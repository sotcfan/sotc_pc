// Memory management (libkernel). The PS4 has two kinds of game memory:
//
// - Direct memory: the console's RAM, shared by CPU and GPU. A game first takes a range of it
//   (sceKernelAllocateDirectMemory returns a "physical" offset) and then maps that range at one
//   or more virtual addresses. Here the physical memory is a memfd of the console's size, and
//   mapping direct memory maps a slice of it, so aliases behave as on the console.
// - Flexible memory: memory allocated and mapped in one step. It comes from the same memfd,
//   from the range after direct memory.
//
// The whole memfd is also mapped a second time ("backing"), never protected, so the GPU layer
// can read and write memory whose pages it protects to track CPU writes (gpu_memory.h).
//
// Virtual addresses chosen by us stay below 1 TiB, as the GPU addresses only 40 bits.


#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cerrno>
#include <cstring>
#include <iterator>
#include <map>
#include <mutex>
#include <format>
#include <shared_mutex>
#include <string>

#include "common/host_memory.h"
#include "common/log.h"
#include "common/platform.h"
#include "common/types.h"
#include "hle/kernel/errno.h"
#include "diag/memory_types.h"
#include "hle/kernel/gpu_memory.h"
#include "hle/registry.h"

namespace hle::kernel {

namespace {

constexpr u64 MiB = 1ull << 20;
constexpr u64 PageSize = 16 * 1024; // PS4 page

// A standard PS4 has 5248 MiB for the game, 512 MiB of which is flexible memory.
constexpr u64 DirectMemorySize = 4736 * MiB;
constexpr u64 FlexibleMemorySize = 512 * MiB;
constexpr u64 PhysicalMemorySize = DirectMemorySize + FlexibleMemorySize;

// The GPU addresses 40 bits; only mappings below this are reported to the GPU layer.
constexpr u64 GpuAddressLimit = 0x100'0000'0000;
constexpr u64 LowGuardSize = 0x40'0000; // mirror of [0, 0x400000): read-only zeros
// Where the mirror of the game's memory starts (gpu_memory.h).
constexpr u64 MirrorOffset = 0x200'0000'0000;

// Where mappings without an address are placed, and the highest address we hand out.
constexpr u64 DefaultMappingBase = 0x2'0000'0000;
constexpr u64 MappingLimit = 0xFC'0000'0000;

// Protection bits.
constexpr s32 ProtCpuRead = 0x01;
constexpr s32 ProtCpuWrite = 0x02; // implies read
constexpr s32 ProtCpuExec = 0x04;
constexpr s32 ProtGpuRead = 0x10;
constexpr s32 ProtGpuWrite = 0x20;

// Mapping flags.
constexpr s32 MapFixed = 0x10;
constexpr s32 MapNoOverwrite = 0x80;

constexpr u64 AlignUp(u64 value, u64 align) {
    return (value + align - 1) / align * align;
}
constexpr bool IsPageAligned(u64 value) {
    return value % PageSize == 0;
}

// The host protection (common/host_memory.h) for a PS4 protection.
int HostProtection(s32 prot) {
    using namespace host_memory;
    int host = ProtNone;
    if (prot & ProtCpuRead) {
        host |= ProtRead;
    }
    if (prot & ProtCpuWrite) {
        host |= ProtRead | ProtWrite;
    }
    if (prot & ProtCpuExec) {
        host |= ProtRead | ProtExec;
    }
    // Until there is a GPU backend, "GPU" access is done by CPU code (copies to Vulkan), so
    // GPU-visible memory must be readable and writable from the CPU.
    if (prot & (ProtGpuRead | ProtGpuWrite)) {
        host |= ProtRead | ProtWrite;
    }
    return host;
}

enum class VmaType { Reserved, Direct, Flexible };

struct Vma {
    u64 size;
    VmaType type;
    s32 prot;
    u64 physical; // offset in the memfd (Direct and Flexible)
    std::string name;
};

// What sceKernelVirtualQuery returns.
struct VirtualQueryInfo {
    u64 start;
    u64 end;
    u64 offset;
    s32 protection;
    s32 memory_type;
    u8 is_flexible : 1;
    u8 is_direct : 1;
    u8 is_stack : 1;
    u8 is_pooled : 1;
    u8 is_committed : 1;
    char name[32];
};
static_assert(sizeof(VirtualQueryInfo) == 72);

struct BatchMapEntry {
    void* start;
    u64 offset;
    u64 length;
    s8 protection;
    s8 type;
    s16 reserved;
    s32 operation; // 0 map direct, 1 unmap, 2 protect, 3 map flexible, 4 type protect
};

// See MemoryManager::ClampToMapped.
struct RunCache {
    u64 generation = ~u64{0};
    struct Run {
        u64 start = 0, end = 0;
    };
    // sotcport: 64 (was 8): the GPU layer looks up every buffer it binds, a few dozen runs a
    // frame, and with 8 they kept evicting each other (~6% of the draw recording thread).
    std::array<Run, 64> runs{};
    u32 next = 0;
};
thread_local RunCache t_runs;

class MemoryManager {
public:
    MemoryManager() {
        backing_ = host_memory::CreatePhysical(PhysicalMemorySize);
        if (!backing_) {
            LOG_CRITICAL("Memory", "could not create the physical memory: {}",
                         platform::LastErrorText());
            return;
        }
#ifdef _WIN32
        // Windows: game mappings replace placeholders, so the range they may use is reserved
        // now (common/host_memory_windows.cpp). Linux maps over free address space directly.
        host_memory::ReserveGuest(DefaultMappingBase, MappingLimit);
#endif
        // The mirror (see GuestMirrorOffset): reserved now, filled as the game maps memory.
        if (host_memory::ReserveGuest(MirrorOffset, MirrorOffset + GpuAddressLimit)) {
            mirror_ = true;
            // The game never maps below its executable (0x400000), but the GPU layer's SRT
            // walker may follow a pointer that is still zero there (a table the queued GPU work
            // has not filled yet; it walks again once that work has run): zeros, not a crash.
            host_memory::MapZeros(MirrorOffset, LowGuardSize, host_memory::ProtRead);
        } else {
            LOG_WARNING("Memory", "no mirror of the game memory: {}", platform::LastErrorText());
        }
    }

    u64 MirrorOffsetOrZero() const {
        return mirror_ ? MirrorOffset : 0;
    }

    s32 AllocateDirect(u64 search_start, u64 search_end, u64 length, u64 alignment, s32 type,
                       s64* out) {
        if (!out || length == 0 || !IsPageAligned(length) || (alignment && !IsPageAligned(alignment))) {
            return ErrorCode(BSD_EINVAL);
        }
        std::unique_lock<TrackedMutex> lock{mutex_};
        const u64 align = std::max(alignment, PageSize);
        search_end = std::min(search_end, DirectMemorySize);
        u64 candidate = AlignUp(search_start, align);
        for (const auto& [start, allocation] : physical_) {
            if (start >= search_end) {
                break;
            }
            if (start >= candidate + length) {
                break; // the gap before this allocation fits
            }
            if (start + allocation.size > candidate) {
                candidate = AlignUp(start + allocation.size, align);
            }
        }
        if (candidate + length > search_end) {
            LOG_ERROR("Memory", "no direct memory for {:#x} bytes", length);
            return ErrorCode(BSD_EAGAIN);
        }
        physical_[candidate] = {length, type};
        lock.unlock();
        // Written after the lock is released: the pointer is the game's memory, which the GPU layer may
        // be tracking; a write there faults, and the fault handler calls back into this manager
        // (on the same thread: with the lock still held, the rwlock reports EDEADLK and the
        // process aborts — seen on 2026-10-10 after 16 minutes of play).
        *out = static_cast<s64>(candidate);
        LOG_INFO("Memory", "direct memory {:#x}-{:#x} allocated (type {})", candidate,
                  candidate + length, type);
        return 0;
    }

    s32 ReleaseDirect(u64 start, u64 length) {
        std::scoped_lock<TrackedMutex> lock{mutex_};
        for (auto it = physical_.lower_bound(start); it != physical_.end() && it->first < start + length;) {
            it = physical_.erase(it);
        }
        // Give the RAM back to the host.
        host_memory::Discard(start, length);
        return 0;
    }

    s32 AvailableDirect(u64 search_start, u64 search_end, u64 alignment, u64* out_start,
                        u64* out_size) {
        if (!out_start || !out_size) {
            return ErrorCode(BSD_EINVAL);
        }
        std::unique_lock<TrackedMutex> lock{mutex_};
        const u64 align = std::max(alignment, PageSize);
        search_end = std::min(search_end, DirectMemorySize);
        u64 best_start = 0, best_size = 0;
        u64 cursor = search_start;
        const auto consider = [&](u64 gap_end) {
            const u64 start = AlignUp(cursor, align);
            if (gap_end > start && gap_end - start > best_size) {
                best_start = start;
                best_size = gap_end - start;
            }
        };
        for (const auto& [start, allocation] : physical_) {
            if (start >= search_end) {
                break;
            }
            if (start + allocation.size <= cursor) {
                continue;
            }
            consider(std::min(start, search_end));
            cursor = std::max(cursor, start + allocation.size);
        }
        consider(search_end);
        if (best_size == 0) {
            return ErrorCode(BSD_ENOMEM);
        }
        lock.unlock();
        // Written after the lock is released (see AllocateDirect).
        *out_start = best_start;
        *out_size = best_size;
        return 0;
    }

    // Maps [*address, +length). Without MapFixed, *address is only a hint.
    s32 Map(void** address, u64 length, s32 prot, s32 flags, VmaType type, u64 physical,
            u64 alignment, const char* name) {
        if (!address || length == 0 || !IsPageAligned(length)) {
            return ErrorCode(BSD_EINVAL);
        }
        // The game's arguments are read before taking the lock: they are guest memory, which
        // the GPU layer may be tracking, and a read fault there is handled by waiting for the GPU
        // command thread, which may itself be waiting for this lock (see AllocateDirect).
        u64 start = reinterpret_cast<u64>(*address);
        const std::string mapping_name = name ? name : "";
        std::unique_lock<TrackedMutex> lock{mutex_};
        const u64 align = std::max<u64>(alignment, PageSize);
        if (flags & MapFixed) {
            if (start == 0 || !IsPageAligned(start)) {
                return ErrorCode(BSD_EINVAL);
            }
            if ((flags & MapNoOverwrite) && Overlaps(start, length)) {
                return ErrorCode(BSD_ENOMEM);
            }
            // Mapping over memory unmaps it: the GPU layer must hear it first, while the old
            // memory is still there (it may copy GPU-written data back into it).
            if (OverlapsMemory(start, length) && start + length <= GpuAddressLimit) {
                lock.unlock();
                Notify(&GpuMemoryObserver::OnUnmapped, start, length);
                lock.lock();
            }
        } else {
            start = FindFree(start ? start : DefaultMappingBase, length, align);
            if (start == 0) {
                LOG_ERROR("Memory", "no address space for {:#x} bytes", length);
                return ErrorCode(BSD_ENOMEM);
            }
        }

        if (type == VmaType::Flexible) {
            if (!AllocateFlexible(length, physical)) {
                LOG_ERROR("Memory", "out of flexible memory");
                return ErrorCode(BSD_ENOMEM);
            }
        }

        if (!HostMap(start, length, prot, type, physical)) {
            LOG_ERROR("Memory", "mapping at {:#x} ({:#x} bytes) failed: {}", start, length,
                      platform::LastErrorText());
            if (type == VmaType::Flexible) {
                FreeFlexible(physical, length);
            }
            return ErrorCode(BSD_ENOMEM);
        }
        const bool replaced_memory = Remove(start, length);
        vmas_[start] = {length, type, prot, physical, mapping_name};
        generation_.fetch_add(1, std::memory_order_release);
        if (type != VmaType::Reserved) {
            AddRun(start, start + length);
            if (mirror_ && start + length <= GpuAddressLimit) {
                if (!host_memory::MapPhysical(start + MirrorOffset, length, physical,
                                              host_memory::ProtRead | host_memory::ProtWrite)) {
                    LOG_ERROR("Memory", "mirror of {:#x}-{:#x} failed: {}", start,
                              start + length, platform::LastErrorText());
                }
            }
        }
        LOG_DEBUG("Memory", "{} {:#x}-{:#x} prot {:#x} ({})", TypeName(type), start,
                  start + length, prot, mapping_name);
        lock.unlock();
        *address = reinterpret_cast<void*>(start); // after the unlock (see AllocateDirect)

        (void)replaced_memory; // the GPU layer was told before the mapping was replaced
        if (start + length <= GpuAddressLimit) {
            if (type != VmaType::Reserved) {
                Notify(&GpuMemoryObserver::OnMapped, start, length);
            }
        }
        return 0;
    }

    s32 Unmap(u64 start, u64 length) {
        std::unique_lock<TrackedMutex> lock{mutex_};
        length = AlignUp(length, PageSize);
        if (!OverlapsOnlyOurs(start, length)) {
            LOG_ERROR("Memory", "munmap of {:#x}-{:#x}, which the game did not map", start,
                      start + length);
            return ErrorCode(BSD_EINVAL);
        }
        // The GPU layer hears it first, while the memory is still there: it may copy
        // GPU-written data back into it (direct memory keeps its contents after an unmap).
        lock.unlock();
        Notify(&GpuMemoryObserver::OnUnmapped, start, length);
        lock.lock();
        Remove(start, length);
        host_memory::Unmap(start, length);
        return 0;
    }

    s32 Protect(u64 start, u64 length, s32 prot) {
        {
            std::scoped_lock<TrackedMutex> lock{mutex_};
            if (!host_memory::ProtectGuest(start, AlignUp(length, PageSize),
                                           HostProtection(prot))) {
                return ErrorCode(BSD_EINVAL);
            }
            for (auto it = vmas_.lower_bound(start); it != vmas_.end() && it->first < start + length;
                 ++it) {
                it->second.prot = prot;
            }
        }
        Notify(&GpuMemoryObserver::OnProtectionChanged, start, length);
        return 0;
    }

    s32 Query(u64 address, s32 flags, VirtualQueryInfo* info) {
        if (!info) {
            return ErrorCode(BSD_EFAULT);
        }
        VirtualQueryInfo result{};
        {
            std::scoped_lock<TrackedMutex> lock{mutex_};
            auto it = vmas_.upper_bound(address);
            if (it != vmas_.begin() &&
                std::prev(it)->first + std::prev(it)->second.size > address) {
                --it;
            } else if (!(flags & 1)) { // 1 = return the next mapping when nothing is at `address`
                return ErrorCode(BSD_EACCES);
            }
            if (it == vmas_.end()) {
                return ErrorCode(BSD_EACCES);
            }
            const Vma& vma = it->second;
            result.start = it->first;
            result.end = it->first + vma.size;
            result.offset = vma.type == VmaType::Direct ? vma.physical : 0;
            result.protection = vma.prot;
            result.is_flexible = vma.type == VmaType::Flexible;
            result.is_direct = vma.type == VmaType::Direct;
            result.is_committed = vma.type != VmaType::Reserved;
            std::strncpy(result.name, vma.name.c_str(), sizeof(result.name) - 1);
        }
        *info = result; // after the unlock (see AllocateDirect)
        return 0;
    }

    // The kind of physical memory behind `address` (see MemoryTypeAt in gpu_memory.h).
    s32 MemoryTypeAt(u64 address) {
        std::shared_lock<TrackedMutex> lock{mutex_};
        auto it = vmas_.upper_bound(address);
        if (it == vmas_.begin()) {
            return -1;
        }
        --it;
        const Vma& vma = it->second;
        if (address >= it->first + vma.size || vma.type == VmaType::Reserved) {
            return -1;
        }
        if (vma.type == VmaType::Flexible) {
            return FlexibleMemoryType;
        }
        const u64 physical = vma.physical + (address - it->first);
        auto allocation = physical_.upper_bound(physical);
        if (allocation == physical_.begin()) {
            return -1;
        }
        --allocation;
        return physical < allocation->first + allocation->second.size ? allocation->second.type
                                                                       : -1;
    }

    u64 FlexibleAvailable() {
        std::scoped_lock<TrackedMutex> lock{mutex_};
        u64 used = 0;
        for (const auto& [start, size] : flexible_physical_) {
            used += size;
        }
        return FlexibleMemorySize - used;
    }

    // ---- For the GPU layer (gpu_memory.h) ----

    void SetObserver(GpuMemoryObserver* observer) {
        observer_ = observer;
    }

    // ClampToMapped (and BackingAddress through it) runs several times per GPU draw and inside the
    // write tracking fault handler: each thread remembers the last runs of contiguous memory it
    // looked up (t_runs), valid while no mapping changes (generation_), so most calls take no
    // lock.
    u64 ClampToMapped(u64 address, u64 size) {
        RunCache& cache = t_runs;
        const u64 generation = generation_.load(std::memory_order_acquire);
        if (cache.generation == generation) {
            for (const auto& run : cache.runs) {
                if (address - run.start < run.end - run.start) {
                    return std::min(run.end - address, size);
                }
            }
        } else {
            cache = {};
            cache.generation = generation;
        }
        std::shared_lock<TrackedMutex> lock{mutex_};
        auto it = runs_.upper_bound(address);
        if (it == runs_.begin()) {
            return 0;
        }
        --it;
        if (it->second <= address) {
            return 0;
        }
        if (generation_.load(std::memory_order_relaxed) == generation) {
            cache.runs[cache.next++ % cache.runs.size()] = {it->first, it->second};
        }
        return std::min(it->second, address + size) - address;
    }

    u8* BackingAddress(u64 address) {
        if (mirror_ && address < GpuAddressLimit) {
            // Below the GPU limit all memory is also mapped at the mirror.
            return ClampToMapped(address, 1) ? reinterpret_cast<u8*>(address + MirrorOffset)
                                             : nullptr;
        }
        std::shared_lock<TrackedMutex> lock{mutex_};
        const auto it = Find(address);
        if (it == vmas_.end() || it->second.type == VmaType::Reserved || !backing_) {
            return nullptr;
        }
        return backing_ + it->second.physical + (address - it->first);
    }

    // The tracker passes ranges of whole pages that may span holes between mappings; mprotect
    // over a hole fails (ENOMEM) and leaves the pages after it unprotected while the tracker
    // believes they are watched. So each mapped piece is protected on its own, never with
    // more access than the game gave it.
    void ProtectForTracking(u64 address, u64 size, bool readable, bool writable) {
        // Shared: this only reads vmas_ (the mprotect calls change no state of ours), and it runs
        // for every CPU write fault on GPU-tracked memory, from several threads at once while the
        // game streams data in. With the exclusive lock those faults queued behind each other's
        // mprotect (a system call with TLB shootdowns: tens of microseconds under load).
        // Mappings still change only under the exclusive lock, so the pieces stay valid.
        std::shared_lock<TrackedMutex> lock{mutex_};
        const u64 end = address + size;
        auto it = vmas_.upper_bound(address);
        if (it != vmas_.begin()) {
            --it;
        }
        for (; it != vmas_.end() && it->first < end; ++it) {
            const u64 piece_start = std::max(it->first, address);
            const u64 piece_end = std::min(it->first + it->second.size, end);
            if (piece_start >= piece_end || it->second.type == VmaType::Reserved) {
                continue;
            }
            int prot = HostProtection(it->second.prot);
            if (!readable) {
                prot &= ~host_memory::ProtRead;
            }
            if (!writable) {
                prot &= ~host_memory::ProtWrite;
            }
            if (!(prot & host_memory::ProtRead)) {
                prot &= ~host_memory::ProtWrite; // write-only is not a valid host protection
            }
            if (!host_memory::ProtectGuest(piece_start, piece_end - piece_start, prot)) {
                LOG_ERROR("Memory", "tracking protection failed at [{:#x}, {:#x}): {}",
                          piece_start, piece_end, platform::LastErrorText());
            }
        }
    }

    std::vector<std::pair<u64, u64>> MappedRanges() {
        std::scoped_lock<TrackedMutex> lock{mutex_};
        std::vector<std::pair<u64, u64>> ranges;
        for (const auto& [start, vma] : vmas_) {
            if (vma.type != VmaType::Reserved) {
                ranges.emplace_back(start, start + vma.size);
            }
        }
        return ranges;
    }

private:
    struct PhysicalAllocation {
        u64 size;
        s32 type;
    };

    // The VMA containing `address`, or end().
    std::map<u64, Vma>::iterator Find(u64 address) {
        auto it = vmas_.upper_bound(address);
        if (it == vmas_.begin()) {
            return vmas_.end();
        }
        --it;
        return it->first + it->second.size > address ? it : vmas_.end();
    }

    void Notify(void (GpuMemoryObserver::*event)(u64, u64), u64 start, u64 length) {
        if (GpuMemoryObserver* observer = observer_.load()) {
            (observer->*event)(start, length);
        }
    }

    // First-fit allocation in the flexible part of the memfd.
    bool AllocateFlexible(u64 length, u64& out) {
        u64 candidate = DirectMemorySize;
        for (const auto& [start, size] : flexible_physical_) {
            if (start >= candidate + length) {
                break;
            }
            candidate = std::max(candidate, start + size);
        }
        if (candidate + length > PhysicalMemorySize) {
            return false;
        }
        flexible_physical_[candidate] = length;
        out = candidate;
        return true;
    }

    // Frees [start, +length) of flexible memory (possibly part of an allocation) and gives the
    // RAM back to the host, so it reads as zero when reused.
    void FreeFlexible(u64 start, u64 length) {
        const u64 end = start + length;
        auto it = flexible_physical_.upper_bound(start);
        if (it != flexible_physical_.begin()) {
            --it;
        }
        while (it != flexible_physical_.end() && it->first < end) {
            const u64 block_start = it->first;
            const u64 block_end = block_start + it->second;
            if (block_end <= start) {
                ++it;
                continue;
            }
            it = flexible_physical_.erase(it);
            if (block_start < start) {
                flexible_physical_[block_start] = start - block_start;
            }
            if (block_end > end) {
                it = flexible_physical_.emplace(end, block_end - end).first;
                ++it;
            }
        }
        host_memory::Discard(start, length);
    }

    static const char* TypeName(VmaType type) {
        switch (type) {
        case VmaType::Reserved: return "reserved";
        case VmaType::Direct: return "direct";
        case VmaType::Flexible: return "flexible";
        }
        return "?";
    }

    bool HostMap(u64 start, u64 length, s32 prot, VmaType type, u64 physical) {
        // Replacing our own mappings (or a reservation) is allowed; anything else at that
        // address belongs to the host process and must not be touched.
        if (!OverlapsOnlyOurs(start, length) && !host_memory::IsFree(start, length)) {
            return false;
        }
        switch (type) {
        case VmaType::Reserved:
            return host_memory::MapReserved(start, length);
        case VmaType::Direct:
        case VmaType::Flexible:
            diag::NoteGameMapping(start, length, physical, type == VmaType::Flexible);
            return host_memory::MapPhysical(start, length, physical, HostProtection(prot));
        }
        return false;
    }

    bool Overlaps(u64 start, u64 length) const {
        return OverlapEnd(start, length) != 0;
    }

    // True if part of the range has memory behind it (not just a reservation).
    bool OverlapsMemory(u64 start, u64 length) const {
        auto it = vmas_.upper_bound(start);
        if (it != vmas_.begin()) {
            --it;
        }
        for (; it != vmas_.end() && it->first < start + length; ++it) {
            if (it->first + it->second.size > start && it->second.type != VmaType::Reserved) {
                return true;
            }
        }
        return false;
    }

    // True if every page of the range is free or covered by one of our mappings.
    bool OverlapsOnlyOurs(u64 start, u64 length) const {
        u64 cursor = start;
        auto it = vmas_.upper_bound(start);
        if (it != vmas_.begin()) {
            --it;
        }
        for (; it != vmas_.end() && it->first < start + length; ++it) {
            if (it->first + it->second.size <= cursor) {
                continue;
            }
            if (it->first > cursor) {
                return false; // a hole we do not own
            }
            cursor = it->first + it->second.size;
        }
        return cursor >= start + length;
    }

    // End of the first of our mappings that overlaps [start, +length), or 0 if none does.
    u64 OverlapEnd(u64 start, u64 length) const {
        auto it = vmas_.upper_bound(start);
        if (it != vmas_.begin() && std::prev(it)->first + std::prev(it)->second.size > start) {
            --it;
        }
        if (it != vmas_.end() && it->first < start + length) {
            return it->first + it->second.size;
        }
        return 0;
    }

    // First free, aligned range at or above `from`. The host may own ranges we do not know
    // about, so a candidate is only accepted after asking the host (host_memory::IsFree).
    u64 FindFree(u64 from, u64 length, u64 align) {
        u64 candidate = AlignUp(from, align);
        while (candidate + length <= MappingLimit) {
            if (const u64 end = OverlapEnd(candidate, length)) {
                candidate = AlignUp(end, align);
                continue;
            }
            if (host_memory::IsFree(candidate, length)) {
                return candidate;
            }
            candidate = AlignUp(candidate + std::max<u64>(length, 256 * MiB), align);
        }
        return 0;
    }

    // Forgets the VMAs inside [start, +length), splitting those that cross the edges. Returns
    // true if any of them was memory (not a reservation).
    bool Remove(u64 start, u64 length) {
        bool removed_memory = false;
        const u64 end = start + length;
        generation_.fetch_add(1, std::memory_order_release);
        SubtractRun(start, end);
        if (mirror_ && end <= GpuAddressLimit) {
            // Zeros rather than no access: the GPU thread may still be copying from here (it
            // looked the range up just before) and must read stale data, not crash.
            host_memory::MapZeros(start + MirrorOffset, length,
                                  host_memory::ProtRead | host_memory::ProtWrite);
        }
        auto it = vmas_.upper_bound(start);
        if (it != vmas_.begin()) {
            --it;
        }
        while (it != vmas_.end() && it->first < end) {
            const u64 vma_start = it->first;
            const u64 vma_end = vma_start + it->second.size;
            if (vma_end <= start) {
                ++it;
                continue;
            }
            Vma vma = it->second;
            it = vmas_.erase(it);
            const u64 cut_start = std::max(vma_start, start);
            const u64 cut_end = std::min(vma_end, end);
            if (vma.type == VmaType::Flexible) {
                FreeFlexible(vma.physical + (cut_start - vma_start), cut_end - cut_start);
            }
            removed_memory |= vma.type != VmaType::Reserved;
            if (vma_start < start) {
                Vma left = vma;
                left.size = start - vma_start;
                vmas_[vma_start] = left;
            }
            if (vma_end > end) {
                Vma right = vma;
                right.size = vma_end - end;
                if (right.type != VmaType::Reserved) {
                    right.physical += end - vma_start;
                }
                it = vmas_.emplace(end, right).first;
                ++it;
            }
        }
        return removed_memory;
    }

    // [start, end) joined into the run of contiguous memory (not reservations) around it.
    void AddRun(u64 start, u64 end) {
        auto it = runs_.upper_bound(start);
        if (it != runs_.begin() && std::prev(it)->second >= start) {
            --it;
        }
        while (it != runs_.end() && it->first <= end) {
            start = std::min(start, it->first);
            end = std::max(end, it->second);
            it = runs_.erase(it);
        }
        runs_.emplace(start, end);
    }

    // [start, end) taken out of the runs of contiguous memory.
    void SubtractRun(u64 start, u64 end) {
        auto it = runs_.upper_bound(start);
        if (it != runs_.begin() && std::prev(it)->second > start) {
            --it;
        }
        while (it != runs_.end() && it->first < end) {
            const u64 run_start = it->first;
            const u64 run_end = it->second;
            it = runs_.erase(it);
            if (run_start < start) {
                runs_.emplace(run_start, start);
            }
            if (run_end > end) {
                it = runs_.emplace(end, run_end).first;
                ++it;
            }
        }
    }

    // Exclusive for changes; shared for the lookups the GPU layer makes all the time.
    // Records who holds it exclusively (diagnostics for freezes: see DescribeMemoryLock).
    class TrackedMutex {
    public:
        void lock() {
            mutex.lock();
            owner.store(CurrentTid(), std::memory_order_relaxed);
            owner_site.store(reinterpret_cast<u64>(__builtin_return_address(0)),
                             std::memory_order_relaxed);
        }
        bool try_lock() {
            if (!mutex.try_lock()) {
                return false;
            }
            owner.store(CurrentTid(), std::memory_order_relaxed);
            return true;
        }
        void unlock() {
            owner.store(0, std::memory_order_relaxed);
            mutex.unlock();
        }
        // A shared lock taken by the thread that already holds the lock exclusively passes
        // through: the game's thread stacks are game memory, so the first push on a stack page
        // the GPU layer write-protects faults, even inside our own code holding this lock, and
        // the fault handler asks the memory manager (ClampToMapped, ...) for a shared lock. A
        // real rwlock refuses that (EDEADLK, "Resource deadlock avoided": a crash on entering
        // the sixth colossus' arena, 2026-10-12). Such a fault comes at a call or a push,
        // normally right after taking the lock, before the tables change.
        static u32 CurrentTid() {
            thread_local const u32 tid = platform::CurrentThreadId();
            return tid;
        }
        static int& ReentrantShared() {
            thread_local int depth = 0;
            return depth;
        }
        void lock_shared() {
            if (owner.load(std::memory_order_relaxed) == CurrentTid()) {
                ReentrantShared()++;
                return;
            }
            mutex.lock_shared();
            shared_holders.fetch_add(1, std::memory_order_relaxed);
        }
        bool try_lock_shared() {
            if (owner.load(std::memory_order_relaxed) == CurrentTid()) {
                ReentrantShared()++;
                return true;
            }
            if (!mutex.try_lock_shared()) {
                return false;
            }
            shared_holders.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
        void unlock_shared() {
            if (ReentrantShared() > 0) {
                ReentrantShared()--;
                return;
            }
            shared_holders.fetch_sub(1, std::memory_order_relaxed);
            mutex.unlock_shared();
        }
        std::shared_mutex mutex;
        std::atomic<u32> owner{0};
        std::atomic<u64> owner_site{0};
        std::atomic<int> shared_holders{0};
    };
    TrackedMutex mutex_;

public:
    std::string DescribeLock() {
#ifdef _WIN32
        return std::format("memory manager lock: exclusive owner tid {} (locked at {:#x}), "
                           "{} shared holders",
                           mutex_.owner.load(), mutex_.owner_site.load(),
                           mutex_.shared_holders.load());
#else
        // glibc's pthread_rwlock_t: __readers, __writers, __wrphase_futex, __writers_futex, ...,
        // __cur_writer at offset 24 (the tid of the thread holding it for writing).
        u32 words[8];
        std::memcpy(words, mutex_.mutex.native_handle(), sizeof(words));
        return std::format("memory manager lock: exclusive owner tid {} (locked at {:#x}), "
                           "{} shared holders; rwlock readers={:#x} writers={:#x} wrphase={:#x} "
                           "writers_futex={:#x} cur_writer={}",
                           mutex_.owner.load(), mutex_.owner_site.load(),
                           mutex_.shared_holders.load(), words[0], words[1], words[2], words[3],
                           words[6]);
#endif
    }

private:
    u8* backing_ = nullptr;
    std::map<u64, PhysicalAllocation> physical_; // direct memory allocations
    std::map<u64, u64> flexible_physical_;       // flexible memory in use: start -> size
    std::map<u64, Vma> vmas_;
    std::map<u64, u64> runs_; // contiguous memory (reservations excluded): start -> end
    bool mirror_ = false;
    std::atomic<u64> generation_{0}; // changes with every change to vmas_
    std::atomic<GpuMemoryObserver*> observer_{nullptr};
};

MemoryManager& Manager() {
    static MemoryManager manager;
    return manager;
}

PS4_SYSV_ABI u64 sceKernelGetDirectMemorySize() {
    return DirectMemorySize;
}

PS4_SYSV_ABI s32 sceKernelAllocateDirectMemory(s64 search_start, s64 search_end, u64 length,
                                               u64 alignment, s32 memory_type, s64* out) {
    if (search_start < 0 || search_end <= search_start) {
        return ErrorCode(BSD_EINVAL);
    }
    return Manager().AllocateDirect(search_start, search_end, length, alignment, memory_type, out);
}

PS4_SYSV_ABI s32 sceKernelReleaseDirectMemory(u64 start, u64 length) {
    return Manager().ReleaseDirect(start, length);
}

PS4_SYSV_ABI s32 sceKernelAvailableDirectMemorySize(u64 search_start, u64 search_end,
                                                    u64 alignment, u64* out_start, u64* out_size) {
    return Manager().AvailableDirect(search_start, search_end, alignment, out_start, out_size);
}

PS4_SYSV_ABI s32 sceKernelMapNamedDirectMemory(void** address, u64 length, s32 prot, s32 flags,
                                               s64 physical, u64 alignment, const char* name) {
    if (physical < 0 || !IsPageAligned(physical) ||
        static_cast<u64>(physical) + length > DirectMemorySize) {
        return ErrorCode(BSD_EINVAL);
    }
    return Manager().Map(address, length, prot, flags, VmaType::Direct, physical, alignment, name);
}

PS4_SYSV_ABI s32 sceKernelMapDirectMemory(void** address, u64 length, s32 prot, s32 flags,
                                          s64 physical, u64 alignment) {
    return sceKernelMapNamedDirectMemory(address, length, prot, flags, physical, alignment, "");
}

PS4_SYSV_ABI s32 sceKernelMapNamedFlexibleMemory(void** address, u64 length, s32 prot, s32 flags,
                                                 const char* name) {
    return Manager().Map(address, length, prot, flags, VmaType::Flexible, 0, 0, name);
}

PS4_SYSV_ABI s32 sceKernelMapFlexibleMemory(void** address, u64 length, s32 prot, s32 flags) {
    return sceKernelMapNamedFlexibleMemory(address, length, prot, flags, "");
}

PS4_SYSV_ABI s32 sceKernelReserveVirtualRange(void** address, u64 length, s32 flags,
                                              u64 alignment) {
    if (alignment && (!std::has_single_bit(alignment) || !IsPageAligned(alignment))) {
        return ErrorCode(BSD_EINVAL);
    }
    return Manager().Map(address, length, 0, flags, VmaType::Reserved, 0, alignment, "reserved");
}

PS4_SYSV_ABI s32 sceKernelMunmap(void* address, u64 length) {
    return Manager().Unmap(reinterpret_cast<u64>(address), length);
}

PS4_SYSV_ABI s32 sceKernelMprotect(const void* address, u64 length, s32 prot) {
    return Manager().Protect(reinterpret_cast<u64>(address), length, prot);
}

PS4_SYSV_ABI s32 sceKernelVirtualQuery(const void* address, s32 flags, VirtualQueryInfo* info,
                                       u64 info_size) {
    if (info_size < sizeof(VirtualQueryInfo)) {
        return ErrorCode(BSD_EINVAL);
    }
    return Manager().Query(reinterpret_cast<u64>(address), flags, info);
}

PS4_SYSV_ABI s32 sceKernelAvailableFlexibleMemorySize(u64* out_size) {
    if (!out_size) {
        return ErrorCode(BSD_EINVAL);
    }
    *out_size = Manager().FlexibleAvailable();
    return 0;
}

PS4_SYSV_ABI s32 sceKernelBatchMap2(BatchMapEntry* entries, s32 count, s32* processed, s32 flags) {
    s32 result = 0;
    s32 done = 0;
    for (; done < count; done++) {
        BatchMapEntry& entry = entries[done];
        switch (entry.operation) {
        case 0:
            result = sceKernelMapNamedDirectMemory(&entry.start, entry.length, entry.protection,
                                                   flags, static_cast<s64>(entry.offset), 0, "");
            break;
        case 1: result = sceKernelMunmap(entry.start, entry.length); break;
        case 2: result = sceKernelMprotect(entry.start, entry.length, entry.protection); break;
        case 3:
            result = sceKernelMapNamedFlexibleMemory(&entry.start, entry.length, entry.protection,
                                                     flags, "");
            break;
        case 4: result = sceKernelMprotect(entry.start, entry.length, entry.protection); break;
        default: result = ErrorCode(BSD_EINVAL); break;
        }
        if (result != 0) {
            break;
        }
    }
    if (processed) {
        *processed = done;
    }
    return result;
}

PS4_SYSV_ABI s32 sceKernelBatchMap(BatchMapEntry* entries, s32 count, s32* processed) {
    return sceKernelBatchMap2(entries, count, processed, MapFixed);
}

} // namespace

void SetGpuMemoryObserver(GpuMemoryObserver* observer) {
    Manager().SetObserver(observer);
}

u8* BackingAddress(u64 address) {
    return Manager().BackingAddress(address);
}

u64 ClampToMapped(u64 address, u64 size) {
    return Manager().ClampToMapped(address, size);
}

u64 GuestMirrorOffset() {
    return Manager().MirrorOffsetOrZero();
}

s32 MemoryTypeAt(u64 address) {
    return Manager().MemoryTypeAt(address);
}

void ProtectForTracking(u64 address, u64 size, bool readable, bool writable) {
    Manager().ProtectForTracking(address, size, readable, writable);
}

std::string DescribeMemoryLock() {
    return Manager().DescribeLock();
}

std::vector<std::pair<u64, u64>> MappedRanges() {
    return Manager().MappedRanges();
}

HLE_FUNCTION("libkernel", "sceKernelGetDirectMemorySize", sceKernelGetDirectMemorySize);
HLE_FUNCTION("libkernel", "sceKernelAllocateDirectMemory", sceKernelAllocateDirectMemory);
HLE_FUNCTION("libkernel", "sceKernelReleaseDirectMemory", sceKernelReleaseDirectMemory);
HLE_FUNCTION("libkernel", "sceKernelAvailableDirectMemorySize", sceKernelAvailableDirectMemorySize);
HLE_FUNCTION("libkernel", "sceKernelMapDirectMemory", sceKernelMapDirectMemory);
HLE_FUNCTION("libkernel", "sceKernelMapNamedDirectMemory", sceKernelMapNamedDirectMemory);
HLE_FUNCTION("libkernel", "sceKernelMapFlexibleMemory", sceKernelMapFlexibleMemory);
HLE_FUNCTION("libkernel", "sceKernelMapNamedFlexibleMemory", sceKernelMapNamedFlexibleMemory);
HLE_FUNCTION("libkernel", "sceKernelReserveVirtualRange", sceKernelReserveVirtualRange);
HLE_FUNCTION("libkernel", "sceKernelMunmap", sceKernelMunmap);
HLE_FUNCTION("libkernel", "sceKernelMprotect", sceKernelMprotect);
HLE_FUNCTION("libkernel", "sceKernelVirtualQuery", sceKernelVirtualQuery);
HLE_FUNCTION("libkernel", "sceKernelAvailableFlexibleMemorySize", sceKernelAvailableFlexibleMemorySize);
HLE_FUNCTION("libkernel", "sceKernelBatchMap", sceKernelBatchMap);
HLE_FUNCTION("libkernel", "sceKernelBatchMap2", sceKernelBatchMap2);

} // namespace hle::kernel
