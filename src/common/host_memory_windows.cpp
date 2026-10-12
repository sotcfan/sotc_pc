// Windows implementation of common/host_memory.h.
//
// Adapted from the Windows fork of bbport (Supermedo, src/runtime_memory.c, GPL-2.0-or-later),
// itself following shadPS4: the game's physical memory is a pagefile-backed section, and each
// guest range (the game's address space, the mirror) is reserved up front as one placeholder.
// Mapping a range carves it out as a single placeholder first (views and private memory
// overlapping it are given back, placeholders coalesced and split), then replaces it with a view
// of the section (MapViewOfFile3) or with private memory (VirtualAlloc2). The parts of a view cut
// off by the carve are mapped again with the page protections they had, since the GPU layer's
// write tracking protects pages inside views.
//
// Calls that change mappings come from hle/kernel/memory.cpp with its exclusive lock held;
// ProtectGuest runs under its shared lock and only reads the region table.
#ifdef _WIN32

#include "common/host_memory.h"

#include <windows.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "common/log.h"

namespace host_memory {

namespace {

enum class Kind : u8 {
    Placeholder, // reserved, inaccessible
    View,        // a view of the section at `physical`
    Private,     // private zero-filled memory (the mirror's stand-ins for unmapped memory)
};

struct Region {
    u64 start;
    u64 end;
    Kind kind;
    u64 physical;
};

// Sorted, contiguous inside each reservation; covers every reserved byte.
std::vector<Region> g_regions;
HANDLE g_section = nullptr;
u8* g_backing = nullptr;

DWORD WinProt(int prot) {
    if (prot & ProtExec) {
        if (prot & ProtWrite) {
            return PAGE_EXECUTE_READWRITE;
        }
        return (prot & ProtRead) ? PAGE_EXECUTE_READ : PAGE_EXECUTE;
    }
    if (prot & ProtWrite) {
        return PAGE_READWRITE;
    }
    return (prot & ProtRead) ? PAGE_READONLY : PAGE_NOACCESS;
}

[[noreturn]] void Fail(const char* what, u64 address, u64 size) {
    std::fprintf(stderr, "STOP: %s failed at %#llx (+%#llx), error %lu\n", what,
                 static_cast<unsigned long long>(address), static_cast<unsigned long long>(size),
                 GetLastError());
    std::fflush(stderr);
    std::_Exit(21);
}

// First region with end > address.
size_t RegionIndex(u64 address) {
    const auto it = std::upper_bound(g_regions.begin(), g_regions.end(), address,
                                     [](u64 a, const Region& r) { return a < r.end; });
    return static_cast<size_t>(it - g_regions.begin());
}

bool InsideReservation(u64 start, u64 end) {
    size_t i = RegionIndex(start);
    u64 cursor = start;
    for (; i < g_regions.size() && cursor < end; ++i) {
        if (g_regions[i].start > cursor) {
            return false; // a gap between reservations
        }
        cursor = g_regions[i].end;
    }
    return cursor >= end;
}

void MapView(const Region& r, DWORD protect) {
    // A view cannot start out inaccessible: map it read/write, then protect it.
    const DWORD initial = protect == PAGE_NOACCESS ? PAGE_READWRITE : protect;
    if (!MapViewOfFile3(g_section, GetCurrentProcess(), reinterpret_cast<void*>(r.start),
                        r.physical, r.end - r.start, MEM_REPLACE_PLACEHOLDER, initial, nullptr,
                        0)) {
        Fail("MapViewOfFile3", r.start, r.end - r.start);
    }
    DWORD old;
    if (initial != protect &&
        !VirtualProtect(reinterpret_cast<void*>(r.start), r.end - r.start, protect, &old)) {
        Fail("VirtualProtect", r.start, r.end - r.start);
    }
}

void MapPrivate(const Region& r, DWORD protect) {
    if (!VirtualAlloc2(GetCurrentProcess(), reinterpret_cast<void*>(r.start), r.end - r.start,
                       MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER, protect, nullptr, 0)) {
        Fail("VirtualAlloc2 (private)", r.start, r.end - r.start);
    }
}

// Gives a view or private region back to a placeholder of the same extent.
void ToPlaceholder(const Region& r) {
    if (r.kind == Kind::View) {
        if (!UnmapViewOfFile2(GetCurrentProcess(), reinterpret_cast<void*>(r.start),
                              MEM_PRESERVE_PLACEHOLDER)) {
            Fail("UnmapViewOfFile2", r.start, r.end - r.start);
        }
    } else if (r.kind == Kind::Private) {
        if (!VirtualFree(reinterpret_cast<void*>(r.start), r.end - r.start,
                         MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) {
            Fail("VirtualFree (private)", r.start, r.end - r.start);
        }
    }
}

struct ProtRun {
    u64 start;
    u64 end;
    DWORD protect;
};

void SaveProtections(u64 start, u64 end, std::vector<ProtRun>& runs) {
    for (u64 at = start; at < end;) {
        MEMORY_BASIC_INFORMATION m;
        if (!VirtualQuery(reinterpret_cast<void*>(at), &m, sizeof(m))) {
            Fail("VirtualQuery", at, 0);
        }
        const u64 next = std::min<u64>(reinterpret_cast<u64>(m.BaseAddress) + m.RegionSize, end);
        runs.push_back({at, next, m.Protect});
        at = next;
    }
}

// Leaves [a, b) as exactly one placeholder; returns its region index. The parts of mapped
// regions outside [a, b) are mapped again, with their page protections.
size_t Carve(u64 a, u64 b) {
    if (!InsideReservation(a, b)) {
        Fail("guest range outside the reservation", a, b - a);
    }
    const size_t first = RegionIndex(a);
    size_t last = first;
    while (last < g_regions.size() && g_regions[last].start < b) {
        ++last;
    }
    const u64 lo = g_regions[first].start;
    const u64 hi = g_regions[last - 1].end;
    const Region left = g_regions[first];
    const Region right = g_regions[last - 1];
    if (first == last - 1 && left.kind == Kind::Placeholder && lo == a && hi == b) {
        return first;
    }
    const bool keep_left = left.kind != Kind::Placeholder && lo < a;
    const bool keep_right = right.kind != Kind::Placeholder && b < hi;
    std::vector<ProtRun> runs;
    if (keep_left) {
        SaveProtections(lo, a, runs);
    }
    if (keep_right) {
        SaveProtections(b, hi, runs);
    }
    for (size_t i = first; i < last; ++i) {
        ToPlaceholder(g_regions[i]);
    }
    // Neighboring placeholders inside one reservation coalesce into one.
    if (last - first > 1 &&
        !VirtualFree(reinterpret_cast<void*>(lo), hi - lo, MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS)) {
        Fail("coalescing placeholders", lo, hi - lo);
    }
    g_regions.erase(g_regions.begin() + first, g_regions.begin() + last);
    g_regions.insert(g_regions.begin() + first, Region{lo, hi, Kind::Placeholder, 0});
    size_t at = first;
    if (lo < a) {
        if (!VirtualFree(reinterpret_cast<void*>(lo), a - lo, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) {
            Fail("splitting a placeholder", lo, a - lo);
        }
        g_regions[at].end = a;
        g_regions.insert(g_regions.begin() + at + 1, Region{a, hi, Kind::Placeholder, 0});
        if (keep_left) {
            g_regions[at].kind = left.kind;
            g_regions[at].physical = left.physical;
            if (left.kind == Kind::View) {
                MapView(g_regions[at], PAGE_READWRITE);
            } else {
                MapPrivate(g_regions[at], PAGE_READWRITE);
            }
        }
        ++at;
    }
    if (b < hi) {
        if (!VirtualFree(reinterpret_cast<void*>(a), b - a, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) {
            Fail("splitting a placeholder", a, b - a);
        }
        g_regions[at].end = b;
        g_regions.insert(g_regions.begin() + at + 1, Region{b, hi, Kind::Placeholder, 0});
        if (keep_right) {
            Region& tail = g_regions[at + 1];
            tail.kind = right.kind;
            tail.physical = right.physical + (b - right.start);
            if (right.kind == Kind::View) {
                MapView(tail, PAGE_READWRITE);
            } else {
                MapPrivate(tail, PAGE_READWRITE);
            }
        }
    }
    // Private memory cut in two keeps nothing of its old contents (only the mirror uses it,
    // and there it only stands for unmapped memory: zeros).
    for (const ProtRun& run : runs) {
        DWORD old;
        if (run.protect != PAGE_READWRITE &&
            !VirtualProtect(reinterpret_cast<void*>(run.start), run.end - run.start, run.protect,
                            &old)) {
            Fail("restoring page protection", run.start, run.end - run.start);
        }
    }
    return at;
}

} // namespace

u64 PageSize() {
    return 4096;
}

void* Allocate(u64 size, int prot) {
    return VirtualAlloc(nullptr, size, MEM_RESERVE | MEM_COMMIT, WinProt(prot));
}

void* AllocateAt(u64 address, u64 size, int prot) {
    void* memory = VirtualAlloc(reinterpret_cast<void*>(address), size, MEM_RESERVE | MEM_COMMIT,
                                WinProt(prot));
    if (memory && memory != reinterpret_cast<void*>(address)) {
        VirtualFree(memory, 0, MEM_RELEASE);
        return nullptr;
    }
    return memory;
}

bool Protect(void* address, u64 size, int prot) {
    DWORD old;
    return VirtualProtect(address, size, WinProt(prot), &old) != 0;
}

void Free(void* address, u64) {
    VirtualFree(address, 0, MEM_RELEASE);
}

u8* CreatePhysical(u64 size) {
    // Executable: the game's code may live in direct memory too (it never does here, but a
    // view cannot be made executable later unless the section allows it).
    g_section = CreateFileMapping2(INVALID_HANDLE_VALUE, nullptr, FILE_MAP_ALL_ACCESS,
                                   PAGE_EXECUTE_READWRITE, SEC_COMMIT, size, nullptr, nullptr, 0);
    if (!g_section) {
        LOG_CRITICAL("Memory",
                     "cannot create {} MiB of game memory (error {}); is the page file large "
                     "enough?",
                     size >> 20, GetLastError());
        return nullptr;
    }
    g_backing = static_cast<u8*>(MapViewOfFile3(g_section, GetCurrentProcess(), nullptr, 0, size, 0,
                                                PAGE_READWRITE, nullptr, 0));
    return g_backing;
}

bool ReserveGuest(u64 start, u64 end) {
    if (!VirtualAlloc2(GetCurrentProcess(), reinterpret_cast<void*>(start), end - start,
                       MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, nullptr, 0)) {
        LOG_CRITICAL("Memory", "cannot reserve [{:#x}, {:#x}) for the game (error {})", start,
                     end, GetLastError());
        return false;
    }
    const size_t at = RegionIndex(start);
    g_regions.insert(g_regions.begin() + at, Region{start, end, Kind::Placeholder, 0});
    return true;
}

bool MapPhysical(u64 address, u64 size, u64 physical, int prot) {
    const size_t i = Carve(address, address + size);
    g_regions[i].kind = Kind::View;
    g_regions[i].physical = physical;
    MapView(g_regions[i], WinProt(prot));
    return true;
}

bool MapReserved(u64 address, u64 size) {
    Carve(address, address + size);
    return true;
}

bool MapZeros(u64 address, u64 size, int prot) {
    const size_t i = Carve(address, address + size);
    g_regions[i].kind = Kind::Private;
    MapPrivate(g_regions[i], WinProt(prot));
    return true;
}

bool Unmap(u64 address, u64 size) {
    Carve(address, address + size);
    return true;
}

bool ProtectGuest(u64 address, u64 size, int prot) {
    // VirtualProtect cannot cross from one view to the next: one call per region.
    const DWORD protect = WinProt(prot);
    const u64 end = address + size;
    bool ok = true;
    for (size_t i = RegionIndex(address); i < g_regions.size() && g_regions[i].start < end; ++i) {
        const Region& r = g_regions[i];
        if (r.kind == Kind::Placeholder) {
            ok = false;
            continue;
        }
        const u64 a = std::max(r.start, address);
        const u64 b = std::min(r.end, end);
        DWORD old;
        if (!VirtualProtect(reinterpret_cast<void*>(a), b - a, protect, &old)) {
            ok = false;
        }
    }
    return ok;
}

bool IsFree(u64 address, u64 size) {
    // Inside our reservations nothing but our own mappings can be.
    return InsideReservation(address, address + size);
}

void Discard(u64 physical, u64 size) {
    // A section has no "punch hole": the pages stay committed. Released memory must read as
    // zeros when it is handed out again, as the memfd's punched holes do on Linux.
    std::memset(g_backing + physical, 0, size);
}

} // namespace host_memory

#endif // _WIN32
