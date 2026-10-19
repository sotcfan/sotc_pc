// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <immintrin.h>
#include <thread>
#include <utility>
#include "common/adaptive_mutex.h"
#include "common/alignment.h"
#include "common/assert.h"
#include "common/debug.h"
#include "common/div_ceil.h"
#include "common/error.h"
#include "common/multi_level_page_table.h"
#include "common/signal_context.h"
#include "common/thread.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "core/signals.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "perf_temp.h" // sotcport TEMP
#include <atomic>
#include <cstdlib>
#include <cstring>

#ifndef _WIN64
#include <sys/mman.h>
#include "common/adaptive_mutex.h"
#else
#include <windows.h>
#endif

#ifdef __linux__
#include <thread>
#include <fcntl.h>
#include <linux/userfaultfd.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#else
#include "common/spin_lock.h"
#endif

namespace VideoCore {
struct PageManager::Impl {
    struct PageState {
        u8 num_write_watchers;
        u8 num_read_watchers;

        Core::MemoryPermission WritePerm() const noexcept {
            return num_write_watchers == 0 ? Core::MemoryPermission::Write
                                           : Core::MemoryPermission::None;
        }

        Core::MemoryPermission ReadPerm() const noexcept {
            return num_read_watchers == 0 ? Core::MemoryPermission::Read
                                          : Core::MemoryPermission::None;
        }

        Core::MemoryPermission Perms() const noexcept {
            return ReadPerm() | WritePerm();
        }

        template <bool is_read>
        u8 GetPage() const {
            if constexpr (is_read) {
                return num_read_watchers;
            } else {
                return num_write_watchers;
            }
        }

        constexpr Core::MemoryPermission Update(PageOp write_op, bool update_write = true,
                                                PageOp read_op = PageOp::None,
                                                bool update_read = false) {
            if (update_read) {
                if (read_op == PageOp::Track) {
                    ASSERT_MSG(num_read_watchers == 0, "Too many watchers");
                } else if (read_op == PageOp::Untrack) {
                    ASSERT_MSG(num_read_watchers > 0, "Not enough watchers");
                }
                num_read_watchers += std::to_underlying(read_op);
            }
            if (update_write) {
                if (write_op == PageOp::Track) {
                    ASSERT_MSG(num_write_watchers < 255, "Too many watchers");
                } else if (write_op == PageOp::Untrack) {
                    ASSERT_MSG(num_write_watchers > 0, "Not enough watchers");
                }
                num_write_watchers += std::to_underlying(write_op);
            }
            return Perms();
        }
    };

    static constexpr size_t ADDRESS_BITS = 40;
    static constexpr size_t NUM_ADDRESS_PAGES = 1ULL << (40 - PM_PAGE_BITS);
    static constexpr size_t NUM_ADDRESS_LOCKS = NUM_ADDRESS_PAGES / NUM_REGION_PAGES;
    inline static Vulkan::Rasterizer* rasterizer;

    Impl() = default;
    virtual ~Impl() = default;

    virtual void OnMap(VAddr address, size_t size) {
        EnsurePages(address, address + size);
        ReapplyProtection(address, size);
    }

    // sotcport: gives the pages of [address, +size) that something still watches the host
    // protection their state says they have. A new mapping (or the game's sceKernelMprotect)
    // replaces the host protection of its pages, and pages can be watched while nothing is
    // mapped there: an image spanning a streamed texture's unmapped mips is tracked whole. Such
    // pages came back writable with their state still saying "protected", so a later Track
    // changed no protection and the game's writes there went unnoticed: textures kept old
    // contents (colored blocks, 2026-10-14). The page locks are held until the protection is
    // applied, as in UpdatePageWatchers, so no other change can land in between.
    // SOTC_NO_REPROTECT=1 turns this off (A/B).
    void ReapplyProtection(VAddr address, size_t size) {
        static const bool disabled = std::getenv("SOTC_NO_REPROTECT") != nullptr;
        if (disabled || size == 0) {
            return;
        }
        const u64 page_start = address >> PM_PAGE_BITS;
        const u64 page_end = Common::DivCeil(address + size, PM_PAGE_SIZE);
        u64 range_begin = 0;
        u64 range_pages = 0;
        Core::MemoryPermission range_perms{};
        u64 reprotected = 0;
        const auto flush = [&] {
            if (range_pages > 0) {
                Protect(range_begin << PM_PAGE_BITS, range_pages << PM_PAGE_BITS, range_perms);
                reprotected += range_pages;
                range_pages = 0;
            }
        };
        for (u64 page = page_start; page != page_end; ++page) {
            PageState* state = cached_pages.find(page);
            if (!state) {
                flush();
                continue;
            }
            locks[page].lock();
            if (state->num_write_watchers == 0 && state->num_read_watchers == 0) {
                flush();
                continue;
            }
            const auto perms = state->Perms();
            if (range_pages > 0 && perms != range_perms) {
                flush();
            }
            if (range_pages == 0) {
                range_begin = page;
                range_perms = perms;
            }
            ++range_pages;
        }
        flush();
        for (u64 page = page_start; page != page_end; ++page) {
            if (auto* lock = locks.find(page)) {
                lock->unlock();
            }
        }
        // TEMP: how often it finds watched pages (SOTC_PERF).
        if (reprotected && perf::Enabled()) {
            static std::atomic<u64> total{0};
            static std::atomic<u32> calls{0};
            total += reprotected;
            if (const u32 n = ++calls; n <= 10 || n % 100 == 0) {
                LOG_INFO(Render, "protection reapplied to {} watched pages of {:#x}-{:#x} "
                                 "({} calls, {} pages so far)",
                         reprotected, address, address + size, n, total.load());
            }
        }
    }

    virtual void OnUnmap(VAddr address, size_t size) {
        // No-op
    }

    virtual void Protect(VAddr address, size_t size, Core::MemoryPermission perms) = 0;

    void EnsurePages(VAddr begin, VAddr end) {
        end = std::min(end, VAddr{1} << ADDRESS_BITS) - 1;
        const size_t start_page = begin >> PM_PAGE_BITS;
        const size_t end_page = end >> PM_PAGE_BITS;
        cached_pages.reserve(start_page, end_page);
        locks.reserve(start_page, end_page);
    }

    void UpdatePageWatchers(VAddr addr, u64 size, PageOp write_op) {
        const u64 page_start = addr >> PM_PAGE_BITS;
        const u64 page_end = Common::DivCeil(addr + size, PM_PAGE_SIZE);

        Core::MemoryPermission perms{};
        u64 range_begin = page_start;
        u64 range_pages = 0;
        u64 potential_pages = 0;

        const auto release_pending = [&] {
            if (range_pages > 0) {
                Protect(range_begin << PM_PAGE_BITS, range_pages << PM_PAGE_BITS, perms);
                range_pages = 0;
                potential_pages = 0;
            }
        };

        // Iterate requested pages
        const u64 aligned_addr = page_start << PM_PAGE_BITS;
        const u64 aligned_end = page_end << PM_PAGE_BITS;
        if (!rasterizer->IsMapped(aligned_addr, aligned_end - aligned_addr)) {
            LOG_WARNING(Render,
                        "Tracking memory region {:#x} - {:#x} which is not fully GPU mapped.",
                        aligned_addr, aligned_end);
            EnsurePages(aligned_addr, aligned_end);
        }

        for (u64 page = page_start; page != page_end; ++page) {
            PageState* state = cached_pages.find(page);
            if (!state) {
                continue;
            }

            locks[page].lock();

            const auto old_perms = state->Perms();
            if (page == page_start) {
                perms = old_perms;
            }

            // Apply the change to the page state
            const auto new_perms = state->Update(write_op);
            if (new_perms != perms) [[unlikely]] {
                // If the protection changed add pending (un)protect action
                release_pending();
                perms = new_perms;
            } else if (range_pages != 0) {
                ++potential_pages;
            }

            // If the page must be (un)protected
            if (new_perms != old_perms) {
                if (range_pages == 0) {
                    // Start a new potential range
                    range_begin = page;
                    potential_pages = 1;
                }
                // Extend current range up to potential range
                range_pages = potential_pages;
            }
        }

        // Add pending (un)protect action
        release_pending();

        for (u64 page = page_start; page != page_end; ++page) {
            if (auto* lock = locks.find(page)) {
                lock->unlock();
            }
        }
    }

    void UpdatePageWatchersForRegion(VAddr base_addr, const Bounds& bounds,
                                     const RegionBits& write_mask, const RegionBits& read_mask,
                                     PageOp write_op, PageOp read_op) {
        const u64 page_start = bounds.start_word * PAGES_PER_WORD + bounds.start_page;
        const u64 page_end = bounds.end_word * PAGES_PER_WORD + bounds.end_page + 1;
        // sotcport: only the stretches of pages whose watchers change (a mask bit set), each
        // with the pages around it up to MaxGap unchanged pages (so their protection can still
        // be applied in one call, as before). The bounds span from the first to the last changed
        // page of a 16 MiB region: after a DMA sync (BufferCache::SynchronizeDmaBuffers) the
        // pages the game wrote are spread over it, and every page in between was locked,
        // looked at and unlocked (~3% of the draw recording thread, 2026-10-19).
        constexpr u64 MaxGap = 16;
        const auto changes = [&](u64 word) {
            u64 bits = 0;
            if (write_op != PageOp::None) {
                bits |= write_mask[word];
            }
            if (read_op != PageOp::None) {
                bits |= read_mask[word];
            }
            return bits;
        };
        // The next page from `page` on whose watchers change, or page_end.
        const auto next_change = [&](u64 page) {
            while (page < page_end) {
                const u64 bits = changes(page / PAGES_PER_WORD) >> (page % PAGES_PER_WORD);
                if (bits != 0) {
                    return std::min(page + std::countr_zero(bits), page_end);
                }
                page = Common::AlignUp(page + 1, PAGES_PER_WORD);
            }
            return page_end;
        };
        // The end of the stretch starting at `page` (a changed page): past its last changed page
        // that is followed by more than MaxGap unchanged ones.
        const auto stretch_end = [&](u64 page) {
            u64 last = page;
            for (u64 next = next_change(page + 1); next < page_end && next - last <= MaxGap;
                 next = next_change(next + 1)) {
                last = next;
            }
            return last + 1;
        };
        for (u64 first = next_change(page_start); first < page_end;) {
            const u64 last = stretch_end(first);
            UpdatePageWatchersForStretch(base_addr, first, last, write_mask, read_mask, write_op,
                                         read_op);
            first = next_change(last);
        }
    }

    // The original UpdatePageWatchersForRegion, for the pages [page_start, page_end) of the
    // region at base_addr: their locks are held until their protection is applied.
    void UpdatePageWatchersForStretch(VAddr base_addr, u64 page_start, u64 page_end,
                                      const RegionBits& write_mask, const RegionBits& read_mask,
                                      PageOp write_op, PageOp read_op) {
        const u64 base_page = base_addr >> PM_PAGE_BITS;

        Core::MemoryPermission perms{};
        u64 range_begin = base_page + page_start;
        u64 range_pages = 0;
        u64 potential_pages = 0;

        const auto release_pending = [&] {
            if (range_pages > 0) {
                Protect(range_begin << PM_PAGE_BITS, range_pages << PM_PAGE_BITS, perms);
                range_pages = 0;
                potential_pages = 0;
            }
        };

        for (u64 page = page_start; page != page_end; ++page) {
            PageState* state = cached_pages.find(base_page + page);
            if (!state) {
                continue;
            }

            locks[base_page + page].lock();

            const auto old_perms = state->Perms();
            if (page == page_start) {
                perms = old_perms;
            }

            // Apply the change to the page state
            const bool update_write = write_op != PageOp::None && write_mask.GetPage(page);
            const bool update_read = read_op != PageOp::None && read_mask.GetPage(page);
            const auto new_perms = state->Update(write_op, update_write, read_op, update_read);

            if (new_perms != perms) [[unlikely]] {
                // If the protection changed add pending (un)protect action
                release_pending();
                perms = new_perms;
            } else if (range_pages != 0) {
                // If the protection did not change, extend the potential range
                ++potential_pages;
            }

            // If the page must be (un)protected
            if (new_perms != old_perms) {
                if (range_pages == 0) {
                    // Start a new potential range
                    range_begin = base_page + page;
                    potential_pages = 1;
                }
                // Extend current rango up to potential range
                range_pages = potential_pages;
            }
        }

        // Add pending (un)protect action
        release_pending();

        for (u64 page = page_start; page != page_end; ++page) {
            if (auto* lock = locks.find(base_page + page)) {
                lock->unlock();
            }
        }
    }

    struct PageTraits {
        using Entry = PageState;
        static constexpr size_t ADDRESS_SPACE_BITS = ADDRESS_BITS;
        static constexpr size_t L1_BITS = 16;
        static constexpr size_t PAGE_BITS = PM_PAGE_BITS;
        static constexpr bool NULL_CHECK = false;
    };
    Common::MultiLevelPageTable<PageTraits> cached_pages;
    // sotcport: one byte per page, taken and released inline. The page locks are taken for
    // every page of every range whose watchers change (UpdatePageWatchersForRegion: the draw
    // recording thread re-protects the pages it uploads, often hundreds at a time), and a
    // pthread adaptive mutex there meant two calls into libc per page and 40 bytes of lock per
    // 4 KiB page (~4% of the draw recording thread, 2026-10-19). Held only for the state update
    // and the mprotect that follows it, so waiting by spinning is enough; a holder that lost its
    // CPU is waited for with yields.
    struct PageLock {
        std::atomic<u8> held; // zero-initialized with the table's pages

        void lock() noexcept {
            while (held.exchange(1, std::memory_order_acquire) != 0) {
                for (u32 spins = 0; held.load(std::memory_order_relaxed) != 0; ++spins) {
                    if (spins < 256) {
                        _mm_pause();
                    } else {
                        std::this_thread::yield();
                    }
                }
            }
        }
        void unlock() noexcept {
            held.store(0, std::memory_order_release);
        }
    };
    struct MutexTraits {
        using Entry = PageLock;
        static constexpr size_t ADDRESS_SPACE_BITS = ADDRESS_BITS;
        static constexpr size_t L1_BITS = 16;
        static constexpr size_t PAGE_BITS = PM_PAGE_BITS;
        static constexpr bool NULL_CHECK = false;
    };
    Common::MultiLevelPageTable<MutexTraits> locks;
};

#ifdef __linux__
struct UffdImpl : public PageManager::Impl {
private:
    std::jthread ufd_thread;
    int uffd;

public:
    UffdImpl(Vulkan::Rasterizer* rasterizer_) : Impl() {
        rasterizer = rasterizer_;
        uffd = syscall(SYS_userfaultfd, O_CLOEXEC | O_NONBLOCK | UFFD_USER_MODE_ONLY);
        if (uffd == -1) {
            LOG_ERROR(Common_Memory,
                      "userfaultfd syscall failed: {}, falling back to signal implementation",
                      Common::GetLastErrorMsg());
            throw std::runtime_error("userfaultfd");
        }

        // Request uffdio features from kernel.
        uffdio_api api;
        api.api = UFFD_API;
        api.features = UFFD_FEATURE_THREAD_ID;
        const int ret = ioctl(uffd, UFFDIO_API, &api);
        if (ret != 0) {
            LOG_ERROR(Common_Memory,
                      "uffdio_api call failed: {}, falling back to signal implementation",
                      Common::GetLastErrorMsg());
            throw std::runtime_error("uffdio_api");
        }

        // Create uffd handler thread
        ufd_thread = std::jthread([&](std::stop_token token) { UffdHandler(token); });
    }

    ~UffdImpl() = default;

    void OnMap(VAddr address, size_t size) override {
        PageManager::Impl::OnMap(address, size);
        uffdio_register reg;
        reg.range.start = address;
        reg.range.len = size;
        reg.mode = UFFDIO_REGISTER_MODE_WP;
        const int ret = ioctl(uffd, UFFDIO_REGISTER, &reg);
        ASSERT_MSG(ret != -1, "Uffdio register failed with error: {}", Common::GetLastErrorMsg());
    }

    void OnUnmap(VAddr address, size_t size) override {
        uffdio_range range;
        range.start = address;
        range.len = size;
        const int ret = ioctl(uffd, UFFDIO_UNREGISTER, &range);
        ASSERT_MSG(ret != -1, "Uffdio unregister failed with error: {}", Common::GetLastErrorMsg());
    }

    void Protect(VAddr address, size_t size, Core::MemoryPermission perms) override {
        bool allow_write = True(perms & Core::MemoryPermission::Write);
        uffdio_writeprotect wp;
        wp.range.start = address;
        wp.range.len = size;
        wp.mode = allow_write ? UFFDIO_WRITEPROTECT_MODE_DONTWAKE : UFFDIO_WRITEPROTECT_MODE_WP;
        const int ret = ioctl(uffd, UFFDIO_WRITEPROTECT, &wp);
        ASSERT_MSG(ret != -1, "Uffdio writeprotect failed with error: {}",
                   Common::GetLastErrorMsg());
    }

    void UffdHandler(std::stop_token token) {
        Common::SetCurrentThreadName("shadPS4:Uffd");

        auto regions = Core::Memory::Instance()->GetAddressSpace().GetUsableRegions();
        for (auto& region : regions) {
            OnMap(region.lower(), region.upper());
        }
        LOG_INFO(Common_Memory, "registered reserved memory with userfaultfd");

        while (!token.stop_requested()) {
            pollfd pollfd;
            pollfd.fd = uffd;
            pollfd.events = POLLIN;

            // Block until the descriptor is ready for data reads.
            const int pollres = poll(&pollfd, 1, -1);
            switch (pollres) {
            case -1:
                perror("Poll userfaultfd");
                continue;
                break;
            case 0:
                continue;
            case 1:
                break;
            default:
                UNREACHABLE_MSG("Unexpected number of descriptors {} out of poll", pollres);
            }

            // We don't want an error condition to have occured.
            ASSERT_MSG(!(pollfd.revents & POLLERR), "POLLERR on userfaultfd");

            // We waited until there is data to read, we don't care about anything else.
            if (!(pollfd.revents & POLLIN)) {
                continue;
            }

            // Read message from kernel.
            uffd_msg msg;
            const int readret = read(uffd, &msg, sizeof(msg));
            if (readret == -1) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    continue;
                }
                LOG_ERROR(Common_Memory, "Unexpected result of uffd read: {}",
                          Common::GetLastErrorMsg());
                break;
            }
            ASSERT_MSG(readret == sizeof(msg), "Unexpected short read, exiting");
            ASSERT(msg.arg.pagefault.flags & UFFD_PAGEFAULT_FLAG_WP);

            // Notify rasterizer about the fault.
            const VAddr addr = msg.arg.pagefault.address;
            const auto ptid = msg.arg.pagefault.feat.ptid;
            rasterizer->InvalidateMemory(addr, 1,
                                         ptid == rasterizer->GetGpuCommandProcessorThreadId());

            // Some calls to InvalidateMemory never reach the UFFDIO_WRITEPROTECT ioctl in
            // ::Protect, therefore we use MODE_DONTWAKE and wake the thread with UFFDIO_WAKE here
            uffdio_range wake;
            wake.start = msg.arg.pagefault.address;
            wake.len = PageManager::PM_PAGE_SIZE;
            const int ret = ioctl(uffd, UFFDIO_WAKE, &wake);
            ASSERT_MSG(ret != -1, "Waking thread {} failed with: {}", ptid,
                       Common::GetLastErrorMsg());
        }
    }
};
#endif // __linux__

struct SignalImpl : public PageManager::Impl {
    SignalImpl(Vulkan::Rasterizer* rasterizer_) : Impl() {
        rasterizer = rasterizer_;

        // Should be called first.
        constexpr auto priority = std::numeric_limits<u32>::min();
        Core::Signals::Instance()->RegisterAccessViolationHandler(GuestFaultSignalHandler,
                                                                  priority);
    }

    void Protect(VAddr address, size_t size, Core::MemoryPermission perms) override {
        RENDERER_TRACE;
        auto* memory = Core::Memory::Instance();
        auto& impl = memory->GetAddressSpace();
        ASSERT_MSG(perms != Core::MemoryPermission::Write,
                   "Attempted to protect region as write-only which is not a valid permission");
        impl.Protect(address, size, perms);
    }

    // sotcport TEMP (SOTC_PERF): the code address behind a fault, for attributing readbacks
    // (only the faulting instruction: following frame pointers here could fault on the game's
    // small fiber stacks).
    static u64 GameFaultSite(void* context) {
        return reinterpret_cast<u64>(Common::GetRip(context));
    }

    static bool GuestFaultSignalHandler(void* context, void* fault_address) {
        const auto addr = reinterpret_cast<VAddr>(fault_address);
        if (perf::Enabled()) {
            perf::SetFaultSite(GameFaultSite(context)); // sotcport TEMP
        }
        const auto size = std::min<u64>(8, PageManager::GetNextPageAddr(addr) - addr);
        // sotcport: with the two-stage draw pipeline the draw recording thread owns the caches
        // too: its faults are handled inline like the GPU command thread's (which first waits
        // for that thread to run dry, in InvalidateMemory/ReadMemory).
        const auto is_gpu_thread = rasterizer->IsGpuSideThread();
        if (Common::IsWriteError(context)) {
            if (!rasterizer->InvalidateMemory(addr, size, is_gpu_thread)) {
                return false;
            }
            rasterizer->ExtendWriteFault(addr); // sotcport
            return true;
        } else {
            return rasterizer->ReadMemory(addr, size, is_gpu_thread);
        }
        return false;
    }
};

PageManager::PageManager(Vulkan::Rasterizer* rasterizer_) {
#ifdef __linux__
    if (EmulatorSettings.IsUserfaultfdTracking()) {
        try {
            impl = std::make_unique<UffdImpl>(rasterizer_);
            LOG_INFO(Config, "Memory tracking method: userfaultfd");
            return;
        } catch (const std::runtime_error& e) {
            // if uffd is unsupported, falls back to SignalImpl
        }
    }
    LOG_INFO(Config, "Memory tracking method: signals");
#endif
    impl = std::make_unique<SignalImpl>(rasterizer_);
}

PageManager::~PageManager() = default;

void PageManager::OnGpuMap(VAddr address, size_t size) {
    impl->OnMap(address, size);
}

void PageManager::OnGpuUnmap(VAddr address, size_t size) {
    impl->OnUnmap(address, size);
}

void PageManager::UpdatePageWatchers(VAddr addr, u64 size, PageOp write_op) const {
    impl->UpdatePageWatchers(addr, size, write_op);
}

void PageManager::UpdatePageWatchersForRegion(VAddr base_addr, const Bounds& bounds,
                                              const RegionBits& write_mask,
                                              const RegionBits& read_mask, PageOp write_op,
                                              PageOp read_op) const {
    impl->UpdatePageWatchersForRegion(base_addr, bounds, write_mask, read_mask, write_op, read_op);
}

} // namespace VideoCore
