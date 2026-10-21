// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <span>

#include <condition_variable>
#include <deque>
#include <mutex>
#include <shared_mutex>
#include <vector>
#include <boost/container/small_vector.hpp>

#include "common/interval_set.h"
#include "common/types.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/fault_manager.h"
#include "video_core/buffer_cache/range_set.h"
#include "video_core/renderer_vulkan/vk_staging_buffer_pool.h"
#include "video_core/renderer_vulkan/vk_semaphore.h"

namespace AmdGpu {
struct Liverpool;
}

namespace Core {
class MemoryManager;
}

namespace Vulkan {
class GraphicsPipeline;
struct SubmitInfo;
class Runtime;
class StagingBufferPool;
} // namespace Vulkan

namespace VideoCore {

class TextureCache;
class MemoryTracker;
class PageManager;

class BufferCache {
    static constexpr u64 ADDRESS_SPACE_BITS = 40;
    static constexpr u64 ARENA_PAGE_BITS = 32;
    static constexpr u64 ARENA_PAGE_SIZE = u64{1} << ARENA_PAGE_BITS;
    static constexpr u64 NUM_ARENA_PAGES = u64{1} << (ADDRESS_SPACE_BITS - ARENA_PAGE_BITS);
    static constexpr u64 MIN_BLOCK_SIZE = 16_KB;
    // sotcport: 16 KiB upstream. Measured on 2026-10-11 with the frame rate uncapped
    // (tools/bench_game.sh, 3 alternating runs each): 64 KiB 36.5 -> 29.5 FPS (copying bigger
    // buffers on every bind costs more than tracking their pages), 8 KiB 36.2 -> 38.3 FPS,
    // 4 KiB the same as 8 KiB.
    static constexpr u64 STREAM_THRESHOLD = 8_KB;

public:
    explicit BufferCache(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                         Vulkan::Runtime& runtime, AmdGpu::Liverpool* liverpool,
                         TextureCache& texture_cache, PageManager& tracker);
    ~BufferCache();

    /// Returns a pointer to GDS device local buffer.
    [[nodiscard]] const Buffer* GetGdsBuffer() const noexcept {
        return &gds_buffer;
    }

    /// Retrieves the device local DBA page table buffer.
    [[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept {
        return bda_pagetable_buffer.get();
    }

    /// Retrieves the fault buffer.
    [[nodiscard]] Buffer* GetFaultBuffer() noexcept {
        return fault_manager->GetFaultBuffer();
    }

    /// Retrieves the stream buffer.
    [[nodiscard]] StreamBuffer& GetStreamBuffer() noexcept {
        return stream_buffer;
    }

    /// Returns minimum granularity of a sparse memory bind.
    u32 GetSparsePageShift() const noexcept {
        return block_shift;
    }

    void TickFrame();

    /// Invalidates any buffer in the logical page range.
    void InvalidateMemory(VAddr device_addr, u64 size, bool assume_locks = false);

    /// Flushes any GPU modified buffer in the logical page range back to CPU memory.
    void ReadMemory(VAddr device_addr, u64 size, bool is_write = false, bool assume_locks = false);

    /// sotcport: after a CPU write fault at device_addr, stops watching the rest of the
    /// surrounding window where that is safe (RegionManager::ExtendWriteFault).
    /// SOTC_FAULT_WINDOW=<KiB> turns it on (off by default: see the .cpp for why).
    void ExtendWriteFault(VAddr device_addr);

    /// Finds a buffer for the specified region.
    /// sotcport: `shadowed` (with is_written): the CPU already holds what the GPU will write
    /// there (Rasterizer::ShadowCopyShader), so the write is not tracked as GPU-modified.
    [[nodiscard]] std::pair<const Buffer*, u64> ObtainBuffer(VAddr device_addr, u32 size,
                                                             bool is_written,
                                                             bool is_texel_buffer = false,
                                                             bool shadowed = false);

    /// sotcport: CPU memory at [addr, +size) now holds what the GPU's copy will hold once the
    /// work recorded so far has run (the GPU command thread wrote it there itself): those bytes
    /// stop counting as GPU-modified. Waits for asynchronous readbacks still bringing older
    /// bytes back there. GPU command thread, with nothing queued for stage B that writes there.
    void ForgetGpuWrites(VAddr addr, u64 size);

    /// Attempts to obtain a buffer without modifying the cache contents.
    [[nodiscard]] std::pair<const Buffer*, u64> ObtainBufferForImage(VAddr device_addr, u32 size);

    /// Return true when a region is modified from the CPU
    [[nodiscard]] bool IsRegionCpuModified(VAddr addr, size_t size);

    /// Return true when a region is modified from the GPU
    [[nodiscard]] bool IsRegionGpuModified(VAddr addr, size_t size);

    /// sotcport: changes whenever memory may have become CPU-modified (MemoryTracker::CpuMarks).
    [[nodiscard]] u64 CpuMarks() const;
    /// sotcport: changes whenever GPU-written ranges are taken back (see gpu_modified_removals).
    [[nodiscard]] u64 GpuModifiedRemovals() const noexcept {
        return gpu_modified_removals.load(std::memory_order_acquire);
    }

    /// sotcport (diagnostics): copies what the GPU's arena holds for [device_addr, +size) into
    /// `out`, changing no state (waits for the GPU). False when it is not all resident.
    bool PeekGpuMemory(VAddr device_addr, u64 size, u8* out);

    /// sotcport: whether the GPU wrote these exact bytes since they were last downloaded
    /// (IsRegionGpuModified answers for whole tracking pages). GPU command thread: with the
    /// two-stage draw pipeline it asks while stage B may add ranges (gpu_modified_mutex).
    [[nodiscard]] bool AreBytesGpuModified(VAddr addr, size_t size) const {
        std::shared_lock lock{gpu_modified_mutex};
        return gpu_modified_ranges.Intersects(addr, size) || readback_ranges.Intersects(addr, size);
    }

    /// Synchronizes all buffers needed for DMA.
    void SynchronizeDmaBuffers();

    /// Commits pending sparse buffer memory binds. Must be called before every scheduler submit.
    void SubmitPendingArenaBinds(Vulkan::SubmitInfo& info);

private:
    struct ArenaBinds {
        const Buffer* arena;
        boost::container::small_vector<vk::SparseMemoryBind, 32> binds;
    };

    ArenaBinds* BindsForArena(const Buffer* arena) {
        auto it = std::ranges::find(pending_binds, arena, &ArenaBinds::arena);
        if (it != pending_binds.end()) {
            return std::addressof(*it);
        }
        return &pending_binds.emplace_back(arena);
    }

    const Buffer* GetArena(u64 first_block, u64 last_block);

    void EnsureResident(const Buffer* arena, u64 first_block, u64 last_block);

    void DownloadMemory(const Buffer* arena, VAddr device_addr, u64 size);

    // sotcport: downloads several windows with a single wait for the GPU (see ReadMemory).
    void DownloadWindows(std::span<const std::pair<VAddr, u64>> windows);

    /// sotcport: one window of a download: its arena, its range clamped to the arena, and its
    /// copies (a slice of the download's copy list).
    struct DownloadWindow {
        const Buffer* arena;
        VAddr start;
        u64 size;
        size_t first_copy;
        size_t num_copies;
    };
    using DownloadWindowList = boost::container::small_vector<DownloadWindow, 33>;
    using DownloadCopyList = boost::container::small_vector<vk::BufferCopy, 64>;
    /// Takes the GPU-written ranges of the windows out of gpu_modified_ranges and lists the copies
    /// that bring them back (destination offsets from 0). Caller holds gpu_modified_mutex.
    u64 CollectDownloads(std::span<const std::pair<VAddr, u64>> windows, DownloadWindowList& out,
                         DownloadCopyList& copies);
    /// Clears the "GPU modified" page state (and its read protection) of [start, +size), except
    /// on pages that still have GPU data pending (gpu_modified_ranges) or on the way back
    /// (readback_ranges). Caller holds gpu_modified_mutex.
    void UnmarkDownloaded(VAddr start, u64 size);
    /// The download windows of a readback of [device_addr, +size) (see ReadMemory).
    boost::container::small_vector<std::pair<VAddr, u64>, 33> ReadbackWindows(VAddr device_addr,
                                                                              u64 size);

    /// sotcport: asynchronous readbacks (SOTC_ASYNC_READBACK=1). A guest thread's readback used to
    /// run entirely on the GPU command thread, which then waited for the GPU (~0.5 ms each,
    /// hundreds per second) with all command processing stopped. Now the command thread only
    /// records and submits the copies (PrepareReadback); the guest thread that faulted waits for
    /// the GPU, writes the data to guest memory and clears the page state (CompleteReadback).
    /// Readbacks complete in the order they were prepared (`sequence`), so older data never
    /// overwrites newer; bytes on the way are in readback_ranges, so nobody takes them for current.
    struct AsyncReadback {
        bool empty = true;           ///< nothing to download
        u64 wait_for = ~u64{0};      ///< (empty) an earlier readback bringing these bytes back
        u64 sequence = 0;
        u64 tick = 0;                ///< GPU tick after which the copies are done
        Vulkan::StagingBufferRef staging{};
        DownloadWindowList windows;
        DownloadCopyList copies;
    };
    AsyncReadback PrepareReadback(VAddr device_addr, u64 size);
    void CompleteReadback(AsyncReadback& readback, VAddr device_addr, u64 size, bool is_write);
    /// Waits until no asynchronous readback is bringing back bytes of [addr, +size).
    void WaitForReadbacks(VAddr addr, u64 size);
    RangeSet readback_ranges;     ///< bytes on the way back (under gpu_modified_mutex)
    std::mutex readback_mutex;    ///< guards the fields below
    std::condition_variable readback_cv;
    u64 readback_next = 0;        ///< next sequence to hand out (GPU command thread)
    u64 readback_completed = 0;   ///< every sequence below this one has completed
    struct InflightReadback {
        u64 sequence;
        VAddr start, end; ///< span of its windows
    };
    std::vector<InflightReadback> readback_inflight;
    std::array<VAddr, 32> hot_windows{}; // where readbacks happened lately (window starts)
    u32 hot_windows_next = 0;

    /// sotcport: a read of guest memory on its way to the GPU (see
    /// Rasterizer::RewindShadowWrites).
    void RewindShadowWrites(VAddr address, u8* data, u64 size);
    bool SynchronizeMemory(const Buffer* arena, VAddr device_addr, u32 size, bool is_written,
                           bool is_texel_buffer);

    bool SynchronizeMemoryFromImage(const Buffer* arena, VAddr device_addr, u32 size);

    const Vulkan::Instance& instance;
    Vulkan::Scheduler& scheduler;
    Vulkan::Runtime& runtime;
    Vulkan::StagingBufferPool& staging_pool;
    AmdGpu::Liverpool* liverpool;
    Core::MemoryManager* memory;
    TextureCache& texture_cache;
    std::unique_ptr<MemoryTracker> memory_tracker;

    StreamBuffer stream_buffer;
    Buffer gds_buffer;
    RangeSet gpu_modified_ranges;
    /// sotcport (two-stage draw pipeline): guards gpu_modified_ranges, so the GPU command thread
    /// can check SRT walker reads without waiting for stage B. Downloads hold it from taking the
    /// ranges out until the data is in guest memory: "not modified" then always means the
    /// memory is current.
    mutable std::shared_mutex gpu_modified_mutex;
    /// sotcport: bumped (under gpu_modified_mutex) whenever ranges leave gpu_modified_ranges:
    /// while it stays the same, a range found there is still there (ObtainBuffer).
    std::atomic<u64> gpu_modified_removals{0};

    std::unique_ptr<FaultManager> fault_manager;
    std::unique_ptr<Buffer> bda_pagetable_buffer;
    bool fault_process_pending{};
    u64 dma_synced_marks = 0; ///< sotcport: MemoryTracker::CpuMarks at the last DMA sync

    std::array<const Buffer*, NUM_ARENA_PAGES> address_space{};
    std::deque<Buffer> arenas;
    std::vector<ArenaBinds> pending_binds;
    Vulkan::Semaphore memory_semaphore;

    struct Backing : public Interval {
        vk::DeviceMemory memory;
        u64 offset;
        constexpr bool CanMergeWith(const Backing& other) const noexcept {
            return memory == other.memory && offset + (end - start) == other.offset;
        }
        constexpr Backing SubRange(u64 a, u64 b) const noexcept {
            return {{a, b}, memory, offset + (a - start)};
        }
    };
    IntervalList<Backing> resident_ranges;

    u32 arena_memory_type_index{};
    u32 block_size{};
    u32 block_shift{};
    u32 blocks_per_arena_page{};
    u32 blocks_per_arena_page_shift{};
};

} // namespace VideoCore
