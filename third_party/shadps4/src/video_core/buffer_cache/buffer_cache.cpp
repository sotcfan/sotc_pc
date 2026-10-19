// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <bit>
#include <magic_enum/magic_enum.hpp>

#include "common/alignment.h"
#include "core/debug_state.h"
#include "core/memory.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/memory_tracker.h"
#include "video_core/buffer_cache/region_definitions.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h" // sotcport: RunOnStageB
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"

#include <vk_mem_alloc.h>
#include "perf_temp.h" // sotcport TEMP
#include "diag/memory_types.h"

namespace VideoCore {

static constexpr size_t GDS_BUFFER_SIZE = 64_KB;
static constexpr size_t STREAM_BUFFER_SIZE = 128_MB;

static constexpr auto ARENA_USAGE =
    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst |
    vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
    vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eVertexBuffer |
    vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress;

std::optional<u32> FindMemoryType(const vk::PhysicalDeviceMemoryProperties& properties,
                                  vk::MemoryPropertyFlags wanted, u32 memory_type_bits) {
    for (u32 i = 0; i < properties.memoryTypeCount; ++i) {
        if (((memory_type_bits >> i) & 1) == 0) {
            continue;
        }
        const auto flags = properties.memoryTypes[i].propertyFlags;
        if ((flags & wanted) == wanted) {
            return i;
        }
    }
    return std::nullopt;
}

BufferCache::BufferCache(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                         Vulkan::Runtime& runtime_, AmdGpu::Liverpool* liverpool_,
                         TextureCache& texture_cache_, PageManager& tracker)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_},
      staging_pool{runtime_.GetStagingPool()}, liverpool{liverpool_},
      memory{Core::Memory::Instance()}, texture_cache{texture_cache_},
      memory_tracker{std::make_unique<MemoryTracker>(tracker)},
      stream_buffer{instance, scheduler, MemoryType::Stream, STREAM_BUFFER_SIZE},
      gds_buffer{instance, 0, GDS_BUFFER_SIZE, MemoryType::Stream, "GDS Buffer"},
      memory_semaphore{instance} {
    stream_buffer.untracked_reads = true; // sotcport: see Buffer::untracked_reads
    const vk::BufferCreateInfo probe_ci = {
        .flags =
            vk::BufferCreateFlagBits::eSparseBinding | vk::BufferCreateFlagBits::eSparseResidency,
        .size = ARENA_PAGE_SIZE,
        .usage = ARENA_USAGE,
        .sharingMode = vk::SharingMode::eExclusive,
    };
    const vk::DeviceBufferMemoryRequirements req_info = {
        .pCreateInfo = &probe_ci,
    };
    const auto device = instance.GetDevice();
    const auto reqs = device.getBufferMemoryRequirements(req_info).memoryRequirements;
    block_size = Common::AlignUp(std::max<u64>(reqs.alignment, MIN_BLOCK_SIZE), reqs.alignment);
    ASSERT_MSG(std::popcount(block_size) == 1, "Sparse block size {} is not a power of 2",
               block_size);
    block_shift = std::bit_width(block_size) - 1;
    blocks_per_arena_page = ARENA_PAGE_SIZE / block_size;
    blocks_per_arena_page_shift = ARENA_PAGE_BITS - block_shift;
    arena_memory_type_index =
        FindMemoryType(instance.GetMemoryProperties(), vk::MemoryPropertyFlagBits::eDeviceLocal,
                       reqs.memoryTypeBits)
            .value();

    const u64 bda_pagetable_size =
        (blocks_per_arena_page * NUM_ARENA_PAGES) * sizeof(vk::DeviceAddress);
    fault_manager = std::make_unique<FaultManager>(instance, scheduler, *this, block_shift,
                                                   blocks_per_arena_page * NUM_ARENA_PAGES);
    bda_pagetable_buffer = std::make_unique<Buffer>(
        instance, 0, bda_pagetable_size, MemoryType::DeviceLocal, "BDA Page Table Buffer");
    runtime.FillBuffer(bda_pagetable_buffer.get(), 0u, bda_pagetable_size, 0u);
}

BufferCache::~BufferCache() = default;

void BufferCache::TickFrame() {
    if (std::exchange(fault_process_pending, false)) {
        fault_manager->ProcessFaultBuffer();
    }
}

void BufferCache::InvalidateMemory(VAddr device_addr, u64 size, bool assume_locks) {
    memory_tracker->InvalidateRegion(device_addr, size, [this, device_addr, size, assume_locks] {
        ReadMemory(device_addr, size, true, assume_locks);
    });
}

void BufferCache::ExtendWriteFault(VAddr device_addr) {
    static const u64 window = [] {
        const char* env = std::getenv("SOTC_FAULT_WINDOW");
        // Off by default. Measured (2026-10-10, tools/bench_game.sh, 3 alternating runs each,
        // nobody at the controls): 256 KiB cut the write faults 15x (46k/s -> 3k/s, ~290 -> ~60
        // ms/s in the handlers) but the frame rate dropped a little (28.4 -> 27.8 mean): here the
        // faults run on the game's threads, which have time to spare, while the limit is the GPU
        // command thread, which the window gives more uploads to.
        const u64 kib = env ? std::strtoull(env, nullptr, 10) : 0;
        return kib ? std::bit_ceil(std::clamp<u64>(kib, 4, 4096)) * 1024 : 0;
    }();
    if (window > 4096) {
        memory_tracker->ExtendWriteFault(Common::AlignDown(device_addr, window), window);
    }
}

boost::container::small_vector<std::pair<VAddr, u64>, 33> BufferCache::ReadbackWindows(
    VAddr device_addr, u64 size) {
    const u32 first_block = device_addr >> block_shift;
    const u32 last_block = (device_addr + size - 1) >> block_shift;
    const auto* arena = GetArena(first_block, last_block);

    // GPU-modified ranges come as many small scattered islands,
    // so the download is widened to a window around the request
    constexpr u64 WindowSize = 512_KB;
    const VAddr arena_end = arena->cpu_addr + arena->size_bytes;
    const VAddr window_start =
        std::max<VAddr>(Common::AlignDown(device_addr, WindowSize), arena->cpu_addr);
    const VAddr window_end =
        std::min<VAddr>(std::max<VAddr>(window_start + WindowSize, device_addr + size), arena_end);
    // sotcport: each download waits for the GPU to go idle, and the game reads back
    // the results of GPU work from the same couple dozen places every frame, one after the
    // other. Fetch whatever the GPU wrote in those places along with this request, so one
    // wait serves them all; downloading early is what a later fault would do anyway.
    boost::container::small_vector<std::pair<VAddr, u64>, 33> windows;
    windows.emplace_back(window_start, window_end - window_start);
    const VAddr hot_key = Common::AlignDown(device_addr, WindowSize);
    bool known = false;
    // SOTC_NO_HOT_WINDOWS=1 downloads only the requested window (diagnostics).
    static const bool hot_windows_enabled = !std::getenv("SOTC_NO_HOT_WINDOWS");
    for (const VAddr hot : hot_windows_enabled ? hot_windows : decltype(hot_windows){}) {
        known |= hot == hot_key;
        if (hot && hot != hot_key) {
            windows.emplace_back(hot, WindowSize);
        }
    }
    if (!known) {
        hot_windows[hot_windows_next++ % hot_windows.size()] = hot_key;
    }
    return windows;
}

void BufferCache::ReadMemory(VAddr device_addr, u64 size, bool is_write, bool assume_locks) {
    perf::NoteReadback(device_addr, size, is_write); // sotcport TEMP
    const auto flush_request = [this, device_addr, size, is_write] {
        DownloadWindows(ReadbackWindows(device_addr, size));
        if (is_write) {
            memory_tracker->MarkRegionAsCpuModified(device_addr, size);
        }
    };
    if (assume_locks) {
        // sotcport: the requested bytes may be on their way back for a guest thread.
        WaitForReadbacks(device_addr, size);
        flush_request();
        return;
    }
    static const bool async = std::getenv("SOTC_ASYNC_READBACK") != nullptr;
    if (!async) {
        liverpool->SendCommand<true>(std::move(flush_request));
        return;
    }
    AsyncReadback readback;
    perf::Mark('R'); // TEMP timeline: a guest thread reads GPU data back
    // sotcport: stage B records the copy right away (the game read after a fence stage B
    // signalled, so the work writing these bytes is recorded already); without the draw
    // pipeline the command processor does it.
    auto* rasterizer = liverpool->GetRasterizer();
    if (!rasterizer ||
        !rasterizer->RunOnStageB([&] { readback = PrepareReadback(device_addr, size); })) {
        liverpool->SendCommand<true>([&] { readback = PrepareReadback(device_addr, size); });
    }
    CompleteReadback(readback, device_addr, size, is_write);
    perf::Mark('r');
}

BufferCache::AsyncReadback BufferCache::PrepareReadback(VAddr device_addr, u64 size) {
    // GPU command thread, stage B of the draw pipeline idle (a pending command drains it).
    AsyncReadback readback;
    const auto windows = ReadbackWindows(device_addr, size);
    std::scoped_lock lock{gpu_modified_mutex};
    const u64 total_size_bytes = CollectDownloads(windows, readback.windows, readback.copies);
    if (total_size_bytes == 0) {
        perf::Add(perf::EmptyDownload, 0); // sotcport TEMP
        // The bytes may already be on their way back for another thread: wait for that one.
        std::scoped_lock readback_lock{readback_mutex};
        for (const auto& inflight : readback_inflight) {
            if (device_addr < inflight.end && inflight.start < device_addr + size) {
                readback.wait_for = inflight.sequence;
            }
        }
        return readback;
    }
    readback.empty = false;
    readback.staging = staging_pool.Request(total_size_bytes, VideoCore::MemoryType::HostCached,
                                            0, /*deferred=*/true);
    VAddr span_start = ~VAddr{0}, span_end = 0;
    for (const DownloadWindow& window : readback.windows) {
        auto copies = std::span{readback.copies}.subspan(window.first_copy, window.num_copies);
        for (auto& copy : copies) {
            copy.dstOffset += readback.staging.offset;
            readback_ranges.Add(window.arena->cpu_addr + copy.srcOffset, copy.size);
        }
        if (!copies.empty()) {
            runtime.CopyBuffer(window.arena, readback.staging.buffer, copies);
        }
        span_start = std::min(span_start, window.start);
        span_end = std::max(span_end, window.start + window.size);
    }
    readback.tick = scheduler.CurrentTick();
    scheduler.Flush();
    std::scoped_lock readback_lock{readback_mutex};
    readback.sequence = readback_next++;
    readback_inflight.push_back({readback.sequence, span_start, span_end});
    return readback;
}

void BufferCache::CompleteReadback(AsyncReadback& readback, VAddr device_addr, u64 size,
                                   bool is_write) {
    // The guest thread that faulted; nothing here needs the GPU command thread.
    // A write fault (is_write) does not mark the bytes CPU-modified here, as the synchronous path
    // does: stage B may be marking the same pages GPU-modified right now, and a page marked both
    // ways would get a write-only protection. The guest's write faults once more instead, and
    // that fault (MemoryTracker::InvalidateRegion, under the region lock) marks it safely.
    (void)is_write;
    if (readback.empty) {
        if (readback.wait_for != ~u64{0}) {
            std::unique_lock lock{readback_mutex};
            readback_cv.wait(lock, [&] { return readback_completed > readback.wait_for; });
        }
        return;
    }
    {
        perf::Scope perf_scope{perf::WaitDownload}; // sotcport TEMP
        scheduler.GetWorkSemaphore()->Wait(readback.tick);
    }
    {
        // In order: an older readback of the same bytes must not land after a newer one.
        std::unique_lock lock{readback_mutex};
        readback_cv.wait(lock, [&] { return readback_completed == readback.sequence; });
    }
    readback.staging.buffer->Invalidate(readback.staging.offset, readback.staging.size);
    {
        std::scoped_lock lock{gpu_modified_mutex};
        for (const DownloadWindow& window : readback.windows) {
            for (const auto& copy :
                 std::span{readback.copies}.subspan(window.first_copy, window.num_copies)) {
                const VAddr address = window.arena->cpu_addr + copy.srcOffset;
                const u8* data = readback.staging.mapped + (copy.dstOffset - readback.staging.offset);
                perf::NoteDownload(address, data, copy.size); // sotcport TEMP
                memory->TryWriteBacking(std::bit_cast<u8*>(address), data, copy.size);
                readback_ranges.Subtract(address, copy.size);
            }
        }
        for (const DownloadWindow& window : readback.windows) {
            UnmarkDownloaded(window.start, window.size);
        }
    }
    {
        std::scoped_lock lock{readback_mutex};
        readback_completed = readback.sequence + 1;
        std::erase_if(readback_inflight, [&](const InflightReadback& inflight) {
            return inflight.sequence == readback.sequence;
        });
    }
    readback_cv.notify_all();
    // The staging memory goes back to the pool at its owner's next request (sotcport: it was a
    // command for the GPU command thread, which drained the draw pipeline for it).
    staging_pool.FreeDeferredFromAnyThread(readback.staging);
}

void BufferCache::WaitForReadbacks(VAddr addr, u64 size) {
    std::unique_lock lock{readback_mutex};
    readback_cv.wait(lock, [&] {
        return std::ranges::none_of(readback_inflight, [&](const InflightReadback& inflight) {
            return addr < inflight.end && inflight.start < addr + size;
        });
    });
}

void BufferCache::DownloadMemory(const Buffer* arena, VAddr device_addr, u64 size) {
    std::scoped_lock lock{gpu_modified_mutex}; // sotcport: see gpu_modified_mutex
    boost::container::small_vector<vk::BufferCopy, 1> copies;
    u64 total_size_bytes = 0;
    const VAddr arena_base = arena->cpu_addr;
    memory_tracker->ForEachDownloadRange<false>(device_addr, size, [&](u64 address, u64 size) {
        const auto add_download = [&](VAddr start, VAddr end) {
            const u64 new_offset = start - arena_base;
            const u64 new_size = end - start;
            copies.push_back(vk::BufferCopy{
                .srcOffset = new_offset,
                .dstOffset = total_size_bytes,
                .size = new_size,
            });
            // Align up to avoid cache conflicts
            constexpr u64 align = 64ULL;
            constexpr u64 mask = ~(align - 1ULL);
            total_size_bytes += (new_size + align - 1) & mask;
        };
        gpu_modified_ranges.ForEachInRange(address, size, add_download);
        gpu_modified_ranges.Subtract(address, size);
        gpu_modified_removals.fetch_add(1, std::memory_order_release);
    });
    if (total_size_bytes == 0) {
        perf::Add(perf::EmptyDownload, 0); // sotcport TEMP
        return;
    }
    const auto download = staging_pool.Request(total_size_bytes, VideoCore::MemoryType::HostCached);
    for (auto& copy : copies) {
        copy.dstOffset += download.offset;
    }
    runtime.CopyBuffer(arena, download.buffer, copies);
    {
        perf::Scope perf_scope{perf::WaitDownload}; // sotcport TEMP
        scheduler.Finish();
    }

    download.buffer->Invalidate(download.offset, download.size);
    for (const auto& copy : copies) {
        auto* dst_addr = std::bit_cast<u8*>(arena_base + copy.srcOffset);
        perf::NoteDownload(arena_base + copy.srcOffset,
                           download.mapped + (copy.dstOffset - download.offset),
                           copy.size); // sotcport TEMP
        memory->TryWriteBacking(dst_addr, download.mapped + (copy.dstOffset - download.offset),
                                copy.size);
    }
    memory_tracker->UnmarkRegionAsGpuModified(device_addr, size, false);
}

u64 BufferCache::CollectDownloads(std::span<const std::pair<VAddr, u64>> windows,
                                  DownloadWindowList& out, DownloadCopyList& copies) {
    u64 total_size_bytes = 0;
    for (auto [start, size] : windows) {
        // Clamp to the arena of the window's first byte (a lookup of one block never migrates).
        const auto* arena = GetArena(start >> block_shift, start >> block_shift);
        const VAddr end = std::min<VAddr>(start + size, arena->cpu_addr + arena->size_bytes);
        start = std::max<VAddr>(start, arena->cpu_addr);
        if (start >= end) {
            continue;
        }
        size = end - start;
        const size_t first_copy = copies.size();
        const VAddr arena_base = arena->cpu_addr;
        memory_tracker->ForEachDownloadRange<false>(start, size, [&](u64 address, u64 size) {
            gpu_modified_ranges.ForEachInRange(address, size, [&](VAddr range_start,
                                                                  VAddr range_end) {
                copies.push_back(vk::BufferCopy{
                    .srcOffset = range_start - arena_base,
                    .dstOffset = total_size_bytes,
                    .size = range_end - range_start,
                });
                // Align up to avoid cache conflicts
                total_size_bytes += Common::AlignUp(range_end - range_start, 64ULL);
            });
            gpu_modified_ranges.Subtract(address, size);
            gpu_modified_removals.fetch_add(1, std::memory_order_release);
        });
        out.push_back({arena, start, size, first_copy, copies.size() - first_copy});
    }
    return total_size_bytes;
}

void BufferCache::UnmarkDownloaded(VAddr start, u64 size) {
    if (readback_ranges.Empty() && !gpu_modified_ranges.Intersects(start, size)) {
        memory_tracker->UnmarkRegionAsGpuModified(start, size, false);
        return;
    }
    // Page by page, in runs of pages that have nothing pending.
    const VAddr end = start + size;
    VAddr run = 0;
    bool in_run = false;
    for (VAddr page = Common::AlignDown(start, BYTES_PER_PAGE); page < end;
         page += BYTES_PER_PAGE) {
        const VAddr lo = std::max(page, start);
        const VAddr hi = std::min(page + BYTES_PER_PAGE, end);
        const bool pending = gpu_modified_ranges.Intersects(page, BYTES_PER_PAGE) ||
                             readback_ranges.Intersects(page, BYTES_PER_PAGE);
        if (!pending && !in_run) {
            run = lo;
            in_run = true;
        } else if (pending && in_run) {
            memory_tracker->UnmarkRegionAsGpuModified(run, lo - run, false);
            in_run = false;
        }
        if (!pending && hi == end) {
            memory_tracker->UnmarkRegionAsGpuModified(run, hi - run, false);
            in_run = false;
        }
    }
}

void BufferCache::DownloadWindows(std::span<const std::pair<VAddr, u64>> windows) {
    std::scoped_lock lock{gpu_modified_mutex}; // sotcport: see gpu_modified_mutex
    DownloadWindowList pending;
    DownloadCopyList copies;
    const u64 total_size_bytes = CollectDownloads(windows, pending, copies);
    if (total_size_bytes == 0) {
        perf::Add(perf::EmptyDownload, 0); // sotcport TEMP
        return;
    }
    const auto download = staging_pool.Request(total_size_bytes, VideoCore::MemoryType::HostCached);
    for (auto& copy : copies) {
        copy.dstOffset += download.offset;
    }
    for (const DownloadWindow& window : pending) {
        if (window.num_copies) {
            runtime.CopyBuffer(window.arena, download.buffer,
                               std::span{copies}.subspan(window.first_copy, window.num_copies));
        }
    }
    {
        perf::Scope perf_scope{perf::WaitDownload}; // sotcport TEMP
        scheduler.Finish();
    }
    download.buffer->Invalidate(download.offset, download.size);
    for (const DownloadWindow& window : pending) {
        if (!window.num_copies) {
            continue;
        }
        for (const auto& copy :
             std::span{copies}.subspan(window.first_copy, window.num_copies)) {
            const VAddr address = window.arena->cpu_addr + copy.srcOffset;
            const u8* data = download.mapped + (copy.dstOffset - download.offset);
            perf::NoteDownload(address, data, copy.size); // sotcport TEMP
            memory->TryWriteBacking(std::bit_cast<u8*>(address), data, copy.size);
        }
        UnmarkDownloaded(window.start, window.size);
    }
}

std::pair<const Buffer*, u64> BufferCache::ObtainBuffer(VAddr device_addr, u32 size,
                                                        bool is_written, bool is_texel_buffer,
                                                        bool shadowed) {
    // For read-only buffers use device local stream buffer to reduce renderpass breaks.
    // sotcport: SOTC_STREAM_THRESHOLD=<bytes> raises the size limit (diagnostics: larger
    // read-only buffers then come straight from memory, not from tracked arena copies).
    static const u64 stream_threshold = [] {
        const char* bytes = std::getenv("SOTC_STREAM_THRESHOLD");
        return bytes ? std::strtoull(bytes, nullptr, 0) : u64{STREAM_THRESHOLD};
    }();
    diag::NoteMemoryAccess(is_written ? diag::MemAccess::BufferWritten : diag::MemAccess::BufferRead,
                           device_addr, size); // sotcport (SOTC_MEMORY_TYPES)
    if (!is_written && size <= stream_threshold && !IsRegionGpuModified(device_addr, size)) {
        diag::NoteMemoryAccess(diag::MemAccess::UploadRead, device_addr, size);
        const auto [data, offset] = stream_buffer.Map(size, instance.UniformMinAlignment());
        memory->CopySparseMemory(device_addr, data, size);
        stream_buffer.Commit();
        return {&stream_buffer, offset};
    }
    const u64 first_block = device_addr >> block_shift;
    const u64 last_block = (device_addr + size - 1) >> block_shift;
    const auto* arena = GetArena(first_block, last_block);
    EnsureResident(arena, first_block, last_block);
    // sotcport: the exact range first, then the page state (SynchronizeMemory): an asynchronous
    // readback completing in between (CompleteReadback) only clears the state of pages with no
    // exact range pending, so it cannot clear this write's mark.
    if (is_written && !shadowed) {
        // sotcport: the same storage buffers are bound for writing draw after draw; the interval
        // set insert (allocation, merging) is skipped when the range is already there.
        // sotcport: and the lock and the search are skipped for a range this thread found or
        // added there since the last removal (gpu_modified_removals): ~5% of the draw recording
        // thread, the game's compute dispatches binding the same output buffers one after the
        // other (2026-10-19).
        struct KnownRange {
            VAddr address;
            u64 size;
            u64 removals;
        };
        static thread_local std::array<KnownRange, 64> known{};
        auto& slot = known[((device_addr >> 6) ^ size) % known.size()];
        if (slot.address != device_addr || slot.size != size ||
            slot.removals != gpu_modified_removals.load(std::memory_order_acquire)) {
            std::scoped_lock lock{gpu_modified_mutex}; // sotcport: two-stage draw pipeline
            if (!gpu_modified_ranges.Contains(device_addr, size)) {
                gpu_modified_ranges.Add(device_addr, size);
            }
            slot = {device_addr, size, gpu_modified_removals.load(std::memory_order_relaxed)};
        }
    }
    SynchronizeMemory(arena, device_addr, size, is_written && !shadowed, is_texel_buffer);
    return {arena, arena->Offset(device_addr)};
}

std::pair<const Buffer*, u64> BufferCache::ObtainBufferForImage(VAddr device_addr, u32 size) {
    if (IsRegionGpuModified(device_addr, size)) {
        return ObtainBuffer(device_addr, size, false);
    }
    const auto staging = staging_pool.Request(size, VideoCore::MemoryType::HostUncached,
                                              instance.StorageMinAlignment());
    memory->CopySparseMemory(device_addr, staging.mapped, staging.size);
    staging.Flush();
    return {staging.buffer, staging.offset};
}

void BufferCache::ForgetGpuWrites(VAddr addr, u64 size) {
    WaitForReadbacks(addr, size);
    std::scoped_lock lock{gpu_modified_mutex};
    if (!gpu_modified_ranges.Intersects(addr, size)) {
        return;
    }
    gpu_modified_ranges.Subtract(addr, size);
    gpu_modified_removals.fetch_add(1, std::memory_order_release);
    UnmarkDownloaded(addr, size);
}

bool BufferCache::IsRegionCpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionCpuModified(addr, size);
}

u64 BufferCache::CpuMarks() const {
    return memory_tracker->CpuMarks();
}

bool BufferCache::IsRegionGpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionGpuModified(addr, size);
}

bool BufferCache::PeekGpuMemory(VAddr device_addr, u64 size, u8* out) {
    const u64 first_block = device_addr >> block_shift;
    const u64 last_block = (device_addr + size - 1) >> block_shift;
    const Buffer* arena = address_space[first_block >> blocks_per_arena_page_shift];
    if (!arena || address_space[last_block >> blocks_per_arena_page_shift] != arena) {
        return false;
    }
    bool resident = true;
    resident_ranges.ForEachGap(first_block, last_block + 1,
                               [&](u64, u64) { resident = false; });
    if (!resident) {
        return false;
    }
    const auto download = staging_pool.Request(size, VideoCore::MemoryType::HostCached);
    const vk::BufferCopy copy{
        .srcOffset = device_addr - arena->cpu_addr,
        .dstOffset = download.offset,
        .size = size,
    };
    runtime.CopyBuffer(arena, download.buffer, std::span{&copy, 1});
    scheduler.Finish();
    download.buffer->Invalidate(download.offset, download.size);
    std::memcpy(out, download.mapped, size);
    return true;
}

void BufferCache::SynchronizeDmaBuffers() {
    fault_process_pending = true;
    // sotcport: nothing to do when no memory became CPU-modified since the last time (draws
    // with DMA access come in runs; the scan below walks all resident memory, ~9% of the draw
    // recording thread). Memory made resident counts too (EnsureResident). The mark is read
    // before scanning: a write landing during the scan changes it and the next call scans.
    const u64 marks = memory_tracker->CpuMarks();
    if (marks == dma_synced_marks) {
        return;
    }
    dma_synced_marks = marks;
    // sotcport: only the high pages marked CPU-dirty since the last time (NoteCpuDirty), where
    // they are resident (what becomes resident is marked too, EnsureResident).
    memory_tracker->TakeCpuDirty([this](VAddr dirty_addr, u64 dirty_size) {
        const u64 first_block = dirty_addr >> block_shift;
        const u64 end_block = (dirty_addr + dirty_size) >> block_shift;
        resident_ranges.ForEachInRange(first_block, end_block, [&](const Backing& backing) {
            const u64 start = std::max(first_block, backing.start);
            const u64 end = std::min(end_block, backing.end);
            if (start >= end) {
                return;
            }
            const u64 page = start >> (ARENA_PAGE_BITS - block_shift);
            SynchronizeMemory(address_space[page], start << block_shift,
                              (end - start) << block_shift, false, false);
        });
    });
}

const Buffer* BufferCache::GetArena(u64 first_block, u64 last_block) {
    const u64 first_page = first_block >> blocks_per_arena_page_shift;
    const u64 last_page = last_block >> blocks_per_arena_page_shift;
    ASSERT_MSG(last_page - first_page <= 1,
               "Buffer request cannot span more than two VA arena pages");

    const auto* first_arena = address_space[first_page];
    const auto* last_arena = address_space[last_page];
    if (first_arena == last_arena) {
        if (!first_arena) {
            const u64 base_block = Common::AlignDownPow2<u64>(first_block, blocks_per_arena_page);
            const u64 num_pages = last_page - first_page + 1;
            const auto* new_arena =
                &arenas.emplace_back(instance, base_block << block_shift,
                                     num_pages << ARENA_PAGE_BITS, MemoryType::Sparse);
            address_space[first_page] = new_arena;
            address_space[last_page] = new_arena;
        }
        return address_space[first_page];
    }

    LOG_WARNING(Render, "Migrating arena");

    const u64 first_addr = first_arena ? first_arena->cpu_addr : (first_page << ARENA_PAGE_BITS);
    const u64 first_size = first_arena ? first_arena->size_bytes : ARENA_PAGE_SIZE;
    const u64 last_size = last_arena ? last_arena->size_bytes : ARENA_PAGE_SIZE;

    const u64 base_block = first_addr >> block_shift;
    const u64 total_size = first_size + last_size;
    const u64 end_block = (first_addr + total_size) >> block_shift;
    auto* new_arena = &arenas.emplace_back(instance, first_addr, total_size, MemoryType::Sparse);
    auto* bind = BindsForArena(new_arena);
    resident_ranges.ForEachInRange(base_block, end_block, [&](const Backing& backing) {
        const u64 start = std::max(base_block, backing.start);
        const u64 end = std::min(end_block, backing.end);
        bind->binds.push_back(vk::SparseMemoryBind{
            .resourceOffset = (start - base_block) << block_shift,
            .size = (end - start) << block_shift,
            .memory = backing.memory,
            .memoryOffset = (backing.offset + start - backing.start) << block_shift,
        });
    });

    u64 base_page = first_addr >> ARENA_PAGE_BITS;
    for (u32 page = 0; page < (first_size >> ARENA_PAGE_BITS); ++page) {
        address_space[base_page + page] = new_arena;
    }
    base_page = last_page;
    for (u32 page = 0; page < (last_size >> ARENA_PAGE_BITS); ++page) {
        address_space[base_page + page] = new_arena;
    }
    return new_arena;
}

void BufferCache::EnsureResident(const Buffer* arena, u64 first_block, u64 last_block) {
    u32 resident_blocks{};
    IntervalList bind_ranges;
    resident_ranges.ForEachGap(first_block, last_block + 1, [&](u64 start, u64 end) {
        resident_blocks += end - start;
        bind_ranges.Add({start, end});
    });

    if (bind_ranges.Empty()) {
        return;
    }

    const vk::MemoryAllocateFlagsInfo alloc_flags = {
        .flags = vk::MemoryAllocateFlagBits::eDeviceAddress,
    };
    const vk::MemoryAllocateInfo alloc_info = {
        .pNext = &alloc_flags,
        .allocationSize = resident_blocks << block_shift,
        .memoryTypeIndex = arena_memory_type_index,
    };
    auto allocated = instance.GetDevice().allocateMemory(alloc_info);
    // sotcport: say how much was asked and how full the VRAM was before failing (4 GB cards).
    if (allocated.result != vk::Result::eSuccess) {
        VmaBudget budgets[VK_MAX_MEMORY_HEAPS]{};
        vmaGetHeapBudgets(instance.GetAllocator(), budgets);
        LOG_CRITICAL(Render,
                     "arena allocation of {} MB failed: {}; heap 0: {} MB used of a {} MB "
                     "budget, {} MB of it in the allocator's blocks",
                     alloc_info.allocationSize >> 20, vk::to_string(allocated.result),
                     budgets[0].usage >> 20, budgets[0].budget >> 20,
                     budgets[0].statistics.blockBytes >> 20);
    }
    const auto device_memory = Vulkan::Check(std::move(allocated));

    boost::container::small_vector<vk::BufferCopy, 8> copies;
    const auto staging =
        staging_pool.Request(resident_blocks * sizeof(vk::DeviceAddress), MemoryType::HostUncached);

    u64 memory_offset{};
    ArenaBinds* binds = BindsForArena(arena);
    auto* bda_addrs = reinterpret_cast<vk::DeviceAddress*>(staging.mapped);
    u64 offset = staging.offset;
    for (const auto& range : bind_ranges) {
        Backing backing;
        backing.start = range.start;
        backing.end = range.end;
        backing.memory = device_memory;
        backing.offset = memory_offset >> block_shift;
        resident_ranges.Add(backing);
        // sotcport: see SynchronizeDmaBuffers
        memory_tracker->NoteCpuDirty(range.start << block_shift,
                                     (range.end - range.start) << block_shift);

        LOG_INFO(Render, "Making range start={}, end={} resident", backing.start, backing.end);

        const auto& bind = binds->binds.emplace_back(vk::SparseMemoryBind{
            .resourceOffset = (range.start << block_shift) - arena->cpu_addr,
            .size = (range.end - range.start) << block_shift,
            .memory = device_memory,
            .memoryOffset = memory_offset,
        });
        memory_offset += bind.size;

        for (u32 block = 0; block < bind.size; block += block_size) {
            *(bda_addrs++) = arena->BufferDeviceAddress() + bind.resourceOffset + block;
        }
        const u64 copy_size = (backing.end - backing.start) * sizeof(vk::DeviceAddress);
        copies.emplace_back(offset, backing.start * sizeof(vk::DeviceAddress), copy_size);
        offset += copy_size;
    }

    staging.Flush();
    runtime.CopyBuffer(staging.buffer, bda_pagetable_buffer.get(), copies);
}

bool BufferCache::SynchronizeMemory(const Buffer* arena, VAddr device_addr, u32 size,
                                    bool is_written, bool is_texel_buffer) {
    boost::container::small_vector<vk::BufferCopy, 4> copies;
    size_t total_size_bytes{};
    memory_tracker->ForEachUploadRange(device_addr, size, is_written, [&](u64 addr, u64 size) {
        copies.emplace_back(total_size_bytes, addr, size);
        total_size_bytes += size;
    });
    if (!copies.empty()) {
        const auto staging = staging_pool.Request(total_size_bytes, MemoryType::HostUncached);
        for (auto& copy : copies) {
            diag::NoteMemoryAccess(diag::MemAccess::UploadRead, copy.dstOffset, copy.size);
            memory->CopySparseMemory(copy.dstOffset, staging.mapped + copy.srcOffset, copy.size);
            copy.srcOffset += staging.offset;
            copy.dstOffset -= arena->cpu_addr;
        }
        staging.Flush();
        runtime.CopyBuffer(staging.buffer, arena, copies);
    }
    if (is_texel_buffer && !is_written) {
        return SynchronizeMemoryFromImage(arena, device_addr, size);
    }
    return false;
}

bool BufferCache::SynchronizeMemoryFromImage(const Buffer* arena, VAddr device_addr, u32 size) {
    if (auto type = texture_cache.IsMeta(device_addr)) {
        if (*type == TextureCache::MetaType::HTile) {
            static constexpr u32 ZmaskUncompressed = 0xf;
            runtime.FillBuffer(arena, arena->Offset(device_addr), size, ZmaskUncompressed);
            return true;
        } else if (*type == TextureCache::MetaType::CMask) {
            static constexpr u32 CmaskCompressed = 0xCCCCCCCC;
            runtime.FillBuffer(arena, arena->Offset(device_addr), size, CmaskCompressed);
            return true;
        } else {
            LOG_WARNING(Render_Vulkan, "Unhandled metadata type {}", magic_enum::enum_name(*type));
        }
    }
    const ImageId image_id = texture_cache.FindImageFromRange(device_addr, size);
    if (!image_id) {
        return false;
    }
    Image& image = texture_cache.GetImage(image_id);
    ASSERT_MSG(device_addr == image.info.guest_address,
               "Texel buffer aliases image subresources {:x} : {:x}", device_addr,
               image.info.guest_address);
    const u64 arena_offset = arena->Offset(device_addr);
    boost::container::small_vector<vk::BufferImageCopy, 8> buffer_copies;
    for (u32 mip = 0; mip < image.info.resources.levels; mip++) {
        const auto& mip_info = image.info.mips_layout[mip];
        const u32 width = std::max(image.info.size.width >> mip, 1u);
        const u32 height = std::max(image.info.size.height >> mip, 1u);
        const u32 depth = std::max(image.info.size.depth >> mip, 1u);
        if (arena_offset + mip_info.offset + mip_info.size > arena->size_bytes) {
            break;
        }
        buffer_copies.push_back(vk::BufferImageCopy{
            .bufferOffset = mip_info.offset,
            .bufferRowLength = mip_info.pitch,
            .bufferImageHeight = mip_info.height,
            .imageSubresource{
                .aspectMask = image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
                .mipLevel = mip,
                .baseArrayLayer = 0,
                .layerCount = image.info.resources.layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {width, height, depth},
        });
    }
    if (buffer_copies.empty()) {
        return false;
    }
    auto& tile_manager = texture_cache.GetTileManager();
    tile_manager.TileImage(image, buffer_copies, arena, arena_offset);
    return true;
}

void BufferCache::SubmitPendingArenaBinds(Vulkan::SubmitInfo& info) {
    if (pending_binds.empty()) {
        return;
    }

    std::vector<vk::SparseBufferMemoryBindInfo> buffer_binds;
    buffer_binds.reserve(pending_binds.size());

    for (const auto& binds : pending_binds) {
        buffer_binds.emplace_back(vk::SparseBufferMemoryBindInfo{
            .buffer = binds.arena->Handle(),
            .bindCount = static_cast<u32>(binds.binds.size()),
            .pBinds = binds.binds.data(),
        });
    }

    const u64 signal_tick = memory_semaphore.NextTick();
    const auto signal_sema = memory_semaphore.Handle();

    const vk::TimelineSemaphoreSubmitInfo timeline_si = {
        .signalSemaphoreValueCount = 1u,
        .pSignalSemaphoreValues = &signal_tick,
    };

    const vk::BindSparseInfo sparse_info = {
        .pNext = &timeline_si,
        .bufferBindCount = static_cast<u32>(buffer_binds.size()),
        .pBufferBinds = buffer_binds.data(),
        .signalSemaphoreCount = 1u,
        .pSignalSemaphores = &signal_sema,
    };

    info.AddWait(signal_sema, signal_tick);
    auto submit_result = instance.GetGraphicsQueue().bindSparse(sparse_info);
    ASSERT_MSG(submit_result != vk::Result::eErrorDeviceLost, "Device lost during submit");

    pending_binds.clear();
}

} // namespace VideoCore
