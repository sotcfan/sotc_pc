// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/interval_set.h"
#include "common/types.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/renderer_vulkan/vk_staging_buffer_pool.h"
#include "video_core/texture_cache/image.h"
#include "video_core/texture_cache/types.h"

namespace VideoCore {
class BlitHelper;
} // namespace VideoCore

namespace Vulkan {

class Instance;
class Scheduler;

class Runtime {
public:
    explicit Runtime(const Instance& instance, Scheduler& scheduler);
    ~Runtime() = default;

    const Instance& GetInstance() const {
        return instance;
    }

    StagingBufferPool& GetStagingPool() {
        return staging_pool;
    }

    void TickFrame();

    void CopyBuffer(const VideoCore::Buffer* src, const VideoCore::Buffer* dst,
                    std::span<const vk::BufferCopy> copies);

    void FillBuffer(const VideoCore::Buffer* dst, u64 offset, u64 size, u32 value);
    /// sotcport: writes `words` into dst at `offset` (4-byte aligned, at most 64 KiB) in
    /// recording order (vkCmdUpdateBuffer).
    void UpdateBuffer(const VideoCore::Buffer* dst, u64 offset, std::span<const u32> words);

    void InlineData(VideoCore::Buffer* dst, u64 offset, u32 value);

    bool Transit(VideoCore::Image* image, vk::ImageLayout dst_layout,
                 vk::PipelineStageFlags2 dst_stage, vk::AccessFlags2 dst_access,
                 std::optional<VideoCore::SubresourceRange> subres_range = {});

    void UploadImage(VideoCore::Image* dst, const VideoCore::Buffer* src,
                     std::span<const vk::BufferImageCopy> upload_copies);
    void DownloadImage(VideoCore::Image* src, const VideoCore::Buffer* dst,
                       std::span<const vk::BufferImageCopy> download_copies);

    void CopyImage(VideoCore::Image* src, VideoCore::Image* dst);
    void CopyImageWithBuffer(VideoCore::Image* src, VideoCore::Image* dst,
                             const VideoCore::Buffer* buffer, u64 offset);
    void CopyMip(VideoCore::Image* src, VideoCore::Image* dst, u32 mip, u32 slice);

    void CopyColorAndDepth(VideoCore::Image* src, VideoCore::Image* dst);

    void CopyDepthStencil(VideoCore::Image* src, VideoCore::Image* dst,
                          const VideoCore::SubresourceRange& sub_range);

    void ResolveImage(VideoCore::Image* src, VideoCore::Image* dst,
                      const VideoCore::SubresourceRange& src_range,
                      const VideoCore::SubresourceRange& dst_range);
    void ClearImage(VideoCore::Image* dst, const VideoCore::SubresourceRange& range,
                    const vk::ClearValue& clear_value);

    void SetBackingSamples(VideoCore::Image* image, u32 num_samples, bool copy_backing = true);

    /// sotcport: `guest`: an access by one of the game's shaders to one of its buffers (a
    /// storage or uniform binding). Hazards between two such accesses are left to the game's own
    /// synchronization (see GuestWritesPending); hazards with anything else (copies, uploads and
    /// downloads, fixed-function reads such as indirect arguments or indices) get a barrier as
    /// before.
    void AccessBuffer(const VideoCore::Buffer* handle, u64 offset, u64 size,
                      vk::PipelineStageFlags2 src_stage, vk::AccessFlags2 src_access,
                      bool guest = false);

    bool IsBufferAccessed(const VideoCore::Buffer* handle, u64 offset, u64 size,
                          bool check_read_access = false, bool guest = false);

    /// sotcport: whether a game shader wrote a buffer since the last barrier. On the PS4 the GPU
    /// starts a dispatch or draw without waiting for the ones before it: a game that needs their
    /// results waits for them itself (ACQUIRE_MEM, WAIT_REG_MEM on an end-of-pipe label...).
    /// Every hazard between the game's shaders used to get a global barrier here (~1,800 a frame
    /// in the forest, between dispatches that write the same buffers one after the other, where
    /// the game asks for ~100), which made the GPU run its small dispatches one at a time. Now
    /// stage B puts one barrier where the game synchronizes (Rasterizer::RunPacket), if a game
    /// shader wrote anything since the last one. SOTC_STRICT_BARRIERS=1 goes back to a barrier
    /// at every hazard.
    bool GuestWritesPending() const noexcept {
        return guest_writes_pending;
    }

    void FlushBarriers();

private:
    void MakeCurrent(const VideoCore::Buffer* handle);

private:
    const Instance& instance;
    Scheduler& scheduler;
    std::unique_ptr<VideoCore::BlitHelper> blit_helper;
    StagingBufferPool staging_pool;
    struct BufferBarriers {
        const VideoCore::Buffer* handle;
        using AccessList = IntervalList<Interval>;
        AccessList read_ranges;
        AccessList write_ranges;
        AccessList guest_read_ranges;  ///< sotcport: see AccessBuffer
        AccessList guest_write_ranges; ///< sotcport: see AccessBuffer
    };
    bool guest_writes_pending = false; ///< sotcport: see GuestWritesPending
    BufferBarriers* resource{};
    std::vector<BufferBarriers> resources;
    u32 num_resources = 0; ///< sotcport: entries of `resources` in use (see FlushBarriers)
    VideoCore::Image::Barriers image_barriers;
    vk::MemoryBarrier2 memory_barrier{};
};

} // namespace Vulkan
