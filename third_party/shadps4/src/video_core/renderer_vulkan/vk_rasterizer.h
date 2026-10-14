// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <functional>
#include "common/recursive_lock.h"
#include "common/shared_first_mutex.h"
#include "common/unique_function.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"

namespace AmdGpu {
struct Liverpool;
union Regs;
struct PM4CmdEventWriteEop;
struct PM4CmdEventWriteEos;
struct PM4CmdWaitRegMem;
struct PM4CmdReleaseMem;
enum class DataSelect : u32;
}

namespace Core {
class MemoryManager;
}

namespace Vulkan {

class GraphicsPipeline;
class Runtime;
class DrawPipe;

class Rasterizer {
public:
    explicit Rasterizer(const Instance& instance, Scheduler& scheduler, Runtime& runtime,
                        AmdGpu::Liverpool* liverpool);
    ~Rasterizer();

    [[nodiscard]] Scheduler& GetScheduler() noexcept {
        return scheduler;
    }

    [[nodiscard]] Runtime& GetRuntime() noexcept {
        return runtime;
    }

    [[nodiscard]] VideoCore::BufferCache& GetBufferCache() noexcept {
        return buffer_cache;
    }

    [[nodiscard]] VideoCore::TextureCache& GetTextureCache() noexcept {
        return texture_cache;
    }

    void Draw(bool is_indexed, u32 index_offset = 0);
    void DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 size, u32 max_count,
                      VAddr count_address, u16 vertex_sgpr_offset, u16 instance_sgpr_offset);

    void DispatchDirect();
    void DispatchIndirect(VAddr address, u32 offset, u32 size);

    void ScopeMarker(fmt::string_view fmt, fmt::format_args args, auto&& func) {
        if (host_markers_enabled) {
            ScopeMarkerBegin(fmt::vformat(fmt, args));
            func();
            ScopeMarkerEnd();
        } else {
            func();
        }
    }

    void ScopeMarkerBegin(const std::string_view& str, bool from_guest = false);
    void ScopeMarkerEnd(bool from_guest = false);
    void ScopedMarkerInsert(const std::string_view& str, bool from_guest = false);
    void ScopedMarkerInsertColor(const std::string_view& str, const u32 color,
                                 bool from_guest = false);

    // sotcport (two-stage draw pipeline, vk_draw_pipe.h): on the GPU command thread (stage A),
    // waits until the draw recording thread (stage B) has run every packet handed to it. Every
    // entry point that touches the caches, the runtime or the scheduler outside the draw path
    // calls it first; no-op elsewhere or when the pipeline is off.
    /// `reason`: DrawPipe::Reason or a graphics queue PM4 opcode (statistics only).
    void DrainDrawPipe(u32 reason = ~0u);
    /// sotcport (two-stage draw pipeline): hands an end-of-pipe fence to stage B, which signals
    /// it in stream order. False with the pipeline off: the caller signals it as before.
    bool PostEopFence(const AmdGpu::PM4CmdEventWriteEop& packet);
    /// sotcport: the same for an EVENT_WRITE_EOS label write.
    bool PostEosFence(const AmdGpu::PM4CmdEventWriteEos& packet);
    /// sotcport: IT_GET_LOD_STATS of the graphics queue, for the LOD statistics buffer at `base`
    /// of `size` bytes (see WriteLodStats). Runs on stage B in order with the packets.
    void GetLodStats(VAddr base, u32 size);
    /// sotcport (two-stage draw pipeline): hands a GPU command processor command (see
    /// Liverpool::SendCommand) to stage B, which runs it in order with the packets. Takes the
    /// callback (it is moved from) and returns true; false with the pipeline off or the option off.
    bool PostCommand(Common::UniqueFunction<void>& callback);
    /// The same for a compute queue's RELEASE_MEM.
    bool PostReleaseMem(const AmdGpu::PM4CmdReleaseMem& packet, u32 pipe_id);
    /// sotcport (two-stage draw pipeline): whether a fence handed to stage B and not signalled
    /// yet will meet this wait. Stage B runs everything in stream order, so whatever stage A
    /// queues after the wait runs after that fence anyway.
    bool PendingFenceMeets(const AmdGpu::PM4CmdWaitRegMem& wait);
    // True on the threads that own the GPU layer's state: stage A, or stage B while it runs.
    [[nodiscard]] bool IsGpuSideThread() const;
    // Stage A: whether a draw still queued for stage B will write [address, +size) (storage
    // buffers and images it binds). Used before stage A reads GPU-written guest memory.
    [[nodiscard]] bool PendingWriteOverlaps(VAddr address, u64 size);

    void FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds);
    void CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds);
    u32 ReadDataFromGds(u32 gsd_offset);
    bool InvalidateMemory(VAddr addr, u64 size, bool assume_locks = false);
    bool ReadMemory(VAddr addr, u64 size, bool assume_locks = false);
    // sotcport: see BufferCache::ExtendWriteFault (CPU write faults only).
    void ExtendWriteFault(VAddr addr) {
        buffer_cache.ExtendWriteFault(addr);
    }
    // sotcport: see BufferCache::AreBytesGpuModified.
    bool AreBytesGpuModified(VAddr addr, u64 size) const {
        return buffer_cache.AreBytesGpuModified(addr, size);
    }
    bool IsMapped(VAddr addr, u64 size);
    void MapMemory(VAddr addr, u64 size);
    void RegisterMemory(VAddr addr, u64 size);
    void UnmapMemory(VAddr addr, u64 size);

    u64 Flush();
    void Finish();
    void OnSubmit();
    /// `reason`: a DrawPipe::Reason for the drain (DrawPipe::ReasonFence when not given).
    void OnFence(u32 reason = 0);

    PipelineCache& GetPipelineCache() {
        return pipeline_cache;
    }

    template <typename Func>
    void ForEachMappedRangeInRange(VAddr addr, u64 size, Func&& func) {
        const auto range = decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
        Common::RecursiveSharedLock lock{mapped_ranges_mutex};
        for (const auto& mapped_range : (mapped_ranges & range)) {
            func(mapped_range);
        }
    }

    std::thread::id GetGpuCommandProcessorThread();
#ifdef __linux__
    u32 GetGpuCommandProcessorThreadId();
#endif

private:
    void PrepareRenderState(const GraphicsPipeline* pipeline);
    RenderState BeginRendering(const GraphicsPipeline* pipeline);
    void Resolve();
    void DepthStencilCopy(bool is_depth, bool is_stencil);
    void EliminateFastClear();
    void EliminateFastClear(const AmdGpu::ColorBuffer& col_buf, AmdGpu::CbDbExtent hint);
    /// sotcport: hands a FilterDraw pass to stage B, or skips one that does nothing; false when
    /// the caller has to drain and run it here.
    bool PostFilterDraw();

    void UpdateDynamicState(const GraphicsPipeline* pipeline, bool is_indexed) const;
    void UpdateViewportScissorState() const;
    void UpdateDepthStencilState() const;
    void UpdatePrimitiveState(bool is_indexed) const;
    void UpdateRasterizationState() const;
    void UpdateColorBlendingState(const GraphicsPipeline* pipeline) const;

    bool FilterDraw();

    // sotcport (two-stage draw pipeline): the parts of a draw or dispatch after the pipeline
    // selection, run on stage B when the pipeline is on (else right after the selection).
    struct IndirectDraw {
        VAddr arg_address;
        VAddr count_address;
        u32 offset;
        u32 stride;
        u32 max_count;
        u32 pad;
    };
    void DrawRecord(const GraphicsPipeline* pipeline, bool is_indexed, u32 index_offset);
    void DrawIndirectRecord(const GraphicsPipeline* pipeline, bool is_indexed,
                            const IndirectDraw& indirect);
    void DispatchRecord(const ComputePipeline* pipeline);
    /// sotcport: computes on the CPU, as the packet is processed, what the game's buffer
    /// fill/copy compute shader will write (see the .cpp). True when it did: the dispatch still
    /// runs on the GPU, with its destination not tracked as GPU-modified (shadow_dispatch).
    bool ShadowCopyShader(const ComputePipeline* pipeline);
    void VerifyShadowCopy();
    bool shadow_dispatch = false; ///< the dispatch being recorded was shadowed on the CPU
    VAddr shadow_verify_address = 0;          ///< SOTC_VERIFY_SHADOW_COPY: copy to compare
    std::vector<u32> shadow_verify_expected;  ///< what the CPU computed for it
    bool next_packet_shadowed = false; ///< stage A: the packet about to be posted was
    void MaybeFlushEarly();
    u32 work_since_flush = 0; ///< draws and dispatches recorded since the last early flush
    void DispatchIndirectRecord(const ComputePipeline* pipeline, VAddr address, u32 offset,
                                u32 size);
    enum class PacketKind : u32 { Draw, DrawIndirect, Dispatch, DispatchIndirect, Task };
    /// A task stage B runs in stream order (fences, events): `run` gets a copy of the payload.
    using TaskFn = void (*)(Rasterizer& self, const u8* payload);
    void PostTask(TaskFn run, const void* payload, u32 payload_size);
    void NotePendingWrite(VAddr address, u64 size);
    struct PendingFence {
        VAddr address;
        u64 value;
        u32 num_bytes;
        u64 position; ///< ring position after its packet
    };
    std::vector<PendingFence> pending_fences; ///< stage A only
    [[nodiscard]] bool PostFencesOnStageB() const;
    void NotePendingFence(VAddr address, AmdGpu::DataSelect data_sel, u64 data, u32 num_bytes);
    [[nodiscard]] bool UseDrawPipe() const noexcept {
        return draw_pipe != nullptr;
    }

public:
    /// sotcport: runs `work` on stage B of the draw pipeline ahead of its queued packets and
    /// returns when it ran (see DrawPipe::RunUrgent); false without the draw pipeline.
    bool RunOnStageB(std::function<void()> work);
    /// sotcport: the command processor is between packets (see DrawPipe::ReleaseStageA).
    void ReleaseStageA();

private:
    [[nodiscard]] bool OnStageA() const;
    // FilterDraw's checks without its side effects: false when it would skip the draw or run a
    // pass of its own (fast clear elimination, resolve, depth/stencil copy).
    [[nodiscard]] bool FilterDrawPasses() const;
    void PostPacket(PacketKind kind, const Pipeline* pipeline, bool is_indexed, u32 index_offset,
                    const IndirectDraw* indirect);
    static void RunPacket(void* context, const u8* data, u32 size);
    void NotePendingWrites(const Pipeline* pipeline);

    void BindBuffers(const Shader::Info& stage, Shader::Backend::Bindings& binding,
                     Shader::PushData& push_data);
    void BindTextures(const Shader::Info& stage, Shader::Backend::Bindings& binding);
    /// sotcport: copies the GPU-written dwords of a flattened user data block (see
    /// Shader::Info::SrtPatch) on the GPU.
    void PatchFlatUserData(VideoCore::StreamBuffer& flatbuf, u64 offset, u32 size,
                           std::span<const Shader::Info::SrtPatch> patches);
    bool BindResources(const Pipeline* pipeline);

    struct IndexBinding {
        const VideoCore::Buffer* buffer;
        u64 offset;
        vk::IndexType type;
    };

    void BindVertexBuffers(const GraphicsPipeline* pipeline);
    IndexBinding ObtainIndexBuffer(u32 index_offset = 0);
    void BindIndexBuffer(const IndexBinding& index);

    void ResetBindings(bool is_compute);

    bool IsComputeMetaClear(const Pipeline* pipeline);
    bool IsComputeImageCopy(const Pipeline* pipeline);
    bool IsComputeImageClear(const Pipeline* pipeline);

private:
    friend class VideoCore::BufferCache;

    const Instance& instance;
    Scheduler& scheduler;
    Runtime& runtime;
    VideoCore::PageManager page_manager;
    VideoCore::BufferCache buffer_cache;
    VideoCore::TextureCache texture_cache;
    AmdGpu::Liverpool* liverpool;

    // sotcport (two-stage draw pipeline): the GPU state the current draw sees. The draw path
    // reads registers, the compute program and the render target size hints only through these:
    // on the draw recording thread (stage B) they return that thread's own copy, which matches
    // the draw being recorded; everywhere else, the command processor's live state.
    struct StageState;
    static inline thread_local const StageState* stage_state = nullptr;
    const AmdGpu::Regs& GpuRegs() const noexcept;
    const AmdGpu::ComputeProgram& CsRegs() const noexcept;
    const std::array<AmdGpu::CbDbExtent, AmdGpu::NUM_COLOR_BUFFERS>& CbExtents() const noexcept;
    const AmdGpu::CbDbExtent& DbExtent() const noexcept;

    Core::MemoryManager* memory;
    boost::icl::interval_set<VAddr> mapped_ranges;
    Common::SharedFirstMutex mapped_ranges_mutex;
    std::atomic<u64> mapped_generation{0}; ///< sotcport: changes with mapped_ranges (IsMapped)
    PipelineCache pipeline_cache;
    const bool host_markers_enabled;
    const bool guest_markers_enabled;

    struct ImageBinding {
        VideoCore::ImageId image_id;
        VideoCore::TextureCache::ImageDesc desc;
    };
    std::array<ImageBinding, Shader::NUM_IMAGES> image_bindings;
    std::array<ImageBinding, AmdGpu::NUM_COLOR_BUFFERS> cb_descs;
    ImageBinding db_desc;

    /// sotcport: the last render target found for each color slot and for depth, kept while the
    /// registers that describe it and the texture cache's registry stay the same (see
    /// PrepareRenderState). Most draws keep their targets, and finding them again was ~15% of
    /// recording an indirect draw.
    struct TargetCacheEntry {
        u64 generation = ~u64{0};
        std::array<u32, 48> key{}; ///< the register words the description is built from
        u32 key_words = 0;
        ImageBinding binding{};
    };
    std::array<TargetCacheEntry, AmdGpu::NUM_COLOR_BUFFERS> cb_target_cache;
    TargetCacheEntry db_target_cache;
    /// Finds the image for `binding.desc` (built by `build` on a miss), or keeps the last one.
    template <typename Build>
    VideoCore::ImageId FindTargetImage(TargetCacheEntry& entry, std::span<const u32> key,
                                       ImageBinding& binding, Build&& build);

    boost::container::static_vector<vk::DescriptorImageInfo, Shader::NUM_IMAGES> image_infos;
    boost::container::static_vector<vk::DescriptorBufferInfo, Shader::NUM_BUFFERS> buffer_infos;

    struct BoundBuffer {
        const VideoCore::Buffer* buffer;
        u64 offset;
        u32 size;
        vk::AccessFlags2 src_access;
    };
    std::vector<BoundBuffer> bound_buffers;
    std::vector<VideoCore::ImageId> bound_images;

    /// sotcport: the game streams its textures' finer mip levels in and out by what the GPU
    /// reports having sampled. Every frame it marks up to 256 textures in their T# (word 6:
    /// COUNTER_BANK_ID [19:12], LOD_HDW_CNT_EN [20]) and then asks for the counters with
    /// IT_GET_LOD_STATS (game code 0x53d0f0 marks them, 0x53d620 asks, 0x53cb10 reads). On real
    /// hardware the texture units count, per bank, the samples and the finest mip level they
    /// needed. Here: how many times a texture with counting enabled was bound per bank since the
    /// last request (draws and dispatches), counted on stage B. See WriteLodStats.
    std::array<u32, 256> lod_stats_binds{};
    void NoteLodStatsBind(const AmdGpu::Image& tsharp);
    void WriteLodStats(VAddr base, u32 size);

    /// sotcport: the bindings of the last draw, reused when the very next packet is a draw that
    /// binds exactly the same descriptors (see BindResources). Stage B only.
    struct BindMemo {
        u64 packet = ~u64{0}; ///< bind_packet of the draw that left it; valid for the next one
        const Pipeline* pipeline = nullptr;
        u64 texture_generation = 0;
        std::vector<u32> key;
        /// What has to be done again for every draw: the flattened user data (address 0) and
        /// small buffers the buffer cache streams (copied from guest memory each time).
        struct Dynamic {
            u32 info_index;   ///< buffer_infos entry
            u32 bound_index;  ///< bound_buffers entry, counted from BindResources' first one
            u32 push_binding; ///< push_data buffer offset slot
            u32 stage;        ///< flattened user data: index into the pipeline's stages
            VAddr address;
            u32 size;
        };
        std::vector<Dynamic> dynamics;
        std::vector<BoundBuffer> bound_buffers;        ///< added by BindResources
        std::vector<VideoCore::ImageId> bound_images; ///< added by BindResources
        std::vector<u8> lod_banks; ///< counter banks of its textures (NoteLodStatsBind)
        std::array<u8, Shader::NUM_BUFFERS> buf_offsets{};
    };
    BindMemo bind_memo;
    std::vector<u32> bind_key;
    u64 bind_packet = 0;         ///< counts stage B packets and drains (see BindMemo::packet)
    bool memo_recording = false; ///< BindResources is recording a BindMemo
    u32 memo_stage = 0;          ///< the stage BindBuffers is binding (for BindMemo::Dynamic)
    size_t memo_first_bound_buffer = 0; ///< bound_buffers size when BindResources started
    template <typename Visit>
    bool VisitBindKey(const Pipeline* pipeline, Visit&& visit);
    bool BindKeyMatches(const Pipeline* pipeline, bool& eligible);
    void BuildBindKey(const Pipeline* pipeline);
    void ReuseBindings(const Pipeline* pipeline);

    u32 set_write_index{};
    Pipeline::DescriptorWrites set_writes;
    Shader::PushData push_data;

    bool attachment_feedback_loop{};
    bool needs_barrier{};

    // sotcport (two-stage draw pipeline).
    std::unique_ptr<DrawPipe> draw_pipe;
    std::unique_ptr<StageState> pipe_state; ///< stage B's copy of the GPU state
    bool pipe_synced = false;               ///< pipe_state started from the live registers
    struct PendingWrite {
        VAddr begin, end;
        u64 position; ///< ring position after the packet that writes it
    };
    std::vector<PendingWrite> pending_writes; ///< stage A only
    VAddr pending_min = ~VAddr{0}, pending_max = 0;
    u64 pending_checks = 0;
    u64 pending_last = 0; ///< sotcport: newest position among pending_writes
};

} // namespace Vulkan
