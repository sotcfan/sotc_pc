// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/debug.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "diag/memory_types.h"
#include "core/memory.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/amdgpu/liverpool.h"
#include "core/libraries/kernel/process.h" // pm4_cmds.h needs it first (shim)
#include "video_core/amdgpu/pm4_cmds.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_draw_pipe.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_hle.h"
#include "video_core/texture_cache/image_view.h"
#include "video_core/texture_cache/texture_cache.h"
#include <unordered_set>
#include <xxhash.h>
#include "perf_temp.h" // sotcport TEMP
#include "diag/toggles.h" // sotcport: F7 diagnostics
#include "hle/kernel/gpu_memory.h"
#include "shader_recompiler/ir/passes/srt.h"

namespace Vulkan {

static Shader::PushData MakeUserData(const AmdGpu::Regs& regs) {
    // TODO(roamic): Add support for multiple viewports and geometry shaders when ViewportIndex
    // is encountered and implemented in the recompiler.
    Shader::PushData push_data{};
    push_data.xoffset = regs.viewport_control.xoffset_enable ? regs.viewports[0].xoffset : 0.f;
    push_data.xscale = regs.viewport_control.xscale_enable ? regs.viewports[0].xscale : 1.f;
    push_data.yoffset = regs.viewport_control.yoffset_enable ? regs.viewports[0].yoffset : 0.f;
    push_data.yscale = regs.viewport_control.yscale_enable ? regs.viewports[0].yscale : 1.f;
    return push_data;
}

namespace {

// sotcport: see Shader::PFN_SrtWalker. The walker read guest memory through the mirror, which
// may hold stale data where the GPU wrote: fetch those bytes like a CPU read fault would.
VideoCore::BufferCache* g_srt_buffer_cache = nullptr;
Rasterizer* g_srt_rasterizer = nullptr; // sotcport: two-stage draw pipeline

bool CheckSrtReads(Shader::Info& info, const u64* begin, const u64* end) {
    const u64 mirror = hle::kernel::GuestMirrorOffset();
    bool fetched = false;
    // sotcport: what each read is (Shader::SrtReadPointer). Data reads the CPU never looks at
    // (no pointer, no descriptor, nothing the walker computes offsets from) are left to the GPU
    // when the GPU wrote them or will write them in a packet still queued for stage B: copied
    // into the flattened user data on the GPU, in GPU order, right before the shader runs
    // (info.srt_patches, BindBuffers). The rest is fetched as before. Without a map that matches
    // the log (it always should), every read counts as one the CPU needs.
    // SOTC_NO_SRT_PATCHES=1 fetches everything (A/B measurements). Without the shadow copies
    // below this was slower (every walk of a never-downloaded table became a patch); with them
    // only data the GPU really computes is left to the GPU.
    static const bool no_patches = std::getenv("SOTC_NO_SRT_PATCHES") != nullptr;
    const std::span<const u16> map = info.srt_read_map;
    const bool use_map = !no_patches && !diag::On(diag::SrtPatchesOff) &&
                         map.size() == static_cast<size_t>(end - begin);
    // Reads come in runs from the same table: decide per 4 KiB page whether the GPU wrote any of
    // it, or may in a queued packet (cheap), and only then look at the exact bytes.
    constexpr u64 PageBits = 12;
    constexpr u64 PageSize = u64{1} << PageBits;
    u64 last_page = ~u64{0};
    bool page_pending = false;
    bool page_modified = false;
    for (const u64* read = begin; read != end; ++read) {
        const VAddr address = *read - mirror;
        diag::NoteMemoryAccess(diag::MemAccess::SrtRead, address, 8); // sotcport (SOTC_MEMORY_TYPES)
        if (address >> PageBits != last_page) {
            last_page = address >> PageBits;
            page_pending = g_srt_rasterizer->PendingWriteOverlaps(last_page << PageBits, PageSize);
            page_modified = g_srt_buffer_cache->IsRegionGpuModified(last_page << PageBits, PageSize);
        }
        if (!page_pending && !page_modified) {
            continue;
        }
        const u16 entry = use_map ? map[read - begin] : Shader::SrtReadPointer;
        const u32 dst_dw = entry & Shader::SrtReadDwordMask;
        if (entry != Shader::SrtReadPointer && !(entry & Shader::SrtReadCpuNeeded) &&
            !info.IsDescriptorDword(dst_dw)) {
            if ((page_pending && g_srt_rasterizer->PendingWriteOverlaps(address, sizeof(u32))) ||
                (page_modified && g_srt_buffer_cache->AreBytesGpuModified(address, sizeof(u32)))) {
                info.srt_patches.push_back({address, dst_dw, 0});
            }
            continue;
        }
        // The CPU needs these bytes. sotcport (two-stage draw pipeline): work still queued for
        // stage B may write here (a shader, or a fence the game uses to patch a table): stage B
        // runs dry first, and the walk is done again (fetched), since what it read may have been
        // stale. After that no packet in flight writes this page, so stage B can only take
        // ranges out of the exact set below (downloads), never add to it, and the check runs
        // without waiting for it (the set has its own lock); only an actual download drains.
        if (page_pending) {
            g_srt_rasterizer->DrainDrawPipe(DrawPipe::ReasonSrtPending);
            fetched = true;
            page_pending = false;
            page_modified = g_srt_buffer_cache->IsRegionGpuModified(last_page << PageBits, PageSize);
        }
        // SOTC_SRT_FETCH_PAGES=1: fetch the whole page whenever the GPU wrote any of it
        // (diagnostics: rules out reads the byte check would miss).
        static const bool whole_pages = std::getenv("SOTC_SRT_FETCH_PAGES") != nullptr;
        if (page_modified && whole_pages) {
            g_srt_rasterizer->DrainDrawPipe(DrawPipe::ReasonSrtGpuModified);
            g_srt_buffer_cache->ReadMemory(last_page << PageBits, PageSize, false, true);
            page_modified = false;
            fetched = true;
        } else if (page_modified && g_srt_buffer_cache->AreBytesGpuModified(address, 8)) {
            // sotcport (SOTC_TRACE_SRT_READBACKS=1, diagnostics): which walks still wait for the
            // GPU, sampled over the whole session.
            static const bool trace = std::getenv("SOTC_TRACE_SRT_READBACKS") != nullptr;
            static u32 traced = 0;
            const u64 before = trace ? *reinterpret_cast<const u64*>(*read) : 0;
            g_srt_rasterizer->DrainDrawPipe(DrawPipe::ReasonSrtGpuModified);
            perf::SetFaultSite(1); // sotcport TEMP: "0x1" marks SRT walker readbacks
            g_srt_buffer_cache->ReadMemory(address, 8, false, true);
            perf::SetFaultSite(0);
            fetched = true;
            if (trace && (traced++ % 97) == 0) { // a sample over the whole session
                LOG_INFO(Render,
                         "SRT readback: shader {:#x} read #{} of {} ({}) at {:#x}: {:#x} -> {:#x}",
                         Shader::SrtWalkProgramHash, read - begin, end - begin,
                         entry == Shader::SrtReadPointer   ? "pointer"
                         : entry & Shader::SrtReadCpuNeeded ? "offset"
                                                            : "descriptor",
                         address, before, *reinterpret_cast<const u64*>(*read));
            }
        }
    }
    perf::NoteSrtReads(end - begin, fetched); // sotcport TEMP
    perf::NoteSrtPatches(info.srt_patches.size()); // sotcport TEMP
    return fetched;
}

// sotcport: the image description of a T# (sizes, mip layout, tiling) is rebuilt for every
// texture of every draw, and the game binds the same few thousand T#s each frame. It only
// depends on the T# and these flags of the resource, so keep the last ones built. GPU command
// processor thread only.
using ImageDesc = VideoCore::TextureCache::ImageDesc;

// sotcport: also keeps the image FindImage found for the description (see FindTexture below):
// valid while the texture cache's registry stays the same (TextureCache::RegistryGeneration).
struct ImageDescEntry {
    AmdGpu::Image sharp;
    u8 flags;
    bool valid;
    ImageDesc desc;
    u64 found_generation = ~u64{0};
    VideoCore::ImageId found_image{};
    ImageDesc found_desc; ///< desc as FindImage left it (overlaps adjust the view)
};

ImageDescEntry* CachedImageEntry(const AmdGpu::Image& sharp, const Shader::ImageResource& resource) {
    // A 2-way set-associative table of 4096 with a cheaper multiplicative hash (bbport step 4)
    // was measured slower (3.64 -> 3.71 us per draw, 3 alternating runs each, 2026-10-11): the
    // T# hash is cheap and the bigger table costs more in cache misses.
    static std::array<ImageDescEntry, 1024> cache{};
    // SOTC_NO_IMAGE_DESC_CACHE=1 builds every description again (diagnostics).
    static const bool disabled = std::getenv("SOTC_NO_IMAGE_DESC_CACHE") != nullptr;
    if (disabled) {
        return nullptr;
    }
    const u8 flags = u8(resource.is_depth) | u8(resource.is_written) << 1 |
                     u8(resource.is_array) << 2;
    const u64 hash = XXH3_64bits(&sharp, sizeof(sharp)) + flags;
    ImageDescEntry& entry = cache[hash % cache.size()];
    if (!entry.valid || entry.flags != flags ||
        std::memcmp(&entry.sharp, &sharp, sizeof(sharp)) != 0) {
        entry.sharp = sharp;
        entry.flags = flags;
        entry.valid = true;
        entry.found_generation = ~u64{0};
        std::construct_at(&entry.desc, sharp, resource);
    }
    return &entry;
}

const ImageDesc& CachedImageDesc(const AmdGpu::Image& sharp, const Shader::ImageResource& resource) {
    if (auto* entry = CachedImageEntry(sharp, resource)) {
        return entry->desc;
    }
    alignas(ImageDesc) static thread_local std::byte scratch[sizeof(ImageDesc)];
    return *std::construct_at(reinterpret_cast<ImageDesc*>(scratch), sharp, resource);
}

} // Anonymous namespace

struct Rasterizer::StageState {
    AmdGpu::Regs regs;
    std::array<AmdGpu::CbDbExtent, AmdGpu::NUM_COLOR_BUFFERS> cb_extent;
    AmdGpu::CbDbExtent db_extent;
    AmdGpu::ComputeProgram cs_program;
};

const AmdGpu::Regs& Rasterizer::GpuRegs() const noexcept {
    return stage_state ? stage_state->regs : liverpool->regs;
}
const AmdGpu::ComputeProgram& Rasterizer::CsRegs() const noexcept {
    return stage_state ? stage_state->cs_program : liverpool->GetCsRegs();
}
const std::array<AmdGpu::CbDbExtent, AmdGpu::NUM_COLOR_BUFFERS>& Rasterizer::CbExtents()
    const noexcept {
    return stage_state ? stage_state->cb_extent : liverpool->last_cb_extent;
}
const AmdGpu::CbDbExtent& Rasterizer::DbExtent() const noexcept {
    return stage_state ? stage_state->db_extent : liverpool->last_db_extent;
}

Rasterizer::Rasterizer(const Instance& instance_, Scheduler& scheduler_, Runtime& runtime_,
                       AmdGpu::Liverpool* liverpool_)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_}, page_manager{this},
      buffer_cache{instance, scheduler, runtime, liverpool_, texture_cache, page_manager},
      texture_cache{instance, scheduler, runtime, liverpool_, buffer_cache, page_manager},
      liverpool{liverpool_}, memory{Core::Memory::Instance()},
      pipeline_cache{instance, scheduler, liverpool, buffer_cache.GetSparsePageShift()},
      host_markers_enabled{EmulatorSettings.IsVkHostMarkersEnabled()},
      guest_markers_enabled{EmulatorSettings.IsVkGuestMarkersEnabled()} {
    if (!EmulatorSettings.IsNullGPU()) {
        liverpool->BindRasterizer(this);
    }
    memory->SetRasterizer(this);
    g_srt_buffer_cache = &buffer_cache;
    g_srt_rasterizer = this;
    Shader::SrtReadCheck = CheckSrtReads;

    scheduler.SetSubmitCallback([this](Vulkan::SubmitInfo& info) {
        runtime.FlushBarriers();
        buffer_cache.SubmitPendingArenaBinds(info);
    });

    // sotcport: two-stage draw pipeline, SOTC_DRAW_PIPE=1 (off by default while it is new).
    // Host debug markers wrap each draw on the GPU command thread, so they keep it off.
    const char* pipe_env = std::getenv("SOTC_DRAW_PIPE");
    if (pipe_env && *pipe_env == '1' && !host_markers_enabled) {
        pipe_state = std::make_unique<StageState>();
        liverpool->pipe_owns_dirty = true;
        draw_pipe = std::make_unique<DrawPipe>(&Rasterizer::RunPacket, this);
        LOG_INFO(Render_Vulkan, "two-stage draw pipeline on");
    }
}

Rasterizer::~Rasterizer() {
    // The draw recording thread uses everything below: stop it first.
    draw_pipe.reset();
}

bool Rasterizer::FilterDraw() {
    const auto& regs = GpuRegs();
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::EliminateFastClear) {
        // Clears the render target if FCE is launched before any draws
        EliminateFastClear();
        return false;
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::FmaskDecompress) {
        // TODO: check for a valid MRT1 to promote the draw to the resolve pass.
        LOG_TRACE(Render_Vulkan, "FMask decompression pass skipped");
        ScopedMarkerInsert("FmaskDecompress");
        return false;
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Resolve) {
        LOG_TRACE(Render_Vulkan, "Resolve pass");
        Resolve();
        return false;
    }
    if (regs.primitive_type == AmdGpu::PrimitiveType::None) {
        LOG_TRACE(Render_Vulkan, "Primitive type 'None' skipped");
        ScopedMarkerInsert("PrimitiveTypeNone");
        return false;
    }

    const bool cb_disabled =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
    const auto depth_copy =
        regs.depth_render_override.force_z_dirty && regs.depth_render_override.force_z_valid &&
        regs.depth_buffer.DepthValid() && regs.depth_buffer.DepthWriteValid() &&
        regs.depth_buffer.DepthAddress() != regs.depth_buffer.DepthWriteAddress();
    const auto stencil_copy =
        regs.depth_render_override.force_stencil_dirty &&
        regs.depth_render_override.force_stencil_valid && regs.depth_buffer.StencilValid() &&
        regs.depth_buffer.StencilWriteValid() &&
        regs.depth_buffer.StencilAddress() != regs.depth_buffer.StencilWriteAddress();
    if (cb_disabled && (depth_copy || stencil_copy)) {
        // Games may disable color buffer and enable force depth/stencil dirty and valid to
        // do a copy from one depth-stencil surface to another, without a pixel shader.
        // We need to detect this case and perform the copy, otherwise it will have no effect.
        LOG_TRACE(Render_Vulkan, "Performing depth-stencil override copy");
        DepthStencilCopy(depth_copy, stencil_copy);
        return false;
    }

    return true;
}

template <typename Build>
VideoCore::ImageId Rasterizer::FindTargetImage(TargetCacheEntry& entry, std::span<const u32> key,
                                    ImageBinding& binding, Build&& build) {
    // SOTC_NO_TARGET_CACHE=1 looks every target up again (A/B measurements).
    static const bool disabled = std::getenv("SOTC_NO_TARGET_CACHE") != nullptr;
    ASSERT(key.size() <= entry.key.size());
    if (!disabled && !diag::On(diag::TargetCacheOff) &&
        entry.generation == texture_cache.RegistryGeneration() &&
        entry.key_words == key.size() &&
        std::equal(key.begin(), key.end(), entry.key.begin())) {
        binding.desc = entry.binding.desc;
        texture_cache.UseFoundImage(entry.binding.image_id);
        return entry.binding.image_id;
    }
    build();
    const VideoCore::ImageId image_id = texture_cache.FindImage(binding.desc);
    if (texture_cache.LastFindWasDirect()) {
        entry.generation = texture_cache.RegistryGeneration();
        std::ranges::copy(key, entry.key.begin());
        entry.key_words = static_cast<u32>(key.size());
        entry.binding = {image_id, binding.desc};
    } else {
        entry.generation = ~u64{0};
    }
    return image_id;
}

void Rasterizer::PrepareRenderState(const GraphicsPipeline* pipeline) {
    // Prefetch render targets to handle overlaps with bound textures (e.g. mipgen)
    const auto& key = pipeline->GetGraphicsKey();
    const auto& regs = GpuRegs();
    if (regs.color_control.degamma_enable) {
        LOG_WARNING(Render_Vulkan, "Color buffers require gamma correction");
    }

    const bool skip_cb_binding =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
    for (s32 cb = 0; cb < std::bit_width(key.mrt_mask); ++cb) {
        auto& [image_id, desc] = cb_descs[cb];
        const auto& col_buf = regs.color_buffers[cb];
        const u32 target_mask = regs.color_target_mask.GetMask(cb);
        if (skip_cb_binding || !col_buf || !target_mask || (key.mrt_mask & (1 << cb)) == 0) {
            image_id = {};
            continue;
        }
        const auto& hint = CbExtents()[cb];
        struct {
            AmdGpu::ColorBuffer buffer;
            AmdGpu::CbDbExtent hint;
        } key{col_buf, hint};
        image_id = bound_images.emplace_back(FindTargetImage(
            cb_target_cache[cb], {reinterpret_cast<const u32*>(&key), sizeof(key) / 4},
            cb_descs[cb], [&] { std::construct_at(&desc, col_buf, hint); }));
        auto& image = texture_cache.GetImage(image_id);
        image.binding.is_target = 1u;
    }

    if ((regs.depth_control.depth_enable && regs.depth_buffer.DepthValid()) ||
        (regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid())) {
        const auto htile_address = regs.depth_htile_data_base.GetAddress();
        const auto& hint = DbExtent();
        auto& [image_id, desc] = db_desc;
        struct {
            AmdGpu::DepthBuffer buffer;
            AmdGpu::DepthView view;
            AmdGpu::DepthControl control;
            u64 htile_address;
            AmdGpu::CbDbExtent hint;
            u32 pad;
        } key{regs.depth_buffer, regs.depth_view, regs.depth_control, htile_address, hint, 0};
        image_id = bound_images.emplace_back(FindTargetImage(
            db_target_cache, {reinterpret_cast<const u32*>(&key), sizeof(key) / 4}, db_desc, [&] {
                std::construct_at(&desc, regs.depth_buffer, regs.depth_view, regs.depth_control,
                                  htile_address, hint);
            }));
        auto& image = texture_cache.GetImage(image_id);
        image.binding.is_target = 1u;
    } else {
        db_desc.image_id = {};
    }
}

static std::pair<u32, u32> GetDrawOffsets(const AmdGpu::Regs& regs, const Shader::Info& info,
                                          const Shader::Gcn::FetchShaderData& fetch_shader) {
    u32 vertex_offset = regs.index_offset;
    u32 instance_offset = 0;
    if (!fetch_shader.Empty()) {
        if (vertex_offset == 0 && fetch_shader.vertex_offset_sgpr != -1) {
            vertex_offset = info.UserData()[fetch_shader.vertex_offset_sgpr];
        }
        if (fetch_shader.instance_offset_sgpr != -1) {
            instance_offset = info.UserData()[fetch_shader.instance_offset_sgpr];
        }
    }
    return {vertex_offset, instance_offset};
}

void Rasterizer::EliminateFastClear() {
    EliminateFastClear(GpuRegs().color_buffers[0], CbExtents()[0]);
}

void Rasterizer::EliminateFastClear(const AmdGpu::ColorBuffer& col_buf, AmdGpu::CbDbExtent hint) {
    if (!col_buf || !col_buf.info.fast_clear) {
        return;
    }
    VideoCore::TextureCache::ImageDesc desc(col_buf, hint);
    const auto image_id = texture_cache.FindImage(desc);
    const auto& image_view = texture_cache.FindRenderTarget(image_id, desc);
    if (!texture_cache.IsMetaCleared(col_buf.CmaskAddress(), col_buf.view.slice_start)) {
        return;
    }
    for (u32 slice = col_buf.view.slice_start; slice <= col_buf.view.slice_max; ++slice) {
        texture_cache.TouchMeta(col_buf.CmaskAddress(), slice, false);
    }
    auto& image = texture_cache.GetImage(image_id);
    const auto clear_value = LiverpoolToVK::ColorBufferClearValue(col_buf);

    ScopeMarkerBegin(fmt::format("EliminateFastClear:MRT={:#x}:M={:#x}", col_buf.Address(),
                                 col_buf.CmaskAddress()));
    runtime.ClearImage(&image, desc.view_info.range, clear_value);
    ScopeMarkerEnd();
}

bool Rasterizer::PostFilterDraw() {
    // sotcport (two-stage draw pipeline): FilterDraw's passes ran on this thread after stage B
    // ran dry, ~1,300 times a second while walking, all of them fast clear eliminations (one per
    // render target the game resolves). Those only need color buffer 0 and its extent: stage B
    // runs them in order instead. The passes that do nothing are skipped here.
    const auto& regs = GpuRegs();
    using Mode = AmdGpu::ColorControl::OperationMode;
    const auto mode = regs.color_control.mode;
    if (mode == Mode::FmaskDecompress || regs.primitive_type == AmdGpu::PrimitiveType::None) {
        return true;
    }
    if (mode != Mode::EliminateFastClear || !OnStageA()) {
        return false;
    }
    const auto& col_buf = regs.color_buffers[0];
    if (!col_buf || !col_buf.info.fast_clear) {
        return true;
    }
    struct Payload {
        AmdGpu::ColorBuffer col_buf;
        AmdGpu::CbDbExtent hint;
    };
    const Payload payload{col_buf, CbExtents()[0]};
    PostTask(
        [](Rasterizer& self, const u8* data) {
            const auto& task = *reinterpret_cast<const Payload*>(data);
            self.EliminateFastClear(task.col_buf, task.hint);
        },
        &payload, sizeof(payload));
    return true;
}

bool Rasterizer::FilterDrawPasses() const {
    const auto& regs = GpuRegs();
    using Mode = AmdGpu::ColorControl::OperationMode;
    const auto mode = regs.color_control.mode;
    if (mode == Mode::EliminateFastClear || mode == Mode::FmaskDecompress ||
        mode == Mode::Resolve || regs.primitive_type == AmdGpu::PrimitiveType::None) {
        return false;
    }
    const bool depth_copy =
        regs.depth_render_override.force_z_dirty && regs.depth_render_override.force_z_valid &&
        regs.depth_buffer.DepthValid() && regs.depth_buffer.DepthWriteValid() &&
        regs.depth_buffer.DepthAddress() != regs.depth_buffer.DepthWriteAddress();
    const bool stencil_copy =
        regs.depth_render_override.force_stencil_dirty &&
        regs.depth_render_override.force_stencil_valid && regs.depth_buffer.StencilValid() &&
        regs.depth_buffer.StencilWriteValid() &&
        regs.depth_buffer.StencilAddress() != regs.depth_buffer.StencilWriteAddress();
    return !(mode == Mode::Disable && (depth_copy || stencil_copy));
}

void Rasterizer::Draw(bool is_indexed, u32 index_offset) {
    perf::Scope perf_scope{perf::Draw}; // sotcport TEMP
    RENDERER_TRACE;

    if (UseDrawPipe()) {
        // sotcport: this thread only selects the pipeline and hands the draw to stage B. Draws
        // FilterDraw handles itself (fast clear elimination, resolve, depth copy) run here,
        // after stage B has run dry.
        if (!FilterDrawPasses()) {
            if (PostFilterDraw()) {
                return;
            }
            DrainDrawPipe(DrawPipe::ReasonFilterDraw);
            scheduler.PopPendingOperations();
            FilterDraw();
            return;
        }
        const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline();
        if (pipeline) {
            PostPacket(PacketKind::Draw, pipeline, is_indexed, index_offset, nullptr);
        }
        return;
    }

    scheduler.PopPendingOperations();

    if (!FilterDraw()) {
        return;
    }

    const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline();
    if (!pipeline) {
        return;
    }
    DrawRecord(pipeline, is_indexed, index_offset);
}

void Rasterizer::DrawRecord(const GraphicsPipeline* pipeline, bool is_indexed, u32 index_offset) {
    ++work_since_flush; // sotcport: see MaybeFlushEarly
    const auto& regs = GpuRegs();
    PrepareRenderState(pipeline);
    if (!BindResources(pipeline)) {
        return;
    }
    const auto state = BeginRendering(pipeline);

    IndexBinding index{};
    if (is_indexed) {
        index = ObtainIndexBuffer(index_offset);
    }
    BindVertexBuffers(pipeline);
    if (is_indexed) {
        BindIndexBuffer(index);
    }

    if (needs_barrier) {
        runtime.FlushBarriers();
    }

    pipeline->BindResources(set_writes, push_data);
    UpdateDynamicState(pipeline, is_indexed);
    scheduler.BeginRendering(state);

    const auto& vs_info = pipeline->GetStage(Shader::SwStage::Vertex);
    const auto& fetch_shader = pipeline->GetFetchShader();
    const auto [vertex_offset, instance_offset] = GetDrawOffsets(regs, vs_info, fetch_shader);

    // sotcport: recorded by value (threaded recording).
    scheduler.Record([handle = pipeline->Handle(), is_indexed, count = regs.num_indices,
                      instances = regs.num_instances.NumInstances(), vertex_offset,
                      instance_offset](vk::CommandBuffer cmdbuf) {
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, handle);
        if (is_indexed) {
            cmdbuf.drawIndexed(count, instances, 0, s32(vertex_offset), instance_offset);
        } else {
            cmdbuf.draw(count, instances, vertex_offset, instance_offset);
        }
    });
    DebugState.IncDrawCall();

    ResetBindings(false);
    scheduler.KickRecording();
}

void Rasterizer::DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 stride,
                              u32 max_count, VAddr count_address, u16 vertex_sgpr_offset,
                              u16 instance_sgpr_offset) {
    perf::Scope perf_scope{perf::Draw}; // sotcport TEMP
    RENDERER_TRACE;

    const DrawIndirectParams params = {
        .vertex_sgpr_offset = vertex_sgpr_offset,
        .instance_sgpr_offset = instance_sgpr_offset,
    };
    const IndirectDraw indirect = {
        .arg_address = arg_address,
        .count_address = count_address,
        .offset = offset,
        .stride = stride,
        .max_count = max_count,
        .pad = 0,
    };

    if (UseDrawPipe()) {
        if (!FilterDrawPasses()) {
            if (PostFilterDraw()) {
                return;
            }
            DrainDrawPipe(DrawPipe::ReasonFilterDraw);
            scheduler.PopPendingOperations();
            FilterDraw();
            return;
        }
        // sotcport: see StageAChain.
        static const bool no_chain = std::getenv("SOTC_NO_DRAW_CHAIN") != nullptr;
        auto& chain = stage_a_chain;
        if (!no_chain && chain.pipeline && chain.posted == posted_packets &&
            chain.epoch == liverpool->chain_epoch && chain.is_indexed == is_indexed &&
            chain.vertex_sgpr_offset == vertex_sgpr_offset &&
            chain.instance_sgpr_offset == instance_sgpr_offset && !RegsChangedSincePost()) {
            PostPacket(PacketKind::DrawIndirect, chain.pipeline, is_indexed, 0, &indirect);
            chain.posted = posted_packets;
            return;
        }
        chain.pipeline = nullptr;
        const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline(params);
        if (pipeline) {
            PostPacket(PacketKind::DrawIndirect, pipeline, is_indexed, 0, &indirect);
            chain = {pipeline,           liverpool->chain_epoch, posted_packets,
                     vertex_sgpr_offset, instance_sgpr_offset,   is_indexed};
        }
        return;
    }

    scheduler.PopPendingOperations();

    if (!FilterDraw()) {
        return;
    }

    const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline(params);
    if (!pipeline) {
        return;
    }
    DrawIndirectRecord(pipeline, is_indexed, indirect);
}

void Rasterizer::DrawIndirectRecord(const GraphicsPipeline* pipeline, bool is_indexed,
                                    const IndirectDraw& indirect) {
    ++work_since_flush; // sotcport: see MaybeFlushEarly
    if (TryChainIndirectDraw(pipeline, is_indexed, indirect)) {
        return;
    }
    const VAddr arg_address = indirect.arg_address;
    const VAddr count_address = indirect.count_address;
    const u32 offset = indirect.offset;
    const u32 stride = indirect.stride;
    const u32 max_count = indirect.max_count;

    PrepareRenderState(pipeline);
    if (!BindResources(pipeline)) {
        return;
    }
    const auto state = BeginRendering(pipeline);

    const auto size = stride * max_count;
    const auto [buffer, base] = buffer_cache.ObtainBuffer(arg_address + offset, size, false);
    needs_barrier |= runtime.IsBufferAccessed(buffer, base, size);
    bound_buffers.emplace_back(buffer, base, size, vk::AccessFlagBits2::eIndirectCommandRead);

    const VideoCore::Buffer* count_buffer;
    u64 count_offset;
    if (count_address != 0) {
        std::tie(count_buffer, count_offset) = buffer_cache.ObtainBuffer(count_address, 4, false);
        needs_barrier |= runtime.IsBufferAccessed(count_buffer, count_offset, 4);
        bound_buffers.emplace_back(count_buffer, count_offset, 4,
                                   vk::AccessFlagBits2::eIndirectCommandRead);
    }

    IndexBinding index{};
    if (is_indexed) {
        index = ObtainIndexBuffer();
    }
    BindVertexBuffers(pipeline);
    if (is_indexed) {
        BindIndexBuffer(index);
    }

    if (needs_barrier) {
        runtime.FlushBarriers();
    }

    pipeline->BindResources(set_writes, push_data);
    UpdateDynamicState(pipeline, is_indexed);
    scheduler.BeginRendering(state);

    if (is_indexed) {
        ASSERT(sizeof(VkDrawIndexedIndirectCommand) == stride);
    } else {
        ASSERT(sizeof(VkDrawIndirectCommand) == stride);
    }
    // sotcport: recorded by value (threaded recording).
    const vk::Buffer args = buffer->Handle();
    const vk::Buffer counts = count_address != 0 ? count_buffer->Handle() : vk::Buffer{};
    scheduler.Record([handle = pipeline->Handle(), is_indexed, args, base, counts, count_offset,
                      max_count, stride](vk::CommandBuffer cmdbuf) {
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, handle);
        if (is_indexed) {
            if (counts) {
                cmdbuf.drawIndexedIndirectCount(args, base, counts, count_offset, max_count,
                                                stride);
            } else {
                cmdbuf.drawIndexedIndirect(args, base, max_count, stride);
            }
        } else {
            if (counts) {
                cmdbuf.drawIndirectCount(args, base, counts, count_offset, max_count, stride);
            } else {
                cmdbuf.drawIndirect(args, base, max_count, stride);
            }
        }
    });
    DebugState.IncDrawCall();
    StartIndirectChain(pipeline, is_indexed, indirect);

    ResetBindings(false);
    scheduler.KickRecording();
}

void Rasterizer::StartIndirectChain(const GraphicsPipeline* pipeline, bool is_indexed,
                                    const IndirectDraw& indirect) {
    // A draw can be chained onto this one if its bindings can be kept as they are: the binding
    // memo holds them (read-only bindings that need nothing redone per draw but the flattened
    // user data, which a chained draw must have the same), see BindResources.
    auto& chain = indirect_chain;
    chain.packet = ~u64{0};
    if (!DrawPipe::OnStageB() || bind_memo.packet != bind_packet ||
        indirect.count_address != 0 || !scheduler.IsRendering()) {
        return;
    }
    chain.stage_bases.clear();
    chain.user_data.clear();
    for (u32 i = 0; i < Shader::Info::num_ud_snapshots; ++i) {
        const auto& snapshot = Shader::Info::ud_snapshots[i];
        if (!snapshot.patches.empty()) {
            return;
        }
        chain.stage_bases.push_back(snapshot.pgm_base);
        chain.stage_bases.push_back(snapshot.user_data.size());
        chain.stage_bases.push_back(snapshot.flat.size());
        chain.user_data.insert(chain.user_data.end(), snapshot.user_data.begin(),
                               snapshot.user_data.end());
        chain.user_data.insert(chain.user_data.end(), snapshot.flat.begin(), snapshot.flat.end());
    }
    chain.packet = bind_packet;
    chain.pipeline = pipeline;
    chain.record_serial = scheduler.RecordSerial();
    chain.tick = scheduler.CurrentTick();
    chain.args_end =
        indirect.arg_address + indirect.offset + u64{indirect.stride} * indirect.max_count;
    chain.stride = indirect.stride;
    chain.is_indexed = is_indexed;
}

bool Rasterizer::TryChainIndirectDraw(const GraphicsPipeline* pipeline, bool is_indexed,
                                      const IndirectDraw& indirect) {
    static const bool disabled = std::getenv("SOTC_NO_DRAW_CHAIN") != nullptr;
    auto& chain = indirect_chain;
    // The draw before this one started or continued a chain, nothing ran or was recorded since
    // (the command buffer, the render pass, the bound pipeline, descriptors and dynamic state
    // are still the ones it left), no register changed its value, and this draw's arguments
    // come right after its arguments.
    const VAddr args = indirect.arg_address + indirect.offset;
    if (disabled || !DrawPipe::OnStageB() || chain.packet + 1 != bind_packet ||
        packet_regs_changed ||
        chain.pipeline != pipeline || chain.is_indexed != is_indexed ||
        indirect.count_address != 0 || indirect.stride != chain.stride || args != chain.args_end ||
        chain.record_serial != scheduler.RecordSerial() || chain.tick != scheduler.CurrentTick() ||
        !scheduler.IsRendering() || bind_memo.packet + 1 != bind_packet ||
        bind_memo.texture_generation != texture_cache.RegistryGeneration()) {
        return false;
    }
    // The same shaders with the same user data.
    if (chain.stage_bases.size() != size_t{Shader::Info::num_ud_snapshots} * 3) {
        return false;
    }
    size_t at = 0;
    for (u32 i = 0; i < Shader::Info::num_ud_snapshots; ++i) {
        const auto& snapshot = Shader::Info::ud_snapshots[i];
        if (!snapshot.patches.empty() || chain.stage_bases[i * 3] != snapshot.pgm_base ||
            chain.stage_bases[i * 3 + 1] != snapshot.user_data.size() ||
            chain.stage_bases[i * 3 + 2] != snapshot.flat.size() ||
            !std::equal(snapshot.user_data.begin(), snapshot.user_data.end(),
                        chain.user_data.begin() + at) ||
            !std::equal(snapshot.flat.begin(), snapshot.flat.end(),
                        chain.user_data.begin() + at + snapshot.user_data.size())) {
            return false;
        }
        at += snapshot.user_data.size() + snapshot.flat.size();
    }
    // The arguments, as the full path binds them (a barrier needed first: take that path).
    const u32 size = indirect.stride * indirect.max_count;
    const auto [buffer, base] = buffer_cache.ObtainBuffer(args, size, false);
    if (runtime.IsBufferAccessed(buffer, base, size)) {
        return false;
    }
    bound_buffers.emplace_back(buffer, base, size, vk::AccessFlagBits2::eIndirectCommandRead);
    // What binding the textures again would have counted (see NoteLodStatsBind).
    for (const u8 bank : bind_memo.lod_banks) {
        ++lod_stats_binds[bank];
    }
    const vk::Buffer handle = buffer->Handle();
    const u32 max_count = indirect.max_count;
    const u32 stride = indirect.stride;
    scheduler.Record([is_indexed, handle, base, max_count, stride](vk::CommandBuffer cmdbuf) {
        if (is_indexed) {
            cmdbuf.drawIndexedIndirect(handle, base, max_count, stride);
        } else {
            cmdbuf.drawIndirect(handle, base, max_count, stride);
        }
    });
    DebugState.IncDrawCall();
    chain.packet = bind_packet;
    chain.record_serial = scheduler.RecordSerial();
    chain.args_end = args + size;
    // The bindings are still the memo's: a draw after this one may reuse them as well.
    bind_memo.packet = bind_packet;
    ResetBindings(false);
    scheduler.KickRecording();
    return true;
}

void Rasterizer::DispatchDirect() {
    perf::Scope perf_scope{perf::Dispatch}; // sotcport TEMP
    RENDERER_TRACE;

    if (!UseDrawPipe()) {
        scheduler.PopPendingOperations();
    }

    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline();
    if (!pipeline) {
        return;
    }
    const bool shadowed = ShadowCopyShader(pipeline); // sotcport
    if (shadowed && !shadow_verify_address) {
        // sotcport: the GPU gets the bytes the CPU computed instead of running the copy.
        PostShadowResult();
        return;
    }
    if (UseDrawPipe()) {
        next_packet_shadowed = shadowed;
        PostPacket(PacketKind::Dispatch, pipeline, false, 0, nullptr);
        next_packet_shadowed = false;
    } else {
        shadow_dispatch = shadowed;
        DispatchRecord(pipeline);
        shadow_dispatch = false;
    }
    if (shadowed && shadow_verify_address) {
        VerifyShadowCopy();
    }
}

void Rasterizer::PostShadowResult() {
    // A copy from the CPU's result (vkCmdUpdateBuffer) where the dispatch would have run: in
    // stream order on stage B, or right here without the two-stage pipeline. The GPU's memory
    // then holds exactly what the CPU computed and the SRT walker reads, by construction. When
    // the GPU ran the copy itself, its source could differ from the one the CPU read (an older
    // value the GPU's memory had not received yet), and the two copies of the destination
    // diverged: groups of objects blinked after the bridge cutscene before the last colossus
    // (2026-10-14), one copy in 755,000 at start-up in the forest (2026-10-19).
    if (!UseDrawPipe()) {
        WriteShadowResult(shadow_result_address, shadow_result);
        return;
    }
    static std::vector<u8> payload;
    const u32 bytes = static_cast<u32>(shadow_result.size() * sizeof(u32));
    payload.resize(sizeof(VAddr) + sizeof(u32) + bytes);
    std::memcpy(payload.data(), &shadow_result_address, sizeof(VAddr));
    std::memcpy(payload.data() + sizeof(VAddr), &bytes, sizeof(u32));
    std::memcpy(payload.data() + sizeof(VAddr) + sizeof(u32), shadow_result.data(), bytes);
    PostTask(
        [](Rasterizer& self, const u8* raw) {
            VAddr address;
            u32 bytes;
            std::memcpy(&address, raw, sizeof(VAddr));
            std::memcpy(&bytes, raw + sizeof(VAddr), sizeof(u32));
            // The payload sits in the ring, 8-byte aligned: the words can be read in place.
            const auto* words = reinterpret_cast<const u32*>(raw + sizeof(VAddr) + sizeof(u32));
            self.WriteShadowResult(address, {words, bytes / sizeof(u32)});
        },
        payload.data(), static_cast<u32>(payload.size()));
}

void Rasterizer::WriteShadowResult(VAddr address, std::span<const u32> words) {
    // Written, but not GPU-modified: CPU memory already holds the same bytes (shadowed).
    const auto [buffer, offset] =
        buffer_cache.ObtainBuffer(address, static_cast<u32>(words.size_bytes()), true, false,
                                  true);
    runtime.UpdateBuffer(buffer, offset, words);
}

void Rasterizer::VerifyShadowCopy() {
    // sotcport (SOTC_VERIFY_SHADOW_COPY=1, diagnostics): right after a sampled shadowed
    // dispatch, before anything later can write there, compares what the GPU wrote with what
    // the CPU computed (stage B records it, the GPU runs it, the arena is read back).
    static u32 verified = 0, mismatches = 0;
    DrainDrawPipe(DrawPipe::ReasonSrtPending);
    std::vector<u32> gpu(shadow_verify_expected.size());
    if (buffer_cache.PeekGpuMemory(shadow_verify_address, gpu.size() * 4,
                                   reinterpret_cast<u8*>(gpu.data()))) {
        ++verified;
        if (gpu != shadow_verify_expected) {
            ++mismatches;
            const auto first =
                std::ranges::mismatch(gpu, shadow_verify_expected).in1 - gpu.begin();
            u64 differing = 0;
            for (size_t i = 0; i < gpu.size(); ++i) {
                differing += gpu[i] != shadow_verify_expected[i];
            }
            // The source dword behind it, as the CPU and the GPU see it now.
            const VAddr src_dword =
                shadow_verify_src + (first % std::max<u32>(shadow_verify_period, 1)) * 4;
            u32 src_cpu = 0, src_gpu = 0;
            memory->CopySparseMemory(src_dword, reinterpret_cast<u8*>(&src_cpu), 4);
            const bool src_peeked =
                buffer_cache.PeekGpuMemory(src_dword, 4, reinterpret_cast<u8*>(&src_gpu));
            LOG_ERROR(Render,
                      "shadow copy at {:#x}: GPU {:#x} / CPU {:#x} at dword {} of {} ({} differ); "
                      "source dword {:#x}: CPU {:#x} / GPU {}; {}",
                      shadow_verify_address, gpu[first], shadow_verify_expected[first], first,
                      gpu.size(), differing, src_dword, src_cpu,
                      src_peeked ? fmt::format("{:#x}", src_gpu) : std::string{"?"},
                      shadow_verify_info);
            LOG_ERROR(Render, "  source page CPU-modified: {}, GPU-modified: {}; label writes in "
                              "the source:{}",
                      buffer_cache.IsRegionCpuModified(src_dword, 4),
                      buffer_cache.IsRegionGpuModified(src_dword, 4),
                      perf::DescribeLabelWrites(shadow_verify_src,
                                                shadow_verify_src + shadow_verify_bytes));
        }
        if (verified % 100 == 0) {
            LOG_INFO(Render, "shadow copies verified: {}, {} differed", verified, mismatches);
        }
    }
    shadow_verify_address = 0;
}

// sotcport: the game's buffer fill/copy helper, a compute shader with three formatted buffers:
// a control block {count, period}, a source and a destination, one dword per thread:
//     for i < min(count, destination records): dst[i] = src[i % period]   (0 past the source)
// The game uses it to copy tables of constants and descriptors that the CPU prepared, and later
// draws read those tables through their SRTs. Done on the GPU only, every such read had to wait
// for the GPU (a readback: drain, submit, wait, ~0.5 ms; ~1,000 per second, almost all of the
// SRT walker's readbacks in the benchmark scene). So the CPU computes the same bytes as the
// packet is processed, in packet order, and writes them to guest memory: the walker reads
// current data. The GPU's memory gets the same bytes in GPU order (PostShadowResult), without
// marking them GPU-modified: the copy itself no longer runs on the GPU.
// Only when the inputs are current in CPU memory (nothing GPU-written or still queued for stage
// B, not inside a cached image: images the GPU draws into are not tracked as GPU-modified
// buffers), the formats copy bits unchanged, and the dispatch has the shape the code relies on.
// +20% frame rate in the forest (2026-10-19). SOTC_NO_SHADOW_COPY=1 turns it off (A/B).
bool Rasterizer::ShadowCopyShader(const ComputePipeline* pipeline) {
    constexpr u64 FillCopyShaderHash = 0x50784801;
    // sotcport: off from 2026-10-14 to 2026-10-19: after the bridge cutscene before the last
    // colossus, groups of objects blinked with it on (the GPU's own copy could diverge from the
    // CPU's; now the GPU gets the CPU's bytes, see PostShadowResult).
    static const bool disabled = std::getenv("SOTC_NO_SHADOW_COPY") != nullptr;
    const auto& cs = pipeline->GetStage(Shader::SwStage::Compute);
    if (disabled || diag::On(diag::ShadowCopyOff) || cs.pgm_hash != FillCopyShaderHash) {
        return false;
    }
    // SOTC_VERIFY_SHADOW_COPY=<n> (diagnostics): every n-th one (1: all) is compared with what
    // the GPU writes (VerifyShadowCopy).
    static const u32 verify_interval = [] {
        const char* env = std::getenv("SOTC_VERIFY_SHADOW_COPY");
        return env ? std::max<u32>(1, static_cast<u32>(std::strtoul(env, nullptr, 10))) : 0u;
    }();
    static u32 verify_counter = 0;
    std::array<const Shader::BufferResource*, 3> descs{};
    u32 found = 0;
    for (const auto& desc : cs.buffers) {
        if (!desc.IsSpecial()) {
            if (found == descs.size()) {
                return false;
            }
            descs[found++] = &desc;
        }
    }
    if (found != descs.size() || descs[0]->is_written || descs[1]->is_written ||
        !descs[2]->is_written) {
        return false;
    }
    const AmdGpu::Buffer ctl = descs[0]->GetSharp(cs);
    const AmdGpu::Buffer src = descs[1]->GetSharp(cs);
    const AmdGpu::Buffer dst = descs[2]->GetSharp(cs);
    const auto copies_bits = [](const AmdGpu::Buffer& buffer) {
        const auto number = buffer.GetNumberFmt();
        return buffer.GetDataFmt() == AmdGpu::DataFormat::Format32 && buffer.GetStride() == 4 &&
               (number == AmdGpu::NumberFormat::Uint || number == AmdGpu::NumberFormat::Sint ||
                number == AmdGpu::NumberFormat::Float) &&
               AmdGpu::CompSwizzle(buffer.dst_sel_x) == AmdGpu::CompSwizzle::Red;
    };
    const auto& program = CsRegs();
    if (!copies_bits(src) || !copies_bits(dst) || ctl.GetSize() < 8 || program.dim_y != 1 ||
        program.dim_z != 1 || program.start_x != 0 || program.num_thread_x.full != 64 ||
        program.num_thread_y.full != 1 || program.num_thread_z.full != 1) {
        return false;
    }
    std::array<u32, 2> control{};
    const VAddr src_address = src.base_address;
    const VAddr dst_address = dst.base_address;
    const u64 src_bytes = u64{src.num_records} * 4;
    // Inputs current in CPU memory: nothing the GPU wrote, nothing queued that writes them.
    const auto current = [&](VAddr address, u64 size) {
        return size == 0 || (hle::kernel::ClampToMapped(address, size) == size &&
                             !PendingWriteOverlaps(address, size) &&
                             !buffer_cache.AreBytesGpuModified(address, size));
    };
    if (!current(ctl.base_address, sizeof(control)) || !current(src_address, src_bytes) ||
        texture_cache.FindImageFromRange(src_address, src_bytes, false)) {
        return false;
    }
    memory->CopySparseMemory(ctl.base_address, reinterpret_cast<u8*>(control.data()),
                             sizeof(control));
    const u64 threads = u64{program.dim_x} * 64;
    const u64 count = std::min<u64>({control[0], dst.num_records, threads});
    const u32 period = control[1];
    const u64 dst_bytes = count * 4;
    // Only small copies: the tables the SRT walker reads are a few KiB; big fills (clearing
    // buffers, up to hundreds of KiB) would only cost CPU time here.
    constexpr u64 MaxShadowBytes = 16_KB;
    if (count == 0 || period == 0 || dst_bytes > MaxShadowBytes || dst_address % 4 != 0 ||
        hle::kernel::ClampToMapped(dst_address, dst_bytes) != dst_bytes ||
        (dst_address < src_address + src_bytes && src_address < dst_address + dst_bytes)) {
        return false;
    }
    // Writes still queued for stage B to the destination come first in GPU order: let stage B
    // record them (they mark what they write), then those marks are dropped for the bytes this
    // copy overwrites.
    if (PendingWriteOverlaps(dst_address, dst_bytes)) {
        DrainDrawPipe(DrawPipe::ReasonSrtPending);
    }
    buffer_cache.ForgetGpuWrites(dst_address, dst_bytes);
    static std::vector<u32> source, result;
    source.resize(src.num_records);
    memory->CopySparseMemory(src_address, reinterpret_cast<u8*>(source.data()), src_bytes);
    result.resize(count);
    for (u64 i = 0; i < count; ++i) {
        const u64 index = i % period;
        result[i] = index < source.size() ? source[index] : 0;
    }
    memory->TryWriteBacking(reinterpret_cast<void*>(dst_address), result.data(), dst_bytes);
    shadow_result_address = dst_address;
    shadow_result.assign(result.begin(), result.end());
    perf::NoteShadowCopy(dst_bytes); // sotcport TEMP
    if (verify_interval && (verify_counter++ % verify_interval) == 0) {
        shadow_verify_address = dst_address;
        shadow_verify_expected.assign(result.begin(), result.end());
        shadow_verify_src = src_address;
        shadow_verify_period = period;
        shadow_verify_bytes = src_bytes;
        shadow_verify_info = fmt::format(
            "control {:#x} (count {}, period {}), source {:#x} ({} bytes{}), dispatch {}x64",
            ctl.base_address, control[0], period, src_address, src_bytes,
            texture_cache.FindImageFromRange(src_address, src_bytes, false)
                ? ", inside a cached image"
                : "",
            program.dim_x);
    }
    return true;
}

void Rasterizer::MaybeFlushEarly() {
    // sotcport: recorded work used to reach the GPU only at the game's own submissions (~40 per
    // second, thousands of draws each): until then the GPU sat idle, and every readback in
    // between (hundreds per second) submitted the whole backlog and waited for the GPU to run it
    // from the start, on the GPU command thread. Submitting every few hundred draws and
    // dispatches keeps the GPU working alongside, so a readback only waits for the tail.
    // Called at dispatches, where rendering has ended anyway (no render pass is split).
    // SOTC_FLUSH_EVERY=<n> sets the interval, 0 turns it off.
    static const u32 interval = [] {
        const char* env = std::getenv("SOTC_FLUSH_EVERY");
        return env ? static_cast<u32>(std::strtoul(env, nullptr, 10)) : 256u;
    }();
    if (interval && ++work_since_flush >= interval) {
        work_since_flush = 0;
        scheduler.Flush();
    }
}

void Rasterizer::DispatchRecord(const ComputePipeline* pipeline) {
    MaybeFlushEarly();
    const auto& cs_program = CsRegs();
    const auto& cs = pipeline->GetStage(Shader::SwStage::Compute);
    if (ExecuteShaderHLE(cs, GpuRegs(), cs_program, *this)) {
        return;
    }

    if (!BindResources(pipeline)) {
        return;
    }

    if (needs_barrier) {
        runtime.FlushBarriers();
    }

    scheduler.EndRendering();
    pipeline->BindResources(set_writes, push_data);

    // sotcport: recorded by value (threaded recording).
    scheduler.Record([handle = pipeline->Handle(), x = cs_program.dim_x, y = cs_program.dim_y,
                      z = cs_program.dim_z](vk::CommandBuffer cmdbuf) {
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, handle);
        cmdbuf.dispatch(x, y, z);
    });
    DebugState.IncDispatch();
    if (bind_memo.packet == bind_packet) {
        bind_memo.record_serial = scheduler.RecordSerial(); // sotcport: see BindMemo::compute
    }

    ResetBindings(true);
    scheduler.KickRecording();
}

void Rasterizer::DispatchIndirect(VAddr address, u32 offset, u32 size) {
    perf::Scope perf_scope{perf::Dispatch}; // sotcport TEMP
    RENDERER_TRACE;

    if (!UseDrawPipe()) {
        scheduler.PopPendingOperations();
    }

    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline();
    if (!pipeline) {
        return;
    }
    if (UseDrawPipe()) {
        // The argument buffer travels like an indirect draw's (offset/stride/max_count unused).
        const IndirectDraw indirect = {
            .arg_address = address,
            .count_address = 0,
            .offset = offset,
            .stride = size,
            .max_count = 1,
            .pad = 0,
        };
        PostPacket(PacketKind::DispatchIndirect, pipeline, false, 0, &indirect);
        return;
    }
    DispatchIndirectRecord(pipeline, address, offset, size);
}

void Rasterizer::DispatchIndirectRecord(const ComputePipeline* pipeline, VAddr address, u32 offset,
                                        u32 size) {
    MaybeFlushEarly();
    if (!BindResources(pipeline)) {
        return;
    }

    const auto [buffer, base] = buffer_cache.ObtainBuffer(address + offset, size, false);
    needs_barrier |= runtime.IsBufferAccessed(buffer, base, size);
    bound_buffers.emplace_back(buffer, base, size, vk::AccessFlagBits2::eIndirectCommandRead);

    if (needs_barrier) {
        runtime.FlushBarriers();
    }

    scheduler.EndRendering();
    pipeline->BindResources(set_writes, push_data);

    // sotcport: recorded by value (threaded recording).
    scheduler.Record([handle = pipeline->Handle(), args = buffer->Handle(),
                      base](vk::CommandBuffer cmdbuf) {
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, handle);
        cmdbuf.dispatchIndirect(args, base);
    });
    DebugState.IncDispatch();
    if (bind_memo.packet == bind_packet) {
        bind_memo.record_serial = scheduler.RecordSerial(); // sotcport: see BindMemo::compute
    }

    ResetBindings(true);
    scheduler.KickRecording();
}

// ---- sotcport: two-stage draw pipeline (vk_draw_pipe.h) ----
//
// Packet layout (8-byte aligned pieces):
//   PacketHeader
//   u16 block indices [num_blocks]                     register blocks written since the last
//   u32 words [num_blocks * RegDirty::BlockWords]      packet, with their current values
//   per stage: PacketStage, u32 user data [ud_size], u32 flattened user data [flat_size]
//   ComputeProgram                                     dispatches only (the queue's cs_state)
//   IndirectDraw                                       indirect draws and dispatches only
//   Regs                                               verification packets only
// A task (PacketKind::Task, events run in stream order) is a TaskHeader and its payload instead.

namespace {

struct PacketHeader {
    u32 kind;
    u8 is_indexed;
    u8 regs_reset;  ///< ClearState since the last packet: defaults before the blocks
    u8 full_regs;   ///< the whole register file follows the header (first packet)
    u8 verify;      ///< the live register file is appended for comparison
    u8 shadowed;    ///< sotcport: a dispatch whose writes the CPU computed (ShadowCopyShader)
    u8 game_sync;   ///< sotcport: the game synchronized since the last packet (GuestWritesPending)
    const void* pipeline;
    u32 index_offset;
    u16 num_blocks;
    u16 num_stages;
    std::array<AmdGpu::CbDbExtent, AmdGpu::NUM_COLOR_BUFFERS> cb_extent;
    AmdGpu::CbDbExtent db_extent;
};

/// A PacketKind::Task packet: the header, then the payload (copied).
struct TaskHeader {
    u32 kind;
    u32 payload_size;
    void* run; ///< Rasterizer::TaskFn
};

struct PacketStage {
    const Shader::Info* info;
    VAddr pgm_base;
    u32 ud_size;
    u32 flat_size;
    u32 num_patches; ///< sotcport: Shader::Info::SrtPatch entries after the user data
    u32 pad;
};

constexpr u32 AlignPacket(u32 size) {
    return (size + 7) & ~7u;
}

u32 PipeVerifyInterval() {
    static const u32 interval = [] {
        const char* env = std::getenv("SOTC_PIPE_VERIFY");
        return env ? static_cast<u32>(std::strtoul(env, nullptr, 10)) : 0u;
    }();
    return interval;
}

} // namespace

bool Rasterizer::OnStageA() const {
    return std::this_thread::get_id() == liverpool->GetGpuCommandProcessorThread();
}

bool Rasterizer::IsGpuSideThread() const {
    return OnStageA() || DrawPipe::OnStageB();
}

void Rasterizer::ReleaseStageA() {
    if (draw_pipe) {
        draw_pipe->ReleaseStageA();
    }
}

bool Rasterizer::RunOnStageB(std::function<void()> work) {
    // SOTC_READBACK_ON_A=1: the command processor does it after draining (A/B measurements).
    static const bool on_a = std::getenv("SOTC_READBACK_ON_A") != nullptr;
    if (!draw_pipe || on_a) {
        return false;
    }
    if (DrawPipe::OnStageB()) {
        work();
    } else {
        draw_pipe->RunUrgent([this, &work] {
            Scheduler::pipe_busy.store(&scheduler, std::memory_order_relaxed); // TEMP check
            work();
            Scheduler::pipe_busy.store(nullptr, std::memory_order_relaxed);
        });
    }
    return true;
}

void Rasterizer::DrainDrawPipe(u32 reason) {
    if (draw_pipe && !DrawPipe::OnStageB() && OnStageA()) {
        draw_pipe->Drain(reason);
        // sotcport: stage A may record work of its own now (readbacks, uploads): bindings kept
        // by stage B for its next draw are no longer known to be current (see BindMemo), nor is
        // the pipeline kept for stage A's next indirect draw (see StageAChain).
        ++bind_packet;
        ++posted_packets;
    }
}

void Rasterizer::NotePendingWrites(const Pipeline* pipeline) {
    // What this draw will write that stage A could read before stage B has marked it: storage
    // buffers and storage images, from the descriptors stage A just flattened.
    const auto note = [&](VAddr address, u64 size) { NotePendingWrite(address, size); };
    for (const auto* stage : pipeline->GetStages()) {
        if (!stage) {
            continue;
        }
        for (const auto& desc : stage->buffers) {
            // sotcport: what a shadowed dispatch writes is already in CPU memory.
            if (!desc.is_written || desc.IsSpecial() || next_packet_shadowed) {
                continue;
            }
            const auto vsharp = desc.GetSharp(*stage);
            note(vsharp.base_address, vsharp.GetSize());
        }
        for (const auto& desc : stage->images) {
            if (!desc.is_written) {
                continue;
            }
            const auto tsharp = desc.GetSharp(*stage);
            const VideoCore::ImageInfo info{tsharp, desc};
            note(info.guest_address, info.guest_size);
        }
    }
}

void Rasterizer::NotePendingWrite(VAddr address, u64 size) {
    // Written by the packet about to be committed: pending until stage B has passed it.
    if (address == 0 || size == 0) {
        return;
    }
    // sotcport: past the start of the packet about to be committed: Reached() only once stage B
    // has finished it (it was its start, so a write counted as done while stage B was still
    // running the packet that writes, before it marked the bytes).
    const u64 position = draw_pipe->Head() + 1;
    pending_last = std::max(pending_last, position);
    const VAddr end = address + size;
    for (auto& write : pending_writes) {
        if (address <= write.end && write.begin <= end) {
            write.begin = std::min(write.begin, address);
            write.end = std::max(write.end, end);
            write.position = position;
            pending_min = std::min(pending_min, write.begin);
            pending_max = std::max(pending_max, write.end);
            return;
        }
    }
    pending_writes.push_back({address, end, position});
    pending_min = std::min(pending_min, address);
    pending_max = std::max(pending_max, end);
}

bool Rasterizer::PendingWriteOverlaps(VAddr address, u64 size) {
    if (!draw_pipe || pending_writes.empty() || address >= pending_max ||
        address + size <= pending_min) {
        return false;
    }
    // sotcport: stage B past the newest of them: none is pending (one check, no scan; the
    // scans were ~7% of this thread while walking).
    if (draw_pipe->Reached(pending_last)) {
        pending_writes.clear();
        pending_min = ~VAddr{0};
        pending_max = 0;
        return false;
    }
    // Now and then drop the writes stage B has passed: the bounds shrink.
    if ((++pending_checks & 63) == 0) {
        std::erase_if(pending_writes,
                      [&](const PendingWrite& write) { return draw_pipe->Reached(write.position); });
        pending_min = ~VAddr{0};
        pending_max = 0;
        for (const auto& write : pending_writes) {
            pending_min = std::min(pending_min, write.begin);
            pending_max = std::max(pending_max, write.end);
        }
    }
    const VAddr end = address + size;
    for (const auto& write : pending_writes) {
        if (address < write.end && write.begin < end && !draw_pipe->Reached(write.position)) {
            return true;
        }
    }
    return false;
}

void Rasterizer::PostPacket(PacketKind kind, const Pipeline* pipeline, bool is_indexed,
                            u32 index_offset, const IndirectDraw* indirect) {
    auto& dirty = liverpool->reg_dirty;
    const bool full = !pipe_synced;
    pipe_synced = true;

    std::array<u16, AmdGpu::RegDirty::NumBlocks> blocks;
    u32 num_blocks = 0;
    if (!full) {
        for (size_t block = dirty.blocks._Find_first(); block < dirty.blocks.size();
             block = dirty.blocks._Find_next(block)) {
            blocks[num_blocks++] = static_cast<u16>(block);
        }
    }
    const auto stages = pipeline->GetStages();
    u32 num_stages = 0;
    u32 size = sizeof(PacketHeader) + AlignPacket(num_blocks * sizeof(u16)) +
               num_blocks * AmdGpu::RegDirty::BlockWords * sizeof(u32);
    if (full) {
        size += sizeof(AmdGpu::Regs);
    }
    for (const auto* stage : stages) {
        if (stage) {
            ++num_stages;
            size += sizeof(PacketStage) +
                    AlignPacket(u32(stage->user_data.size() + stage->flattened_ud_buf.size()) *
                                sizeof(u32)) +
                    u32(stage->srt_patches.size() * sizeof(Shader::Info::SrtPatch));
        }
    }
    const bool dispatch = kind == PacketKind::Dispatch || kind == PacketKind::DispatchIndirect;
    if (dispatch) {
        size += AlignPacket(sizeof(AmdGpu::ComputeProgram));
    }
    if (indirect) {
        size += AlignPacket(sizeof(IndirectDraw));
    }
    const u32 interval = PipeVerifyInterval();
    const bool verify =
        interval && (DrawPipe::packets.load(std::memory_order_relaxed) % interval) == 0;
    if (verify) {
        size += sizeof(AmdGpu::Regs);
    }

    NotePendingWrites(pipeline);

    u8* out = draw_pipe->Begin(size);
    u8* const start = out;
    auto& header = *reinterpret_cast<PacketHeader*>(out);
    header.kind = static_cast<u32>(kind);
    header.is_indexed = is_indexed;
    header.regs_reset = !full && dirty.reset;
    header.full_regs = full;
    header.verify = verify;
    header.shadowed = next_packet_shadowed;
    header.game_sync = std::exchange(liverpool->game_sync_pending, false);
    header.pipeline = pipeline;
    header.index_offset = index_offset;
    header.num_blocks = static_cast<u16>(num_blocks);
    header.num_stages = static_cast<u16>(num_stages);
    header.cb_extent = liverpool->last_cb_extent;
    header.db_extent = liverpool->last_db_extent;
    out += sizeof(PacketHeader);
    if (full) {
        std::memcpy(out, &liverpool->regs, sizeof(AmdGpu::Regs));
        out += sizeof(AmdGpu::Regs);
    }
    std::memcpy(out, blocks.data(), num_blocks * sizeof(u16));
    out += AlignPacket(num_blocks * sizeof(u16));
    for (u32 i = 0; i < num_blocks; ++i) {
        std::memcpy(out,
                    liverpool->regs.reg_array.data() + blocks[i] * AmdGpu::RegDirty::BlockWords,
                    AmdGpu::RegDirty::BlockWords * sizeof(u32));
        out += AmdGpu::RegDirty::BlockWords * sizeof(u32);
    }
    // sotcport: stage A's copy of what stage B now has (see RegsChangedSincePost).
    if (full || !posted_regs) {
        if (!posted_regs) {
            posted_regs = std::make_unique<AmdGpu::Regs>();
        }
        std::memcpy(posted_regs.get(), &liverpool->regs, sizeof(AmdGpu::Regs));
    } else {
        if (header.regs_reset) {
            posted_regs->SetDefaults();
        }
        for (u32 i = 0; i < num_blocks; ++i) {
            const u32 word = blocks[i] * AmdGpu::RegDirty::BlockWords;
            std::memcpy(posted_regs->reg_array.data() + word,
                        liverpool->regs.reg_array.data() + word,
                        AmdGpu::RegDirty::BlockWords * sizeof(u32));
        }
    }
    posted_cb_extent = liverpool->last_cb_extent;
    posted_db_extent = liverpool->last_db_extent;
    ++posted_packets;
    for (const auto* stage : stages) {
        if (!stage) {
            continue;
        }
        auto& stage_header = *reinterpret_cast<PacketStage*>(out);
        stage_header = {stage,
                        stage->pgm_base,
                        static_cast<u32>(stage->user_data.size()),
                        static_cast<u32>(stage->flattened_ud_buf.size()),
                        static_cast<u32>(stage->srt_patches.size()),
                        0};
        out += sizeof(PacketStage);
        std::memcpy(out, stage->user_data.data(), stage->user_data.size_bytes());
        std::memcpy(out + stage->user_data.size_bytes(), stage->flattened_ud_buf.data(),
                    stage->flattened_ud_buf.size() * sizeof(u32));
        out += AlignPacket(u32(stage->user_data.size() + stage->flattened_ud_buf.size()) *
                           sizeof(u32));
        const u32 patch_bytes = u32(stage->srt_patches.size() * sizeof(Shader::Info::SrtPatch));
        std::memcpy(out, stage->srt_patches.data(), patch_bytes);
        out += patch_bytes;
    }
    if (dispatch) {
        std::memcpy(out, &liverpool->GetCsRegs(), sizeof(AmdGpu::ComputeProgram));
        out += AlignPacket(sizeof(AmdGpu::ComputeProgram));
    }
    if (indirect) {
        std::memcpy(out, indirect, sizeof(IndirectDraw));
        out += AlignPacket(sizeof(IndirectDraw));
    }
    if (verify) {
        std::memcpy(out, &liverpool->regs, sizeof(AmdGpu::Regs));
        out += sizeof(AmdGpu::Regs);
    }
    dirty.Clear();
    draw_pipe->Commit(static_cast<u32>(out - start));
}

bool Rasterizer::RegsChangedSincePost() const {
    const auto& dirty = liverpool->reg_dirty;
    if (!pipe_synced || dirty.reset || !posted_regs ||
        std::memcmp(&posted_cb_extent, &liverpool->last_cb_extent, sizeof(posted_cb_extent)) != 0 ||
        std::memcmp(&posted_db_extent, &liverpool->last_db_extent, sizeof(posted_db_extent)) != 0) {
        return true;
    }
    for (size_t block = dirty.blocks._Find_first(); block < dirty.blocks.size();
         block = dirty.blocks._Find_next(block)) {
        const u32 word = static_cast<u32>(block) * AmdGpu::RegDirty::BlockWords;
        if (std::memcmp(posted_regs->reg_array.data() + word,
                        liverpool->regs.reg_array.data() + word,
                        AmdGpu::RegDirty::BlockWords * sizeof(u32)) != 0) {
            return true;
        }
    }
    return false;
}

void Rasterizer::PostTask(TaskFn run, const void* payload, u32 payload_size) {
    ++posted_packets; // sotcport: see StageAChain
    const u32 size = sizeof(TaskHeader) + AlignPacket(payload_size);
    u8* out = draw_pipe->Begin(size);
    auto& header = *reinterpret_cast<TaskHeader*>(out);
    header.kind = static_cast<u32>(PacketKind::Task);
    header.payload_size = payload_size;
    header.run = reinterpret_cast<void*>(run);
    std::memcpy(out + sizeof(TaskHeader), payload, payload_size);
    draw_pipe->Commit(size);
}

bool Rasterizer::PostFencesOnStageB() const {
    // SOTC_PIPE_SYNC_FENCES=1: stage A drains and signals fences itself (A/B measurements).
    static const bool sync_fences = std::getenv("SOTC_PIPE_SYNC_FENCES") != nullptr;
    return draw_pipe && !sync_fences && OnStageA();
}

void Rasterizer::NotePendingFence(VAddr address, AmdGpu::DataSelect data_sel, u64 data,
                                  u32 num_bytes) {
    // Call before posting the fence's packet. What it writes is pending like a draw's writes;
    // a known value can also satisfy waits before stage B signals it (after posting: Head()).
    if (data_sel == AmdGpu::DataSelect::None) {
        return;
    }
    NotePendingWrite(address, num_bytes);
    std::erase_if(pending_fences,
                  [&](const PendingFence& fence) { return draw_pipe->Reached(fence.position); });
    if (data_sel == AmdGpu::DataSelect::Data32Low) {
        pending_fences.push_back({address, data & 0xffffffffu, 4, 0});
    } else if (data_sel == AmdGpu::DataSelect::Data64) {
        pending_fences.push_back({address, data, 8, 0});
    }
}

bool Rasterizer::PostCommand(Common::UniqueFunction<void>& callback) {
    // SOTC_PIPE_COMMANDS_ON_B=1 (experiment): see Liverpool::ProcessCommands.
    static const bool enabled = std::getenv("SOTC_PIPE_COMMANDS_ON_B") != nullptr;
    if (!enabled || !draw_pipe || !OnStageA()) {
        return false;
    }
    auto* moved = new Common::UniqueFunction<void>(std::move(callback));
    PostTask(
        [](Rasterizer&, const u8* payload) {
            Common::UniqueFunction<void>* command;
            std::memcpy(&command, payload, sizeof(command));
            (*command)();
            delete command;
        },
        &moved, sizeof(moved));
    return true;
}

bool Rasterizer::PostEopFence(const AmdGpu::PM4CmdEventWriteEop& packet) {
    if (!PostFencesOnStageB()) {
        return false;
    }
    NotePendingFence(reinterpret_cast<VAddr>(packet.Address<u8>()), packet.data_sel,
                     packet.DataQWord(), sizeof(u64));
    PostTask(
        [](Rasterizer& self, const u8* payload) {
            const auto& eop = *reinterpret_cast<const AmdGpu::PM4CmdEventWriteEop*>(payload);
            self.texture_cache.ProcessDownloadImages(); // what OnFence does on stage A
            AmdGpu::Liverpool::SignalEopFence(eop);
        },
        &packet, sizeof(packet));
    if (!pending_fences.empty() && pending_fences.back().position == 0) {
        pending_fences.back().position = draw_pipe->Head();
    }
    return true;
}

void Rasterizer::NoteLodStatsBind(const AmdGpu::Image& tsharp) {
    const u8 bank = static_cast<u8>(tsharp.counter_bank_id);
    ++lod_stats_binds[bank];
    if (memo_recording) {
        bind_memo.lod_banks.push_back(bank);
    }
}

void Rasterizer::GetLodStats(VAddr base, u32 size) {
    // SOTC_LOD_STATS=1 answers the requests (WriteLodStats); without it the buffer is left as
    // the game cleared it, as before (every texture then reads as needing its finest level).
    static const bool enabled = std::getenv("SOTC_LOD_STATS") != nullptr;
    if (!enabled && !diag::On(diag::LodStatsOn)) {
        return;
    }
    if (draw_pipe && OnStageA()) {
        // The counts are taken by stage B as it binds; the label the game writes next is
        // signalled by stage B after this (or after stage B is drained).
        struct Request {
            VAddr base;
            u32 size;
        } const request{base, size};
        PostTask(
            [](Rasterizer& self, const u8* payload) {
                Request r;
                std::memcpy(&r, payload, sizeof(r));
                self.WriteLodStats(r.base, r.size);
            },
            &request, sizeof(request));
        return;
    }
    WriteLodStats(base, size);
}

void Rasterizer::WriteLodStats(VAddr base, u32 size) {
    // The buffer the game reads (0x53cb10): after a 0x40 byte header (unused by the game), two
    // slots of 256 u64 counters, one per bank, which the game reads alternately (the packet
    // carries the buffer size minus 0x20). Each counter: bits [59:56] the finest mip level
    // sampled (0xf: not sampled at all), bits [55:32] how many samples (used as the texture's
    // priority). Without the texture units' counts, a texture bound since the last request is
    // reported as needing its finest level (0) with its bind count; one never bound, as not
    // sampled, so the game can drop its finer levels. Both slots get the same counters.
    std::array<u64, 256> counters;
    u32 banks_bound = 0;
    for (u32 bank = 0; bank < counters.size(); ++bank) {
        const u32 binds = lod_stats_binds[bank];
        counters[bank] = binds ? u64{std::min<u32>(binds, 0xffffff)} << 32 : u64{0xf} << 56;
        banks_bound += binds ? 1 : 0;
    }
    lod_stats_binds.fill(0);
    // With SOTC_PERF, every 600th request (~10 s): how many banks had a texture bound.
    static u32 requests = 0;
    if (perf::Enabled() && ++requests % 600 == 0) {
        LOG_INFO(Render_Vulkan, "LOD statistics: {} of 256 counter banks bound", banks_bound);
    }
    constexpr u32 Header = 0x40;
    constexpr u32 SlotBytes = sizeof(counters);
    auto* memory = Core::Memory::Instance();
    for (u32 slot = 0; slot < 2 && Header + (slot + 1) * SlotBytes <= size + 0x20; ++slot) {
        void* dst = reinterpret_cast<void*>(base + Header + slot * SlotBytes);
        if (!memory->TryWriteBacking(dst, counters.data(), SlotBytes)) {
            static bool logged = false;
            if (!std::exchange(logged, true)) {
                LOG_WARNING(Render_Vulkan, "LOD statistics buffer {:#x} not writable", base);
            }
            return;
        }
    }
}

bool Rasterizer::PostWriteData(VAddr address, const void* data, u32 bytes) {
    // WRITE_DATA (about one per frame) used to drain stage B before writing: ~60 ms a second of
    // the GPU command thread in the forest. Like a fence, it is written by stage B in order with
    // the packets; until then its bytes count as a pending write for stage A. Written with a
    // plain copy, as before: a page a cache watches faults and is invalidated (handled inline on
    // stage B). SOTC_SYNC_WRITE_DATA=1 drains as before (A/B).
    static const bool sync = std::getenv("SOTC_SYNC_WRITE_DATA") != nullptr;
    constexpr u32 MaxBytes = 256;
    if (sync || bytes > MaxBytes || !PostFencesOnStageB()) {
        return false;
    }
    NotePendingWrite(address, bytes);
    struct Payload {
        VAddr address;
        u32 bytes;
        u8 data[MaxBytes];
    } payload;
    payload.address = address;
    payload.bytes = bytes;
    std::memcpy(payload.data, data, bytes);
    PostTask(
        [](Rasterizer& self, const u8* raw) {
            Payload copy;
            std::memcpy(&copy, raw, offsetof(Payload, data));
            std::memcpy(copy.data, raw + offsetof(Payload, data), copy.bytes);
            self.texture_cache.ProcessDownloadImages(); // what OnFence does on stage A
            std::memcpy(reinterpret_cast<void*>(copy.address), copy.data, copy.bytes);
        },
        &payload, static_cast<u32>(offsetof(Payload, data) + bytes));
    return true;
}

bool Rasterizer::PostEosFence(const AmdGpu::PM4CmdEventWriteEos& packet) {
    if (!PostFencesOnStageB()) {
        return false;
    }
    NotePendingFence(reinterpret_cast<VAddr>(packet.Address()), AmdGpu::DataSelect::Data32Low,
                     packet.DataDWord(), sizeof(u32));
    PostTask(
        [](Rasterizer& self, const u8* payload) {
            const auto& eos = *reinterpret_cast<const AmdGpu::PM4CmdEventWriteEos*>(payload);
            self.texture_cache.ProcessDownloadImages(); // what OnFence does on stage A
            eos.SignalFence([](void* address, u64 data, u32 num_bytes) {
                auto* memory = Core::Memory::Instance();
                ASSERT(memory->TryWriteBacking(address, &data, num_bytes));
                perf::NoteLabelWrite(reinterpret_cast<u64>(address), num_bytes); // sotcport TEMP
            });
        },
        &packet, sizeof(packet));
    if (!pending_fences.empty() && pending_fences.back().position == 0) {
        pending_fences.back().position = draw_pipe->Head();
    }
    return true;
}

bool Rasterizer::PostReleaseMem(const AmdGpu::PM4CmdReleaseMem& packet, u32 pipe_id) {
    if (!PostFencesOnStageB()) {
        return false;
    }
    const u32 num_bytes = packet.data_sel == AmdGpu::DataSelect::GdsMemStore
                              ? packet.num_dw * u32{sizeof(u32)}
                              : u32{sizeof(u64)};
    NotePendingFence(packet.Address<VAddr>(), packet.data_sel, packet.DataQWord(), num_bytes);
    struct Payload {
        AmdGpu::PM4CmdReleaseMem packet;
        u32 pipe_id;
    };
    const Payload payload{packet, pipe_id};
    PostTask(
        [](Rasterizer& self, const u8* data) {
            const auto& task = *reinterpret_cast<const Payload*>(data);
            self.texture_cache.ProcessDownloadImages(); // what OnFence does on stage A
            self.liverpool->SignalReleaseMem(task.packet, task.pipe_id);
        },
        &payload, sizeof(payload));
    if (!pending_fences.empty() && pending_fences.back().position == 0) {
        pending_fences.back().position = draw_pipe->Head();
    }
    return true;
}

bool Rasterizer::PendingFenceMeets(const AmdGpu::PM4CmdWaitRegMem& wait) {
    // SOTC_PIPE_NO_FENCE_WAITS=1: waits drain instead (diagnostics).
    static const bool disabled = std::getenv("SOTC_PIPE_NO_FENCE_WAITS") != nullptr;
    if (disabled || !draw_pipe || pending_fences.empty() ||
        wait.mem_space != AmdGpu::PM4CmdWaitRegMem::MemSpace::Memory) {
        return false;
    }
    const auto address = reinterpret_cast<VAddr>(wait.Address());
    for (const auto& fence : pending_fences) {
        if (draw_pipe->Reached(fence.position)) {
            continue; // signalled: the memory itself answers
        }
        if (address == fence.address) {
            if (wait.TestValue(static_cast<u32>(fence.value))) {
                return true;
            }
        } else if (fence.num_bytes == 8 && address == fence.address + 4) {
            if (wait.TestValue(static_cast<u32>(fence.value >> 32))) {
                return true;
            }
        }
    }
    return false;
}

void Rasterizer::RunPacket(void* context, const u8* data, u32 size) {
    auto& self = *static_cast<Rasterizer*>(context);
    ++self.bind_packet; // sotcport: see BindMemo
    if (reinterpret_cast<const TaskHeader*>(data)->kind == static_cast<u32>(PacketKind::Task)) {
        const auto& task = *reinterpret_cast<const TaskHeader*>(data);
        Scheduler::pipe_busy.store(&self.scheduler, std::memory_order_relaxed); // TEMP
        self.scheduler.PopPendingOperations();
        reinterpret_cast<TaskFn>(task.run)(self, data + sizeof(TaskHeader));
        Scheduler::pipe_busy.store(nullptr, std::memory_order_relaxed); // TEMP
        return;
    }
    auto& state = *self.pipe_state;
    const auto& header = *reinterpret_cast<const PacketHeader*>(data);
    const u8* in = data + sizeof(PacketHeader);
    // sotcport: whether a register's value changes (blocks are sent when written, often with
    // the same values again): see TryChainIndirectDraw.
    self.packet_regs_changed = header.full_regs || header.regs_reset;
    if (header.full_regs) {
        std::memcpy(&state.regs, in, sizeof(AmdGpu::Regs));
        in += sizeof(AmdGpu::Regs);
    }
    if (header.regs_reset) {
        state.regs.SetDefaults();
    }
    const auto* blocks = reinterpret_cast<const u16*>(in);
    in += AlignPacket(header.num_blocks * sizeof(u16));
    for (u32 i = 0; i < header.num_blocks; ++i) {
        u32* const block = state.regs.reg_array.data() + blocks[i] * AmdGpu::RegDirty::BlockWords;
        constexpr u32 block_bytes = AmdGpu::RegDirty::BlockWords * sizeof(u32);
        if (!self.packet_regs_changed && std::memcmp(block, in, block_bytes) != 0) {
            self.packet_regs_changed = true;
        }
        std::memcpy(block, in, block_bytes);
        in += block_bytes;
    }
    ASSERT(header.num_stages <= Shader::Info::MaxUdSnapshots);
    for (u32 i = 0; i < header.num_stages; ++i) {
        const auto& stage = *reinterpret_cast<const PacketStage*>(in);
        in += sizeof(PacketStage);
        const auto* words = reinterpret_cast<const u32*>(in);
        in += AlignPacket((stage.ud_size + stage.flat_size) * sizeof(u32));
        const auto* patches = reinterpret_cast<const Shader::Info::SrtPatch*>(in);
        in += stage.num_patches * sizeof(Shader::Info::SrtPatch);
        Shader::Info::ud_snapshots[i] = {stage.info,
                                         {words, stage.ud_size},
                                         {words + stage.ud_size, stage.flat_size},
                                         stage.pgm_base,
                                         {patches, stage.num_patches}};
    }
    Shader::Info::num_ud_snapshots = header.num_stages;
    const auto kind = static_cast<PacketKind>(header.kind);
    if (kind == PacketKind::Dispatch || kind == PacketKind::DispatchIndirect) {
        std::memcpy(&state.cs_program, in, sizeof(AmdGpu::ComputeProgram));
        in += AlignPacket(sizeof(AmdGpu::ComputeProgram));
    }
    const IndirectDraw* indirect = nullptr;
    if (kind == PacketKind::DrawIndirect || kind == PacketKind::DispatchIndirect) {
        indirect = reinterpret_cast<const IndirectDraw*>(in);
        in += AlignPacket(sizeof(IndirectDraw));
    }
    if (header.verify) {
        // SOTC_PIPE_VERIFY=N: every Nth packet carries the live register file; stage B's copy
        // must be identical.
        const auto* live = reinterpret_cast<const AmdGpu::Regs*>(in);
        in += sizeof(AmdGpu::Regs);
        static u64 checks = 0, mismatches = 0;
        ++checks;
        if (std::memcmp(state.regs.reg_array.data(), live->reg_array.data(),
                        sizeof(live->reg_array)) != 0) {
            if (++mismatches <= 20) {
                for (u32 w = 0; w < AmdGpu::Regs::NumRegs; ++w) {
                    if (state.regs.reg_array[w] != live->reg_array[w]) {
                        LOG_ERROR(Render_Vulkan,
                                  "PIPE VERIFY: stage B register copy differs at word {:#x} "
                                  "({:#x} vs {:#x})",
                                  w, state.regs.reg_array[w], live->reg_array[w]);
                        break;
                    }
                }
            }
        }
        if (checks % 2000 == 0) {
            LOG_INFO(Render_Vulkan, "PIPE VERIFY: {} packet checks, {} mismatches", checks,
                     mismatches);
        }
    }
    ASSERT(static_cast<u32>(in - data) == size);
    // sotcport: the target extents found by the command processor count as registers here.
    if (!self.packet_regs_changed &&
        (std::memcmp(&state.cb_extent, &header.cb_extent, sizeof(state.cb_extent)) != 0 ||
         std::memcmp(&state.db_extent, &header.db_extent, sizeof(state.db_extent)) != 0)) {
        self.packet_regs_changed = true;
    }
    state.cb_extent = header.cb_extent;
    state.db_extent = header.db_extent;
    stage_state = &state;

    Scheduler::pipe_busy.store(&self.scheduler, std::memory_order_relaxed); // TEMP
    self.scheduler.PopPendingOperations();
    // sotcport: where the game synchronizes, everything its shaders wrote before is made
    // visible to what comes after (see Runtime::GuestWritesPending).
    if (header.game_sync && self.runtime.GuestWritesPending()) {
        self.runtime.FlushBarriers();
    }
    switch (kind) {
    case PacketKind::Draw:
        self.DrawRecord(static_cast<const GraphicsPipeline*>(header.pipeline), header.is_indexed,
                        header.index_offset);
        break;
    case PacketKind::DrawIndirect:
        self.DrawIndirectRecord(static_cast<const GraphicsPipeline*>(header.pipeline),
                                header.is_indexed, *indirect);
        break;
    case PacketKind::Dispatch:
        self.shadow_dispatch = header.shadowed;
        self.DispatchRecord(static_cast<const ComputePipeline*>(header.pipeline));
        self.shadow_dispatch = false;
        break;
    case PacketKind::DispatchIndirect:
        self.DispatchIndirectRecord(static_cast<const ComputePipeline*>(header.pipeline),
                                    indirect->arg_address, indirect->offset, indirect->stride);
        break;
    }
    Shader::Info::num_ud_snapshots = 0;
    Scheduler::pipe_busy.store(nullptr, std::memory_order_relaxed); // TEMP
}

u64 Rasterizer::Flush() {
    DrainDrawPipe(DrawPipe::ReasonFlush); // sotcport: two-stage draw pipeline
    const u64 current_tick = scheduler.CurrentTick();
    SubmitInfo info{};
    scheduler.Flush(info);
    return current_tick;
}

void Rasterizer::Finish() {
    DrainDrawPipe(DrawPipe::ReasonFinish); // sotcport: two-stage draw pipeline
    scheduler.Finish();
}

void Rasterizer::OnSubmit() {
    DrainDrawPipe(DrawPipe::ReasonSubmit); // sotcport: two-stage draw pipeline
    buffer_cache.TickFrame();
    texture_cache.ProcessDownloadImages();
    texture_cache.RunGarbageCollector();
    runtime.TickFrame();
}

void Rasterizer::OnFence(u32 reason) {
    DrainDrawPipe(reason ? reason : DrawPipe::ReasonFence); // sotcport: two-stage draw pipeline
    texture_cache.ProcessDownloadImages();
}

bool Rasterizer::BindResources(const Pipeline* pipeline) {
    // sotcport: ~70% of the draws bind exactly what the draw before them did (same pipeline,
    // buffers, textures, samplers and targets; only the flattened user data differs). Then the
    // descriptors of the previous draw are reused and only the per-draw parts are redone
    // (ReuseBindings), as long as nothing ran in between (BindMemo::packet) and the texture
    // cache's registry is the same. SOTC_NO_BIND_MEMO=1 binds everything every time.
    bool memo_eligible = false;
    // sotcport: dispatches reuse bindings too, under stricter conditions (see BindMemo::compute):
    // ~half of the game's dispatches bind exactly what the one before them bound.
    const bool compute = pipeline->IsCompute();
    const u64 cpu_marks = compute ? buffer_cache.CpuMarks() : 0;
    const u64 gpu_removals = compute ? buffer_cache.GpuModifiedRemovals() : 0;
    const auto& cs = CsRegs();
    const std::array<u32, 3> dims = compute ? std::array<u32, 3>{cs.dim_x, cs.dim_y, cs.dim_z}
                                            : std::array<u32, 3>{};
    const bool memo_valid =
        bind_memo.packet + 1 == bind_packet &&
        bind_memo.texture_generation == texture_cache.RegistryGeneration() &&
        bind_memo.compute == compute &&
        (!compute || (bind_memo.cpu_marks == cpu_marks && bind_memo.gpu_removals == gpu_removals &&
                      bind_memo.record_serial == scheduler.RecordSerial() &&
                      bind_memo.dims == dims));
    const bool memo_hit = BindKeyMatches(pipeline, memo_eligible) && memo_valid;
    if (memo_hit) {
        // The dispatch before, with the same bindings and sizes, was not one of the copies or
        // clears done another way below (else it would have left no memo).
        ReuseBindings(pipeline);
        bind_memo.packet = bind_packet;
        return true;
    }
    if (IsComputeImageCopy(pipeline) || IsComputeMetaClear(pipeline) ||
        IsComputeImageClear(pipeline)) {
        return false;
    }
    bind_memo.packet = ~u64{0};
    memo_recording = memo_eligible;
    bind_memo.dynamics.clear();
    bind_memo.lod_banks.clear();
    const size_t first_bound_buffer = bound_buffers.size();
    const size_t first_bound_image = bound_images.size();
    memo_first_bound_buffer = first_bound_buffer;

    set_write_index = 0;
    set_writes.clear();
    buffer_infos.clear();
    image_infos.clear();

    bool uses_dma = false;

    // Bind resource buffers and textures.
    Shader::Backend::Bindings binding{};
    push_data = MakeUserData(GpuRegs());
    memo_stage = 0;
    for (const auto* stage : pipeline->GetStages()) {
        if (!stage) {
            ++memo_stage;
            continue;
        }
        set_writes.resize(set_writes.size() + stage->buffers.size() + stage->images.size() +
                          stage->samplers.size());
        BindBuffers(*stage, binding, push_data);
        BindTextures(*stage, binding);
        uses_dma |= stage->uses_dma;
        ++memo_stage;
    }

    if (uses_dma) {
        buffer_cache.SynchronizeDmaBuffers();
    }

    if (memo_recording) {
        memo_recording = false;
        // Not with a texture that is also a target here, or bound as storage elsewhere: its
        // layout changes around the draw.
        bool reusable = true;
        for (size_t i = first_bound_image; i < bound_images.size(); ++i) {
            const auto& image = texture_cache.GetImage(bound_images[i]);
            reusable &= !image.binding.is_target && !image.binding.force_general;
        }
        if (reusable) {
            bind_memo.packet = bind_packet;
            bind_memo.pipeline = pipeline;
            bind_memo.texture_generation = texture_cache.RegistryGeneration();
            bind_memo.compute = compute;
            bind_memo.dims = dims;
            bind_memo.cpu_marks = cpu_marks;
            bind_memo.gpu_removals = gpu_removals;
            bind_memo.record_serial = ~u64{0}; // set once the dispatch is recorded
            // bind_memo.key holds this draw's key already (BindKeyMatches).
            bind_memo.bound_buffers.assign(bound_buffers.begin() + first_bound_buffer,
                                           bound_buffers.end());
            bind_memo.bound_images.assign(bound_images.begin() + first_bound_image,
                                          bound_images.end());
            bind_memo.buf_offsets = push_data.buf_offsets;
        }
    }
    return true;
}

template <typename Visit>
bool Rasterizer::VisitBindKey(const Pipeline* pipeline, Visit&& visit) {
    // What the bindings of a draw depend on besides the texture cache's registry: the pipeline,
    // the targets found for it, the border color base, and every descriptor (V#, T#, S#), in a
    // fixed order: `visit(words, count)` gets them piece by piece and returns false to stop.
    // False when the draw cannot reuse bindings: only draws whose bindings are all read-only and
    // need nothing done per draw but the flattened user data and streamed buffers (no GDS, user
    // clip planes, DMA, storage, SRT patches). ~63% of the eligible draws reuse their bindings.
    // Dispatches may write buffers (not images), see BindMemo::compute.
    static const bool disabled = std::getenv("SOTC_NO_BIND_MEMO") != nullptr;
    // sotcport: dispatches too (BindMemo::compute), but not those whose writes the CPU computes
    // (their buffers are not marked GPU-written: a different binding path).
    static const bool no_compute = std::getenv("SOTC_NO_COMPUTE_MEMO") != nullptr;
    const bool compute = pipeline->IsCompute();
    if (disabled || diag::On(diag::BindMemoOff) || !DrawPipe::OnStageB() ||
        (compute && (no_compute || shadow_dispatch))) {
        return false;
    }
    bool go_on = true;
    const auto add = [&](const auto& value) {
        static_assert(sizeof(value) % sizeof(u32) == 0);
        if (go_on) {
            go_on = visit(reinterpret_cast<const u32*>(&value), sizeof(value) / sizeof(u32));
        }
    };
    for (const auto& cb : cb_descs) {
        add(cb.image_id);
    }
    add(db_desc.image_id);
    add(GpuRegs().ta_bc_base);
    for (const auto* stage : pipeline->GetStages()) {
        if (!stage) {
            continue;
        }
        if (stage->uses_dma || !stage->SrtPatches().empty()) {
            return false;
        }
        for (const auto& desc : stage->buffers) {
            // A dispatch's written buffers are bound again as they were: the conditions of
            // BindMemo::compute keep their state the same.
            if (desc.is_written && !compute) {
                return false;
            }
            if (desc.IsSpecial()) {
                // The flattened user data is redone per draw; clip planes are a null descriptor
                // while user clip planes are off (else copied from registers per draw: not here);
                // the BDA page table and the fault buffer are fixed buffers.
                const auto type = desc.buffer_type;
                if (type == Shader::BufferType::ClipPlanes) {
                    if (GpuRegs().clipper_control.user_clip_plane_enable != 0) {
                        return false;
                    }
                } else if (type != Shader::BufferType::Flatbuf &&
                           type != Shader::BufferType::BdaPagetable &&
                           type != Shader::BufferType::FaultBuffer) {
                    return false;
                }
                continue;
            }
            if (go_on) {
                add(desc.GetSharp(*stage));
            }
        }
        for (const auto& desc : stage->images) {
            if (desc.is_written) {
                return false;
            }
            if (go_on) {
                add(desc.GetSharp(*stage));
            }
        }
        for (const auto& desc : stage->samplers) {
            if (go_on) {
                add(desc.GetSharp(*stage));
            }
        }
    }
    return true;
}

bool Rasterizer::BindKeyMatches(const Pipeline* pipeline, bool& eligible) {
    // Compared in place with the kept key: no key is built for a draw that reuses. From the
    // first difference on, the rest of this draw's key replaces the kept one's: the memo cannot
    // serve this draw anyway, and if this draw leaves a memo, its key is ready (building it in a
    // second pass, piece by piece, was ~2% of the draw recording thread, 2026-10-19).
    auto& key = bind_memo.key;
    size_t at = 0;
    bool match = bind_memo.pipeline == pipeline;
    if (!match) {
        key.clear();
    }
    eligible = VisitBindKey(pipeline, [&](const u32* words, size_t count) {
        if (match) {
            if (at + count <= key.size() && std::equal(words, words + count, key.begin() + at)) {
                at += count;
                return true;
            }
            match = false;
            key.resize(at);
        }
        for (size_t i = 0; i < count; ++i) {
            key.push_back(words[i]);
        }
        return true;
    });
    if (match && at != key.size()) {
        // This draw's key is a prefix of the kept one: not the same.
        match = false;
        key.resize(at);
    }
    return eligible && match;
}

void Rasterizer::ReuseBindings(const Pipeline* pipeline) {
    // The descriptor writes of the previous draw are still in set_writes, buffer_infos and
    // image_infos (cleared only by the next full BindResources). Redo what is per draw.
    push_data = MakeUserData(GpuRegs());
    push_data.buf_offsets = bind_memo.buf_offsets;
    const auto stages = pipeline->GetStages();
    const u64 alignment = instance.StorageMinAlignment();
    auto& stream = buffer_cache.GetStreamBuffer();
    const size_t first_bound_buffer = bound_buffers.size();
    bound_buffers.insert(bound_buffers.end(), bind_memo.bound_buffers.begin(),
                         bind_memo.bound_buffers.end());
    for (const auto& dynamic : bind_memo.dynamics) {
        auto& info = buffer_infos[dynamic.info_index];
        if (dynamic.address == 0) {
            const auto flat = stages[dynamic.stage]->FlatUserData();
            info.offset = stream.Copy(flat.data(), flat.size_bytes(), alignment);
            continue;
        }
        // As BufferCache::ObtainBuffer streams it, then BindBuffers aligns it.
        const auto [data, offset] = stream.Map(dynamic.size, instance.UniformMinAlignment());
        memory->CopySparseMemory(dynamic.address, data, dynamic.size);
        stream.Commit();
        const u64 offset_aligned = Common::AlignDown(offset, alignment);
        const u64 adjust = offset - offset_aligned;
        push_data.AddOffset(dynamic.push_binding, adjust);
        info.offset = offset_aligned;
        info.range = dynamic.size + adjust;
        bound_buffers[first_bound_buffer + dynamic.bound_index].offset = offset;
    }
    for (const auto image_id : bind_memo.bound_images) {
        bound_images.push_back(image_id);
        texture_cache.GetImage(image_id).binding.is_bound = 1u;
        texture_cache.UseFoundImage(image_id);
    }
    for (const u8 bank : bind_memo.lod_banks) {
        ++lod_stats_binds[bank];
    }
}

void Rasterizer::BindVertexBuffers(const GraphicsPipeline* pipeline) {
    const auto& regs = GpuRegs();
    VertexInputs<vk::VertexInputAttributeDescription2EXT> attributes;
    VertexInputs<vk::VertexInputBindingDescription2EXT> bindings;
    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT> divisors;
    VertexInputs<AmdGpu::Buffer> guest_buffers;
    pipeline->GetVertexInputs(attributes, bindings, divisors, guest_buffers,
                              regs.vgt_instance_step_rate_0, regs.vgt_instance_step_rate_1);

    // Obtaining buffers can submit the current command buffer, so state is recorded afterwards.
    const auto set_vertex_input = [&] {
        if (instance.IsVertexInputDynamicState()) {
            // sotcport: recorded by value (threaded recording).
            scheduler.Record([bindings, attributes](vk::CommandBuffer cmdbuf) {
                cmdbuf.setVertexInputEXT(bindings, attributes);
            });
        }
    };

    if (bindings.empty()) {
        // If there are no bindings, there is nothing further to do.
        set_vertex_input();
        return;
    }

    struct BufferRange {
        VAddr base_address;
        VAddr end_address;
        const VideoCore::Buffer* buffer;
        u64 offset;

        [[nodiscard]] size_t GetSize() const {
            return end_address - base_address;
        }
    };

    // Build list of ranges covering the requested buffers
    VertexInputs<BufferRange> ranges{};
    for (const auto& buffer : guest_buffers) {
        if (buffer.base_address != 0 && buffer.GetSize() > 0) {
            ranges.emplace_back(buffer.base_address, buffer.base_address + buffer.GetSize());
        }
    }

    // Merge connecting ranges together
    VertexInputs<BufferRange> ranges_merged{};
    if (!ranges.empty()) {
        std::ranges::sort(ranges, [](const BufferRange& lhv, const BufferRange& rhv) {
            return lhv.base_address < rhv.base_address;
        });
        ranges_merged.emplace_back(ranges[0]);
        for (auto range : ranges) {
            auto& prev_range = ranges_merged.back();
            if (prev_range.end_address < range.base_address) {
                ranges_merged.emplace_back(range);
            } else {
                prev_range.end_address = std::max(prev_range.end_address, range.end_address);
            }
        }
    }

    // Map buffers for merged ranges
    for (auto& range : ranges_merged) {
        const u64 size = memory->ClampRangeSize(range.base_address, range.GetSize());
        std::tie(range.buffer, range.offset) =
            buffer_cache.ObtainBuffer(range.base_address, size, false);
        needs_barrier |= runtime.IsBufferAccessed(range.buffer, range.offset, size);
        bound_buffers.emplace_back(range.buffer, range.offset, size,
                                   vk::AccessFlagBits2::eVertexAttributeRead);
    }

    // Bind vertex buffers
    VertexInputs<vk::Buffer> host_buffers;
    VertexInputs<vk::DeviceSize> host_offsets;
    VertexInputs<vk::DeviceSize> host_sizes;
    VertexInputs<vk::DeviceSize> host_strides;
    for (const auto& buffer : guest_buffers) {
        if (buffer.base_address != 0 && buffer.GetSize() > 0) {
            const auto host_buffer_info =
                std::ranges::find_if(ranges_merged, [&](const BufferRange& range) {
                    return buffer.base_address >= range.base_address &&
                           buffer.base_address < range.end_address;
                });
            ASSERT(host_buffer_info != ranges_merged.cend());
            host_buffers.emplace_back(host_buffer_info->buffer->Handle());
            host_offsets.push_back(host_buffer_info->offset + buffer.base_address -
                                   host_buffer_info->base_address);
        } else {
            host_buffers.emplace_back(VK_NULL_HANDLE);
            host_offsets.push_back(0);
        }
        host_sizes.push_back(buffer.GetSize());
        host_strides.push_back(buffer.GetStride());
    }

    set_vertex_input();

    // sotcport: recorded by value (threaded recording).
    const u32 num_buffers = static_cast<u32>(guest_buffers.size());
    scheduler.Record([dynamic = instance.IsVertexInputDynamicState(), num_buffers, host_buffers,
                      host_offsets, host_sizes, host_strides](vk::CommandBuffer cmdbuf) {
        if (dynamic) {
            cmdbuf.bindVertexBuffers(0, num_buffers, host_buffers.data(), host_offsets.data());
        } else {
            cmdbuf.bindVertexBuffers2(0, num_buffers, host_buffers.data(), host_offsets.data(),
                                      host_sizes.data(), host_strides.data());
        }
    });
}

Rasterizer::IndexBinding Rasterizer::ObtainIndexBuffer(u32 index_offset) {
    const auto& regs = GpuRegs();

    // Figure out index type and size.
    const bool is_index16 = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
    const vk::IndexType index_type = is_index16 ? vk::IndexType::eUint16 : vk::IndexType::eUint32;
    const u32 index_size = is_index16 ? sizeof(u16) : sizeof(u32);
    const VAddr index_address =
        regs.index_base_address.Address<VAddr>() + index_offset * index_size;

    // Bind index buffer.
    const u32 index_buffer_size = regs.num_indices * index_size;
    const auto [buffer, offset] =
        buffer_cache.ObtainBuffer(index_address, index_buffer_size, false);
    needs_barrier |= runtime.IsBufferAccessed(buffer, offset, index_buffer_size);
    bound_buffers.emplace_back(buffer, offset, index_buffer_size, vk::AccessFlagBits2::eIndexRead);
    return {buffer, offset, index_type};
}

void Rasterizer::BindIndexBuffer(const IndexBinding& index) {
    // sotcport: recorded by value (threaded recording).
    scheduler.Record([buffer = index.buffer->Handle(), offset = index.offset,
                      type = index.type](vk::CommandBuffer cmdbuf) {
        cmdbuf.bindIndexBuffer(buffer, offset, type);
    });
}

void Rasterizer::PatchFlatUserData(VideoCore::StreamBuffer& flatbuf, u64 offset, u32 size,
                                   std::span<const Shader::Info::SrtPatch> patches) {
    // sotcport: the walker left these dwords to the GPU (CheckSrtReads): the GPU wrote them, or
    // will in work recorded before this draw. Copy each from the GPU's copy of guest memory into
    // the flattened user data the CPU just wrote, in GPU order: as the console's shader would
    // read them itself. Copies cannot run inside rendering (Runtime::CopyBuffer ends it).
    boost::container::small_vector<vk::BufferCopy, 8> copies;
    const VideoCore::Buffer* source = nullptr;
    const auto copy_pending = [&] {
        if (!copies.empty()) {
            runtime.CopyBuffer(source, &flatbuf, copies);
            copies.clear();
        }
    };
    for (const auto& patch : patches) {
        const auto [buffer, buffer_offset] =
            buffer_cache.ObtainBuffer(patch.address, sizeof(u32), false);
        if (buffer != source) {
            copy_pending();
            source = buffer;
        }
        copies.push_back({buffer_offset, offset + u64{patch.dst_dw} * sizeof(u32), sizeof(u32)});
    }
    copy_pending();
    // The shader reads what the copies wrote.
    needs_barrier |= runtime.IsBufferAccessed(&flatbuf, offset, size);
    bound_buffers.emplace_back(&flatbuf, offset, size, vk::AccessFlagBits2::eShaderRead);
}

void Rasterizer::ResetBindings(bool is_compute) {
    for (auto& image_id : bound_images) {
        texture_cache.GetImage(image_id).binding = {};
    }
    for (const auto [buffer, offset, size, src_access, guest] : bound_buffers) {
        const auto dst_stage = is_compute ? vk::PipelineStageFlagBits2::eComputeShader
                                          : vk::PipelineStageFlagBits2::eAllGraphics;
        runtime.AccessBuffer(buffer, offset, size, dst_stage, src_access, guest);
    }
    bound_images.clear();
    bound_buffers.clear();
    needs_barrier = false;
}

bool Rasterizer::IsComputeMetaClear(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Most of the time when a metadata is updated with a shader it gets cleared. It means
    // we can skip the whole dispatch and update the tracked state instead. Also, it is not
    // intended to be consumed and in such rare cases (e.g. HTile introspection, CRAA) we
    // will need its full emulation anyways.
    const auto& info = pipeline->GetStage(Shader::SwStage::Compute);

    // Assume if a shader reads metadata, it is a copy shader.
    for (const auto& desc : info.buffers) {
        const VAddr address = desc.GetSharp(info).base_address;
        if (!desc.IsSpecial() && !desc.is_written && texture_cache.IsMeta(address)) {
            return false;
        }
    }

    // Metadata surfaces are tiled and thus need address calculation to be written properly.
    // If a shader wants to encode HTILE, for example, from a depth image it will have to compute
    // proper tile address from dispatch invocation id. This address calculation contains an xor
    // operation so use it as a heuristic for metadata writes that are probably not clears.
    if (!info.has_bitwise_xor) {
        // Assume if a shader writes metadata without address calculation, it is a clear shader.
        for (const auto& desc : info.buffers) {
            const VAddr address = desc.GetSharp(info).base_address;
            if (!desc.IsSpecial() && desc.is_written && texture_cache.ClearMeta(address)) {
                // Assume all slices were updates
                LOG_TRACE(Render_Vulkan, "Metadata update skipped");
                return true;
            }
        }
    }
    return false;
}

bool Rasterizer::IsComputeImageCopy(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Ensure shader only has 2 bound buffers
    const auto& cs_pgm = CsRegs();
    const auto& info = pipeline->GetStage(Shader::SwStage::Compute);
    if (cs_pgm.num_thread_x.full != 64 || info.buffers.size() != 2 || !info.images.empty()) {
        return false;
    }

    // Those 2 buffers must both be formatted. One must be source and another destination.
    const auto& desc0 = info.buffers[0];
    const auto& desc1 = info.buffers[1];
    if (!desc0.is_formatted || !desc1.is_formatted || desc0.is_written == desc1.is_written) {
        return false;
    }

    // Buffers must have the same size and each thread of the dispatch must copy 1 dword of data
    const AmdGpu::Buffer buf0 = desc0.GetSharp(info);
    const AmdGpu::Buffer buf1 = desc1.GetSharp(info);
    if (buf0.GetSize() != buf1.GetSize() || cs_pgm.dim_x != (buf0.GetSize() / 256)) {
        return false;
    }

    // Find images the buffer alias
    const auto image0_id = texture_cache.FindImageFromRange(buf0.base_address, buf0.GetSize());
    if (!image0_id) {
        return false;
    }
    const auto image1_id =
        texture_cache.FindImageFromRange(buf1.base_address, buf1.GetSize(), false);
    if (!image1_id) {
        return false;
    }

    // Image copy must be valid
    VideoCore::Image& image0 = texture_cache.GetImage(image0_id);
    VideoCore::Image& image1 = texture_cache.GetImage(image1_id);
    if (image0.info.guest_size != image1.info.guest_size ||
        image0.info.pitch != image1.info.pitch || image0.info.guest_size != buf0.GetSize() ||
        image0.info.num_bits != image1.info.num_bits) {
        return false;
    }

    // Perform image copy
    VideoCore::Image& src_image = desc0.is_written ? image1 : image0;
    VideoCore::Image& dst_image = desc0.is_written ? image0 : image1;
    runtime.CopyColorAndDepth(&src_image, &dst_image);
    return true;
}

bool Rasterizer::IsComputeImageClear(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Ensure shader only has 2 bound buffers
    const auto& cs_pgm = CsRegs();
    const auto& info = pipeline->GetStage(Shader::SwStage::Compute);
    if (cs_pgm.num_thread_x.full != 64 || info.buffers.size() != 2 || !info.images.empty()) {
        return false;
    }

    // From those 2 buffers, first must hold the clear vector and second the image being cleared
    const auto& desc0 = info.buffers[0];
    const auto& desc1 = info.buffers[1];
    if (desc0.is_formatted || !desc1.is_formatted || desc0.is_written || !desc1.is_written) {
        return false;
    }

    // First buffer must have size of vec4 and second the size of a single layer
    const AmdGpu::Buffer buf0 = desc0.GetSharp(info);
    const AmdGpu::Buffer buf1 = desc1.GetSharp(info);
    const u32 buf1_bpp = AmdGpu::NumBitsPerBlock(buf1.GetDataFmt());
    if (buf0.GetSize() != 16 || (cs_pgm.dim_x * 128ULL * (buf1_bpp / 8)) != buf1.GetSize()) {
        return false;
    }

    // Find image the buffer alias
    const auto image1_id =
        texture_cache.FindImageFromRange(buf1.base_address, buf1.GetSize(), false);
    if (!image1_id) {
        return false;
    }

    // Image clear must be valid
    VideoCore::Image& image1 = texture_cache.GetImage(image1_id);
    if (image1.info.guest_size != buf1.GetSize() || image1.info.num_bits != buf1_bpp ||
        image1.info.props.is_depth) {
        return false;
    }

    // Perform image clear
    const float* values = reinterpret_cast<float*>(buf0.base_address);
    const vk::ClearValue clear = {
        .color = {.float32 = std::array<float, 4>{values[0], values[1], values[2], values[3]}},
    };
    const VideoCore::SubresourceRange range = {
        .base =
            {
                .level = 0,
                .layer = 0,
            },
        .extent = image1.info.resources,
    };
    runtime.ClearImage(&image1, range, clear);
    return true;
}

void Rasterizer::BindBuffers(const Shader::Info& stage, Shader::Backend::Bindings& binding,
                             Shader::PushData& push_data) {
    const u64 alignment = instance.StorageMinAlignment();
    for (const auto& desc : stage.buffers) {
        const auto src_access =
            vk::AccessFlagBits2::eShaderRead |
            (desc.is_written ? vk::AccessFlagBits2::eShaderWrite : vk::AccessFlagBits2::eNone);
        if (desc.IsSpecial()) {
            if (desc.buffer_type == Shader::BufferType::GdsBuffer) {
                const auto* gds_buf = buffer_cache.GetGdsBuffer();
                buffer_infos.emplace_back(gds_buf->Handle(), 0, gds_buf->SizeBytes());
                needs_barrier |=
                    runtime.IsBufferAccessed(gds_buf, 0, gds_buf->SizeBytes(), desc.is_written);
                bound_buffers.emplace_back(gds_buf, 0, gds_buf->SizeBytes(), src_access);
            } else if (desc.buffer_type == Shader::BufferType::Flatbuf) {
                auto& vk_buffer = buffer_cache.GetStreamBuffer();
                const u32 ubo_size = stage.FlatUserData().size() * sizeof(u32);
                // sotcport: SOTC_TRACE_FLATBUF=<shader hash, hex> logs the flattened user data
                // a shader gets (diagnostics: the native renderer's validation compares it).
                static const u64 trace_flatbuf = [] {
                    const char* hash = std::getenv("SOTC_TRACE_FLATBUF");
                    return hash ? std::strtoull(hash, nullptr, 16) : 0ull;
                }();
                if (trace_flatbuf && (stage.pgm_hash & 0xffffffffull) == trace_flatbuf) {
                    std::string words;
                    for (const u32 word : stage.FlatUserData()) {
                        words += fmt::format(" {:08x}", word);
                    }
                    LOG_INFO(Render, "flatbuf of {:#x}:{}", stage.pgm_hash, words);
                }
                const u64 offset =
                    vk_buffer.Copy(stage.FlatUserData().data(), ubo_size, alignment);
                buffer_infos.emplace_back(vk_buffer.Handle(), offset, ubo_size);
                if (memo_recording) { // sotcport: see BindMemo
                    bind_memo.dynamics.push_back({static_cast<u32>(buffer_infos.size() - 1), 0,
                                                  0, memo_stage, 0, ubo_size});
                }
                // sotcport: dwords the GPU wrote come from the GPU (Shader::Info::SrtPatch).
                if (const auto patches = stage.SrtPatches(); !patches.empty()) {
                    PatchFlatUserData(vk_buffer, offset, ubo_size, patches);
                }
            } else if (desc.buffer_type == Shader::BufferType::ClipPlanes) {
                // Permutations compiled without enabled planes never read the buffer, so the
                // declared binding is satisfied with a null descriptor instead of a copy.
                if (GpuRegs().clipper_control.user_clip_plane_enable == 0) {
                    buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
                } else {
                    auto& vk_buffer = buffer_cache.GetStreamBuffer();
                    std::array<float, AmdGpu::NUM_CLIP_PLANES * 4> planes{};
                    for (u32 i = 0; i < AmdGpu::NUM_CLIP_PLANES; ++i) {
                        const auto& plane = GpuRegs().clip_user_data[i];
                        planes[i * 4 + 0] = std::bit_cast<float>(plane.data_x);
                        planes[i * 4 + 1] = std::bit_cast<float>(plane.data_y);
                        planes[i * 4 + 2] = std::bit_cast<float>(plane.data_z);
                        planes[i * 4 + 3] = std::bit_cast<float>(plane.data_w);
                    }
                    const u32 ubo_size = static_cast<u32>(sizeof(planes));
                    const u64 offset = vk_buffer.Copy(planes.data(), ubo_size, alignment);
                    buffer_infos.emplace_back(vk_buffer.Handle(), offset, ubo_size);
                }
            } else if (desc.buffer_type == Shader::BufferType::BdaPagetable) {
                const auto* bda_buffer = buffer_cache.GetBdaPageTableBuffer();
                buffer_infos.emplace_back(bda_buffer->Handle(), 0, bda_buffer->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::FaultBuffer) {
                const auto* fault_buffer = buffer_cache.GetFaultBuffer();
                buffer_infos.emplace_back(fault_buffer->Handle(), 0, fault_buffer->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::SharedMemory) {
                auto& lds_buffer = buffer_cache.GetStreamBuffer();
                const auto& cs_program = CsRegs();
                const auto lds_size = cs_program.SharedMemSize() * cs_program.NumWorkgroups();
                const auto [data, offset] = lds_buffer.Map(lds_size, alignment);
                std::memset(data, 0, lds_size);
                lds_buffer.Commit();
                buffer_infos.emplace_back(lds_buffer.Handle(), offset, lds_size);
            } else {
                UNREACHABLE_MSG("Unexpected buffer type {}", u32(desc.buffer_type));
            }
        } else {
            const auto vsharp = desc.GetSharp(stage);
            if (vsharp.base_address == 0 || vsharp.GetSize() == 0) {
                buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
            } else {
                u64 size = memory->ClampRangeSize(vsharp.base_address, vsharp.GetSize());
                // sotcport: the game binds many read-only buffers without a real size
                // (num_records ~4 GiB: no bounds on the console), each draw pointing the base at
                // its own data. Clamped to the end of their mapping, some covered 161 MB and
                // 2.7 GB of memory, all of it made resident in VRAM and kept up to date with the
                // CPU's writes: a 4 GB card ran out of VRAM and paged (2026-10-19, GTX 1650). They
                // get a window from their base instead, far more than a draw reads: Pro mode on
                // the GTX 1650 27 -> 39 FPS, same image at a still scene.
                // SOTC_UNBOUNDED_WINDOW=<MB> sets it (0: the whole mapping, as before).
                static const u64 unbounded_window = [] {
                    const char* mb = std::getenv("SOTC_UNBOUNDED_WINDOW");
                    return (mb ? std::strtoull(mb, nullptr, 0) : u64{64}) << 20;
                }();
                if (unbounded_window && !desc.is_written && vsharp.GetSize() >= (u64{1} << 30)) {
                    size = std::min(size, unbounded_window);
                }
                if (size != vsharp.GetSize()) {
                    // sotcport: the game binds buffers without a real size (num_records of
                    // ~4 GiB) thousands of times a second; say it once per shader.
                    static std::unordered_set<u64> reported;
                    if (reported.insert(stage.pgm_hash).second) {
                        LOG_INFO(Render, "Clamped size from {} to {} for stage {:#x}",
                                 vsharp.GetSize(), size, stage.pgm_hash);
                    }
                }
                const auto [buffer, offset] =
                    buffer_cache.ObtainBuffer(vsharp.base_address, size, desc.is_written,
                                              desc.is_formatted, shadow_dispatch);
                const u64 offset_aligned = Common::AlignDown(offset, alignment);
                const u64 adjust = offset - offset_aligned;
                if (adjust % 4 != 0) {
                    LOG_WARNING(Render_Vulkan, "Buffer binding in shader {:#x} isn't dword aligned",
                                stage.pgm_hash);
                }
                push_data.AddOffset(binding.buffer, adjust);
                buffer_infos.emplace_back(buffer->Handle(), offset_aligned, size + adjust);
                // sotcport: a game shader's own binding (see Runtime::AccessBuffer).
                static const bool strict = std::getenv("SOTC_STRICT_BARRIERS") != nullptr;
                const bool guest = !strict && DrawPipe::OnStageB();
                bound_buffers.emplace_back(buffer, offset, size, src_access, guest);
                // sotcport: a streamed copy is made again for every draw (see BindMemo).
                if (memo_recording && buffer == &buffer_cache.GetStreamBuffer()) {
                    bind_memo.dynamics.push_back(
                        {static_cast<u32>(buffer_infos.size() - 1),
                         static_cast<u32>(bound_buffers.size() - 1 - memo_first_bound_buffer),
                         binding.buffer, memo_stage, vsharp.base_address,
                         static_cast<u32>(size)});
                }
                if (desc.is_written) {
                    // Raw storage-buffer writes can also make an aliased cached image stale.
                    texture_cache.InvalidateMemoryFromGPU(vsharp.base_address, size);
                }
                needs_barrier |=
                    runtime.IsBufferAccessed(buffer, offset, size, desc.is_written, guest);
            }
        }

        auto& set_write = set_writes[set_write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = binding.unified++;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = 1;
        set_write.descriptorType = vk::DescriptorType::eStorageBuffer;
        set_write.pBufferInfo = &buffer_infos.back();
        ++binding.buffer;
    }
}

void Rasterizer::BindTextures(const Shader::Info& stage, Shader::Backend::Bindings& binding) {
    const u32 first_image_idx = image_infos.size();
    // To emulate storing to explicit mip levels, build a descriptor array with each mip level.
    boost::container::small_vector<u32, 8> image_descriptor_array_sizes;

    u32 num_images{};
    for (const auto& image_desc : stage.images) {
        const auto tsharp = image_desc.GetSharp(stage);
        if (tsharp.lod_hw_cnt_en) {
            NoteLodStatsBind(tsharp);
        }
        const auto data_fmt = tsharp.GetDataFmt();
        const auto num_fmt = tsharp.GetNumberFmt();
        const u32 num_bindings = image_desc.NumBindings(stage);

        // Null bindings must still match the descriptor type and array size of the layout.
        const auto bind_null_image = [&] {
            for (u32 i = 0; i < num_bindings; i++) {
                auto& [image_id, desc] = image_bindings[num_images++];
                image_id = {};
                desc.type = image_desc.is_written ? VideoCore::TextureCache::BindingType::Storage
                                                  : VideoCore::TextureCache::BindingType::Texture;
            }
            image_descriptor_array_sizes.push_back(num_bindings);
        };

        if (tsharp.Address() == 0 || data_fmt == AmdGpu::DataFormat::FormatInvalid) {
            bind_null_image();
            continue;
        }

        if (!memory->IsValidGpuMapping(tsharp.Address(), 0) ||
            !magic_enum::enum_contains(data_fmt) || !magic_enum::enum_contains(num_fmt)) {
            LOG_WARNING(Render_Vulkan,
                        "Rejecting invalid T# address={:#x}, pitch={}, width={}, "
                        "data_format={}, num_format={}",
                        tsharp.Address(), tsharp.pitch, tsharp.width, static_cast<u32>(data_fmt),
                        static_cast<u32>(num_fmt));
            bind_null_image();
            continue;
        }

        const Shader::MipStorageFallbackMode mip_fallback_mode = image_desc.mip_fallback_mode;

        for (auto i = 0; i < num_bindings; i++) {
            auto& [image_id, desc] = image_bindings[num_images++];
            // sotcport: the image found for this T# last time, while the texture cache's
            // registry is the same (FindImage was ~4% of the draw recording thread).
            // SOTC_NO_TEXTURE_CACHE_LOOKUP=1 finds every texture again (A/B).
            static const bool no_lookup = std::getenv("SOTC_NO_TEXTURE_CACHE_LOOKUP") != nullptr;
            ImageDescEntry* entry = mip_fallback_mode == Shader::MipStorageFallbackMode::None &&
                                            !no_lookup && !diag::On(diag::TextureLookupOff)
                                        ? CachedImageEntry(tsharp, image_desc)
                                        : nullptr;
            if (entry && entry->found_generation == texture_cache.RegistryGeneration()) {
                desc = entry->found_desc;
                image_id = entry->found_image;
                texture_cache.UseFoundImage(image_id);
            } else {
                desc = CachedImageDesc(tsharp, image_desc);

                if (mip_fallback_mode == Shader::MipStorageFallbackMode::ConstantIndex) {
                    ASSERT(num_bindings == 1);
                    desc.view_info.range.base.level += image_desc.constant_mip_index;
                    desc.view_info.range.extent.levels = 1;
                } else if (mip_fallback_mode == Shader::MipStorageFallbackMode::DynamicIndex) {
                    desc.view_info.range.base.level += i;
                    desc.view_info.range.extent.levels = 1;
                }

                image_id = texture_cache.FindImage(desc);
                if (entry && texture_cache.LastFindWasDirect()) {
                    entry->found_generation = texture_cache.RegistryGeneration();
                    entry->found_image = image_id;
                    entry->found_desc = desc;
                }
            }
            auto* image = &texture_cache.GetImage(image_id);
            diag::NoteMemoryAccess(diag::MemAccess::ImageBind, image->info.guest_address,
                                   image->info.guest_size); // sotcport (SOTC_MEMORY_TYPES)
            if (auto depth_image_id = texture_cache.GetAssociatedDepth(*image)) {
                // If this image has an associated depth image, it's a stencil attachment.
                // Redirect the access to the actual depth-stencil buffer.
                image_id = depth_image_id;
                image = &texture_cache.GetImage(image_id);
            }
            if (image->binding.is_bound) {
                // The image is already bound. In case if it is about to be used as storage we
                // need to force general layout on it.
                image->binding.force_general |= image_desc.is_written;
            }
            image->binding.is_bound = 1u;
        }

        image_descriptor_array_sizes.push_back(num_bindings);
    }

    // Second pass to re-bind images that were updated after binding
    for (u32 i = 0; i < num_images; ++i) {
        auto& [image_id, desc] = image_bindings[i];
        bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        if (!image_id) {
            image_infos.emplace_back(VK_NULL_HANDLE, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
        } else {
            if (auto& old_image = texture_cache.GetImage(image_id);
                old_image.binding.needs_rebind) {
                old_image.binding = {};
                image_id = texture_cache.FindImage(desc);
            }

            bound_images.emplace_back(image_id);

            auto& image = texture_cache.GetImage(image_id);
            auto& image_view = texture_cache.FindTexture(image_id, desc);
            texture_cache.VerifyImage(image_id); // sotcport (SOTC_VERIFY_IMAGES)
            const auto binding = image.binding;

            // The image is either bound as storage in a separate descriptor or bound as render
            // target in feedback loop. Depth images are excluded because they can't be bound as
            // storage and feedback loop doesn't make sense for them
            if ((binding.force_general || binding.is_target) && !image.info.props.is_depth) {
                if (instance.IsAttachmentFeedbackLoopLayoutSupported() && image.binding.is_target) {
                    needs_barrier |= runtime.Transit(
                        &image, vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT,
                        vk::PipelineStageFlagBits2::eAllGraphics, vk::AccessFlagBits2::eShaderRead);
                } else {
                    needs_barrier |= runtime.Transit(
                        &image, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eAllCommands,
                        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
                        desc.view_info.range);
                }
            } else {
                if (is_storage) {
                    needs_barrier |= runtime.Transit(
                        &image, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eAllCommands,
                        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
                        desc.view_info.range);
                } else {
                    const auto new_layout = image.info.props.is_depth
                                                ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                                : vk::ImageLayout::eShaderReadOnlyOptimal;
                    needs_barrier |= runtime.Transit(
                        &image, new_layout, vk::PipelineStageFlagBits2::eAllCommands,
                        vk::AccessFlagBits2::eShaderRead, desc.view_info.range);
                }
            }
            image.usage.storage |= is_storage;
            image.usage.texture |= !is_storage;

            image_infos.emplace_back(VK_NULL_HANDLE, *image_view.image_view,
                                     image.backing->state.layout);
        }
    }

    u32 image_info_idx = first_image_idx;
    u32 image_binding_idx = 0;
    for (u32 array_size : image_descriptor_array_sizes) {
        const auto& [_, desc] = image_bindings[image_binding_idx];
        const bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        auto& set_write = set_writes[set_write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = binding.unified;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = array_size;
        set_write.descriptorType =
            is_storage ? vk::DescriptorType::eStorageImage : vk::DescriptorType::eSampledImage;
        set_write.pImageInfo = &image_infos[image_info_idx];

        image_info_idx += array_size;
        image_binding_idx += array_size;
        binding.unified += array_size;
    }

    for (const auto& sampler : stage.samplers) {
        auto ssharp = sampler.GetSharp(stage);
        if (!ssharp.Valid() || (ssharp.border_color_type.Value() == AmdGpu::BorderColor::Custom &&
                                GpuRegs().ta_bc_base.Address() == 0)) {
            LOG_WARNING(Render_Vulkan,
                        "Rejecting invalid S# max_aniso={}, filter_mode={}, mip_filter={}, "
                        "border_color_type={}, border_color_base={:#x}",
                        static_cast<u32>(ssharp.max_aniso.Value()),
                        static_cast<u32>(ssharp.filter_mode.Value()),
                        static_cast<u32>(ssharp.mip_filter.Value()),
                        static_cast<u32>(ssharp.border_color_type.Value()),
                        GpuRegs().ta_bc_base.Address());
            ssharp = AmdGpu::Sampler{};
        }
        const auto vk_sampler =
            texture_cache.GetSampler(ssharp, GpuRegs().ta_bc_base, sampler.is_depth);
        // sotcport: SOTC_TRACE_FLATBUF=<shader hash> also logs the samplers it gets.
        static const u64 trace_sampler = [] {
            const char* hash = std::getenv("SOTC_TRACE_FLATBUF");
            return hash ? std::strtoull(hash, nullptr, 16) : 0ull;
        }();
        if (trace_sampler && (stage.pgm_hash & 0xffffffffull) == trace_sampler) {
            using namespace Vulkan::LiverpoolToVK;
            LOG_INFO(Render, "sampler of {:#x}: filter {}/{} mip {} address {}/{}/{} lod bias {} "
                             "lod {}..{} aniso {} ({}) depth {}",
                     stage.pgm_hash, u32(Filter(ssharp.xy_mag_filter)),
                     u32(Filter(ssharp.xy_min_filter)), u32(MipFilter(ssharp.mip_filter)),
                     u32(ClampMode(ssharp.clamp_x)), u32(ClampMode(ssharp.clamp_y)),
                     u32(ClampMode(ssharp.clamp_z)), ssharp.LodBias(), ssharp.MinLod(),
                     ssharp.MaxLod(),
                     AmdGpu::IsAnisoFilter(ssharp.xy_mag_filter) ||
                         AmdGpu::IsAnisoFilter(ssharp.xy_min_filter),
                     ssharp.MaxAniso(), sampler.is_depth);
        }
        image_infos.emplace_back(vk_sampler, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
        auto& set_write = set_writes[set_write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = binding.unified++;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = 1;
        set_write.descriptorType = vk::DescriptorType::eSampler;
        set_write.pImageInfo = &image_infos.back();
    }
}

RenderState Rasterizer::BeginRendering(const GraphicsPipeline* pipeline) {
    attachment_feedback_loop = false;
    const auto& regs = GpuRegs();
    const auto& key = pipeline->GetGraphicsKey();
    RenderState state;
    state.width = instance.GetMaxFramebufferWidth();
    state.height = instance.GetMaxFramebufferHeight();
    state.num_layers = std::numeric_limits<u16>::max();
    state.num_color_attachments = std::bit_width(key.mrt_mask);
    for (auto cb = 0u; cb < state.num_color_attachments; ++cb) {
        auto& [image_id, desc] = cb_descs[cb];
        if (!image_id) {
            state.color_attachments[cb] = {};
            continue;
        }
        auto* image = &texture_cache.GetImage(image_id);
        if (image->binding.needs_rebind) {
            image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
            image = &texture_cache.GetImage(image_id);
        }
        texture_cache.UpdateImage(image_id);
        runtime.SetBackingSamples(image, key.color_samples[cb]);
        const auto& image_view = texture_cache.FindRenderTarget(image_id, desc);
        const auto slice = image_view.info.range.base.layer;
        const auto mip = image_view.info.range.base.level;

        const auto& col_buf = regs.color_buffers[cb];
        const bool is_clear = texture_cache.IsMetaCleared(col_buf.CmaskAddress(), slice);
        texture_cache.TouchMeta(col_buf.CmaskAddress(), slice, false);

        if (image->binding.is_bound) {
            ASSERT_MSG(!image->binding.force_general,
                       "Having image both as storage and render target is unsupported");
            runtime.FlushBarriers();
            needs_barrier |=
                runtime.Transit(image,
                                instance.IsAttachmentFeedbackLoopLayoutSupported()
                                    ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
                                    : vk::ImageLayout::eGeneral,
                                vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                                vk::AccessFlagBits2::eColorAttachmentWrite |
                                    vk::AccessFlagBits2::eColorAttachmentRead);
            attachment_feedback_loop = true;
        } else {
            needs_barrier |= runtime.Transit(image, vk::ImageLayout::eColorAttachmentOptimal,
                                             vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                                             vk::AccessFlagBits2::eColorAttachmentWrite |
                                                 vk::AccessFlagBits2::eColorAttachmentRead,
                                             desc.view_info.range);
        }

        state.width = std::min<u32>(state.width, std::max(image->info.size.width >> mip, 1u));
        state.height = std::min<u32>(state.height, std::max(image->info.size.height >> mip, 1u));
        state.num_layers = std::min<u32>(state.num_layers, image_view.info.range.extent.layers);

        const auto clear_value =
            is_clear ? LiverpoolToVK::ColorBufferClearValue(col_buf) : vk::ClearValue{};
        auto& attachment = state.color_attachments[cb];
        attachment.image_view = *image_view.image_view;
        attachment.image_layout = image->backing->state.layout;
        attachment.clear_value = clear_value.color.uint32;
        attachment.is_clear = is_clear;

        image->usage.render_target = 1u;
    }
    for (u32 cb = state.num_color_attachments; cb < state.color_attachments.size(); ++cb) {
        state.color_attachments[cb] = {};
    }

    if (auto image_id = db_desc.image_id; image_id) {
        auto& desc = db_desc.desc;
        const auto htile_address = regs.depth_htile_data_base.GetAddress();
        const auto& image_view = texture_cache.FindDepthTarget(image_id, desc);
        auto& image = texture_cache.GetImage(image_id);

        const auto slice = image_view.info.range.base.layer;
        const bool is_depth_clear =
            (regs.depth_render_control.depth_clear_enable && regs.depth_control.depth_enable &&
             regs.depth_control.depth_write_enable) ||
            texture_cache.IsMetaCleared(htile_address, slice);
        const bool is_stencil_clear = regs.depth_render_control.stencil_clear_enable;
        texture_cache.TouchMeta(htile_address, slice, false);
        ASSERT(desc.view_info.range.extent.levels == 1 && !image.binding.needs_rebind);

        const bool has_stencil = image.info.props.has_stencil;
        // Stencil writes can be enabled while depth writes are off.
        const bool stencil_write =
            has_stencil && regs.depth_control.stencil_enable && !desc.view_info.is_storage;
        const auto new_layout = desc.view_info.is_storage
                                    ? has_stencil ? vk::ImageLayout::eDepthStencilAttachmentOptimal
                                                  : vk::ImageLayout::eDepthAttachmentOptimal
                                : stencil_write
                                    ? vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal
                                : has_stencil ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                              : vk::ImageLayout::eDepthReadOnlyOptimal;
        needs_barrier |= runtime.Transit(&image, new_layout,
                                         vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                                             vk::PipelineStageFlagBits2::eLateFragmentTests,
                                         vk::AccessFlagBits2::eDepthStencilAttachmentWrite |
                                             vk::AccessFlagBits2::eDepthStencilAttachmentRead,
                                         desc.view_info.range);

        state.width = std::min<u32>(state.width, image.info.size.width);
        state.height = std::min<u32>(state.height, image.info.size.height);
        state.num_layers = std::min<u32>(state.num_layers, image_view.info.range.extent.layers);

        auto& attachment = state.depth_stencil_attachment;
        attachment.image_view = *image_view.image_view;
        attachment.image_layout = image.backing->state.layout;
        attachment.clear_value = {};
        attachment.is_clear = 0;

        if (regs.depth_buffer.DepthValid()) {
            attachment.clear_value[0] = is_depth_clear ? std::bit_cast<u32>(regs.depth_clear) : 0u;
            attachment.has_depth = true;
            attachment.depth_clear = is_depth_clear;
        }
        if (regs.depth_buffer.StencilValid()) {
            attachment.clear_value[1] = is_stencil_clear ? regs.stencil_clear : 0u;
            attachment.has_stencil = true;
            attachment.stencil_clear = is_stencil_clear;
        }

        image.usage.depth_target = true;
    } else {
        state.depth_stencil_attachment = {};
    }

    if (state.num_layers == std::numeric_limits<u16>::max()) {
        state.num_layers = 1;
    }

    return state;
}

void Rasterizer::Resolve() {
    const auto& mrt0_hint = CbExtents()[0];
    const auto& mrt1_hint = CbExtents()[1];
    VideoCore::TextureCache::ImageDesc mrt0_desc{GpuRegs().color_buffers[0], mrt0_hint};
    VideoCore::TextureCache::ImageDesc mrt1_desc{GpuRegs().color_buffers[1], mrt1_hint};
    auto& mrt0_image = texture_cache.GetImage(texture_cache.FindImage(mrt0_desc, true));
    auto& mrt1_image = texture_cache.GetImage(texture_cache.FindImage(mrt1_desc, true));

    ScopeMarkerBegin(fmt::format("Resolve:MRT0={:#x}:MRT1={:#x}",
                                 GpuRegs().color_buffers[0].Address(),
                                 GpuRegs().color_buffers[1].Address()));
    runtime.ResolveImage(&mrt0_image, &mrt1_image, mrt0_desc.view_info.range,
                         mrt1_desc.view_info.range);
    ScopeMarkerEnd();
}

void Rasterizer::DepthStencilCopy(bool is_depth, bool is_stencil) {
    auto& regs = GpuRegs();

    auto read_desc = VideoCore::TextureCache::ImageDesc(
        regs.depth_buffer, regs.depth_view, regs.depth_control,
        regs.depth_htile_data_base.GetAddress(), DbExtent(), false);
    auto write_desc = VideoCore::TextureCache::ImageDesc(
        regs.depth_buffer, regs.depth_view, regs.depth_control,
        regs.depth_htile_data_base.GetAddress(), DbExtent(), true);

    auto& read_image = texture_cache.GetImage(texture_cache.FindImage(read_desc));
    auto& write_image = texture_cache.GetImage(texture_cache.FindImage(write_desc));

    VideoCore::SubresourceRange sub_range;
    sub_range.base.layer = GpuRegs().depth_view.slice_start;
    sub_range.extent.layers = GpuRegs().depth_view.NumSlices() - sub_range.base.layer;

    ScopeMarkerBegin(fmt::format(
        "DepthStencilCopy:DR={:#x}:SR={:#x}:DW={:#x}:SW={:#x}", regs.depth_buffer.DepthAddress(),
        regs.depth_buffer.StencilAddress(), regs.depth_buffer.DepthWriteAddress(),
        regs.depth_buffer.StencilWriteAddress()));

    runtime.CopyDepthStencil(&read_image, &write_image, sub_range);

    ScopeMarkerEnd();
}

void Rasterizer::FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds) {
    DrainDrawPipe(DrawPipe::ReasonFillBuffer); // sotcport: two-stage draw pipeline
    ASSERT_MSG(address % 4 == 0 && num_bytes % 4 == 0,
               "FillBuffer address and size must be a multiple of 4 bytes");
    if (!is_gds) {
        texture_cache.ClearMeta(address);
        if (!buffer_cache.IsRegionGpuModified(address, num_bytes)) {
            u32* buffer = std::bit_cast<u32*>(address);
            std::fill(buffer, buffer + (num_bytes / sizeof(u32)), value);
            return;
        }
    }
    const auto [buffer, offset] = [&] -> std::pair<const VideoCore::Buffer*, u64> {
        if (is_gds) {
            return {buffer_cache.GetGdsBuffer(), address};
        }
        return buffer_cache.ObtainBuffer(address, num_bytes, true);
    }();
    runtime.FillBuffer(buffer, offset, num_bytes, value);
}

void Rasterizer::CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds) {
    DrainDrawPipe(DrawPipe::ReasonCopyBuffer); // sotcport: two-stage draw pipeline
    if (!dst_gds && !buffer_cache.IsRegionGpuModified(dst, num_bytes)) {
        if (!src_gds && !buffer_cache.IsRegionGpuModified(src, num_bytes) &&
            !texture_cache.FindImageFromRange(src, num_bytes)) {
            // Both buffers were not transferred to GPU yet. Can safely copy in host memory.
            std::memcpy(std::bit_cast<void*>(dst), std::bit_cast<void*>(src), num_bytes);
            return;
        }
    }
    texture_cache.InvalidateMemoryFromGPU(dst, num_bytes);
    const auto* gds_buffer = buffer_cache.GetGdsBuffer();
    const auto [src_buffer, src_offset] = [&] -> std::pair<const VideoCore::Buffer*, u64> {
        if (src_gds) {
            return {gds_buffer, src};
        }
        return buffer_cache.ObtainBuffer(src, num_bytes, false, true);
    }();
    const auto [dst_buffer, dst_offset] = [&] -> std::pair<const VideoCore::Buffer*, u64> {
        if (dst_gds) {
            return {gds_buffer, dst};
        }
        return buffer_cache.ObtainBuffer(dst, num_bytes, true, true);
    }();
    const vk::BufferCopy copy = {
        .srcOffset = src_offset,
        .dstOffset = dst_offset,
        .size = num_bytes,
    };
    runtime.CopyBuffer(src_buffer, dst_buffer, std::span{&copy, 1});
}

u32 Rasterizer::ReadDataFromGds(u32 gds_offset) {
    DrainDrawPipe(DrawPipe::ReasonGds); // sotcport: two-stage draw pipeline
    auto* gds_buf = buffer_cache.GetGdsBuffer();
    u32 value;
    std::memcpy(&value, gds_buf->mapped_data.data() + gds_offset, sizeof(u32));
    return value;
}

bool Rasterizer::InvalidateMemory(VAddr addr, u64 size, bool assume_locks) {
    perf::Scope perf_scope{perf::Invalidate}; // sotcport TEMP
    if (assume_locks) {
        DrainDrawPipe(DrawPipe::ReasonInvalidate); // sotcport: inline on the GPU command thread: stage B must be idle
    }
    if (!IsMapped(addr, size)) {
        // Not GPU mapped memory, can skip invalidation logic entirely.
        return false;
    }
    buffer_cache.InvalidateMemory(addr, size, assume_locks);
    texture_cache.InvalidateMemory(addr, size);
    return true;
}

bool Rasterizer::ReadMemory(VAddr addr, u64 size, bool assume_locks) {
    perf::Scope perf_scope{perf::Readback}; // sotcport TEMP
    if (assume_locks) {
        DrainDrawPipe(DrawPipe::ReasonReadback); // sotcport: inline on the GPU command thread: stage B must be idle
    }
    if (!IsMapped(addr, size)) {
        // Not GPU mapped memory, can skip invalidation logic entirely.
        return false;
    }
    buffer_cache.ReadMemory(addr, size, false, assume_locks);
    return true;
}

bool Rasterizer::IsMapped(VAddr addr, u64 size) {
    if (size == 0) {
        // There is no memory, so not mapped.
        return false;
    }
    if (static_cast<u64>(addr) > std::numeric_limits<u64>::max() - size) {
        // Memory range wrapped the address space, cannot be mapped.
        return false;
    }
    // sotcport: every CPU write fault on tracked memory asks this, from several threads at once
    // while the game streams data in, and the shared lock (a mutex inside) made them queue. Each
    // thread keeps the last mapped intervals it found, valid while the mappings do not change
    // (mapped_generation, bumped under the exclusive lock), so most calls take no lock.
    struct IntervalCache {
        u64 generation = ~u64{0};
        std::array<std::pair<VAddr, VAddr>, 8> intervals{};
        u32 next = 0;
    };
    thread_local IntervalCache cache;
    const u64 generation = mapped_generation.load(std::memory_order_acquire);
    if (cache.generation == generation) {
        for (const auto& [start, end] : cache.intervals) {
            if (addr >= start && addr + size <= end) {
                return true;
            }
        }
    } else {
        cache = {};
        cache.generation = generation;
    }
    Common::RecursiveSharedLock lock{mapped_ranges_mutex};
    const auto it = mapped_ranges.find(addr);
    if (it == mapped_ranges.end()) {
        return false;
    }
    // [first, last_next): the interval as a half-open range, whatever its bound type.
    const VAddr start = boost::icl::first(*it);
    const VAddr end = boost::icl::last_next(*it);
    if (addr + size > end) {
        return false;
    }
    if (mapped_generation.load(std::memory_order_relaxed) == generation) {
        cache.intervals[cache.next++ % cache.intervals.size()] = {start, end};
    }
    return true;
}

void Rasterizer::MapMemory(VAddr addr, u64 size) {
    {
        std::scoped_lock lock{mapped_ranges_mutex};
        mapped_ranges += decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
        mapped_generation.fetch_add(1, std::memory_order_release);
    }
}

void Rasterizer::RegisterMemory(VAddr addr, u64 size) {
    page_manager.OnGpuMap(addr, size);
}

void Rasterizer::UnmapMemory(VAddr addr, u64 size) {
    DrainDrawPipe(DrawPipe::ReasonUnmap); // sotcport: two-stage draw pipeline
    // sotcport: this runs on the GPU command thread (gpu_glue.cpp, OnUnmapped): a readback
    // must be done right here, not sent to that same thread (it would wait for itself).
    buffer_cache.InvalidateMemory(addr, size, IsGpuSideThread());
    texture_cache.UnmapMemory(addr, size);
    {
        std::scoped_lock lock{mapped_ranges_mutex};
        mapped_ranges -= decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
        mapped_generation.fetch_add(1, std::memory_order_release); // sotcport: see IsMapped
    }
}

void Rasterizer::UpdateDynamicState(const GraphicsPipeline* pipeline, const bool is_indexed) const {
    UpdateViewportScissorState();
    UpdateDepthStencilState();
    UpdatePrimitiveState(is_indexed);
    UpdateRasterizationState();
    UpdateColorBlendingState(pipeline);

    auto& dynamic_state = scheduler.GetDynamicState();
    if (!scheduler.IsRecordingDeferred()) {
        dynamic_state.Commit(instance, scheduler.CommandBuffer());
        return;
    }
    // sotcport: a copy commits on the recording thread (threaded recording).
    if (dynamic_state.AnyDirty()) {
        scheduler.Record([state = dynamic_state, inst = &instance](vk::CommandBuffer cmdbuf) mutable {
            state.Commit(*inst, cmdbuf);
        });
        dynamic_state.ClearDirty();
    }
}

void Rasterizer::UpdateViewportScissorState() const {
    const auto& regs = GpuRegs();

    const auto combined_scissor_value_tl = [](s16 scr, s16 win, s16 gen, s16 win_offset) {
        return std::max({scr, s16(win + win_offset), s16(gen + win_offset)});
    };
    const auto combined_scissor_value_br = [](s16 scr, s16 win, s16 gen, s16 win_offset) {
        return std::min({scr, s16(win + win_offset), s16(gen + win_offset)});
    };
    const bool enable_offset = !regs.window_scissor.window_offset_disable;

    AmdGpu::Scissor scsr{};
    scsr.top_left_x = combined_scissor_value_tl(
        regs.screen_scissor.top_left_x, s16(regs.window_scissor.top_left_x),
        s16(regs.generic_scissor.top_left_x),
        enable_offset ? regs.window_offset.window_x_offset : 0);
    scsr.top_left_y = combined_scissor_value_tl(
        regs.screen_scissor.top_left_y, s16(regs.window_scissor.top_left_y),
        s16(regs.generic_scissor.top_left_y),
        enable_offset ? regs.window_offset.window_y_offset : 0);
    scsr.bottom_right_x = combined_scissor_value_br(
        regs.screen_scissor.bottom_right_x, regs.window_scissor.bottom_right_x,
        regs.generic_scissor.bottom_right_x,
        enable_offset ? regs.window_offset.window_x_offset : 0);
    scsr.bottom_right_y = combined_scissor_value_br(
        regs.screen_scissor.bottom_right_y, regs.window_scissor.bottom_right_y,
        regs.generic_scissor.bottom_right_y,
        enable_offset ? regs.window_offset.window_y_offset : 0);

    boost::container::static_vector<vk::Viewport, AmdGpu::NUM_VIEWPORTS> viewports;
    boost::container::static_vector<vk::Rect2D, AmdGpu::NUM_VIEWPORTS> scissors;

    if (regs.polygon_control.enable_window_offset &&
        (regs.window_offset.window_x_offset != 0 || regs.window_offset.window_y_offset != 0)) {
        LOG_ERROR(Render_Vulkan,
                  "PA_SU_SC_MODE_CNTL.VTX_WINDOW_OFFSET_ENABLE support is not yet implemented.");
    }

    const auto& vp_ctl = regs.viewport_control;
    for (u32 i = 0; i < AmdGpu::NUM_VIEWPORTS; i++) {
        const auto& vp = regs.viewports[i];
        const auto& vp_d = regs.viewport_depths[i];
        if (vp.xscale == 0) {
            continue;
        }

        const auto zoffset = vp_ctl.zoffset_enable ? vp.zoffset : 0.f;
        const auto zscale = vp_ctl.zscale_enable ? vp.zscale : 1.f;

        vk::Viewport viewport{};

        // https://gitlab.freedesktop.org/mesa/mesa/-/blob/209a0ed/src/amd/vulkan/radv_pipeline_graphics.c#L688-689
        // https://gitlab.freedesktop.org/mesa/mesa/-/blob/209a0ed/src/amd/vulkan/radv_cmd_buffer.c#L3103-3109
        // When the clip space is ranged [-1...1], the zoffset is centered.
        // By reversing the above viewport calculations, we get the following:
        if (regs.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW) {
            viewport.minDepth = zoffset - zscale;
            viewport.maxDepth = zoffset + zscale;
        } else {
            viewport.minDepth = zoffset;
            viewport.maxDepth = zoffset + zscale;
        }

        if (!instance.IsDepthRangeUnrestrictedSupported()) {
            // Unrestricted depth range not supported by device. Restrict to valid range.
            viewport.minDepth = std::max(viewport.minDepth, 0.f);
            viewport.maxDepth = std::min(viewport.maxDepth, 1.f);
        }

        if (regs.IsClipDisabled()) {
            // In case if clipping is disabled we patch the shader to convert vertex position
            // from screen space coordinates to NDC by defining a render space as full hardware
            // window range [0..16383, 0..16383] and setting the viewport to its size.
            viewport.x = 0.f;
            viewport.y = 0.f;
            viewport.width = float(std::min<u32>(instance.GetMaxViewportWidth(), 16_KB));
            viewport.height = float(std::min<u32>(instance.GetMaxViewportHeight(), 16_KB));
        } else {
            const auto xoffset = vp_ctl.xoffset_enable ? vp.xoffset : 0.f;
            const auto xscale = vp_ctl.xscale_enable ? vp.xscale : 1.f;
            const auto yoffset = vp_ctl.yoffset_enable ? vp.yoffset : 0.f;
            const auto yscale = vp_ctl.yscale_enable ? vp.yscale : 1.f;

            viewport.x = xoffset - xscale;
            viewport.y = yoffset - yscale;
            viewport.width = xscale * 2.0f;
            viewport.height = yscale * 2.0f;
        }

        viewports.push_back(viewport);

        auto vp_scsr = scsr;
        if (regs.mode_control.vport_scissor_enable) {
            vp_scsr.top_left_x =
                std::max(vp_scsr.top_left_x, s16(regs.viewport_scissors[i].top_left_x));
            vp_scsr.top_left_y =
                std::max(vp_scsr.top_left_y, s16(regs.viewport_scissors[i].top_left_y));
            vp_scsr.bottom_right_x = std::min(AmdGpu::Scissor::Clamp(vp_scsr.bottom_right_x),
                                              regs.viewport_scissors[i].bottom_right_x);
            vp_scsr.bottom_right_y = std::min(AmdGpu::Scissor::Clamp(vp_scsr.bottom_right_y),
                                              regs.viewport_scissors[i].bottom_right_y);
        }
        scissors.push_back({
            .offset = {vp_scsr.top_left_x, vp_scsr.top_left_y},
            .extent = {vp_scsr.GetWidth(), vp_scsr.GetHeight()},
        });
    }

    if (viewports.empty()) {
        // Vulkan requires providing at least one viewport.
        constexpr vk::Viewport empty_viewport = {
            .x = -1.0f,
            .y = -1.0f,
            .width = 1.0f,
            .height = 1.0f,
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        constexpr vk::Rect2D empty_scissor = {
            .offset = {0, 0},
            .extent = {1, 1},
        };
        viewports.push_back(empty_viewport);
        scissors.push_back(empty_scissor);
    }

    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetViewports(viewports);
    dynamic_state.SetScissors(scissors);
}

void Rasterizer::UpdateDepthStencilState() const {
    const auto& regs = GpuRegs();
    auto& dynamic_state = scheduler.GetDynamicState();

    const auto depth_test_enabled =
        regs.depth_control.depth_enable && regs.depth_buffer.DepthValid();
    dynamic_state.SetDepthTestEnabled(depth_test_enabled);
    if (depth_test_enabled) {
        dynamic_state.SetDepthWriteEnabled(regs.depth_control.depth_write_enable &&
                                           !regs.depth_render_control.depth_clear_enable);
        dynamic_state.SetDepthCompareOp(LiverpoolToVK::CompareOp(regs.depth_control.depth_func));
    }

    const auto depth_bounds_test_enabled = regs.depth_control.depth_bounds_enable;
    dynamic_state.SetDepthBoundsTestEnabled(depth_bounds_test_enabled);
    if (depth_bounds_test_enabled) {
        dynamic_state.SetDepthBounds(regs.depth_bounds_min, regs.depth_bounds_max);
    }

    const auto depth_bias_enabled = regs.polygon_control.NeedsBias();
    dynamic_state.SetDepthBiasEnabled(depth_bias_enabled);
    if (depth_bias_enabled) {
        const bool front = regs.polygon_control.enable_polygon_offset_front;
        dynamic_state.SetDepthBias(
            front ? regs.poly_offset.front_offset : regs.poly_offset.back_offset,
            regs.poly_offset.depth_bias,
            (front ? regs.poly_offset.front_scale : regs.poly_offset.back_scale) / 16.f);
    }

    const auto stencil_test_enabled =
        regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid();
    dynamic_state.SetStencilTestEnabled(stencil_test_enabled);
    if (stencil_test_enabled) {
        const StencilOps front_ops{
            .fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_fail_front),
            .pass_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zpass_front),
            .depth_fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zfail_front),
            .compare_op = LiverpoolToVK::CompareOp(regs.depth_control.stencil_ref_func),
        };
        const StencilOps back_ops = regs.depth_control.backface_enable ? StencilOps{
            .fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_fail_back),
            .pass_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zpass_back),
            .depth_fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zfail_back),
            .compare_op = LiverpoolToVK::CompareOp(regs.depth_control.stencil_bf_func),
        } : front_ops;
        dynamic_state.SetStencilOps(front_ops, back_ops);

        const bool stencil_clear = regs.depth_render_control.stencil_clear_enable;
        const auto front = regs.stencil_ref_front;
        const auto back =
            regs.depth_control.backface_enable ? regs.stencil_ref_back : regs.stencil_ref_front;
        // GCN REPLACE_OP writes DB_STENCILREFMASK.STENCILOPVAL, so a face whose stencil ops
        // include ReplaceOp takes its Vulkan reference from op_val.
        const auto& sc = regs.stencil_control;
        const auto uses_op_val = [](AmdGpu::StencilFunc fail, AmdGpu::StencilFunc zpass,
                                    AmdGpu::StencilFunc zfail) {
            return fail == AmdGpu::StencilFunc::ReplaceOp ||
                   zpass == AmdGpu::StencilFunc::ReplaceOp ||
                   zfail == AmdGpu::StencilFunc::ReplaceOp;
        };
        const bool front_op =
            uses_op_val(sc.stencil_fail_front, sc.stencil_zpass_front, sc.stencil_zfail_front);
        const bool back_op =
            regs.depth_control.backface_enable
                ? uses_op_val(sc.stencil_fail_back, sc.stencil_zpass_back, sc.stencil_zfail_back)
                : front_op;
        const auto ref_conflict = [](AmdGpu::CompareFunc func, const AmdGpu::StencilRefMask& ref) {
            return func != AmdGpu::CompareFunc::Always && func != AmdGpu::CompareFunc::Never &&
                   ref.stencil_test_val != ref.stencil_op_val;
        };
        if ((front_op && ref_conflict(regs.depth_control.stencil_ref_func, front)) ||
            (back_op && regs.depth_control.backface_enable &&
             ref_conflict(regs.depth_control.stencil_bf_func, back))) {
            LOG_WARNING(Render_Vulkan, "Stencil test requires test_val while ReplaceOp requires "
                                       "op_val; the stencil test will use op_val");
        }
        dynamic_state.SetStencilReferences(front_op ? front.stencil_op_val : front.stencil_test_val,
                                           back_op ? back.stencil_op_val : back.stencil_test_val);
        dynamic_state.SetStencilWriteMasks(!stencil_clear ? front.stencil_write_mask : 0U,
                                           !stencil_clear ? back.stencil_write_mask : 0U);
        dynamic_state.SetStencilCompareMasks(front.stencil_mask, back.stencil_mask);
    }
}

void Rasterizer::UpdatePrimitiveState(const bool is_indexed) const {
    const auto& regs = GpuRegs();
    auto& dynamic_state = scheduler.GetDynamicState();

    const auto is_list_topology = [](const AmdGpu::PrimitiveType type) {
        const auto topology = LiverpoolToVK::PrimitiveType(type);
        return topology == vk::PrimitiveTopology::ePointList ||
               topology == vk::PrimitiveTopology::eLineList ||
               topology == vk::PrimitiveTopology::eTriangleList ||
               topology == vk::PrimitiveTopology::eLineListWithAdjacency ||
               topology == vk::PrimitiveTopology::eTriangleListWithAdjacency;
    };
    const auto is_patch_list_topology = [](const AmdGpu::PrimitiveType type) {
        // Quad and rect lists are emulated using tessellation.
        return type == AmdGpu::PrimitiveType::PatchPrimitive ||
               type == AmdGpu::PrimitiveType::QuadList || type == AmdGpu::PrimitiveType::RectList;
    };

    const auto prim_restart =
        (regs.enable_primitive_restart & 1) != 0 &&
        (instance.IsListRestartSupported() || !is_list_topology(regs.primitive_type)) &&
        (instance.IsPatchListRestartSupported() || !is_patch_list_topology(regs.primitive_type));
    ASSERT_MSG(!is_indexed || !prim_restart || regs.primitive_restart_index == 0xFFFF ||
                   regs.primitive_restart_index == 0xFFFFFFFF,
               "Primitive restart index other than -1 is not supported yet");

    const auto cull_mode = LiverpoolToVK::IsPrimitiveCulled(regs.primitive_type)
                               ? LiverpoolToVK::CullMode(regs.polygon_control.CullingMode())
                               : vk::CullModeFlagBits::eNone;
    const auto front_face = LiverpoolToVK::FrontFace(regs.polygon_control.front_face);

    dynamic_state.SetPrimitiveRestartEnabled(prim_restart);
    dynamic_state.SetRasterizerDiscardEnabled(regs.clipper_control.dx_rasterization_kill);
    dynamic_state.SetCullMode(cull_mode);
    dynamic_state.SetFrontFace(front_face);
}

void Rasterizer::UpdateRasterizationState() const {
    const auto& regs = GpuRegs();
    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetLineWidth(regs.line_control.Width());
}

void Rasterizer::UpdateColorBlendingState(const GraphicsPipeline* pipeline) const {
    const auto& regs = GpuRegs();
    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetBlendConstants(regs.blend_constants);
    dynamic_state.SetColorWriteMasks(pipeline->GetGraphicsKey().write_masks);
    dynamic_state.SetAttachmentFeedbackLoopEnabled(attachment_feedback_loop);
}

void Rasterizer::ScopeMarkerBegin(const std::string_view& str, bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.beginDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
    });
}

void Rasterizer::ScopeMarkerEnd(bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.endDebugUtilsLabelEXT();
}

void Rasterizer::ScopedMarkerInsert(const std::string_view& str, bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.insertDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
    });
}

void Rasterizer::ScopedMarkerInsertColor(const std::string_view& str, const u32 color,
                                         bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.insertDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
        .color = std::array<f32, 4>(
            {(f32)((color >> 16) & 0xff) / 255.0f, (f32)((color >> 8) & 0xff) / 255.0f,
             (f32)(color & 0xff) / 255.0f, (f32)((color >> 24) & 0xff) / 255.0f})});
}

std::thread::id Rasterizer::GetGpuCommandProcessorThread() {
    return liverpool->GetGpuCommandProcessorThread();
}

#ifdef __linux__
u32 Rasterizer::GetGpuCommandProcessorThreadId() {
    return liverpool->GetGpuCommandProcessorThreadId();
}
#endif

} // namespace Vulkan
