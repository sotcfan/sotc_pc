// sotcport: the shader recompiler as a tool of the native renderer (interface and binding
// conventions in src/render/shader_translate.h). Mirrors what the GPU layer's pipeline cache does
// for a compute dispatch (vk_pipeline_cache.cpp: RefreshComputeKey, BuildRuntimeInfo,
// CompileModule) and what its rasterizer binds (vk_rasterizer.cpp: BindBuffers), without its
// Vulkan objects.

#include "render/shader_translate.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <span>
#include <format>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/backend/spirv/emit_spirv_quad_rect.h"
#include "shader_recompiler/frontend/fetch_shader.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/amdgpu/regs.h"
#include "video_core/amdgpu/regs_shader.h"
#include "video_core/amdgpu/resource.h"
#include "video_core/amdgpu/tiling.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/texture_cache/image_info.h"
#include "video_core/texture_cache/image_view.h"
#include "video_core/host_shaders/tiling_macro_8_comp.h"
#include "video_core/host_shaders/tiling_macro_16_comp.h"
#include "video_core/host_shaders/tiling_macro_32_comp.h"
#include "video_core/host_shaders/tiling_macro_64_comp.h"
#include "video_core/host_shaders/tiling_macro_96_comp.h"
#include "video_core/host_shaders/tiling_macro_128_comp.h"
#include "video_core/host_shaders/tiling_micro_8_comp.h"
#include "video_core/host_shaders/tiling_micro_16_comp.h"
#include "video_core/host_shaders/tiling_micro_32_comp.h"
#include "video_core/host_shaders/tiling_micro_64_comp.h"
#include "video_core/host_shaders/tiling_micro_96_comp.h"
#include "video_core/host_shaders/tiling_micro_128_comp.h"

namespace render::translate {

namespace {

// What the native device offers the recompiler. The values are RADV's on the user's RX 7900 XT
// (the same as tools/shader_survey); the native device enables the matching features.
Shader::Profile NativeProfile() {
    return Shader::Profile{
        .max_viewport_width = 16384,
        .max_viewport_height = 16384,
        .max_shared_memory_size = 65536,
        .supported_spirv = 0x00010600,
        .subgroup_size = 64,
        .sparse_page_shift = DmaPageBits,
        .support_int8 = true,
        .support_int16 = true,
        .support_int64 = true,
        .support_float16 = true,
        .support_float64 = true,
        .supports_denorm_behavior_independence = true,
        .supports_rounding_mode_independence = true,
        .support_fp16_denorm_preserve = true,
        .support_fp16_denorm_flush = true,
        .support_fp32_denorm_preserve = true,
        .support_fp32_denorm_flush = true,
        .support_fp64_denorm_preserve = true,
        .support_fp64_denorm_flush = true,
        .support_fp16_signed_zero_inf_nan_preserve = true,
        .support_fp32_signed_zero_inf_nan_preserve = true,
        .support_fp64_signed_zero_inf_nan_preserve = true,
        .supports_image_load_store_lod = true,
        .supports_native_cube_calc = true,
        .supports_trinary_minmax = true,
        .supports_buffer_fp32_atomic_min_max = true,
        .supports_image_fp32_atomic_min_max = true,
        .supports_buffer_int64_atomics = true,
        .supports_shared_int64_atomics = true,
        .supports_workgroup_explicit_memory_layout = true,
        .supports_amd_shader_explicit_vertex_parameter = true,
        // Like the GPU layer (Instance: barycentrics only without the AMD extension): with both,
        // the recompiler declares per-vertex inputs in a way no driver accepts.
        .supports_fragment_shader_barycentric = false,
        .supports_shader_stencil_export = true,
    };
}

const char* SpecialName(Shader::BufferType type) {
    switch (type) {
    case Shader::BufferType::BdaPagetable: return "bda-pagetable";
    case Shader::BufferType::FaultBuffer: return "fault-buffer";
    case Shader::BufferType::GdsBuffer: return "gds";
    case Shader::BufferType::ClipPlanes: return "clip-planes";
    default: return "?";
    }
}

} // namespace

// The image's layout, detiling and view, from the GPU layer's own computations
// (VideoCore::ImageInfo / ImageViewInfo, TileManager::GetTilingPipeline / DetileImage), so the
// native renderer reads the PS4 layout exactly as the layer does.
static void FillImageLayout(ImageBinding& out, const VideoCore::ImageInfo& info) {
    out.vk_format = static_cast<u32>(info.pixel_format);
    out.volume = info.props.is_volume;
    out.levels = info.resources.levels;
    out.layers = info.resources.layers;
    out.guest_size = info.guest_size;
    out.tiled = info.props.is_tiled;
    out.extent = {info.size.width, info.size.height, info.size.depth};
    out.num_bits = info.num_bits;
    out.block = info.props.is_block;
    out.has_stencil = info.props.has_stencil;
    switch (info.pixel_format) {
    case vk::Format::eD16UnormS8Uint: out.element_bytes = 2; break;
    case vk::Format::eD24UnormS8Uint:
    case vk::Format::eD32SfloatS8Uint: out.element_bytes = 4; break;
    default: out.element_bytes = info.num_bits / 8; break;
    }

    // Copies from the linear bytes, as TextureCache::RefreshImage builds them.
    for (u32 m = 0; m < info.resources.levels; m++) {
        const u32 width = std::max(info.size.width >> m, 1u);
        const u32 height = std::max(info.size.height >> m, 1u);
        const u32 depth = info.props.is_volume ? std::max(info.size.depth >> m, 1u) : 1u;
        const auto& mip = info.mips_layout[m];
        out.copies.push_back({
            .buffer_offset = mip.offset,
            .row_length = mip.pitch,
            .image_height = mip.height,
            .width = mip.pitch ? std::min<u32>(mip.pitch, width) : width,
            .height = mip.height ? std::min<u32>(mip.height, height) : height,
            .depth = depth,
        });
    }

    if (info.props.is_tiled) {
        // TileManager::GetTilingPipeline: the shader variant and its specialization constants.
        const auto micro_tile_mode = AmdGpu::GetMicroTileMode(info.tile_mode);
        const bool is_macro = AmdGpu::IsMacroTiled(info.array_mode);
        out.tiling_spec = {info.num_samples, u32(micro_tile_mode),
                           AmdGpu::GetMicroTileThickness(info.array_mode), 0u /* detiler */};
        out.tiling_spec_count = 4;
        if (is_macro) {
            const auto macro_tile_mode =
                AmdGpu::CalculateMacrotileMode(info.tile_mode, info.num_bits, info.num_samples);
            out.tiling_spec[4] = u32(info.array_mode);
            out.tiling_spec[5] = u32(AmdGpu::GetPipeConfig(info.tile_mode));
            out.tiling_spec[6] = AmdGpu::GetBankWidth(macro_tile_mode);
            out.tiling_spec[7] = AmdGpu::GetBankHeight(macro_tile_mode);
            out.tiling_spec[8] = AmdGpu::GetNumBanks(macro_tile_mode);
            out.tiling_spec[9] = std::bit_width(out.tiling_spec[8]) - 1;
            out.tiling_spec[10] = AmdGpu::CalculateTileSplit(info.tile_mode, info.array_mode,
                                                             micro_tile_mode, info.num_bits);
            out.tiling_spec[11] = AmdGpu::GetMacrotileAspect(macro_tile_mode);
            out.tiling_spec_count = 12;
        }
        std::span<const u32> code;
        switch (info.num_bits) {
        case 8: code = is_macro ? std::span<const u32>{TILING_MACRO_8_COMP} : std::span<const u32>{TILING_MICRO_8_COMP}; break;
        case 16: code = is_macro ? std::span<const u32>{TILING_MACRO_16_COMP} : std::span<const u32>{TILING_MICRO_16_COMP}; break;
        case 32: code = is_macro ? std::span<const u32>{TILING_MACRO_32_COMP} : std::span<const u32>{TILING_MICRO_32_COMP}; break;
        case 64: code = is_macro ? std::span<const u32>{TILING_MACRO_64_COMP} : std::span<const u32>{TILING_MICRO_64_COMP}; break;
        case 96: code = is_macro ? std::span<const u32>{TILING_MACRO_96_COMP} : std::span<const u32>{TILING_MICRO_96_COMP}; break;
        case 128: code = is_macro ? std::span<const u32>{TILING_MACRO_128_COMP} : std::span<const u32>{TILING_MICRO_128_COMP}; break;
        default: break; // unsupported texel size: left without a detiler (the caller skips it)
        }
        out.tiling_spirv = code.data();
        out.tiling_spirv_words = code.size();
        out.tiling_groups = (info.guest_size / (info.num_bits / 8)) / 64;

        // TileManager::DetileImage: the TilingInfo uniform block (scalar layout: three u32, then
        // 16 MipInfo of 12 bytes: u32 size, u16 pitch, u16 height, u32 offset). Compressed formats
        // count pitch and height in 4x4 blocks.
        struct TilingInfo {
            u32 bank_swizzle;
            u32 micro_tiled_mips;
            u32 num_mips;
            std::array<VideoCore::MipInfo, VideoCore::MAX_MIPS> mips;
        } params{};
        static_assert(sizeof(TilingInfo) == 12 + 12 * VideoCore::MAX_MIPS);
        params.bank_swizzle = info.bank_swizzle;
        params.micro_tiled_mips = info.micro_tiled_mips;
        params.num_mips = info.resources.levels;
        for (u32 mip = 0; mip < params.num_mips; ++mip) {
            auto& mip_info = params.mips[mip];
            mip_info = info.mips_layout[mip];
            if (info.props.is_block) {
                mip_info.pitch = std::max<u16>((mip_info.pitch + 3) / 4, 1U);
                mip_info.height = std::max<u16>((mip_info.height + 3) / 4, 1U);
            }
        }
        out.tiling_info.resize(sizeof(params));
        std::memcpy(out.tiling_info.data(), &params, sizeof(params));
    }
}

static void FillImage(ImageBinding& out, const AmdGpu::Image& sharp, const Shader::ImageResource& desc) {
    FillImageLayout(out, VideoCore::ImageInfo{sharp, desc});

    // The view (VideoCore::ImageViewInfo; view types as ConvertImageViewType in image_view.cpp:
    // cubes are seen as 2D arrays, the recompiler samples them that way).
    const VideoCore::ImageViewInfo view{sharp, desc};
    out.storage = view.is_storage;
    switch (view.type) {
    case AmdGpu::ImageType::Color1DArray:
    case AmdGpu::ImageType::Color2DArray:
    case AmdGpu::ImageType::Cube: out.view_type = VK_IMAGE_VIEW_TYPE_2D_ARRAY; break;
    case AmdGpu::ImageType::Color3D: out.view_type = VK_IMAGE_VIEW_TYPE_3D; break;
    default: out.view_type = VK_IMAGE_VIEW_TYPE_2D; break;
    }
    out.view_format = static_cast<u32>(view.format);
    if (!(view.mapping == AmdGpu::IdentityMapping)) {
        const vk::ComponentMapping mapping = Vulkan::LiverpoolToVK::ComponentMapping(view.mapping);
        out.swizzle = {static_cast<u32>(mapping.r), static_cast<u32>(mapping.g),
                       static_cast<u32>(mapping.b), static_cast<u32>(mapping.a)};
    }
    out.view_base_level = view.range.base.level;
    out.view_levels = view.range.extent.levels;
    if (view.min_lod != 0) {
        const float last_level = static_cast<float>(view.range.base.level +
                                                    view.range.extent.levels - 1);
        out.view_min_lod = std::min(static_cast<float>(view.min_lod) / 256.f, last_level);
    }
    out.view_base_layer = view.range.base.layer;
    out.view_layers = view.range.extent.layers;
}

// VideoCore::Sampler's VkSamplerCreateInfo (texture_cache/sampler.cpp), without the device limits
// (the native device clamps them).
static void FillSampler(SamplerBinding& out, const AmdGpu::Sampler& sampler, bool is_depth) {
    using namespace Vulkan;
    out.mag_filter = static_cast<u32>(LiverpoolToVK::Filter(sampler.xy_mag_filter));
    out.min_filter = static_cast<u32>(LiverpoolToVK::Filter(sampler.xy_min_filter));
    out.mipmap_mode = static_cast<u32>(LiverpoolToVK::MipFilter(sampler.mip_filter));
    out.address_u = static_cast<u32>(LiverpoolToVK::ClampMode(sampler.clamp_x));
    out.address_v = static_cast<u32>(LiverpoolToVK::ClampMode(sampler.clamp_y));
    out.address_w = static_cast<u32>(LiverpoolToVK::ClampMode(sampler.clamp_z));
    out.lod_bias = sampler.LodBias();
    out.anisotropy =
        AmdGpu::IsAnisoFilter(sampler.xy_mag_filter) || AmdGpu::IsAnisoFilter(sampler.xy_min_filter);
    out.max_anisotropy = out.anisotropy ? std::max(sampler.MaxAniso(), 1.0f) : 1.0f;
    out.compare = is_depth;
    out.compare_op = static_cast<u32>(LiverpoolToVK::DepthCompare(sampler.depth_compare_func));
    out.min_lod = sampler.MinLod();
    out.max_lod = sampler.MaxLod();
    auto border = LiverpoolToVK::BorderColor(sampler.border_color_type);
    if (border == vk::BorderColor::eFloatCustomEXT) {
        border = vk::BorderColor::eFloatOpaqueBlack;
    }
    out.border = static_cast<u32>(border);
}

// The resources of a translated stage, in the recompiler's binding order (buffers, images,
// samplers), with what the native renderer needs to provide each, and its flattened user data.
static void FillBindings(const Shader::Info& info, TranslatedProgram& out) {
    out.flatbuf = info.flattened_ud_buf;
    out.flatbuf_reads.clear();
    if (info.srt_info.walker_func) {
        // The walk again, for its log of reads (Info::RefreshFlatBuf keeps it to itself).
        static std::vector<u64> log(Shader::SrtReadLogSize);
        std::vector<u32> scratch(info.flattened_ud_buf.size());
        const u64* end =
            info.srt_info.walker_func(info.user_data.data(), scratch.data(), log.data());
        out.flatbuf_reads.assign(static_cast<const u64*>(log.data()), end);
    }
    out.images = static_cast<u32>(info.images.size());
    out.samplers = static_cast<u32>(info.samplers.size());
    out.image_bindings.clear();
    for (const auto& desc : info.images) {
        const AmdGpu::Image sharp = desc.GetSharp(info);
        ImageBinding& image = out.image_bindings.emplace_back();
        image.address = sharp.Address();
        image.width = sharp.width + 1;
        image.height = sharp.height + 1;
        image.depth = sharp.depth + 1;
        image.pitch = sharp.Pitch();
        image.base_level = sharp.base_level;
        image.last_level = sharp.last_level;
        image.type = static_cast<u32>(sharp.GetType());
        image.data_format = static_cast<u32>(sharp.GetDataFmt());
        image.number_format = static_cast<u32>(sharp.GetNumberFmt());
        image.tile_mode = static_cast<u32>(sharp.GetTileMode());
        image.written = desc.is_written;
        image.atomic = desc.is_atomic;
        image.depth_image = desc.is_depth;
        image.per_mip_descriptors = desc.NumBindings(info) != 1;
        FillImage(image, sharp, desc);
    }
    out.sampler_bindings.clear();
    for (const auto& desc : info.samplers) {
        FillSampler(out.sampler_bindings.emplace_back(), desc.GetSharp(info), desc.is_depth);
    }
    out.buffers.clear();
    for (const auto& desc : info.buffers) {
        BufferBinding& binding = out.buffers.emplace_back();
        binding.written = desc.is_written;
        binding.formatted = desc.is_formatted;
        switch (desc.buffer_type) {
        case Shader::BufferType::Guest: {
            const AmdGpu::Buffer sharp = desc.GetSharp(info);
            binding.kind = BufferKind::Guest;
            binding.address = sharp.base_address;
            binding.size = sharp.base_address ? sharp.GetSize() : 0;
            break;
        }
        case Shader::BufferType::Flatbuf:
            binding.kind = BufferKind::Flatbuf;
            break;
        case Shader::BufferType::SharedMemory:
            binding.kind = BufferKind::SharedMemory;
            break;
        case Shader::BufferType::BdaPagetable:
            binding.kind = BufferKind::PageTable;
            break;
        case Shader::BufferType::FaultBuffer:
            binding.kind = BufferKind::FaultBuffer;
            break;
        case Shader::BufferType::ClipPlanes:
            binding.kind = BufferKind::Inline; // filled by TranslateDraw
            break;
        default:
            binding.kind = BufferKind::Unsupported;
            binding.name = SpecialName(desc.buffer_type);
            break;
        }
    }
}

bool TranslateCompute(const ComputeProgram& program, TranslatedProgram& out, std::string& error) {
    // The registers as the GPU layer keeps them (AmdGpu::ComputeProgram, COMPUTE_* order).
    AmdGpu::ComputeProgram cs{};
    cs.address = program.code_address >> 8;
    const std::array<u32, 2> settings = {program.rsrc1, program.rsrc2};
    std::memcpy(&cs.settings, settings.data(), sizeof(cs.settings));
    cs.num_thread_x.full = static_cast<u16>(program.num_threads[0]);
    cs.num_thread_y.full = static_cast<u16>(program.num_threads[1]);
    cs.num_thread_z.full = static_cast<u16>(program.num_threads[2]);
    cs.user_data = program.user_data;

    const Shader::ShaderParams params = AmdGpu::GetParams(cs);
    if (params.code.empty()) {
        error = "no shader binary info (OrbShdr) after the code";
        return false;
    }

    // As PipelineCache::BuildRuntimeInfo does for the compute stage.
    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(Shader::HwStage::Compute, Shader::SwStage::Compute);
    runtime_info.props.num_user_data = cs.settings.num_user_regs;
    runtime_info.props.num_allocated_vgprs = cs.settings.num_vgprs * 4;
    runtime_info.props.fp_denorm_mode32 = cs.settings.fp_denorm_mode32;
    runtime_info.props.fp_denorm_mode16_64 = cs.settings.fp_denorm_mode64;
    runtime_info.props.fp_round_mode32 = cs.settings.fp_round_mode32;
    runtime_info.props.fp_round_mode16_64 = cs.settings.fp_round_mode64;
    runtime_info.props.dx10_clamp = cs.settings.dx10_clamp;
    runtime_info.hw.cs.workgroup_size = {cs.num_thread_x.full, cs.num_thread_y.full,
                                         cs.num_thread_z.full};
    runtime_info.hw.cs.tgid_enable = {cs.IsTgidEnabled(0), cs.IsTgidEnabled(1),
                                      cs.IsTgidEnabled(2)};
    runtime_info.hw.cs.shared_memory_size = cs.SharedMemSize();

    const Shader::Profile profile = NativeProfile();
    Shader::Info info{Shader::HwStage::Compute, Shader::SwStage::Compute, params};
    Shader::Pools pools;
    Shader::Backend::Bindings bindings{};
    const auto ir_program = Shader::TranslateProgram(params.code, pools, info, runtime_info, profile);
    out.spirv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, ir_program, bindings);
    if (info.translation_failed) {
        error = "the recompiler could not translate some instructions";
        return false;
    }
    info.RefreshFlatBuf();

    out.hash = params.hash;
    out.workgroup_size = runtime_info.hw.cs.workgroup_size;
    out.shared_memory_bytes = cs.SharedMemSize();
    FillBindings(info, out);
    return true;
}

// --- draws ------------------------------------------------------------------------------------

namespace {

// PipelineCache's MapOutputs (vk_pipeline_cache.cpp, shadPS4): which VS output vectors hold what
// besides the parameters (point size, clip and cull distances...), from PA_CL_VS_OUT_CNTL.
u32 MapOutputs(std::span<Shader::OutputMap, 3> outputs, const AmdGpu::VsOutputControl& ctl) {
    using Shader::Output;
    u32 num_outputs = 0;
    if (ctl.vs_out_misc_enable) {
        auto& misc_vec = outputs[num_outputs++];
        misc_vec[0] = ctl.use_vtx_point_size ? Output::PointSize : Output::None;
        misc_vec[1] = ctl.use_vtx_edge_flag
                          ? Output::EdgeFlag
                          : (ctl.use_vtx_gs_cut_flag ? Output::GsCutFlag : Output::None);
        misc_vec[2] =
            ctl.use_vtx_kill_flag
                ? Output::KillFlag
                : (ctl.use_vtx_render_target_idx ? Output::RenderTargetIndex : Output::None);
        misc_vec[3] = ctl.use_vtx_viewport_idx ? Output::ViewportIndex : Output::None;
    }
    const auto distances = [&](u32 first, Output clip, Output cull) {
        auto& vec = outputs[num_outputs++];
        for (u32 i = 0; i < 4; i++) {
            vec[i] = ctl.IsClipDistEnabled(first + i)
                         ? Output(u32(clip) + i)
                         : (ctl.IsCullDistEnabled(first + i) ? Output(u32(cull) + i) : Output::None);
        }
    };
    if (ctl.vs_out_ccdist0_enable) {
        distances(0, Output::ClipDist0, Output::CullDist0);
    }
    if (ctl.vs_out_ccdist1_enable) {
        distances(4, Output::ClipDist4, Output::CullDist4);
    }
    return num_outputs;
}

// PipelineCache::BuildRuntimeInfo's part common to every stage.
void BuildCommon(Shader::RuntimeInfo& info, const AmdGpu::ShaderProgram& program) {
    info.props.fp_denorm_mode32 = program.settings.fp_denorm_mode32;
    info.props.fp_denorm_mode16_64 = program.settings.fp_denorm_mode64;
    info.props.fp_round_mode32 = program.settings.fp_round_mode32;
    info.props.fp_round_mode16_64 = program.settings.fp_round_mode64;
    info.props.num_allocated_vgprs = program.NumVgprs();
    info.props.num_user_data = program.settings.num_user_regs;
    info.props.num_input_vgprs = program.settings.vgpr_comp_cnt;
    info.props.dx10_clamp = program.settings.dx10_clamp;
}

} // namespace

bool TranslateDraw(const DrawState& state, const u32* packet, TranslatedDraw& out,
                   std::string& error) {
    out = TranslatedDraw{};
    if (state.regs.size() != AmdGpu::Regs::NumRegs) {
        error = "incomplete register file";
        return false;
    }
    const auto regs_storage = std::make_unique<AmdGpu::Regs>();
    AmdGpu::Regs& regs = *regs_storage;
    std::memcpy(regs.reg_array.data(), state.regs.data(), sizeof(regs.reg_array));

    // What the draw packet puts in the registers (Liverpool::ProcessGraphics; the PM4Cmd*
    // structs in pm4_cmds.h, dword 0 the header).
    u32 index_offset = 0;
    switch ((packet[0] >> 8) & 0xff) {
    case 0x27: // DRAW_INDEX_2: max size, index base lo, index base hi, count, initiator
        regs.max_index_size = packet[1];
        regs.index_base_address.base_addr_lo = packet[2];
        regs.index_base_address.base_addr_hi = packet[3];
        regs.num_indices = packet[4];
        out.indexed = true;
        break;
    case 0x35: // DRAW_INDEX_OFFSET_2: max size, index offset, count, initiator
        regs.max_index_size = packet[1];
        index_offset = packet[2];
        regs.num_indices = packet[3];
        out.indexed = true;
        break;
    case 0x2d: // DRAW_INDEX_AUTO: count, initiator
        regs.num_indices = packet[1];
        break;
    default:
        out.skipped = std::format("draw packet {:#x}", (packet[0] >> 8) & 0xff);
        return true;
    }

    // Draws the GPU layer does not render as draws (Rasterizer::FilterDrawPasses).
    using Mode = AmdGpu::ColorControl::OperationMode;
    const auto mode = regs.color_control.mode;
    if (mode != Mode::Normal && mode != Mode::Disable) {
        out.skipped = std::format("color control mode {}", u32(mode));
        return true;
    }
    if (regs.primitive_type == AmdGpu::PrimitiveType::None || !regs.num_indices) {
        out.skipped = "empty draw";
        return true;
    }
    if (regs.stage_enable.raw != AmdGpu::ShaderStageEnable::VgtStages::Vs) {
        out.skipped = std::format("shader stages {:#x}", u32(regs.stage_enable.raw));
        return true;
    }

    if (!regs.ps_program.Address<u32*>() || !regs.vs_program.Address<u32*>()) {
        out.skipped = "no pixel or vertex shader";
        return true;
    }

    // Color targets as the shader recompiler sees them (PipelineCache::RefreshGraphicsKey, first
    // pass).
    std::array<Shader::PsColorBuffer, Shader::MaxColorBuffers> color_buffers{};
    const bool skip_cb_binding = mode == Mode::Disable;
    for (u32 cb = 0; cb < AmdGpu::NUM_COLOR_BUFFERS && !skip_cb_binding; ++cb) {
        const auto& col_buf = regs.color_buffers[cb];
        if (!col_buf || !regs.color_target_mask.GetMask(cb)) {
            continue;
        }
        if (col_buf.NumSamples() > 1) {
            out.skipped = "multisampled color target";
            return true;
        }
        auto& color_buffer = color_buffers[cb];
        color_buffer.data_format = col_buf.GetDataFmt();
        color_buffer.num_format = col_buf.GetNumberFmt();
        color_buffer.num_conversion = col_buf.GetNumberConversion();
        color_buffer.export_format = regs.color_export_format.GetFormat(cb);
        color_buffer.swizzle = col_buf.Swizzle();
        const auto& bc = regs.blend_control[cb];
        color_buffer.blend_self_scale =
            bc.enable && !col_buf.info.blend_bypass &&
            (bc.color_func == AmdGpu::BlendControl::BlendFunc::Min ||
             bc.color_func == AmdGpu::BlendControl::BlendFunc::Max) &&
            bc.color_src_factor == AmdGpu::BlendControl::BlendFactor::SrcColor &&
            bc.color_dst_factor == AmdGpu::BlendControl::BlendFactor::DstColor;
    }

    // --- shaders: the pixel shader first (its bindings come first), then the vertex shader ---
    const Shader::Profile profile = NativeProfile();
    Shader::Backend::Bindings bindings{};

    const Shader::ShaderParams ps_params = AmdGpu::GetParams(regs.ps_program);
    const Shader::ShaderParams vs_params = AmdGpu::GetParams(regs.vs_program);
    if (ps_params.code.empty() || vs_params.code.empty()) {
        error = "no shader binary info (OrbShdr) after the code";
        return false;
    }

    // PipelineCache::BuildRuntimeInfo, fragment stage.
    Shader::RuntimeInfo ps_runtime{};
    ps_runtime.Initialize(Shader::HwStage::Fragment, Shader::SwStage::Fragment);
    BuildCommon(ps_runtime, regs.ps_program);
    auto& fs = ps_runtime.hw.fs;
    fs.en_flags = regs.ps_input_ena;
    fs.addr_flags = regs.ps_input_addr;
    fs.num_inputs = regs.num_interp;
    fs.front_face_all_bits = regs.barycentric_control.front_face_all_bits;
    fs.depth_before_shader = regs.depth_shader_control.depth_before_shader;
    fs.num_samples = 1;
    fs.z_export_format = regs.z_export_format;
    const u8 stencil_ref_export_enable = regs.depth_shader_control.stencil_op_val_export_enable |
                                         regs.depth_shader_control.stencil_test_val_export_enable;
    fs.mrtz_mask = regs.depth_shader_control.z_export_enable | (stencil_ref_export_enable << 1) |
                   (regs.depth_shader_control.mask_export_enable << 2) |
                   (regs.depth_shader_control.coverage_to_mask_enable << 3);
    const auto& cb0_blend = regs.blend_control[0];
    fs.dual_source_blending = false;
    if (cb0_blend.enable) {
        using Vulkan::LiverpoolToVK::IsDualSourceBlendFactor;
        fs.dual_source_blending = IsDualSourceBlendFactor(cb0_blend.color_dst_factor) ||
                                  IsDualSourceBlendFactor(cb0_blend.color_src_factor);
        if (cb0_blend.separate_alpha_blend) {
            fs.dual_source_blending |= IsDualSourceBlendFactor(cb0_blend.alpha_dst_factor) ||
                                       IsDualSourceBlendFactor(cb0_blend.alpha_src_factor);
        }
    }
    if (fs.dual_source_blending) {
        out.skipped = "dual source blending";
        return true;
    }
    for (u32 i = 0; i < regs.num_interp; i++) {
        fs.inputs[i] = {
            .param_index = u16(regs.ps_inputs[i].input_offset),
            .is_default = u16(regs.ps_inputs[i].use_default),
            .is_flat = u16(regs.ps_inputs[i].flat_shade),
            .default_value = u16(regs.ps_inputs[i].default_value),
        };
    }
    fs.color_buffers = color_buffers;
    fs.clip_distance_emulation = false; // only needed on NVIDIA's driver

    Shader::Info ps_info{Shader::HwStage::Fragment, Shader::SwStage::Fragment, ps_params};
    Shader::Pools ps_pools;
    const auto ps_ir = Shader::TranslateProgram(ps_params.code, ps_pools, ps_info, ps_runtime, profile);
    out.ps.spirv = Shader::Backend::SPIRV::EmitSPIRV(profile, ps_runtime, ps_ir, bindings);
    if (ps_info.translation_failed) {
        error = "the recompiler could not translate some pixel shader instructions";
        return false;
    }
    ps_info.RefreshFlatBuf();
    out.ps.hash = ps_params.hash;
    FillBindings(ps_info, out.ps);

    // PipelineCache::BuildRuntimeInfo, vertex stage.
    Shader::RuntimeInfo vs_runtime{};
    vs_runtime.Initialize(Shader::HwStage::Vertex, Shader::SwStage::Vertex);
    BuildCommon(vs_runtime, regs.vs_program);
    auto& vs = vs_runtime.hw.vs;
    vs.user_clip_plane_mask = regs.clipper_control.user_clip_plane_enable;
    vs.num_outputs = MapOutputs(vs.outputs, regs.vs_output_control);
    vs.emulate_depth_negative_one_to_one = false; // VK_EXT_depth_clip_control instead
    vs.clip_disable = regs.IsClipDisabled();
    auto& sw_vs = vs_runtime.sw.vs;
    sw_vs.step_rate_0 = regs.vgt_instance_step_rate_0;
    sw_vs.step_rate_1 = regs.vgt_instance_step_rate_1;
    sw_vs.vertex_sgpr_offset = 0;
    sw_vs.instance_sgpr_offset = 0;
    sw_vs.tess_emulated_primitive = regs.primitive_type == AmdGpu::PrimitiveType::RectList ||
                                    regs.primitive_type == AmdGpu::PrimitiveType::QuadList;

    Shader::Info vs_info{Shader::HwStage::Vertex, Shader::SwStage::Vertex, vs_params};
    Shader::Pools vs_pools;
    const auto vs_ir = Shader::TranslateProgram(vs_params.code, vs_pools, vs_info, vs_runtime, profile);
    out.vs.spirv = Shader::Backend::SPIRV::EmitSPIRV(profile, vs_runtime, vs_ir, bindings);
    if (vs_info.translation_failed) {
        error = "the recompiler could not translate some vertex shader instructions";
        return false;
    }
    vs_info.RefreshFlatBuf();
    out.vs.hash = vs_params.hash;
    FillBindings(vs_info, out.vs);
    // The user clip planes (Rasterizer::BindBuffers): PA_CL_UCP_n as 6 vec4, read only when
    // enabled.
    for (BufferBinding& binding : out.vs.buffers) {
        if (binding.kind != BufferKind::Inline) {
            continue;
        }
        binding.data.assign(AmdGpu::NUM_CLIP_PLANES * 4, 0);
        if (regs.clipper_control.user_clip_plane_enable) {
            for (u32 i = 0; i < AmdGpu::NUM_CLIP_PLANES; ++i) {
                const auto& plane = regs.clip_user_data[i];
                binding.data[i * 4 + 0] = plane.data_x;
                binding.data[i * 4 + 1] = plane.data_y;
                binding.data[i * 4 + 2] = plane.data_z;
                binding.data[i * 4 + 3] = plane.data_w;
            }
        }
    }

    // Vertex attributes (GraphicsPipeline::GetVertexInputs; one binding per attribute).
    Shader::Gcn::FetchShaderData fetch{};
    if (Shader::Gcn::ParseFetchShader(vs_info, fetch)) {
        using InstanceIdType = Shader::Gcn::VertexAttribute::InstanceIdType;
        for (const auto& attrib : fetch.attributes) {
            const AmdGpu::Buffer buffer = attrib.GetSharp(vs_info);
            const auto step_rate = attrib.GetStepRate();
            out.vertex_inputs.push_back({
                .vk_format = u32(Vulkan::LiverpoolToVK::SurfaceFormat(buffer.GetDataFmt(),
                                                                       buffer.GetNumberFmt())),
                .address = buffer.base_address,
                .size = buffer.base_address ? buffer.GetSize() : 0,
                .stride = buffer.GetStride(),
                .per_instance = step_rate != InstanceIdType::None,
                .divisor = step_rate == InstanceIdType::OverStepRate0
                               ? regs.vgt_instance_step_rate_0
                               : (step_rate == InstanceIdType::OverStepRate1
                                      ? regs.vgt_instance_step_rate_1
                                      : 1u),
            });
        }
    }

    // --- fixed function state (GraphicsPipeline's constructor) ---------------------------------
    const auto prim_type = regs.primitive_type;
    out.topology = u32(Vulkan::LiverpoolToVK::PrimitiveType(prim_type));
    const bool is_rect_list = prim_type == AmdGpu::PrimitiveType::RectList;
    const bool is_quad_list = prim_type == AmdGpu::PrimitiveType::QuadList;
    if (prim_type == AmdGpu::PrimitiveType::PatchPrimitive) {
        out.skipped = "patches";
        return true;
    }
    if (is_rect_list || is_quad_list) {
        using Shader::Backend::SPIRV::AuxShaderType;
        out.patch_points = is_rect_list ? 3 : 4;
        out.tcs_spirv = Shader::Backend::SPIRV::EmitAuxilaryTessShader(
            is_quad_list ? AuxShaderType::QuadListTCS : AuxShaderType::RectListTCS, fs);
        out.tes_spirv =
            Shader::Backend::SPIRV::EmitAuxilaryTessShader(AuxShaderType::PassthroughTES, fs);
    }
    out.polygon_mode = u32(Vulkan::LiverpoolToVK::PolygonMode(regs.polygon_control.PolyMode()));
    out.depth_clip = regs.clipper_control.ZclipEnable();
    out.depth_clamp = !regs.depth_render_override.disable_viewport_clamp;
    out.provoking_last = regs.polygon_control.provoking_vtx_last != AmdGpu::ProvokingVtxLast::First;
    out.negative_one_to_one = regs.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW;

    // Rasterizer::UpdatePrimitiveState.
    out.primitive_restart = (regs.enable_primitive_restart & 1) != 0;
    if (out.primitive_restart && out.indexed && regs.primitive_restart_index != 0xFFFF &&
        regs.primitive_restart_index != 0xFFFFFFFF) {
        out.skipped = "primitive restart index other than -1";
        return true;
    }
    out.cull_mode = Vulkan::LiverpoolToVK::IsPrimitiveCulled(prim_type)
                        ? u32(Vulkan::LiverpoolToVK::CullMode(regs.polygon_control.CullingMode()))
                        : 0u;
    out.front_face = u32(Vulkan::LiverpoolToVK::FrontFace(regs.polygon_control.front_face));
    out.rasterizer_discard = regs.clipper_control.dx_rasterization_kill;

    // Rasterizer::UpdateViewportScissorState, for the first viewport in use.
    {
        const auto tl = [](s16 scr, s16 win, s16 gen, s16 win_offset) {
            return std::max({scr, s16(win + win_offset), s16(gen + win_offset)});
        };
        const auto br = [](s16 scr, s16 win, s16 gen, s16 win_offset) {
            return std::min({scr, s16(win + win_offset), s16(gen + win_offset)});
        };
        const bool enable_offset = !regs.window_scissor.window_offset_disable;
        const s16 x_offset = enable_offset ? regs.window_offset.window_x_offset : 0;
        const s16 y_offset = enable_offset ? regs.window_offset.window_y_offset : 0;
        AmdGpu::Scissor scsr{};
        scsr.top_left_x = tl(regs.screen_scissor.top_left_x, s16(regs.window_scissor.top_left_x),
                             s16(regs.generic_scissor.top_left_x), x_offset);
        scsr.top_left_y = tl(regs.screen_scissor.top_left_y, s16(regs.window_scissor.top_left_y),
                             s16(regs.generic_scissor.top_left_y), y_offset);
        scsr.bottom_right_x = br(regs.screen_scissor.bottom_right_x,
                                 regs.window_scissor.bottom_right_x,
                                 regs.generic_scissor.bottom_right_x, x_offset);
        scsr.bottom_right_y = br(regs.screen_scissor.bottom_right_y,
                                 regs.window_scissor.bottom_right_y,
                                 regs.generic_scissor.bottom_right_y, y_offset);
        if (regs.polygon_control.enable_window_offset && (x_offset != 0 || y_offset != 0)) {
            out.skipped = "window offset";
            return true;
        }
        const auto& vp_ctl = regs.viewport_control;
        bool found = false;
        for (u32 i = 0; i < AmdGpu::NUM_VIEWPORTS && !found; i++) {
            const auto& vp = regs.viewports[i];
            if (vp.xscale == 0) {
                continue;
            }
            found = true;
            const float zoffset = vp_ctl.zoffset_enable ? vp.zoffset : 0.f;
            const float zscale = vp_ctl.zscale_enable ? vp.zscale : 1.f;
            float min_depth, max_depth;
            if (regs.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW) {
                min_depth = zoffset - zscale;
                max_depth = zoffset + zscale;
            } else {
                min_depth = zoffset;
                max_depth = zoffset + zscale;
            }
            if (regs.IsClipDisabled()) {
                out.viewport = {0.f, 0.f, 16384.f, 16384.f, min_depth, max_depth};
            } else {
                const float xoffset = vp_ctl.xoffset_enable ? vp.xoffset : 0.f;
                const float xscale = vp_ctl.xscale_enable ? vp.xscale : 1.f;
                const float yoffset = vp_ctl.yoffset_enable ? vp.yoffset : 0.f;
                const float yscale = vp_ctl.yscale_enable ? vp.yscale : 1.f;
                out.viewport = {xoffset - xscale, yoffset - yscale, xscale * 2.0f,
                                yscale * 2.0f, min_depth, max_depth};
            }
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
            out.scissor_offset = {vp_scsr.top_left_x, vp_scsr.top_left_y};
            out.scissor_extent = {vp_scsr.GetWidth(), vp_scsr.GetHeight()};
        }
        if (!found) {
            out.viewport = {-1.f, -1.f, 1.f, 1.f, 0.f, 1.f};
            out.scissor_offset = {0, 0};
            out.scissor_extent = {1, 1};
        }
        // MakeUserData (vk_rasterizer.cpp): the shader's own viewport transform when clipping is
        // off.
        out.push_viewport = {
            vp_ctl.xoffset_enable ? regs.viewports[0].xoffset : 0.f,
            vp_ctl.yoffset_enable ? regs.viewports[0].yoffset : 0.f,
            vp_ctl.xscale_enable ? regs.viewports[0].xscale : 1.f,
            vp_ctl.yscale_enable ? regs.viewports[0].yscale : 1.f,
        };
    }

    // --- color targets and blending (RefreshGraphicsKey, second pass; GraphicsPipeline;
    // PrepareRenderState / BeginRendering) ----------------------------------------------------
    out.blend_constants = regs.blend_constants;
    out.logic_op_enable = regs.color_control.rop3 != AmdGpu::ColorControl::LogicOp::Copy;
    out.logic_op = u32(Vulkan::LiverpoolToVK::LogicOp(regs.color_control.rop3));
    const u32 mrt_mask = ps_info.mrt_mask;
    const u32 num_attachments = std::bit_width(mrt_mask);
    out.colors.resize(num_attachments);
    out.width = 16384;
    out.height = 16384;
    for (u32 cb = 0; cb < num_attachments; cb++) {
        const auto& col_buf = regs.color_buffers[cb];
        const u32 target_mask = regs.color_target_mask.GetMask(cb);
        if (skip_cb_binding || !col_buf || !target_mask || !(mrt_mask & (1u << cb))) {
            continue;
        }
        ColorTarget& target = out.colors[cb];
        target.used = true;
        const AmdGpu::CbDbExtent hint{.raw = state.cb_extent[cb]};
        const VideoCore::ImageInfo info{col_buf, hint};
        FillImageLayout(target.image, info);
        target.image.address = info.guest_address;
        const VideoCore::ImageViewInfo view{col_buf};
        const auto format =
            Vulkan::LiverpoolToVK::SurfaceFormat(col_buf.GetDataFmt(), col_buf.GetNumberFmt());
        target.view_format = u32(format);
        target.view_level = view.range.base.level;
        target.view_layer = view.range.base.layer;
        target.view_layers = view.range.extent.layers;
        if (!(view.mapping == AmdGpu::IdentityMapping)) {
            const vk::ComponentMapping mapping =
                Vulkan::LiverpoolToVK::ComponentMapping(view.mapping);
            target.swizzle = {u32(mapping.r), u32(mapping.g), u32(mapping.b), u32(mapping.a)};
        }
        target.cmask_address = col_buf.CmaskAddress();
        target.meta_tracked = info.meta_info.cmask_addr != 0;
        const vk::ClearValue clear = Vulkan::LiverpoolToVK::ColorBufferClearValue(col_buf);
        std::memcpy(target.clear_value.data(), clear.color.uint32.data(), 16);
        out.width = std::min(out.width, std::max(info.size.width >> target.view_level, 1u));
        out.height = std::min(out.height, std::max(info.size.height >> target.view_level, 1u));

        target.write_mask = color_buffers[cb].swizzle.ApplyMask(target_mask);
        const auto& control = regs.blend_control[cb];
        if (!control.enable || col_buf.info.blend_bypass) {
            continue;
        }
        using namespace Vulkan;
        const auto src_color = LiverpoolToVK::BlendFactor(control.color_src_factor);
        const auto dst_color = LiverpoolToVK::BlendFactor(control.color_dst_factor);
        const auto color_blend = LiverpoolToVK::BlendOp(control.color_func);
        const auto src_alpha = control.separate_alpha_blend
                                   ? LiverpoolToVK::BlendFactor(control.alpha_src_factor)
                                   : src_color;
        const auto dst_alpha = control.separate_alpha_blend
                                   ? LiverpoolToVK::BlendFactor(control.alpha_dst_factor)
                                   : dst_color;
        const auto alpha_blend =
            control.separate_alpha_blend ? LiverpoolToVK::BlendOp(control.alpha_func) : color_blend;
        auto eff_src_alpha = src_alpha;
        auto eff_dst_alpha = dst_alpha;
        auto eff_alpha_blend = alpha_blend;
        if (alpha_blend == vk::BlendOp::eMin || alpha_blend == vk::BlendOp::eMax) {
            const bool takes_max = alpha_blend == vk::BlendOp::eMax;
            if (src_alpha == vk::BlendFactor::eOne && dst_alpha == vk::BlendFactor::eZero) {
                eff_alpha_blend = vk::BlendOp::eAdd;
                eff_src_alpha = takes_max ? vk::BlendFactor::eOne : vk::BlendFactor::eZero;
                eff_dst_alpha = vk::BlendFactor::eZero;
            } else if (src_alpha == vk::BlendFactor::eZero && dst_alpha == vk::BlendFactor::eOne) {
                eff_alpha_blend = vk::BlendOp::eAdd;
                eff_src_alpha = vk::BlendFactor::eZero;
                eff_dst_alpha = takes_max ? vk::BlendFactor::eOne : vk::BlendFactor::eZero;
            }
        }
        target.blend = true;
        target.src_color = u32(src_color);
        target.dst_color = u32(dst_color);
        target.color_op = u32(color_blend);
        target.src_alpha = u32(eff_src_alpha);
        target.dst_alpha = u32(eff_dst_alpha);
        target.alpha_op = u32(eff_alpha_blend);
        if (color_buffers[cb].blend_self_scale) {
            target.src_color = u32(vk::BlendFactor::eOne);
            target.dst_color = u32(vk::BlendFactor::eOne);
        }
        // Alpha masked out of the export but used by the blend: the hardware blends with 1.
        const bool alpha_masked_out =
            (regs.color_shader_mask.GetMask(cb) & AmdGpu::ColorBufferMask::ComponentA) == 0;
        if (alpha_masked_out && (src_color == vk::BlendFactor::eSrcAlpha ||
                                 src_color == vk::BlendFactor::eOneMinusSrcAlpha)) {
            target.src_color = u32(src_color == vk::BlendFactor::eSrcAlpha
                                       ? vk::BlendFactor::eOne
                                       : vk::BlendFactor::eZero);
        }
        if (alpha_masked_out && (dst_color == vk::BlendFactor::eSrcAlpha ||
                                 dst_color == vk::BlendFactor::eOneMinusSrcAlpha)) {
            target.dst_color = u32(dst_color == vk::BlendFactor::eSrcAlpha
                                       ? vk::BlendFactor::eOne
                                       : vk::BlendFactor::eZero);
        }
    }

    // --- depth target and test (PrepareRenderState, BeginRendering, UpdateDepthStencilState) ---
    const bool depth_test = regs.depth_control.depth_enable && regs.depth_buffer.DepthValid();
    const bool stencil_test = regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid();
    if (stencil_test && !regs.depth_buffer.DepthValid()) {
        out.skipped = "stencil target without depth";
        return true;
    }
    if (depth_test || stencil_test) {
        DepthTarget& depth = out.depth;
        depth.used = true;
        const auto htile_address = regs.depth_htile_data_base.GetAddress();
        const AmdGpu::CbDbExtent hint{.raw = state.db_extent};
        const VideoCore::ImageInfo info{regs.depth_buffer, regs.depth_view.NumSlices(),
                                        htile_address, hint, false};
        FillImageLayout(depth.image, info);
        depth.image.address = info.guest_address;
        const VideoCore::ImageViewInfo view{regs.depth_buffer, regs.depth_view,
                                            regs.depth_control};
        if (view.range.extent.levels != 1 || view.range.base.level != 0) {
            out.skipped = "depth target mip";
            return true;
        }
        depth.view_layer = view.range.base.layer;
        depth.view_layers = view.range.extent.layers;
        depth.test = depth_test;
        depth.write = depth_test && regs.depth_control.depth_write_enable &&
                      !regs.depth_render_control.depth_clear_enable;
        if (stencil_test) {
            using Vulkan::LiverpoolToVK::CompareOp;
            using Vulkan::LiverpoolToVK::StencilOp;
            const auto& sc = regs.stencil_control;
            const bool backface = regs.depth_control.backface_enable;
            const bool stencil_clear = regs.depth_render_control.stencil_clear_enable;
            // GCN REPLACE_OP writes STENCILOPVAL: a face whose ops replace takes its Vulkan
            // reference from op_val, the others from test_val.
            const auto uses_op_val = [](AmdGpu::StencilFunc fail, AmdGpu::StencilFunc zpass,
                                        AmdGpu::StencilFunc zfail) {
                return fail == AmdGpu::StencilFunc::ReplaceOp ||
                       zpass == AmdGpu::StencilFunc::ReplaceOp ||
                       zfail == AmdGpu::StencilFunc::ReplaceOp;
            };
            const bool front_op =
                uses_op_val(sc.stencil_fail_front, sc.stencil_zpass_front, sc.stencil_zfail_front);
            const bool back_op =
                backface ? uses_op_val(sc.stencil_fail_back, sc.stencil_zpass_back,
                                       sc.stencil_zfail_back)
                         : front_op;
            const auto& ref_front = regs.stencil_ref_front;
            const auto& ref_back = backface ? regs.stencil_ref_back : regs.stencil_ref_front;
            depth.stencil = true;
            depth.front = {
                .fail_op = u32(StencilOp(sc.stencil_fail_front)),
                .pass_op = u32(StencilOp(sc.stencil_zpass_front)),
                .depth_fail_op = u32(StencilOp(sc.stencil_zfail_front)),
                .compare_op = u32(CompareOp(regs.depth_control.stencil_ref_func)),
                .compare_mask = ref_front.stencil_mask,
                .write_mask = stencil_clear ? 0u : u32(ref_front.stencil_write_mask),
                .reference = front_op ? ref_front.stencil_op_val : ref_front.stencil_test_val,
            };
            depth.back = backface ? StencilFace{
                .fail_op = u32(StencilOp(sc.stencil_fail_back)),
                .pass_op = u32(StencilOp(sc.stencil_zpass_back)),
                .depth_fail_op = u32(StencilOp(sc.stencil_zfail_back)),
                .compare_op = u32(CompareOp(regs.depth_control.stencil_bf_func)),
                .compare_mask = ref_back.stencil_mask,
                .write_mask = stencil_clear ? 0u : u32(ref_back.stencil_write_mask),
                .reference = back_op ? ref_back.stencil_op_val : ref_back.stencil_test_val,
            } : depth.front;
            depth.stencil_clear = stencil_clear;
            depth.stencil_clear_value = regs.stencil_clear;
        }
        depth.compare_op = u32(Vulkan::LiverpoolToVK::CompareOp(regs.depth_control.depth_func));
        depth.bounds = regs.depth_control.depth_bounds_enable;
        depth.bounds_min = regs.depth_bounds_min;
        depth.bounds_max = regs.depth_bounds_max;
        depth.clear_pass = depth_test && regs.depth_render_control.depth_clear_enable &&
                           regs.depth_control.depth_write_enable;
        depth.htile_address = htile_address;
        depth.meta_tracked = info.meta_info.htile_addr != 0;
        depth.clear_value = regs.depth_clear;
        out.width = std::min(out.width, info.size.width);
        out.height = std::min(out.height, info.size.height);
    }
    out.depth_bias = regs.polygon_control.NeedsBias();
    if (out.depth_bias) {
        const bool front = regs.polygon_control.enable_polygon_offset_front;
        out.bias_constant = front ? regs.poly_offset.front_offset : regs.poly_offset.back_offset;
        out.bias_clamp = regs.poly_offset.depth_bias;
        out.bias_slope = (front ? regs.poly_offset.front_scale : regs.poly_offset.back_scale) / 16.f;
    }

    // --- draw arguments (Rasterizer::DrawRecord, GetDrawOffsets, ObtainIndexBuffer) -----------
    out.count = regs.num_indices;
    out.instances = regs.num_instances.NumInstances();
    out.vertex_offset = regs.index_offset;
    if (!fetch.Empty()) {
        if (out.vertex_offset == 0 && fetch.vertex_offset_sgpr != -1) {
            out.vertex_offset = vs_info.UserData()[fetch.vertex_offset_sgpr];
        }
        if (fetch.instance_offset_sgpr != -1) {
            out.instance_offset = vs_info.UserData()[fetch.instance_offset_sgpr];
        }
    }
    if (out.indexed) {
        out.index32 = regs.index_buffer_type.index_type != AmdGpu::IndexType::Index16;
        out.index_address = regs.index_base_address.Address<VAddr>() +
                            u64{index_offset} * (out.index32 ? 4 : 2);
    }
    return true;
}

} // namespace render::translate
