// sotcport: the shader recompiler as a tool of the native renderer (interface and binding
// conventions in src/render/shader_translate.h). Mirrors what the GPU layer's pipeline cache does
// for a compute dispatch (vk_pipeline_cache.cpp: RefreshComputeKey, BuildRuntimeInfo,
// CompileModule) and what its rasterizer binds (vk_rasterizer.cpp: BindBuffers), without its
// Vulkan objects.

#include "render/shader_translate.h"

#include <cstring>
#include <format>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/amdgpu/regs_shader.h"
#include "video_core/amdgpu/resource.h"

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
        .sparse_page_shift = 16,
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
        .supports_fragment_shader_barycentric = true,
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
    out.flatbuf = info.flattened_ud_buf;
    out.workgroup_size = runtime_info.hw.cs.workgroup_size;
    out.shared_memory_bytes = cs.SharedMemSize();
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
        default:
            binding.kind = BufferKind::Unsupported;
            binding.name = SpecialName(desc.buffer_type);
            break;
        }
    }
    return true;
}

} // namespace render::translate
