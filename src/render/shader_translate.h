#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "common/types.h"

// Translation of the game's GCN shaders to SPIR-V for the native renderer, through the shader
// recompiler we already have (shadPS4's, used as a tool). The implementation lives with the GPU
// layer (third_party/shadps4/shim/recompiler_glue.cpp) because the recompiler's headers belong
// to it; this interface keeps the native renderer free of them.
//
// The SPIR-V follows the recompiler's binding conventions, which the native renderer mirrors:
// descriptor set 0 with one binding per resource in this order — every buffer (storage buffer),
// then images, then samplers — and a push constant block (Shader::PushData, 128 bytes) whose
// bytes 16 on hold, per buffer binding, how far the bound offset was moved down to meet the
// device's alignment.

namespace render::translate {

// What a compute dispatch runs, as the state model knows it (COMPUTE_* registers).
struct ComputeProgram {
    u64 code_address = 0; // COMPUTE_PGM_LO << 8 (PGM_HI is always 0)
    u32 rsrc1 = 0;
    u32 rsrc2 = 0;
    std::array<u32, 3> num_threads{}; // COMPUTE_NUM_THREAD_X/Y/Z
    std::array<u32, 16> user_data{};  // COMPUTE_USER_DATA_0..15
};

enum class BufferKind : u8 {
    Guest,        // game memory described by a V#: [address, address + size)
    Flatbuf,      // the user data and SRT data the shader reads, flattened (TranslatedProgram::flatbuf)
    SharedMemory, // LDS emulated in a buffer (shared_memory_bytes per workgroup), zeroed
    // Scalar loads from addresses only known while the shader runs ("dynamic" ReadConst) go
    // through a page table: one u64 per DmaPageBits page of game memory, the device address of
    // a copy of that page, or 0 when absent; the load then sets the page's bit in the fault
    // buffer (one bit per page) and reads 0.
    PageTable,
    FaultBuffer,
    Inline,       // data the translation provides (BufferBinding::data: the user clip planes)
    Unsupported,  // something the native renderer does not provide yet (see `name`)
};

// The page size of the page table above (the recompiler's Profile::sparse_page_shift).
constexpr u32 DmaPageBits = 16;
// Pages of the 40-bit address space the game uses.
constexpr u64 DmaPageCount = u64{1} << (40 - DmaPageBits);

struct BufferBinding {
    BufferKind kind = BufferKind::Unsupported;
    u64 address = 0;
    u64 size = 0;
    bool written = false;
    bool formatted = false; // read or written with buffer_load/store_format (typed)
    const char* name = "";  // for Unsupported: which special buffer
    std::vector<u32> data;  // for Inline
};

// An image the shader uses (T#): what the native renderer needs to create it, fill it from game
// memory and bind it as the shader expects. The GPU layer's own computations are reused for every
// part that depends on the PS4's layout (VideoCore::ImageInfo, ImageViewInfo, TileManager).
struct ImageBinding {
    // The T# (for logs and lookups).
    u64 address = 0;
    u32 width = 0, height = 0, depth = 0; // depth: 3D depth or array layers (T# depth + 1)
    u32 pitch = 0;
    u32 base_level = 0, last_level = 0;
    u32 type = 0;          // AmdGpu::ImageType (9 2D, 10 3D, 11 cube, 13 2D array; 8/12 1D)
    u32 data_format = 0;   // AmdGpu::DataFormat
    u32 number_format = 0; // AmdGpu::NumberFormat
    u32 tile_mode = 0;     // AmdGpu::TileMode (tiling_index): 8 = linear aligned
    bool written = false, atomic = false, depth_image = false;
    // Bound as one descriptor per mip (the recompiler's DynamicIndex fallback for storage
    // images with a mip chain): not supported by the native renderer yet.
    bool per_mip_descriptors = false;

    // The image (VideoCore::ImageInfo of the T#).
    u32 vk_format = 0;  // VkFormat
    bool volume = false; // 3D image; otherwise 2D (1D and cubes too, as the GPU layer does)
    u32 levels = 1, layers = 1;
    u32 guest_size = 0; // bytes of game memory it covers (every mip and layer)
    bool tiled = false;
    std::array<u32, 3> extent{}; // ImageInfo::size (what the GPU layer matches images by)
    u32 num_bits = 0;            // ImageInfo::num_bits: bits per texel, or per block
    bool block = false;          // block-compressed (4x4 texels per element)
    // Bytes per element in a buffer copy: num_bits / 8, but only the depth plane of
    // depth-stencil formats (2 or 4 bytes).
    u32 element_bytes = 0;
    // A depth format with a stencil plane. The stencil plane has no layout in game memory here:
    // the GPU layer keeps it on the GPU only; it moves as 1 byte per texel of mip 0, rows
    // extent[0] apart, layer after layer (see StencilBytes).
    bool has_stencil = false;
    u64 StencilBytes() const {
        return u64{extent[0]} * extent[1] * (volume ? 1 : layers);
    }

    // Detiling: the GPU layer's tiling.comp (TileManager::DetileImage) turns the tiled bytes into a
    // linear buffer. Its SPIR-V, specialization constants (4 for micro tiling, 12 for macro) and
    // TilingInfo uniform block (bank swizzle, micro-tiled mips, mip layout), and the dispatch
    // width in workgroups.
    const u32* tiling_spirv = nullptr;
    size_t tiling_spirv_words = 0;
    std::array<u32, 12> tiling_spec{};
    u32 tiling_spec_count = 0;
    std::vector<u8> tiling_info;
    u32 tiling_groups = 0;

    // Copies from the linear bytes into the image, one per mip (TextureCache::RefreshImage).
    struct MipCopy {
        u64 buffer_offset = 0;
        u32 row_length = 0, image_height = 0; // in texels (blocks for compressed formats: 0 = tight)
        u32 width = 0, height = 0, depth = 1;
    };
    std::vector<MipCopy> copies;

    // Calls func(offset, bytes) for every row of texels (of blocks, when compressed) the copies
    // of the first `levels` mips cover in the linear buffer, as vkCmdCopyBufferToImage reads it:
    // rows `row_length` texels apart, slices (depth or array layers, one after another)
    // `image_height` rows apart.
    template <typename F>
    void ForEachRow(u32 levels, F&& func) const {
        const u64 dim = block ? 4 : 1;
        const u64 slices_per_copy = volume ? 1 : layers;
        for (u32 m = 0; m < levels && m < copies.size(); m++) {
            const MipCopy& c = copies[m];
            const u64 row_bytes = ((c.row_length ? c.row_length : c.width) + dim - 1) / dim *
                                  element_bytes;
            const u64 slice_bytes =
                row_bytes * (((c.image_height ? c.image_height : c.height) + dim - 1) / dim);
            const u64 width_bytes = (c.width + dim - 1) / dim * element_bytes;
            const u64 rows = (c.height + dim - 1) / dim;
            for (u64 s = 0; s < c.depth * slices_per_copy; s++) {
                for (u64 y = 0; y < rows; y++) {
                    func(c.buffer_offset + s * slice_bytes + y * row_bytes, width_bytes);
                }
            }
        }
    }

    // The view the shader sees (VideoCore::ImageViewInfo).
    bool storage = false;           // storage image (written); otherwise sampled
    u32 view_type = 0;              // VkImageViewType
    u32 view_format = 0;            // VkFormat
    std::array<u32, 4> swizzle{};   // VkComponentSwizzle r, g, b, a (0: identity)
    u32 view_base_level = 0, view_levels = 1, view_base_layer = 0, view_layers = 1;
    // The T#'s MIN_LOD clamp (VkImageViewMinLodCreateInfoEXT, as the GPU layer's ImageView);
    // 0: none.
    float view_min_lod = 0;
};

// A sampler the shader uses (S#), as a VkSamplerCreateInfo (texture_cache/sampler.cpp). Custom
// border colors fall back to opaque black.
struct SamplerBinding {
    u32 mag_filter = 0, min_filter = 0, mipmap_mode = 0;    // VkFilter, VkSamplerMipmapMode
    u32 address_u = 0, address_v = 0, address_w = 0;        // VkSamplerAddressMode
    float lod_bias = 0, max_anisotropy = 1, min_lod = 0, max_lod = 0;
    bool anisotropy = false, compare = false;
    u32 compare_op = 0;  // VkCompareOp
    u32 border = 0;      // VkBorderColor
};

struct TranslatedProgram {
    u64 hash = 0; // from the shader's OrbShdr footer
    std::vector<u32> spirv;
    std::vector<BufferBinding> buffers; // in binding order
    u32 images = 0;   // bindings after the buffers (not supported by the native renderer yet)
    u32 samplers = 0;
    std::vector<ImageBinding> image_bindings;     // in binding order, after the buffers
    std::vector<SamplerBinding> sampler_bindings; // in binding order, after the images
    std::vector<u32> flatbuf;
    // The reads the SRT walker made to fill flatbuf (mirror addresses, in order; diagnostics).
    std::vector<u64> flatbuf_reads;
    std::array<u32, 3> workgroup_size{};
    u32 shared_memory_bytes = 0;
};

// --- draws ----------------------------------------------------------------------------------

// The GPU register file as a draw sees it (AmdGpu::Regs::reg_array, RegisterCount dwords: context
// registers at 0xA000, SH at 0x2C00, uconfig at 0xC000) and what Gnm writes beside the registers:
// the size of each render target, in a NOP after its registers (AmdGpu::CbDbExtent, raw).
inline constexpr u32 RegisterCount = 0xD000;
struct DrawState {
    std::vector<u32> regs;
    std::array<u32, 8> cb_extent{};
    u32 db_extent = 0;
    // An indirect draw's arguments, read from game memory (the GPU writes them, culling on the
    // GPU): VkDrawIndirectCommand (DRAW_INDIRECT, 4 dwords) or VkDrawIndexedIndirectCommand
    // (DRAW_INDEX_INDIRECT, 5 dwords), at the SET_BASE address + the packet's data offset.
    bool indirect = false;
    std::array<u32, 5> indirect_args{};
};

// A vertex attribute, from the fetch shader the vertex shader calls and the V# it reads (one
// binding per attribute, as the GPU layer does).
struct VertexInput {
    u32 vk_format = 0;
    u64 address = 0; // game memory [address, address + size)
    u64 size = 0;
    u32 stride = 0;
    bool per_instance = false;
    u32 divisor = 1;
};

// A color render target (attachment index = CB slot): its image as the GPU layer describes it
// (VideoCore::ImageInfo / ImageViewInfo of the color buffer), its blending and write mask.
struct ColorTarget {
    bool used = false; // slots below the last used one may be empty
    ImageBinding image;
    u32 view_format = 0; // VkFormat
    u32 view_level = 0, view_layer = 0, view_layers = 1;
    std::array<u32, 4> swizzle{};
    // VkPipelineColorBlendAttachmentState (factors and ops as VkBlendFactor / VkBlendOp)
    bool blend = false;
    u32 src_color = 0, dst_color = 0, color_op = 0, src_alpha = 0, dst_alpha = 0, alpha_op = 0;
    u32 write_mask = 0xf; // VkColorComponentFlags
    // A fast clear the GPU layer has not applied yet (its CMASK says cleared): the attachment
    // starts as clear_value (VkClearColorValue, as uint32) instead of its contents.
    u64 cmask_address = 0;
    // The GPU layer tracks the target's fast clears (CB_COLOR_INFO.FAST_CLEAR): binding it
    // registers its CMASK, as cleared when new (see FastClearQuery).
    bool meta_tracked = false;
    std::array<u32, 4> clear_value{};
};

// The depth target (VideoCore::ImageInfo / ImageViewInfo of the depth buffer) and the depth and
// stencil tests (Rasterizer::UpdateDepthStencilState).
struct StencilFace {
    u32 fail_op = 0, pass_op = 0, depth_fail_op = 0, compare_op = 0; // VkStencilOp, VkCompareOp
    u32 compare_mask = 0, write_mask = 0, reference = 0;
};
struct DepthTarget {
    bool used = false;
    ImageBinding image;
    u32 view_layer = 0, view_layers = 1;
    bool stencil = false; // stencil test on (the image has a stencil plane)
    StencilFace front, back;
    bool stencil_clear = false; // DB_RENDER_CONTROL.STENCIL_CLEAR_ENABLE: starts as stencil_clear_value
    u32 stencil_clear_value = 0;
    bool test = false, write = false;
    u32 compare_op = 0; // VkCompareOp
    bool bounds = false;
    float bounds_min = 0, bounds_max = 1;
    // The draw clears the depth (DB_RENDER_CONTROL.DEPTH_CLEAR_ENABLE with depth writes), or the
    // GPU layer holds a fast clear of it (HTILE, see FastClearQuery): it starts as clear_value.
    bool clear_pass = false;
    u64 htile_address = 0;
    bool meta_tracked = false; // DB_Z_INFO.TILE_SURFACE_ENABLE (as ColorTarget::meta_tracked)
    float clear_value = 0;
};

// What a draw needs besides its shaders: fixed function state as Vulkan values (the GPU layer's
// own conversions: GraphicsPipeline, Rasterizer::UpdateDynamicState), targets and draw arguments.
struct TranslatedDraw {
    TranslatedProgram ps, vs; // descriptor bindings: the pixel shader's, then the vertex shader's
    std::vector<VertexInput> vertex_inputs;

    u32 topology = 0;      // VkPrimitiveTopology (a patch list for rect and quad lists)
    u32 patch_points = 0;  // rect lists (3) and quad lists (4): tessellation shaders turn each
    std::vector<u32> tcs_spirv, tes_spirv; // patch into a quad (the GPU layer's own)
    bool primitive_restart = false;
    u32 cull_mode = 0, front_face = 0, polygon_mode = 0;
    bool rasterizer_discard = false, depth_clamp = false, depth_clip = true;
    bool negative_one_to_one = false; // clip space z in [-w, w] (VK_EXT_depth_clip_control)
    bool provoking_last = false;
    std::array<float, 6> viewport{}; // x, y, width, height, min depth, max depth
    std::array<s32, 2> scissor_offset{};
    std::array<u32, 2> scissor_extent{};
    std::array<float, 4> push_viewport{}; // Shader::PushData xoffset, yoffset, xscale, yscale
    std::array<float, 4> blend_constants{};
    bool logic_op_enable = false;
    u32 logic_op = 0; // VkLogicOp
    std::vector<ColorTarget> colors; // attachment i = CB slot i
    DepthTarget depth;
    bool depth_bias = false;
    float bias_constant = 0, bias_clamp = 0, bias_slope = 0;
    u32 width = 0, height = 0;       // render area

    bool indexed = false;
    bool index32 = false;
    u64 index_address = 0; // the first index used
    u32 count = 0;         // indices (or vertices)
    u32 instances = 1;
    u32 vertex_offset = 0, instance_offset = 0;

    std::string skipped; // set when the draw uses something not supported yet
};

// Translates a draw: `packet` is its PM4 packet (DRAW_INDEX_2, DRAW_INDEX_OFFSET_2 or
// DRAW_INDEX_AUTO; header first). Same thread and limits as TranslateCompute.
bool TranslateDraw(const DrawState& state, const u32* packet, TranslatedDraw& out,
                   std::string& error);

// Translates a compute program for the current contents of game memory (the SPIR-V depends on
// the descriptors it reads: buffer formats, image types...). Only on the GPU command processor
// thread (the SRT walker it runs is not thread-safe). Returns false with `error` set.
bool TranslateCompute(const ComputeProgram& program, TranslatedProgram& out, std::string& error);

} // namespace render::translate
