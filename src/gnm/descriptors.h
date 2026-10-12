#pragma once

#include <span>
#include <string>

#include "common/types.h"

// The GPU resource descriptors the game builds with Sony's Gnm code (GCN 1.1 "Sea Islands"
// layout): textures (T#, 8 dwords), buffers (V#, 4 dwords) and color render targets (the
// CB_COLOR* registers Gnm keeps in its RenderTarget object). Decoded field by field from the
// dwords, without depending on the GPU layer, for the diagnostic tools now and the native
// renderer later.

namespace gnm {

// T# "type" field.
enum class ImageType : u8 {
    Invalid = 0,
    Color1D = 8,
    Color2D = 9,
    Color3D = 10,
    Cube = 11,
    Color1DArray = 12,
    Color2DArray = 13,
    Color2DMsaa = 14,
    Color2DMsaaArray = 15,
};

struct TextureInfo {
    u64 address;      // bytes
    u32 width, height, depth;
    u32 data_format;  // IMG_DATA_FORMAT (also the CB color format numbering)
    u32 num_format;   // IMG_NUM_FORMAT
    ImageType type;
    u32 base_level, last_level;
    u32 base_array, last_array;
    u32 tiling_index;
    u32 swizzle;      // dst_sel x, y, z, w: 3 bits each
    u32 pitch;

    bool Valid() const {
        return (static_cast<u8>(type) & 0x8) != 0;
    }
};

struct BufferInfo {
    u64 address;
    u32 stride;
    u32 num_records;
    u32 data_format;  // BUF_DATA_FORMAT (same numbering as the image one, 4 bits)
    u32 num_format;   // BUF_NUM_FORMAT (3 bits)
    u32 swizzle;
    u32 type;         // 0 for a buffer (a T# has a non-zero type in the same bits)
};

// The first registers of Gnm's RenderTarget object: CB_COLORn_BASE, PITCH, SLICE, VIEW, INFO,
// ATTRIB, ...
struct RenderTargetInfo {
    u64 address;
    u32 pitch;        // pixels
    u32 height;       // pixels (slice size / pitch)
    u32 format;       // CB_COLOR_INFO.FORMAT (data format numbering)
    u32 number_type;  // CB_COLOR_INFO.NUMBER_TYPE
    u32 tile_mode;    // CB_COLOR_ATTRIB.TILE_MODE_INDEX
};

TextureInfo DecodeTexture(std::span<const u32, 8> dwords);
BufferInfo DecodeBuffer(std::span<const u32, 4> dwords);
RenderTargetInfo DecodeRenderTarget(std::span<const u32, 6> registers);

const char* DataFormatName(u32 format);
const char* NumberFormatName(u32 format);    // textures and buffers
const char* ColorNumberTypeName(u32 type);  // render targets
const char* ImageTypeName(ImageType type);

// One-line summaries for logs and reports.
std::string Describe(const TextureInfo& texture);
std::string Describe(const BufferInfo& buffer);
std::string Describe(const RenderTargetInfo& target);

} // namespace gnm
