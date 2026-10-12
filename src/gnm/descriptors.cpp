// Decoding of the GCN resource descriptors. The bit layouts below come from AMD's Sea Islands
// documentation and match shadPS4's AmdGpu::Image / AmdGpu::Buffer (video_core/amdgpu/
// resource.h), which were checked against real games; field positions are given as bit numbers
// counted over the whole descriptor (bit 32 = bit 0 of dword 1, and so on).

#include "gnm/descriptors.h"

#include <format>

namespace gnm {

namespace {

// Bits [first, first + count) of a little-endian dword array. Slow (bit by bit) but simple and
// independent of the compiler's bitfield layout; descriptors are decoded only for diagnostics
// and once per bind, never per pixel.
u64 Bits(std::span<const u32> dwords, u32 first, u32 count) {
    u64 value = 0;
    for (u32 i = 0; i < count; i++) {
        const u32 bit = first + i;
        value |= u64{(dwords[bit / 32] >> (bit % 32)) & 1} << i;
    }
    return value;
}

} // namespace

// T#, 8 dwords:
//   bits   0-37  base address >> 8 (256-byte aligned)
//         38-39  mtype_l2           40-51  min_lod (4.8 fixed point)
//         52-57  data format        58-61  number format         62-63  mtype
//         64-77  width - 1          78-91  height - 1            92-94  perf_mod, 95 interlaced
//         96-107 dst_sel x, y, z, w (3 bits each: 0 zero, 1 one, 4 red, 5 green, 6 blue, 7 alpha)
//        108-111 base mip level    112-115 last mip level
//        116-120 tiling index (into the GPU's tile mode table)
//        121 pow2pad, 122 mtype2, 123 atc, 124-127 type (ImageType; 0 = not a texture)
//        128-140 depth - 1 (3D) or last slice (arrays)   141-154 pitch - 1
//        160-172 base array slice  173-185 last array slice
//   (dwords 6-7: LOD warning, counter bank, Neo-only compression bits; not needed here)
TextureInfo DecodeTexture(std::span<const u32, 8> t) {
    TextureInfo info{};
    info.address = Bits(t, 0, 38) << 8;
    info.data_format = static_cast<u32>(Bits(t, 52, 6));
    info.num_format = static_cast<u32>(Bits(t, 58, 4));
    info.width = static_cast<u32>(Bits(t, 64, 14)) + 1;
    info.height = static_cast<u32>(Bits(t, 78, 14)) + 1;
    info.swizzle = static_cast<u32>(Bits(t, 96, 12));
    info.base_level = static_cast<u32>(Bits(t, 108, 4));
    info.last_level = static_cast<u32>(Bits(t, 112, 4));
    info.tiling_index = static_cast<u32>(Bits(t, 116, 5));
    info.type = static_cast<ImageType>(Bits(t, 124, 4));
    info.depth = static_cast<u32>(Bits(t, 128, 13)) + 1;
    info.pitch = static_cast<u32>(Bits(t, 141, 14)) + 1;
    info.base_array = static_cast<u32>(Bits(t, 160, 13));
    info.last_array = static_cast<u32>(Bits(t, 173, 13));
    return info;
}

// V#, 4 dwords:
//   bits   0-39  base address (bytes)         48-61  stride in bytes (0 = raw buffer)
//         62 cache swizzle, 63 swizzle enable
//         64-95  number of records (elements if stride != 0, else bytes)
//         96-107 dst_sel x, y, z, w           108-110 number format   111-114 data format
//        115-116 element size, 117-118 index stride, 119 add thread id
//        123-125 mtype, 126-127 type (0 for a buffer; a T# has its type in the same bits)
BufferInfo DecodeBuffer(std::span<const u32, 4> v) {
    BufferInfo info{};
    info.address = Bits(v, 0, 40);
    info.stride = static_cast<u32>(Bits(v, 48, 14));
    info.num_records = v[2];
    info.swizzle = static_cast<u32>(Bits(v, 96, 12));
    info.num_format = static_cast<u32>(Bits(v, 108, 3));
    info.data_format = static_cast<u32>(Bits(v, 111, 4));
    info.type = static_cast<u32>(Bits(v, 126, 2));
    return info;
}

// Gnm's RenderTarget object starts with the CB_COLORn registers in hardware order (the game's
// setRenderTarget, 0xc5cbc0, copies 14 of them into a SET_CONTEXT_REG packet):
//   [0] CB_COLOR_BASE   address >> 8
//   [1] CB_COLOR_PITCH  bits 0-10: pitch / 8 - 1 (in tiles of 8 pixels)
//   [2] CB_COLOR_SLICE  bits 0-21: slice size / 64 - 1 (in tiles of 8x8 pixels)
//   [3] CB_COLOR_VIEW   first/last array slice
//   [4] CB_COLOR_INFO   bits 2-6 format, 8-10 number type, 11-12 component swap, ...
//   [5] CB_COLOR_ATTRIB bits 0-4 tile mode index, ...
// The height is not stored; slice / pitch gives it rounded up to the tiling (1080 -> 1088).
RenderTargetInfo DecodeRenderTarget(std::span<const u32, 6> r) {
    RenderTargetInfo info{};
    info.address = u64{r[0]} << 8;
    info.pitch = ((r[1] & 0x7FF) + 1) * 8;
    const u64 slice = (u64{r[2] & 0x3FFFFF} + 1) * 64;
    info.height = static_cast<u32>(slice / info.pitch);
    info.format = (r[4] >> 2) & 0x1F;
    info.number_type = (r[4] >> 8) & 0x7;
    info.tile_mode = r[5] & 0x1F;
    return info;
}

// IMG_DATA_FORMAT / BUF_DATA_FORMAT / CB COLOR_* (the three share the numbering up to 22).
// The name lists the bits per component, first component in the lowest bits.
const char* DataFormatName(u32 format) {
    switch (format) {
    case 0: return "Invalid";
    case 1: return "8";
    case 2: return "16";
    case 3: return "8_8";
    case 4: return "32";
    case 5: return "16_16";
    case 6: return "10_11_11";
    case 7: return "11_11_10";
    case 8: return "10_10_10_2";
    case 9: return "2_10_10_10";
    case 10: return "8_8_8_8";
    case 11: return "32_32";
    case 12: return "16_16_16_16";
    case 13: return "32_32_32";
    case 14: return "32_32_32_32";
    case 16: return "5_6_5";
    case 17: return "1_5_5_5";
    case 18: return "5_5_5_1";
    case 19: return "4_4_4_4";
    case 20: return "8_24";
    case 21: return "24_8";
    case 22: return "X24_8_32";
    case 32: return "GB_GR";
    case 33: return "BG_RG";
    case 34: return "5_9_9_9";
    case 35: return "BC1";
    case 36: return "BC2";
    case 37: return "BC3";
    case 38: return "BC4";
    case 39: return "BC5";
    case 40: return "BC6";
    case 41: return "BC7";
    case 57: return "4_4";
    case 58: return "6_5_5";
    case 59: return "1";
    case 60: return "1_Reversed";
    case 61: return "32_As_8";
    case 62: return "32_As_8_8";
    case 63: return "32_As_32_32_32_32";
    default: return (format >= 44 && format <= 56) ? "Fmask" : "?";
    }
}

const char* NumberFormatName(u32 format) {
    switch (format) {
    case 0: return "Unorm";
    case 1: return "Snorm";
    case 2: return "Uscaled";
    case 3: return "Sscaled";
    case 4: return "Uint";
    case 5: return "Sint";
    case 6: return "SnormNz";
    case 7: return "Float";
    case 9: return "Srgb";
    case 10: return "Ubnorm";
    case 11: return "UbnormNz";
    case 12: return "Ubint";
    case 13: return "Ubscaled";
    default: return "?";
    }
}

// CB_COLOR_INFO.NUMBER_TYPE: like the texture number format, but sRGB is 6 instead of 9.
const char* ColorNumberTypeName(u32 type) {
    switch (type) {
    case 0: return "Unorm";
    case 1: return "Snorm";
    case 2: return "Uscaled";
    case 3: return "Sscaled";
    case 4: return "Uint";
    case 5: return "Sint";
    case 6: return "Srgb";
    case 7: return "Float";
    default: return "?";
    }
}

const char* ImageTypeName(ImageType type) {
    switch (type) {
    case ImageType::Invalid: return "Invalid";
    case ImageType::Color1D: return "1D";
    case ImageType::Color2D: return "2D";
    case ImageType::Color3D: return "3D";
    case ImageType::Cube: return "Cube";
    case ImageType::Color1DArray: return "1DArray";
    case ImageType::Color2DArray: return "2DArray";
    case ImageType::Color2DMsaa: return "2DMsaa";
    case ImageType::Color2DMsaaArray: return "2DMsaaArray";
    }
    return "?";
}

std::string Describe(const TextureInfo& t) {
    if (!t.Valid()) {
        return "null";
    }
    std::string text = std::format("{:#x} {} {}x{}", t.address, ImageTypeName(t.type), t.width,
                                   t.height);
    if (t.type == ImageType::Color3D) {
        text += std::format("x{}", t.depth);
    }
    if (t.last_array > t.base_array) {
        text += std::format(" layers {}-{}", t.base_array, t.last_array);
    }
    text += std::format(" {}/{} mips {}-{} tile {}", DataFormatName(t.data_format),
                        NumberFormatName(t.num_format), t.base_level, t.last_level,
                        t.tiling_index);
    return text;
}

std::string Describe(const BufferInfo& b) {
    return std::format("{:#x} stride {} records {} {}/{}", b.address, b.stride, b.num_records,
                       DataFormatName(b.data_format), NumberFormatName(b.num_format));
}

std::string Describe(const RenderTargetInfo& r) {
    return std::format("{:#x} {}x{} {}/{} tile {}", r.address, r.pitch, r.height,
                       DataFormatName(r.format), ColorNumberTypeName(r.number_type), r.tile_mode);
}

} // namespace gnm
