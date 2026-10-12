#pragma once

// The game's shader files (.csdr, Bluepoint's format around Sony's Gnmx shader binaries).
//
// Layout, all little-endian:
//   u32 version            1: one shader, 2: a vertex + pixel shader pair (materials)
//   u32 stage              version 1: 0 compute, 1 pixel, 2 vertex...; version 2: 2
//   u32 first_block_size
//   InputResourceOffsets   228 bytes per shader, one after the other: where each resource slot
//                          goes (a user data SGPR or a dword of the stage's resource table)
//   (Bluepoint metadata)
//   per shader:
//     "Shdr" header        16 bytes: magic, u16 major (7), u16 minor (2), u8 type,
//                          u8 struct size in dwords, ...
//     Gnmx shader struct   u32 size/flags (bits 0-22: code size in bytes), u32, then the stage
//                          registers as passed to sceGnmSet*Shader, then stage extras
//     code block           the size above: the GCN code (as long as the binary info says),
//                          input usage slots and other metadata, and in its last 28 bytes the
//                          "OrbShdr" binary info (shadPS4's AmdGpu::BinaryInfo: code length,
//                          shader hash)
//
// Most of the game's shaders are Oodle-compressed inside the PSARC archives; the decompressed
// files come from SOTC_DUMP_RESOURCES (diag/resource_dump.h).

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "common/types.h"

namespace shader {

// The "type" byte of the Shdr header (Gnmx shader type).
enum class GnmxType : u8 {
    Vertex = 1,
    Pixel = 2,
    Geometry = 3,
    Compute = 4,
    Export = 5,
    Local = 6, // not seen in the game's files
    Hull = 7,  // tesselationshadertest/hs_main
};

constexpr size_t ResourceOffsetsSize = 228;
constexpr size_t BinaryInfoSize = 28;

struct ShaderBinary {
    GnmxType type;
    // InputResourceOffsets of this shader (see renderizador/CATALOGO_GNM.md in the analysis
    // notes for its fields).
    std::span<const u8> resource_offsets;
    // The Gnmx shader struct, whole (common data, stage registers, stage extras).
    std::span<const u32> header;
    // The stage registers: header from dword 2 on (PGM_LO, PGM_HI, RSRC1, RSRC2, ...).
    std::span<const u32> registers;
    // The GCN code alone (what shadPS4 translates).
    std::span<const u32> code;
    u64 hash; // binary info shader_hash: the key shadPS4 uses for the shader
    u32 crc32;
};

// Splits a .csdr file into its shaders. The spans point into `file`. Returns nullopt and sets
// `error` if the file does not follow the layout above.
std::optional<std::vector<ShaderBinary>> ParseCsdr(std::span<const u8> file, std::string& error);

const char* GnmxTypeName(GnmxType type);

} // namespace shader
