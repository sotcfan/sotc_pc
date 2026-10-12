#include "shader/csdr.h"

#include <cstring>
#include <format>
#include <string_view>

namespace shader {

namespace {

constexpr size_t FileHeaderSize = 12;
constexpr size_t ShdrHeaderSize = 16;
constexpr u32 ShdrMagic = 0x72646853; // "Shdr"
constexpr std::string_view BinaryInfoMagic{"OrbShdr"};

u32 ReadU32(std::span<const u8> data, size_t offset) {
    u32 value;
    std::memcpy(&value, data.data() + offset, sizeof(value));
    return value;
}

std::span<const u32> Dwords(std::span<const u8> data, size_t offset, size_t count) {
    return {reinterpret_cast<const u32*>(data.data() + offset), count};
}

size_t FindShdr(std::span<const u8> file, size_t from) {
    for (size_t offset = (from + 3) & ~size_t{3}; offset + 4 <= file.size(); offset += 4) {
        if (ReadU32(file, offset) == ShdrMagic) {
            return offset;
        }
    }
    return std::string_view::npos;
}

} // namespace

std::optional<std::vector<ShaderBinary>> ParseCsdr(std::span<const u8> file, std::string& error) {
    if (reinterpret_cast<uintptr_t>(file.data()) % 4 != 0 || file.size() < FileHeaderSize) {
        error = "file too small or misaligned";
        return std::nullopt;
    }
    const u32 version = ReadU32(file, 0);
    if (version != 1 && version != 2) {
        error = std::format("unknown version {:#x} (compressed?)", version);
        return std::nullopt;
    }
    const size_t count = version == 1 ? 1 : 2;
    if (FileHeaderSize + count * ResourceOffsetsSize > file.size()) {
        error = "truncated resource offset tables";
        return std::nullopt;
    }

    std::vector<ShaderBinary> shaders;
    size_t search_from = FileHeaderSize + count * ResourceOffsetsSize;
    for (size_t i = 0; i < count; i++) {
        const size_t shdr = FindShdr(file, search_from);
        if (shdr == std::string_view::npos || shdr + ShdrHeaderSize + 8 > file.size()) {
            error = std::format("shader {}: no Shdr header", i);
            return std::nullopt;
        }
        const u8 type = file[shdr + 8];
        const size_t header_dwords = file[shdr + 9];
        const size_t header_offset = shdr + ShdrHeaderSize;
        if (header_dwords < 4 || header_offset + header_dwords * 4 > file.size()) {
            error = std::format("shader {}: bad struct size {}", i, header_dwords);
            return std::nullopt;
        }
        const size_t code_offset = header_offset + header_dwords * 4;
        const size_t code_size = ReadU32(file, header_offset) & 0x7FFFFF;
        if (code_size < BinaryInfoSize || code_size % 4 != 0 ||
            code_offset + code_size > file.size()) {
            error = std::format("shader {}: bad code size {:#x}", i, code_size);
            return std::nullopt;
        }
        const size_t info = code_offset + code_size - BinaryInfoSize;
        if (std::memcmp(file.data() + info, BinaryInfoMagic.data(), BinaryInfoMagic.size()) != 0) {
            error = std::format("shader {}: no binary info at the end of the code", i);
            return std::nullopt;
        }
        // Binary info: u32 at +8 holds the code length in bits 8-31. Between the code and the
        // binary info lie the input usage slots and other metadata.
        const size_t code_length = ReadU32(file, info + 8) >> 8;
        if (code_length == 0 || code_length % 4 != 0 || code_offset + code_length > info) {
            error = std::format("shader {}: bad code length {:#x}", i, code_length);
            return std::nullopt;
        }
        u64 hash;
        std::memcpy(&hash, file.data() + info + 16, sizeof(hash));
        const std::span<const u32> header = Dwords(file, header_offset, header_dwords);
        shaders.push_back(ShaderBinary{
            .type = static_cast<GnmxType>(type),
            .resource_offsets = file.subspan(FileHeaderSize + i * ResourceOffsetsSize,
                                             ResourceOffsetsSize),
            .header = header,
            .registers = header.subspan(2),
            .code = Dwords(file, code_offset, code_length / 4),
            .hash = hash,
            .crc32 = ReadU32(file, info + 24),
        });
        search_from = code_offset + code_size;
    }
    return shaders;
}

const char* GnmxTypeName(GnmxType type) {
    switch (type) {
    case GnmxType::Vertex: return "vs";
    case GnmxType::Pixel: return "ps";
    case GnmxType::Geometry: return "gs";
    case GnmxType::Compute: return "cs";
    case GnmxType::Export: return "es";
    case GnmxType::Hull: return "hs";
    case GnmxType::Local: return "ls";
    }
    return "?";
}

} // namespace shader
