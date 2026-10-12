#include "loader/self.h"

#include <algorithm>
#include <cstring>
#include <format>
#include <fstream>
#include <stdexcept>

namespace loader {

namespace {

constexpr u32 SelfMagic = 0x1D3D154F;

// Flags of a SELF entry.
constexpr u64 EntryEncrypted = 0x2;
constexpr u64 EntryCompressed = 0x8;
constexpr u64 EntryHasSegment = 0x800; // bits 20..31 hold the program header index

struct SelfEntry {
    u64 properties;
    u64 offset;
    u64 file_size;
    u64 memory_size;
};

template <typename T>
T ReadAt(const std::vector<u8>& data, u64 offset) {
    if (offset + sizeof(T) > data.size()) {
        throw std::runtime_error("truncated file");
    }
    T value;
    std::memcpy(&value, data.data() + offset, sizeof(T));
    return value;
}

std::vector<u8> ReadWholeFile(const std::filesystem::path& path) {
    std::ifstream file{path, std::ios::binary};
    if (!file) {
        throw std::runtime_error(std::format("could not open {}", path.string()));
    }
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

void ParseHeaders(ElfFile& elf, const std::vector<u8>& bytes, u64 header_offset) {
    elf.header = ReadAt<Elf64_Ehdr>(bytes, header_offset);
    if (std::memcmp(elf.header.e_ident, ELFMAG, SELFMAG) != 0) {
        throw std::runtime_error("ELF header not found");
    }
    if (elf.header.e_ident[EI_CLASS] != ELFCLASS64 || elf.header.e_machine != EM_X86_64) {
        throw std::runtime_error("not an x86-64 ELF");
    }
    elf.phdrs.resize(elf.header.e_phnum);
    for (u16 i = 0; i < elf.header.e_phnum; i++) {
        elf.phdrs[i] = ReadAt<Elf64_Phdr>(
            bytes, header_offset + elf.header.e_phoff + u64{i} * elf.header.e_phentsize);
    }
}

// Rebuilds the ELF stored in a SELF: the ELF header and program headers follow the entry
// table, and each entry flagged with a segment index holds that segment's file bytes.
void UnwrapSelf(ElfFile& elf, const std::vector<u8>& self) {
    const u64 self_size = ReadAt<u64>(self, 0x10);
    const u16 entry_count = ReadAt<u16>(self, 0x18);
    const u64 header_offset = 0x20 + u64{entry_count} * sizeof(SelfEntry);
    ParseHeaders(elf, self, header_offset);

    const u64 headers_end = elf.header.e_phoff + u64{elf.header.e_phnum} * elf.header.e_phentsize;
    u64 image_size = headers_end;
    for (const auto& ph : elf.phdrs) {
        image_size = std::max(image_size, ph.p_offset + ph.p_filesz);
    }
    elf.data.assign(image_size, 0);
    std::memcpy(elf.data.data(), self.data() + header_offset, headers_end);

    for (u16 i = 0; i < entry_count; i++) {
        const auto entry = ReadAt<SelfEntry>(self, 0x20 + u64{i} * sizeof(SelfEntry));
        if (entry.properties & EntryEncrypted) {
            throw std::runtime_error(
                "eboot.bin is encrypted; a decrypted dump of the game is required");
        }
        if (entry.properties & EntryCompressed) {
            throw std::runtime_error("compressed segments are not supported yet");
        }
        if (!(entry.properties & EntryHasSegment)) {
            continue;
        }
        const u64 index = (entry.properties >> 20) & 0xFFF;
        if (index >= elf.phdrs.size() || entry.offset + entry.file_size > self.size() ||
            elf.phdrs[index].p_offset + entry.file_size > elf.data.size()) {
            throw std::runtime_error("SELF entry out of bounds");
        }
        std::memcpy(elf.data.data() + elf.phdrs[index].p_offset, self.data() + entry.offset,
                    entry.file_size);
    }

    // The version segment is stored after the SELF body instead of in an entry.
    for (const auto& ph : elf.phdrs) {
        if (ph.p_type == PT_SCE_VERSION && ph.p_filesz != 0 &&
            self_size + ph.p_filesz <= self.size()) {
            std::memcpy(elf.data.data() + ph.p_offset, self.data() + self_size, ph.p_filesz);
        }
    }
}

} // namespace

ElfFile ElfFile::Load(const std::filesystem::path& path) {
    std::vector<u8> bytes = ReadWholeFile(path);
    ElfFile elf;
    if (bytes.size() >= 4 && std::memcmp(bytes.data(), ELFMAG, SELFMAG) == 0) {
        ParseHeaders(elf, bytes, 0);
        elf.data = std::move(bytes);
    } else if (ReadAt<u32>(bytes, 0) == SelfMagic) {
        UnwrapSelf(elf, bytes);
    } else {
        throw std::runtime_error(std::format("{} is neither an ELF nor a PS4 SELF", path.string()));
    }
    return elf;
}

const Elf64_Phdr* ElfFile::FindSegment(u32 type) const {
    for (const auto& ph : phdrs) {
        if (ph.p_type == type) {
            return &ph;
        }
    }
    return nullptr;
}

} // namespace loader
