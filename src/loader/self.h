#pragma once

#include <filesystem>
#include <vector>

#include "loader/elf_ps4.h"

namespace loader {

// An ELF image read from disk. PS4 executables ship inside a SELF container; for a decrypted
// (f)SELF this rebuilds the plain ELF, with every segment at its p_offset, so the rest of the
// loader never has to know about the container.
struct ElfFile {
    std::vector<u8> data;
    Elf64_Ehdr header{};
    std::vector<Elf64_Phdr> phdrs;

    // Throws std::runtime_error with a message for the user when the file cannot be used.
    static ElfFile Load(const std::filesystem::path& path);

    const Elf64_Phdr* FindSegment(u32 type) const;
};

} // namespace loader
