#pragma once

// The generic ELF64 structures and constants the loader uses, for systems without <elf.h>
// (Windows). Layouts and values follow the System V ABI (and glibc's <elf.h>, which Linux
// builds keep using; see elf_ps4.h).

#include <cstdint>

using Elf64_Half = std::uint16_t;
using Elf64_Word = std::uint32_t;
using Elf64_Sword = std::int32_t;
using Elf64_Xword = std::uint64_t;
using Elf64_Sxword = std::int64_t;
using Elf64_Addr = std::uint64_t;
using Elf64_Off = std::uint64_t;
using Elf64_Section = std::uint16_t;

constexpr int EI_NIDENT = 16;

struct Elf64_Ehdr { // 64 bytes
    unsigned char e_ident[EI_NIDENT];
    Elf64_Half e_type;
    Elf64_Half e_machine;
    Elf64_Word e_version;
    Elf64_Addr e_entry;
    Elf64_Off e_phoff;
    Elf64_Off e_shoff;
    Elf64_Word e_flags;
    Elf64_Half e_ehsize;
    Elf64_Half e_phentsize;
    Elf64_Half e_phnum;
    Elf64_Half e_shentsize;
    Elf64_Half e_shnum;
    Elf64_Half e_shstrndx;
};

struct Elf64_Phdr { // 56 bytes
    Elf64_Word p_type;
    Elf64_Word p_flags;
    Elf64_Off p_offset;
    Elf64_Addr p_vaddr;
    Elf64_Addr p_paddr;
    Elf64_Xword p_filesz;
    Elf64_Xword p_memsz;
    Elf64_Xword p_align;
};

struct Elf64_Dyn { // 16 bytes
    Elf64_Sxword d_tag;
    union {
        Elf64_Xword d_val;
        Elf64_Addr d_ptr;
    } d_un;
};

struct Elf64_Rela { // 24 bytes
    Elf64_Addr r_offset;
    Elf64_Xword r_info;
    Elf64_Sxword r_addend;
};

struct Elf64_Sym { // 24 bytes
    Elf64_Word st_name;
    unsigned char st_info;
    unsigned char st_other;
    Elf64_Section st_shndx;
    Elf64_Addr st_value;
    Elf64_Xword st_size;
};

#define ELFMAG "\177ELF"
constexpr int SELFMAG = 4;
constexpr int EI_CLASS = 4;
constexpr unsigned char ELFCLASS64 = 2;
constexpr Elf64_Half EM_X86_64 = 62;

constexpr Elf64_Word PT_LOAD = 1;
constexpr Elf64_Word PT_DYNAMIC = 2;
constexpr Elf64_Word PT_TLS = 7;

constexpr Elf64_Word PF_X = 1;
constexpr Elf64_Word PF_W = 2;
constexpr Elf64_Word PF_R = 4;

constexpr Elf64_Sxword DT_NULL = 0;

constexpr Elf64_Word R_X86_64_64 = 1;
constexpr Elf64_Word R_X86_64_GLOB_DAT = 6;
constexpr Elf64_Word R_X86_64_JUMP_SLOT = 7;
constexpr Elf64_Word R_X86_64_RELATIVE = 8;

constexpr unsigned char STB_LOCAL = 0;
constexpr unsigned char STT_OBJECT = 1;

#define ELF64_R_SYM(info) ((info) >> 32)
#define ELF64_R_TYPE(info) ((info) & 0xffffffff)
#define ELF64_ST_BIND(info) ((info) >> 4)
#define ELF64_ST_TYPE(info) ((info) & 0xf)
