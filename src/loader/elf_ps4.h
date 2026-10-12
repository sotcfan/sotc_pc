#pragma once

// PS4-specific values used in the game's ELF file. The generic ELF structures (Elf64_Ehdr,
// Elf64_Phdr, Elf64_Dyn, Elf64_Rela, Elf64_Sym) come from <elf.h>, or from loader/elf64.h where
// the system has none (Windows).

#ifdef _WIN32
#include "loader/elf64.h"
#else
#include <elf.h>
#endif

#include "common/types.h"

namespace loader {

// e_type
constexpr u16 ET_SCE_EXEC = 0xFE00;
constexpr u16 ET_SCE_DYNEXEC = 0xFE10; // the game's eboot.bin
constexpr u16 ET_SCE_DYNAMIC = 0xFE18; // .prx/.sprx modules

// p_type
constexpr u32 PT_SCE_DYNLIBDATA = 0x61000000; // symbol, string and relocation tables
constexpr u32 PT_SCE_PROCPARAM = 0x61000001;  // process parameters (see ProcParam)
constexpr u32 PT_SCE_MODULE_PARAM = 0x61000002;
constexpr u32 PT_SCE_RELRO = 0x61000010; // writable during relocation, read-only after
constexpr u32 PT_SCE_COMMENT = 0x6FFFFF00;
constexpr u32 PT_SCE_VERSION = 0x6FFFFF01;

// d_tag. Values marked "offset" are offsets into the PT_SCE_DYNLIBDATA blob.
constexpr s64 DT_SCE_FINGERPRINT = 0x61000007;
constexpr s64 DT_SCE_ORIGINAL_FILENAME = 0x61000009;
constexpr s64 DT_SCE_MODULE_INFO = 0x6100000D;
constexpr s64 DT_SCE_NEEDED_MODULE = 0x6100000F;
constexpr s64 DT_SCE_MODULE_ATTR = 0x61000011;
constexpr s64 DT_SCE_EXPORT_LIB = 0x61000013;
constexpr s64 DT_SCE_IMPORT_LIB = 0x61000015;
constexpr s64 DT_SCE_EXPORT_LIB_ATTR = 0x61000017;
constexpr s64 DT_SCE_IMPORT_LIB_ATTR = 0x61000019;
constexpr s64 DT_SCE_HASH = 0x61000025;
constexpr s64 DT_SCE_PLTGOT = 0x61000027;
constexpr s64 DT_SCE_JMPREL = 0x61000029; // offset
constexpr s64 DT_SCE_PLTREL = 0x6100002B;
constexpr s64 DT_SCE_PLTRELSZ = 0x6100002D;
constexpr s64 DT_SCE_RELA = 0x6100002F; // offset
constexpr s64 DT_SCE_RELASZ = 0x61000031;
constexpr s64 DT_SCE_RELAENT = 0x61000033;
constexpr s64 DT_SCE_STRTAB = 0x61000035; // offset
constexpr s64 DT_SCE_STRSZ = 0x61000037;
constexpr s64 DT_SCE_SYMTAB = 0x61000039; // offset
constexpr s64 DT_SCE_SYMENT = 0x6100003B;
constexpr s64 DT_SCE_HASHSZ = 0x6100003D;
constexpr s64 DT_SCE_SYMTABSZ = 0x6100003F;

// Contents of the PT_SCE_PROCPARAM segment.
struct ProcParam {
    u64 size;
    u32 magic; // "ORBI"
    u32 entry_count;
    u64 sdk_version;
    const char* process_name;
    const char* main_thread_name;
    const u32* main_thread_priority;
    const u32* main_thread_stack_size;
    const void* libc_param;
    const void* memory_param;
    const void* fs_param;
    const u32* process_preload_enabled;
};

} // namespace loader
