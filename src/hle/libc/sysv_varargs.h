#pragma once

#include <cstddef>
#include <string>

#include "common/types.h"

// Variable arguments as the game passes them: the System V x86-64 calling convention, on any
// host. On Linux the host's own va_list has this layout, so the host's vsnprintf can read it; on
// Windows the host convention differs (and `long` has 32 bits there, 64 on the PS4), so the
// printf/scanf family formats and scans from this structure itself.
//
// Variadic functions the game calls (printf, snprintf, ...) enter through
// SOTC_SYSV_VARIADIC_ENTRY: a small assembly stub stores the argument registers the way a System V
// va_start would and calls the function's v* implementation with a SysvVaList*.

namespace hle::libc {

// The System V va_list (24 bytes). `va_list` itself is an array of one, so functions receive a
// pointer to it.
struct SysvVaList {
    u32 gp_offset;          // +0: offset in reg_save_area of the next integer register (to 48)
    u32 fp_offset;          // +4: offset of the next vector register (48 + 16 * n, to 176)
    u8* overflow_arg_area;  // +8: next argument passed on the stack
    u8* reg_save_area;      // +16: rdi, rsi, rdx, rcx, r8, r9 (8 bytes each), xmm0-7 (16 each)
};
static_assert(sizeof(SysvVaList) == 24);

// The next argument of each class (the list is advanced).
u64 NextInteger(SysvVaList& args);
double NextDouble(SysvVaList& args);
long double NextLongDouble(SysvVaList& args);

// printf formatting from a copy of `args` (the PS4's v*printf leave the caller's list untouched;
// the game formats the same list twice, first to measure). Returns the full length; at most
// size - 1 characters and a terminator are written.
int FormatToBuffer(char* buffer, std::size_t size, const char* format, const SysvVaList* args);
// The whole text.
std::string Format(const char* format, const SysvVaList* args);

// sscanf from a copy of `args`.
int Scan(const char* text, const char* format, const SysvVaList* args);

} // namespace hle::libc

// Defines `entry`, a variadic System V function of the game, as a stub that builds the
// SysvVaList (named arguments: `named` integer/pointer registers, no vector ones) and jumps to
// `impl`, which takes the same named arguments followed by the SysvVaList* in register `va_reg`
// (the next integer argument register: rsi after 1 named argument, rdx after 2, rcx after 3).
// `impl` must be an extern "C" PS4_SYSV_ABI function.
//
// Stack frame (rsp 16-aligned after `sub`): [rsp, +48) rdi..r9, [rsp+48, +176) xmm0..7 (AL, the
// number of vector registers used, is ignored: all eight are stored), [rsp+176, +24) the list.
#define SOTC_SYSV_VARIADIC_ENTRY(entry, impl, named, va_reg)                                       \
    extern "C" __attribute__((naked)) PS4_SYSV_ABI int entry(...) {                                \
        asm(".intel_syntax noprefix\n"                                                             \
            "push rbp\n"                                                                           \
            "mov rbp, rsp\n"                                                                       \
            "sub rsp, 208\n"                                                                       \
            "mov [rsp + 0], rdi\n"                                                                 \
            "mov [rsp + 8], rsi\n"                                                                 \
            "mov [rsp + 16], rdx\n"                                                                \
            "mov [rsp + 24], rcx\n"                                                                \
            "mov [rsp + 32], r8\n"                                                                 \
            "mov [rsp + 40], r9\n"                                                                 \
            "movaps [rsp + 48], xmm0\n"                                                            \
            "movaps [rsp + 64], xmm1\n"                                                            \
            "movaps [rsp + 80], xmm2\n"                                                            \
            "movaps [rsp + 96], xmm3\n"                                                            \
            "movaps [rsp + 112], xmm4\n"                                                           \
            "movaps [rsp + 128], xmm5\n"                                                           \
            "movaps [rsp + 144], xmm6\n"                                                           \
            "movaps [rsp + 160], xmm7\n"                                                           \
            "mov dword ptr [rsp + 176], " #named " * 8\n"                                          \
            "mov dword ptr [rsp + 180], 48\n"                                                      \
            "lea rax, [rbp + 16]\n"                                                                \
            "mov [rsp + 184], rax\n"                                                               \
            "mov [rsp + 192], rsp\n"                                                               \
            "lea " #va_reg ", [rsp + 176]\n"                                                       \
            "call " #impl "\n"                                                                     \
            "leave\n"                                                                              \
            "ret\n"                                                                                \
            ".att_syntax prefix\n");                                                               \
    }
