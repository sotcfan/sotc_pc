#include "loader/tls.h"

#ifdef _WIN32
#include <windows.h>
#endif

#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "common/log.h"

namespace loader::tls {

namespace {

#ifdef _WIN32
// Windows: the game's thread pointer lives in one of the TEB's 64 TLS slots (TlsAlloc), which
// the TEB keeps at gs:[0x1480 + 8 * slot]. The rewritten instruction reads it through gs (as
// the Windows fork of bbport does, Supermedo, src/runtime_thread.c).
DWORD g_tls_slot = TLS_OUT_OF_INDEXES;

s32 ThreadPointerOffset() {
    if (g_tls_slot == TLS_OUT_OF_INDEXES) {
        g_tls_slot = TlsAlloc();
        if (g_tls_slot >= 64) {
            throw std::runtime_error("no TLS slot below 64 for the game's thread pointer");
        }
    }
    return static_cast<s32>(0x1480 + 8 * g_tls_slot);
}
#else
// The game's thread pointer for the current thread. local-exec places it at a fixed offset from
// the host thread pointer, the same for every thread, which is what the patched code reads.
thread_local u64 t_game_thread_pointer __attribute__((tls_model("local-exec"))) = 0;

s32 ThreadPointerOffset() {
    u64 host_thread_pointer;
    asm("mov %%fs:0, %0" : "=r"(host_thread_pointer));
    const s64 offset =
        static_cast<s64>(reinterpret_cast<u64>(&t_game_thread_pointer) - host_thread_pointer);
    if (offset < INT32_MIN || offset > INT32_MAX) {
        throw std::runtime_error("TLS variable out of reach of fs:[disp32]");
    }
    return static_cast<s32>(offset);
}
#endif
thread_local u8* t_block = nullptr;

Module::TlsInfo g_template;

// Space above the thread pointer; the game only reads the self pointer at offset 0.
constexpr u64 ControlBlockSize = 0x40;

constexpr u64 AlignUp(u64 value, u64 align) {
    return (value + align - 1) / align * align;
}

void SetGameThreadPointer(u64 value) {
#ifdef _WIN32
    TlsSetValue(g_tls_slot, reinterpret_cast<void*>(value));
#else
    t_game_thread_pointer = value;
#endif
}

} // namespace

size_t PatchThreadPointerReads(u8* code, u64 size) {
    const s32 offset = ThreadPointerOffset();
    size_t count = 0;
    for (u64 i = 0; i + 9 <= size; i++) {
        // 64 | REX.W (48 or 4C) | 8B | ModRM 00rrr100 | SIB 25 | disp32 = 0  ->  mov reg, fs:[0]
        if (code[i] != 0x64 || (code[i + 1] != 0x48 && code[i + 1] != 0x4C) ||
            code[i + 2] != 0x8B || (code[i + 3] & 0xC7) != 0x04 || code[i + 4] != 0x25) {
            continue;
        }
        u32 displacement;
        std::memcpy(&displacement, code + i + 5, 4);
        if (displacement != 0) {
            continue;
        }
        std::memcpy(code + i + 5, &offset, 4);
#ifdef _WIN32
        code[i] = 0x65; // fs -> gs segment prefix
#endif
        count++;
        i += 8;
    }
    return count;
}

void SetTemplate(const Module::TlsInfo& info) {
    g_template = info;
}

void InitializeThread() {
    // The TLS image sits right below the thread pointer, rounded up to its alignment.
    const u64 tls_size = AlignUp(g_template.size, std::max<u64>(g_template.align, 16));
#ifdef _WIN32
    auto* block = static_cast<u8*>(_aligned_malloc(AlignUp(tls_size + ControlBlockSize, 64), 64));
#else
    auto* block = static_cast<u8*>(std::aligned_alloc(64, AlignUp(tls_size + ControlBlockSize, 64)));
#endif
    std::memset(block, 0, tls_size + ControlBlockSize);
    if (g_template.init_size != 0) {
        std::memcpy(block, reinterpret_cast<const void*>(g_template.image), g_template.init_size);
    }
    const u64 thread_pointer = reinterpret_cast<u64>(block) + tls_size;
    std::memcpy(reinterpret_cast<void*>(thread_pointer), &thread_pointer, sizeof(thread_pointer));
    SetGameThreadPointer(thread_pointer);
    t_block = block;
}

u64 ThreadPointer() {
#ifdef _WIN32
    return g_tls_slot == TLS_OUT_OF_INDEXES
               ? 0
               : reinterpret_cast<u64>(TlsGetValue(g_tls_slot));
#else
    return t_game_thread_pointer;
#endif
}

void FreeThread() {
#ifdef _WIN32
    _aligned_free(t_block);
#else
    std::free(t_block);
#endif
    t_block = nullptr;
    SetGameThreadPointer(0);
}

} // namespace loader::tls
