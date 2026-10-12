#pragma once

#include "loader/module.h"

// Thread-local storage of the game.
//
// The game reaches its thread-local variables through the thread pointer stored at fs:[0],
// at small negative offsets (the x86-64 "variant II" layout). On Linux the fs register belongs
// to glibc, and the bytes below glibc's thread pointer hold our own thread_local variables. So
// the loader rewrites every "mov reg, fs:[0]" in the game into "mov reg, fs:[offset]", which
// reads the game's thread pointer from one of our thread_local variables instead. On Windows the
// rewrite is "mov reg, gs:[0x1480 + 8 * slot]", a TLS slot of the TEB. Each thread gets its own
// block of game TLS, laid out as on the PS4.

namespace loader::tls {

// Rewrites the "mov reg, fs:[0]" instructions in `code`. Returns how many were changed.
size_t PatchThreadPointerReads(u8* code, u64 size);

// The module whose TLS image new threads copy.
void SetTemplate(const Module::TlsInfo& info);

// Gives the calling thread its block of game TLS. Every thread that runs game code must call
// this before running it, and FreeThread() when done.
void InitializeThread();
void FreeThread();

// The calling thread's game thread pointer (0 if it has no game TLS).
u64 ThreadPointer();

} // namespace loader::tls
