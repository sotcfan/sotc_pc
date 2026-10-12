#pragma once

#include "common/types.h"

// Redirecting functions inside the game's executable (not imports): the start of the function
// becomes a jump to our code, and its first instructions move to a trampoline that runs the
// function as it was.

namespace loader {

// Redirects the function of the game's executable at `address` to `replacement` and returns a
// pointer that runs the original function. Must be called between Linker::LoadExecutable and
// Linker::Link, while the game's code is still writable. Returns nullptr, and logs why, if the
// start of the function cannot be moved (too short, or a short jump that cannot be relocated).
void* HookFunction(u64 address, const void* replacement);

// Executable memory within ±2 GiB of the game's code, for code that the game's code jumps to
// with 32-bit offsets (trampolines, generated stubs). Never freed.
u8* AllocateNearCode(size_t size);

} // namespace loader
