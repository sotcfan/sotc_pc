#pragma once

#include <cstdio>

namespace hle::libc {

// Reads the game's libc parameters (its own malloc and operator new) and runs its malloc_init.
// Must run on the game's main thread, before the entry point.
void Initialize(const void* libc_param);

// Runs the functions the game registered with atexit/__cxa_atexit, newest first.
void RunExitHandlers();

// The FILE the game passes to stdio functions, as a host FILE (handles _Stdout and _Stderr).
std::FILE* HostFile(void* guest_file);

} // namespace hle::libc
