#pragma once

#include "common/types.h"

#include <string>
#include <vector>

namespace hle::kernel {

// The PS4's (FreeBSD amd64) time structures, 16 bytes with 64-bit fields. The host's own types
// may differ: Windows' `long` tv_nsec/tv_usec has 32 bits.
struct Ps4Timespec {
    s64 tv_sec;
    s64 tv_nsec;
};
struct Ps4Timeval {
    s64 tv_sec;
    s64 tv_usec;
};

// Registers the thread that runs the game's entry point as the game's main thread.
void InitializeMainThread();

// Event flags that have threads waiting on them, one line each (for thread dumps).
std::vector<std::string> WaitedEventFlags();

} // namespace hle::kernel
