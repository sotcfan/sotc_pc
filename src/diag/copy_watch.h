#pragma once

#include <cstddef>

// Diagnostic tool for studying how the game uses GPU memory: which of the game's memcpy/memmove calls read from or write to GPU memory (Garlic/Onion), counted
// exactly rather than sampled.
//
// SOTC_WATCH_COPIES=1: every copy through the game's libc imports is attributed to its call
// site and to the memory types of its source and destination; every 10 seconds the log gets
// the totals and the call sites that touch Garlic or Onion. Costs a memory manager lookup per
// copy, so only for measuring.

namespace diag {

extern bool g_watch_copies;

// Reads SOTC_WATCH_COPIES. Called once at start-up, before the game runs.
void InstallCopyWatch();

void NoteCopy(void* dst, const void* src, size_t size, const void* caller);

} // namespace diag
