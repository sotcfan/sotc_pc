#pragma once

// Diagnostic tool for studying how the game uses memory: which memory the game's code reads and
// writes, sampled while the user plays.
//
// F11 starts 30 seconds of sampling: `perf record` samples the game's user-mode instructions
// with their registers (so the address each one touches can be recomputed offline), while this
// process saves /proc/self/maps every second (mappings change a lot when the game loads an
// area). Everything goes to ~/.local/share/sotcport/samples/<n>/, to be analyzed offline.

namespace diag {

// Starts a sampling window unless one is running. Safe to call from the window thread.
void StartAccessSampling();

} // namespace diag
