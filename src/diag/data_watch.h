#pragma once

#include "common/types.h"

// Diagnostic tool for reverse engineering: which instructions of the game read or write a given
// piece of memory, with the CPU's debug registers (hardware watchpoints, through perf_event_open,
// so no debugger is needed and the game runs at full speed between hits).
//
// Every thread gets the watchpoint (and the threads they create later inherit it). On a hit the
// kernel stops the thread right after the instruction with a SIGTRAP; the handler notes where the
// instruction ends (RIP) and lets the thread go on. A thread logs every 2 s the instructions that
// hit, with their counts, as "<RIP> x<count>": the instruction that touched the memory is the one
// just before that address (disassemble a little before it).
//
// SOTC_WATCH_DATA=<address>[:<length>[:w]] watches from the start: length 1, 2, 4 or 8 (default
// 4), "w" for writes only (default: reads and writes; x86 has no read-only watchpoints). Up to four
// watches at once (the CPU has four debug registers); WatchData() adds them from code, for
// addresses only known at run time (a heap object found by a hook).

namespace diag {

// Watches [address, address + length) on every thread. length: 1, 2, 4 or 8, and address
// aligned to it. Returns false (and logs why) if the kernel refused it.
bool WatchData(u64 address, u32 length, bool writes_only);

// Reads SOTC_WATCH_DATA. After crash::Install (it chains to the crash handler's SIGTRAP).
void InstallDataWatch();

} // namespace diag
