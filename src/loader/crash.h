#pragma once

namespace crash {

// Installs handlers that, when the game crashes, print where it stopped (in the same addresses
// used by disassemblers), the registers, the call stack and the last imports called.
void Install();

// Every thread that runs game code calls this when it starts: it gets its own signal stack, so
// a stack overflow can still be reported.
void InstallAltStack();

// Development aid: every game thread prints where it is and its last imports. The game keeps
// running. Triggered by SIGUSR1 (kill -USR1 <pid>) or by --dump-after.
void DumpAllThreads();

// Handlers that may resolve a fault before it is reported as a crash (the GPU layer protects
// pages to track CPU writes, and patches code it generates). Each returns true if it handled
// the fault; lower priorities run first.
using AccessFaultHandler = bool (*)(void* context, void* fault_address);
using IllegalInstructionHandler = bool (*)(void* context);
void AddAccessFaultHandler(AccessFaultHandler handler, unsigned priority);
void AddIllegalInstructionHandler(IllegalInstructionHandler handler, unsigned priority);

// Prints the same report as a crash, starting at the caller of the function that calls it,
// and ends the process. For fatal errors detected by our code (abort, __stack_chk_fail...).
[[noreturn]] void Fatal(const char* reason);

} // namespace crash
