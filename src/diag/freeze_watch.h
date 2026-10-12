#pragma once

// Freeze watchdog: when the game stops presenting frames for a while (the picture freezes
// without a crash), writes to the log what every thread of the process is doing, so the cause
// can be found afterwards:
//   - per thread: name, state (R running, S sleeping, D uninterruptible...), the kernel function
//     it waits in (wchan) and the system call it is blocked in, with arguments (a futex address
//     identifies the lock);
//   - then the usual thread dump (crash::DumpAllThreads: stacks, event flags, fibers).
// Each freeze is reported once; the watchdog re-arms when frames come again.

namespace diag {

// Called regularly from the main loop with the number of frames presented so far.
void CheckForFreeze(unsigned long long frames_presented);

} // namespace diag
