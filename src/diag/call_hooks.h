#pragma once

#include <array>
#include <span>
#include <vector>

#include "common/types.h"

// Watching calls to functions inside the game's executable without changing what they do: each
// hooked function runs as it was, with the observer called before it starts and after it
// returns. Used by the diagnostic tools that map the game's renderer (gnm_trace, gnm_recorder).
//
// The functions are hooked with loader::HookFunction, so this must run between
// Linker::LoadExecutable and Linker::Link. Their return address is swapped for a stub of ours
// and kept on a per-thread stack, so the observer sees the return even for tail calls.

namespace diag {

struct ObservedCall {
    u32 id;                   // index of the function in the list given to ObserveCalls
    u32 depth;                // calls of the same observer already running on this thread
    std::array<u64, 6> args;  // rdi, rsi, rdx, rcx, r8, r9 at entry
    std::array<u64, 4> float_args; // low 64 bits of xmm0-xmm3 (float/double arguments)
    // The first two arguments passed on the stack (the 7th and 8th integer arguments). Read
    // from the caller's stack whatever the function takes: meaningless if it has fewer.
    std::array<u64, 2> stack_args;
    u64 return_address;       // where the function returns to
    std::array<u64, 8> user;  // free for the observer, kept from on_enter to on_exit
};

struct CallObserver {
    // Before the function runs. May change `user`, nothing else.
    void (*on_enter)(ObservedCall& call);
    // After the function returned (its return value is not visible here).
    void (*on_exit)(const ObservedCall& call);
};

// Hooks every function in `addresses`; `observer` must live as long as the program. Returns,
// per function, whether it could be hooked (a function shorter than a jump cannot).
std::vector<bool> ObserveCalls(std::span<const u64> addresses, const CallObserver* observer);

} // namespace diag
