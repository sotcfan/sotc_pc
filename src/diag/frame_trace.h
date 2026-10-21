#pragma once

// Frame trace (diagnostics): what every thread waited for, and who woke it, in a window of a
// few seconds, to find the critical path of a frame (which wait of which thread holds the next
// present back). SOTC_FRAME_TRACE=<from>-<to> (seconds since start) records, from every thread:
//   - the game's blocking calls (event queues, event flags, condition variables, contended
//     mutexes, sleeps), with the object waited on and two return addresses into the game;
//   - the calls that release them (event flag set, condition broadcast, event trigger);
//   - the GPU layer's timeline marks (perf::Mark: submissions, flips, presents, readbacks, stage
//     A/B states, GPU command buffer spans; OPTIONS.md, SOTC_FRAME_TIMELINE).
// After <to> the events go to <data folder>/frame_trace.bin (tools/frame_trace.py reads it).
// Header only for the checks, so the shadPS4 sources can use it too.

#include <atomic>
#include <cstdint>

namespace diag::trace {

// What a thread blocks in (Event::sub of 'B'/'E'/'K' events).
enum Wait : std::uint8_t {
    Equeue = 1,
    EventFlag,
    Cond,
    CondTimed,
    Mutex,
    Usleep,
    Nanosleep,
};

// Set once at start when SOTC_FRAME_TRACE is given; the window is checked inside.
inline std::atomic<bool> g_enabled{false};

inline bool Enabled() {
    return g_enabled.load(std::memory_order_relaxed);
}

// Reads SOTC_FRAME_TRACE (main(), before the game starts).
void Initialize();

// `caller`, `caller2`: return addresses into the game (Callers below), 0 if unknown.
void WaitBegin(Wait kind, const void* object, std::uint64_t caller, std::uint64_t caller2);
void WaitEnd(Wait kind, const void* object);
// A wait already over, from `begin` to `end` (Now()): for frequent short waits recorded only when
// long (contended mutexes: the game's job workers wait ~1.5 million times a second, 0.15 µs at
// the median).
void WaitSpan(Wait kind, const void* object, std::uint64_t begin, std::uint64_t end);
// steady_clock in nanoseconds, the trace's clock.
std::uint64_t Now();
void Wake(Wait kind, const void* object, std::uint64_t caller);
// A timeline mark (perf::Mark / perf::MarkAt), with its own time on steady_clock.
void Mark(char kind, std::uint64_t ns, std::uint64_t value);

// The return address of the frame one level above `frame` (a frame pointer of the HLE function,
// __builtin_frame_address(0)): the game function that called the game function that called us.
// The game keeps frame pointers (perf follows its call chains); a saved frame pointer outside the
// next 64 KiB of the stack is not followed.
inline std::uint64_t CallerAbove(void* frame) {
    const auto* fp = static_cast<const std::uint64_t*>(frame);
    const auto* next = reinterpret_cast<const std::uint64_t*>(fp[0]);
    if (next <= fp || next - fp > 64 * 1024 / 8) {
        return 0;
    }
    return next[1];
}

// WaitBegin/WaitEnd around a scope (the callers are taken where it is declared).
struct WaitScope {
    WaitScope(Wait kind_, const void* object_, std::uint64_t caller, std::uint64_t caller2)
        : kind{kind_}, object{object_}, on{Enabled()} {
        if (on) {
            WaitBegin(kind, object, caller, caller2);
        }
    }
    ~WaitScope() {
        if (on) {
            WaitEnd(kind, object);
        }
    }
    Wait kind;
    const void* object;
    bool on;
};

} // namespace diag::trace

// The two return addresses into the game, from inside an HLE function.
#define SOTC_TRACE_CALLERS                                                                       \
    reinterpret_cast<std::uint64_t>(__builtin_return_address(0)),                                \
        ::diag::trace::CallerAbove(__builtin_frame_address(0))
