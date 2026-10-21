// Process-level libkernel functions: stack protection, clocks, sleeping, console model.

#include <cstdlib>
#include <time.h>
#include <x86intrin.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <ctime>
#include <random>
#include <thread>

#include "common/log.h"
#include "common/types.h"
#include "diag/frame_trace.h"
#include "hle/kernel/errno.h"
#include "hle/kernel/kernel.h"
#include "hle/registry.h"
#include "loader/crash.h"

// sotcport TEMP: game wait times (third_party/shadps4/shim/perf_temp.h).
namespace perf {
bool Enabled();
uint64_t Now();
void AddWait(const char* kind, uint64_t ns);
struct WaitScope {
    explicit WaitScope(const char* kind_) : kind{kind_}, start{Enabled() ? Now() : 0} {}
    ~WaitScope() {
        if (start) {
            AddWait(kind, Now() - start);
        }
    }
    const char* kind;
    uint64_t start;
};
} // namespace perf

namespace hle::kernel {

namespace {

// Value the game's functions copy to the stack and check before returning (-fstack-protector).
u64 g_stack_guard = 0x5348414457303031ULL;

[[noreturn]] PS4_SYSV_ABI void stack_chk_fail() {
    crash::Fatal("the game's stack protector detected corrupted data (__stack_chk_fail)");
}

const auto g_process_start = std::chrono::steady_clock::now();

// Microseconds since the process started.
PS4_SYSV_ABI u64 sceKernelGetProcessTime() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now() - g_process_start)
        .count();
}

PS4_SYSV_ABI u64 sceKernelReadTsc() {
    return __rdtsc();
}

// The TSC rate, measured once against the system clock.
PS4_SYSV_ABI u64 sceKernelGetTscFrequency() {
    static const u64 frequency = [] {
        const auto start_time = std::chrono::steady_clock::now();
        const u64 start_tsc = __rdtsc();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        const u64 ticks = __rdtsc() - start_tsc;
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
        const u64 measured = static_cast<u64>(ticks / seconds);
        LOG_INFO("Kernel", "TSC frequency: {:.1f} MHz", measured / 1e6);
        return measured;
    }();
    return frequency;
}

PS4_SYSV_ABI s32 sceKernelUsleep(u32 microseconds) {
    perf::WaitScope perf_wait{"usleep"};
    diag::trace::WaitScope trace_wait{diag::trace::Usleep, nullptr, SOTC_TRACE_CALLERS};
    std::this_thread::sleep_for(std::chrono::microseconds(microseconds));
    return 0;
}

PS4_SYSV_ABI s32 sceKernelSleep(u32 seconds) {
    std::this_thread::sleep_for(std::chrono::seconds(seconds));
    return 0;
}

// 0 = original PS4, 1 = PS4 Pro. The game picks resolution and effects from this, and only
// offers 60 FPS (its "performance" mode) on the Pro: on an original PS4 its logic was never meant
// to run at 60 (at 60 FPS the fifth colossus flies straight up out of its arena, user test
// 2026-10-12). SOTC_NEO_MODE=1 reports a Pro. The GPU layer asks too (gpu_glue.cpp) and must
// get the same answer.
PS4_SYSV_ABI s32 sceKernelIsNeoMode() {
    static const s32 neo = [] {
        const char* value = std::getenv("SOTC_NEO_MODE");
        return value && *value == '1' ? 1 : 0;
    }();
    return neo;
}

// ---- POSIX time (libScePosix) ----

// FreeBSD clock ids.
clockid_t HostClock(s32 clock) {
    switch (clock) {
    case 0:  // CLOCK_REALTIME
    case 9:  // CLOCK_REALTIME_PRECISE
    case 10: // CLOCK_REALTIME_FAST
        return CLOCK_REALTIME;
    case 13: // CLOCK_SECOND
        return CLOCK_REALTIME_COARSE;
    case 14: // CLOCK_THREAD_CPUTIME_ID
        return CLOCK_THREAD_CPUTIME_ID;
    case 2:  // CLOCK_PROF
    case 15: // CLOCK_PROCESS_CPUTIME_ID
        return CLOCK_PROCESS_CPUTIME_ID;
    default: // CLOCK_MONOTONIC (4), CLOCK_UPTIME (5) and their variants
        return CLOCK_MONOTONIC;
    }
}

PS4_SYSV_ABI s32 posix_clock_gettime(s32 clock, Ps4Timespec* out) {
    if (!out) {
        SetErrno(BSD_EFAULT);
        return -1;
    }
    timespec now;
    clock_gettime(HostClock(clock), &now);
    *out = {static_cast<s64>(now.tv_sec), static_cast<s64>(now.tv_nsec)};
    return 0;
}

PS4_SYSV_ABI s32 posix_gettimeofday(Ps4Timeval* out, void*) {
    if (out) {
        timespec now;
        clock_gettime(CLOCK_REALTIME, &now);
        out->tv_sec = now.tv_sec;
        out->tv_usec = now.tv_nsec / 1000;
    }
    return 0;
}

PS4_SYSV_ABI s32 posix_nanosleep(const Ps4Timespec* duration, Ps4Timespec* remaining) {
    perf::WaitScope perf_wait{"nanosleep"};
    diag::trace::WaitScope trace_wait{diag::trace::Nanosleep, nullptr, SOTC_TRACE_CALLERS};
    if (!duration) {
        SetErrno(BSD_EFAULT);
        return -1;
    }
    const timespec host{static_cast<std::time_t>(duration->tv_sec),
                        static_cast<long>(duration->tv_nsec)};
    timespec left{};
    if (nanosleep(&host, &left) != 0) {
        if (remaining) {
            *remaining = {static_cast<s64>(left.tv_sec), static_cast<s64>(left.tv_nsec)};
        }
        SetErrno(ToBsdErrno(errno));
        return -1;
    }
    return 0;
}

// A random (version 4) UUID.
PS4_SYSV_ABI s32 sceKernelUuidCreate(u8* uuid) {
    if (!uuid) {
        return ErrorCode(BSD_EINVAL);
    }
    static thread_local std::mt19937_64 generator{std::random_device{}()};
    for (int i = 0; i < 16; i += 8) {
        const u64 bits = generator();
        std::memcpy(uuid + i, &bits, 8);
    }
    uuid[6] = (uuid[6] & 0x0F) | 0x40;
    uuid[8] = (uuid[8] & 0x3F) | 0x80;
    return 0;
}

} // namespace

HLE_FUNCTION("libScePosix", "clock_gettime", posix_clock_gettime);
HLE_FUNCTION("libScePosix", "gettimeofday", posix_gettimeofday);
HLE_FUNCTION("libScePosix", "nanosleep", posix_nanosleep);
HLE_FUNCTION("libkernel", "sceKernelUuidCreate", sceKernelUuidCreate);
HLE_OBJECT("libkernel", "__stack_chk_guard", g_stack_guard);
HLE_FUNCTION("libkernel", "__stack_chk_fail", stack_chk_fail);
HLE_FUNCTION("libkernel", "sceKernelGetProcessTime", sceKernelGetProcessTime);
HLE_FUNCTION("libkernel", "sceKernelReadTsc", sceKernelReadTsc);
HLE_FUNCTION("libkernel", "sceKernelGetTscFrequency", sceKernelGetTscFrequency);
HLE_FUNCTION("libkernel", "sceKernelUsleep", sceKernelUsleep);
HLE_FUNCTION("libkernel", "sceKernelSleep", sceKernelSleep);
HLE_FUNCTION("libkernel", "sceKernelIsNeoMode", sceKernelIsNeoMode);

} // namespace hle::kernel
