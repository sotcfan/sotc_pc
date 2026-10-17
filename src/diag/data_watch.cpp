#include "diag/data_watch.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <format>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "common/log.h"

#ifndef _WIN32
#include <linux/hw_breakpoint.h>
#include <linux/perf_event.h>
#include <signal.h>
#include <sys/syscall.h>
#include <ucontext.h>
#include <unistd.h>
#endif

namespace diag {

#ifdef _WIN32

bool WatchData(u64, u32, bool) {
    LOG_WARNING("Watch", "data watchpoints are only implemented on Linux");
    return false;
}

void InstallDataWatch() {}

#else

namespace {

// Instructions that hit, filled by the signal handler (which may only use atomics): an open
// addressing table keyed by RIP. Entries are never removed.
struct Hit {
    std::atomic<u64> rip{0};
    std::atomic<u64> count{0};
    std::atomic<u64> watch{0}; // the watched address it touched
    u64 reported = 0;          // count at the last report (reporter thread only)
};
constexpr u32 HitSlots = 1024;
std::array<Hit, HitSlots> g_hits;
std::atomic<u64> g_lost{0}; // hits that found the table full

struct Watch {
    u64 address;
    u32 length;
    bool writes_only;
};
std::mutex g_mutex;
std::vector<Watch> g_watches;
struct sigaction g_previous_trap{};
std::atomic<bool> g_installed{false};

void NoteHit(u64 rip, u64 watched) {
    u32 slot = static_cast<u32>((rip * 0x9e3779b97f4a7c15ull) >> 54) % HitSlots;
    for (u32 probe = 0; probe < HitSlots; ++probe, slot = (slot + 1) % HitSlots) {
        Hit& hit = g_hits[slot];
        u64 expected = 0;
        if (hit.rip.load(std::memory_order_relaxed) == rip ||
            hit.rip.compare_exchange_strong(expected, rip, std::memory_order_relaxed) ||
            expected == rip) {
            hit.watch.store(watched, std::memory_order_relaxed);
            hit.count.fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }
    g_lost.fetch_add(1, std::memory_order_relaxed);
}

void TrapHandler(int signal, siginfo_t* info, void* context) {
    if (info->si_code == TRAP_PERF) {
        const auto* uc = static_cast<const ucontext_t*>(context);
        // si_addr: the watched address (glibc has no accessor for the event's sig_data).
        NoteHit(static_cast<u64>(uc->uc_mcontext.gregs[REG_RIP]),
                reinterpret_cast<u64>(info->si_addr));
        return;
    }
    // Not ours (an int3, a single step): what the crash handler would have done.
    if (g_previous_trap.sa_flags & SA_SIGINFO) {
        g_previous_trap.sa_sigaction(signal, info, context);
    } else if (g_previous_trap.sa_handler != SIG_IGN && g_previous_trap.sa_handler != SIG_DFL) {
        g_previous_trap.sa_handler(signal);
    }
}

void ReportLoop() {
    for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds{2});
        for (Hit& hit : g_hits) {
            const u64 rip = hit.rip.load(std::memory_order_relaxed);
            const u64 count = hit.count.load(std::memory_order_relaxed);
            if (rip == 0 || count == hit.reported) {
                continue;
            }
            LOG_INFO("Watch", "{:#x} x{} (+{}) on {:#x}", rip, count, count - hit.reported,
                     hit.watch.load(std::memory_order_relaxed));
            hit.reported = count;
        }
        if (const u64 lost = g_lost.exchange(0, std::memory_order_relaxed)) {
            LOG_WARNING("Watch", "{} hits lost (table full)", lost);
        }
    }
}

void InstallHandler() {
    if (g_installed.exchange(true)) {
        return;
    }
    struct sigaction action{};
    action.sa_sigaction = TrapHandler;
    action.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_NODEFER | SA_RESTART;
    sigemptyset(&action.sa_mask);
    sigaction(SIGTRAP, &action, &g_previous_trap);
    std::thread{ReportLoop}.detach();
}

} // namespace

bool WatchData(u64 address, u32 length, bool writes_only) {
    if ((length != 1 && length != 2 && length != 4 && length != 8) || address % length != 0) {
        LOG_ERROR("Watch", "{:#x}:{}: the length must be 1, 2, 4 or 8 and the address aligned "
                           "to it", address, length);
        return false;
    }
    InstallHandler();
    std::scoped_lock lock{g_mutex};
    if (g_watches.size() == 4) {
        LOG_ERROR("Watch", "{:#x}: four watches already (the CPU has four debug registers)",
                  address);
        return false;
    }
    perf_event_attr attr{};
    attr.type = PERF_TYPE_BREAKPOINT;
    attr.size = sizeof(attr);
    attr.bp_type = writes_only ? HW_BREAKPOINT_W : HW_BREAKPOINT_RW;
    attr.bp_addr = address;
    attr.bp_len = length;
    attr.sample_period = 1;
    attr.exclude_kernel = 1;
    attr.exclude_hv = 1;
    attr.sigtrap = 1;        // a synchronous SIGTRAP on the thread that hit
    attr.remove_on_exec = 1; // required with sigtrap
    attr.inherit = 1;        // threads created later get it too
    attr.inherit_thread = 1;
    u32 opened = 0, failed = 0;
    int error = 0;
    for (const auto& entry : std::filesystem::directory_iterator{"/proc/self/task"}) {
        const pid_t tid = std::atoi(entry.path().filename().c_str());
        const long fd = syscall(SYS_perf_event_open, &attr, tid, -1, -1, PERF_FLAG_FD_CLOEXEC);
        if (fd < 0) {
            ++failed;
            error = errno;
        } else {
            ++opened; // kept open for the rest of the run
        }
    }
    if (opened == 0) {
        LOG_ERROR("Watch", "{:#x}: perf_event_open failed: {}", address, std::strerror(error));
        return false;
    }
    g_watches.push_back({address, length, writes_only});
    LOG_INFO("Watch", "watching {} of {:#x}..{:#x} on {} threads{}",
             writes_only ? "writes" : "reads and writes", address, address + length, opened,
             failed ? std::format(" ({} failed: {})", failed, std::strerror(error)) : "");
    return true;
}

void InstallDataWatch() {
    const char* text = std::getenv("SOTC_WATCH_DATA");
    if (!text || !*text) {
        return;
    }
    char* end = nullptr;
    const u64 address = std::strtoull(text, &end, 16);
    u32 length = 4;
    bool writes_only = false;
    if (*end == ':') {
        length = static_cast<u32>(std::strtoul(end + 1, &end, 10));
        if (*end == ':') {
            writes_only = end[1] == 'w';
        }
    }
    WatchData(address, length, writes_only);
}

#endif

} // namespace diag
