// Event flags (libkernel): a 64-bit pattern that threads set and wait on, with AND/OR conditions
// and optional clearing when the wait succeeds.

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <vector>
#include <algorithm>

#include "common/log.h"
#include "common/types.h"
#include "diag/frame_trace.h"
#include "hle/kernel/errno.h"
#include "hle/kernel/kernel.h"
#include "hle/registry.h"

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

constexpr u32 WaitAnd = 0x01;
constexpr u32 WaitOr = 0x02;
constexpr u32 WaitClearAll = 0x10;
constexpr u32 WaitClearPattern = 0x20;

constexpr u32 AttrMultipleWaiters = 0x20;

class EventFlag {
public:
    EventFlag(std::string name, u32 attributes, u64 bits)
        : name_{std::move(name)}, attributes_{attributes}, bits_{bits} {}

    const std::string& Name() const {
        return name_;
    }

    // `timeout` (microseconds, may be null = forever) receives the time left.
    //
    // As in the PS4 kernel, a waiter is released by the Set() that satisfies it, at that moment:
    // Set() checks the queued waiters in order, hands each satisfied one the pattern it saw and
    // applies its clear mode. Re-checking the bits only when the thread wakes up would lose
    // wake-ups: the game sets two flags back to back, and the thread woken by the first one may
    // clear the second (to start the next frame) before the second waiter gets to look at it.
    s32 Wait(u64 pattern, u32 mode, u64* result, u32* timeout) {
        std::unique_lock lock{mutex_};
        if (!(attributes_ & AttrMultipleWaiters) && !waiters_.empty()) {
            return ErrorCode(BSD_EPERM);
        }
        last_waiter_ = logging::CurrentThreadName();
        if (Matches(bits_, pattern, mode)) {
            if (result) {
                *result = bits_;
            }
            ApplyClear(pattern, mode);
            wait_count_++;
            return 0;
        }
        Waiter waiter{pattern, mode};
        waiters_.push_back(&waiter);
        const auto released = [&] { return waiter.satisfied || deleted_; };
        if (!timeout) {
            condition_.wait(lock, released);
        } else {
            const auto start = std::chrono::steady_clock::now();
            condition_.wait_for(lock, std::chrono::microseconds(*timeout), released);
            const auto spent = std::chrono::duration_cast<std::chrono::microseconds>(
                                   std::chrono::steady_clock::now() - start)
                                   .count();
            *timeout = waiter.satisfied && spent < *timeout ? static_cast<u32>(*timeout - spent) : 0;
        }
        std::erase(waiters_, &waiter);
        if (waiter.satisfied) {
            if (result) {
                *result = waiter.result;
            }
            wait_count_++;
            return 0;
        }
        if (result) {
            *result = bits_;
        }
        return deleted_ ? ErrorCode(BSD_EACCES) : ErrorCode(BSD_ETIMEDOUT);
    }

    s32 Poll(u64 pattern, u32 mode, u64* result) {
        std::scoped_lock lock{mutex_};
        if (result) {
            *result = bits_;
        }
        if (!Matches(bits_, pattern, mode)) {
            return ErrorCode(BSD_EBUSY);
        }
        ApplyClear(pattern, mode);
        return 0;
    }

    void Set(u64 pattern, u64 caller) {
        std::scoped_lock lock{mutex_};
        set_count_++;
        last_setter_ = caller;
        bits_ |= pattern;
        bool released = false;
        for (Waiter* waiter : waiters_) {
            if (!waiter->satisfied && Matches(bits_, waiter->pattern, waiter->mode)) {
                waiter->satisfied = true;
                waiter->result = bits_;
                ApplyClear(waiter->pattern, waiter->mode);
                released = true;
            }
        }
        if (released) {
            condition_.notify_all();
        }
    }

    // Keeps only the bits that are set in `pattern`.
    void Clear(u64 pattern, u64 caller) {
        std::scoped_lock lock{mutex_};
        clear_count_++;
        last_clearer_ = caller;
        bits_ &= pattern;
    }

    std::string Describe() {
        std::scoped_lock lock{mutex_};
        if (waiters_.empty()) {
            return {};
        }
        return std::format("{} \"{}\": bits {:#x}, waiting: {} ({}), set {} times, last by {:#x}; "
                           "{} waits satisfied; cleared {} times, last by {:#x}",
                           static_cast<void*>(this), name_, bits_, waiters_.size(), last_waiter_, set_count_,
                           last_setter_, wait_count_, clear_count_, last_clearer_);
    }

    void MarkDeleted() {
        std::scoped_lock lock{mutex_};
        deleted_ = true;
        condition_.notify_all();
    }

private:
    struct Waiter {
        u64 pattern;
        u32 mode;
        bool satisfied = false;
        u64 result = 0;
    };

    static bool Matches(u64 bits, u64 pattern, u32 mode) {
        return (mode & WaitOr) ? (bits & pattern) != 0 : (bits & pattern) == pattern;
    }

    void ApplyClear(u64 pattern, u32 mode) {
        if (mode & WaitClearAll) {
            bits_ = 0;
        } else if (mode & WaitClearPattern) {
            bits_ &= ~pattern;
        }
    }

    std::string name_;
    u32 attributes_;
    u64 bits_;
    std::vector<Waiter*> waiters_; // in arrival order (FIFO; priority order is not modeled)
    std::string last_waiter_;
    u64 set_count_ = 0;
    u64 last_setter_ = 0;
    u64 wait_count_ = 0;
    u64 clear_count_ = 0;
    u64 last_clearer_ = 0;
    bool deleted_ = false;
    std::mutex mutex_;
    std::condition_variable condition_;
};

std::mutex g_flags_mutex;
std::vector<EventFlag*> g_flags;

bool ValidMode(u32 mode) {
    const u32 wait = mode & 0x0F;
    const u32 clear = mode & 0xF0;
    return (wait == WaitAnd || wait == WaitOr) &&
           (clear == 0 || clear == WaitClearAll || clear == WaitClearPattern);
}

PS4_SYSV_ABI s32 sceKernelCreateEventFlag(EventFlag** out, const char* name, u32 attributes,
                                          u64 initial, const void* options) {
    if (!out || !name || options) {
        return ErrorCode(BSD_EINVAL);
    }
    *out = new EventFlag(name, attributes, initial);
    {
        std::scoped_lock lock{g_flags_mutex};
        g_flags.push_back(*out);
    }
    LOG_DEBUG("Events", "event flag \"{}\" created (attributes {:#x}, value {:#x})", name,
              attributes, initial);
    return 0;
}

PS4_SYSV_ABI s32 sceKernelDeleteEventFlag(EventFlag* flag) {
    if (!flag) {
        return ErrorCode(BSD_ESRCH);
    }
    // Waiters may still be inside Wait(); the object is kept so they can leave safely.
    flag->MarkDeleted();
    return 0;
}

PS4_SYSV_ABI s32 sceKernelSetEventFlag(EventFlag* flag, u64 pattern) {
    if (!flag) {
        return ErrorCode(BSD_ESRCH);
    }
    if (diag::trace::Enabled()) {
        diag::trace::Wake(diag::trace::EventFlag, flag,
                          reinterpret_cast<u64>(__builtin_return_address(0)));
    }
    flag->Set(pattern, reinterpret_cast<u64>(__builtin_return_address(0)));
    return 0;
}

PS4_SYSV_ABI s32 sceKernelClearEventFlag(EventFlag* flag, u64 pattern) {
    if (!flag) {
        return ErrorCode(BSD_ESRCH);
    }
    flag->Clear(pattern, reinterpret_cast<u64>(__builtin_return_address(0)));
    return 0;
}

PS4_SYSV_ABI s32 sceKernelWaitEventFlag(EventFlag* flag, u64 pattern, u32 mode, u64* result,
                                        u32* timeout) {
    if (!flag) {
        return ErrorCode(BSD_ESRCH);
    }
    if (pattern == 0 || !ValidMode(mode)) {
        return ErrorCode(BSD_EINVAL);
    }
    perf::WaitScope perf_wait{"eventflag"};
    if (diag::trace::Enabled()) {
        diag::trace::WaitBegin(diag::trace::EventFlag, flag, SOTC_TRACE_CALLERS);
        const s32 status = flag->Wait(pattern, mode, result, timeout);
        diag::trace::WaitEnd(diag::trace::EventFlag, flag);
        return status;
    }
    return flag->Wait(pattern, mode, result, timeout);
}

PS4_SYSV_ABI s32 sceKernelPollEventFlag(EventFlag* flag, u64 pattern, u32 mode, u64* result) {
    if (!flag) {
        return ErrorCode(BSD_ESRCH);
    }
    if (pattern == 0 || !ValidMode(mode)) {
        return ErrorCode(BSD_EINVAL);
    }
    return flag->Poll(pattern, mode, result);
}

} // namespace

std::vector<std::string> WaitedEventFlags() {
    std::vector<std::string> result;
    std::scoped_lock lock{g_flags_mutex};
    for (EventFlag* flag : g_flags) {
        if (std::string line = flag->Describe(); !line.empty()) {
            result.push_back(std::move(line));
        }
    }
    return result;
}

HLE_FUNCTION("libkernel", "sceKernelCreateEventFlag", sceKernelCreateEventFlag);
HLE_FUNCTION("libkernel", "sceKernelDeleteEventFlag", sceKernelDeleteEventFlag);
HLE_FUNCTION("libkernel", "sceKernelSetEventFlag", sceKernelSetEventFlag);
HLE_FUNCTION("libkernel", "sceKernelClearEventFlag", sceKernelClearEventFlag);
HLE_FUNCTION("libkernel", "sceKernelWaitEventFlag", sceKernelWaitEventFlag);
HLE_FUNCTION("libkernel", "sceKernelPollEventFlag", sceKernelPollEventFlag);

} // namespace hle::kernel
