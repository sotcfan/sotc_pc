// Event queues (libkernel), the PS4's version of FreeBSD's kqueue. The game creates queues,
// registers events on them (its own "user" events, or display flips through VideoOut) and
// sleeps in sceKernelWaitEqueue until one of them fires.

#include "hle/kernel/equeue.h"

#include <algorithm>
#include <optional>
#include <x86intrin.h>

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <vector>

#include "common/log.h"
#include "hle/kernel/errno.h"
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

class EventQueue {
public:
    explicit EventQueue(std::string name) : name_{std::move(name)} {}

    void Add(const KernelEvent& event) {
        std::scoped_lock lock{mutex_};
        for (Entry& entry : entries_) {
            if (entry.event.ident == event.ident && entry.event.filter == event.filter) {
                entry = {event, false};
                return;
            }
        }
        entries_.push_back({event, false});
    }

    bool Remove(u64 ident, s16 filter) {
        std::scoped_lock lock{mutex_};
        return std::erase_if(entries_, [&](const Entry& entry) {
                   return entry.event.ident == ident && entry.event.filter == filter;
               }) != 0;
    }

    // Returns false if no such event was added. See TriggerEvent() for `hint`.
    bool Trigger(u64 ident, s16 filter, void* hint) {
        bool found = false;
        {
            std::scoped_lock lock{mutex_};
            for (Entry& entry : entries_) {
                if (entry.event.ident != ident || entry.event.filter != filter) {
                    continue;
                }
                entry.triggered = true;
                if (filter == FilterUser) {
                    entry.event.udata = hint;
                } else if (filter == FilterVideoOut) {
                    ApplyVideoOutHint(entry.event, reinterpret_cast<u64>(hint));
                } else {
                    entry.event.data = reinterpret_cast<u64>(hint);
                }
                found = true;
            }
        }
        condition_.notify_all();
        return found;
    }

    // Waits for at least one fired event and copies up to `capacity` of them. `timeout` null
    // waits forever. As in kqueue, a delivered event with EventClear waits for a new trigger,
    // one with EventOneShot is removed, and any other stays fired.
    s32 Wait(KernelEvent* out, int capacity, int* count, const u32* timeout) {
        // The game's memory (timeout, the output array, count) is read and written outside the
        // lock: it may be tracked by the GPU layer, and a fault there is handled by waiting for
        // the GPU command thread, which needs this lock to signal flip events.
        const std::optional<u32> timeout_us = timeout ? std::optional<u32>{*timeout} : std::nullopt;
        std::vector<KernelEvent> delivered_events;
        std::unique_lock lock{mutex_};
        const auto ready = [&] {
            for (const Entry& entry : entries_) {
                if (entry.triggered) {
                    return true;
                }
            }
            return false;
        };
        if (!timeout_us) {
            condition_.wait(lock, ready);
        } else if (!condition_.wait_for(lock, std::chrono::microseconds(*timeout_us), ready)) {
            lock.unlock();
            if (count) {
                *count = 0;
            }
            return ErrorCode(BSD_ETIMEDOUT);
        }
        for (auto it = entries_.begin();
             it != entries_.end() && static_cast<int>(delivered_events.size()) < capacity;) {
            if (!it->triggered) {
                ++it;
                continue;
            }
            delivered_events.push_back(it->event);
            // Display events are delivered once per flip/vblank, as on the console (the game's
            // RenderFlip thread counts them); shadPS4 leaves them fired, which makes that thread
            // spin and miscount.
            if ((it->event.flags & EventClear) || it->event.filter == FilterVideoOut) {
                it->triggered = false;
                it->event.data = 0;
                it->event.fflags = 0;
            }
            if (it->event.flags & EventOneShot) {
                it = entries_.erase(it);
            } else {
                ++it;
            }
        }
        lock.unlock();
        std::copy(delivered_events.begin(), delivered_events.end(), out);
        if (count) {
            *count = static_cast<int>(delivered_events.size());
        }
        return 0;
    }

    const std::string& Name() const {
        return name_;
    }

private:
    struct Entry {
        KernelEvent event;
        bool triggered;
    };

    // VideoOut events: data = TSC low 12 bits | counter (4 bits, saturating) << 12 |
    // the hint from bit 16 (flip argument). The hint's low byte must name this event.
    static void ApplyVideoOutHint(KernelEvent& event, u64 hint) {
        if (hint == 0 || (hint & 0xFF) != event.ident || event.ident == 0xFE) {
            return;
        }
        u64 counter = (event.data >> 12) & 0xF;
        if (counter != 0xF) {
            counter++;
        }
        event.data = (__rdtsc() & 0xFFF) | (counter << 12) | (hint & 0xFFFFFFFFFFFF0000ULL);
    }

    std::string name_;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::vector<Entry> entries_;
};

void AddEvent(EventQueue* queue, const KernelEvent& event) {
    if (queue) {
        queue->Add(event);
    }
}

bool RemoveEvent(EventQueue* queue, u64 ident, s16 filter) {
    return queue && queue->Remove(ident, filter);
}

bool TriggerEvent(EventQueue* queue, u64 ident, s16 filter, void* hint) {
    return queue && queue->Trigger(ident, filter, hint);
}

namespace {

PS4_SYSV_ABI s32 sceKernelCreateEqueue(EventQueue** out, const char* name) {
    if (!out) {
        return ErrorCode(BSD_EINVAL);
    }
    *out = new EventQueue(name ? name : "");
    LOG_DEBUG("Events", "event queue \"{}\" created", name ? name : "");
    return 0;
}

PS4_SYSV_ABI s32 sceKernelDeleteEqueue(EventQueue* queue) {
    if (!queue) {
        return ErrorCode(BSD_EBADF);
    }
    // Kept alive: other threads may still be waiting on it.
    return 0;
}

PS4_SYSV_ABI s32 sceKernelAddUserEventEdge(EventQueue* queue, int id) {
    if (!queue) {
        return ErrorCode(BSD_EBADF);
    }
    queue->Add({static_cast<u64>(id), FilterUser, EventAdd | EventClear, 0, 0, nullptr});
    return 0;
}

PS4_SYSV_ABI s32 sceKernelTriggerUserEvent(EventQueue* queue, int id, void* udata) {
    if (!queue) {
        return ErrorCode(BSD_EBADF);
    }
    return queue->Trigger(static_cast<u64>(id), FilterUser, udata) ? 0 : ErrorCode(BSD_ENOENT);
}

PS4_SYSV_ABI s32 sceKernelDeleteUserEvent(EventQueue* queue, int id) {
    if (!queue) {
        return ErrorCode(BSD_EBADF);
    }
    return queue->Remove(static_cast<u64>(id), FilterUser) ? 0 : ErrorCode(BSD_ENOENT);
}

PS4_SYSV_ABI s32 sceKernelWaitEqueue(EventQueue* queue, KernelEvent* events, int capacity,
                                     int* count, const u32* timeout) {
    if (!queue) {
        return ErrorCode(BSD_EBADF);
    }
    if (!events || capacity < 1) {
        return ErrorCode(BSD_EINVAL);
    }
    perf::WaitScope perf_wait{"equeue"};
    return queue->Wait(events, capacity, count, timeout);
}

PS4_SYSV_ABI s32 sceKernelGetEventFilter(const KernelEvent* event) {
    return event ? event->filter : 0;
}

} // namespace

HLE_FUNCTION("libkernel", "sceKernelCreateEqueue", sceKernelCreateEqueue);
HLE_FUNCTION("libkernel", "sceKernelDeleteEqueue", sceKernelDeleteEqueue);
HLE_FUNCTION("libkernel", "sceKernelAddUserEventEdge", sceKernelAddUserEventEdge);
HLE_FUNCTION("libkernel", "sceKernelTriggerUserEvent", sceKernelTriggerUserEvent);
HLE_FUNCTION("libkernel", "sceKernelDeleteUserEvent", sceKernelDeleteUserEvent);
HLE_FUNCTION("libkernel", "sceKernelWaitEqueue", sceKernelWaitEqueue);
HLE_FUNCTION("libkernel", "sceKernelGetEventFilter", sceKernelGetEventFilter);

} // namespace hle::kernel
