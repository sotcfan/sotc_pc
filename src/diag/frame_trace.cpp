// Frame trace (see frame_trace.h). One array of fixed-size events filled by every thread with an
// atomic index; written to a file once, by the first thread that records after the window.
//
// The file is written by the presentation thread at the first present after the window: events
// also come from signal handlers (readbacks in the GPU layer's fault handler), where file I/O
// could deadlock on a lock the interrupted code holds.
//
// File layout (little endian), read by tools/frame_trace.py:
//   "SOTCTRC1"                     8 bytes
//   u32 thread count, u32 0
//   per thread: u32 tid, u32 0, char name[16]   (from /proc/self/task/<tid>/comm at the end)
//   u64 event count
//   per event: Event (32 bytes, below)

#include "diag/frame_trace.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "common/log.h"
#include "common/platform.h"

namespace diag::trace {

namespace {

struct Event {
    std::uint64_t ns;      // steady_clock
    std::uint64_t object;  // the object waited on / released ('B', 'E', 'K'); the value ('M')
    std::uint64_t caller;  // return address into the game ('B', 'K')
    std::uint32_t caller2; // the return address one frame above (game code is below 4 GiB)
    std::uint16_t thread;  // index into the thread table
    char kind;             // 'B' wait begins, 'E' wait ends, 'K' wake, 'M' timeline mark
    std::uint8_t sub;      // Wait for 'B'/'E'/'K'; the mark's letter for 'M'
};
static_assert(sizeof(Event) == 32);

// 4 Mi events (128 MiB, touched only as they are filled): a busy second has ~300 thousand.
constexpr std::size_t Capacity = std::size_t{1} << 22;

std::uint64_t g_from_ns = 0, g_to_ns = 0; // the window, on steady_clock
std::unique_ptr<Event[]> g_events;
std::atomic<std::size_t> g_next{0};
std::atomic<bool> g_written{false};

std::mutex g_threads_mutex;
std::vector<std::uint32_t> g_thread_ids; // index -> tid
thread_local int t_thread = -1;

int ThreadIndex() {
    if (t_thread < 0) {
        std::scoped_lock lock{g_threads_mutex};
        t_thread = static_cast<int>(g_thread_ids.size());
        g_thread_ids.push_back(platform::CurrentThreadId());
    }
    return t_thread;
}

void Write() {
    // Threads still recording may be filling their slot: the ones claimed after this point are
    // left out, and a slot claimed just before may be half written (the tool drops events with
    // a zero time).
    const std::size_t count = std::min(g_next.load(), Capacity);
    const auto path = platform::DataDir() / "frame_trace.bin";
    std::FILE* file = std::fopen(path.string().c_str(), "wb");
    if (!file) {
        LOG_ERROR("Trace", "frame trace: cannot write {}", path.string());
        return;
    }
    std::vector<std::uint32_t> threads;
    {
        std::scoped_lock lock{g_threads_mutex};
        threads = g_thread_ids;
    }
    std::fwrite("SOTCTRC1", 1, 8, file);
    const std::uint32_t header[2] = {static_cast<std::uint32_t>(threads.size()), 0};
    std::fwrite(header, sizeof(header), 1, file);
    for (const std::uint32_t tid : threads) {
        char name[16] = {};
        const std::string comm = "/proc/self/task/" + std::to_string(tid) + "/comm";
        if (std::FILE* f = std::fopen(comm.c_str(), "r")) {
            if (std::fgets(name, sizeof(name), f)) {
                name[std::strcspn(name, "\n")] = 0;
            }
            std::fclose(f);
        }
        const std::uint32_t ids[2] = {tid, 0};
        std::fwrite(ids, sizeof(ids), 1, file);
        std::fwrite(name, sizeof(name), 1, file);
    }
    const std::uint64_t events = count;
    std::fwrite(&events, sizeof(events), 1, file);
    std::fwrite(g_events.get(), sizeof(Event), count, file);
    std::fclose(file);
    LOG_INFO("Trace", "frame trace: {} events of {} threads written to {}{}", count,
             threads.size(), path.string(),
             g_next.load() > Capacity ? " (full: the end of the window is missing)" : "");
}

void Record(char kind, std::uint8_t sub, std::uint64_t ns, std::uint64_t object,
            std::uint64_t caller, std::uint64_t caller2) {
    if (ns < g_from_ns) {
        return;
    }
    if (ns > g_to_ns) {
        if (kind == 'M' && sub == 'P' && !g_written.exchange(true)) {
            Write();
        }
        return;
    }
    const std::size_t index = g_next.fetch_add(1, std::memory_order_relaxed);
    if (index >= Capacity) {
        return;
    }
    g_events[index] = {ns,
                       object,
                       caller,
                       static_cast<std::uint32_t>(caller2),
                       static_cast<std::uint16_t>(ThreadIndex()),
                       kind,
                       sub};
}

} // namespace

std::uint64_t Now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

void Initialize() {
    const char* env = std::getenv("SOTC_FRAME_TRACE");
    double from = 0, to = 0;
    if (!env || std::sscanf(env, "%lf-%lf", &from, &to) != 2 || to <= from) {
        if (env) {
            LOG_ERROR("Trace", "SOTC_FRAME_TRACE={}: expected <from>-<to> in seconds", env);
        }
        return;
    }
    const std::uint64_t start = Now();
    g_from_ns = start + static_cast<std::uint64_t>(from * 1e9);
    g_to_ns = start + static_cast<std::uint64_t>(to * 1e9);
    // Not value-initialized: the pages are only touched as events fill them.
    g_events.reset(new Event[Capacity]);
    g_enabled = true;
    LOG_INFO("Trace", "frame trace from {} s to {} s", from, to);
}

void WaitBegin(Wait kind, const void* object, std::uint64_t caller, std::uint64_t caller2) {
    Record('B', kind, Now(), reinterpret_cast<std::uint64_t>(object), caller, caller2);
}

void WaitEnd(Wait kind, const void* object) {
    Record('E', kind, Now(), reinterpret_cast<std::uint64_t>(object), 0, 0);
}

void WaitSpan(Wait kind, const void* object, std::uint64_t begin, std::uint64_t end) {
    Record('B', kind, begin, reinterpret_cast<std::uint64_t>(object), 0, 0);
    Record('E', kind, end, reinterpret_cast<std::uint64_t>(object), 0, 0);
}

void Wake(Wait kind, const void* object, std::uint64_t caller) {
    Record('K', kind, Now(), reinterpret_cast<std::uint64_t>(object), caller, 0);
}

void Mark(char kind, std::uint64_t ns, std::uint64_t value) {
    Record('M', static_cast<std::uint8_t>(kind), ns, value, 0, 0);
}

} // namespace diag::trace
