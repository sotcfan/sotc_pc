// TEMP (2026-10-16): finding the game's camera code, for turning the camera with the mouse
// directly (src/input/input.cpp, "native mouse camera" in the notes). Remove once found.
//
// The engine turns the controller into an array of floats, one per "action" (the input bindings
// of the game's data): 0xe2b2e0(float* actions, bindings*, int device) reads each bound axis
// through 0xe2bcf0 (axis 2/3 = right stick, normalized with the dead zone the controller
// reports), clamps and scales it, and stores it at actions[slot]. That array is a local of the
// main loop (0x65c080); each player's is then copied by 0x5c6de0(game manager, actions, player,
// flag) into the game manager, at +0x2041d8 + player * 0x484 (0x480 bytes, 288 floats), where
// the game reads it (0x5c6e20 returns a pointer to it). The right stick's X is in slots 101, 151
// and 157 (2026-10-16). Whoever reads those is the camera.
//
// SOTC_PROBE_CAMERA=1: logs, about once a second, the action array's address and its nonzero
// slots, to see which slots follow the right stick (hold an arrow key).
// SOTC_PROBE_CAMERA=stick: watches the floats the camera keeps the right stick in. The original
// game's code (Team Ico's, "cameraman") reads the stick through 0xbc9a70(float* x, float* y),
// called once per update by the camera's update (0xbb3640) with x = camera + 0x16c8 and
// y = camera + 0x16cc; the watchpoints show who turns them into rotation.
// SOTC_PROBE_CAMERA=<slot>[,<slot>...]: also puts a hardware watchpoint (diag/data_watch.h) on
// those slots (four at most), so the log lists the instructions that read them.

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <format>
#include <string>
#include <vector>

#include "common/log.h"
#include "common/types.h"
#include "diag/call_hooks.h"
#include "diag/camera_probe.h"
#include "diag/data_watch.h"

namespace diag {

namespace {

constexpr u64 StorePlayerActionsAddress = 0x5c6de0;
constexpr u64 ReadRightStickAddress = 0xbc9a70;
// The camera's orientation controller update: (controller, ...) with the right stick in
// xmm0/xmm1. Keeps the last 5 stick samples (averaged), turns them into a target turning speed
// and eases toward it.
constexpr u64 OrientationUpdateAddress = 0xbcf490;
constexpr u64 PlayerActionsOffset = 0x2041d8;
constexpr u64 PlayerActionsStride = 0x484;
constexpr u32 LoggedSlots = 288;

std::vector<int> g_watch_slots;
std::atomic<bool> g_watch_armed{false};

void OnExit(const ObservedCall& call) {
    // The game manager's copy (args: manager, the main loop's array or null, player).
    if (call.args[1] == 0 || call.args[2] != 0) {
        return; // player 0 only
    }
    const auto* actions =
        reinterpret_cast<const float*>(call.args[0] + PlayerActionsOffset);
    // There are several action arrays (one per set of bindings): arm on the first one whose
    // watched slot moves (hold the stick), not simply the first one seen.
    if (!g_watch_slots.empty() && actions[g_watch_slots[0]] != 0.0f &&
        !g_watch_armed.exchange(true)) {
        LOG_INFO("CamProbe", "arming on the action array at {}", static_cast<const void*>(actions));
        for (const int slot : g_watch_slots) {
            WatchData(reinterpret_cast<u64>(actions + slot), 4, false);
        }
    }
    static std::atomic<s64> last_log{0};
    const s64 now = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count();
    s64 last = last_log.load(std::memory_order_relaxed);
    if (now - last < 1000 || !last_log.compare_exchange_strong(last, now)) {
        return;
    }
    std::string slots;
    for (u32 i = 0; i < LoggedSlots; ++i) {
        if (actions[i] != 0.0f) {
            slots += std::format(" [{}]={:.3f}", i, actions[i]);
        }
    }
    LOG_INFO("CamProbe", "player 0 actions {}:{}", static_cast<const void*>(actions),
             slots.empty() ? " (all zero)" : slots);
}

void OnEnter(ObservedCall&) {}

const CallObserver g_observer{OnEnter, OnExit}; // ObserveCalls calls both without a null check

std::atomic<bool> g_stick_armed{false};
void OnStickExit(const ObservedCall& call) {
    // Once the stick moves (hold an arrow key), watch both floats.
    const auto* x = reinterpret_cast<const float*>(call.args[0]);
    if (x && *x != 0.0f && !g_stick_armed.exchange(true)) {
        LOG_INFO("CamProbe", "right stick floats at {:#x} / {:#x} (x = {})", call.args[0],
                 call.args[1], *x);
        WatchData(call.args[0], 4, false);
        WatchData(call.args[1], 4, false);
    }
}
const CallObserver g_stick_observer{OnEnter, OnStickExit};

// SOTC_PROBE_CAMERA=orient: logs the orientation controller's fields every 15 updates.
void OnOrientEnter(ObservedCall& call) {
    call.user[0] = call.float_args[0];
    call.user[1] = call.float_args[1];
}
void OnOrientExit(const ObservedCall& call) {
    static std::atomic<u32> count{0};
    if (count.fetch_add(1, std::memory_order_relaxed) % 15 != 0) {
        return;
    }
    const auto* c = reinterpret_cast<const float*>(call.args[0]);
    float stick_x, stick_y;
    std::memcpy(&stick_x, &call.user[0], 4);
    std::memcpy(&stick_y, &call.user[1], 4);
    std::string fields;
    for (u32 offset = 0x2c; offset <= 0x5c; offset += 4) {
        fields += std::format(" {:x}={:.4f}", offset, c[offset / 4]);
    }
    LOG_INFO("CamProbe", "orient {:#x} in ({:.3f},{:.3f}):{} | f8={:.3f} fc={:.3f} 100={:.3f} "
             "108={:.3f} 10c={:.3f}", call.args[0], stick_x, stick_y, fields, c[0xf8 / 4],
             c[0xfc / 4], c[0x100 / 4], c[0x108 / 4], c[0x10c / 4]);
}
const CallObserver g_orient_observer{OnOrientEnter, OnOrientExit};

// SOTC_PROBE_CAMERA=menu: the menu grid's input handling (0x722780, (grid, update args)), once a
// second: the selected index (+0x190), the items (+0xf0..+0xf8, UI nodes) and the rectangle
// fields of each item's node (+0x1e4 position, +0x1ec size, +0x1f4/+0x1fc offsets, +0x204/+0x20c
// bounds), to see in which space they are.
constexpr u64 MenuGridInputAddress = 0x722780;
void OnMenuExit(const ObservedCall& call) {
    static std::atomic<s64> last{0};
    const s64 now = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count();
    s64 previous = last.load(std::memory_order_relaxed);
    if (now - previous < 1000 || !last.compare_exchange_strong(previous, now)) {
        return;
    }
    const auto* grid = reinterpret_cast<const u8*>(call.args[0]);
    const auto begin = *reinterpret_cast<u8* const* const*>(grid + 0xf0);
    const auto end = *reinterpret_cast<u8* const* const*>(grid + 0xf8);
    const auto* config = *reinterpret_cast<const u8* const*>(grid + 0xa0);
    const auto* update = reinterpret_cast<const u64*>(call.args[1]);
    LOG_INFO("CamProbe", "menu grid {:#x} selected {} of {}, orientation {}, args node {:#x}",
             call.args[0], *reinterpret_cast<const s32*>(grid + 0x190), end - begin,
             *reinterpret_cast<const s32*>(config + 0xec), update[0]);
    // Once per grid: find where a node keeps its children (a field pointing to nodes whose
    // parent, +0x1b0, is that node: a pointer or a vector begin/end), then print the item trees
    // with absolute rectangles (position + offset summed up the parents, size + size offset).
    static std::atomic<u64> dumped{0};
    if (dumped.exchange(call.args[0]) == call.args[0]) {
        return;
    }
    const auto plausible = [](u64 v) {
        return v >= 0x100000000ull && v < 0x800000000ull && v % 8 == 0;
    };
    const auto parent_of = [](const u8* n) { return *reinterpret_cast<const u8* const*>(n + 0x1b0); };
    const auto absolute = [&](const u8* n, float& x, float& y) {
        x = y = 0;
        for (; n; n = parent_of(n)) {
            const auto* f = reinterpret_cast<const float*>(n + 0x1e4);
            x += f[0] + f[4];
            y += f[1] + f[5];
        }
    };
    const u8* item0 = *begin;
    for (u32 off = 0; off < 0x240; off += 8) {
        const u64 v = *reinterpret_cast<const u64*>(item0 + off);
        if (!plausible(v)) {
            continue;
        }
        if (parent_of(reinterpret_cast<const u8*>(v)) == item0) {
            LOG_INFO("CamProbe", "  item 0 +{:#x}: points to a child", off);
        }
        const u64 e = *reinterpret_cast<const u64*>(item0 + off + 8);
        if (e > v && e - v <= 64 * 8 && plausible(*reinterpret_cast<const u64*>(v))) {
            int children = 0;
            for (u64 q = v; q < e; q += 8) {
                const u64 c = *reinterpret_cast<const u64*>(q);
                children += plausible(c) && parent_of(reinterpret_cast<const u8*>(c)) == item0;
            }
            if (children) {
                LOG_INFO("CamProbe", "  item 0 +{:#x}: vector of {} with {} children", off,
                         (e - v) / 8, children);
            }
        }
    }
    // The chain of item 0's ancestors, raw.
    for (const u8* n = item0; n; n = parent_of(n)) {
        const auto* f = reinterpret_cast<const float*>(n + 0x1e4);
        LOG_INFO("CamProbe", "  up {}: pos {:.1f},{:.1f} size {:.1f},{:.1f} off {:.1f},{:.1f} "
                 "soff {:.1f},{:.1f} b {:.1f},{:.1f},{:.1f},{:.1f} s {:.2f} {:#x} flags {:#x}",
                 static_cast<const void*>(n), f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7],
                 f[8], f[9], f[10], f[11], f[12], *reinterpret_cast<const u32*>(n + 0x218),
                 *reinterpret_cast<const u32*>(n + 0x1d8));
    }
    // The children of item 0's parent: a field of the parent that is a vector of nodes whose
    // parent is it.
    {
        const u8* box = parent_of(item0);
        for (u32 off = 0; box && off < 0x240; off += 8) {
            const u64 v = *reinterpret_cast<const u64*>(box + off);
            const u64 e = *reinterpret_cast<const u64*>(box + off + 8);
            if (!plausible(v) || e <= v || e - v > 64 * 8) {
                continue;
            }
            int children = 0;
            for (u64 q = v; q < e; q += 8) {
                const u64 c = *reinterpret_cast<const u64*>(q);
                children += plausible(c) && parent_of(reinterpret_cast<const u8*>(c)) == box;
            }
            if (!children) {
                continue;
            }
            LOG_INFO("CamProbe", "  box +{:#x}: vector of {} with {} children", off, (e - v) / 8,
                     children);
            for (u64 q = v; q < e; q += 8) {
                const auto* c = *reinterpret_cast<const u8* const*>(q);
                float x, y;
                absolute(c, x, y);
                const auto* f = reinterpret_cast<const float*>(c + 0x1e4);
                LOG_INFO("CamProbe", "    child {}: abs {:.1f},{:.1f} size {:.1f},{:.1f} local "
                         "{:.1f},{:.1f} off {:.1f},{:.1f} flags {:#x}{}",
                         static_cast<const void*>(c), x, y, f[2] + f[6], f[3] + f[7], f[0], f[1],
                         f[4], f[5], *reinterpret_cast<const u32*>(c + 0x1d8),
                         c == item0 ? " (the item)" : "");
            }
        }
    }
    // Also: which items have the grid's node (or something under it) as parent.
    for (auto it = begin; it != end && it - begin < 12; ++it) {
        float x, y;
        absolute(*it, x, y);
        const auto* f = reinterpret_cast<const float*>(*it + 0x1e4);
        LOG_INFO("CamProbe", "  item {} {}: abs {:.1f},{:.1f} size {:.1f},{:.1f} parent {} flags "
                 "{:#x}", it - begin, static_cast<const void*>(*it), x, y, f[2] + f[6],
                 f[3] + f[7], static_cast<const void*>(parent_of(*it)),
                 *reinterpret_cast<const u32*>(*it + 0x1d8));
    }
}

const CallObserver g_menu_observer{OnEnter, OnMenuExit};

} // namespace

void InstallCameraProbe() {
    const char* text = std::getenv("SOTC_PROBE_CAMERA");
    if (!text || !*text) {
        return;
    }
    if (std::strcmp(text, "menu") == 0) {
        const u64 addresses[] = {MenuGridInputAddress};
        const auto hooked = ObserveCalls(addresses, &g_menu_observer);
        LOG_INFO("CamProbe", "menu probe {}", hooked[0] ? "on" : "FAILED");
        return;
    }
    if (std::strcmp(text, "orient") == 0) {
        const u64 addresses[] = {OrientationUpdateAddress};
        const auto hooked = ObserveCalls(addresses, &g_orient_observer);
        LOG_INFO("CamProbe", "orientation probe {}", hooked[0] ? "on" : "FAILED");
        return;
    }
    if (std::strcmp(text, "stick") == 0) {
        const u64 addresses[] = {ReadRightStickAddress};
        const auto hooked = ObserveCalls(addresses, &g_stick_observer);
        LOG_INFO("CamProbe", "right stick probe {}", hooked[0] ? "on" : "FAILED");
        return;
    }
    if (std::strcmp(text, "1") != 0) {
        for (const char* p = text; *p;) {
            char* end = nullptr;
            g_watch_slots.push_back(static_cast<int>(std::strtol(p, &end, 10)));
            p = *end == ',' ? end + 1 : end;
            if (end == p && *p) {
                break;
            }
        }
    }
    const u64 addresses[] = {StorePlayerActionsAddress};
    const auto hooked = ObserveCalls(addresses, &g_observer);
    LOG_INFO("CamProbe", "action array probe {}{}", hooked[0] ? "on" : "FAILED",
             g_watch_slots.empty() ? "" : std::format(", watching {} slots", g_watch_slots.size()));
}

} // namespace diag
