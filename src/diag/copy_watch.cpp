#include "diag/copy_watch.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <format>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "common/log.h"
#include "common/types.h"
#include "hle/kernel/gpu_memory.h"

namespace diag {

bool g_watch_copies = false;

namespace {

struct Totals {
    u64 calls = 0;
    u64 bytes = 0;
};

using Key = std::tuple<s32, s32, u64>; // source type, destination type, call site

std::mutex g_mutex;
std::map<Key, Totals> g_sites;
std::map<std::pair<s32, s32>, Totals> g_classes;

std::string TypeName(s32 type) {
    switch (type) {
    case -1: return "other";
    case 0: return "onion";
    case 3: return "garlic";
    case hle::kernel::FlexibleMemoryType: return "flexible";
    default: return std::format("type{}", type);
    }
}

bool IsGpuMemory(s32 type) {
    return type == 0 || type == 3;
}

void Report() {
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(10));
        std::vector<std::pair<Key, Totals>> sites;
        std::map<std::pair<s32, s32>, Totals> classes;
        {
            std::scoped_lock lock{g_mutex};
            sites.assign(g_sites.begin(), g_sites.end());
            classes = g_classes;
        }
        std::string text = "copies so far (calls, MiB) by source -> destination:";
        for (const auto& [types, totals] : classes) {
            text += std::format(" | {} -> {}: {} {:.1f}", TypeName(types.first),
                                TypeName(types.second), totals.calls, totals.bytes / 1048576.0);
        }
        LOG_INFO("Copies", "{}", text);
        std::ranges::sort(sites, [](const auto& a, const auto& b) { return a.second.bytes > b.second.bytes; });
        std::string reads = "reading GPU memory (call site: calls, MiB):";
        size_t shown = 0;
        for (const auto& [key, totals] : sites) {
            if (IsGpuMemory(std::get<0>(key)) && shown++ < 15) {
                reads += std::format(" | {:#x} {}->{}: {} {:.2f}", std::get<2>(key),
                                     TypeName(std::get<0>(key)), TypeName(std::get<1>(key)),
                                     totals.calls, totals.bytes / 1048576.0);
            }
        }
        LOG_INFO("Copies", "{}", reads);
    }
}

} // namespace

void InstallCopyWatch() {
    const char* value = std::getenv("SOTC_WATCH_COPIES");
    if (!value || !*value || *value == '0') {
        return;
    }
    g_watch_copies = true;
    std::thread{Report}.detach();
    LOG_INFO("Copies", "watching memcpy/memmove (SOTC_WATCH_COPIES)");
}

void NoteCopy(void* dst, const void* src, size_t size, const void* caller) {
    const s32 from = hle::kernel::MemoryTypeAt(reinterpret_cast<u64>(src));
    const s32 to = hle::kernel::MemoryTypeAt(reinterpret_cast<u64>(dst));
    std::scoped_lock lock{g_mutex};
    auto& totals = g_classes[{from, to}];
    totals.calls++;
    totals.bytes += size;
    if (IsGpuMemory(from) || IsGpuMemory(to)) {
        auto& site = g_sites[{from, to, reinterpret_cast<u64>(caller)}];
        site.calls++;
        site.bytes += size;
    }
}

} // namespace diag
