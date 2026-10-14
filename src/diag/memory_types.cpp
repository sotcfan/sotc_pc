#include "diag/memory_types.h"

#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <format>
#include <string>
#include <thread>

#include "common/log.h"
#include "hle/kernel/gpu_memory.h"

namespace diag {

const bool g_memory_types_enabled = std::getenv("SOTC_MEMORY_TYPES") != nullptr;

namespace {

bool Enabled() {
    return g_memory_types_enabled;
}

// Columns of the report: the memory types the game uses, then everything else.
enum Column : u32 { Onion, Garlic, Flexible, OtherDirect, Unmapped, NumColumns };
constexpr std::array<const char*, NumColumns> ColumnNames = {"onion", "garlic", "flexible",
                                                             "other", "unmapped"};
constexpr std::array<const char*, static_cast<u32>(MemAccess::Count)> KindNames = {
    "gfx commands", "ce commands", "compute commands", "indirect commands", "SRT reads",
    "labels",       "buffer read",  "buffer written",  "images bound",      "upload reads"};

struct Counter {
    std::atomic<u64> bytes{0};
    std::atomic<u64> count{0};
};
std::array<std::array<Counter, NumColumns>, static_cast<u32>(MemAccess::Count)> g_counters{};

// MemoryTypeAt takes the memory manager's lock: remember the type per 64 KiB page, per thread.
// The tables are dropped at every report (the game remaps memory now and then).
constexpr u32 CachePageBits = 16;
constexpr u32 CacheEntries = 4096;
std::atomic<u32> g_cache_generation{0};

Column ColumnAt(u64 address) {
    struct Entry {
        u64 page = ~u64{0};
        Column column = Unmapped;
    };
    thread_local std::array<Entry, CacheEntries> cache{};
    thread_local u32 generation = ~0u;
    const u32 current = g_cache_generation.load(std::memory_order_relaxed);
    if (generation != current) {
        cache.fill({});
        generation = current;
    }
    const u64 page = address >> CachePageBits;
    Entry& entry = cache[page % CacheEntries];
    if (entry.page != page) {
        const s32 type = hle::kernel::MemoryTypeAt(address);
        entry.page = page;
        entry.column = type == 0                                ? Onion
                       : type == 3                              ? Garlic
                       : type == hle::kernel::FlexibleMemoryType ? Flexible
                       : type < 0                               ? Unmapped
                                                                : OtherDirect;
    }
    return entry.column;
}

// Game mappings: [direct/flexible][log2 size][log2 alignment of the address / physical offset].
constexpr u32 SizeClasses = 33;  // 2^0 .. 2^32 and above
constexpr u32 AlignClasses = 25; // 2^0 .. 2^24 and above
std::array<std::array<std::atomic<u64>, SizeClasses>, 2> g_map_sizes{};
std::array<std::array<std::atomic<u64>, AlignClasses>, 2> g_map_address_align{};
std::array<std::array<std::atomic<u64>, AlignClasses>, 2> g_map_physical_align{};

u32 Log2Class(u64 value, u32 classes) {
    return value ? std::min<u32>(63 - std::countl_zero(value), classes - 1) : 0;
}
u32 AlignClass(u64 value, u32 classes) {
    return value ? std::min<u32>(std::countr_zero(value), classes - 1) : classes - 1;
}

std::string Histogram(const auto& counts, const char* unit_suffix) {
    std::string text;
    for (u32 i = 0; i < counts.size(); ++i) {
        if (const u64 n = counts[i].load(std::memory_order_relaxed)) {
            const u64 value = u64{1} << i;
            text += value >= 1024 * 1024 ? std::format(" {}M{}:{}", value >> 20, unit_suffix, n)
                    : value >= 1024      ? std::format(" {}K{}:{}", value >> 10, unit_suffix, n)
                                         : std::format(" {}{}:{}", value, unit_suffix, n);
        }
    }
    return text.empty() ? " -" : text;
}

void ReportLoop() {
    constexpr auto Interval = std::chrono::seconds{10};
    for (;;) {
        std::this_thread::sleep_for(Interval);
        g_cache_generation.fetch_add(1, std::memory_order_relaxed);
        LOG_INFO("MemTypes", "GPU layer accesses by memory type, last {} s (MB / count):",
                 Interval.count());
        for (u32 kind = 0; kind < static_cast<u32>(MemAccess::Count); ++kind) {
            std::string line;
            for (u32 column = 0; column < NumColumns; ++column) {
                Counter& counter = g_counters[kind][column];
                const u64 bytes = counter.bytes.exchange(0, std::memory_order_relaxed);
                const u64 count = counter.count.exchange(0, std::memory_order_relaxed);
                if (count) {
                    line += std::format("  {} {:.2f}/{}", ColumnNames[column],
                                        static_cast<double>(bytes) / 1e6, count);
                }
            }
            if (!line.empty()) {
                LOG_INFO("MemTypes", "  {}:{}", KindNames[kind], line);
            }
        }
        for (u32 flexible = 0; flexible < 2; ++flexible) {
            const char* name = flexible ? "flexible" : "direct";
            LOG_INFO("MemTypes", "  {} maps (since start), size:{}", name,
                     Histogram(g_map_sizes[flexible], ""));
            LOG_INFO("MemTypes", "  {} maps, address alignment:{}", name,
                     Histogram(g_map_address_align[flexible], ""));
            LOG_INFO("MemTypes", "  {} maps, physical alignment:{}", name,
                     Histogram(g_map_physical_align[flexible], ""));
        }
    }
}

} // namespace

bool MemoryTypesEnabled() {
    return Enabled();
}

void NoteMemoryAccessSlow(MemAccess kind, u64 address, u64 bytes) {
    if (!Enabled() || kind >= MemAccess::Count) {
        return;
    }
    Counter& counter = g_counters[static_cast<u32>(kind)][ColumnAt(address)];
    counter.bytes.fetch_add(bytes, std::memory_order_relaxed);
    counter.count.fetch_add(1, std::memory_order_relaxed);
}

void NoteGameMapping(u64 address, u64 size, u64 physical, bool flexible) {
    if (!Enabled()) {
        return;
    }
    const u32 f = flexible ? 1 : 0;
    g_map_sizes[f][Log2Class(size, SizeClasses)].fetch_add(1, std::memory_order_relaxed);
    g_map_address_align[f][AlignClass(address, AlignClasses)].fetch_add(1,
                                                                        std::memory_order_relaxed);
    g_map_physical_align[f][AlignClass(physical, AlignClasses)].fetch_add(
        1, std::memory_order_relaxed);
}

void InstallMemoryTypes() {
    if (!Enabled()) {
        return;
    }
    LOG_INFO("MemTypes", "counting GPU layer accesses by memory type (SOTC_MEMORY_TYPES)");
    std::thread{ReportLoop}.detach();
}

} // namespace diag
