// Glue between sotcport and the parts of shadPS4 it reuses for the GPU (see
// third_party/shadps4/README.md). Compiled with shadPS4's include paths; sotcport's own
// headers used here (src/hle, src/loader, src/gpu) only depend on standard types.

// XXH3_state_t on the stack (HashBacking runs inside a fault handler: no allocation).
#define XXH_STATIC_LINKING_ONLY
#include <magic_enum/magic_enum.hpp>
#include <xxhash.h>

#include <SDL3/SDL.h>
#include "host_module.h"
#include "common/thread.h"
#include <x86intrin.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <tuple>
#include <mutex>
#include <string>
#include <thread>
#include <vector>


#include "common/logging/log.h"
#include "common/path_util.h"
#include "core/address_space.h"
#include "core/libraries/kernel/process.h"
#include "core/libraries/kernel/time.h"
#include "core/memory.h"
#include "core/signals.h"
#include "gpu/gpu.h"
#include "hle/kernel/gpu_memory.h"
#include "input/input.h"
#include "loader/crash.h"
#include "render/shader_translate.h"
#include "perf_temp.h"
#include "sdl_window.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/amdgpu/packet_observer.h"
#include "video_core/amdgpu/pm4_opcodes.h"
#include "video_core/renderdoc.h"
#include "video_core/renderer_vulkan/vk_draw_pipe.h"
#include "video_core/renderer_vulkan/vk_presenter.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_runtime.h"

namespace Core::Loader {
class SymbolsResolver;
}

// Used by shadPS4's GnmDriver to create its presenter.
Frontend::WindowSDL* g_window = nullptr;

namespace Libraries::GnmDriver {
void RegisterLib(Core::Loader::SymbolsResolver* sym);
}
namespace Libraries::VideoOut {
void RegisterLib(Core::Loader::SymbolsResolver* sym);
}

namespace {

int g_sdk_version = 0;
const auto g_start = std::chrono::steady_clock::now();

} // namespace

// ---- Window ----

namespace Frontend {

WindowSDL::WindowSDL(s32 width_, s32 height_, std::string_view title)
    : width{width_}, height{height_} {
    const std::string name(title);
    window = SDL_CreateWindow(name.c_str(), width, height, SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
    if (!window) {
        LOG_CRITICAL(Frontend, "could not create the window: {}", SDL_GetError());
        return;
    }
    const SDL_PropertiesID props = SDL_GetWindowProperties(window);
    if (std::strcmp(SDL_GetCurrentVideoDriver(), "x11") == 0) {
        info.type = WindowSystemType::X11;
        info.display_connection = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_X11_DISPLAY_POINTER, nullptr);
        info.render_surface = reinterpret_cast<void*>(SDL_GetNumberProperty(props, SDL_PROP_WINDOW_X11_WINDOW_NUMBER, 0));
    } else {
        info.type = WindowSystemType::Wayland;
        info.display_connection = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WAYLAND_DISPLAY_POINTER, nullptr);
        info.render_surface = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WAYLAND_SURFACE_POINTER, nullptr);
    }
}

WindowSDL::~WindowSDL() {
    if (window) {
        SDL_DestroyWindow(window);
    }
}

bool WindowSDL::ProcessEvents(int timeout_ms) {
    SDL_Event event;
    // The mouse's mode (captured, or a cursor in menus) follows the game's state, not only
    // events: checked at every turn of the loop.
    input::PollWindow(window);
    if (!SDL_WaitEventTimeout(&event, timeout_ms)) {
        return true;
    }
    do {
        input::HandleEvent(event);
        switch (event.type) {
        case SDL_EVENT_QUIT:
        case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
            return false;
        case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
            SDL_GetWindowSizeInPixels(window, &width, &height);
            break;
        default:
            break;
        }
    } while (SDL_PollEvent(&event));
    return true;
}

} // namespace Frontend

// ---- Signals: chained into sotcport's crash handler ----

// The GPU layer's command processor (core/libraries/gnmdriver/gnmdriver.cpp).
extern std::unique_ptr<AmdGpu::Liverpool> liverpool;

namespace Core {

Signals* Signals::Instance() {
    static Signals instance;
    return &instance;
}

void Signals::RegisterAccessViolationHandler(const AccessViolationHandler& handler, u32 priority) {
    crash::AddAccessFaultHandler(handler, priority);
}

void Signals::RegisterIllegalInstructionHandler(const IllegalInstructionHandler& handler,
                                                u32 priority) {
    crash::AddIllegalInstructionHandler(handler, priority);
}

// ---- Memory ----

VAddr AddressSpace::SystemReservedVirtualBase() noexcept {
    return hle::kernel::SystemReservedBase;
}

u64 AddressSpace::SystemReservedVirtualSize() const noexcept {
    return hle::kernel::SystemReservedSize;
}

void AddressSpace::Protect(VAddr virtual_addr, u64 size, MemoryPermission perms) {
    hle::kernel::ProtectForTracking(virtual_addr, size, True(perms & MemoryPermission::Read),
                                    True(perms & MemoryPermission::Write));
}

boost::icl::interval_set<VAddr> AddressSpace::GetUsableRegions() {
    boost::icl::interval_set<VAddr> regions;
    for (const auto& [start, end] : hle::kernel::MappedRanges()) {
        regions += boost::icl::interval<VAddr>::right_open(start, end);
    }
    return regions;
}

namespace {

// Tells the rasterizer about the game's mappings, as shadPS4's memory manager does.
class RasterizerObserver final : public hle::kernel::GpuMemoryObserver {
public:
    explicit RasterizerObserver(Vulkan::Rasterizer* rasterizer_) : rasterizer{rasterizer_} {}

    void OnMapped(u64 address, u64 size) override {
        perf::NoteMapEvent('M', address, size); // sotcport (SOTC_VERIFY_IMAGES)
        rasterizer->MapMemory(address, size);
        rasterizer->RegisterMemory(address, size);
        // The game streams in the large mips of a texture by mapping memory into the range it
        // reserved for it, under an image the GPU already has. The new mapping comes without
        // the tracking protection, so its contents and later writes would go unnoticed: mark
        // what is there as modified, to be uploaded (and tracked) again on its next use.
        rasterizer->InvalidateMemory(address, size);
    }
    void OnUnmapped(u64 address, u64 size) override {
        perf::NoteMapEvent('U', address, size); // sotcport (SOTC_VERIFY_IMAGES)
        // Unmapping frees the images and buffers in the range. A game thread doing that while
        // the GPU threads use them (the command thread, or stage B of the two-stage draw
        // pipeline) is a use after free: shadPS4 and bbport do it that way, and it can corrupt
        // the heap. The GPU command thread runs it instead, between packets and after stage B
        // has run dry. SOTC_UNMAP_ON_GUEST_THREAD=1 keeps the old way (diagnostics).
        static const bool on_guest_thread = std::getenv("SOTC_UNMAP_ON_GUEST_THREAD") != nullptr;
        if (on_guest_thread || !liverpool || rasterizer->IsGpuSideThread()) {
            rasterizer->UnmapMemory(address, size);
            return;
        }
        const uint64_t start = perf::Enabled() ? perf::Now() : 0; // sotcport TEMP
        liverpool->SendCommand<true>([&] { rasterizer->UnmapMemory(address, size); });
        if (start) {
            perf::AddWait("unmap (waits for GpuComm)", perf::Now() - start); // sotcport TEMP
        }
    }
    void OnProtectionChanged(u64 address, u64 size) override {
        perf::NoteMapEvent('P', address, size); // sotcport (SOTC_VERIFY_IMAGES)
        rasterizer->RegisterMemory(address, size);
    }

private:
    Vulkan::Rasterizer* rasterizer;
};

constexpr u64 TrackPage = 16 * 1024;

} // namespace

// The rasterizer, for gpu::SyncGuestRange.
Vulkan::Rasterizer* g_rasterizer = nullptr;

void MemoryManager::SetRasterizer(Vulkan::Rasterizer* rasterizer) {
    g_rasterizer = rasterizer;
    static RasterizerObserver* observer = nullptr;
    delete observer;
    observer = rasterizer ? new RasterizerObserver(rasterizer) : nullptr;
    hle::kernel::SetGpuMemoryObserver(observer);
}

u64 MemoryManager::ClampRangeSize(VAddr virtual_addr, u64 size) {
    return hle::kernel::ClampToMapped(virtual_addr, size);
}

void MemoryManager::CopySparseMemory(VAddr source, u8* dest, u64 size) {
    // Whole runs of mapped memory at once (one lookup each); unmapped pages read as zeros.
    while (size) {
        if (const u64 mapped = hle::kernel::ClampToMapped(source, size)) {
            // The mirror covers the GPU's 40-bit range (hle::kernel::GuestMirrorOffset).
            if (const u8* backing = hle::kernel::BackingAddress(source);
                backing && hle::kernel::GuestMirrorOffset() &&
                source + mapped <= 0x100'0000'0000ULL) {
                std::memcpy(dest, backing, mapped); // the mirror is contiguous like the game's view
                source += mapped;
                dest += mapped;
                size -= mapped;
                continue;
            }
        }
        const u64 copy = std::min<u64>(TrackPage - source % TrackPage, size);
        if (const u8* backing = hle::kernel::BackingAddress(source)) {
            std::memcpy(dest, backing, copy);
        } else {
            std::memset(dest, 0, copy);
        }
        source += copy;
        dest += copy;
        size -= copy;
    }
}

bool MemoryManager::TryWriteBacking(void* address, const void* data, u64 size) {
    VAddr target = reinterpret_cast<VAddr>(address);
    const auto* source = static_cast<const u8*>(data);
    if (!hle::kernel::BackingAddress(target)) {
        return false;
    }
    while (size) {
        const u64 copy = std::min<u64>(TrackPage - target % TrackPage, size);
        if (u8* backing = hle::kernel::BackingAddress(target)) {
            std::memcpy(backing, source, copy);
        }
        target += copy;
        source += copy;
        size -= copy;
    }
    return true;
}

u64 MemoryManager::HashBacking(VAddr address, u64 size) {
    static constexpr u8 Zeros[TrackPage] = {};
    XXH3_state_t state;
    XXH3_64bits_reset(&state);
    while (size) {
        const u64 chunk = std::min<u64>(TrackPage - address % TrackPage, size);
        const u8* backing = hle::kernel::BackingAddress(address);
        XXH3_64bits_update(&state, backing ? backing : Zeros, chunk);
        address += chunk;
        size -= chunk;
    }
    return XXH3_64bits_digest(&state);
}

} // namespace Core

// ---- libkernel functions called by shadPS4's code ----

namespace Libraries::Kernel {

// sotcport: the same answer the game gets (src/hle/kernel/process.cpp, SOTC_NEO_MODE).
s32 PS4_SYSV_ABI sceKernelIsNeoMode() {
    static const s32 neo = [] {
        const char* value = std::getenv("SOTC_NEO_MODE");
        return value && *value == '1' ? 1 : 0;
    }();
    return neo;
}

u64 PS4_SYSV_ABI sceKernelReadTsc() {
    return __rdtsc();
}

u64 PS4_SYSV_ABI sceKernelGetTscFrequency() {
    static const u64 frequency = [] {
        const auto start_time = std::chrono::steady_clock::now();
        const u64 start_tsc = __rdtsc();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
        return static_cast<u64>((__rdtsc() - start_tsc) / seconds);
    }();
    return frequency;
}

u64 PS4_SYSV_ABI sceKernelGetProcessTime() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() -
                                                                 g_start)
        .count();
}

s32 PS4_SYSV_ABI sceKernelGetCompiledSdkVersion(s32* version) {
    if (!version) {
        return -1;
    }
    *version = g_sdk_version;
    return 0;
}

} // namespace Libraries::Kernel

// ---- sotcport TEMP: performance counters (perf_temp.h) ----

namespace perf {

namespace {

// SOTC_SPIKES=<ms>: log what every thread waited on during each frame that took longer than
// <ms> between flips (see NoteFrameForSpikes). It needs the counters, so it turns them on too.
const double g_spike_ms = [] {
    const char* env = std::getenv("SOTC_SPIKES");
    return env ? std::max(1.0, std::atof(env)) : 0.0;
}();
const bool g_enabled = std::getenv("SOTC_PERF") != nullptr || g_spike_ms > 0;
std::atomic<uint64_t> g_count[NumSlots];
std::atomic<uint64_t> g_ns[NumSlots];
std::mutex g_mutex;
std::map<std::string, std::pair<uint64_t, uint64_t>> g_waits;                   // count, ns
// (64 KiB region, is_write, game code address of the fault) -> count, bytes
std::map<std::tuple<uint64_t, bool, uint64_t>, std::pair<uint64_t, uint64_t>> g_readbacks;
std::atomic<uint64_t> g_walker_runs, g_walker_reads, g_walker_fetches, g_walker_patched,
    g_walker_patches, g_shadow_copies, g_shadow_bytes;
// The same counters for the frame being made (SOTC_SPIKES); reset at every flip.
std::atomic<uint64_t> g_frame_count[NumSlots];
std::atomic<uint64_t> g_frame_ns[NumSlots];
std::map<std::string, std::pair<uint64_t, uint64_t>> g_frame_waits; // under g_mutex
std::atomic<uint64_t> g_download_bytes, g_download_changed, g_downloads, g_downloads_changed;
constexpr const char* SlotNames[NumSlots] = {
    "GpuComm busy", "Draw", "Dispatch", "Readback", "Scheduler::Wait", "New pipeline",
    "OnSubmit+Flush", "Invalidate (fault)", "Flip", "Wait: download", "Wait: stream buffer",
    "Wait: fault buffer", "Wait: image download", "Empty download", "Texture upload",
    "Barrier", "Rendering end",
};

// The `limit` largest entries of `entries` (sorted by their first member), joined.
std::string Largest(std::vector<std::pair<uint64_t, std::string>> entries, size_t limit) {
    std::ranges::sort(entries, std::greater{});
    std::string text;
    for (size_t i = 0; i < entries.size() && i < limit; i++) {
        text += " | " + entries[i].second;
    }
    return text;
}

} // namespace

bool Enabled() {
    return g_enabled;
}

uint64_t Now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

void Add(Slot slot, uint64_t ns) {
    g_count[slot].fetch_add(1, std::memory_order_relaxed);
    g_ns[slot].fetch_add(ns, std::memory_order_relaxed);
    if (g_spike_ms > 0) {
        g_frame_count[slot].fetch_add(1, std::memory_order_relaxed);
        g_frame_ns[slot].fetch_add(ns, std::memory_order_relaxed);
    }
    if (slot == Readback || slot == SchedulerWait || slot == Invalidate) {
        AddWait(SlotNames[slot], ns);
    }
}

void AddWait(const char* kind, uint64_t ns) {
    thread_local char name[32] = {};
    if (!name[0]) {
        const std::string thread_name = Common::GetCurrentThreadName();
        std::snprintf(name, sizeof(name), "%s", thread_name.c_str());
    }
    std::scoped_lock lock{g_mutex};
    auto& entry = g_waits[std::string{kind} + " / " + name];
    entry.first++;
    entry.second += ns;
    if (g_spike_ms > 0) {
        auto& frame_entry = g_frame_waits[std::string{kind} + " / " + name];
        frame_entry.first++;
        frame_entry.second += ns;
    }
}

thread_local uint64_t t_fault_site = 0;

std::atomic<uint64_t> g_srt_readbacks, g_srt_readbacks_changed;
std::map<uint64_t, std::pair<uint64_t, uint64_t>> g_srt_regions; // 64 KiB region -> total, changed

void NoteSrtReadback(uint64_t address, bool changed) {
    if (!g_enabled) {
        return;
    }
    g_srt_readbacks++;
    g_srt_readbacks_changed += changed;
    std::scoped_lock lock{g_mutex};
    auto& entry = g_srt_regions[address >> 16];
    entry.first++;
    entry.second += changed;
}

std::map<uint64_t, std::array<uint64_t, 3>> g_waitregmem; // address -> met, unmet, last ref

void NoteWaitRegMem(uint64_t address, bool met, bool equal, uint32_t ref) {
    if (!g_enabled) {
        return;
    }
    std::scoped_lock lock{g_mutex};
    auto& entry = g_waitregmem[address];
    entry[met ? 0 : 1]++;
    entry[2] = ref | (uint64_t{equal} << 32);
}

std::vector<uint32_t> g_frame_intervals_us; // under g_mutex

void NoteFrameInterval(uint64_t ns) {
    if (!g_enabled) {
        return;
    }
    std::scoped_lock lock{g_mutex};
    g_frame_intervals_us.push_back(static_cast<uint32_t>(ns / 1000));
}

void SetFaultSite(uint64_t game_address) {
    t_fault_site = game_address;
}

void NoteReadback(uint64_t address, uint64_t size, bool is_write) {
    if (!g_enabled) {
        return;
    }
    std::scoped_lock lock{g_mutex};
    auto& entry = g_readbacks[{address >> 16, is_write, t_fault_site}];
    entry.first++;
    entry.second += size;
}

void NoteSrtReads(uint64_t reads, bool fetched) {
    if (!g_enabled) {
        return;
    }
    g_walker_runs++;
    g_walker_reads += reads;
    g_walker_fetches += fetched;
}

void NoteShadowCopy(uint64_t bytes) {
    if (!g_enabled) {
        return;
    }
    g_shadow_copies++;
    g_shadow_bytes += bytes;
}

void NoteSrtPatches(uint64_t patches) {
    if (!g_enabled || !patches) {
        return;
    }
    g_walker_patched++;
    g_walker_patches += patches;
}

namespace {
struct MapEvent {
    std::atomic<uint64_t> address{0};
    std::atomic<uint64_t> size{0};
    std::atomic<uint64_t> time{0};
    std::atomic<char> kind{0};
};
constexpr uint32_t MapEventCount = 8192;
std::array<MapEvent, MapEventCount> g_map_events;
std::atomic<uint32_t> g_map_event_next{0};
} // namespace

void NoteMapEvent(char kind, uint64_t address, uint64_t size) {
    static const bool enabled = std::getenv("SOTC_VERIFY_IMAGES") != nullptr;
    if (!enabled) {
        return;
    }
    MapEvent& event = g_map_events[g_map_event_next.fetch_add(1) % MapEventCount];
    event.address = address;
    event.size = size;
    event.time = Now();
    event.kind = kind;
}

std::string DescribeMapEvents(uint64_t begin, uint64_t end) {
    const uint64_t now = Now();
    std::string text;
    for (const MapEvent& event : g_map_events) {
        const uint64_t address = event.address;
        const uint64_t size = event.size;
        if (event.kind && address < end && begin < address + size) {
            text += fmt::format(" {}:{:#x}-{:#x}@-{}ms", event.kind.load(), address,
                                address + size, (now - event.time) / 1000000);
        }
    }
    return text.empty() ? " none" : text;
}

namespace {
std::array<MapEvent, MapEventCount> g_label_writes;
std::atomic<uint32_t> g_label_write_next{0};
} // namespace

void NoteLabelWrite(uint64_t address, uint64_t size) {
    static const bool enabled = std::getenv("SOTC_VERIFY_SHADOW_COPY") != nullptr;
    if (!enabled) {
        return;
    }
    MapEvent& event = g_label_writes[g_label_write_next.fetch_add(1) % MapEventCount];
    event.address = address;
    event.size = size;
    event.time = Now();
    event.kind = 'L';
}

std::string DescribeLabelWrites(uint64_t begin, uint64_t end) {
    const uint64_t now = Now();
    std::string text;
    u32 shown = 0;
    for (const MapEvent& event : g_label_writes) {
        const uint64_t address = event.address;
        const uint64_t size = event.size;
        if (event.kind && address < end && begin < address + size && shown++ < 8) {
            text += fmt::format(" {:#x}+{}@-{}ms", address, size, (now - event.time) / 1000000);
        }
    }
    return text.empty() ? " none" : text;
}

void NoteDownload(uint64_t address, const uint8_t* data, uint64_t size) {
    if (!g_enabled) {
        return;
    }
    uint64_t changed = 0;
    for (uint64_t done = 0; done < size;) {
        const uint64_t chunk =
            std::min<uint64_t>(Core::TrackPage - (address + done) % Core::TrackPage, size - done);
        const uint8_t* backing = hle::kernel::BackingAddress(address + done);
        // Counted per 64-byte block (a byte loop over ~140 MB/s of downloads took ~6% of the
        // GPU command thread and skewed the benchmarks that enable these counters).
        for (uint64_t i = 0; i < chunk; i += 64) {
            const uint64_t n = std::min<uint64_t>(64, chunk - i);
            if (!backing || std::memcmp(backing + i, data + done + i, n) != 0) {
                changed += n;
            }
        }
        done += chunk;
    }
    g_download_bytes += size;
    g_download_changed += changed;
    g_downloads++;
    g_downloads_changed += changed != 0;
}

// Last values from NoteVram (bytes), for the PERF report.
std::atomic<uint64_t> g_vram_used{0}, g_vram_pressure{0}, g_vram_critical{0};

void NoteVram(uint64_t used, uint64_t pressure, uint64_t critical) {
    g_vram_used.store(used, std::memory_order_relaxed);
    g_vram_pressure.store(pressure, std::memory_order_relaxed);
    g_vram_critical.store(critical, std::memory_order_relaxed);
}

// TEMP timeline (see perf::Mark).
struct TimelineEvent {
    uint64_t ns;
    char kind;
    uint64_t value;
};
constexpr size_t TimelineSize = size_t{1} << 16;
std::array<TimelineEvent, TimelineSize> g_timeline{};
std::atomic<uint64_t> g_timeline_next{0};
const int g_timeline_every = [] {
    const char* env = std::getenv("SOTC_FRAME_TIMELINE");
    return env ? std::max(1, std::atoi(env)) : 0;
}();

void Mark(char kind, uint64_t value) {
    if (!g_timeline_every) {
        return;
    }
    const uint64_t index = g_timeline_next.fetch_add(1, std::memory_order_relaxed);
    g_timeline[index % TimelineSize] = {Now(), kind, value};
    if (kind != 'P') {
        return;
    }
    static uint64_t last_present = 0, frames = 0;
    if (++frames % g_timeline_every == 0 && last_present && index - last_present < TimelineSize) {
        const uint64_t t0 = g_timeline[last_present % TimelineSize].ns;
        std::string text;
        for (uint64_t i = last_present; i <= index; ++i) {
            const auto& event = g_timeline[i % TimelineSize];
            text += fmt::format(" {:.2f}{}", (event.ns - t0) / 1e6, event.kind);
            if (event.value) {
                text += fmt::format("({})", event.value);
            }
        }
        LOG_INFO(Render, "TIMELINE frame {} ({:.2f} ms):{}", frames,
                 (g_timeline[index % TimelineSize].ns - t0) / 1e6, text);
    }
    last_present = index;
}

void NoteFrameForSpikes() {
    // Called at every flip (presentation thread). A frame here is the time between two flips.
    static uint64_t last = 0, frames = 0;
    static uint64_t last_drain = 0;
    if (g_spike_ms <= 0) {
        return;
    }
    const uint64_t now = Now();
    const uint64_t drain = Vulkan::DrawPipe::drain_cycles_total.load(std::memory_order_relaxed);
    const double frame_ms = last ? (now - last) / 1e6 : 0.0;
    ++frames;
    if (last && frame_ms > g_spike_ms) {
        // GPU command thread work by kind, then the waits of every thread, largest first.
        std::vector<std::pair<uint64_t, std::string>> slots;
        for (int i = 0; i < NumSlots; i++) {
            const uint64_t ns = g_frame_ns[i].load(std::memory_order_relaxed);
            if (i != GpuCommBusy && ns >= 500'000) {
                slots.emplace_back(ns, fmt::format("{} {}x {:.1f} ms", SlotNames[i],
                                                   g_frame_count[i].load(), ns / 1e6));
            }
        }
        std::vector<std::pair<uint64_t, std::string>> waits;
        {
            std::scoped_lock lock{g_mutex};
            for (const auto& [key, value] : g_frame_waits) {
                if (value.second >= 1'000'000) {
                    waits.emplace_back(value.second, fmt::format("{} {}x {:.1f} ms", key,
                                                                 value.first, value.second / 1e6));
                }
            }
        }
        static const double tsc_per_ms = Libraries::Kernel::sceKernelGetTscFrequency() / 1e3;
        LOG_WARNING(Render, "SPIKE frame {} took {:.1f} ms; draw pipe drains waited {:.1f} ms; "
                            "GPU thread:{}; waits:{}",
                    frames, frame_ms, (drain - last_drain) / tsc_per_ms,
                    Largest(std::move(slots), 8), Largest(std::move(waits), 10));
    }
    for (int i = 0; i < NumSlots; i++) {
        g_frame_count[i].store(0, std::memory_order_relaxed);
        g_frame_ns[i].store(0, std::memory_order_relaxed);
    }
    {
        std::scoped_lock lock{g_mutex};
        g_frame_waits.clear();
    }
    last = now;
    last_drain = drain;
}

void Report() {
    static uint64_t last = Now();
    const uint64_t now = Now();
    if (!g_enabled || now - last < 5'000'000'000) {
        return;
    }
    const double seconds = (now - last) / 1e9;
    last = now;
    std::string text = "PERF:";
    for (int i = 0; i < NumSlots; i++) {
        const uint64_t count = g_count[i].exchange(0);
        const uint64_t ns = g_ns[i].exchange(0);
        text += fmt::format(" | {} {:.0f}/s {:.1f} ms/s", SlotNames[i], count / seconds,
                            ns / 1e6 / seconds);
    }
    LOG_INFO(Render, "{}", text);
    LOG_INFO(Render,
             "PERF downloads: {:.0f}/s, {:.0f}/s with some change; {:.0f} KB/s downloaded, "
             "{:.1f} KB/s changed",
             g_downloads.exchange(0) / seconds, g_downloads_changed.exchange(0) / seconds,
             g_download_bytes.exchange(0) / 1024.0 / seconds,
             g_download_changed.exchange(0) / 1024.0 / seconds);
    LOG_INFO(Render,
             "PERF walker: {:.0f} runs/s, {:.0f} reads/s, {:.0f}/s needed a readback, {:.0f}/s "
             "left GPU-written data to the GPU ({:.0f} dwords/s); {:.0f} copy dispatches/s "
             "computed on the CPU too ({:.0f} KB/s)",
             g_walker_runs.exchange(0) / seconds, g_walker_reads.exchange(0) / seconds,
             g_walker_fetches.exchange(0) / seconds, g_walker_patched.exchange(0) / seconds,
             g_walker_patches.exchange(0) / seconds, g_shadow_copies.exchange(0) / seconds,
             g_shadow_bytes.exchange(0) / 1024.0 / seconds);
    // VRAM: when it fills, the driver moves memory to system RAM and the GPU slows down a lot.
    LOG_INFO(Render, "PERF VRAM: {} MB used; texture GC pressured at {} MB, critical at {} MB",
             g_vram_used.load() >> 20, g_vram_pressure.load() >> 20, g_vram_critical.load() >> 20);
    // Threaded recording: CommandBuffer() calls that had to wait for the recording thread
    // (code not converted to Scheduler::Record).
    LOG_INFO(Render, "PERF recording: {:.0f} direct recordings/s",
             Vulkan::Scheduler::direct_recordings.exchange(0) / seconds);
    // Two-stage draw pipeline (SOTC_DRAW_PIPE=1): TSC cycles converted with the rate measured
    // between two reports.
    static uint64_t last_tsc = __rdtsc();
    const uint64_t tsc = __rdtsc();
    const double tsc_per_ms = (tsc - last_tsc) / (seconds * 1e3);
    last_tsc = tsc;
    LOG_INFO(Render,
             "PERF draw pipe: {:.0f} packets/s, {:.0f} drains/s waiting {:.1f} ms/s, "
             "stage B busy {:.1f} ms/s",
             Vulkan::DrawPipe::packets.exchange(0) / seconds,
             Vulkan::DrawPipe::drains.exchange(0) / seconds,
             Vulkan::DrawPipe::drain_cycles.exchange(0) / tsc_per_ms / seconds,
             Vulkan::DrawPipe::busy_cycles.exchange(0) / tsc_per_ms / seconds);
    {
        using Vulkan::DrawPipe;
        std::vector<std::pair<uint64_t, std::string>> reasons;
        for (u32 i = 0; i < DrawPipe::NumReasons; i++) {
            const uint64_t count = DrawPipe::drains_by_reason[i].exchange(0);
            const uint64_t cycles = DrawPipe::cycles_by_reason[i].exchange(0);
            if (!count) {
                continue;
            }
            const char* name = DrawPipe::ReasonName(i);
            std::string label = name ? name : "";
            if (!name) {
                const u32 opcode = i % DrawPipe::ReasonComputeOpcode;
                label = i >= DrawPipe::ReasonComputeOpcode ? "compute " : "";
                label += magic_enum::enum_name(static_cast<AmdGpu::PM4ItOpcode>(opcode));
                label += fmt::format("({:#x})", opcode);
            }
            reasons.emplace_back(cycles, fmt::format("{} {:.0f}/s {:.1f} ms/s", label,
                                                     count / seconds,
                                                     cycles / tsc_per_ms / seconds));
        }
        if (!reasons.empty()) {
            LOG_INFO(Render, "PERF drains by reason:{}", Largest(std::move(reasons), 12));
        }
    }
    std::scoped_lock lock{g_mutex};
    std::vector<std::pair<uint64_t, std::string>> regions;
    // Sites outside the game executable are shown relative to sotcport's own image (addr2line).
    static const uintptr_t host_base = [] {
        return sotc_host::ModuleBaseOf(reinterpret_cast<void*>(&Report));
    }();
    for (const auto& [key, value] : g_readbacks) {
        const auto& [region, is_write, site] = key;
        const std::string where = site < 0x4000000 ? fmt::format("{:#x}", site)
                                                   : fmt::format("host+{:#x}", site - host_base);
        regions.emplace_back(value.first, fmt::format("{:#x}{} from {} {:.0f}/s ~{} B",
                                                      region << 16, is_write ? " WRITE" : " read",
                                                      where, value.first / seconds,
                                                      value.second / value.first));
    }
    {
        std::vector<std::pair<uint64_t, std::string>> srt;
        for (const auto& [region, value] : g_srt_regions) {
            srt.emplace_back(value.first, fmt::format("{:#x} {:.0f}/s changed {:.0f}/s",
                                                      region << 16, value.first / seconds,
                                                      value.second / seconds));
        }
        g_srt_regions.clear();
        LOG_INFO(Render, "PERF SRT readbacks: {:.0f}/s, {:.0f}/s changed the data:{}",
                 g_srt_readbacks.exchange(0) / seconds,
                 g_srt_readbacks_changed.exchange(0) / seconds, Largest(std::move(srt), 6));
    }
    {
        std::vector<std::pair<uint64_t, std::string>> waits;
        for (const auto& [address, value] : g_waitregmem) {
            waits.emplace_back(value[0] + value[1],
                               fmt::format("{:#x} met {:.0f}/s unmet {:.0f}/s ref {:#x}{}",
                                           address, value[0] / seconds, value[1] / seconds,
                                           u32(value[2]), value[2] >> 32 ? " (==)" : ""));
        }
        g_waitregmem.clear();
        LOG_INFO(Render, "PERF WaitRegMem drains:{}", Largest(std::move(waits), 8));
    }
    if (!g_frame_intervals_us.empty()) {
        auto& v = g_frame_intervals_us;
        std::ranges::sort(v);
        double mean = 0, var = 0;
        for (const u32 x : v) {
            mean += x;
        }
        mean /= v.size();
        for (const u32 x : v) {
            var += (x - mean) * (x - mean);
        }
        LOG_INFO(Render,
                 "PERF frame pacing: {} frames, mean {:.2f} ms, stdev {:.2f} ms, p1 {:.1f} "
                 "median {:.1f} p99 {:.1f} max {:.1f} ms",
                 v.size(), mean / 1000, std::sqrt(var / v.size()) / 1000, v[v.size() / 100] / 1000.0,
                 v[v.size() / 2] / 1000.0, v[v.size() * 99 / 100] / 1000.0, v.back() / 1000.0);
        v.clear();
    }
    LOG_INFO(Render, "PERF readbacks ({} 64K regions):{}", regions.size(),
             Largest(std::move(regions), 15));
    g_readbacks.clear();
    std::vector<std::pair<uint64_t, std::string>> waits;
    for (const auto& [key, value] : g_waits) {
        waits.emplace_back(value.second, fmt::format("{} {:.0f}/s {:.0f} ms/s", key,
                                                     value.first / seconds,
                                                     value.second / 1e6 / seconds));
    }
    LOG_INFO(Render, "PERF waits:{}", Largest(std::move(waits), 20));
    g_waits.clear();
}

} // namespace perf

// ---- sotcport's interface (src/gpu/gpu.h) ----


namespace gpu {

bool 
Initialize(int sdk_version) {
    g_sdk_version = sdk_version;
    Common::Log::Setup("");
    // Before any Vulkan instance exists, so RenderDoc can hook it.
    VideoCore::LoadRenderDoc();
    VideoCore::SetOutputDir(Common::FS::GetUserPath(Common::FS::PathType::UserDir) / "captures",
                            "sotcport");
    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS | SDL_INIT_GAMEPAD)) {
        LOG_CRITICAL(Frontend, "SDL_Init failed: {}", SDL_GetError());
        return false;
    }
    // The diagnostics that record or check every GPU command make the game much slower (the
    // logos alone take minutes): say so in the title, so such a window is not taken for a
    // normal session.
    const bool diagnostics = std::getenv("SOTC_RECORD_GNM") ||
                             std::getenv("SOTC_VALIDATE_NATIVE") ||
                             std::getenv("SOTC_VALIDATE_NATIVE_DRAWS");
    g_window = new Frontend::WindowSDL(
        1280, 720, diagnostics ? "sotcport (diagnostics on: much slower)" : "sotcport");
    input::Initialize();
    Libraries::GnmDriver::RegisterLib(nullptr);
    Libraries::VideoOut::RegisterLib(nullptr);
    return true;
}


void FinishScreenshots() {
    // A screenshot is saved once the GPU has run the frame's copy (a deferred operation of the
    // draw scheduler), then encoded (~0.2 s each, one after the other).
    if (!Vulkan::WaitForScreenshots(std::chrono::seconds{5})) {
        LOG_WARNING(Render_Vulkan, "exiting with screenshots not saved yet");
    }
}

bool ProcessEvents(int timeout_ms) {
    // SOTC_SCREENSHOTS="60,95.5,120": screenshots at these times (seconds since start), for
    // looking at the game without anyone at the keyboard.
    static const auto start = std::chrono::steady_clock::now();
    static std::vector<double> captures = [] {
        std::vector<double> times;
        if (const char* list = std::getenv("SOTC_SCREENSHOTS")) {
            for (char* end = nullptr;; list = end + (*end == ',')) {
                const double time = std::strtod(list, &end);
                if (end == list) {
                    break;
                }
                times.push_back(time);
            }
        }
        std::ranges::sort(times, std::greater{});
        return times;
    }();
    const double now = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    while (!captures.empty() && captures.back() <= now) {
        captures.pop_back();
        RequestScreenshot();
    }
    // SOTC_RENDERDOC: the same for RenderDoc frame captures.
    static std::vector<double> frame_captures = [] {
        std::vector<double> times;
        if (const char* list = std::getenv("SOTC_RENDERDOC")) {
            for (char* end = nullptr;; list = end + (*end == ',')) {
                const double time = std::strtod(list, &end);
                if (end == list) {
                    break;
                }
                times.push_back(time);
            }
        }
        std::ranges::sort(times, std::greater{});
        return times;
    }();
    while (!frame_captures.empty() && frame_captures.back() <= now) {
        frame_captures.pop_back();
        CaptureFrame();
    }
    perf::Report(); // sotcport TEMP
    return g_window && g_window->ProcessEvents(timeout_ms);
}

namespace {
std::atomic<unsigned long long> g_frames_presented{0};
} // namespace

void NoteFramePresented() {
    g_frames_presented.fetch_add(1, std::memory_order_relaxed);
    perf::NoteFrameForSpikes(); // sotcport TEMP (SOTC_SPIKES)
    perf::Mark('P'); // TEMP timeline
}

namespace {
PacketObserver g_packet_observer = nullptr;
} // namespace

void SetPacketObserver(PacketObserver observer) {
    g_packet_observer = observer;
    AmdGpu::g_packet_observer =
        observer ? [](const u32* packet, u32 dwords, AmdGpu::PacketQueue queue) {
            g_packet_observer(packet, dwords, static_cast<u32>(queue));
        } : nullptr;
}

std::uint32_t ReadRegister(std::uint32_t index) {
    return index < AmdGpu::Regs::NumRegs ? liverpool->regs.reg_array[index] : 0;
}

// The command processor stores SET_SH_REG writes to 0x200 on (COMPUTE_*) in the running
// queue's ComputeProgram, laid out as the registers (user data at word 0x40).
static_assert(offsetof(AmdGpu::ComputeProgram, user_data) == 0x40 * 4);
std::uint32_t ReadComputeRegister(std::uint32_t index) {
    const std::uint32_t offset = index - (AmdGpu::Regs::ShRegWordOffset + 0x200);
    if (offset >= sizeof(AmdGpu::ComputeProgram) / 4) {
        return 0;
    }
    return reinterpret_cast<const std::uint32_t*>(&liverpool->GetCsRegs())[offset];
}

std::uint64_t ReadIndirectArgsAddress() {
    return liverpool->IndirectArgsAddress();
}

void ReadRegisterFile(std::uint32_t* out) {
    std::memcpy(out, liverpool->regs.reg_array.data(), sizeof(liverpool->regs.reg_array));
}

void ReadRenderTargetExtents(std::uint32_t* color, std::uint32_t& depth) {
    for (u32 i = 0; i < AmdGpu::NUM_COLOR_BUFFERS; i++) {
        color[i] = liverpool->last_cb_extent[i].raw;
    }
    depth = liverpool->last_db_extent.raw;
}

bool IsFastCleared(std::uint64_t meta_address, std::uint32_t slice, bool tracked) {
    if (!Core::g_rasterizer) {
        return false;
    }
    Core::g_rasterizer->DrainDrawPipe(Vulkan::DrawPipe::ReasonReadback);
    const auto& cache = Core::g_rasterizer->GetTextureCache();
    if (cache.IsMeta(meta_address)) {
        return cache.IsMetaCleared(meta_address, slice);
    }
    // FindRenderTarget / FindDepthTarget register it before the check, with clear_mask -1
    // (MetaDataInfo; a depth image made by overlap resolution starts at 0 instead: not seen).
    return tracked;
}

void SyncGuestRange(std::uint64_t address, std::uint64_t size) {
    // What a CPU read of GPU-written memory does (readbacks "Precise"), called directly: the
    // packet observer runs on the command processor thread, which holds what assume_locks means.
    if (Core::g_rasterizer && size) {
        Core::g_rasterizer->ReadMemory(address, size, /*assume_locks=*/true);
    }
}

// The GPU layer's image for a binding: like TextureCache::FindImage's perfect match, but with
// the same texel size instead of a compatible format (the bytes are compared, so they must be
// laid out alike) and any number of mips laid out alike; the image with the most of the
// binding's mips (`levels`). From the packet observer (stage B is drained first).
// `stencil_plane`: the binding is a depth target's stencil plane seen as an 8-bit texture (an
// image at the stencil address that FindDepthTarget associated with the depth image): binding it,
// the GPU layer redirects to the depth image's stencil aspect (BindTextures, GetAssociatedDepth),
// which is where the stencil values live (only on its GPU); that depth image is returned.
static VideoCore::Image* FindLayerImage(const render::translate::ImageBinding& binding,
                                        u32& levels, bool& stencil_plane) {
    auto& rasterizer = *Core::g_rasterizer;
    rasterizer.DrainDrawPipe(Vulkan::DrawPipe::ReasonReadback); // stage B touches the caches
    auto& cache = rasterizer.GetTextureCache();
    VideoCore::Image* found = nullptr;
    levels = 0;
    stencil_plane = false;
    if (binding.num_bits == 8 && !binding.block) {
        cache.ForEachImageInRegion(binding.address, 1, [&](VideoCore::ImageId, VideoCore::Image& image) {
            if (found || image.info.guest_address != binding.address) {
                return;
            }
            if (const auto depth_id = cache.GetAssociatedDepth(image)) {
                VideoCore::Image& depth = cache.GetImage(depth_id);
                if (depth.info.props.has_stencil && depth.info.size.width == binding.extent[0] &&
                    depth.info.size.height == binding.extent[1]) {
                    found = &depth;
                }
            }
        });
        if (found) {
            levels = 1;
            stencil_plane = true;
            return found;
        }
    }
    cache.ForEachImageInRegion(binding.address, 1, [&](VideoCore::ImageId, VideoCore::Image& image) {
        const auto& info = image.info;
        if (info.guest_address != binding.address || info.size.width != binding.extent[0] ||
            info.size.height != binding.extent[1] || info.size.depth != binding.extent[2] ||
            info.num_bits != binding.num_bits || bool(info.props.is_block) != binding.block ||
            info.num_samples != 1 || info.resources.layers < binding.layers) {
            return;
        }
        u32 common = std::min<u32>(info.resources.levels, binding.copies.size());
        for (u32 m = 0; m < common; m++) {
            if (info.mips_layout[m].offset != binding.copies[m].buffer_offset) {
                common = m;
            }
        }
        if (common > levels) {
            found = &image;
            levels = common;
        }
    });
    return found;
}

std::uint32_t ReadLayerImage(const render::translate::ImageBinding& binding, bool newer_only,
                             std::uint8_t* out, std::string& why) {
    if (!Core::g_rasterizer) {
        why = "no GPU layer";
        return 0;
    }
    u32 levels = 0;
    bool stencil_plane = false;
    VideoCore::Image* found = FindLayerImage(binding, levels, stencil_plane);
    if (!found) {
        why = "the GPU layer has no image for it";
        return 0;
    }
    auto& rasterizer = *Core::g_rasterizer;
    auto& cache = rasterizer.GetTextureCache();
    VideoCore::Image& image = *found;
    cache.RefreshImage(image);
    if (newer_only && False(image.flags & VideoCore::ImageFlagBits::GpuModified)) {
        why = "the GPU layer's image holds no GPU writes";
        return 0;
    }
    auto& runtime = rasterizer.GetRuntime();
    const auto staging = runtime.GetStagingPool().Request(binding.guest_size,
                                                          VideoCore::MemoryType::HostCached, 16);
    const auto aspect = stencil_plane            ? vk::ImageAspectFlagBits::eStencil
                        : image.info.props.is_depth ? vk::ImageAspectFlagBits::eDepth
                                                    : vk::ImageAspectFlagBits::eColor;
    const u32 layers = binding.volume ? 1 : binding.layers;
    std::vector<vk::BufferImageCopy> copies;
    for (u32 m = 0; m < levels; m++) {
        const auto& c = binding.copies[m];
        copies.push_back({
            .bufferOffset = staging.offset + c.buffer_offset,
            .bufferRowLength = c.row_length,
            .bufferImageHeight = c.image_height,
            .imageSubresource = {aspect, m, 0, layers},
            .imageOffset = {0, 0, 0},
            .imageExtent = {c.width, c.height, c.depth},
        });
    }
    runtime.DownloadImage(&image, staging.buffer, copies);
    rasterizer.GetScheduler().Finish();
    staging.Invalidate();
    binding.ForEachRow(levels, [&](u64 offset, u64 bytes) {
        std::memcpy(out + offset, staging.mapped + offset, bytes);
    });
    return levels;
}

bool ReadLayerStencil(const render::translate::ImageBinding& binding, std::uint8_t* out,
                      std::string& why) {
    if (!Core::g_rasterizer) {
        why = "no GPU layer";
        return false;
    }
    u32 levels = 0;
    bool stencil_plane = false;
    VideoCore::Image* image = FindLayerImage(binding, levels, stencil_plane);
    if (!image || !image->info.props.has_stencil) {
        why = image ? "the GPU layer's image has no stencil" : "the GPU layer has no image for it";
        return false;
    }
    auto& rasterizer = *Core::g_rasterizer;
    auto& runtime = rasterizer.GetRuntime();
    const u64 size = binding.StencilBytes();
    const auto staging =
        runtime.GetStagingPool().Request(size, VideoCore::MemoryType::HostCached, 16);
    const vk::BufferImageCopy copy{
        .bufferOffset = staging.offset,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = {vk::ImageAspectFlagBits::eStencil, 0, 0,
                             binding.volume ? 1u : binding.layers},
        .imageOffset = {0, 0, 0},
        .imageExtent = {binding.extent[0], binding.extent[1], 1},
    };
    runtime.DownloadImage(image, staging.buffer, std::span{&copy, 1});
    rasterizer.GetScheduler().Finish();
    staging.Invalidate();
    std::memcpy(out, staging.mapped, size);
    return true;
}

std::string DescribeLayerImages(const render::translate::ImageBinding& binding) {
    if (!Core::g_rasterizer || !binding.address) {
        return {};
    }
    u32 levels = 0;
    bool stencil_plane = false;
    const VideoCore::Image* taken = FindLayerImage(binding, levels, stencil_plane);
    std::string text;
    if (stencil_plane) {
        text += fmt::format("    the stencil plane of the GPU layer's depth image at {:#x} "
                            "(taken)\n",
                            taken->info.guest_address);
    }
    Core::g_rasterizer->GetTextureCache().ForEachImageInRegion(
        binding.address, binding.guest_size, [&](VideoCore::ImageId id, VideoCore::Image& image) {
            const auto& info = image.info;
            text += fmt::format(
                "    layer image {}: at {:+#x} {} {}x{}x{} {} bits{} tile {} array {} pitch {} "
                "size {:#x} mips {} layers {} flags {:#x}{}\n",
                id.index, s64(info.guest_address) - s64(binding.address),
                vk::to_string(info.pixel_format), info.size.width, info.size.height,
                info.size.depth, info.num_bits, info.props.is_block ? " block" : "",
                u32(info.tile_mode), u32(info.array_mode), info.pitch, info.guest_size,
                info.resources.levels, info.resources.layers, u32(image.flags),
                &image == taken ? " (taken)" : "");
        });
    return text;
}

bool IsGpuModified(std::uint64_t address, std::uint64_t size) {
    return Core::g_rasterizer &&
           Core::g_rasterizer->GetBufferCache().IsRegionGpuModified(address, size);
}

bool PeekLayerBuffer(std::uint64_t address, std::uint64_t size, std::uint8_t* out) {
    if (!Core::g_rasterizer) {
        return false;
    }
    Core::g_rasterizer->DrainDrawPipe(Vulkan::DrawPipe::ReasonReadback);
    return Core::g_rasterizer->GetBufferCache().PeekGpuMemory(address, size, out);
}

unsigned long long FramesPresented() {
    return g_frames_presented.load(std::memory_order_relaxed);
}

void RequestScreenshot() {
    VideoCore::RequestScreenshot(VideoCore::ScreenshotRequest::GameOnly);
}

void CaptureFrame() {
    if (!VideoCore::IsRenderDocLoaded()) {
        LOG_WARNING(Render, "RenderDoc not loaded (run with SOTC_RENDERDOC=...)");
        return;
    }
    LOG_INFO(Render, "capturing a frame in RenderDoc");
    VideoCore::TriggerCapture();
}

} // namespace gpu
