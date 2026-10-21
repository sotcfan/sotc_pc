#pragma once

// GPU layer optimizations that can be switched at run time (diagnostics). Each also has its
// environment variable (OPTIONS.md), read once at start; these are for switching while the
// game runs, to see which one changes a defect that only shows in one place of the game (F7,
// src/input/input.cpp: a sequence that turns them off one by one and takes screenshots).
// Header only, so the shadPS4 sources can use it too.

#include <atomic>
#include <cstdint>

namespace diag {

enum Toggle : std::uint32_t {
    ShadowCopyOff,         // as SOTC_NO_SHADOW_COPY
    SrtPatchesOff,         // as SOTC_NO_SRT_PATCHES
    BindMemoOff,           // as SOTC_NO_BIND_MEMO
    TargetCacheOff,        // as SOTC_NO_TARGET_CACHE
    TextureLookupOff,      // as SOTC_NO_TEXTURE_CACHE_LOOKUP
    LodStatsOn,            // as SOTC_LOD_STATS=1
    StrictBarriersOn,      // as SOTC_STRICT_BARRIERS=1
    WholeBuffersOn,        // as SOTC_UNBOUNDED_WINDOW=0
    SyncReadbackOn,        // as SOTC_ASYNC_READBACK unset
    Cap60On,               // as SOTC_FPS_LIMIT=60 (with SOTC_UNCAPPED or SOTC_FPS_LIMIT)
    FsrOff,                // as SOTC_UPSCALER unset: the game's image and jitter (no TAA)
};

inline std::atomic<std::uint32_t> g_toggles{0};

inline bool On(Toggle toggle) {
    return (g_toggles.load(std::memory_order_relaxed) >> toggle) & 1;
}

inline void SetToggles(std::uint32_t mask) {
    g_toggles.store(mask, std::memory_order_relaxed);
}

} // namespace diag
