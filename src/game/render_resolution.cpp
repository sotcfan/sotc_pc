// The resolution the game renders at.
//
// ==== How the game sizes its render targets (2026-10-20) ========================================
//
// On a PS4 Pro the game has two video modes (the options menu's "PS4 Pro mode", or the engine's
// ConVar r_Wants4K): "1080p 60FPS" renders the 3D scene at 1600x900 and outputs 1920x1080, "4K
// 30FPS" renders at 2560x1440 and outputs 3840x2160 (SOTC_LOG_RT_SIZES). The render target
// manager (Renderer/Backend/PS4) keeps, for each of its 144 targets (full, half, quarter size...),
// the memory, allocated at the 4K mode's size and used as it is in that mode, and smaller views
// of it for the other two modes, with their sizes in two tables (SOTC_LOG_RT_TABLE):
//
//   manager +0x0000  base target[144]   (8 bytes each; null = slot unused)
//   manager +0x0480  sizes in the base PS4's mode[144]: u32 width, u32 height, u32 flags (12
//                    bytes); the full-size ones 1920x1080
//   manager +0x0b40  view in that mode[144]
//   manager +0x0fc0  sizes in the PS4 Pro's 1080p mode[144] (same layout); full size 1600x900
//   manager +0x1680  view in that mode[144]
//   manager +0x1f80  the active view[144]
//   manager +0x3b70  u32 the mode: 0 the base PS4's, 1 the 4K one (the base targets), 2 the
//                    1080p one
//
// 0x49fa90(manager) (re)creates the views from the size tables (0x496f70(base, 0, width, height,
// flags) makes a view); 0x49fba0 makes the views of the current mode the active ones.
//
// The sizes of the 4K mode come from the backend's start (0x495a90), which passes the render
// target manager (constructor 0x49aef0) {1600, 900, 1920, 1080, 2560, 1440, 3840, 2160}: the
// 1080p mode's scene, the base PS4's, the 4K mode's scene (the size the base targets are
// allocated at) and its output (the "DisplayBuffer" allocation; the 1080p modes use 1920x1080
// views of it). The 4K mode's scene and output sizes are immediates of 0x495a90, in Pro mode (on a
// base PS4 a cmove picks 1920x1080 instead):
//
//   0x495bf1  41 bf 70 08 00 00  mov r15d, 0x870   output height (2160)
//   0x495bfc  41 bc 00 0f 00 00  mov r12d, 0xf00   output width (3840)
//   0x495d5b  41 bd 00 0a 00 00  mov r13d, 0xa00   scene width (2560)
//   0x495d76  41 bf a0 05 00 00  mov r15d, 0x5a0   scene height (1440)
//
// The rest of the game asks what the screen is with the mode: 0x56f2e0 / 0x56f300 return
// 1920 / 1080, or 3840 / 2160 when the backend's mode (backend +0x3c, the backend at 0x32d3670) is
// 2, and 0x575a60 and 0x4a2aa0 do the same inline; the interface lays itself out on a 1920x1080
// screen and scales it by that. Others read float tables indexed by the same test, pairs of the
// value in the 1080p modes and in the 4K one: {1920, 3840} and {1080, 2160}, and the half sizes
// {960, 1920}, {540, 1080} (at 0x13c34a0, 0x13d7518 and 0x1667f14-0x1667fa3).
//
// The targets come from the render target allocator ("RTAllocator", 0x49aef0), a pool of a fixed
// size: 573 MiB on a PS4 Pro (mov eax, 0x23d10000 at 0x49b760; 322.5 MiB on a base PS4), enough
// for the 4K mode's 2560x1440 scene. A larger scene ran out of it ("Out of Memory, Requesting
// 21.33 MiB from RTAllocator - Free: 11.91 MiB / 573.06 MiB" at 3840x2160), so above 2560x1440
// the pool grows with the number of pixels.
//
// The game uses the 4K mode only on a TV taller than 1080 lines (0x495a90 checks
// sceVideoOutGetResolutionStatus) and when its options, or the ConVar r_Wants4K, ask for it.
//
// ==== Native resolution =========================================================================
//
// SOTC_RESOLUTION=<w>x<h> (with SOTC_NEO_MODE=1) makes the 4K mode render its scene and output at
// that size, so the picture leaves the game at the size it was drawn at: no upscaling (the game's
// own pass from the scene to the output runs at 1:1). Every 3840/2160 of the 4K mode above
// becomes the width/height (their halves, the halves): with only the targets resized, the
// interface still laid itself out for 3840x2160, 1.5 times too large at 2560x1440 (2026-10-20). sotcport then also reports a 4K TV (unless
// SOTC_VIDEO_OUT says otherwise) and asks for the 4K mode on the game's command line
// (+r_Wants4K=WantsHigh) unless the command line already sets r_Wants4K. The game's camera
// assumes 16:9: other aspect ratios come out stretched.
//
// SOTC_LOG_RT_TABLE=1 logs the tables each time the views are made.

#include "game/render_resolution.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "common/log.h"
#include "common/types.h"
#include "loader/hook.h"

namespace game {

namespace {

constexpr u64 MakeViewsAddress = 0x49fa90;

// The immediates of 0x495a90 that size the 4K mode (see above): the address of the instruction,
// its bytes as the game ships them, and which dimension it holds.
struct SizeImmediate {
    u64 address;
    u8 length;      // of the instruction; the immediate is its last four bytes
    u8 original[6];
    bool width;     // else the height
};
// The render target pool's size in Pro mode (see above).
constexpr u64 TargetPoolAddress = 0x49b760;
constexpr u8 TargetPoolOriginal[5] = {0xb8, 0x00, 0x00, 0xd1, 0x23}; // mov eax, 0x23d10000
constexpr u64 TargetPoolSize = 0x23d10000;

constexpr SizeImmediate SizeImmediates[] = {
    {0x495bf1, 6, {0x41, 0xbf, 0x70, 0x08, 0x00, 0x00}, false},
    {0x495bfc, 6, {0x41, 0xbc, 0x00, 0x0f, 0x00, 0x00}, true},
    {0x495d5b, 6, {0x41, 0xbd, 0x00, 0x0a, 0x00, 0x00}, true},
    {0x495d76, 6, {0x41, 0xbf, 0xa0, 0x05, 0x00, 0x00}, false},
    // The screen size the rest of the game asks for in the 4K mode.
    {0x4a2ae4, 5, {0xb8, 0x00, 0x0f, 0x00, 0x00}, true},  // mov eax, 0xf00
    {0x4a2b08, 5, {0xb8, 0x70, 0x08, 0x00, 0x00}, false}, // mov eax, 0x870
    {0x56f2e7, 5, {0xb9, 0x00, 0x0f, 0x00, 0x00}, true},  // mov ecx, 0xf00
    {0x56f307, 5, {0xb9, 0x70, 0x08, 0x00, 0x00}, false}, // mov ecx, 0x870
    {0x575c58, 5, {0xb9, 0x70, 0x08, 0x00, 0x00}, false}, // mov ecx, 0x870
};

// The float tables of the screen size per mode (see above): where they start and how many
// {1080p modes, 4K mode} pairs follow.
struct SizeTable {
    u64 address;
    u32 pairs;
};
constexpr SizeTable SizeTables[] = {{0x13c34a0, 2}, {0x13d7518, 2}, {0x1667f14, 18}};

constexpr u32 TargetCount = 144;
constexpr u32 BaseTargetsOffset = 0x0;
constexpr u32 BaseSizesOffset = 0x480;
constexpr u32 ProSizesOffset = 0xfc0;
constexpr u32 ModeOffset = 0x3b70;

struct TargetSize {
    u32 width;
    u32 height;
    u32 flags;
};
static_assert(sizeof(TargetSize) == 12);

using MakeViewsFn = void PS4_SYSV_ABI (*)(u8* manager);
MakeViewsFn g_original_make_views = nullptr;

bool g_log_table = false;

TargetSize* Sizes(u8* manager, u32 offset) {
    return reinterpret_cast<TargetSize*>(manager + offset);
}

void LogTable(u8* manager) {
    const auto* base = reinterpret_cast<u64*>(manager + BaseTargetsOffset);
    const auto* base_ps4 = Sizes(manager, BaseSizesOffset);
    const auto* pro_1080p = Sizes(manager, ProSizesOffset);
    u32 mode = 0;
    std::memcpy(&mode, manager + ModeOffset, sizeof(mode));
    LOG_INFO("Resolution", "render target views made (manager {}, mode {}):",
             fmt::ptr(manager), mode);
    for (u32 i = 0; i < TargetCount; i++) {
        if (!base[i]) {
            continue;
        }
        LOG_INFO("Resolution", "  {:3}: base PS4 mode {}x{} ({:#x}), PS4 Pro 1080p mode {}x{} ({:#x})",
                 i, base_ps4[i].width, base_ps4[i].height, base_ps4[i].flags, pro_1080p[i].width,
                 pro_1080p[i].height, pro_1080p[i].flags);
    }
}

void PS4_SYSV_ABI MakeViews(u8* manager) {
    if (g_log_table) {
        LogTable(manager);
    }
    g_original_make_views(manager);
}

bool EnvOn(const char* name) {
    const char* value = std::getenv(name);
    return value && *value && std::strcmp(value, "0") != 0;
}

// Rewrites the 4K mode's scene and output sizes, after checking every instruction is the one the
// game ships (another version of the executable is left alone).
bool PatchSizes(u32 width, u32 height) {
    for (const auto& immediate : SizeImmediates) {
        if (std::memcmp(reinterpret_cast<const void*>(immediate.address), immediate.original,
                        immediate.length) != 0) {
            LOG_ERROR("Resolution", "unexpected code at {:#x}: the resolution stays the game's",
                      immediate.address);
            return false;
        }
    }
    // The tables first: a pair that is not one of the four kinds means another executable.
    for (const auto& table : SizeTables) {
        const auto* pairs = reinterpret_cast<const float*>(table.address);
        for (u32 i = 0; i < table.pairs; i++) {
            const float small = pairs[2 * i];
            const bool known = small == 1920.0f || small == 1080.0f || small == 960.0f ||
                               small == 540.0f;
            if (!known || pairs[2 * i + 1] != 2.0f * small) {
                LOG_ERROR("Resolution", "unexpected size table at {:#x}: the resolution stays the "
                                        "game's",
                          table.address);
                return false;
            }
        }
    }
    for (const auto& table : SizeTables) {
        auto* pairs = reinterpret_cast<float*>(table.address);
        for (u32 i = 0; i < table.pairs; i++) {
            const float small = pairs[2 * i];
            const float full = small == 1920.0f || small == 960.0f ? float(width) : float(height);
            pairs[2 * i + 1] = small == 960.0f || small == 540.0f ? full / 2.0f : full;
        }
    }
    // A scene larger than the game's own 2560x1440 needs a larger render target pool.
    const u64 pixels = u64{width} * height;
    const u64 game_pixels = 2560 * 1440;
    if (pixels > game_pixels) {
        if (std::memcmp(reinterpret_cast<const void*>(TargetPoolAddress), TargetPoolOriginal,
                        sizeof(TargetPoolOriginal)) != 0) {
            LOG_ERROR("Resolution", "unexpected code at {:#x}: the resolution stays the game's",
                      TargetPoolAddress);
            return false;
        }
        // Rounded up to the pool's 64 KiB alignment.
        const u64 pool = (TargetPoolSize * pixels / game_pixels + 0xffff) & ~u64{0xffff};
        if (pool > 0xffffffffu) {
            LOG_ERROR("Resolution", "{}x{} needs a render target pool over 4 GiB", width, height);
            return false;
        }
        const u32 pool32 = static_cast<u32>(pool);
        std::memcpy(reinterpret_cast<void*>(TargetPoolAddress + 1), &pool32, sizeof(pool32));
        LOG_INFO("Resolution", "render target pool {} MiB (the game's {} MiB)", pool >> 20,
                 TargetPoolSize >> 20);
    }
    for (const auto& immediate : SizeImmediates) {
        const u32 value = immediate.width ? width : height;
        std::memcpy(reinterpret_cast<void*>(immediate.address + immediate.length - 4), &value,
                    sizeof(value));
    }
    return true;
}

} // namespace

void InstallRenderResolution(std::vector<std::string>& game_args) {
    g_log_table = EnvOn("SOTC_LOG_RT_TABLE");
    if (g_log_table) {
        g_original_make_views = reinterpret_cast<MakeViewsFn>(
            loader::HookFunction(MakeViewsAddress, reinterpret_cast<const void*>(&MakeViews)));
        if (!g_original_make_views) {
            LOG_ERROR("Resolution", "could not hook the render target views at {:#x}",
                      MakeViewsAddress);
        }
    }

    const char* resolution = std::getenv("SOTC_RESOLUTION");
    if (!resolution || !*resolution) {
        return;
    }
    u32 width = 0, height = 0;
    if (std::sscanf(resolution, "%ux%u", &width, &height) != 2 || width < 640 || height < 360 ||
        width > 7680 || height > 4320) {
        LOG_ERROR("Resolution", "SOTC_RESOLUTION={}: expected <width>x<height>, 640x360 to "
                                "7680x4320",
                  resolution);
        return;
    }
    // Render targets are tiled in 8x8 blocks and halved for the smaller passes: even sizes.
    width &= ~1u;
    height &= ~1u;
    if (!EnvOn("SOTC_NEO_MODE")) {
        LOG_ERROR("Resolution", "SOTC_RESOLUTION needs SOTC_NEO_MODE=1 (it sizes the PS4 Pro's "
                                "4K mode); the resolution stays the game's");
        return;
    }
    if (!PatchSizes(width, height)) {
        return;
    }
    // Ask for the 4K mode unless the game's command line already chooses one.
    const bool chosen = std::ranges::any_of(game_args, [](const std::string& arg) {
        return arg.starts_with("+r_Wants4K=");
    });
    if (!chosen) {
        game_args.push_back("+r_Wants4K=WantsHigh");
    }
    LOG_INFO("Resolution", "the game renders and outputs at {}x{} (its 4K mode{})", width, height,
             chosen ? ", if the command line's r_Wants4K picks it" : "");
}

} // namespace game
