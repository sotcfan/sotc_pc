// The game's camera jitter, for a temporal upscaler (FSR).
//
// ==== How the game jitters its camera (2026-10-20) ==============================================
//
// For its temporal anti-aliasing the engine moves the projection by a sub-pixel offset every
// frame. 0x4bb950(view state, const float2* scale, u32 phases) returns it: the Halton (2, 3)
// sequence from index 1 (a table of 47 pairs copied from 0x13a0580 to 0x32d3b80), entry
// (frame counter % phases), as scale * (2t - 1); zero when the anti-aliasing (render settings
// +0x3a8) or the jitter (+0x3b0, ConVar r_aaEnableTemporalProjectMatrixJitter) is off, unless
// r_aaForceJitterIndex (+0x3b4) forces an entry. The frame counter is [[view state + 0x10] +
// 0xd70].
//
// Its main caller, 0x4bbb00 (the call returns to 0x4bbbb1), asks for 8 phases with scale =
// k / render target size (k from a table at 0x13a06e0 by a byte of the view: 0.5 or 1.0) and
// keeps the result in the view's constants (+0x80); the main view asks with k = 0.5 (logged), an
// offset of up to half a pixel. Another caller (0x4c0a62, 16 phases, scale in pixels) uses it for
// something else.
//
// ==== With FSR (SOTC_UPSCALER=fsr, src/gpu/upscaler.h) ===========================================
//
// The game's temporal anti-aliasing is turned off (+r_aaMethod=Off: FSR takes its place) and the
// main caller's jitter becomes FSR's: Halton (2, 3) from index 1 over ffxFsr3GetJitterPhaseCount
// phases (8 * (output / render)^2), which the GPU layer gets in pixels (gpu::upscaler::SetJitter)
// to hand FSR. The value returned to the game is that offset in the game's units: its main view
// asks with k = 0.5, which makes (t - 0.5) pixels out of scale * (2t - 1), so r = 2 * jitter *
// scale. SOTC_FSR_JITTER_SCALE=<f> (default 2) and SOTC_FSR_JITTER_SIGN=<sx>,<sy> (default 1,1)
// change that, for checking the convention by the picture's sharpness: on 2026-10-20 the defaults
// gave the sharpest still image (Laplacian of the temple door: 93.5, against 89.6 with scale 4,
// 90.9 / 91.4 with y / x flipped; native 1440p 97.9, the game's own enlargement 62.5).
//
// SOTC_LOG_JITTER=1 logs the calls (rate-limited).

#include "game/upscaler.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common/log.h"
#include "common/types.h"
#include "gpu/upscaler.h"
#include "loader/hook.h"

namespace game {

namespace {

constexpr u64 JitterAddress = 0x4bb950;
constexpr u64 MainJitterReturn = 0x4bbbb1;

struct Float2 {
    float x;
    float y;
};

using JitterFn = Float2 PS4_SYSV_ABI (*)(u8* view_state, const Float2* scale, u32 phases);
JitterFn g_original_jitter = nullptr;

bool g_log = false;
bool g_fsr = false;
float g_jitter_scale = 2.0f;
float g_sign_x = 1.0f, g_sign_y = 1.0f;
std::atomic<u32> g_logged{0};

float Halton(u32 index, u32 base) {
    float fraction = 1.0f, result = 0.0f;
    while (index > 0) {
        fraction /= float(base);
        result += fraction * float(index % base);
        index /= base;
    }
    return result;
}

u32 FrameCounter(const u8* view_state) {
    const u8* game = nullptr;
    std::memcpy(&game, view_state + 0x10, sizeof(game));
    u32 counter = 0;
    if (game) {
        std::memcpy(&counter, game + 0xd70, sizeof(counter));
    }
    return counter;
}

Float2 PS4_SYSV_ABI Jitter(u8* view_state, const Float2* scale, u32 phases) {
    const u64 return_address = reinterpret_cast<u64>(__builtin_return_address(0));
    if (g_fsr && return_address == MainJitterReturn) {
        // FSR's jitter, as ffxFsr3GetJitterOffset: Halton from index 1, minus a half.
        const u32 frame = FrameCounter(view_state);
        const u32 index = frame % gpu::upscaler::JitterPhases();
        const float x = Halton(index + 1, 2) - 0.5f;
        const float y = Halton(index + 1, 3) - 0.5f;
        gpu::upscaler::SetJitter({x, y, frame});
        const Float2 result{g_sign_x * g_jitter_scale * x * scale->x,
                            g_sign_y * g_jitter_scale * y * scale->y};
        if (g_log && g_logged.fetch_add(1) < 400) {
            LOG_INFO("Jitter", "FSR frame {} phase {} ({:.4f}, {:.4f}) px -> ({:.6g}, {:.6g})",
                     frame, index, x, y, result.x, result.y);
        }
        return result;
    }
    const Float2 result = g_original_jitter(view_state, scale, phases);
    if (g_log && g_logged.fetch_add(1) < 400) {
        const u64 caller = reinterpret_cast<u64>(__builtin_return_address(0));
        LOG_INFO("Jitter", "caller {:#x} frame {} phases {} scale ({:.6g}, {:.6g}) -> ({:.6g}, "
                           "{:.6g})",
                 caller, FrameCounter(view_state), phases, scale->x, scale->y, result.x,
                 result.y);
    }
    return result;
}

} // namespace

void InstallUpscaler(std::vector<std::string>& game_args) {
    const char* log = std::getenv("SOTC_LOG_JITTER");
    g_log = log && *log && *log != '0';
    g_fsr = gpu::upscaler::GetConfig().enabled;
    if (g_fsr) {
        // FSR replaces the game's temporal anti-aliasing (unless the command line sets one).
        const bool chosen = std::ranges::any_of(game_args, [](const std::string& arg) {
            return arg.starts_with("+r_aaMethod=");
        });
        if (!chosen) {
            game_args.push_back("+r_aaMethod=Off");
        }
        if (const char* scale = std::getenv("SOTC_FSR_JITTER_SCALE")) {
            g_jitter_scale = std::strtof(scale, nullptr);
        }
        if (const char* sign = std::getenv("SOTC_FSR_JITTER_SIGN")) {
            std::sscanf(sign, "%f,%f", &g_sign_x, &g_sign_y);
        }
    }
    if (!g_log && !g_fsr) {
        return;
    }
    g_original_jitter = reinterpret_cast<JitterFn>(
        loader::HookFunction(JitterAddress, reinterpret_cast<const void*>(&Jitter)));
    if (!g_original_jitter) {
        LOG_ERROR("Jitter", "could not hook the camera jitter at {:#x}", JitterAddress);
    }
}

} // namespace game
