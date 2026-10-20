// The temporal upscaler's settings and shared jitter (see upscaler.h).

#include "gpu/upscaler.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include "common/log.h"

namespace gpu::upscaler {

namespace {

Config g_config;
std::mutex g_jitter_mutex;
Jitter g_jitter;

const char* Env(const char* name) {
    const char* value = std::getenv(name);
    return value && *value ? value : nullptr;
}

} // namespace

bool Requested() {
    const char* upscaler = Env("SOTC_UPSCALER");
    return upscaler && (std::strcmp(upscaler, "fsr") == 0 || std::strcmp(upscaler, "fsr3") == 0);
}

bool RenderSizeForMode(std::uint32_t output_width, std::uint32_t output_height,
                       std::uint32_t& render_width, std::uint32_t& render_height) {
    const char* mode = Env("SOTC_FSR_MODE");
    if (!mode) {
        return false;
    }
    // FSR's own ratios (FFX_FSR3UPSCALER_QUALITY_MODE_*); "native" is FSR as anti-aliasing only.
    float ratio = 0.0f;
    if (std::strcmp(mode, "native") == 0) {
        ratio = 1.0f;
    } else if (std::strcmp(mode, "quality") == 0) {
        ratio = 1.5f;
    } else if (std::strcmp(mode, "balanced") == 0) {
        ratio = 1.7f;
    } else if (std::strcmp(mode, "performance") == 0) {
        ratio = 2.0f;
    } else if (std::strcmp(mode, "ultra") == 0) {
        ratio = 3.0f;
    } else {
        LOG_ERROR("Upscaler", "SOTC_FSR_MODE={}: native, quality, balanced, performance or ultra",
                  mode);
        return false;
    }
    // Even sizes, as the render targets need.
    render_width = static_cast<std::uint32_t>(std::lround(output_width / ratio)) & ~1u;
    render_height = static_cast<std::uint32_t>(std::lround(output_height / ratio)) & ~1u;
    return true;
}

void Configure(std::uint32_t render_width, std::uint32_t render_height,
               std::uint32_t output_width, std::uint32_t output_height) {
    if (!Requested()) {
        return;
    }
    if (render_width > output_width || render_height > output_height) {
        LOG_ERROR("Upscaler", "FSR needs a render size ({}x{}) no larger than the output ({}x{}); "
                              "it stays off",
                  render_width, render_height, output_width, output_height);
        return;
    }
    g_config.enabled = true;
    g_config.render_width = render_width;
    g_config.render_height = render_height;
    g_config.output_width = output_width;
    g_config.output_height = output_height;
    if (const char* sharpness = Env("SOTC_FSR_SHARPNESS")) {
        g_config.sharpness = std::clamp(std::strtof(sharpness, nullptr), 0.0f, 1.0f);
    }
    LOG_INFO("Upscaler", "FSR 3.1: {}x{} -> {}x{}, {} jitter phases, sharpness {}", render_width,
             render_height, output_width, output_height, JitterPhases(), g_config.sharpness);
}

const Config& GetConfig() {
    return g_config;
}

std::uint32_t JitterPhases() {
    if (!g_config.enabled || g_config.render_width == 0) {
        return 8;
    }
    const float ratio = float(g_config.output_width) / float(g_config.render_width);
    return static_cast<std::uint32_t>(std::ceil(8.0f * ratio * ratio));
}

void SetJitter(const Jitter& jitter) {
    std::scoped_lock lock{g_jitter_mutex};
    g_jitter = jitter;
}

Jitter GetJitter() {
    std::scoped_lock lock{g_jitter_mutex};
    return g_jitter;
}

} // namespace gpu::upscaler
