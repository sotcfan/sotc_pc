#pragma once

// The temporal upscaler (FSR 3.1): its settings and the camera jitter it shares between the
// game's hook (src/game/upscaler.cpp, on the game's thread) and the GPU layer
// (third_party/shadps4/.../vk_fsr.cpp, on the draw recording thread).
//
// SOTC_UPSCALER=fsr turns it on; the scene is then drawn at the render size (SOTC_RENDER_RESOLUTION,
// or SOTC_FSR_MODE=quality|balanced|performance|ultra: 1.5, 1.7, 2 or 3 times smaller than the
// output) and FSR enlarges it to the output size (SOTC_RESOLUTION), replacing the game's own
// temporal anti-aliasing. SOTC_FSR_SHARPNESS=<0..1> (default 0.5; 0 turns sharpening off).

#include <cstdint>

namespace gpu::upscaler {

struct Config {
    bool enabled = false;
    std::uint32_t render_width = 0, render_height = 0; // the scene
    std::uint32_t output_width = 0, output_height = 0; // what the game shows
    float sharpness = 0.5f;
};

// Whether SOTC_UPSCALER asks for FSR (before the sizes are known).
bool Requested();

// The render size SOTC_FSR_MODE asks for an output of that size, or false (no mode given).
bool RenderSizeForMode(std::uint32_t output_width, std::uint32_t output_height,
                       std::uint32_t& render_width, std::uint32_t& render_height);

// Called once the game's render and output sizes are set (src/game/render_resolution.cpp):
// enables FSR when it was requested.
void Configure(std::uint32_t render_width, std::uint32_t render_height,
               std::uint32_t output_width, std::uint32_t output_height);

const Config& GetConfig();

// The number of jitter phases FSR wants for this ratio (8 * (output / render)^2, as
// ffxFsr3GetJitterPhaseCount).
std::uint32_t JitterPhases();

// The jitter of the frame being drawn, in render pixels (x right, y down), and the counter of the
// game frame it belongs to.
struct Jitter {
    float x = 0.0f;
    float y = 0.0f;
    std::uint32_t frame = 0;
};
void SetJitter(const Jitter& jitter);
Jitter GetJitter();

} // namespace gpu::upscaler
