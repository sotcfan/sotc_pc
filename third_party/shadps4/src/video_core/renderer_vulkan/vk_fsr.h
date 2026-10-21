// SPDX-FileCopyrightText: Copyright 2026 sotcport
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

// sotcport: FSR 3.1 temporal upscaling of the game's scene (SOTC_UPSCALER=fsr, src/gpu/upscaler.h).
//
// The game's frame in its PS4 Pro 4K mode (RenderDoc, 2026-10-20): the scene is drawn at the
// render size (HDR color RGBA16F, motion vectors R16G16 float, reversed-Z D32S8 depth), the
// temporal anti-aliasing and the post-processing (motion blur, depth of field, bloom) run at that
// size into an RGBA16F image, the interface is drawn at the output size, and one final draw
// composes them into the output buffer: it samples the scene image (enlarging it with a plain
// filter, and tonemapping it) and the interface. FSR goes right before that draw: the
// post-processed scene, the scene's depth and motion vectors and the camera jitter
// (src/game/upscaler.cpp) make an image of the output size, which the final draw samples
// instead. The game's own anti-aliasing is off (FSR does it).
//
// The motion vectors are uv(now) - uv(previous frame) in texture units (the game's TAA looks the
// history up at uv - v), so FSR gets motionVectorScale = -render size.

#include <chrono>
#include <memory>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/texture_cache/image.h"

struct FfxVkPortableUpscaleContext;

namespace VideoCore {
class TextureCache;
}

namespace Vulkan {

class Instance;
class Scheduler;
class Runtime;

class FsrPass {
public:
    FsrPass(const Instance& instance, Scheduler& scheduler, Runtime& runtime,
            VideoCore::TextureCache& texture_cache);
    ~FsrPass();

    /// Whether FSR is on. The settings are read on first use: the renderer is made before the
    /// game's resolution (and so FSR's sizes) is set up (src/game/render_resolution.cpp).
    [[nodiscard]] bool Enabled() noexcept {
        if (!configured) {
            LoadConfig();
        }
        return enabled;
    }

    /// A render target of a draw: the scene's depth and motion vectors are the last render-size
    /// depth and R16G16 float targets before the final draw.
    void NoteTarget(VideoCore::ImageId id, const VideoCore::Image& image);

    /// Whether `image` is the post-processed scene (render size, RGBA16F).
    [[nodiscard]] bool IsScene(const VideoCore::Image& image);
    /// Whether `image`, a draw's first color target, is a display buffer of the output size (the
    /// game's final draw writes the frame there: tiled for display, which no scene target is).
    [[nodiscard]] bool IsDisplayTarget(const VideoCore::Image& image);

    /// The final draw samples `scene`: records FSR (once per frame, before the draw's render
    /// pass) and gives the view and layout of its result to bind instead. False if FSR could not
    /// run (the game's image is then bound as it is).
    bool ReplaceScene(VideoCore::ImageId scene_id, vk::ImageView& view, vk::ImageLayout& layout);

private:
    void LoadConfig();
    bool CreateContext();
    void CreateOutput();

    const Instance& instance;
    Scheduler& scheduler;
    Runtime& runtime;
    VideoCore::TextureCache& texture_cache;
    bool configured = false;
    bool enabled = false;
    bool failed = false;
    u32 render_width = 0, render_height = 0, output_width = 0, output_height = 0;
    float sharpness = 0.5f;

    FfxVkPortableUpscaleContext* context = nullptr;
    VideoCore::UniqueImage output;
    vk::UniqueImageView output_view;
    bool output_initialized = false;
    bool resume = false; ///< switched off at run time (F6): the history is stale when it is back

    VideoCore::ImageId depth_id{};
    VideoCore::ImageId motion_id{};
    u64 frame_id = 0;
    unsigned long long dispatched_for = ~0ull; // gpu::FramesPresented() of the last dispatch
    std::chrono::steady_clock::time_point last_dispatch{};
};

} // namespace Vulkan
