#pragma once

#include <functional>
#include <string>
#include <vector>

#include "common/types.h"
#include "render/native_resources.h"
#include "render/shader_translate.h"

// The native renderer running one draw on its own Vulkan device (render/vulkan), next to the GPU
// layer, which keeps showing the game: the shaders are translated for the current game memory
// (translate::TranslateDraw), their resources prepared as for a dispatch (native_resources.h),
// the vertex and index buffers copied from game memory, and each color target starts as the GPU
// layer holds it (or as game memory does), so that after the draw it can be compared with the
// GPU layer's (diag/native_validate.cpp).
//
// Supported for now: the vertex + pixel shader pipeline (no tessellation or geometry shaders),
// single-sampled color targets, a depth target without stencil test, non-indirect draws.

namespace render {

struct NativeDrawResult : ResourceReport {
    u64 vs_hash = 0, ps_hash = 0;
    bool stencil_from_layer = false; // the stencil plane came from the GPU layer
    std::string skipped; // set when the draw uses something not supported yet
    std::string error;   // set when something failed
    translate::TranslatedDraw draw; // what ran (for diagnostics)
};

// Whether the GPU layer holds a fast clear it has not applied yet of the target whose metadata
// (CMASK of a color target, HTILE of a depth target) is at `meta_address`, slice `slice`: the
// target then starts as its clear value. `tracked`: the target's meta_tracked (the layer
// registers metadata it does not know yet when it binds the target, as cleared).
using FastClearQuery = std::function<bool(u64 meta_address, u32 slice, bool tracked)>;
// The GPU layer's stencil plane of a depth target (ImageBinding::StencilBytes bytes; the stencil
// lives on its GPU only). False: unknown (it starts as 0).
using LayerStencilReader =
    std::function<bool(const translate::ImageBinding& binding, std::vector<u8>& out)>;

// Runs the draw whose PM4 packet is `packet` with the registers of `state`; on the GPU command
// processor thread (see TranslateCompute).
void RunNativeDraw(const translate::DrawState& state, const u32* packet, const GuestReader& read,
                   const LayerImageReader& read_image, const LayerStencilReader& read_stencil,
                   const FastClearQuery& fast_cleared, NativeDrawResult& result);

} // namespace render
