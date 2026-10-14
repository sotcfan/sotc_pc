#pragma once

#include <array>
#include <functional>
#include <string>
#include <vector>

#include "common/types.h"
#include "render/native_resources.h"
#include "render/shader_translate.h"

// The native renderer running one compute dispatch on its own Vulkan device (render/vulkan),
// from the state the model knows: the COMPUTE_* registers and the dispatch size. The shader is
// translated for the current game memory; the buffers it uses are copied from game memory into
// buffers of the native device, the dispatch runs, and the bytes of every buffer it may write
// come back.
//
// This is the first piece of step 2 of the native renderer (PLANO_RENDERIZADOR_NATIVO.md): it
// runs next to the GPU layer, which keeps showing the game, and the recorder compares the two
// (diag/native_validate.cpp). The resources are prepared by render/native_resources.h: game
// memory buffers, the flattened user data, LDS, dynamic loads, images and samplers; images the
// dispatch writes come back linear, for comparison with the GPU layer's.

namespace render {

struct ComputeDispatch {
    translate::ComputeProgram program;
    std::array<u32, 3> groups{}; // DISPATCH_DIRECT dimensions (workgroups)
};

// What a dispatch did (the written buffers and images, ResourceReport) and how it went.
struct NativeComputeResult : ResourceReport {
    u64 hash = 0;
    std::string skipped; // set when the dispatch uses something not supported yet
    std::string error;   // set when something failed
    translate::TranslatedProgram program; // what ran (for diagnostics)
};

// Runs the dispatch; on the GPU command processor thread (see TranslateCompute).
void RunNativeCompute(const ComputeDispatch& dispatch, const GuestReader& read,
                      const LayerImageReader& read_image, NativeComputeResult& result);

} // namespace render
