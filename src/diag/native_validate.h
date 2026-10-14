#pragma once

#include <cstdio>

#include "common/types.h"
#include "render/state_model.h"

// Differential check of the native renderer against the GPU layer, dispatch by dispatch
// (needs SOTC_RECORD_GNM=1, which feeds the state model):
//
//   SOTC_VALIDATE_NATIVE=all            every compute shader
//   SOTC_VALIDATE_NATIVE=50784801,...   only these shaders (hash from the OrbShdr footer)
//   SOTC_VALIDATE_NATIVE_COUNT=3        dispatches checked per shader (default 3)
//
// When the GPU layer reaches a selected dispatch (the recorder's packet observer, before the
// layer runs it), the GPU layer's results so far are brought into game memory, and the native
// renderer runs the same dispatch on its own device from the model's state
// (render/native_compute). At the next packet the layer has recorded the dispatch: the bytes it
// wrote are brought into game memory too and compared with the native renderer's. Every check
// waits for the GPU, so the game slows down while checks are running. Results go to the
// recorder's report.

namespace diag {

bool NativeValidationEnabled();

// At a DISPATCH_DIRECT / DISPATCH_INDIRECT packet, with the model's state for it.
// `compute_queue`: the packet runs on an asynchronous compute queue (its DISPATCH_INDIRECT
// carries the arguments' address; on the graphics queue it is an offset from SET_BASE).
void ValidateDispatchBefore(const render::RegisterSnapshot& snapshot, const u32* packet,
                            bool indirect, bool compute_queue);

// At a draw packet (graphics queue), with the model's state for it (SOTC_VALIDATE_NATIVE_DRAWS:
// "all" or vertex/pixel shader hashes). The draw runs on the native renderer
// (render/native_draw) with each color target as the GPU layer holds it; at the next packet the
// targets are compared with the GPU layer's.
void ValidateDrawBefore(const render::RegisterSnapshot& snapshot, const u32* packet);

// At every packet, before anything else: completes the check of the previous dispatch or draw.
void ValidateAfterPacket();

void WriteNativeValidationReport(std::FILE* file);

} // namespace diag
