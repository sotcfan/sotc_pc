#pragma once

#include <array>
#include <functional>
#include <string>
#include <vector>

#include "common/types.h"
#include "render/shader_translate.h"

// The native renderer running one compute dispatch on its own Vulkan device (render/vulkan),
// from the state the model knows: the COMPUTE_* registers and the dispatch size. The shader is
// translated for the current game memory; the buffers it uses are copied from game memory into
// buffers of the native device, the dispatch runs, and the bytes of every buffer it may write
// come back.
//
// This is the first piece of step 2 of the native renderer (PLANO_RENDERIZADOR_NATIVO.md): it
// runs next to the GPU layer, which keeps showing the game, and the recorder compares the two
// (diag/native_validate.cpp). Supported for now: buffers only (game memory, the flattened user
// data, LDS in memory); shaders with images or samplers are skipped.

namespace render {

struct ComputeDispatch {
    translate::ComputeProgram program;
    std::array<u32, 3> groups{}; // DISPATCH_DIRECT dimensions (workgroups)
};

// Reads game memory as the GPU sees it right now (the caller syncs it first).
using GuestReader = std::function<bool(u64 address, void* out, u64 size)>;

struct WrittenRange {
    u64 address = 0;
    std::vector<u8> bytes; // the range after the native dispatch
    u64 changed = 0;       // bytes the dispatch changed (a check that compares nothing is weak)
};

struct NativeComputeResult {
    u64 hash = 0;
    std::vector<WrittenRange> written;
    std::string skipped; // set when the dispatch uses something not supported yet
    std::string error;   // set when something failed
    bool clamped = false; // a buffer was larger than the native renderer copies (see .cpp)
    translate::TranslatedProgram program; // what ran (for diagnostics)
};

// Runs the dispatch; on the GPU command processor thread (see TranslateCompute).
void RunNativeCompute(const ComputeDispatch& dispatch, const GuestReader& read,
                      NativeComputeResult& result);

} // namespace render
