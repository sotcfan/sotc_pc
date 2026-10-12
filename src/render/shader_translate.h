#pragma once

#include <array>
#include <string>
#include <vector>

#include "common/types.h"

// Translation of the game's GCN shaders to SPIR-V for the native renderer, through the shader
// recompiler we already have (shadPS4's, used as a tool). The implementation lives with the GPU
// layer (third_party/shadps4/shim/recompiler_glue.cpp) because the recompiler's headers belong
// to it; this interface keeps the native renderer free of them.
//
// The SPIR-V follows the recompiler's binding conventions, which the native renderer mirrors:
// descriptor set 0 with one binding per resource in this order — every buffer (storage buffer),
// then images, then samplers — and a push constant block (Shader::PushData, 128 bytes) whose
// bytes 16 on hold, per buffer binding, how far the bound offset was moved down to meet the
// device's alignment.

namespace render::translate {

// What a compute dispatch runs, as the state model knows it (COMPUTE_* registers).
struct ComputeProgram {
    u64 code_address = 0; // COMPUTE_PGM_LO << 8 (PGM_HI is always 0)
    u32 rsrc1 = 0;
    u32 rsrc2 = 0;
    std::array<u32, 3> num_threads{}; // COMPUTE_NUM_THREAD_X/Y/Z
    std::array<u32, 16> user_data{};  // COMPUTE_USER_DATA_0..15
};

enum class BufferKind : u8 {
    Guest,        // game memory described by a V#: [address, address + size)
    Flatbuf,      // the user data and SRT data the shader reads, flattened (TranslatedProgram::flatbuf)
    SharedMemory, // LDS emulated in a buffer (shared_memory_bytes per workgroup), zeroed
    Unsupported,  // something the native renderer does not provide yet (see `name`)
};

struct BufferBinding {
    BufferKind kind = BufferKind::Unsupported;
    u64 address = 0;
    u64 size = 0;
    bool written = false;
    bool formatted = false; // read or written with buffer_load/store_format (typed)
    const char* name = "";  // for Unsupported: which special buffer
};

// An image the shader uses (T#), as the native renderer needs to create and fill it.
struct ImageBinding {
    u64 address = 0;
    u32 width = 0, height = 0, depth = 0; // depth: 3D depth or array layers (T# depth + 1)
    u32 pitch = 0;
    u32 base_level = 0, last_level = 0;
    u32 type = 0;          // AmdGpu::ImageType (9 1D, 10 2D, 11 3D, 13 cube, 14 1D array, 15 2D array)
    u32 data_format = 0;   // AmdGpu::DataFormat
    u32 number_format = 0; // AmdGpu::NumberFormat
    u32 tile_mode = 0;     // AmdGpu::TileMode (tiling_index): 8 = linear aligned
    bool written = false, atomic = false, depth_image = false;
};

struct TranslatedProgram {
    u64 hash = 0; // from the shader's OrbShdr footer
    std::vector<u32> spirv;
    std::vector<BufferBinding> buffers; // in binding order
    u32 images = 0;   // bindings after the buffers (not supported by the native renderer yet)
    u32 samplers = 0;
    std::vector<ImageBinding> image_bindings; // in binding order, after the buffers
    std::vector<u32> flatbuf;
    std::array<u32, 3> workgroup_size{};
    u32 shared_memory_bytes = 0;
};

// Translates a compute program for the current contents of game memory (the SPIR-V depends on
// the descriptors it reads: buffer formats, image types...). Only on the GPU command processor
// thread (the SRT walker it runs is not thread-safe). Returns false with `error` set.
bool TranslateCompute(const ComputeProgram& program, TranslatedProgram& out, std::string& error);

} // namespace render::translate
