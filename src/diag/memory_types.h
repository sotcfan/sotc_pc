#pragma once

#include "common/types.h"

// Which kinds of game memory (WB_ONION, WC_GARLIC, flexible) the GPU layer touches, and how
// (SOTC_MEMORY_TYPES=1; logged every 10 s). Groundwork for the unified memory model, where the
// game's memory becomes the GPU's memory: with the Garlic in video memory every CPU read of it
// crosses PCIe uncached (~43 MB/s measured), so whatever the GPU layer reads on the CPU (command
// buffers, SRT tables, labels) has to live in cached system memory. Also records how the game
// maps memory (sizes and alignments), since the GPU's page granularity is 64 KiB.

namespace diag {

enum class MemAccess : u32 {
    GfxCommands,      ///< graphics command buffers (DCB), read by the command processor
    CeCommands,       ///< constant engine command buffers (CCB)
    ComputeCommands,  ///< compute queue command buffers
    IndirectCommands, ///< INDIRECT_BUFFER targets (both queues)
    SrtRead,          ///< 8-byte reads of the SRT walker
    Label,            ///< fence/WriteData/ReleaseMem destinations
    BufferRead,       ///< buffers bound for the GPU to read
    BufferWritten,    ///< buffers bound for the GPU to write
    ImageBind,        ///< textures and render targets bound
    UploadRead,       ///< CPU copies of guest memory into staging (buffer uploads)
    Count,
};

bool MemoryTypesEnabled();

// Set from SOTC_MEMORY_TYPES before anything runs; read inline so that the calls in hot paths
// (every SRT walker read, every buffer bound) cost a load and a branch when it is off.
extern const bool g_memory_types_enabled;

void NoteMemoryAccessSlow(MemAccess kind, u64 address, u64 bytes);

// Counts an access of `bytes` at `address`.
inline void NoteMemoryAccess(MemAccess kind, u64 address, u64 bytes) {
    if (g_memory_types_enabled) [[unlikely]] {
        NoteMemoryAccessSlow(kind, address, bytes);
    }
}

// A game mapping of physical memory: counted by size and alignment (address and physical).
void NoteGameMapping(u64 address, u64 size, u64 physical, bool flexible);

// Starts the report thread. Does nothing without the variable.
void InstallMemoryTypes();

} // namespace diag
