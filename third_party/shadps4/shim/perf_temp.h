#pragma once

// sotcport TEMP: time spent in parts of the GPU layer and in the game's waits, logged every
// 5 s when SOTC_PERF is set (gpu_glue.cpp).

#include <cstdint>

namespace perf {

enum Slot : int {
    GpuCommBusy,
    Draw,
    Dispatch,
    Readback,
    SchedulerWait,
    NewPipeline,
    SubmitFlush,
    Invalidate,
    Flip,
    WaitDownload,
    WaitStream,
    WaitFaultBuffer,
    WaitImageDownload,
    EmptyDownload,
    TextureUpload,
    NumSlots,
};

bool Enabled();
uint64_t Now();
void Add(Slot slot, uint64_t ns);
// Time a thread of the game spent blocked in `kind` (equeue, event flag...).
void AddWait(const char* kind, uint64_t ns);
// The game code address of the access fault being handled on this thread (0: none), so the
// readbacks it causes can be attributed (set by the fault handler).
void SetFaultSite(uint64_t game_address);
// A readback of [address, +size) (is_write: the CPU is about to write there).
void NoteReadback(uint64_t address, uint64_t size, bool is_write);
// One SRT walker run: guest reads it logged, and whether any needed a readback.
void NoteSrtReads(uint64_t reads, bool fetched);
// GPU-written dwords of a walk left to the GPU (Shader::Info::srt_patches).
void NoteSrtPatches(uint64_t patches);
// A fill/copy dispatch computed on the CPU too (Rasterizer::ShadowCopyShader).
void NoteShadowCopy(uint64_t bytes);
// One SRT walker readback of 8 bytes at `address`: whether the downloaded data differed.
void NoteSrtReadback(uint64_t address, bool changed);
// A WAIT_REG_MEM that drained stage B: polled address, whether it was met right after.
void NoteWaitRegMem(uint64_t address, bool met, bool equal, uint32_t ref);
// Time between two presented frames (uncapped present loop), for the frame pacing report.
void NoteFrameInterval(uint64_t ns);
// Called at every flip: with SOTC_SPIKES=<ms>, logs the waits of each frame longer than <ms>.
void NoteFrameForSpikes();
// TEMP timeline (SOTC_FRAME_TIMELINE=<n>): events stamped by any thread; every n-th presented
// frame, the events since the previous one are logged.
void Mark(char kind, uint64_t value = 0);
// Device-local memory in use (VK_EXT_memory_budget) and the texture cache's garbage collector
// thresholds, all in bytes; refreshed at every submit by TextureCache::GarbageCollectImages.
void NoteVram(uint64_t used, uint64_t pressure, uint64_t critical);
// Downloaded GPU data about to replace [address, +size): counts the bytes that actually change.
void NoteDownload(uint64_t address, const uint8_t* data, uint64_t size);

struct Scope {
    explicit Scope(Slot slot_) : slot{slot_}, start{Enabled() ? Now() : 0} {}
    ~Scope() {
        if (start) {
            Add(slot, Now() - start);
        }
    }
    Slot slot;
    uint64_t start;
};

} // namespace perf
