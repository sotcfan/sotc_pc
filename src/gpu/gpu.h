#pragma once

#include <cstdint>

// The GPU layer: window, VideoOut and Gnm (command buffers and shaders translated to Vulkan).
// For now this is shadPS4's renderer (third_party/shadps4), a temporary solution until the
// game's renderer gets a native interface of its own.

namespace gpu {

// Creates the window and the renderer, and registers the Gnm and VideoOut functions. Runs on the
// main thread, before the game is linked (the renderer must exist before the game maps memory).
bool Initialize(int sdk_version);

// Handles window events for up to `timeout_ms`. Returns false once the window was closed.
bool ProcessEvents(int timeout_ms);

// Saves the next frame the game shows as a PNG (in <data>/gpu/screenshots; the log says where).
void RequestScreenshot();

// How many frames have been presented so far (for the freeze watchdog in main.cpp).
unsigned long long FramesPresented();

// Captures the next frame in RenderDoc (<data>/gpu/captures), when it was loaded at startup
// (SOTC_RENDERDOC set).
void CaptureFrame();

// Diagnostics: `observer` sees every PM4 packet the GPU executes (address in game memory, size
// in dwords with the header, queue: 0 graphics, 1 constant engine, 2 compute). Set before the
// game starts; not thread-safe to change later.
using PacketObserver = void (*)(const std::uint32_t* packet, std::uint32_t dwords,
                                std::uint32_t queue);
void SetPacketObserver(PacketObserver observer);

// Diagnostics: a register of the GPU layer's register file (shadPS4's reg_array index: context
// registers at 0xA000 + offset, SH at 0x2C00, uconfig at 0xC000) as it is right now. Only
// meaningful on the GPU thread, from inside a packet observer.
std::uint32_t ReadRegister(std::uint32_t index);
// A compute register (COMPUTE_*, 0x2E00-0x2E4F) of the queue whose commands are running; only
// meaningful from the packet observer (the GPU layer keeps one set per queue).
std::uint32_t ReadComputeRegister(std::uint32_t index);
// The base address of indirect draw arguments (set by SET_BASE; not a register).
std::uint64_t ReadIndirectArgsAddress();
// Makes game memory [address, address + size) hold what the GPU layer's GPU wrote there:
// finishes the queued GPU work and downloads it (waits). From the packet observer only.
void SyncGuestRange(std::uint64_t address, std::uint64_t size);

} // namespace gpu
