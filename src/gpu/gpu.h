#pragma once

#include <cstdint>
#include <string>

namespace render::translate {
struct ImageBinding;
}

// The GPU layer: window, VideoOut and Gnm (command buffers and shaders translated to Vulkan).
// For now this is shadPS4's renderer (third_party/shadps4), a temporary solution until the
// game's renderer gets a native interface of its own.

namespace gpu {

// Creates the window and the renderer, and registers the Gnm and VideoOut functions. Runs on the
// main thread, before the game is linked (the renderer must exist before the game maps memory).
bool Initialize(int sdk_version);

// `sotcport --list-displays`: one line per monitor, tab separated: its number (what SOTC_DISPLAY
// takes, 1 = the first), name, desktop size (<w>x<h>), refresh rate in Hz and 1 for the primary
// one. For the launcher. Returns the exit status.
int PrintDisplays();

// Handles window events for up to `timeout_ms`. Returns false once the window was closed.
bool ProcessEvents(int timeout_ms);

// Saves the next frame the game shows as a PNG (in <data>/gpu/screenshots; the log says where).
void RequestScreenshot();

// Waits (up to a few seconds) for the screenshots asked for to be written, before exiting.
void FinishScreenshots();

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
// The whole register file (0xD000 dwords, ReadRegister's indices) as it is right now, and the
// render target extents Gnm passes in NOPs (AmdGpu::CbDbExtent, raw): what the next draw sees,
// from the packet observer. For the native renderer's validation (diag/native_validate.cpp).
void ReadRegisterFile(std::uint32_t* out);
void ReadRenderTargetExtents(std::uint32_t* color /* 8 */, std::uint32_t& depth);
// Whether the GPU layer holds a fast clear it has not applied yet of the target whose metadata
// (a color target's CMASK, a depth target's HTILE) is at `meta_address` (slice `slice`), when it
// binds the target for the next draw. `tracked`: the layer tracks the target's metadata (it then
// registers metadata it does not know yet, and new metadata starts cleared: the first draw to a
// new target clears it).
bool IsFastCleared(std::uint64_t meta_address, std::uint32_t slice, bool tracked);
// Makes game memory [address, address + size) hold what the GPU layer's GPU wrote there:
// finishes the queued GPU work and downloads it (waits). From the packet observer only.
void SyncGuestRange(std::uint64_t address, std::uint64_t size);
// Validation of the native renderer (diag/native_validate.cpp): the contents of the GPU layer's
// own image for `binding`, copied out linearly with the binding's copies into `out`
// (binding.guest_size bytes; only the rows ImageBinding::ForEachRow lists are written). The
// layer's image starts at the same address with the same extent and texel size and the same
// layout for the mips both have; it may have fewer or more mips than the binding (a T# per mip
// of a mip chain describes the image with last_level = that mip, and the layer keeps one image
// for them all). It is first brought up to date the way binding it does (CPU writes uploaded).
// `newer_only`: only an image that then holds GPU writes, which game memory lacks. Returns how
// many of the binding's mips (from 0) were copied: 0, with `why`, when none.
// From the packet observer only; waits for the GPU.
std::uint32_t ReadLayerImage(const render::translate::ImageBinding& binding, bool newer_only,
                             std::uint8_t* out, std::string& why);
// The stencil plane of the GPU layer's depth image for `binding` (found as ReadLayerImage does):
// binding.StencilBytes() bytes. False, with `why`, when it has no such image or no stencil.
bool ReadLayerStencil(const render::translate::ImageBinding& binding, std::uint8_t* out,
                      std::string& why);
// The GPU layer's images that overlap the binding's memory, one line each (offset from it,
// format, extent, texel size, tiling, size, mips, flags, and which one ReadLayerImage takes);
// for dumps.
std::string DescribeLayerImages(const render::translate::ImageBinding& binding);
// Whether the GPU layer's buffer cache holds GPU writes in [address, address + size) that game
// memory may lack (for dumps).
bool IsGpuModified(std::uint64_t address, std::uint64_t size);
// What the GPU layer's buffer arena holds for [address, address + size) (for dumps: may differ
// from game memory when its tracking missed a write). False when not resident there.
bool PeekLayerBuffer(std::uint64_t address, std::uint64_t size, std::uint8_t* out);

} // namespace gpu
