#pragma once

#include <cstddef>
#include <iterator>

#include "common/types.h"

// The frontier between the game's renderer and the GPU: the functions of Sony's Gnm/Gnmx code
// (SDK 4.508, embedded in the executable between 0xc5c000 and 0xc9bb00) that the game calls to
// build GPU commands. Everything the GPU runs for the game comes out of these 88 functions —
// proven at run time by the renderer recorder (src/diag/gnm_recorder.cpp; analysis notes:
// renderizador/CATALOGO_GNM.md). The native renderer replaces them.
//
// Addresses are those of the executable as loaded at 0x400000. Names are the probable Gnm/Gnmx
// ones, deduced from the packets each function writes.

namespace render::frontier {

// --- addresses -----------------------------------------------------------------------------
constexpr u64 DrawIndexIndirect = 0xc6b3d0;
constexpr u64 DrawIndexOffset = 0xc6b040;
constexpr u64 DrawIndexAuto = 0xc6ac10;
constexpr u64 DrawIndirect = 0xc6b150;
constexpr u64 Dispatch = 0xc6b9e0;
constexpr u64 SetIndexBuffer = 0xc6bf50;
constexpr u64 SetIndexCount = 0xc6bfb0;
constexpr u64 SetIndexSize = 0xc6bed0;
constexpr u64 SetNumInstances = 0xc6c000;
constexpr u64 SetBaseIndirectArgs = 0xc6ad40;
constexpr u64 SetRenderTarget = 0xc5cbc0;
constexpr u64 SetDepthRenderTarget = 0xc6a290;
constexpr u64 SetBlendControl = 0xc69a50;
constexpr u64 SetRenderTargetMask = 0xc69990;
constexpr u64 SetCbControl = 0xc69ce0;
constexpr u64 SetDepthStencilControl = 0xc69d60;
constexpr u64 SetStencilOpControl = 0xc69dc0;
constexpr u64 SetStencil = 0xc69bb0;
constexpr u64 SetDepthClearValue = 0xc69f50;
constexpr u64 SetDbRenderControl = 0xc6dd20;
constexpr u64 SetDbRenderOverride = 0xc6be10;
constexpr u64 SetPrimitiveSetup = 0xc6e1a0;
constexpr u64 SetPolygonOffsetFront = 0xc6e410;
constexpr u64 SetPolygonOffsetBack = 0xc6e480;
constexpr u64 SetPolygonOffsetZFormat = 0xc6e3a0;
constexpr u64 SetScanModeControl = 0xc6e4f0;
constexpr u64 SetClipControl = 0xc6ded0;
constexpr u64 SetupScreenViewport = 0xc5de40;
constexpr u64 SetViewportScissor = 0xc6d890;
constexpr u64 SetPrimitiveType = 0xc6a900;
constexpr u64 SetActiveShaderStages = 0xc6d160;
constexpr u64 SetPsShaderUsage = 0xc6bc00;
constexpr u64 SetBorderColorTableAddr = 0xc6c840;
constexpr u64 InitializeDefaultHardwareState = 0xc69290;
constexpr u64 WriteAtEndOfPipe = 0xc6eb60;
constexpr u64 WriteAtEndOfShader = 0xc6f5e0;
constexpr u64 TriggerEvent = 0xc6ced0;
constexpr u64 WaitOnAddress = 0xc6edc0;
constexpr u64 WaitOnAddress2 = 0xc6eec0;
constexpr u64 WaitForFlipDone = 0xc6f700;
constexpr u64 WaitForGraphicsWrites = 0xc6cf40;
constexpr u64 FlushShaderCachesAndWait = 0xc6f1d0;
constexpr u64 SetPredication = 0xc6e800;
constexpr u64 PushMarker = 0xc6f2a0;
constexpr u64 PopMarker = 0xc6f3a0;
constexpr u64 AllocateFromCommandBuffer = 0xc6d210;
constexpr u64 GetLodStats = 0xc6a880;
constexpr u64 ComputeDispatch = 0xc83870;
constexpr u64 ComputeDispatchIndirect = 0xc83950;
constexpr u64 ComputeWriteReleaseMemEvent = 0xc846f0;
constexpr u64 ComputeWaitOnAddress = 0xc84d40;
constexpr u64 ComputePushMarker = 0xc84f10;
constexpr u64 ComputePopMarker = 0xc85010;
constexpr u64 ComputeNop3 = 0xc83f80;
constexpr u64 ComputeInitializeDefaultHardwareState = 0xc830e0;
constexpr u64 SetVsShader = 0xc61520;
constexpr u64 SetPsShader = 0xc61600;
constexpr u64 SetCsShader = 0xc616f0;
constexpr u64 SetTextures = 0xc61fe0;
constexpr u64 SetBuffers = 0xc61ce0;
constexpr u64 SetRwTextures = 0xc62160;
constexpr u64 SetRwBuffers = 0xc61e60;
constexpr u64 SetSamplers = 0xc622e0;
constexpr u64 SetConstantBuffers = 0xc619a0;
constexpr u64 SetSrtPointer = 0xc62490;
constexpr u64 SetAppendConsumeRange = 0xc618e0;
constexpr u64 SetIndirectOffsets = 0xc62510;
constexpr u64 PreDraw = 0xc5fc10;
constexpr u64 PreDispatch = 0xc5f220;
constexpr u64 ComputeSetCsShader = 0xc5e7a0;
constexpr u64 ComputePreDispatch = 0xc5e5d0;
constexpr u64 ComputeSetBuffers = 0xc5e8e0;
constexpr u64 ComputeSetRwBuffers = 0xc5e9a0;
constexpr u64 ComputeSetRwTextures = 0xc5eb20;
constexpr u64 ComputeSetConstantBuffers = 0xc5e820;
constexpr u64 ComputeSetSrtPointer = 0xc5ecd0;
constexpr u64 ComputeSetAppendConsumeRange = 0xc5e7e0;
constexpr u64 ComputeSetTextures = 0xc5ea60;
constexpr u64 ComputeSetSamplers = 0xc5ebe0;
constexpr u64 InitializeContext = 0xc62690;
constexpr u64 ResetContext = 0xc62900;
constexpr u64 Submit = 0xc636e0;
constexpr u64 SubmitDirect = 0xc63310;
constexpr u64 SubmitAndFlip = 0xc63aa0;
constexpr u64 ComputeResetContext = 0xc5d0c0;
constexpr u64 ComputeSubmit = 0xc5d600;
constexpr u64 ComputeRingIndirectBuffer = 0xc84c40;
constexpr u64 ComputeInitializeQueue = 0xc5d300;

// --- what each function touches --------------------------------------------------------------

// How a function reaches the command buffers it may write to. A command buffer object starts
// with {u32* begin, u32* end, u32* cursor} (Gnm's DrawCommandBuffer/DispatchCommandBuffer).
enum class Buffers : u8 {
    None,
    Self,           // the first argument is the command buffer
    GraphicsLcue,   // graphics LCUE (context + 0x2e8): command buffer pointers at +0xc090 and
                    // +0xc098 (the second one is used when the VS also runs as an ES)
    ComputeLcue,    // compute LCUE: command buffer pointer at +0x90
    ComputeContext, // compute context: its command buffer at +0x10
    ComputeQueue,   // asynchronous compute queue: its ring at +0x18
};

enum class Role : u8 {
    Normal,
    // (context, memory, size, ...): the context's command buffer starts over in
    // [memory, memory + size), so whatever was marked there belongs to an earlier use.
    InitBuffer,
    Flip, // ends a frame
};

struct FrontierFunction {
    u64 address;
    const char* name;
    Buffers buffers;
    Role role = Role::Normal;
};

inline constexpr FrontierFunction Functions[] = {
    // C: DrawCommandBuffer methods.
    {DrawIndexIndirect, "drawIndexIndirect", Buffers::Self},
    {DrawIndexOffset, "drawIndexOffset", Buffers::Self},
    {DrawIndexAuto, "drawIndexAuto", Buffers::Self},
    {DrawIndirect, "drawIndirect", Buffers::Self},
    {Dispatch, "dispatch", Buffers::Self},
    {SetIndexBuffer, "setIndexBuffer", Buffers::Self},
    {SetIndexCount, "setIndexCount", Buffers::Self},
    {SetIndexSize, "setIndexSize", Buffers::Self},
    {SetNumInstances, "setNumInstances", Buffers::Self},
    {SetBaseIndirectArgs, "setBaseIndirectArgs", Buffers::Self},
    {SetRenderTarget, "setRenderTarget", Buffers::Self},
    {SetDepthRenderTarget, "setDepthRenderTarget", Buffers::Self},
    {SetBlendControl, "setBlendControl", Buffers::Self},
    {SetRenderTargetMask, "setRenderTargetMask", Buffers::Self},
    {SetCbControl, "setCbControl", Buffers::Self},
    {SetDepthStencilControl, "setDepthStencilControl", Buffers::Self},
    {SetStencilOpControl, "setStencilOpControl", Buffers::Self},
    {SetStencil, "setStencil", Buffers::Self},
    {SetDepthClearValue, "setDepthClearValue", Buffers::Self},
    {SetDbRenderControl, "setDbRenderControl", Buffers::Self},
    {SetDbRenderOverride, "setDbRenderOverride", Buffers::Self},
    {SetPrimitiveSetup, "setPrimitiveSetup", Buffers::Self},
    {SetPolygonOffsetFront, "setPolygonOffsetFront", Buffers::Self},
    {SetPolygonOffsetBack, "setPolygonOffsetBack", Buffers::Self},
    {SetPolygonOffsetZFormat, "setPolygonOffsetZFormat", Buffers::Self},
    {SetScanModeControl, "setScanModeControl", Buffers::Self},
    {SetClipControl, "setClipControl", Buffers::Self},
    {SetupScreenViewport, "setupScreenViewport", Buffers::Self},
    {SetViewportScissor, "setViewportScissor", Buffers::Self},
    {SetPrimitiveType, "setPrimitiveType", Buffers::Self},
    {SetActiveShaderStages, "setActiveShaderStages", Buffers::Self},
    {SetPsShaderUsage, "setPsShaderUsage", Buffers::Self},
    {SetBorderColorTableAddr, "setBorderColorTableAddr", Buffers::Self},
    {InitializeDefaultHardwareState, "initializeDefaultHardwareState", Buffers::Self},
    {WriteAtEndOfPipe, "writeAtEndOfPipe", Buffers::Self},
    {WriteAtEndOfShader, "writeAtEndOfShader", Buffers::Self},
    {TriggerEvent, "triggerEvent", Buffers::Self},
    {WaitOnAddress, "waitOnAddress", Buffers::Self},
    {WaitOnAddress2, "waitOnAddress2", Buffers::Self},
    {WaitForFlipDone, "waitForFlipDone", Buffers::Self},
    {WaitForGraphicsWrites, "waitForGraphicsWrites", Buffers::Self},
    {FlushShaderCachesAndWait, "flushShaderCachesAndWait", Buffers::Self},
    {SetPredication, "setPredication", Buffers::Self},
    {PushMarker, "pushMarker", Buffers::Self},
    {PopMarker, "popMarker", Buffers::Self},
    {AllocateFromCommandBuffer, "allocateFromCommandBuffer", Buffers::Self},
    {GetLodStats, "getLodStats", Buffers::Self},
    // C: DispatchCommandBuffer methods (asynchronous compute).
    {ComputeDispatch, "compute.dispatch", Buffers::Self},
    {ComputeDispatchIndirect, "compute.dispatchIndirect", Buffers::Self},
    {ComputeWriteReleaseMemEvent, "compute.writeReleaseMemEvent", Buffers::Self},
    {ComputeWaitOnAddress, "compute.waitOnAddress", Buffers::Self},
    {ComputePushMarker, "compute.pushMarker", Buffers::Self},
    {ComputePopMarker, "compute.popMarker", Buffers::Self},
    {ComputeNop3, "compute.nop3", Buffers::Self},
    {ComputeInitializeDefaultHardwareState, "compute.initializeDefaultHardwareState", Buffers::Self},
    // B: graphics LCUE.
    {SetVsShader, "setVsShader", Buffers::GraphicsLcue},
    {SetPsShader, "setPsShader", Buffers::GraphicsLcue},
    {SetCsShader, "setCsShader", Buffers::GraphicsLcue},
    {SetTextures, "setTextures", Buffers::GraphicsLcue},
    {SetBuffers, "setBuffers", Buffers::GraphicsLcue},
    {SetRwTextures, "setRwTextures", Buffers::GraphicsLcue},
    {SetRwBuffers, "setRwBuffers", Buffers::GraphicsLcue},
    {SetSamplers, "setSamplers", Buffers::GraphicsLcue},
    {SetConstantBuffers, "setConstantBuffers", Buffers::GraphicsLcue},
    {SetSrtPointer, "setSrtPointer", Buffers::GraphicsLcue},
    {SetAppendConsumeRange, "setAppendConsumeRange", Buffers::GraphicsLcue},
    {SetIndirectOffsets, "setIndirectOffsets", Buffers::GraphicsLcue},
    {PreDraw, "preDraw", Buffers::GraphicsLcue},
    {PreDispatch, "preDispatch", Buffers::GraphicsLcue},
    // B: compute LCUE.
    {ComputeSetCsShader, "compute.setCsShader", Buffers::ComputeLcue},
    {ComputePreDispatch, "compute.preDispatch", Buffers::ComputeLcue},
    {ComputeSetBuffers, "compute.setBuffers", Buffers::ComputeLcue},
    {ComputeSetRwBuffers, "compute.setRwBuffers", Buffers::ComputeLcue},
    {ComputeSetRwTextures, "compute.setRwTextures", Buffers::ComputeLcue},
    {ComputeSetConstantBuffers, "compute.setConstantBuffers", Buffers::ComputeLcue},
    {ComputeSetSrtPointer, "compute.setSrtPointer", Buffers::ComputeLcue},
    {ComputeSetAppendConsumeRange, "compute.setAppendConsumeRange", Buffers::ComputeLcue},
    {ComputeSetTextures, "compute.setTextures", Buffers::ComputeLcue},
    {ComputeSetSamplers, "compute.setSamplers", Buffers::ComputeLcue},
    // D: life cycle, submission and flip.
    {InitializeContext, "initializeContext", Buffers::Self, Role::InitBuffer},
    {ResetContext, "resetContext", Buffers::Self},
    {Submit, "submit", Buffers::Self},
    {SubmitDirect, "submitDirect", Buffers::Self},
    {SubmitAndFlip, "submitAndFlip", Buffers::Self, Role::Flip},
    {ComputeResetContext, "compute.resetContext", Buffers::ComputeContext, Role::InitBuffer},
    {ComputeSubmit, "compute.submit", Buffers::ComputeQueue},
    {ComputeRingIndirectBuffer, "compute.ringIndirectBuffer", Buffers::Self},
    {ComputeInitializeQueue, "compute.initializeQueue", Buffers::None},
};

constexpr size_t FunctionCount = std::size(Functions);

} // namespace render::frontier
