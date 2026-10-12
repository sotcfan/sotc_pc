#pragma once

// Replaces shadPS4's settings system. Only the settings its GPU code reads, with the values the
// game was run with in shadPS4 (its defaults, plus readbacks_mode = Precise from the game's own
// configuration file).

#include <cstdlib>
#include <memory>
#include <string>

#include "common/types.h"

enum GpuReadbacksMode : int {
    Disabled,
    Relaxed,
    Precise,
};

class EmulatorSettingsImpl {
public:
    static std::shared_ptr<EmulatorSettingsImpl> GetInstance() {
        static const auto instance = std::make_shared<EmulatorSettingsImpl>();
        return instance;
    }

    s32 GetGpuId() const { return -1; }
    u32 GetInternalScreenWidth() const { return 1280; }
    u32 GetInternalScreenHeight() const { return 720; }
    std::string GetPresentMode() const { return "Mailbox"; }
    int GetRcasAttenuation() const { return 250; }
    u32 GetReadbacksMode() const { return GpuReadbacksMode::Precise; }
    u32 GetVblankFrequency() const { return 60; }

    bool IsCopyGpuBuffers() const { return false; }
    bool IsDirectMemoryAccessEnabled() const { return false; }
    // SOTC_DUMP_SHADERS=1: dump each shader (GCN binary, IR, SPIR-V) to <data>/gpu/shader/dumps.
    bool IsDumpShaders() const { return std::getenv("SOTC_DUMP_SHADERS") != nullptr; }
    bool IsFsrEnabled() const { return false; }
    bool IsHdrAllowed() const { return false; }
    bool IsInlineFetchShader() const { return false; }
    bool IsNullGPU() const { return false; }
    bool IsPatchShaders() const { return false; }
    bool IsPipelineCacheArchived() const { return false; }
    // sotcport: shaders translated in earlier sessions are loaded at startup instead of
    // being translated in the middle of play (cache_storage.cpp). SOTC_NO_SHADER_CACHE=1
    // turns it off.
    bool IsPipelineCacheEnabled() const {
        return std::getenv("SOTC_NO_SHADER_CACHE") == nullptr;
    }
    bool IsRcasEnabled() const { return true; }
    bool IsReadbackLinearImagesEnabled() const { return false; }
    // SOTC_RENDERDOC="seconds,...": load RenderDoc and capture the frames at these times.
    bool IsRenderdocEnabled() const { return std::getenv("SOTC_RENDERDOC") != nullptr; }
    bool IsShaderCollect() const { return false; }
    bool IsUserfaultfdTracking() const { return false; }
    bool IsVkCrashDiagnosticEnabled() const { return false; }
    bool IsVkGuestMarkersEnabled() const { return false; }
    bool IsVkHostMarkersEnabled() const { return false; }
    bool IsVkValidationEnabled() const { return false; }
    bool IsVkValidationCoreEnabled() const { return true; }
    bool IsVkValidationGpuEnabled() const { return false; }
    bool IsVkValidationSyncEnabled() const { return false; }
};

#define EmulatorSettings (*EmulatorSettingsImpl::GetInstance())
