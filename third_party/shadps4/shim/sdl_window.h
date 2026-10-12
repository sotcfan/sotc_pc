#pragma once

// Replaces shadPS4's window: an SDL3 window with what its Vulkan code needs.

#include <string_view>

#include "common/types.h"

struct SDL_Window;

namespace Frontend {

enum class WindowSystemType : u8 {
    Headless,
    Windows,
    X11,
    Wayland,
    Metal,
};

struct WindowSystemInfo {
    void* display_connection = nullptr;
    void* render_surface = nullptr;
    float render_surface_scale = 1.0f;
    WindowSystemType type = WindowSystemType::Headless;
};

class WindowSDL {
public:
    WindowSDL(s32 width, s32 height, std::string_view title);
    ~WindowSDL();

    s32 GetWidth() const {
        return width;
    }
    s32 GetHeight() const {
        return height;
    }
    WindowSystemInfo GetWindowInfo() const {
        return info;
    }

    // Handles pending window events, waiting up to `timeout_ms` for one. Returns false once the
    // window has been closed.
    bool ProcessEvents(int timeout_ms);

private:
    SDL_Window* window = nullptr;
    WindowSystemInfo info;
    s32 width;
    s32 height;
};

} // namespace Frontend
