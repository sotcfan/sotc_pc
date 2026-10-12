// The PC's keyboard and game controllers, combined into the state of one PS4 controller.
//
// Keyboard (key positions of a QWERTY keyboard, so they stay put on other layouts):
//   W A S D          left stick (move)          arrow keys    right stick (camera)
//   Space, Enter     Cross                      Backspace, F  Circle
//   E                Triangle                   Q             Square
//   Left Shift       R1                         Left Ctrl     L1
//   C                R2                         Z             L2
//   B                R3                         V             L3
//   I J K L          directional buttons        Escape        Options
//   Tab              touch pad button
//   F9               writes a marker to the log and saves a screenshot (for finding a
//                    moment of play later)
//   F10              captures the frame in RenderDoc (when started with SOTC_RENDERDOC)
//   F11              samples for 30 s which memory the game's code touches (perf)
// SOTC_KEY_SCRIPT="20:Return,24.5:Return:0.3,60:W:30" presses keys by itself: seconds since the
// start, SDL key name and, optionally, how long to hold it (0.15 s by default). For reaching a
// scene and testing it without anyone at the keyboard.
// Controllers go through SDL's gamepad mapping, by button position: the bottom face button is
// Cross on any brand, Start is Options and Back (or a touch pad) is the touch pad button.

#include "input/input.h"

#include <SDL3/SDL.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <string>
#include <mutex>
#include <vector>

#include "common/log.h"
#include "diag/access_sampling.h"
#include "gpu/gpu.h"

namespace input {

namespace {

struct KeyButton {
    SDL_Scancode key;
    std::uint32_t button;
};

constexpr KeyButton KeyButtons[] = {
    {SDL_SCANCODE_SPACE, button::Cross},     {SDL_SCANCODE_RETURN, button::Cross},
    {SDL_SCANCODE_BACKSPACE, button::Circle}, {SDL_SCANCODE_F, button::Circle},
    {SDL_SCANCODE_E, button::Triangle},      {SDL_SCANCODE_Q, button::Square},
    {SDL_SCANCODE_LSHIFT, button::R1},       {SDL_SCANCODE_LCTRL, button::L1},
    {SDL_SCANCODE_C, button::R2},            {SDL_SCANCODE_Z, button::L2},
    {SDL_SCANCODE_B, button::R3},            {SDL_SCANCODE_V, button::L3},
    {SDL_SCANCODE_I, button::Up},            {SDL_SCANCODE_K, button::Down},
    {SDL_SCANCODE_J, button::Left},          {SDL_SCANCODE_L, button::Right},
    {SDL_SCANCODE_ESCAPE, button::Options},  {SDL_SCANCODE_TAB, button::TouchPad},
};

struct GamepadButton {
    SDL_GamepadButton sdl;
    std::uint32_t button;
};

constexpr GamepadButton GamepadButtons[] = {
    {SDL_GAMEPAD_BUTTON_SOUTH, button::Cross},
    {SDL_GAMEPAD_BUTTON_EAST, button::Circle},
    {SDL_GAMEPAD_BUTTON_WEST, button::Square},
    {SDL_GAMEPAD_BUTTON_NORTH, button::Triangle},
    {SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, button::L1},
    {SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, button::R1},
    {SDL_GAMEPAD_BUTTON_LEFT_STICK, button::L3},
    {SDL_GAMEPAD_BUTTON_RIGHT_STICK, button::R3},
    {SDL_GAMEPAD_BUTTON_START, button::Options},
    {SDL_GAMEPAD_BUTTON_BACK, button::TouchPad},
    {SDL_GAMEPAD_BUTTON_TOUCHPAD, button::TouchPad},
    {SDL_GAMEPAD_BUTTON_DPAD_UP, button::Up},
    {SDL_GAMEPAD_BUTTON_DPAD_DOWN, button::Down},
    {SDL_GAMEPAD_BUTTON_DPAD_LEFT, button::Left},
    {SDL_GAMEPAD_BUTTON_DPAD_RIGHT, button::Right},
};

// The PS4 reports L2/R2 as pressed buttons from this much travel on.
constexpr std::uint8_t TriggerButtonThreshold = 0x20;

std::mutex g_mutex;
PadState g_state;                     // what Read() returns
bool g_keys[SDL_SCANCODE_COUNT] = {}; // keyboard keys held, window thread only
std::vector<SDL_Gamepad*> g_gamepads; // open controllers, guarded by g_mutex

std::uint8_t StickValue(Sint16 axis) {
    return static_cast<std::uint8_t>((axis + 32768) >> 8);
}

std::uint8_t TriggerValue(Sint16 axis) {
    return static_cast<std::uint8_t>(std::max<int>(axis, 0) * 255 / 32767);
}

// Keeps whichever value is farthest from the center, so any controller can move the stick.
void TakeStronger(std::uint8_t& current, std::uint8_t candidate) {
    if (std::abs(candidate - 0x80) > std::abs(current - 0x80)) {
        current = candidate;
    }
}

// The stick position given by two opposite keys, or `fallback` if neither or both are held.
std::uint8_t KeyAxis(SDL_Scancode negative, SDL_Scancode positive, std::uint8_t fallback) {
    if (g_keys[negative] == g_keys[positive]) {
        return fallback;
    }
    return g_keys[negative] ? 0x00 : 0xFF;
}

// Recomputes the state from the keyboard and the controllers. Called with g_mutex held.
void Update() {
    PadState state;
    for (SDL_Gamepad* gamepad : g_gamepads) {
        for (const GamepadButton& mapping : GamepadButtons) {
            if (SDL_GetGamepadButton(gamepad, mapping.sdl)) {
                state.buttons |= mapping.button;
            }
        }
        TakeStronger(state.left_x, StickValue(SDL_GetGamepadAxis(gamepad, SDL_GAMEPAD_AXIS_LEFTX)));
        TakeStronger(state.left_y, StickValue(SDL_GetGamepadAxis(gamepad, SDL_GAMEPAD_AXIS_LEFTY)));
        TakeStronger(state.right_x, StickValue(SDL_GetGamepadAxis(gamepad, SDL_GAMEPAD_AXIS_RIGHTX)));
        TakeStronger(state.right_y, StickValue(SDL_GetGamepadAxis(gamepad, SDL_GAMEPAD_AXIS_RIGHTY)));
        state.l2 = std::max(state.l2, TriggerValue(SDL_GetGamepadAxis(gamepad, SDL_GAMEPAD_AXIS_LEFT_TRIGGER)));
        state.r2 = std::max(state.r2, TriggerValue(SDL_GetGamepadAxis(gamepad, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER)));
    }

    for (const KeyButton& mapping : KeyButtons) {
        if (g_keys[mapping.key]) {
            state.buttons |= mapping.button;
        }
    }
    state.left_x = KeyAxis(SDL_SCANCODE_A, SDL_SCANCODE_D, state.left_x);
    state.left_y = KeyAxis(SDL_SCANCODE_W, SDL_SCANCODE_S, state.left_y);
    state.right_x = KeyAxis(SDL_SCANCODE_LEFT, SDL_SCANCODE_RIGHT, state.right_x);
    state.right_y = KeyAxis(SDL_SCANCODE_UP, SDL_SCANCODE_DOWN, state.right_y);

    // Keyboard triggers press all the way; analog triggers also set the button bits.
    if (state.buttons & button::L2) {
        state.l2 = 0xFF;
    }
    if (state.buttons & button::R2) {
        state.r2 = 0xFF;
    }
    if (state.l2 >= TriggerButtonThreshold) {
        state.buttons |= button::L2;
    }
    if (state.r2 >= TriggerButtonThreshold) {
        state.buttons |= button::R2;
    }
    g_state = state;
}

void OpenGamepad(SDL_JoystickID id) {
    if (SDL_GetGamepadFromID(id)) {
        return; // already open
    }
    SDL_Gamepad* gamepad = SDL_OpenGamepad(id);
    if (!gamepad) {
        LOG_WARNING("Input", "could not open controller {}: {}", id, SDL_GetError());
        return;
    }
    LOG_INFO("Input", "controller connected: {}", SDL_GetGamepadName(gamepad));
    g_gamepads.push_back(gamepad);
}

void CloseGamepad(SDL_JoystickID id) {
    SDL_Gamepad* gamepad = SDL_GetGamepadFromID(id);
    if (!gamepad) {
        return;
    }
    LOG_INFO("Input", "controller disconnected: {}", SDL_GetGamepadName(gamepad));
    std::erase(g_gamepads, gamepad);
    SDL_CloseGamepad(gamepad);
}

} // namespace

// The function keys for diagnostics (see the top of the file), from the keyboard or the script.
void OnToolKey(SDL_Scancode key) {
    switch (key) {
    case SDL_SCANCODE_F9: {
        // A marker in the log and a screenshot, so a moment of play can be found and seen later.
        static int markers = 0;
        LOG_INFO("Input", "==== MARKER {} (F9) ====", ++markers);
        gpu::RequestScreenshot();
        break;
    }
    case SDL_SCANCODE_F10:
        gpu::CaptureFrame();
        break;
    case SDL_SCANCODE_F11:
        diag::StartAccessSampling();
        break;
    default:
        break;
    }
}

// SOTC_KEY_SCRIPT (see the top of the file).
struct ScriptedKey {
    double start;
    double end;
    SDL_Scancode key;
    bool held = false;
};

std::vector<ScriptedKey> ParseScript() {
    std::vector<ScriptedKey> script;
    const char* text = std::getenv("SOTC_KEY_SCRIPT");
    if (!text) {
        return script;
    }
    std::string list = text;
    size_t begin = 0;
    while (begin < list.size()) {
        size_t end = list.find(',', begin);
        if (end == std::string::npos) {
            end = list.size();
        }
        const std::string item = list.substr(begin, end - begin);
        begin = end + 1;
        const size_t colon = item.find(':');
        if (colon == std::string::npos) {
            continue;
        }
        const size_t colon2 = item.find(':', colon + 1);
        const double start = std::strtod(item.c_str(), nullptr);
        const std::string name = item.substr(colon + 1, colon2 == std::string::npos
                                                            ? std::string::npos
                                                            : colon2 - colon - 1);
        const double hold =
            colon2 == std::string::npos ? 0.15 : std::strtod(item.c_str() + colon2 + 1, nullptr);
        const SDL_Scancode key = SDL_GetScancodeFromName(name.c_str());
        if (key == SDL_SCANCODE_UNKNOWN) {
            LOG_WARNING("Input", "SOTC_KEY_SCRIPT: unknown key \"{}\"", name);
            continue;
        }
        script.push_back({start, start + hold, key});
    }
    LOG_INFO("Input", "SOTC_KEY_SCRIPT: {} keys scheduled", script.size());
    return script;
}

// Presses and releases the scripted keys whose time has come.
void RunScript() {
    static std::vector<ScriptedKey> script = ParseScript();
    static const auto start = std::chrono::steady_clock::now();
    if (script.empty()) {
        return;
    }
    const double now =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    bool changed = false;
    for (ScriptedKey& entry : script) {
        const bool held = now >= entry.start && now < entry.end;
        if (held != entry.held) {
            entry.held = held;
            g_keys[entry.key] = held;
            changed = true;
            if (held) {
                OnToolKey(entry.key);
            }
        }
    }
    if (changed) {
        Update();
    }
}

void Initialize() {
    std::scoped_lock lock{g_mutex};
    int count = 0;
    if (SDL_JoystickID* ids = SDL_GetGamepads(&count)) {
        for (int i = 0; i < count; i++) {
            OpenGamepad(ids[i]);
        }
        SDL_free(ids);
    }
    if (g_gamepads.empty()) {
        LOG_INFO("Input", "no controller connected; using the keyboard");
    }
    Update();
}

void HandleEvent(const SDL_Event& event) {
    std::scoped_lock lock{g_mutex};
    switch (event.type) {
    case SDL_EVENT_KEY_DOWN:
        if (!event.key.repeat) {
            OnToolKey(event.key.scancode);
        }
        [[fallthrough]];
    case SDL_EVENT_KEY_UP:
        if (event.key.scancode < SDL_SCANCODE_COUNT) {
            g_keys[event.key.scancode] = event.key.down;
        }
        break;
    case SDL_EVENT_WINDOW_FOCUS_LOST:
        // Key releases go to the window that has the focus: forget everything held.
        std::fill(std::begin(g_keys), std::end(g_keys), false);
        break;
    case SDL_EVENT_GAMEPAD_ADDED:
        OpenGamepad(event.gdevice.which);
        break;
    case SDL_EVENT_GAMEPAD_REMOVED:
        CloseGamepad(event.gdevice.which);
        break;
    case SDL_EVENT_GAMEPAD_AXIS_MOTION:
    case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
    case SDL_EVENT_GAMEPAD_BUTTON_UP:
        break;
    default:
        return;
    }
    Update();
}

PadState Read() {
    std::scoped_lock lock{g_mutex};
    RunScript();
    return g_state;
}

void SetVibration(std::uint8_t large_motor, std::uint8_t small_motor) {
    std::scoped_lock lock{g_mutex};
    for (SDL_Gamepad* gamepad : g_gamepads) {
        // The PS4 keeps vibrating until told otherwise; SDL needs a duration, so use the longest.
        SDL_RumbleGamepad(gamepad, large_motor * 257, small_motor * 257, SDL_MAX_UINT32);
    }
}

void SetLightBar(std::uint8_t red, std::uint8_t green, std::uint8_t blue) {
    std::scoped_lock lock{g_mutex};
    for (SDL_Gamepad* gamepad : g_gamepads) {
        SDL_SetGamepadLED(gamepad, red, green, blue);
    }
}

} // namespace input
