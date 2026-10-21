// The PC's keyboard, mouse and game controllers, combined into the state of one PS4 controller.
//
// Keyboard and mouse: bound in <data folder>/controls.ini (DefaultControls below is what a new
// file gets; delete the file to get the defaults back). Keys are named by their position on a
// QWERTY keyboard (SDL scancodes), so they stay put on other layouts. In short:
//   W A S D          left stick (move), Left Alt held: walk (a shorter push of the stick)
//   mouse            the camera, once captured: click in the window to capture it (that click
//                    does nothing else), F1 or leaving the window releases it; in menus it is a
//                    cursor (src/input/menu_mouse.cpp): pointing selects, the left button
//                    confirms, the right button goes back, the wheel moves the selection; on
//                    the world map dragging moves the map and the wheel zooms
//   Escape           Options (the pause menu); in menus, Circle (back)
//   arrow keys       right stick, at full speed
//   the buttons      see DefaultControls
//
// The mouse turns the game's free camera directly (mouse_camera = native, the default): the
// camera's update is hooked (src/input/mouse_camera.cpp) and takes the movement through
// TakeMouseLook. Elsewhere (menus, the bow, the map) and with mouse_camera = stick, the mouse
// turns the camera through the right stick, because the game only knows a stick: the
// speed of the mouse becomes how far the stick is pushed (MouseLook). A stick sets how fast the
// camera turns, so this gives "the camera turns as far as the mouse moved" only as long as the
// game turns the camera at a speed proportional to the push. Beyond full push the camera cannot
// turn faster: the rest of a fast flick is carried over to the next reads (a fraction of a
// second at most), so the turn still ends where the mouse sent it. Writing the angle straight
// into the game's camera would remove that limit; that needs the camera code found in the
// executable first (same way as src/input/game_hooks.cpp).
//
// Tool keys (fixed, not in controls.ini):
//   F1               releases the mouse, or captures it again
//   F6               safe mode switch: each press moves to the next of "strict barriers",
//                    "whole buffers", "synchronous readbacks", "all three" and back to normal
//                    (ModeSwitches below; a "MODE" line in the log), and it stays while playing:
//                    for defects that only show while moving, which F7 cannot catch
//   F7               diagnostic sequence: for ~4 s each, switches one GPU layer optimization
//                    after another back to the safe way (src/diag/toggles.h, DiagPhases below)
//                    and takes 12 screenshots, with
//                    "DIAG PHASE" lines in the log (for finding which one causes a defect that
//                    only shows in one place of the game; keep the camera still meanwhile)
//   F9               writes a marker to the log and saves a screenshot (for finding a
//                    moment of play later)
//   F10              captures the frame in RenderDoc (when started with SOTC_RENDERDOC)
//   F11              samples for 30 s which memory the game's code touches (perf)
// SOTC_KEY_SCRIPT="20:Return,24.5:Return:0.3,60:W:30" presses keys by itself: seconds since the
// start, SDL key name and, optionally, how long to hold it (0.15 s by default). For reaching a
// scene and testing it without anyone at the keyboard.
// SOTC_INPUT_RECORD=<file> writes down the controller state the game reads, the F9 presses and
// when a progress save was loaded; SOTC_INPUT_REPLAY=<file> plays such a recording back from
// that load on, instead of the keyboard and controllers (taking the screenshots at the recorded
// F9 presses). The clock is the count of the game's controller reads, not seconds: the game
// reads it once per update, so a replay follows the game whatever the frame rate. Before the
// load (the menus) the replay does nothing: the game boots at its own pace and the controller
// reads there do not line up, so the menus are left to SOTC_KEY_SCRIPT (tools/bench_game.sh).
// SOTC_INPUT_REPLAY_ANCHOR=<read> gives the load's read for a recording made without the LOAD
// line. For reproducing a stretch of play on foot (on horseback it drifts off, 2026-10-14).
// Controllers go through SDL's gamepad mapping, by button position: the bottom face button is
// Cross on any brand, Start is Options and Back (or a touch pad) is the touch pad button.

#include "input/input.h"

#include <SDL3/SDL.h>

#include <algorithm>
#include <bit>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <string>
#include <utility>
#include <mutex>
#include <sstream>
#include <vector>

#include "common/log.h"
#include "diag/access_sampling.h"
#include "diag/toggles.h"
#include "gpu/gpu.h"

namespace input {

namespace {

// ---- Bindings (controls.ini) ----------------------------------------------------------------
//
// The file is lines of "<input> = <control>[, <control>...]", plus a few "<setting> = <value>"
// lines; "#" starts a comment. This text is both the defaults and what a new file is written
// with, so the two never disagree.
//
// <input>   a key by its SDL name ("W", "Space", "Left Shift", "Return", "Up" = the up arrow...;
//           https://wiki.libsdl.org/SDL3/SDL_Scancode), or Mouse Left, Mouse Right, Mouse Middle
//           (also called Wheel Press: pressing the wheel), Mouse Back, Mouse Forward (the two
//           side buttons), Wheel Up, Wheel Down.
// <control> a PS4 button (Cross Circle Square Triangle L1 R1 L2 R2 L3 R3 Options TouchPad
//           DpadUp DpadDown DpadLeft DpadRight), a stick direction (LeftStickUp, LeftStickDown,
//           LeftStickLeft, LeftStickRight, and the same with RightStick), or Walk (while held,
//           the left stick is pushed only walk_speed of the way).
//           A button may carry a tag: "R2@Grab". The game still just sees R2; the tag only
//           tells hooks in the game's code which input pressed it (see input.h, TagBit).
// The wheel has no "held": each notch presses the button for a few of the game's reads.
//
// The PS4 layout below is the game's "Modern" control scheme (Options > Settings > Controls):
// Cross jumps, Circle rolls, R2 grabs, the touch pad opens the map. What each other button does
// depends on the scheme picked in the game; the game's Controls screen shows it.
constexpr const char* DefaultControls = R"(# sotcport keyboard and mouse bindings.
# <input> = <PS4 control>[, <PS4 control>...]   (see src/input/input.cpp for the names)
# Delete this file to get the defaults back. Gamepads are not configured here.

# Moving
W = LeftStickUp
S = LeftStickDown
A = LeftStickLeft
D = LeftStickRight
Left Alt = Walk

# Camera: the mouse (click in the window to capture it, F1 releases it; in menus it is a
# cursor that points, clicks and scrolls) and the arrows
Up = RightStickUp
Down = RightStickDown
Left = RightStickLeft
Right = RightStickRight

# Actions
Space = Cross
Return = Circle
Mouse Left = Square
Q = Square
Mouse Right = R2
C = R2
F = Circle
Backspace = Circle
Mouse Back = Circle
E = Triangle
Mouse Forward = Triangle
Left Shift = R1
Mouse Middle = L1
Left Ctrl = L1
Z = L2
B = R3
V = L3
M = TouchPad

# Directional buttons; the wheel presses them one notch at a time
Wheel Up = DpadLeft
Wheel Down = DpadRight
Wheel press = DpadUp
P = DpadDown


Escape = Options
Tab = TouchPad

# Mouse camera. native: the mouse turns the game's camera directly (as far as it moves, no top
# speed, no easing); where the game's camera is not the free camera (menus, aiming the bow...)
# it falls back to pushing the right stick. stick: always the right stick.
mouse_camera = native
# How fast it turns: 1 = 0.001 radian per mouse count in native (about 20 cm for a full turn at
# 800 dpi), or a full push of the stick at 1500 counts per second as a stick. The stick push a
# small movement starts from (the game ignores pushes smaller than its dead zone; the controller
# reports 13/127 = 0.10), and whether moving the mouse up looks down (in native, the game's
# invert options are left to the stick; as a stick, they apply).
mouse_sensitivity = 1.0
mouse_deadzone = 0.10
mouse_invert_y = 0
# How far Walk pushes the left stick (0..1).
walk_speed = 0.4
)";

// Inputs that can be bound, numbered in one space: the keyboard's scancodes first, then the
// mouse buttons (SDL's numbers 1-5 after MouseSource), then the two wheel directions.
constexpr int MouseSource = SDL_SCANCODE_COUNT;       // + SDL_BUTTON_LEFT (1) .. SDL_BUTTON_X2 (5)
constexpr int WheelUpSource = MouseSource + 6;
constexpr int WheelDownSource = MouseSource + 7;
constexpr int SourceCount = MouseSource + 8;

constexpr struct {
    const char* name;
    int source;
} MouseSourceNames[] = {
    {"Mouse Left", MouseSource + SDL_BUTTON_LEFT},  {"Mouse Right", MouseSource + SDL_BUTTON_RIGHT},
    {"Mouse Middle", MouseSource + SDL_BUTTON_MIDDLE}, {"Mouse Back", MouseSource + SDL_BUTTON_X1},
    {"Wheel Press", MouseSource + SDL_BUTTON_MIDDLE}, // the same button as Mouse Middle
    {"Mouse Forward", MouseSource + SDL_BUTTON_X2}, {"Wheel Up", WheelUpSource},
    {"Wheel Down", WheelDownSource},
};

// What an input does.
struct Control {
    enum Kind : std::uint8_t { Button, Stick, Walk } kind;
    std::uint32_t button = 0; // Button: the PS4 button bit
    std::uint32_t tag = 0;    // Button: the bit of its tag ("R2@Grab"), 0 without one
    std::uint8_t stick = 0;   // Stick: 0 = left, 1 = right
    std::int8_t dx = 0;       // Stick: direction, -1/0/+1 on each axis (y: -1 = up)
    std::int8_t dy = 0;
};

constexpr struct {
    const char* name;
    Control control;
} ControlNames[] = {
    {"Cross", {Control::Button, button::Cross}},
    {"Circle", {Control::Button, button::Circle}},
    {"Square", {Control::Button, button::Square}},
    {"Triangle", {Control::Button, button::Triangle}},
    {"L1", {Control::Button, button::L1}},
    {"R1", {Control::Button, button::R1}},
    {"L2", {Control::Button, button::L2}},
    {"R2", {Control::Button, button::R2}},
    {"L3", {Control::Button, button::L3}},
    {"R3", {Control::Button, button::R3}},
    {"Options", {Control::Button, button::Options}},
    {"TouchPad", {Control::Button, button::TouchPad}},
    {"DpadUp", {Control::Button, button::Up}},
    {"DpadDown", {Control::Button, button::Down}},
    {"DpadLeft", {Control::Button, button::Left}},
    {"DpadRight", {Control::Button, button::Right}},
    {"LeftStickUp", {Control::Stick, 0, 0, 0, 0, -1}},
    {"LeftStickDown", {Control::Stick, 0, 0, 0, 0, 1}},
    {"LeftStickLeft", {Control::Stick, 0, 0, 0, -1, 0}},
    {"LeftStickRight", {Control::Stick, 0, 0, 0, 1, 0}},
    {"RightStickUp", {Control::Stick, 0, 0, 1, 0, -1}},
    {"RightStickDown", {Control::Stick, 0, 0, 1, 0, 1}},
    {"RightStickLeft", {Control::Stick, 0, 0, 1, -1, 0}},
    {"RightStickRight", {Control::Stick, 0, 0, 1, 1, 0}},
    {"Walk", {Control::Walk}},
};

std::vector<Control> g_bindings[SourceCount]; // per input, what it does (set before any read)
std::vector<std::string> g_tag_names;          // tag bit i is g_tag_names[i] (32 at most)
bool g_bindings_loaded = false;

struct MouseSettings {
    float sensitivity = 1.0f;
    float deadzone = 0.10f;
    bool invert_y = false;
    bool native_camera = true; // mouse_camera = native (see TakeMouseLook)
};
MouseSettings g_mouse_settings;
float g_walk_speed = 0.4f;

std::string Trim(std::string text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const auto last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

bool SameName(std::string_view a, std::string_view b) {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(), [](char x, char y) {
               return std::tolower(static_cast<unsigned char>(x)) ==
                      std::tolower(static_cast<unsigned char>(y));
           });
}

int SourceFromName(const std::string& name) {
    for (const auto& mouse : MouseSourceNames) {
        if (SameName(name, mouse.name)) {
            return mouse.source;
        }
    }
    const SDL_Scancode key = SDL_GetScancodeFromName(name.c_str());
    return key == SDL_SCANCODE_UNKNOWN ? -1 : static_cast<int>(key);
}

// "R2" or "R2@Grab" (a tag gets the next free bit the first time its name is seen).
bool ControlFromName(const std::string& text, Control& out) {
    const size_t at = text.find('@');
    const std::string name = Trim(text.substr(0, at));
    for (const auto& entry : ControlNames) {
        if (!SameName(name, entry.name)) {
            continue;
        }
        out = entry.control;
        if (at != std::string::npos) {
            const std::string tag = Trim(text.substr(at + 1));
            if (out.kind != Control::Button || tag.empty()) {
                return false; // only buttons can be told apart
            }
            auto it = std::find_if(g_tag_names.begin(), g_tag_names.end(),
                                   [&](const std::string& known) { return SameName(known, tag); });
            if (it == g_tag_names.end()) {
                if (g_tag_names.size() == 32) {
                    return false;
                }
                it = g_tag_names.insert(g_tag_names.end(), tag);
            }
            out.tag = 1u << (it - g_tag_names.begin());
        }
        return true;
    }
    return false;
}

void ParseControls(const std::string& text, const char* origin) {
    for (auto& list : g_bindings) {
        list.clear();
    }
    g_tag_names.clear();
    g_mouse_settings = {};
    g_walk_speed = 0.4f;
    size_t begin = 0;
    int line_number = 0;
    while (begin < text.size()) {
        size_t end = text.find('\n', begin);
        if (end == std::string::npos) {
            end = text.size();
        }
        std::string line = text.substr(begin, end - begin);
        begin = end + 1;
        ++line_number;
        if (const size_t hash = line.find('#'); hash != std::string::npos) {
            line.resize(hash);
        }
        line = Trim(line);
        if (line.empty()) {
            continue;
        }
        const size_t equals = line.find('=');
        if (equals == std::string::npos) {
            LOG_WARNING("Input", "{}:{}: no '=' in \"{}\"", origin, line_number, line);
            continue;
        }
        const std::string left = Trim(line.substr(0, equals));
        const std::string right = Trim(line.substr(equals + 1));
        // Settings first: their names are not key names.
        if (left == "mouse_sensitivity") {
            g_mouse_settings.sensitivity = std::max(0.01f, std::strtof(right.c_str(), nullptr));
            continue;
        }
        if (left == "mouse_deadzone") {
            g_mouse_settings.deadzone = std::clamp(std::strtof(right.c_str(), nullptr), 0.0f, 0.9f);
            continue;
        }
        if (left == "mouse_camera") {
            g_mouse_settings.native_camera = !SameName(right, "stick");
            continue;
        }
        if (left == "mouse_invert_y") {
            g_mouse_settings.invert_y = right == "1" || SameName(right, "true");
            continue;
        }
        if (left == "walk_speed") {
            g_walk_speed = std::clamp(std::strtof(right.c_str(), nullptr), 0.05f, 1.0f);
            continue;
        }
        const int source = SourceFromName(left);
        if (source < 0) {
            LOG_WARNING("Input", "{}:{}: unknown input \"{}\"", origin, line_number, left);
            continue;
        }
        size_t item_begin = 0;
        while (item_begin <= right.size()) {
            size_t item_end = right.find(',', item_begin);
            if (item_end == std::string::npos) {
                item_end = right.size();
            }
            const std::string item = Trim(right.substr(item_begin, item_end - item_begin));
            item_begin = item_end + 1;
            Control control{};
            if (item.empty()) {
                continue;
            }
            if (ControlFromName(item, control)) {
                g_bindings[source].push_back(control);
            } else {
                LOG_WARNING("Input", "{}:{}: unknown PS4 control \"{}\"", origin, line_number,
                            item);
            }
        }
    }
    g_bindings_loaded = true;
}

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

// SOTC_INPUT_RECORD / SOTC_INPUT_REPLAY (see the top of the file). One line per change:
// "<read> <buttons, hex> <lx> <ly> <rx> <ry> <l2> <r2>", or "<read> F9" for a screenshot.
std::uint64_t g_reads = 0;            // controller reads so far (the recording's clock)
std::FILE* g_record = nullptr;
PadState g_recorded;                  // the last state written down
bool g_recorded_any = false;
struct ReplayEvent {
    std::uint64_t read;
    bool screenshot;
    PadState state;
};
std::vector<ReplayEvent> g_replay;
std::size_t g_replay_next = 0;
bool g_replaying = false;
PadState g_replay_state;
std::uint64_t g_recorded_load = 0;         // the recording's read at the save load (0: none)
std::int64_t g_replay_offset = 0;          // live read - recorded read, once loaded
bool g_replay_live = false;                // the save was loaded: the replay drives the pad
bool g_record_loaded = false;
const auto g_start = std::chrono::steady_clock::now();

// F7: the diagnostic sequence (see the top of the file).
struct DiagPhase {
    const char* name;
    std::uint32_t toggles;
};
// The phases that change bindings (barriers, buffer sizes) also turn the bind memo off: a draw
// that reuses its bindings would keep the ones made before the switch. Shadow copies are off by
// default since 2026-10-17 (SOTC_SHADOW_COPY=1), so they have no phase of their own.
constexpr std::uint32_t AllOff =
    (1u << diag::ShadowCopyOff) | (1u << diag::SrtPatchesOff) | (1u << diag::BindMemoOff) |
    (1u << diag::TargetCacheOff) | (1u << diag::TextureLookupOff) |
    (1u << diag::StrictBarriersOn) | (1u << diag::WholeBuffersOn) | (1u << diag::SyncReadbackOn);
constexpr DiagPhase DiagPhases[] = {
    {"as launched", 0},
    {"strict barriers (bind memo off)", (1u << diag::StrictBarriersOn) | (1u << diag::BindMemoOff)},
    {"whole buffers (bind memo off)", (1u << diag::WholeBuffersOn) | (1u << diag::BindMemoOff)},
    {"synchronous readbacks", 1u << diag::SyncReadbackOn},
    {"bind memo off", 1u << diag::BindMemoOff},
    {"SRT patches off", 1u << diag::SrtPatchesOff},
    {"target cache off", 1u << diag::TargetCacheOff},
    {"texture lookup cache off", 1u << diag::TextureLookupOff},
    {"all of them", AllOff},
    {"LOD statistics on", 1u << diag::LodStatsOn},
    {"as launched again", 0},
};
constexpr double DiagPhaseSeconds = 4.0;
constexpr double DiagSettleSeconds = 1.5; // before the first screenshot of a phase
constexpr double DiagShotInterval = 0.2;
constexpr int DiagShotsPerPhase = 12;
// F6: GPU layer optimizations switched back to the safe way while playing (see the top of the
// file). The ones that change bindings turn the bind memo off too (see DiagPhases).
struct ModeSwitch {
    const char* name;
    std::uint32_t toggles;
};
constexpr std::uint32_t SafeThree =
    (1u << diag::StrictBarriersOn) | (1u << diag::WholeBuffersOn) | (1u << diag::SyncReadbackOn);
constexpr ModeSwitch ModeSwitches[] = {
    {"normal (as launched)", 0},
    {"strict barriers (bind memo off)", (1u << diag::StrictBarriersOn) | (1u << diag::BindMemoOff)},
    {"whole buffers (bind memo off)", (1u << diag::WholeBuffersOn) | (1u << diag::BindMemoOff)},
    {"synchronous readbacks", 1u << diag::SyncReadbackOn},
    {"all three (bind memo off)", SafeThree | (1u << diag::BindMemoOff)},
};
std::size_t g_mode_switch = 0;

bool g_diag_running = false;
std::chrono::steady_clock::time_point g_diag_start;
int g_diag_phase = -1;
int g_diag_shots = 0;

// Called with g_mutex held, at every controller read (the game reads it every update).
void RunDiagSequence() {
    if (!g_diag_running) {
        return;
    }
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - g_diag_start).count();
    const int phase = static_cast<int>(elapsed / DiagPhaseSeconds);
    if (phase >= static_cast<int>(std::size(DiagPhases))) {
        diag::SetToggles(ModeSwitches[g_mode_switch].toggles); // back to the F6 mode
        g_diag_running = false;
        LOG_INFO("Input", "==== DIAG sequence done ====");
        return;
    }
    if (phase != g_diag_phase) {
        g_diag_phase = phase;
        g_diag_shots = 0;
        diag::SetToggles(DiagPhases[phase].toggles);
        LOG_INFO("Input", "==== DIAG PHASE {}: {} ====", phase, DiagPhases[phase].name);
    }
    const double in_phase = elapsed - phase * DiagPhaseSeconds;
    if (g_diag_shots < DiagShotsPerPhase &&
        in_phase >= DiagSettleSeconds + g_diag_shots * DiagShotInterval) {
        ++g_diag_shots;
        gpu::RequestScreenshot();
    }
}

bool SameState(const PadState& a, const PadState& b) {
    return a.buttons == b.buttons && a.left_x == b.left_x && a.left_y == b.left_y &&
           a.right_x == b.right_x && a.right_y == b.right_y && a.l2 == b.l2 && a.r2 == b.r2;
}

void OpenRecording() {
    if (const char* path = std::getenv("SOTC_INPUT_RECORD")) {
        g_record = std::fopen(path, "w");
        if (g_record) {
            std::fprintf(g_record, "# sotcport input recording: <read> <buttons> <lx> <ly> <rx> "
                                   "<ry> <l2> <r2> | <read> F9\n");
            LOG_INFO("Input", "recording the controller to {}", path);
        } else {
            LOG_WARNING("Input", "cannot write {}", path);
        }
    }
    if (const char* path = std::getenv("SOTC_INPUT_REPLAY")) {
        std::FILE* file = std::fopen(path, "r");
        if (!file) {
            LOG_WARNING("Input", "cannot read {}", path);
            return;
        }
        char line[256];
        while (std::fgets(line, sizeof(line), file)) {
            if (line[0] == '#') {
                continue;
            }
            ReplayEvent event{};
            unsigned long long read = 0;
            unsigned buttons = 0, lx = 0, ly = 0, rx = 0, ry = 0, l2 = 0, r2 = 0;
            if (std::strstr(line, "LOAD") && std::sscanf(line, "%llu", &read) == 1) {
                if (g_recorded_load == 0) {
                    g_recorded_load = read;
                }
                continue;
            }
            if (std::strstr(line, "F9") && std::sscanf(line, "%llu", &read) == 1) {
                event.read = read;
                event.screenshot = true;
            } else if (std::sscanf(line, "%llu %x %u %u %u %u %u %u", &read, &buttons, &lx, &ly,
                                   &rx, &ry, &l2, &r2) == 8) {
                event.read = read;
                event.state = {buttons,
                               static_cast<std::uint8_t>(lx),
                               static_cast<std::uint8_t>(ly),
                               static_cast<std::uint8_t>(rx),
                               static_cast<std::uint8_t>(ry),
                               static_cast<std::uint8_t>(l2),
                               static_cast<std::uint8_t>(r2)};
            } else {
                continue;
            }
            g_replay.push_back(event);
        }
        std::fclose(file);
        if (const char* anchor = std::getenv("SOTC_INPUT_REPLAY_ANCHOR")) {
            g_recorded_load = std::strtoull(anchor, nullptr, 10);
        }
        // What came before the load (the menus) is not replayed.
        std::erase_if(g_replay, [](const ReplayEvent& event) {
            return g_recorded_load != 0 && event.read < g_recorded_load;
        });
        g_replaying = true;
        g_replay_live = g_recorded_load == 0; // an old recording without a load: from the start
        LOG_INFO("Input", "replaying {} controller events from {}{}", g_replay.size(), path,
                 g_recorded_load ? std::format(", from the save load (read {})", g_recorded_load)
                                 : std::string{});
    }
}

// Called with g_mutex held, once per controller read.
// `state` is what the game gets from the keyboard, mouse and gamepads at this read.
void RecordOrReplay(const PadState& state) {
    static const bool opened = (OpenRecording(), true);
    (void)opened;
    ++g_reads;
    if (g_record && (!g_recorded_any || !SameState(state, g_recorded))) {
        g_recorded = state;
        g_recorded_any = true;
        // The last column (milliseconds since start) is for reading the file; replays ignore it.
        std::fprintf(g_record, "%llu %x %u %u %u %u %u %u %lld\n",
                     static_cast<unsigned long long>(g_reads), state.buttons, state.left_x,
                     state.left_y, state.right_x, state.right_y, state.l2, state.r2,
                     static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                                std::chrono::steady_clock::now() - g_start)
                                                .count()));
        std::fflush(g_record);
    }
    if (g_replaying && g_replay_live) {
        while (g_replay_next < g_replay.size() &&
               static_cast<std::int64_t>(g_replay[g_replay_next].read) + g_replay_offset <=
                   static_cast<std::int64_t>(g_reads)) {
            const ReplayEvent& event = g_replay[g_replay_next++];
            if (event.screenshot) {
                static int markers = 0;
                LOG_INFO("Input", "==== MARKER {} (F9, replayed) ====", ++markers);
                gpu::RequestScreenshot();
            } else {
                g_replay_state = event.state;
            }
        }
        if (g_replay_next == g_replay.size() && !g_replay.empty()) {
            static bool told = false;
            if (!std::exchange(told, true)) {
                LOG_INFO("Input", "controller replay finished at read {}", g_reads);
            }
        }
    }
}
bool g_keys[SDL_SCANCODE_COUNT] = {}; // keyboard keys held, window thread only
// Keys SOTC_KEY_SCRIPT holds. Kept apart from the real keyboard: losing the window's focus
// releases the real keys (their releases go to another window), and it released the
// scripted ones too, which left a benchmark standing still (2026-10-14).
bool g_script_keys[SDL_SCANCODE_COUNT] = {};

bool KeyHeld(SDL_Scancode key) {
    return g_keys[key] || g_script_keys[key];
}

// ---- Mouse ------------------------------------------------------------------------------------
// Everything here is guarded by g_mutex: the window's thread writes it from HandleEvent, the
// game's thread consumes it in Read.

// Capture: the mouse turns the camera (SDL's relative mode: no cursor, raw movement) while the
// player wants it (g_capture_wanted: a click into the window sets it, F1 and leaving the window
// clear it), the window has the focus and no menu is open. ApplyMouseMode() keeps
// g_mouse_captured in step with that, on the window's thread.
SDL_Window* g_window = nullptr; // the game's window (from its events)
bool g_capture_wanted = false;
bool g_mouse_captured = false;   // relative mode is on

// Menus (src/input/menu_mouse.cpp): while a menu has the input, the mouse is a cursor. The
// game's menu code tells when it runs (NoteMenuActive); a menu is open while it keeps doing so.
std::chrono::steady_clock::time_point g_menu_seen{};
std::chrono::steady_clock::time_point g_map_seen{}; // the world map, one of those "menus"
constexpr auto MenuTimeout = std::chrono::milliseconds{200};
// The cursor in the game's UI coordinates (1920 x 1080, whatever the window's size), and how
// many times it moved (the menu code changes the selection only when the cursor moves, so that
// a cursor left over an item does not fight the keyboard).
float g_cursor_x = -1.0f;
float g_cursor_y = -1.0f;
std::uint32_t g_cursor_moves = 0;
// Escape is Circle (back) in menus, Options (the pause menu) elsewhere: decided when it is pressed
// and kept until it is released, so that the press that opens the pause menu does not go on as
// a "back" that closes it again (NoteEscapePressed).
bool g_escape_as_circle = false;

// The left button held in a menu (dragging the map).
bool g_menu_left_held = false;
// A PS4 button held as long as the left button is (a click on a value control's arrow:
// HoldButtonWhileClicked), at least MenuHoldMinReads reads.
std::uint32_t g_menu_hold = 0;
int g_menu_hold_reads = 0;
constexpr int MenuHoldMinReads = 3;
// A left click in a menu, waiting for the menu code to find what it hit (MenuClickTimeout).
bool g_click_pending = false;
float g_click_x = 0.0f;
float g_click_y = 0.0f;
std::chrono::steady_clock::time_point g_click_time{};
constexpr auto MenuClickTimeout = std::chrono::milliseconds{250};

bool MenuOpen() {
    return std::chrono::steady_clock::now() - g_menu_seen < MenuTimeout;
}
std::uint32_t g_mouse_held = 0;       // bit n: SDL mouse button n held
// Mouse movement not given to the game yet, in mouse counts (SDL's relative motion). Read()
// turns it into a stick push and keeps what a full push could not carry (see MouseLook).
float g_mouse_dx = 0.0f;
float g_mouse_dy = 0.0f;
// The same movement for the native mouse camera (TakeMouseLook, src/input/mouse_camera.cpp),
// which takes it on the game's thread whenever the game updates its free camera.
float g_look_dx = 0.0f;
float g_look_dy = 0.0f;
// When the free camera last took the mouse: while it keeps taking it, the mouse stays off the
// right stick. In menus, cinematics or other cameras (the bow, the map) the hook is not called,
// and after NativeCameraTimeout the mouse goes back to pushing the stick.
std::chrono::steady_clock::time_point g_native_camera_seen{};
// When the captured mouse last moved (MouseLookActive).
std::chrono::steady_clock::time_point g_last_motion{};
constexpr auto NativeCameraTimeout = std::chrono::milliseconds{150};
std::chrono::steady_clock::time_point g_last_read = std::chrono::steady_clock::now();

// The wheel has no "held": every notch becomes a press of WheelPressReads controller reads
// followed by a release of WheelReleaseReads reads (the game sees a press only as a change
// between two of its reads, and two presses only with a release between them). Notches that
// come faster wait their turn, up to WheelMaxPending.
constexpr int WheelPressReads = 3;
constexpr int WheelReleaseReads = 2;
constexpr int WheelMaxPending = 4;
struct WheelPulse {
    int pending = 0;    // notches waiting
    int reads_left = 0; // reads left in the current press or release
    bool pressed = false;
};
WheelPulse g_wheel[2]; // [0] up, [1] down

// One controller read went by: advances a wheel direction's press/release cycle.
void StepWheel(WheelPulse& wheel) {
    if (wheel.reads_left > 0 && --wheel.reads_left > 0) {
        return;
    }
    if (wheel.pressed) {
        wheel.pressed = false;
        wheel.reads_left = WheelReleaseReads;
    } else if (wheel.pending > 0) {
        --wheel.pending;
        wheel.pressed = true;
        wheel.reads_left = WheelPressReads;
    }
}

// Puts the mouse in the mode it should be in now (see g_capture_wanted). On the window's thread.
void ApplyMouseMode() {
    if (!g_window) {
        return;
    }
    const bool focused = (SDL_GetWindowFlags(g_window) & SDL_WINDOW_INPUT_FOCUS) != 0;
    const bool captured = g_capture_wanted && focused && !MenuOpen();
    if (captured == g_mouse_captured) {
        return;
    }
    // Relative mode hides the cursor, keeps it in the window and reports raw movement even at
    // the window's edge; out of it the cursor shows again.
    SDL_SetWindowRelativeMouseMode(g_window, captured);
    g_mouse_captured = captured;
    g_mouse_held = 0;
    // A button held across the change is released on the other side: forget it here.
    g_menu_left_held = false;
    g_mouse_dx = g_mouse_dy = 0.0f;
    g_look_dx = g_look_dy = 0.0f;
    LOG_INFO("Input", "mouse {}", captured ? "captured (F1 releases it)"
                                  : MenuOpen() ? "free (menu)" : "released");
}

// A position in the window (SDL's window coordinates) in the game's UI coordinates. The
// presenter fits the 16:9 picture in the window and centers it (FitImage in vk_presenter.cpp);
// -1 outside the picture.
void WindowToUi(float x, float y, float& ui_x, float& ui_y) {
    int width = 0, height = 0;
    if (!g_window || !SDL_GetWindowSize(g_window, &width, &height) || width <= 0 || height <= 0) {
        ui_x = ui_y = -1.0f;
        return;
    }
    constexpr float UiWidth = 1920.0f, UiHeight = 1080.0f;
    const float scale = std::min(width / UiWidth, height / UiHeight);
    ui_x = (x - (width - UiWidth * scale) / 2) / scale;
    ui_y = (y - (height - UiHeight * scale) / 2) / scale;
    if (ui_x < 0 || ui_y < 0 || ui_x >= UiWidth || ui_y >= UiHeight) {
        ui_x = ui_y = -1.0f;
    }
}

// Presses of PS4 buttons the menus ask for (a click is Cross, the right button Circle, the wheel
// the directional buttons): each is held for PulsePressReads reads, then released for
// PulseReleaseReads, one after the other (like the wheel notches, for the same reason).
constexpr int PulsePressReads = 3;
constexpr int PulseReleaseReads = 2;
std::vector<std::uint32_t> g_pulse_queue;
std::uint32_t g_pulse_button = 0; // the one being pressed or released
int g_pulse_reads = 0;
bool g_pulse_pressed = false;

void QueuePulse(std::uint32_t button) {
    if (g_pulse_queue.size() < 4) {
        g_pulse_queue.push_back(button);
    }
}

// The wheel in a menu moves the selection up and down; on the map it zooms (R2 in, L2 out, as
// the map's legend says).
void MenuWheel(float notches) {
    if (!MenuOpen() || notches == 0) {
        return;
    }
    const bool map = std::chrono::steady_clock::now() - g_map_seen < MenuTimeout;
    QueuePulse(map ? (notches > 0 ? button::R2 : button::L2)
                   : (notches > 0 ? button::Up : button::Down));
}

// One controller read went by.
void StepPulse() {
    if (g_pulse_reads > 0 && --g_pulse_reads > 0) {
        return;
    }
    if (g_pulse_pressed) {
        g_pulse_pressed = false;
        g_pulse_reads = PulseReleaseReads;
    } else if (!g_pulse_queue.empty()) {
        g_pulse_button = g_pulse_queue.front();
        g_pulse_queue.erase(g_pulse_queue.begin());
        g_pulse_pressed = true;
        g_pulse_reads = PulsePressReads;
    }
}

// Turns the mouse movement since the last read into a push of the right stick. Called by
// Read() once per controller read, with g_mutex held.
//
// A stick sets the camera's turning *speed*, so what matters is the mouse's speed: counts per
// second over the time since the last read (not counts per read, which would make the camera
// turn faster at a higher frame rate). MouseFullSpeed counts per second (divided by
// mouse_sensitivity) push the stick all the way.
//
//   push = speed / full speed            (a vector: the direction the mouse moved)
//   push > 1: only 1 fits; the part that did not fit stays in g_mouse_dx/dy for the next
//             reads (at most MouseMaxCarrySeconds of full push), so a fast flick turns the
//             camera as far as a slow one, only spread over a few frames
//   0 < push: raised past the game's dead zone, deadzone + push * (1 - deadzone), so that a
//             slow movement turns the camera slowly instead of not at all
//
// Returns false when the mouse did not move (the stick is left to the gamepad and the keys).
constexpr float MouseFullSpeed = 1500.0f;
constexpr float MouseMaxCarrySeconds = 0.15f;
bool MouseLook(std::uint8_t& out_x, std::uint8_t& out_y) {
    const auto now = std::chrono::steady_clock::now();
    // Clamped: a long gap (a loading screen, the first read) must not turn a small movement
    // into nothing, and two reads at almost the same time must not turn it into a huge push.
    const float seconds = std::clamp(std::chrono::duration<float>(now - g_last_read).count(),
                                     0.001f, 0.1f);
    g_last_read = now;
    if (!g_mouse_captured || (g_mouse_dx == 0.0f && g_mouse_dy == 0.0f)) {
        g_mouse_dx = g_mouse_dy = 0.0f;
        return false;
    }
    const float full_speed = MouseFullSpeed / g_mouse_settings.sensitivity;
    float push_x = g_mouse_dx / seconds / full_speed;
    float push_y = g_mouse_dy / seconds / full_speed;
    float push = std::sqrt(push_x * push_x + push_y * push_y);
    if (push > 1.0f) {
        // Keep the part that did not fit, in counts, capped.
        const float kept = 1.0f - 1.0f / push;
        float carry_x = g_mouse_dx * kept;
        float carry_y = g_mouse_dy * kept;
        const float carry = std::sqrt(carry_x * carry_x + carry_y * carry_y);
        const float max_carry = full_speed * MouseMaxCarrySeconds;
        if (carry > max_carry) {
            carry_x *= max_carry / carry;
            carry_y *= max_carry / carry;
        }
        g_mouse_dx = carry_x;
        g_mouse_dy = carry_y;
        push_x /= push;
        push_y /= push;
        push = 1.0f;
    } else {
        g_mouse_dx = g_mouse_dy = 0.0f;
    }
    if (g_mouse_settings.invert_y) {
        push_y = -push_y;
    }
    const float deadzone = g_mouse_settings.deadzone;
    const float scale = (deadzone + push * (1.0f - deadzone)) / push; // push > 0 here
    const auto to_stick = [](float value) {
        return static_cast<std::uint8_t>(std::clamp<int>(0x80 + std::lround(value * 127.0f), 0, 255));
    };
    out_x = to_stick(push_x * scale);
    out_y = to_stick(push_y * scale);
    return true;
}
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

// Buttons that keyboard/mouse inputs hold, by tag (see input.h, ButtonHeldFor). Set by Update()
// from what is held now; Read() copies them to the g_read_* pair, which is what the game saw.
std::uint32_t g_untagged_buttons = 0;  // from gamepads and inputs without a tag
std::uint32_t g_tagged_buttons[32] = {}; // [tag bit index]: from inputs with that tag
std::uint32_t g_read_untagged_buttons = 0;
std::uint32_t g_read_tagged_buttons[32] = {};

bool SourceHeld(int source) {
    if (source < MouseSource) {
        return KeyHeld(static_cast<SDL_Scancode>(source));
    }
    if (source == WheelUpSource || source == WheelDownSource) {
        return g_wheel[source == WheelDownSource].pressed;
    }
    return (g_mouse_held >> (source - MouseSource)) & 1;
}

// A stick pushed by keys: the sum of the held directions, put on the circle the real stick
// moves in. Two keys at once (W + D) give a diagonal of the same length as one key, not
// sqrt(2) longer (which the game would read as running faster on diagonals, or clip to a
// square); opposite keys cancel out. `length` 1 is a full push.
void KeyStick(int dx, int dy, float length, std::uint8_t& out_x, std::uint8_t& out_y) {
    if (dx == 0 && dy == 0) {
        return; // nothing held: the gamepad's value stays
    }
    const float scale = length / std::sqrt(static_cast<float>(dx * dx + dy * dy));
    out_x = static_cast<std::uint8_t>(std::clamp<int>(0x80 + std::lround(dx * scale * 127.0f), 0, 255));
    out_y = static_cast<std::uint8_t>(std::clamp<int>(0x80 + std::lround(dy * scale * 127.0f), 0, 255));
}

// Recomputes the state from the keyboard, the mouse buttons, the wheel and the controllers
// (the mouse's movement is added later, in Read). Called with g_mutex held.
void Update() {
    PadState state;
    std::uint32_t untagged = 0;
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
    // A gamepad's buttons are never told apart, its analog triggers included.
    untagged |= state.buttons;
    if (state.l2 >= TriggerButtonThreshold) {
        untagged |= button::L2;
    }
    if (state.r2 >= TriggerButtonThreshold) {
        untagged |= button::R2;
    }

    // Keyboard, mouse buttons and wheel, through the bindings.
    std::uint32_t tagged[32] = {};
    int stick_dx[2] = {}, stick_dy[2] = {};
    bool walk = false;
    for (int source = 0; source < SourceCount; ++source) {
        if (g_bindings[source].empty() || !SourceHeld(source)) {
            continue;
        }
        if (source == SDL_SCANCODE_ESCAPE && g_escape_as_circle) {
            state.buttons |= button::Circle;
            untagged |= button::Circle;
            continue;
        }
        for (const Control& control : g_bindings[source]) {
            switch (control.kind) {
            case Control::Button:
                state.buttons |= control.button;
                if (control.tag) {
                    tagged[std::countr_zero(control.tag)] |= control.button;
                } else {
                    untagged |= control.button;
                }
                break;
            case Control::Stick:
                stick_dx[control.stick] += control.dx;
                stick_dy[control.stick] += control.dy;
                break;
            case Control::Walk:
                walk = true;
                break;
            }
        }
    }
    // Opposite keys can sum to more than 1 per axis (two keys bound to the same direction).
    for (int stick = 0; stick < 2; ++stick) {
        stick_dx[stick] = std::clamp(stick_dx[stick], -1, 1);
        stick_dy[stick] = std::clamp(stick_dy[stick], -1, 1);
    }
    KeyStick(stick_dx[0], stick_dy[0], walk ? g_walk_speed : 1.0f, state.left_x, state.left_y);
    KeyStick(stick_dx[1], stick_dy[1], 1.0f, state.right_x, state.right_y);

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
    g_untagged_buttons = untagged;
    std::copy(std::begin(tagged), std::end(tagged), std::begin(g_tagged_buttons));
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
        if (g_record) {
            std::fprintf(g_record, "%llu F9\n", static_cast<unsigned long long>(g_reads));
            std::fflush(g_record);
        }
        break;
    }
    case SDL_SCANCODE_F6: {
        // g_mutex is held (both callers), as for RunDiagSequence, which also sets the toggles.
        g_mode_switch = (g_mode_switch + 1) % std::size(ModeSwitches);
        if (!g_diag_running) {
            diag::SetToggles(ModeSwitches[g_mode_switch].toggles);
        }
        LOG_INFO("Input", "==== MODE {}: {} (F6) ====", g_mode_switch,
                 ModeSwitches[g_mode_switch].name);
        break;
    }
    case SDL_SCANCODE_F7:
        if (!g_diag_running) {
            g_diag_running = true;
            g_diag_start = std::chrono::steady_clock::now();
            g_diag_phase = -1;
            LOG_INFO("Input", "==== DIAG sequence started (F7) ====");
        }
        break;
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

// SOTC_MOUSE_SCRIPT="80:move:400:500,82:click:400:545": moves the menu cursor, clicks, presses
// (down) and releases (up) the left button, clicks the right button (right) or turns the wheel
// (wheelup, wheeldown) at
// those seconds (same clock as SOTC_KEY_SCRIPT), in the game's UI coordinates (1920 x 1080):
// for testing the mouse in menus with nobody at the mouse. Only while the mouse is not captured.
struct ScriptedMouse {
    double time;
    enum Action { Move, Click, Down, Up, Right, WheelUp, WheelDown } action;
    float x, y;
    bool done = false;
};

std::vector<ScriptedMouse> ParseMouseScript() {
    std::vector<ScriptedMouse> script;
    const char* text = std::getenv("SOTC_MOUSE_SCRIPT");
    for (const char* p = text; p && *p;) {
        ScriptedMouse entry{};
        char action[16] = {};
        int used = 0;
        if (std::sscanf(p, "%lf:%15[a-z]:%f:%f%n", &entry.time, action, &entry.x, &entry.y,
                        &used) != 4) {
            LOG_WARNING("Input", "SOTC_MOUSE_SCRIPT: cannot read \"{}\"", p);
            break;
        }
        entry.action = std::strcmp(action, "click") == 0  ? ScriptedMouse::Click
                       : std::strcmp(action, "down") == 0 ? ScriptedMouse::Down
                       : std::strcmp(action, "up") == 0   ? ScriptedMouse::Up
                       : std::strcmp(action, "right") == 0 ? ScriptedMouse::Right
                       : std::strcmp(action, "wheelup") == 0   ? ScriptedMouse::WheelUp
                       : std::strcmp(action, "wheeldown") == 0 ? ScriptedMouse::WheelDown
                                                               : ScriptedMouse::Move;
        script.push_back(entry);
        p += used;
        p += *p == ',';
    }
    return script;
}

void RunMouseScript(double now) {
    static std::vector<ScriptedMouse> script = ParseMouseScript();
    for (ScriptedMouse& entry : script) {
        if (entry.done || now < entry.time) {
            continue;
        }
        entry.done = true;
        if (g_mouse_captured) {
            continue;
        }
        g_cursor_x = entry.x;
        g_cursor_y = entry.y;
        ++g_cursor_moves;
        if (entry.action == ScriptedMouse::Click || entry.action == ScriptedMouse::Down) {
            g_click_x = entry.x;
            g_click_y = entry.y;
            g_click_pending = true;
            g_click_time = std::chrono::steady_clock::now();
        }
        if (entry.action == ScriptedMouse::Down || entry.action == ScriptedMouse::Up) {
            g_menu_left_held = entry.action == ScriptedMouse::Down;
        }
        if (entry.action == ScriptedMouse::Right && MenuOpen()) {
            QueuePulse(button::Circle);
        }
        if (entry.action == ScriptedMouse::WheelUp || entry.action == ScriptedMouse::WheelDown) {
            MenuWheel(entry.action == ScriptedMouse::WheelUp ? 1.0f : -1.0f);
        }
        static constexpr const char* Names[] = {"move",  "click",   "down",     "up",
                                                "right", "wheelup", "wheeldown"};
        LOG_INFO("Input", "mouse script: {} {:.0f},{:.0f}", Names[entry.action], entry.x,
                 entry.y);
    }
}

// Presses and releases the scripted keys whose time has come.
void RunScript() {
    static std::vector<ScriptedKey> script = ParseScript();
    static const auto start = std::chrono::steady_clock::now();
    RunMouseScript(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
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
            if (held && entry.key == SDL_SCANCODE_ESCAPE) {
                g_escape_as_circle = MenuOpen();
            }
            g_script_keys[entry.key] = held;
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

void LoadControls(const char* data_dir) {
    std::scoped_lock lock{g_mutex};
    const std::filesystem::path path = std::filesystem::path{data_dir} / "controls.ini";
    std::ifstream file{path};
    if (!file) {
        // First run (or the file was deleted to get the defaults back): write the defaults, so
        // there is a file to edit.
        std::ofstream out{path};
        out << DefaultControls;
        ParseControls(DefaultControls, "defaults");
        if (out) {
            LOG_INFO("Input", "wrote the default controls to {}", path.string());
        } else {
            LOG_WARNING("Input", "{} could not be written: using the default controls",
                        path.string());
        }
        return;
    }
    std::ostringstream text;
    text << file.rdbuf();
    ParseControls(text.str(), path.string().c_str());
    LOG_INFO("Input", "controls from {}{}", path.string(),
             g_tag_names.empty() ? std::string{}
                                 : std::format(" ({} split-action tags)", g_tag_names.size()));
}

void Initialize() {
    std::scoped_lock lock{g_mutex};
    if (!g_bindings_loaded) {
        ParseControls(DefaultControls, "defaults");
    }
    int count = 0;
    if (SDL_JoystickID* ids = SDL_GetGamepads(&count)) {
        for (int i = 0; i < count; i++) {
            OpenGamepad(ids[i]);
        }
        SDL_free(ids);
    }
    if (g_gamepads.empty()) {
        LOG_INFO("Input", "no controller connected; using the keyboard and mouse");
    }
    Update();
}

// On the window's thread (SDL wants the mouse mode changed there).
void HandleEvent(const SDL_Event& event) {
    std::scoped_lock lock{g_mutex};
    switch (event.type) {
    case SDL_EVENT_KEY_DOWN:
        if (!event.key.repeat) {
            // F1 releases the mouse, or captures it again (menus free it by themselves).
            // Handled here and not in OnToolKey: that also runs for SOTC_KEY_SCRIPT on the
            // game's thread, and a benchmark must not grab the mouse of whoever is at the
            // computer.
            if (event.key.scancode == SDL_SCANCODE_ESCAPE) {
                g_escape_as_circle = MenuOpen();
            }
            if (event.key.scancode == SDL_SCANCODE_F1) {
                g_window = SDL_GetWindowFromID(event.key.windowID);
                g_capture_wanted = !g_capture_wanted;
                ApplyMouseMode();
            }
            OnToolKey(event.key.scancode);
        }
        [[fallthrough]];
    case SDL_EVENT_KEY_UP:
        if (event.key.scancode < SDL_SCANCODE_COUNT) {
            g_keys[event.key.scancode] = event.key.down;
        }
        break;
    case SDL_EVENT_MOUSE_MOTION:
        if (g_mouse_captured) {
            g_mouse_dx += event.motion.xrel;
            g_mouse_dy += event.motion.yrel;
            g_look_dx += event.motion.xrel;
            g_look_dy += event.motion.yrel;
            g_last_motion = std::chrono::steady_clock::now();
        } else {
            // A cursor: where it is in the game's UI, for the menus.
            WindowToUi(event.motion.x, event.motion.y, g_cursor_x, g_cursor_y);
            ++g_cursor_moves;
        }
        return; // nothing to recompute: Read() takes the movement
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
    case SDL_EVENT_MOUSE_BUTTON_UP:
        if (!g_mouse_captured) {
            g_window = SDL_GetWindowFromID(event.button.windowID);
            if (event.button.button == SDL_BUTTON_LEFT) {
                g_menu_left_held = event.button.down && MenuOpen();
            }
            if (!event.button.down) {
                return;
            }
            // Clicking into the window means playing with the mouse: once no menu is open, it
            // is captured.
            g_capture_wanted = true;
            if (MenuOpen()) {
                // In a menu: the left button picks what is under the cursor (the menu code
                // finds it), the right button goes back.
                if (event.button.button == SDL_BUTTON_LEFT) {
                    WindowToUi(event.button.x, event.button.y, g_click_x, g_click_y);
                    g_click_pending = g_click_x >= 0;
                    g_click_time = std::chrono::steady_clock::now();
                } else if (event.button.button == SDL_BUTTON_RIGHT) {
                    QueuePulse(button::Circle);
                }
                return;
            }
            // Out of menus a click into a free cursor only captures the mouse: it was meant for
            // the window, not for the game (and its release must not reach the game either).
            ApplyMouseMode();
            return;
        }
        if (event.button.button >= SDL_BUTTON_LEFT && event.button.button <= SDL_BUTTON_X2) {
            const std::uint32_t bit = 1u << event.button.button;
            g_mouse_held = event.button.down ? g_mouse_held | bit : g_mouse_held & ~bit;
        }
        break;
    case SDL_EVENT_MOUSE_WHEEL: {
        // A notch is y = +-1 (more in one event when the wheel spins fast); a "flipped" wheel
        // (natural scrolling) reports the opposite sign, turned back here.
        float notches = event.wheel.y;
        if (event.wheel.direction == SDL_MOUSEWHEEL_FLIPPED) {
            notches = -notches;
        }
        if (!g_mouse_captured) {
            MenuWheel(notches);
            return;
        }
        WheelPulse& wheel = g_wheel[notches < 0 ? 1 : 0];
        const int count = std::max(1, static_cast<int>(std::lround(std::abs(notches))));
        wheel.pending = std::min(wheel.pending + count, WheelMaxPending);
        return; // Read() presses the button, one notch at a time
    }
    case SDL_EVENT_WINDOW_FOCUS_LOST:
        // Key releases go to the window that has the focus: forget everything held, and let
        // the mouse go (it is captured again with a click).
        std::fill(std::begin(g_keys), std::end(g_keys), false);
        g_menu_left_held = false;
        g_capture_wanted = false;
        ApplyMouseMode();
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

// The game reads the controller once per update of its world: this is the clock of the
// mouse movement, the wheel's presses and the recordings.
PadState Read() {
    std::scoped_lock lock{g_mutex};
    RunScript();
    // The wheel's presses and releases advance one read at a time.
    const bool wheel_was_pressed[2] = {g_wheel[0].pressed, g_wheel[1].pressed};
    StepWheel(g_wheel[0]);
    StepWheel(g_wheel[1]);
    if (g_wheel[0].pressed != wheel_was_pressed[0] || g_wheel[1].pressed != wheel_was_pressed[1]) {
        Update();
    }
    // The presses the menus asked for.
    StepPulse();
    PadState state = g_state;
    if (g_menu_hold) {
        if (g_menu_left_held || g_menu_hold_reads < MenuHoldMinReads) {
            state.buttons |= g_menu_hold;
            ++g_menu_hold_reads;
        } else {
            g_menu_hold = 0;
        }
    }
    if (g_pulse_pressed) {
        state.buttons |= g_pulse_button;
        // The triggers are analog too: the game may read how far they are pressed.
        if (g_pulse_button & button::L2) {
            state.l2 = 0xFF;
        }
        if (g_pulse_button & button::R2) {
            state.r2 = 0xFF;
        }
    }
    // The mouse, when it moved, takes the right stick over from the gamepad and the arrows,
    // unless the game's free camera is taking it directly (native mouse camera). Each path
    // drops the movement it does not use, so none piles up for when the other takes over.
    if (std::chrono::steady_clock::now() - g_native_camera_seen < NativeCameraTimeout) {
        g_mouse_dx = g_mouse_dy = 0.0f;
    } else {
        g_look_dx = g_look_dy = 0.0f;
        MouseLook(state.right_x, state.right_y);
    }
    RecordOrReplay(state);
    RunDiagSequence();
    // What the game sees now, for ButtonHeldFor. A replay has no tags: all its buttons count as
    // untagged, so hooks leave them alone.
    const bool replay = g_replaying && g_replay_live;
    g_read_untagged_buttons = replay ? g_replay_state.buttons : g_untagged_buttons;
    for (int i = 0; i < 32; ++i) {
        g_read_tagged_buttons[i] = replay ? 0 : g_tagged_buttons[i];
    }
    return replay ? g_replay_state : state;
}

void PollWindow(SDL_Window* window) {
    std::scoped_lock lock{g_mutex};
    g_window = window;
    ApplyMouseMode();
}

void NoteMenuActive(bool map) {
    std::scoped_lock lock{g_mutex};
    g_menu_seen = std::chrono::steady_clock::now();
    if (map) {
        g_map_seen = g_menu_seen;
    }
}

bool MenuCursor(float& x, float& y, std::uint32_t& moves) {
    std::scoped_lock lock{g_mutex};
    if (g_mouse_captured || g_cursor_x < 0) {
        return false;
    }
    x = g_cursor_x;
    y = g_cursor_y;
    moves = g_cursor_moves;
    return true;
}

bool PeekMenuClick(float& x, float& y) {
    std::scoped_lock lock{g_mutex};
    if (g_click_pending && std::chrono::steady_clock::now() - g_click_time > MenuClickTimeout) {
        g_click_pending = false; // no menu found anything under it
    }
    x = g_click_x;
    y = g_click_y;
    return g_click_pending;
}

bool MenuDragging() {
    std::scoped_lock lock{g_mutex};
    return g_menu_left_held && !g_mouse_captured;
}

void TakeMenuClick() {
    std::scoped_lock lock{g_mutex};
    g_click_pending = false;
}

void HoldButtonWhileClicked(std::uint32_t button) {
    std::scoped_lock lock{g_mutex};
    g_menu_hold = button;
    g_menu_hold_reads = 0;
}

void PressButtonOnce(std::uint32_t button) {
    std::scoped_lock lock{g_mutex};
    QueuePulse(button);
}

bool MouseLookActive() {
    std::scoped_lock lock{g_mutex};
    // A mouse sends its movement in bursts (and none between two frames at a high polling
    // rate): count it as moving for a little while after its last movement.
    constexpr auto Linger = std::chrono::milliseconds{100};
    return g_mouse_settings.native_camera && !g_replaying && g_mouse_captured &&
           (g_look_dx != 0.0f || g_look_dy != 0.0f ||
            std::chrono::steady_clock::now() - g_last_motion < Linger);
}

bool TakeMouseLook(float& yaw, float& pitch) {
    std::scoped_lock lock{g_mutex};
    if (!g_mouse_settings.native_camera || g_replaying) {
        return false;
    }
    g_native_camera_seen = std::chrono::steady_clock::now();
    if (!g_mouse_captured || (g_look_dx == 0.0f && g_look_dy == 0.0f)) {
        g_look_dx = g_look_dy = 0.0f;
        return false;
    }
    constexpr float RadiansPerCount = 0.001f; // at mouse_sensitivity 1
    const float scale = RadiansPerCount * g_mouse_settings.sensitivity;
    yaw = g_look_dx * scale;
    pitch = (g_mouse_settings.invert_y ? -g_look_dy : g_look_dy) * scale;
    g_look_dx = g_look_dy = 0.0f;
    return true;
}

std::uint32_t TagBit(const char* name) {
    std::scoped_lock lock{g_mutex};
    for (size_t i = 0; i < g_tag_names.size(); ++i) {
        if (SameName(g_tag_names[i], name)) {
            return 1u << i;
        }
    }
    return 0;
}

bool ButtonHeldFor(std::uint32_t button, std::uint32_t tag_bit) {
    std::scoped_lock lock{g_mutex};
    if (g_read_untagged_buttons & button) {
        return true; // a gamepad or an untagged input: it means every action
    }
    return tag_bit != 0 && (g_read_tagged_buttons[std::countr_zero(tag_bit)] & button) != 0;
}

void NoteSaveLoaded() {
    std::scoped_lock lock{g_mutex};
    if (g_record && !g_record_loaded) {
        g_record_loaded = true;
        std::fprintf(g_record, "%llu LOAD\n", static_cast<unsigned long long>(g_reads));
        std::fflush(g_record);
    }
    if (g_replaying && !g_replay_live) {
        g_replay_live = true;
        g_replay_offset = static_cast<std::int64_t>(g_reads) -
                          static_cast<std::int64_t>(g_recorded_load);
        LOG_INFO("Input", "save loaded at read {}: the controller replay starts", g_reads);
    }
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
