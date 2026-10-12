#pragma once

// The PC's keyboard and game controllers, seen as one PS4 controller.
//
// The thread that owns the window feeds every SDL event to HandleEvent(); libScePad reads the
// combined state with Read() from any thread. Also compiled into the GPU glue (with shadPS4's
// include paths), so this header only uses standard types.

#include <cstdint>

union SDL_Event;

namespace input {

// PS4 button bits, as in ScePadData::buttons.
namespace button {
constexpr std::uint32_t L3 = 0x2;
constexpr std::uint32_t R3 = 0x4;
constexpr std::uint32_t Options = 0x8;
constexpr std::uint32_t Up = 0x10;
constexpr std::uint32_t Right = 0x20;
constexpr std::uint32_t Down = 0x40;
constexpr std::uint32_t Left = 0x80;
constexpr std::uint32_t L2 = 0x100;
constexpr std::uint32_t R2 = 0x200;
constexpr std::uint32_t L1 = 0x400;
constexpr std::uint32_t R1 = 0x800;
constexpr std::uint32_t Triangle = 0x1000;
constexpr std::uint32_t Circle = 0x2000;
constexpr std::uint32_t Cross = 0x4000;
constexpr std::uint32_t Square = 0x8000;
constexpr std::uint32_t TouchPad = 0x100000;
} // namespace button

struct PadState {
    std::uint32_t buttons = 0;
    // Sticks: 0 = left/up, 0x80 = centered, 0xFF = right/down.
    std::uint8_t left_x = 0x80;
    std::uint8_t left_y = 0x80;
    std::uint8_t right_x = 0x80;
    std::uint8_t right_y = 0x80;
    // Triggers: 0 = released, 0xFF = fully pressed.
    std::uint8_t l2 = 0;
    std::uint8_t r2 = 0;
};

// On the window's thread, after SDL is initialized with SDL_INIT_GAMEPAD.
void Initialize();
void HandleEvent(const SDL_Event& event);

// From any thread.
PadState Read();
// Motor strengths 0–255 (left = large, right = small), as in ScePadVibrationParam.
void SetVibration(std::uint8_t large_motor, std::uint8_t small_motor);
void SetLightBar(std::uint8_t red, std::uint8_t green, std::uint8_t blue);

} // namespace input
