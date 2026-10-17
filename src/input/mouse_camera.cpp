// The native mouse camera: the mouse turns the game's free camera directly, instead of pushing
// the right stick.
//
// ==== Why ======================================================================================
//
// The game only knows a stick, and its camera treats the stick as a *speed*: the orientation
// controller (0xbcf490, from the original game's code, gamesys/camorictrler.cpp) averages the
// last 5 stick samples, turns the average into a target turning speed through a response curve,
// and eases the camera's speed toward it, up to 150 degrees per second. A mouse pushing the stick
// therefore turns with a delay (the average), an acceleration (the easing) and a top speed: it
// feels like "a mouse moving a stick", whatever the tuning. Here the mouse movement is added to
// the camera's angles after the controller's update, so the camera turns exactly as far as the
// mouse moved, at once; the stick path is left as it is for gamepads.
//
// ==== How the stick reaches the camera (found 2026-10-16 with diag/data_watch.h) ================
//
//   scePadRead -> engine device (0xe2bcf0: axis 2/3 = right stick)
//   -> engine action array (0xe2b2e0; right stick X = slot 157, Y = 158)
//   -> game manager copy per player (0x5c6de0, +0x2041d8)
//   -> PS2-style pad bytes (0x95ff50 -> 0x3308020, right stick at +6/+7)
//   -> original game's pad state (0x97bd20 -> 0x33359a0) and its history (0xa88e40 -> 0x3412e10)
//   -> "read right stick" (0xbc9a70, with its dead zone) into the camera (+0x16c8 / +0x16cc),
//      called by the camera's update (cameraman.cpp, 0xbb3640)
//   -> free camera mode (0xbbc690), which negates the stick and calls the orientation
//      controller's update 0xbcf490(controller, ?, ?, stick x, stick y, 4 more floats).
//
// The orientation controller's fields used here (offsets from its address, the first argument):
//   +0x44 / +0x48 / +0x4c   yaw: speed (rad/s), step of the last update, angle (rad, -pi..pi)
//   +0x50 / +0x54 / +0x58   pitch: the same three (rad/s, rad, rad)
//   +0x108 / +0x10c         pitch limits (+-1.222 rad = +-70 degrees, seen on foot)
//   +0x118 / +0x119         the game's "invert X / invert Y" camera options (bytes)
// Directions, from screenshots of SOTC_MOUSE_CAMERA_TEST runs: a growing yaw angle turns the view
// to the right, a growing pitch angle makes the camera look up. The mouse follows the PC
// convention (right turns right, down looks down) whatever the game's invert options say: they
// are for the stick (and were on in the player's profile, which turned the mouse round,
// 2026-10-16); mouse_invert_y in controls.ini flips the vertical for those who want it.
//
// The controller's update is also called by a scripted camera (0xb0799a, in a boss's script):
// only the call from the free camera mode is changed.
//
// The free camera mode only runs while the stick is off center: with the stick still, the
// camera's update (0xbb3640) tests the stick floats (camera + 0x16c8 / 0x16cc, at 0xbb420c and
// 0xbb42f0) and, when both are zero, lets the automatic camera follow Wander instead. So while
// the mouse moves, the stick reading (0xbc9a70) is made to report a push too small to turn the
// camera (StickNudge), and the game switches to its manual camera exactly as for a stick; when
// the mouse stops, the game's own delay brings the automatic camera back, as after a stick.
// Without this the mouse never reached the camera: the free camera mode was not running, so
// the mouse fell back to pushing the stick, which started the free camera mode, which took the
// mouse off the stick, and round again (2026-10-16, "it barely moved").

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "common/log.h"
#include "common/types.h"
#include "input/game_hooks.h"
#include "input/input.h"
#include "loader/hook.h"

namespace input {

namespace {

constexpr u64 OrientationUpdateAddress = 0xbcf490;
// Where the free camera mode's call (0xbbc8b7) returns to.
constexpr u64 FreeCameraReturnAddress = 0xbbc8bc;

constexpr u32 YawAngleOffset = 0x4c;
constexpr u32 PitchAngleOffset = 0x58;
constexpr u32 PitchLimitAOffset = 0x108;
constexpr u32 PitchLimitBOffset = 0x10c;
constexpr u32 InvertXOffset = 0x118;
constexpr u32 InvertYOffset = 0x119;

constexpr float Pi = 3.14159265358979f;

constexpr u64 ReadRightStickAddress = 0xbc9a70;
// The push reported while the mouse moves: nonzero for the game's tests, far too small to turn
// the camera (full push = 1, 150 degrees per second).
constexpr float StickNudge = 1e-4f;

using ReadRightStickFn = void PS4_SYSV_ABI (*)(float* x, float* y);
ReadRightStickFn g_original_read_stick = nullptr;

using OrientationUpdateFn = void PS4_SYSV_ABI (*)(u8* controller, s32 a, s32 b, float stick_x,
                                                  float stick_y, float c, float d, float e,
                                                  float f);
OrientationUpdateFn g_original = nullptr;

float& Field(u8* controller, u32 offset) {
    return *reinterpret_cast<float*>(controller + offset);
}

// SOTC_MOUSE_CAMERA_TEST="<from s>:<to s>:<yaw>:<pitch>" adds that many radians at every free
// camera update between those seconds of the run, as if the mouse moved right (yaw) and down
// (pitch) (a key script cannot move the mouse); every 10 updates the log gets the angles before
// the game's update and after it, to see whether the additions stick. With tools/bench_game.sh
// and screenshots, this is how the directions above were found.
struct CameraTest {
    double from = 0, to = 0;
    float yaw = 0, pitch = 0;
    bool on = false;
};
const CameraTest g_test = [] {
    CameraTest test;
    if (const char* text = std::getenv("SOTC_MOUSE_CAMERA_TEST")) {
        test.on = std::sscanf(text, "%lf:%lf:%f:%f", &test.from, &test.to, &test.yaw,
                              &test.pitch) == 4;
    }
    return test;
}();
const auto g_start = std::chrono::steady_clock::now();

double Seconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - g_start).count();
}

bool TestMoving() {
    if (!g_test.on) {
        return false;
    }
    const double now = Seconds();
    return now >= g_test.from && now < g_test.to;
}

void PS4_SYSV_ABI ReadRightStick(float* x, float* y) {
    g_original_read_stick(x, y);
    if (*x == 0.0f && *y == 0.0f && (MouseLookActive() || TestMoving())) {
        *x = StickNudge;
    }
}

void PS4_SYSV_ABI OrientationUpdate(u8* controller, s32 a, s32 b, float stick_x, float stick_y,
                                    float c, float d, float e, float f) {
    const float entry_yaw = Field(controller, YawAngleOffset);
    const float entry_pitch = Field(controller, PitchAngleOffset);
    g_original(controller, a, b, stick_x, stick_y, c, d, e, f);
    // The hook replaces the function's first instructions with a jump, so the return address
    // is still the game's call site.
    if (reinterpret_cast<u64>(__builtin_return_address(0)) != FreeCameraReturnAddress) {
        return;
    }
    float yaw = 0, pitch = 0;
    bool moved = TakeMouseLook(yaw, pitch);
    if (g_test.on) {
        const double now = Seconds();
        if (TestMoving()) {
            yaw += g_test.yaw;
            pitch += g_test.pitch;
            moved = true;
        }
        static u32 calls = 0;
        if (++calls % 10 == 0) {
            LOG_INFO("Input", "camera test t={:.1f}: yaw {:.4f} -> {:.4f} (+{:.4f}), pitch {:.4f} "
                              "-> {:.4f} (+{:.4f}), stick ({:.2f},{:.2f}), invert {}/{}",
                     now, entry_yaw, Field(controller, YawAngleOffset), yaw, entry_pitch,
                     Field(controller, PitchAngleOffset), pitch, stick_x, stick_y,
                     controller[InvertXOffset], controller[InvertYOffset]);
        }
    }
    if (!moved) {
        return;
    }
    // TakeMouseLook gives down as positive; the pitch angle grows looking up.
    pitch = -pitch;
    float& yaw_angle = Field(controller, YawAngleOffset);
    yaw_angle += yaw;
    // Kept in -pi..pi like the game keeps it.
    while (yaw_angle > Pi) {
        yaw_angle -= 2.0f * Pi;
    }
    while (yaw_angle < -Pi) {
        yaw_angle += 2.0f * Pi;
    }
    const float limit_a = Field(controller, PitchLimitAOffset);
    const float limit_b = Field(controller, PitchLimitBOffset);
    float& pitch_angle = Field(controller, PitchAngleOffset);
    pitch_angle = std::clamp(pitch_angle + pitch, std::min(limit_a, limit_b),
                             std::max(limit_a, limit_b));
}

} // namespace

void InstallMouseCamera() {
    g_original_read_stick = reinterpret_cast<ReadRightStickFn>(loader::HookFunction(
        ReadRightStickAddress, reinterpret_cast<const void*>(&ReadRightStick)));
    if (!g_original_read_stick) {
        LOG_ERROR("Input", "native mouse camera: could not hook the stick reading at {:#x}",
                  ReadRightStickAddress);
        return;
    }
    g_original = reinterpret_cast<OrientationUpdateFn>(loader::HookFunction(
        OrientationUpdateAddress, reinterpret_cast<const void*>(&OrientationUpdate)));
    if (!g_original) {
        LOG_ERROR("Input", "native mouse camera: could not hook the camera at {:#x}; the mouse "
                           "turns it through the right stick",
                  OrientationUpdateAddress);
    }
}

} // namespace input
