// The game's step of time per frame, and the physics that did not follow it.
//
// ==== The defect ================================================================================
//
// Played at exactly 60 or exactly 30 frames per second everything is fine; at any other rate, or
// when the frame time wobbles, Wander's cape shakes wildly and Agro runs ahead and snaps back.
// The cause (found 2026-10-19 with a probe on the functions below): the world and the physics are
// moved by two different clocks.
//
// ==== How the game keeps time ===================================================================
//
// The engine's main loop (GameCoreLoop, 0x65c080) keeps a frame clock on its stack. At the start
// of each frame 0x65ccc0(clock, reset) reads the TSC and works out the step of this frame:
//
//   clock +0x08  the game core object; its +0x28c8 points to a bool, "quantized step" (on)
//   clock +0x18  double, the TSC time of the previous frame, in seconds
//   clock +0x28  float, the step of this frame (seconds)
//   clock +0x2c  float, the same step, the one handed out: 0x6610a0 (slot 3 of the clock's
//                vtable) returns it and zeroes it, and 0x65ccc0 adds what is left of it to the
//                next frame's time
//   clock +0x30  float, the time measured but not yet handed out (see below)
//
// The "nominal" step, a float at 0x32ee7b0, is 1 / the rate the video controller paces the game at:
// 1/59.94 s in the 60 FPS modes, 1/29.97 s when the game asks for 30. With the quantized step on
// (always, in the runs measured), the step is the nominal one plus or minus whole quanta of half
// of it (8.34 ms at 60), and the rest of the measured time waits in +0x30 for the next frames: at
// a steady 60 every step is 16.68 ms; at 85 FPS they go 8.34, 16.68, 8.34, 8.34, 16.68...; above
// 120 FPS some steps are 0. The step is capped by a cvar (100 ms).
//
// 0x65ce80 (the rest of "PreProcessGameLoop") takes the step and gives it to the time scale
// manager (0x6616f0, game core + 0x3c10), which scales it per time group for the game objects:
// Wander, Agro, the colossi, the animations and the original game's code all move by it.
//
// The physics does not: the game's update (0x65bb30) starts the physics with
// 0xe3f140(nominal step, physics manager) (BPE/Source/Engine/Physics/PhysX/
// PhysXPhysicsManager.cpp; 0 while the game is paused), which stores the step at manager + 0x130
// and runs "PhysicsManager::PhysicsJob" (0xe3f0f0): the APEX scene's simulate(step), with the
// clothing (Wander's cape) in it. So the physics always moves 1/60 s per frame (1/30 in the 30 FPS
// mode), whatever time went by. At exactly 60 (or 30) that is the same as the world's step; at 85
// FPS the physics runs 1.4 times faster than the world, and against steps that alternate between
// half and a whole of its own. The cape is pulled by Wander's bones (moved by the step) and
// integrated with the other clock: it shakes. Agro's physics runs ahead of her animation and is
// pulled back. A steady 40 or 45 does the same (steps of 25.03 ms, or 16.68 and 25.03 alternating,
// against 16.68 for the physics), which is why only 30 and 60 ever worked.
//
// ==== The fix ===================================================================================
//
// 1. The physics gets the step of the frame instead of the nominal one (0xe3f140 is hooked). At a
//    steady 60 or 30 the step is exactly the nominal value, so nothing changes there. The step
//    given to the physics is capped at 1/29.97 s (never more than the game ever gave it), and a
//    step of 0 skips the physics for that frame, as the game's pause does.
// 2. The step itself is smoothed (0x65ccc0 is hooked; the game's value is replaced after it runs):
//    instead of jumping between whole quanta it follows the measured frame time, averaged, and
//    pays back the difference a little every frame, so that the game's time keeps up with real
//    time. Even with both clocks agreeing, a cape integrated with steps that double and halve
//    from one frame to the next is jerky, and so is everything on screen. A hitch (a frame much
//    longer than the ones before it: loading, a shader being built) is not paid back: the game
//    stands still for it instead of jumping ahead.
//
// SOTC_GAME_STEP=game keeps the game's own (quantized) step; SOTC_PHYSICS_STEP=nominal gives the
// physics the nominal step again, as the game does. SOTC_LOG_FRAME_STEP=1 logs every frame: the
// measured time, the game's step, the step given and the physics step.

#include "game/frame_step.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string_view>

#include "common/log.h"
#include "common/types.h"
#include "loader/hook.h"

namespace game {

namespace {

constexpr u64 FrameClockAddress = 0x65ccc0;
constexpr u64 PhysicsStepAddress = 0xe3f140;
constexpr u64 NominalStepAddress = 0x32ee7b0;
// The cvar that caps the step: a pointer to a float, seconds.
constexpr u64 MaxStepCvarAddress = 0x32ecd70;

constexpr u32 ClockLastTimeOffset = 0x18;
constexpr u32 ClockStepOffset = 0x28;
constexpr u32 ClockStepOutOffset = 0x2c;

// The longest step the physics ever got from the game (the nominal step of the 30 FPS mode).
constexpr float MaxPhysicsStep = 1.0f / 29.97f;

// The reset argument is a bool in the low byte of esi; taken whole so that it reaches the
// original exactly as the game passed it.
using FrameClockFn = void PS4_SYSV_ABI (*)(u8* clock, u32 reset);
using PhysicsStepFn = void PS4_SYSV_ABI (*)(float step, void* manager);

FrameClockFn g_original_clock = nullptr;
PhysicsStepFn g_original_physics = nullptr;

bool g_smooth = true;           // SOTC_GAME_STEP != game
bool g_physics_follows = true;  // SOTC_PHYSICS_STEP != nominal
bool g_log = false;             // SOTC_LOG_FRAME_STEP

// Both hooks run on the game's main thread, one after the other in each frame.
float g_frame_step = 0.0f;  // the step handed out this frame (seconds)
float g_game_step = 0.0f;   // what the game had worked out (for the log)
double g_elapsed = 0.0;     // the measured frame time (for the log)

template <typename T>
T& Field(u8* base, u32 offset) {
    return *reinterpret_cast<T*>(base + offset);
}

float NominalStep() {
    return *reinterpret_cast<const float*>(NominalStepAddress);
}

// The smoothed step. Times in seconds.
class StepSmoother {
public:
    void Reset(double nominal) {
        average_ = nominal;
        debt_ = 0.0;
    }

    double Next(double elapsed, double max_step) {
        // A frame much longer than the average is a hitch: only part of it counts.
        elapsed = std::clamp(elapsed, 0.0, average_ * 2.0 + 0.002);
        average_ += (elapsed - average_) * AverageWeight;
        // debt_: real time that went by and was not given to the game yet. Each frame gets the
        // average frame time plus a tenth of what is owed beyond it: at a steady rate the debt
        // fades out and the step settles on the frame time.
        debt_ += elapsed;
        double step = average_ + (debt_ - average_) * PaybackWeight;
        step = std::clamp(step, 0.0, max_step);
        debt_ = std::clamp(debt_ - step, -MaxDebt, MaxDebt);
        return step;
    }

private:
    static constexpr double AverageWeight = 0.1;
    static constexpr double PaybackWeight = 0.1;
    // More than this behind (or ahead of) real time is dropped instead of paid back.
    static constexpr double MaxDebt = 0.05;

    double average_ = 1.0 / 59.94;
    double debt_ = 0.0;
};

StepSmoother g_smoother;

void PS4_SYSV_ABI FrameClock(u8* clock, u32 reset) {
    const double last_time = Field<double>(clock, ClockLastTimeOffset);
    g_original_clock(clock, reset);
    const double elapsed = Field<double>(clock, ClockLastTimeOffset) - last_time;
    const float game_step = Field<float>(clock, ClockStepOutOffset);
    float step = game_step;
    if (g_smooth) {
        if ((reset & 0xff) != 0) {
            // After a load the game starts over from one nominal step: so does the smoothing.
            g_smoother.Reset(NominalStep());
        } else {
            const float max_step = **reinterpret_cast<float* const*>(MaxStepCvarAddress);
            step = static_cast<float>(g_smoother.Next(elapsed, max_step));
            Field<float>(clock, ClockStepOffset) = step;
            Field<float>(clock, ClockStepOutOffset) = step;
        }
    }
    g_frame_step = step;
    g_game_step = game_step;
    g_elapsed = elapsed;
}

void PS4_SYSV_ABI PhysicsStep(float step, void* manager) {
    const float nominal = step;
    // 0: the game is paused; leave it so.
    if (g_physics_follows && step > 0.0f) {
        step = std::min(g_frame_step, std::max(MaxPhysicsStep, NominalStep()));
    }
    if (g_log) {
        LOG_INFO("FrameStep", "elapsed {:6.2f} ms  game step {:6.2f}  step {:6.2f}  physics {:6.2f} "
                              "(nominal {:.2f})",
                 g_elapsed * 1000.0, g_game_step * 1000.0f, g_frame_step * 1000.0f,
                 step * 1000.0f, nominal * 1000.0f);
    }
    g_original_physics(step, manager);
}

bool EnvIs(const char* name, std::string_view value) {
    const char* env = std::getenv(name);
    return env && value == env;
}

} // namespace

void InstallFrameStep() {
    g_smooth = !EnvIs("SOTC_GAME_STEP", "game");
    g_physics_follows = !EnvIs("SOTC_PHYSICS_STEP", "nominal");
    const char* log = std::getenv("SOTC_LOG_FRAME_STEP");
    g_log = log && *log && *log != '0';
    if (!g_smooth && !g_physics_follows && !g_log) {
        return;
    }
    g_original_clock = reinterpret_cast<FrameClockFn>(
        loader::HookFunction(FrameClockAddress, reinterpret_cast<const void*>(&FrameClock)));
    if (!g_original_clock) {
        LOG_ERROR("FrameStep", "could not hook the frame clock at {:#x}; the game keeps its own "
                               "step and the physics its nominal one",
                  FrameClockAddress);
        return;
    }
    g_original_physics = reinterpret_cast<PhysicsStepFn>(
        loader::HookFunction(PhysicsStepAddress, reinterpret_cast<const void*>(&PhysicsStep)));
    if (!g_original_physics) {
        LOG_ERROR("FrameStep", "could not hook the physics step at {:#x}", PhysicsStepAddress);
    }
    LOG_INFO("FrameStep", "game step: {}; physics step: {}", g_smooth ? "smoothed" : "the game's",
             g_physics_follows ? "the frame's" : "nominal");
}

} // namespace game
