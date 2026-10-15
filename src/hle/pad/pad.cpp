// libScePad: the controller. One controller is always connected: the PC's keyboard and game
// controllers combined (src/input). No motion sensors or touch positions yet: the controller
// lies flat and still and the touch pad is never touched, but its button works.

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_set>

#include "common/log.h"
#include "common/types.h"
#include "hle/registry.h"
#include "input/input.h"

namespace hle::pad {

namespace {

constexpr s32 PadHandle = 1;
constexpr s32 ErrorInvalidArgument = static_cast<s32>(0x80920001);
constexpr s32 ErrorInvalidPort = static_cast<s32>(0x80920002);
constexpr s32 ErrorInvalidHandle = static_cast<s32>(0x80920003);

struct PadData {
    u32 buttons;
    u8 left_stick[2];
    u8 right_stick[2];
    u8 analog_buttons[4]; // l2, r2, padding
    float orientation[4];
    float acceleration[3];
    float angular_velocity[3];
    struct {
        u8 count;
        u8 reserved[3];
        u32 held_time;
        struct {
            u16 x;
            u16 y;
            u8 id;
            u8 reserved[3];
        } touch[2];
    } touch_data;
    bool connected;
    u64 timestamp;
    struct {
        u32 id;
        u8 reserved;
        u8 length;
        u8 data[10];
    } extension_unit;
    u8 connected_count;
    u8 reserved[2];
    u8 device_unique_data_length;
    u8 device_unique_data[12];
};

struct ControllerInformation {
    float touch_pixel_density;
    u16 touch_resolution[2];
    u8 dead_zone_left;
    u8 dead_zone_right;
    u8 connection_type;
    u8 connected_count;
    bool connected;
    s32 device_class;
    u8 reserved[8];
};

struct VibrationParam {
    u8 large_motor;
    u8 small_motor;
};

struct LightBarParam {
    u8 red;
    u8 green;
    u8 blue;
    u8 reserved;
};

void FillState(PadData* data) {
    const input::PadState state = input::Read();
    std::memset(data, 0, sizeof(*data));
    data->buttons = state.buttons;
    data->left_stick[0] = state.left_x;
    data->left_stick[1] = state.left_y;
    data->right_stick[0] = state.right_x;
    data->right_stick[1] = state.right_y;
    data->analog_buttons[0] = state.l2;
    data->analog_buttons[1] = state.r2;
    data->orientation[3] = 1.0f; // identity quaternion: lying flat
    data->acceleration[1] = 1.0f; // gravity
    data->connected = true;
    data->connected_count = 1;
    data->timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
                          std::chrono::steady_clock::now().time_since_epoch())
                          .count();
}

// SOTC_LOG_PAD_CALLERS=1: logs every place of the game's code that reads the controller (once
// per place) and the buffer it reads into. The starting point for finding the game's own input
// code (src/input/game_hooks.cpp). `caller` is where the game's call returns to: the call
// instruction is just before it. Imports reach these functions with a jump, so the return
// address on entry is still the game's.
void LogCaller(const char* function, const void* caller, const PadData* data) {
    static const bool enabled = std::getenv("SOTC_LOG_PAD_CALLERS") != nullptr;
    if (!enabled) {
        return;
    }
    static std::mutex mutex;
    static std::unordered_set<const void*> seen;
    std::scoped_lock lock{mutex};
    if (seen.insert(caller).second) {
        LOG_INFO("Pad", "{} called from {} (returns to), into the buffer at {}", function, caller,
                 static_cast<const void*>(data));
    }
}

PS4_SYSV_ABI s32 scePadInit() {
    return 0;
}

PS4_SYSV_ABI s32 scePadOpen(s32, s32 type, s32 index, const void*) {
    // Only the standard controller port of the first user.
    return type == 0 && index == 0 ? PadHandle : ErrorInvalidPort;
}

PS4_SYSV_ABI s32 scePadClose(s32 handle) {
    return handle == PadHandle ? 0 : ErrorInvalidHandle;
}

PS4_SYSV_ABI s32 scePadReadState(s32 handle, PadData* data) {
    if (handle != PadHandle) {
        return ErrorInvalidHandle;
    }
    if (!data) {
        return ErrorInvalidArgument;
    }
    LogCaller("scePadReadState", __builtin_return_address(0), data);
    FillState(data);
    return 0;
}

// Returns how many samples were written: only the current state is kept, so one.
PS4_SYSV_ABI s32 scePadRead(s32 handle, PadData* data, s32 count) {
    if (handle != PadHandle) {
        return ErrorInvalidHandle;
    }
    if (!data || count < 1) {
        return ErrorInvalidArgument;
    }
    LogCaller("scePadRead", __builtin_return_address(0), data);
    FillState(data);
    return 1;
}

PS4_SYSV_ABI s32 scePadGetControllerInformation(s32 handle, ControllerInformation* info) {
    if (handle != PadHandle) {
        return ErrorInvalidHandle;
    }
    if (!info) {
        return ErrorInvalidArgument;
    }
    std::memset(info, 0, sizeof(*info));
    info->touch_pixel_density = 44.86f;
    info->touch_resolution[0] = 1920;
    info->touch_resolution[1] = 943;
    info->dead_zone_left = info->dead_zone_right = 13;
    info->connected_count = 1;
    info->connected = true;
    return 0;
}

PS4_SYSV_ABI s32 scePadSetLightBar(s32 handle, const LightBarParam* param) {
    if (handle != PadHandle) {
        return ErrorInvalidHandle;
    }
    if (!param) {
        return ErrorInvalidArgument;
    }
    input::SetLightBar(param->red, param->green, param->blue);
    return 0;
}

PS4_SYSV_ABI s32 scePadSetVibration(s32 handle, const VibrationParam* param) {
    if (handle != PadHandle) {
        return ErrorInvalidHandle;
    }
    if (!param) {
        return ErrorInvalidArgument;
    }
    input::SetVibration(param->large_motor, param->small_motor);
    return 0;
}

PS4_SYSV_ABI s32 scePadSetMotionSensorState(s32 handle, bool) {
    return handle == PadHandle ? 0 : ErrorInvalidHandle;
}

} // namespace

HLE_FUNCTION("libScePad", "scePadInit", scePadInit);
HLE_FUNCTION("libScePad", "scePadOpen", scePadOpen);
HLE_FUNCTION("libScePad", "scePadClose", scePadClose);
HLE_FUNCTION("libScePad", "scePadReadState", scePadReadState);
HLE_FUNCTION("libScePad", "scePadRead", scePadRead);
HLE_FUNCTION("libScePad", "scePadGetControllerInformation", scePadGetControllerInformation);
HLE_FUNCTION("libScePad", "scePadSetLightBar", scePadSetLightBar);
HLE_FUNCTION("libScePad", "scePadSetVibration", scePadSetVibration);
HLE_FUNCTION("libScePad", "scePadSetMotionSensorState", scePadSetMotionSensorState);

} // namespace hle::pad
