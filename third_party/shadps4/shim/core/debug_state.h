#pragma once

// Replaces shadPS4's debugger state: nothing is recorded or paused.

#include <string>
#include <utility>
#include <vector>

#include "common/types.h"

namespace DebugStateType {

enum class QueueType {
    dcb = 0,
    ccb = 1,
    acb = 2,
};

struct QueueDump {
    QueueType type;
    u32 submit_num;
    u32 num2;
    std::vector<u32> data;
    uintptr_t base_addr;
};

class DebugStateImpl {
public:
    std::pair<u32, u32> game_resolution{};
    std::pair<u32, u32> output_resolution{};
    bool is_using_fsr{};

    void PauseGuestThreads() {}
    bool IsGuestThreadsPaused() const { return false; }
    void IncFlipFrameNum() {}
    void IncDrawCall() noexcept {}
    void IncDispatch() noexcept {}
    void IncGnmFrameNum() {}
    bool DumpingCurrentFrame() const { return false; }
    bool DumpingCurrentReg() { return false; }
    bool ShouldPauseInSubmit() const { return false; }
    void PushQueueDump(QueueDump) {}
    template <typename... Args>
    void PushRegsDump(Args&&...) {}
    template <typename... Args>
    void PushRegsDumpCompute(Args&&...) {}
    template <typename... Args>
    void CollectShader(Args&&...) {}
};

} // namespace DebugStateType

inline DebugStateType::DebugStateImpl DebugStateInstance;
#define DebugState DebugStateInstance
