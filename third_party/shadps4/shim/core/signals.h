#pragma once

// Replaces shadPS4's signal handling: handlers are chained into sotcport's crash handler
// (src/loader/crash.h), which runs them before reporting a crash.

#include "common/types.h"

namespace Core {

using AccessViolationHandler = bool (*)(void* context, void* fault_address);
using IllegalInstructionHandler = bool (*)(void* context);

class Signals {
public:
    static Signals* Instance();
    void RegisterAccessViolationHandler(const AccessViolationHandler& handler, u32 priority);
    void RegisterIllegalInstructionHandler(const IllegalInstructionHandler& handler, u32 priority);
    void RemoveHandlers() {}
};

} // namespace Core
