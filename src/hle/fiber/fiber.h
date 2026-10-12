#pragma once

#include <string>
#include <vector>

#include "common/types.h"

namespace hle::fiber {

struct SuspendedFiber {
    std::string name;
    u64 frame_pointer;  // rbp saved when it was suspended
    u64 resume_address; // where it continues
};

// Fibers that have run and are now waiting to be resumed (for thread dumps).
std::vector<SuspendedFiber> SuspendedFibers();

// One line per fiber that is not idle in a pool: name and state (for thread dumps).
std::vector<std::string> BusyFibers();

} // namespace hle::fiber
