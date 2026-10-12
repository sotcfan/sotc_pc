#pragma once

// Replaces shadPS4's GPU memory viewer (a debugger window): always disabled.

#include <string>
#include <vector>

#include "common/types.h"

namespace Core::Devtools::Widget {

struct GpuMemoryRow {
    std::string name;
    u64 bytes{};
};

struct GpuMemoryGroup {
    std::string name;
    std::vector<GpuMemoryRow> rows;
};

class GpuMemoryViewer {
public:
    static bool IsEnabled() { return false; }
    static void Publish(GpuMemoryGroup) {}
};

} // namespace Core::Devtools::Widget
