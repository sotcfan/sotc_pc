// Replaces shadPS4's log backend (spdlog-based): its messages go to sotcport's log, with
// shadPS4's category names ("Render.Vulkan", "Lib.GnmDriver"...).

#include <fmt/format.h>

#include "common/logging/log.h"

// From sotcport's src/common/log.h. Declared here because that header's LOG_* macros
// would clash with shadPS4's.
namespace logging {
enum class Level : int { Trace, Debug, Info, Warning, Error, Critical };
bool Enabled(Level level);
void Write(Level level, std::string_view category, std::string_view message);
} // namespace logging

namespace Common::Log {

std::array<Level, NUM_LOG_CLASSES> g_class_levels{};

void Setup(std::string_view) {
    // Messages below sotcport's level are not even formatted.
    Level lowest = Level::Critical;
    for (int i = static_cast<int>(Level::Critical); i >= 0; i--) {
        if (logging::Enabled(static_cast<logging::Level>(i))) {
            lowest = static_cast<Level>(i);
        }
    }
    g_class_levels.fill(lowest);
}

void Switch(std::string_view, bool) {}
void Shutdown() {}
void Flush() {}
void Terminate() {}
void UpdateSinks() {}
void UpdateLogLevels(std::string_view) {}
void UpdateLogFlushLevel(std::string_view) {}

void VLog(Class log_class, Level level, const char*, int, const char*, fmt::string_view format,
          fmt::format_args args) {
    const auto ours = static_cast<logging::Level>(std::min<int>(static_cast<int>(level), 5));
    logging::Write(ours, NameOf(log_class), fmt::vformat(format, args));
}

} // namespace Common::Log
