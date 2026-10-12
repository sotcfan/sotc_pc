#include "common/log.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <string>

namespace logging {

namespace {

std::atomic<int> g_level{static_cast<int>(Level::Info)};
std::mutex g_mutex;
const auto g_start = std::chrono::steady_clock::now();
thread_local std::string t_thread_name = "?";

constexpr std::array<std::string_view, 6> LevelNames = {"Trace", "Debug", "Info",
                                                        "Warning", "Error", "CRITICAL"};

} // namespace

void SetLevel(Level level) {
    g_level = static_cast<int>(level);
}

bool Enabled(Level level) {
    return static_cast<int>(level) >= g_level.load(std::memory_order_relaxed);
}

bool ParseLevel(std::string_view text, Level& out) {
    constexpr std::array<std::string_view, 6> keys = {"trace", "debug", "info",
                                                      "warning", "error", "critical"};
    for (size_t i = 0; i < keys.size(); i++) {
        if (text == keys[i]) {
            out = static_cast<Level>(i);
            return true;
        }
    }
    return false;
}

void SetThreadName(std::string_view name) {
    t_thread_name = name;
}

const char* CurrentThreadName() {
    return t_thread_name.c_str();
}

void Write(Level level, std::string_view category, std::string_view message) {
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - g_start).count();
    const std::string line =
        std::format("[{:10.4f}] [{}] <{}> {}: {}\n", seconds, t_thread_name,
                    LevelNames[static_cast<int>(level)], category, message);
    std::scoped_lock lock{g_mutex};
    std::fwrite(line.data(), 1, line.size(), stderr);
}

} // namespace logging
