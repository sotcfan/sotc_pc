#include "diag/freeze_watch.h"

#include <dirent.h>

#include <chrono>
#include <cstdlib>
#include <format>
#include <fstream>
#include <sstream>
#include <string>

#include "common/log.h"
#include "hle/kernel/gpu_memory.h"
#include "loader/crash.h"

namespace diag {

namespace {

constexpr auto FreezeTime = std::chrono::seconds(10);

std::string ReadSmallFile(const std::string& path) {
    std::ifstream in{path};
    std::stringstream text;
    text << in.rdbuf();
    std::string result = text.str();
    while (!result.empty() && (result.back() == '\n' || result.back() == '\0')) {
        result.pop_back();
    }
    return result;
}

// Field 3 of /proc/<pid>/task/<tid>/stat, after the parenthesized name.
char ThreadState(const std::string& stat) {
    const size_t close = stat.rfind(')');
    return close != std::string::npos && close + 2 < stat.size() ? stat[close + 2] : '?';
}

void ReportThreads(double seconds) {
    LOG_ERROR("Freeze", "==== no frame for {:.0f} s: what every thread is doing ====", seconds);
    DIR* tasks = opendir("/proc/self/task");
    if (!tasks) {
        return;
    }
    while (const dirent* entry = readdir(tasks)) {
        if (entry->d_name[0] == '.') {
            continue;
        }
        const std::string base = std::format("/proc/self/task/{}/", entry->d_name);
        const std::string name = ReadSmallFile(base + "comm");
        const char state = ThreadState(ReadSmallFile(base + "stat"));
        const std::string wchan = ReadSmallFile(base + "wchan");
        // "syscall": number and arguments of the call it is blocked in (202 = futex, with the
        // lock address first), or "running".
        const std::string syscall = ReadSmallFile(base + "syscall");
        LOG_ERROR("Freeze", "  {:>7} {:<16} {} wchan={:<24} syscall={}", entry->d_name, name,
                  state, wchan.empty() ? "-" : wchan, syscall.substr(0, 120));
    }
    closedir(tasks);
    LOG_ERROR("Freeze", "{}", hle::kernel::DescribeMemoryLock());
    crash::DumpAllThreads();
}

} // namespace

void CheckForFreeze(unsigned long long frames_presented) {
    using Clock = std::chrono::steady_clock;
    static unsigned long long last_frames = 0;
    static Clock::time_point last_change = Clock::now();
    static bool reported = false;
    const auto now = Clock::now();
    if (frames_presented != last_frames) {
        if (reported) {
            LOG_INFO("Freeze", "frames are coming again");
        }
        last_frames = frames_presented;
        last_change = now;
        reported = false;
        return;
    }
    // Only once the game has started showing frames, and once per freeze.
    if (frames_presented == 0 || reported || now - last_change < FreezeTime) {
        return;
    }
    reported = true;
    ReportThreads(std::chrono::duration<double>(now - last_change).count());
}

} // namespace diag
