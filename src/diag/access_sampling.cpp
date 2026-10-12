#ifndef _WIN32 // perf sampling (F11) is Linux-only; Windows gets the stub at the end

#include "diag/access_sampling.h"

#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <string>
#include <thread>
#include <vector>

#include "common/log.h"
#include "common/types.h"

extern char** environ;

namespace diag {

namespace {

constexpr int WindowSeconds = 30;

std::atomic<bool> g_running{false};

std::filesystem::path SamplesDir() {
    if (const char* xdg = std::getenv("XDG_DATA_HOME"); xdg && *xdg) {
        return std::filesystem::path{xdg} / "sotcport/samples";
    }
    return std::filesystem::path{std::getenv("HOME")} / ".local/share/sotcport/samples";
}

u64 MonotonicNs() {
    timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return static_cast<u64>(now.tv_sec) * 1'000'000'000 + now.tv_nsec;
}

void Run(std::filesystem::path dir) {
    // perf uses CLOCK_MONOTONIC (-k), so each sample can be matched with the maps snapshot
    // taken before it (snapshot file names are monotonic nanoseconds).
    const std::string pid = std::to_string(getpid());
    const std::string output = (dir / "perf.data").string();
    const std::string duration = std::to_string(WindowSeconds);
    std::vector<std::string> args = {
        "perf", "record", "-e", "cycles:u", "-F", "2999", "-k", "CLOCK_MONOTONIC",
        "--call-graph", "fp",
        "--intr-regs=AX,BX,CX,DX,SI,DI,BP,SP,IP,R8,R9,R10,R11,R12,R13,R14,R15",
        "-p", pid, "-o", output, "--", "sleep", duration};
    std::vector<char*> argv;
    for (auto& arg : args) {
        argv.push_back(arg.data());
    }
    argv.push_back(nullptr);

    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    const std::string log = (dir / "perf.txt").string();
    posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, log.c_str(),
                                     O_WRONLY | O_CREAT | O_TRUNC, 0644);
    posix_spawn_file_actions_adddup2(&actions, STDOUT_FILENO, STDERR_FILENO);
    pid_t perf = 0;
    const int error = posix_spawnp(&perf, "perf", &actions, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    if (error != 0) {
        LOG_ERROR("Sampling", "could not start perf: {}", std::strerror(error));
        g_running = false;
        return;
    }

    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(WindowSeconds + 1);
    while (std::chrono::steady_clock::now() < end) {
        std::error_code ignored;
        std::filesystem::copy_file("/proc/self/maps", dir / std::format("maps_{}.txt", MonotonicNs()),
                                   ignored);
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    int status = 0;
    waitpid(perf, &status, 0);
    LOG_INFO("Sampling", "==== sampling finished: {} ====", dir.string());
    g_running = false;
}

} // namespace

void StartAccessSampling() {
    if (g_running.exchange(true)) {
        LOG_INFO("Sampling", "sampling already running");
        return;
    }
    static int count = 0;
    const auto base = SamplesDir();
    std::error_code error;
    std::filesystem::create_directories(base, error);
    std::filesystem::path dir;
    do {
        dir = base / std::to_string(++count);
    } while (std::filesystem::exists(dir));
    std::filesystem::create_directories(dir, error);
    LOG_INFO("Sampling", "==== SAMPLING {} started (F11), {} s: {} ====", count, WindowSeconds,
             dir.string());
    std::thread{Run, dir}.detach();
}

} // namespace diag

#else

#include "diag/access_sampling.h"

namespace diag {
void StartAccessSampling() {}
} // namespace diag

#endif
