#ifndef _WIN32
#include <signal.h>
#include <unistd.h>
#endif

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "common/platform.h"
#include "common/log.h"
#include "diag/gnm_recorder.h"
#include "diag/gnm_trace.h"
#include "diag/resource_dump.h"
#include "diag/copy_watch.h"
#include "diag/freeze_watch.h"
#include "diag/memory_types.h"
#include "diag/packet_stats.h"
#include "hle/kernel/filesystem.h"
#include "hle/savedata/savedata.h"
#include "gpu/gpu.h"
#include "input/game_hooks.h"
#include "input/input.h"
#include "loader/crash.h"
#include "loader/game_version.h"
#include "loader/linker.h"

namespace {

// The address the game's executable was linked for. Keeping it means every address in our logs
// matches the game executable as disassemblers load it.
constexpr u64 EbootBase = 0x400000;

constexpr std::string_view Usage = R"(Usage: sotcport <game folder> [options] [-- game arguments]

The game folder is the decrypted dump (the one with eboot.bin and the pak*.psarc files).
Nothing is written to it.

Options:
  --trace <file>       records, in order, the first call to each Sony function
  --log <level>        trace, debug, info, warning, error (default: info)
  --data <folder>      where the game's writable data lives
                       (default: ~/.local/share/sotcport)
  --dump-after <s>     after s seconds, every game thread shows where it is
                       (kill -USR1 <pid> does the same at any time)
)";

std::filesystem::path DefaultDataDir() {
    return platform::DataDir();
}

// The main thread handles the window (SDL needs that) and thread dumps while the game runs on
// its own threads. Closing the window ends the program.
[[noreturn]] void RunMainLoop(int dump_after_seconds) {
#ifndef _WIN32
    sigset_t dump_signal;
    sigemptyset(&dump_signal);
    sigaddset(&dump_signal, SIGUSR1);
#endif
    const auto start = std::chrono::steady_clock::now();
    bool timed_dump_done = dump_after_seconds <= 0;
    while (gpu::ProcessEvents(50)) {
#ifndef _WIN32 // Windows: no SIGUSR1; --dump-after and the freeze watch remain
        const timespec no_wait{0, 0};
        if (sigtimedwait(&dump_signal, nullptr, &no_wait) == SIGUSR1) {
            crash::DumpAllThreads();
        }
#endif
        diag::CheckForFreeze(gpu::FramesPresented());
        if (!timed_dump_done &&
            std::chrono::steady_clock::now() - start >= std::chrono::seconds(dump_after_seconds)) {
            crash::DumpAllThreads();
            timed_dump_done = true;
        }
    }
    LOG_INFO("Main", "window closed; exiting");
    gpu::FinishScreenshots(); // the last F9 screenshots may still be being written
    std::fflush(nullptr);
    std::_Exit(0);
}

} // namespace

int main(int argc, char** argv) {
    logging::SetThreadName("Main");
#ifndef _WIN32
    // SIGUSR1 asks for a thread dump (RunMainLoop waits for it). Blocked before any thread
    // exists: every thread inherits the mask, so none of them (the window system's, the GPU
    // layer's) is killed by it.
    sigset_t dump_signal;
    sigemptyset(&dump_signal);
    sigaddset(&dump_signal, SIGUSR1);
    pthread_sigmask(SIG_BLOCK, &dump_signal, nullptr);
#endif

    std::filesystem::path game_dir, data_dir = DefaultDataDir();
    loader::LinkerOptions options;
    int dump_after_seconds = 0;
    std::vector<std::string> game_args = {"/app0/eboot.bin"};

    for (int i = 1; i < argc; i++) {
        const std::string_view arg = argv[i];
        const bool has_value = i + 1 < argc;
        if (arg == "--") {
            game_args.insert(game_args.end(), argv + i + 1, argv + argc);
            break;
        } else if (arg == "--trace" && has_value) {
            options.trace_file = argv[++i];
        } else if (arg == "--dump-after" && has_value) {
            dump_after_seconds = std::atoi(argv[++i]);
        } else if (arg == "--data" && has_value) {
            data_dir = argv[++i];
        } else if (arg == "--log" && has_value) {
            logging::Level level;
            if (!logging::ParseLevel(argv[++i], level)) {
                std::fprintf(stderr, "unknown log level: %s\n", argv[i]);
                return 2;
            }
            logging::SetLevel(level);
        } else if (arg == "-h" || arg == "--help") {
            std::fputs(Usage.data(), stdout);
            return 0;
        } else if (game_dir.empty() && !arg.starts_with("-")) {
            game_dir = arg;
        } else {
            std::fprintf(stderr, "unknown option: %s\n\n%s", argv[i], Usage.data());
            return 2;
        }
    }
    if (game_dir.empty()) {
        std::fputs(Usage.data(), stderr);
        return 2;
    }

    // The game's own messages go to stdout; keep them in order with our log on stderr.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    try {
        crash::Install();
        // In the default data folder whatever --data says: two at once also fight over the
        // GPU and the window (benchmarks run with a data folder of their own).
        if (!platform::AcquireInstanceLock(DefaultDataDir())) {
            std::fprintf(stderr, "error: sotcport is already running; close it first\n");
            return 1;
        }
        // The options this run was started with, for reading a log later.
        std::string options_text;
#ifdef _WIN32
        char** const environment = _environ;
#else
        char** const environment = environ;
#endif
        for (char** env = environment; *env; ++env) {
            if (std::strncmp(*env, "SOTC_", 5) == 0 && std::strncmp(*env, "SOTC_GAME=", 10) != 0) {
                options_text += ' ';
                options_text += *env;
            }
        }
        LOG_INFO("Main", "options:{}", options_text.empty() ? " (none)" : options_text);
        hle::kernel::InitializeFilesystem(std::filesystem::absolute(game_dir), data_dir);
        hle::savedata::Initialize(data_dir);
        input::LoadControls(data_dir.string().c_str()); // controls.ini, before the window opens

        if (!loader::CheckGameVersion(game_dir / "eboot.bin")) {
            return 1;
        }
        loader::Linker linker{options};
        loader::Module& eboot = linker.LoadExecutable(game_dir / "eboot.bin", EbootBase);

        // The renderer must exist before the game maps memory, and its functions must be
        // registered before the game is linked.
        const auto* param = eboot.GetProcParam();
        if (!gpu::Initialize(param ? static_cast<int>(param->sdk_version) : 0)) {
            return 1;
        }
        diag::InstallGnmTracer();    // SOTC_TRACE_GNM
        diag::InstallGnmRecorder();  // SOTC_RECORD_GNM
        diag::InstallResourceDump(); // SOTC_DUMP_RESOURCES
        diag::InstallCopyWatch();    // SOTC_WATCH_COPIES
        diag::InstallPacketStats();  // SOTC_PACKET_STATS
        diag::InstallMemoryTypes();  // SOTC_MEMORY_TYPES
        input::InstallGameHooks();   // split actions (src/input/game_hooks.cpp)
        linker.Link(eboot);
        linker.Start(eboot, game_args);
        RunMainLoop(dump_after_seconds);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
