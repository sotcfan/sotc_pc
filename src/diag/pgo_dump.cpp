// Profile dump for instrumented builds (SOTC_PGO=generate, see the top-level CMakeLists.txt).
//
// Clang's profile runtime writes the counters when the process exits normally, but the game
// usually ends through _exit, a signal or `timeout`, so a background thread writes them every
// 30 s instead. The counters keep accumulating in memory and every write replaces the file
// with the totals so far, so the last write of a session is its whole profile. One file per
// process (pid in the name); merge them with llvm-profdata.

#ifdef SOTC_PGO_GENERATE

#include <chrono>
#include <filesystem>
#include <string>
#include <thread>

#include <unistd.h>


extern "C" {
int __llvm_profile_write_file(void);
void __llvm_profile_set_filename(const char* name);
}

namespace {

struct PgoDump {
    PgoDump() {
        std::filesystem::create_directories(SOTC_PGO_DIR);
        // The runtime keeps the pointer: the string must outlive the process.
        static const std::string path =
            std::string(SOTC_PGO_DIR) + "/sotcport-" + std::to_string(getpid()) + ".profraw";
        __llvm_profile_set_filename(path.c_str());
        std::thread([] {
            for (;;) {
                std::this_thread::sleep_for(std::chrono::seconds(30));
                __llvm_profile_write_file();
            }
        }).detach();
    }
};

PgoDump g_pgo_dump;

} // namespace

#endif
