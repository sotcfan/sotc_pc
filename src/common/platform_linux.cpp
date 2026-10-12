// Linux implementation of common/platform.h.
#ifndef _WIN32

#include "common/platform.h"

#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>

namespace platform {

u32 CurrentThreadId() {
    return static_cast<u32>(syscall(SYS_gettid));
}

std::string LastErrorText() {
    return std::strerror(errno);
}

std::filesystem::path DataDir() {
    if (const char* xdg = std::getenv("XDG_DATA_HOME"); xdg && *xdg) {
        return std::filesystem::path(xdg) / "sotcport";
    }
    if (const char* home = std::getenv("HOME"); home && *home) {
        return std::filesystem::path(home) / ".local/share/sotcport";
    }
    return "sotcport-data";
}

} // namespace platform

#endif // !_WIN32
