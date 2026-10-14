// Linux implementation of common/platform.h.
#ifndef _WIN32

#include "common/platform.h"

#include <fcntl.h>
#include <sys/file.h>
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

bool AcquireInstanceLock(const std::filesystem::path& data_dir) {
    std::error_code error;
    std::filesystem::create_directories(data_dir, error);
    const auto path = data_dir / "sotcport.lock";
    // Never closed: the lock lasts as long as the process (the kernel drops it at exit, also
    // after a crash).
    const int fd = open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    if (fd < 0) {
        return true; // cannot lock (read-only folder?): do not refuse to run
    }
    return flock(fd, LOCK_EX | LOCK_NB) == 0;
}

} // namespace platform

#endif // !_WIN32
