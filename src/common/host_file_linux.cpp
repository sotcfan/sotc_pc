// Linux implementation of common/host_file.h.
#ifndef _WIN32

#include "common/host_file.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>

namespace host_file {

namespace {

u64 Nanoseconds(const timespec& time) {
    return static_cast<u64>(time.tv_sec) * 1'000'000'000ull + static_cast<u64>(time.tv_nsec);
}

} // namespace

Handle Open(const std::filesystem::path& path, int flags, OpenError* error) {
    int host = (flags & OpenWrite) ? ((flags & OpenRead) ? O_RDWR : O_WRONLY) : O_RDONLY;
    if (flags & OpenAppend) {
        host |= O_APPEND;
    }
    if (flags & OpenCreate) {
        host |= O_CREAT;
    }
    if (flags & OpenTruncate) {
        host |= O_TRUNC;
    }
    const int fd = open(path.c_str(), host | O_CLOEXEC, 0644);
    if (error) {
        *error = fd >= 0 ? OpenError::None
                 : (errno == ENOENT || errno == ENOTDIR) ? OpenError::NotFound
                                                        : OpenError::Other;
    }
    return fd >= 0 ? fd : Invalid;
}

void Close(Handle handle) {
    if (handle != Invalid) {
        close(static_cast<int>(handle));
    }
}

s64 ReadAt(Handle handle, void* buffer, u64 size, u64 offset) {
    u64 count = 0;
    while (count < size) {
        const ssize_t got = pread(static_cast<int>(handle), static_cast<u8*>(buffer) + count,
                                  size - count, static_cast<off_t>(offset + count));
        if (got < 0) {
            return -1;
        }
        if (got == 0) {
            break;
        }
        count += static_cast<u64>(got);
    }
    return static_cast<s64>(count);
}

s64 WriteAt(Handle handle, const void* buffer, u64 size, u64 offset) {
    return pwrite(static_cast<int>(handle), buffer, size, static_cast<off_t>(offset));
}

s64 Size(Handle handle) {
    struct stat info;
    return fstat(static_cast<int>(handle), &info) == 0 ? info.st_size : -1;
}

bool Truncate(Handle handle, u64 size) {
    return ftruncate(static_cast<int>(handle), static_cast<off_t>(size)) == 0;
}

void Sync(Handle handle) {
    fsync(static_cast<int>(handle));
}

bool Stat(const std::filesystem::path& path, Info& info) {
    struct stat host;
    if (stat(path.c_str(), &host) != 0) {
        return false;
    }
    info.is_directory = S_ISDIR(host.st_mode);
    info.size = static_cast<u64>(host.st_size);
    info.access_ns = Nanoseconds(host.st_atim);
    info.modify_ns = Nanoseconds(host.st_mtim);
    info.change_ns = Nanoseconds(host.st_ctim);
    info.mode = host.st_mode;
    return true;
}

std::FILE* OpenStdio(const std::filesystem::path& path, const char* mode) {
    return std::fopen(path.c_str(), mode);
}

bool Remove(const std::filesystem::path& path) {
    return unlink(path.c_str()) == 0;
}

} // namespace host_file

#endif // !_WIN32
