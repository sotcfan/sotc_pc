// Windows implementation of common/host_file.h.
#ifdef _WIN32

#include "common/host_file.h"

#include <windows.h>

#include <algorithm>

namespace host_file {

namespace {

HANDLE Native(Handle handle) {
    return reinterpret_cast<HANDLE>(handle);
}

// FILETIME counts 100 ns intervals since 1601-01-01; 11644473600 s separate it from 1970.
u64 Nanoseconds(const FILETIME& time) {
    const u64 ticks = (static_cast<u64>(time.dwHighDateTime) << 32) | time.dwLowDateTime;
    constexpr u64 Epoch1970 = 116444736000000000ull;
    return ticks >= Epoch1970 ? (ticks - Epoch1970) * 100 : 0;
}

} // namespace

Handle Open(const std::filesystem::path& path, int flags, OpenError* error) {
    DWORD access = 0;
    if (flags & OpenRead) {
        access |= GENERIC_READ;
    }
    if (flags & OpenWrite) {
        access |= GENERIC_WRITE;
    }
    if (access == 0) {
        access = GENERIC_READ;
    }
    DWORD disposition = OPEN_EXISTING;
    if ((flags & OpenCreate) && (flags & OpenTruncate)) {
        disposition = CREATE_ALWAYS;
    } else if (flags & OpenCreate) {
        disposition = OPEN_ALWAYS;
    } else if (flags & OpenTruncate) {
        disposition = TRUNCATE_EXISTING;
    }
    HANDLE file = CreateFileW(path.c_str(), access, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                              nullptr, disposition, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        if (error) {
            const DWORD code = GetLastError();
            *error = (code == ERROR_FILE_NOT_FOUND || code == ERROR_PATH_NOT_FOUND)
                         ? OpenError::NotFound
                         : OpenError::Other;
        }
        return Invalid;
    }
    if (error) {
        *error = OpenError::None;
    }
    if (flags & OpenAppend) {
        LARGE_INTEGER zero{};
        SetFilePointerEx(file, zero, nullptr, FILE_END);
    }
    return reinterpret_cast<Handle>(file);
}

void Close(Handle handle) {
    if (handle != Invalid) {
        CloseHandle(Native(handle));
    }
}

s64 ReadAt(Handle handle, void* buffer, u64 size, u64 offset) {
    u64 count = 0;
    while (count < size) {
        OVERLAPPED at{};
        at.Offset = static_cast<DWORD>(offset + count);
        at.OffsetHigh = static_cast<DWORD>((offset + count) >> 32);
        const DWORD chunk = static_cast<DWORD>(std::min<u64>(size - count, 1u << 30));
        DWORD got = 0;
        if (!ReadFile(Native(handle), static_cast<u8*>(buffer) + count, chunk, &got, &at)) {
            if (GetLastError() == ERROR_HANDLE_EOF) {
                break;
            }
            return -1;
        }
        if (got == 0) {
            break;
        }
        count += got;
    }
    return static_cast<s64>(count);
}

s64 WriteAt(Handle handle, const void* buffer, u64 size, u64 offset) {
    OVERLAPPED at{};
    at.Offset = static_cast<DWORD>(offset);
    at.OffsetHigh = static_cast<DWORD>(offset >> 32);
    DWORD written = 0;
    if (!WriteFile(Native(handle), buffer, static_cast<DWORD>(size), &written, &at)) {
        return -1;
    }
    return written;
}

s64 Size(Handle handle) {
    LARGE_INTEGER size;
    return GetFileSizeEx(Native(handle), &size) ? size.QuadPart : -1;
}

bool Truncate(Handle handle, u64 size) {
    FILE_END_OF_FILE_INFO info{};
    info.EndOfFile.QuadPart = static_cast<LONGLONG>(size);
    return SetFileInformationByHandle(Native(handle), FileEndOfFileInfo, &info, sizeof(info)) != 0;
}

void Sync(Handle handle) {
    FlushFileBuffers(Native(handle));
}

bool Stat(const std::filesystem::path& path, Info& info) {
    WIN32_FILE_ATTRIBUTE_DATA data;
    if (!GetFileAttributesExW(path.c_str(), GetFileExInfoStandard, &data)) {
        return false;
    }
    info.is_directory = (data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
    info.size = (static_cast<u64>(data.nFileSizeHigh) << 32) | data.nFileSizeLow;
    info.access_ns = Nanoseconds(data.ftLastAccessTime);
    info.modify_ns = Nanoseconds(data.ftLastWriteTime);
    info.change_ns = Nanoseconds(data.ftLastWriteTime); // no inode change time on Windows
    info.mode = info.is_directory ? 0040755 : 0100644;
    return true;
}

std::FILE* OpenStdio(const std::filesystem::path& path, const char* mode) {
    std::wstring wide_mode;
    for (const char* c = mode; *c; ++c) {
        wide_mode += static_cast<wchar_t>(*c);
    }
    if (wide_mode.find(L'b') == std::wstring::npos) {
        wide_mode += L'b';
    }
    return _wfopen(path.c_str(), wide_mode.c_str());
}

bool Remove(const std::filesystem::path& path) {
    return DeleteFileW(path.c_str()) != 0;
}

} // namespace host_file

#endif // _WIN32
