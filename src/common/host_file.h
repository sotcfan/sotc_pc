#pragma once

#include <cstdio>
#include <filesystem>

#include "common/types.h"

// Host files with one interface for Linux (POSIX descriptors) and Windows (HANDLEs): positioned
// reads and writes, sizes and file information, as the game's file APIs (Fios2, PSARC archives)
// need them. Always binary.
namespace host_file {

// A descriptor (Linux) or a HANDLE (Windows); Invalid when closed.
using Handle = intptr_t;
constexpr Handle Invalid = -1;

enum OpenFlags : int {
    OpenRead = 1,
    OpenWrite = 2,
    OpenCreate = 4,   // create if missing
    OpenTruncate = 8, // empty an existing file
    OpenAppend = 16,  // writes go to the end
};

enum class OpenError { None, NotFound, Other };

Handle Open(const std::filesystem::path& path, int flags, OpenError* error = nullptr);
void Close(Handle handle);
// Bytes read at `offset` (fewer at the end of the file), or -1.
s64 ReadAt(Handle handle, void* buffer, u64 size, u64 offset);
// Bytes written at `offset`, or -1.
s64 WriteAt(Handle handle, const void* buffer, u64 size, u64 offset);
// Size in bytes, or -1.
s64 Size(Handle handle);
bool Truncate(Handle handle, u64 size);
void Sync(Handle handle);

struct Info {
    bool is_directory = false;
    u64 size = 0;
    // Nanoseconds since 1970-01-01 UTC.
    u64 access_ns = 0;
    u64 modify_ns = 0;
    u64 change_ns = 0;
    u32 mode = 0; // POSIX st_mode (Windows: synthesized, 0040755 or 0100644)
};
bool Stat(const std::filesystem::path& path, Info& info);

// fopen for a host path. On Windows the path is opened as UTF-16 (any characters) and in binary
// mode even if `mode` has no 'b' (text mode would translate line endings in the game's files).
std::FILE* OpenStdio(const std::filesystem::path& path, const char* mode);
bool Remove(const std::filesystem::path& path);

} // namespace host_file
