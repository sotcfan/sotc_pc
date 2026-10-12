#pragma once

#include <filesystem>
#include <string_view>

namespace hle::kernel {

// Sets where the game's paths live on the PC. "/app0" is the game folder and is never written
// to; the writable PS4 locations (/data, /temp0, ...) go under `data_dir`.
void InitializeFilesystem(const std::filesystem::path& game_dir,
                          const std::filesystem::path& data_dir);

struct HostPath {
    std::filesystem::path path;
    bool read_only = false;
    bool valid = false; // false when the path is outside every mount point
};

// Save data folders appear at "/savedata0".."/savedata15" only while mounted.
void MountDirectory(std::string_view guest, const std::filesystem::path& host, bool read_only);
void UnmountDirectory(std::string_view guest);

// Translates a path used by the game ("/app0/pak0.psarc", relative paths start at /app0).
// When the exact name does not exist, a case-insensitive match is tried, in case the game spells
// a name with different capitalization than the file on disk.
HostPath ToHostPath(std::string_view guest_path);

} // namespace hle::kernel
