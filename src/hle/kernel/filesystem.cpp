#include "hle/kernel/filesystem.h"

#include <algorithm>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <system_error>
#include <vector>

#include "common/log.h"

namespace hle::kernel {

namespace {

struct Mount {
    std::string guest;
    std::filesystem::path host;
    bool read_only;
};

std::shared_mutex g_mounts_mutex;
std::vector<Mount> g_mounts;

// "/app0//a/./b/../c" -> "/app0/a/c". ".." never climbs above the root.
std::string Normalize(std::string_view path) {
    std::vector<std::string_view> parts;
    size_t start = 0;
    while (start <= path.size()) {
        const size_t end = std::min(path.find('/', start), path.size());
        const std::string_view part = path.substr(start, end - start);
        if (part == "..") {
            if (!parts.empty()) {
                parts.pop_back();
            }
        } else if (!part.empty() && part != ".") {
            parts.push_back(part);
        }
        start = end + 1;
    }
    std::string result;
    for (const auto part : parts) {
        result += '/';
        result += part;
    }
    return result.empty() ? "/" : result;
}

bool EqualsIgnoringCase(std::string_view a, std::string_view b) {
    return std::ranges::equal(a, b, [](char x, char y) {
        return std::tolower(static_cast<unsigned char>(x)) ==
               std::tolower(static_cast<unsigned char>(y));
    });
}

// Resolves `relative` under `root` one component at a time, accepting a name that differs only
// in case when the exact one does not exist. Components that match nothing are kept as given.
std::filesystem::path MatchCase(const std::filesystem::path& root,
                                const std::filesystem::path& relative) {
    std::filesystem::path current = root;
    for (const auto& component : relative) {
        std::filesystem::path exact = current / component;
        std::error_code error;
        if (!std::filesystem::exists(exact, error)) {
            for (const auto& entry : std::filesystem::directory_iterator(current, error)) {
                if (EqualsIgnoringCase(entry.path().filename().string(), component.string())) {
                    exact = entry.path();
                    break;
                }
            }
        }
        current = exact;
    }
    return current;
}

} // namespace

void InitializeFilesystem(const std::filesystem::path& game_dir,
                          const std::filesystem::path& data_dir) {
    std::scoped_lock lock{g_mounts_mutex};
    g_mounts = {
        {"/app0", game_dir, true},
        {"/hostapp", game_dir, true},
        {"/data", data_dir / "data", false},
        {"/temp0", data_dir / "temp0", false},
        {"/temp", data_dir / "temp0", false},
        {"/download0", data_dir / "download0", false},
    };
    for (const auto& mount : g_mounts) {
        if (!mount.read_only) {
            std::filesystem::create_directories(mount.host);
        }
    }
    LOG_INFO("Files", "/app0 -> {} (read-only); writes go to {}", game_dir.string(),
             data_dir.string());
}

void MountDirectory(std::string_view guest, const std::filesystem::path& host, bool read_only) {
    std::scoped_lock lock{g_mounts_mutex};
    std::erase_if(g_mounts, [&](const Mount& mount) { return mount.guest == guest; });
    g_mounts.push_back({std::string(guest), host, read_only});
}

void UnmountDirectory(std::string_view guest) {
    std::scoped_lock lock{g_mounts_mutex};
    std::erase_if(g_mounts, [&](const Mount& mount) { return mount.guest == guest; });
}

HostPath ToHostPath(std::string_view guest_path) {
    std::shared_lock lock{g_mounts_mutex};
    const std::string path =
        Normalize(guest_path.starts_with('/') ? std::string(guest_path)
                                              : "/app0/" + std::string(guest_path));
    for (const auto& mount : g_mounts) {
        if (path != mount.guest && !path.starts_with(mount.guest + "/")) {
            continue;
        }
        const std::filesystem::path relative =
            path.size() > mount.guest.size() ? path.substr(mount.guest.size() + 1) : "";
        std::filesystem::path host = mount.host / relative;
        std::error_code error;
        if (!relative.empty() && !std::filesystem::exists(host, error)) {
            host = MatchCase(mount.host, relative);
        }
        return {host, mount.read_only, true};
    }
    return {};
}

} // namespace hle::kernel
