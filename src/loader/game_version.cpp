#include "loader/game_version.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <fstream>
#include <string>
#include <vector>

#include "common/log.h"
#include "common/sha256.h"

namespace loader {
namespace {

struct KnownVersion {
    const char* description;
    const char* sha256; // of eboot.bin, as `sha256sum` prints it
};

// The versions the port is tested with.
constexpr std::array KnownVersions = {
    KnownVersion{"Shadow of the Colossus, CUSA08809 (Europe), version 01.00",
                 "21982727c6f97fd415b3af5e99c5e071619c4dcc2248787a701a31a45ab0a4e0"},
};

std::string HashFile(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return {};
    }
    Sha256 hash;
    std::vector<char> chunk(1 << 20);
    while (file) {
        file.read(chunk.data(), std::streamsize(chunk.size()));
        hash.Update(chunk.data(), std::size_t(file.gcount()));
    }
    std::string text;
    for (const u8 byte : hash.Final()) {
        text += std::format("{:02x}", byte);
    }
    return text;
}

} // namespace

bool CheckGameVersion(const std::filesystem::path& eboot) {
    const std::string sha256 = HashFile(eboot);
    if (sha256.empty()) {
        LOG_ERROR("Loader", "cannot read {}", eboot.string());
        return false;
    }
    for (const auto& known : KnownVersions) {
        if (sha256 == known.sha256) {
            LOG_INFO("Loader", "game: {}", known.description);
            return true;
        }
    }
    std::string expected;
    for (const auto& known : KnownVersions) {
        expected += std::format("\n  {}  ({})", known.sha256, known.description);
    }
    const bool allowed = [] {
        const char* env = std::getenv("SOTC_ALLOW_UNKNOWN_EBOOT");
        return env && *env == '1';
    }();
    std::fprintf(stderr,
                 "%s: %s is not a version sotcport was made for.\n"
                 "  found    %s\n"
                 "  expected one of:%s\n"
                 "The port hooks functions inside the executable, so another version will most "
                 "likely crash.\n%s",
                 allowed ? "warning" : "error", eboot.string().c_str(), sha256.c_str(),
                 expected.c_str(),
                 allowed ? "Continuing anyway (SOTC_ALLOW_UNKNOWN_EBOOT=1).\n"
                         : "Set SOTC_ALLOW_UNKNOWN_EBOOT=1 to try anyway.\n");
    return allowed;
}

} // namespace loader
