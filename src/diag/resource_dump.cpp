#include "common/platform.h"
#include "diag/resource_dump.h"

#include <strings.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "common/log.h"
#include "common/types.h"
#include "loader/hook.h"

namespace diag {

namespace {

// BPE/Source/Core/Util/Compression.cpp: decompresses a resource into `dst`. `info` points into
// the resource load request: bits 29-31 are the compression type (3 = Oodle), bits 0-28 the
// uncompressed size. The request's path ("$/coredata/...") is a char* 0x48 bytes before it
// (request + 0xa8, info = request + 0xf0). Called from the resource factory (sub_547160).
constexpr u64 DecompressAddress = 0x5aa2a0;
constexpr s64 PathOffsetFromInfo = 0xa8 - 0xf0;

using DecompressFn = u64 PS4_SYSV_ABI (*)(const u32* info, const void* src, u64 src_size, void* dst,
                                          u64 dst_size);

DecompressFn g_original = nullptr;
std::vector<std::string> g_extensions;
std::filesystem::path g_root;
std::mutex g_mutex;
std::unordered_set<std::string> g_written;
u64 g_count = 0;

std::filesystem::path DataDir() {
    return platform::DataDir();
}

bool Wanted(std::string_view path) {
    for (const auto& extension : g_extensions) {
        if (path.size() >= extension.size() &&
            strncasecmp(path.data() + path.size() - extension.size(), extension.c_str(),
                        extension.size()) == 0) {
            return true;
        }
    }
    return false;
}

void Save(const u32* info, const void* dst, u64 size) {
    const char* const* path_slot = reinterpret_cast<const char* const*>(
        reinterpret_cast<const u8*>(info) + PathOffsetFromInfo);
    const char* path = *path_slot;
    if (!path || !Wanted(path)) {
        return;
    }
    std::string relative{path};
    if (relative.starts_with("$/")) {
        relative.erase(0, 2);
    }
    for (char& c : relative) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    std::scoped_lock lock{g_mutex};
    if (!g_written.insert(relative).second) {
        return;
    }
    const std::filesystem::path file = g_root / relative;
    std::error_code error;
    std::filesystem::create_directories(file.parent_path(), error);
    std::ofstream out{file, std::ios::binary | std::ios::trunc};
    out.write(static_cast<const char*>(dst), static_cast<std::streamsize>(size));
    if (++g_count % 1000 == 0) {
        LOG_INFO("Dump", "{} resources saved to {}", g_count, g_root.string());
    }
}

u64 PS4_SYSV_ABI Decompress(const u32* info, const void* src, u64 src_size, void* dst,
                            u64 dst_size) {
    // Returns the decompressed size (the caller uses it as the resource's size), 0 on failure.
    const u64 result = g_original(info, src, src_size, dst, dst_size);
    if (result != 0 && result <= dst_size) {
        Save(info, dst, result);
    }
    return result;
}

} // namespace

void InstallResourceDump() {
    const char* list = std::getenv("SOTC_DUMP_RESOURCES");
    if (!list || !*list) {
        return;
    }
    std::string_view rest{list};
    while (!rest.empty()) {
        const size_t comma = rest.find(',');
        std::string extension{rest.substr(0, comma)};
        if (!extension.empty()) {
            g_extensions.push_back(extension.starts_with('.') ? extension : "." + extension);
        }
        rest = comma == std::string_view::npos ? std::string_view{} : rest.substr(comma + 1);
    }
    g_root = DataDir() / "resources";
    g_original = reinterpret_cast<DecompressFn>(
        loader::HookFunction(DecompressAddress, reinterpret_cast<const void*>(&Decompress)));
    if (!g_original) {
        LOG_ERROR("Dump", "SOTC_DUMP_RESOURCES: could not hook the decompressor");
        return;
    }
    LOG_INFO("Dump", "saving decompressed {} resources to {}", list, g_root.string());
}

} // namespace diag
