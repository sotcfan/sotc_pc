#include "common/platform.h"
#include "diag/native_validate.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/log.h"
#include "gpu/gpu.h"
#include "hle/kernel/gpu_memory.h"
#include "render/native_compute.h"

namespace diag {

namespace {

struct Settings {
    bool enabled = false;
    bool all = false;
    std::unordered_set<u64> hashes;
    u32 per_shader = 3;
    bool dump = false; // SOTC_VALIDATE_NATIVE_DUMP=1: SPIR-V and bindings of each check
};

const Settings& GetSettings() {
    static const Settings settings = [] {
        Settings s;
        const char* value = std::getenv("SOTC_VALIDATE_NATIVE");
        if (!value || !*value) {
            return s;
        }
        s.enabled = true;
        const std::string list = value;
        if (list == "all") {
            s.all = true;
        } else {
            for (size_t at = 0; at < list.size();) {
                const size_t comma = std::min(list.find(',', at), list.size());
                s.hashes.insert(std::strtoull(list.substr(at, comma - at).c_str(), nullptr, 16));
                at = comma + 1;
            }
        }
        if (const char* dump = std::getenv("SOTC_VALIDATE_NATIVE_DUMP")) {
            s.dump = *dump == '1';
        }
        if (const char* count = std::getenv("SOTC_VALIDATE_NATIVE_COUNT")) {
            s.per_shader = static_cast<u32>(std::max(1L, std::strtol(count, nullptr, 10)));
        }
        return s;
    }();
    return settings;
}

// Game memory through the mirror that write tracking never protects (gpu_memory.h).
bool ReadMirror(u64 address, void* out, u64 size) {
    if (hle::kernel::ClampToMapped(address, size) < size || !hle::kernel::GuestMirrorOffset()) {
        return false;
    }
    std::memcpy(out, reinterpret_cast<const void*>(address + hle::kernel::GuestMirrorOffset()),
                size);
    return true;
}

// The shader's hash from the OrbShdr footer after its code, by code address.
u64 ShaderHashAt(u64 code) {
    static std::unordered_map<u64, u64> cache;
    if (const auto it = cache.find(code); it != cache.end()) {
        return it->second;
    }
    u64 hash = 0;
    std::vector<u8> bytes(256 * 1024);
    const u64 size = hle::kernel::ClampToMapped(code, bytes.size());
    if (size >= 28 && ReadMirror(code, bytes.data(), size)) {
        for (u64 at = 0; at + 28 <= size; at += 4) {
            if (std::memcmp(bytes.data() + at, "OrbShdr", 7) == 0) {
                std::memcpy(&hash, bytes.data() + at + 16, 8);
                break;
            }
        }
    }
    cache[code] = hash;
    return hash;
}

// Shader file names from the shader survey (tools/shader_survey), for the report.
const std::unordered_map<u64, std::string>& ShaderNames() {
    static const auto names = [] {
        std::unordered_map<u64, std::string> map;
        std::ifstream in{platform::DataDir() / "shader_survey/survey.txt"};
        for (std::string line; std::getline(in, line);) {
            // stage \t hash \t status \t details \t path
            std::vector<std::string> fields;
            for (size_t at = 0; at <= line.size();) {
                const size_t tab = std::min(line.find('\t', at), line.size());
                fields.push_back(line.substr(at, tab - at));
                at = tab + 1;
            }
            if (fields.size() < 5 || fields[1].size() != 16) {
                continue;
            }
            std::string path = fields[4];
            if (const size_t cut = path.find("/shaders/_ps4/"); cut != std::string::npos) {
                path = path.substr(cut + 14);
            }
            map.try_emplace(std::strtoull(fields[1].c_str(), nullptr, 16), path);
        }
        return map;
    }();
    return names;
}

// ~/.local/share/sotcport/native/<hash>_<n>.spv and .txt: the SPIR-V that ran and what it
// was given (disassemble with spirv-dis).
void Dump(const render::ComputeDispatch& dispatch, const render::NativeComputeResult& result,
          u32 attempt) {
    const auto dir = platform::DataDir() / "native";
    std::filesystem::create_directories(dir);
    const auto base = dir / std::format("{:016x}_{}", result.hash, attempt);
    std::ofstream spv{base.string() + ".spv", std::ios::binary};
    spv.write(reinterpret_cast<const char*>(result.program.spirv.data()),
              static_cast<std::streamsize>(result.program.spirv.size() * 4));
    std::ofstream txt{base.string() + ".txt"};
    txt << std::format("code {:#x} rsrc1 {:08x} rsrc2 {:08x} threads {} {} {} groups {} {} {}\n",
                       dispatch.program.code_address, dispatch.program.rsrc1,
                       dispatch.program.rsrc2, dispatch.program.num_threads[0],
                       dispatch.program.num_threads[1], dispatch.program.num_threads[2],
                       dispatch.groups[0], dispatch.groups[1], dispatch.groups[2]);
    txt << "user data";
    for (u32 value : dispatch.program.user_data) {
        txt << std::format(" {:08x}", value);
    }
    txt << "\nflatbuf";
    for (u32 value : result.program.flatbuf) {
        txt << std::format(" {:08x}", value);
    }
    txt << "\n";
    constexpr const char* kinds[] = {"guest", "flatbuf", "lds", "unsupported"};
    for (size_t i = 0; i < result.program.buffers.size(); i++) {
        const auto& b = result.program.buffers[i];
        txt << std::format("binding {}: {} {:#x} size {:#x}{}{}\n", i,
                           kinds[static_cast<int>(b.kind)], b.address, b.size,
                           b.written ? " written" : "", b.formatted ? " formatted" : "");
    }
    for (const auto& range : result.written) {
        txt << std::format("written {:#x}: {} bytes, {} changed by the native dispatch\n",
                           range.address, range.bytes.size(), range.changed);
    }
}

struct ShaderStats {
    u64 changed = 0; // bytes the native dispatches changed (0: the check proves little)
    u32 attempts = 0;
    u32 matched = 0;
    u32 mismatched = 0;
    std::string skipped; // the last reason
    std::string error;
    std::string first_mismatch;
    bool clamped = false;
};

std::mutex g_mutex;
std::map<u64, ShaderStats> g_stats;
u64 g_bytes_compared = 0;
u64 g_bytes_changed = 0; // of those, bytes the dispatches changed

struct Pending {
    u64 hash = 0;
    std::vector<render::WrittenRange> written;
};
std::optional<Pending> g_pending;

// Value of a modelled register in the snapshot (nullopt if the model does not know it).
// A pointer to the pre-dispatch's copy of the LCUE resource table is known to the model only as
// "the table + offset" (its content is proven to be the model's table, CheckTable in the
// recorder); its address, chosen per dispatch, is taken from the GPU layer's register.
std::optional<u32> Value(const render::RegisterSnapshot& snapshot, u32 index) {
    static const auto slots = [] {
        std::unordered_map<u32, size_t> map;
        for (size_t i = 0; i < render::ModelledCount; i++) {
            map[render::ModelledRegisters[i].index] = i;
        }
        return map;
    }();
    const auto it = slots.find(index);
    if (it == slots.end() || !snapshot.known.test(it->second)) {
        return std::nullopt;
    }
    if (snapshot.table_pointer.test(it->second)) {
        return gpu::ReadComputeRegister(index);
    }
    return snapshot.values[it->second];
}

} // namespace

bool NativeValidationEnabled() {
    return GetSettings().enabled;
}

void ValidateDispatchBefore(const render::RegisterSnapshot& snapshot, const u32* packet,
                            bool indirect, bool compute_queue) {
    const Settings& settings = GetSettings();
    constexpr u32 Sh = render::ShBase;
    const auto pgm = Value(snapshot, Sh + 0x20c);
    if (!pgm) {
        return;
    }
    const u64 code = u64{*pgm} << 8;
    const u64 hash = ShaderHashAt(code);
    if (!settings.all && !settings.hashes.contains(hash)) {
        return;
    }
    std::scoped_lock lock{g_mutex};
    ShaderStats& stats = g_stats[hash];
    if (stats.attempts >= settings.per_shader) {
        return;
    }
    stats.attempts++;

    render::ComputeDispatch dispatch;
    dispatch.program.code_address = code;
    const auto rsrc1 = Value(snapshot, Sh + 0x212), rsrc2 = Value(snapshot, Sh + 0x213);
    const auto x = Value(snapshot, Sh + 0x207), y = Value(snapshot, Sh + 0x208),
               z = Value(snapshot, Sh + 0x209);
    if (!rsrc1 || !rsrc2 || !x || !y || !z) {
        stats.skipped = "program registers unknown to the model";
        return;
    }
    dispatch.program.rsrc1 = *rsrc1;
    dispatch.program.rsrc2 = *rsrc2;
    dispatch.program.num_threads = {*x, *y, *z};
    // The user data the shader is given must all be known (the state proof says they are).
    const u32 given = (*rsrc2 >> 1) & 0x1f;
    for (u32 i = 0; i < 16; i++) {
        const auto value = Value(snapshot, render::UserDataBase[render::StageCompute] + i);
        if (!value && i < given) {
            stats.skipped = "user data unknown to the model";
            return;
        }
        dispatch.program.user_data[i] = value.value_or(0);
    }
    if (!indirect) {
        // DISPATCH_DIRECT: header, dim_x, dim_y, dim_z, initiator.
        dispatch.groups = {packet[1], packet[2], packet[3]};
    } else {
        // The workgroup counts are in memory, often written by an earlier dispatch.
        u64 args = 0;
        if (compute_queue) {
            args = packet[1] | u64{packet[2] & 0xffff} << 32; // header, address lo, address hi
        } else {
            const auto lo = Value(snapshot, render::IndirectArgsBase);
            const auto hi = Value(snapshot, render::IndirectArgsBase + 1);
            if (!lo || !hi) {
                stats.skipped = "indirect arguments address unknown to the model";
                return;
            }
            args = (u64{*lo} | u64{*hi} << 32) + packet[1]; // header, data offset
        }
        gpu::SyncGuestRange(args, 12);
        if (!ReadMirror(args, dispatch.groups.data(), 12)) {
            stats.error = std::format("could not read the indirect arguments at {:#x}", args);
            return;
        }
        if (!dispatch.groups[0] || !dispatch.groups[1] || !dispatch.groups[2]) {
            stats.attempts--; // an empty dispatch checks nothing: try another one
            return;
        }
    }

    render::NativeComputeResult result;
    const auto read = [](u64 address, void* out, u64 size) {
        gpu::SyncGuestRange(address, size); // the GPU layer's results so far
        return ReadMirror(address, out, size);
    };
    render::RunNativeCompute(dispatch, read, result);
    if (settings.dump && !result.program.spirv.empty()) {
        Dump(dispatch, result, stats.attempts);
    }
    stats.clamped |= result.clamped;
    if (!result.skipped.empty()) {
        stats.skipped = result.skipped;
        return;
    }
    if (!result.error.empty()) {
        stats.error = result.error;
        return;
    }
    g_pending = Pending{hash, std::move(result.written)};
}

void ValidateAfterPacket() {
    std::scoped_lock lock{g_mutex};
    if (!g_pending) {
        return;
    }
    Pending pending = std::move(*g_pending);
    g_pending.reset();
    ShaderStats& stats = g_stats[pending.hash];
    bool matched = true;
    for (const auto& range : pending.written) {
        std::vector<u8> layer(range.bytes.size());
        gpu::SyncGuestRange(range.address, layer.size());
        if (!ReadMirror(range.address, layer.data(), layer.size())) {
            stats.error = std::format("could not read the GPU layer's result at {:#x}",
                                      range.address);
            return;
        }
        g_bytes_compared += layer.size();
        g_bytes_changed += range.changed;
        stats.changed += range.changed;
        const auto [native_it, layer_it] =
            std::ranges::mismatch(range.bytes, layer);
        if (native_it != range.bytes.end()) {
            matched = false;
            if (stats.first_mismatch.empty()) {
                const size_t offset = (native_it - range.bytes.begin()) & ~size_t{3};
                u32 native = 0, theirs = 0;
                std::memcpy(&native, range.bytes.data() + offset,
                            std::min<size_t>(4, range.bytes.size() - offset));
                std::memcpy(&theirs, layer.data() + offset,
                            std::min<size_t>(4, layer.size() - offset));
                size_t differing = 0;
                for (size_t i = 0; i < layer.size(); i++) {
                    differing += range.bytes[i] != layer[i];
                }
                stats.first_mismatch = std::format(
                    "buffer {:#x} ({} bytes): first difference at +{:#x}, native {:08x} / GPU "
                    "layer {:08x}; {} bytes differ",
                    range.address, layer.size(), offset, native, theirs, differing);
            }
        }
    }
    (matched ? stats.matched : stats.mismatched)++;
}

void WriteNativeValidationReport(std::FILE* file) {
    std::scoped_lock lock{g_mutex};
    u32 matched = 0, mismatched = 0, skipped = 0, failed = 0;
    for (const auto& [hash, stats] : g_stats) {
        if (stats.matched && !stats.mismatched) matched++;
        if (stats.mismatched) mismatched++;
        if (!stats.matched && !stats.mismatched && !stats.skipped.empty()) skipped++;
        if (!stats.error.empty()) failed++;
    }
    std::fprintf(file,
                 "\n== native renderer vs the GPU layer (SOTC_VALIDATE_NATIVE), compute: %zu "
                 "shaders tried; %u always matched, %u differed, %u skipped, %u with errors; "
                 "%llu bytes compared, %llu of them changed by the dispatches\n",
                 g_stats.size(), matched, mismatched, skipped, failed,
                 static_cast<unsigned long long>(g_bytes_compared),
                 static_cast<unsigned long long>(g_bytes_changed));
    const auto& names = ShaderNames();
    for (const auto& [hash, stats] : g_stats) {
        const auto name = names.find(hash);
        std::string line = std::format("  {:016x} {:<60} matched {} differed {} (changed {} bytes)",
                                       hash, name != names.end() ? name->second : "?",
                                       stats.matched, stats.mismatched, stats.changed);
        if (stats.clamped) line += " (buffers cut at 64 MiB)";
        if (!stats.skipped.empty()) line += "  skipped: " + stats.skipped;
        if (!stats.error.empty()) line += "  error: " + stats.error;
        if (!stats.first_mismatch.empty()) line += "\n      " + stats.first_mismatch;
        std::fprintf(file, "%s\n", line.c_str());
    }
}

} // namespace diag
