#include "common/platform.h"
#include "diag/gnm_trace.h"

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "common/log.h"
#include "common/types.h"
#include "diag/call_hooks.h"
#include "hle/kernel/gpu_memory.h"

namespace diag {

namespace {

// How much of the object in the first argument is compared before and after the call to find
// the command buffer write pointers that moved.
constexpr size_t SnapshotSize = 0x200;
// Packets recorded per call (the rest is only counted).
constexpr size_t MaxPackets = 48;
constexpr size_t MaxSampleDwords = 96;

struct Shape {
    u64 calls = 0;
    std::array<u64, 6> sample_args{};
    std::vector<u32> sample_dwords;
};

// Calls whose whole first argument is compared before and after: the first ones of each
// function, then one in SampleEvery.
constexpr u64 SampleFirst = 64;
constexpr u64 SampleEvery = 4096;
constexpr size_t FullObjectSize = 0x10000;

struct Function {
    u64 address = 0;
    u64 calls = 0;
    u64 calls_without_packets = 0;
    std::map<std::string, Shape> shapes;
    u64 sampled = 0;
    std::map<std::pair<u32, u32>, u64> changed; // (offset, length) in the first argument
    std::vector<std::array<u64, 6>> arg_samples;
};

// A piece of memory whose pointers are compared before and after the call.
struct Watched {
    char label[16]; // "@20" (field of the first argument) or "@20>" (object it points to)
    u64 address;
    u32 size;
    std::array<u8, SnapshotSize> bytes;
};

// A command buffer write pointer seen moving over packets: the address of the pointer.
constexpr size_t MaxCursors = 32;

struct Frame {
    u32 id;
    std::array<u64, 6> args;
    std::vector<Watched> watched;
    std::vector<std::pair<u64, u64>> cursors; // (where the pointer is, its value)
    std::vector<u8> full_before;              // the whole first argument, when sampled
};

std::vector<Function> g_functions;
// Calls started, per function (counted before the call, without the lock).
std::unique_ptr<std::atomic<u64>[]> g_entries;
std::mutex g_mutex;
// The calls running on this thread, innermost last (ObservedCall::user[0] is the index).
thread_local std::vector<Frame> t_frames;
// The command buffer write pointers this thread has been seen to use.
thread_local std::vector<u64> t_cursors;
// Which thread wrote through each cursor last: a context moves between the job system's
// threads, and only its current thread should look at its cursors.
std::mutex g_owner_mutex;
std::map<u64, std::thread::id> g_cursor_owner;

void ClaimCursor(u64 location) {
    std::scoped_lock lock{g_owner_mutex};
    g_cursor_owner[location] = std::this_thread::get_id();
}

bool OwnsCursor(u64 location) {
    std::scoped_lock lock{g_owner_mutex};
    const auto it = g_cursor_owner.find(location);
    return it != g_cursor_owner.end() && it->second == std::this_thread::get_id();
}

// Copies game memory through the view the GPU layer never protects.
bool ReadGame(u64 address, void* out, size_t size) {
    if (hle::kernel::ClampToMapped(address, size) < size) {
        return false;
    }
    auto* dest = static_cast<u8*>(out);
    while (size) {
        const u64 chunk = std::min<u64>(0x4000 - address % 0x4000, size);
        const u8* backing = hle::kernel::BackingAddress(address);
        if (!backing) {
            return false;
        }
        std::memcpy(dest, backing, chunk);
        address += chunk;
        dest += chunk;
        size -= chunk;
    }
    return true;
}

// "SET_CONTEXT_REG 0x10b+2" etc. for the PM4 packets in `dwords`; the register offsets are
// named later, by the analysis script.
std::string DescribePackets(const std::vector<u32>& dwords) {
    std::string text;
    size_t i = 0, packets = 0;
    while (i < dwords.size() && packets < MaxPackets) {
        const u32 header = dwords[i];
        const u32 type = header >> 30;
        if (type == 2) { // filler
            i++;
            continue;
        }
        if (type != 3) {
            text += "?";
            break;
        }
        const u32 count = ((header >> 16) & 0x3FFF) + 1;
        const u32 opcode = (header >> 8) & 0xFF;
        char item[64];
        if ((opcode == 0x68 || opcode == 0x69 || opcode == 0x76 || opcode == 0x79) &&
            i + 1 < dwords.size()) {
            std::snprintf(item, sizeof(item), "%02x:%x+%u ", opcode, dwords[i + 1] & 0xFFFF,
                          count - 1);
        } else {
            std::snprintf(item, sizeof(item), "%02x/%u ", opcode, count);
        }
        text += item;
        i += 1 + count;
        packets++;
    }
    if (i < dwords.size()) {
        text += "…";
    }
    return text;
}

// Whether `dwords` is a run of whole PM4 type 3 packets (or filler).
bool LooksLikePackets(const std::vector<u32>& dwords) {
    size_t i = 0;
    while (i < dwords.size()) {
        const u32 type = dwords[i] >> 30;
        if (type == 2) {
            i++;
            continue;
        }
        if (type != 3) {
            return false;
        }
        i += 2 + ((dwords[i] >> 16) & 0x3FFF);
    }
    return i == dwords.size();
}

// How much of each object the first argument points to is watched (a command buffer keeps its
// write pointer near the start).
constexpr size_t PointeeSize = 0x40;
constexpr size_t MaxPointees = 16;

void OnEnter(ObservedCall& call) {
    call.user[0] = t_frames.size();
    Frame& frame = t_frames.emplace_back();
    frame.id = call.id;
    frame.args = call.args;
    const u64 rdi = call.args[0];
    const u64 entry = g_entries[call.id].fetch_add(1, std::memory_order_relaxed);
    if (entry < SampleFirst || entry % SampleEvery == 0) {
        const u64 size = hle::kernel::ClampToMapped(rdi, FullObjectSize) & ~u64{7};
        if (size >= 8 && rdi > 0x100000000ULL) {
            frame.full_before.resize(size);
            if (!ReadGame(rdi, frame.full_before.data(), size)) {
                frame.full_before.clear();
            }
        }
    }
    for (const u64 cursor : t_cursors) {
        u64 value;
        if (OwnsCursor(cursor) && ReadGame(cursor, &value, 8)) {
            frame.cursors.emplace_back(cursor, value);
        }
    }
    Watched object{};
    if (!ReadGame(rdi, object.bytes.data(), SnapshotSize)) {
        return;
    }
    std::snprintf(object.label, sizeof(object.label), "@");
    object.address = rdi;
    object.size = SnapshotSize;
    frame.watched.push_back(object);
    // The objects it points to (a context keeps pointers to its command buffers).
    for (size_t offset = 0; offset + 8 <= SnapshotSize && frame.watched.size() <= MaxPointees;
         offset += 8) {
        u64 pointer;
        std::memcpy(&pointer, object.bytes.data() + offset, 8);
        if (pointer < 0x100000000ULL || pointer % 8 != 0 || pointer == rdi ||
            std::ranges::any_of(frame.watched,
                                [&](const Watched& w) { return w.address == pointer; })) {
            continue;
        }
        Watched pointee{};
        if (!ReadGame(pointer, pointee.bytes.data(), PointeeSize)) {
            continue;
        }
        std::snprintf(pointee.label, sizeof(pointee.label), "@%zx>", offset);
        pointee.address = pointer;
        pointee.size = PointeeSize;
        frame.watched.push_back(pointee);
    }
}

void OnExit(const ObservedCall& call) {
    // Anything above this call's frame was left by a call that never returned the normal way.
    Frame frame = std::move(t_frames[call.user[0]]);
    t_frames.resize(call.user[0]);

    // Every pointer in the watched memory that moved by whole dwords: forward over what the
    // call wrote (command packets or another buffer), or backward over what it allocated from
    // the end of a buffer.
    std::string shape_text;
    std::vector<u32> sample;
    for (const Watched& watched : frame.watched) {
        std::array<u8, SnapshotSize> after;
        if (!ReadGame(watched.address, after.data(), watched.size)) {
            continue;
        }
        for (size_t offset = 0; offset + 8 <= watched.size; offset += 8) {
            u64 old_value, new_value;
            std::memcpy(&old_value, watched.bytes.data() + offset, 8);
            std::memcpy(&new_value, after.data() + offset, 8);
            if (old_value == new_value || old_value < 0x100000000ULL || old_value % 4 != 0 ||
                new_value % 4 != 0) {
                continue;
            }
            const bool forward = new_value > old_value;
            const u64 distance = forward ? new_value - old_value : old_value - new_value;
            if (distance > 0x40000) {
                continue;
            }
            const u64 start = forward ? old_value : new_value;
            std::vector<u32> written(distance / 4);
            if (!ReadGame(start, written.data(), written.size() * 4)) {
                continue;
            }
            const u64 location = watched.address + offset;
            if (std::ranges::any_of(frame.cursors,
                                    [&](const auto& cursor) { return cursor.first == location; })) {
                continue; // reported below, as a known command buffer
            }
            char field[48];
            std::snprintf(field, sizeof(field), "%s%zx%s ", watched.label, offset,
                          forward ? "" : "<");
            shape_text += field;
            if (forward && LooksLikePackets(written)) {
                shape_text += DescribePackets(written);
                ClaimCursor(location);
                if (std::ranges::find(t_cursors, location) == t_cursors.end() &&
                    t_cursors.size() < MaxCursors) {
                    t_cursors.push_back(location);
                }
            } else {
                char data[32];
                std::snprintf(data, sizeof(data), "data %zu B ", written.size() * 4);
                shape_text += data;
            }
            if (sample.empty()) {
                sample.assign(written.begin(),
                              written.begin() + std::min(written.size(), MaxSampleDwords));
            }
        }
    }

    // The command buffers this thread is known to use, however the call reached them.
    for (const auto& [location, old_value] : frame.cursors) {
        u64 new_value;
        if (!ReadGame(location, &new_value, 8) || new_value <= old_value ||
            new_value - old_value > 0x40000 || (new_value - old_value) % 4 != 0) {
            continue;
        }
        std::vector<u32> written((new_value - old_value) / 4);
        if (!ReadGame(old_value, written.data(), written.size() * 4) ||
            !LooksLikePackets(written)) {
            continue;
        }
        char field[48];
        std::snprintf(field, sizeof(field), "cb%zx ",
                      static_cast<size_t>(std::ranges::find(t_cursors, location) -
                                          t_cursors.begin()));
        shape_text += field;
        shape_text += DescribePackets(written);
        if (sample.empty()) {
            sample.assign(written.begin(),
                          written.begin() + std::min(written.size(), MaxSampleDwords));
        }
    }

    // What the call changed in its first argument (sampled).
    std::vector<std::pair<u32, u32>> changed;
    if (!frame.full_before.empty()) {
        std::vector<u8> after(frame.full_before.size());
        if (ReadGame(frame.args[0], after.data(), after.size())) {
            size_t i = 0;
            while (i < after.size()) {
                if (after[i] == frame.full_before[i]) {
                    i++;
                    continue;
                }
                size_t end = i + 1;
                // Join differences closer than 8 bytes into one range.
                for (size_t gap = 0; end < after.size() && gap < 8; end++) {
                    gap = after[end] == frame.full_before[end] ? gap + 1 : 0;
                }
                while (end > i + 1 && after[end - 1] == frame.full_before[end - 1]) {
                    end--;
                }
                changed.emplace_back(static_cast<u32>(i), static_cast<u32>(end - i));
                i = end;
            }
        }
    }

    std::scoped_lock lock{g_mutex};
    Function& function = g_functions[frame.id];
    function.calls++;
    if (!frame.full_before.empty()) {
        function.sampled++;
        for (const auto& range : changed) {
            if (function.changed.size() < 256 || function.changed.contains(range)) {
                function.changed[range]++;
            }
        }
    }
    if (function.arg_samples.size() < 4 &&
        std::ranges::find(function.arg_samples, frame.args) == function.arg_samples.end()) {
        function.arg_samples.push_back(frame.args);
    }
    if (shape_text.empty()) {
        function.calls_without_packets++;
    } else {
        Shape& shape = function.shapes[shape_text];
        if (shape.calls++ == 0) {
            shape.sample_args = frame.args;
            shape.sample_dwords = std::move(sample);
        }
    }
}

std::filesystem::path ReportPath() {
    return platform::DataDir() / "gnm_trace.txt";
}

void WriteReport() {
    std::scoped_lock lock{g_mutex};
    const auto path = ReportPath();
    std::FILE* file = std::fopen((path.string() + ".tmp").c_str(), "w");
    if (!file) {
        return;
    }
    std::vector<const Function*> order;
    for (const Function& function : g_functions) {
        order.push_back(&function);
    }
    std::ranges::sort(order, [](auto* a, auto* b) { return a->calls > b->calls; });
    for (const Function* function : order) {
        std::fprintf(file, "== %#" PRIx64 " calls %" PRIu64 " (no writes %" PRIu64 ")\n",
                     function->address, function->calls, function->calls_without_packets);
        std::vector<std::pair<u64, const std::pair<const std::string, Shape>*>> shapes;
        for (const auto& entry : function->shapes) {
            shapes.emplace_back(entry.second.calls, &entry);
        }
        std::ranges::sort(shapes, std::greater{});
        for (size_t i = 0; i < shapes.size() && i < 12; i++) {
            const auto& [text, shape] = *shapes[i].second;
            std::fprintf(file, "  %" PRIu64 "x  %s\n    args", shape.calls, text.c_str());
            for (u64 arg : shape.sample_args) {
                std::fprintf(file, " %#" PRIx64, arg);
            }
            std::fprintf(file, "\n    dwords");
            for (u32 dword : shape.sample_dwords) {
                std::fprintf(file, " %08x", dword);
            }
            std::fprintf(file, "\n");
        }
        if (function->sampled) {
            std::vector<std::pair<u64, std::pair<u32, u32>>> ranges;
            for (const auto& [range, count] : function->changed) {
                ranges.emplace_back(count, range);
            }
            std::ranges::sort(ranges, std::greater{});
            std::fprintf(file, "  changes (in %" PRIu64 " samples):", function->sampled);
            for (size_t i = 0; i < ranges.size() && i < 16; i++) {
                std::fprintf(file, " +%x[%u]x%" PRIu64, ranges[i].second.first,
                             ranges[i].second.second, ranges[i].first);
            }
            std::fprintf(file, "\n");
        }
        for (const auto& args : function->arg_samples) {
            std::fprintf(file, "  arg");
            for (u64 arg : args) {
                std::fprintf(file, " %#" PRIx64, arg);
            }
            std::fprintf(file, "\n");
        }
        if (shapes.size() > 12) {
            std::fprintf(file, "  ... %zu shapes\n", shapes.size());
        }
    }
    std::fclose(file);
    std::filesystem::rename(path.string() + ".tmp", path);
}

const CallObserver g_observer{OnEnter, OnExit};

} // namespace

void InstallGnmTracer() {
    const char* list = std::getenv("SOTC_TRACE_GNM");
    if (!list) {
        return;
    }
    std::ifstream file{list};
    if (!file) {
        LOG_ERROR("Trace", "SOTC_TRACE_GNM: could not open {}", list);
        return;
    }
    std::vector<u64> addresses;
    for (std::string line; std::getline(file, line);) {
        if (line.size() < 3 || line[0] != '0') {
            continue;
        }
        addresses.push_back(std::strtoull(line.c_str(), nullptr, 16));
    }
    g_functions.resize(addresses.size());
    for (size_t i = 0; i < addresses.size(); i++) {
        g_functions[i].address = addresses[i];
    }
    g_entries = std::make_unique<std::atomic<u64>[]>(addresses.size());
    const std::vector<bool> hooked = ObserveCalls(addresses, &g_observer);
    const size_t installed = std::ranges::count(hooked, true);
    LOG_INFO("Trace", "Gnm: {} of {} functions hooked; report in {}", installed,
             addresses.size(), ReportPath().string());
    std::thread{[] {
        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds{20});
            WriteReport();
        }
    }}.detach();
}

} // namespace diag
