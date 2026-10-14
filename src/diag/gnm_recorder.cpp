// The renderer recorder (overview in gnm_recorder.h). Parts of this file, in order:
//
//   1. The frontier table: the 88 functions of Sony's Gnm/Gnmx code the game calls, with how
//      each one reaches its command buffer (from renderizador/CATALOGO_GNM.md in the analysis
//      notes, verified with the tracer).
//   2. Owner marks: a byte per dword of game memory saying which function wrote it into a
//      command buffer. Written in OnExit from the movement of the command buffer's cursor.
//   3. Reading the game's objects (shaders, descriptors, render targets) and following the state
//      it sets, to list the shader variants.
//   4. Frame recording: the calls between two flips, with decoded arguments.
//   5. The hooks themselves (OnEnter/OnExit, through diag::ObserveCalls).
//   6. The GPU side: OnPacket, called by the PM4 command processor for every packet it runs,
//      looks the packet up in the owner marks.
//   7. The reports, written every 20 seconds by a thread of their own.
//   8. The survey (SOTC_SURVEY_GNM=1): the other functions of Sony's block the game calls
//      directly, watched to tell which ones change renderer state the frontier table misses.
//
// Only calls made by the game (depth 0) are recorded or change the tracked state; calls between
// frontier functions (setupScreenViewport calling setViewportScissor, compute.submit calling
// compute.ringIndirectBuffer) only refine the owner marks.

#include "common/platform.h"
#include "common/host_file.h"
#include "diag/gnm_recorder.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <initializer_list>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/log.h"
#include "common/types.h"
#include "diag/call_hooks.h"
#include "diag/native_validate.h"
#include "gnm/descriptors.h"
#include "render/frontier.h"
#include "render/state_model.h"
#include "gpu/gpu.h"
#include "hle/kernel/gpu_memory.h"

namespace diag {

namespace {

// ---------------------------------------------------------------------------------------------
// The frontier: the functions of Sony's code the game calls to talk to the GPU
// (renderizador/CATALOGO_GNM.md, tables B, C and D). Names are the probable Gnm/Gnmx ones.

using render::frontier::Buffers;
using render::frontier::FrontierFunction;
using render::frontier::FunctionCount;
using render::frontier::Functions;
using render::frontier::Role;
namespace fn = render::frontier;

static_assert(FunctionCount < 254, "owners are stored in a byte, 0 = nobody");

// Owner given, on the GPU side, to packets outside the game's memory: the GPU layer's own
// commands (shadPS4 sends its InitSequence, the default hardware state, before the first
// submission of each frame; on the PS4 the system does that, not the game).
constexpr u8 OwnerGpuLayer = 255;

// A call that moves a cursor further than this did not write it all (a buffer switch, or a
// wrong guess about the object): nothing is marked.
constexpr u64 MaxCallBytes = 4ull << 20;

struct FunctionStats {
    std::atomic<u64> calls{0};
    std::atomic<u64> bad_buffer{0};     // the object did not look like a command buffer
    std::atomic<u64> buffer_switches{0}; // the buffer's start changed during the call
    std::atomic<u64> too_far{0};         // the cursor moved more than MaxCallBytes
    std::atomic<u64> dwords{0};          // dwords marked
};

std::array<FunctionStats, FunctionCount> g_stats;

const auto g_start = std::chrono::steady_clock::now();

double Seconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - g_start).count();
}

std::filesystem::path DataPath(const char* name) {
    return platform::DataDir() / name;
}

// ---------------------------------------------------------------------------------------------
// Owner marks: one byte per dword of game memory that a frontier function wrote into a command
// buffer (function index + 1; 0 = nobody). Chunks are allocated when first written. The game's
// threads write marks and the GPU thread reads them; a race between the two would itself be a
// bug of the game (executing commands still being written), so plain bytes are enough.

constexpr u64 ChunkShift = 24; // 16 MiB of game memory per chunk
constexpr u64 ChunkBytes = 1ull << ChunkShift;
constexpr u64 AddressLimit = 1ull << 40;

std::array<std::atomic<u8*>, (AddressLimit >> ChunkShift)> g_chunks{};

u8* ChunkOf(u64 address, bool create) {
    if (address >= AddressLimit) {
        return nullptr;
    }
    auto& slot = g_chunks[address >> ChunkShift];
    u8* chunk = slot.load(std::memory_order_acquire);
    if (!chunk && create) {
        auto* fresh = static_cast<u8*>(std::calloc(ChunkBytes / 4, 1));
        if (slot.compare_exchange_strong(chunk, fresh, std::memory_order_acq_rel)) {
            chunk = fresh;
        } else {
            std::free(fresh);
        }
    }
    return chunk;
}

// Calls `apply(marks, count)` over the marks of [begin, end), chunk by chunk.
template <typename Apply>
void ForEachMarkRun(u64 begin, u64 end, bool create, Apply&& apply) {
    begin &= ~u64{3};
    while (begin < end) {
        const u64 chunk_end = (begin | (ChunkBytes - 1)) + 1;
        const u64 run_end = std::min(end, chunk_end);
        if (u8* chunk = ChunkOf(begin, create)) {
            apply(chunk + ((begin & (ChunkBytes - 1)) >> 2), (run_end - begin + 3) >> 2);
        }
        begin = run_end;
    }
}

void Mark(u64 begin, u64 end, u8 owner) {
    ForEachMarkRun(begin, end, true, [&](u8* marks, u64 count) { std::memset(marks, owner, count); });
}

void ClearMarks(u64 begin, u64 end) {
    ForEachMarkRun(begin, end, false, [](u8* marks, u64 count) { std::memset(marks, 0, count); });
}

u8 OwnerAt(u64 address) {
    const u8* chunk = ChunkOf(address, false);
    return chunk ? chunk[(address & (ChunkBytes - 1)) >> 2] : 0;
}

// ---------------------------------------------------------------------------------------------
// Command buffer objects.

struct BufferState {
    u64 begin, end, cursor;
};

bool ReadBuffer(u64 object, BufferState& state) {
    if (object < 0x10000 || object % 8 != 0 || hle::kernel::ClampToMapped(object, 24) < 24) {
        return false;
    }
    std::memcpy(&state, reinterpret_cast<const void*>(object), sizeof(state));
    return state.begin % 4 == 0 && state.cursor % 4 == 0 && state.begin <= state.cursor &&
           state.cursor <= state.end && state.end - state.begin <= (64ull << 20);
}

u64 ReadPointer(u64 address) {
    if (address < 0x10000 || hle::kernel::ClampToMapped(address, 8) < 8) {
        return 0;
    }
    u64 value;
    std::memcpy(&value, reinterpret_cast<const void*>(address), sizeof(value));
    return value;
}

std::array<u64, 2> BufferObjects(Buffers buffers, u64 arg0) {
    switch (buffers) {
    case Buffers::None:
        return {};
    case Buffers::Self:
        return {arg0, 0};
    case Buffers::GraphicsLcue:
        return {ReadPointer(arg0 + 0xc090), ReadPointer(arg0 + 0xc098)};
    case Buffers::ComputeLcue:
        return {ReadPointer(arg0 + 0x90), 0};
    case Buffers::ComputeContext:
        return {arg0 + 0x10, 0};
    case Buffers::ComputeQueue:
        return {arg0 + 0x18, 0};
    }
    return {};
}

// What the frontier calls running on this thread inside an outer one wrote: the outer call
// marks its whole range as its own, then these are marked again with their own owner.
struct InnerMark {
    u64 begin, end;
    u8 owner;
};
thread_local std::vector<InnerMark> t_inner_marks;

// ---------------------------------------------------------------------------------------------
// Reading the game's objects: shaders, descriptors, render targets.

// Copies `size` bytes of game memory, if all of it is mapped.
bool ReadGame(u64 address, void* out, size_t size) {
    if (address < 0x10000 || hle::kernel::ClampToMapped(address, size) < size) {
        return false;
    }
    std::memcpy(out, reinterpret_cast<const void*>(address), size);
    return true;
}

// The hash of a shader (the one in the "OrbShdr" binary info at the end of its code block, also
// the key of shadPS4's shader cache and of the shader survey), from the Gnmx shader struct the
// game passes to set*Shader: dword 0 bits 0-22 = code block size, dwords 2-3 = PGM_LO/PGM_HI
// (the code's GPU address >> 8). 0 if it cannot be read.
u64 ShaderHash(u64 shader) {
    std::array<u32, 4> header;
    if (!ReadGame(shader, header.data(), sizeof(header))) {
        return 0;
    }
    const u64 code = (u64{header[3]} << 32 | header[2]) << 8;
    const u32 size = header[0] & 0x7FFFFF;
    if (size < 28) {
        return 0;
    }
    std::array<u8, 28> info;
    if (!ReadGame(code + size - info.size(), info.data(), info.size()) ||
        std::memcmp(info.data(), "OrbShdr", 7) != 0) {
        return 0;
    }
    u64 hash;
    std::memcpy(&hash, info.data() + 16, sizeof(hash));
    return hash;
}

std::optional<gnm::TextureInfo> ReadTexture(u64 address) {
    std::array<u32, 8> dwords;
    if (!ReadGame(address, dwords.data(), sizeof(dwords))) {
        return std::nullopt;
    }
    return gnm::DecodeTexture(dwords);
}

std::optional<gnm::BufferInfo> ReadBufferDescriptor(u64 address) {
    std::array<u32, 4> dwords;
    if (!ReadGame(address, dwords.data(), sizeof(dwords))) {
        return std::nullopt;
    }
    return gnm::DecodeBuffer(dwords);
}

std::optional<gnm::RenderTargetInfo> ReadRenderTarget(u64 address) {
    std::array<u32, 6> registers;
    if (!ReadGame(address, registers.data(), sizeof(registers))) {
        return std::nullopt;
    }
    return gnm::DecodeRenderTarget(registers);
}

// A C string of the game (marker names live in the executable's read-only data or in the game's
// heap), up to `limit` characters; printable ASCII only.
std::string ReadGameString(u64 address, size_t limit = 120) {
    std::string text;
    for (size_t i = 0; i < limit; i++) {
        char c;
        if (!ReadGame(address + i, &c, 1) || c == 0) {
            break;
        }
        text += (c >= 0x20 && c < 0x7F) ? c : '?';
    }
    return text;
}

// What a shader specialization depends on, for a texture: type and formats, in one number
// (0 = nothing bound).
u32 TextureKey(const gnm::TextureInfo& texture) {
    if (!texture.Valid()) {
        return 0;
    }
    return static_cast<u32>(texture.type) << 16 | texture.data_format << 8 | texture.num_format;
}

// The same for a buffer in a resource slot (slots take textures or buffers): formatted buffers
// are read with format conversion, raw ones are not.
constexpr u32 BufferKeyBit = 1u << 31;

u32 BufferKey(const gnm::BufferInfo& buffer) {
    return BufferKeyBit | buffer.data_format << 8 | buffer.num_format;
}

std::string DescribeTextureKey(u32 key) {
    if (key == 0) {
        return "null";
    }
    if (key & BufferKeyBit) {
        const u32 format = (key >> 8) & 0xFF;
        return format == 0 ? std::string{"buffer"}
                           : std::format("buffer {}/{}", gnm::DataFormatName(format),
                                         gnm::NumberFormatName(key & 0xFF));
    }
    return std::format("{} {}/{}", gnm::ImageTypeName(static_cast<gnm::ImageType>(key >> 16)),
                       gnm::DataFormatName((key >> 8) & 0xFF), gnm::NumberFormatName(key & 0xFF));
}

// ---------------------------------------------------------------------------------------------
// Shader specializations: for each draw and dispatch, which shaders run and with which kinds of
// textures (and, for pixel shaders, render target formats) — what the translation of a shader
// depends on besides its code. Tracked per LCUE and per command buffer, from the setters.

// Gnm stage numbers.
constexpr u32 StageCompute = 0;
constexpr u32 StagePixel = 1;
constexpr u32 StageVertex = 2;
constexpr u32 StageCount = 7;
constexpr u32 SlotCount = 16;
constexpr u32 NotSet = 0xFFFFFFFF;

struct StageState {
    u64 hash = 0;
    u64 table = 0; // InputResourceOffsets of the shader
    std::array<u32, SlotCount> textures;
    std::array<u32, SlotCount> rw_textures;
    StageState() {
        textures.fill(NotSet);
        rw_textures.fill(NotSet);
    }
};

struct LcueState {
    std::array<StageState, StageCount> stages;
};

struct CommandBufferState {
    std::array<u32, 8> targets{}; // format << 8 | number type, + 1 << 16 when bound
};

constexpr size_t Shards = 16;
struct StateShard {
    std::mutex mutex;
    std::unordered_map<u64, LcueState> lcues;
    std::unordered_map<u64, CommandBufferState> buffers;
};
std::array<StateShard, Shards> g_state;

StateShard& ShardOf(u64 object) {
    return g_state[(object >> 4 ^ object >> 12) % Shards];
}

struct VariantKey {
    u32 stage;
    u64 hash;
    std::vector<u32> parts; // (slot << 24 | kind << 20) followed by the value, per used slot
    auto operator<=>(const VariantKey&) const = default;
};

std::mutex g_variants_mutex;
std::map<VariantKey, u64> g_variants;

// Which texture slots a shader uses, from its InputResourceOffsets (0xFFFF = unused).
std::array<bool, SlotCount * 2> UsedSlots(u64 table) {
    std::array<bool, SlotCount * 2> used{};
    std::array<u16, SlotCount * 2> slots;
    if (!ReadGame(table + 0x22, slots.data(), sizeof(slots))) { // t# then u# arrays
        return used;
    }
    for (size_t i = 0; i < slots.size(); i++) {
        used[i] = slots[i] != 0xFFFF;
    }
    return used;
}

void NoteVariant(const StageState& stage, u32 stage_number, const CommandBufferState* buffer) {
    if (stage.hash == 0) {
        return;
    }
    VariantKey key{stage_number, stage.hash, {}};
    const auto used = UsedSlots(stage.table);
    for (u32 slot = 0; slot < SlotCount; slot++) {
        if (used[slot]) {
            key.parts.push_back(slot << 24 | 0u << 20);
            key.parts.push_back(stage.textures[slot]);
        }
        if (used[SlotCount + slot]) {
            key.parts.push_back(slot << 24 | 1u << 20);
            key.parts.push_back(stage.rw_textures[slot]);
        }
    }
    if (buffer && stage_number == StagePixel) {
        for (u32 slot = 0; slot < buffer->targets.size(); slot++) {
            if (buffer->targets[slot]) {
                key.parts.push_back(slot << 24 | 2u << 20);
                key.parts.push_back(buffer->targets[slot]);
            }
        }
    }
    std::scoped_lock lock{g_variants_mutex};
    g_variants[key]++;
}

// Textures (32-byte T#) or buffers (16-byte V#) put in resource slots.
void SetResources(u64 lcue, u32 stage, u32 slot, u32 count, u64 descriptors, bool writable,
                  bool buffers) {
    if (stage >= StageCount || slot >= SlotCount || count > SlotCount) {
        return;
    }
    std::array<u32, SlotCount> keys;
    for (u32 i = 0; i < count && slot + i < SlotCount; i++) {
        if (buffers) {
            const auto buffer =
                descriptors ? ReadBufferDescriptor(descriptors + i * 16) : std::nullopt;
            keys[i] = buffer ? BufferKey(*buffer) : 0;
        } else {
            const auto texture = descriptors ? ReadTexture(descriptors + i * 32) : std::nullopt;
            keys[i] = texture ? TextureKey(*texture) : 0;
        }
    }
    StateShard& shard = ShardOf(lcue);
    std::scoped_lock lock{shard.mutex};
    StageState& state = shard.lcues[lcue].stages[stage];
    for (u32 i = 0; i < count && slot + i < SlotCount; i++) {
        (writable ? state.rw_textures : state.textures)[slot + i] = keys[i];
    }
}

void SetShader(u64 lcue, u32 stage, u64 shader, u64 table) {
    const u64 hash = shader ? ShaderHash(shader) : 0;
    StateShard& shard = ShardOf(lcue);
    std::scoped_lock lock{shard.mutex};
    StageState& state = shard.lcues[lcue].stages[stage];
    state.hash = hash;
    state.table = table;
}

void SetRenderTarget(u64 buffer, u32 slot, u64 target) {
    if (slot >= 8) {
        return;
    }
    u32 key = 0;
    if (target) {
        if (const auto info = ReadRenderTarget(target); info && info->address) {
            key = 1u << 16 | info->format << 8 | info->number_type;
        }
    }
    StateShard& shard = ShardOf(buffer);
    std::scoped_lock lock{shard.mutex};
    shard.buffers[buffer].targets[slot] = key;
}

void BeforeDrawOrDispatch(u64 lcue, std::initializer_list<u32> stages, bool graphics) {
    const u64 buffer = graphics ? ReadPointer(lcue + 0xc090) : 0;
    LcueState lcue_state;
    CommandBufferState buffer_state;
    bool has_buffer = false;
    {
        StateShard& shard = ShardOf(lcue);
        std::scoped_lock lock{shard.mutex};
        const auto it = shard.lcues.find(lcue);
        if (it == shard.lcues.end()) {
            return;
        }
        lcue_state = it->second;
    }
    if (buffer) {
        StateShard& shard = ShardOf(buffer);
        std::scoped_lock lock{shard.mutex};
        if (const auto it = shard.buffers.find(buffer); it != shard.buffers.end()) {
            buffer_state = it->second;
            has_buffer = true;
        }
    }
    for (const u32 stage : stages) {
        NoteVariant(lcue_state.stages[stage], stage, has_buffer ? &buffer_state : nullptr);
    }
}

void ForgetContext(u64 lcue, u64 buffer) {
    {
        StateShard& shard = ShardOf(lcue);
        std::scoped_lock lock{shard.mutex};
        shard.lcues.erase(lcue);
    }
    StateShard& shard = ShardOf(buffer);
    std::scoped_lock lock{shard.mutex};
    shard.buffers.erase(buffer);
}

// Follows the state the game sets through the frontier (calls made by the game only).
void TrackState(const ObservedCall& call) {
    const auto& a = call.args;
    switch (Functions[call.id].address) {
    case fn::SetVsShader: SetShader(a[0], StageVertex, a[1], a[4]); break; // (lcue, shader, modifier, fetch, table)
    case fn::SetPsShader: SetShader(a[0], StagePixel, a[1], a[2]); break;  // (lcue, shader, table)
    case fn::SetCsShader: SetShader(a[0], StageCompute, a[1], a[2]); break;
    case fn::ComputeSetCsShader: SetShader(a[0], StageCompute, a[1], a[2]); break;
    // (lcue, stage, slot, n, descriptors)
    case fn::SetTextures: SetResources(a[0], u32(a[1]), u32(a[2]), u32(a[3]), a[4], false, false); break;
    case fn::SetRwTextures: SetResources(a[0], u32(a[1]), u32(a[2]), u32(a[3]), a[4], true, false); break;
    case fn::SetBuffers: SetResources(a[0], u32(a[1]), u32(a[2]), u32(a[3]), a[4], false, true); break;
    case fn::SetRwBuffers: SetResources(a[0], u32(a[1]), u32(a[2]), u32(a[3]), a[4], true, true); break;
    // compute: (lcue, slot, n, descriptors)
    case fn::ComputeSetTextures: SetResources(a[0], StageCompute, u32(a[1]), u32(a[2]), a[3], false, false); break;
    case fn::ComputeSetRwTextures: SetResources(a[0], StageCompute, u32(a[1]), u32(a[2]), a[3], true, false); break;
    case fn::ComputeSetBuffers: SetResources(a[0], StageCompute, u32(a[1]), u32(a[2]), a[3], false, true); break;
    case fn::ComputeSetRwBuffers: SetResources(a[0], StageCompute, u32(a[1]), u32(a[2]), a[3], true, true); break;
    case fn::SetRenderTarget: SetRenderTarget(a[0], u32(a[1]), a[2]); break; // (buffer, slot, target)
    case fn::PreDraw: BeforeDrawOrDispatch(a[0], {StageVertex, StagePixel}, true); break;
    case fn::PreDispatch: BeforeDrawOrDispatch(a[0], {StageCompute}, true); break;
    case fn::ComputePreDispatch: BeforeDrawOrDispatch(a[0], {StageCompute}, false); break;
    case fn::InitializeContext: ForgetContext(a[0] + 0x2e8, a[0]); break;            // (context = its buffer)
    case fn::ComputeResetContext: ForgetContext(a[0] + 0x50, a[0] + 0x10); break;      // compute context
    default: break;
    }
}

// The arguments of a call, decoded where it helps reading a recorded frame.
std::string DescribeArgs(const ObservedCall& call) {
    const auto& a = call.args;
    auto hex = [&](size_t count) {
        std::string text;
        for (size_t i = 0; i < count; i++) {
            text += std::format(" {:#x}", a[i]);
        }
        return text;
    };
    auto list = [&](u64 descriptors, u32 count, bool texture) {
        std::string text;
        for (u32 i = 0; i < count && i < SlotCount; i++) {
            if (texture) {
                const auto info = descriptors ? ReadTexture(descriptors + i * 32) : std::nullopt;
                text += " [" + (info ? gnm::Describe(*info) : std::string{"?"}) + "]";
            } else {
                const auto info = descriptors ? ReadBufferDescriptor(descriptors + i * 16)
                                              : std::nullopt;
                text += " [" + (info ? gnm::Describe(*info) : std::string{"?"}) + "]";
            }
        }
        return text;
    };
    switch (Functions[call.id].address) {
    case fn::SetVsShader:
        return std::format(" {:#x} shader {:016x} fetch {:#x}", a[0], ShaderHash(a[1]), a[3]);
    case fn::SetPsShader: case fn::SetCsShader: case fn::ComputeSetCsShader:
        return std::format(" {:#x} shader {:016x}", a[0], a[1] ? ShaderHash(a[1]) : 0);
    case fn::SetTextures: case fn::SetRwTextures:
        return std::format(" {:#x} stage {} slot {}", a[0], a[1], a[2]) + list(a[4], u32(a[3]), true);
    case fn::SetBuffers: case fn::SetRwBuffers: case fn::SetConstantBuffers:
        return std::format(" {:#x} stage {} slot {}", a[0], a[1], a[2]) + list(a[4], u32(a[3]), false);
    case fn::ComputeSetTextures: case fn::ComputeSetRwTextures:
        return std::format(" {:#x} slot {}", a[0], a[1]) + list(a[3], u32(a[2]), true);
    case fn::ComputeSetBuffers: case fn::ComputeSetRwBuffers: case fn::ComputeSetConstantBuffers:
        return std::format(" {:#x} slot {}", a[0], a[1]) + list(a[3], u32(a[2]), false);
    case fn::SetRenderTarget: {
        const auto info = a[2] ? ReadRenderTarget(a[2]) : std::nullopt;
        return std::format(" {:#x} slot {} ", a[0], a[1]) +
               (a[2] == 0 ? std::string{"null"} : info ? gnm::Describe(*info) : std::string{"?"});
    }
    case fn::Dispatch: case fn::ComputeDispatch:
        return std::format(" {:#x} groups {} {} {}", a[0], a[1], a[2], a[3]);
    case fn::DrawIndexAuto:
        return std::format(" {:#x} count {}", a[0], a[1]);
    // pushMarker(buffer, name): the engine's debug name for the section that follows, written
    // into the command buffer as a NOP for GPU debuggers (the frame's structure, in its own words).
    case fn::PushMarker: case fn::ComputePushMarker:
        return std::format(" {:#x} \"{}\"", a[0], ReadGameString(a[1]));
    case fn::DrawIndexOffset:
        return std::format(" {:#x} first {} count {}", a[0], a[1], a[2]);
    default:
        return hex(6);
    }
}

// ---------------------------------------------------------------------------------------------
// Frame recording.

std::mutex g_record_mutex;
std::vector<double> g_record_times; // seconds, latest first
std::atomic<bool> g_recording{false};
std::vector<std::string> g_record_lines;
double g_record_start = 0;
int g_frames_written = 0;
constexpr size_t MaxRecordedLines = 2'000'000;
// The engine works on more than one frame at a time: when a frame is flipped, the next one's
// late passes (lighting, post-processing) may not be built yet. A recording therefore spans
// several flip intervals, so that at least one frame is complete in it.
constexpr int FlipsPerRecording = 3;
int g_flips_left = 0;

std::atomic<u32> g_thread_count{0};
thread_local u32 t_thread_index = g_thread_count.fetch_add(1);

void WriteRecordedFrame() {
    const auto directory = DataPath("gnm_frames");
    std::filesystem::create_directories(directory);
    const auto path = directory / std::format("frame_{}.txt", g_frames_written++);
    std::FILE* file = host_file::OpenStdio(path, "w");
    if (!file) {
        return;
    }
    std::fprintf(file,
                 "# %d flip intervals from %.1f s: ms since the first flip, thread, function, "
                 "arguments (submitAndFlip lines are the frame boundaries)\n",
                 FlipsPerRecording, g_record_start);
    for (const std::string& line : g_record_lines) {
        std::fputs(line.c_str(), file);
    }
    std::fclose(file);
    LOG_INFO("Recorder", "frame recorded: {} calls in {}", g_record_lines.size(), path.string());
    g_record_lines.clear();
}

void OnFlip() {
    std::scoped_lock lock{g_record_mutex};
    if (g_recording.load(std::memory_order_relaxed)) {
        if (--g_flips_left > 0) {
            return; // the flip itself is recorded by the caller, as a frame boundary
        }
        g_recording.store(false, std::memory_order_relaxed);
        WriteRecordedFrame();
    }
    const double now = Seconds();
    if (!g_record_times.empty() && g_record_times.back() <= now) {
        while (!g_record_times.empty() && g_record_times.back() <= now) {
            g_record_times.pop_back();
        }
        g_record_start = now;
        g_flips_left = FlipsPerRecording;
        g_recording.store(true, std::memory_order_relaxed);
    }
}

void RecordCall(const ObservedCall& call) {
    std::string line = std::format("{:9.3f} t{:<2} {:<34}", (Seconds() - g_record_start) * 1000,
                                   t_thread_index, Functions[call.id].name);
    line += DescribeArgs(call);
    line += '\n';
    std::scoped_lock lock{g_record_mutex};
    if (g_recording.load(std::memory_order_relaxed) && g_record_lines.size() < MaxRecordedLines) {
        g_record_lines.push_back(std::move(line));
    }
}

// ---------------------------------------------------------------------------------------------
// The state proof: the native renderer's state model (render/state_model.h) is fed with the
// game's calls; at each draw call its snapshot is kept under the address of the draw packet the
// call writes (the command buffer's cursor before the call), and when the GPU layer executes
// that packet, OnPacket compares the snapshot with the registers the GPU actually has.

render::StateModel g_model;

struct PendingDraw {
    u64 buffer;
    render::RegisterSnapshot snapshot;
};
std::mutex g_pending_mutex;
std::unordered_map<u64, PendingDraw> g_pending_draws;
// What each command buffer inherited from the queue when it was submitted.
std::unordered_map<u64, render::RegisterSnapshot> g_inherited;
// Draw packets nobody executes (a command buffer dropped by the game) would pile up.
constexpr size_t MaxPendingDraws = 200'000;

struct RegisterProof {
    u64 compared = 0;
    u64 mismatched = 0;
    u64 unknown = 0; // draws where the model had not set it since the buffer started
    u64 excluded = 0; // draws where something the model does not follow yet wrote it
    // User data the model does not know (or excludes) that the draw's or dispatch's shader is
    // not even given (at or above its USER_SGPR count), and the indirect arguments address at a
    // direct draw: whatever they hold does not matter.
    u64 unused = 0;
    std::map<u32, u64> unknown_values; // what the GPU had in those draws (a few distinct ones)
    std::vector<std::pair<u32, u32>> samples; // (model, GPU) of the first mismatches
};
std::mutex g_proof_mutex;
std::array<RegisterProof, render::ModelledCount> g_proof;
u64 g_draws_checked = 0;
u64 g_dispatches_checked = 0;
u64 g_draws_without_snapshot = 0;

// The resource table copies (render::ResourceTableCopy) behind the pointer SGPRs, per stage.
struct TableProof {
    u64 draws = 0;           // draws whose SGPRs point at a copy of this stage's table
    u64 inconsistent = 0;    // pointers of one draw that disagree about where the copy is
    u64 no_table = 0;        // the model had no copy for the stage
    u64 unreadable = 0;      // the copy's memory could not be read
    u64 dwords_compared = 0;
    u64 dwords_mismatched = 0;
    u64 dwords_unknown = 0;  // dwords no setter wrote (since the LCUE was initialized)
    std::vector<std::array<u32, 3>> samples; // (dword, model, GPU memory) of first mismatches
    std::map<u32, u64> unknown_dwords;        // dword index -> draws, for the unknown ones
    std::map<u32, u32> unknown_last;          // dword index -> the last value seen there
};
std::array<TableProof, render::ModelledStages> g_table_proof; // CS, PS, VS

// Slot in ModelledRegisters of the user data register `sgpr` of `stage` (0 CS, 1 PS, 2 VS).
size_t UserDataSlot(u32 stage, u32 sgpr) {
    static const auto slots = [] {
        std::array<size_t, render::ModelledStages * render::UserDataCount> all{};
        for (size_t i = 0; i < render::ModelledCount; i++) {
            const u32 index = render::ModelledRegisters[i].index;
            for (u32 s = 0; s < render::ModelledStages; s++) {
                const u32 base = render::UserDataBase[s];
                if (index >= base && index < base + render::UserDataCount) {
                    all[s * render::UserDataCount + index - base] = i;
                }
            }
        }
        return all;
    }();
    return slots[stage * render::UserDataCount + sgpr];
}

// A register as the GPU layer has it now: compute registers from the running queue's set.
u32 GpuRegister(u32 index) {
    if (index == render::IndirectArgsBase || index == render::IndirectArgsBase + 1) {
        const u64 address = gpu::ReadIndirectArgsAddress();
        return index == render::IndirectArgsBase ? static_cast<u32>(address)
                                                 : static_cast<u32>(address >> 32);
    }
    return render::IsComputeRegister(index) ? gpu::ReadComputeRegister(index)
                                            : gpu::ReadRegister(index);
}

// Checks the table copy the GPU's SGPRs point at for one stage of one draw or dispatch (called
// with g_proof_mutex held); true when the pointers agree and every known dword matches.
bool CheckTable(const render::RegisterSnapshot& snapshot, u32 stage) {
    TableProof& proof = g_table_proof[stage];
    u64 base = 0;
    bool any = false, consistent = true;
    for (u32 sgpr = 0; sgpr + 1 < render::UserDataCount; sgpr++) {
        const size_t slot = UserDataSlot(stage, sgpr);
        if (!snapshot.table_pointer.test(slot) || (snapshot.values[slot] & render::TablePointerHigh)) {
            continue;
        }
        const size_t high = UserDataSlot(stage, sgpr + 1);
        const u32 offset = snapshot.values[slot];
        if (!snapshot.table_pointer.test(high) ||
            snapshot.values[high] != (offset | render::TablePointerHigh)) {
            consistent = false;
        }
        const u32 index = render::UserDataBase[stage] + sgpr;
        const u64 address = u64{GpuRegister(index)} | u64{GpuRegister(index + 1)} << 32;
        const u64 start = address - u64{offset} * 4;
        if (any && start != base) {
            consistent = false;
        }
        base = start;
        any = true;
    }
    if (!any) {
        return true;
    }
    proof.draws++;
    if (!consistent) {
        proof.inconsistent++;
        return false;
    }
    const auto& table = snapshot.tables[stage];
    if (!table) {
        proof.no_table++;
        return false;
    }
    std::vector<u32> memory(table->values.size());
    if (!memory.empty() && !ReadGame(base, memory.data(), memory.size() * 4)) {
        proof.unreadable++;
        return false;
    }
    bool matched = true;
    for (size_t i = 0; i < memory.size(); i++) {
        if (!table->known[i]) {
            proof.dwords_unknown++;
            if (proof.unknown_dwords.size() < 16 || proof.unknown_dwords.contains(i)) {
                proof.unknown_dwords[static_cast<u32>(i)]++;
                proof.unknown_last[static_cast<u32>(i)] = memory[i];
            }
            continue;
        }
        proof.dwords_compared++;
        if (memory[i] != table->values[i]) {
            proof.dwords_mismatched++;
            matched = false;
            if (proof.samples.size() < 8) {
                proof.samples.push_back({static_cast<u32>(i), table->values[i], memory[i]});
            }
        }
    }
    return matched;
}

// Calls whose packet gets a snapshot: draws, and dispatches of either queue.
bool IsDrawOrDispatch(u64 function) {
    return function == fn::DrawIndexIndirect || function == fn::DrawIndexOffset ||
           function == fn::DrawIndexAuto || function == fn::DrawIndirect ||
           function == fn::Dispatch || function == fn::ComputeDispatch ||
           function == fn::ComputeDispatchIndirect;
}

// PM4 opcodes of the packets those methods write first.
bool IsDrawPacket(u32 kind) {
    return kind == 0x25 || kind == 0x35 || kind == 0x2d || kind == 0x24 || kind == 0x27;
}

void FeedModel(const ObservedCall& call, u64 cursor_before) {
    const render::FrontierCall model_call{Functions[call.id].address, call.args, call.float_args,
                                          call.stack_args};
    const u64 function = model_call.function;
    // A draw's own effects (VGT_NUM_INDICES...) reach the GPU's registers only when its packet
    // runs, after the observer compared them: snapshot first, then apply.
    if (IsDrawOrDispatch(function) && cursor_before) {
        render::RegisterSnapshot snapshot = g_model.Snapshot(call.args[0]);
        {
            std::scoped_lock lock{g_pending_mutex};
            if (g_pending_draws.size() >= MaxPendingDraws) {
                g_pending_draws.clear();
            }
            g_pending_draws[cursor_before] = {call.args[0], snapshot};
        }
        g_model.Apply(model_call);
        return;
    }
    g_model.Apply(model_call);
    if (function == fn::Submit || function == fn::SubmitAndFlip ||
               function == fn::SubmitDirect) {
        render::RegisterSnapshot inherited = g_model.Submit(call.args[0]);
        std::scoped_lock lock{g_pending_mutex};
        g_inherited[call.args[0]] = inherited;
    } else if (function == fn::ComputeSubmit) {
        // (queue, workload, compute context): its DispatchCommandBuffer is at context + 0x10.
        const u64 buffer = call.args[2] + 0x10;
        {
            static std::mutex queues_mutex;
            static std::vector<u64> queues;
            std::scoped_lock lock{queues_mutex};
            if (std::ranges::find(queues, call.args[0]) == queues.end()) {
                queues.push_back(call.args[0]);
                LOG_INFO("Recorder", "compute queue object {:#x} (#{})", call.args[0],
                         queues.size());
            }
        }
        render::RegisterSnapshot inherited = g_model.Submit(buffer, call.args[0]);
        std::scoped_lock lock{g_pending_mutex};
        g_inherited[buffer] = inherited;
    }
}

// At a draw packet the graphics registers are compared, at a dispatch packet the compute ones.
// `indirect`: the draw reads its arguments from memory (only then is the SET_BASE address used).
void CheckState(u64 address, bool dispatch, bool indirect = false, bool compute_queue = false) {
    render::RegisterSnapshot snapshot;
    {
        std::scoped_lock lock{g_pending_mutex};
        const auto it = g_pending_draws.find(address);
        if (it == g_pending_draws.end()) {
            std::scoped_lock proof_lock{g_proof_mutex};
            g_draws_without_snapshot++;
            return;
        }
        snapshot = it->second.snapshot;
        // The GPU runs a buffer after it was submitted, so what it inherits is known by now.
        if (const auto inherited = g_inherited.find(it->second.buffer);
            inherited != g_inherited.end()) {
            render::StateModel::Inherit(snapshot, inherited->second);
        }
        g_pending_draws.erase(it);
    }
    if (dispatch && NativeValidationEnabled()) {
        // The native renderer runs the same dispatch from this state (before the GPU layer does).
        ValidateDispatchBefore(snapshot, reinterpret_cast<const u32*>(address), indirect,
                               compute_queue);
    } else if (!dispatch && NativeValidationEnabled()) {
        ValidateDrawBefore(snapshot, reinterpret_cast<const u32*>(address));
    }
    std::scoped_lock lock{g_proof_mutex};
    std::array<bool, render::ModelledStages> tables_ok{true, true, true};
    if (dispatch) {
        g_dispatches_checked++;
        tables_ok[render::StageCompute] = CheckTable(snapshot, render::StageCompute);
    } else {
        g_draws_checked++;
        tables_ok[render::StagePixel] = CheckTable(snapshot, render::StagePixel);
        tables_ok[render::StageVertex] = CheckTable(snapshot, render::StageVertex);
    }
    // How many user data SGPRs each stage's shader is given (RSRC2 bits 1-5, USER_SGPR).
    auto user_sgprs = [](u32 stage) {
        constexpr std::array<u32, render::ModelledStages> rsrc2 = {0x2e13, 0x2c0b, 0x2c4b};
        return (GpuRegister(rsrc2[stage]) >> 1) & 0x1f;
    };
    const std::array<u32, render::ModelledStages> given = {
        dispatch ? user_sgprs(render::StageCompute) : 0,
        dispatch ? 0 : user_sgprs(render::StagePixel),
        dispatch ? 0 : user_sgprs(render::StageVertex)};
    for (size_t i = 0; i < render::ModelledCount; i++) {
        RegisterProof& proof = g_proof[i];
        const u32 index = render::ModelledRegisters[i].index;
        if (render::IsComputeRegister(index) != dispatch) {
            continue;
        }
        if (!snapshot.known.test(i) || snapshot.excluded.test(i)) {
            bool is_unused = !indirect && (index == render::IndirectArgsBase ||
                                           index == render::IndirectArgsBase + 1);
            for (u32 stage = 0; stage < render::ModelledStages; stage++) {
                const u32 base = render::UserDataBase[stage];
                if (index >= base && index < base + render::UserDataCount) {
                    is_unused = index - base >= given[stage];
                }
            }
            if (is_unused) {
                proof.unused++;
                continue;
            }
        }
        if (snapshot.excluded.test(i)) {
            proof.excluded++;
            continue;
        }
        if (snapshot.table_pointer.test(i)) {
            // Proven through the copy it points at (CheckTable).
            const u32 index = render::ModelledRegisters[i].index;
            const u32 stage = dispatch ? render::StageCompute
                              : index >= render::UserDataBase[render::StageVertex]
                                  ? render::StageVertex
                                  : render::StagePixel;
            proof.compared++;
            if (!tables_ok[stage]) {
                proof.mismatched++;
                if (proof.samples.size() < 4) {
                    proof.samples.emplace_back(snapshot.values[i], GpuRegister(index));
                }
            }
            continue;
        }
        if (!snapshot.known.test(i)) {
            proof.unknown++;
            const u32 gpu = GpuRegister(render::ModelledRegisters[i].index);
            if (proof.unknown_values.size() < 4 || proof.unknown_values.contains(gpu)) {
                proof.unknown_values[gpu]++;
            }
            continue;
        }
        proof.compared++;
        const u32 gpu = GpuRegister(render::ModelledRegisters[i].index);
        if (gpu != snapshot.values[i]) {
            proof.mismatched++;
            if (proof.samples.size() < 4) {
                proof.samples.emplace_back(snapshot.values[i], gpu);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The survey. The frontier proof only sees functions that write commands; a Sony function that
// just stores state in an LCUE (which the pre-draw turns into commands later) passes unseen. The
// functions below are every function of Sony's block (0xc5c000-0xc9bba0) that the game's code
// calls directly and that is not in the frontier table (static scan of call/jmp instructions,
// 2026-10-10). Each is watched without changing it: calls from the game and from Sony's own code,
// and what its first two arguments are (an object the frontier functions use, or something else).

constexpr u64 SurveyFunctions[] = {
    0xc5c040, 0xc5c050, 0xc5cf20, 0xc5cf30, 0xc5d2e0, 0xc5d770, 0xc5d8e0, 0xc5db40,
    0xc60bb0, 0xc60c50, 0xc61260, 0xc612b0, 0xc652d0, 0xc65360, 0xc65410, 0xc65500,
    0xc655d0, 0xc65600, 0xc65620, 0xc66960, 0xc66a80, 0xc66aa0, 0xc66b30, 0xc66e10,
    0xc66f50, 0xc66fa0, 0xc67420, 0xc67770, 0xc67830, 0xc678f0, 0xc67990, 0xc68620,
    0xc68710, 0xc688e0, 0xc689d0, 0xc68ba0, 0xc68ca0, 0xc68e90, 0xc69180, 0xc691f0,
    0xc6a700, 0xc6bac0, 0xc6c5f0, 0xc6e670, 0xc6ee70, 0xc7c380, 0xc7c390, 0xc7c3b0,
    0xc7c3d0, 0xc7c400, 0xc7c4a0, 0xc7c4b0, 0xc7c4c0, 0xc7c4d0, 0xc7c510, 0xc7c620,
    0xc7c650, 0xc7c670, 0xc7c680, 0xc7c7a0, 0xc7c840, 0xc7c8c0, 0xc7cb90, 0xc7cba0,
    0xc7cbc0, 0xc7cbe0, 0xc7cc00, 0xc7cc10, 0xc7cc20, 0xc7cc40, 0xc7cd70, 0xc7cd80,
    0xc7cda0, 0xc7d070, 0xc7d0a0, 0xc7d0c0, 0xc7d0e0, 0xc7d100, 0xc7d120, 0xc7da70,
    0xc7dac0, 0xc7dea0, 0xc7e280, 0xc7e320, 0xc7e5f0, 0xc7eb10, 0xc7ec70, 0xc7eda0,
    0xc7efd0, 0xc7f0d0, 0xc7f3d0, 0xc7f3e0, 0xc7f400, 0xc7f790, 0xc80360, 0xc803d0,
    0xc80530, 0xc805c0, 0xc806a0, 0xc806d0, 0xc806f0, 0xc80c30, 0xc80c80, 0xc81090,
    0xc81b10, 0xc81e60, 0xc82150, 0xc82420, 0xc82710, 0xc829e0, 0xc82a60, 0xc82ce0,
    0xc82d60, 0xc82ec0, 0xc82f10, 0xc83640, 0xc843d0, 0xc85540, 0xc85640, 0xc85890,
    0xc85bd0, 0xc8f000,
};
constexpr size_t SurveyCount = std::size(SurveyFunctions);

// What an argument is, judged by the objects the frontier functions were called with.
enum class ObjectKind : u8 {
    Other,          // anything else (a descriptor, a plain value, memory of the game's)
    Stack,          // a local variable of the caller (within 1 MiB of the current stack)
    GraphicsContext, // a LightweightGfxContext = its DrawCommandBuffer (at +0)
    GraphicsLcue,   // context + 0x2e8
    InsideGraphicsContext, // a field inside a graphics context (0xcbb8 bytes)
    CommandBuffer,  // a command buffer a frontier method was called on (not a context's)
    ComputeLcue,
    ComputeContext,
    ComputeQueue,
    Count,
};
constexpr const char* ObjectKindNames[] = {"other",        "stack", "gfx-context", "gfx-lcue",
                                           "in-gfx-context", "cmd-buffer",    "compute-lcue",
                                           "compute-context", "compute-queue"};
constexpr u64 GraphicsContextSize = 0xcbb8;

std::mutex g_objects_mutex;
std::unordered_map<u64, ObjectKind> g_objects;
// start -> {end, when a frontier function last used it}. The game allocates contexts in
// transient memory: only one used in the last 100 ms counts as live.
std::map<u64, std::pair<u64, double>> g_graphics_contexts;
bool g_survey_enabled = false;

// Called for every frontier call made by the game: remembers its objects.
void NoteObjects(const FrontierFunction& function, u64 arg0) {
    if (!g_survey_enabled) {
        return;
    }
    std::scoped_lock lock{g_objects_mutex};
    switch (function.buffers) {
    case Buffers::GraphicsLcue:
        g_objects[arg0] = ObjectKind::GraphicsLcue;
        g_objects[arg0 - 0x2e8] = ObjectKind::GraphicsContext;
        g_graphics_contexts[arg0 - 0x2e8] = {arg0 - 0x2e8 + GraphicsContextSize, Seconds()};
        break;
    case Buffers::Self:
        g_objects.try_emplace(arg0, ObjectKind::CommandBuffer);
        break;
    case Buffers::ComputeLcue:
        g_objects[arg0] = ObjectKind::ComputeLcue;
        break;
    case Buffers::ComputeContext:
        g_objects[arg0] = ObjectKind::ComputeContext;
        break;
    case Buffers::ComputeQueue:
        g_objects[arg0] = ObjectKind::ComputeQueue;
        break;
    case Buffers::None:
        break;
    }
}

// `offset` receives the offset inside the object for InsideGraphicsContext.
ObjectKind KindOf(u64 value, u64* offset = nullptr) {
    // The hook runs on the game's own stack (fibers included), so a local of ours marks it.
    const u64 stack = reinterpret_cast<u64>(&value);
    if (value - (stack - (1 << 20)) < (2 << 20)) {
        return ObjectKind::Stack;
    }
    std::scoped_lock lock{g_objects_mutex};
    if (const auto it = g_objects.find(value); it != g_objects.end()) {
        return it->second;
    }
    if (auto it = g_graphics_contexts.upper_bound(value); it != g_graphics_contexts.begin()) {
        --it;
        if (value < it->second.first && Seconds() - it->second.second < 0.1) {
            if (offset) {
                *offset = value - it->first;
            }
            return ObjectKind::InsideGraphicsContext;
        }
    }
    return ObjectKind::Other;
}

struct SurveyStats {
    std::atomic<u64> game_calls{0};
    std::atomic<u64> sony_calls{0}; // from Sony's own block (nested in a frontier call)
    // Calls from the game by the kind of the first and of the second argument.
    std::array<std::array<std::atomic<u64>, size_t(ObjectKind::Count)>, 2> kinds{};
    std::mutex sample_mutex;
    std::vector<std::array<u64, 4>> samples; // arguments of the first game calls
    u64 caller = 0;                          // one return address in the game
    std::array<u64, 2> inside_offset{~0ull, ~0ull}; // an offset seen for in-gfx-context
};
std::array<SurveyStats, SurveyCount> g_survey;

bool InSonyBlock(u64 address) {
    return address >= 0xc5c000 && address < 0xc9bba0;
}

void SurveyEnter(ObservedCall& call) {
    SurveyStats& stats = g_survey[call.id];
    if (InSonyBlock(call.return_address)) {
        stats.sony_calls.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    stats.game_calls.fetch_add(1, std::memory_order_relaxed);
    std::array<u64, 2> offsets{~0ull, ~0ull};
    for (size_t i = 0; i < 2; i++) {
        stats.kinds[i][size_t(KindOf(call.args[i], &offsets[i]))].fetch_add(
            1, std::memory_order_relaxed);
    }
    std::scoped_lock lock{stats.sample_mutex};
    for (size_t i = 0; i < 2; i++) {
        if (offsets[i] != ~0ull) {
            stats.inside_offset[i] = offsets[i];
        }
    }
    if (stats.samples.size() < 3) {
        stats.samples.push_back({call.args[0], call.args[1], call.args[2], call.args[3]});
        stats.caller = call.return_address;
    }
}

void SurveyExit(const ObservedCall&) {}

const CallObserver g_survey_observer{SurveyEnter, SurveyExit};
std::vector<bool> g_survey_hooked;

void WriteSurveyReport(std::FILE* file) {
    std::fprintf(file, "\n== survey: other Sony functions the game calls directly (calls from the "
                       "game / from Sony's code; kinds of the 1st and 2nd arguments; samples)\n");
    {
        std::scoped_lock lock{g_objects_mutex};
        std::string contexts;
        for (const auto& [start, range] : g_graphics_contexts) {
            contexts += std::format(" {:#x}", start);
        }
        std::fprintf(file, "  graphics contexts seen (%zu):%s\n", g_graphics_contexts.size(),
                     contexts.substr(0, 400).c_str());
    }
    for (size_t i = 0; i < SurveyCount; i++) {
        SurveyStats& stats = g_survey[i];
        if (!g_survey_hooked[i]) {
            std::fprintf(file, "  %#llx  (not hooked: too short)\n",
                         static_cast<unsigned long long>(SurveyFunctions[i]));
            continue;
        }
        std::string kinds;
        for (size_t a = 0; a < 2; a++) {
            kinds += a == 0 ? "  arg0:" : "  arg1:";
            for (size_t k = 0; k < size_t(ObjectKind::Count); k++) {
                if (const u64 count = stats.kinds[a][k].load(std::memory_order_relaxed)) {
                    kinds += std::format(" {} {}", ObjectKindNames[k], count);
                }
            }
        }
        std::string samples;
        {
            std::scoped_lock lock{stats.sample_mutex};
            for (const auto& args : stats.samples) {
                samples += std::format(" ({:#x}, {:#x}, {:#x}, {:#x})", args[0], args[1], args[2],
                                       args[3]);
            }
            if (stats.caller) {
                samples += std::format("  caller {:#x}", stats.caller);
            }
            for (size_t a = 0; a < 2; a++) {
                if (stats.inside_offset[a] != ~0ull) {
                    samples += std::format("  arg{} at context+{:#x}", a, stats.inside_offset[a]);
                }
            }
        }
        std::fprintf(file, "  %#llx  %10llu %10llu%s%s\n",
                     static_cast<unsigned long long>(SurveyFunctions[i]),
                     static_cast<unsigned long long>(stats.game_calls.load()),
                     static_cast<unsigned long long>(stats.sony_calls.load()), kinds.c_str(),
                     samples.c_str());
    }
}

// ---------------------------------------------------------------------------------------------
// The calls.

// ObservedCall::user: [i * 3] buffer object i (0 = none), [i * 3 + 1] its start, [i * 3 + 2]
// its cursor before the call.
void OnEnter(ObservedCall& call) {
    const FrontierFunction& function = Functions[call.id];
    g_stats[call.id].calls.fetch_add(1, std::memory_order_relaxed);
    if (call.depth == 0) {
        t_inner_marks.clear();
    }
    if (function.role == Role::InitBuffer && call.args[2] <= (64ull << 20)) {
        ClearMarks(call.args[1], call.args[1] + call.args[2]);
    }
    if (function.role == Role::Flip && call.depth == 0) {
        OnFlip();
    }
    if (call.depth == 0) {
        NoteObjects(function, call.args[0]);
        TrackState(call);
        if (g_recording.load(std::memory_order_relaxed)) {
            RecordCall(call);
        }
    }
    const auto objects = BufferObjects(function.buffers, call.args[0]);
    // (the model is fed below, once the cursor before the call is known)
    for (size_t i = 0; i < objects.size(); i++) {
        BufferState state;
        if (objects[i] == 0) {
            continue;
        }
        if (!ReadBuffer(objects[i], state)) {
            // The second buffer of a graphics LCUE is often absent, and a context being
            // initialized may not have a buffer yet.
            if (i == 0 && function.role != Role::InitBuffer) {
                g_stats[call.id].bad_buffer.fetch_add(1, std::memory_order_relaxed);
            }
            continue;
        }
        call.user[i * 3] = objects[i];
        call.user[i * 3 + 1] = state.begin;
        call.user[i * 3 + 2] = state.cursor;
    }
    if (call.depth == 0) {
        FeedModel(call, call.user[0] ? call.user[2] : 0);
    }
}

void OnExit(const ObservedCall& call) {
    FunctionStats& stats = g_stats[call.id];
    const u8 owner = static_cast<u8>(call.id + 1);
    for (size_t i = 0; i < 2; i++) {
        const u64 object = call.user[i * 3];
        BufferState state;
        if (object == 0 || !ReadBuffer(object, state)) {
            continue;
        }
        u64 begin = call.user[i * 3 + 2];
        if (state.begin != call.user[i * 3 + 1]) {
            // The buffer was replaced during the call (it was full, or the context is being
            // initialized): what is in the new one was written now; the end of the old one is
            // not known.
            if (Functions[call.id].role != Role::InitBuffer) {
                stats.buffer_switches.fetch_add(1, std::memory_order_relaxed);
            }
            begin = state.begin;
        }
        if (state.cursor <= begin) {
            continue;
        }
        if (state.cursor - begin > MaxCallBytes) {
            stats.too_far.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        Mark(begin, state.cursor, owner);
        stats.dwords.fetch_add((state.cursor - begin) / 4, std::memory_order_relaxed);
        if (call.depth > 0) {
            t_inner_marks.push_back({begin, state.cursor, owner});
        }
    }
    if (call.depth == 0) {
        for (const InnerMark& inner : t_inner_marks) {
            Mark(inner.begin, inner.end, inner.owner);
        }
        t_inner_marks.clear();
    }
}

const CallObserver g_observer{OnEnter, OnExit};

// ---------------------------------------------------------------------------------------------
// The GPU side.

const char* QueueName(u32 queue) {
    switch (queue) {
    case 0: return "graphics";
    case 1: return "constant";
    case 2: return "compute";
    default: return "?";
    }
}

std::string PacketName(u32 kind) {
    if (kind >= 0x100) {
        return kind == 0x102 ? "filler" : std::format("type{}", kind - 0x100);
    }
    static const std::map<u32, const char*> names = {
        {0x10, "NOP"}, {0x11, "SET_BASE"}, {0x12, "CLEAR_STATE"}, {0x13, "INDEX_BUFFER_SIZE"},
        {0x15, "DISPATCH_DIRECT"}, {0x16, "DISPATCH_INDIRECT"}, {0x20, "SET_PREDICATION"},
        {0x22, "COND_EXEC"}, {0x23, "PRED_EXEC"}, {0x24, "DRAW_INDIRECT"},
        {0x25, "DRAW_INDEX_INDIRECT"}, {0x26, "INDEX_BASE"}, {0x27, "DRAW_INDEX_2"},
        {0x28, "CONTEXT_CONTROL"}, {0x2a, "INDEX_TYPE"}, {0x2c, "DRAW_INDIRECT_MULTI"},
        {0x2d, "DRAW_INDEX_AUTO"}, {0x2f, "NUM_INSTANCES"}, {0x30, "DRAW_INDEX_MULTI_AUTO"},
        {0x33, "INDIRECT_BUFFER_CONST"}, {0x35, "DRAW_INDEX_OFFSET_2"}, {0x37, "WRITE_DATA"},
        {0x38, "DRAW_INDEX_INDIRECT_MULTI"}, {0x39, "MEM_SEMAPHORE"}, {0x3c, "WAIT_REG_MEM"},
        {0x3f, "INDIRECT_BUFFER"}, {0x40, "COPY_DATA"}, {0x42, "PFP_SYNC_ME"},
        {0x46, "EVENT_WRITE"}, {0x47, "EVENT_WRITE_EOP"}, {0x48, "EVENT_WRITE_EOS"},
        {0x49, "RELEASE_MEM"}, {0x50, "DMA_DATA"}, {0x58, "ACQUIRE_MEM"}, {0x59, "REWIND"},
        {0x68, "SET_CONFIG_REG"}, {0x69, "SET_CONTEXT_REG"}, {0x76, "SET_SH_REG"},
        {0x79, "SET_UCONFIG_REG"}, {0x81, "WRITE_CONST_RAM"}, {0x83, "DUMP_CONST_RAM"},
        {0x84, "INCREMENT_CE_COUNTER"}, {0x85, "INCREMENT_DE_COUNTER"},
        {0x86, "WAIT_ON_CE_COUNTER"}, {0x88, "WAIT_ON_DE_COUNTER_DIFF"}, {0x8e, "GET_LOD_STATS"},
    };
    const auto it = names.find(kind);
    return it != names.end() ? it->second : std::format("op{:02x}", kind);
}

const char* OwnerName(u8 owner) {
    if (owner == OwnerGpuLayer) {
        return "(GPU layer, not the game)";
    }
    return owner == 0 ? "(nobody)" : Functions[owner - 1].name;
}

struct PacketStats {
    u64 packets = 0;
    u64 dwords = 0;
    u64 uncovered_dwords = 0;
};

struct UncoveredSample {
    u32 queue, kind, dwords, covered;
    u64 address;
    u8 owner_before, owner_after;
    std::array<u32, 4> first;
    double seconds;
};

std::mutex g_gpu_mutex;
// Key: queue << 24 | packet kind << 8 | owner of the header.
std::map<u32, PacketStats> g_packets;
std::vector<UncoveredSample> g_samples;
std::map<u32, u32> g_samples_per_kind; // queue << 16 | kind
constexpr size_t MaxSamples = 128;
constexpr u32 MaxSamplesPerKind = 8;

void OnPacket(const u32* packet, u32 dwords, u32 queue) {
    const u64 address = reinterpret_cast<u64>(packet);
    const u32 header = packet[0];
    const u32 type = header >> 30;
    const u32 kind = type == 3 ? (header >> 8) & 0xFF : 0x100 + type;
    u8 owner = OwnerAt(address);
    u32 covered = 0;
    for (u32 i = 0; i < dwords; i++) {
        covered += OwnerAt(address + i * 4) != 0;
    }
    if (covered == 0 && hle::kernel::ClampToMapped(address, 4) == 0) {
        owner = OwnerGpuLayer;
        covered = dwords;
    }
    if (NativeValidationEnabled()) {
        ValidateAfterPacket(); // the previous packet's dispatch has been recorded by now
    }
    if (queue == 0 && IsDrawPacket(kind)) {
        CheckState(address, false, kind == 0x24 || kind == 0x25); // DRAW_(INDEX_)INDIRECT
    } else if ((queue == 0 || queue == 2) && (kind == 0x15 || kind == 0x16)) {
        CheckState(address, true, kind == 0x16, queue == 2); // DISPATCH_DIRECT / _INDIRECT
    }
    std::scoped_lock lock{g_gpu_mutex};
    PacketStats& stats = g_packets[queue << 24 | kind << 8 | owner];
    stats.packets++;
    stats.dwords += dwords;
    stats.uncovered_dwords += dwords - covered;
    if (covered < dwords && g_samples.size() < MaxSamples &&
        g_samples_per_kind[queue << 16 | kind]++ < MaxSamplesPerKind) {
        UncoveredSample& sample = g_samples.emplace_back();
        sample.queue = queue;
        sample.kind = kind;
        sample.dwords = dwords;
        sample.covered = covered;
        sample.address = address;
        sample.owner_before = OwnerAt(address - 4);
        sample.owner_after = OwnerAt(address + u64{dwords} * 4);
        for (u32 i = 0; i < 4; i++) {
            sample.first[i] = i < dwords ? packet[i] : 0;
        }
        sample.seconds = Seconds();
    }
}

// ---------------------------------------------------------------------------------------------
// The report.

void WriteReport(std::FILE* file) {
    std::fprintf(file, "gnm_recorder: frontier check after %.0f s\n\n", Seconds());

    std::scoped_lock lock{g_gpu_mutex};
    for (u32 queue = 0; queue < 3; queue++) {
        PacketStats total;
        std::map<u8, PacketStats> by_owner;
        std::map<u32, PacketStats> uncovered_by_kind;
        for (const auto& [key, stats] : g_packets) {
            if (key >> 24 != queue) {
                continue;
            }
            total.packets += stats.packets;
            total.dwords += stats.dwords;
            total.uncovered_dwords += stats.uncovered_dwords;
            PacketStats& owner = by_owner[key & 0xFF];
            owner.packets += stats.packets;
            owner.dwords += stats.dwords;
            owner.uncovered_dwords += stats.uncovered_dwords;
            if (stats.uncovered_dwords) {
                PacketStats& kind = uncovered_by_kind[(key >> 8) & 0xFFFF];
                kind.packets += stats.packets;
                kind.uncovered_dwords += stats.uncovered_dwords;
            }
        }
        if (total.packets == 0) {
            continue;
        }
        std::fprintf(file,
                     "== %s queue: %llu packets, %llu dwords executed; %llu dwords in game "
                     "memory not written by a frontier function (%.4f%%)\n",
                     QueueName(queue), static_cast<unsigned long long>(total.packets),
                     static_cast<unsigned long long>(total.dwords),
                     static_cast<unsigned long long>(total.uncovered_dwords),
                     100.0 * total.uncovered_dwords / total.dwords);
        std::vector<std::pair<u64, u8>> owners;
        for (const auto& [owner, stats] : by_owner) {
            owners.emplace_back(stats.dwords, owner);
        }
        std::ranges::sort(owners, std::greater{});
        std::fprintf(file, "  by the function that wrote the packet header (dwords, packets: types):\n");
        for (const auto& [dwords, owner] : owners) {
            std::string kinds;
            std::vector<std::pair<u64, u32>> list;
            for (const auto& [key, stats] : g_packets) {
                if (key >> 24 == queue && (key & 0xFF) == owner) {
                    list.emplace_back(stats.packets, (key >> 8) & 0xFFFF);
                }
            }
            std::ranges::sort(list, std::greater{});
            for (size_t i = 0; i < list.size() && i < 8; i++) {
                kinds += std::format(" {} {}", PacketName(list[i].second), list[i].first);
            }
            if (list.size() > 8) {
                kinds += " …";
            }
            std::fprintf(file, "    %-36s %12llu %10llu:%s\n", OwnerName(owner),
                         static_cast<unsigned long long>(dwords),
                         static_cast<unsigned long long>(by_owner[owner].packets), kinds.c_str());
        }
        if (!uncovered_by_kind.empty()) {
            std::fprintf(file, "  not covered, by packet type (packets of that type, uncovered dwords):\n");
            for (const auto& [kind, stats] : uncovered_by_kind) {
                std::fprintf(file, "    %-28s %10llu %10llu\n", PacketName(kind).c_str(),
                             static_cast<unsigned long long>(stats.packets),
                             static_cast<unsigned long long>(stats.uncovered_dwords));
            }
        }
        std::fprintf(file, "\n");
    }

    if (!g_samples.empty()) {
        std::fprintf(file, "== samples of packets not (fully) covered\n");
        for (const UncoveredSample& sample : g_samples) {
            std::fprintf(file,
                         "  %7.1f s %-8s %-24s at %#llx, %u dwords (%u covered); before: %s, "
                         "after: %s; %08x %08x %08x %08x\n",
                         sample.seconds, QueueName(sample.queue), PacketName(sample.kind).c_str(),
                         static_cast<unsigned long long>(sample.address), sample.dwords,
                         sample.covered, OwnerName(sample.owner_before),
                         OwnerName(sample.owner_after), sample.first[0], sample.first[1],
                         sample.first[2], sample.first[3]);
        }
        std::fprintf(file, "\n");
    }

    {
        std::scoped_lock proof_lock{g_proof_mutex};
        std::fprintf(file,
                     "== state model vs the GPU's registers at each draw (graphics registers) "
                     "and dispatch (compute registers): %llu draws and %llu dispatches checked, "
                     "%llu without a snapshot\n   register: compared, mismatched, unknown to "
                     "the model (GPU values then); samples model/GPU. User data the shader is not "
                     "given (USER_SGPR) are not compared.\n",
                     static_cast<unsigned long long>(g_draws_checked),
                     static_cast<unsigned long long>(g_dispatches_checked),
                     static_cast<unsigned long long>(g_draws_without_snapshot));
        size_t proven = 0;
        for (const RegisterProof& proof : g_proof) {
            proven += proof.compared > 0 && proof.mismatched == 0 && proof.unknown == 0 &&
                      proof.excluded == 0;
        }
        std::fprintf(file, "   %zu of %zu registers matched at every draw or dispatch; the others:\n", proven,
                     render::ModelledCount);
        for (size_t i = 0; i < render::ModelledCount; i++) {
            const RegisterProof& proof = g_proof[i];
            if (proof.compared > 0 && proof.mismatched == 0 && proof.unknown == 0 &&
                proof.excluded == 0) {
                continue;
            }
            std::string samples;
            for (const auto& [model, gpu_value] : proof.samples) {
                samples += std::format(" {:08x}/{:08x}", model, gpu_value);
            }
            if (proof.excluded) {
                samples += std::format("  excluded (not modelled yet): {}", proof.excluded);
            }
            if (proof.unused) {
                samples += std::format("  not given to the shader: {}", proof.unused);
            }
            if (!proof.unknown_values.empty()) {
                samples += "  unknown:";
                for (const auto& [value, count] : proof.unknown_values) {
                    samples += std::format(" {:08x} x{}", value, count);
                }
            }
            std::fprintf(file, "  %-32s %12llu %10llu %12llu%s\n",
                         render::ModelledRegisters[i].name,
                         static_cast<unsigned long long>(proof.compared),
                         static_cast<unsigned long long>(proof.mismatched),
                         static_cast<unsigned long long>(proof.unknown), samples.c_str());
        }
        std::fprintf(file, "   resource table copies behind the pointer SGPRs (draws; pointers "
                           "disagreeing; no copy in the model; unreadable; dwords compared, "
                           "mismatched, not known to the model; samples dword: model/GPU):\n");
        for (u32 stage = 0; stage < render::ModelledStages; stage++) {
            const TableProof& proof = g_table_proof[stage];
            std::string samples;
            for (const auto& [dword, model, gpu_value] : proof.samples) {
                samples += std::format(" {}: {:08x}/{:08x}", dword, model, gpu_value);
            }
            if (!proof.unknown_dwords.empty()) {
                samples += "  unknown dwords (index x draws):";
                for (const auto& [dword, count] : proof.unknown_dwords) {
                    samples += std::format(" {} x{} (last {:08x})", dword, count,
                                           proof.unknown_last.at(dword));
                }
            }
            std::fprintf(file, "  %s  %llu  %llu  %llu  %llu  %llu %llu %llu%s\n",
                         std::array{"CS", "PS", "VS"}[stage],
                         static_cast<unsigned long long>(proof.draws),
                         static_cast<unsigned long long>(proof.inconsistent),
                         static_cast<unsigned long long>(proof.no_table),
                         static_cast<unsigned long long>(proof.unreadable),
                         static_cast<unsigned long long>(proof.dwords_compared),
                         static_cast<unsigned long long>(proof.dwords_mismatched),
                         static_cast<unsigned long long>(proof.dwords_unknown), samples.c_str());
        }
        std::fprintf(file, "\n");
    }

    std::fprintf(file, "== frontier functions: calls, dwords written, bad buffer object, buffer "
                       "switches (full buffer), cursor moved too far\n");
    for (size_t i = 0; i < FunctionCount; i++) {
        const FunctionStats& stats = g_stats[i];
        std::fprintf(file, "  %#8llx %-38s %12llu %12llu %8llu %8llu %8llu\n",
                     static_cast<unsigned long long>(Functions[i].address), Functions[i].name,
                     static_cast<unsigned long long>(stats.calls.load()),
                     static_cast<unsigned long long>(stats.dwords.load()),
                     static_cast<unsigned long long>(stats.bad_buffer.load()),
                     static_cast<unsigned long long>(stats.buffer_switches.load()),
                     static_cast<unsigned long long>(stats.too_far.load()));
    }
}

const char* StageName(u32 stage) {
    switch (stage) {
    case StageCompute: return "cs";
    case StagePixel: return "ps";
    case StageVertex: return "vs";
    default: return "?";
    }
}

std::string DescribeVariant(const VariantKey& key) {
    std::string text;
    for (size_t i = 0; i + 1 < key.parts.size(); i += 2) {
        const u32 slot = key.parts[i] >> 24;
        const u32 kind = (key.parts[i] >> 20) & 0xF;
        const u32 value = key.parts[i + 1];
        if (kind == 2) {
            text += std::format(" rt{}={}/{}", slot, gnm::DataFormatName((value >> 8) & 0xFF),
                                gnm::ColorNumberTypeName(value & 0xFF));
        } else {
            text += std::format(" {}{}={}", kind == 0 ? "t" : "u", slot,
                                value == NotSet ? std::string{"unset"} : DescribeTextureKey(value));
        }
    }
    return text.empty() ? " (no textures)" : text;
}

// gnm_shaders.txt (to read) and gnm_variants.tsv (for tools: stage, hash, uses, then the key as
// hexadecimal pairs slot<<24|kind<<20, value; kind 0 texture, 1 writable texture, 2 render
// target).
void WriteShaderReports() {
    std::map<VariantKey, u64> variants;
    {
        std::scoped_lock lock{g_variants_mutex};
        variants = g_variants;
    }
    struct Shader {
        u32 stage;
        u64 uses = 0;
        std::vector<std::pair<u64, const VariantKey*>> variants;
    };
    std::map<u64, Shader> shaders;
    for (const auto& [key, uses] : variants) {
        Shader& shader = shaders[key.hash];
        shader.stage = key.stage;
        shader.uses += uses;
        shader.variants.emplace_back(uses, &key);
    }
    std::array<u64, 3> per_stage{};
    std::map<size_t, u64> histogram;
    for (auto& [hash, shader] : shaders) {
        per_stage[std::min<u32>(shader.stage, 2)]++;
        histogram[std::min<size_t>(shader.variants.size(), 4)]++;
        std::ranges::sort(shader.variants, std::greater{});
    }

    const auto text_path = DataPath("gnm_shaders.txt");
    if (std::FILE* file = std::fopen((text_path.string() + ".tmp").c_str(), "w")) {
        std::fprintf(file,
                     "shader specializations after %.0f s: %zu shaders (cs %llu, ps %llu, vs %llu), "
                     "%zu variants\n",
                     Seconds(), shaders.size(), static_cast<unsigned long long>(per_stage[0]),
                     static_cast<unsigned long long>(per_stage[1]),
                     static_cast<unsigned long long>(per_stage[2]), variants.size());
        std::fprintf(file, "shaders with 1 / 2 / 3 / 4+ variants:");
        for (size_t count = 1; count <= 4; count++) {
            std::fprintf(file, " %llu", static_cast<unsigned long long>(histogram[count]));
        }
        std::fprintf(file, "\n(variant = kinds of the textures and buffers in the slots the shader "
                           "uses, and for pixel shaders the render target formats; resources a "
                           "shader reads through its SRT are not seen here)\n\n");
        std::vector<std::pair<u64, u64>> order;
        for (const auto& [hash, shader] : shaders) {
            order.emplace_back(shader.variants.size() * 1'000'000'000'000ull + shader.uses, hash);
        }
        std::ranges::sort(order, std::greater{});
        for (const auto& [_, hash] : order) {
            const Shader& shader = shaders[hash];
            std::fprintf(file, "%s %016llx  uses %llu, %zu variants\n", StageName(shader.stage),
                         static_cast<unsigned long long>(hash),
                         static_cast<unsigned long long>(shader.uses), shader.variants.size());
            for (size_t i = 0; i < shader.variants.size() && i < 8; i++) {
                std::fprintf(file, "  %10llu%s\n",
                             static_cast<unsigned long long>(shader.variants[i].first),
                             DescribeVariant(*shader.variants[i].second).c_str());
            }
        }
        std::fclose(file);
        std::filesystem::rename(text_path.string() + ".tmp", text_path);
    }

    const auto table_path = DataPath("gnm_variants.tsv");
    if (std::FILE* file = std::fopen((table_path.string() + ".tmp").c_str(), "w")) {
        for (const auto& [key, uses] : variants) {
            std::fprintf(file, "%s\t%016llx\t%llu\t", StageName(key.stage),
                         static_cast<unsigned long long>(key.hash),
                         static_cast<unsigned long long>(uses));
            for (const u32 part : key.parts) {
                std::fprintf(file, " %08x", part);
            }
            std::fprintf(file, "\n");
        }
        std::fclose(file);
        std::filesystem::rename(table_path.string() + ".tmp", table_path);
    }
}

void ReportLoop() {
    const auto path = DataPath("gnm_recorder.txt");
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds{20});
        const auto temporary = path.string() + ".tmp";
        if (std::FILE* file = std::fopen(temporary.c_str(), "w")) {
            WriteReport(file);
            if (NativeValidationEnabled()) {
                WriteNativeValidationReport(file);
            }
            if (g_survey_enabled) {
                WriteSurveyReport(file);
            }
            std::fclose(file);
            std::filesystem::rename(temporary, path);
        }
        WriteShaderReports();
    }
}

} // namespace

void InstallGnmRecorder() {
    const char* value = std::getenv("SOTC_RECORD_GNM");
    if (!value || !*value || *value == '0') {
        return;
    }
    if (std::getenv("SOTC_TRACE_GNM")) {
        LOG_ERROR("Recorder", "SOTC_RECORD_GNM and SOTC_TRACE_GNM watch the same functions; "
                              "use one at a time");
        return;
    }
    if (const char* times = std::getenv("SOTC_RECORD_FRAMES")) {
        for (const char* p = times; *p;) {
            char* end;
            const double seconds = std::strtod(p, &end);
            if (end == p) {
                break;
            }
            g_record_times.push_back(seconds);
            p = *end == ',' ? end + 1 : end;
        }
        std::ranges::sort(g_record_times, std::greater{});
    }
    std::array<u64, FunctionCount> addresses;
    for (size_t i = 0; i < FunctionCount; i++) {
        addresses[i] = Functions[i].address;
    }
    const std::vector<bool> hooked = ObserveCalls(addresses, &g_observer);
    for (size_t i = 0; i < FunctionCount; i++) {
        if (!hooked[i]) {
            LOG_ERROR("Recorder", "could not hook {} ({:#x})", Functions[i].name,
                      Functions[i].address);
        }
    }
    if (const char* survey = std::getenv("SOTC_SURVEY_GNM"); survey && *survey == '1') {
        g_survey_enabled = true;
        g_survey_hooked = ObserveCalls(SurveyFunctions, &g_survey_observer);
        LOG_INFO("Recorder", "survey: watching {} of {} other Sony functions",
                 std::ranges::count(g_survey_hooked, true), SurveyCount);
    }
    gpu::SetPacketObserver(OnPacket);
    std::thread{ReportLoop}.detach();
    LOG_INFO("Recorder", "watching {} frontier functions; report in {}",
             std::ranges::count(hooked, true), DataPath("gnm_recorder.txt").string());
}

} // namespace diag
