#include "render/state_model.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <string>

#include "hle/kernel/gpu_memory.h"

#include "render/frontier.h"

namespace render {

namespace {

// Position of each modelled register in RegisterSnapshot (or -1), indexed by register index.
// Built once; the register file indices fit below 0xD000.
const std::array<s16, 0xD000>& SlotTable() {
    static const auto table = [] {
        std::array<s16, 0xD000> slots;
        slots.fill(-1);
        for (size_t i = 0; i < ModelledCount; i++) {
            slots[ModelledRegisters[i].index] = static_cast<s16>(i);
        }
        return slots;
    }();
    return table;
}

u32 Low32(u64 value) {
    return static_cast<u32>(value);
}

// Copies `size` bytes of the game's memory (an object the game passed), if all of it is mapped.
bool ReadGame(u64 address, void* out, size_t size) {
    if (address < 0x10000 || hle::kernel::ClampToMapped(address, size) < size) {
        return false;
    }
    std::memcpy(out, reinterpret_cast<const void*>(address), size);
    return true;
}

// setPolygonOffsetZFormat looks its register value up in a table of the executable
// (0x168fee0), indexed by the depth format: no offset / 16-bit / 24-bit / 32-bit float depth.
constexpr std::array<u32, 4> PolygonOffsetFormats = {0x0, 0xf0, 0xe8, 0x1e9};

// The modelled registers after initializeDefaultHardwareState (sceGnmDrawInitDefaultHardwareState350:
// CLEAR_STATE, then the driver's default sequence; values as the GPU layer sets them,
// third_party/shadps4 regs.cpp SetDefaults and gnmdriver_init.h InitSequence350). The GPU
// layer's CLEAR_STATE zeroes the whole register file, uconfig included (the real GPU's resets
// only context registers). Registers not listed are left unknown: VGT_PRIMITIVE_TYPE (always set
// before drawing) and CB_TARGET_MASK (a placeholder in the GPU layer, 0xdeadbeef).
struct DefaultValue {
    u32 index;
    u32 value;
};
constexpr u32 C = ContextBase;
constexpr DefaultValue DefaultsAfterInit[] = {
    {C + 0x000, 0},          {C + 0x003, 0},          {C + 0x00b, 0},
    // screen viewport (setupScreenViewport): CLEAR_STATE values, VTE from the default sequence
    {C + 0x00c, 0},          {C + 0x00d, 0x40004000}, {C + 0x0b4, 0},
    {C + 0x0b5, 0x3f800000}, {C + 0x10f, 0},          {C + 0x110, 0},
    {C + 0x111, 0},          {C + 0x112, 0},          {C + 0x113, 0},
    {C + 0x114, 0},          {C + 0x206, 0x43f},
    // shader outputs/inputs (from the shaders): CLEAR_STATE values, SPI_VS_OUT_CONFIG from the
    // default sequence; CB_SHADER_MASK stays unknown (a placeholder in the GPU layer)
    {C + 0x1b1, 2},          {C + 0x1b3, 0},          {C + 0x1b4, 0},
    {C + 0x1b6, 2},          {C + 0x1b8, 0},          {C + 0x1c3, 0},
    {C + 0x1c4, 0},          {C + 0x1c5, 0},          {C + 0x203, 0},
    {C + 0x207, 0},
    // Shader program registers: the GPU layer's CLEAR_STATE zeroes them too (the real GPU's
    // leaves SH registers alone). Only visible in a stage without a program (PGM_LO = 0), where
    // the values do not matter; followed so the proof stays exact.
    {ShBase + 0x08, 0},      {ShBase + 0x09, 0},      {ShBase + 0x0a, 0},
    {ShBase + 0x0b, 0},      {ShBase + 0x48, 0},      {ShBase + 0x49, 0},
    {ShBase + 0x4a, 0},      {ShBase + 0x4b, 0},
    // ... and the user data (SGPRs) of every stage (zeroed in InitializeDefaultHardwareState
    // below). The game sets the ones a shader reads before drawing with it, so only stale values
    // of earlier shaders are lost. Checked with the recorder (2026-10-10): the only shaders given
    // an SGPR last written by a CLEAR_STATE (VS 0xddd376ab and 0x4acdff26, s2 of 3) never read it.
    // index buffer and instancing: zeroed, then NUM_INSTANCES = 1 by the default sequence
    {C + 0x1f9, 0},          {C + 0x1fa, 0},          {C + 0x1fc, 0},
    {C + 0x29e, 0},          {C + 0x29f, 0},          {UconfigBase + 0x24c, 0},
    {UconfigBase + 0x24d, 1},
    // depth target: CLEAR_STATE zeroes them
    {C + 0x002, 0},          {C + 0x005, 0},          {C + 0x00f, 0},
    {C + 0x010, 0},          {C + 0x011, 0},          {C + 0x012, 0},
    {C + 0x013, 0},          {C + 0x014, 0},          {C + 0x015, 0},
    {C + 0x016, 0},          {C + 0x017, 0},          {C + 0x2af, 0},
    {C + 0x020, 0},          {C + 0x094, 0x80000000}, {C + 0x095, 0x40004000},
    {C + 0x10b, 0},          {C + 0x10c, 0},          {C + 0x10d, 0},
    {C + 0x1e0, 0},          {C + 0x1e1, 0},          {C + 0x1e2, 0},
    {C + 0x1e3, 0},          {C + 0x1e4, 0},          {C + 0x1e5, 0},
    {C + 0x1e6, 0},          {C + 0x1e7, 0},          {C + 0x200, 0},
    {C + 0x202, 0x00cc0010}, {C + 0x204, 0},          {C + 0x205, 4},
    {C + 0x292, 0},          {C + 0x2de, 0x1e9},      {C + 0x2e0, 0},
    {C + 0x2e1, 0},          {C + 0x2e2, 0},          {C + 0x2e3, 0},
};

} // namespace

namespace {

// The graphics LCUE (context + 0x2e8) keeps a pointer to its command buffer at +0xc090, the
// compute LCUE (compute context + 0x50) at +0x90.
u64 LcueCommandBuffer(u64 lcue, bool compute_lcue = false) {
    u64 buffer = 0;
    ReadGame(lcue + (compute_lcue ? 0x90 : 0xc090), &buffer, sizeof(buffer));
    return buffer;
}

} // namespace

const char* ColorTargetRegisterName(u32 slot, u32 field) {
    // Built once: "CB_COLOR0_BASE" ... "CB_COLOR7_DCC_BASE".
    static const auto names = [] {
        std::array<std::string, 8 * ColorTargetRegisters> all;
        for (u32 s = 0; s < 8; s++) {
            for (u32 f = 0; f < ColorTargetRegisters; f++) {
                all[s * ColorTargetRegisters + f] =
                    "CB_COLOR" + std::to_string(s) + "_" + ColorTargetRegisterNames[f];
            }
        }
        return all;
    }();
    return names[slot * ColorTargetRegisters + field].c_str();
}

const char* UserDataRegisterName(u32 stage, u32 sgpr) {
    static const auto names = [] {
        std::array<std::string, 7 * UserDataCount> all;
        constexpr const char* stages[] = {"CS", "PS", "VS", "GS", "ES", "HS", "LS"};
        for (u32 s = 0; s < 7; s++) {
            for (u32 n = 0; n < UserDataCount; n++) {
                all[s * UserDataCount + n] =
                    std::string{"SPI_SHADER_USER_DATA_"} + stages[s] + "_" + std::to_string(n);
            }
        }
        return all;
    }();
    return names[stage * UserDataCount + sgpr].c_str();
}

void StateModel::SetResources(u64 lcue, u32 stage, u32 slot, u32 count, u64 descriptors,
                              u32 array_offset, u32 dwords, bool compute_lcue) {
    const u64 table = StageTable(lcue, stage);
    const u64 target = LcueCommandBuffer(lcue, compute_lcue);
    if (!table || !target || slot >= 16 || count > 16) {
        return;
    }
    std::array<u16, 16> slots;
    if (!ReadGame(table + array_offset, slots.data(), sizeof(slots))) {
        return;
    }
    for (u32 i = 0; i < count && slot + i < 16; i++) {
        const u16 entry = slots[slot + i];
        // 0xFFFF: unused; bit 15 clear: into the LCUE's resource table of the stage at dword
        // bits 0-13 (copied to the ring by the pre-draw); bit 15 set: straight into the SGPRs
        // from bits 0-13 on.
        if (entry == 0xFFFF) {
            continue;
        }
        const u32 position = entry & 0x3FFF;
        std::array<u32, 8> descriptor;
        if (!ReadGame(descriptors + u64{i} * dwords * 4, descriptor.data(), dwords * 4)) {
            continue;
        }
        if (!(entry & 0x8000)) {
            if (stage >= ModelledStages) {
                continue;
            }
            Shard& shard = ShardOf(lcue);
            std::scoped_lock lock{shard.mutex};
            LcueState::TableImage& image = shard.lcues[lcue].images[stage];
            for (u32 d = 0; d < dwords && position + d < image.values.size(); d++) {
                image.values[position + d] = descriptor[d];
                image.known.set(position + d);
            }
            continue;
        }
        for (u32 d = 0; d < dwords && position + d < UserDataCount; d++) {
            Set(target, UserDataBase[stage] + position + d, descriptor[d]);
        }
    }
}

const char* PsInputRegisterName(u32 input) {
    static const auto names = [] {
        std::array<std::string, PsInputCount> all;
        for (u32 n = 0; n < PsInputCount; n++) {
            all[n] = "SPI_PS_INPUT_CNTL_" + std::to_string(n);
        }
        return all;
    }();
    return names[input].c_str();
}


StateModel::Shard& StateModel::ShardOf(u64 buffer) {
    return shards_[(buffer >> 6 ^ buffer >> 14) % ShardCount];
}

void StateModel::Set(u64 buffer, u32 index, u32 value) {
    const s16 slot = SlotTable()[index];
    if (slot < 0) {
        return;
    }
    Shard& shard = ShardOf(buffer);
    std::scoped_lock lock{shard.mutex};
    RegisterSnapshot& state = shard.buffers[buffer];
    state.values[slot] = value;
    state.known.set(slot);
    state.excluded.reset(slot);
    state.table_pointer.reset(slot);
}

void StateModel::SetTablePointer(u64 buffer, u32 index, u32 offset) {
    const s16 slot = SlotTable()[index];
    if (slot < 0) {
        return;
    }
    Shard& shard = ShardOf(buffer);
    std::scoped_lock lock{shard.mutex};
    RegisterSnapshot& state = shard.buffers[buffer];
    state.values[slot] = offset;
    state.known.set(slot);
    state.excluded.reset(slot);
    state.table_pointer.set(slot);
}

void StateModel::Exclude(u64 buffer, u32 index) {
    const s16 slot = SlotTable()[index];
    if (slot < 0) {
        return;
    }
    Shard& shard = ShardOf(buffer);
    std::scoped_lock lock{shard.mutex};
    RegisterSnapshot& state = shard.buffers[buffer];
    state.known.reset(slot);
    state.excluded.set(slot);
    state.table_pointer.reset(slot);
}

u64 StateModel::StageTable(u64 lcue, u32 stage) {
    Shard& shard = ShardOf(lcue);
    std::scoped_lock lock{shard.mutex};
    const auto it = shard.lcues.find(lcue);
    return it != shard.lcues.end() && stage < 7 ? it->second.tables[stage] : 0;
}

void StateModel::SetStageTable(u64 lcue, u32 stage, u64 table) {
    if (stage >= 7) {
        return;
    }
    Shard& shard = ShardOf(lcue);
    std::scoped_lock lock{shard.mutex};
    shard.lcues[lcue].tables[stage] = table;
}

RegisterSnapshot StateModel::Submit(u64 buffer, u64 compute_queue) {
    const RegisterSnapshot own = Snapshot(buffer);
    std::scoped_lock lock{queue_mutex_};
    RegisterSnapshot& queue_ = compute_queue ? compute_queues_[compute_queue] : this->queue_;
    const RegisterSnapshot inherited = queue_;
    for (size_t i = 0; i < ModelledCount; i++) {
        if (own.known.test(i)) {
            queue_.values[i] = own.values[i];
            queue_.known.set(i);
            queue_.excluded.reset(i);
            queue_.table_pointer.set(i, own.table_pointer.test(i));
        } else if (own.excluded.test(i)) {
            queue_.known.reset(i);
            queue_.excluded.set(i);
            queue_.table_pointer.reset(i);
        }
    }
    for (size_t stage = 0; stage < own.tables.size(); stage++) {
        if (own.tables[stage]) {
            queue_.tables[stage] = own.tables[stage];
        }
    }
    return inherited;
}

void StateModel::Inherit(RegisterSnapshot& snapshot, const RegisterSnapshot& inherited) {
    for (size_t i = 0; i < ModelledCount; i++) {
        if (snapshot.known.test(i) || snapshot.excluded.test(i)) {
            continue;
        }
        if (inherited.known.test(i)) {
            snapshot.values[i] = inherited.values[i];
            snapshot.known.set(i);
            snapshot.table_pointer.set(i, inherited.table_pointer.test(i));
        } else if (inherited.excluded.test(i)) {
            snapshot.excluded.set(i);
        }
    }
    for (size_t stage = 0; stage < snapshot.tables.size(); stage++) {
        if (!snapshot.tables[stage]) {
            snapshot.tables[stage] = inherited.tables[stage];
        }
    }
}

RegisterSnapshot StateModel::Snapshot(u64 buffer) {
    Shard& shard = ShardOf(buffer);
    std::scoped_lock lock{shard.mutex};
    const auto it = shard.buffers.find(buffer);
    return it != shard.buffers.end() ? it->second : RegisterSnapshot{};
}

// The pre-draw (0xc5fc10), for the PS and VS. For each stage whose state changed since the last
// draw it copies the bound shader's part of the LCUE's resource table to fresh memory (its size
// is the resource offsets table's first u16, in dwords) and writes SGPRs; each byte field of the
// resource offsets table names the first SGPR (0xFF: none):
//   +0x04 fetch shader address (VS)          +0x05 vertex buffers: copy + u16 [+0x18] dwords (VS)
//   +0x06 stream-out: copy + u16 [+0x20] (VS) +0x07 the copy itself
//   +0x08 constant buffers: copy + u16 [+0x16] +0x09 textures/buffers: copy + u16 [+0x1a]
//   +0x0a RW resources: copy + u16 [+0x1c]   +0x0b samplers: copy + u16 [+0x1e]
//   +0x0c the LCUE's global table (lcue + 0x78)
//   +0x0d, +0x0e one dword each (lcue + 0xc0e0 / 0xc100 + stage * 4); +0x0f one computed
//   dword (VS: the product of the two u16 at lcue + 0xc5e8)
// Pointers take 2 SGPRs (low, high). Every write to the table image marks the stage as changed,
// so whatever the GPU's SGPRs point at holds the image as it is now: the model does not need
// the LCUE's "changed" flags. The pointers and their table copy are proven by content (the
// address is chosen per draw); +0x0c to +0x0f are not modelled yet and stay excluded.
void StateModel::PreDraw(u64 lcue, bool compute_lcue, std::initializer_list<u32> stages) {
    struct PointerField {
        u32 field;  // byte offset of the SGPR number in the resource offsets table
        u32 offset; // byte offset of the u16 dword offset into the copy (0: the copy itself)
        bool vs_only;
    };
    constexpr PointerField pointers[] = {
        {0x05, 0x18, true}, {0x06, 0x20, true}, {0x07, 0, false}, {0x08, 0x16, false},
        {0x09, 0x1a, false}, {0x0a, 0x1c, false}, {0x0b, 0x1e, false},
    };
    constexpr std::pair<u32, u32> excluded_fields[] = {{0x0c, 2}, {0x0d, 1}, {0x0e, 1}, {0x0f, 1}};
    const u64 target = LcueCommandBuffer(lcue, compute_lcue);
    if (!target) {
        return;
    }
    for (u32 stage : stages) {
        const u64 table = StageTable(lcue, stage);
        std::array<u8, 0x22> header;
        if (!table || !ReadGame(table, header.data(), sizeof(header))) {
            continue;
        }
        auto u16_at = [&](u32 offset) { return u32{header[offset]} | u32{header[offset + 1]} << 8; };
        const u32 base = UserDataBase[stage];
        for (const PointerField& pointer : pointers) {
            const u32 sgpr = header[pointer.field];
            if (sgpr == 0xFF || sgpr + 1 >= UserDataCount || (pointer.vs_only && stage != 2)) {
                continue;
            }
            const u32 offset = pointer.offset ? u16_at(pointer.offset) : 0;
            SetTablePointer(target, base + sgpr, offset);
            SetTablePointer(target, base + sgpr + 1, offset | TablePointerHigh);
        }
        if (const u32 sgpr = header[0x04]; stage == 2 && sgpr != 0xFF && sgpr + 1 < UserDataCount) {
            u64 fetch_shader = 0;
            {
                Shard& shard = ShardOf(lcue);
                std::scoped_lock lock{shard.mutex};
                fetch_shader = shard.lcues[lcue].fetch_shader;
            }
            Set(target, base + sgpr, static_cast<u32>(fetch_shader));
            Set(target, base + sgpr + 1, static_cast<u32>(fetch_shader >> 32));
        }
        for (const auto& [field, sgprs] : excluded_fields) {
            const u32 sgpr = header[field];
            if (field == 0x0f && stage != 2) {
                continue;
            }
            for (u32 d = 0; d < sgprs && sgpr != 0xFF && sgpr + d < UserDataCount; d++) {
                Exclude(target, base + sgpr + d);
            }
        }
        // The copy, as the model knows the image now.
        auto copy = std::make_shared<ResourceTableCopy>();
        const u32 size = std::min<u32>(u16_at(0), 0x600);
        copy->values.resize(size);
        copy->known.resize(size);
        {
            Shard& shard = ShardOf(lcue);
            std::scoped_lock lock{shard.mutex};
            const LcueState::TableImage& image = shard.lcues[lcue].images[stage];
            for (u32 i = 0; i < size; i++) {
                copy->values[i] = image.values[i];
                copy->known[i] = image.known.test(i);
            }
        }
        Shard& shard = ShardOf(target);
        std::scoped_lock lock{shard.mutex};
        shard.buffers[target].tables[stage] = std::move(copy);
    }
}

// sceGnmSetCsShaderWithModifier with modifier 0, as both pre-dispatches call it: the Gnmx
// CsShader's registers from dword 2 on are PGM_LO, PGM_HI (must be 0), RSRC1, RSRC2,
// NUM_THREAD_X/Y/Z. Modelled when the shader is set: the pre-dispatch writes them only when it
// changed, so the values are the same.
void StateModel::SetCsProgram(u64 target, u64 shader) {
    std::array<u32, 7> regs{};
    if (!target || !shader || !ReadGame(shader + 8, regs.data(), sizeof(regs))) {
        return;
    }
    Set(target, ShBase + 0x20c, regs[0]); // COMPUTE_PGM_LO
    Set(target, ShBase + 0x20d, 0);       // COMPUTE_PGM_HI
    Set(target, ShBase + 0x212, regs[2]); // COMPUTE_PGM_RSRC1
    Set(target, ShBase + 0x213, regs[3]); // COMPUTE_PGM_RSRC2
    Set(target, ShBase + 0x207, regs[4]); // COMPUTE_NUM_THREAD_X
    Set(target, ShBase + 0x208, regs[5]); // COMPUTE_NUM_THREAD_Y
    Set(target, ShBase + 0x209, regs[6]); // COMPUTE_NUM_THREAD_Z
}

void StateModel::Apply(const FrontierCall& call) {
    namespace fn = frontier;
    const auto& a = call.args;
    const u64 buffer = a[0];
    const u32 c = ContextBase;
    switch (call.function) {
    case fn::InitializeContext: {
        // (context = its command buffer, memory, size, ...): a new command buffer, which
        // starts with the default state.
        {
            Shard& shard = ShardOf(buffer);
            std::scoped_lock lock{shard.mutex};
            shard.buffers.erase(buffer);
        }
        // Its LCUE (context + 0x2e8) starts over too; the memory of its resource tables keeps
        // whatever was there, which the model does not know.
        Shard& shard = ShardOf(buffer + 0x2e8);
        std::scoped_lock lock{shard.mutex};
        shard.lcues.erase(buffer + 0x2e8);
        break;
    }
    case fn::InitializeDefaultHardwareState: // (buffer, ...): resets the GPU state
        for (const DefaultValue& reset : DefaultsAfterInit) {
            Set(buffer, reset.index, reset.value);
        }
        // CLEAR_STATE zeroes the color target and pixel shader input registers; the default
        // sequence leaves them.
        for (u32 i = 0; i < 8 * ColorTargetStride; i++) {
            Set(buffer, c + ColorTargetBase + i, 0);
        }
        for (u32 i = 0; i < PsInputCount; i++) {
            Set(buffer, c + PsInputBase + i, 0);
        }
        Set(buffer, c + 0x2d5, 0);
        for (u32 i = 0; i < UserDataCount; i++) {
            Set(buffer, UserDataBase[StagePixel] + i, 0);
            Set(buffer, UserDataBase[StageVertex] + i, 0);
        }
        // The default sequence (InitSequence350) also sets these compute registers of the
        // graphics queue; CLEAR_STATE leaves the compute registers alone.
        Set(buffer, ShBase + 0x216, 0xffffffff); // COMPUTE_STATIC_THREAD_MGMT_SE0
        Set(buffer, ShBase + 0x217, 0xffffffff); // COMPUTE_STATIC_THREAD_MGMT_SE1
        Set(buffer, ShBase + 0x215, 0);          // COMPUTE_RESOURCE_LIMITS
        break;
    // Each case: the method's arguments -> register value, as its code computes it.
    // --- index buffer, instancing, indirect arguments ----------------------------------------
    case fn::SetIndexBuffer: // (buffer, address): INDEX_BASE
        Set(buffer, c + 0x1fa, Low32(a[1]) & ~1u);       // VGT_DMA_BASE
        Set(buffer, c + 0x1f9, (a[1] >> 32) & 0xff);     // VGT_DMA_BASE_HI
        break;
    case fn::SetIndexCount: // (buffer, count): INDEX_BUFFER_SIZE
        Set(buffer, UconfigBase + 0x24c, Low32(a[1])); // VGT_NUM_INDICES
        break;
    case fn::SetIndexSize: { // (buffer, size, cache policy): INDEX_TYPE
        const u32 policy = Low32(a[2]);
        const u32 policy_bits = policy != 2 ? (policy & 3) << 6 | 1u << 10 : 0;
        Set(buffer, c + 0x29f, (Low32(a[1]) & 0xfffffb3f) | policy_bits); // VGT_DMA_INDEX_TYPE
        break;
    }
    case fn::SetNumInstances: // (buffer, count): NUM_INSTANCES
        Set(buffer, UconfigBase + 0x24d, Low32(a[1])); // VGT_NUM_INSTANCES
        break;
    case fn::SetBaseIndirectArgs: // (buffer, shader type, address): SET_BASE, base index 1
        Set(buffer, IndirectArgsBase, Low32(a[2]) & ~7u);
        Set(buffer, IndirectArgsBase + 1, (a[2] >> 32) & 0xff);
        break;
    // Draws (applied after their snapshot): the packet's fields the GPU layer keeps.
    case fn::DrawIndexOffset: // (buffer, index offset, count, modifier): DRAW_INDEX_OFFSET_2
        Set(buffer, c + 0x29e, Low32(a[2]));          // VGT_DMA_MAX_SIZE (= the count)
        Set(buffer, UconfigBase + 0x24c, Low32(a[2])); // VGT_NUM_INDICES
        Set(buffer, c + 0x1fc, 0);                    // VGT_DRAW_INITIATOR (non-Pro mode)
        break;
    case fn::DrawIndexAuto: // (buffer, count, modifier): DRAW_INDEX_AUTO
        Set(buffer, UconfigBase + 0x24c, Low32(a[1])); // VGT_NUM_INDICES
        Set(buffer, c + 0x1fc, 2);                    // VGT_DRAW_INITIATOR: auto index
        break;
    case fn::SetDepthStencilControl: // (buffer, value)
        Set(buffer, c + 0x200, Low32(a[1]) & 0xc07007ff);
        break;
    case fn::SetBlendControl: // (buffer, slot, value)
        if (a[1] < 8) {
            Set(buffer, c + 0x1e0 + static_cast<u32>(a[1]), Low32(a[2]) & 0xffff1fff);
        }
        break;
    case fn::SetPrimitiveSetup: // (buffer, value)
        Set(buffer, c + 0x205, Low32(a[1]) & 0x393fff);
        break;
    case fn::SetRenderTargetMask: // (buffer, mask)
        Set(buffer, c + 0x08e, Low32(a[1]));
        break;
    case fn::SetCbControl: // (buffer, mode, raster op)
        Set(buffer, c + 0x202, (Low32(a[1]) << 4 & 0x70) | (Low32(a[2]) << 16 & 0xff0000));
        break;
    case fn::SetStencilOpControl: // (buffer, value)
        Set(buffer, c + 0x10b, Low32(a[1]) & 0xffffff);
        break;
    case fn::SetStencil: // (buffer, front reference/masks, back reference/masks)
        Set(buffer, c + 0x10c, Low32(a[1]));
        Set(buffer, c + 0x10d, Low32(a[2]));
        break;
    case fn::SetDepthClearValue: // (buffer, float depth in xmm0)
        Set(buffer, c + 0x00b, Low32(call.float_args[0]));
        break;
    case fn::SetDbRenderControl: // (buffer, value)
        Set(buffer, c + 0x000, Low32(a[1]) & 0x1fff);
        break;
    case fn::SetDbRenderOverride: // (buffer, value)
        Set(buffer, c + 0x003, Low32(a[1]) & 0xfc1f1fff);
        break;
    case fn::SetPolygonOffsetFront: // (buffer, float scale in xmm0, float offset in xmm1)
        Set(buffer, c + 0x2e0, Low32(call.float_args[0]));
        Set(buffer, c + 0x2e1, Low32(call.float_args[1]));
        break;
    case fn::SetPolygonOffsetBack:
        Set(buffer, c + 0x2e2, Low32(call.float_args[0]));
        Set(buffer, c + 0x2e3, Low32(call.float_args[1]));
        break;
    case fn::SetPolygonOffsetZFormat: // (buffer, depth format)
        if (a[1] < PolygonOffsetFormats.size()) {
            Set(buffer, c + 0x2de, PolygonOffsetFormats[a[1]]);
        }
        break;
    case fn::SetScanModeControl: // (buffer, msaa enable, vport scissor enable)
        Set(buffer, c + 0x292, (Low32(a[1]) & 1) + (Low32(a[2]) & 1) * 2);
        break;
    case fn::SetClipControl: // (buffer, value)
        Set(buffer, c + 0x204, Low32(a[1]) & 0xf7fe03f);
        break;
    case fn::SetViewportScissor: // (buffer, viewport, left, top, right, bottom, window offset on)
        // Only viewport 0 is modelled; the game uses only that one.
        if (a[1] == 0) {
            // The 7th argument is a 32-bit int on the stack: the upper half of its slot is
            // whatever was there before.
            const u32 window_offset_disable = Low32(call.stack_args[0]) == 0 ? 1u << 31 : 0;
            Set(buffer, c + 0x094,
                window_offset_disable | (Low32(a[3]) << 16 & 0x7fff0000) | (Low32(a[2]) & 0x7fff));
            Set(buffer, c + 0x095, (Low32(a[5]) << 16 & 0x7fff0000) | (Low32(a[4]) & 0x7fff));
        }
        break;
    case fn::SetBorderColorTableAddr: // (buffer, address)
        Set(buffer, c + 0x020, static_cast<u32>(a[1] >> 8));
        break;
    case fn::SetRenderTarget: { // (buffer, slot, RenderTarget* or null)
        // A null target (or one whose base address is 0) only switches the slot off:
        // CB_COLORn_INFO = 0, the other registers keep their values. Otherwise the 14 registers
        // the RenderTarget object starts with are copied as they are. (The method also adjusts
        // one register for fast-cleared targets with a CMASK and no FMASK; not modelled yet.)
        const u32 slot = static_cast<u32>(a[1]);
        if (slot >= 8) {
            break;
        }
        const u32 first = c + ColorTargetBase + slot * ColorTargetStride;
        std::array<u32, ColorTargetRegisters> registers;
        if (a[2] == 0 || !ReadGame(a[2], registers.data(), sizeof(registers)) || registers[0] == 0) {
            Set(buffer, first + ColorTargetInfo, 0);
            break;
        }
        for (u32 field = 0; field < ColorTargetRegisters; field++) {
            Set(buffer, first + field, registers[field]);
        }
        break;
    }
    case fn::SetDepthRenderTarget: { // (buffer, DepthRenderTarget* or null)
        // Gnm's DepthRenderTarget object, in dwords: [0..7] DB_Z_INFO, DB_STENCIL_INFO,
        // DB_Z_READ_BASE, DB_STENCIL_READ_BASE, DB_Z_WRITE_BASE, DB_STENCIL_WRITE_BASE,
        // DB_DEPTH_SIZE, DB_DEPTH_SLICE (written in one packet), [8] DB_DEPTH_VIEW,
        // [9] DB_HTILE_DATA_BASE, [10] DB_HTILE_SURFACE, [11] DB_DEPTH_INFO.
        // It counts as a target if it has a depth buffer (Z format != 0 and a read address) or
        // a stencil buffer (stencil format != 0 and a read address) — what the method's two
        // helpers (0xc68710, 0xc68ca0) check. Otherwise depth and stencil are switched off:
        // DB_Z_INFO = DB_STENCIL_INFO = 0.
        std::array<u32, 12> target;
        const bool read = a[1] != 0 && ReadGame(a[1], target.data(), sizeof(target));
        const bool has_depth = read && (target[0] & 0x3) != 0 && target[2] != 0;
        const bool has_stencil = read && (target[1] & 0x1) != 0 && target[3] != 0;
        if (!has_depth && !has_stencil) {
            Set(buffer, c + 0x010, 0);
            Set(buffer, c + 0x011, 0);
            break;
        }
        for (u32 i = 0; i < 8; i++) {
            Set(buffer, c + 0x010 + i, target[i]);
        }
        Set(buffer, c + 0x002, target[8]);
        Set(buffer, c + 0x005, target[9]);
        Set(buffer, c + 0x2af, target[10]);
        Set(buffer, c + 0x00f, target[11]);
        break;
    }
    case fn::SetupScreenViewport: {
        // (buffer, left, top, right, bottom: 32-bit ints; z scale, z offset: floats in xmm0/1)
        // Gnmx's helper: viewport 0 covering the rectangle, depth range 0..1, Y flipped. The
        // arithmetic is single precision, in the method's order, so the bits match.
        // Not modelled: PA_SU_HARDWARE_SCREEN_OFFSET (0x8d) and the guard band (0x2fa-0x2fd),
        // which only tune the GPU's clipping and have no Vulkan counterpart.
        const s32 left = static_cast<s32>(a[1]), top = static_cast<s32>(a[2]);
        const s32 right = static_cast<s32>(a[3]), bottom = static_cast<s32>(a[4]);
        const float x_scale = static_cast<float>(right - left) * 0.5f;
        const float y_scale = static_cast<float>(bottom - top) * -0.5f;
        // The offsets convert the corner as an unsigned 32-bit value (zero-extended).
        const float x_offset = static_cast<float>(static_cast<u32>(left)) + x_scale;
        const float y_offset =
            static_cast<float>(static_cast<u32>(top)) + static_cast<float>(bottom - top) * 0.5f;
        Set(buffer, c + 0x0b4, std::bit_cast<u32>(0.0f));
        Set(buffer, c + 0x0b5, std::bit_cast<u32>(1.0f));
        Set(buffer, c + 0x10f, std::bit_cast<u32>(x_scale));
        Set(buffer, c + 0x110, std::bit_cast<u32>(x_offset));
        Set(buffer, c + 0x111, std::bit_cast<u32>(y_scale));
        Set(buffer, c + 0x112, std::bit_cast<u32>(y_offset));
        Set(buffer, c + 0x113, Low32(call.float_args[0]));
        Set(buffer, c + 0x114, Low32(call.float_args[1]));
        Set(buffer, c + 0x206, 0x43f); // all scales/offsets on, W0 format (0xc7c9c0/0xc7c9d0)
        Set(buffer, c + 0x00c, static_cast<u32>(top) << 16 | (static_cast<u32>(left) & 0xffff));
        Set(buffer, c + 0x00d,
            static_cast<u32>(bottom) << 16 | (static_cast<u32>(right) & 0xffff));
        break;
    }
    case fn::SetActiveShaderStages: // (buffer, value)
        // Only the stage combinations the GPU supports are written; anything else is ignored.
        // 0: VS+PS; 0x45/0xad/0xb0/0x145/0x1ad/0x200: with ES/GS, LS/HS and copy shaders.
        switch (Low32(a[1])) {
        case 0x0: case 0x45: case 0xad: case 0xb0: case 0x145: case 0x1ad: case 0x200:
            Set(buffer, c + 0x2d5, Low32(a[1]));
            break;
        default:
            break;
        }
        break;
    case fn::SetPsShaderUsage: { // (buffer, u32 inputs[], count)
        // Copies `count` (at most 32) dwords into SPI_PS_INPUT_CNTL_0 onwards.
        const u32 count = Low32(a[2]);
        std::array<u32, PsInputCount> inputs;
        if (count == 0 || count > PsInputCount || !ReadGame(a[1], inputs.data(), count * 4)) {
            break;
        }
        for (u32 i = 0; i < count; i++) {
            Set(buffer, c + PsInputBase + i, inputs[i]);
        }
        break;
    }
    case fn::SetCsShader: // (lcue, CsShader*, resource offsets table): dispatches on this queue
        SetStageTable(buffer, StageCompute, a[2]);
        SetCsProgram(LcueCommandBuffer(buffer), a[1]);
        break;
    // --- shaders (graphics LCUE: first argument = LCUE, not the command buffer) -------------
    // The LCUE stores the shader and emits its registers in the pre-draw, through the driver's
    // sceGnmSet/UpdatePsShader350 and sceGnmSet/UpdateVsShader, which take the Gnmx shader
    // struct's registers (from dword 2 on: PGM_LO, PGM_HI, RSRC1, RSRC2, then per stage).
    // Modelled when the shader is set: no draw sits between the two.
    case fn::SetPsShader: { // (lcue, PsShader* or null, resource offsets table)
        SetStageTable(buffer, 1, a[2]);
        const u64 target = LcueCommandBuffer(buffer);
        std::array<u32, 12> regs{};
        if (a[1] == 0 || !ReadGame(a[1] + 8, regs.data(), sizeof(regs))) {
            // No pixel shader: no program, depth from the rasterizer, all channels written.
            Set(target, ShBase + 0x08, 0);
            Set(target, ShBase + 0x09, 0);
            Set(target, c + 0x203, 0);
            Set(target, c + 0x08f, 0xf);
            break;
        }
        Set(target, ShBase + 0x08, regs[0]);   // PGM_LO (the code's GPU address >> 8)
        Set(target, ShBase + 0x09, 0);         // PGM_HI: the driver rejects anything else
        Set(target, ShBase + 0x0a, regs[2]);   // RSRC1
        Set(target, ShBase + 0x0b, regs[3]);   // RSRC2
        Set(target, c + 0x1c4, regs[4]);       // SPI_SHADER_Z_FORMAT
        Set(target, c + 0x1c5, regs[5]);       // SPI_SHADER_COL_FORMAT
        Set(target, c + 0x1b3, regs[6]);       // SPI_PS_INPUT_ENA
        Set(target, c + 0x1b4, regs[7]);       // SPI_PS_INPUT_ADDR
        Set(target, c + 0x1b6, regs[8]);       // SPI_PS_IN_CONTROL
        Set(target, c + 0x1b8, regs[9]);       // SPI_BARYC_CNTL
        Set(target, c + 0x203, regs[10]);      // DB_SHADER_CONTROL
        Set(target, c + 0x08f, regs[11]);      // CB_SHADER_MASK
        break;
    }
    case fn::SetVsShader: { // (lcue, VsShader*, modifier, fetch shader, resource offsets table)
        SetStageTable(buffer, 2, a[4]);
        {
            Shard& shard = ShardOf(buffer);
            std::scoped_lock lock{shard.mutex};
            shard.lcues[buffer].fetch_shader = a[3];
        }
        const u64 target = LcueCommandBuffer(buffer);
        std::array<u32, 7> regs{};
        if (a[1] == 0 || !ReadGame(a[1] + 8, regs.data(), sizeof(regs))) {
            break;
        }
        // A non-zero modifier replaces RSRC1's VGPR/SGPR count fields (bits outside 0xfcfffc3f).
        const u32 modifier = Low32(a[2]);
        const u32 rsrc1 = modifier == 0 ? regs[2] : (regs[2] & 0xfcfffc3f) | modifier;
        Set(target, ShBase + 0x48, regs[0]);   // PGM_LO
        Set(target, ShBase + 0x49, 0);         // PGM_HI
        Set(target, ShBase + 0x4a, rsrc1);     // RSRC1
        Set(target, ShBase + 0x4b, regs[3]);   // RSRC2
        Set(target, c + 0x1b1, regs[4]);       // SPI_VS_OUT_CONFIG
        Set(target, c + 0x1c3, regs[5]);       // SPI_SHADER_POS_FORMAT
        Set(target, c + 0x207, regs[6]);       // PA_CL_VS_OUT_CNTL
        break;
    }
    // --- resources and user data (graphics LCUE) -----------------------------------------
    // (lcue, stage, first slot, count, descriptors); arrays of the resource offsets table:
    // +0x22 resources (t#), +0x42 writable resources (u#), +0x62 samplers, +0x82 constant
    // buffers. Textures are 8 dwords, buffers and samplers 4.
    case fn::SetTextures:
        SetResources(buffer, Low32(a[1]), Low32(a[2]), Low32(a[3]), a[4], 0x22, 8);
        break;
    case fn::SetRwTextures:
        SetResources(buffer, Low32(a[1]), Low32(a[2]), Low32(a[3]), a[4], 0x42, 8);
        break;
    case fn::SetBuffers:
        SetResources(buffer, Low32(a[1]), Low32(a[2]), Low32(a[3]), a[4], 0x22, 4);
        break;
    case fn::SetRwBuffers:
        SetResources(buffer, Low32(a[1]), Low32(a[2]), Low32(a[3]), a[4], 0x42, 4);
        break;
    case fn::SetSamplers:
        SetResources(buffer, Low32(a[1]), Low32(a[2]), Low32(a[3]), a[4], 0x62, 4);
        break;
    case fn::SetConstantBuffers:
        SetResources(buffer, Low32(a[1]), Low32(a[2]), Low32(a[3]), a[4], 0x82, 4);
        break;
    case fn::SetSrtPointer: { // (lcue, stage, u32 data[])
        // The table's u16 at +0x10: low byte = first SGPR (0xFF: none), high byte = dwords;
        // the data are copied into those SGPRs (sub_c6d5e0). For the VS, when the table's byte
        // +0x14 is not 0xFF, setSrtPointer also writes them into the LCUE's second command
        // buffer (+0xc098, with the table at +0xc1b8): not modelled.
        const u32 stage = Low32(a[1]);
        const u64 table = StageTable(buffer, stage);
        u16 field = 0xFFFF;
        if (stage >= 7 || !table || !ReadGame(table + 0x10, &field, sizeof(field))) {
            break;
        }
        const u32 sgpr = field & 0xFF, count = field >> 8;
        std::array<u32, UserDataCount> data;
        if (sgpr == 0xFF || count == 0 || count > UserDataCount ||
            !ReadGame(a[2], data.data(), count * 4)) {
            break;
        }
        const u64 target = LcueCommandBuffer(buffer);
        for (u32 d = 0; d < count && sgpr + d < UserDataCount; d++) {
            Set(target, UserDataBase[stage] + sgpr + d, data[d]);
        }
        break;
    }
    case fn::PreDraw: // (lcue)
        PreDraw(buffer, false, {StagePixel, StageVertex});
        break;
    case fn::PreDispatch: // (lcue): graphics queue dispatch; same scheme, stage 0
        PreDraw(buffer, false, {StageCompute});
        break;
    // --- compute LCUE (compute context + 0x50; first argument = LCUE) ------------------------
    case fn::ComputeSetCsShader: // (lcue, CsShader*, resource offsets table)
        SetStageTable(buffer, StageCompute, a[2]);
        SetCsProgram(LcueCommandBuffer(buffer, true), a[1]);
        break;
    case fn::ComputePreDispatch: // (lcue)
        PreDraw(buffer, true, {StageCompute});
        break;
    // Resource setters: (lcue, first slot, count, descriptors); arrays as in the graphics LCUE.
    case fn::ComputeSetBuffers:
        SetResources(buffer, StageCompute, Low32(a[1]), Low32(a[2]), a[3], 0x22, 4, true);
        break;
    case fn::ComputeSetTextures:
        SetResources(buffer, StageCompute, Low32(a[1]), Low32(a[2]), a[3], 0x22, 8, true);
        break;
    case fn::ComputeSetRwBuffers:
        SetResources(buffer, StageCompute, Low32(a[1]), Low32(a[2]), a[3], 0x42, 4, true);
        break;
    case fn::ComputeSetRwTextures:
        SetResources(buffer, StageCompute, Low32(a[1]), Low32(a[2]), a[3], 0x42, 8, true);
        break;
    case fn::ComputeSetSamplers:
        SetResources(buffer, StageCompute, Low32(a[1]), Low32(a[2]), a[3], 0x62, 4, true);
        break;
    case fn::ComputeSetConstantBuffers:
        SetResources(buffer, StageCompute, Low32(a[1]), Low32(a[2]), a[3], 0x82, 4, true);
        break;
    case fn::ComputeSetSrtPointer: { // (lcue, u32 data[]): table's u16 at +0x10, as graphics
        const u64 table = StageTable(buffer, StageCompute);
        u16 field = 0xFFFF;
        if (!table || !ReadGame(table + 0x10, &field, sizeof(field))) {
            break;
        }
        const u32 sgpr = field & 0xFF, count = field >> 8;
        std::array<u32, UserDataCount> data;
        if (sgpr == 0xFF || count == 0 || count > UserDataCount ||
            !ReadGame(a[1], data.data(), count * 4)) {
            break;
        }
        const u64 target = LcueCommandBuffer(buffer, true);
        for (u32 d = 0; d < count && sgpr + d < UserDataCount; d++) {
            Set(target, UserDataBase[StageCompute] + sgpr + d, data[d]);
        }
        break;
    }
    case fn::ComputeResetContext: { // (compute context, memory, size, ...)
        // A new command buffer (the DispatchCommandBuffer at context + 0x10) and an LCUE
        // (context + 0x50) cleared by sub_c5e510: memset(lcue + 0x90, 0, 0x1810) zeroes its
        // resource table, so every dword of it is known to be 0.
        const u64 command_buffer = buffer + 0x10, lcue = buffer + 0x50;
        {
            Shard& shard = ShardOf(command_buffer);
            std::scoped_lock lock{shard.mutex};
            shard.buffers.erase(command_buffer);
        }
        Shard& shard = ShardOf(lcue);
        std::scoped_lock lock{shard.mutex};
        LcueState& state = shard.lcues[lcue];
        state = LcueState{};
        state.images[StageCompute].known.set();
        break;
    }
    case fn::ComputeInitializeDefaultHardwareState: // (dispatch command buffer, ...)
        // sceGnmDispatchInitDefaultHardwareState
        Set(buffer, ShBase + 0x216, 0xffffffff); // COMPUTE_STATIC_THREAD_MGMT_SE0
        Set(buffer, ShBase + 0x217, 0xffffffff); // COMPUTE_STATIC_THREAD_MGMT_SE1
        Set(buffer, ShBase + 0x215, 0x170);      // COMPUTE_RESOURCE_LIMITS
        break;
    case fn::SetPrimitiveType: // (buffer, type)
        Set(buffer, UconfigBase + 0x242, Low32(a[1]) & 0x3f);
        break;
    default:
        break;
    }
}

} // namespace render
