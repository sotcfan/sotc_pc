#pragma once

#include <array>
#include <bitset>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "common/types.h"

// The native renderer's model of the GPU state, built from the game's calls to the frontier
// functions (render/frontier.h) without any GPU command (PM4) in between.
//
// On the PS4, the Gnm methods turn their arguments into values of GCN registers and write them
// into the command buffer; the GPU then draws with whatever the registers hold. The model does
// the same arithmetic but keeps the values in a table per command buffer, so that at every draw
// the native renderer knows the complete state. The formulas come from the methods' code in the
// executable (decompiled; one comment per method below in state_model.cpp).
//
// It is proven against the real thing by the renderer recorder (src/diag/gnm_recorder.cpp,
// SOTC_RECORD_GNM=1): at each draw the GPU layer executes, the registers it received are
// compared with the model's snapshot for that draw.
//
// It covers the graphics registers: the simple methods (one register = one argument with a
// mask), render targets, depth target, viewport, shader stages and programs, pixel shader inputs
// and the PS/VS user data (descriptors and SRT data in the SGPRs, and the pointers the pre-draw
// sets to its copies of the LCUE's resource tables, proven by the content they point at), the
// compute state of the graphics queue and of each asynchronous compute queue, index buffers,
// instancing and the indirect arguments address.
//
// State carries over between command buffers. Only the first command buffer of each frame
// resets the GPU state (initializeDefaultHardwareState: CLEAR_STATE plus the driver's default
// sequence, about once per frame against ~12 context initializations); every other buffer starts
// with whatever the buffer submitted before it left. The game builds buffers in parallel, so the
// order is only known when they are submitted: Submit() gives each buffer the state it
// inherits, and a register a buffer did not set itself takes its value from there.

namespace render {

// A register of the GCN register file, as an index of shadPS4's reg_array (and of the PM4
// SET_*_REG offsets plus their base): context registers start at 0xA000, uconfig at 0xC000, SH
// (shader) registers at 0x2C00.
struct ModelledRegister {
    u32 index;
    const char* name;
};

inline constexpr u32 ContextBase = 0xA000;
inline constexpr u32 UconfigBase = 0xC000;
inline constexpr u32 ShBase = 0x2C00;

// The CB_COLORn_* registers of one color render target slot, in hardware order (slot n starts at
// context register 0x318 + n * 15). Gnm's RenderTarget object holds the first 14 in this order.
inline constexpr const char* ColorTargetRegisterNames[] = {
    "BASE", "PITCH", "SLICE", "VIEW", "INFO", "ATTRIB", "DCC_CONTROL", "CMASK", "CMASK_SLICE",
    "FMASK", "FMASK_SLICE", "CLEAR_WORD0", "CLEAR_WORD1", "DCC_BASE",
};
inline constexpr u32 ColorTargetRegisters = 14;
inline constexpr u32 ColorTargetBase = 0x318;
inline constexpr u32 ColorTargetStride = 15;
inline constexpr u32 ColorTargetInfo = 4; // CB_COLORn_INFO: 0 = slot off

// Names of the CB_COLORn_* registers ("CB_COLOR3_INFO"), for reports.
const char* ColorTargetRegisterName(u32 slot, u32 field);

// The registers the model follows, in report order (the color target registers follow these).
inline constexpr ModelledRegister SimpleRegisters[] = {
    {ContextBase + 0x000, "DB_RENDER_CONTROL"},
    {ContextBase + 0x002, "DB_DEPTH_VIEW"},
    {ContextBase + 0x005, "DB_HTILE_DATA_BASE"},
    {ContextBase + 0x00c, "PA_SC_SCREEN_SCISSOR_TL"},
    {ContextBase + 0x00d, "PA_SC_SCREEN_SCISSOR_BR"},
    {ContextBase + 0x00f, "DB_DEPTH_INFO"},
    {ContextBase + 0x010, "DB_Z_INFO"},
    {ContextBase + 0x011, "DB_STENCIL_INFO"},
    {ContextBase + 0x012, "DB_Z_READ_BASE"},
    {ContextBase + 0x013, "DB_STENCIL_READ_BASE"},
    {ContextBase + 0x014, "DB_Z_WRITE_BASE"},
    {ContextBase + 0x015, "DB_STENCIL_WRITE_BASE"},
    {ContextBase + 0x016, "DB_DEPTH_SIZE"},
    {ContextBase + 0x017, "DB_DEPTH_SLICE"},
    {ContextBase + 0x2af, "DB_HTILE_SURFACE"},
    {ContextBase + 0x003, "DB_RENDER_OVERRIDE"},
    {ContextBase + 0x00b, "DB_DEPTH_CLEAR"},
    {ContextBase + 0x020, "TA_BC_BASE_ADDR"},
    {ContextBase + 0x08e, "CB_TARGET_MASK"},
    {ContextBase + 0x08f, "CB_SHADER_MASK"},
    {ContextBase + 0x094, "PA_SC_VPORT_SCISSOR_0_TL"},
    {ContextBase + 0x095, "PA_SC_VPORT_SCISSOR_0_BR"},
    {ContextBase + 0x0b4, "PA_SC_VPORT_ZMIN_0"},
    {ContextBase + 0x0b5, "PA_SC_VPORT_ZMAX_0"},
    {ContextBase + 0x10b, "DB_STENCIL_CONTROL"},
    {ContextBase + 0x10c, "DB_STENCILREFMASK"},
    {ContextBase + 0x10d, "DB_STENCILREFMASK_BF"},
    {ContextBase + 0x1e0, "CB_BLEND0_CONTROL"},
    {ContextBase + 0x1e1, "CB_BLEND1_CONTROL"},
    {ContextBase + 0x1e2, "CB_BLEND2_CONTROL"},
    {ContextBase + 0x1e3, "CB_BLEND3_CONTROL"},
    {ContextBase + 0x1e4, "CB_BLEND4_CONTROL"},
    {ContextBase + 0x1e5, "CB_BLEND5_CONTROL"},
    {ContextBase + 0x1e6, "CB_BLEND6_CONTROL"},
    {ContextBase + 0x1e7, "CB_BLEND7_CONTROL"},
    {ContextBase + 0x10f, "PA_CL_VPORT_XSCALE"},
    {ContextBase + 0x110, "PA_CL_VPORT_XOFFSET"},
    {ContextBase + 0x111, "PA_CL_VPORT_YSCALE"},
    {ContextBase + 0x112, "PA_CL_VPORT_YOFFSET"},
    {ContextBase + 0x113, "PA_CL_VPORT_ZSCALE"},
    {ContextBase + 0x114, "PA_CL_VPORT_ZOFFSET"},
    {ContextBase + 0x1b1, "SPI_VS_OUT_CONFIG"},
    {ContextBase + 0x1b3, "SPI_PS_INPUT_ENA"},
    {ContextBase + 0x1b4, "SPI_PS_INPUT_ADDR"},
    {ContextBase + 0x1b6, "SPI_PS_IN_CONTROL"},
    {ContextBase + 0x1b8, "SPI_BARYC_CNTL"},
    {ContextBase + 0x1c3, "SPI_SHADER_POS_FORMAT"},
    {ContextBase + 0x1c4, "SPI_SHADER_Z_FORMAT"},
    {ContextBase + 0x1c5, "SPI_SHADER_COL_FORMAT"},
    {ContextBase + 0x200, "DB_DEPTH_CONTROL"},
    {ContextBase + 0x202, "CB_COLOR_CONTROL"},
    {ContextBase + 0x203, "DB_SHADER_CONTROL"},
    {ContextBase + 0x204, "PA_CL_CLIP_CNTL"},
    {ContextBase + 0x205, "PA_SU_SC_MODE_CNTL"},
    {ContextBase + 0x206, "PA_CL_VTE_CNTL"},
    {ContextBase + 0x207, "PA_CL_VS_OUT_CNTL"},
    {ContextBase + 0x292, "PA_SC_MODE_CNTL_0"},
    {ContextBase + 0x2de, "PA_SU_POLY_OFFSET_DB_FMT_CNTL"},
    {ContextBase + 0x2e0, "PA_SU_POLY_OFFSET_FRONT_SCALE"},
    {ContextBase + 0x2e1, "PA_SU_POLY_OFFSET_FRONT_OFFSET"},
    {ContextBase + 0x2e2, "PA_SU_POLY_OFFSET_BACK_SCALE"},
    {ContextBase + 0x2e3, "PA_SU_POLY_OFFSET_BACK_OFFSET"},
    {ContextBase + 0x2d5, "VGT_SHADER_STAGES_EN"},
    {UconfigBase + 0x242, "VGT_PRIMITIVE_TYPE"},
    // index buffer and instancing (the GPU layer keeps these in the same register file)
    {ContextBase + 0x1f9, "VGT_DMA_BASE_HI"},
    {ContextBase + 0x1fa, "VGT_DMA_BASE"},
    {ContextBase + 0x1fc, "VGT_DRAW_INITIATOR"},
    {ContextBase + 0x29e, "VGT_DMA_MAX_SIZE"},
    {ContextBase + 0x29f, "VGT_DMA_INDEX_TYPE"},
    {UconfigBase + 0x24c, "VGT_NUM_INDICES"},
    {UconfigBase + 0x24d, "VGT_NUM_INSTANCES"},
    {0xcffe, "INDIRECT_ARGS_BASE_LO"}, // not registers: see IndirectArgsBase
    {0xcfff, "INDIRECT_ARGS_BASE_HI"},
    // shader program registers (SH registers, 0x2C00 + offset)
    {ShBase + 0x08, "SPI_SHADER_PGM_LO_PS"},
    {ShBase + 0x09, "SPI_SHADER_PGM_HI_PS"},
    {ShBase + 0x0a, "SPI_SHADER_PGM_RSRC1_PS"},
    {ShBase + 0x0b, "SPI_SHADER_PGM_RSRC2_PS"},
    {ShBase + 0x48, "SPI_SHADER_PGM_LO_VS"},
    {ShBase + 0x49, "SPI_SHADER_PGM_HI_VS"},
    {ShBase + 0x4a, "SPI_SHADER_PGM_RSRC1_VS"},
    {ShBase + 0x4b, "SPI_SHADER_PGM_RSRC2_VS"},
    // compute registers (SH 0x200 on): on the GPU they belong to the queue that runs the
    // dispatch, and are compared at dispatches only (see IsComputeRegister)
    {ShBase + 0x207, "COMPUTE_NUM_THREAD_X"},
    {ShBase + 0x208, "COMPUTE_NUM_THREAD_Y"},
    {ShBase + 0x209, "COMPUTE_NUM_THREAD_Z"},
    {ShBase + 0x20c, "COMPUTE_PGM_LO"},
    {ShBase + 0x20d, "COMPUTE_PGM_HI"},
    {ShBase + 0x212, "COMPUTE_PGM_RSRC1"},
    {ShBase + 0x213, "COMPUTE_PGM_RSRC2"},
    {ShBase + 0x215, "COMPUTE_RESOURCE_LIMITS"},
    {ShBase + 0x216, "COMPUTE_STATIC_THREAD_MGMT_SE0"},
    {ShBase + 0x217, "COMPUTE_STATIC_THREAD_MGMT_SE1"},
};

// SPI_PS_INPUT_CNTL_0..31 (context 0x191 + n): how each pixel shader input is fed from the
// vertex shader's outputs.
inline constexpr u32 PsInputBase = 0x191;
inline constexpr u32 PsInputCount = 32;

// The user data registers (SPI_SHADER_USER_DATA_<stage>_0..15): the 16 SGPRs a shader starts
// with, filled by the LCUE with descriptors and pointers. Base per Gnm stage number (0 CS, 1 PS,
// 2 VS, 3 GS, 4 ES, 5 HS, 6 LS), from the executable's table at 0x1690410. Only PS and VS are
// modelled.
inline constexpr std::array<u32, 7> UserDataBase = {0x2e40, 0x2c0c, 0x2c4c, 0x2c8c,
                                                    0x2ccc, 0x2d0c, 0x2d4c};
inline constexpr u32 UserDataCount = 16;
const char* UserDataRegisterName(u32 stage, u32 sgpr);
// The stages whose user data are modelled, by Gnm stage number.
inline constexpr u32 StageCompute = 0, StagePixel = 1, StageVertex = 2, ModelledStages = 3;

// The address SET_BASE sets for indirect draw arguments is not a register; the model keeps it
// as two pseudo-registers at the (unused) top of the register file.
inline constexpr u32 IndirectArgsBase = 0xcffe;

// Compute registers (COMPUTE_*, user data included) live per queue on the GPU: graphics draws
// do not use them, and dispatches use only them.
inline constexpr bool IsComputeRegister(u32 index) {
    return index >= ShBase + 0x200 && index < ShBase + 0x250;
}

inline constexpr size_t ModelledCount = std::size(SimpleRegisters) + 8 * ColorTargetRegisters +
                                        PsInputCount + ModelledStages * UserDataCount;

// Names of SPI_PS_INPUT_CNTL_n, for reports.
const char* PsInputRegisterName(u32 input);


// All modelled registers: the simple ones, then CB_COLOR0_BASE ... CB_COLOR7_DCC_BASE.
inline const std::array<ModelledRegister, ModelledCount> ModelledRegisters = [] {
    std::array<ModelledRegister, ModelledCount> all{};
    size_t i = 0;
    for (const ModelledRegister& simple : SimpleRegisters) {
        all[i++] = simple;
    }
    for (u32 slot = 0; slot < 8; slot++) {
        for (u32 field = 0; field < ColorTargetRegisters; field++) {
            all[i++] = {ContextBase + ColorTargetBase + slot * ColorTargetStride + field,
                        ColorTargetRegisterName(slot, field)};
        }
    }
    for (u32 input = 0; input < PsInputCount; input++) {
        all[i++] = {ContextBase + PsInputBase + input, PsInputRegisterName(input)};
    }
    for (u32 stage = 0; stage < ModelledStages; stage++) {
        for (u32 sgpr = 0; sgpr < UserDataCount; sgpr++) {
            all[i++] = {UserDataBase[stage] + sgpr, UserDataRegisterName(stage, sgpr)};
        }
    }
    return all;
}();

// A resource table as the pre-draw copied it into its ring: the LCUE keeps one per shader stage
// (descriptors whose slot is not marked for the SGPRs), and before a draw copies the bound
// shader's part of it (its resource offsets table's first u16, in dwords) to fresh memory and
// points SGPRs at it. `known` marks the dwords the model saw a setter write.
struct ResourceTableCopy {
    std::vector<u32> values;
    std::vector<u8> known;
};

// The modelled registers of one command buffer at one moment. A register is "known" once a
// method set it after the buffer was initialized; until then its value is inherited from the
// buffers submitted before (see StateModel::Submit).
struct RegisterSnapshot {
    std::array<u32, ModelledCount> values{};
    std::bitset<ModelledCount> known;
    // Written by something the model does not follow yet: neither known nor inherited, not
    // compared.
    std::bitset<ModelledCount> excluded;
    // Known registers that hold half of a pointer into the resource table copy of their stage
    // (the address itself is chosen per draw by the pre-draw): `values` holds the dword offset
    // into the table, with TablePointerHigh set for the high half.
    std::bitset<ModelledCount> table_pointer;
    // The resource table copies the pointers refer to, by stage (CS, PS, VS).
    std::array<std::shared_ptr<const ResourceTableCopy>, ModelledStages> tables;
};
inline constexpr u32 TablePointerHigh = 0x80000000;

// A call of the game to a frontier function, as the model needs it.
struct FrontierCall {
    u64 function;                 // address (render::frontier constants)
    std::array<u64, 6> args;      // integer arguments (rdi...r9)
    std::array<u64, 4> float_args; // xmm0-xmm3, low 64 bits
    std::array<u64, 2> stack_args; // 7th and 8th integer arguments
};

class StateModel {
public:
    // Applies a call made by the game (calls between frontier functions must not be passed).
    void Apply(const FrontierCall& call);

    // The current state of a command buffer (empty if it was never seen).
    RegisterSnapshot Snapshot(u64 buffer);

    // The game submitted `buffer` to the graphics queue (or to the asynchronous compute queue
    // object `compute_queue`; each has its own registers): returns the state it starts from
    // (what the queue held before it) and adds the buffer's own state on top of the queue's.
    RegisterSnapshot Submit(u64 buffer, u64 compute_queue = 0);

    // Fills the registers `snapshot` does not know from `inherited`.
    static void Inherit(RegisterSnapshot& snapshot, const RegisterSnapshot& inherited);

private:
    // Per graphics LCUE: the resource offsets table of the shader bound to each stage (where
    // each resource slot goes), as set*Shader passed it.
    struct LcueState {
        std::array<u64, 7> tables{};
        // The LCUE's resource table of each stage as far as the setters wrote it (graphics:
        // lcue + 0x90 + stage * 0x1800; compute: lcue + 0x98, stage 0; 0x600 dwords).
        struct TableImage {
            std::array<u32, 0x600> values{};
            std::bitset<0x600> known;
        };
        std::array<TableImage, ModelledStages> images;
        u64 fetch_shader = 0; // set by setVsShader, kept at lcue + 0xc130
    };

    struct Shard {
        std::mutex mutex;
        std::unordered_map<u64, RegisterSnapshot> buffers;
        std::unordered_map<u64, LcueState> lcues;
    };
    static constexpr size_t ShardCount = 16;

    Shard& ShardOf(u64 buffer);
    void Set(u64 buffer, u32 index, u32 value);
    void Exclude(u64 buffer, u32 index);
    void SetTablePointer(u64 buffer, u32 index, u32 offset);
    // The pre-draw / pre-dispatch: what they point the SGPRs of `stages` at (see PreDraw).
    void PreDraw(u64 lcue, bool compute_lcue, std::initializer_list<u32> stages);
    // The CS program registers from a Gnmx CsShader (setCsShader of either LCUE).
    void SetCsProgram(u64 target, u64 shader);

    u64 StageTable(u64 lcue, u32 stage);
    void SetStageTable(u64 lcue, u32 stage, u64 table);
    // A resource setter: `count` descriptors of `dwords` each, for consecutive slots of the
    // table's array at `array_offset`; slots marked for user data go straight to the SGPRs.
    void SetResources(u64 lcue, u32 stage, u32 slot, u32 count, u64 descriptors, u32 array_offset,
                      u32 dwords, bool compute_lcue = false);

    std::array<Shard, ShardCount> shards_;
    std::mutex queue_mutex_;
    RegisterSnapshot queue_; // the graphics queue's state after the last submitted buffer
    // The same per asynchronous compute queue (by the game's queue object).
    std::unordered_map<u64, RegisterSnapshot> compute_queues_;
};

} // namespace render
