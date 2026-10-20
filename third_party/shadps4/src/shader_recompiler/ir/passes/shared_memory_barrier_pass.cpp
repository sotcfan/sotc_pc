// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <unordered_set>
#include <boost/container/small_vector.hpp>
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/profile.h"

namespace Shader::Optimization {

static bool IsLoadShared(const IR::Inst& inst) {
    return inst.GetOpcode() == IR::Opcode::LoadSharedU16 ||
           inst.GetOpcode() == IR::Opcode::LoadSharedU32 ||
           inst.GetOpcode() == IR::Opcode::LoadSharedU64;
}

static bool IsWriteShared(const IR::Inst& inst) {
    const IR::Opcode opcode = inst.GetOpcode();
    if (opcode >= IR::Opcode::SharedAtomicIAdd32 && opcode <= IR::Opcode::SharedAtomicCmpSwap64) {
        return !inst.Flags<bool>();
    }
    return opcode == IR::Opcode::WriteSharedU16 || opcode == IR::Opcode::WriteSharedU32 ||
           opcode == IR::Opcode::WriteSharedU64;
}

// Inserts barriers when a shared memory write and read occur in the same basic block.
static void EmitBarrierInBlock(IR::Block* block) {
    enum class BarrierAction : u32 {
        None,
        BarrierOnWrite,
        BarrierOnRead,
    };
    BarrierAction action{};
    for (IR::Inst& inst : block->Instructions()) {
        if (IsLoadShared(inst)) {
            if (action == BarrierAction::BarrierOnRead) {
                IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
                ir.Barrier();
            }
            action = BarrierAction::BarrierOnWrite;
            continue;
        }
        if (IsWriteShared(inst)) {
            if (action == BarrierAction::BarrierOnWrite) {
                IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
                ir.Barrier();
            }
            action = BarrierAction::BarrierOnRead;
        }
    }
    if (action != BarrierAction::None) {
        IR::IREmitter ir{*block, --block->end()};
        ir.Barrier();
    }
}

using NodeSet = std::unordered_set<const IR::Block*>;

static void EmitBarrierAtBlockStart(IR::Block* block) {
    auto insert_point = std::ranges::find_if_not(block->Instructions(), IR::IsPhi);
    // sotcport: the console's code often has its own barrier there already (the merge block of
    // one step of a reduction); a second one in a row only costs time.
    if (insert_point != block->end() && insert_point->GetOpcode() == IR::Opcode::Barrier) {
        return;
    }
    IR::IREmitter ir{*block, insert_point};
    ir.Barrier();
}

// sotcport (2026-10-20): whether a condition can differ between the invocations of a workgroup.
// It does when it depends on the invocation's id (LocalInvocationId, the flat index, the lane
// id). In a workgroup of one console wave (64 threads or fewer) a value read from one lane for
// the whole wave (ReadLane, Ballot) is the same in the whole workgroup, and on a GPU with smaller
// subgroups LowerWave64BallotPass (after this pass) makes it so through shared memory: the search
// does not go through those. Upstream followed them, so a condition on, say, the minimum depth of
// an 8x8 tile (each lane's depth, reduced across the wave and read from lanes 0 and 32) counted as
// divergent, and nothing inside it got barriers: Shadow of the Colossus's tiled light culling (cs
// 0x57bd9f27) builds a list of lights in shared memory inside such a condition (reset a counter,
// each invocation appends the lights it finds, then all of them read the list), and on a GTX 1650
// the two subgroups of 32 raced over it: one half of every tile (4 rows) got the wrong lights,
// regular horizontal stripes over the shadowed areas. ReadFirstLane is followed: it is not
// lowered, and reads the first lane of each subgroup.
static bool IsDivergent(const IR::U1& cond, bool single_wave) {
    if (cond.IsImmediate()) {
        return false;
    }
    boost::container::small_vector<const IR::Inst*, 64> stack{cond.Inst()};
    std::unordered_set<const IR::Inst*> visited;
    while (!stack.empty()) {
        const IR::Inst* const inst = stack.back();
        stack.pop_back();
        if (!visited.insert(inst).second) {
            continue;
        }
        switch (inst->GetOpcode()) {
        case IR::Opcode::LaneId:
            return true;
        case IR::Opcode::GetAttributeU32:
            if (inst->Arg(0).Attribute() == IR::Attribute::LocalInvocationId ||
                inst->Arg(0).Attribute() == IR::Attribute::LocalInvocationIndex) {
                return true;
            }
            break;
        case IR::Opcode::ReadLane:
        case IR::Opcode::Ballot:
            if (single_wave) {
                continue; // the same in the whole workgroup: not followed
            }
            break;
        default:
            break;
        }
        for (size_t arg = 0; arg < inst->NumArgs(); ++arg) {
            const IR::Value value = inst->Arg(arg);
            if (!value.IsImmediate()) {
                stack.push_back(value.Inst());
            }
        }
    }
    return false;
}

// Inserts a barrier after divergent conditional blocks to avoid undefined
// behavior when some threads write and others read from shared memory.
static void EmitBarrierInMergeBlock(const IR::AbstractSyntaxNode::Data& data,
                                    NodeSet& divergence_end, u32& divergence_depth,
                                    bool single_wave) {
    const IR::U1 cond = data.if_node.cond;
    if (IsDivergent(cond, single_wave)) {
        if (divergence_depth == 0) {
            EmitBarrierAtBlockStart(data.if_node.merge);
        }
        ++divergence_depth;
        divergence_end.emplace(data.if_node.merge);
    }
}

// A barrier inside a loop is invalid when different invocations leave on different iterations.
// Mark such loops so their shared-memory synchronization can be deferred to the merge block.
static NodeSet FindDivergentLoops(const IR::AbstractSyntaxList& syntax_list, bool single_wave) {
    NodeSet divergent_loops;
    for (const IR::AbstractSyntaxNode& node : syntax_list) {
        switch (node.type) {
        case IR::AbstractSyntaxNode::Type::Repeat:
            if (IsDivergent(node.data.repeat.cond, single_wave)) {
                divergent_loops.emplace(node.data.repeat.merge);
            }
            break;
        case IR::AbstractSyntaxNode::Type::Break:
            if (IsDivergent(node.data.break_node.cond, single_wave)) {
                divergent_loops.emplace(node.data.break_node.merge);
            }
            break;
        default:
            break;
        }
    }
    return divergent_loops;
}

void SharedMemoryBarrierPass(IR::Program& program, const RuntimeInfo& runtime_info,
                             const Profile& profile) {
    if (program.info.hw_stage != HwStage::Compute) {
        return;
    }
    const auto& cs_info = runtime_info.hw.cs;
    const u32 shared_memory_size = cs_info.shared_memory_size;
    // sotcport (2026-10-20): every workgroup size, not only 64 (upstream: "the compiler can only
    // omit barriers when the local workgroup size is the same as the HW subgroup"). The console's
    // code leaves out the barriers between shared memory steps that one wave of 64 does alone,
    // and that is not only a whole 64-thread group: the last steps of a reduction in a larger
    // group are written the same way, `if (id < 32) s[id] += s[id + 32]; if (id < 16) s[id] +=
    // s[id + 16]; ...`, each step reading what other lanes wrote in the one before. In a wave of
    // 64 the lanes run in lockstep and that works. On a GPU with subgroups of 32 (NVIDIA) it does
    // not, even with all of those lanes in one subgroup: without a barrier nothing orders one
    // lane's write before another lane's read. Shadow of the Colossus's auto exposure (cs
    // 0x276f682a, 512 threads, one luminance sum per row of the half-size image) got sums of 41%
    // of the right value on a GTX 1650, different on every replay of the same frame, and the
    // whole image came out overexposed. The barriers this pass adds are only placed where the
    // control flow is the same in the whole workgroup, whatever its size.
    if (shared_memory_size == 0 || !profile.needs_lds_barriers) {
        return;
    }
    // One console wave: what one lane reads for the whole wave is the same in the whole group.
    constexpr u32 GcnSubgroupSize = 64;
    const u32 threadgroup_size =
        cs_info.workgroup_size[0] * cs_info.workgroup_size[1] * cs_info.workgroup_size[2];
    const bool single_wave = threadgroup_size <= GcnSubgroupSize;
    using Type = IR::AbstractSyntaxNode::Type;
    u32 divergence_depth{};
    NodeSet divergence_end;
    const NodeSet divergent_loops = FindDivergentLoops(program.syntax_list, single_wave);
    for (const IR::AbstractSyntaxNode& node : program.syntax_list) {
        if (node.type == Type::EndIf) {
            if (divergence_end.contains(node.data.end_if.merge)) {
                --divergence_depth;
            }
            continue;
        }
        // Check if branch depth is zero, we don't want to insert barrier in potentially divergent
        // code.
        if (node.type == Type::If) {
            EmitBarrierInMergeBlock(node.data, divergence_end, divergence_depth, single_wave);
            continue;
        }
        if (node.type == Type::Loop && divergent_loops.contains(node.data.loop.merge)) {
            ++divergence_depth;
            continue;
        }
        if (node.type == Type::Repeat && divergent_loops.contains(node.data.repeat.merge)) {
            ASSERT(divergence_depth > 0);
            --divergence_depth;
            if (divergence_depth == 0) {
                EmitBarrierAtBlockStart(node.data.repeat.merge);
            }
            continue;
        }
        if (node.type == Type::Block && divergence_depth == 0) {
            EmitBarrierInBlock(node.data.block);
        }
    }
}

} // namespace Shader::Optimization
