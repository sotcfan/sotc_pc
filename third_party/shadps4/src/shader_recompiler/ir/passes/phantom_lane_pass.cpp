// sotcport: reads of lanes that do not exist in a small compute workgroup.
//
// On the PS4 a compute workgroup of up to 32 threads still runs as a 64-lane wave; lanes 32-63
// start with their exec bit off. Wave reductions in the game's shaders force exec on for the
// whole wave, fill every lane with the operation's identity, copy the real values into the lanes
// that were active (v_cndmask on the saved exec mask), reduce with lane swizzles and finally add
// v_readlane 0 and v_readlane 32:
//
//     exec = all lanes;  v = saved_exec ? x : identity;  v = op(v, swizzle(v)) ...
//     s0 = readlane(v, 0);  s1 = readlane(v, 32);  result = op(s0, s1)
//
// In Vulkan those lanes do not exist (or are inactive), so readlane 32 returns whatever the
// register holds. Shadow of the Colossus computes its characters' ambient lighting this way
// (compute shader 0x87cf9b8a) and the garbage made them neon colored.
//
// This pass replaces such a ReadLane with the value the lane would have on the PS4, evaluating
// its expression as seen from a lane that was never active: any Select on an exec mask takes
// its false operand and swizzles within the upper half return the lane's own value. When the
// expression contains anything else, the ReadLane is left alone.

#include <optional>
#include <unordered_map>

#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/basic_block.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/runtime_info.h"

namespace Shader::Optimization {

namespace {

bool IsExecMask(const IR::Value& value) {
    if (value.IsImmediate()) {
        return false;
    }
    const IR::Inst* inst = value.Inst();
    if (inst->GetOpcode() == IR::Opcode::GetExec) {
        return true;
    }
    const IR::RegTag tag = inst->GetRegTag();
    return inst->GetOpcode() == IR::Opcode::Phi && tag && tag.type == IR::RegType::Exec;
}

bool IsLaneLocalArithmetic(IR::Opcode op) {
    switch (op) {
    case IR::Opcode::BitCastU16F16:
    case IR::Opcode::BitCastU32F32:
    case IR::Opcode::BitCastF16U16:
    case IR::Opcode::BitCastF32U32:
    case IR::Opcode::FPAbs32:
    case IR::Opcode::FPAdd32:
    case IR::Opcode::FPSub32:
    case IR::Opcode::FPFma32:
    case IR::Opcode::FPMax32:
    case IR::Opcode::FPMin32:
    case IR::Opcode::FPMul32:
    case IR::Opcode::FPNeg32:
    case IR::Opcode::IAdd32:
    case IR::Opcode::ISub32:
    case IR::Opcode::IMul32:
    case IR::Opcode::ShiftLeftLogical32:
    case IR::Opcode::ShiftRightLogical32:
    case IR::Opcode::ShiftRightArithmetic32:
    case IR::Opcode::BitwiseAnd32:
    case IR::Opcode::BitwiseOr32:
    case IR::Opcode::BitwiseXor32:
    case IR::Opcode::BitwiseNot32:
    case IR::Opcode::SMin32:
    case IR::Opcode::UMin32:
    case IR::Opcode::SMax32:
    case IR::Opcode::UMax32:
        return true;
    default:
        return false;
    }
}

class PhantomEvaluator {
public:
    PhantomEvaluator(IR::Block& block, IR::Block::iterator insert_point)
        : block{block}, insert_point{insert_point} {}

    // The value of `value` in a lane of the upper half that was never active.
    std::optional<IR::Value> Evaluate(const IR::Value& value, int depth = 0) {
        if (value.IsImmediate()) {
            return value;
        }
        if (depth > 64) {
            return std::nullopt;
        }
        IR::Inst* inst = value.Inst();
        if (const auto it = cache.find(inst); it != cache.end()) {
            return it->second;
        }
        std::optional<IR::Value> result;
        const IR::Opcode op = inst->GetOpcode();
        if (op == IR::Opcode::ShuffleXor) {
            // A swizzle by less than 32 stays within the upper half, where every lane is alike.
            if (inst->Arg(1).IsImmediate() && inst->Arg(1).U32() < 32) {
                result = Evaluate(inst->Arg(0), depth + 1);
            }
        } else if (op == IR::Opcode::SelectU32 || op == IR::Opcode::SelectF32) {
            if (IsExecMask(inst->Arg(0))) {
                result = Evaluate(inst->Arg(2), depth + 1);
            }
        } else if (op == IR::Opcode::ReadLane || op == IR::Opcode::ReadFirstLane ||
                   op == IR::Opcode::GetUserData) {
            // Uniform across the wave.
            result = IR::Value{inst};
        } else if (IsLaneLocalArithmetic(op)) {
            result = Rebuild(inst, depth);
        }
        cache.emplace(inst, result);
        return result;
    }

private:
    std::optional<IR::Value> Rebuild(IR::Inst* inst, int depth) {
        std::array<IR::Value, 3> args{};
        const size_t num_args = inst->NumArgs();
        if (num_args > args.size()) {
            return std::nullopt;
        }
        for (size_t i = 0; i < num_args; i++) {
            const auto arg = Evaluate(inst->Arg(i), depth + 1);
            if (!arg) {
                return std::nullopt;
            }
            args[i] = *arg;
        }
        IR::Block::iterator it;
        switch (num_args) {
        case 1:
            it = block.PrependNewInst(insert_point, inst->GetOpcode(), {args[0]});
            break;
        case 2:
            it = block.PrependNewInst(insert_point, inst->GetOpcode(), {args[0], args[1]});
            break;
        case 3:
            it = block.PrependNewInst(insert_point, inst->GetOpcode(), {args[0], args[1], args[2]});
            break;
        default:
            return std::nullopt;
        }
        it->SetFlags(inst->Flags<u64>());
        return IR::Value{&*it};
    }

    IR::Block& block;
    IR::Block::iterator insert_point;
    std::unordered_map<IR::Inst*, std::optional<IR::Value>> cache;
};

} // Anonymous namespace

void PhantomLaneReadPass(IR::Program& program, const RuntimeInfo& runtime_info) {
    if (program.info.hw_stage != HwStage::Compute) {
        return;
    }
    const auto [size_x, size_y, size_z] = runtime_info.hw.cs.workgroup_size;
    const u32 num_threads = size_x * size_y * size_z;
    if (num_threads == 0 || num_threads > 32) {
        return;
    }
    for (IR::Block* const block : program.blocks) {
        for (auto it = block->begin(); it != block->end(); ++it) {
            IR::Inst& inst = *it;
            if (inst.GetOpcode() != IR::Opcode::ReadLane || !inst.Arg(1).IsImmediate()) {
                continue;
            }
            const u32 lane = inst.Arg(1).U32();
            if (lane < 32 || lane >= 64) {
                continue;
            }
            PhantomEvaluator evaluator{*block, it};
            if (const auto value = evaluator.Evaluate(inst.Arg(0))) {
                inst.ReplaceUsesWith(*value);
            }
        }
    }
}

} // namespace Shader::Optimization
