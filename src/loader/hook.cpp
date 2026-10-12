#include "common/host_memory.h"
#include "common/platform.h"
#include "loader/hook.h"


#include <Zydis/Zydis.h>

#include <cerrno>
#include <cstring>
#include <limits>
#include <mutex>

#include "common/log.h"

namespace loader {

namespace {

// Below the executable (mapped at 0x400000): the game never maps memory there, and every jump
// from its code fits in 32 bits.
constexpr u64 NearCodeBase = 0x200000;
constexpr size_t NearCodeSize = 0x200000;

constexpr size_t JumpSize = 5; // E9 rel32

std::mutex g_mutex;
u8* g_near_code = nullptr;
size_t g_near_used = 0;

bool FitsRel32(s64 value) {
    return value >= std::numeric_limits<s32>::min() && value <= std::numeric_limits<s32>::max();
}

void WriteJump(u8* from, const void* to) {
    const s64 offset = reinterpret_cast<s64>(to) - reinterpret_cast<s64>(from + JumpSize);
    from[0] = 0xE9;
    const s32 rel = static_cast<s32>(offset);
    std::memcpy(from + 1, &rel, sizeof(rel));
}

// Copies the instructions covering the first `JumpSize` bytes of `source` to `dest`, fixing
// everything that is relative to the instruction pointer. Returns the number of source bytes
// moved and sets `written` to the bytes written, or returns 0 if it cannot be done.
size_t RelocateStart(const u8* source, u8* dest, size_t& written, u64 address) {
    ZydisDecoder decoder;
    ZydisDecoderInit(&decoder, ZYDIS_MACHINE_MODE_LONG_64, ZYDIS_STACK_WIDTH_64);
    size_t moved = 0;
    written = 0;
    while (moved < JumpSize) {
        ZydisDecodedInstruction instruction;
        ZydisDecodedOperand operands[ZYDIS_MAX_OPERAND_COUNT];
        if (!ZYAN_SUCCESS(ZydisDecoderDecodeFull(&decoder, source + moved, 15, &instruction,
                                                 operands))) {
            LOG_ERROR("Hook", "{:#x}: undecodable instruction at +{}", address, moved);
            return 0;
        }
        const u8* old_ip = source + moved;
        u8* new_ip = dest + written;
        const u8 length = instruction.length;
        if (instruction.mnemonic == ZYDIS_MNEMONIC_RET || instruction.mnemonic == ZYDIS_MNEMONIC_INT3) {
            LOG_ERROR("Hook", "{:#x}: the function ends before {} bytes", address, JumpSize);
            return 0;
        }
        std::memcpy(new_ip, old_ip, length);
        const s64 delta = old_ip - new_ip;

        if (instruction.attributes & ZYDIS_ATTRIB_IS_RELATIVE) {
            // A memory operand relative to RIP: the displacement moves with the instruction.
            if (instruction.raw.disp.size == 32) {
                s32 disp;
                std::memcpy(&disp, new_ip + instruction.raw.disp.offset, 4);
                const s64 fixed = disp + delta;
                if (!FitsRel32(fixed)) {
                    LOG_ERROR("Hook", "{:#x}: relative address out of range", address);
                    return 0;
                }
                const s32 value = static_cast<s32>(fixed);
                std::memcpy(new_ip + instruction.raw.disp.offset, &value, 4);
            }
            // A relative jump or call.
            if (instruction.raw.imm[0].is_relative) {
                const u64 target = reinterpret_cast<u64>(old_ip) + length +
                                   static_cast<s64>(instruction.raw.imm[0].value.s);
                if (target > reinterpret_cast<u64>(source) &&
                    target < reinterpret_cast<u64>(source) + JumpSize) {
                    LOG_ERROR("Hook", "{:#x}: jump into the moved bytes", address);
                    return 0;
                }
                if (instruction.raw.imm[0].size == 32) {
                    const s64 fixed = static_cast<s64>(target) -
                                      reinterpret_cast<s64>(new_ip + length);
                    if (!FitsRel32(fixed)) {
                        return 0;
                    }
                    const s32 value = static_cast<s32>(fixed);
                    std::memcpy(new_ip + instruction.raw.imm[0].offset, &value, 4);
                } else if (old_ip[0] == 0xEB) { // jmp rel8 -> jmp rel32
                    WriteJump(new_ip, reinterpret_cast<const void*>(target));
                    written += JumpSize;
                    moved += length;
                    continue;
                } else if ((old_ip[0] & 0xF0) == 0x70) { // jcc rel8 -> jcc rel32
                    new_ip[0] = 0x0F;
                    new_ip[1] = static_cast<u8>(0x80 | (old_ip[0] & 0x0F));
                    const s64 fixed = static_cast<s64>(target) - reinterpret_cast<s64>(new_ip + 6);
                    const s32 value = static_cast<s32>(fixed);
                    std::memcpy(new_ip + 2, &value, 4);
                    written += 6;
                    moved += length;
                    continue;
                } else {
                    LOG_ERROR("Hook", "{:#x}: short jump I cannot move", address);
                    return 0;
                }
            }
        }
        written += length;
        moved += length;
    }
    return moved;
}

} // namespace

u8* AllocateNearCode(size_t size) {
    std::scoped_lock lock{g_mutex};
    if (!g_near_code) {
        void* memory = host_memory::AllocateAt(
            NearCodeBase, NearCodeSize,
            host_memory::ProtRead | host_memory::ProtWrite | host_memory::ProtExec);
        if (!memory) {
            LOG_CRITICAL("Hook", "could not reserve code memory at {:#x}: {}",
                         NearCodeBase, platform::LastErrorText());
            return nullptr;
        }
        g_near_code = static_cast<u8*>(memory);
    }
    size = (size + 15) & ~size_t{15};
    if (g_near_used + size > NearCodeSize) {
        LOG_CRITICAL("Hook", "out of code memory near the game");
        return nullptr;
    }
    u8* result = g_near_code + g_near_used;
    g_near_used += size;
    return result;
}

void* HookFunction(u64 address, const void* replacement) {
    // The trampoline: the moved instructions (each grows by at most 4 bytes when a short jump
    // becomes a long one), a jump back to the rest of the function, and a jump to the
    // replacement for the function's start to reach with 32 bits.
    u8* trampoline = AllocateNearCode(64);
    if (!trampoline) {
        return nullptr;
    }
    auto* function = reinterpret_cast<u8*>(address);
    size_t written = 0;
    const size_t moved = RelocateStart(function, trampoline, written, address);
    if (moved == 0) {
        return nullptr;
    }
    WriteJump(trampoline + written, function + moved);
    u8* to_replacement = trampoline + written + JumpSize;
    // jmp [rip+0]; dq replacement — the replacement may be anywhere.
    to_replacement[0] = 0xFF;
    to_replacement[1] = 0x25;
    std::memset(to_replacement + 2, 0, 4);
    std::memcpy(to_replacement + 6, &replacement, sizeof(replacement));

    WriteJump(function, to_replacement);
    for (size_t i = JumpSize; i < moved; i++) {
        function[i] = 0xCC; // the rest of the moved instructions is never reached
    }
    return trampoline;
}

} // namespace loader
