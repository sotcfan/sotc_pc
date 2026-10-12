// How a hooked call flows (see call_hooks.h for the interface):
//
//   game code ── call ──> function start (overwritten by loader::HookFunction with a jump)
//     ──> entry stub (generated below, one per function): saves the argument registers, calls
//         CallHooksEnter, which records the call on this thread's stack of frames, runs the
//         observer's on_enter and replaces the return address on the game's stack with the
//         exit stub; restores the registers and jumps to the trampoline
//     ──> trampoline (made by HookFunction): the function's first instructions, moved, and a
//         jump back into the rest of the function, which runs unchanged
//     ── ret ──> exit stub (shared): keeps the return registers, calls CallHooksExit, which
//         finds the frame by the stack slot of the return address, runs on_exit and gives
//         back the real return address; the stub returns there.
//
// Replacing the return address (instead of calling the original function from our code) keeps
// the game's stack layout exactly as it was, so functions that read their stack arguments, use
// tail calls or never return normally still work.

#include "diag/call_hooks.h"

#include <xbyak/xbyak.h>

#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <vector>

#include "common/log.h"
#include "loader/hook.h"

namespace diag {

namespace {

// The integer argument registers as the entry stub stores them (rax carries the number of vector
// registers used by variadic calls, so it is kept too).
struct SavedRegisters {
    u64 rdi, rsi, rdx, rcx, r8, r9, rax, pad;
    std::array<u64, 16> xmm; // xmm0-xmm7, 16 bytes each (low and high halves)
};
static_assert(offsetof(SavedRegisters, xmm) == 0x40, "the entry stub stores xmm0 at rsp + 0x40");

// One running hooked call. `return_slot` is where the game's return address was on its stack
// (now holding the exit stub): it identifies the call when the exit stub runs.
struct Frame {
    const CallObserver* observer;
    u64* return_slot;
    ObservedCall call;
};

// The calls running on this thread, innermost last.
thread_local std::vector<Frame> t_frames;

std::mutex g_code_mutex;

Xbyak::CodeGenerator& Code() {
    static Xbyak::CodeGenerator code{1 << 20};
    return code;
}

u64 g_exit_stub = 0;

extern "C" void CallHooksEnter(const CallObserver* observer, u32 id, const SavedRegisters* regs,
                               u64* return_slot) {
    u32 depth = 0;
    for (const Frame& frame : t_frames) {
        depth += frame.observer == observer;
    }
    Frame& frame = t_frames.emplace_back();
    frame.observer = observer;
    frame.return_slot = return_slot;
    frame.call.id = id;
    frame.call.depth = depth;
    frame.call.args = {regs->rdi, regs->rsi, regs->rdx, regs->rcx, regs->r8, regs->r9};
    frame.call.float_args = {regs->xmm[0], regs->xmm[2], regs->xmm[4], regs->xmm[6]};
    // Stack arguments start right after the return address.
    frame.call.stack_args = {return_slot[1], return_slot[2]};
    frame.call.return_address = *return_slot;
    frame.call.user = {};
    observer->on_enter(frame.call);
    *return_slot = g_exit_stub;
}

extern "C" u64 CallHooksExit(u64* return_slot) {
    // Find this call's frame: normally the last one; anything above it was left by a call that
    // never returned the normal way.
    size_t index = t_frames.size();
    while (index > 0 && t_frames[index - 1].return_slot != return_slot) {
        index--;
    }
    if (index == 0) {
        LOG_CRITICAL("Hook", "return without a record (stack {:#x}); aborting",
                     reinterpret_cast<u64>(return_slot));
        std::abort();
    }
    const Frame frame = t_frames[index - 1];
    t_frames.resize(index - 1);
    frame.observer->on_exit(frame.call);
    return frame.call.return_address;
}

// The code a hooked function jumps to: saves the argument registers, calls
// CallHooksEnter(observer, id, registers, return slot), restores them and goes on to the
// original function, which then returns to the shared exit stub.
//
// Stack while CallHooksEnter runs (rbp = return slot - 8, after the push below):
//   rbp + 8          the game's return address (the "return slot")
//   rbp              saved rbp
//   rsp + 0x00-0x37  rdi, rsi, rdx, rcx, r8, r9, rax (SavedRegisters)
//   rsp + 0x40-0xbf  xmm0-xmm7 (floating point arguments)
// rsp is aligned to 16 bytes before the call, as the System V ABI requires.
const void* MakeEntryStub(const CallObserver* observer, u32 id, void* const* cell) {
    using namespace Xbyak::util;
    auto& c = Code();
    const void* start = c.getCurr();
    c.push(rbp);
    c.mov(rbp, rsp);
    c.sub(rsp, 0xC0);
    c.and_(rsp, -16);
    c.mov(ptr[rsp + 0x00], rdi);
    c.mov(ptr[rsp + 0x08], rsi);
    c.mov(ptr[rsp + 0x10], rdx);
    c.mov(ptr[rsp + 0x18], rcx);
    c.mov(ptr[rsp + 0x20], r8);
    c.mov(ptr[rsp + 0x28], r9);
    c.mov(ptr[rsp + 0x30], rax);
    for (int i = 0; i < 8; i++) {
        c.movdqu(ptr[rsp + 0x40 + i * 16], Xbyak::Xmm(i));
    }
    c.mov(rdi, reinterpret_cast<u64>(observer));
    c.mov(esi, id);
    c.mov(rdx, rsp);
    c.lea(rcx, ptr[rbp + 8]);
    c.mov(rax, reinterpret_cast<u64>(&CallHooksEnter));
    c.call(rax);
    c.mov(rdi, ptr[rsp + 0x00]);
    c.mov(rsi, ptr[rsp + 0x08]);
    c.mov(rdx, ptr[rsp + 0x10]);
    c.mov(rcx, ptr[rsp + 0x18]);
    c.mov(r8, ptr[rsp + 0x20]);
    c.mov(r9, ptr[rsp + 0x28]);
    c.mov(rax, ptr[rsp + 0x30]);
    for (int i = 0; i < 8; i++) {
        c.movdqu(Xbyak::Xmm(i), ptr[rsp + 0x40 + i * 16]);
    }
    c.mov(rsp, rbp);
    c.pop(rbp);
    c.mov(r11, reinterpret_cast<u64>(cell));
    c.jmp(ptr[r11]);
    c.ready();
    return start;
}

// Where the hooked functions return: asks CallHooksExit for the real return address (keeping
// the return registers) and goes there.
//
// On entry rsp points just past the consumed return address, which is exactly the return
// slot CallHooksEnter recorded (rbx + 24 below, after the 8-byte hole and three pushes). The
// hole receives the real return address, so the final `ret` goes back to the game.
// rax/rdx (integer results) and xmm0/xmm1 (floating point results) are preserved.
u64 MakeExitStub() {
    using namespace Xbyak::util;
    auto& c = Code();
    const u64 start = reinterpret_cast<u64>(c.getCurr());
    c.sub(rsp, 8); // the return address goes here
    c.push(rax);
    c.push(rdx);
    c.push(rbx);
    c.mov(rbx, rsp);
    c.and_(rsp, -16);
    c.sub(rsp, 32);
    c.movdqu(ptr[rsp], xmm0);
    c.movdqu(ptr[rsp + 16], xmm1);
    c.lea(rdi, ptr[rbx + 24]);
    c.mov(rax, reinterpret_cast<u64>(&CallHooksExit));
    c.call(rax);
    c.mov(ptr[rbx + 24], rax);
    c.movdqu(xmm0, ptr[rsp]);
    c.movdqu(xmm1, ptr[rsp + 16]);
    c.mov(rsp, rbx);
    c.pop(rbx);
    c.pop(rdx);
    c.pop(rax);
    c.ret();
    c.ready();
    return start;
}

} // namespace

std::vector<bool> ObserveCalls(std::span<const u64> addresses, const CallObserver* observer) {
    std::scoped_lock lock{g_code_mutex};
    if (!g_exit_stub) {
        g_exit_stub = MakeExitStub();
    }
    std::vector<bool> hooked;
    hooked.reserve(addresses.size());
    for (size_t id = 0; id < addresses.size(); id++) {
        // The trampoline that runs the original function is known only once the hook exists,
        // and the hook needs the stub first: the stub jumps through this cell.
        auto* cell = new void*{nullptr};
        const void* entry = MakeEntryStub(observer, static_cast<u32>(id), cell);
        *cell = loader::HookFunction(addresses[id], entry);
        hooked.push_back(*cell != nullptr);
    }
    return hooked;
}

} // namespace diag
