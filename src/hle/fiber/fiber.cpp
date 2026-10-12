// Fibers (libSceFiber): execution contexts that the game switches between by hand. The BPE
// engine's job system runs its jobs on fibers.
//
// - sceFiberRun leaves the thread and enters a fiber;
// - sceFiberSwitch goes from the running fiber to another one;
// - sceFiberReturnToThread goes back to where sceFiberRun was called.
// Each switch hands one 64-bit argument to the context that resumes.
//
// The game allocates each SceFiber object (256 bytes) and the stack the fiber runs on; our state
// lives inside that object. A fiber suspended on one thread may be resumed on another, so the
// per-thread state is looked up again after every switch, never kept from before it.

#include "hle/fiber/fiber.h"

#ifdef _WIN32
#include <windows.h>
#endif

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <vector>

#include "common/log.h"
#include "common/types.h"
#include "hle/registry.h"

// fiber_switch.S: the System V convention (arguments in rdi, rsi, rdx, ecx) on every host.
extern "C" PS4_SYSV_ABI void fiber_switch(u64* save_rsp, u64 resume_rsp, u32* release_state,
                                          u32 release_value);
extern "C" void fiber_start();

namespace hle::fiber {

namespace {

using Entry = PS4_SYSV_ABI void (*)(u64 arg_on_initialize, u64 arg_on_run);

constexpr s32 ErrorNull = static_cast<s32>(0x80590001);
constexpr s32 ErrorAlignment = static_cast<s32>(0x80590002);
constexpr s32 ErrorRange = static_cast<s32>(0x80590003);
constexpr s32 ErrorInvalid = static_cast<s32>(0x80590004);
constexpr s32 ErrorPermission = static_cast<s32>(0x80590005);
constexpr s32 ErrorState = static_cast<s32>(0x80590006);

constexpr u32 Magic = 0x45424946; // "FIBE"
constexpr u64 MinimumContextSize = 512;

enum class State : u32 { Idle = 1, Running = 2, Terminated = 3 };

struct Fiber {
    u32 magic;
    State state;
    Entry entry;
    u64 arg_on_initialize;
    u8* context;
    u64 context_size;
    char name[32];
    u64 saved_rsp;
    bool started;
};
static_assert(sizeof(Fiber) <= 256, "must fit in the game's SceFiber object");

struct ThreadState {
    Fiber* current = nullptr;       // fiber running on this thread, if any
    u64 thread_rsp = 0;             // the thread's own context, saved by sceFiberRun
    u64* run_return_out = nullptr;  // where sceFiberRun's caller wants the return argument
    u64 transfer = 0;               // argument handed over by the last switch
#ifdef _WIN32
    void* own_stack_base = nullptr;  // the thread's stack bounds in its TEB, before any fiber
    void* own_stack_limit = nullptr;
#endif
};

// Not inlined: every call reads the state of the thread running at that moment.
[[gnu::noinline]] ThreadState& Current() {
    static thread_local ThreadState state;
    return state;
}

// Windows keeps the running stack's bounds in the thread's TEB (NT_TIB StackBase/StackLimit),
// and code there relies on them (stack probes of large frames, exception dispatch). A fiber's
// stack is game memory elsewhere, so the bounds follow every switch: the target fiber's stack,
// or the thread's own when it goes back (null). Nothing to do on Linux.
void UseStackOf(const Fiber* target);

struct Info {
    u64 size;
    Entry entry;
    u64 arg_on_initialize;
    void* context;
    u64 context_size;
    char name[32];
    u64 context_size_margin;
    u8 padding[48];
};
static_assert(sizeof(Info) == 128);

// Every initialized fiber, for SuspendedFibers().
std::mutex g_fibers_mutex;
std::vector<Fiber*> g_fibers;

// Builds, at the top of the fiber's stack, the frame fiber_switch() expects, so that the first
// switch into the fiber "returns" into fiber_start with the fiber in r12.
void PrepareStack(Fiber* fiber) {
    const u64 top = ((reinterpret_cast<u64>(fiber->context) + fiber->context_size) & ~u64{15}) - 16;
    auto* frame = reinterpret_cast<u64*>(top - 64);
    u32 mxcsr;
    u16 x87_control;
    asm volatile("stmxcsr %0" : "=m"(mxcsr));
    asm volatile("fnstcw %0" : "=m"(x87_control));
    std::memset(frame, 0, 64 + 16);
    std::memcpy(frame, &mxcsr, 4);
    std::memcpy(reinterpret_cast<u8*>(frame) + 4, &x87_control, 2);
    frame[4] = reinterpret_cast<u64>(fiber);        // r12
    frame[7] = reinterpret_cast<u64>(&fiber_start); // return address
    fiber->saved_rsp = reinterpret_cast<u64>(frame);
}

std::atomic_ref<u32> StateOf(Fiber* fiber) {
    return std::atomic_ref<u32>(*reinterpret_cast<u32*>(&fiber->state));
}

// Claims an idle fiber for this thread. Several threads may try to run the same fiber; only
// one gets it.
bool Claim(Fiber* fiber) {
    u32 expected = static_cast<u32>(State::Idle);
    return StateOf(fiber).compare_exchange_strong(expected, static_cast<u32>(State::Running));
}

void UseStackOf(const Fiber* target) {
#ifdef _WIN32
    ThreadState& thread = Current();
    auto* tib = reinterpret_cast<NT_TIB*>(NtCurrentTeb());
    if (!thread.own_stack_base) {
        thread.own_stack_base = tib->StackBase;
        thread.own_stack_limit = tib->StackLimit;
    }
    if (target) {
        tib->StackLimit = target->context;
        tib->StackBase = static_cast<u8*>(target->context) + target->context_size;
    } else {
        tib->StackBase = thread.own_stack_base;
        tib->StackLimit = thread.own_stack_limit;
    }
#else
    (void)target;
#endif
}

// Starts or resumes `fiber` (already claimed) on this thread. The current context is saved in
// *save_rsp; if `leaving` is given, it becomes Idle only after that save (see fiber_switch.S).
void Enter(Fiber* fiber, u64 argument, u64* save_rsp, Fiber* leaving) {
    ThreadState& thread = Current();
    thread.current = fiber;
    thread.transfer = argument;
    if (!fiber->started) {
        fiber->started = true;
        PrepareStack(fiber);
    }
    UseStackOf(fiber);
    fiber_switch(save_rsp, fiber->saved_rsp,
                 leaving ? reinterpret_cast<u32*>(&leaving->state) : nullptr,
                 static_cast<u32>(State::Idle));
}

s32 Check(const Fiber* fiber) {
    if (!fiber) {
        return ErrorNull;
    }
    return fiber->magic == Magic ? 0 : ErrorInvalid;
}

PS4_SYSV_ABI s32 _sceFiberInitializeImpl(Fiber* fiber, const char* name, Entry entry,
                                         u64 arg_on_initialize, void* context, u64 context_size,
                                         const void* /*options*/, u32 /*build_version*/) {
    if (!fiber || !name || !entry || !context) {
        return ErrorNull;
    }
    if (reinterpret_cast<u64>(fiber) % 8 != 0 || reinterpret_cast<u64>(context) % 16 != 0) {
        return ErrorAlignment;
    }
    if (context_size < MinimumContextSize || context_size % 16 != 0) {
        return ErrorRange;
    }
    std::memset(fiber, 0, sizeof(Fiber));
    fiber->magic = Magic;
    fiber->state = State::Idle;
    fiber->entry = entry;
    fiber->arg_on_initialize = arg_on_initialize;
    fiber->context = static_cast<u8*>(context);
    fiber->context_size = context_size;
    std::strncpy(fiber->name, name, sizeof(fiber->name) - 1);
    {
        std::scoped_lock lock{g_fibers_mutex};
        if (std::ranges::find(g_fibers, fiber) == g_fibers.end()) {
            g_fibers.push_back(fiber);
        }
    }
    LOG_DEBUG("Fibers", "fiber \"{}\" created (stack {:#x} bytes)", fiber->name, context_size);
    return 0;
}

PS4_SYSV_ABI s32 sceFiberOptParamInitialize(u32* options) {
    if (!options) {
        return ErrorNull;
    }
    *options = Magic;
    return 0;
}

PS4_SYSV_ABI s32 sceFiberFinalize(Fiber* fiber) {
    if (const s32 error = Check(fiber)) {
        return error;
    }
    if (fiber->state == State::Running) {
        return ErrorState;
    }
    fiber->magic = 0;
    std::scoped_lock lock{g_fibers_mutex};
    std::erase(g_fibers, fiber);
    return 0;
}

PS4_SYSV_ABI s32 sceFiberRun(Fiber* fiber, u64 arg_on_run_to, u64* arg_on_return) {
    if (const s32 error = Check(fiber)) {
        return error;
    }
    ThreadState& thread = Current();
    if (thread.current) {
        return ErrorPermission; // already inside a fiber: that is what sceFiberSwitch is for
    }
    if (!Claim(fiber)) {
        LOG_WARNING("Fibers", "sceFiberRun: \"{}\" is not idle (state {})", fiber->name,
                    static_cast<u32>(fiber->state));
        return ErrorState;
    }
    thread.run_return_out = arg_on_return;
    LOG_TRACE("Fibers", "run \"{}\"", fiber->name);
    Enter(fiber, arg_on_run_to, &thread.thread_rsp, nullptr);
    // Back on the thread: a fiber called sceFiberReturnToThread (or its entry returned).
    return 0;
}

PS4_SYSV_ABI s32 sceFiberSwitch(Fiber* target, u64 arg_on_run_to, u64* arg_on_run) {
    if (const s32 error = Check(target)) {
        return error;
    }
    Fiber* self = Current().current;
    if (!self) {
        LOG_WARNING("Fibers", "sceFiberSwitch outside a fiber");
        return ErrorPermission;
    }
    if (!Claim(target)) {
        LOG_WARNING("Fibers", "sceFiberSwitch from \"{}\" to \"{}\": target is not idle (state {})",
                    self->name, target->name, static_cast<u32>(target->state));
        return ErrorState;
    }
    LOG_TRACE("Fibers", "switch {} -> {}", static_cast<void*>(self), static_cast<void*>(target));
    Enter(target, arg_on_run_to, &self->saved_rsp, self);
    // Resumed, possibly on another thread.
    if (arg_on_run) {
        *arg_on_run = Current().transfer;
    }
    return 0;
}

PS4_SYSV_ABI s32 sceFiberReturnToThread(u64 arg_on_return, u64* arg_on_run) {
    ThreadState& thread = Current();
    Fiber* self = thread.current;
    if (!self) {
        return ErrorPermission;
    }
    if (thread.run_return_out) {
        *thread.run_return_out = arg_on_return;
    }
    thread.current = nullptr;
    LOG_TRACE("Fibers", "return \"{}\"", self->name);
    UseStackOf(nullptr);
    fiber_switch(&self->saved_rsp, thread.thread_rsp, reinterpret_cast<u32*>(&self->state),
                 static_cast<u32>(State::Idle));
    // Resumed, possibly on another thread.
    if (arg_on_run) {
        *arg_on_run = Current().transfer;
    }
    return 0;
}

PS4_SYSV_ABI s32 sceFiberGetInfo(Fiber* fiber, Info* info) {
    if (const s32 error = Check(fiber)) {
        return error;
    }
    if (!info) {
        return ErrorNull;
    }
    if (info->size != sizeof(Info)) {
        return ErrorInvalid;
    }
    info->entry = fiber->entry;
    info->arg_on_initialize = fiber->arg_on_initialize;
    info->context = fiber->context;
    info->context_size = fiber->context_size;
    std::memcpy(info->name, fiber->name, sizeof(info->name));
    info->context_size_margin = 0; // unknown: the stack is not measured
    return 0;
}

PS4_SYSV_ABI s32 sceFiberRename(Fiber* fiber, const char* name) {
    if (const s32 error = Check(fiber)) {
        return error;
    }
    if (!name) {
        return ErrorNull;
    }
    std::strncpy(fiber->name, name, sizeof(fiber->name) - 1);
    fiber->name[sizeof(fiber->name) - 1] = '\0';
    return 0;
}

} // namespace

std::vector<SuspendedFiber> SuspendedFibers() {
    std::vector<SuspendedFiber> result;
    std::scoped_lock lock{g_fibers_mutex};
    for (const Fiber* fiber : g_fibers) {
        if (fiber->magic == Magic && fiber->started && fiber->state == State::Idle) {
            // fiber_switch's saved frame: [mxcsr+fcw] r15 r14 r13 r12 rbx rbp return-address
            const auto* frame = reinterpret_cast<const u64*>(fiber->saved_rsp);
            result.push_back({fiber->name, frame[6], frame[7]});
        }
    }
    return result;
}

std::vector<std::string> BusyFibers() {
    std::vector<std::string> result;
    std::scoped_lock lock{g_fibers_mutex};
    for (const Fiber* fiber : g_fibers) {
        if (fiber->magic != Magic || fiber->state == State::Idle) {
            continue;
        }
        result.push_back(std::format("\"{}\": {}", fiber->name,
                                     fiber->state == State::Running ? "running" : "finished"));
    }
    return result;
}

// Called by fiber_start (fiber_switch.S) the first time a fiber runs.
extern "C" [[noreturn]] PS4_SYSV_ABI void FiberMain(void* raw_fiber) {
    auto* fiber = static_cast<Fiber*>(raw_fiber);
    fiber->entry(fiber->arg_on_initialize, Current().transfer);

    // The entry function returned: the fiber is finished and the thread takes over again.
    ThreadState& thread = Current();
    LOG_DEBUG("Fibers", "fiber \"{}\" finished", fiber->name);
    fiber->state = State::Terminated;
    thread.current = nullptr;
    if (thread.run_return_out) {
        *thread.run_return_out = 0;
    }
    u64 unused;
    UseStackOf(nullptr);
    fiber_switch(&unused, thread.thread_rsp, nullptr, 0);
    __builtin_unreachable();
}

HLE_FUNCTION("libSceFiber", "_sceFiberInitializeImpl", _sceFiberInitializeImpl);
HLE_FUNCTION("libSceFiber", "sceFiberOptParamInitialize", sceFiberOptParamInitialize);
HLE_FUNCTION("libSceFiber", "sceFiberFinalize", sceFiberFinalize);
HLE_FUNCTION("libSceFiber", "sceFiberRun", sceFiberRun);
HLE_FUNCTION("libSceFiber", "sceFiberSwitch", sceFiberSwitch);
HLE_FUNCTION("libSceFiber", "sceFiberReturnToThread", sceFiberReturnToThread);
HLE_FUNCTION("libSceFiber", "sceFiberGetInfo", sceFiberGetInfo);
HLE_FUNCTION("libSceFiber", "sceFiberRename", sceFiberRename);

} // namespace hle::fiber
