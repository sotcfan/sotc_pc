#include "loader/crash.h"

#ifdef _WIN32
#include <windows.h>
#include <tlhelp32.h>
#include <io.h>
#else
#include <dirent.h>
#include <execinfo.h>
#include <pthread.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <ucontext.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <utility>

#include "common/log.h"
#include "common/types.h"
#include "hle/fiber/fiber.h"
#include "hle/kernel/kernel.h"
#include "loader/linker.h"
#include "loader/tls.h"

#ifndef _WIN32
extern "C" char __executable_start; // start of our executable (linker-provided)
#endif

namespace crash {

namespace {

// The GPU layer's write-tracking fault handler does real work on this stack (cache
// invalidation, logging): 64 KiB overflowed, and an overflow inside a handler kills the process
// without a report. A guard page below the stack turns a future overflow into a clean fault.
constexpr size_t AltStackSize = 1024 * 1024;
constexpr size_t AltStackGuard = 4096;
constexpr int MaxFrames = 32;
constexpr int StderrFd = 2;

template <typename Handler>
struct HandlerList {
    static constexpr int Max = 8;
    std::pair<unsigned, Handler> entries[Max];
    std::atomic<int> count{0};

    void Add(Handler handler, unsigned priority) {
        int n = count.load();
        if (n == Max) {
            return;
        }
        // Kept sorted by priority; additions happen at start-up, before any fault.
        int i = n;
        while (i > 0 && entries[i - 1].first > priority) {
            entries[i] = entries[i - 1];
            i--;
        }
        entries[i] = {priority, handler};
        count.store(n + 1);
    }
};
HandlerList<AccessFaultHandler> g_access_handlers;
HandlerList<IllegalInstructionHandler> g_illegal_handlers;

// Serializes the output of threads dumping at the same time.
std::atomic_flag g_dump_lock = ATOMIC_FLAG_INIT;

// The general registers of a crashed thread, as both systems give them.
struct Registers {
    u64 rax, rbx, rcx, rdx, rsi, rdi, rbp, rsp, r8, r9, r10, r11, r12, r13, r14, r15, rip;
};

void Write(const char* text) {
    size_t length = std::strlen(text);
    while (length > 0) {
#ifdef _WIN32
        const int written = _write(StderrFd, text, static_cast<unsigned>(length));
#else
        const ssize_t written = write(StderrFd, text, length);
#endif
        if (written <= 0) {
            return;
        }
        text += written;
        length -= static_cast<size_t>(written);
    }
}

// Reads memory without faulting again if the address is bad.
bool SafeRead(u64 address, void* out, size_t size) {
#ifdef _WIN32
    SIZE_T read = 0;
    return ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(address), out, size,
                             &read) &&
           read == size;
#else
    iovec local{out, size};
    iovec remote{reinterpret_cast<void*>(address), size};
    return process_vm_readv(getpid(), &local, 1, &remote, 1, 0) == static_cast<ssize_t>(size);
#endif
}

// Start of our executable: host addresses minus this are what addr2line expects.
const void* ExecutableBase() {
#ifdef _WIN32
    return GetModuleHandleW(nullptr);
#else
    return &__executable_start;
#endif
}

[[noreturn]] void ExitNow(int code) {
    std::fflush(nullptr);
    std::_Exit(code);
}

void PrintStack(u64 frame) {
    char line[256], where[160];
    // The game's code keeps frame pointers (push rbp; mov rbp, rsp), so the return addresses
    // can be found by following the rbp chain.
    Write("Call stack (by rbp):\n");
    for (int i = 0; i < MaxFrames && frame != 0; i++) {
        u64 pair[2]; // saved rbp, return address
        if (!SafeRead(frame, pair, sizeof(pair)) || pair[1] == 0) {
            break;
        }
        loader::DescribeAddress(pair[1], where, sizeof(where));
        std::snprintf(line, sizeof(line), "  #%-2d %s\n", i, where);
        Write(line);
        if (pair[0] <= frame) {
            break;
        }
        frame = pair[0];
    }
}

void PrintStackAndImports(u64 frame) {
    PrintStack(frame);
    loader::PrintRecentImports(StderrFd);
    loader::PrintMissingImports(StderrFd);
    Write("====\n");
}

// How deep the current thread is in the fault handler. A fault inside a fault handler (the GPU
// layer's write tracking does real work there) comes back to it and gets reported, instead of
// the process dying silently.
thread_local int t_handler_depth = 0;

struct DepthGuard {
    DepthGuard() {
        t_handler_depth++;
    }
    ~DepthGuard() {
        t_handler_depth--;
    }
};

// Runs the registered handlers for a fault. True if one of them resolved it.
bool TryHandlers(bool access_fault, void* context, void* address) {
    if (t_handler_depth > 1) {
        Write("\n==== fault INSIDE a fault handler (see the stack below) ====\n");
        return false;
    }
    if (access_fault) {
        for (int i = 0; i < g_access_handlers.count; i++) {
            if (g_access_handlers.entries[i].second(context, address)) {
                return true;
            }
        }
    } else {
        for (int i = 0; i < g_illegal_handlers.count; i++) {
            if (g_illegal_handlers.entries[i].second(context)) {
                return true;
            }
        }
    }
    return false;
}

// The crash report: what happened, the registers, the game's call stack, the last imports.
void Report(const char* what, u64 address, const Registers& r) {
    char line[512], where[160];
    std::snprintf(line, sizeof(line), "\n==== CRASH in thread [%s]: %s, address %#llx ====\n",
                  logging::CurrentThreadName(), what, static_cast<unsigned long long>(address));
    Write(line);
    loader::DescribeAddress(r.rip, where, sizeof(where));
    std::snprintf(line, sizeof(line), "rip = %s (sotcport base %p)\n", where, ExecutableBase());
    Write(line);
    using ull = unsigned long long;
    std::snprintf(line, sizeof(line),
                  "rax=%016llx rbx=%016llx rcx=%016llx rdx=%016llx\n"
                  "rsi=%016llx rdi=%016llx rbp=%016llx rsp=%016llx\n"
                  "r8 =%016llx r9 =%016llx r10=%016llx r11=%016llx\n"
                  "r12=%016llx r13=%016llx r14=%016llx r15=%016llx\n",
                  ull(r.rax), ull(r.rbx), ull(r.rcx), ull(r.rdx), ull(r.rsi), ull(r.rdi),
                  ull(r.rbp), ull(r.rsp), ull(r.r8), ull(r.r9), ull(r.r10), ull(r.r11),
                  ull(r.r12), ull(r.r13), ull(r.r14), ull(r.r15));
    Write(line);
    PrintStackAndImports(r.rbp);
}

// Prints the fibers and event flags the game is waiting on (both systems).
void DumpGameState() {
    Write("\n==== THREAD DUMP ====\n");
    loader::PrintMissingImports(StderrFd);
    for (const auto& line : hle::kernel::WaitedEventFlags()) {
        Write(("event flag " + line + "\n").c_str());
    }
    for (const auto& line : hle::fiber::BusyFibers()) {
        Write(("fiber " + line + "\n").c_str());
    }
    for (const auto& fiber : hle::fiber::SuspendedFibers()) {
        char line[256], where[160];
        loader::DescribeAddress(fiber.resume_address, where, sizeof(where));
        std::snprintf(line, sizeof(line), "\n--- suspended fiber \"%s\", resumes at %s\n",
                      fiber.name.c_str(), where);
        Write(line);
        PrintStack(fiber.frame_pointer);
    }
}

// std::terminate: an exception nobody caught (in our code or the GPU layer's), e.g. a
// std::system_error from a lock. Reported like a crash, with the thrower's stack (still intact:
// terminate runs before any unwinding).
[[noreturn]] void TerminateHandler() {
    char line[512];
    const char* what = "no exception (std::terminate called directly)";
    std::string text;
    if (const std::exception_ptr current = std::current_exception()) {
        try {
            std::rethrow_exception(current);
        } catch (const std::exception& exception) {
            text = exception.what();
            what = text.c_str();
        } catch (...) {
            what = "an exception that is not a std::exception";
        }
    }
    std::snprintf(line, sizeof(line),
                  "\n==== UNCAUGHT EXCEPTION in thread [%s]: %s ====\n(sotcport base %p)\n",
                  logging::CurrentThreadName(), what, ExecutableBase());
    Write(line);
    // The unwinder's view (host frames, through code without frame pointers) ...
    void* frames[48];
#ifdef _WIN32
    const int count = RtlCaptureStackBackTrace(0, 48, frames, nullptr);
#else
    const int count = backtrace(frames, 48);
#endif
    Write("Call stack (unwinder):\n");
    for (int i = 0; i < count; i++) {
        char where[160];
        loader::DescribeAddress(reinterpret_cast<u64>(frames[i]), where, sizeof(where));
        std::snprintf(line, sizeof(line), "  #%-2d %s\n", i, where);
        Write(line);
    }
    // ... and the rbp chain, which goes on into the game's code.
    PrintStackAndImports(reinterpret_cast<u64>(__builtin_frame_address(0)));
    ExitNow(134);
}

#ifdef _WIN32

// ---- Windows: vectored exception handler ----
//
// The fault handlers receive the EXCEPTION_POINTERS as their context, which is what the GPU
// layer's code expects on Windows (shadPS4's common/signal_context.cpp).

Registers FromContext(const CONTEXT& c) {
    return {c.Rax, c.Rbx, c.Rcx, c.Rdx, c.Rsi, c.Rdi, c.Rbp, c.Rsp, c.R8,
            c.R9,  c.R10, c.R11, c.R12, c.R13, c.R14, c.R15, c.Rip};
}

const char* ExceptionName(DWORD code) {
    switch (code) {
    case EXCEPTION_ACCESS_VIOLATION: return "access violation (invalid memory access)";
    case EXCEPTION_ILLEGAL_INSTRUCTION: return "illegal instruction";
    case EXCEPTION_PRIV_INSTRUCTION: return "privileged instruction";
    case EXCEPTION_INT_DIVIDE_BY_ZERO: return "integer division by zero";
    case EXCEPTION_INT_OVERFLOW: return "integer overflow";
    case EXCEPTION_STACK_OVERFLOW: return "stack overflow";
    case EXCEPTION_BREAKPOINT: return "breakpoint (int3)";
    case EXCEPTION_IN_PAGE_ERROR: return "in-page error";
    default: return nullptr; // not a crash (C++ exceptions, debugger messages, thread names)
    }
}

LONG CALLBACK VectoredHandler(EXCEPTION_POINTERS* info) {
    const DWORD code = info->ExceptionRecord->ExceptionCode;
    const char* name = ExceptionName(code);
    if (!name) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    const DepthGuard depth;
    if (code == EXCEPTION_ACCESS_VIOLATION || code == EXCEPTION_IN_PAGE_ERROR) {
        // ExceptionInformation[0]: 0 read, 1 write, 8 execute; [1]: the address.
        void* address = reinterpret_cast<void*>(info->ExceptionRecord->ExceptionInformation[1]);
        if (TryHandlers(true, info, address)) {
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        Report(name, reinterpret_cast<u64>(address), FromContext(*info->ContextRecord));
        ExitNow(139);
    }
    if (code == EXCEPTION_ILLEGAL_INSTRUCTION && TryHandlers(false, info, nullptr)) {
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    Report(name, reinterpret_cast<u64>(info->ExceptionRecord->ExceptionAddress),
           FromContext(*info->ContextRecord));
    ExitNow(134);
}

#else

// ---- Linux: signals ----

Registers FromContext(const ucontext_t& context) {
    const greg_t* r = context.uc_mcontext.gregs;
    return {u64(r[REG_RAX]), u64(r[REG_RBX]), u64(r[REG_RCX]), u64(r[REG_RDX]),
            u64(r[REG_RSI]), u64(r[REG_RDI]), u64(r[REG_RBP]), u64(r[REG_RSP]),
            u64(r[REG_R8]),  u64(r[REG_R9]),  u64(r[REG_R10]), u64(r[REG_R11]),
            u64(r[REG_R12]), u64(r[REG_R13]), u64(r[REG_R14]), u64(r[REG_R15]),
            u64(r[REG_RIP])};
}

const char* SignalName(int signal) {
    switch (signal) {
    case SIGSEGV: return "SIGSEGV (invalid memory access)";
    case SIGBUS: return "SIGBUS (bus error)";
    case SIGILL: return "SIGILL (illegal instruction)";
    case SIGFPE: return "SIGFPE (arithmetic error)";
    case SIGTRAP: return "SIGTRAP (int3)";
    default: return "signal";
    }
}

// SIGUSR2, sent to each thread by DumpAllThreads(): prints where it is.
void DumpHandler(int, siginfo_t*, void* raw_context) {
    const auto* context = static_cast<ucontext_t*>(raw_context);
    const greg_t* r = context->uc_mcontext.gregs;
    while (g_dump_lock.test_and_set(std::memory_order_acquire)) {
    }
    char line[256], where[160], kernel_name[17] = {};
    prctl(PR_GET_NAME, kernel_name);
    loader::DescribeAddress(r[REG_RIP], where, sizeof(where));
    // The game keeps per-thread values right below its thread pointer (the job system's worker
    // index is at -0x10).
    const u64 tp = loader::tls::ThreadPointer();
    u64 tls_words[4] = {};
    if (tp) {
        SafeRead(tp - 0x20, tls_words, sizeof(tls_words));
    }
    std::snprintf(line, sizeof(line),
                  "\n--- thread [%s / %s], rip = %s, game TLS [-0x20..0) = %lx %lx %lx %lx\n",
                  logging::CurrentThreadName(), kernel_name, where, tls_words[0], tls_words[1],
                  tls_words[2], tls_words[3]);
    Write(line);
    PrintStack(r[REG_RBP]);
    loader::PrintRecentImports(StderrFd);
    g_dump_lock.clear(std::memory_order_release);
}

void Handler(int signal, siginfo_t* info, void* raw_context) {
    const DepthGuard depth;
    const bool access = signal == SIGSEGV || signal == SIGBUS;
    if ((access || signal == SIGILL) && TryHandlers(access, raw_context, info->si_addr)) {
        return;
    }
    Report(SignalName(signal), reinterpret_cast<u64>(info->si_addr),
           FromContext(*static_cast<ucontext_t*>(raw_context)));
    _exit(128 + signal);
}

#endif

} // namespace

void Install() {
    std::set_terminate(TerminateHandler);
#ifdef _WIN32
    // First in line: the GPU layer's page tracking must see its faults before anything else.
    AddVectoredExceptionHandler(1, VectoredHandler);
    InstallAltStack();
#else
    // The process maps several GB; a core file would be huge. Run under gdb to inspect a crash.
    const rlimit no_core{0, 0};
    setrlimit(RLIMIT_CORE, &no_core);

    InstallAltStack();
    struct sigaction action{};
    action.sa_sigaction = Handler;
    action.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_NODEFER;
    sigemptyset(&action.sa_mask);
    for (int signal : {SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGTRAP}) {
        sigaction(signal, &action, nullptr);
    }

    struct sigaction dump{};
    dump.sa_sigaction = DumpHandler;
    dump.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_RESTART;
    sigemptyset(&dump.sa_mask);
    sigaction(SIGUSR2, &dump, nullptr);
#endif
}

void AddAccessFaultHandler(AccessFaultHandler handler, unsigned priority) {
    g_access_handlers.Add(handler, priority);
}

void AddIllegalInstructionHandler(IllegalInstructionHandler handler, unsigned priority) {
    g_illegal_handlers.Add(handler, priority);
}

void DumpAllThreads() {
    while (g_dump_lock.test_and_set(std::memory_order_acquire)) {
    }
    DumpGameState();
    g_dump_lock.clear(std::memory_order_release);
#ifdef _WIN32
    // Every other thread of the process is stopped in turn and its registers read. Per-thread
    // data (last imports, game TLS) is only known to the thread itself, so it is not printed.
    const DWORD self = GetCurrentThreadId();
    const DWORD process = GetCurrentProcessId();
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snapshot == INVALID_HANDLE_VALUE) {
        return;
    }
    THREADENTRY32 entry{};
    entry.dwSize = sizeof(entry);
    for (BOOL more = Thread32First(snapshot, &entry); more; more = Thread32Next(snapshot, &entry)) {
        if (entry.th32OwnerProcessID != process || entry.th32ThreadID == self) {
            continue;
        }
        HANDLE thread = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT |
                                       THREAD_QUERY_LIMITED_INFORMATION,
                                   FALSE, entry.th32ThreadID);
        if (!thread) {
            continue;
        }
        if (SuspendThread(thread) != static_cast<DWORD>(-1)) {
            CONTEXT context{};
            context.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
            if (GetThreadContext(thread, &context)) {
                std::string name = "?";
                PWSTR description = nullptr;
                if (SUCCEEDED(GetThreadDescription(thread, &description)) && description) {
                    name.clear();
                    for (const wchar_t* c = description; *c; ++c) {
                        name += static_cast<char>(*c < 128 ? *c : '?');
                    }
                    LocalFree(description);
                }
                char line[256], where[160];
                loader::DescribeAddress(context.Rip, where, sizeof(where));
                std::snprintf(line, sizeof(line), "\n--- thread %lu [%s], rip = %s\n",
                              entry.th32ThreadID, name.c_str(), where);
                Write(line);
                PrintStack(context.Rbp);
            }
            ResumeThread(thread);
        }
        CloseHandle(thread);
    }
    CloseHandle(snapshot);
#else
    // Every thread of the process, also those created by the GPU layer and by libraries.
    const pid_t self = static_cast<pid_t>(syscall(SYS_gettid));
    if (DIR* tasks = opendir("/proc/self/task")) {
        while (const dirent* entry = readdir(tasks)) {
            const pid_t tid = static_cast<pid_t>(std::atoi(entry->d_name));
            if (tid > 0 && tid != self) {
                syscall(SYS_tgkill, getpid(), tid, SIGUSR2);
            }
        }
        closedir(tasks);
    }
#endif
}

void InstallAltStack() {
#ifdef _WIN32
    // Windows runs the exception handler on the faulting thread's own stack; keep enough of it
    // in reserve that a stack overflow can still be reported.
    ULONG reserve = 64 * 1024;
    SetThreadStackGuarantee(&reserve);
#else
    void* memory = mmap(nullptr, AltStackGuard + AltStackSize, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (memory == MAP_FAILED) {
        return;
    }
    mprotect(memory, AltStackGuard, PROT_NONE);
    stack_t stack{};
    stack.ss_sp = static_cast<u8*>(memory) + AltStackGuard;
    stack.ss_size = AltStackSize;
    stack.ss_flags = 0;
    sigaltstack(&stack, nullptr);
#endif
}

void Fatal(const char* reason) {
    char line[512], where[160];
    loader::DescribeAddress(reinterpret_cast<u64>(__builtin_return_address(0)), where,
                            sizeof(where));
    std::snprintf(line, sizeof(line), "\n==== FATAL ERROR in thread [%s]: %s ====\ncalled from %s\n",
                  logging::CurrentThreadName(), reason, where);
    Write(line);
    PrintStackAndImports(reinterpret_cast<u64>(__builtin_frame_address(0)));
    ExitNow(134);
}

} // namespace crash
