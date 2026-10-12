// Process start and exit, and the small part of the C++ runtime the game uses. The game is
// built without C++ exceptions, so the functions that would throw just report and abort.

#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

#include "common/log.h"
#include "common/types.h"
#include "hle/libc/libc.h"
#include "hle/registry.h"
#include "loader/crash.h"

namespace hle::libc {

namespace {

struct ExitHandler {
    PS4_SYSV_ABI void (*function)(void*);
    void* argument;
};

std::mutex g_exit_mutex;
std::vector<ExitHandler> g_exit_handlers;

[[noreturn]] void Fatal(const std::string& what) {
    crash::Fatal(what.c_str());
}

// Called by the entry point before anything else; on the PS4 it sets up `environ`.
PS4_SYSV_ABI void init_env() {}

PS4_SYSV_ABI int cxa_atexit(PS4_SYSV_ABI void (*function)(void*), void* argument, void*) {
    std::scoped_lock lock{g_exit_mutex};
    g_exit_handlers.push_back({function, argument});
    return 0;
}

PS4_SYSV_ABI int atexit(PS4_SYSV_ABI void (*function)()) {
    // A function without argument ignores the one it is given.
    return cxa_atexit(reinterpret_cast<PS4_SYSV_ABI void (*)(void*)>(function), nullptr, nullptr);
}

[[noreturn]] PS4_SYSV_ABI void exit(int code) {
    LOG_INFO("Process", "the game called exit({})", code);
    RunExitHandlers();
    std::fflush(nullptr);
    _exit(code);
}

[[noreturn]] PS4_SYSV_ABI void quick_exit(int code) {
    LOG_INFO("Process", "the game called quick_exit({})", code);
    std::fflush(nullptr);
    _exit(code);
}

[[noreturn]] PS4_SYSV_ABI void abort() {
    Fatal("the game called abort()");
}

// Sony hook run with main()'s return value, before exit().
PS4_SYSV_ABI void catchReturnFromMain(int code) {
    LOG_INFO("Process", "the game's main() returned {}", code);
}

// Guards for function-local statics (Itanium C++ ABI): the first byte of the 64-bit guard is
// set once the object is constructed. The compiler checks it inline and only calls acquire
// when it is still zero. One recursive mutex keeps it simple and lets a static's constructor
// initialize other statics on the same thread.
std::recursive_mutex g_guard_mutex;

std::atomic_ref<u8> GuardByte(u64* guard) {
    return std::atomic_ref<u8>(*reinterpret_cast<u8*>(guard));
}

PS4_SYSV_ABI int cxa_guard_acquire(u64* guard) {
    if (GuardByte(guard).load(std::memory_order_acquire) != 0) {
        return 0;
    }
    g_guard_mutex.lock();
    if (GuardByte(guard).load(std::memory_order_acquire) != 0) {
        g_guard_mutex.unlock();
        return 0;
    }
    return 1; // caller constructs the object, then calls release (or abort)
}

PS4_SYSV_ABI void cxa_guard_release(u64* guard) {
    GuardByte(guard).store(1, std::memory_order_release);
    g_guard_mutex.unlock();
}

PS4_SYSV_ABI void cxa_pure_virtual() {
    Fatal("pure virtual function call");
}

PS4_SYSV_ABI void throw_bad_alloc() {
    Fatal("std::bad_alloc (out of memory)");
}

PS4_SYSV_ABI void throw_length_error(const char* message) {
    Fatal(std::format("std::length_error: {}", message ? message : ""));
}

PS4_SYSV_ABI void throw_out_of_range(const char* message) {
    Fatal(std::format("std::out_of_range: {}", message ? message : ""));
}

// std::exception::_Raise() const: Dinkumware's "throw *this" when exceptions are disabled.
PS4_SYSV_ABI void exception_raise(const void*) {
    Fatal("C++ exception thrown (the game does not handle exceptions)");
}

// Marker variable the game imports to require the libc.
u64 g_need_libc = 1;

} // namespace

void RunExitHandlers() {
    std::vector<ExitHandler> handlers;
    {
        std::scoped_lock lock{g_exit_mutex};
        handlers.swap(g_exit_handlers);
    }
    for (auto it = handlers.rbegin(); it != handlers.rend(); ++it) {
        it->function(it->argument);
    }
}

HLE_FUNCTION("libc", "_init_env", init_env);
HLE_FUNCTION("libc", "atexit", atexit);
HLE_FUNCTION("libc", "__cxa_atexit", cxa_atexit);
HLE_FUNCTION("libc", "exit", exit);
HLE_FUNCTION("libc", "quick_exit", quick_exit);
HLE_FUNCTION("libc", "abort", abort);
HLE_FUNCTION("libc", "catchReturnFromMain", catchReturnFromMain);
HLE_FUNCTION("libc", "__cxa_guard_acquire", cxa_guard_acquire);
HLE_FUNCTION("libc", "__cxa_guard_release", cxa_guard_release);
HLE_FUNCTION("libc", "__cxa_pure_virtual", cxa_pure_virtual);
HLE_FUNCTION("libc", "_ZSt11_Xbad_allocv", throw_bad_alloc);
HLE_FUNCTION("libc", "_ZSt14_Xlength_errorPKc", throw_length_error);
HLE_FUNCTION("libc", "_ZSt14_Xout_of_rangePKc", throw_out_of_range);
HLE_FUNCTION("libc", "_ZNKSt9exception6_RaiseEv", exception_raise);
HLE_OBJECT("libc", "Need_sceLibc", g_need_libc);

} // namespace hle::libc
