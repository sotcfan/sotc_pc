#pragma once

// sotcport: small host helpers for the diagnostics in the GPU layer (perf counters, scheduler
// checks), with the same meaning on Linux and Windows.

#include <cstdint>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#include <execinfo.h>
#endif

namespace sotc_host {

// Load address of the module (executable or library) containing `address`, 0 if unknown.
// Addresses minus this are what addr2line expects.
inline uintptr_t ModuleBaseOf(const void* address) {
#ifdef _WIN32
    HMODULE module = nullptr;
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           static_cast<LPCSTR>(address), &module)) {
        return reinterpret_cast<uintptr_t>(module);
    }
    return 0;
#else
    Dl_info info{};
    return dladdr(address, &info) ? reinterpret_cast<uintptr_t>(info.dli_fbase) : 0;
#endif
}

// Return addresses of the calling thread's stack, innermost first.
inline int Backtrace(void** frames, int max) {
#ifdef _WIN32
    return RtlCaptureStackBackTrace(0, static_cast<DWORD>(max), frames, nullptr);
#else
    return backtrace(frames, max);
#endif
}

} // namespace sotc_host
