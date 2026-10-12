// Windows implementation of common/platform.h.
#ifdef _WIN32

#include "common/platform.h"

#include <windows.h>

#include <cstdlib>
#include <format>

namespace platform {

u32 CurrentThreadId() {
    return static_cast<u32>(GetCurrentThreadId());
}

std::string LastErrorText() {
    const DWORD error = GetLastError();
    char text[256] = {};
    const DWORD length =
        FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, error,
                       0, text, sizeof(text), nullptr);
    std::string result(text, length);
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) {
        result.pop_back();
    }
    return std::format("{} (error {})", result, error);
}

std::filesystem::path DataDir() {
    if (const wchar_t* local = _wgetenv(L"LOCALAPPDATA"); local && *local) {
        return std::filesystem::path(local) / "sotcport";
    }
    return "sotcport-data";
}

} // namespace platform

#endif // _WIN32
