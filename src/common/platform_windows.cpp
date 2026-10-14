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

bool AcquireInstanceLock(const std::filesystem::path& data_dir) {
    std::error_code error;
    std::filesystem::create_directories(data_dir, error);
    const auto path = data_dir / "sotcport.lock";
    // No sharing: a second process cannot open the file while this one runs. Never closed
    // (released when the process exits, also after a crash).
    const HANDLE file = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr,
                                    OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file != INVALID_HANDLE_VALUE) {
        return true;
    }
    return GetLastError() != ERROR_SHARING_VIOLATION; // other errors: do not refuse to run
}

} // namespace platform

#endif // _WIN32
