#pragma once

#include <filesystem>
#include <string>

#include "common/types.h"

// Small operating system services with one interface for Linux and Windows
// (common/platform_linux.cpp, common/platform_windows.cpp).
namespace platform {

// The calling thread's id as the system knows it (gettid / GetCurrentThreadId).
u32 CurrentThreadId();

// Text of the last system error of this thread (errno / GetLastError).
std::string LastErrorText();

// sotcport's data folder (saves, caches, logs of the diagnostic tools): $XDG_DATA_HOME/sotcport
// or ~/.local/share/sotcport on Linux, %LOCALAPPDATA%\sotcport on Windows.
std::filesystem::path DataDir();

} // namespace platform
