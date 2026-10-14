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

// Takes the lock that only one sotcport at a time may hold (a file in `data_dir`, held open
// until the process exits). False if another sotcport holds it: two of them at once corrupted
// each other's rendering (2026-10-14).
bool AcquireInstanceLock(const std::filesystem::path& data_dir);

} // namespace platform
