#pragma once

#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "loader/module.h"

namespace loader {

struct LinkerOptions {
    // When set, every import goes through the recording thunk and the first call of each one is
    // appended to this file, in order.
    std::filesystem::path trace_file;

    // If not zero, every game thread prints where it is after this many seconds.
    int dump_after_seconds = 0;
};

class Linker {
public:
    explicit Linker(LinkerOptions options);
    ~Linker();

    // The game's eboot.bin, mapped at the address its code was linked for.
    Module& LoadExecutable(const std::filesystem::path& path, u64 base);

    // Binds the module's imports to our implementations (or to recording stubs) and applies
    // the final memory permissions.
    void Link(Module& module);

    // Starts the game's entry point on a new thread and returns; the caller's thread goes on
    // to run the window's main loop.
    void Start(Module& module, std::vector<std::string> args);

    const Module* FindModule(u64 address) const;

private:
    LinkerOptions options_;
    std::vector<std::unique_ptr<Module>> modules_;
};

// Writes "0x936730 (eboot.bin)" or "stub libkernel::sceKernelOpen" into `out`.
// Safe to call from the crash handler (no allocation).
void DescribeAddress(u64 address, char* out, size_t size);

// For the crash handler: the last imports called by the current thread, and every import
// called so far that has no implementation.
void PrintRecentImports(int fd);
void PrintMissingImports(int fd);

} // namespace loader
