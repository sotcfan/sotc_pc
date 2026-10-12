#include "common/host_file.h"
#include "common/host_memory.h"
#include "common/platform.h"
#include "loader/linker.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <pthread.h>
#include <signal.h>
#include <unistd.h>

#include <atomic>
#include <cstring>
#include <deque>
#include <format>
#include <mutex>
#include <stdexcept>

#include "common/log.h"
#include "hle/process.h"
#include "hle/registry.h"
#include "loader/crash.h"
#include "loader/tls.h"

extern "C" void import_thunk();
extern "C" void import_return_zero();

namespace loader {
namespace {

struct ImportInfo {
    std::string library;
    std::string name;
    u64 target = 0; // where a call continues: our implementation or import_return_zero
    bool implemented = false;
    std::atomic<u64> calls{0};
};

// Each stub is "movabs r11, <ImportInfo*>; jmp [rip+0]; dq import_thunk", padded with int3.
constexpr size_t StubSize = 32;
constexpr size_t MaxStubs = 4096;

// Variables we do not implement yet point to a zeroed block of this size.
constexpr size_t MissingObjectSize = 0x1000;

constexpr size_t MainThreadStackSize = 8ull << 20;

struct State {
    std::mutex mutex;
    std::deque<ImportInfo> imports; // deque: addresses stay valid, stubs point to them
    u8* stubs = nullptr;
    size_t stub_count = 0;
    std::vector<std::unique_ptr<u8[]>> missing_objects;
    std::FILE* trace = nullptr;
    const std::vector<std::unique_ptr<Module>>* modules = nullptr;
};
State g_state;

struct RecentCall {
    const ImportInfo* info;
    u64 return_address;
};
constexpr size_t RecentCallCount = 16;
thread_local RecentCall t_recent_calls[RecentCallCount];
thread_local size_t t_recent_next = 0;

u64 MakeStub(const ImportInfo& info) {
    if (g_state.stub_count == MaxStubs) {
        throw std::runtime_error("too many imports for the stub area");
    }
    u8* stub = g_state.stubs + g_state.stub_count++ * StubSize;
    const u64 info_address = reinterpret_cast<u64>(&info);
    const u64 thunk_address = reinterpret_cast<u64>(&import_thunk);
    stub[0] = 0x49; // movabs r11, imm64
    stub[1] = 0xBB;
    std::memcpy(stub + 2, &info_address, 8);
    stub[10] = 0xFF; // jmp qword [rip+0]
    stub[11] = 0x25;
    std::memset(stub + 12, 0, 4);
    std::memcpy(stub + 16, &thunk_address, 8);
    std::memset(stub + 24, 0xCC, StubSize - 24);
    return reinterpret_cast<u64>(stub);
}

const ImportInfo* StubOwner(u64 address) {
    const u64 start = reinterpret_cast<u64>(g_state.stubs);
    if (!g_state.stubs || address < start || address >= start + g_state.stub_count * StubSize) {
        return nullptr;
    }
    const u64 stub = start + (address - start) / StubSize * StubSize;
    u64 info;
    std::memcpy(&info, reinterpret_cast<const void*>(stub + 2), 8);
    return reinterpret_cast<const ImportInfo*>(info);
}

void FirstCall(const ImportInfo& info, u64 return_address) {
    char caller[160];
    DescribeAddress(return_address, caller, sizeof(caller));
    if (!info.implemented) {
        LOG_WARNING("Import", "{}::{} not implemented yet (called from {}); returning 0",
                    info.library, info.name, caller);
    } else {
        LOG_DEBUG("Import", "first call to {}::{} (from {})", info.library, info.name,
                  caller);
    }
    if (g_state.trace) {
        std::scoped_lock lock{g_state.mutex};
        std::fprintf(g_state.trace, "%s::%s\t%s\t%s\n", info.library.c_str(), info.name.c_str(),
                     info.implemented ? "implemented" : "MISSING", caller);
        std::fflush(g_state.trace);
    }
}

void WriteText(int fd, const char* text) {
    size_t length = std::strlen(text);
    while (length > 0) {
        const ssize_t written = write(fd, text, length);
        if (written <= 0) {
            return;
        }
        text += written;
        length -= written;
    }
}

PS4_SYSV_ABI void ProgramExit() {
    LOG_INFO("Process", "the game called the process exit function");
}

// What the PS4 kernel passes to the entry point.
struct EntryParams {
    int argc;
    u32 padding;
    const char* argv[33];
};

struct MainThreadStart {
    Module* module;
    std::vector<std::string> args;
};

void* GuestMainThread(void* raw) {
    auto* start = static_cast<MainThreadStart*>(raw);
    logging::SetThreadName("Game:Main");
    crash::InstallAltStack();
    tls::InitializeThread();

    hle::InitializeProcess(start->module->GetProcParam());

    EntryParams params{};
    for (const std::string& arg : start->args) {
        if (params.argc == std::size(params.argv) - 1) {
            break;
        }
        params.argv[params.argc++] = arg.c_str();
    }

    using EntryFunction = PS4_SYSV_ABI void (*)(EntryParams*, PS4_SYSV_ABI void (*)());
    const auto entry = reinterpret_cast<EntryFunction>(start->module->EntryAddress());
    LOG_INFO("Process", "calling the game entry point at {:#x}", start->module->EntryAddress());
    entry(&params, &ProgramExit);

    // The entry point ends with exit(); it never returns on the PS4.
    LOG_ERROR("Process", "the game entry point returned");
    return nullptr;
}

} // namespace
} // namespace loader

// Called by import_thunk (see import_thunk.S).
extern "C" PS4_SYSV_ABI u64 OnImportCall(void* raw_info, u64 return_address) {
    using namespace loader;
    auto* info = static_cast<ImportInfo*>(raw_info);
    t_recent_calls[t_recent_next++ % RecentCallCount] = {info, return_address};
    if (info->calls.fetch_add(1, std::memory_order_relaxed) == 0) {
        FirstCall(*info, return_address);
    }
    return info->target;
}

namespace loader {

Linker::Linker(LinkerOptions options) : options_{std::move(options)} {
    void* stubs = host_memory::Allocate(MaxStubs * StubSize,
                                        host_memory::ProtRead | host_memory::ProtWrite);
    if (!stubs) {
        throw std::runtime_error("could not allocate the stub area");
    }
    g_state.stubs = static_cast<u8*>(stubs);
    g_state.modules = &modules_;
    if (!options_.trace_file.empty()) {
        g_state.trace = host_file::OpenStdio(options_.trace_file, "w");
        if (!g_state.trace) {
            throw std::runtime_error(
                std::format("could not create {}", options_.trace_file.string()));
        }
        std::fprintf(g_state.trace, "# import\tstatus\tcalled from\n");
    }
}

Linker::~Linker() {
    if (g_state.trace) {
        std::fclose(g_state.trace);
        g_state.trace = nullptr;
    }
}

Module& Linker::LoadExecutable(const std::filesystem::path& path, u64 base) {
    const ElfFile elf = ElfFile::Load(path);
    if (elf.header.e_type != ET_SCE_EXEC && elf.header.e_type != ET_SCE_DYNEXEC) {
        throw std::runtime_error(std::format("{} is not a PS4 executable", path.string()));
    }
    modules_.push_back(std::make_unique<Module>(path.filename().string(), elf, base));
    return *modules_.back();
}

void Linker::Link(Module& module) {
    const bool trace_all = g_state.trace != nullptr;
    size_t implemented = 0, stubbed = 0, objects_missing = 0;
    host_memory::Protect(g_state.stubs, MaxStubs * StubSize,
                         host_memory::ProtRead | host_memory::ProtWrite);

    module.Relocate([&](const ImportRef& ref) -> u64 {
        const hle::Symbol* symbol = hle::FindByNid(ref.library, ref.nid);
        const std::string_view known = hle::NameOfNid(ref.nid);
        const std::string name = known.empty() ? "nid_" + ref.nid : std::string(known);

        if (ref.is_object) {
            if (symbol) {
                implemented++;
                return reinterpret_cast<u64>(symbol->address);
            }
            objects_missing++;
            LOG_WARNING("Import", "variable {}::{} not implemented; points to a zeroed block",
                        ref.library, name);
            auto& block = g_state.missing_objects.emplace_back(new u8[MissingObjectSize]());
            return reinterpret_cast<u64>(block.get());
        }

        ImportInfo& info = g_state.imports.emplace_back();
        info.library = ref.library;
        info.name = name;
        if (symbol) {
            if (symbol->is_object) {
                LOG_ERROR("Import", "{}::{} is a function in the game and a variable here", ref.library, name);
            }
            implemented++;
            info.implemented = true;
            info.target = reinterpret_cast<u64>(symbol->address);
            if (!trace_all) {
                return info.target;
            }
        } else {
            stubbed++;
            info.target = reinterpret_cast<u64>(&import_return_zero);
        }
        return MakeStub(info);
    });

    host_memory::Protect(g_state.stubs, MaxStubs * StubSize,
                         host_memory::ProtRead | host_memory::ProtExec);

    size_t tls_patches = 0;
    for (const auto& [address, size] : module.ExecutableRanges()) {
        tls_patches += tls::PatchThreadPointerReads(reinterpret_cast<u8*>(address), size);
    }
    tls::SetTemplate(module.Tls());
    LOG_INFO("Loader", "{}: {} thread pointer reads (fs:[0]) redirected",
             module.Name(), tls_patches);

    module.Protect();
    LOG_INFO("Loader", "{}: {} imports implemented, {} functions and {} variables missing",
             module.Name(), implemented, stubbed, objects_missing);
}

void Linker::Start(Module& module, std::vector<std::string> args) {
    // SIGUSR1 asks for a dump of every game thread. It is blocked here, before any game thread
    // exists, so all of them inherit the mask and only the main thread receives it.
#ifndef _WIN32 // Windows has no signals to block (thread dumps suspend each thread instead)
    sigset_t dump_signal;
    sigemptyset(&dump_signal);
    sigaddset(&dump_signal, SIGUSR1);
    pthread_sigmask(SIG_BLOCK, &dump_signal, nullptr);
#endif

    auto* start = new MainThreadStart{&module, std::move(args)};
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, MainThreadStackSize);
    pthread_t thread;
    if (const int error = pthread_create(&thread, &attr, GuestMainThread, start); error != 0) {
        throw std::runtime_error(std::format("pthread_create failed: {}", std::strerror(error)));
    }
    pthread_attr_destroy(&attr);
    pthread_detach(thread);
}

const Module* Linker::FindModule(u64 address) const {
    for (const auto& module : modules_) {
        if (module->Contains(address)) {
            return module.get();
        }
    }
    return nullptr;
}

void DescribeAddress(u64 address, char* out, size_t size) {
    if (g_state.modules) {
        for (const auto& module : *g_state.modules) {
            if (module->Contains(address)) {
                std::snprintf(out, size, "%#lx (%s)", address, module->Name().c_str());
                return;
            }
        }
    }
    if (const ImportInfo* info = StubOwner(address)) {
        std::snprintf(out, size, "%#lx (stub of %s::%s)", address, info->library.c_str(),
                      info->name.c_str());
        return;
    }
    // Host code: library + offset (for addr2line) and the nearest exported symbol. dladdr is not
    // async-signal-safe (it takes the dynamic loader's lock); this is only used for crash reports
    // and thread dumps, where a hang is the worst outcome.
#ifdef _WIN32
    // Windows: the module (executable or DLL) and the offset in it; no symbol names.
    HMODULE module = nullptr;
    char module_path[MAX_PATH];
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCSTR>(address), &module) &&
        GetModuleFileNameA(module, module_path, sizeof(module_path))) {
        const char* slash = std::strrchr(module_path, '\\');
        std::snprintf(out, size, "%#llx (%s+%#llx)", static_cast<unsigned long long>(address),
                      slash ? slash + 1 : module_path,
                      static_cast<unsigned long long>(address - reinterpret_cast<u64>(module)));
        return;
    }
#else
    Dl_info dl{};
    if (dladdr(reinterpret_cast<void*>(address), &dl) && dl.dli_fname) {
        const char* slash = std::strrchr(dl.dli_fname, '/');
        const char* library = slash ? slash + 1 : dl.dli_fname;
        const u64 offset = address - reinterpret_cast<u64>(dl.dli_fbase);
        if (dl.dli_sname) {
            std::snprintf(out, size, "%#lx (%s+%#lx, %s)", address, library, offset,
                          dl.dli_sname);
        } else {
            std::snprintf(out, size, "%#lx (%s+%#lx)", address, library, offset);
        }
        return;
    }
#endif
    std::snprintf(out, size, "%#llx", static_cast<unsigned long long>(address));
}

void PrintRecentImports(int fd) {
    WriteText(fd, "Last imports called by this thread (most recent last):\n");
    char line[320], caller[160];
    for (size_t i = 0; i < RecentCallCount; i++) {
        const RecentCall& call = t_recent_calls[(t_recent_next + i) % RecentCallCount];
        if (!call.info) {
            continue;
        }
        DescribeAddress(call.return_address, caller, sizeof(caller));
        std::snprintf(line, sizeof(line), "  %s::%s%s, called from %s\n",
                      call.info->library.c_str(), call.info->name.c_str(),
                      call.info->implemented ? "" : " [not implemented]", caller);
        WriteText(fd, line);
    }
}

void PrintMissingImports(int fd) {
    WriteText(fd, "Unimplemented functions the game called:\n");
    char line[256];
    for (const ImportInfo& info : g_state.imports) {
        const u64 calls = info.calls.load(std::memory_order_relaxed);
        if (!info.implemented && calls != 0) {
            std::snprintf(line, sizeof(line), "  %s::%s (%lu calls)\n", info.library.c_str(),
                          info.name.c_str(), calls);
            WriteText(fd, line);
        }
    }
}

} // namespace loader
