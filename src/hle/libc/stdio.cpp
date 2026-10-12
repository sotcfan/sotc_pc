// <stdio.h>. Files the game opens are host FILEs, with paths translated by the kernel file
// system layer. The game's stdout and stderr are Dinkumware objects (_Stdout, _Stderr) whose
// addresses it passes around; we hand out placeholders and map them back to the host streams.

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>

#include "common/host_file.h"
#include "common/log.h"
#include "common/types.h"
#include "hle/kernel/errno.h"
#include "hle/kernel/filesystem.h"
#include "hle/libc/libc.h"
#include "hle/libc/sysv_varargs.h"
#include "hle/registry.h"

namespace hle::libc {

namespace {

struct GuestFile {
    u8 unused[0x100];
};
GuestFile g_stdout{};
GuestFile g_stderr{};

PS4_SYSV_ABI void* fopen(const char* path, const char* mode) {
    const kernel::HostPath host = kernel::ToHostPath(path);
    const bool writes = std::strpbrk(mode, "wa+") != nullptr;
    if (!host.valid || (host.read_only && writes)) {
        LOG_WARNING("libc", "fopen(\"{}\", \"{}\") refused: {}", path, mode,
                    host.valid ? "read-only location" : "path outside the mount points");
        kernel::SetErrno(host.valid ? kernel::BSD_EROFS : kernel::BSD_ENOENT);
        return nullptr;
    }
    std::FILE* file = host_file::OpenStdio(host.path, mode);
    if (!file) {
        kernel::SetErrno(kernel::ToBsdErrno(errno));
    }
    LOG_DEBUG("libc", "fopen(\"{}\", \"{}\") -> {}", path, mode, file ? "ok" : "failed");
    return file;
}

PS4_SYSV_ABI int fclose(void* f) { return std::fclose(HostFile(f)); }
PS4_SYSV_ABI size_t fread(void* d, size_t s, size_t n, void* f) { return std::fread(d, s, n, HostFile(f)); }
PS4_SYSV_ABI size_t fwrite(const void* d, size_t s, size_t n, void* f) { return std::fwrite(d, s, n, HostFile(f)); }
// `long` is 64 bits on the PS4 (32 on Windows): s64, and the host's 64-bit calls.
PS4_SYSV_ABI int fseek(void* f, s64 offset, int whence) {
#ifdef _WIN32
    return _fseeki64(HostFile(f), offset, whence);
#else
    return std::fseek(HostFile(f), offset, whence);
#endif
}
PS4_SYSV_ABI s64 ftell(void* f) {
#ifdef _WIN32
    return _ftelli64(HostFile(f));
#else
    return std::ftell(HostFile(f));
#endif
}
PS4_SYSV_ABI int feof(void* f) { return std::feof(HostFile(f)); }
PS4_SYSV_ABI int fflush(void* f) { return std::fflush(f ? HostFile(f) : nullptr); }
PS4_SYSV_ABI char* fgets(char* s, int n, void* f) { return std::fgets(s, n, HostFile(f)); }
PS4_SYSV_ABI int fputs(const char* s, void* f) { return std::fputs(s, HostFile(f)); }
PS4_SYSV_ABI int fputc(int c, void* f) { return std::fputc(c, HostFile(f)); }
PS4_SYSV_ABI int puts(const char* s) { return std::puts(s); }
PS4_SYSV_ABI int putchar(int c) { return std::putchar(c); }

// The PS4's v*printf functions leave the caller's va_list untouched, and the game relies on it:
// it formats the same list twice, first to measure the text. Ours always work on a copy
// (sysv_varargs.h). A va_list arrives as a pointer to the System V list.
PS4_SYSV_ABI int vfprintf(void* f, const char* format, const SysvVaList* args) {
    const std::string text = Format(format, args);
    return static_cast<int>(std::fwrite(text.data(), 1, text.size(), HostFile(f)));
}

PS4_SYSV_ABI int vsnprintf(char* buffer, size_t size, const char* format, const SysvVaList* args) {
    return FormatToBuffer(buffer, size, format, args);
}

PS4_SYSV_ABI int vsprintf(char* buffer, const char* format, const SysvVaList* args) {
    return FormatToBuffer(buffer, static_cast<size_t>(-1) >> 1, format, args);
}

// C11 Annex K: like vsnprintf, but an error if the text does not fit.
PS4_SYSV_ABI int vsprintf_s(char* buffer, size_t size, const char* format,
                            const SysvVaList* args) {
    const int result = vsnprintf(buffer, size, format, args);
    if (result < 0 || static_cast<size_t>(result) >= size) {
        if (size > 0) {
            buffer[0] = '\0';
        }
        return -1;
    }
    return result;
}

} // namespace

// The variadic functions: SOTC_SYSV_VARIADIC_ENTRY builds the list and calls these. `used`:
// only the stubs' assembly calls them, which link-time optimization does not see.
extern "C" {
__attribute__((used)) PS4_SYSV_ABI int sotc_fprintf_va(void* f, const char* format, const SysvVaList* args) {
    return vfprintf(f, format, args);
}
__attribute__((used)) PS4_SYSV_ABI int sotc_printf_va(const char* format, const SysvVaList* args) {
    return vfprintf(&g_stdout, format, args);
}
__attribute__((used)) PS4_SYSV_ABI int sotc_snprintf_va(char* buffer, size_t size, const char* format,
                                  const SysvVaList* args) {
    return FormatToBuffer(buffer, size, format, args);
}
__attribute__((used)) PS4_SYSV_ABI int sotc_sprintf_va(char* buffer, const char* format, const SysvVaList* args) {
    return vsprintf(buffer, format, args);
}
__attribute__((used)) PS4_SYSV_ABI int sotc_sscanf_va(const char* text, const char* format, const SysvVaList* args) {
    return Scan(text, format, args);
}
__attribute__((used)) PS4_SYSV_ABI int sotc_fscanf_va(void* f, const char* format, const SysvVaList* args) {
    // Only used on small text files: read a line and scan it.
    char line[4096];
    if (!std::fgets(line, sizeof(line), HostFile(f))) {
        return EOF;
    }
    return Scan(line, format, args);
}
} // extern "C"

namespace {
SOTC_SYSV_VARIADIC_ENTRY(sotc_fprintf, sotc_fprintf_va, 2, rdx)
SOTC_SYSV_VARIADIC_ENTRY(sotc_printf, sotc_printf_va, 1, rsi)
SOTC_SYSV_VARIADIC_ENTRY(sotc_snprintf, sotc_snprintf_va, 3, rcx)
SOTC_SYSV_VARIADIC_ENTRY(sotc_sprintf, sotc_sprintf_va, 2, rdx)
SOTC_SYSV_VARIADIC_ENTRY(sotc_sscanf, sotc_sscanf_va, 2, rdx)
SOTC_SYSV_VARIADIC_ENTRY(sotc_fscanf, sotc_fscanf_va, 2, rdx)
} // namespace

std::FILE* HostFile(void* guest_file) {
    if (guest_file == &g_stdout) {
        return stdout;
    }
    if (guest_file == &g_stderr) {
        return stderr;
    }
    return static_cast<std::FILE*>(guest_file);
}

HLE_OBJECT("libc", "_Stdout", g_stdout);
HLE_OBJECT("libc", "_Stderr", g_stderr);
HLE_FUNCTION("libc", "fopen", fopen);
HLE_FUNCTION("libc", "fclose", fclose);
HLE_FUNCTION("libc", "fread", fread);
HLE_FUNCTION("libc", "fwrite", fwrite);
HLE_FUNCTION("libc", "fseek", fseek);
HLE_FUNCTION("libc", "ftell", ftell);
HLE_FUNCTION("libc", "feof", feof);
HLE_FUNCTION("libc", "fflush", fflush);
HLE_FUNCTION("libc", "fgets", fgets);
HLE_FUNCTION("libc", "fputs", fputs);
HLE_FUNCTION("libc", "fputc", fputc);
HLE_FUNCTION("libc", "puts", puts);
HLE_FUNCTION("libc", "putchar", putchar);
HLE_FUNCTION("libc", "vfprintf", vfprintf);
HLE_FUNCTION("libc", "fprintf", sotc_fprintf);
HLE_FUNCTION("libc", "printf", sotc_printf);
HLE_FUNCTION("libc", "vsnprintf", vsnprintf);
HLE_FUNCTION("libc", "vsprintf", vsprintf);
HLE_FUNCTION("libc", "vsprintf_s", vsprintf_s);
HLE_FUNCTION("libc", "snprintf", sotc_snprintf);
HLE_FUNCTION("libc", "sprintf", sotc_sprintf);
HLE_FUNCTION("libc", "sscanf", sotc_sscanf);
HLE_FUNCTION("libc", "fscanf", sotc_fscanf);

} // namespace hle::libc
