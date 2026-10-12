// <string.h> and <wchar.h> memory functions. Plain forwarding to the host C library, except for
// the wide-character ones: wchar_t is 2 bytes on the PS4 and 4 bytes on Linux.

#include <strings.h>

#include <cstring>

#include "common/types.h"
#include "diag/copy_watch.h"
#include "hle/registry.h"

namespace hle::libc {

namespace {

using ps4_wchar = char16_t;

// Copies can be attributed to their call site in the game (diag/copy_watch.h): imports reach
// these functions with the game's return address on the stack.
PS4_SYSV_ABI void* memcpy(void* d, const void* s, size_t n) {
    if (diag::g_watch_copies) [[unlikely]] {
        diag::NoteCopy(d, s, n, __builtin_return_address(0));
    }
    return std::memcpy(d, s, n);
}
PS4_SYSV_ABI void* memmove(void* d, const void* s, size_t n) {
    if (diag::g_watch_copies) [[unlikely]] {
        diag::NoteCopy(d, s, n, __builtin_return_address(0));
    }
    return std::memmove(d, s, n);
}
PS4_SYSV_ABI void* memset(void* d, int c, size_t n) { return std::memset(d, c, n); }
PS4_SYSV_ABI int memcmp(const void* a, const void* b, size_t n) { return std::memcmp(a, b, n); }
PS4_SYSV_ABI const void* memchr(const void* s, int c, size_t n) { return std::memchr(s, c, n); }

PS4_SYSV_ABI size_t strlen(const char* s) { return std::strlen(s); }
PS4_SYSV_ABI size_t strnlen(const char* s, size_t n) { return ::strnlen(s, n); }
PS4_SYSV_ABI int strcmp(const char* a, const char* b) { return std::strcmp(a, b); }
PS4_SYSV_ABI int strncmp(const char* a, const char* b, size_t n) { return std::strncmp(a, b, n); }
PS4_SYSV_ABI int strcasecmp(const char* a, const char* b) { return ::strcasecmp(a, b); }
PS4_SYSV_ABI int strncasecmp(const char* a, const char* b, size_t n) { return ::strncasecmp(a, b, n); }
PS4_SYSV_ABI char* strcpy(char* d, const char* s) { return std::strcpy(d, s); }
PS4_SYSV_ABI char* strncpy(char* d, const char* s, size_t n) { return std::strncpy(d, s, n); }
PS4_SYSV_ABI char* strcat(char* d, const char* s) { return std::strcat(d, s); }
PS4_SYSV_ABI char* strncat(char* d, const char* s, size_t n) { return std::strncat(d, s, n); }
PS4_SYSV_ABI const char* strchr(const char* s, int c) { return std::strchr(s, c); }
PS4_SYSV_ABI const char* strrchr(const char* s, int c) { return std::strrchr(s, c); }
PS4_SYSV_ABI const char* strstr(const char* s, const char* f) { return std::strstr(s, f); }
PS4_SYSV_ABI size_t strspn(const char* s, const char* set) { return std::strspn(s, set); }
PS4_SYSV_ABI size_t strcspn(const char* s, const char* set) { return std::strcspn(s, set); }

PS4_SYSV_ABI ps4_wchar* wmemcpy(ps4_wchar* d, const ps4_wchar* s, size_t n) {
    std::memcpy(d, s, n * sizeof(ps4_wchar));
    return d;
}

PS4_SYSV_ABI ps4_wchar* wmemmove(ps4_wchar* d, const ps4_wchar* s, size_t n) {
    std::memmove(d, s, n * sizeof(ps4_wchar));
    return d;
}

PS4_SYSV_ABI ps4_wchar* wmemset(ps4_wchar* d, ps4_wchar c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        d[i] = c;
    }
    return d;
}

} // namespace

HLE_FUNCTION("libc", "memcpy", memcpy);
HLE_FUNCTION("libc", "memmove", memmove);
HLE_FUNCTION("libc", "memset", memset);
HLE_FUNCTION("libc", "memcmp", memcmp);
HLE_FUNCTION("libc", "memchr", memchr);
HLE_FUNCTION("libc", "strlen", strlen);
HLE_FUNCTION("libc", "strnlen", strnlen);
HLE_FUNCTION("libc", "strcmp", strcmp);
HLE_FUNCTION("libc", "strncmp", strncmp);
HLE_FUNCTION("libc", "strcasecmp", strcasecmp);
HLE_FUNCTION("libc", "strncasecmp", strncasecmp);
HLE_FUNCTION("libc", "strcpy", strcpy);
HLE_FUNCTION("libc", "strncpy", strncpy);
HLE_FUNCTION("libc", "strcat", strcat);
HLE_FUNCTION("libc", "strncat", strncat);
HLE_FUNCTION("libc", "strchr", strchr);
HLE_FUNCTION("libc", "strrchr", strrchr);
HLE_FUNCTION("libc", "strstr", strstr);
HLE_FUNCTION("libc", "strspn", strspn);
HLE_FUNCTION("libc", "strcspn", strcspn);
HLE_FUNCTION("libc", "wmemcpy", wmemcpy);
HLE_FUNCTION("libc", "wmemmove", wmemmove);
HLE_FUNCTION("libc", "wmemset", wmemset);

} // namespace hle::libc
