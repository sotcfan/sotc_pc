// Character classification. The compiler inlined isdigit(), tolower() and friends into the game
// as lookups in Dinkumware's tables, which the game gets from _Getpctype() and similar. The
// tables hold one short per character from -1 (EOF) to 255; we build them for the "C" locale.

#include <array>

#include "common/types.h"
#include "hle/registry.h"

namespace hle::libc {

namespace {

// Dinkumware's classification bits.
constexpr s16 XD = 0x001;  // hexadecimal digit
constexpr s16 UP = 0x002;  // upper case letter
constexpr s16 SP = 0x004;  // space character ' '
constexpr s16 PU = 0x008;  // punctuation
constexpr s16 LO = 0x010;  // lower case letter
constexpr s16 DI = 0x020;  // decimal digit
constexpr s16 CN = 0x040;  // \t \n \v \f \r
constexpr s16 BB = 0x080;  // other control characters
constexpr s16 XB = 0x400;  // extra blank ('\t')

struct Tables {
    std::array<s16, 257> classes{};
    std::array<s16, 257> lower{};
    std::array<s16, 257> upper{};

    Tables() {
        for (int c = -1; c < 256; c++) {
            const size_t i = c + 1;
            lower[i] = static_cast<s16>(c >= 'A' && c <= 'Z' ? c + 32 : c);
            upper[i] = static_cast<s16>(c >= 'a' && c <= 'z' ? c - 32 : c);
            s16 bits = 0;
            if (c >= 0 && c < 0x20) {
                bits |= BB;
                if (c >= '\t' && c <= '\r') {
                    bits |= CN;
                }
                if (c == '\t') {
                    bits |= XB;
                }
            } else if (c == 0x7F) {
                bits |= BB;
            } else if (c == ' ') {
                bits |= SP;
            } else if (c >= '0' && c <= '9') {
                bits |= DI | XD;
            } else if (c >= 'A' && c <= 'Z') {
                bits |= UP | (c <= 'F' ? XD : 0);
            } else if (c >= 'a' && c <= 'z') {
                bits |= LO | (c <= 'f' ? XD : 0);
            } else if (c > ' ' && c < 0x7F) {
                bits |= PU;
            }
            classes[i] = bits;
        }
    }
};

const Tables g_tables;

// Each returns a pointer to the entry of character 0, so [-1] is EOF.
PS4_SYSV_ABI const s16* Getpctype() { return g_tables.classes.data() + 1; }
PS4_SYSV_ABI const s16* Getptolower() { return g_tables.lower.data() + 1; }
PS4_SYSV_ABI const s16* Getptoupper() { return g_tables.upper.data() + 1; }

// iswctype() for the 16-bit wchar_t: only ASCII is classified.
PS4_SYSV_ABI int Iswctype(u16 c, s16 mask) {
    return c < 0x80 ? (g_tables.classes[c + 1] & mask) : 0;
}

// Per-thread conversion states used by the multibyte functions.
thread_local u64 t_mbstate[16];
thread_local u64 t_wcstate[16];
PS4_SYSV_ABI void* Getpmbstate() { return t_mbstate; }
PS4_SYSV_ABI void* Getpwcstate() { return t_wcstate; }

} // namespace

HLE_FUNCTION("libc", "_Getpctype", Getpctype);
HLE_FUNCTION("libc", "_Getptolower", Getptolower);
HLE_FUNCTION("libc", "_Getptoupper", Getptoupper);
HLE_FUNCTION("libc", "_Iswctype", Iswctype);
HLE_FUNCTION("libc", "_Getpmbstate", Getpmbstate);
HLE_FUNCTION("libc", "_Getpwcstate", Getpwcstate);

} // namespace hle::libc
