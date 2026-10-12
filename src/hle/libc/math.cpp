// <math.h>. Results come from the host math library and can differ from the PS4's in the last
// bit; the game does not depend on bit-exact math for anything we have seen so far.

#include <cmath>

#include "common/types.h"
#include "hle/registry.h"

namespace hle::libc {

namespace {

PS4_SYSV_ABI float acosf(float x) { return std::acos(x); }
PS4_SYSV_ABI float asinf(float x) { return std::asin(x); }
PS4_SYSV_ABI float atanf(float x) { return std::atan(x); }
PS4_SYSV_ABI float atan2f(float y, float x) { return std::atan2(y, x); }
PS4_SYSV_ABI double atan2(double y, double x) { return std::atan2(y, x); }
PS4_SYSV_ABI float tanf(float x) { return std::tan(x); }
PS4_SYSV_ABI float expf(float x) { return std::exp(x); }
PS4_SYSV_ABI float exp2f(float x) { return std::exp2(x); }
PS4_SYSV_ABI float powf(float x, float y) { return std::pow(x, y); }
PS4_SYSV_ABI double pow(double x, double y) { return std::pow(x, y); }
PS4_SYSV_ABI float fmodf(float x, float y) { return std::fmod(x, y); }
PS4_SYSV_ABI double fmod(double x, double y) { return std::fmod(x, y); }
PS4_SYSV_ABI float roundf(float x) { return std::round(x); }
PS4_SYSV_ABI float frexpf(float x, int* e) { return std::frexp(x, e); }
PS4_SYSV_ABI double ldexp(double x, int e) { return std::ldexp(x, e); }
PS4_SYSV_ABI double modf(double x, double* i) { return std::modf(x, i); }
PS4_SYSV_ABI int isfinite(double x) { return std::isfinite(x); }
PS4_SYSV_ABI int isfinitef(float x) { return std::isfinite(x); }

// Dinkumware internals behind sinf/cosf and logf/log10f/log2f, which the compiler inlined
// into the game. `quadrant` 0 gives the sine, 1 the cosine.
PS4_SYSV_ABI double Sin(double x, unsigned int quadrant) {
    return quadrant != 0 ? std::cos(x) : std::sin(x);
}

PS4_SYSV_ABI float FSin(float x, unsigned int quadrant) {
    return quadrant != 0 ? std::cos(x) : std::sin(x);
}

// `base` > 0 gives log10, 0 the natural log, < 0 log2.
PS4_SYSV_ABI float FLog(float x, int base) {
    if (base > 0) {
        return std::log10(x);
    }
    return base == 0 ? std::log(x) : std::log2(x);
}

} // namespace

HLE_FUNCTION("libc", "acosf", acosf);
HLE_FUNCTION("libc", "asinf", asinf);
HLE_FUNCTION("libc", "atanf", atanf);
HLE_FUNCTION("libc", "atan2f", atan2f);
HLE_FUNCTION("libc", "atan2", atan2);
HLE_FUNCTION("libc", "tanf", tanf);
HLE_FUNCTION("libc", "expf", expf);
HLE_FUNCTION("libc", "exp2f", exp2f);
HLE_FUNCTION("libc", "powf", powf);
HLE_FUNCTION("libc", "pow", pow);
HLE_FUNCTION("libc", "fmodf", fmodf);
HLE_FUNCTION("libc", "fmod", fmod);
HLE_FUNCTION("libc", "roundf", roundf);
HLE_FUNCTION("libc", "frexpf", frexpf);
HLE_FUNCTION("libc", "ldexp", ldexp);
HLE_FUNCTION("libc", "modf", modf);
HLE_FUNCTION("libc", "__isfinite", isfinite);
HLE_FUNCTION("libc", "__isfinitef", isfinitef);
HLE_FUNCTION("libc", "_Sin", Sin);
HLE_FUNCTION("libc", "_FSin", FSin);
HLE_FUNCTION("libc", "_FLog", FLog);

} // namespace hle::libc
