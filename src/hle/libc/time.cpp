// <time.h>. time_t is 64-bit on both sides, but the PS4's struct tm has only the nine standard
// int fields (36 bytes); glibc's adds tm_gmtoff and tm_zone (56 bytes). Writing a host tm into
// the game's would overrun it (the game keeps a 36-byte tm on the stack), so
// every function converts between the two.

#include <ctime>

#include "common/types.h"
#include "hle/registry.h"

namespace hle::libc {

namespace {

struct Ps4Tm {
    int tm_sec;
    int tm_min;
    int tm_hour;
    int tm_mday;
    int tm_mon;
    int tm_year;
    int tm_wday;
    int tm_yday;
    int tm_isdst;
};
static_assert(sizeof(Ps4Tm) == 36);

Ps4Tm ToPs4(const std::tm& host) {
    return {host.tm_sec,  host.tm_min,  host.tm_hour, host.tm_mday, host.tm_mon,
            host.tm_year, host.tm_wday, host.tm_yday, host.tm_isdst};
}

std::tm ToHost(const Ps4Tm& tm) {
    std::tm host{};
    host.tm_sec = tm.tm_sec;
    host.tm_min = tm.tm_min;
    host.tm_hour = tm.tm_hour;
    host.tm_mday = tm.tm_mday;
    host.tm_mon = tm.tm_mon;
    host.tm_year = tm.tm_year;
    host.tm_wday = tm.tm_wday;
    host.tm_yday = tm.tm_yday;
    host.tm_isdst = tm.tm_isdst;
    return host;
}

PS4_SYSV_ABI std::time_t time(std::time_t* out) { return std::time(out); }
PS4_SYSV_ABI double difftime(std::time_t end, std::time_t start) { return std::difftime(end, start); }

// Normalizes the fields and returns the time they describe (the fields are written back).
// localtime_r/gmtime_r (POSIX) or their Windows equivalents, which take the arguments in the
// other order. True on success.
bool HostLocalTime(const std::time_t* t, std::tm* out) {
#ifdef _WIN32
    return ::localtime_s(out, t) == 0;
#else
    return ::localtime_r(t, out) != nullptr;
#endif
}
bool HostGmTime(const std::time_t* t, std::tm* out) {
#ifdef _WIN32
    return ::gmtime_s(out, t) == 0;
#else
    return ::gmtime_r(t, out) != nullptr;
#endif
}

PS4_SYSV_ABI std::time_t mktime(Ps4Tm* tm) {
    std::tm host = ToHost(*tm);
    const std::time_t result = std::mktime(&host);
    *tm = ToPs4(host);
    return result;
}

PS4_SYSV_ABI Ps4Tm* localtime(const std::time_t* t) {
    thread_local Ps4Tm result;
    std::tm host;
    if (!t || !HostLocalTime(t, &host)) {
        return nullptr;
    }
    result = ToPs4(host);
    return &result;
}

// C11 Annex K versions: (time, result), returning result or null.
PS4_SYSV_ABI Ps4Tm* localtime_s(const std::time_t* t, Ps4Tm* result) {
    std::tm host;
    if (!t || !result || !HostLocalTime(t, &host)) {
        return nullptr;
    }
    *result = ToPs4(host);
    return result;
}

PS4_SYSV_ABI Ps4Tm* gmtime_s(const std::time_t* t, Ps4Tm* result) {
    std::tm host;
    if (!t || !result || !HostGmTime(t, &host)) {
        return nullptr;
    }
    *result = ToPs4(host);
    return result;
}

PS4_SYSV_ABI size_t strftime(char* out, size_t size, const char* format, const Ps4Tm* tm) {
    const std::tm host = ToHost(*tm);
    return std::strftime(out, size, format, &host);
}

} // namespace

HLE_FUNCTION("libc", "time", time);
HLE_FUNCTION("libc", "difftime", difftime);
HLE_FUNCTION("libc", "mktime", mktime);
HLE_FUNCTION("libc", "localtime", localtime);
HLE_FUNCTION("libc", "localtime_s", localtime_s);
HLE_FUNCTION("libc", "gmtime_s", gmtime_s);
HLE_FUNCTION("libc", "strftime", strftime);

} // namespace hle::libc
