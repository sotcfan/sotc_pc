// The printf/scanf family over System V variable arguments (see sysv_varargs.h).
//
// Linux: the host's va_list has the same layout, so a copy of the list goes to the host's
// vsnprintf/vsscanf (SOTC_SYSV_FORMAT=1 uses our formatter instead, to test it on Linux).
// Windows: each conversion of the format is read from the list here and formatted by the host's
// snprintf on its own, with the PS4's sizes: `long` is 64 bits on the PS4 and 32 on Windows, so
// l-sized integers are passed as long long; wide strings are 16-bit on both.

#include "hle/libc/sysv_varargs.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace hle::libc {

u64 NextInteger(SysvVaList& args) {
    if (args.gp_offset < 48) {
        u64 value;
        std::memcpy(&value, args.reg_save_area + args.gp_offset, 8);
        args.gp_offset += 8;
        return value;
    }
    u64 value;
    std::memcpy(&value, args.overflow_arg_area, 8);
    args.overflow_arg_area += 8;
    return value;
}

double NextDouble(SysvVaList& args) {
    double value;
    if (args.fp_offset < 176) {
        std::memcpy(&value, args.reg_save_area + args.fp_offset, 8);
        args.fp_offset += 16;
        return value;
    }
    std::memcpy(&value, args.overflow_arg_area, 8);
    args.overflow_arg_area += 8;
    return value;
}

long double NextLongDouble(SysvVaList& args) {
    // Always in memory, 16-byte aligned, 16 bytes (80-bit x87 value in the low 10).
    auto address = reinterpret_cast<uintptr_t>(args.overflow_arg_area);
    address = (address + 15) & ~uintptr_t{15};
    long double value = 0;
    std::memcpy(&value, reinterpret_cast<const void*>(address), std::min<size_t>(sizeof(value), 16));
    args.overflow_arg_area = reinterpret_cast<u8*>(address + 16);
    return value;
}

namespace {

bool UseOwnFormatter() {
#ifdef _WIN32
    return true;
#else
    static const bool own = [] {
        const char* value = std::getenv("SOTC_SYSV_FORMAT");
        return value && *value == '1';
    }();
    return own;
#endif
}

#ifndef _WIN32
// On Linux a SysvVaList is the host's va_list element.
static_assert(sizeof(va_list) == sizeof(SysvVaList));
#endif

// Appends one formatted conversion. `spec` is the host printf spec ("%-08.3lld").
template <typename T>
void AppendHost(std::string& out, const std::string& spec, T value) {
    char small[128];
    const int length = std::snprintf(small, sizeof(small), spec.c_str(), value);
    if (length < 0) {
        return;
    }
    if (static_cast<size_t>(length) < sizeof(small)) {
        out.append(small, static_cast<size_t>(length));
        return;
    }
    std::string big(static_cast<size_t>(length) + 1, '\0');
    std::snprintf(big.data(), big.size(), spec.c_str(), value);
    out.append(big.data(), static_cast<size_t>(length));
}

std::string Utf16ToUtf8(const char16_t* text, int max) {
    std::string out;
    for (int i = 0; text[i] && (max < 0 || i < max); ++i) {
        const u32 c = text[i];
        if (c < 0x80) {
            out += static_cast<char>(c);
        } else if (c < 0x800) {
            out += static_cast<char>(0xC0 | (c >> 6));
            out += static_cast<char>(0x80 | (c & 0x3F));
        } else {
            out += static_cast<char>(0xE0 | (c >> 12));
            out += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (c & 0x3F));
        }
    }
    return out;
}

std::string FormatOwn(const char* format, SysvVaList args) {
    std::string out;
    for (const char* p = format; *p;) {
        if (*p != '%') {
            out += *p++;
            continue;
        }
        const char* start = p++;
        if (*p == '%') {
            out += '%';
            ++p;
            continue;
        }
        std::string spec = "%";
        while (*p && std::strchr("-+ #0'", *p)) {
            spec += *p++;
        }
        if (*p == '*') {
            spec += std::to_string(static_cast<int>(NextInteger(args)));
            ++p;
        } else {
            while (*p >= '0' && *p <= '9') {
                spec += *p++;
            }
        }
        int precision = -1;
        if (*p == '.') {
            spec += *p++;
            if (*p == '*') {
                precision = static_cast<int>(NextInteger(args));
                spec += std::to_string(precision);
                ++p;
            } else {
                precision = 0;
                while (*p >= '0' && *p <= '9') {
                    precision = precision * 10 + (*p - '0');
                    spec += *p++;
                }
            }
        }
        // Length modifier: how wide the argument is on the PS4.
        enum class Size { Int, Char, Short, Long, LongLong, LongDouble } size = Size::Int;
        if (p[0] == 'h' && p[1] == 'h') {
            size = Size::Char;
            p += 2;
        } else if (*p == 'h') {
            size = Size::Short;
            ++p;
        } else if (p[0] == 'l' && p[1] == 'l') {
            size = Size::LongLong;
            p += 2;
        } else if (*p == 'l' || *p == 'j' || *p == 'z' || *p == 't' || *p == 'q') {
            size = Size::Long; // 64 bits on the PS4
            ++p;
        } else if (*p == 'L') {
            size = Size::LongDouble;
            ++p;
        }
        const char conversion = *p;
        if (!conversion) {
            out.append(start); // a lone '%' at the end: print as is
            break;
        }
        ++p;
        switch (conversion) {
        case 'd':
        case 'i': {
            const u64 raw = NextInteger(args);
            long long value;
            switch (size) {
            case Size::Char: value = static_cast<signed char>(raw); break;
            case Size::Short: value = static_cast<short>(raw); break;
            case Size::Int: value = static_cast<int>(raw); break;
            default: value = static_cast<long long>(raw); break;
            }
            AppendHost(out, spec + "ll" + conversion, value);
            break;
        }
        case 'u':
        case 'o':
        case 'x':
        case 'X': {
            const u64 raw = NextInteger(args);
            unsigned long long value;
            switch (size) {
            case Size::Char: value = static_cast<unsigned char>(raw); break;
            case Size::Short: value = static_cast<unsigned short>(raw); break;
            case Size::Int: value = static_cast<unsigned int>(raw); break;
            default: value = raw; break;
            }
            AppendHost(out, spec + "ll" + conversion, value);
            break;
        }
        case 'c':
            if (size == Size::Long) {
                const char16_t wide[2] = {static_cast<char16_t>(NextInteger(args)), 0};
                AppendHost(out, spec + "s", Utf16ToUtf8(wide, 1).c_str());
            } else {
                AppendHost(out, spec + "c", static_cast<int>(NextInteger(args)));
            }
            break;
        case 's': {
            const u64 pointer = NextInteger(args);
            if (size == Size::Long) {
                // PS4 wide strings are 16-bit (wchar_t is 2 bytes there).
                const auto* wide = reinterpret_cast<const char16_t*>(pointer);
                const std::string text = wide ? Utf16ToUtf8(wide, precision) : "(null)";
                AppendHost(out, spec + "s", text.c_str());
            } else {
                const auto* text = reinterpret_cast<const char*>(pointer);
                AppendHost(out, spec + "s", text ? text : "(null)");
            }
            break;
        }
        case 'p':
            AppendHost(out, spec + "p", reinterpret_cast<void*>(NextInteger(args)));
            break;
        case 'f':
        case 'F':
        case 'e':
        case 'E':
        case 'g':
        case 'G':
        case 'a':
        case 'A':
            if (size == Size::LongDouble) {
                AppendHost(out, spec + "L" + conversion, NextLongDouble(args));
            } else {
                AppendHost(out, spec + conversion, NextDouble(args));
            }
            break;
        case 'n': {
            const u64 pointer = NextInteger(args);
            const auto count = static_cast<long long>(out.size());
            switch (size) {
            case Size::Char: *reinterpret_cast<signed char*>(pointer) = static_cast<signed char>(count); break;
            case Size::Short: *reinterpret_cast<short*>(pointer) = static_cast<short>(count); break;
            case Size::Int: *reinterpret_cast<int*>(pointer) = static_cast<int>(count); break;
            default: *reinterpret_cast<long long*>(pointer) = count; break;
            }
            break;
        }
        default:
            out.append(start, p); // unknown conversion: print as is
            break;
        }
    }
    return out;
}

// The format with every l-sized integer conversion widened to ll, for a host where `long` has
// 32 bits (scanf writes through the pointers with the host's sizes).
std::string WidenLongs(const char* format) {
    std::string out;
    for (const char* p = format; *p;) {
        if (*p != '%') {
            out += *p++;
            continue;
        }
        out += *p++;
        if (*p == '%') { // "%%"
            out += *p++;
            continue;
        }
        while (*p && std::strchr("*0123456789", *p)) {
            out += *p++;
        }
        if (*p == '[') { // a scanset is copied whole
            out += *p++;
            if (*p == '^') {
                out += *p++;
            }
            if (*p == ']') {
                out += *p++;
            }
            while (*p && *p != ']') {
                out += *p++;
            }
            if (*p) {
                out += *p++;
            }
            continue;
        }
        if (p[0] == 'l' && p[1] != 'l' && p[1] && std::strchr("diouxXn", p[1])) {
            out += "ll";
            ++p;
        }
    }
    return out;
}

} // namespace

std::string Format(const char* format, const SysvVaList* args) {
    if (UseOwnFormatter()) {
        return FormatOwn(format, *args);
    }
#ifndef _WIN32
    va_list copy;
    std::memcpy(copy, args, sizeof(SysvVaList));
    va_list measure;
    va_copy(measure, copy);
    const int length = std::vsnprintf(nullptr, 0, format, measure);
    va_end(measure);
    std::string out(length > 0 ? static_cast<size_t>(length) : 0, '\0');
    if (length > 0) {
        std::vsnprintf(out.data(), out.size() + 1, format, copy);
    }
    return out;
#else
    return {};
#endif
}

int FormatToBuffer(char* buffer, std::size_t size, const char* format, const SysvVaList* args) {
    if (!UseOwnFormatter()) {
#ifndef _WIN32
        va_list copy;
        std::memcpy(copy, args, sizeof(SysvVaList));
        return std::vsnprintf(buffer, size, format, copy);
#endif
    }
    const std::string text = FormatOwn(format, *args);
    if (size > 0) {
        const size_t n = std::min(text.size(), size - 1);
        std::memcpy(buffer, text.data(), n);
        buffer[n] = '\0';
    }
    return static_cast<int>(text.size());
}

int Scan(const char* text, const char* format, const SysvVaList* args) {
#ifndef _WIN32
    if (!UseOwnFormatter()) {
        va_list copy;
        std::memcpy(copy, args, sizeof(SysvVaList));
        return std::vsscanf(text, format, copy);
    }
#endif
    // Every scanf argument is a pointer: take enough of them for any format the game uses and
    // pass them all (the host's sscanf ignores the extra ones).
    SysvVaList list = *args;
    void* p[24];
    for (void*& pointer : p) {
        pointer = reinterpret_cast<void*>(NextInteger(list));
    }
    const std::string widened = WidenLongs(format);
    return std::sscanf(text, widened.c_str(), p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7],
                       p[8], p[9], p[10], p[11], p[12], p[13], p[14], p[15], p[16], p[17], p[18],
                       p[19], p[20], p[21], p[22], p[23]);
}

} // namespace hle::libc
