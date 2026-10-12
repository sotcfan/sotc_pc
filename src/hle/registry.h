#pragma once

#include <string_view>

#include "common/types.h"

// "HLE" (high-level emulation) is the usual name for this layer: our own implementations of the
// Sony library functions the game calls. Each one registers itself under the library name and
// the symbol name the game uses; the loader matches them to the game's imports by NID.

namespace hle {

struct Symbol {
    std::string_view library;
    std::string_view name;
    void* address;
    bool is_object;
};

// Our implementation of `nid` in `library`, or nullptr.
const Symbol* FindByNid(std::string_view library, std::string_view nid);

// Readable name for a NID, also for functions we do not implement yet; empty if unknown.
std::string_view NameOfNid(std::string_view nid);

// For code that knows a function only by NID (the shadPS4 libraries we reuse register this
// way). Must be called before the game is linked.
void RegisterByNid(std::string_view library, std::string_view nid, std::string_view name,
                   void* address);

struct Registrar {
    Registrar(std::string_view library, std::string_view name, void* address, bool is_object);
};

} // namespace hle

namespace hle {

// The address of a function the game will call. The game uses the System V calling convention;
// on Linux that is the default, but on Windows a function without PS4_SYSV_ABI would use
// Microsoft's and crash on the first call. These overloads only accept System V function
// pointers there, so a missing PS4_SYSV_ABI is a compile error.
template <typename R, typename... Args>
void* SysvAddress(R(PS4_SYSV_ABI* function)(Args...)) {
    return reinterpret_cast<void*>(function);
}
template <typename R, typename... Args>
void* SysvAddress(R(PS4_SYSV_ABI* function)(Args..., ...)) {
    return reinterpret_cast<void*>(function);
}

} // namespace hle

#define HLE_CONCAT_INNER(a, b) a##b
#define HLE_CONCAT(a, b) HLE_CONCAT_INNER(a, b)

// Registers `function` as the implementation of `name` in `library`.
#define HLE_FUNCTION(library, name, function)                                                      \
    static const ::hle::Registrar HLE_CONCAT(hle_registrar_, __COUNTER__) {                        \
        library, name, ::hle::SysvAddress(&(function)), false                                      \
    }

// Registers `object` as the variable `name` in `library`.
#define HLE_OBJECT(library, name, object)                                                          \
    static const ::hle::Registrar HLE_CONCAT(hle_registrar_, __COUNTER__) {                        \
        library, name, reinterpret_cast<void*>(&(object)), true                                    \
    }
