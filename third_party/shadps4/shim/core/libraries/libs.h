#pragma once

// shadPS4 registers its library functions by NID; here they go into sotcport's registry
// (src/hle/registry.h), so the loader binds the game's imports to them.

#include <string_view>

#include "common/types.h"

namespace Core::Loader {
class SymbolsResolver;
}

namespace hle {
void RegisterByNid(std::string_view library, std::string_view nid, std::string_view name,
                   void* address);
}

#define LIB_FUNCTION(nid, lib, libversion, mod, function)                                          \
    ::hle::RegisterByNid(lib, nid, #function, reinterpret_cast<void*>(function))

#define LIB_OBJ(nid, lib, libversion, mod, obj)                                                    \
    ::hle::RegisterByNid(lib, nid, #obj, reinterpret_cast<void*>(obj))
