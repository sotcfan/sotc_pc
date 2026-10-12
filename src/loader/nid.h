#pragma once

#include <string>
#include <string_view>

#include "common/types.h"

namespace loader {

// PS4 modules refer to functions and variables by NID instead of by name: the first 8 bytes of
// SHA-1(name + fixed suffix), written as 11 characters of a base64 variant.
std::string NameToNid(std::string_view name);

// Library and module ids inside symbol names ("NID#lib#module") use the same alphabet.
bool DecodeId(std::string_view text, u64& out);

} // namespace loader
