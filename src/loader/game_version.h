#pragma once

#include <filesystem>

namespace loader {

// Checks that `eboot` is the game executable sotcport was made for: the port hooks functions
// inside it and relies on its data layout, so another version would crash or misbehave in ways
// that look like bugs of the port. Prints what it found and what it expects.
// Returns false when the executable is unknown, unless SOTC_ALLOW_UNKNOWN_EBOOT=1.
bool CheckGameVersion(const std::filesystem::path& eboot);

} // namespace loader
