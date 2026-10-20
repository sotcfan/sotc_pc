#pragma once

#include <filesystem>

namespace hle::np {

// The trophies unlocked so far are kept in <data_dir>/trophies.txt (see trophy.cpp). Call before
// the game starts.
void InitializeTrophies(const std::filesystem::path& data_dir);

} // namespace hle::np
