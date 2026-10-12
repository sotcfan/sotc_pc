#pragma once

#include <filesystem>

namespace hle::savedata {

// Saves go under <data_dir>/saves. Call before the game starts.
void Initialize(const std::filesystem::path& data_dir);

} // namespace hle::savedata
