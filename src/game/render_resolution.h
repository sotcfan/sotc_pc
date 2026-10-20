#pragma once

// The resolution the game renders at (see render_resolution.cpp).

#include <string>
#include <vector>

namespace game {

// Between Linker::LoadExecutable and Linker::Link (the game's code is still writable).
// SOTC_RESOLUTION may add the ConVar that picks the 4K mode to the game's command line.
void InstallRenderResolution(std::vector<std::string>& game_args);

} // namespace game
