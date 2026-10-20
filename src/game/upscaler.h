#pragma once

// The game's camera jitter, for a temporal upscaler (see upscaler.cpp).

#include <string>
#include <vector>

namespace game {

// Between Linker::LoadExecutable and Linker::Link (the game's code is still writable), after
// InstallRenderResolution (which sets the upscaler's sizes). With FSR it turns the game's own
// temporal anti-aliasing off on its command line.
void InstallUpscaler(std::vector<std::string>& game_args);

} // namespace game
