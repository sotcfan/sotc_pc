#pragma once

// The game's languages (see language.cpp).

namespace game {

// Between Linker::LoadExecutable and Linker::Link (the game's code is still writable).
void InstallLanguage();

} // namespace game
