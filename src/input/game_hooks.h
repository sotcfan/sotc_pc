#pragma once

// Hooks into the game's own input code, for the split actions of the keyboard and mouse (one PS4
// button, two inputs that mean different things; see input.h, ButtonHeldFor, and the long
// explanation in game_hooks.cpp).

namespace input {

// Between Linker::LoadExecutable and Linker::Link (the game's code is still writable), after
// LoadControls (the tags come from controls.ini). Does nothing for a hook whose address is not
// known yet.
void InstallGameHooks();

} // namespace input
