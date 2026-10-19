#pragma once

// The game's step of time per frame, and the physics made to follow it, so that Wander's cape and
// Agro behave at any frame rate (see frame_step.cpp).

namespace game {

// Between Linker::LoadExecutable and Linker::Link (the game's code is still writable).
void InstallFrameStep();

} // namespace game
