#pragma once

#include "loader/elf_ps4.h"

namespace hle {

// Prepares our libraries for the game process. Runs on the game's main thread right before
// its entry point, as the PS4's libkernel and libc start-up code would.
void InitializeProcess(const loader::ProcParam* param);

} // namespace hle
