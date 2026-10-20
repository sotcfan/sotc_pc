#pragma once

#include "common/types.h"

namespace hle::system {

// The system language the game is told (SCE_SYSTEM_PARAM_LANG_*): SOTC_LANGUAGE, as a name
// ("pt-BR", "en-GB"...) or a code, else English (US).
s32 SystemLanguage();

// Whether SOTC_LANGUAGE chose it (the game then also gets it instead of the one in its profile:
// src/game/language.cpp).
bool SystemLanguageChosen();

} // namespace hle::system
