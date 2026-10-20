// The game's languages.
//
// ==== How the game picks its language (2026-10-20) ==============================================
//
// The localization manager (BPE, the object at *0x36f4fe0) keeps four uses of language: text,
// subtitles, speech and system (ConVars loc_text, loc_subtitles, loc_speech, loc_system; enum
// ELanguage). Its layout:
//
//   manager +0x000         s32 the default language (ELanguage)
//   manager +slot*0x3b0    one block per use (slot 0-3):
//              +0x0c       s32 the language in use
//              +0x10       u32 mask of the languages this use may take (bit = ELanguage)
//              +0x20/0x28  std::vector<s32> the languages offered, in order (begin, end)
//              +0x40/0x48  std::vector<s32> the languages whose text is loaded: the offered ones
//                          and the ones they fall back to
//              +0x58+l*0x20  the text tables of language l
//   manager +0xec8 + l*0x70  one entry per language l (ELanguage order):
//              +0x00       s32 the language
//              +0x04       s32 the one to fall back to (-1 none)
//              +0x10       const char* the suffix of its text files ("pt-br": text/<x>/_cmn/
//                          <file>.<suffix>.ctxt)
//              +0x58/0x60  std::vector<s32> the system languages (SCE_SYSTEM_PARAM_LANG_*) that
//                          choose it
//
// At start (0x931d60) the manager is set up by 0xe2caf0: the language entries, then each use by
// 0xe2dae0(manager, use, allowed), where `allowed` is the build's mask for that use (a game
// setting, *(*0x36f79a8 + 0xf8) + 0x2c/0x30/0x34/0x38): it keeps the mask (for the system use,
// only languages with system codes), lists the offered languages in ELanguage order and the
// loaded ones (those plus their fallbacks). Then 0xe2ed00(manager, use, folder, file), once per
// text file (ui, story, system, tutorial, ps4, win...), reads <folder>/_cmn/<file>.<suffix>.ctxt
// for every loaded language (a missing file is skipped: ps4/win exist for a few languages only).
// A language without its own text shows its fallback's: pt-BR shows pt.
// 0xe2ea00(manager, use) then looks the system language up among the use's languages
// (sceSystemServiceParamGetInt(1)); 0xe2e5d0(manager, language) then sets all four uses with
// 0xe2e620(manager, use, language), which the game also calls itself when the profile is loaded
// (with the language kept there) or the player picks one.
// The first time (no profile save) the game shows its own language list, starting at that
// language, and keeps the choice in the profile: later the system language no longer matters.
//
// The languages (ELanguage = entry index): 0 global, 1 ar, 2 zh, 3 zh-MO, 4 cs, 5 da, 6 nl, 7 en,
// 8 en-GB, 9 fi, 10 fr, 11 fr-CA, 12 de, 13 el, 14 it, 15 ja, 16 ko, 17 no, 18 pl, 19 pt,
// 20 pt-BR, 21 ru, 22 es, 23 es-419, 24 sv, 25 tr, 26 th. This dump (CUSA08809, the European one)
// offers ar, da, nl, en-GB, fi, fr, de, it, no, pl, pt, ru, es, sv, tr (mask 0x36e5762); its data
// still has the complete text of the Americas' variants: en, fr-CA, pt-BR and es-419
// (text/<x>/_cmn/{ui,story,system,tutorial}.<suffix>.ctxt).
//
// ==== What sotcport changes =====================================================================
//
// 1. The Americas' variants are offered too, wherever their sibling is (en with en-GB, fr-CA with
//    fr, pt-BR with pt, es-419 with es): 0xe2dae0 is hooked and they are added to `allowed`, so
//    the game lists them (ELanguage order puts each next to its sibling) and loads their text.
//    Adding them later, to the lists alone, offered pt-BR with pt's text (2026-10-20: its text
//    had not been loaded).
// 2. SOTC_LANGUAGE (src/hle/system/language.h) also replaces every language the game sets
//    (0xe2e620 is hooked), the profile's included: the launcher's choice holds on every start,
//    not only before the first profile. Without it the game's own choice stands.
//
// SOTC_LOG_LANGUAGES=1 logs the manager's tables when the languages are set.

#include "game/language.h"

#include <cstdlib>
#include <cstring>
#include <string>

#include <fmt/format.h>

#include "common/log.h"
#include "common/types.h"
#include "hle/system/language.h"
#include "loader/hook.h"

namespace game {

namespace {

constexpr u64 SetUpUseAddress = 0xe2dae0;
constexpr u64 SetUseLanguageAddress = 0xe2e620;

constexpr u32 UseCount = 4;
constexpr u32 UseSize = 0x3b0;
constexpr u32 UseCurrentOffset = 0x0c;
constexpr u32 UseMaskOffset = 0x10;
constexpr u32 UseOfferedOffset = 0x20;
constexpr u32 UseLoadedOffset = 0x40;
constexpr u32 LanguagesOffset = 0xec8;
constexpr u32 LanguageSize = 0x70;
constexpr u32 LanguageCount = 27;
constexpr u32 LanguageFallbackOffset = 0x04;
constexpr u32 LanguageSuffixOffset = 0x10;
constexpr u32 LanguageSystemCodesOffset = 0x58;

using SetUpUseFn = void PS4_SYSV_ABI (*)(u8* manager, s32 use, u32 allowed);
using SetUseLanguageFn = void PS4_SYSV_ABI (*)(u8* manager, s32 use, s32 language);
SetUpUseFn g_original_set_up_use = nullptr;
SetUseLanguageFn g_original_set_use_language = nullptr;

bool g_log = false;

// The extra languages, each with the sibling that brings it in: {language, sibling}.
struct Extra {
    s32 language;
    s32 sibling;
};
constexpr Extra Extras[] = {{7, 8}, {11, 10}, {20, 19}, {23, 22}};

template <typename T>
T Read(const u8* base, u32 offset) {
    T value;
    std::memcpy(&value, base + offset, sizeof(value));
    return value;
}

std::string IntVector(const u8* base, u32 offset) {
    const auto* begin = Read<const s32*>(base, offset);
    const auto* end = Read<const s32*>(base, offset + 8);
    std::string text;
    for (const s32* it = begin; it && it < end && it - begin < 64; ++it) {
        text += fmt::format("{}{}", text.empty() ? "" : ",", *it);
    }
    return text;
}

void LogManager(const u8* manager, s32 language) {
    LOG_INFO("Language", "languages set to {} (manager {}, default {})", language,
             fmt::ptr(manager), Read<s32>(manager, 0));
    for (u32 use = 0; use < UseCount; use++) {
        const u8* block = manager + use * UseSize;
        LOG_INFO("Language", "  use {}: current {}, mask {:#x}, offered [{}], loaded [{}]", use,
                 Read<s32>(block, UseCurrentOffset), Read<u32>(block, UseMaskOffset),
                 IntVector(block, UseOfferedOffset), IntVector(block, UseLoadedOffset));
    }
    for (u32 l = 0; l < LanguageCount; l++) {
        const u8* entry = manager + LanguagesOffset + l * LanguageSize;
        const char* suffix = Read<const char*>(entry, LanguageSuffixOffset);
        LOG_INFO("Language", "  language {:2}: id {}, fallback {}, files '{}', system [{}]", l,
                 Read<s32>(entry, 0), Read<s32>(entry, LanguageFallbackOffset),
                 suffix ? suffix : "", IntVector(entry, LanguageSystemCodesOffset));
    }
}

// 0xe2dae0: sets a use up from the build's mask of the languages it allows (see above).
void PS4_SYSV_ABI SetUpUse(u8* manager, s32 use, u32 allowed) {
    const u32 original = allowed;
    for (const auto& extra : Extras) {
        if (allowed & (1u << extra.sibling)) {
            allowed |= 1u << extra.language;
        }
    }
    if (allowed != original) {
        LOG_INFO("Language", "use {}: languages {:#x} instead of {:#x} (the Americas' variants)",
                 use, allowed, original);
    }
    g_original_set_up_use(manager, use, allowed);
}

// The language whose entry lists `code` among its system languages, or -1.
s32 LanguageOfSystemCode(const u8* manager, s32 code) {
    for (u32 l = 0; l < LanguageCount; l++) {
        const u8* entry = manager + LanguagesOffset + l * LanguageSize;
        const auto* begin = Read<const s32*>(entry, LanguageSystemCodesOffset);
        const auto* end = Read<const s32*>(entry, LanguageSystemCodesOffset + 8);
        for (const s32* it = begin; it && it < end; ++it) {
            if (*it == code) {
                return Read<s32>(entry, 0);
            }
        }
    }
    return -1;
}

void PS4_SYSV_ABI SetUseLanguage(u8* manager, s32 use, s32 language) {
    // Only where the use takes that language (speech has none: the game sets it to -1).
    const u32 mask = use >= 0 && use < s32(UseCount) ? Read<u32>(manager + use * UseSize,
                                                                  UseMaskOffset)
                                                     : 0;
    if (hle::system::SystemLanguageChosen() && language >= 0) {
        const s32 chosen = LanguageOfSystemCode(manager, hle::system::SystemLanguage());
        if (chosen >= 0 && chosen < 32 && (mask & (1u << chosen)) && chosen != language) {
            LOG_INFO("Language", "use {}: language {} (SOTC_LANGUAGE) instead of the game's {}", use,
                     chosen, language);
            language = chosen;
        }
    }
    if (g_log && use == 0) {
        LogManager(manager, language);
    }
    g_original_set_use_language(manager, use, language);
}

} // namespace

void InstallLanguage() {
    const char* log = std::getenv("SOTC_LOG_LANGUAGES");
    g_log = log && *log && *log != '0';
    g_original_set_up_use = reinterpret_cast<SetUpUseFn>(
        loader::HookFunction(SetUpUseAddress, reinterpret_cast<const void*>(&SetUpUse)));
    if (!g_original_set_up_use) {
        LOG_ERROR("Language", "could not hook the language setup at {:#x}", SetUpUseAddress);
    }
    g_original_set_use_language = reinterpret_cast<SetUseLanguageFn>(loader::HookFunction(
        SetUseLanguageAddress, reinterpret_cast<const void*>(&SetUseLanguage)));
    if (!g_original_set_use_language) {
        LOG_ERROR("Language", "could not hook the language setter at {:#x}",
                  SetUseLanguageAddress);
    }
}

} // namespace game
