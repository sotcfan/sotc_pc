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
//   manager +0xec8 + l*0x70  one entry per language l (ELanguage order):
//              +0x00       s32 the language
//              +0x04       s32 the one to fall back to (-1 none)
//              +0x10       const char* the suffix of its text files ("pt-br": text/<x>/_cmn/
//                          <file>.<suffix>.ctxt)
//              +0x58/0x60  std::vector<s32> the system languages (SCE_SYSTEM_PARAM_LANG_*) that
//                          choose it
//
// At start 0xe2ea00(manager, use) looks the system language up among the use's languages
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
// 1. The Americas' variants are offered too, each right after its sibling (en before en-GB,
//    fr-CA after fr, pt-BR after pt, es-419 after es): the uses' lists are rebuilt in the game's
//    own heap (its operator new, 0x417070) and the masks get their bits.
// 2. SOTC_LANGUAGE (src/hle/system/language.h) also replaces every language the game sets
//    (0xe2e620 is hooked), the profile's included: the launcher's choice holds on every start,
//    not only before the first profile. Without it the game's own choice stands.
//
// SOTC_LOG_LANGUAGES=1 logs the manager's tables when the languages are set.

#include "game/language.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "common/log.h"
#include "common/types.h"
#include "hle/system/language.h"
#include "loader/hook.h"

namespace game {

namespace {

constexpr u64 SetUseLanguageAddress = 0xe2e620;
constexpr u64 OperatorNewAddress = 0x417070;

constexpr u32 UseCount = 4;
constexpr u32 UseSize = 0x3b0;
constexpr u32 UseCurrentOffset = 0x0c;
constexpr u32 UseMaskOffset = 0x10;
constexpr u32 UseOfferedOffset = 0x20;
constexpr u32 LanguagesOffset = 0xec8;
constexpr u32 LanguageSize = 0x70;
constexpr u32 LanguageCount = 27;
constexpr u32 LanguageFallbackOffset = 0x04;
constexpr u32 LanguageSuffixOffset = 0x10;
constexpr u32 LanguageSystemCodesOffset = 0x58;

using SetUseLanguageFn = void PS4_SYSV_ABI (*)(u8* manager, s32 use, s32 language);
using OperatorNewFn = void* PS4_SYSV_ABI (*)(u64 size);
SetUseLanguageFn g_original_set_use_language = nullptr;

bool g_log = false;
bool g_offered = false; // the extra languages were added

// The extra languages, each with the one it goes next to: {language, sibling, before it}.
struct Extra {
    s32 language;
    s32 sibling;
    bool before;
};
constexpr Extra Extras[] = {{7, 8, true}, {11, 10, false}, {20, 19, false}, {23, 22, false}};

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
        LOG_INFO("Language", "  use {}: current {}, mask {:#x}, offered [{}]", use,
                 Read<s32>(block, UseCurrentOffset), Read<u32>(block, UseMaskOffset),
                 IntVector(block, UseOfferedOffset));
    }
    for (u32 l = 0; l < LanguageCount; l++) {
        const u8* entry = manager + LanguagesOffset + l * LanguageSize;
        const char* suffix = Read<const char*>(entry, LanguageSuffixOffset);
        LOG_INFO("Language", "  language {:2}: id {}, fallback {}, files '{}', system [{}]", l,
                 Read<s32>(entry, 0), Read<s32>(entry, LanguageFallbackOffset),
                 suffix ? suffix : "", IntVector(entry, LanguageSystemCodesOffset));
    }
}

template <typename T>
void Write(u8* base, u32 offset, T value) {
    std::memcpy(base + offset, &value, sizeof(value));
}

// Adds the extra languages to a use's offered list and mask (see above). Uses that offer nothing
// (speech: the game has none) are left alone.
void OfferExtras(u8* block) {
    const auto* begin = Read<const s32*>(block, UseOfferedOffset);
    const auto* end = Read<const s32*>(block, UseOfferedOffset + 8);
    if (!begin || end <= begin || end - begin > 64) {
        return;
    }
    std::vector<s32> offered(begin, end);
    u32 mask = Read<u32>(block, UseMaskOffset);
    for (const auto& extra : Extras) {
        if (std::ranges::find(offered, extra.language) != offered.end()) {
            continue;
        }
        auto sibling = std::ranges::find(offered, extra.sibling);
        if (sibling == offered.end()) {
            continue;
        }
        offered.insert(extra.before ? sibling : sibling + 1, extra.language);
        mask |= 1u << extra.language;
    }
    if (offered.size() == static_cast<size_t>(end - begin)) {
        return;
    }
    // The game's vector<s32> {begin, end, capacity}: a new buffer from the game's own heap (the
    // old one is left behind: it was allocated once, at start).
    const auto operator_new = reinterpret_cast<OperatorNewFn>(OperatorNewAddress);
    auto* storage = static_cast<s32*>(operator_new(offered.size() * sizeof(s32)));
    std::memcpy(storage, offered.data(), offered.size() * sizeof(s32));
    Write(block, UseOfferedOffset, storage);
    Write(block, UseOfferedOffset + 8, storage + offered.size());
    Write(block, UseOfferedOffset + 16, storage + offered.size());
    Write(block, UseMaskOffset, mask);
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
    if (!g_offered) {
        g_offered = true;
        for (u32 use = 0; use < UseCount; use++) {
            OfferExtras(manager + use * UseSize);
        }
    }
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
    g_original_set_use_language = reinterpret_cast<SetUseLanguageFn>(loader::HookFunction(
        SetUseLanguageAddress, reinterpret_cast<const void*>(&SetUseLanguage)));
    if (!g_original_set_use_language) {
        LOG_ERROR("Language", "could not hook the language setter at {:#x}",
                  SetUseLanguageAddress);
    }
}

} // namespace game
