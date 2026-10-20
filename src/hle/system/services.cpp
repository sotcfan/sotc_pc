// libSceUserService and libSceSystemService: who is playing and console-wide settings.
// There is one local user, id 1, logged in from the start.

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <string_view>

#include "common/log.h"
#include "common/types.h"
#include "hle/registry.h"
#include "hle/system/language.h"

namespace hle::system {

namespace {

constexpr s32 UserId = 1;
constexpr s32 InvalidUserId = -1;
constexpr const char* UserName = "Player";

constexpr s32 ErrorUserInvalidArgument = static_cast<s32>(0x80960005);
constexpr s32 ErrorUserNoEvent = static_cast<s32>(0x80960007);
constexpr s32 ErrorSystemParameter = static_cast<s32>(0x80A10003);
constexpr s32 ErrorSystemNoEvent = static_cast<s32>(0x80A10004);

// ---- libSceUserService ----

struct LoginUserIdList {
    s32 ids[4];
};

struct UserEvent {
    s32 type; // 0 = login, 1 = logout
    s32 user_id;
};

std::atomic<bool> g_login_reported{false};

PS4_SYSV_ABI s32 sceUserServiceInitialize(const void*) {
    return 0;
}

PS4_SYSV_ABI s32 sceUserServiceGetInitialUser(s32* user) {
    if (!user) {
        return ErrorUserInvalidArgument;
    }
    *user = UserId;
    return 0;
}

PS4_SYSV_ABI s32 sceUserServiceGetLoginUserIdList(LoginUserIdList* list) {
    if (!list) {
        return ErrorUserInvalidArgument;
    }
    list->ids[0] = UserId;
    list->ids[1] = list->ids[2] = list->ids[3] = InvalidUserId;
    return 0;
}

PS4_SYSV_ABI s32 sceUserServiceGetUserName(s32, char* name, u64 size) {
    if (!name || size == 0) {
        return ErrorUserInvalidArgument;
    }
    std::strncpy(name, UserName, size - 1);
    name[size - 1] = '\0';
    return 0;
}

// The user's login is reported once, as on the console after start-up.
PS4_SYSV_ABI s32 sceUserServiceGetEvent(UserEvent* event) {
    if (!event) {
        return ErrorUserInvalidArgument;
    }
    if (g_login_reported.exchange(true)) {
        return ErrorUserNoEvent;
    }
    *event = {0, UserId};
    return 0;
}

// ---- libSceSystemService ----

constexpr s32 ParamLanguage = 1;
constexpr s32 ParamDateFormat = 2;
constexpr s32 ParamTimeFormat = 3;
constexpr s32 ParamTimeZone = 4;
constexpr s32 ParamSummerTime = 5;
constexpr s32 ParamParentalLevel = 7;
constexpr s32 ParamEnterButton = 1000;

constexpr s32 LanguageEnglishUs = 1;
constexpr s32 EnterButtonCross = 1;

// The system language (SCE_SYSTEM_PARAM_LANG_*), as BCP 47-like names. Shadow of the Colossus
// has text and menus in English, Japanese, French (also Canada's), Spanish (Spain's and Latin
// America's), German, Italian, Dutch, Portuguese (Portugal's and Brazil's), Russian, Korean,
// Chinese, Polish, Turkish and Arabic (the names in its executable).
struct Language {
    std::string_view name;
    s32 code;
};
constexpr Language Languages[] = {
    {"ja", 0},       {"en", 1},       {"en-US", 1},    {"fr", 2},       {"es", 3},
    {"de", 4},       {"it", 5},       {"nl", 6},       {"pt", 7},       {"pt-PT", 7},
    {"ru", 8},       {"ko", 9},       {"zh-Hant", 10}, {"zh-Hans", 11}, {"zh", 11},
    {"fi", 12},      {"sv", 13},      {"da", 14},      {"no", 15},      {"pl", 16},
    {"pt-BR", 17},   {"en-GB", 18},   {"tr", 19},      {"es-419", 20},  {"ar", 21},
    {"fr-CA", 22},
};


struct DisplaySafeAreaInfo {
    float ratio;
    u8 reserved[128];
};

struct SystemStatus {
    s32 event_count;
    bool system_ui_overlaid;
    bool in_background;
    bool cpu_mode7_normal;
    bool live_streaming;
    bool out_of_vr_area;
};

PS4_SYSV_ABI s32 sceSystemServiceGetDisplaySafeAreaInfo(DisplaySafeAreaInfo* info) {
    if (!info) {
        return ErrorSystemParameter;
    }
    std::memset(info, 0, sizeof(*info));
    info->ratio = 1.0f; // the whole screen is safe
    return 0;
}

PS4_SYSV_ABI s32 sceSystemServiceGetStatus(SystemStatus* status) {
    if (!status) {
        return ErrorSystemParameter;
    }
    *status = {0, false, false, true, false, false};
    return 0;
}

PS4_SYSV_ABI s32 sceSystemServiceHideSplashScreen() {
    return 0;
}

PS4_SYSV_ABI s32 sceSystemServiceParamGetInt(s32 id, s32* value) {
    if (!value) {
        return ErrorSystemParameter;
    }
    switch (id) {
    case ParamLanguage: *value = SystemLanguage(); break;
    case ParamDateFormat: *value = 1; break; // DD/MM/YYYY
    case ParamTimeFormat: *value = 1; break; // 24 hours
    case ParamTimeZone: *value = 0; break;
    case ParamSummerTime: *value = 0; break;
    case ParamParentalLevel: *value = 0; break;
    case ParamEnterButton: *value = EnterButtonCross; break;
    default:
        LOG_WARNING("System", "unknown system parameter {}", id);
        *value = 0;
    }
    return 0;
}

PS4_SYSV_ABI s32 sceSystemServiceReceiveEvent(void*) {
    return ErrorSystemNoEvent;
}

PS4_SYSV_ABI s32 sceSystemServiceLoadExec(const char* path, char* const*) {
    LOG_ERROR("System", "the game asked to execute \"{}\" (not supported)", path ? path : "");
    return 0;
}

} // namespace

// SOTC_LANGUAGE=<name or code> (OPTIONS.md): the language the game is told the system is in;
// English (US) otherwise.
s32 SystemLanguage() {
    static const s32 language = [] {
        const char* wanted = std::getenv("SOTC_LANGUAGE");
        if (!wanted || !*wanted) {
            return LanguageEnglishUs;
        }
        for (const auto& language : Languages) {
            if (language.name == wanted) {
                return language.code;
            }
        }
        char* end = nullptr;
        const long code = std::strtol(wanted, &end, 10);
        if (*end == '\0' && code >= 0 && code <= 29) {
            return static_cast<s32>(code);
        }
        LOG_WARNING("System", "SOTC_LANGUAGE={}: unknown language; English", wanted);
        return LanguageEnglishUs;
    }();
    return language;
}

bool SystemLanguageChosen() {
    const char* wanted = std::getenv("SOTC_LANGUAGE");
    return wanted && *wanted;
}

HLE_FUNCTION("libSceUserService", "sceUserServiceInitialize", sceUserServiceInitialize);
HLE_FUNCTION("libSceUserService", "sceUserServiceGetInitialUser", sceUserServiceGetInitialUser);
HLE_FUNCTION("libSceUserService", "sceUserServiceGetLoginUserIdList", sceUserServiceGetLoginUserIdList);
HLE_FUNCTION("libSceUserService", "sceUserServiceGetUserName", sceUserServiceGetUserName);
HLE_FUNCTION("libSceUserService", "sceUserServiceGetEvent", sceUserServiceGetEvent);
HLE_FUNCTION("libSceSystemService", "sceSystemServiceGetDisplaySafeAreaInfo", sceSystemServiceGetDisplaySafeAreaInfo);
HLE_FUNCTION("libSceSystemService", "sceSystemServiceGetStatus", sceSystemServiceGetStatus);
HLE_FUNCTION("libSceSystemService", "sceSystemServiceHideSplashScreen", sceSystemServiceHideSplashScreen);
HLE_FUNCTION("libSceSystemService", "sceSystemServiceParamGetInt", sceSystemServiceParamGetInt);
HLE_FUNCTION("libSceSystemService", "sceSystemServiceReceiveEvent", sceSystemServiceReceiveEvent);
HLE_FUNCTION("libSceSystemService", "sceSystemServiceLoadExec", sceSystemServiceLoadExec);

} // namespace hle::system
