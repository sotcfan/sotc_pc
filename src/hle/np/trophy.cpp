// libSceNpTrophy: the game's trophies. There is no trophy service here: registration always
// succeeds and every unlock is accepted and written to the log ("trophy N unlocked").
//
// The game creates its trophy context on a task thread and, while it has none, asks again from
// its main loop (the context stays at -1 when the call does not fill it in); some of its startup
// waits for the registration to finish. One context and one handle are enough.

#include <mutex>
#include <set>

#include "common/log.h"
#include "common/types.h"
#include "hle/registry.h"

namespace hle::np {

namespace {

constexpr s32 Context = 1;
constexpr s32 Handle = 1;
constexpr s32 InvalidTrophyId = -1; // SCE_NP_TROPHY_INVALID_TROPHY_ID: no platinum unlocked

constexpr s32 ErrorInvalidArgument = static_cast<s32>(0x80551604);

std::mutex g_mutex;
std::set<s32> g_unlocked; // this session's unlocks (each is logged once)

PS4_SYSV_ABI s32 sceNpTrophyCreateContext(s32* context, s32 /*user_id*/, u32 /*service_label*/,
                                          u64 /*options*/) {
    if (!context) {
        return ErrorInvalidArgument;
    }
    *context = Context;
    return 0;
}

PS4_SYSV_ABI s32 sceNpTrophyCreateHandle(s32* handle) {
    if (!handle) {
        return ErrorInvalidArgument;
    }
    *handle = Handle;
    return 0;
}

PS4_SYSV_ABI s32 sceNpTrophyRegisterContext(s32, s32, u64) {
    LOG_INFO("Trophy", "trophies registered");
    return 0;
}

PS4_SYSV_ABI s32 sceNpTrophyUnlockTrophy(s32, s32, s32 trophy, s32* platinum) {
    if (platinum) {
        *platinum = InvalidTrophyId;
    }
    std::scoped_lock lock{g_mutex};
    if (g_unlocked.insert(trophy).second) {
        LOG_INFO("Trophy", "trophy {} unlocked", trophy);
    }
    return 0;
}

PS4_SYSV_ABI s32 sceNpTrophyDestroyHandle(s32) {
    return 0;
}

PS4_SYSV_ABI s32 sceNpTrophyDestroyContext(s32) {
    return 0;
}

} // namespace

HLE_FUNCTION("libSceNpTrophy", "sceNpTrophyCreateContext", sceNpTrophyCreateContext);
HLE_FUNCTION("libSceNpTrophy", "sceNpTrophyCreateHandle", sceNpTrophyCreateHandle);
HLE_FUNCTION("libSceNpTrophy", "sceNpTrophyRegisterContext", sceNpTrophyRegisterContext);
HLE_FUNCTION("libSceNpTrophy", "sceNpTrophyUnlockTrophy", sceNpTrophyUnlockTrophy);
HLE_FUNCTION("libSceNpTrophy", "sceNpTrophyDestroyHandle", sceNpTrophyDestroyHandle);
HLE_FUNCTION("libSceNpTrophy", "sceNpTrophyDestroyContext", sceNpTrophyDestroyContext);

} // namespace hle::np
