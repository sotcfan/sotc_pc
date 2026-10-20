// libSceNpTrophy: the game's trophies. There is no trophy service here: registration always
// succeeds and every unlock is accepted, written to the log ("trophy N unlocked") and kept in
// <data>/trophies.txt, so that the trophies earned stay recorded from one session to the next.
//
// The game creates its trophy context on a task thread and, while it has none, asks again from
// its main loop (the context stays at -1 when the call does not fill it in); some of its startup
// waits for the registration to finish. One context and one handle are enough.
//
// The game never asks which trophies it has (it imports only the six functions below): it calls
// sceNpTrophyUnlockTrophy whenever a trophy's condition is met, so the same trophy comes again in
// later sessions and only its first unlock is recorded. On a PS4 the system itself awards the
// platinum when the last of the other trophies is unlocked, and tells the game through the
// `platinum` argument; this does the same.
//
// trophies.txt, one line per trophy, in the order they were unlocked (lines starting with '#'
// are comments):
//
//   <trophy id> <date> <local time>        e.g. "5 2026-10-20 14:03:27"
//
// The names and descriptions are in the game's sce_sys/trophy/trophy00.trp, encrypted with a
// console key: not read here. Its 38 icons (TROP000.PNG to TROP037.PNG) are plain PNG files, the
// id is the number in the name.

#include "hle/np/trophy.h"

#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <system_error>

#include "common/log.h"
#include "common/types.h"
#include "hle/registry.h"

namespace hle::np {

namespace {

constexpr s32 Context = 1;
constexpr s32 Handle = 1;
constexpr s32 InvalidTrophyId = -1; // SCE_NP_TROPHY_INVALID_TROPHY_ID: no platinum unlocked

constexpr s32 ErrorInvalidArgument = static_cast<s32>(0x80551604);

// Shadow of the Colossus (CUSA08809) has 38 trophies, ids 0 to 37 (TROP000..TROP037.PNG in
// trophy00.trp); 0 is the platinum (its icon is Wander; 1-5 are gold, 6-12 silver, the rest
// bronze).
constexpr s32 TrophyCount = 38;
constexpr s32 PlatinumId = 0;

std::mutex g_mutex;
std::filesystem::path g_path;          // <data>/trophies.txt; empty: not recorded
std::map<s32, std::string> g_unlocked; // trophy id -> when it was first unlocked
std::string g_order;                   // the file's lines, in the order the trophies came

std::string LocalTimeNow() {
    const std::time_t now = std::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &now);
#else
    localtime_r(&now, &tm);
#endif
    char text[32];
    std::strftime(text, sizeof(text), "%Y-%m-%d %H:%M:%S", &tm);
    return text;
}

// Writes the whole file again (it is tiny): to a temporary file first, then renamed over the old
// one, so that a crash while writing never leaves it half written. Called with g_mutex held.
void Save() {
    if (g_path.empty()) {
        return;
    }
    std::filesystem::path temp = g_path;
    temp += ".tmp";
    {
        std::ofstream out{temp, std::ios::trunc};
        out << "# Trophies unlocked in sotcport (Shadow of the Colossus, CUSA08809), in the order\n"
               "# they came: trophy id, date, local time. 38 trophies, id 0 is the platinum; the\n"
               "# icons are TROP000.PNG..TROP037.PNG in the game's sce_sys/trophy/trophy00.trp.\n"
            << g_order;
        if (!out) {
            LOG_ERROR("Trophy", "could not write {}", temp.string());
            return;
        }
    }
    std::error_code error;
    std::filesystem::rename(temp, g_path, error);
    if (error) {
        LOG_ERROR("Trophy", "could not replace {}: {}", g_path.string(), error.message());
    }
}

// Records a trophy's first unlock. False if it was already recorded. Called with g_mutex held.
bool Record(s32 trophy) {
    if (g_unlocked.contains(trophy)) {
        return false;
    }
    const std::string when = LocalTimeNow();
    g_unlocked.emplace(trophy, when);
    g_order += std::to_string(trophy) + ' ' + when + '\n';
    return true;
}

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
    if (trophy < 0 || trophy >= TrophyCount) {
        LOG_WARNING("Trophy", "unlock of unknown trophy {}", trophy);
        return 0;
    }
    std::scoped_lock lock{g_mutex};
    if (!Record(trophy)) {
        return 0; // earned in an earlier session (or earlier in this one)
    }
    LOG_INFO("Trophy", "trophy {} unlocked ({} of {})", trophy, g_unlocked.size(), TrophyCount);
    // The platinum comes with the last of the others.
    if (trophy != PlatinumId && !g_unlocked.contains(PlatinumId) &&
        g_unlocked.size() == static_cast<size_t>(TrophyCount - 1)) {
        Record(PlatinumId);
        if (platinum) {
            *platinum = PlatinumId;
        }
        LOG_INFO("Trophy", "platinum trophy {} unlocked: all {} trophies", PlatinumId,
                 TrophyCount);
    }
    Save();
    return 0;
}

PS4_SYSV_ABI s32 sceNpTrophyDestroyHandle(s32) {
    return 0;
}

PS4_SYSV_ABI s32 sceNpTrophyDestroyContext(s32) {
    return 0;
}

} // namespace

void InitializeTrophies(const std::filesystem::path& data_dir) {
    std::scoped_lock lock{g_mutex};
    g_path = data_dir / "trophies.txt";
    g_unlocked.clear();
    g_order.clear();
    std::ifstream in{g_path};
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream fields{line};
        s32 trophy = -1;
        std::string date, time;
        if (!(fields >> trophy >> date >> time) || trophy < 0 || trophy >= TrophyCount) {
            LOG_WARNING("Trophy", "{}: line not understood, dropped: \"{}\"", g_path.string(), line);
            continue;
        }
        if (g_unlocked.emplace(trophy, date + ' ' + time).second) {
            g_order += line + '\n';
        }
    }
    LOG_INFO("Trophy", "{} of {} trophies unlocked so far ({})", g_unlocked.size(), TrophyCount,
             g_path.string());
}

HLE_FUNCTION("libSceNpTrophy", "sceNpTrophyCreateContext", sceNpTrophyCreateContext);
HLE_FUNCTION("libSceNpTrophy", "sceNpTrophyCreateHandle", sceNpTrophyCreateHandle);
HLE_FUNCTION("libSceNpTrophy", "sceNpTrophyRegisterContext", sceNpTrophyRegisterContext);
HLE_FUNCTION("libSceNpTrophy", "sceNpTrophyUnlockTrophy", sceNpTrophyUnlockTrophy);
HLE_FUNCTION("libSceNpTrophy", "sceNpTrophyDestroyHandle", sceNpTrophyDestroyHandle);
HLE_FUNCTION("libSceNpTrophy", "sceNpTrophyDestroyContext", sceNpTrophyDestroyContext);

} // namespace hle::np
