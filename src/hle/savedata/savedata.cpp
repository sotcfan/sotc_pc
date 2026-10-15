// libSceSaveData: the game's saves, one folder per save under <data>/saves/<user id>/<name>.
//
// While the game has a save mounted, its folder appears at "/savedataN" and the game reads and
// writes ordinary files there. Next to them, "sce_sys" holds what the PS4 keeps about a save:
// param.bin (our own format: the size it was created with and the title/detail texts the game
// sets) and icon0.png. Unmounting "with backup" copies the folder to <data>/saves/<user
// id>/.backup/<name>, which the game can check and restore if the save gets damaged.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "common/log.h"
#include "input/input.h"
#include "common/types.h"
#include "hle/kernel/filesystem.h"
#include "hle/registry.h"
#include "hle/savedata/savedata.h"

namespace hle::savedata {

namespace fs = std::filesystem;

namespace {

constexpr s32 ErrorParameter = static_cast<s32>(0x809F0000);
constexpr s32 ErrorNotInitialized = static_cast<s32>(0x809F0001);
constexpr s32 ErrorBusy = static_cast<s32>(0x809F0003);
constexpr s32 ErrorNotMounted = static_cast<s32>(0x809F0004);
constexpr s32 ErrorExists = static_cast<s32>(0x809F0007);
constexpr s32 ErrorNotFound = static_cast<s32>(0x809F0008);
constexpr s32 ErrorInternal = static_cast<s32>(0x809F000B);
constexpr s32 ErrorMountFull = static_cast<s32>(0x809F000C);
constexpr s32 ErrorBadMounted = static_cast<s32>(0x809F000D);
constexpr s32 ErrorBroken = static_cast<s32>(0x809F000F);

// Save sizes are counted in 32 KiB blocks.
constexpr u64 BlockSize = 32 * 1024;
constexpr u64 MinBlocks = 96;
constexpr u64 MaxBlocks = 32768;
constexpr int MountSlots = 16;

struct DirName {
    char data[32];
};

struct TitleId {
    char data[10];
    char padding[6];
};

struct MountPoint {
    char data[16];
};

struct Param {
    char title[128];
    char subtitle[128];
    char detail[1024];
    u32 user_param;
    u32 padding;
    s64 mtime; // seconds since 1970
    u8 reserved[32];
};
static_assert(sizeof(Param) == 0x530);

enum class ParamType : u32 { All = 0, Title = 1, Subtitle = 2, Detail = 3, UserParam = 4 };

namespace mount_mode {
constexpr u32 ReadOnly = 1 << 0;
constexpr u32 Create = 1 << 2;          // fails if the save exists
constexpr u32 CreateIfMissing = 1 << 5; // "CREATE2"
} // namespace mount_mode

struct Mount2 {
    s32 user_id;
    u32 padding;
    const DirName* dir_name;
    u64 blocks;
    u32 mode;
    u8 reserved[36];
};

struct MountResult {
    MountPoint mount_point;
    u64 required_blocks;
    u32 unused;
    u32 status; // 1 = created by this call
    u8 reserved[32];
};

struct MountInfo {
    u64 blocks;
    u64 free_blocks;
    u8 reserved[32];
};

struct Icon {
    void* buffer;
    u64 buffer_size;
    u64 data_size;
    u8 reserved[32];
};

struct CheckBackupData {
    s32 user_id;
    u32 padding;
    const TitleId* title_id;
    const DirName* dir_name;
    Param* param;
    Icon* icon;
    u8 reserved[32];
};

struct RestoreBackupData {
    s32 user_id;
    u32 padding;
    const TitleId* title_id;
    const DirName* dir_name;
    const void* fingerprint;
    u32 unused;
    u8 reserved[36];
};

struct Delete {
    s32 user_id;
    u32 padding;
    const TitleId* title_id;
    const DirName* dir_name;
    u32 unused;
    u8 reserved[36];
};

enum class SortKey : u32 { DirName = 0, UserParam = 1, Blocks = 2, Mtime = 3, FreeBlocks = 5 };

struct DirNameSearchCond {
    s32 user_id;
    u32 padding;
    const TitleId* title_id;
    const DirName* dir_name; // pattern: '%' any run of characters, '_' any one character
    SortKey key;
    u32 descending;
    u8 reserved[32];
};

struct SearchInfo {
    u64 blocks;
    u64 free_blocks;
    u8 reserved[32];
};

struct DirNameSearchResult {
    u32 hit_count;
    u32 padding;
    DirName* dir_names;
    u32 dir_names_capacity;
    u32 set_count;
    Param* params; // optional
    SearchInfo* infos; // optional
    u8 reserved[16];
};

// sce_sys/param.bin.
struct StoredInfo {
    char magic[8];
    u64 blocks;
    Param param;
};
constexpr char StoredMagic[8] = {'S', 'H', 'A', 'D', 'S', 'A', 'V', '1'};

struct Mounted {
    s32 user_id;
    std::string dir_name;
    fs::path path;
    bool read_only;
};

std::mutex g_mutex;
fs::path g_root;
bool g_initialized = false;
std::array<std::optional<Mounted>, MountSlots> g_slots;

std::string ToString(const char* text, size_t max) {
    return std::string(text, strnlen(text, max));
}

template <size_t N>
void CopyString(char (&out)[N], std::string_view text) {
    std::memset(out, 0, N);
    std::memcpy(out, text.data(), std::min(text.size(), N - 1));
}

fs::path UserDir(s32 user_id) {
    return g_root / std::to_string(user_id);
}

fs::path SavePath(s32 user_id, std::string_view dir_name) {
    return UserDir(user_id) / dir_name;
}

fs::path BackupPath(s32 user_id, std::string_view dir_name) {
    return UserDir(user_id) / ".backup" / dir_name;
}

fs::path InfoPath(const fs::path& save) {
    return save / "sce_sys" / "param.bin";
}

std::optional<StoredInfo> ReadInfo(const fs::path& save) {
    StoredInfo info{};
    std::ifstream file(InfoPath(save), std::ios::binary);
    if (!file.read(reinterpret_cast<char*>(&info), sizeof(info)) ||
        std::memcmp(info.magic, StoredMagic, sizeof(StoredMagic)) != 0) {
        return std::nullopt;
    }
    std::error_code error;
    const auto written = fs::last_write_time(InfoPath(save), error);
    if (!error) {
        info.param.mtime = std::chrono::duration_cast<std::chrono::seconds>(
                               std::chrono::file_clock::to_sys(written)
                                   .time_since_epoch())
                               .count();
    }
    return info;
}

bool WriteInfo(const fs::path& save, const StoredInfo& info) {
    std::error_code error;
    fs::create_directories(save / "sce_sys", error);
    std::ofstream file(InfoPath(save), std::ios::binary | std::ios::trunc);
    return static_cast<bool>(file.write(reinterpret_cast<const char*>(&info), sizeof(info)));
}

u64 DirectorySize(const fs::path& path) {
    u64 size = 0;
    std::error_code error;
    for (const auto& entry : fs::recursive_directory_iterator(path, error)) {
        if (entry.is_regular_file(error)) {
            size += entry.file_size(error);
        }
    }
    return size;
}

u64 FreeBlocks(const fs::path& save, u64 blocks) {
    const u64 used = (DirectorySize(save) + BlockSize - 1) / BlockSize;
    return used < blocks ? blocks - used : 0;
}

// Replaces `to` with a copy of `from`.
bool CopyFolder(const fs::path& from, const fs::path& to) {
    std::error_code error;
    fs::remove_all(to, error);
    fs::create_directories(to.parent_path(), error);
    fs::copy(from, to, fs::copy_options::recursive, error);
    if (error) {
        LOG_ERROR("Saves", "copy from {} to {} failed: {}", from.string(), to.string(),
                  error.message());
        return false;
    }
    return true;
}

// The PS4's save name patterns, compared ignoring case.
bool MatchPattern(std::string_view name, std::string_view pattern) {
    if (pattern.empty()) {
        return name.empty();
    }
    if (pattern[0] == '%') {
        for (size_t skip = 0; skip <= name.size(); skip++) {
            if (MatchPattern(name.substr(skip), pattern.substr(1))) {
                return true;
            }
        }
        return false;
    }
    if (name.empty()) {
        return false;
    }
    if (pattern[0] != '_' && std::tolower(static_cast<unsigned char>(pattern[0])) !=
                                 std::tolower(static_cast<unsigned char>(name[0]))) {
        return false;
    }
    return MatchPattern(name.substr(1), pattern.substr(1));
}

std::optional<Mounted>* FindMounted(const MountPoint* mount_point) {
    if (!mount_point) {
        return nullptr;
    }
    const std::string wanted = ToString(mount_point->data, sizeof(mount_point->data));
    for (int i = 0; i < MountSlots; i++) {
        if (g_slots[i] && wanted == "/savedata" + std::to_string(i)) {
            return &g_slots[i];
        }
    }
    return nullptr;
}

bool IsMounted(const fs::path& save) {
    return std::ranges::any_of(g_slots, [&](const auto& slot) { return slot && slot->path == save; });
}

PS4_SYSV_ABI s32 sceSaveDataInitialize3(const void*) {
    std::scoped_lock lock{g_mutex};
    g_initialized = true;
    return 0;
}

PS4_SYSV_ABI s32 sceSaveDataTerminate() {
    std::scoped_lock lock{g_mutex};
    if (std::ranges::any_of(g_slots, [](const auto& slot) { return slot.has_value(); })) {
        return ErrorBusy;
    }
    g_initialized = false;
    return 0;
}

PS4_SYSV_ABI s32 sceSaveDataMount2(const Mount2* mount, MountResult* result) {
    if (!mount || !mount->dir_name || !result) {
        return ErrorParameter;
    }
    std::scoped_lock lock{g_mutex};
    if (!g_initialized) {
        return ErrorNotInitialized;
    }
    const std::string name = ToString(mount->dir_name->data, sizeof(mount->dir_name->data));
    const fs::path save = SavePath(mount->user_id, name);
    if (IsMounted(save)) {
        return ErrorBusy;
    }
    const auto free_slot = std::ranges::find_if(g_slots, [](const auto& slot) { return !slot; });
    if (free_slot == g_slots.end()) {
        return ErrorMountFull;
    }

    std::error_code error;
    const bool exists = fs::exists(save, error);
    const bool create = mount->mode & mount_mode::Create;
    const bool create_if_missing = mount->mode & mount_mode::CreateIfMissing;
    if (!exists && !create && !create_if_missing) {
        LOG_INFO("Saves", "Mount2 \"{}\": does not exist", name);
        return ErrorNotFound;
    }
    if (exists && create) {
        return ErrorExists;
    }
    std::optional<StoredInfo> info;
    if (exists) {
        info = ReadInfo(save);
        if (!info) {
            LOG_ERROR("Saves", "Mount2 \"{}\": {} has no valid sce_sys/param.bin", name,
                      save.string());
            return ErrorBroken;
        }
    } else {
        info.emplace();
        std::memcpy(info->magic, StoredMagic, sizeof(StoredMagic));
        info->blocks = std::clamp<u64>(mount->blocks, MinBlocks, MaxBlocks);
        if (!WriteInfo(save, *info)) {
            LOG_ERROR("Saves", "could not create {}", save.string());
            return ErrorInternal;
        }
    }

    const int slot = static_cast<int>(free_slot - g_slots.begin());
    const std::string mount_point = "/savedata" + std::to_string(slot);
    const bool read_only = mount->mode & mount_mode::ReadOnly;
    *free_slot = Mounted{mount->user_id, name, save, read_only};
    hle::kernel::MountDirectory(mount_point, save, read_only);

    std::memset(result, 0, sizeof(*result));
    CopyString(result->mount_point.data, mount_point);
    result->status = !exists && create_if_missing ? 1 : 0;
    LOG_INFO("Saves", "\"{}\" mounted at {} ({}{}, mode {:#x})", name, mount_point,
             read_only ? "read-only" : "read-write", exists ? "" : ", just created",
             mount->mode);
    if (read_only && name.find("Progress") != std::string::npos) {
        input::NoteSaveLoaded(); // a controller replay starts here
    }
    return 0;
}

s32 Unmount(const MountPoint* mount_point, bool backup) {
    std::scoped_lock lock{g_mutex};
    std::optional<Mounted>* slot = FindMounted(mount_point);
    if (!slot) {
        return ErrorNotMounted;
    }
    const Mounted mounted = **slot;
    hle::kernel::UnmountDirectory(ToString(mount_point->data, sizeof(mount_point->data)));
    slot->reset();
    if (backup && !mounted.read_only) {
        CopyFolder(mounted.path, BackupPath(mounted.user_id, mounted.dir_name));
    }
    LOG_INFO("Saves", "\"{}\" unmounted{}", mounted.dir_name, backup ? " (with backup)" : "");
    return 0;
}

PS4_SYSV_ABI s32 sceSaveDataUmount(const MountPoint* mount_point) {
    return Unmount(mount_point, false);
}

PS4_SYSV_ABI s32 sceSaveDataUmountWithBackup(const MountPoint* mount_point) {
    return Unmount(mount_point, true);
}

PS4_SYSV_ABI s32 sceSaveDataGetMountInfo(const MountPoint* mount_point, MountInfo* info) {
    if (!info) {
        return ErrorParameter;
    }
    std::scoped_lock lock{g_mutex};
    std::optional<Mounted>* slot = FindMounted(mount_point);
    if (!slot) {
        return ErrorNotMounted;
    }
    const auto stored = ReadInfo((*slot)->path);
    const u64 blocks = stored ? stored->blocks : MinBlocks;
    std::memset(info, 0, sizeof(*info));
    info->blocks = blocks;
    info->free_blocks = FreeBlocks((*slot)->path, blocks);
    return 0;
}

PS4_SYSV_ABI s32 sceSaveDataSetParam(const MountPoint* mount_point, ParamType type,
                                     const void* buffer, u64 size) {
    if (!buffer) {
        return ErrorParameter;
    }
    std::scoped_lock lock{g_mutex};
    std::optional<Mounted>* slot = FindMounted(mount_point);
    if (!slot) {
        return ErrorNotMounted;
    }
    if ((*slot)->read_only) {
        return ErrorBadMounted;
    }
    auto info = ReadInfo((*slot)->path);
    if (!info) {
        return ErrorBroken;
    }
    const char* text = static_cast<const char*>(buffer);
    switch (type) {
    case ParamType::All:
        if (size < sizeof(Param)) {
            return ErrorParameter;
        }
        std::memcpy(&info->param, buffer, sizeof(Param));
        break;
    case ParamType::Title:
        CopyString(info->param.title, ToString(text, sizeof(info->param.title)));
        break;
    case ParamType::Subtitle:
        CopyString(info->param.subtitle, ToString(text, sizeof(info->param.subtitle)));
        break;
    case ParamType::Detail:
        CopyString(info->param.detail, ToString(text, sizeof(info->param.detail)));
        break;
    case ParamType::UserParam:
        std::memcpy(&info->param.user_param, buffer, sizeof(u32));
        break;
    default:
        return ErrorParameter;
    }
    if (!WriteInfo((*slot)->path, *info)) {
        return ErrorInternal;
    }
    LOG_INFO("Saves", "\"{}\": title \"{}\", subtitle \"{}\"", (*slot)->dir_name,
             ToString(info->param.title, sizeof(info->param.title)),
             ToString(info->param.subtitle, sizeof(info->param.subtitle)));
    return 0;
}

PS4_SYSV_ABI s32 sceSaveDataSaveIcon(const MountPoint* mount_point, const Icon* icon) {
    if (!icon || !icon->buffer) {
        return ErrorParameter;
    }
    std::scoped_lock lock{g_mutex};
    std::optional<Mounted>* slot = FindMounted(mount_point);
    if (!slot) {
        return ErrorNotMounted;
    }
    if ((*slot)->read_only) {
        return ErrorBadMounted;
    }
    std::error_code error;
    fs::create_directories((*slot)->path / "sce_sys", error);
    std::ofstream file((*slot)->path / "sce_sys" / "icon0.png", std::ios::binary | std::ios::trunc);
    file.write(static_cast<const char*>(icon->buffer),
               static_cast<std::streamsize>(std::min(icon->buffer_size, icon->data_size)));
    return file ? 0 : ErrorInternal;
}

PS4_SYSV_ABI s32 sceSaveDataCheckBackupData(const CheckBackupData* check) {
    if (!check || !check->dir_name) {
        return ErrorParameter;
    }
    std::scoped_lock lock{g_mutex};
    const std::string name = ToString(check->dir_name->data, sizeof(check->dir_name->data));
    if (IsMounted(SavePath(check->user_id, name))) {
        return ErrorBusy;
    }
    const fs::path backup = BackupPath(check->user_id, name);
    const auto info = ReadInfo(backup);
    if (!info) {
        return ErrorNotFound;
    }
    if (check->param) {
        *check->param = info->param;
    }
    if (check->icon && check->icon->buffer) {
        std::ifstream file(backup / "sce_sys" / "icon0.png", std::ios::binary);
        file.read(static_cast<char*>(check->icon->buffer),
                  static_cast<std::streamsize>(check->icon->buffer_size));
        check->icon->data_size = static_cast<u64>(file.gcount());
    }
    return 0;
}

PS4_SYSV_ABI s32 sceSaveDataRestoreBackupData(const RestoreBackupData* restore) {
    if (!restore || !restore->dir_name) {
        return ErrorParameter;
    }
    std::scoped_lock lock{g_mutex};
    const std::string name = ToString(restore->dir_name->data, sizeof(restore->dir_name->data));
    const fs::path save = SavePath(restore->user_id, name);
    if (IsMounted(save)) {
        return ErrorBusy;
    }
    const fs::path backup = BackupPath(restore->user_id, name);
    if (!ReadInfo(backup)) {
        return ErrorNotFound;
    }
    LOG_INFO("Saves", "\"{}\" restored from backup", name);
    return CopyFolder(backup, save) ? 0 : ErrorInternal;
}

PS4_SYSV_ABI s32 sceSaveDataDelete(const Delete* del) {
    if (!del || !del->dir_name) {
        return ErrorParameter;
    }
    std::scoped_lock lock{g_mutex};
    const std::string name = ToString(del->dir_name->data, sizeof(del->dir_name->data));
    const fs::path save = SavePath(del->user_id, name);
    if (IsMounted(save)) {
        return ErrorBusy;
    }
    std::error_code error;
    if (!fs::exists(save, error)) {
        return ErrorNotFound;
    }
    fs::remove_all(save, error);
    fs::remove_all(BackupPath(del->user_id, name), error);
    LOG_INFO("Saves", "\"{}\" deleted", name);
    return error ? ErrorInternal : 0;
}

PS4_SYSV_ABI s32 sceSaveDataDirNameSearch(const DirNameSearchCond* cond,
                                          DirNameSearchResult* result) {
    if (!cond || !result) {
        return ErrorParameter;
    }
    std::scoped_lock lock{g_mutex};
    if (!g_initialized) {
        return ErrorNotInitialized;
    }
    const std::string pattern =
        cond->dir_name ? ToString(cond->dir_name->data, sizeof(cond->dir_name->data)) : "";

    struct Found {
        std::string name;
        StoredInfo info;
        u64 free_blocks;
    };
    std::vector<Found> found;
    std::error_code error;
    for (const auto& entry : fs::directory_iterator(UserDir(cond->user_id), error)) {
        const std::string name = entry.path().filename().string();
        if (!entry.is_directory(error) || name.starts_with('.') || name.starts_with("sce_") ||
            (!pattern.empty() && !MatchPattern(name, pattern))) {
            continue;
        }
        if (auto info = ReadInfo(entry.path())) {
            found.push_back({name, *info, FreeBlocks(entry.path(), info->blocks)});
        }
    }

    auto sort_by = [&](auto key) {
        std::ranges::stable_sort(found, {}, key);
    };
    switch (cond->key) {
    case SortKey::UserParam:
        sort_by([](const Found& f) { return f.info.param.user_param; });
        break;
    case SortKey::Blocks:
        sort_by([](const Found& f) { return f.info.blocks; });
        break;
    case SortKey::Mtime:
        sort_by([](const Found& f) { return f.info.param.mtime; });
        break;
    case SortKey::FreeBlocks:
        sort_by([](const Found& f) { return f.free_blocks; });
        break;
    default:
        sort_by([](const Found& f) { return f.name; });
        break;
    }
    if (cond->descending) {
        std::ranges::reverse(found);
    }

    const u32 count = std::min<u32>(result->dir_names_capacity, static_cast<u32>(found.size()));
    result->hit_count = static_cast<u32>(found.size());
    result->set_count = count;
    for (u32 i = 0; i < count && result->dir_names; i++) {
        CopyString(result->dir_names[i].data, found[i].name);
        if (result->params) {
            result->params[i] = found[i].info.param;
        }
        if (result->infos) {
            std::memset(&result->infos[i], 0, sizeof(SearchInfo));
            result->infos[i].blocks = found[i].info.blocks;
            result->infos[i].free_blocks = found[i].free_blocks;
        }
    }
    LOG_INFO("Saves", "search \"{}\": {} save(s)", pattern, found.size());
    return 0;
}

} // namespace

void Initialize(const std::filesystem::path& data_dir) {
    std::scoped_lock lock{g_mutex};
    g_root = data_dir / "saves";
    std::error_code error;
    fs::create_directories(g_root, error);
    LOG_INFO("Saves", "saves in {}", g_root.string());
}

HLE_FUNCTION("libSceSaveData", "sceSaveDataInitialize3", sceSaveDataInitialize3);
HLE_FUNCTION("libSceSaveData", "sceSaveDataTerminate", sceSaveDataTerminate);
HLE_FUNCTION("libSceSaveData", "sceSaveDataMount2", sceSaveDataMount2);
HLE_FUNCTION("libSceSaveData", "sceSaveDataUmount", sceSaveDataUmount);
HLE_FUNCTION("libSceSaveData", "sceSaveDataUmountWithBackup", sceSaveDataUmountWithBackup);
HLE_FUNCTION("libSceSaveData", "sceSaveDataGetMountInfo", sceSaveDataGetMountInfo);
HLE_FUNCTION("libSceSaveData", "sceSaveDataSetParam", sceSaveDataSetParam);
HLE_FUNCTION("libSceSaveData", "sceSaveDataSaveIcon", sceSaveDataSaveIcon);
HLE_FUNCTION("libSceSaveData", "sceSaveDataCheckBackupData", sceSaveDataCheckBackupData);
HLE_FUNCTION("libSceSaveData", "sceSaveDataRestoreBackupData", sceSaveDataRestoreBackupData);
HLE_FUNCTION("libSceSaveData", "sceSaveDataDelete", sceSaveDataDelete);
HLE_FUNCTION("libSceSaveData", "sceSaveDataDirNameSearch", sceSaveDataDirNameSearch);

} // namespace hle::savedata
