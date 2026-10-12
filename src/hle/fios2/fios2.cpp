// libSceFios2, the PS4 file I/O library the game uses for all its files. It opens plain files
// and files inside mounted PSARC archives (the pak*.psarc files).
//
// Operations the game starts asynchronously (sceFiosFHOpen, sceFiosFHRead...) are carried out
// right away, and the operation handle the game gets back is already complete.
//
// Signatures and structures come from how the game calls the library
// and from the parameter names in the library's own trace messages.

#include <atomic>

#include <cerrno>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "common/host_file.h"
#include "common/log.h"
#include "common/types.h"
#include "hle/fios2/psarc.h"
#include "hle/kernel/filesystem.h"
#include "hle/registry.h"

namespace hle::fios2 {

namespace {

// SCE_FIOS_ERROR_*, numbered in the order of the library's own table.
constexpr s32 FiosError(u32 index) {
    return static_cast<s32>(0x80820000u + index);
}
constexpr s32 ErrorBadPath = FiosError(0x05);
constexpr s32 ErrorBadPtr = FiosError(0x06);
constexpr s32 ErrorBadOffset = FiosError(0x07);
constexpr s32 ErrorBadSize = FiosError(0x08);
constexpr s32 ErrorBadOp = FiosError(0x0A);
constexpr s32 ErrorBadFh = FiosError(0x0B);
constexpr s32 ErrorEof = FiosError(0x10);
constexpr s32 ErrorAccess = FiosError(0x13);
constexpr s32 ErrorDecompression = FiosError(0x14);
constexpr s32 ErrorReadOnly = FiosError(0x15);
constexpr s32 ErrorBadArchive = FiosError(0x1F);

// Open flags (low 16 bits of OpenParams::flags).
constexpr u32 OpenRead = 0x01;
constexpr u32 OpenWrite = 0x02;
constexpr u32 OpenAppend = 0x04;
constexpr u32 OpenCreate = 0x08;
constexpr u32 OpenTruncate = 0x10;

constexpr s32 SeekSet = 0;
constexpr s32 SeekCurrent = 1;
constexpr s32 SeekEnd = 2;

struct OpAttr {
    s64 deadline;
    void* callback;
    void* callback_context;
    u32 priority_and_flags;
    u32 user_tag;
    void* user_pointer;
    void* reserved;
};

struct OpenParams {
    u32 flags; // open flags in the low 16 bits
    u32 reserved;
    void* buffer;
    u64 buffer_length;
};

struct Buffer {
    void* pointer;
    u64 length;
};

// Dates are nanoseconds since 1970.
struct Stat {
    s64 size;
    s64 access_date;
    s64 modification_date;
    s64 creation_date;
    u32 flags;
    u32 reserved;
    s64 uid;
    s64 gid;
    s64 device;
    s64 inode;
    s64 mode;
};
static_assert(sizeof(Stat) == 0x50, "the game reserves exactly 0x50 bytes");

constexpr u32 StatDirectory = 0x1;
constexpr u32 StatReadable = 0x2;
constexpr u32 StatWritable = 0x4;

// An open file: a host file, or a file inside a mounted archive.
struct OpenFile {
    std::string path;
    host_file::Handle file = host_file::Invalid;
    const PsarcArchive* archive = nullptr;
    const PsarcArchive::Entry* entry = nullptr;
    std::mutex mutex; // guards position
    s64 position = 0;

    ~OpenFile() {
        host_file::Close(file);
    }

    s64 Size() const {
        if (entry) {
            return static_cast<s64>(entry->size);
        }
        const s64 size = host_file::Size(file);
        return size >= 0 ? size : ErrorAccess;
    }

    // Bytes read, or an error code.
    s64 ReadAt(void* buffer, s64 length, s64 offset) const {
        if (length == 0) {
            return 0;
        }
        s64 count;
        if (entry) {
            count = archive->Read(*entry, offset, buffer, length);
            if (count < 0) {
                return ErrorDecompression;
            }
        } else {
            count = host_file::ReadAt(file, buffer, static_cast<u64>(length),
                                      static_cast<u64>(offset));
            if (count < 0) {
                return ErrorAccess;
            }
        }
        return count == 0 ? ErrorEof : count;
    }

    s64 WriteAt(const void* buffer, s64 length, s64 offset) const {
        if (entry) {
            return ErrorReadOnly;
        }
        const s64 written = host_file::WriteAt(file, buffer, static_cast<u64>(length),
                                               static_cast<u64>(offset));
        return written < 0 ? ErrorAccess : written;
    }
};

struct Mount {
    s32 handle;
    std::string point; // ends with '/'
    std::unique_ptr<PsarcArchive> archive;
};

struct Operation {
    s32 error;
    s64 count;
};

struct State {
    std::mutex mutex;
    std::map<s32, std::shared_ptr<OpenFile>> files;
    std::map<s32, Operation> operations;
    std::vector<Mount> mounts;
    s32 next_handle = 1;
    s32 next_operation = 1;
};

State& Fios() {
    static State state;
    return state;
}

// "/app0//a/./b" -> "/app0/a/b"
std::string Normalize(std::string_view path) {
    std::string result = "/";
    size_t start = 0;
    while (start < path.size()) {
        const size_t end = std::min(path.find('/', start), path.size());
        const std::string_view part = path.substr(start, end - start);
        if (!part.empty() && part != ".") {
            if (result.size() > 1) {
                result += '/';
            }
            result += part;
        }
        start = end + 1;
    }
    return result;
}

struct ArchiveFile {
    const PsarcArchive* archive = nullptr;
    const PsarcArchive::Entry* entry = nullptr;
};

ArchiveFile FindInArchives(const std::string& path) {
    std::scoped_lock lock{Fios().mutex};
    for (const Mount& mount : Fios().mounts) {
        if (path.starts_with(mount.point)) {
            if (const auto* entry = mount.archive->Find(path.substr(mount.point.size()))) {
                return {mount.archive.get(), entry};
            }
        }
    }
    return {};
}

std::shared_ptr<OpenFile> GetFile(s32 handle) {
    std::scoped_lock lock{Fios().mutex};
    const auto it = Fios().files.find(handle);
    return it == Fios().files.end() ? nullptr : it->second;
}

s32 NewOperation(s32 error, s64 count) {
    std::scoped_lock lock{Fios().mutex};
    const s32 id = Fios().next_operation++;
    Fios().operations[id] = {error, count};
    return id;
}

// Error part and count part of a result that is either a count or an error code.
s32 ErrorOf(s64 result) {
    return result < 0 ? static_cast<s32>(result) : 0;
}
s64 CountOf(s64 result) {
    return result < 0 ? 0 : result;
}

s32 OpenPath(const char* raw_path, const OpenParams* params, s32* out) {
    if (!raw_path || !out) {
        return ErrorBadPtr;
    }
    const std::string path = Normalize(raw_path);
    const u32 flags = params ? (params->flags & 0xFFFF) : OpenRead;
    const bool writes = flags & (OpenWrite | OpenAppend | OpenCreate | OpenTruncate);

    auto file = std::make_shared<OpenFile>();
    file->path = path;
    if (const ArchiveFile found = FindInArchives(path); found.entry) {
        if (writes) {
            return ErrorReadOnly;
        }
        file->archive = found.archive;
        file->entry = found.entry;
    } else {
        const kernel::HostPath host = kernel::ToHostPath(path);
        if (!host.valid) {
            LOG_INFO("Fios2", "\"{}\" is outside the mount points", path);
            return ErrorBadPath;
        }
        if (writes && host.read_only) {
            LOG_WARNING("Fios2", "\"{}\": write refused (read-only)", path);
            return ErrorReadOnly;
        }
        int host_flags = 0;
        if (flags & OpenRead) {
            host_flags |= host_file::OpenRead;
        }
        if (flags & OpenWrite) {
            host_flags |= host_file::OpenWrite;
        }
        if (flags & OpenAppend) {
            host_flags |= host_file::OpenAppend;
        }
        if (flags & OpenCreate) {
            host_flags |= host_file::OpenCreate;
        }
        if (flags & OpenTruncate) {
            host_flags |= host_file::OpenTruncate;
        }
        host_file::OpenError error;
        file->file = host_file::Open(host.path, host_flags, &error);
        if (file->file == host_file::Invalid) {
            LOG_INFO("Fios2", "\"{}\" not found", path);
            return error == host_file::OpenError::NotFound ? ErrorBadPath : ErrorAccess;
        }
    }

    std::scoped_lock lock{Fios().mutex};
    const s32 handle = Fios().next_handle++;
    Fios().files[handle] = file;
    *out = handle;
    LOG_DEBUG("Fios2", "opened \"{}\" (fh {}, {})", path, handle, file->entry ? "psarc" : "disk");
    return 0;
}

s32 CloseHandle(s32 handle) {
    std::scoped_lock lock{Fios().mutex};
    return Fios().files.erase(handle) ? 0 : ErrorBadFh;
}

s64 ReadSequential(s32 handle, void* buffer, s64 length) {
    if (length < 0) {
        return ErrorBadSize;
    }
    const auto file = GetFile(handle);
    if (!file) {
        return ErrorBadFh;
    }
    std::scoped_lock lock{file->mutex};
    const s64 result = file->ReadAt(buffer, length, file->position);
    if (result > 0) {
        file->position += result;
    }
    return result;
}

s64 ReadPositioned(s32 handle, void* buffer, s64 length, s64 offset) {
    if (length < 0) {
        return ErrorBadSize;
    }
    if (offset < 0) {
        return ErrorBadOffset;
    }
    const auto file = GetFile(handle);
    return file ? file->ReadAt(buffer, length, offset) : ErrorBadFh;
}

s64 ReadVector(s32 handle, const Buffer* buffers, s32 count, s64 offset, bool positioned) {
    if (!buffers || count < 0) {
        return ErrorBadPtr;
    }
    s64 total = 0;
    for (s32 i = 0; i < count; i++) {
        const s64 result =
            positioned ? ReadPositioned(handle, buffers[i].pointer, buffers[i].length, offset + total)
                       : ReadSequential(handle, buffers[i].pointer, buffers[i].length);
        if (result < 0) {
            return total > 0 ? total : result;
        }
        total += result;
        if (static_cast<u64>(result) < buffers[i].length) {
            break;
        }
    }
    return total;
}

s32 FillStat(const std::string& path, Stat* out) {
    *out = {};
    if (const ArchiveFile found = FindInArchives(path); found.entry) {
        out->size = static_cast<s64>(found.entry->size);
        out->flags = StatReadable;
        return 0;
    }
    const kernel::HostPath host = kernel::ToHostPath(path);
    host_file::Info info;
    if (!host.valid || !host_file::Stat(host.path, info)) {
        return ErrorBadPath;
    }
    out->size = static_cast<s64>(info.size);
    out->access_date = static_cast<s64>(info.access_ns);
    out->modification_date = static_cast<s64>(info.modify_ns);
    out->creation_date = static_cast<s64>(info.change_ns);
    out->flags = StatReadable | (host.read_only ? 0 : StatWritable) |
                 (info.is_directory ? StatDirectory : 0);
    out->mode = info.mode;
    return 0;
}

// ---- Library ----

PS4_SYSV_ABI s32 sceFiosInitialize(const void*) {
    LOG_INFO("Fios2", "initialized");
    return 0;
}

// The game hands these two filters to sceFiosIOFilterAdd; it never calls them itself. Archives
// are always handled here, so adding them changes nothing.
PS4_SYSV_ABI s32 sceFiosIOFilterPsarcDearchiver() {
    LOG_ERROR("Fios2", "sceFiosIOFilterPsarcDearchiver called directly");
    return 0;
}

PS4_SYSV_ABI s32 sceFiosIOFilterCache() {
    LOG_ERROR("Fios2", "sceFiosIOFilterCache called directly");
    return 0;
}

PS4_SYSV_ABI s32 sceFiosIOFilterAdd(s32 index, void* filter, void*) {
    LOG_DEBUG("Fios2", "filter {:#x} added at position {}", reinterpret_cast<u64>(filter), index);
    return 0;
}

// ---- Archives ----

PS4_SYSV_ABI s64 sceFiosArchiveGetMountBufferSizeSync(const OpAttr*, const char* path, const void*) {
    if (!path) {
        return ErrorBadPtr;
    }
    const kernel::HostPath host = kernel::ToHostPath(Normalize(path));
    host_file::Info info;
    if (!host.valid || !host_file::Stat(host.path, info)) {
        LOG_ERROR("Fios2", "archive to mount not found: {}", path);
        return ErrorBadPath;
    }
    // The game allocates this much and passes it to the mount call; we do not need it.
    return 64 * 1024;
}

PS4_SYSV_ABI s32 sceFiosArchiveMountSync(const OpAttr*, s32* out, const char* archive_path,
                                         const char* mount_point, Buffer /*buffer*/,
                                         const void* /*params*/) {
    if (!out || !archive_path || !mount_point) {
        return ErrorBadPtr;
    }
    const kernel::HostPath host = kernel::ToHostPath(Normalize(archive_path));
    auto archive = host.valid ? PsarcArchive::Open(host.path) : nullptr;
    if (!archive) {
        return ErrorBadArchive;
    }
    std::string point = Normalize(mount_point);
    if (!point.ends_with('/')) {
        point += '/';
    }
    LOG_INFO("Fios2", "{} mounted at {} ({} files)", archive_path, point, archive->FileCount());
    std::scoped_lock lock{Fios().mutex};
    const s32 handle = Fios().next_handle++;
    // Later mounts take precedence, as in an overlay.
    Fios().mounts.insert(Fios().mounts.begin(), Mount{handle, point, std::move(archive)});
    *out = handle;
    return 0;
}

PS4_SYSV_ABI s32 sceFiosArchiveUnmountSync(const OpAttr*, s32 handle) {
    std::scoped_lock lock{Fios().mutex};
    auto& mounts = Fios().mounts;
    for (auto it = mounts.begin(); it != mounts.end(); ++it) {
        if (it->handle == handle) {
            LOG_INFO("Fios2", "{} unmounted", it->archive->Path().filename().string());
            mounts.erase(it);
            return 0;
        }
    }
    return ErrorBadFh;
}

// ---- Synchronous file functions ----

PS4_SYSV_ABI s32 sceFiosFHOpenSync(const OpAttr*, s32* out, const char* path,
                                   const OpenParams* params) {
    return OpenPath(path, params, out);
}

PS4_SYSV_ABI s32 sceFiosFHCloseSync(const OpAttr*, s32 handle) {
    return CloseHandle(handle);
}

PS4_SYSV_ABI s64 sceFiosFHReadSync(const OpAttr*, s32 handle, void* buffer, s64 length) {
    return ReadSequential(handle, buffer, length);
}

PS4_SYSV_ABI s64 sceFiosFHPreadSync(const OpAttr*, s32 handle, void* buffer, s64 length,
                                    s64 offset) {
    return ReadPositioned(handle, buffer, length, offset);
}

PS4_SYSV_ABI s64 sceFiosFHWriteSync(const OpAttr*, s32 handle, const void* buffer, s64 length) {
    if (length < 0) {
        return ErrorBadSize;
    }
    const auto file = GetFile(handle);
    if (!file) {
        return ErrorBadFh;
    }
    std::scoped_lock lock{file->mutex};
    const s64 result = file->WriteAt(buffer, length, file->position);
    if (result > 0) {
        file->position += result;
    }
    return result;
}

PS4_SYSV_ABI s32 sceFiosFHTruncateSync(const OpAttr*, s32 handle, s64 length) {
    const auto file = GetFile(handle);
    if (!file) {
        return ErrorBadFh;
    }
    if (file->entry) {
        return ErrorReadOnly;
    }
    return host_file::Truncate(file->file, static_cast<u64>(length)) ? 0 : ErrorAccess;
}

PS4_SYSV_ABI s32 sceFiosFHSyncSync(const OpAttr*, s32 handle) {
    const auto file = GetFile(handle);
    if (!file) {
        return ErrorBadFh;
    }
    if (file->file != host_file::Invalid) {
        host_file::Sync(file->file);
    }
    return 0;
}

PS4_SYSV_ABI s64 sceFiosFHGetSize(s32 handle) {
    const auto file = GetFile(handle);
    return file ? file->Size() : ErrorBadFh;
}

PS4_SYSV_ABI s64 sceFiosFHSeek(s32 handle, s64 offset, s32 whence) {
    const auto file = GetFile(handle);
    if (!file) {
        return ErrorBadFh;
    }
    std::scoped_lock lock{file->mutex};
    s64 base;
    switch (whence) {
    case SeekSet: base = 0; break;
    case SeekCurrent: base = file->position; break;
    case SeekEnd: base = file->Size(); break;
    default: return ErrorBadOffset;
    }
    if (base + offset < 0) {
        return ErrorBadOffset;
    }
    file->position = base + offset;
    return file->position;
}

PS4_SYSV_ABI s64 sceFiosFHTell(s32 handle) {
    const auto file = GetFile(handle);
    if (!file) {
        return ErrorBadFh;
    }
    std::scoped_lock lock{file->mutex};
    return file->position;
}

PS4_SYSV_ABI s32 sceFiosFHStatSync(const OpAttr*, s32 handle, Stat* out) {
    if (!out) {
        return ErrorBadPtr;
    }
    const auto file = GetFile(handle);
    return file ? FillStat(file->path, out) : ErrorBadFh;
}

PS4_SYSV_ABI s32 sceFiosStatSync(const OpAttr*, const char* path, Stat* out) {
    if (!path || !out) {
        return ErrorBadPtr;
    }
    return FillStat(Normalize(path), out);
}

PS4_SYSV_ABI bool sceFiosFileExistsSync(const OpAttr*, const char* path) {
    Stat stat;
    return path && FillStat(Normalize(path), &stat) == 0 && !(stat.flags & StatDirectory);
}

PS4_SYSV_ABI s64 sceFiosFileGetSizeSync(const OpAttr*, const char* path) {
    if (!path) {
        return ErrorBadPtr;
    }
    Stat stat;
    const s32 error = FillStat(Normalize(path), &stat);
    return error ? error : stat.size;
}

PS4_SYSV_ABI s32 sceFiosFileDeleteSync(const OpAttr*, const char* path) {
    if (!path) {
        return ErrorBadPtr;
    }
    const kernel::HostPath host = kernel::ToHostPath(Normalize(path));
    if (!host.valid) {
        return ErrorBadPath;
    }
    if (host.read_only) {
        return ErrorReadOnly;
    }
    return host_file::Remove(host.path) ? 0 : ErrorBadPath;
}

// ---- Asynchronous file functions (completed immediately) ----

PS4_SYSV_ABI s32 sceFiosFHOpen(const OpAttr*, s32* out, const char* path, const OpenParams* params) {
    return NewOperation(OpenPath(path, params, out), 0);
}

PS4_SYSV_ABI s32 sceFiosFHClose(const OpAttr*, s32 handle) {
    return NewOperation(CloseHandle(handle), 0);
}

PS4_SYSV_ABI s32 sceFiosFHRead(const OpAttr*, s32 handle, void* buffer, s64 length) {
    const s64 result = ReadSequential(handle, buffer, length);
    return NewOperation(ErrorOf(result), CountOf(result));
}

PS4_SYSV_ABI s32 sceFiosFHPread(const OpAttr*, s32 handle, void* buffer, s64 length, s64 offset) {
    const s64 result = ReadPositioned(handle, buffer, length, offset);
    return NewOperation(ErrorOf(result), CountOf(result));
}

PS4_SYSV_ABI s32 sceFiosFHReadv(const OpAttr*, s32 handle, const Buffer* buffers, s32 count) {
    const s64 result = ReadVector(handle, buffers, count, 0, false);
    return NewOperation(ErrorOf(result), CountOf(result));
}

PS4_SYSV_ABI s32 sceFiosFHPreadv(const OpAttr*, s32 handle, const Buffer* buffers, s32 count,
                                 s64 offset) {
    const s64 result = ReadVector(handle, buffers, count, offset, true);
    return NewOperation(ErrorOf(result), CountOf(result));
}

// ---- Operations ----

template <typename F>
auto WithOperation(s32 id, F&& function) -> decltype(function(std::declval<Operation&>())) {
    std::scoped_lock lock{Fios().mutex};
    const auto it = Fios().operations.find(id);
    if (it == Fios().operations.end()) {
        return ErrorBadOp;
    }
    return function(it->second);
}

PS4_SYSV_ABI bool sceFiosOpIsDone(s32) {
    return true;
}

PS4_SYSV_ABI s32 sceFiosOpWait(s32 id) {
    return WithOperation(id, [](Operation& op) { return op.error; });
}

PS4_SYSV_ABI s32 sceFiosOpGetError(s32 id) {
    return WithOperation(id, [](Operation& op) { return op.error; });
}

PS4_SYSV_ABI s64 sceFiosOpGetActualCount(s32 id) {
    return WithOperation(id, [](Operation& op) -> s64 { return op.count; });
}

PS4_SYSV_ABI s32 sceFiosOpDelete(s32 id) {
    std::scoped_lock lock{Fios().mutex};
    return Fios().operations.erase(id) ? 0 : ErrorBadOp;
}

PS4_SYSV_ABI s32 sceFiosOpCancel(s32) {
    return 0; // already complete
}

} // namespace

HLE_FUNCTION("libSceFios2", "sceFiosInitialize", sceFiosInitialize);
HLE_FUNCTION("libSceFios2", "sceFiosIOFilterAdd", sceFiosIOFilterAdd);
HLE_FUNCTION("libSceFios2", "sceFiosIOFilterPsarcDearchiver", sceFiosIOFilterPsarcDearchiver);
HLE_FUNCTION("libSceFios2", "sceFiosIOFilterCache", sceFiosIOFilterCache);
HLE_FUNCTION("libSceFios2", "sceFiosArchiveGetMountBufferSizeSync", sceFiosArchiveGetMountBufferSizeSync);
HLE_FUNCTION("libSceFios2", "sceFiosArchiveMountSync", sceFiosArchiveMountSync);
HLE_FUNCTION("libSceFios2", "sceFiosArchiveUnmountSync", sceFiosArchiveUnmountSync);
HLE_FUNCTION("libSceFios2", "sceFiosFHOpenSync", sceFiosFHOpenSync);
HLE_FUNCTION("libSceFios2", "sceFiosFHCloseSync", sceFiosFHCloseSync);
HLE_FUNCTION("libSceFios2", "sceFiosFHReadSync", sceFiosFHReadSync);
HLE_FUNCTION("libSceFios2", "sceFiosFHPreadSync", sceFiosFHPreadSync);
HLE_FUNCTION("libSceFios2", "sceFiosFHWriteSync", sceFiosFHWriteSync);
HLE_FUNCTION("libSceFios2", "sceFiosFHTruncateSync", sceFiosFHTruncateSync);
HLE_FUNCTION("libSceFios2", "sceFiosFHSyncSync", sceFiosFHSyncSync);
HLE_FUNCTION("libSceFios2", "sceFiosFHGetSize", sceFiosFHGetSize);
HLE_FUNCTION("libSceFios2", "sceFiosFHSeek", sceFiosFHSeek);
HLE_FUNCTION("libSceFios2", "sceFiosFHTell", sceFiosFHTell);
HLE_FUNCTION("libSceFios2", "sceFiosFHStatSync", sceFiosFHStatSync);
HLE_FUNCTION("libSceFios2", "sceFiosStatSync", sceFiosStatSync);
HLE_FUNCTION("libSceFios2", "sceFiosFileExistsSync", sceFiosFileExistsSync);
HLE_FUNCTION("libSceFios2", "sceFiosFileGetSizeSync", sceFiosFileGetSizeSync);
HLE_FUNCTION("libSceFios2", "sceFiosFileDeleteSync", sceFiosFileDeleteSync);
HLE_FUNCTION("libSceFios2", "sceFiosFHOpen", sceFiosFHOpen);
HLE_FUNCTION("libSceFios2", "sceFiosFHClose", sceFiosFHClose);
HLE_FUNCTION("libSceFios2", "sceFiosFHRead", sceFiosFHRead);
HLE_FUNCTION("libSceFios2", "sceFiosFHPread", sceFiosFHPread);
HLE_FUNCTION("libSceFios2", "sceFiosFHReadv", sceFiosFHReadv);
HLE_FUNCTION("libSceFios2", "sceFiosFHPreadv", sceFiosFHPreadv);
HLE_FUNCTION("libSceFios2", "sceFiosOpIsDone", sceFiosOpIsDone);
HLE_FUNCTION("libSceFios2", "sceFiosOpWait", sceFiosOpWait);
HLE_FUNCTION("libSceFios2", "sceFiosOpGetError", sceFiosOpGetError);
HLE_FUNCTION("libSceFios2", "sceFiosOpGetActualCount", sceFiosOpGetActualCount);
HLE_FUNCTION("libSceFios2", "sceFiosOpDelete", sceFiosOpDelete);
HLE_FUNCTION("libSceFios2", "sceFiosOpCancel", sceFiosOpCancel);

} // namespace hle::fios2
