#include "hle/fios2/psarc.h"

#include <zlib.h>

#include <algorithm>
#include <atomic>
#include <cstring>

#include "common/log.h"
#include "common/md5.h"

namespace hle::fios2 {

namespace {

// Header, all fields big-endian.
struct Header {
    char magic[4]; // "PSAR"
    u16 version_major;
    u16 version_minor;
    char compression[4]; // "zlib" or "lzma"
    u32 toc_length;      // includes this header
    u32 entry_size;      // 30
    u32 entry_count;     // entry 0 is the list of names
    u32 block_size;
    u32 flags;
};
static_assert(sizeof(Header) == 32);

u64 BigEndian(const u8* data, int bytes) {
    u64 value = 0;
    for (int i = 0; i < bytes; i++) {
        value = (value << 8) | data[i];
    }
    return value;
}

bool ReadExact(host_file::Handle file, void* buffer, u64 length, u64 offset) {
    return host_file::ReadAt(file, buffer, length, offset) == static_cast<s64>(length);
}

// The last block decompressed by this thread, so sequential small reads do not decompress the
// same block again.
struct BlockCache {
    u64 archive_id = 0;
    u64 block = ~u64{0};
    std::vector<u8> data;
};
thread_local BlockCache t_cache;

std::atomic<u64> g_next_archive_id{1};

} // namespace

std::unique_ptr<PsarcArchive> PsarcArchive::Open(const std::filesystem::path& path) {
    std::unique_ptr<PsarcArchive> archive{new PsarcArchive};
    archive->path_ = path;
    archive->id_ = g_next_archive_id++;
    archive->file_ = host_file::Open(path, host_file::OpenRead);
    if (archive->file_ == host_file::Invalid) {
        LOG_ERROR("PSARC", "could not open {}", path.string());
        return nullptr;
    }

    u8 raw_header[sizeof(Header)];
    if (!ReadExact(archive->file_, raw_header, sizeof(raw_header), 0) ||
        std::memcmp(raw_header, "PSAR", 4) != 0) {
        LOG_ERROR("PSARC", "{} is not a PSARC archive", path.string());
        return nullptr;
    }
    const u32 toc_length = static_cast<u32>(BigEndian(raw_header + 12, 4));
    const u32 entry_size = static_cast<u32>(BigEndian(raw_header + 16, 4));
    const u32 entry_count = static_cast<u32>(BigEndian(raw_header + 20, 4));
    archive->block_size_ = static_cast<u32>(BigEndian(raw_header + 24, 4));
    if (std::memcmp(raw_header + 8, "zlib", 4) != 0) {
        LOG_ERROR("PSARC", "{}: unsupported compression {:.4s}", path.string(),
                  reinterpret_cast<const char*>(raw_header + 8));
        return nullptr;
    }
    if (entry_size < 30 || entry_count == 0 || toc_length < sizeof(Header) + u64{entry_size} * entry_count) {
        LOG_ERROR("PSARC", "{}: invalid index", path.string());
        return nullptr;
    }

    std::vector<u8> toc(toc_length - sizeof(Header));
    if (!ReadExact(archive->file_, toc.data(), toc.size(), sizeof(Header))) {
        LOG_ERROR("PSARC", "{}: truncated index", path.string());
        return nullptr;
    }

    // Entry: MD5 of the name (16), first block (4), size (5), offset (5). The entries are not in
    // the order of the name list (in these archives they are sorted by hash), so names are
    // matched to entries through the MD5.
    std::vector<Entry> entries(entry_count);
    std::unordered_map<std::string, u32> by_hash;
    for (u32 i = 0; i < entry_count; i++) {
        const u8* e = toc.data() + u64{i} * entry_size;
        entries[i] = {BigEndian(e + 20, 5), BigEndian(e + 25, 5), static_cast<u32>(BigEndian(e + 16, 4))};
        by_hash.emplace(std::string(reinterpret_cast<const char*>(e), 16), i);
    }

    // Block size table: 2, 3 or 4 bytes per block depending on the block size.
    const int width = archive->block_size_ <= 0x10000 ? 2 : archive->block_size_ <= 0x1000000 ? 3 : 4;
    const u64 table_start = u64{entry_count} * entry_size;
    for (u64 at = table_start; at + width <= toc.size(); at += width) {
        archive->block_sizes_.push_back(static_cast<u32>(BigEndian(toc.data() + at, width)));
    }
    archive->block_starts_.resize(archive->block_sizes_.size() + 1);
    for (size_t i = 0; i < archive->block_sizes_.size(); i++) {
        const u32 stored = archive->block_sizes_[i] ? archive->block_sizes_[i] : archive->block_size_;
        archive->block_starts_[i + 1] = archive->block_starts_[i] + stored;
    }

    // Entry 0 is the list of names, separated by '\0' or '\n'.
    std::string names(entries[0].size, '\0');
    if (archive->Read(entries[0], 0, names.data(), names.size()) != static_cast<s64>(names.size())) {
        LOG_ERROR("PSARC", "{}: unreadable name list", path.string());
        return nullptr;
    }
    const char separator = names.find('\0') != std::string::npos ? '\0' : '\n';
    size_t start = 0, unmatched = 0;
    while (start < names.size()) {
        const size_t end = std::min(names.find(separator, start), names.size());
        const std::string_view name = std::string_view(names).substr(start, end - start);
        start = end + 1;
        if (name.empty()) {
            continue;
        }
        const auto digest = Md5(name);
        const auto it = by_hash.find(std::string(reinterpret_cast<const char*>(digest.data()), 16));
        if (it == by_hash.end()) {
            unmatched++;
            continue;
        }
        archive->entries_[archive->Key(name)] = entries[it->second];
    }
    if (unmatched != 0) {
        LOG_WARNING("PSARC", "{}: {} names without a matching entry", path.filename().string(),
                    unmatched);
    }
    LOG_INFO("PSARC", "{}: {} files", path.filename().string(), archive->entries_.size());
    return archive;
}

PsarcArchive::~PsarcArchive() {
    if (file_ != host_file::Invalid) {
        host_file::Close(file_);
    }
}

std::string PsarcArchive::Key(std::string_view name) const {
    while (name.starts_with('/')) {
        name.remove_prefix(1);
    }
    // The game does not always use the same capitalization as the archive
    // ("Shaders/_cmn" for "shaders/_cmn"), so names are compared ignoring case.
    std::string key(name);
    std::ranges::transform(key, key.begin(), [](unsigned char c) { return std::tolower(c); });
    return key;
}

const PsarcArchive::Entry* PsarcArchive::Find(std::string_view name) const {
    const auto it = entries_.find(Key(name));
    return it == entries_.end() ? nullptr : &it->second;
}

bool PsarcArchive::ReadBlock(const Entry& entry, u64 block_in_entry, std::vector<u8>& out) const {
    const u64 block = entry.first_block + block_in_entry;
    if (block >= block_sizes_.size()) {
        return false;
    }
    const u64 expected = std::min<u64>(block_size_, entry.size - block_in_entry * block_size_);
    const u64 stored = block_sizes_[block] ? block_sizes_[block] : block_size_;
    const u64 offset = entry.offset + (block_starts_[block] - block_starts_[entry.first_block]);

    std::vector<u8> raw(stored);
    if (!ReadExact(file_, raw.data(), stored, offset)) {
        return false;
    }
    out.resize(expected);
    // A block whose stored size equals its real size was kept uncompressed.
    if (stored == expected) {
        std::memcpy(out.data(), raw.data(), expected);
        return true;
    }
    uLongf out_size = expected;
    if (uncompress(out.data(), &out_size, raw.data(), stored) != Z_OK || out_size != expected) {
        LOG_ERROR("PSARC", "{}: corrupted block {}", path_.filename().string(), block);
        return false;
    }
    return true;
}

s64 PsarcArchive::Read(const Entry& entry, u64 position, void* buffer, u64 length) const {
    auto* out = static_cast<u8*>(buffer);
    const u64 end = std::min(entry.size, position + length);
    u64 copied = 0;
    while (position < end) {
        const u64 block_in_entry = position / block_size_;
        const u64 block = entry.first_block + block_in_entry;
        if (t_cache.archive_id != id_ || t_cache.block != block) {
            if (!ReadBlock(entry, block_in_entry, t_cache.data)) {
                t_cache.archive_id = 0;
                return -1;
            }
            t_cache.archive_id = id_;
            t_cache.block = block;
        }
        const u64 within = position % block_size_;
        const u64 count = std::min<u64>(end - position, t_cache.data.size() - within);
        std::memcpy(out + copied, t_cache.data.data() + within, count);
        copied += count;
        position += count;
    }
    return static_cast<s64>(copied);
}

} // namespace hle::fios2
