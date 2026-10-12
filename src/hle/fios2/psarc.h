#pragma once

#include <filesystem>

#include "common/host_file.h"
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "common/types.h"

namespace hle::fios2 {

// A read-only PSARC archive, the format of the game's pak*.psarc files. Each file inside is
// stored as a run of blocks (64 KiB here), each either zlib-compressed or stored as is.
class PsarcArchive {
public:
    struct Entry {
        u64 size;
        u64 offset;      // of its first block in the archive
        u32 first_block; // index in the block size table
    };

    // nullptr (and a message in the log) if the file cannot be used.
    static std::unique_ptr<PsarcArchive> Open(const std::filesystem::path& path);
    ~PsarcArchive();

    // `name` is the path inside the archive, with or without the leading '/'; case is ignored.
    const Entry* Find(std::string_view name) const;

    // Copies up to `length` bytes starting at `position` of the entry. Returns the number of
    // bytes copied, or -1 if the archive data is damaged.
    s64 Read(const Entry& entry, u64 position, void* buffer, u64 length) const;

    const std::filesystem::path& Path() const {
        return path_;
    }
    size_t FileCount() const {
        return entries_.size();
    }

private:
    PsarcArchive() = default;
    std::string Key(std::string_view name) const;
    bool ReadBlock(const Entry& entry, u64 block_in_entry, std::vector<u8>& out) const;

    std::filesystem::path path_;
    u64 id_ = 0; // unique per archive opened, for the block cache
    host_file::Handle file_ = host_file::Invalid;
    u32 block_size_ = 0;
    std::vector<u32> block_sizes_;    // stored size of each block; 0 means a full block
    std::vector<u64> block_starts_;   // prefix sums of the stored sizes
    std::unordered_map<std::string, Entry> entries_;
};

} // namespace hle::fios2
