#pragma once

#include <array>
#include <cstddef>

#include "common/types.h"

// SHA-256 (FIPS 180-4), incremental. Used to recognize the game executable the port was made
// for (src/loader/game_version.cpp), not for security.
class Sha256 {
public:
    Sha256();
    void Update(const void* data, std::size_t size);
    std::array<u8, 32> Final();

private:
    void Block(const u8* block);

    std::array<u32, 8> state;
    std::array<u8, 64> buffer{};
    std::size_t buffered = 0;
    u64 length = 0; // bytes hashed so far
};
