#include "common/sha256.h"

#include <bit>
#include <cstring>

namespace {

// First 32 bits of the fractional parts of the cube roots of the first 64 primes (FIPS 180-4,
// section 4.2.2).
constexpr std::array<u32, 64> K = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

} // namespace

Sha256::Sha256()
    : state{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
            0x5be0cd19} {}

void Sha256::Block(const u8* block) {
    std::array<u32, 64> w;
    for (int i = 0; i < 16; i++) {
        w[i] = u32(block[i * 4]) << 24 | u32(block[i * 4 + 1]) << 16 |
               u32(block[i * 4 + 2]) << 8 | u32(block[i * 4 + 3]);
    }
    for (int i = 16; i < 64; i++) {
        const u32 s0 = std::rotr(w[i - 15], 7) ^ std::rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const u32 s1 = std::rotr(w[i - 2], 17) ^ std::rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    u32 a = state[0], b = state[1], c = state[2], d = state[3];
    u32 e = state[4], f = state[5], g = state[6], h = state[7];
    for (int i = 0; i < 64; i++) {
        const u32 s1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
        const u32 ch = (e & f) ^ (~e & g);
        const u32 t1 = h + s1 + ch + K[i] + w[i];
        const u32 s0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
        const u32 maj = (a & b) ^ (a & c) ^ (b & c);
        const u32 t2 = s0 + maj;
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    state[0] += a, state[1] += b, state[2] += c, state[3] += d;
    state[4] += e, state[5] += f, state[6] += g, state[7] += h;
}

void Sha256::Update(const void* data, std::size_t size) {
    const auto* bytes = static_cast<const u8*>(data);
    length += size;
    while (size > 0) {
        const std::size_t n = std::min(size, buffer.size() - buffered);
        std::memcpy(buffer.data() + buffered, bytes, n);
        buffered += n, bytes += n, size -= n;
        if (buffered == buffer.size()) {
            Block(buffer.data());
            buffered = 0;
        }
    }
}

std::array<u8, 32> Sha256::Final() {
    // Padding: 0x80, zeros up to 56 bytes in the last block, then the length in bits (big-endian).
    const u64 bits = length * 8;
    const u8 one = 0x80;
    Update(&one, 1);
    const u8 zero = 0;
    while (buffered != 56) {
        Update(&zero, 1);
    }
    for (int i = 7; i >= 0; i--) {
        const u8 byte = u8(bits >> (i * 8));
        Update(&byte, 1);
    }
    std::array<u8, 32> digest;
    for (int i = 0; i < 8; i++) {
        digest[i * 4] = u8(state[i] >> 24);
        digest[i * 4 + 1] = u8(state[i] >> 16);
        digest[i * 4 + 2] = u8(state[i] >> 8);
        digest[i * 4 + 3] = u8(state[i]);
    }
    return digest;
}
