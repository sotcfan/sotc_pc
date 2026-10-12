#include "loader/nid.h"

#include <array>
#include <bit>
#include <cstring>
#include <vector>

namespace loader {

namespace {

constexpr std::string_view Alphabet =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-";

constexpr std::array<u8, 16> NidSuffix = {0x51, 0x8D, 0x64, 0xA6, 0x35, 0xDE, 0xD8, 0xC1,
                                          0xE6, 0xB0, 0x39, 0xB1, 0xC3, 0xE5, 0x52, 0x30};

// Plain SHA-1 (FIPS 180-4). Only used to compute NIDs, never for security.
std::array<u8, 20> Sha1(const std::vector<u8>& message) {
    u32 h[5] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};

    std::vector<u8> data = message;
    const u64 bit_length = u64{message.size()} * 8;
    data.push_back(0x80);
    while (data.size() % 64 != 56) {
        data.push_back(0);
    }
    for (int i = 7; i >= 0; i--) {
        data.push_back(static_cast<u8>(bit_length >> (i * 8)));
    }

    for (size_t block = 0; block < data.size(); block += 64) {
        u32 w[80];
        for (int i = 0; i < 16; i++) {
            const u8* p = &data[block + i * 4];
            w[i] = (u32{p[0]} << 24) | (u32{p[1]} << 16) | (u32{p[2]} << 8) | p[3];
        }
        for (int i = 16; i < 80; i++) {
            w[i] = std::rotl(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
        }
        u32 a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
        for (int i = 0; i < 80; i++) {
            u32 f, k;
            if (i < 20) {
                f = (b & c) | (~b & d);
                k = 0x5A827999;
            } else if (i < 40) {
                f = b ^ c ^ d;
                k = 0x6ED9EBA1;
            } else if (i < 60) {
                f = (b & c) | (b & d) | (c & d);
                k = 0x8F1BBCDC;
            } else {
                f = b ^ c ^ d;
                k = 0xCA62C1D6;
            }
            const u32 t = std::rotl(a, 5) + f + e + k + w[i];
            e = d;
            d = c;
            c = std::rotl(b, 30);
            b = a;
            a = t;
        }
        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
    }

    std::array<u8, 20> digest;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 4; j++) {
            digest[i * 4 + j] = static_cast<u8>(h[i] >> (24 - j * 8));
        }
    }
    return digest;
}

} // namespace

std::string NameToNid(std::string_view name) {
    std::vector<u8> input(name.begin(), name.end());
    input.insert(input.end(), NidSuffix.begin(), NidSuffix.end());
    const auto digest = Sha1(input);

    // The first 8 digest bytes, read as a little-endian number, are written 6 bits at a time
    // from the most significant end; the 11th character carries the last 4 bits.
    u64 value;
    std::memcpy(&value, digest.data(), sizeof(value));
    std::string nid(11, '\0');
    for (int i = 0; i < 10; i++) {
        nid[i] = Alphabet[(value >> (58 - i * 6)) & 0x3F];
    }
    nid[10] = Alphabet[(value & 0xF) * 4];
    return nid;
}

bool DecodeId(std::string_view text, u64& out) {
    out = 0;
    for (char c : text) {
        const size_t digit = Alphabet.find(c);
        if (digit == std::string_view::npos) {
            return false;
        }
        out = out * 64 + digit;
    }
    return !text.empty();
}

} // namespace loader
