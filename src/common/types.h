#pragma once

#include <cstddef>
#include <cstdint>

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using s8 = std::int8_t;
using s16 = std::int16_t;
using s32 = std::int32_t;
using s64 = std::int64_t;

// The game was built for the PS4, whose calling convention is System V AMD64. Every function
// the game calls directly (and every game function we call) uses it. On Linux this is already
// the default; the attribute keeps that true when the project is built for Windows.
#define PS4_SYSV_ABI __attribute__((sysv_abi))
