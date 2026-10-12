// SPDX-FileCopyrightText: Copyright 2026 sotcport
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace AmdGpu {

// sotcport: diagnostic hook called for every PM4 packet the command processor executes,
// with the packet's address in game memory and its size in dwords (header included). Used by
// the game's renderer recorder (src/diag/gnm_recorder.cpp) to check that every command the GPU
// runs was written by a known function. Set once before the GPU starts; null = off.
enum class PacketQueue : u32 { Graphics, Constant, Compute };
using PacketObserver = void (*)(const u32* packet, u32 dwords, PacketQueue queue);
inline PacketObserver g_packet_observer = nullptr;

} // namespace AmdGpu
