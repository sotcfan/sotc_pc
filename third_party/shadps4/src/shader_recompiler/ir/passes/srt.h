// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <boost/container/set.hpp>
#include <boost/container/small_vector.hpp>
#include "common/types.h"

namespace Serialization {
struct Archive;
}

namespace Shader {

// sotcport: the walker reads guest memory through the read-only mirror
// (hle::kernel::GuestMirrorOffset), so pages the GPU wrote never fault and stall it. It records
// the guest address of every such read in `read_log` and returns the end of the log, for the
// caller to fetch whatever the GPU really wrote there (SrtReadCheck) and run it again.
using PFN_SrtWalker = u64* PS4_SYSV_ABI (*)(const u32* /*user_data*/, u32* /*flat_dst*/,
                                            u64* /*read_log*/);
PFN_SrtWalker RegisterWalkerCode(const u8* ptr, size_t size);

// Set by the GPU layer: makes CPU memory current for the logged reads [begin, end) the GPU wrote
// to, returning true if it had to (the walker's output is then stale).
using PFN_SrtReadCheck = bool (*)(const u64* begin, const u64* end);
extern PFN_SrtReadCheck SrtReadCheck;
constexpr size_t SrtReadLogSize = 1 << 16;

struct PersistentSrtInfo {
    // Special case when fetch shader uses step rates.
    struct SrtSharpReservation {
        u32 sgpr_base;
        u32 dword_offset;
        u32 num_dwords;
    };

    PFN_SrtWalker walker_func{};
    size_t walker_func_size{};
    u32 flattened_bufsize_dw = 16; // NumUserDataRegs

    void Serialize(Serialization::Archive& ar) const;
    bool Deserialize(Serialization::Archive& ar);
};

} // namespace Shader
