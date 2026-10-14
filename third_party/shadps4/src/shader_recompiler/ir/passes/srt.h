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

struct Info;

// sotcport: what each entry of the walker's read log is (Info::srt_read_map, generated with the
// walker, whose code is straight-line: every walk logs the same reads in the same order). A data
// read gives the flattened user data dword it fills (SrtReadDwordMask); SrtReadCpuNeeded marks
// data the walker itself computes later offsets from. SrtReadPointer is a pointer read (8 bytes).
constexpr u16 SrtReadPointer = 0xFFFF;
constexpr u16 SrtReadCpuNeeded = 0x8000;
constexpr u16 SrtReadDwordMask = 0x7FFF;

// Set by the GPU layer: makes CPU memory current for the logged reads [begin, end) the GPU wrote
// to, returning true if it had to (the walker's output is then stale). sotcport: data the CPU
// does not need (no pointer, no descriptor, no offset) is left to the GPU instead: listed in
// info.srt_patches, copied on the GPU into the flattened user data before the shader runs.
using PFN_SrtReadCheck = bool (*)(Info& info, const u64* begin, const u64* end);
extern PFN_SrtReadCheck SrtReadCheck;
// sotcport: cache-neutral begin
// sotcport (diagnostics, SOTC_TRACE_SRT_READBACKS): the program whose SRT is being walked.
extern u64 SrtWalkProgramHash;
// sotcport: cache-neutral end
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
