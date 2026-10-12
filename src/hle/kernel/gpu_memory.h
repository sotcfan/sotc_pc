#pragma once

#include <string>

#include <utility>
#include <vector>

#include "common/types.h"

// What the GPU layer needs from the memory manager (see memory.cpp).

namespace hle::kernel {

// Told about changes to the game's mappings, outside the memory manager's lock.
class GpuMemoryObserver {
public:
    virtual ~GpuMemoryObserver() = default;
    // New memory at [address, +size). Host protections there were reset.
    virtual void OnMapped(u64 address, u64 size) = 0;
    virtual void OnUnmapped(u64 address, u64 size) = 0;
    // The game changed the protection of [address, +size).
    virtual void OnProtectionChanged(u64 address, u64 size) = 0;
};

void SetGpuMemoryObserver(GpuMemoryObserver* observer);

// The same memory through a second mapping that is never protected, so the GPU layer can read
// and write pages it is tracking. Null if `address` has no physical memory behind it.
u8* BackingAddress(u64 address);

// `size`, shortened so the range stays within contiguous mapped memory starting at `address`.
u64 ClampToMapped(u64 address, u64 size);

// The game's memory below the GPU's 40-bit limit is also mapped at `address + offset`,
// a view the GPU layer's write tracking never protects (0 if the mirror could not be reserved).
u64 GuestMirrorOffset();

// Changes the host protection of game memory to track CPU accesses to GPU resources. The
// mapping itself (and what the game asked for) is not changed.
void ProtectForTracking(u64 address, u64 size, bool readable, bool writable);

// The kind of physical memory behind `address`: the type the game gave the direct memory
// (0 = WB_ONION, 3 = WC_GARLIC...), FlexibleMemoryType for flexible memory, -1 if nothing is
// mapped there. Takes the memory manager's lock: for diagnostics, not for hot paths.
constexpr s32 FlexibleMemoryType = 100;
s32 MemoryTypeAt(u64 address);

// [start, end) of every mapping the game has (reservations excluded).
std::vector<std::pair<u64, u64>> MappedRanges();

// Where the PS4 places its "system reserved" area; the GPU's tessellation ring lives at its end.
constexpr u64 SystemReservedBase = 0x7'FFFF'C000;
constexpr u64 SystemReservedSize = 0x7'C000'4000;

// Who holds the memory manager's lock, for freeze reports (diagnostics).
std::string DescribeMemoryLock();

} // namespace hle::kernel
