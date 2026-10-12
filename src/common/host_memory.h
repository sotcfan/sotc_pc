#pragma once

#include "common/types.h"

// The host operating system's virtual memory, behind one interface for Linux (mmap, memfd) and
// Windows (placeholders, section views: VirtualAlloc2/MapViewOfFile3).
//
// Two kinds of memory go through here:
// - Plain host memory for our own use (the executable image, import stubs, hook trampolines,
//   the crash handler's stack): Allocate/AllocateAt/Protect/Free.
// - The game's memory, managed by hle/kernel/memory.cpp: one "physical" object of the console's
//   size (a memfd on Linux, a pagefile-backed section on Windows), mapped at the game's addresses
//   and a second time as the "backing" view the GPU layer reads through, plus the "mirror", a
//   copy of the game's address space at a fixed offset that never gets protected. Game mappings
//   may only go inside the ranges given to ReserveGuest (Windows needs them reserved up front, as
//   placeholders that views later replace; on Linux the reservation only records the bounds).
namespace host_memory {

// Page protection, as a combination of these bits.
enum Prot : int {
    ProtNone = 0,
    ProtRead = 1,
    ProtWrite = 2,
    ProtExec = 4,
};

// Size of a host page (4 KiB on both systems).
u64 PageSize();

// ---- Plain host memory ----

// Anywhere; nullptr on failure.
void* Allocate(u64 size, int prot);
// Exactly at `address` and only if the range is free; nullptr on failure.
void* AllocateAt(u64 address, u64 size, int prot);
bool Protect(void* address, u64 size, int prot);
void Free(void* address, u64 size);

// ---- The game's memory ----

// Creates the physical memory object and maps it whole as the backing view (read/write).
// Returns the backing view, or nullptr.
u8* CreatePhysical(u64 size);

// Reserves [start, end) for game mappings. On Windows the range becomes one inaccessible
// placeholder; game mappings must stay inside reserved ranges. Returns false if the range is
// not free.
bool ReserveGuest(u64 start, u64 end);

// Maps [address, +size) to physical memory at `physical`, replacing whatever we mapped there.
bool MapPhysical(u64 address, u64 size, u64 physical, int prot);
// Makes [address, +size) inaccessible address space, replacing whatever we mapped there.
bool MapReserved(u64 address, u64 size);
// Maps fresh zero-filled private memory at [address, +size), replacing whatever we mapped there.
bool MapZeros(u64 address, u64 size, int prot);
// Gives [address, +size) back (inside a reservation it stays reserved, inaccessible).
bool Unmap(u64 address, u64 size);
// Changes the protection of game memory we mapped. On Windows a range may span several views.
bool ProtectGuest(u64 address, u64 size, int prot);
// True if the host has nothing at [address, +size) that we did not map (a candidate address for
// a mapping without a fixed address).
bool IsFree(u64 address, u64 size);
// The physical range's contents are no longer needed: the host may drop them (it reads as zeros
// afterwards).
void Discard(u64 physical, u64 size);

} // namespace host_memory
