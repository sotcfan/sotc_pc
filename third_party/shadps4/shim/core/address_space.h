#pragma once

// Replaces shadPS4's address space with what its GPU code needs from sotcport's memory
// manager (src/hle/kernel/gpu_memory.h).

#include <boost/icl/interval_set.hpp>

#include "common/enum.h"
#include "common/types.h"

namespace Core {

enum class MemoryPermission : u32 {
    None = 0,
    Read = 1 << 0,
    Write = 1 << 1,
    ReadWrite = Read | Write,
    Execute = 1 << 2,
    ReadWriteExecute = Read | Write | Execute,
};
DECLARE_ENUM_FLAG_OPERATORS(MemoryPermission)

class AddressSpace {
public:
    VAddr SystemReservedVirtualBase() noexcept;
    u64 SystemReservedVirtualSize() const noexcept;

    // Host protection used to track CPU writes to GPU resources.
    void Protect(VAddr virtual_addr, u64 size, MemoryPermission perms);

    boost::icl::interval_set<VAddr> GetUsableRegions();
};

} // namespace Core
