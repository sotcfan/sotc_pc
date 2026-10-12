#pragma once

// Replaces shadPS4's memory manager with the part its GPU code uses, on top of sotcport's
// (src/hle/kernel/gpu_memory.h).

#include "common/singleton.h"
#include "common/types.h"
#include "core/address_space.h"

namespace Vulkan {
class Rasterizer;
}

namespace Core {

class MemoryManager {
public:
    // The rasterizer is told about the game's mappings from then on.
    void SetRasterizer(Vulkan::Rasterizer* rasterizer);

    AddressSpace& GetAddressSpace() {
        return address_space;
    }

    bool IsValidGpuMapping(VAddr virtual_addr, u64 size) {
        // The PS4's GPU can only handle 40 bit addresses.
        return virtual_addr + size < 0x10000000000;
    }

    u64 ClampRangeSize(VAddr virtual_addr, u64 size);
    void CopySparseMemory(VAddr source, u8* dest, u64 size);
    bool TryWriteBacking(void* address, const void* data, u64 size);
    // XXH3 of [address, +size) read through the unprotected backing mapping (unmapped pages
    // count as zeros), so it can run inside the write-tracking fault handler.
    u64 HashBacking(VAddr address, u64 size);

private:
    AddressSpace address_space;
};

using Memory = Common::Singleton<MemoryManager>;

} // namespace Core
