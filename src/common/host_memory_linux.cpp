// Linux implementation of common/host_memory.h: mmap and a memfd. This is what
// hle/kernel/memory.cpp and the loader did directly before the Windows port.
#ifndef _WIN32

#include "common/host_memory.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace host_memory {

namespace {

int g_physical_fd = -1;

int HostProt(int prot) {
    int host = PROT_NONE;
    if (prot & ProtRead) {
        host |= PROT_READ;
    }
    if (prot & ProtWrite) {
        host |= PROT_WRITE;
    }
    if (prot & ProtExec) {
        host |= PROT_EXEC;
    }
    return host;
}

void* Result(void* mapped) {
    return mapped == MAP_FAILED ? nullptr : mapped;
}

} // namespace

u64 PageSize() {
    return 4096;
}

void* Allocate(u64 size, int prot) {
    return Result(mmap(nullptr, size, HostProt(prot), MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
                       -1, 0));
}

void* AllocateAt(u64 address, u64 size, int prot) {
    void* mapped = mmap(reinterpret_cast<void*>(address), size, HostProt(prot),
                        MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
    if (mapped == MAP_FAILED) {
        return nullptr;
    }
    if (mapped != reinterpret_cast<void*>(address)) { // old kernels ignore NOREPLACE
        munmap(mapped, size);
        return nullptr;
    }
    return mapped;
}

bool Protect(void* address, u64 size, int prot) {
    return mprotect(address, size, HostProt(prot)) == 0;
}

void Free(void* address, u64 size) {
    munmap(address, size);
}

u8* CreatePhysical(u64 size) {
    g_physical_fd = memfd_create("ps4-physical-memory", MFD_CLOEXEC);
    if (g_physical_fd < 0 || ftruncate(g_physical_fd, static_cast<off_t>(size)) != 0) {
        return nullptr;
    }
    return static_cast<u8*>(
        Result(mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, g_physical_fd, 0)));
}

bool ReserveGuest(u64 start, u64 end) {
    // Linux maps game memory with MAP_FIXED over our own mappings and MAP_FIXED_NOREPLACE
    // elsewhere (hle/kernel/memory.cpp decides), so nothing needs reserving up front. The mirror
    // is the exception: it is reserved whole so nothing else lands inside it.
    void* mapped = mmap(reinterpret_cast<void*>(start), end - start, PROT_NONE,
                        MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED_NOREPLACE, -1, 0);
    if (mapped == reinterpret_cast<void*>(start)) {
        return true;
    }
    if (mapped != MAP_FAILED) {
        munmap(mapped, end - start);
    }
    return false;
}

bool MapPhysical(u64 address, u64 size, u64 physical, int prot) {
    return mmap(reinterpret_cast<void*>(address), size, HostProt(prot), MAP_SHARED | MAP_FIXED,
                g_physical_fd, static_cast<off_t>(physical)) != MAP_FAILED;
}

bool MapReserved(u64 address, u64 size) {
    return mmap(reinterpret_cast<void*>(address), size, PROT_NONE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED, -1, 0) != MAP_FAILED;
}

bool MapZeros(u64 address, u64 size, int prot) {
    return mmap(reinterpret_cast<void*>(address), size, HostProt(prot),
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED, -1, 0) != MAP_FAILED;
}

bool Unmap(u64 address, u64 size) {
    return munmap(reinterpret_cast<void*>(address), size) == 0;
}

bool ProtectGuest(u64 address, u64 size, int prot) {
    return mprotect(reinterpret_cast<void*>(address), size, HostProt(prot)) == 0;
}

bool IsFree(u64 address, u64 size) {
    void* probe = mmap(reinterpret_cast<void*>(address), size, PROT_NONE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED_NOREPLACE, -1, 0);
    if (probe == MAP_FAILED) {
        return false;
    }
    munmap(probe, size);
    return probe == reinterpret_cast<void*>(address);
}

void Discard(u64 physical, u64 size) {
    fallocate(g_physical_fd, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE,
              static_cast<off_t>(physical), static_cast<off_t>(size));
}

} // namespace host_memory

#endif // !_WIN32
