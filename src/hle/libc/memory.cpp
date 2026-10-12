// malloc and operator new. The game supplies its own allocator through the libc parameters in
// its ProcParam (Shadow of the Colossus does), so these functions
// only forward to it, exactly as the PS4 libc does. The host allocator is the fallback for
// games that do not replace it.

#include <algorithm>
#include <cstdlib>
#include <new>

#include "common/log.h"
#include "common/types.h"
#include "hle/libc/libc.h"
#include "hle/registry.h"

namespace hle::libc {

namespace {

// The host's heap, used until the game installs its own allocator (and for what it allocated
// before). On Windows, memory with a chosen alignment must be freed with _aligned_free, so every
// fallback allocation goes through the _aligned_* family there (16-byte alignment, like malloc).
void* HostMalloc(size_t size) {
#ifdef _WIN32
    return _aligned_malloc(size ? size : 1, 16);
#else
    return std::malloc(size);
#endif
}
void HostFree(void* pointer) {
#ifdef _WIN32
    _aligned_free(pointer);
#else
    std::free(pointer);
#endif
}
void* HostRealloc(void* pointer, size_t size) {
#ifdef _WIN32
    return _aligned_realloc(pointer, size ? size : 1, 16);
#else
    return std::realloc(pointer, size);
#endif
}
void* HostMemalign(size_t alignment, size_t size) {
#ifdef _WIN32
    return _aligned_malloc(size ? size : 1, std::max<size_t>(alignment, 16));
#else
    void* pointer = nullptr;
    return ::posix_memalign(&pointer, alignment, size) == 0 ? pointer : nullptr;
#endif
}

} // namespace

namespace {

struct MallocReplace {
    u64 size;
    u64 unknown;
    PS4_SYSV_ABI int (*malloc_init)();
    PS4_SYSV_ABI int (*malloc_finalize)();
    PS4_SYSV_ABI void* (*malloc)(size_t size);
    PS4_SYSV_ABI void (*free)(void* pointer);
    PS4_SYSV_ABI void* (*calloc)(size_t count, size_t size);
    PS4_SYSV_ABI void* (*realloc)(void* pointer, size_t size);
    PS4_SYSV_ABI void* (*memalign)(size_t alignment, size_t size);
    PS4_SYSV_ABI void* (*reallocalign)(void* pointer, size_t size, size_t alignment);
    PS4_SYSV_ABI int (*posix_memalign)(void** out, size_t alignment, size_t size);
    PS4_SYSV_ABI int (*malloc_stats)(void* stats);
    PS4_SYSV_ABI int (*malloc_stats_fast)(void* stats);
    PS4_SYSV_ABI size_t (*malloc_usable_size)(void* pointer);
};

struct NewReplace {
    u64 size;
    u64 unknown;
    PS4_SYSV_ABI void* (*new_)(size_t size);
    PS4_SYSV_ABI void* (*new_nothrow)(size_t size, const void* nothrow);
    PS4_SYSV_ABI void* (*new_array)(size_t size);
    PS4_SYSV_ABI void* (*new_array_nothrow)(size_t size, const void* nothrow);
    PS4_SYSV_ABI void (*delete_)(void* pointer);
    PS4_SYSV_ABI void (*delete_nothrow)(void* pointer, const void* nothrow);
    PS4_SYSV_ABI void (*delete_array)(void* pointer);
    PS4_SYSV_ABI void (*delete_array_nothrow)(void* pointer, const void* nothrow);
};

// Only the fields we use; the layout was read from the game's own parameters.
struct LibcParam {
    u64 size;
    u64 entry_count;
    const u64* heap_size;
    u64 unknown_18;
    const u32* unknown_20;
    u64 unknown_28;
    const MallocReplace* malloc_replace;
    const NewReplace* new_replace;
};

const MallocReplace* g_malloc = nullptr;
const NewReplace* g_new = nullptr;

PS4_SYSV_ABI void* malloc(size_t size) {
    return g_malloc ? g_malloc->malloc(size) : HostMalloc(size);
}

PS4_SYSV_ABI void free(void* pointer) {
    if (g_malloc) {
        g_malloc->free(pointer);
    } else {
        HostFree(pointer);
    }
}

PS4_SYSV_ABI void* realloc(void* pointer, size_t size) {
    return g_malloc ? g_malloc->realloc(pointer, size) : HostRealloc(pointer, size);
}

PS4_SYSV_ABI void* memalign(size_t alignment, size_t size) {
    if (g_malloc) {
        return g_malloc->memalign(alignment, size);
    }
    return HostMemalign(alignment, size);
}

// operator new(size_t, const std::nothrow_t&)
PS4_SYSV_ABI void* new_nothrow(size_t size, const void* nothrow) {
    if (g_new && g_new->new_nothrow) {
        return g_new->new_nothrow(size, nothrow);
    }
    return ::operator new(size, std::nothrow);
}

// std::nothrow: an empty object whose address is all that matters.
struct {
    u8 unused;
} g_nothrow;

using NewHandler = PS4_SYSV_ABI void (*)();
NewHandler g_new_handler = nullptr;

PS4_SYSV_ABI NewHandler set_new_handler(NewHandler handler) {
    const NewHandler previous = g_new_handler;
    g_new_handler = handler;
    return previous;
}

} // namespace

void Initialize(const void* raw_param) {
    const auto* param = static_cast<const LibcParam*>(raw_param);
    if (!param || param->size < sizeof(LibcParam)) {
        LOG_INFO("libc", "the game does not replace malloc; using the host one");
        return;
    }
    if (param->malloc_replace && param->malloc_replace->size >= sizeof(MallocReplace)) {
        g_malloc = param->malloc_replace;
    }
    if (param->new_replace && param->new_replace->size >= sizeof(NewReplace)) {
        g_new = param->new_replace;
    }
    LOG_INFO("libc", "game malloc: {}, game operator new: {}", g_malloc != nullptr,
             g_new != nullptr);
    if (g_malloc && g_malloc->malloc_init) {
        const int result = g_malloc->malloc_init();
        LOG_INFO("libc", "game malloc_init returned {}", result);
    }
}

HLE_FUNCTION("libc", "malloc", malloc);
HLE_FUNCTION("libc", "free", free);
HLE_FUNCTION("libc", "realloc", realloc);
HLE_FUNCTION("libc", "memalign", memalign);
HLE_FUNCTION("libc", "_ZnwmRKSt9nothrow_t", new_nothrow);
HLE_OBJECT("libc", "_ZSt7nothrow", g_nothrow);
HLE_FUNCTION("libc", "_ZSt15set_new_handlerPFvvE", set_new_handler);

} // namespace hle::libc
