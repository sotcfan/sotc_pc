// <stdlib.h> conversions, sorting and rand, plus Dinkumware's lock and atomic helpers.

#include <atomic>
#include <cstdlib>
#include <mutex>

#include "common/log.h"
#include "common/types.h"
#include "hle/registry.h"

namespace hle::libc {

namespace {

PS4_SYSV_ABI double atof(const char* s) { return std::atof(s); }
PS4_SYSV_ABI double strtod(const char* s, char** end) { return std::strtod(s, end); }
// `long` is 64 bits on the PS4 (32 on Windows).
PS4_SYSV_ABI s64 strtol(const char* s, char** end, int base) { return std::strtoll(s, end, base); }

// Dinkumware's strtoul/strtoll/strtoull.
PS4_SYSV_ABI u64 Stoul(const char* s, char** end, int base) {
    return std::strtoull(s, end, base);
}
PS4_SYSV_ABI long long Stoll(const char* s, char** end, int base) {
    return std::strtoll(s, end, base);
}
PS4_SYSV_ABI unsigned long long Stoull(const char* s, char** end, int base) {
    return std::strtoull(s, end, base);
}

using Compare = PS4_SYSV_ABI int (*)(const void*, const void*);

// The game's comparison function uses the System V convention; the host's qsort/bsearch call
// theirs with the host's (Microsoft's on Windows). A host-convention function forwards to the
// game's, which this thread keeps here for the duration of the call (qsort may nest: the
// previous one is restored).
thread_local Compare t_compare = nullptr;

int HostCompare(const void* a, const void* b) {
    return t_compare(a, b);
}

PS4_SYSV_ABI void qsort(void* base, size_t count, size_t size, Compare compare) {
    const Compare previous = t_compare;
    t_compare = compare;
    std::qsort(base, count, size, HostCompare);
    t_compare = previous;
}

PS4_SYSV_ABI void* bsearch(const void* key, const void* base, size_t count, size_t size,
                           Compare compare) {
    const Compare previous = t_compare;
    t_compare = compare;
    void* found = std::bsearch(key, base, count, size, HostCompare);
    t_compare = previous;
    return found;
}

// The PS4's rand(): a 64-bit linear congruential generator returning bits 32..61, so the game
// sees the same sequence as on the console. RAND_MAX is 0x3FFFFFFF.
std::atomic<u64> g_rand_state{1};

PS4_SYSV_ABI int rand() {
    u64 state = g_rand_state.load(std::memory_order_relaxed);
    u64 next;
    do {
        next = state * 0x5851F42D4C957F2DULL + 1;
    } while (!g_rand_state.compare_exchange_weak(state, next, std::memory_order_relaxed));
    return static_cast<int>((next >> 32) & 0x3FFFFFFF);
}

PS4_SYSV_ABI void srand(unsigned int seed) {
    g_rand_state.store(seed, std::memory_order_relaxed);
}

// Dinkumware's library-wide locks, indexed by a small number (stdio, locale, ...).
constexpr int SystemLockCount = 16;
std::recursive_mutex g_system_locks[SystemLockCount];

PS4_SYSV_ABI void Locksyslock(int index) {
    if (index >= 0 && index < SystemLockCount) {
        g_system_locks[index].lock();
    } else {
        LOG_ERROR("libc", "_Locksyslock({}) out of range", index);
    }
}

PS4_SYSV_ABI void Unlocksyslock(int index) {
    if (index >= 0 && index < SystemLockCount) {
        g_system_locks[index].unlock();
    }
}

PS4_SYSV_ABI u32 Atomic_fetch_add_4(volatile u32* target, u32 value, int) {
    return __atomic_fetch_add(target, value, __ATOMIC_SEQ_CST);
}

PS4_SYSV_ABI u32 Atomic_fetch_sub_4(volatile u32* target, u32 value, int) {
    return __atomic_fetch_sub(target, value, __ATOMIC_SEQ_CST);
}

} // namespace

HLE_FUNCTION("libc", "atof", atof);
HLE_FUNCTION("libc", "strtod", strtod);
HLE_FUNCTION("libc", "strtol", strtol);
HLE_FUNCTION("libc", "_Stoul", Stoul);
HLE_FUNCTION("libc", "_Stoll", Stoll);
HLE_FUNCTION("libc", "_Stoull", Stoull);
HLE_FUNCTION("libc", "qsort", qsort);
HLE_FUNCTION("libc", "bsearch", bsearch);
HLE_FUNCTION("libc", "rand", rand);
HLE_FUNCTION("libc", "srand", srand);
HLE_FUNCTION("libc", "_Locksyslock", Locksyslock);
HLE_FUNCTION("libc", "_Unlocksyslock", Unlocksyslock);
HLE_FUNCTION("libc", "_Atomic_fetch_add_4", Atomic_fetch_add_4);
HLE_FUNCTION("libc", "_Atomic_fetch_sub_4", Atomic_fetch_sub_4);

} // namespace hle::libc
