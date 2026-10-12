// Threads, mutexes, condition variables and thread-specific keys: libkernel's scePthread*
// functions and libScePosix's pthread_* functions, on top of the host's pthreads.
//
// Every PS4 pthread type (thread, mutex, attributes, condition) is an 8-byte handle that points
// to an object the library allocates. A null handle is the static initializer
// (PTHREAD_MUTEX_INITIALIZER); its object is created on first use.
//
// The sce* functions return 0 or a negative Sony error code (0x8002xxxx); the POSIX ones return
// a FreeBSD errno value.

#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <atomic>
#include <csetjmp>
#include <cstring>
#include <string>

#include "common/log.h"
#include "common/types.h"
#include "hle/kernel/errno.h"
#include "hle/kernel/kernel.h"
#include "hle/registry.h"
#include "loader/crash.h"
#include "loader/tls.h"

// sotcport TEMP: game wait times (third_party/shadps4/shim/perf_temp.h).
namespace perf {
bool Enabled();
uint64_t Now();
void AddWait(const char* kind, uint64_t ns);
struct WaitScope {
    explicit WaitScope(const char* kind_) : kind{kind_}, start{Enabled() ? Now() : 0} {}
    ~WaitScope() {
        if (start) {
            AddWait(kind, Now() - start);
        }
    }
    const char* kind;
    uint64_t start;
};
} // namespace perf

namespace hle::kernel {

namespace {

using GuestEntry = PS4_SYSV_ABI void* (*)(void*);
using KeyDestructor = PS4_SYSV_ABI void (*)(void*);

// PS4 (FreeBSD) mutex types.
constexpr int MutexErrorCheck = 1;
constexpr int MutexRecursive = 2;
constexpr int MutexNormal = 3;
constexpr int MutexAdaptive = 4;

constexpr int DetachStateJoinable = 0;
constexpr int DetachStateDetached = 1;

// PS4 priorities: 256 is the highest, 767 the lowest.
constexpr int PriorityHighest = 256;
constexpr int PriorityLowest = 767;
constexpr int PriorityDefault = 700;

constexpr u64 DefaultStackSize = 2ull << 20;
// Our own functions run on the game's threads too (logging, file access...), so every thread
// gets this much on top of the stack size the game asks for.
constexpr u64 HostStackReserve = 512ull << 10;

struct MutexAttr {
    int type = MutexErrorCheck;
    int protocol = 0;
};

struct Mutex {
    pthread_mutex_t host;
    std::string name;
};

struct Cond {
    pthread_cond_t host;
};

struct ThreadAttr {
    u64 stack_size = DefaultStackSize;
    int detach_state = DetachStateJoinable;
    int policy = 0;
    int priority = PriorityDefault;
    int inherit_sched = 0;
    u64 affinity = 0x7F; // the 7 cores a game can use
};

struct Thread {
    pthread_t host{};
    GuestEntry entry = nullptr;
    void* argument = nullptr;
    std::string name;
    u32 id = 0;
    ThreadAttr attr;
    void* result = nullptr;
    std::jmp_buf exit_point; // where pthread_exit returns to, see ThreadStart
    bool can_exit = false;
};

std::atomic<u32> g_next_thread_id{1};
thread_local Thread* t_self = nullptr;

s32 PosixResult(int host_error) {
    return ToBsdErrno(host_error);
}

// Sony error from a FreeBSD errno value, for the sce* variants.
s32 SceFromPosix(s32 bsd_errno) {
    return bsd_errno == 0 ? 0 : ErrorCode(bsd_errno);
}

// ---- Mutexes ----

int HostMutexType(int type) {
    switch (type) {
    case MutexRecursive: return PTHREAD_MUTEX_RECURSIVE;
    case MutexErrorCheck: return PTHREAD_MUTEX_ERRORCHECK;
    default: return PTHREAD_MUTEX_NORMAL;
    }
}

Mutex* NewMutex(int type, const char* name) {
    auto* mutex = new Mutex;
    pthread_mutexattr_t attr;
    ::pthread_mutexattr_init(&attr);
    ::pthread_mutexattr_settype(&attr, HostMutexType(type));
    ::pthread_mutex_init(&mutex->host, &attr);
    ::pthread_mutexattr_destroy(&attr);
    mutex->name = name ? name : "";
    return mutex;
}

// Values of the 8-byte handle below this are static initializers, not objects:
// 0 = PTHREAD_MUTEX_INITIALIZER, 1 = PTHREAD_ADAPTIVE_MUTEX_INITIALIZER_NP.
constexpr u64 MaxStaticInitializer = 0x100;

Mutex* GetMutex(Mutex** handle) {
    if (!handle) {
        return nullptr;
    }
    std::atomic_ref<Mutex*> slot{*handle};
    Mutex* current = slot.load(std::memory_order_acquire);
    if (reinterpret_cast<u64>(current) >= MaxStaticInitializer) {
        return current;
    }
    const int type = reinterpret_cast<u64>(current) == 1 ? MutexAdaptive : MutexErrorCheck;
    Mutex* created = NewMutex(type, "");
    if (slot.compare_exchange_strong(current, created, std::memory_order_acq_rel)) {
        return created;
    }
    ::pthread_mutex_destroy(&created->host);
    delete created;
    return current;
}

s32 MutexInit(Mutex** handle, MutexAttr* const* attr, const char* name) {
    if (!handle) {
        return BSD_EINVAL;
    }
    const MutexAttr* a = attr ? *attr : nullptr;
    *handle = NewMutex(a ? a->type : MutexErrorCheck, name);
    return 0;
}

s32 MutexDestroy(Mutex** handle) {
    if (!handle) {
        return BSD_EINVAL;
    }
    if (reinterpret_cast<u64>(*handle) >= MaxStaticInitializer) {
        ::pthread_mutex_destroy(&(*handle)->host);
        delete *handle;
    }
    *handle = nullptr;
    return 0;
}

s32 MutexLock(Mutex** handle) {
    Mutex* mutex = GetMutex(handle);
    return mutex ? PosixResult(::pthread_mutex_lock(&mutex->host)) : BSD_EINVAL;
}

s32 MutexTrylock(Mutex** handle) {
    Mutex* mutex = GetMutex(handle);
    return mutex ? PosixResult(::pthread_mutex_trylock(&mutex->host)) : BSD_EINVAL;
}

s32 MutexUnlock(Mutex** handle) {
    Mutex* mutex = GetMutex(handle);
    return mutex ? PosixResult(::pthread_mutex_unlock(&mutex->host)) : BSD_EINVAL;
}

s32 MutexattrInit(MutexAttr** attr) {
    if (!attr) {
        return BSD_EINVAL;
    }
    *attr = new MutexAttr;
    return 0;
}

s32 MutexattrDestroy(MutexAttr** attr) {
    if (!attr) {
        return BSD_EINVAL;
    }
    delete *attr;
    *attr = nullptr;
    return 0;
}

s32 MutexattrSettype(MutexAttr** attr, int type) {
    if (!attr || !*attr || type < MutexErrorCheck || type > MutexAdaptive) {
        return BSD_EINVAL;
    }
    (*attr)->type = type;
    return 0;
}

s32 MutexattrSetprotocol(MutexAttr** attr, int protocol) {
    if (!attr || !*attr) {
        return BSD_EINVAL;
    }
    (*attr)->protocol = protocol;
    return 0;
}

// ---- Condition variables ----

Cond* GetCond(Cond** handle) {
    if (!handle) {
        return nullptr;
    }
    std::atomic_ref<Cond*> slot{*handle};
    Cond* current = slot.load(std::memory_order_acquire);
    if (current) {
        return current;
    }
    auto* created = new Cond;
    ::pthread_cond_init(&created->host, nullptr);
    if (slot.compare_exchange_strong(current, created, std::memory_order_acq_rel)) {
        return created;
    }
    ::pthread_cond_destroy(&created->host);
    delete created;
    return current;
}

PS4_SYSV_ABI s32 pthread_cond_init(Cond** handle, const void*) {
    if (!handle) {
        return BSD_EINVAL;
    }
    *handle = nullptr;
    return GetCond(handle) ? 0 : BSD_ENOMEM;
}

PS4_SYSV_ABI s32 pthread_cond_destroy(Cond** handle) {
    if (!handle) {
        return BSD_EINVAL;
    }
    if (*handle) {
        ::pthread_cond_destroy(&(*handle)->host);
        delete *handle;
    }
    *handle = nullptr;
    return 0;
}

PS4_SYSV_ABI s32 pthread_cond_wait(Cond** cond, Mutex** mutex) {
    Cond* c = GetCond(cond);
    Mutex* m = GetMutex(mutex);
    perf::WaitScope perf_wait{"cond"};
    return c && m ? PosixResult(::pthread_cond_wait(&c->host, &m->host)) : BSD_EINVAL;
}

PS4_SYSV_ABI s32 pthread_cond_timedwait(Cond** cond, Mutex** mutex,
                                        const Ps4Timespec* deadline) {
    Cond* c = GetCond(cond);
    Mutex* m = GetMutex(mutex);
    if (!c || !m || !deadline) {
        return BSD_EINVAL;
    }
    perf::WaitScope perf_wait{"condtimed"};
    const timespec host{static_cast<std::time_t>(deadline->tv_sec),
                        static_cast<long>(deadline->tv_nsec)};
    return PosixResult(::pthread_cond_timedwait(&c->host, &m->host, &host));
}

PS4_SYSV_ABI s32 pthread_cond_broadcast(Cond** cond) {
    Cond* c = GetCond(cond);
    return c ? PosixResult(::pthread_cond_broadcast(&c->host)) : BSD_EINVAL;
}

// ---- Thread-specific keys ----
// Implemented here rather than with host keys so that the game's destructors run in
// ThreadStart, while the thread's game TLS still exists.

constexpr int MaxKeys = 256;
constexpr int DestructorPasses = 4;

struct KeySlot {
    std::atomic<bool> used{false};
    std::atomic<KeyDestructor> destructor{nullptr};
};
KeySlot g_keys[MaxKeys];
thread_local void* t_key_values[MaxKeys];

PS4_SYSV_ABI s32 pthread_key_create(int* key, KeyDestructor destructor) {
    if (!key) {
        return BSD_EINVAL;
    }
    for (int i = 0; i < MaxKeys; i++) {
        bool expected = false;
        if (g_keys[i].used.compare_exchange_strong(expected, true)) {
            g_keys[i].destructor = destructor;
            *key = i;
            return 0;
        }
    }
    return BSD_EAGAIN;
}

PS4_SYSV_ABI s32 pthread_key_delete(int key) {
    if (key < 0 || key >= MaxKeys || !g_keys[key].used) {
        return BSD_EINVAL;
    }
    g_keys[key].destructor = nullptr;
    g_keys[key].used = false;
    return 0;
}

PS4_SYSV_ABI void* pthread_getspecific(int key) {
    return key >= 0 && key < MaxKeys ? t_key_values[key] : nullptr;
}

PS4_SYSV_ABI s32 pthread_setspecific(int key, const void* value) {
    if (key < 0 || key >= MaxKeys || !g_keys[key].used) {
        return BSD_EINVAL;
    }
    t_key_values[key] = const_cast<void*>(value);
    return 0;
}

void RunKeyDestructors() {
    for (int pass = 0; pass < DestructorPasses; pass++) {
        bool called = false;
        for (int i = 0; i < MaxKeys; i++) {
            void* value = t_key_values[i];
            const KeyDestructor destructor = g_keys[i].destructor;
            if (value && destructor) {
                t_key_values[i] = nullptr;
                destructor(value);
                called = true;
            }
        }
        if (!called) {
            break;
        }
    }
}

// ---- Threads ----

void* ThreadStart(void* raw) {
    auto* thread = static_cast<Thread*>(raw);
    t_self = thread;
    const std::string name = thread->name.empty() ? std::format("Game:{}", thread->id) : thread->name;
    logging::SetThreadName(name);
    ::pthread_setname_np(::pthread_self(), name.substr(0, 15).c_str());
    crash::InstallAltStack();
    loader::tls::InitializeThread();
    LOG_DEBUG("Threads", "thread {} \"{}\" started (entry {:#x})", thread->id, thread->name,
              reinterpret_cast<u64>(thread->entry));

    // pthread_exit() jumps back here. The frames it skips are the game's, which have nothing
    // to clean up (the game is built without exceptions).
    thread->can_exit = true;
    if (setjmp(thread->exit_point) == 0) {
        thread->result = thread->entry(thread->argument);
    }
    thread->can_exit = false;

    RunKeyDestructors();
    loader::tls::FreeThread();
    LOG_DEBUG("Threads", "thread {} \"{}\" finished", thread->id, thread->name);
    void* result = thread->result;
    if (thread->attr.detach_state == DetachStateDetached) {
        delete thread;
    }
    return result;
}

s32 CreateThread(Thread** out, ThreadAttr* const* attr, GuestEntry entry, void* argument,
                 const char* name) {
    if (!out || !entry) {
        return BSD_EINVAL;
    }
    auto* thread = new Thread;
    thread->entry = entry;
    thread->argument = argument;
    thread->name = name ? name : "";
    thread->id = g_next_thread_id++;
    if (attr && *attr) {
        thread->attr = **attr;
    }

    pthread_attr_t host_attr;
    ::pthread_attr_init(&host_attr);
    ::pthread_attr_setstacksize(&host_attr, std::max<u64>(thread->attr.stack_size, 64 * 1024) +
                                              HostStackReserve);
    ::pthread_attr_setdetachstate(&host_attr, thread->attr.detach_state == DetachStateDetached
                                                ? PTHREAD_CREATE_DETACHED
                                                : PTHREAD_CREATE_JOINABLE);
    // The handle must be valid before the thread runs: it may use it right away.
    *out = thread;
    const int error = ::pthread_create(&thread->host, &host_attr, ThreadStart, thread);
    ::pthread_attr_destroy(&host_attr);
    if (error != 0) {
        LOG_ERROR("Threads", "could not create thread \"{}\": {}", thread->name,
                  std::strerror(error));
        *out = nullptr;
        delete thread;
        return PosixResult(error);
    }
    LOG_INFO("Threads", "thread {} \"{}\" created (stack {:#x}, priority {})", thread->id,
             thread->name, thread->attr.stack_size, thread->attr.priority);
    return 0;
}

s32 JoinThread(Thread* thread, void** result) {
    if (!thread) {
        return BSD_ESRCH;
    }
    void* value = nullptr;
    const int error = ::pthread_join(thread->host, &value);
    if (error != 0) {
        return PosixResult(error);
    }
    if (result) {
        *result = value;
    }
    delete thread;
    return 0;
}

[[noreturn]] void ExitThread(void* value) {
    if (t_self && t_self->can_exit) {
        t_self->result = value;
        std::longjmp(t_self->exit_point, 1);
    }
    crash::Fatal("pthread_exit called on a thread the game did not create");
}

ThreadAttr* GetAttr(ThreadAttr* const* attr) {
    return attr ? *attr : nullptr;
}

// ---- libkernel: scePthread* ----

PS4_SYSV_ABI s32 scePthreadMutexInit(Mutex** m, MutexAttr* const* attr, const char* name) {
    return SceFromPosix(MutexInit(m, attr, name));
}
PS4_SYSV_ABI s32 scePthreadMutexDestroy(Mutex** m) { return SceFromPosix(MutexDestroy(m)); }
PS4_SYSV_ABI s32 scePthreadMutexLock(Mutex** m) { return SceFromPosix(MutexLock(m)); }
PS4_SYSV_ABI s32 scePthreadMutexTrylock(Mutex** m) { return SceFromPosix(MutexTrylock(m)); }
PS4_SYSV_ABI s32 scePthreadMutexUnlock(Mutex** m) { return SceFromPosix(MutexUnlock(m)); }
PS4_SYSV_ABI s32 scePthreadMutexattrInit(MutexAttr** a) { return SceFromPosix(MutexattrInit(a)); }
PS4_SYSV_ABI s32 scePthreadMutexattrDestroy(MutexAttr** a) { return SceFromPosix(MutexattrDestroy(a)); }
PS4_SYSV_ABI s32 scePthreadMutexattrSettype(MutexAttr** a, int type) {
    return SceFromPosix(MutexattrSettype(a, type));
}

PS4_SYSV_ABI s32 scePthreadAttrInit(ThreadAttr** attr) {
    if (!attr) {
        return ErrorCode(BSD_EINVAL);
    }
    *attr = new ThreadAttr;
    return 0;
}

PS4_SYSV_ABI s32 scePthreadAttrDestroy(ThreadAttr** attr) {
    if (!attr) {
        return ErrorCode(BSD_EINVAL);
    }
    delete *attr;
    *attr = nullptr;
    return 0;
}

PS4_SYSV_ABI s32 scePthreadAttrSetstacksize(ThreadAttr** attr, u64 size) {
    if (!GetAttr(attr) || size < 16 * 1024) {
        return ErrorCode(BSD_EINVAL);
    }
    (*attr)->stack_size = size;
    return 0;
}

PS4_SYSV_ABI s32 scePthreadAttrGetstacksize(ThreadAttr* const* attr, u64* size) {
    if (!GetAttr(attr) || !size) {
        return ErrorCode(BSD_EINVAL);
    }
    *size = (*attr)->stack_size;
    return 0;
}

PS4_SYSV_ABI s32 scePthreadAttrSetdetachstate(ThreadAttr** attr, int state) {
    if (!GetAttr(attr) || (state != DetachStateJoinable && state != DetachStateDetached)) {
        return ErrorCode(BSD_EINVAL);
    }
    (*attr)->detach_state = state;
    return 0;
}

PS4_SYSV_ABI s32 scePthreadAttrSetinheritsched(ThreadAttr** attr, int inherit) {
    if (!GetAttr(attr)) {
        return ErrorCode(BSD_EINVAL);
    }
    (*attr)->inherit_sched = inherit;
    return 0;
}

PS4_SYSV_ABI s32 scePthreadAttrSetschedpolicy(ThreadAttr** attr, int policy) {
    if (!GetAttr(attr)) {
        return ErrorCode(BSD_EINVAL);
    }
    (*attr)->policy = policy;
    return 0;
}

PS4_SYSV_ABI s32 scePthreadAttrSetschedparam(ThreadAttr** attr, const int* priority) {
    if (!GetAttr(attr) || !priority) {
        return ErrorCode(BSD_EINVAL);
    }
    (*attr)->priority = *priority;
    return 0;
}

PS4_SYSV_ABI s32 scePthreadAttrSetaffinity(ThreadAttr** attr, u64 mask) {
    if (!GetAttr(attr)) {
        return ErrorCode(BSD_EINVAL);
    }
    (*attr)->affinity = mask;
    return 0;
}

PS4_SYSV_ABI s32 scePthreadAttrGetaffinity(ThreadAttr* const* attr, u64* mask) {
    if (!GetAttr(attr) || !mask) {
        return ErrorCode(BSD_EINVAL);
    }
    *mask = (*attr)->affinity;
    return 0;
}

// Fills `attr` (already initialized) with the attributes of a running thread.
PS4_SYSV_ABI s32 scePthreadAttrGet(Thread* thread, ThreadAttr** attr) {
    if (!thread || !GetAttr(attr)) {
        return ErrorCode(BSD_EINVAL);
    }
    **attr = thread->attr;
    return 0;
}

PS4_SYSV_ABI s32 scePthreadCreate(Thread** thread, ThreadAttr* const* attr, GuestEntry entry,
                                  void* argument, const char* name) {
    return SceFromPosix(CreateThread(thread, attr, entry, argument, name));
}

PS4_SYSV_ABI s32 scePthreadJoin(Thread* thread, void** result) {
    return SceFromPosix(JoinThread(thread, result));
}

[[noreturn]] PS4_SYSV_ABI void scePthreadExit(void* value) {
    ExitThread(value);
}

PS4_SYSV_ABI Thread* scePthreadSelf() {
    return t_self;
}

PS4_SYSV_ABI s32 scePthreadGetthreadid() {
    return t_self ? static_cast<s32>(t_self->id) : 0;
}

PS4_SYSV_ABI s32 scePthreadYield() {
    ::sched_yield();
    return 0;
}

PS4_SYSV_ABI s32 scePthreadGetprio(Thread* thread, int* priority) {
    if (!thread || !priority) {
        return ErrorCode(BSD_EINVAL);
    }
    *priority = thread->attr.priority;
    return 0;
}

// Priorities and core affinity are only recorded: the host scheduler does a good job on its
// own, and the PS4's 7 cores do not map to the PC's.
PS4_SYSV_ABI s32 scePthreadSetprio(Thread* thread, int priority) {
    if (!thread) {
        return ErrorCode(BSD_ESRCH);
    }
    thread->attr.priority = priority;
    return 0;
}

PS4_SYSV_ABI s32 scePthreadGetaffinity(Thread* thread, u64* mask) {
    if (!thread || !mask) {
        return ErrorCode(BSD_EINVAL);
    }
    *mask = thread->attr.affinity;
    return 0;
}

PS4_SYSV_ABI s32 scePthreadSetaffinity(Thread* thread, u64 mask) {
    if (!thread) {
        return ErrorCode(BSD_ESRCH);
    }
    thread->attr.affinity = mask;
    return 0;
}

// ---- libScePosix: pthread_* ----

PS4_SYSV_ABI s32 pthread_mutex_init(Mutex** m, MutexAttr* const* attr) { return MutexInit(m, attr, nullptr); }
PS4_SYSV_ABI s32 pthread_mutex_destroy(Mutex** m) { return MutexDestroy(m); }
PS4_SYSV_ABI s32 pthread_mutex_lock(Mutex** m) { return MutexLock(m); }
PS4_SYSV_ABI s32 pthread_mutex_trylock(Mutex** m) { return MutexTrylock(m); }
PS4_SYSV_ABI s32 pthread_mutex_unlock(Mutex** m) { return MutexUnlock(m); }
PS4_SYSV_ABI s32 pthread_mutexattr_init(MutexAttr** a) { return MutexattrInit(a); }
PS4_SYSV_ABI s32 pthread_mutexattr_destroy(MutexAttr** a) { return MutexattrDestroy(a); }
PS4_SYSV_ABI s32 pthread_mutexattr_settype(MutexAttr** a, int type) { return MutexattrSettype(a, type); }
PS4_SYSV_ABI s32 pthread_mutexattr_setprotocol(MutexAttr** a, int p) { return MutexattrSetprotocol(a, p); }

PS4_SYSV_ABI s32 pthread_attr_init(ThreadAttr** attr) {
    return scePthreadAttrInit(attr) == 0 ? 0 : BSD_EINVAL;
}

PS4_SYSV_ABI s32 pthread_attr_destroy(ThreadAttr** attr) {
    return scePthreadAttrDestroy(attr) == 0 ? 0 : BSD_EINVAL;
}

PS4_SYSV_ABI s32 pthread_attr_setstacksize(ThreadAttr** attr, u64 size) {
    return scePthreadAttrSetstacksize(attr, size) == 0 ? 0 : BSD_EINVAL;
}

PS4_SYSV_ABI s32 pthread_create(Thread** thread, ThreadAttr* const* attr, GuestEntry entry,
                                void* argument) {
    return CreateThread(thread, attr, entry, argument, nullptr);
}

PS4_SYSV_ABI s32 pthread_join(Thread* thread, void** result) {
    return JoinThread(thread, result);
}

[[noreturn]] PS4_SYSV_ABI void pthread_exit(void* value) {
    ExitThread(value);
}

PS4_SYSV_ABI Thread* pthread_self() {
    return t_self;
}

PS4_SYSV_ABI s32 pthread_cancel(Thread* thread) {
    LOG_ERROR("Threads", "pthread_cancel of thread \"{}\" is not supported",
              thread ? thread->name : "?");
    return BSD_ENOSYS;
}

struct SchedParam {
    int priority;
};

PS4_SYSV_ABI s32 pthread_getschedparam(Thread* thread, int* policy, SchedParam* param) {
    if (!thread || !policy || !param) {
        return BSD_EINVAL;
    }
    *policy = thread->attr.policy;
    param->priority = thread->attr.priority;
    return 0;
}

PS4_SYSV_ABI s32 pthread_setschedparam(Thread* thread, int policy, const SchedParam* param) {
    if (!thread || !param) {
        return BSD_EINVAL;
    }
    thread->attr.policy = policy;
    thread->attr.priority = param->priority;
    return 0;
}

PS4_SYSV_ABI int sched_get_priority_max(int) {
    return PriorityHighest;
}

PS4_SYSV_ABI int sched_get_priority_min(int) {
    return PriorityLowest;
}

PS4_SYSV_ABI int sched_yield() {
    return ::sched_yield();
}

} // namespace

void InitializeMainThread() {
    auto* thread = new Thread;
    thread->host = ::pthread_self();
    thread->name = "Game:Main";
    thread->id = g_next_thread_id++;
    t_self = thread;
}

HLE_FUNCTION("libkernel", "scePthreadMutexInit", scePthreadMutexInit);
HLE_FUNCTION("libkernel", "scePthreadMutexDestroy", scePthreadMutexDestroy);
HLE_FUNCTION("libkernel", "scePthreadMutexLock", scePthreadMutexLock);
HLE_FUNCTION("libkernel", "scePthreadMutexTrylock", scePthreadMutexTrylock);
HLE_FUNCTION("libkernel", "scePthreadMutexUnlock", scePthreadMutexUnlock);
HLE_FUNCTION("libkernel", "scePthreadMutexattrInit", scePthreadMutexattrInit);
HLE_FUNCTION("libkernel", "scePthreadMutexattrDestroy", scePthreadMutexattrDestroy);
HLE_FUNCTION("libkernel", "scePthreadMutexattrSettype", scePthreadMutexattrSettype);
HLE_FUNCTION("libkernel", "scePthreadAttrInit", scePthreadAttrInit);
HLE_FUNCTION("libkernel", "scePthreadAttrDestroy", scePthreadAttrDestroy);
HLE_FUNCTION("libkernel", "scePthreadAttrSetstacksize", scePthreadAttrSetstacksize);
HLE_FUNCTION("libkernel", "scePthreadAttrGetstacksize", scePthreadAttrGetstacksize);
HLE_FUNCTION("libkernel", "scePthreadAttrSetdetachstate", scePthreadAttrSetdetachstate);
HLE_FUNCTION("libkernel", "scePthreadAttrSetinheritsched", scePthreadAttrSetinheritsched);
HLE_FUNCTION("libkernel", "scePthreadAttrSetschedpolicy", scePthreadAttrSetschedpolicy);
HLE_FUNCTION("libkernel", "scePthreadAttrSetschedparam", scePthreadAttrSetschedparam);
HLE_FUNCTION("libkernel", "scePthreadAttrSetaffinity", scePthreadAttrSetaffinity);
HLE_FUNCTION("libkernel", "scePthreadAttrGetaffinity", scePthreadAttrGetaffinity);
HLE_FUNCTION("libkernel", "scePthreadAttrGet", scePthreadAttrGet);
HLE_FUNCTION("libkernel", "scePthreadCreate", scePthreadCreate);
HLE_FUNCTION("libkernel", "scePthreadJoin", scePthreadJoin);
HLE_FUNCTION("libkernel", "scePthreadExit", scePthreadExit);
HLE_FUNCTION("libkernel", "scePthreadSelf", scePthreadSelf);
HLE_FUNCTION("libkernel", "scePthreadGetthreadid", scePthreadGetthreadid);
HLE_FUNCTION("libkernel", "scePthreadYield", scePthreadYield);
HLE_FUNCTION("libkernel", "scePthreadGetprio", scePthreadGetprio);
HLE_FUNCTION("libkernel", "scePthreadSetprio", scePthreadSetprio);
HLE_FUNCTION("libkernel", "scePthreadGetaffinity", scePthreadGetaffinity);
HLE_FUNCTION("libkernel", "scePthreadSetaffinity", scePthreadSetaffinity);

HLE_FUNCTION("libScePosix", "pthread_mutex_init", pthread_mutex_init);
HLE_FUNCTION("libScePosix", "pthread_mutex_destroy", pthread_mutex_destroy);
HLE_FUNCTION("libScePosix", "pthread_mutex_lock", pthread_mutex_lock);
HLE_FUNCTION("libScePosix", "pthread_mutex_trylock", pthread_mutex_trylock);
HLE_FUNCTION("libScePosix", "pthread_mutex_unlock", pthread_mutex_unlock);
HLE_FUNCTION("libScePosix", "pthread_mutexattr_init", pthread_mutexattr_init);
HLE_FUNCTION("libScePosix", "pthread_mutexattr_destroy", pthread_mutexattr_destroy);
HLE_FUNCTION("libScePosix", "pthread_mutexattr_settype", pthread_mutexattr_settype);
HLE_FUNCTION("libScePosix", "pthread_mutexattr_setprotocol", pthread_mutexattr_setprotocol);
HLE_FUNCTION("libScePosix", "pthread_cond_init", pthread_cond_init);
HLE_FUNCTION("libScePosix", "pthread_cond_destroy", pthread_cond_destroy);
HLE_FUNCTION("libScePosix", "pthread_cond_wait", pthread_cond_wait);
HLE_FUNCTION("libScePosix", "pthread_cond_timedwait", pthread_cond_timedwait);
HLE_FUNCTION("libScePosix", "pthread_cond_broadcast", pthread_cond_broadcast);
HLE_FUNCTION("libScePosix", "pthread_key_create", pthread_key_create);
HLE_FUNCTION("libScePosix", "pthread_key_delete", pthread_key_delete);
HLE_FUNCTION("libScePosix", "pthread_getspecific", pthread_getspecific);
HLE_FUNCTION("libScePosix", "pthread_setspecific", pthread_setspecific);
HLE_FUNCTION("libScePosix", "pthread_attr_init", pthread_attr_init);
HLE_FUNCTION("libScePosix", "pthread_attr_destroy", pthread_attr_destroy);
HLE_FUNCTION("libScePosix", "pthread_attr_setstacksize", pthread_attr_setstacksize);
HLE_FUNCTION("libScePosix", "pthread_create", pthread_create);
HLE_FUNCTION("libScePosix", "pthread_join", pthread_join);
HLE_FUNCTION("libScePosix", "pthread_exit", pthread_exit);
HLE_FUNCTION("libScePosix", "pthread_self", pthread_self);
HLE_FUNCTION("libScePosix", "pthread_cancel", pthread_cancel);
HLE_FUNCTION("libScePosix", "pthread_getschedparam", pthread_getschedparam);
HLE_FUNCTION("libScePosix", "pthread_setschedparam", pthread_setschedparam);
HLE_FUNCTION("libScePosix", "sched_get_priority_max", sched_get_priority_max);
HLE_FUNCTION("libScePosix", "sched_get_priority_min", sched_get_priority_min);
HLE_FUNCTION("libScePosix", "sched_yield", sched_yield);

} // namespace hle::kernel
