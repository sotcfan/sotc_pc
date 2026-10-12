#pragma once

// Replaces shadPS4's event queues with a view over sotcport's (src/hle/kernel/equeue.h).
// The queue handle the game holds is sotcport's EventQueue*; EqueueInternal has no data
// of its own and only forwards to it.

#include "common/types.h"
#include "hle/kernel/equeue.h"

namespace Libraries::Kernel {

struct OrbisKernelEvent {
    enum Filter : s16 {
        None = 0,
        Read = -1,
        Write = -2,
        Aio = -3,
        Vnode = -4,
        Proc = -5,
        Signal = -6,
        Timer = -7,
        Fs = -9,
        Lio = -10,
        User = -11,
        Polling = -12,
        VideoOut = -13,
        GraphicsCore = -14,
        HrTimer = -15,
    };

    enum Flags : u16 {
        Add = 1u,
        Delete = 2u,
        Enable = 4u,
        Disable = 8u,
        OneShot = 0x10u,
        Clear = 0x20u,
        Receipt = 0x40u,
        Dispatch = 0x80u,
        Flag1 = 0x2000u,
        System = 0xf000u,
    };

    u64 ident = 0;
    Filter filter = Filter::None;
    u16 flags = 0;
    u32 fflags = 0;
    u64 data = 0;
    void* udata = nullptr;
};
static_assert(sizeof(OrbisKernelEvent) == sizeof(hle::kernel::KernelEvent));

struct EqueueEvent {
    OrbisKernelEvent event;
    void* data = nullptr;
};

class EqueueInternal {
public:
    bool AddEvent(const EqueueEvent& added) {
        const OrbisKernelEvent& e = added.event;
        hle::kernel::AddEvent(Queue(), {e.ident, e.filter, e.flags, e.fflags, e.data, e.udata});
        return true;
    }
    bool RemoveEvent(u64 ident, s16 filter) {
        return hle::kernel::RemoveEvent(Queue(), ident, filter);
    }
    bool TriggerEvent(u64 ident, s16 filter, void* hint) {
        return hle::kernel::TriggerEvent(Queue(), ident, filter, hint);
    }

private:
    hle::kernel::EventQueue* Queue() {
        return reinterpret_cast<hle::kernel::EventQueue*>(this);
    }
};

using OrbisKernelEqueue = EqueueInternal*;

inline EqueueInternal* GetEqueue(OrbisKernelEqueue queue) {
    return queue;
}

inline u64 PS4_SYSV_ABI sceKernelGetEventData(const OrbisKernelEvent* event) {
    return event ? event->data : 0;
}

} // namespace Libraries::Kernel
