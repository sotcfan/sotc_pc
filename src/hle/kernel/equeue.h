#pragma once

#include "common/types.h"

namespace hle::kernel {

// struct kevent of the PS4 (FreeBSD) kernel.
struct KernelEvent {
    u64 ident;
    s16 filter;
    u16 flags;
    u32 fflags;
    u64 data;
    void* udata;
};
static_assert(sizeof(KernelEvent) == 32);

constexpr s16 FilterUser = -11;
constexpr s16 FilterVideoOut = -13;
constexpr s16 FilterGraphicsCore = -14;

constexpr u16 EventAdd = 0x1;
constexpr u16 EventOneShot = 0x10; // removed after it is delivered
constexpr u16 EventClear = 0x20;   // waits for a new trigger after it is delivered

class EventQueue;

// For other libraries (VideoOut, Gnm) that deliver events through a queue the game created.
// Adding replaces an event with the same ident and filter.
void AddEvent(EventQueue* queue, const KernelEvent& event);
bool RemoveEvent(EventQueue* queue, u64 ident, s16 filter);

// Fires an event. What `hint` carries depends on the filter, as in the PS4 kernel:
// - user events: the new udata;
// - VideoOut: event id in the low byte and the flip argument from bit 16; the event's data
//   becomes a timestamp, a counter and that argument;
// - anything else: the new data.
bool TriggerEvent(EventQueue* queue, u64 ident, s16 filter, void* hint);

} // namespace hle::kernel
