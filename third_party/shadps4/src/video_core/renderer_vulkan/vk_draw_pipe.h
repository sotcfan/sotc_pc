// SPDX-License-Identifier: GPL-2.0-or-later
// sotcport: two-stage draw pipeline. Adapted from bbport's vk_draw_pipe.h (the native Bloodborne
// port, GPL-2.0-or-later); see the "Two-stage draw pipeline" notes there.
//
// The GPU command thread (stage A) decodes PM4, keeps the register file and selects pipelines;
// for each draw or dispatch it writes a packet into this ring (the register blocks written since
// the previous packet, each shader stage's user data, the draw parameters) and goes on decoding.
// The draw recording thread (stage B) owns the buffer and texture caches, the barriers and the
// command recording while packets are in flight: it applies each packet to its own copy of the
// registers and runs the rest of the draw. Anything else stage A does on that state first waits
// for stage B to run dry (Drain) and then runs on stage A as before.

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>

#include <x86intrin.h>

#include "common/assert.h"
#include "common/thread.h"
#include "common/types.h"

namespace Vulkan {

class DrawPipe {
public:
    /// Runs one packet on stage B.
    using Handler = void (*)(void* context, const u8* packet, u32 size);

    DrawPipe(Handler handler_, void* context_) : handler{handler_}, context{context_} {
        ring = std::make_unique<u8[]>(Capacity);
        thread = std::jthread([this](std::stop_token stop) { Run(stop); });
    }

    ~DrawPipe() {
        Drain();
        thread.request_stop();
        wake.fetch_add(1, std::memory_order_seq_cst);
        wake.notify_one();
    }

    /// True on the draw recording thread.
    [[nodiscard]] static bool OnStageB() noexcept {
        return on_stage_b;
    }

    /// Stage A: space for a packet of `size` bytes, valid until Commit().
    u8* Begin(u32 size) {
        size = Align(size + sizeof(Header));
        ASSERT(size <= Capacity / 4);
        u64 at = head;
        const u64 offset = at % Capacity;
        if (offset + size > Capacity) {
            // Wrap: the rest of the ring is skipped (a header with zero size marks it).
            WaitForSpace(at, Capacity - offset);
            reinterpret_cast<Header*>(ring.get() + offset)->size = 0;
            at += Capacity - offset;
            head = at;
            published.store(at, std::memory_order_release);
        }
        WaitForSpace(at, size);
        pending_size = size;
        auto* header = reinterpret_cast<Header*>(ring.get() + at % Capacity);
        header->size = size;
        return reinterpret_cast<u8*>(header + 1);
    }

    /// Stage A: hands the packet from Begin() to stage B.
    void Commit(u32 payload_size) {
        reinterpret_cast<Header*>(ring.get() + head % Capacity)->payload = payload_size;
        head += pending_size;
        published.store(head, std::memory_order_seq_cst);
        packets.fetch_add(1, std::memory_order_relaxed);
        if (sleeping.load(std::memory_order_seq_cst)) {
            wake.fetch_add(1, std::memory_order_seq_cst);
            wake.notify_one();
        }
    }

    /// Why stage A drained (statistics): a graphics queue PM4 opcode (0-255), a compute queue
    /// one (ReasonComputeOpcode + opcode), or one of these.
    enum Reason : u32 {
        ReasonComputeOpcode = 256,
        ReasonCommands = 512, ///< commands queued by other threads (readbacks, flips)
        ReasonGpuIdle,        ///< all submissions done, the GPU is about to be reported idle
        ReasonConstRam,       ///< constant engine packet
        ReasonSrtPending,     ///< SRT walker read a page a packet in flight may write
        ReasonSrtGpuModified, ///< SRT walker read a page the GPU wrote
        ReasonFilterDraw,     ///< draw handled outside the normal path (clear, resolve...)
        ReasonFlush,
        ReasonFinish,
        ReasonSubmit,
        ReasonFence,
        ReasonFillBuffer,
        ReasonCopyBuffer,
        ReasonGds,
        ReasonInvalidate, ///< CPU write fault handled on the GPU side
        ReasonReadback,   ///< readback handled on the GPU side
        ReasonUnmap,
        ReasonOther,
        // sotcport: fences signalled on stage A (OnFence), by packet.
        ReasonFenceEos,            ///< graphics EVENT_WRITE_EOS
        ReasonFenceEopInCmdbuf,    ///< graphics EVENT_WRITE_EOP into the command buffer
        ReasonFenceWriteData,      ///< graphics WRITE_DATA
        ReasonFenceAscWriteData,   ///< compute WRITE_DATA
        ReasonFenceReleaseInCmdbuf, ///< compute RELEASE_MEM into the command buffer
        ReasonCommandsPending, ///< a command buffer about to be read has fences queued into it
        NumReasons,
    };
    static constexpr const char* ReasonName(u32 reason) {
        constexpr const char* names[] = {
            "commands", "gpu idle", "const ram", "srt pending", "srt gpu", "filter",
            "flush",    "finish",   "submit",    "fence",       "fill",    "copy",
            "gds",      "invalidate", "readback", "unmap",      "other",
            "fence: eos", "fence: eop in cmdbuf", "fence: write data", "fence: compute write data",
            "fence: release in cmdbuf", "cmdbuf pending",
        };
        return reason >= ReasonCommands && reason < NumReasons ? names[reason - ReasonCommands]
                                                               : nullptr;
    }
    static inline std::array<std::atomic<u64>, NumReasons> drains_by_reason{};
    static inline std::array<std::atomic<u64>, NumReasons> cycles_by_reason{};

    /// Stage A: waits until stage B has run every committed packet.
    void Drain(u32 reason = ReasonOther) {
        if (consumed.load(std::memory_order_acquire) == head) {
            return;
        }
        drains.fetch_add(1, std::memory_order_relaxed);
        const u32 slot = reason < NumReasons ? reason : ReasonOther;
        drains_by_reason[slot].fetch_add(1, std::memory_order_relaxed);
        const u64 start = __rdtsc();
        for (u32 spins = 0; consumed.load(std::memory_order_acquire) != head; ++spins) {
            if (spins < 4096) {
                _mm_pause();
            } else {
                std::this_thread::yield();
            }
        }
        const u64 waited = __rdtsc() - start;
        drain_cycles.fetch_add(waited, std::memory_order_relaxed);
        drain_cycles_total.fetch_add(waited, std::memory_order_relaxed);
        cycles_by_reason[slot].fetch_add(waited, std::memory_order_relaxed);
    }

    /// Stage A: position after the last committed packet; Reached(position) once B ran it.
    [[nodiscard]] u64 Head() const noexcept {
        return head;
    }
    [[nodiscard]] bool Reached(u64 position) const noexcept {
        return consumed.load(std::memory_order_acquire) >= position;
    }
    [[nodiscard]] bool Idle() const noexcept {
        return consumed.load(std::memory_order_acquire) == head;
    }

    /// Statistics, process-wide (read by the SOTC_PERF report): packets, drains that had to
    /// wait, TSC cycles stage A waited in them, TSC cycles stage B spent running packets.
    static inline std::atomic<u64> packets{0}, drains{0}, drain_cycles{0}, busy_cycles{0};
    /// Like drain_cycles, but never reset (the SOTC_SPIKES frame log takes differences).
    static inline std::atomic<u64> drain_cycles_total{0};

private:
    struct Header {
        u32 size;    ///< bytes to the next packet (0: wrap to the ring start)
        u32 payload; ///< bytes written by stage A
    };
    static constexpr u64 Capacity = 16ull << 20;

    static constexpr u32 Align(u32 size) {
        return (size + 63) & ~63u; // one cache line per packet start
    }

    void WaitForSpace(u64 at, u64 size) {
        while (at + size - consumed.load(std::memory_order_acquire) > Capacity) {
            _mm_pause();
        }
    }

    void Run(std::stop_token stop) {
        Common::SetCurrentThreadName("sotc:DrawRec");
        on_stage_b = true;
        u64 at = 0;
        // Packets follow each other within microseconds while a frame is decoded: spin, and
        // sleep only after a longer pause (between frames). SOTC_PIPE_SPIN_US overrides.
        static const auto spin_time = std::chrono::microseconds([] {
            const char* env = std::getenv("SOTC_PIPE_SPIN_US");
            return env ? std::max(0, std::atoi(env)) : 200;
        }());
        while (true) {
            u64 available = published.load(std::memory_order_acquire);
            if (available == at) {
                const auto spin_until = std::chrono::steady_clock::now() + spin_time;
                for (u32 spins = 1; available == at; ++spins) {
                    if (stop.stop_requested()) {
                        return;
                    }
                    _mm_pause();
                    if (!(spins & 255) && std::chrono::steady_clock::now() >= spin_until) {
                        const u32 seen = wake.load(std::memory_order_seq_cst);
                        sleeping.store(true, std::memory_order_seq_cst);
                        if (published.load(std::memory_order_seq_cst) == at &&
                            !stop.stop_requested()) {
                            wake.wait(seen, std::memory_order_seq_cst);
                        }
                        sleeping.store(false, std::memory_order_relaxed);
                    }
                    available = published.load(std::memory_order_acquire);
                }
            }
            const auto* header = reinterpret_cast<const Header*>(ring.get() + at % Capacity);
            if (header->size == 0) {
                at += Capacity - at % Capacity;
                consumed.store(at, std::memory_order_release);
                continue;
            }
            const u64 start = __rdtsc();
            handler(context, reinterpret_cast<const u8*>(header + 1), header->payload);
            busy_cycles.fetch_add(__rdtsc() - start, std::memory_order_relaxed);
            at += header->size;
            consumed.store(at, std::memory_order_release);
        }
    }

    static inline thread_local bool on_stage_b = false;
    Handler handler;
    void* context;
    std::unique_ptr<u8[]> ring;
    u64 head = 0; ///< stage A's write position
    u32 pending_size = 0;
    alignas(64) std::atomic<u64> published{0};
    alignas(64) std::atomic<u64> consumed{0};
    alignas(64) std::atomic<u32> wake{0};
    std::atomic<bool> sleeping{false};
    std::jthread thread;
};

} // namespace Vulkan
