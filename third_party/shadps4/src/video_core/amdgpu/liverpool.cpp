// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <boost/preprocessor/stringize.hpp>

#include "common/assert.h"
#include "common/debug.h"
#include "common/polyfill_thread.h"
#include "common/thread.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/libraries/kernel/process.h"
#include "core/libraries/videoout/driver.h"
#include "core/memory.h"
#include "core/platform.h"
#include "video_core/amdgpu/packet_observer.h"
#include "video_core/amdgpu/liverpool.h"

// sotcport: packets the game sends every frame; the warning once is enough.
#define LOG_WARNING_ONCE_SHADOW(category, ...)                                                    \
    do {                                                                                          \
        static bool warned = false;                                                               \
        if (!std::exchange(warned, true)) {                                                       \
            LOG_WARNING(category, __VA_ARGS__);                                                   \
        }                                                                                         \
    } while (false)
#include "video_core/amdgpu/pm4_cmds.h"
#include "video_core/renderdoc.h"
#include "video_core/renderer_vulkan/vk_draw_pipe.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "perf_temp.h" // sotcport TEMP
#include "hle/kernel/gpu_memory.h"
#include "diag/memory_types.h"

namespace AmdGpu {

static const char* dcb_task_name{"DCB_TASK"};
static const char* ccb_task_name{"CCB_TASK"};

#define MAX_NAMES 56
static_assert(Liverpool::NumComputeRings <= MAX_NAMES);

#define NAME_NUM(z, n, name) BOOST_PP_STRINGIZE(name) BOOST_PP_STRINGIZE(n),
#define NAME_ARRAY(name, num) {BOOST_PP_REPEAT(num, NAME_NUM, name)}

static const char* acb_task_name[] = NAME_ARRAY(ACB_TASK, MAX_NAMES);

#define YIELD(name)                                                                                \
    FIBER_EXIT;                                                                                    \
    co_yield {};                                                                                   \
    FIBER_ENTER(name);

#define YIELD_CE() YIELD(ccb_task_name)
#define YIELD_GFX() YIELD(dcb_task_name)
#define YIELD_ASC(id) YIELD(acb_task_name[id])

#define RESUME(task, name)                                                                         \
    FIBER_EXIT;                                                                                    \
    task.handle.resume();                                                                          \
    FIBER_ENTER(name);

#define RESUME_CE(task) RESUME(task, ccb_task_name)
#define RESUME_GFX(task) RESUME(task, dcb_task_name)
#define RESUME_ASC(task, id) RESUME(task, acb_task_name[id])

std::array<u8, 48_KB> Liverpool::ConstantEngine::constants_heap;

static std::span<const u32> NextPacket(std::span<const u32> span, size_t offset) {
    if (offset > span.size()) {
        LOG_ERROR(
            Lib_GnmDriver,
            ": packet length exceeds remaining submission size. Packet dword count={}, remaining "
            "submission dwords={}",
            offset, span.size());
        // Return empty subspan so check for next packet bails out
        return {};
    }

    return span.subspan(offset);
}

Liverpool::Liverpool() : guest_markers_enabled{EmulatorSettings.IsVkGuestMarkersEnabled()} {
    num_counter_pairs = Libraries::Kernel::sceKernelIsNeoMode() ? 16 : 8;
    process_thread = std::jthread{std::bind_front(&Liverpool::Process, this)};
}

bool Liverpool::PipelinedComputeOpcode(PM4ItOpcode opcode) {
    // SOTC_PIPE_SYNC_COMPUTE=1: every compute queue packet drains (A/B measurements).
    static const bool sync_compute = std::getenv("SOTC_PIPE_SYNC_COMPUTE") != nullptr;
    switch (opcode) {
    case PM4ItOpcode::Nop:
        return true;
    case PM4ItOpcode::IndirectBuffer: // only recurses into ProcessCompute
    case PM4ItOpcode::SetShReg:
    case PM4ItOpcode::DispatchDirect:
    case PM4ItOpcode::DispatchIndirect:
    case PM4ItOpcode::AcquireMem: // no effect here
    case PM4ItOpcode::EventWrite: // no effect here
    case PM4ItOpcode::Rewind:     // waits on the command buffer itself
    case PM4ItOpcode::WaitRegMem: // drains in WaitRegMemMet when needed
    case PM4ItOpcode::ReleaseMem: // signalled by stage B (or drains in OnFence)
        return !sync_compute;
    default:
        return false;
    }
}

bool Liverpool::PipelinedOpcode(PM4ItOpcode opcode, const u32* packet) {
    switch (opcode) {
    case PM4ItOpcode::Nop: // debug markers only when guest markers are on (checked by the caller)
    case PM4ItOpcode::ContextControl:
    case PM4ItOpcode::ClearState:
    case PM4ItOpcode::SetConfigReg:
    case PM4ItOpcode::SetContextReg:
    case PM4ItOpcode::SetShReg:
    case PM4ItOpcode::SetUconfigReg:
    case PM4ItOpcode::SetPredication: // not implemented: only a warning
    case PM4ItOpcode::IndexType:
    case PM4ItOpcode::DrawIndex2:
    case PM4ItOpcode::DrawIndexOffset2:
    case PM4ItOpcode::DrawIndexAuto:
    case PM4ItOpcode::DrawIndirect:
    case PM4ItOpcode::DrawIndirectMulti:
    case PM4ItOpcode::DrawIndexIndirect:
    case PM4ItOpcode::DrawIndexIndirectMulti:
    case PM4ItOpcode::DrawIndexIndirectCountMulti:
    case PM4ItOpcode::DispatchDirect:
    case PM4ItOpcode::DispatchIndirect:
    case PM4ItOpcode::NumInstances:
    case PM4ItOpcode::IndexBase:
    case PM4ItOpcode::IndexBufferSize:
    case PM4ItOpcode::SetBase:
    case PM4ItOpcode::AcquireMem: // no effect here
    case PM4ItOpcode::PfpSyncMe:  // no effect here
    case PM4ItOpcode::IncrementDeCounter:
    case PM4ItOpcode::WaitOnCeCounter:
    case PM4ItOpcode::EventWriteEop: // signalled by stage B (or drains in OnFence)
    case PM4ItOpcode::WaitRegMem:    // drains in WaitRegMemMet when needed
        return true;
    case PM4ItOpcode::EventWriteEos: // label writes: signalled by stage B (or drain in OnFence)
        return reinterpret_cast<const PM4CmdEventWriteEos*>(packet)->command ==
               PM4CmdEventWriteEos::Command::SignalFence;
    case PM4ItOpcode::DmaData:
        // The ~100k/s DmaData to 0x3022C (a register the handler skips) do nothing.
        return reinterpret_cast<const PM4DmaData*>(packet)->dst_addr_lo == 0x3022C;
    case PM4ItOpcode::EventWrite: {
        // ~1,000/s in the benchmark scene. The handler only writes a register (streamout flush:
        // register state travels with the packets) or, for an occlusion query dump, fake counters
        // to guest memory: only that one waits for stage B.
        const auto* event = reinterpret_cast<const PM4CmdEventWrite*>(packet);
        return !(event->event_index.Value() == EventIndex::ZpassDone &&
                 event->event_type.Value() == EventType::PixelPipeStatDump);
    }
    default:
        return false;
    }
}

// sotcport: two-stage draw pipeline groundwork. Called on the GPU command thread right before
// each draw and dispatch of the graphics queue; `reg_dirty` then holds the register blocks written
// since the previous one, which is what the second stage would receive with this draw.
// SOTC_PIPE_VERIFY=N checks that this is complete: a copy of the registers is kept by applying
// only those blocks (and the defaults after a ClearState), and every Nth checkpoint it is
// compared word for word with the real register file; differences go to the log.
void Liverpool::PipeCheckpoint() {
    if (pipe_owns_dirty) {
        return; // the draw pipeline's packets take the blocks (and verify them themselves)
    }
    static const u32 verify_every = [] {
        const char* env = std::getenv("SOTC_PIPE_VERIFY");
        return env ? u32(std::max(1, std::atoi(env))) : 0u;
    }();
    if (verify_every == 0) {
        reg_dirty.Clear();
        return;
    }
    static std::unique_ptr<Regs> copy;
    static u64 checkpoints = 0, checks = 0, mismatches = 0;
    if (!copy) {
        copy = std::make_unique<Regs>(regs);
        reg_dirty.Clear();
        return;
    }
    if (reg_dirty.reset) {
        copy->SetDefaults();
    }
    for (u32 block = 0; block < RegDirty::NumBlocks; ++block) {
        if (reg_dirty.blocks.test(block)) {
            std::memcpy(copy->reg_array.data() + block * RegDirty::BlockWords,
                        regs.reg_array.data() + block * RegDirty::BlockWords,
                        RegDirty::BlockWords * sizeof(u32));
        }
    }
    reg_dirty.Clear();
    if (++checkpoints % verify_every != 0) {
        return;
    }
    ++checks;
    if (std::memcmp(copy->reg_array.data(), regs.reg_array.data(), sizeof(regs.reg_array)) != 0) {
        ++mismatches;
        std::string words;
        u32 count = 0;
        for (u32 i = 0; i < Regs::NumRegs; ++i) {
            if (copy->reg_array[i] != regs.reg_array[i]) {
                if (count++ < 8) {
                    words += fmt::format(" {:#06x}:{:#x}->{:#x}", i, copy->reg_array[i],
                                         regs.reg_array[i]);
                }
            }
        }
        if (mismatches <= 20) {
            LOG_ERROR(Render, "PIPE VERIFY: {} register words differ from the tracked copy:{}",
                      count, words);
        }
        *copy = regs; // resynchronize so later checks are independent
    }
    if (checks % 20000 == 0) {
        LOG_INFO(Render, "PIPE VERIFY: {} checks, {} mismatches", checks, mismatches);
    }
}

Liverpool::~Liverpool() {
    process_thread.request_stop();
    process_thread.join();
}

void Liverpool::SignalEopFence(const PM4CmdEventWriteEop& packet) {
    diag::NoteMemoryAccess(diag::MemAccess::Label, reinterpret_cast<u64>(packet.Address<u8>()),
                           sizeof(u64)); // sotcport (SOTC_MEMORY_TYPES)
    packet.SignalFence(
        [](void* address, u64 data, u32 num_bytes) {
            auto* memory = Core::Memory::Instance();
            ASSERT(memory->TryWriteBacking(address, &data, num_bytes));
        },
        [] { Platform::IrqC::Instance()->Signal(Platform::InterruptId::GfxEop); });
}

void Liverpool::SignalReleaseMem(const PM4CmdReleaseMem& packet, u32 pipe_id) {
    diag::NoteMemoryAccess(diag::MemAccess::Label, packet.Address<VAddr>(),
                           sizeof(u64)); // sotcport (SOTC_MEMORY_TYPES)
    packet.SignalFence(
        [pipe_id] {
            Platform::IrqC::Instance()->Signal(static_cast<Platform::InterruptId>(pipe_id));
        },
        [this](VAddr dst, u16 gds_index, u16 num_dwords) {
            rasterizer->CopyBuffer(dst, gds_index, num_dwords * sizeof(u32), false, true);
        });
}

// sotcport: whether a fence writing [dst, +bytes) can change commands this thread has yet to
// read in [begin, end), the command buffer being processed; `current` is where the packets not
// read yet start (the end of the packet being processed, which has been read already). The game allocates its labels from its command buffers (inside NOPs, ahead of the
// packets that use them), so ~9,000 fences a second land there: always behind the current packet
// in the benchmark scene (measured 2026-10-14). Those cannot change what is left to read and are
// signalled by stage B in order; a command buffer read again later drains first if a fence still
// queued writes into it (DrainIfCommandsPending). Only fences ahead are written right away.
static bool WritesUnreadCommands(VAddr begin, VAddr end, const u32* current, VAddr dst,
                                 u32 bytes) {
    const auto current_address = reinterpret_cast<VAddr>(current);
    if (dst >= end || dst + bytes <= begin) {
        return false;
    }
    // A packet copied out of a ring (it wrapped) is not where the ring is: be conservative.
    if (current_address < begin || current_address >= end) {
        return true;
    }
    return dst + bytes > current_address;
}

// sotcport: advances a compute queue's read pointer, which lives in guest memory. Written through
// the backing view (not the guest address): the pointer often shares a tracking page with data the
// GPU wrote, and an access to such a page is a read fault that downloads GPU data and waits for
// the whole GPU queue to finish. That happened after every compute packet, ~170 times per second
// in the benchmark scene, each a full GPU sync on this thread. Only the command processor writes
// the pointer (the GPU never does), so no GPU data can be pending for these 4 bytes.
static void AdvanceReadPointer(u32* read_addr, u32 dwords, u32 ring_size_dw) {
    auto* pointer = reinterpret_cast<u32*>(
        hle::kernel::BackingAddress(reinterpret_cast<u64>(read_addr)));
    if (!pointer) {
        pointer = read_addr;
    }
    std::atomic_ref<u32> value{*pointer};
    value.store((value.load(std::memory_order_relaxed) + dwords) % ring_size_dw,
                std::memory_order_release);
}

bool Liverpool::WaitRegMemMet(const PM4CmdWaitRegMem& wait) {
    if (wait.Test(regs.reg_array)) {
        return true;
    }
    if (!rasterizer) {
        return false;
    }
    if (rasterizer->PendingFenceMeets(wait)) {
        return true;
    }
    // Draining stage B can only help if work queued there writes the polled memory (a fence of a
    // kind PendingFenceMeets cannot predict, or a shader whose writes become visible once the
    // write is recorded). Most of these waits poll a label another queue has yet to write (the
    // game's queues waiting for each other: ~1,300 per second in the benchmark scene, every one
    // still unmet after the drain); those just yield to the other queues.
    if (wait.mem_space == PM4CmdWaitRegMem::MemSpace::Memory &&
        !rasterizer->PendingWriteOverlaps(reinterpret_cast<VAddr>(wait.Address()), 4)) {
        return false;
    }
    rasterizer->DrainDrawPipe(static_cast<u32>(PM4ItOpcode::WaitRegMem));
    const bool met = wait.Test(regs.reg_array);
    perf::NoteWaitRegMem(wait.mem_space == PM4CmdWaitRegMem::MemSpace::Memory
                             ? reinterpret_cast<u64>(wait.Address())
                             : wait.Reg(),
                         met, wait.function.Value() == PM4CmdWaitRegMem::Function::Equal,
                         wait.ref); // sotcport TEMP
    return met;
}

void Liverpool::ProcessCommands() {
    // Process incoming commands with high priority
    while (num_commands) {
        Common::UniqueFunction<void> callback{};
        {
            std::scoped_lock lk{submit_mutex};
            callback = std::move(command_queue.front());
            command_queue.pop();
            --num_commands;
        }
        // sotcport: the commands (flips, readbacks for guest threads' faults, unmaps) work on the
        // caches and the scheduler, which stage B of the draw pipeline owns while packets are in
        // flight. They are handed to stage B to run in order with the packets
        // (SOTC_PIPE_COMMANDS_ON_B=1), or stage B runs dry first and they run here.
        if (rasterizer && rasterizer->PostCommand(callback)) {
            continue;
        }
        if (rasterizer) {
            rasterizer->DrainDrawPipe(Vulkan::DrawPipe::ReasonCommands);
        }
        callback();
    }
}

void Liverpool::Process(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:GpuCommandProcessor");
    gpu_id = std::this_thread::get_id();
#ifdef __linux__
    gpu_tid = gettid();
#endif

    while (!stoken.stop_requested()) {
        {
            std::unique_lock lk{submit_mutex};
            Common::CondvarWait(submit_cv, lk, stoken,
                                [this] { return num_commands || num_submits || submit_done; });
        }
        if (stoken.stop_requested()) {
            break;
        }

        VideoCore::StartCapture();
        perf::Scope perf_busy{perf::GpuCommBusy}; // sotcport TEMP
        perf::Mark('a'); // TEMP timeline: command processor wakes

        curr_qid = -1;

        while (num_submits || num_commands) {
            // sotcport: between packets: stage B may run urgent work (readbacks) again.
            if (rasterizer) {
                rasterizer->ReleaseStageA();
            }
            ProcessCommands();

            curr_qid = (curr_qid + 1) % num_mapped_queues;

            auto& queue = mapped_queues[curr_qid];

            Task::Handle task{};
            {
                std::scoped_lock lock{queue.m_access};
                if (queue.submits.empty()) {
                    continue;
                }
                task = queue.submits.front();
            }
            task.resume();

            if (task.done()) {
                task.destroy();

                std::scoped_lock lock{queue.m_access};
                queue.submits.pop();

                --num_submits;
                std::scoped_lock lock2{submit_mutex};
                submit_cv.notify_all();
            }
        }

        if (submit_done) {
            VideoCore::EndCapture();
            if (rasterizer) {
                perf::Scope perf_flush{perf::SubmitFlush}; // sotcport TEMP
                rasterizer->OnSubmit();
                rasterizer->Flush();
            }
            submit_done = false;
        }

        // sotcport: the guest may free what queued work still touches once the GPU is idle
        // (bbport's heap corruption): stage B runs dry first.
        perf::Mark('i'); // TEMP timeline: command processor out of work
        if (rasterizer) {
            rasterizer->DrainDrawPipe(Vulkan::DrawPipe::ReasonGpuIdle);
        }
        perf::Mark('d'); // TEMP timeline: stage B drained, GPU idle signalled
        if (rasterizer) {
            rasterizer->ReleaseStageA(); // sotcport: idle
        }
        Platform::IrqC::Instance()->Signal(Platform::InterruptId::GpuIdle);
    }
}

Liverpool::Task Liverpool::ProcessCeUpdate(std::span<const u32> ccb) {
    FIBER_ENTER(ccb_task_name);

    while (!ccb.empty()) {
        ProcessCommands();

        const auto* header = reinterpret_cast<const PM4Header*>(ccb.data());
        const u32 type = header->type;
        if (g_packet_observer) [[unlikely]] {
            g_packet_observer(ccb.data(), type == 3 ? header->type3.NumWords() + 1 : 1,
                              PacketQueue::Constant);
        }
        if (type != 3) {
            // No other types of packets were spotted so far
            UNREACHABLE_MSG("Invalid PM4 type {}", type);
        }

        const PM4ItOpcode opcode = header->type3.opcode;
        const auto* it_body = reinterpret_cast<const u32*>(header) + 1;
        // sotcport: two-stage draw pipeline: constant RAM dumps write guest memory.
        if (rasterizer && opcode != PM4ItOpcode::Nop && opcode != PM4ItOpcode::WriteConstRam &&
            opcode != PM4ItOpcode::IncrementCeCounter &&
            opcode != PM4ItOpcode::WaitOnDeCounterDiff) {
            rasterizer->DrainDrawPipe(Vulkan::DrawPipe::ReasonConstRam);
        }
        switch (opcode) {
        case PM4ItOpcode::Nop: {
            // const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
            break;
        }
        case PM4ItOpcode::WriteConstRam: {
            const auto* write_const = reinterpret_cast<const PM4WriteConstRam*>(header);
            memcpy(cblock.constants_heap.data() + write_const->Offset(), &write_const->data,
                   write_const->Size());
            break;
        }
        case PM4ItOpcode::DumpConstRam: {
            const auto* dump_const = reinterpret_cast<const PM4DumpConstRam*>(header);
            memcpy(dump_const->Address<void*>(),
                   cblock.constants_heap.data() + dump_const->Offset(), dump_const->Size());
            break;
        }
        case PM4ItOpcode::IncrementCeCounter: {
            ++cblock.ce_count;
            break;
        }
        case PM4ItOpcode::WaitOnDeCounterDiff: {
            const auto diff = it_body[0];
            while ((cblock.de_count - cblock.ce_count) >= diff) {
                YIELD_CE();
            }
            break;
        }
        case PM4ItOpcode::IndirectBufferConst: {
            const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
            diag::NoteMemoryAccess(diag::MemAccess::IndirectCommands, // sotcport
                                   reinterpret_cast<u64>(indirect_buffer->Address<const u32>()),
                                   u64{indirect_buffer->ib_size} * sizeof(u32));
            auto task =
                ProcessCeUpdate({indirect_buffer->Address<const u32>(), indirect_buffer->ib_size});
            RESUME_CE(task);

            while (!task.handle.done()) {
                YIELD_CE();
                RESUME_CE(task);
            }
            break;
        }
        default:
            const u32 count = header->type3.NumWords();
            UNREACHABLE_MSG("Unknown PM4 type 3 opcode {:#x} with count {}",
                            static_cast<u32>(opcode), count);
        }
        ccb = NextPacket(ccb, header->type3.NumWords() + 1);
    }

    FIBER_EXIT;
}

Liverpool::Task Liverpool::ProcessGraphics(std::span<const u32> dcb, std::span<const u32> ccb) {
    FIBER_ENTER(dcb_task_name);
    // sotcport: the command buffer being read (fences that patch it are signalled at once).
    const auto cmd_begin = reinterpret_cast<VAddr>(dcb.data());
    const auto cmd_end = cmd_begin + dcb.size_bytes();
    // sotcport (two-stage draw pipeline): fences still queued for stage B may write into this
    // command buffer (labels in it, see WritesUnreadCommands): they land before it is read.
    if (rasterizer && rasterizer->PendingWriteOverlaps(cmd_begin, dcb.size_bytes())) {
        rasterizer->DrainDrawPipe(Vulkan::DrawPipe::ReasonCommandsPending);
    }

    cblock.Reset();

    // TODO: potentially, ASCs also can depend on CE and in this case the
    // CE task should be moved into more global scope
    Task ce_task{};

    if (!ccb.empty()) {
        // In case of CCB provided kick off CE asap to have the constant heap ready to use
        ce_task = ProcessCeUpdate(ccb);
        RESUME_GFX(ce_task);
    }

    const auto base_addr = reinterpret_cast<uintptr_t>(dcb.data());
    while (!dcb.empty()) {
        ProcessCommands();

        const auto* header = reinterpret_cast<const PM4Header*>(dcb.data());
        const u32 type = header->type;
        if (g_packet_observer) [[unlikely]] {
            g_packet_observer(dcb.data(), type == 3 ? header->type3.NumWords() + 1 : 1,
                              PacketQueue::Graphics);
        }

        switch (type) {
        default:
            UNREACHABLE_MSG("Wrong PM4 type {}", type);
            break;
        case 0:
            UNREACHABLE_MSG("Unimplemented PM4 type 0, base reg: {}, size: {}",
                            header->type0.base.Value(), header->type0.NumWords());
            break;
        case 2:
            // Type-2 packet are used for padding purposes
            dcb = NextPacket(dcb, 1);
            continue;
        case 3:
            const u32 count = header->type3.NumWords();
            const PM4ItOpcode opcode = header->type3.opcode;
            // sotcport: two-stage draw pipeline: everything but register writes, draws,
            // dispatches and no-ops waits for stage B to run dry.
            if (rasterizer && (!PipelinedOpcode(opcode, dcb.data()) ||
                               (opcode == PM4ItOpcode::Nop && guest_markers_enabled))) {
                rasterizer->DrainDrawPipe(static_cast<u32>(opcode));
            }
            switch (opcode) {
            case PM4ItOpcode::Nop: {
                const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
                if (nop->header.count.Value() == 0) {
                    break;
                }

                switch (nop->data_block[0]) {
                case PM4CmdNop::PayloadType::PatchedFlip: {
                    // sotcport: the flip handler prepares the frame for presentation right here
                    // (PrepareFrame copies the image, uses the texture cache and records on the
                    // draw scheduler): stage B must have recorded every draw of the frame first.
                    // Without this drain both threads used those at once: random corruption
                    // (a crash at startup in about half the runs; shadows of the wrong frame).
                    if (rasterizer) {
                        rasterizer->DrainDrawPipe(static_cast<u32>(opcode));
                    }
                    // There is no evidence that GPU CP drives flip events by parsing
                    // special NOP packets. For convenience lets assume that it does.
                    Platform::IrqC::Instance()->Signal(Platform::InterruptId::GfxFlip);
                    break;
                }
                case PM4CmdNop::PayloadType::DebugMarkerPush: {
                    if (guest_markers_enabled) {
                        const auto marker_sz = nop->header.count.Value() * 2;
                        const std::string_view label{
                            reinterpret_cast<const char*>(&nop->data_block[1]), marker_sz};
                        rasterizer->ScopeMarkerBegin(label, true);
                    }
                    break;
                }
                case PM4CmdNop::PayloadType::DebugColorMarkerPush: {
                    if (guest_markers_enabled) {
                        const auto marker_sz = nop->header.count.Value() * 2;
                        const std::string_view label{
                            reinterpret_cast<const char*>(&nop->data_block[1]), marker_sz};
                        const u32 color = *reinterpret_cast<const u32*>(
                            reinterpret_cast<const u8*>(&nop->data_block[1]) + marker_sz);
                        rasterizer->ScopedMarkerInsertColor(label, color, true);
                    }
                    break;
                }
                case PM4CmdNop::PayloadType::DebugMarkerPop: {
                    if (guest_markers_enabled) {
                        rasterizer->ScopeMarkerEnd(true);
                    }
                    break;
                }
                default:
                    break;
                }
                break;
            }
            case PM4ItOpcode::ContextControl: {
                break;
            }
            case PM4ItOpcode::ClearState: {
                regs.SetDefaults();
                reg_dirty.Clear();
                reg_dirty.reset = true;
                break;
            }
            case PM4ItOpcode::SetConfigReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto reg_addr = Regs::ConfigRegWordOffset + set_data->reg_offset;
                const auto* payload = reinterpret_cast<const u32*>(header + 2);
                std::memcpy(&regs.reg_array[reg_addr], payload, (count - 1) * sizeof(u32));
                reg_dirty.Mark(reg_addr, count - 1);
                break;
            }
            case PM4ItOpcode::SetContextReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto reg_addr = Regs::ContextRegWordOffset + set_data->reg_offset;
                const auto* payload = reinterpret_cast<const u32*>(header + 2);

                std::memcpy(&regs.reg_array[reg_addr], payload, (count - 1) * sizeof(u32));
                reg_dirty.Mark(reg_addr, count - 1);

                // In the case of HW, render target memory has alignment as color block operates on
                // tiles. There is no information of actual resource extents stored in CB context
                // regs, so any deduction of it from slices/pitch will lead to a larger surface
                // created. The same applies to the depth targets. Fortunately, the guest always
                // sends a trailing NOP packet right after the context regs setup, so we can use the
                // heuristic below and extract the hint to determine actual resource dims.

                switch (reg_addr) {
                case ContextRegs::CbColor0Base:
                case ContextRegs::CbColor1Base:
                case ContextRegs::CbColor2Base:
                case ContextRegs::CbColor3Base:
                case ContextRegs::CbColor4Base:
                case ContextRegs::CbColor5Base:
                case ContextRegs::CbColor6Base:
                case ContextRegs::CbColor7Base: {
                    const auto col_buf_id = (reg_addr - ContextRegs::CbColor0Base) /
                                            (ContextRegs::CbColor1Base - ContextRegs::CbColor0Base);
                    ASSERT(col_buf_id < NUM_COLOR_BUFFERS);

                    const auto nop_offset = header->type3.count;
                    if (nop_offset == 0x0e || nop_offset == 0x0d || nop_offset == 0x0b) {
                        ASSERT_MSG(payload[nop_offset] == 0xc0001000,
                                   "NOP hint is missing in CB setup sequence");
                        last_cb_extent[col_buf_id].raw = payload[nop_offset + 1];
                    } else {
                        last_cb_extent[col_buf_id].raw = 0;
                    }
                    break;
                }
                case ContextRegs::CbColor0Cmask:
                case ContextRegs::CbColor1Cmask:
                case ContextRegs::CbColor2Cmask:
                case ContextRegs::CbColor3Cmask:
                case ContextRegs::CbColor4Cmask:
                case ContextRegs::CbColor5Cmask:
                case ContextRegs::CbColor6Cmask:
                case ContextRegs::CbColor7Cmask: {
                    const auto col_buf_id =
                        (reg_addr - ContextRegs::CbColor0Cmask) /
                        (ContextRegs::CbColor1Cmask - ContextRegs::CbColor0Cmask);
                    ASSERT(col_buf_id < NUM_COLOR_BUFFERS);

                    const auto nop_offset = header->type3.count;
                    if (nop_offset == 0x04) {
                        ASSERT_MSG(payload[nop_offset] == 0xc0001000,
                                   "NOP hint is missing in CB setup sequence");
                        last_cb_extent[col_buf_id].raw = payload[nop_offset + 1];
                    }
                    break;
                }
                case ContextRegs::DbZInfo: {
                    if (header->type3.count == 8) {
                        ASSERT_MSG(payload[20] == 0xc0001000,
                                   "NOP hint is missing in DB setup sequence");
                        last_db_extent.raw = payload[21];
                    } else {
                        last_db_extent.raw = 0;
                    }
                    break;
                }
                default:
                    break;
                }
                break;
            }
            case PM4ItOpcode::SetShReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto set_size = (count - 1) * sizeof(u32);

                if (set_data->reg_offset >= 0x200 &&
                    set_data->reg_offset <= (0x200 + sizeof(ComputeProgram) / 4)) {
                    ASSERT(set_size <= sizeof(ComputeProgram));
                    auto* addr = reinterpret_cast<u32*>(&mapped_queues[GfxQueueId].cs_state) +
                                 (set_data->reg_offset - 0x200);
                    std::memcpy(addr, header + 2, set_size);
                } else {
                    std::memcpy(&regs.reg_array[Regs::ShRegWordOffset + set_data->reg_offset],
                                header + 2, set_size);
                    reg_dirty.Mark(Regs::ShRegWordOffset + set_data->reg_offset, count - 1);
                }
                break;
            }
            case PM4ItOpcode::SetUconfigReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                std::memcpy(&regs.reg_array[Regs::UconfigRegWordOffset + set_data->reg_offset],
                            header + 2, (count - 1) * sizeof(u32));
                reg_dirty.Mark(Regs::UconfigRegWordOffset + set_data->reg_offset, count - 1);
                break;
            }
            case PM4ItOpcode::SetPredication: {
                LOG_WARNING_ONCE_SHADOW(Render, "Unimplemented IT_SET_PREDICATION"); // sotcport: once
                break;
            }
            case PM4ItOpcode::IndexType: {
                const auto* index_type = reinterpret_cast<const PM4CmdDrawIndexType*>(header);
                regs.index_buffer_type.raw = index_type->raw;
                reg_dirty.MarkField(regs, regs.index_buffer_type);
                break;
            }
            case PM4ItOpcode::DrawIndex2: {
                const auto* draw_index = reinterpret_cast<const PM4CmdDrawIndex2*>(header);
                regs.max_index_size = draw_index->max_size;
                regs.index_base_address.base_addr_lo = draw_index->index_base_lo;
                regs.index_base_address.base_addr_hi = draw_index->index_base_hi;
                regs.num_indices = draw_index->index_count;
                regs.draw_initiator = draw_index->draw_initiator;
                reg_dirty.MarkField(regs, regs.max_index_size);
                reg_dirty.MarkField(regs, regs.index_base_address);
                reg_dirty.MarkField(regs, regs.num_indices);
                reg_dirty.MarkField(regs, regs.draw_initiator);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                PipeCheckpoint();
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker("gfx:{}:DrawIndex2", fmt::make_format_args(cmd_address),
                                        [&] { rasterizer->Draw(true); });
                break;
            }
            case PM4ItOpcode::DrawIndexOffset2: {
                const auto* draw_index_off =
                    reinterpret_cast<const PM4CmdDrawIndexOffset2*>(header);
                regs.max_index_size = draw_index_off->max_size;
                regs.num_indices = draw_index_off->index_count;
                regs.draw_initiator = draw_index_off->draw_initiator;
                reg_dirty.MarkField(regs, regs.max_index_size);
                reg_dirty.MarkField(regs, regs.num_indices);
                reg_dirty.MarkField(regs, regs.draw_initiator);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                PipeCheckpoint();
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker(
                    "gfx:{}:DrawIndexOffset2", fmt::make_format_args(cmd_address),
                    [&] { rasterizer->Draw(true, draw_index_off->index_offset); });
                break;
            }
            case PM4ItOpcode::DrawIndexAuto: {
                const auto* draw_index = reinterpret_cast<const PM4CmdDrawIndexAuto*>(header);
                regs.num_indices = draw_index->index_count;
                regs.draw_initiator = draw_index->draw_initiator;
                reg_dirty.MarkField(regs, regs.num_indices);
                reg_dirty.MarkField(regs, regs.draw_initiator);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                PipeCheckpoint();
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker("gfx:{}:DrawIndexAuto", fmt::make_format_args(cmd_address),
                                        [&] { rasterizer->Draw(false); });
                break;
            }
            case PM4ItOpcode::DrawIndirect: {
                const auto* draw_indirect = reinterpret_cast<const PM4CmdDrawIndirect*>(header);
                const auto offset = draw_indirect->data_offset;
                const auto stride = sizeof(DrawIndirectArgs);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                PipeCheckpoint();
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker(
                    "gfx:{}:DrawIndirect", fmt::make_format_args(cmd_address), [&] {
                        rasterizer->DrawIndirect(false, indirect_args_addr, offset, stride, 1, 0,
                                                 draw_indirect->base_vtx_loc,
                                                 draw_indirect->start_inst_loc);
                    });
                break;
            }
            case PM4ItOpcode::DrawIndirectMulti: {
                const auto* draw_indirect =
                    reinterpret_cast<const PM4CmdDrawIndirectMulti*>(header);
                const auto offset = draw_indirect->data_offset;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                PipeCheckpoint();
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker(
                    "gfx:{}:DrawIndirectMulti", fmt::make_format_args(cmd_address), [&] {
                        rasterizer->DrawIndirect(false, indirect_args_addr, offset,
                                                 draw_indirect->stride, draw_indirect->count, 0,
                                                 draw_indirect->base_vtx_loc,
                                                 draw_indirect->start_inst_loc);
                    });
                break;
            }
            case PM4ItOpcode::DrawIndexIndirect: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirect*>(header);
                const auto offset = draw_index_indirect->data_offset;
                const auto stride = sizeof(DrawIndexedIndirectArgs);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                PipeCheckpoint();
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker(
                    "gfx:{}:DrawIndexIndirect", fmt::make_format_args(cmd_address), [&] {
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset, stride, 1, 0,
                                                 draw_index_indirect->base_vtx_loc,
                                                 draw_index_indirect->start_inst_loc);
                    });
                break;
            }
            case PM4ItOpcode::DrawIndexIndirectMulti: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirectMulti*>(header);
                const auto offset = draw_index_indirect->data_offset;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                PipeCheckpoint();
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker(
                    "gfx:{}:DrawIndexIndirectMulti", fmt::make_format_args(cmd_address), [&] {
                        rasterizer->DrawIndirect(
                            true, indirect_args_addr, offset, draw_index_indirect->stride,
                            draw_index_indirect->count, 0, draw_index_indirect->base_vtx_loc,
                            draw_index_indirect->start_inst_loc);
                    });
                break;
            }
            case PM4ItOpcode::DrawIndexIndirectCountMulti: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirectCountMulti*>(header);
                const auto offset = draw_index_indirect->data_offset;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                PipeCheckpoint();
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker(
                    "gfx:{}:DrawIndexIndirectCountMulti", fmt::make_format_args(cmd_address), [&] {
                        rasterizer->DrawIndirect(
                            true, indirect_args_addr, offset, draw_index_indirect->stride,
                            draw_index_indirect->count,
                            draw_index_indirect->count_indirect_enable.Value()
                                ? draw_index_indirect->count_addr
                                : 0,
                            draw_index_indirect->base_vtx_loc, draw_index_indirect->start_inst_loc);
                    });
                break;
            }
            case PM4ItOpcode::DispatchDirect: {
                const auto* dispatch_direct = reinterpret_cast<const PM4CmdDispatchDirect*>(header);
                auto& cs_program = GetCsRegs();
                cs_program.dim_x = dispatch_direct->dim_x;
                cs_program.dim_y = dispatch_direct->dim_y;
                cs_program.dim_z = dispatch_direct->dim_z;
                cs_program.dispatch_initiator = dispatch_direct->dispatch_initiator;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                                   cs_program);
                }
                if (!rasterizer || (cs_program.dispatch_initiator & 1) == 0) {
                    break;
                }
                PipeCheckpoint();
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker("gfx:{}:DispatchDirect", fmt::make_format_args(cmd_address),
                                        [&] { rasterizer->DispatchDirect(); });
                break;
            }
            case PM4ItOpcode::DispatchIndirect: {
                const auto* dispatch_indirect =
                    reinterpret_cast<const PM4CmdDispatchIndirect*>(header);
                auto& cs_program = GetCsRegs();
                const auto offset = dispatch_indirect->data_offset;
                const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                                   cs_program);
                }
                if (!rasterizer || (cs_program.dispatch_initiator & 1) == 0) {
                    break;
                }
                PipeCheckpoint();
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker(
                    "gfx:{}:DispatchIndirect", fmt::make_format_args(cmd_address),
                    [&] { rasterizer->DispatchIndirect(indirect_args_addr, offset, size); });
                break;
            }
            case PM4ItOpcode::NumInstances: {
                const auto* num_instances = reinterpret_cast<const PM4CmdDrawNumInstances*>(header);
                regs.num_instances.num_instances = num_instances->num_instances;
                reg_dirty.MarkField(regs, regs.num_instances);
                break;
            }
            case PM4ItOpcode::IndexBase: {
                const auto* index_base = reinterpret_cast<const PM4CmdDrawIndexBase*>(header);
                regs.index_base_address.base_addr_lo = index_base->addr_lo;
                regs.index_base_address.base_addr_hi = index_base->addr_hi;
                reg_dirty.MarkField(regs, regs.index_base_address);
                break;
            }
            case PM4ItOpcode::IndexBufferSize: {
                const auto* index_size = reinterpret_cast<const PM4CmdDrawIndexBufferSize*>(header);
                regs.num_indices = index_size->num_indices;
                reg_dirty.MarkField(regs, regs.num_indices);
                break;
            }
            case PM4ItOpcode::SetBase: {
                const auto* set_base = reinterpret_cast<const PM4CmdSetBase*>(header);
                ASSERT(set_base->base_index == PM4CmdSetBase::BaseIndex::DrawIndexIndirPatchTable);
                indirect_args_addr = set_base->Address<u64>();
                break;
            }
            case PM4ItOpcode::EventWrite: {
                const auto* event = reinterpret_cast<const PM4CmdEventWrite*>(header);
                LOG_DEBUG(Render, "Encountered EventWrite: event_type = {}, event_index = {}",
                          magic_enum::enum_name(event->event_type.Value()),
                          magic_enum::enum_name(event->event_index.Value()));
                if (event->event_type.Value() == EventType::SoVgtStreamoutFlush) {
                    // TODO: handle proper synchronization, for now signal that update is done
                    // immediately
                    regs.cp_strmout_cntl.offset_update_done = 1;
                    reg_dirty.MarkField(regs, regs.cp_strmout_cntl);
                } else if (event->event_index.Value() == EventIndex::ZpassDone) {
                    if (event->event_type.Value() == EventType::PixelPipeStatDump) {
                        static constexpr u64 OcclusionCounterValidMask = 0x8000000000000000ULL;
                        static constexpr u64 OcclusionCounterStep = 0x2FFFFFFULL;
                        u64* results = event->Address<u64*>();
                        for (s32 i = 0; i < num_counter_pairs; ++i, results += 2) {
                            *results = pixel_counter | OcclusionCounterValidMask;
                        }
                        pixel_counter += OcclusionCounterStep;
                    }
                }
                break;
            }
            case PM4ItOpcode::EventWriteEos: {
                const auto* event_eos = reinterpret_cast<const PM4CmdEventWriteEos*>(header);
                // sotcport: a label write is signalled by stage B in order, like EOP (~800/s in
                // the benchmark scene used to drain). Not a GDS store: it reads the GPU's GDS.
                if (rasterizer &&
                    event_eos->command == PM4CmdEventWriteEos::Command::SignalFence &&
                    !WritesUnreadCommands(cmd_begin, cmd_end,
                                          dcb.data() + header->type3.NumWords() + 1,
                                          reinterpret_cast<VAddr>(event_eos->Address()),
                                          sizeof(u32)) &&
                    rasterizer->PostEosFence(*event_eos)) {
                    break;
                }
                if (rasterizer) {
                    rasterizer->OnFence(Vulkan::DrawPipe::ReasonFenceEos);
                }
                event_eos->SignalFence([](void* address, u64 data, u32 num_bytes) {
                    auto* memory = Core::Memory::Instance();
                    ASSERT(memory->TryWriteBacking(address, &data, num_bytes));
                });
                if (event_eos->command == PM4CmdEventWriteEos::Command::GdsStore) {
                    ASSERT(event_eos->size == 1);
                    if (rasterizer) {
                        rasterizer->Finish();
                        const u32 value = rasterizer->ReadDataFromGds(event_eos->gds_index);
                        *event_eos->Address() = value;
                    }
                }
                break;
            }
            case PM4ItOpcode::EventWriteEop: {
                const auto* event_eop = reinterpret_cast<const PM4CmdEventWriteEop*>(header);
                // sotcport: with the two-stage draw pipeline stage B signals it in order. Not when
                // it writes into the command buffer being read: the game patches its own packets
                // with fences (see ReleaseMem in ProcessCompute).
                const auto fence_dst = reinterpret_cast<VAddr>(event_eop->Address<u8>());
                static const bool post_in_cmdbuf =
                    std::getenv("SOTC_PIPE_FENCE_IN_CMDBUF") != nullptr; // TEMP experiment
                // The game's labels sit right before the packet that writes them: a 32-bit one
                // ends where this packet starts (~600/s drained while counted as 8 bytes).
                const u32 fence_bytes =
                    event_eop->data_sel == DataSelect::Data32Low ? sizeof(u32) : sizeof(u64);
                const bool patches_commands =
                    !post_in_cmdbuf &&
                    WritesUnreadCommands(cmd_begin, cmd_end,
                                         dcb.data() + header->type3.NumWords() + 1, fence_dst,
                                         fence_bytes);
                if (rasterizer && !patches_commands && rasterizer->PostEopFence(*event_eop)) {
                    break;
                }
                if (rasterizer) {
                    rasterizer->OnFence(Vulkan::DrawPipe::ReasonFenceEopInCmdbuf);
                }
                SignalEopFence(*event_eop);
                break;
            }
            case PM4ItOpcode::DmaData: {
                const auto* dma_data = reinterpret_cast<const PM4DmaData*>(header);
                if (dma_data->dst_addr_lo == 0x3022C || !rasterizer) {
                    break;
                }
                ASSERT(dma_data->command.das == 0);
                if (dma_data->src_sel == DmaDataSrc::Data && dma_data->dst_sel == DmaDataDst::Gds) {
                    rasterizer->FillBuffer(dma_data->dst_addr_lo, dma_data->NumBytes(),
                                           dma_data->data, true);
                } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                            dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                           dma_data->dst_sel == DmaDataDst::Gds) {
                    rasterizer->CopyBuffer(dma_data->dst_addr_lo, dma_data->SrcAddress<VAddr>(),
                                           dma_data->NumBytes(), true, false);
                } else if (dma_data->src_sel == DmaDataSrc::Data &&
                           (dma_data->dst_sel == DmaDataDst::Memory ||
                            dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                    rasterizer->FillBuffer(dma_data->DstAddress<VAddr>(), dma_data->NumBytes(),
                                           dma_data->data, false);
                } else if (dma_data->src_sel == DmaDataSrc::Gds &&
                           (dma_data->dst_sel == DmaDataDst::Memory ||
                            dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                    rasterizer->CopyBuffer(dma_data->DstAddress<VAddr>(), dma_data->src_addr_lo,
                                           dma_data->NumBytes(), false, true);
                } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                            dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                           (dma_data->dst_sel == DmaDataDst::Memory ||
                            dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                    rasterizer->CopyBuffer(dma_data->DstAddress<VAddr>(),
                                           dma_data->SrcAddress<VAddr>(), dma_data->NumBytes(),
                                           false, false);
                } else {
                    UNREACHABLE_MSG("WriteData src_sel = {}, dst_sel = {}", u32(dma_data->src_sel),
                                    u32(dma_data->dst_sel));
                }
                break;
            }
            case PM4ItOpcode::WriteData: {
                const auto* write_data = reinterpret_cast<const PM4CmdWriteData*>(header);
                ASSERT(write_data->dst_sel.Value() == 2 || write_data->dst_sel.Value() == 5);
                const u32 data_size = (header->type3.count.Value() - 2) * 4;
                u64* address = write_data->Address<u64*>();
                diag::NoteMemoryAccess(diag::MemAccess::Label, reinterpret_cast<u64>(address),
                                       data_size); // sotcport (SOTC_MEMORY_TYPES)
                if (!write_data->wr_one_addr.Value()) {
                    if (rasterizer) {
                        rasterizer->OnFence(Vulkan::DrawPipe::ReasonFenceWriteData);
                    }
                    std::memcpy(address, write_data->data, data_size);
                } else {
                    UNREACHABLE();
                }
                break;
            }
            case PM4ItOpcode::CopyData: {
                const auto* copy_data = reinterpret_cast<const PM4CmdCopyData*>(header);
                LOG_WARNING(Render,
                            "unhandled IT_COPY_DATA src_sel = {}, dst_sel = {}, "
                            "count_sel = {}, wr_confirm = {}, engine_sel = {}",
                            u32(copy_data->src_sel.Value()), u32(copy_data->dst_sel.Value()),
                            copy_data->count_sel.Value(), copy_data->wr_confirm.Value(),
                            u32(copy_data->engine_sel.Value()));
                break;
            }
            case PM4ItOpcode::MemSemaphore: {
                const auto* mem_semaphore = reinterpret_cast<const PM4CmdMemSemaphore*>(header);
                if (mem_semaphore->IsSignaling()) {
                    mem_semaphore->Signal();
                } else {
                    while (!mem_semaphore->Signaled()) {
                        YIELD_GFX();
                    }
                    mem_semaphore->Decrement();
                }
                break;
            }
            case PM4ItOpcode::AcquireMem: {
                // const auto* acquire_mem = reinterpret_cast<PM4CmdAcquireMem*>(header);
                break;
            }
            case PM4ItOpcode::Rewind: {
                if (!rasterizer) {
                    break;
                }
                const PM4CmdRewind* rewind = reinterpret_cast<const PM4CmdRewind*>(header);
                while (!rewind->Valid()) {
                    YIELD_GFX();
                }
                break;
            }
            case PM4ItOpcode::WaitRegMem: {
                const auto* wait_reg_mem = reinterpret_cast<const PM4CmdWaitRegMem*>(header);
                // ASSERT(wait_reg_mem->engine.Value() == PM4CmdWaitRegMem::Engine::Me);
                // Optimization: VO label waits are special because the emulator
                // will write to the label when presentation is finished. So if
                // there are no other submits to yield to we can sleep the thread
                // instead and allow other tasks to run.
                const u64* wait_addr = wait_reg_mem->Address<u64*>();
                if (vo_port->IsVoLabel(wait_addr) &&
                    num_submits == mapped_queues[GfxQueueId].submits.size()) {
                    vo_port->WaitVoLabel([&] { return wait_reg_mem->Test(regs.reg_array); });
                    break;
                }
                while (!WaitRegMemMet(*wait_reg_mem)) {
                    YIELD_GFX();
                }
                break;
            }
            case PM4ItOpcode::IndirectBuffer: {
                const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
                diag::NoteMemoryAccess(diag::MemAccess::IndirectCommands, // sotcport
                                       reinterpret_cast<u64>(indirect_buffer->Address<const u32>()),
                                       u64{indirect_buffer->ib_size} * sizeof(u32));
                auto task = ProcessGraphics(
                    {indirect_buffer->Address<const u32>(), indirect_buffer->ib_size}, {});
                RESUME_GFX(task);

                while (!task.handle.done()) {
                    YIELD_GFX();
                    RESUME_GFX(task);
                }
                break;
            }
            case PM4ItOpcode::IncrementDeCounter: {
                ++cblock.de_count;
                break;
            }
            case PM4ItOpcode::WaitOnCeCounter: {
                while (cblock.ce_count <= cblock.de_count && !ce_task.handle.done()) {
                    RESUME_GFX(ce_task);
                }
                break;
            }
            case PM4ItOpcode::PfpSyncMe: {
                break;
            }
            case PM4ItOpcode::StrmoutBufferUpdate: {
                const auto* strmout = reinterpret_cast<const PM4CmdStrmoutBufferUpdate*>(header);
                LOG_WARNING(Render_Vulkan,
                            "Unimplemented IT_STRMOUT_BUFFER_UPDATE, update_memory = {}, "
                            "source_select = {}, buffer_select = {}",
                            strmout->update_memory.Value(),
                            magic_enum::enum_name(strmout->source_select.Value()),
                            strmout->buffer_select.Value());
                break;
            }
            case PM4ItOpcode::GetLodStats: {
                LOG_WARNING_ONCE_SHADOW(Render_Vulkan, "Unimplemented IT_GET_LOD_STATS"); // sotcport: once
                break;
            }
            case PM4ItOpcode::CondExec: {
                const auto* cond_exec = reinterpret_cast<const PM4CmdCondExec*>(header);
                if (cond_exec->command.Value() != 0) {
                    LOG_WARNING(Render, "IT_COND_EXEC used a reserved command");
                }
                const auto skip = *cond_exec->Address() == false;
                if (skip) {
                    dcb = NextPacket(dcb,
                                     header->type3.NumWords() + 1 + cond_exec->exec_count.Value());
                    continue;
                }
                break;
            }
            default:
                UNREACHABLE_MSG("Unknown PM4 type 3 opcode {:#x} with count {}",
                                static_cast<u32>(opcode), count);
            }
            dcb = NextPacket(dcb, header->type3.NumWords() + 1);
            break;
        }
    }

    if (ce_task.handle) {
        while (!ce_task.handle.done()) {
            RESUME_GFX(ce_task);
        }
        ce_task.handle.destroy();
    }

    FIBER_EXIT;
}

template <bool is_indirect>
Liverpool::Task Liverpool::ProcessCompute(std::span<const u32> acb, u32 vqid) {
    FIBER_ENTER(acb_task_name[vqid]);
    auto& queue = asc_queues[{vqid}];

    struct IndirectPatch {
        const PM4Header* header;
        VAddr indirect_addr;
    };
    boost::container::small_vector<IndirectPatch, 4> indirect_patches;

    auto base_addr = reinterpret_cast<VAddr>(acb.data());
    size_t acb_size = acb.size_bytes();
    // sotcport (two-stage draw pipeline): see ProcessGraphics.
    if (rasterizer && rasterizer->PendingWriteOverlaps(base_addr, acb_size)) {
        rasterizer->DrainDrawPipe(Vulkan::DrawPipe::ReasonCommandsPending);
    }
    while (!acb.empty()) {
        ProcessCommands();

        auto* header = reinterpret_cast<const PM4Header*>(acb.data());
        u32 next_dw_off = header->type3.NumWords() + 1;

        // If we have a buffered packet, use it.
        if (queue.tmp_dwords > 0) [[unlikely]] {
            header = reinterpret_cast<const PM4Header*>(queue.tmp_packet.data());
            next_dw_off = header->type3.NumWords() + 1 - queue.tmp_dwords;
            std::memcpy(queue.tmp_packet.data() + queue.tmp_dwords, acb.data(),
                        next_dw_off * sizeof(u32));
            queue.tmp_dwords = 0;
        }

        // If the packet is split across ring boundary, buffer until next submission
        if (next_dw_off > acb.size()) [[unlikely]] {
            std::memcpy(queue.tmp_packet.data(), acb.data(), acb.size_bytes());
            queue.tmp_dwords = acb.size();
            if constexpr (!is_indirect) {
                AdvanceReadPointer(queue.read_addr, acb.size(), queue.ring_size_dw);
            }
            break;
        }

        // A packet split across the ring's end is not reported (it was buffered above).
        if (g_packet_observer && queue.tmp_dwords == 0 &&
            header == reinterpret_cast<const PM4Header*>(acb.data())) [[unlikely]] {
            g_packet_observer(acb.data(), header->type == 2 ? 1 : next_dw_off,
                              PacketQueue::Compute);
        }

        if (header->type == 2) {
            // Type-2 packet are used for padding purposes
            next_dw_off = 1;
            acb = NextPacket(acb, next_dw_off);
            if constexpr (!is_indirect) {
                AdvanceReadPointer(queue.read_addr, next_dw_off, queue.ring_size_dw);
            }
            continue;
        }

        if (header->type != 3) {
            // No other types of packets were spotted so far
            UNREACHABLE_MSG("Invalid PM4 type {}", header->type.Value());
        }

        const PM4ItOpcode opcode = header->type3.opcode;
        // sotcport: two-stage draw pipeline: compute queue packets that touch the caches stage B
        // owns wait for it to run dry. Register writes and dispatches go through the ring like
        // the graphics queue's (a dispatch packet carries this queue's cs_state); waits drain
        // only when needed (WaitRegMemMet); the rest has no effect here.
        if (rasterizer && !PipelinedComputeOpcode(opcode)) {
            rasterizer->DrainDrawPipe(Vulkan::DrawPipe::ReasonComputeOpcode +
                                      static_cast<u32>(opcode));
        }

        const auto* it_body = reinterpret_cast<const u32*>(header) + 1;
        switch (opcode) {
        case PM4ItOpcode::Nop: {
            const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
            break;
        }
        case PM4ItOpcode::IndirectBuffer: {
            const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
            diag::NoteMemoryAccess(diag::MemAccess::IndirectCommands, // sotcport
                                   reinterpret_cast<u64>(indirect_buffer->Address<const u32>()),
                                   u64{indirect_buffer->ib_size} * sizeof(u32));
            // sotcport: a fence or shader queued for stage B may still write into it.
            const auto ib_address = reinterpret_cast<VAddr>(indirect_buffer->Address<const u32>());
            if (rasterizer && rasterizer->PendingWriteOverlaps(
                                  ib_address, indirect_buffer->ib_size * sizeof(u32))) {
                rasterizer->DrainDrawPipe(Vulkan::DrawPipe::ReasonComputeOpcode +
                                          static_cast<u32>(opcode));
            }
            auto task = ProcessCompute<true>(
                {indirect_buffer->Address<const u32>(), indirect_buffer->ib_size}, vqid);
            RESUME_ASC(task, vqid);

            while (!task.handle.done()) {
                YIELD_ASC(vqid);
                RESUME_ASC(task, vqid);
            }
            break;
        }
        case PM4ItOpcode::DmaData: {
            const auto* dma_data = reinterpret_cast<const PM4DmaData*>(header);
            if (dma_data->dst_addr_lo == 0x3022C || !rasterizer) {
                break;
            }
            ASSERT(dma_data->command.das == 0);
            if (dma_data->src_sel == DmaDataSrc::Data && dma_data->dst_sel == DmaDataDst::Gds) {
                rasterizer->FillBuffer(dma_data->dst_addr_lo, dma_data->NumBytes(), dma_data->data,
                                       true);
            } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                        dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                       dma_data->dst_sel == DmaDataDst::Gds) {
                rasterizer->CopyBuffer(dma_data->dst_addr_lo, dma_data->SrcAddress<VAddr>(),
                                       dma_data->NumBytes(), true, false);
            } else if (dma_data->src_sel == DmaDataSrc::Data &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                rasterizer->FillBuffer(dma_data->DstAddress<VAddr>(), dma_data->NumBytes(),
                                       dma_data->data, false);
            } else if (dma_data->src_sel == DmaDataSrc::Gds &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                rasterizer->CopyBuffer(dma_data->DstAddress<VAddr>(), dma_data->src_addr_lo,
                                       dma_data->NumBytes(), false, true);
            } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                        dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                const u32 num_bytes = dma_data->NumBytes();
                const VAddr src_addr = dma_data->SrcAddress<VAddr>();
                const VAddr dst_addr = dma_data->DstAddress<VAddr>();
                const PM4Header* header =
                    reinterpret_cast<const PM4Header*>(dst_addr - sizeof(PM4Header));
                if (dst_addr >= base_addr && dst_addr < base_addr + acb_size &&
                    num_bytes == sizeof(PM4CmdDispatchIndirect::GroupDimensions) &&
                    header->type == 3 && header->type3.opcode == PM4ItOpcode::DispatchDirect) {
                    indirect_patches.emplace_back(header, src_addr);
                } else {
                    rasterizer->CopyBuffer(dst_addr, src_addr, num_bytes, false, false);
                }
            } else {
                UNREACHABLE_MSG("WriteData src_sel = {}, dst_sel = {}", u32(dma_data->src_sel),
                                u32(dma_data->dst_sel));
            }
            break;
        }
        case PM4ItOpcode::AcquireMem: {
            break;
        }
        case PM4ItOpcode::Rewind: {
            if (!rasterizer) {
                break;
            }
            const PM4CmdRewind* rewind = reinterpret_cast<const PM4CmdRewind*>(header);
            while (!rewind->Valid()) {
                YIELD_ASC(vqid);
            }
            break;
        }
        case PM4ItOpcode::SetShReg: {
            const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
            const auto set_size = (header->type3.NumWords() - 1) * sizeof(u32);

            if (set_data->reg_offset >= 0x200 &&
                set_data->reg_offset <= (0x200 + sizeof(ComputeProgram) / 4)) {
                ASSERT(set_size <= sizeof(ComputeProgram));
                auto* addr = reinterpret_cast<u32*>(&mapped_queues[vqid + 1].cs_state) +
                             (set_data->reg_offset - 0x200);
                std::memcpy(addr, header + 2, set_size);
            } else {
                std::memcpy(&regs.reg_array[Regs::ShRegWordOffset + set_data->reg_offset],
                            header + 2, set_size);
                reg_dirty.Mark(Regs::ShRegWordOffset + set_data->reg_offset,
                               u32(set_size / sizeof(u32)));
            }
            break;
        }
        case PM4ItOpcode::SetQueueReg: {
            const auto* set_data = reinterpret_cast<const PM4CmdSetQueueReg*>(header);
            LOG_WARNING(Render, "Encountered compute SetQueueReg: vqid = {}, reg_offset = {:#x}",
                        set_data->vqid.Value(), set_data->reg_offset.Value());
            break;
        }
        case PM4ItOpcode::DispatchDirect: {
            const auto* dispatch_direct = reinterpret_cast<const PM4CmdDispatchDirect*>(header);
            if (auto it = std::ranges::find(indirect_patches, header, &IndirectPatch::header);
                it != indirect_patches.end()) {
                const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
                rasterizer->DispatchIndirect(it->indirect_addr, 0, size);
                break;
            }
            auto& cs_program = GetCsRegs();
            cs_program.dim_x = dispatch_direct->dim_x;
            cs_program.dim_y = dispatch_direct->dim_y;
            cs_program.dim_z = dispatch_direct->dim_z;
            cs_program.dispatch_initiator = dispatch_direct->dispatch_initiator;
            if (DebugState.DumpingCurrentReg()) {
                DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                               cs_program);
            }
            if (!rasterizer || (cs_program.dispatch_initiator & 1) == 0) {
                break;
            }
            const auto cmd_address = reinterpret_cast<const void*>(header);
            rasterizer->ScopeMarker("asc[{}]:{}:DispatchDirect",
                                    fmt::make_format_args(vqid, cmd_address),
                                    [&] { rasterizer->DispatchDirect(); });
            break;
        }
        case PM4ItOpcode::DispatchIndirect: {
            const auto* dispatch_indirect =
                reinterpret_cast<const PM4CmdDispatchIndirectMec*>(header);
            auto& cs_program = GetCsRegs();
            const auto ib_address = dispatch_indirect->Address<VAddr>();
            const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
            if (DebugState.DumpingCurrentReg()) {
                DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                               cs_program);
            }
            if (!rasterizer || (cs_program.dispatch_initiator & 1) == 0) {
                break;
            }
            const auto cmd_address = reinterpret_cast<const void*>(header);
            rasterizer->ScopeMarker("asc[{}]:{}:DispatchIndirect",
                                    fmt::make_format_args(vqid, cmd_address),
                                    [&] { rasterizer->DispatchIndirect(ib_address, 0, size); });
            break;
        }
        case PM4ItOpcode::WriteData: {
            const auto* write_data = reinterpret_cast<const PM4CmdWriteData*>(header);
            ASSERT(write_data->dst_sel.Value() == 2 || write_data->dst_sel.Value() == 5);
            const u32 data_size = (header->type3.count.Value() - 2) * 4;
            diag::NoteMemoryAccess(diag::MemAccess::Label, write_data->Address<u64>(),
                                   data_size); // sotcport (SOTC_MEMORY_TYPES)
            if (!write_data->wr_one_addr.Value()) {
                if (rasterizer) {
                    rasterizer->OnFence(Vulkan::DrawPipe::ReasonFenceAscWriteData);
                }
                std::memcpy(write_data->Address<void*>(), write_data->data, data_size);
            } else {
                UNREACHABLE();
            }
            break;
        }
        case PM4ItOpcode::MemSemaphore: {
            const auto* mem_semaphore = reinterpret_cast<const PM4CmdMemSemaphore*>(header);
            if (mem_semaphore->IsSignaling()) {
                mem_semaphore->Signal();
            } else {
                while (!mem_semaphore->Signaled()) {
                    YIELD_ASC(vqid);
                }
                mem_semaphore->Decrement();
            }
            break;
        }
        case PM4ItOpcode::WaitRegMem: {
            const auto* wait_reg_mem = reinterpret_cast<const PM4CmdWaitRegMem*>(header);
            ASSERT(wait_reg_mem->engine.Value() == PM4CmdWaitRegMem::Engine::Me);
            while (!WaitRegMemMet(*wait_reg_mem)) {
                YIELD_ASC(vqid);
            }
            break;
        }
        case PM4ItOpcode::ReleaseMem: {
            const auto* release_mem = reinterpret_cast<const PM4CmdReleaseMem*>(header);
            // sotcport: with the two-stage draw pipeline stage B signals it in order. Not when it
            // writes into the command buffer being read: the game patches its own packets this
            // way (a 1 written a few packets ahead, seen ~20 times at startup), and they must be
            // patched before this thread reads them.
            const VAddr fence_dst = release_mem->Address<VAddr>();
            // SOTC_PIPE_FENCE_IN_CMDBUF=1 posts these too (TEMP experiment: the labels written
            // there are what the queues' WAIT_REG_MEMs poll; see OTIMIZACAO notes).
            static const bool post_in_cmdbuf = std::getenv("SOTC_PIPE_FENCE_IN_CMDBUF") != nullptr;
            const bool patches_commands =
                !post_in_cmdbuf &&
                WritesUnreadCommands(base_addr, base_addr + acb_size,
                                     acb.data() + header->type3.NumWords() + 1, fence_dst,
                                     release_mem->data_sel == DataSelect::Data32Low
                                         ? sizeof(u32)
                                         : sizeof(u64));
            if (rasterizer && !patches_commands &&
                rasterizer->PostReleaseMem(*release_mem, queue.pipe_id)) {
                break;
            }
            if (rasterizer) {
                rasterizer->OnFence(Vulkan::DrawPipe::ReasonFenceReleaseInCmdbuf);
            }
            SignalReleaseMem(*release_mem, queue.pipe_id);
            break;
        }
        case PM4ItOpcode::EventWrite: {
            // const auto* event = reinterpret_cast<const PM4CmdEventWrite*>(header);
            break;
        }
        default:
            UNREACHABLE_MSG("Unknown PM4 type 3 opcode {:#x} with count {}",
                            static_cast<u32>(opcode), header->type3.NumWords());
        }

        acb = NextPacket(acb, next_dw_off);

        if constexpr (!is_indirect) {
            AdvanceReadPointer(queue.read_addr, next_dw_off, queue.ring_size_dw);
        }
    }

    FIBER_EXIT;
}

Liverpool::CmdBuffer Liverpool::CopyCmdBuffers(std::span<const u32> dcb, std::span<const u32> ccb) {
    auto& queue = mapped_queues[GfxQueueId];
    ASSERT_MSG(queue.dcb_buffer.capacity() >= queue.dcb_buffer_offset + dcb.size(),
               "dcb copy buffer out of reserved space");
    ASSERT_MSG(queue.ccb_buffer.capacity() >= queue.ccb_buffer_offset + ccb.size(),
               "ccb copy buffer out of reserved space");

    queue.dcb_buffer.resize(
        std::max(queue.dcb_buffer.size(), queue.dcb_buffer_offset + dcb.size()));
    queue.ccb_buffer.resize(
        std::max(queue.ccb_buffer.size(), queue.ccb_buffer_offset + ccb.size()));

    const u32 prev_dcb_buffer_offset = queue.dcb_buffer_offset;
    const u32 prev_ccb_buffer_offset = queue.ccb_buffer_offset;
    if (!dcb.empty()) {
        std::memcpy(queue.dcb_buffer.data() + queue.dcb_buffer_offset, dcb.data(),
                    dcb.size_bytes());
        queue.dcb_buffer_offset += dcb.size();
        dcb = std::span<const u32>{queue.dcb_buffer.begin() + prev_dcb_buffer_offset,
                                   queue.dcb_buffer.begin() + queue.dcb_buffer_offset};
    }

    if (!ccb.empty()) {
        std::memcpy(queue.ccb_buffer.data() + queue.ccb_buffer_offset, ccb.data(),
                    ccb.size_bytes());
        queue.ccb_buffer_offset += ccb.size();
        ccb = std::span<const u32>{queue.ccb_buffer.begin() + prev_ccb_buffer_offset,
                                   queue.ccb_buffer.begin() + queue.ccb_buffer_offset};
    }

    return std::make_pair(dcb, ccb);
}

void Liverpool::SubmitGfx(std::span<const u32> dcb, std::span<const u32> ccb) {
    perf::Mark('S', dcb.size()); // TEMP timeline: the game submits
    auto& queue = mapped_queues[GfxQueueId];
    // sotcport (SOTC_MEMORY_TYPES): where the command buffers live.
    diag::NoteMemoryAccess(diag::MemAccess::GfxCommands, reinterpret_cast<u64>(dcb.data()),
                           dcb.size_bytes());
    if (!ccb.empty()) {
        diag::NoteMemoryAccess(diag::MemAccess::CeCommands, reinterpret_cast<u64>(ccb.data()),
                               ccb.size_bytes());
    }

    if (EmulatorSettings.IsCopyGpuBuffers()) {
        std::tie(dcb, ccb) = CopyCmdBuffers(dcb, ccb);
    }

    auto task = ProcessGraphics(dcb, ccb);
    {
        std::scoped_lock lock{queue.m_access};
        queue.submits.emplace(task.handle);
    }

    std::scoped_lock lk{submit_mutex};
    ++num_submits;
    submit_cv.notify_one();
}

void Liverpool::SubmitAsc(u32 gnm_vqid, std::span<const u32> acb) {
    ASSERT_MSG(gnm_vqid > 0 && gnm_vqid < NumTotalQueues, "Invalid virtual ASC queue index");
    auto& queue = mapped_queues[gnm_vqid];
    diag::NoteMemoryAccess(diag::MemAccess::ComputeCommands, reinterpret_cast<u64>(acb.data()),
                           acb.size_bytes()); // sotcport (SOTC_MEMORY_TYPES)

    const auto vqid = gnm_vqid - 1;
    const auto& task = ProcessCompute(acb, vqid);
    {
        std::scoped_lock lock{queue.m_access};
        queue.submits.emplace(task.handle);
    }

    std::scoped_lock lk{submit_mutex};
    num_mapped_queues = std::max(num_mapped_queues, gnm_vqid + 1);
    ++num_submits;
    submit_cv.notify_one();
}

} // namespace AmdGpu
