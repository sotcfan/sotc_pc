#include "diag/packet_stats.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/log.h"
#include "gpu/gpu.h"

namespace diag {
namespace {

// PM4 type-3 opcodes (AMD's pm4 packet headers, as in shadPS4's pm4_opcodes.h).
enum : uint32_t {
    OpNop = 0x10, OpSetBase = 0x11, OpClearState = 0x12, OpIndexBufferSize = 0x13,
    OpDispatchDirect = 0x15, OpDispatchIndirect = 0x16, OpSetPredication = 0x20,
    OpCondExec = 0x22, OpIndexBase = 0x26, OpDrawIndex2 = 0x27, OpContextControl = 0x28,
    OpIndexType = 0x2A, OpDrawIndirect = 0x24, OpDrawIndexIndirect = 0x25,
    OpDrawIndirectMulti = 0x2C, OpDrawIndexAuto = 0x2D, OpNumInstances = 0x2F,
    OpDrawIndexMultiAuto = 0x30, OpIndirectBufferConst = 0x33, OpStrmoutBufferUpdate = 0x34,
    OpDrawIndexOffset2 = 0x35, OpDrawIndexIndirectMulti = 0x38, OpWriteData = 0x37,
    OpMemSemaphore = 0x39, OpWaitRegMem = 0x3C, OpIndirectBuffer = 0x3F, OpCopyData = 0x40,
    OpPfpSyncMe = 0x42, OpSurfaceSync = 0x43, OpEventWrite = 0x46, OpEventWriteEop = 0x47,
    OpEventWriteEos = 0x48, OpReleaseMem = 0x49, OpDmaData = 0x50, OpAcquireMem = 0x58,
    OpRewind = 0x59, OpSetConfigReg = 0x68, OpSetContextReg = 0x69, OpSetShReg = 0x76,
    OpSetQueueReg = 0x78, OpSetUconfigReg = 0x79, OpWriteConstRam = 0x81, OpDumpConstRam = 0x83,
    OpIncrementCeCounter = 0x84, OpIncrementDeCounter = 0x85, OpWaitOnCeCounter = 0x86,
    OpWaitOnDeCounterDiff = 0x88,
};

constexpr int NumQueues = 3;
std::array<std::array<std::atomic<uint64_t>, 256>, NumQueues> g_counts{};
// Estimated drains of the graphics queue: a packet the second stage cannot take, coming after
// at least one draw handed to it. "naive" = the first version (only register writes, direct
// draws and no-ops pipelined); "final" = after the planned work (dispatches, fences, WriteData,
// DmaData and indirect draws pipelined too).
std::atomic<uint64_t> g_drains_naive{0}, g_drains_final{0};
std::array<std::atomic<uint64_t>, 256> g_naive_cause{}, g_final_cause{};
bool g_pending_naive = false, g_pending_final = false; // graphics thread only

// Who last wrote each guest address that a fence or WriteData targets: (queue + 1) << 8 | opcode.
// WaitRegMem on the graphics queue is classified by it: a value written by a graphics fence can be
// waited for in order on the second stage (no drain); one from a compute queue or the CPU cannot.
std::unordered_map<uint64_t, uint32_t> g_last_writer; // GPU command thread only
std::array<std::atomic<uint64_t>, 4> g_wait_kind{};   // gfx fence, compute fence, gfx WriteData, unknown
std::atomic<uint64_t> g_wait_register{0};

void NoteWrite(uint64_t address, uint32_t queue, uint32_t op) {
    if (g_last_writer.size() > (1u << 16)) {
        g_last_writer.clear();
    }
    g_last_writer[address & ~3ull] = (queue + 1) << 8 | op;
}

void ClassifyWait(const uint32_t* packet) {
    if (((packet[1] >> 4) & 1) == 0) { // mem_space: register
        g_wait_register.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const uint64_t address = (packet[2] & ~3u) | uint64_t(packet[3]) << 32;
    const auto it = g_last_writer.find(address);
    int kind = 3;
    if (it != g_last_writer.end()) {
        const uint32_t queue = (it->second >> 8) - 1;
        const uint32_t op = it->second & 0xFF;
        kind = queue == 0 ? (op == 0x37 ? 2 : 0) : 1;
    }
    g_wait_kind[kind].fetch_add(1, std::memory_order_relaxed);
}

bool IsDirectDraw(uint32_t op) {
    return op == OpDrawIndex2 || op == OpDrawIndexAuto || op == OpDrawIndexOffset2 ||
           op == OpDrawIndexMultiAuto;
}
bool IsRegisterOrNop(uint32_t op) {
    return op == OpNop || op == OpSetContextReg || op == OpSetShReg || op == OpSetUconfigReg ||
           op == OpSetConfigReg || op == OpSetQueueReg || op == OpIndexType ||
           op == OpNumInstances || op == OpIndexBase || op == OpIndexBufferSize ||
           op == OpContextControl || op == OpClearState || op == OpSetBase ||
           op == OpIncrementDeCounter || op == OpWaitOnCeCounter;
}
bool PipelinedFinally(uint32_t op) {
    return op == OpDispatchDirect || op == OpDispatchIndirect || op == OpEventWrite ||
           op == OpEventWriteEop || op == OpEventWriteEos || op == OpReleaseMem ||
           op == OpWriteData || op == OpDmaData || op == OpDrawIndirect ||
           op == OpDrawIndexIndirect || op == OpDrawIndirectMulti ||
           op == OpDrawIndexIndirectMulti || op == OpAcquireMem;
}

void OnPacket(const uint32_t* packet, uint32_t, uint32_t queue) {
    const uint32_t header = packet[0];
    if (header >> 30 != 3 || queue >= NumQueues) {
        return;
    }
    const uint32_t op = (header >> 8) & 0xFF;
    g_counts[queue][op].fetch_add(1, std::memory_order_relaxed);
    // Fence and memory write targets (layouts as in shadPS4's pm4_cmds.h).
    if (op == 0x47 || op == 0x48) { // EVENT_WRITE_EOP / EOS: address in dwords 2-3
        NoteWrite(packet[2] | uint64_t(packet[3] & 0xFFFF) << 32, queue, op);
    } else if (op == 0x49) { // RELEASE_MEM: address in dwords 3-4
        NoteWrite(packet[3] | uint64_t(packet[4]) << 32, queue, op);
    } else if (op == 0x37) { // WRITE_DATA: destination in dwords 2-3
        NoteWrite(packet[2] | uint64_t(packet[3]) << 32, queue, op);
    }
    if (queue != 0) {
        return;
    }
    if (op == 0x3C) {
        ClassifyWait(packet);
    }
    const bool draw = IsDirectDraw(op);
    if (draw) {
        g_pending_naive = g_pending_final = true;
        return;
    }
    if (IsRegisterOrNop(op)) {
        return;
    }
    if (g_pending_naive) {
        g_drains_naive.fetch_add(1, std::memory_order_relaxed);
        g_naive_cause[op].fetch_add(1, std::memory_order_relaxed);
        g_pending_naive = false;
    }
    if (PipelinedFinally(op)) {
        if (op != OpWriteData && op != OpDmaData && op != OpEventWriteEop &&
            op != OpEventWriteEos && op != OpReleaseMem && op != OpEventWrite &&
            op != OpAcquireMem) {
            g_pending_final = true; // dispatches and indirect draws go to the second stage too
        }
        return;
    }
    if (g_pending_final) {
        g_drains_final.fetch_add(1, std::memory_order_relaxed);
        g_final_cause[op].fetch_add(1, std::memory_order_relaxed);
        g_pending_final = false;
    }
}

std::string Top(std::array<std::atomic<uint64_t>, 256>& counts, double seconds, int n) {
    std::vector<std::pair<uint64_t, int>> list;
    for (int op = 0; op < 256; op++) {
        if (const uint64_t c = counts[op].exchange(0)) {
            list.emplace_back(c, op);
        }
    }
    std::ranges::sort(list, std::greater{});
    std::string text;
    for (int i = 0; i < std::min<int>(n, list.size()); i++) {
        text += std::format(" {:#04x}:{:.0f}/s", list[i].second, list[i].first / seconds);
    }
    return text;
}

void ReportLoop() {
    constexpr double Seconds = 5.0;
    for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        static constexpr const char* Names[NumQueues] = {"graphics", "constant", "compute"};
        for (int q = 0; q < NumQueues; q++) {
            LOG_INFO("Packets", "{}:{}", Names[q], Top(g_counts[q], Seconds, 24));
        }
        LOG_INFO("Packets",
                 "graphics WaitRegMem: on graphics fences {:.0f}/s, on compute fences {:.0f}/s, "
                 "on graphics WriteData {:.0f}/s, unknown writer (CPU?) {:.0f}/s, registers {:.0f}/s",
                 g_wait_kind[0].exchange(0) / Seconds, g_wait_kind[1].exchange(0) / Seconds,
                 g_wait_kind[2].exchange(0) / Seconds, g_wait_kind[3].exchange(0) / Seconds,
                 g_wait_register.exchange(0) / Seconds);
        LOG_INFO("Packets", "drains naive {:.0f}/s:{} | final {:.0f}/s:{}",
                 g_drains_naive.exchange(0) / Seconds, Top(g_naive_cause, Seconds, 8),
                 g_drains_final.exchange(0) / Seconds, Top(g_final_cause, Seconds, 8));
    }
}

} // namespace

void InstallPacketStats() {
    const char* env = std::getenv("SOTC_PACKET_STATS");
    if (!env || *env != '1') {
        return;
    }
    gpu::SetPacketObserver(OnPacket);
    std::thread{ReportLoop}.detach();
    LOG_INFO("Packets", "counting PM4 packets (report every 5 s)");
}

} // namespace diag
