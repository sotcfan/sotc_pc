// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <queue>

#include "common/interval_set.h"
#include "common/unique_function.h"
#include "video_core/amdgpu/regs_color.h"
#include "video_core/amdgpu/regs_primitive.h"
#include "video_core/renderer_vulkan/vk_resource_pool.h"
#include "video_core/renderer_vulkan/vk_semaphore.h"
#include "vulkan/vulkan.hpp"

namespace tracy {
class VkCtxScope;
}

namespace Vulkan {

class Instance;

struct RenderAttachment {
    vk::ImageView image_view;
    vk::ImageLayout image_layout;
    std::array<u32, 4> clear_value;
    union {
        u32 is_clear;
        struct {
            bool has_depth;
            bool depth_clear;
            bool has_stencil;
            bool stencil_clear;
        };
    };
};
static_assert(std::has_unique_object_representations_v<RenderAttachment>);

struct RenderState {
    std::array<RenderAttachment, 8> color_attachments;
    RenderAttachment depth_stencil_attachment;
    u16 width;
    u16 height;
    u16 num_layers;
    u16 num_color_attachments;

    bool operator==(const RenderState& other) const noexcept {
        return std::memcmp(this, &other, sizeof(RenderState)) == 0;
    }
};
static_assert(std::has_unique_object_representations_v<RenderState>);

struct SubmitInfo {
    std::array<vk::Semaphore, 4> wait_semas;
    std::array<u64, 4> wait_ticks;
    std::array<vk::Semaphore, 4> signal_semas;
    std::array<u64, 4> signal_ticks;
    vk::Fence fence;
    u32 num_wait_semas;
    u32 num_signal_semas;

    void AddWait(vk::Semaphore semaphore, u64 tick = 1) {
        wait_semas[num_wait_semas] = semaphore;
        wait_ticks[num_wait_semas++] = tick;
    }

    void AddSignal(vk::Semaphore semaphore, u64 tick = 1) {
        signal_semas[num_signal_semas] = semaphore;
        signal_ticks[num_signal_semas++] = tick;
    }

    void AddSignal(vk::Fence fence) {
        this->fence = fence;
    }
};

using Viewports = boost::container::static_vector<vk::Viewport, AmdGpu::NUM_VIEWPORTS>;
using Scissors = boost::container::static_vector<vk::Rect2D, AmdGpu::NUM_VIEWPORTS>;
using ColorWriteMasks = std::array<vk::ColorComponentFlags, AmdGpu::NUM_COLOR_BUFFERS>;
struct StencilOps {
    vk::StencilOp fail_op{};
    vk::StencilOp pass_op{};
    vk::StencilOp depth_fail_op{};
    vk::CompareOp compare_op{};

    bool operator==(const StencilOps& other) const {
        return fail_op == other.fail_op && pass_op == other.pass_op &&
               depth_fail_op == other.depth_fail_op && compare_op == other.compare_op;
    }
};
struct DynamicState {
    struct {
        bool viewports : 1;
        bool scissors : 1;

        bool depth_test_enabled : 1;
        bool depth_write_enabled : 1;
        bool depth_compare_op : 1;

        bool depth_bounds_test_enabled : 1;
        bool depth_bounds : 1;

        bool depth_bias_enabled : 1;
        bool depth_bias : 1;

        bool stencil_test_enabled : 1;
        bool stencil_front_ops : 1;
        bool stencil_front_reference : 1;
        bool stencil_front_write_mask : 1;
        bool stencil_front_compare_mask : 1;
        bool stencil_back_ops : 1;
        bool stencil_back_reference : 1;
        bool stencil_back_write_mask : 1;
        bool stencil_back_compare_mask : 1;

        bool primitive_restart_enable : 1;
        bool rasterizer_discard_enable : 1;
        bool cull_mode : 1;
        bool front_face : 1;

        bool blend_constants : 1;
        bool color_write_masks : 1;
        bool line_width : 1;
        bool feedback_loop_enabled : 1;
    } dirty_state{};

    Viewports viewports{};
    Scissors scissors{};

    bool depth_test_enabled{};
    bool depth_write_enabled{};
    vk::CompareOp depth_compare_op{};

    bool depth_bounds_test_enabled{};
    float depth_bounds_min{};
    float depth_bounds_max{};

    bool depth_bias_enabled{};
    float depth_bias_constant{};
    float depth_bias_clamp{};
    float depth_bias_slope{};

    bool stencil_test_enabled{};
    StencilOps stencil_front_ops{};
    u32 stencil_front_reference{};
    u32 stencil_front_write_mask{};
    u32 stencil_front_compare_mask{};
    StencilOps stencil_back_ops{};
    u32 stencil_back_reference{};
    u32 stencil_back_write_mask{};
    u32 stencil_back_compare_mask{};

    bool primitive_restart_enable{};
    bool rasterizer_discard_enable{};
    vk::CullModeFlags cull_mode{};
    vk::FrontFace front_face{};

    std::array<float, 4> blend_constants{};
    ColorWriteMasks color_write_masks{};
    float line_width{};
    bool feedback_loop_enabled{};

    /// Commits the dynamic state to the provided command buffer.
    void Commit(const Instance& instance, const vk::CommandBuffer& cmdbuf);

    /// Invalidates all dynamic state to be flushed into the next command buffer.
    void Invalidate() {
        std::memset(&dirty_state, 0xFF, sizeof(dirty_state));
    }

    /// sotcport: whether Commit() would record anything (for threaded recording, which
    /// commits a copy on the recording thread and then clears the flags here).
    [[nodiscard]] bool AnyDirty() const {
        decltype(dirty_state) clean{};
        return std::memcmp(&dirty_state, &clean, sizeof(dirty_state)) != 0;
    }
    void ClearDirty() {
        dirty_state = {};
    }

    void SetViewports(const Viewports& viewports_) {
        if (!std::ranges::equal(viewports, viewports_)) {
            viewports = viewports_;
            dirty_state.viewports = true;
        }
    }

    void SetScissors(const Scissors& scissors_) {
        if (!std::ranges::equal(scissors, scissors_)) {
            scissors = scissors_;
            dirty_state.scissors = true;
        }
    }

    void SetDepthTestEnabled(const bool enabled) {
        if (depth_test_enabled != enabled) {
            depth_test_enabled = enabled;
            dirty_state.depth_test_enabled = true;
        }
    }

    void SetDepthWriteEnabled(const bool enabled) {
        if (depth_write_enabled != enabled) {
            depth_write_enabled = enabled;
            dirty_state.depth_write_enabled = true;
        }
    }

    void SetDepthCompareOp(const vk::CompareOp compare_op) {
        if (depth_compare_op != compare_op) {
            depth_compare_op = compare_op;
            dirty_state.depth_compare_op = true;
        }
    }

    void SetDepthBoundsTestEnabled(const bool enabled) {
        if (depth_bounds_test_enabled != enabled) {
            depth_bounds_test_enabled = enabled;
            dirty_state.depth_bounds_test_enabled = true;
        }
    }

    void SetDepthBounds(const float min, const float max) {
        if (depth_bounds_min != min || depth_bounds_max != max) {
            depth_bounds_min = min;
            depth_bounds_max = max;
            dirty_state.depth_bounds = true;
        }
    }

    void SetDepthBiasEnabled(const bool enabled) {
        if (depth_bias_enabled != enabled) {
            depth_bias_enabled = enabled;
            dirty_state.depth_bias_enabled = true;
        }
    }

    void SetDepthBias(const float constant, const float clamp, const float slope) {
        if (depth_bias_constant != constant || depth_bias_clamp != clamp ||
            depth_bias_slope != slope) {
            depth_bias_constant = constant;
            depth_bias_clamp = clamp;
            depth_bias_slope = slope;
            dirty_state.depth_bias = true;
        }
    }

    void SetStencilTestEnabled(const bool enabled) {
        if (stencil_test_enabled != enabled) {
            stencil_test_enabled = enabled;
            dirty_state.stencil_test_enabled = true;
        }
    }

    void SetStencilOps(const StencilOps& front_ops, const StencilOps& back_ops) {
        if (stencil_front_ops != front_ops) {
            stencil_front_ops = front_ops;
            dirty_state.stencil_front_ops = true;
        }
        if (stencil_back_ops != back_ops) {
            stencil_back_ops = back_ops;
            dirty_state.stencil_back_ops = true;
        }
    }

    void SetStencilReferences(const u32 front_reference, const u32 back_reference) {
        if (stencil_front_reference != front_reference) {
            stencil_front_reference = front_reference;
            dirty_state.stencil_front_reference = true;
        }
        if (stencil_back_reference != back_reference) {
            stencil_back_reference = back_reference;
            dirty_state.stencil_back_reference = true;
        }
    }

    void SetStencilWriteMasks(const u32 front_write_mask, const u32 back_write_mask) {
        if (stencil_front_write_mask != front_write_mask) {
            stencil_front_write_mask = front_write_mask;
            dirty_state.stencil_front_write_mask = true;
        }
        if (stencil_back_write_mask != back_write_mask) {
            stencil_back_write_mask = back_write_mask;
            dirty_state.stencil_back_write_mask = true;
        }
    }

    void SetStencilCompareMasks(const u32 front_compare_mask, const u32 back_compare_mask) {
        if (stencil_front_compare_mask != front_compare_mask) {
            stencil_front_compare_mask = front_compare_mask;
            dirty_state.stencil_front_compare_mask = true;
        }
        if (stencil_back_compare_mask != back_compare_mask) {
            stencil_back_compare_mask = back_compare_mask;
            dirty_state.stencil_back_compare_mask = true;
        }
    }

    void SetPrimitiveRestartEnabled(const bool enabled) {
        if (primitive_restart_enable != enabled) {
            primitive_restart_enable = enabled;
            dirty_state.primitive_restart_enable = true;
        }
    }

    void SetCullMode(const vk::CullModeFlags cull_mode_) {
        if (cull_mode != cull_mode_) {
            cull_mode = cull_mode_;
            dirty_state.cull_mode = true;
        }
    }

    void SetFrontFace(const vk::FrontFace front_face_) {
        if (front_face != front_face_) {
            front_face = front_face_;
            dirty_state.front_face = true;
        }
    }

    void SetBlendConstants(const std::array<float, 4> blend_constants_) {
        if (blend_constants != blend_constants_) {
            blend_constants = blend_constants_;
            dirty_state.blend_constants = true;
        }
    }

    void SetRasterizerDiscardEnabled(const bool enabled) {
        if (rasterizer_discard_enable != enabled) {
            rasterizer_discard_enable = enabled;
            dirty_state.rasterizer_discard_enable = true;
        }
    }

    void SetColorWriteMasks(const ColorWriteMasks& color_write_masks_) {
        if (!std::ranges::equal(color_write_masks, color_write_masks_)) {
            color_write_masks = color_write_masks_;
            dirty_state.color_write_masks = true;
        }
    }

    void SetLineWidth(const float width) {
        if (line_width != width) {
            line_width = width;
            dirty_state.line_width = true;
        }
    }

    void SetAttachmentFeedbackLoopEnabled(const bool enabled) {
        if (feedback_loop_enabled != enabled) {
            feedback_loop_enabled = enabled;
            dirty_state.feedback_loop_enabled = true;
        }
    }
};

using SessionFunc = Common::UniqueFunction<void>;
using SubmitFunc = Common::UniqueFunction<void, SubmitInfo&>;

/// sotcport: a block of Vulkan commands waiting for the recording thread (threaded
/// recording, from bbport — the Bloodborne PC port, vk_scheduler.h there). Commands are closures
/// placed in the block's storage (no allocation per command) and run in order.
class RecordChunk {
public:
    static constexpr size_t Capacity = 128 * 1024;

    /// Returns false (and leaves `func` untouched) when the block has no room.
    template <typename Func>
    bool Push(Func&& func) {
        using Command = TypedCommand<std::decay_t<Func>>;
        static_assert(sizeof(Command) <= Capacity, "recorded command is too large");
        const size_t offset = (used + alignof(Command) - 1) & ~(alignof(Command) - 1);
        if (offset + sizeof(Command) > Capacity) {
            return false;
        }
        auto* command = new (storage + offset) Command(std::forward<Func>(func));
        if (last) {
            last->next = command;
        } else {
            first = command;
        }
        last = command;
        used = offset + sizeof(Command);
        return true;
    }

    /// Raw storage for a command's variable-length data (lives until the block is executed);
    /// null when the block is full.
    void* Allocate(size_t bytes, size_t align) {
        const size_t offset = (used + align - 1) & ~(align - 1);
        if (offset + bytes > Capacity) {
            return nullptr;
        }
        used = offset + bytes;
        return storage + offset;
    }

    void Execute(vk::CommandBuffer cmdbuf) {
        for (CommandBase* command = first; command;) {
            CommandBase* const next = command->next;
            command->Execute(cmdbuf);
            command->~CommandBase();
            command = next;
        }
        first = last = nullptr;
        used = 0;
    }

    [[nodiscard]] bool Empty() const noexcept {
        return first == nullptr;
    }
    [[nodiscard]] size_t Size() const noexcept {
        return used;
    }

private:
    struct CommandBase {
        virtual ~CommandBase() = default;
        virtual void Execute(vk::CommandBuffer cmdbuf) = 0;
        CommandBase* next{};
    };
    template <typename Func>
    struct TypedCommand final : CommandBase {
        explicit TypedCommand(Func&& func_) : func{std::move(func_)} {}
        explicit TypedCommand(const Func& func_) : func{func_} {}
        void Execute(vk::CommandBuffer cmdbuf) override {
            func(cmdbuf);
        }
        Func func;
    };

    alignas(64) std::byte storage[Capacity];
    size_t used = 0;
    CommandBase* first{};
    CommandBase* last{};
};

class Scheduler {
public:
    /// sotcport: `threaded_recording`: commands passed to Record() are recorded into the
    /// Vulkan command buffer by a thread of their own (SOTC_VK_RECORD_THREAD=0 turns it off).
    explicit Scheduler(const Instance& instance, bool threaded_recording = false);
    ~Scheduler();

    /// Sends the current execution context to the GPU
    /// and increments the scheduler timeline semaphore.
    void Flush(SubmitInfo& info);

    /// Sends the current execution context to the GPU
    /// and increments the scheduler timeline semaphore.
    void Flush();

    /// Sends the current execution context to the GPU and waits for it to complete.
    void Finish();

    /// Waits for the given tick to trigger on the GPU.
    void Wait(u64 tick);

    /// Attempts to execute operations whose tick the GPU has caught up with.
    void PopPendingOperations();

    /// Starts a new rendering scope with provided state.
    void BeginRendering(const RenderState& new_state);

    /// Ends current rendering scope.
    /// sotcport: whether a render pass is open.
    [[nodiscard]] bool IsRendering() const noexcept {
        return is_rendering;
    }

    void EndRendering();

    /// Starts a new session.
    void BeginSession();

    /// Returns the current command buffer used for uploads.
    vk::CommandBuffer UploadCommandBuffer();

    /// Sets a function to be called on every session finalization.
    void SetSessionCallback(SessionFunc&& on_session) {
        this->on_session = std::move(on_session);
    }

    /// Sets a function to be called on every scheduler submission.
    void SetSubmitCallback(SubmitFunc&& on_submit) {
        this->on_submit = std::move(on_submit);
    }

    /// Returns the current render state.
    const RenderState& GetRenderState() const {
        return render_state;
    }

    /// Returns the current pipeline dynamic state tracking.
    DynamicState& GetDynamicState() {
        return dynamic_state;
    }

    /// Returns the current command buffer, for recording on the calling thread. With threaded
    /// recording this first waits until every command passed to Record() is in the buffer, then
    /// records directly (Record() too) until the next KickRecording() or submission: code that
    /// was not converted to Record() stays correct.
    // TEMP (two-stage draw pipeline): while the draw recording thread runs a packet
    // (pipe_busy), only it may record into this scheduler; anyone else is reported once.
    static inline std::atomic<const Scheduler*> pipe_busy{nullptr};
    void CheckOwner() const;

    [[gnu::noinline]] vk::CommandBuffer CommandBuffer() {
        CheckOwner(); // TEMP
        if (recording_threaded && !direct_mode) {
            SyncRecording();
            direct_mode = true;
            direct_recordings.fetch_add(1, std::memory_order_relaxed);
            TraceDirectRecording(__builtin_return_address(0));
        }
        return sessions.back().primary;
    }

    /// sotcport: SOTC_RECORDER_TRACE=1 logs the most frequent callers of CommandBuffer()
    /// that had to wait for the recording thread (offsets in the executable, for addr2line).
    static void TraceDirectRecording(void* caller);

    /// sotcport: records `func(vk::CommandBuffer)` in order with the other commands. The
    /// closure must own everything it uses (capture by value): it may run later, on the
    /// recording thread.
    template <typename Func>
    void Record(Func&& func) {
        CheckOwner(); // TEMP (two-stage draw pipeline)
        if (!recording_threaded || direct_mode) {
            func(sessions.back().primary);
            return;
        }
        if (!record_chunk->Push(std::forward<Func>(func))) {
            full_chunks.push_back(std::move(record_chunk));
            record_chunk = AcquireChunk();
            const bool pushed = record_chunk->Push(std::forward<Func>(func));
            ASSERT(pushed);
        }
    }

    /// sotcport: whether Record() defers commands to the recording thread right now.
    [[nodiscard]] bool IsRecordingDeferred() const noexcept {
        return recording_threaded && !direct_mode;
    }

    /// sotcport: makes room for `bytes` of RecordAllocate() data plus the command that
    /// uses them in the current block: a block is recycled once executed, so the data must
    /// share the block of the command that reads it.
    void ReserveRecordData(size_t bytes) {
        if (!IsRecordingDeferred()) {
            return;
        }
        ASSERT(bytes + 1024 <= RecordChunk::Capacity);
        if (RecordChunk::Capacity - record_chunk->Size() < bytes + 1024) {
            full_chunks.push_back(std::move(record_chunk));
            record_chunk = AcquireChunk();
        }
    }

    /// sotcport: storage in the current block (after ReserveRecordData covering it); only
    /// while IsRecordingDeferred().
    void* RecordAllocate(size_t bytes, size_t align) {
        void* data = record_chunk->Allocate(bytes, align);
        ASSERT(data);
        return data;
    }

    /// sotcport: copies `data` into storage that lives until the command recorded next
    /// (which must use it) has run; without threaded recording, returns `data` itself.
    template <typename T>
    std::span<const T> RecordData(std::span<const T> data) {
        if (!recording_threaded || direct_mode || data.empty()) {
            return data;
        }
        const size_t bytes = data.size_bytes();
        // Room for the data and the command after it, so that both share a block.
        if (RecordChunk::Capacity - record_chunk->Size() < bytes + alignof(T) + 1024) {
            full_chunks.push_back(std::move(record_chunk));
            record_chunk = AcquireChunk();
        }
        auto* dst = static_cast<T*>(record_chunk->Allocate(bytes, alignof(T)));
        std::memcpy(static_cast<void*>(dst), data.data(), bytes);
        return {dst, data.size()};
    }

    /// sotcport: hands the recorded commands to the recording thread (in batches); called
    /// where nobody holds the raw command buffer (end of draws and dispatches).
    void KickRecording(bool force = false);

    /// sotcport: waits until every recorded command is in the command buffer.
    void SyncRecording();

    /// sotcport: CommandBuffer() calls that had to wait for the recording thread.
    static inline std::atomic<u64> direct_recordings{0};

    /// Returns the current command buffer tick.
    [[nodiscard]] u64 CurrentTick() const noexcept {
        return work_semaphore.CurrentTick();
    }

    /// Returns true when a tick has been triggered by the GPU.
    [[nodiscard]] bool IsFree(u64 tick) noexcept {
        if (work_semaphore.IsFree(tick)) {
            return true;
        }
        work_semaphore.Refresh();
        return work_semaphore.IsFree(tick);
    }

    /// Returns the scheduler timeline semaphore.
    [[nodiscard]] Semaphore* GetWorkSemaphore() noexcept {
        return &work_semaphore;
    }

    /// Defers an operation until the gpu has reached the current cpu tick.
    /// Will be run when submitting or calling PopPendingOperations.
    void DeferOperation(Common::UniqueFunction<void>&& func) {
        std::unique_lock lk(pending_ops_mutex);
        pending_ops.emplace(std::move(func), CurrentTick());
        num_pending_ops.fetch_add(1, std::memory_order_release); // sotcport
    }

    /// Defers an operation until the gpu has reached the current cpu tick.
    /// Runs as soon as possible in another thread.
    void DeferPriorityOperation(Common::UniqueFunction<void>&& func) {
        {
            std::unique_lock lk(priority_pending_ops_mutex);
            priority_pending_ops.emplace(std::move(func), CurrentTick());
        }
        priority_pending_ops_cv.notify_one();
    }

    static std::mutex submit_mutex;

private:
    void EndSession();

    void SubmitExecution(SubmitInfo& info);

    void PriorityPendingOpsThread(std::stop_token stoken);

    std::unique_ptr<RecordChunk> AcquireChunk();
    void RecorderThread(std::stop_token stoken);

private:
    const Instance& instance;
    Semaphore work_semaphore;
    CommandPool command_pool;
    /// sotcport: the primary command buffers come from a pool of their own: Vulkan wants
    /// the pool of a command buffer being recorded externally synchronized, and the recording
    /// thread records primaries while the GPU thread records upload buffers.
    CommandPool primary_pool;
    DynamicState dynamic_state;
    SessionFunc on_session{};
    SubmitFunc on_submit{};
    struct Session {
        vk::CommandBuffer upload{};
        vk::CommandBuffer primary{};
    };
    std::vector<Session> sessions;

    /// sotcport (diagnostics, SOTC_GPU_TIMING=1): GPU time of every submitted command buffer, from
    /// timestamps written at its start and end (see BeginSession/EndSession).
    void GpuTimingBegin(vk::CommandBuffer cmdbuf);
    void GpuTimingEnd(vk::CommandBuffer cmdbuf);
    void GpuTimingCollect();
    vk::UniqueQueryPool timing_pool;
    static constexpr u32 TimingSlots = 1024; ///< pairs of queries
    u32 timing_next = 0;
    struct TimingPending {
        u32 slot;
        u64 tick;
    };
    std::deque<TimingPending> timing_pending;
    std::vector<TimingPending> timing_unsubmitted;
    u64 timing_last_end = 0;
    double timing_busy_ns = 0, timing_gap_ns = 0, timing_max_ns = 0;
    u64 timing_count = 0;
    std::chrono::steady_clock::time_point timing_report{};
    std::condition_variable_any event_cv;
    struct PendingOp {
        Common::UniqueFunction<void> callback;
        u64 gpu_tick;
    };
    std::queue<PendingOp> pending_ops;
    std::recursive_mutex pending_ops_mutex;
    // sotcport (bbport step 1b): entries in pending_ops, read without the lock so the
    // per-draw PopPendingOperations returns at once when nothing waits; and a counter that
    // spaces out the GPU tick queries (an ioctl each) while the oldest operation is not due.
    std::atomic<u32> num_pending_ops{0};
    u32 pending_polls = 0;
    std::queue<PendingOp> priority_pending_ops;
    std::mutex priority_pending_ops_mutex;
    std::condition_variable_any priority_pending_ops_cv;
    std::jthread priority_pending_ops_thread;
    RenderState render_state;
    bool is_rendering = false;
    tracy::VkCtxScope* profiler_scope{};
    // sotcport: threaded recording (see Record()).
    bool recording_threaded = false;
    bool direct_mode = false; ///< the command buffer is being recorded on the caller's thread
    std::unique_ptr<RecordChunk> record_chunk;
    std::vector<std::unique_ptr<RecordChunk>> full_chunks;
    std::mutex recorder_mutex;
    std::condition_variable_any recorder_cv;
    std::condition_variable_any recorder_idle_cv;
    std::deque<std::unique_ptr<RecordChunk>> recorder_queue;
    std::vector<std::unique_ptr<RecordChunk>> free_chunks;
    std::atomic<size_t> queued_chunks{0}; ///< recorder_queue.size(), polled without the lock
    bool recorder_busy = false;
    bool recorder_sleeping = false;
    /// The command buffer the recording thread writes to; changes only while it is idle.
    vk::CommandBuffer recording_cmdbuf{};
    std::jthread recorder_thread;
};

} // namespace Vulkan
