// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <vector>
#include <bit>
#include <algorithm>
#include "common/assert.h"
#include "common/debug.h"
#include "common/thread.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/libraries/kernel/time.h"
#include "core/libraries/videoout/driver.h"
#include "core/libraries/videoout/videoout_error.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/renderer_vulkan/vk_presenter.h"
#include "perf_temp.h" // sotcport TEMP
#include "diag/toggles.h" // sotcport: F6 diagnostics

extern std::unique_ptr<Vulkan::Presenter> presenter;
extern std::unique_ptr<AmdGpu::Liverpool> liverpool;

namespace gpu {
void NoteFramePresented(); // sotcport: shim/gpu_glue.cpp
}

namespace Libraries::VideoOut {

constexpr static bool Is32BppPixelFormat(PixelFormat format) {
    switch (format) {
    case PixelFormat::A8R8G8B8Srgb:
    case PixelFormat::A8B8G8R8Srgb:
    case PixelFormat::A2R10G10B10:
    case PixelFormat::A2R10G10B10Srgb:
    case PixelFormat::A2R10G10B10Bt2020Pq:
        return true;
    default:
        return false;
    }
}

constexpr u32 PixelFormatBpp(PixelFormat pixel_format) {
    switch (pixel_format) {
    case PixelFormat::A16R16G16B16Float:
        return 8;
    default:
        return 4;
    }
}

VideoOutDriver::VideoOutDriver(u32 width, u32 height) {
    main_port.resolution.full_width = width;
    main_port.resolution.full_height = height;
    main_port.resolution.pane_width = width;
    main_port.resolution.pane_height = height;
    present_thread = std::jthread([&](std::stop_token token) { PresentThread(token); });
}

VideoOutDriver::~VideoOutDriver() = default;

int VideoOutDriver::Open(const ServiceThreadParams* params) {
    if (main_port.is_open) {
        return ORBIS_VIDEO_OUT_ERROR_RESOURCE_BUSY;
    }
    main_port.is_open = true;
    liverpool->SetVoPort(&main_port);
    return 1;
}

void VideoOutDriver::Close(s32 handle) {
    std::scoped_lock lock{mutex};

    // Mark as closed
    main_port.is_open = false;
    main_port.flip_rate = 0;
    main_port.prev_index = -1;

    // Clear port information
    std::memset(main_port.buffer_labels.data(), 0, sizeof(main_port.buffer_labels));
    std::memset(main_port.groups.data(), 0, sizeof(main_port.groups));
    std::memset(&main_port.vblank_status, 0, sizeof(main_port.vblank_status));
    main_port.flip_status = FlipStatus{};

    // Re-initialize buffers
    std::memset(main_port.buffer_slots.data(), 0, sizeof(main_port.buffer_slots));
    for (auto& buffer : main_port.buffer_slots) {
        buffer.group_index = -1;
    }

    // Clear events
    for (auto event : main_port.flip_events) {
        auto equeue = Kernel::GetEqueue(event);
        if (equeue != nullptr) {
            equeue->RemoveEvent(static_cast<u64>(OrbisVideoOutInternalEventId::Flip),
                                Kernel::OrbisKernelEvent::Filter::VideoOut);
        }
    }
    main_port.flip_events.clear();
    for (auto event : main_port.vblank_events) {
        auto equeue = Kernel::GetEqueue(event);
        if (equeue != nullptr) {
            equeue->RemoveEvent(static_cast<u64>(OrbisVideoOutInternalEventId::Vblank),
                                Kernel::OrbisKernelEvent::Filter::VideoOut);
        }
    }
    main_port.vblank_events.clear();
}

VideoOutPort* VideoOutDriver::GetPort(int handle) {
    if (handle != 1) [[unlikely]] {
        return nullptr;
    }
    return &main_port;
}

int VideoOutDriver::RegisterBuffers(VideoOutPort* port, s32 startIndex, void* const* addresses,
                                    s32 bufferNum, const BufferAttribute* attribute) {
    const s32 group_index = port->FindFreeGroup();
    if (group_index >= MaxDisplayBufferGroups) {
        return ORBIS_VIDEO_OUT_ERROR_NO_EMPTY_SLOT;
    }

    if (startIndex + bufferNum > MaxDisplayBuffers || startIndex > MaxDisplayBuffers ||
        bufferNum > MaxDisplayBuffers) {
        LOG_ERROR(Lib_VideoOut,
                  "Attempted to register too many buffers startIndex = {}, bufferNum = {}",
                  startIndex, bufferNum);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }

    const s32 end_index = startIndex + bufferNum;
    if (bufferNum > 0 &&
        std::any_of(port->buffer_slots.begin() + startIndex, port->buffer_slots.begin() + end_index,
                    [](auto& buffer) { return buffer.group_index != -1; })) {
        return ORBIS_VIDEO_OUT_ERROR_SLOT_OCCUPIED;
    }

    if (attribute->reserved0 != 0 || attribute->reserved1 != 0) {
        LOG_ERROR(Lib_VideoOut, "Invalid reserved members");
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }
    if (attribute->aspect_ratio != 0) {
        LOG_ERROR(Lib_VideoOut, "Invalid aspect ratio = {}", attribute->aspect_ratio);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_ASPECT_RATIO;
    }
    if (attribute->width > attribute->pitch_in_pixel) {
        LOG_ERROR(Lib_VideoOut, "Buffer width {} is larger than pitch {}", attribute->width,
                  attribute->pitch_in_pixel);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_PITCH;
    }
    if (attribute->tiling_mode < TilingMode::Tile || attribute->tiling_mode > TilingMode::Linear) {
        LOG_ERROR(Lib_VideoOut, "Invalid tilingMode = {}",
                  static_cast<u32>(attribute->tiling_mode));
        return ORBIS_VIDEO_OUT_ERROR_INVALID_TILING_MODE;
    }

    LOG_INFO(Lib_VideoOut,
             "startIndex = {}, bufferNum = {}, pixelFormat = {}, aspectRatio = {}, "
             "tilingMode = {}, width = {}, height = {}, pitchInPixel = {}, option = {:#x}",
             startIndex, bufferNum, GetPixelFormatString(attribute->pixel_format),
             attribute->aspect_ratio, static_cast<u32>(attribute->tiling_mode), attribute->width,
             attribute->height, attribute->pitch_in_pixel, attribute->option);

    auto& group = port->groups[group_index];
    std::memcpy(&group.attrib, attribute, sizeof(BufferAttribute));
    group.is_occupied = true;

    for (u32 i = 0; i < bufferNum; i++) {
        const uintptr_t address = reinterpret_cast<uintptr_t>(addresses[i]);
        port->buffer_slots[startIndex + i] = VideoOutBuffer{
            .group_index = group_index,
            .address_left = address,
            .address_right = 0,
        };

        // Reset flip label also when registering buffer
        port->buffer_labels[startIndex + i] = 0;
        port->SignalVoLabel();

        presenter->RegisterVideoOutSurface(group, address);
        LOG_INFO(Lib_VideoOut, "buffers[{}] = {:#x}", i + startIndex, address);
    }

    return group_index;
}

int VideoOutDriver::UnregisterBuffers(VideoOutPort* port, s32 attributeIndex) {
    if (attributeIndex >= MaxDisplayBufferGroups || !port->groups[attributeIndex].is_occupied) {
        LOG_ERROR(Lib_VideoOut, "Invalid attribute index {}", attributeIndex);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }

    auto& group = port->groups[attributeIndex];
    group.is_occupied = false;

    for (auto& buffer : port->buffer_slots) {
        if (buffer.group_index != attributeIndex) {
            continue;
        }
        buffer.group_index = -1;
    }

    return ORBIS_OK;
}

int VideoOutDriver::ChangeBufferAttribute(VideoOutPort* port, s32 attributeIndex,
                                          const BufferAttribute* attribute) {
    if (attributeIndex >= MaxDisplayBufferGroups || !port->groups[attributeIndex].is_occupied) {
        LOG_ERROR(Lib_VideoOut, "Invalid attribute index {}", attributeIndex);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }

    if (attribute->reserved0 != 0 || attribute->reserved1 != 0) {
        LOG_ERROR(Lib_VideoOut, "Invalid reserved members");
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }
    if (attribute->aspect_ratio != 0) {
        LOG_ERROR(Lib_VideoOut, "Invalid aspect ratio = {}", attribute->aspect_ratio);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_ASPECT_RATIO;
    }
    if (attribute->width > attribute->pitch_in_pixel) {
        LOG_ERROR(Lib_VideoOut, "Buffer width {} is larger than pitch {}", attribute->width,
                  attribute->pitch_in_pixel);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_PITCH;
    }
    if (attribute->tiling_mode < TilingMode::Tile || attribute->tiling_mode > TilingMode::Linear) {
        LOG_ERROR(Lib_VideoOut, "Invalid tilingMode = {}",
                  static_cast<u32>(attribute->tiling_mode));
        return ORBIS_VIDEO_OUT_ERROR_INVALID_TILING_MODE;
    }

    LOG_INFO(Lib_VideoOut,
             "attributeIndex = {}, pixelFormat = {}, aspectRatio = {}, "
             "tilingMode = {}, width = {}, height = {}, pitchInPixel = {}, option = {:#x}",
             attributeIndex, GetPixelFormatString(attribute->pixel_format), attribute->aspect_ratio,
             static_cast<u32>(attribute->tiling_mode), attribute->width, attribute->height,
             attribute->pitch_in_pixel, attribute->option);

    std::unique_lock lock{port->port_mutex};
    std::memcpy(&port->groups[attributeIndex].attrib, attribute, sizeof(BufferAttribute));
    return 0;
}

void VideoOutDriver::Flip(const Request& req) {
    // Update HDR status before presenting.
    presenter->SetHDR(req.port->is_hdr);

    // Present the frame.
    presenter->Present(req.frame);
    gpu::NoteFramePresented(); // sotcport: for the freeze watchdog (src/gpu/gpu.h)

    // Update flip status.
    auto* port = req.port;
    {
        std::unique_lock lock{port->port_mutex};
        auto& flip_status = port->flip_status;
        flip_status.count++;
        flip_status.process_time = Libraries::Kernel::sceKernelGetProcessTime();
        flip_status.tsc = Libraries::Kernel::sceKernelReadTsc();
        flip_status.flip_arg = req.flip_arg;
        flip_status.current_buffer = req.index;
        if (req.eop) {
            --flip_status.gc_queue_num;
        }
        --flip_status.flip_pending_num;
    }

    // Trigger flip events for the port.
    for (auto event : port->flip_events) {
        auto equeue = Kernel::GetEqueue(event);
        if (equeue != nullptr) {
            equeue->TriggerEvent(
                static_cast<u64>(OrbisVideoOutInternalEventId::Flip),
                Kernel::OrbisKernelEvent::Filter::VideoOut,
                reinterpret_cast<void*>(static_cast<u64>(OrbisVideoOutInternalEventId::Flip) |
                                        (req.flip_arg << 16)));
        }
    }

    // Reset prev flip label
    if (port->prev_index != -1) {
        port->buffer_labels[port->prev_index] = 0;
        port->SignalVoLabel();
    }
    // save to prev buf index
    port->prev_index = req.index;
}

void VideoOutDriver::DrawBlankFrame() {
    const auto empty_frame = presenter->PrepareBlankFrame(true);
    presenter->Present(empty_frame, false, false);
}

void VideoOutDriver::DrawLastFrame() {
    const auto frame = presenter->PrepareLastFrame();
    if (frame != nullptr) {
        presenter->Present(frame, true);
    }
}

bool VideoOutDriver::SubmitFlip(VideoOutPort* port, s32 index, s64 flip_arg,
                                bool is_eop /*= false*/) {
    {
        std::unique_lock lock{port->port_mutex};
        if (index != -1 && port->flip_status.flip_pending_num > 16) {
            LOG_ERROR(Lib_VideoOut, "Flip queue is full");
            return false;
        }

        if (is_eop) {
            ++port->flip_status.gc_queue_num;
        }
        ++port->flip_status.flip_pending_num; // integral GPU and CPU pending flips counter
        port->flip_status.submit_tsc = Libraries::Kernel::sceKernelReadTsc();
    }

    if (!is_eop) {
        // Non EOP flips can arrive from any thread so ask GPU thread to perform them
        liverpool->SendCommand([=, this]() { SubmitFlipInternal(port, index, flip_arg, is_eop); });
    } else {
        SubmitFlipInternal(port, index, flip_arg, is_eop);
    }

    return true;
}

void VideoOutDriver::SubmitFlipInternal(VideoOutPort* port, s32 index, s64 flip_arg, bool is_eop) {
    Vulkan::Frame* frame;
    if (index == -1) {
        frame = presenter->PrepareBlankFrame(false);
    } else {
        const auto& buffer = port->buffer_slots[index];
        ASSERT_MSG(buffer.group_index >= 0, "Trying to flip an unregistered buffer!");
        const auto& group = port->groups[buffer.group_index];
        frame = presenter->PrepareFrame(group, buffer.address_left);
    }

    {
        std::scoped_lock lock{mutex};
        requests.push({
            .frame = frame,
            .port = port,
            .flip_arg = flip_arg,
            .index = index,
            .eop = is_eop,
            .submitted = std::chrono::steady_clock::now(),
        });
    }
    requests_cv.notify_one(); // sotcport: see PresentThread (uncapped mode)
}

bool UncappedFrameRate() {
    // sotcport: SOTC_UNCAPPED=1 takes away every frame rate limit: the game's flip rate is
    // ignored (see sceVideoOutSetFlipRate) and each flip is presented as soon as it is submitted
    // instead of at the next 60 Hz vblank. The vblank events themselves keep their 60 Hz pace (the
    // game counts them). The swapchain uses Mailbox, so the display does not cap it either.
    // SOTC_FPS_LIMIT=<fps> (implies the same) presents at most that many frames per second.
    static const bool uncapped =
        std::getenv("SOTC_UNCAPPED") != nullptr || std::getenv("SOTC_FPS_LIMIT") != nullptr;
    return uncapped;
}

// sotcport: SOTC_FPS_LIMIT=<fps>[,<fps>...]: frames are presented at most every 1/fps seconds, on
// an even beat. The game waits for its flips to be done before starting more work, so holding a
// flip back paces the game itself. Even frame times matter here: the game's cloth (Wander's cape)
// and Agro's movement were tuned for the console's fixed 30 FPS and misbehave whenever the frame
// time changes from one frame to the next (user tests, 2026-10-11), even by a little.
// With several rates (for example "60,45,30") the limiter is adaptive: it runs at the first one,
// drops to the next when the game keeps missing the beat, and climbs back only after the game has
// shown for a while that it has room for the faster one. Each rate is held steady, so the
// simulation only sees the rare switches.
class FrameRateLimiter {
public:
    using Clock = std::chrono::steady_clock;

    FrameRateLimiter() {
        if (const char* env = std::getenv("SOTC_FPS_LIMIT")) {
            for (const char* p = env; *p;) {
                char* end = nullptr;
                const double fps = std::strtod(p, &end);
                if (end == p) {
                    break;
                }
                if (fps > 0) {
                    periods.push_back(std::chrono::nanoseconds(static_cast<s64>(1e9 / fps)));
                }
                p = *end == ',' ? end + 1 : end;
            }
        }
        std::ranges::sort(periods); // fastest first
        if (periods.size() > 8) {
            periods.resize(8);
        }
    }

    [[nodiscard]] bool Active() const {
        return !periods.empty();
    }
    [[nodiscard]] std::chrono::nanoseconds Period() const {
        return periods.empty() ? std::chrono::nanoseconds{0} : periods[level];
    }

    /// A frame is being presented now: `submitted` is when the game handed it over, `slot` the
    /// beat it was meant for.
    void OnPresent(Clock::time_point now, Clock::time_point submitted, Clock::time_point slot) {
        if (periods.size() < 2) {
            return;
        }
        frames_at_level++;
        // Late: presented over 1 ms after its beat (the game was not ready in time).
        const bool late = now - slot > std::chrono::milliseconds(1);
        recent_late = (recent_late << 1) | (late ? 1 : 0);
        constexpr u64 Window = 32;
        // A whole window at this rate first: one long hitch makes several frames in a row late,
        // which must not drop two or three rates at once.
        if (level + 1 < periods.size() && frames_at_level >= Window &&
            std::popcount(recent_late & ((u64{1} << Window) - 1)) >= 4) {
            // Coming back up to the rate that failed has to wait twice as long as last time.
            climb_wait[level] = std::min<u64>(climb_wait[level] * 2, 64);
            Switch(level + 1, "the game kept missing the beat");
            return;
        }
        if (level == 0) {
            return;
        }
        // Room for the faster rate: the frame was ready early enough before its beat that it
        // would have made the faster beat too, with 15% of that period to spare. The game is
        // paced by its flips, so how early frames are ready is its spare capacity.
        const auto faster = periods[level - 1];
        const auto needed_early = (Period() - faster) + faster * 15 / 100;
        if (!late && slot - submitted >= needed_early) {
            fast_enough++;
        } else {
            fast_enough = 0;
        }
        // Every frame of the last few seconds (more after failed climbs) had that room.
        const u64 frames = static_cast<u64>(climb_wait[level - 1] * 1e9 / Period().count());
        if (fast_enough >= frames) {
            Switch(level - 1, "the game has room for it");
        }
    }

private:
    void Switch(size_t new_level, const char* why) {
        LOG_INFO(Lib_VideoOut, "frame rate limit: {:.0f} -> {:.0f} FPS ({})",
                 1e9 / periods[level].count(), 1e9 / periods[new_level].count(), why);
        level = new_level;
        recent_late = 0;
        fast_enough = 0;
        frames_at_level = 0;
    }

    std::vector<std::chrono::nanoseconds> periods; ///< allowed frame periods, shortest first
    size_t level = 0;
    u64 recent_late = 0;    ///< one bit per recent frame: presented after its beat
    u64 fast_enough = 0;    ///< consecutive frames produced fast enough for the faster rate
    u64 frames_at_level = 0;
    /// Seconds of room needed before climbing back to each rate (doubles after each failure).
    std::array<u64, 8> climb_wait{3, 3, 3, 3, 3, 3, 3, 3};
};

void VideoOutDriver::PresentThread(std::stop_token token) {
    const std::chrono::nanoseconds vblank_period(1000000000 /
                                                 EmulatorSettings.GetVblankFrequency());

    Common::SetCurrentThreadName("shadPS4:PresentThread");
    Common::SetCurrentThreadRealtime(vblank_period);

    Common::AccurateTimer timer{vblank_period};

    const auto receive_request = [this] -> Request {
        std::scoped_lock lk{mutex};
        if (!requests.empty()) {
            const auto request = requests.front();
            requests.pop();
            return request;
        }
        return {};
    };

    if (UncappedFrameRate()) {
        UncappedPresentLoop(token, vblank_period);
        return;
    }

    while (!token.stop_requested()) {
        timer.Start();

        if (DebugState.IsGuestThreadsPaused()) {
            DrawLastFrame();
            timer.End();
            continue;
        }

        // Check if it's time to take a request.
        auto& vblank_status = main_port.vblank_status;
        if (vblank_status.count % (main_port.flip_rate + 1) == 0) {
            const auto request = receive_request();
            if (!request) {
                if (timer.GetTotalWait().count() < 0) { // Dont draw too fast
                    if (!main_port.is_open) {
                        DrawBlankFrame();
                    }
                    // sotcport: shadPS4 redraws the last frame here while its ImGui
                    // overlay is animating; there is no overlay here.
                }
            } else {
                perf::Add(perf::Flip, 0); // sotcport TEMP
                Flip(request);
                FRAME_END;
            }
        }

        {
            // Needs lock here as can be concurrently read by `sceVideoOutGetVblankStatus`
            std::scoped_lock lock{main_port.vo_mutex};

            // Trigger flip events for the port
            for (auto event : main_port.vblank_events) {
                auto equeue = Kernel::GetEqueue(event);
                if (equeue != nullptr) {
                    equeue->TriggerEvent(
                        static_cast<u64>(OrbisVideoOutInternalEventId::Vblank),
                        Kernel::OrbisKernelEvent::Filter::VideoOut,
                        reinterpret_cast<void*>(
                            static_cast<u64>(OrbisVideoOutInternalEventId::Vblank) |
                            (vblank_status.count << 16)));
                }
            }

            // Update vblank status
            vblank_status.count++;
            vblank_status.process_time = Libraries::Kernel::sceKernelGetProcessTime();
            vblank_status.tsc = Libraries::Kernel::sceKernelReadTsc();
            main_port.vblank_cv.notify_all();
        }

        timer.End();
    }
}

// sotcport: the present loop without a frame rate limit (UncappedFrameRate): flips are presented
// as they come; vblank events still fire every `vblank_period`.
void VideoOutDriver::UncappedPresentLoop(std::stop_token token, std::chrono::nanoseconds period) {
    using Clock = std::chrono::steady_clock;
    FrameRateLimiter limiter;
    auto next_vblank = Clock::now() + period;
    auto last_flip = Clock::now();
    auto next_flip = Clock::now(); // with a frame rate limit: the earliest time for the next flip
    // F6 can cap the frame rate at 60 while playing (diag::Cap60On), in place of the limiter.
    constexpr auto Period60 = std::chrono::nanoseconds(1'000'000'000 / 60);
    while (!token.stop_requested()) {
        Request request{};
        const bool cap60 = diag::On(diag::Cap60On);
        const bool limited = cap60 || limiter.Active();
        {
            std::unique_lock lk{mutex};
            const auto flip_allowed = [&] {
                return !limited || Clock::now() >= next_flip;
            };
            if (!requests.empty() && !flip_allowed()) {
                // A frame waits for its beat (frame rate limit): sleep until then or the vblank.
                requests_cv.wait_until(lk, std::min(next_vblank, next_flip));
            } else {
                // Wait for a request or the next vblank.
                requests_cv.wait_until(lk, next_vblank, [&] {
                    return !requests.empty() || token.stop_requested();
                });
            }
            if (!requests.empty() && flip_allowed()) {
                request = requests.front();
                requests.pop();
            }
        }
        if (request) {
            const auto now = Clock::now();
            perf::NoteFrameInterval( // sotcport TEMP: pacing report
                std::chrono::duration_cast<std::chrono::nanoseconds>(now - last_flip).count());
            if (cap60) {
                next_flip = std::max(next_flip + Period60, now);
            } else if (limiter.Active()) {
                limiter.OnPresent(now, request.submitted, next_flip);
                // An even beat: the next slot follows this one, unless the game fell behind.
                next_flip = std::max(next_flip + limiter.Period(), now);
            }
            last_flip = now;
            perf::Add(perf::Flip, 0); // sotcport TEMP
            Flip(request);
            FRAME_END;
        }
        if (Clock::now() < next_vblank) {
            continue;
        }
        next_vblank += period;
        if (next_vblank < Clock::now()) {
            next_vblank = Clock::now() + period; // fell behind (debugger, hitch): no burst
        }
        std::scoped_lock lock{main_port.vo_mutex};
        auto& vblank_status = main_port.vblank_status;
        for (auto event : main_port.vblank_events) {
            auto equeue = Kernel::GetEqueue(event);
            if (equeue != nullptr) {
                equeue->TriggerEvent(static_cast<u64>(OrbisVideoOutInternalEventId::Vblank),
                                     Kernel::OrbisKernelEvent::Filter::VideoOut,
                                     reinterpret_cast<void*>(
                                         static_cast<u64>(OrbisVideoOutInternalEventId::Vblank) |
                                         (vblank_status.count << 16)));
            }
        }
        vblank_status.count++;
        vblank_status.process_time = Libraries::Kernel::sceKernelGetProcessTime();
        vblank_status.tsc = Libraries::Kernel::sceKernelReadTsc();
        main_port.vblank_cv.notify_all();
    }
}

} // namespace Libraries::VideoOut
