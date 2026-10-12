// libSceAudioOut: the game's audio output ports, each played through its own SDL audio stream on
// the PC's default output device (SDL mixes the streams). 8-channel ports are mixed down to
// stereo here: left to the system's conversion, the center channel (where the game puts
// cutscene dialogue) went missing. An output call blocks while the
// stream already holds enough audio, so the game's audio thread runs at the device's pace.
// If no device can be opened, the samples are dropped and each call takes as long as the
// samples would take to play.

#include <SDL3/SDL.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

#include "common/log.h"
#include "common/types.h"
#include "hle/registry.h"

namespace hle::audio {

namespace {

using Clock = std::chrono::steady_clock;

constexpr s32 ErrorInvalidPort = static_cast<s32>(0x80260003);
constexpr s32 ErrorInvalidPointer = static_cast<s32>(0x80260004);
constexpr s32 ErrorInvalidSize = static_cast<s32>(0x80260006);
constexpr s32 ErrorInvalidFormat = static_cast<s32>(0x80260007);

// Low byte of the open call's format parameter.
enum class Format : u32 {
    S16Mono = 0,
    S16Stereo = 1,
    S16_8Ch = 2,
    FloatMono = 3,
    FloatStereo = 4,
    Float_8Ch = 5,
    S16_8ChStd = 6,
    Float_8ChStd = 7,
};

// How many blocks a stream may hold before an output call waits: about 21 ms at 256 samples.
constexpr u32 QueuedBlocks = 4;

struct Port {
    u32 samples;   // per output call
    u32 frequency; // Hz
    u32 channels;
    u32 sample_size; // bytes per sample of one channel
    bool is_float;
    SDL_AudioStream* stream; // null if no device
    std::vector<float> stereo; // 8-channel blocks mixed down
    Clock::time_point next_output; // pacing without a device
};

std::mutex g_mutex;
std::map<s32, Port> g_ports;
s32 g_next_handle = 1;

struct PortState {
    u16 output;
    u8 channel;
    u8 reserved_8;
    s16 volume;
    u16 reroute_counter;
    u64 flag;
    u64 reserved_64[2];
};

struct OutputParam {
    s32 handle;
    const void* data;
};

u32 BlockBytes(const Port& port) {
    return port.samples * port.channels * port.sample_size;
}

// 8 channels (front L/R, center, LFE, then two surround pairs, in either PS4 order) to stereo.
// The LFE is dropped, as usual for headphones and small speakers.
void MixDownToStereo(const void* source, Port& port) {
    constexpr float Center = 0.7071f;
    constexpr float Surround = 0.5f;
    port.stereo.resize(static_cast<size_t>(port.samples) * 2);
    for (u32 i = 0; i < port.samples; i++) {
        float in[8];
        for (u32 c = 0; c < 8; c++) {
            if (port.is_float) {
                std::memcpy(&in[c], static_cast<const u8*>(source) + (i * 8 + c) * 4, 4);
            } else {
                s16 value;
                std::memcpy(&value, static_cast<const u8*>(source) + (i * 8 + c) * 2, 2);
                in[c] = value / 32768.0f;
            }
        }
        const float left = in[0] + Center * in[2] + Surround * (in[4] + in[6]);
        const float right = in[1] + Center * in[2] + Surround * (in[5] + in[7]);
        port.stereo[i * 2] = std::clamp(left, -1.0f, 1.0f);
        port.stereo[i * 2 + 1] = std::clamp(right, -1.0f, 1.0f);
    }
}

// Queues one block (if `data` is not null). Returns the stream to wait on, or null to pace by
// the clock.
SDL_AudioStream* Queue(Port& port, const void* data) {
    if (!port.stream) {
        return nullptr;
    }
    if (data) {
        if (port.channels == 8) {
            MixDownToStereo(data, port);
            SDL_PutAudioStreamData(port.stream, port.stereo.data(),
                                   static_cast<int>(port.stereo.size() * sizeof(float)));
        } else {
            SDL_PutAudioStreamData(port.stream, data, static_cast<int>(BlockBytes(port)));
        }
    }
    return port.stream;
}

// Waits until the port can take another block. Called without the lock held.
void Wait(SDL_AudioStream* stream, int limit_bytes, Clock::time_point clock_wake) {
    if (!stream) {
        std::this_thread::sleep_until(clock_wake);
        return;
    }
    while (SDL_GetAudioStreamQueued(stream) > limit_bytes) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

// Queues `data` on the port and waits for room. Returns the samples per call, or an error.
s32 Output(s32 handle, const void* data, bool wait) {
    SDL_AudioStream* stream;
    int limit;
    Clock::time_point clock_wake;
    u32 samples;
    {
        std::scoped_lock lock{g_mutex};
        const auto it = g_ports.find(handle);
        if (it == g_ports.end()) {
            return ErrorInvalidPort;
        }
        Port& port = it->second;
        stream = Queue(port, data);
        limit = static_cast<int>(port.samples * QueuedBlocks *
                                 (port.channels == 8 ? 2 * sizeof(float) : port.channels * port.sample_size));
        // Without a device: one block's worth of time per call. After a long pause (the game
        // was busy), start again from now instead of catching up.
        const auto now = Clock::now();
        if (port.next_output < now - std::chrono::milliseconds(100)) {
            port.next_output = now;
        }
        clock_wake = port.next_output;
        port.next_output += std::chrono::nanoseconds(1'000'000'000LL * port.samples / port.frequency);
        samples = port.samples;
    }
    if (wait) {
        Wait(stream, limit, clock_wake);
    }
    return static_cast<s32>(samples);
}

PS4_SYSV_ABI s32 sceAudioOutInit() {
    if (!SDL_InitSubSystem(SDL_INIT_AUDIO)) {
        LOG_ERROR("Audio", "SDL_InitSubSystem(audio) failed: {}; the game will be silent", SDL_GetError());
    }
    return 0;
}

PS4_SYSV_ABI s32 sceAudioOutOpen(s32 user_id, s32 type, s32 index, u32 samples, u32 frequency,
                                 u32 param) {
    if (samples == 0 || frequency == 0) {
        return ErrorInvalidSize;
    }
    Port port{};
    port.samples = samples;
    port.frequency = frequency;
    SDL_AudioFormat sdl_format;
    switch (static_cast<Format>(param & 0xFF)) {
    case Format::S16Mono:
        port.channels = 1, sdl_format = SDL_AUDIO_S16;
        break;
    case Format::S16Stereo:
        port.channels = 2, sdl_format = SDL_AUDIO_S16;
        break;
    case Format::S16_8Ch:
        port.channels = 8, sdl_format = SDL_AUDIO_S16;
        break;
    case Format::S16_8ChStd:
        port.channels = 8, sdl_format = SDL_AUDIO_S16;
        break;
    case Format::FloatMono:
        port.channels = 1, sdl_format = SDL_AUDIO_F32;
        break;
    case Format::FloatStereo:
        port.channels = 2, sdl_format = SDL_AUDIO_F32;
        break;
    case Format::Float_8Ch:
        port.channels = 8, sdl_format = SDL_AUDIO_F32;
        break;
    case Format::Float_8ChStd:
        port.channels = 8, sdl_format = SDL_AUDIO_F32;
        break;
    default:
        return ErrorInvalidFormat;
    }
    port.sample_size = SDL_AUDIO_BYTESIZE(sdl_format);
    port.is_float = sdl_format == SDL_AUDIO_F32;
    port.next_output = Clock::now();

    // 8-channel ports reach SDL already mixed down to stereo floats.
    const SDL_AudioSpec spec = port.channels == 8
                                   ? SDL_AudioSpec{SDL_AUDIO_F32, 2, static_cast<int>(frequency)}
                                   : SDL_AudioSpec{sdl_format, static_cast<int>(port.channels),
                                                   static_cast<int>(frequency)};
    port.stream = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, nullptr,
                                            nullptr);
    if (port.stream) {
        SDL_ResumeAudioStreamDevice(port.stream);
    } else {
        LOG_ERROR("Audio", "could not open the audio output: {}", SDL_GetError());
    }

    std::scoped_lock lock{g_mutex};
    const s32 handle = g_next_handle++;
    LOG_INFO("Audio", "port {} opened (user {}, type {}, {} samples at {} Hz, {} channels, format {:#x})",
             handle, user_id, type, samples, frequency, port.channels, param);
    g_ports.emplace(handle, std::move(port));
    (void)index;
    return handle;
}

PS4_SYSV_ABI s32 sceAudioOutClose(s32 handle) {
    std::scoped_lock lock{g_mutex};
    const auto it = g_ports.find(handle);
    if (it == g_ports.end()) {
        return ErrorInvalidPort;
    }
    if (it->second.stream) {
        SDL_DestroyAudioStream(it->second.stream);
    }
    g_ports.erase(it);
    return 0;
}

// A null `data` only waits for the port.
PS4_SYSV_ABI s32 sceAudioOutOutput(s32 handle, const void* data) {
    return Output(handle, data, true);
}

// Several ports at once: queue all, then wait on the first (they share the same timing).
PS4_SYSV_ABI s32 sceAudioOutOutputs(const OutputParam* params, u32 count) {
    if (!params || count == 0) {
        return ErrorInvalidPointer;
    }
    for (u32 i = 1; i < count; i++) {
        if (const s32 result = Output(params[i].handle, params[i].data, false); result < 0) {
            return result;
        }
    }
    return Output(params[0].handle, params[0].data, true);
}

PS4_SYSV_ABI s32 sceAudioOutGetPortState(s32 handle, PortState* state) {
    if (!state) {
        return ErrorInvalidPointer;
    }
    std::scoped_lock lock{g_mutex};
    const auto it = g_ports.find(handle);
    if (it == g_ports.end()) {
        return ErrorInvalidPort;
    }
    std::memset(state, 0, sizeof(*state));
    state->output = 1; // main output
    state->channel = static_cast<u8>(it->second.channels);
    state->volume = 127;
    return 0;
}

PS4_SYSV_ABI s32 sceAudioOutSetMixLevelPadSpk(s32, s32) {
    return 0;
}

} // namespace

HLE_FUNCTION("libSceAudioOut", "sceAudioOutInit", sceAudioOutInit);
HLE_FUNCTION("libSceAudioOut", "sceAudioOutOpen", sceAudioOutOpen);
HLE_FUNCTION("libSceAudioOut", "sceAudioOutClose", sceAudioOutClose);
HLE_FUNCTION("libSceAudioOut", "sceAudioOutOutput", sceAudioOutOutput);
HLE_FUNCTION("libSceAudioOut", "sceAudioOutOutputs", sceAudioOutOutputs);
HLE_FUNCTION("libSceAudioOut", "sceAudioOutGetPortState", sceAudioOutGetPortState);
HLE_FUNCTION("libSceAudioOut", "sceAudioOutSetMixLevelPadSpk", sceAudioOutSetMixLevelPadSpk);

} // namespace hle::audio
