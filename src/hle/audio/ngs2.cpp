// libSceNgs2, the PS4's software audio mixer. The game's sound library (libSceSndx, linked into
// the eboot) builds racks of voices and asks the system to render them every 256 samples:
//
//   sampler voices (decode ATRAC9/PCM)  ->  submixers (THRU, COMP, EQ, AMP...)  ->  mastering
//
// Each voice sends its output through its ports; a patch connects a port to another voice's
// input, and each port has a volume and an optional channel matrix. Mastering voice i becomes
// render buffer i (main output, controller speaker, background music).
//
// This is a simplified Ngs2: samplers play their waveform blocks (no envelopes or filters),
// submixers pass their input through (no compressor or EQ), and reverb and delay racks are
// silent for now, since they only receive effect sends. Volumes, matrices and routing are real.
//
// Parameter blocks, rack options and state layouts were deduced from what the game sends.

#include <algorithm>
#include <format>
#include <string>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <libatrac9.h>

#include "hevag_coefs.inc" // third_party/externals/vgmstream_hevag

#include "common/log.h"
#include "common/types.h"
#include "hle/registry.h"

namespace hle::audio {

namespace {

constexpr s32 ErrorInvalidSystem = static_cast<s32>(0x804A0230);
constexpr s32 ErrorInvalidRack = static_cast<s32>(0x804A0261);
constexpr s32 ErrorInvalidVoice = static_cast<s32>(0x804A0300);
constexpr s32 ErrorInvalidAddress = static_cast<s32>(0x804A0207);

constexpr u32 OutputRate = 48000;
constexpr u32 MaxChannels = 8;
constexpr u32 MaxPorts = 8;

enum RackId : u32 {
    RackSampler = 0x1000,
    RackSubmixer = 0x2000,
    RackReverb = 0x2001,
    RackMastering = 0x3000,
    RackCustomSampler = 0x4001,
    RackCustomSubmixer = 0x4002, // the game's "DELAY"
    RackCustomMastering = 0x4003,
};

// Parameter ids common to all voices.
enum VoiceParam : u32 {
    ParamMatrixLevels = 1,
    ParamPortVolume = 2,
    ParamPortMatrix = 3,
    ParamPortDelay = 4,
    ParamPatch = 5,
    ParamEvent = 6,
};

// Sampler parameter ids.
enum SamplerParam : u32 {
    SamplerSetup = 0x10000000,
    SamplerWaveformBlocks = 0x10000001,
    SamplerPitch = 0x10000005,
    SamplerEnvelope = 0x10000006,
    SamplerFilter = 0x1000000A,
};

constexpr u32 SubmixerSetup = 0x20000000;
constexpr u32 SubmixerUserFx = 0x20000004;
constexpr u32 ReverbSetup = 0x20010000;
constexpr u32 MasteringSetup = 0x30000000;
constexpr u32 CustomSubmixerSetup = 0x40020000;

// Events (ParamEvent). Deduced from the game: it starts a cutscene's audio with Play + Pause
// during loading and resumes it when the scene begins, and sends Kill when done with a voice.
enum VoiceEvent : u32 {
    EventPlay = 0,
    EventStop = 1,
    EventStopImmediate = 2,
    EventKill = 3,
    EventPause = 4,
    EventResume = 5,
    EventResumeB = 6, // not seen yet; assumed to resume too
};

// Voice state flags (sceNgs2VoiceGetState*); libSceSndx tests InUse and Playing|Empty.
constexpr u32 StateInUse = 0x1;
constexpr u32 StatePlaying = 0x2;
constexpr u32 StatePaused = 0x4;
constexpr u32 StateEmpty = 0x20;

// Waveform types.
constexpr u32 WaveformPcmI16 = 0x12;
constexpr u32 WaveformPcmF32 = 0x18;
constexpr u32 WaveformVag = 0x1C;
constexpr u32 WaveformAtrac9 = 0x40;

constexpr u32 RepeatForever = 0xFFFFFFFF;

struct ContextBufferInfo {
    void* host_buffer;
    u64 host_buffer_size;
    u64 reserved[5];
    u64 user_data;
};

struct RenderBufferInfo {
    void* buffer;
    u64 size;
    u32 waveform_type;
    u32 channels;
};

struct ParamHeader {
    u16 size;
    s16 next; // offset to the next header, 0 = last
    u32 id;
};

struct WaveformFormat {
    u32 type;
    u32 channels;
    u32 sample_rate;
    u32 config_data; // ATRAC9: the 4 configuration bytes, first byte in the top bits
    u32 frame_offset;
    u32 frame_margin;
};

struct WaveformBlock {
    u32 data_offset;
    u32 data_size;
    u32 num_repeats;
    u32 num_skip_samples;
    u32 num_samples;
    u32 reserved;
    u64 user_data;
};
static_assert(sizeof(WaveformBlock) == 32);

struct SamplerVoiceState {
    u32 flags;
    float envelope_height;
    float peak_height;
    u32 reserved;
    u64 num_decoded_samples;
    u64 decoded_data_size;
    u64 user_data;
    const void* waveform_data;
};
static_assert(sizeof(SamplerVoiceState) == 48);

// A DSP callback supplied by the game (libSceSndx implements its compressor, gain stages and
// such this way). Called once per render with the voice's channels, one buffer per channel.
struct UserFxContext {
    float** channels; // the main input's channels, then (from index 8) the sidechain's
    u64 user_data[3];
    u32 flags; // 0x1: first call (reset the effect's state)
    u32 num_channels;
    u32 num_samples;
    u32 sample_rate;
};
static_assert(sizeof(UserFxContext) == 0x30);

using UserFxHandler = PS4_SYSV_ABI s32 (*)(UserFxContext*);
constexpr u32 UserFxFirstCall = 0x1;

struct UserFx {
    UserFxHandler handler = nullptr;
    u64 user_data[3] = {};
    bool started = false;
};

struct Rack;
struct System;

// Decodes a sampler voice's waveform blocks, one sample frame (all channels) at a time.
class WaveformReader {
public:
    ~WaveformReader() {
        ReleaseDecoder();
    }

    void Setup(const WaveformFormat& format) {
        format_ = format;
        Clear();
    }

    void SetBlocks(const u8* data, std::span<const WaveformBlock> blocks, bool append) {
        if (!append) {
            Clear();
        }
        data_ = data;
        for (const WaveformBlock& block : blocks) {
            blocks_.push_back({data, block});
        }
    }

    void Rewind() {
        block_index_ = 0;
        block_started_ = false;
        decoded_samples_ = 0;
        decoded_bytes_ = 0;
    }

    const WaveformFormat& Format() const {
        return format_;
    }

    u32 Channels() const {
        return std::clamp<u32>(format_.channels, 1, MaxChannels);
    }

    u32 SampleRate() const {
        return format_.sample_rate ? format_.sample_rate : OutputRate;
    }

    bool HasData() const {
        return block_index_ < blocks_.size();
    }

    // Writes Channels() samples; false when the blocks are exhausted.
    bool Next(float* out) {
        while (block_index_ < blocks_.size()) {
            if (!block_started_ && !StartBlock()) {
                NextBlock();
                continue;
            }
            if (remaining_ > 0 && ReadSample(out)) {
                remaining_--;
                decoded_samples_++;
                return true;
            }
            // Block done: repeat it or move on.
            const WaveformBlock& block = blocks_[block_index_].block;
            if (block.num_repeats == RepeatForever || repeats_done_ < block.num_repeats) {
                repeats_done_++;
                block_started_ = false;
                const u32 repeats = repeats_done_;
                if (!StartBlock()) {
                    NextBlock();
                    continue;
                }
                repeats_done_ = repeats;
                continue;
            }
            NextBlock();
        }
        return false;
    }

    const WaveformBlock* CurrentBlock() const {
        return block_index_ < blocks_.size() ? &blocks_[block_index_].block : nullptr;
    }
    const void* CurrentData() const {
        return block_index_ < blocks_.size() ? blocks_[block_index_].data : data_;
    }
    u64 DecodedSamples() const {
        return decoded_samples_;
    }
    u64 DecodedBytes() const {
        return decoded_bytes_;
    }

private:
    struct QueuedBlock {
        const u8* data;
        WaveformBlock block;
    };

    void Clear() {
        blocks_.clear();
        Rewind();
    }

    void NextBlock() {
        block_index_++;
        block_started_ = false;
        repeats_done_ = 0;
    }

    void ReleaseDecoder() {
        if (atrac9_) {
            Atrac9ReleaseHandle(atrac9_);
            atrac9_ = nullptr;
        }
    }

    bool StartBlock() {
        const QueuedBlock& queued = blocks_[block_index_];
        if (!queued.data || queued.block.data_size == 0) {
            return false;
        }
        cursor_ = queued.data + queued.block.data_offset;
        end_ = cursor_ + queued.block.data_size;
        remaining_ = queued.block.num_samples;
        frame_.clear();
        frame_pos_ = 0;
        frames_in_superframe_ = 0;
        std::memset(vag_history_, 0, sizeof(vag_history_));
        superframe_start_ = cursor_;
        if (format_.type == WaveformAtrac9 && !StartAtrac9()) {
            return false;
        }
        block_started_ = true;
        // Encoder delay and other leading samples the block says to drop.
        float discard[MaxChannels];
        for (u32 i = 0; i < queued.block.num_skip_samples; i++) {
            if (!ReadSample(discard)) {
                break;
            }
        }
        return true;
    }

    bool StartAtrac9() {
        ReleaseDecoder();
        atrac9_ = Atrac9GetHandle();
        u8 config[4] = {
            static_cast<u8>(format_.config_data >> 24), static_cast<u8>(format_.config_data >> 16),
            static_cast<u8>(format_.config_data >> 8), static_cast<u8>(format_.config_data)};
        if (Atrac9InitDecoder(atrac9_, config) != 0 ||
            Atrac9GetCodecInfo(atrac9_, &codec_) != 0) {
            LOG_WARNING("Audio", "Ngs2: invalid ATRAC9 configuration {:#010x}", format_.config_data);
            ReleaseDecoder();
            return false;
        }
        return true;
    }

    // One sample frame from the current block, without counting it against num_samples.
    bool ReadSample(float* out) {
        const u32 channels = Channels();
        switch (format_.type) {
        case WaveformAtrac9: {
            if (frame_pos_ >= frame_.size() && !DecodeAtrac9Frame()) {
                return false;
            }
            const u32 decoded_channels = static_cast<u32>(codec_.channels);
            for (u32 ch = 0; ch < channels; ch++) {
                out[ch] = ch < decoded_channels ? frame_[frame_pos_ + ch] : 0.0f;
            }
            frame_pos_ += decoded_channels;
            return true;
        }
        case WaveformPcmI16: {
            if (cursor_ + 2 * channels > end_) {
                return false;
            }
            for (u32 ch = 0; ch < channels; ch++) {
                s16 value;
                std::memcpy(&value, cursor_ + 2 * ch, 2);
                out[ch] = value / 32768.0f;
            }
            cursor_ += 2 * channels;
            decoded_bytes_ += 2 * channels;
            return true;
        }
        case WaveformVag: {
            if (frame_pos_ >= frame_.size() && !DecodeVagFrame()) {
                return false;
            }
            std::copy_n(frame_.data() + frame_pos_, channels, out);
            frame_pos_ += channels;
            return true;
        }
        case WaveformPcmF32: {
            if (cursor_ + 4 * channels > end_) {
                return false;
            }
            std::memcpy(out, cursor_, 4 * channels);
            cursor_ += 4 * channels;
            decoded_bytes_ += 4 * channels;
            return true;
        }
        default: {
            static bool warned = false;
            if (!warned) {
                warned = true;
                LOG_WARNING("Audio", "Ngs2: unsupported waveform {:#x} (stays silent)", format_.type);
            }
            return false;
        }
        }
    }

    // HE-VAG (Sony ADPCM, VAG's PS Vita/PS4 successor): 16-byte frames of 28 samples, one frame
    // per channel in turn. Byte 0: shift (low nibble) and filter bits 0-3 (high nibble); byte 1:
    // filter bits 4-6 (high nibble) and flags (low nibble, 7 = silent frame). Each filter weighs
    // the last four samples.
    bool DecodeVagFrame() {
        const u32 channels = Channels();
        if (cursor_ + 16 * channels > end_) {
            return false;
        }
        frame_.assign(static_cast<size_t>(28) * channels, 0.0f);
        for (u32 ch = 0; ch < channels; ch++) {
            const u8* frame = cursor_ + 16 * ch;
            const u32 shift = frame[0] & 0x0F;
            u32 filter = (frame[1] & 0xF0) | (frame[0] >> 4);
            if (filter > 127) {
                filter = 0;
            }
            const u32 flags = frame[1] & 0x0F;
            const float* coefs = hevag_coefs_f[filter];
            s32* history = vag_history_[ch];
            for (u32 i = 0; i < 28; i++) {
                s32 sample = 0;
                if (flags != 7) {
                    const u8 byte = frame[2 + i / 2];
                    s32 nibble = (i & 1) ? (byte >> 4) : (byte & 0x0F);
                    if (nibble >= 8) {
                        nibble -= 16;
                    }
                    const s32 code = (nibble << 12) >> shift;
                    sample = code + static_cast<s32>(history[0] * coefs[0] + history[1] * coefs[1] +
                                                     history[2] * coefs[2] + history[3] * coefs[3]);
                }
                history[3] = history[2];
                history[2] = history[1];
                history[1] = history[0];
                history[0] = sample;
                frame_[i * channels + ch] = std::clamp(sample, -32768, 32767) / 32768.0f;
            }
        }
        cursor_ += 16 * channels;
        decoded_bytes_ += 16 * channels;
        frame_pos_ = 0;
        return true;
    }

    bool DecodeAtrac9Frame() {
        if (!atrac9_ || cursor_ >= end_) {
            return false;
        }
        frame_.resize(static_cast<size_t>(codec_.frameSamples) * codec_.channels);
        int used = 0;
        if (Atrac9DecodeF32(atrac9_, cursor_, static_cast<int>(end_ - cursor_), frame_.data(), &used,
                            0) != 0) {
            return false;
        }
        cursor_ += used;
        decoded_bytes_ += used;
        frame_pos_ = 0;
        // A superframe may be padded after its last frame. The padding counts as consumed data:
        // libSceSndx compares the consumed size with what it queued to know when to stream more.
        if (++frames_in_superframe_ == codec_.framesInSuperframe) {
            frames_in_superframe_ = 0;
            const u8* next = std::max(cursor_, superframe_start_ + codec_.superframeSize);
            decoded_bytes_ += next - cursor_;
            cursor_ = next;
            superframe_start_ = cursor_;
        }
        return true;
    }

    WaveformFormat format_{};
    const u8* data_ = nullptr;
    std::vector<QueuedBlock> blocks_;
    size_t block_index_ = 0;
    bool block_started_ = false;
    u32 repeats_done_ = 0;
    u32 remaining_ = 0;
    const u8* cursor_ = nullptr;
    const u8* end_ = nullptr;
    u64 decoded_samples_ = 0;
    u64 decoded_bytes_ = 0;

    void* atrac9_ = nullptr;
    Atrac9CodecInfo codec_{};
    std::vector<float> frame_;
    size_t frame_pos_ = 0;
    int frames_in_superframe_ = 0;
    const u8* superframe_start_ = nullptr;

    s32 vag_history_[MaxChannels][4] = {};
};

struct Voice {
    Rack* rack;
    u32 id;
    u32 flags = 0;

    struct Port {
        float volume = 1.0f;
        s32 matrix = -1;
        Voice* destination = nullptr;
        u32 destination_input = 0; // 0: main input, 1: sidechain (for compressors)
    };
    Port ports[MaxPorts];
    std::unordered_map<s32, std::vector<float>> matrices;

    // Non-sampler voices: channels in and out, the input mixed this render, and the game's DSP.
    u32 io_channels = MaxChannels;
    std::vector<float> input;
    std::vector<float> sidechain;
    UserFx user_fx;

    // Sampler voices.
    WaveformReader reader;
    float pitch = 1.0f;
    double position = 0.0;    // fraction between `previous` and `current`
    float previous[MaxChannels] = {};
    float current[MaxChannels] = {};
    bool primed = false;
};

struct Rack {
    System* system;
    u32 id;
    u32 max_voices;
    std::string name;
    std::vector<std::unique_ptr<Voice>> voices;
};

struct System {
    std::mutex mutex;
    std::vector<Rack*> racks;
    std::unordered_set<const void*> voices; // every handle given out, to validate patches
    std::vector<float> scratch;
    std::vector<float> planar;
    u64 renders = 0;
};

bool IsSampler(const Rack& rack) {
    return rack.id == RackSampler || rack.id == RackCustomSampler;
}

bool IsMastering(const Rack& rack) {
    return rack.id == RackMastering || rack.id == RackCustomMastering;
}

// Effect-only racks, silent for now.
bool IsEffectSend(const Rack& rack) {
    return rack.id == RackReverb || rack.id == RackCustomSubmixer;
}

template <typename T>
const T& Body(const ParamHeader* param) {
    return *reinterpret_cast<const T*>(param + 1);
}

void HandleEvent(Voice& voice, u32 event) {
    const bool sampler = IsSampler(*voice.rack);
    switch (event) {
    case EventPlay:
        voice.reader.Rewind();
        voice.primed = false;
        voice.position = 0.0;
        voice.flags = StateInUse | StatePlaying;
        break;
    case EventPause:
        voice.flags |= StatePaused;
        break;
    case EventResume:
    case EventResumeB:
        voice.flags &= ~StatePaused;
        break;
    case EventStop:
    case EventStopImmediate:
    case EventKill:
        voice.flags = sampler ? 0 : StateInUse;
        break;
    default:
        LOG_WARNING("Audio", "Ngs2: unknown event {}", event);
        break;
    }
}

Voice& GetVoice(Rack& rack, u32 voice_id) {
    if (rack.voices.size() <= voice_id) {
        rack.voices.resize(voice_id + 1);
    }
    if (!rack.voices[voice_id]) {
        rack.voices[voice_id] = std::make_unique<Voice>();
        rack.voices[voice_id]->rack = &rack;
        rack.voices[voice_id]->id = voice_id;
        rack.system->voices.insert(rack.voices[voice_id].get());
    }
    return *rack.voices[voice_id];
}

// libSceSndx patches to rack handles (its submixer racks have one voice each); a voice handle
// works too. Anything else unpatches.
Voice* ResolveDestination(const Voice& source, const void* handle) {
    if (!handle) {
        return nullptr;
    }
    System& system = *source.rack->system;
    if (system.voices.contains(handle)) {
        return const_cast<Voice*>(static_cast<const Voice*>(handle));
    }
    for (Rack* rack : system.racks) {
        if (rack == handle) {
            return &GetVoice(*rack, 0);
        }
    }
    static int warnings = 0;
    if (warnings++ < 10) {
        LOG_WARNING("Audio", "Ngs2: patch from voice {} of rack {:#x} to an unknown destination {}",
                    source.id, source.rack->id, handle);
    }
    return nullptr;
}

void ApplyParam(Voice& voice, const ParamHeader* param) {
    switch (param->id) {
    case ParamMatrixLevels: {
        struct Levels {
            s32 matrix;
            u32 count;
            const float* levels;
        };
        const auto& body = Body<Levels>(param);
        if (body.levels && body.count <= MaxChannels * MaxChannels) {
            voice.matrices[body.matrix].assign(body.levels, body.levels + body.count);
        }
        break;
    }
    case ParamPortVolume: {
        struct PortVolume {
            u32 port;
            float level;
        };
        const auto& body = Body<PortVolume>(param);
        if (body.port < MaxPorts) {
            voice.ports[body.port].volume = body.level;
        }
        break;
    }
    case ParamPortMatrix: {
        struct PortMatrix {
            u32 port;
            s32 matrix;
        };
        const auto& body = Body<PortMatrix>(param);
        if (body.port < MaxPorts) {
            voice.ports[body.port].matrix = body.matrix;
        }
        break;
    }
    case ParamPatch: {
        struct Patch {
            u32 port;
            u32 destination_input;
            Voice* destination;
        };
        const auto& body = Body<Patch>(param);
        if (body.port < MaxPorts) {
            voice.ports[body.port].destination = ResolveDestination(voice, body.destination);
            voice.ports[body.port].destination_input = body.destination_input;
        }
        break;
    }
    case ParamEvent:
        HandleEvent(voice, Body<u32>(param));
        break;
    case SamplerSetup:
        voice.reader.Setup(Body<WaveformFormat>(param));
        voice.primed = false;
        voice.position = 0.0;
        break;
    case SamplerWaveformBlocks: {
        struct Blocks {
            const u8* data;
            u32 flags;
            u32 count;
            const WaveformBlock* blocks;
        };
        const auto& body = Body<Blocks>(param);
        // The game follows the real list with an empty one; that changes nothing.
        if (!body.blocks || body.count == 0) {
            break;
        }
        // While a voice plays, new blocks continue the stream; otherwise they replace it.
        const bool append = (voice.flags & StatePlaying) && voice.reader.HasData();
        voice.reader.SetBlocks(body.data,
                               body.blocks ? std::span(body.blocks, body.count)
                                           : std::span<const WaveformBlock>{},
                               append);
        if (voice.flags & StateInUse) {
            voice.flags &= ~StateEmpty;
        }
        break;
    }
    case SamplerPitch:
        voice.pitch = Body<float>(param);
        break;
    case SubmixerUserFx: {
        struct Fx {
            UserFxHandler handler;
            u64 user_data[3];
        };
        const auto& body = Body<Fx>(param);
        voice.user_fx.handler = body.handler;
        std::copy_n(body.user_data, 3, voice.user_fx.user_data);
        voice.user_fx.started = false;
        break;
    }
    case SubmixerSetup:
    case MasteringSetup:
    case ReverbSetup:
    case CustomSubmixerSetup:
        voice.io_channels = std::clamp<u32>(Body<u32>(param), 1, MaxChannels);
        voice.flags = StateInUse;
        break;
    default:
        // Envelopes, filters, compressors, EQ, reverb and delay settings, limiters: not modeled.
        break;
    }
}

// Mixes `frames` frames of `in` (in_channels wide) into `out` (out_channels wide).
void MixInto(const float* in, u32 in_channels, float* out, u32 out_channels, u32 frames,
             float volume, const std::vector<float>* matrix) {
    if (volume == 0.0f) {
        return;
    }
    if (matrix && matrix->size() == static_cast<size_t>(in_channels) * out_channels) {
        // One row of input levels per output channel (a 6-in, 8-out identity has its ones at
        // 0, 7, 14...).
        const float* levels = matrix->data();
        for (u32 f = 0; f < frames; f++) {
            for (u32 o = 0; o < out_channels; o++) {
                float sum = 0.0f;
                for (u32 i = 0; i < in_channels; i++) {
                    sum += in[f * in_channels + i] * levels[o * in_channels + i];
                }
                out[f * out_channels + o] += sum * volume;
            }
        }
        return;
    }
    if (in_channels == 1 && out_channels > 1) {
        // Mono to the front pair.
        for (u32 f = 0; f < frames; f++) {
            const float sample = in[f] * volume * 0.7071f;
            out[f * out_channels] += sample;
            out[f * out_channels + 1] += sample;
        }
        return;
    }
    if (out_channels == 1 && in_channels > 1) {
        // Down to mono from the front pair.
        for (u32 f = 0; f < frames; f++) {
            out[f] += (in[f * in_channels] + in[f * in_channels + 1]) * 0.5f * volume;
        }
        return;
    }
    const u32 channels = std::min(in_channels, out_channels);
    for (u32 f = 0; f < frames; f++) {
        for (u32 c = 0; c < channels; c++) {
            out[f * out_channels + c] += in[f * in_channels + c] * volume;
        }
    }
}

bool Accepts(const Voice* voice) {
    return voice && !IsSampler(*voice->rack) && (voice->flags & StateInUse) &&
           !voice->input.empty();
}

// Sends a voice's output through its patched ports.
void Route(Voice& voice, const float* out, u32 channels, u32 frames) {
    for (const Voice::Port& port : voice.ports) {
        if (!Accepts(port.destination)) {
            continue;
        }
        const auto matrix = voice.matrices.find(port.matrix);
        Voice& destination = *port.destination;
        float* target = port.destination_input == 1 ? destination.sidechain.data()
                                                    : destination.input.data();
        MixInto(out, channels, target, destination.io_channels, frames, port.volume,
                matrix != voice.matrices.end() ? &matrix->second : nullptr);
    }
}

// Renders a playing sampler voice into `out` (its channel count wide).
void RenderSampler(Voice& voice, float* out, u32 frames) {
    const u32 channels = voice.reader.Channels();
    const double step = voice.pitch * voice.reader.SampleRate() / OutputRate;
    if (!voice.primed) {
        if (!voice.reader.Next(voice.previous) || !voice.reader.Next(voice.current)) {
            std::fill_n(voice.current, MaxChannels, 0.0f);
        }
        voice.primed = true;
        voice.position = 0.0;
    }
    bool ended = false;
    for (u32 f = 0; f < frames; f++) {
        const float t = static_cast<float>(voice.position);
        for (u32 c = 0; c < channels; c++) {
            out[f * channels + c] = voice.previous[c] + (voice.current[c] - voice.previous[c]) * t;
        }
        voice.position += step;
        while (voice.position >= 1.0) {
            voice.position -= 1.0;
            std::copy_n(voice.current, MaxChannels, voice.previous);
            if (!ended && !voice.reader.Next(voice.current)) {
                ended = true;
            }
            if (ended) {
                std::fill_n(voice.current, MaxChannels, 0.0f);
            }
        }
    }
    if (ended) {
        // Out of data: the voice is done.
        voice.flags = 0;
    }
}

// Runs the game's DSP callback on a mixer voice's input, in place.
void RunUserFx(System& system, Voice& voice, u32 frames) {
    UserFx& fx = voice.user_fx;
    if (!fx.handler) {
        return;
    }
    const u32 channels = voice.io_channels;
    system.planar.resize(static_cast<size_t>(frames) * 2 * MaxChannels);
    float* pointers[2 * MaxChannels];
    for (u32 c = 0; c < 2 * MaxChannels; c++) {
        pointers[c] = system.planar.data() + static_cast<size_t>(c) * frames;
    }
    for (u32 c = 0; c < channels; c++) {
        for (u32 f = 0; f < frames; f++) {
            pointers[c][f] = voice.input[f * channels + c];
            pointers[MaxChannels + c][f] = voice.sidechain[f * channels + c];
        }
    }
    UserFxContext context{};
    context.channels = pointers;
    std::copy_n(fx.user_data, 3, context.user_data);
    context.flags = fx.started ? 0 : UserFxFirstCall;
    context.num_channels = channels;
    context.num_samples = frames;
    context.sample_rate = OutputRate;
    fx.started = true;
    fx.handler(&context);
    for (u32 c = 0; c < channels; c++) {
        for (u32 f = 0; f < frames; f++) {
            voice.input[f * channels + c] = pointers[c][f];
        }
    }
}

void Render(System& system, const RenderBufferInfo* buffers, u32 count) {
    // All voices render the same number of frames: the first buffer's.
    u32 frames = 256;
    if (count > 0 && buffers[0].channels > 0) {
        frames = static_cast<u32>(buffers[0].size / (4 * buffers[0].channels));
    }
    frames = std::clamp<u32>(frames, 1, 4096);

    std::vector<Voice*> mixers;
    for (Rack* rack : system.racks) {
        for (auto& voice : rack->voices) {
            if (voice && !IsSampler(*rack) && (voice->flags & StateInUse)) {
                voice->input.assign(static_cast<size_t>(frames) * voice->io_channels, 0.0f);
                voice->sidechain.assign(voice->input.size(), 0.0f);
                mixers.push_back(voice.get());
            } else if (voice) {
                voice->input.clear();
                voice->sidechain.clear();
            }
        }
    }

    // Samplers first.
    system.scratch.resize(static_cast<size_t>(frames) * MaxChannels);
    for (Rack* rack : system.racks) {
        if (!IsSampler(*rack)) {
            continue;
        }
        for (auto& voice : rack->voices) {
            if (!voice || (voice->flags & (StatePlaying | StatePaused)) != StatePlaying) {
                continue;
            }
            RenderSampler(*voice, system.scratch.data(), frames);
            Route(*voice, system.scratch.data(), voice->reader.Channels(), frames);
        }
    }

    // Then the mixers, each after everything that feeds it.
    std::unordered_map<Voice*, u32> pending;
    for (Voice* voice : mixers) {
        pending.try_emplace(voice, 0);
        for (const Voice::Port& port : voice->ports) {
            if (Accepts(port.destination)) {
                pending[port.destination]++;
            }
        }
    }
    std::vector<Voice*> ready;
    for (Voice* voice : mixers) {
        if (pending[voice] == 0) {
            ready.push_back(voice);
        }
    }
    size_t processed = 0;
    while (!ready.empty()) {
        Voice* voice = ready.back();
        ready.pop_back();
        processed++;
        if (!IsEffectSend(*voice->rack)) {
            RunUserFx(system, *voice, frames);
        }
        if (!IsMastering(*voice->rack) && !IsEffectSend(*voice->rack)) {
            Route(*voice, voice->input.data(), voice->io_channels, frames);
        }
        for (const Voice::Port& port : voice->ports) {
            if (Accepts(port.destination) && --pending[port.destination] == 0) {
                ready.push_back(port.destination);
            }
        }
    }
    if (processed != mixers.size() && system.renders == 0) {
        LOG_WARNING("Audio", "Ngs2: the mixing graph has a cycle; {} voices left out",
                    mixers.size() - processed);
    }

    // Mastering voice i is render buffer i.
    Rack* mastering = nullptr;
    for (Rack* rack : system.racks) {
        if (IsMastering(*rack)) {
            mastering = rack;
            break;
        }
    }
    for (u32 i = 0; i < count; i++) {
        const RenderBufferInfo& info = buffers[i];
        if (!info.buffer) {
            continue;
        }
        std::memset(info.buffer, 0, info.size);
        if (info.waveform_type != WaveformPcmF32 || info.channels == 0) {
            continue;
        }
        Voice* voice = mastering && i < mastering->voices.size() ? mastering->voices[i].get() : nullptr;
        if (!voice || voice->input.empty()) {
            continue;
        }
        const u32 out_frames = std::min<u32>(frames, static_cast<u32>(info.size / (4 * info.channels)));
        auto* out = static_cast<float*>(info.buffer);
        MixInto(voice->input.data(), voice->io_channels, out, info.channels, out_frames, 1.0f, nullptr);
        for (u32 s = 0; s < out_frames * info.channels; s++) {
            out[s] = std::clamp(out[s], -1.0f, 1.0f);
        }
    }
    system.renders++;
}

PS4_SYSV_ABI s32 sceNgs2SystemCreateWithAllocator(const void*, const void*, System** out) {
    if (!out) {
        return ErrorInvalidAddress;
    }
    *out = new System{};
    LOG_INFO("Audio", "Ngs2 initialized");
    return 0;
}

PS4_SYSV_ABI s32 sceNgs2SystemDestroy(System* system, ContextBufferInfo*) {
    delete system;
    return 0;
}

PS4_SYSV_ABI s32 sceNgs2SystemRender(System* system, const RenderBufferInfo* buffers, u32 count) {
    if (!system) {
        return ErrorInvalidSystem;
    }
    if (!buffers && count > 0) {
        return ErrorInvalidAddress;
    }
    std::scoped_lock lock{system->mutex};
    Render(*system, buffers, count);
    return 0;
}

// What a rack would need for its buffers; the game allocates it, we keep our own state.
constexpr u64 RackBufferSize = 64 * 1024;

PS4_SYSV_ABI s32 sceNgs2RackQueryBufferSize(u32, const void*, ContextBufferInfo* out) {
    if (!out) {
        return ErrorInvalidAddress;
    }
    std::memset(out, 0, sizeof(*out));
    out->host_buffer_size = RackBufferSize;
    return 0;
}

// Rack options: size (8 bytes), name (16), flags, max grain samples, max voices, ...
struct RackOptions {
    u64 size;
    char name[16];
    u32 flags;
    u32 max_grain_samples;
    u32 max_voices;
};

PS4_SYSV_ABI s32 sceNgs2RackCreate(System* system, u32 rack_id, const RackOptions* options,
                                   const ContextBufferInfo*, Rack** out) {
    if (!system) {
        return ErrorInvalidSystem;
    }
    if (!out) {
        return ErrorInvalidAddress;
    }
    const u32 max_voices = options && options->max_voices ? options->max_voices : 1;
    auto* rack = new Rack{system, rack_id, max_voices,
                          options ? std::string(options->name, strnlen(options->name, 16)) : "", {}};
    {
        std::scoped_lock lock{system->mutex};
        system->racks.push_back(rack);
    }
    *out = rack;
    LOG_DEBUG("Audio", "Ngs2: rack {:#x} \"{}\" with {} voices at {}", rack_id,
              options ? std::string(options->name, strnlen(options->name, 16)) : "",
              max_voices, static_cast<void*>(rack));
    return 0;
}

PS4_SYSV_ABI s32 sceNgs2RackCreateWithAllocator(System* system, u32 rack_id,
                                                const RackOptions* options, const void*, Rack** out) {
    return sceNgs2RackCreate(system, rack_id, options, nullptr, out);
}

PS4_SYSV_ABI s32 sceNgs2RackDestroy(Rack* rack, ContextBufferInfo*) {
    if (!rack) {
        return ErrorInvalidRack;
    }
    System& system = *rack->system;
    std::scoped_lock lock{system.mutex};
    // Nothing may keep sending to the voices going away.
    for (Rack* other : system.racks) {
        for (auto& voice : other->voices) {
            if (!voice) {
                continue;
            }
            for (Voice::Port& port : voice->ports) {
                if (port.destination && port.destination->rack == rack) {
                    port.destination = nullptr;
                }
            }
        }
    }
    for (auto& voice : rack->voices) {
        system.voices.erase(voice.get());
    }
    std::erase(system.racks, rack);
    delete rack;
    return 0;
}

PS4_SYSV_ABI s32 sceNgs2RackGetVoiceHandle(Rack* rack, u32 voice_id, Voice** out) {
    if (!rack) {
        return ErrorInvalidRack;
    }
    if (!out) {
        return ErrorInvalidAddress;
    }
    std::scoped_lock lock{rack->system->mutex};
    *out = &GetVoice(*rack, voice_id);
    return 0;
}

PS4_SYSV_ABI s32 sceNgs2VoiceControl(Voice* voice, const ParamHeader* param) {
    if (!voice) {
        return ErrorInvalidVoice;
    }
    std::scoped_lock lock{voice->rack->system->mutex};
    for (int i = 0; param && i < 256; i++) {
        ApplyParam(*voice, param);
        if (param->next == 0) {
            break;
        }
        param = reinterpret_cast<const ParamHeader*>(reinterpret_cast<const u8*>(param) + param->next);
    }
    return 0;
}

PS4_SYSV_ABI s32 sceNgs2VoiceGetState(Voice* voice, void* state, u64 size) {
    if (!voice) {
        return ErrorInvalidVoice;
    }
    if (!state) {
        return ErrorInvalidAddress;
    }
    std::scoped_lock lock{voice->rack->system->mutex};
    SamplerVoiceState full{};
    full.flags = voice->flags;
    if (IsSampler(*voice->rack) && (voice->flags & StateInUse) && !voice->reader.HasData()) {
        full.flags |= StateEmpty;
    }
    full.envelope_height = 1.0f;
    full.peak_height = 0.0f;
    full.num_decoded_samples = voice->reader.DecodedSamples();
    full.decoded_data_size = voice->reader.DecodedBytes();
    if (const WaveformBlock* block = voice->reader.CurrentBlock()) {
        full.user_data = block->user_data;
    }
    full.waveform_data = voice->reader.CurrentData();
    std::memcpy(state, &full, std::min<u64>(size, sizeof(full)));
    if (size > sizeof(full)) {
        std::memset(static_cast<u8*>(state) + sizeof(full), 0, size - sizeof(full));
    }
    return 0;
}

PS4_SYSV_ABI s32 sceNgs2VoiceGetStateFlags(Voice* voice, u32* flags) {
    if (!voice) {
        return ErrorInvalidVoice;
    }
    if (!flags) {
        return ErrorInvalidAddress;
    }
    std::scoped_lock lock{voice->rack->system->mutex};
    *flags = voice->flags;
    return 0;
}

} // namespace

HLE_FUNCTION("libSceNgs2", "sceNgs2SystemCreateWithAllocator", sceNgs2SystemCreateWithAllocator);
HLE_FUNCTION("libSceNgs2", "sceNgs2SystemDestroy", sceNgs2SystemDestroy);
HLE_FUNCTION("libSceNgs2", "sceNgs2SystemRender", sceNgs2SystemRender);
HLE_FUNCTION("libSceNgs2", "sceNgs2RackQueryBufferSize", sceNgs2RackQueryBufferSize);
HLE_FUNCTION("libSceNgs2", "sceNgs2RackCreate", sceNgs2RackCreate);
HLE_FUNCTION("libSceNgs2", "sceNgs2RackCreateWithAllocator", sceNgs2RackCreateWithAllocator);
HLE_FUNCTION("libSceNgs2", "sceNgs2RackDestroy", sceNgs2RackDestroy);
HLE_FUNCTION("libSceNgs2", "sceNgs2RackGetVoiceHandle", sceNgs2RackGetVoiceHandle);
HLE_FUNCTION("libSceNgs2", "sceNgs2VoiceControl", sceNgs2VoiceControl);
HLE_FUNCTION("libSceNgs2", "sceNgs2VoiceGetState", sceNgs2VoiceGetState);
HLE_FUNCTION("libSceNgs2", "sceNgs2VoiceGetStateFlags", sceNgs2VoiceGetStateFlags);

} // namespace hle::audio
