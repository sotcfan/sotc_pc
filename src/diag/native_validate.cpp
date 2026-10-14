#include "common/platform.h"
#include "diag/native_validate.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <tuple>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/log.h"
#include "gpu/gpu.h"
#include "hle/kernel/gpu_memory.h"
#include "render/native_compute.h"
#include "render/native_draw.h"

namespace diag {

namespace {

// Which shaders are checked: "all" or a list of hashes.
struct Selection {
    bool on = false;
    bool all = false;
    std::unordered_set<u64> hashes;

    bool Selected(u64 hash) const {
        return all || hashes.contains(hash);
    }
    static Selection From(const char* variable) {
        Selection selection;
        const char* value = std::getenv(variable);
        if (!value || !*value) {
            return selection;
        }
        selection.on = true;
        const std::string list = value;
        if (list == "all") {
            selection.all = true;
            return selection;
        }
        for (size_t at = 0; at < list.size();) {
            const size_t comma = std::min(list.find(',', at), list.size());
            selection.hashes.insert(
                std::strtoull(list.substr(at, comma - at).c_str(), nullptr, 16));
            at = comma + 1;
        }
        return selection;
    }
};

struct Settings {
    Selection compute; // SOTC_VALIDATE_NATIVE: dispatches, by compute shader
    Selection draws;   // SOTC_VALIDATE_NATIVE_DRAWS: draws, by vertex or pixel shader
    u32 per_shader = 3;
    bool dump = false; // SOTC_VALIDATE_NATIVE_DUMP=1: SPIR-V and bindings of each check
    // SOTC_VALIDATE_NATIVE_START=<seconds>: no checks before (the logos run at full speed, so
    // SOTC_KEY_SCRIPT times stay those of a normal session).
    std::chrono::steady_clock::time_point start{};
};

const Settings& GetSettings() {
    static const Settings settings = [] {
        Settings s;
        s.compute = Selection::From("SOTC_VALIDATE_NATIVE");
        s.draws = Selection::From("SOTC_VALIDATE_NATIVE_DRAWS");
        if (const char* dump = std::getenv("SOTC_VALIDATE_NATIVE_DUMP")) {
            s.dump = *dump == '1';
        }
        if (const char* count = std::getenv("SOTC_VALIDATE_NATIVE_COUNT")) {
            s.per_shader = static_cast<u32>(std::max(1L, std::strtol(count, nullptr, 10)));
        }
        // Settings are first read at the first command buffer, about when the game starts.
        s.start = std::chrono::steady_clock::now();
        if (const char* start = std::getenv("SOTC_VALIDATE_NATIVE_START")) {
            s.start += std::chrono::seconds(std::max(0L, std::strtol(start, nullptr, 10)));
        }
        return s;
    }();
    return settings;
}

// Game memory through the mirror that write tracking never protects (gpu_memory.h).
bool ReadMirror(u64 address, void* out, u64 size) {
    if (hle::kernel::ClampToMapped(address, size) < size || !hle::kernel::GuestMirrorOffset()) {
        return false;
    }
    std::memcpy(out, reinterpret_cast<const void*>(address + hle::kernel::GuestMirrorOffset()),
                size);
    return true;
}

// The shader's hash from the OrbShdr footer after its code, by code address.
u64 ShaderHashAt(u64 code) {
    static std::unordered_map<u64, u64> cache;
    if (const auto it = cache.find(code); it != cache.end()) {
        return it->second;
    }
    u64 hash = 0;
    std::vector<u8> bytes(256 * 1024);
    const u64 size = hle::kernel::ClampToMapped(code, bytes.size());
    if (size >= 28 && ReadMirror(code, bytes.data(), size)) {
        for (u64 at = 0; at + 28 <= size; at += 4) {
            if (std::memcmp(bytes.data() + at, "OrbShdr", 7) == 0) {
                std::memcpy(&hash, bytes.data() + at + 16, 8);
                break;
            }
        }
    }
    cache[code] = hash;
    return hash;
}

// Shader file names from the shader survey (tools/shader_survey), for the report.
const std::unordered_map<u64, std::string>& ShaderNames() {
    static const auto names = [] {
        std::unordered_map<u64, std::string> map;
        std::ifstream in{platform::DataDir() / "shader_survey/survey.txt"};
        for (std::string line; std::getline(in, line);) {
            // stage \t hash \t status \t details \t path
            std::vector<std::string> fields;
            for (size_t at = 0; at <= line.size();) {
                const size_t tab = std::min(line.find('\t', at), line.size());
                fields.push_back(line.substr(at, tab - at));
                at = tab + 1;
            }
            if (fields.size() < 5 || fields[1].size() != 16) {
                continue;
            }
            std::string path = fields[4];
            if (const size_t cut = path.find("/shaders/_ps4/"); cut != std::string::npos) {
                path = path.substr(cut + 14);
            }
            map.try_emplace(std::strtoull(fields[1].c_str(), nullptr, 16), path);
        }
        return map;
    }();
    return names;
}

// ~/.local/share/sotcport/native/<hash>_<n>.spv and .txt: the SPIR-V that ran and what it
// was given (disassemble with spirv-dis).
void Dump(const render::ComputeDispatch& dispatch, const render::NativeComputeResult& result,
          u32 attempt) {
    const auto dir = platform::DataDir() / "native";
    std::filesystem::create_directories(dir);
    const auto base = dir / std::format("{:016x}_{}", result.hash, attempt);
    std::ofstream spv{base.string() + ".spv", std::ios::binary};
    spv.write(reinterpret_cast<const char*>(result.program.spirv.data()),
              static_cast<std::streamsize>(result.program.spirv.size() * 4));
    std::ofstream txt{base.string() + ".txt"};
    txt << std::format("code {:#x} rsrc1 {:08x} rsrc2 {:08x} threads {} {} {} groups {} {} {}\n",
                       dispatch.program.code_address, dispatch.program.rsrc1,
                       dispatch.program.rsrc2, dispatch.program.num_threads[0],
                       dispatch.program.num_threads[1], dispatch.program.num_threads[2],
                       dispatch.groups[0], dispatch.groups[1], dispatch.groups[2]);
    txt << "user data";
    for (u32 value : dispatch.program.user_data) {
        txt << std::format(" {:08x}", value);
    }
    txt << "\nflatbuf";
    for (u32 value : result.program.flatbuf) {
        txt << std::format(" {:08x}", value);
    }
    txt << "\n";
    // In translate::BufferKind order.
    constexpr const char* kinds[] = {"guest",        "flatbuf", "lds",        "page-table",
                                     "fault-buffer", "inline",  "unsupported"};
    for (size_t i = 0; i < result.program.buffers.size(); i++) {
        const auto& b = result.program.buffers[i];
        txt << std::format("binding {}: {} {:#x} size {:#x}{}{}\n", i,
                           kinds[static_cast<int>(b.kind)], b.address, b.size,
                           b.written ? " written" : "", b.formatted ? " formatted" : "");
    }
    for (const auto& range : result.written) {
        txt << std::format("written {:#x}: {} bytes, {} changed by the native dispatch\n",
                           range.address, range.bytes.size(), range.changed);
    }
    for (size_t i = 0; i < result.program.image_bindings.size(); i++) {
        const auto& b = result.program.image_bindings[i];
        txt << std::format(
            "image {}: {:#x} type {} {}x{}x{} pitch {} levels {}-{} data/num format {}/{} tile {} "
            "vk format {} view {} format {} swizzle {} {} {} {} levels {}+{} layers {}+{}{}{}{}{}\n",
            i, b.address, b.type, b.width, b.height, b.depth, b.pitch, b.base_level, b.last_level,
            b.data_format, b.number_format, b.tile_mode, b.vk_format, b.view_type, b.view_format,
            b.swizzle[0], b.swizzle[1], b.swizzle[2], b.swizzle[3], b.view_base_level,
            b.view_levels, b.view_base_layer, b.view_layers, b.written ? " written" : "",
            b.storage ? " storage" : "", b.depth_image ? " depth" : "",
            i < result.image_layer_levels.size() && result.image_layer_levels[i]
                ? std::format(" ({} mips from the GPU layer)", result.image_layer_levels[i])
                : std::string{});
    }
    for (size_t i = 0; i < result.program.sampler_bindings.size(); i++) {
        const auto& s = result.program.sampler_bindings[i];
        txt << std::format("sampler {}: filters {}/{} mip {} address {} {} {} lod {}..{} bias {} "
                           "aniso {} {} compare {} {} border {}\n",
                           i, s.mag_filter, s.min_filter, s.mipmap_mode, s.address_u, s.address_v,
                           s.address_w, s.min_lod, s.max_lod, s.lod_bias, s.anisotropy,
                           s.max_anisotropy, s.compare, s.compare_op, s.border);
    }
    // The written images after the native dispatch (the GPU layer's go next to them, see
    // ValidateAfterPacket): linear, ImageBinding::copies layout.
    for (size_t i = 0; i < result.written.size(); i++) {
        std::ofstream bin{base.string() + std::format(".buffer{}.native.bin", i), std::ios::binary};
        bin.write(reinterpret_cast<const char*>(result.written[i].bytes.data()),
                  static_cast<std::streamsize>(result.written[i].bytes.size()));
    }
    for (size_t i = 0; i < result.images.size(); i++) {
        std::ofstream bin{base.string() + std::format(".image{}.native.bin", i), std::ios::binary};
        bin.write(reinterpret_cast<const char*>(result.images[i].bytes.data()),
                  static_cast<std::streamsize>(result.images[i].bytes.size()));
        if (!result.images[i].before.empty()) {
            std::ofstream before{base.string() + std::format(".image{}.before.bin", i),
                                 std::ios::binary};
            before.write(reinterpret_cast<const char*>(result.images[i].before.data()),
                         static_cast<std::streamsize>(result.images[i].before.size()));
        }
    }
}

// The same for a draw: <ps>_<n>.vs.spv / .ps.spv / .txt (state, targets, bindings) and the
// targets after the native draw (.imageN.native.bin; the GPU layer's go next to them).
void DumpDraw(const render::NativeDrawResult& result, u32 attempt) {
    const auto dir = platform::DataDir() / "native";
    std::filesystem::create_directories(dir);
    const auto base = dir / std::format("{:016x}_{}", result.ps_hash, attempt);
    const auto& d = result.draw;
    for (const auto& [suffix, spirv] : {std::pair{".vs.spv", &d.vs.spirv}, {".ps.spv", &d.ps.spirv}}) {
        std::ofstream spv{base.string() + suffix, std::ios::binary};
        spv.write(reinterpret_cast<const char*>(spirv->data()),
                  static_cast<std::streamsize>(spirv->size() * 4));
    }
    std::ofstream txt{base.string() + ".txt"};
    txt << std::format("vs {:016x} ps {:016x}\n", result.vs_hash, result.ps_hash);
    txt << std::format("{} count {} instances {} vertex offset {} instance offset {} index {:#x}{}\n",
                       d.indexed ? "indexed" : "auto", d.count, d.instances, d.vertex_offset,
                       d.instance_offset, d.index_address, d.index32 ? " (32-bit)" : "");
    txt << std::format("topology {} patch {} cull {} front {} polygon {} discard {} clamp {} "
                       "clip {} -1..1 {} provoking last {}\n",
                       d.topology, d.patch_points, d.cull_mode, d.front_face, d.polygon_mode,
                       d.rasterizer_discard, d.depth_clamp, d.depth_clip, d.negative_one_to_one,
                       d.provoking_last);
    txt << std::format("viewport {} {} {} {} depth {}..{} scissor {},{} {}x{} area {}x{} push "
                       "{} {} {} {}\n",
                       d.viewport[0], d.viewport[1], d.viewport[2], d.viewport[3], d.viewport[4],
                       d.viewport[5], d.scissor_offset[0], d.scissor_offset[1],
                       d.scissor_extent[0], d.scissor_extent[1], d.width, d.height,
                       d.push_viewport[0], d.push_viewport[1], d.push_viewport[2],
                       d.push_viewport[3]);
    for (size_t i = 0; i < d.vertex_inputs.size(); i++) {
        const auto& v = d.vertex_inputs[i];
        txt << std::format("vertex input {}: {:#x} size {:#x} stride {} vk format {}{}\n", i,
                           v.address, v.size, v.stride, v.vk_format,
                           v.per_instance ? " per instance" : "");
    }
    for (size_t i = 0; i < d.colors.size(); i++) {
        const auto& c = d.colors[i];
        if (!c.used) {
            continue;
        }
        txt << std::format("color {}: {:#x} {}x{} vk format {} view {} level {} layer {}+{} "
                           "mask {:x} blend {} ({} {} {} / {} {} {})\n",
                           i, c.image.address, c.image.extent[0], c.image.extent[1],
                           c.image.vk_format, c.view_format, c.view_level, c.view_layer,
                           c.view_layers, c.write_mask, c.blend, c.src_color, c.dst_color,
                           c.color_op, c.src_alpha, c.dst_alpha, c.alpha_op);
    }
    if (d.depth.used) {
        const auto& z = d.depth;
        txt << std::format("depth: {:#x} {}x{} vk format {} test {} write {} op {} stencil {} "
                           "(ref {} op {} pass {} fail {} zfail {} masks {:x}/{:x}) clear {} "
                           "bounds {} ({}..{})\n",
                           z.image.address, z.image.extent[0], z.image.extent[1],
                           z.image.vk_format, z.test, z.write, z.compare_op, z.stencil,
                           z.front.reference, z.front.compare_op, z.front.pass_op,
                           z.front.fail_op, z.front.depth_fail_op, z.front.compare_mask,
                           z.front.write_mask, z.clear_pass, z.bounds, z.bounds_min,
                           z.bounds_max);
    }
    constexpr const char* kinds[] = {"guest",        "flatbuf", "lds",        "page-table",
                                     "fault-buffer", "inline",  "unsupported"};
    // Mips each image took from the GPU layer (shader images in binding order, then the
    // targets: ResourceSet prepares them in that order).
    size_t prepared = 0;
    const auto layer_mips = [&] {
        const size_t i = prepared++;
        return i < result.image_layer_levels.size() ? result.image_layer_levels[i] : 0u;
    };
    for (const auto& [stage, program] : {std::pair{"ps", &d.ps}, {"vs", &d.vs}}) {
        if (!program->flatbuf.empty()) {
            txt << stage << " flatbuf:";
            for (const u32 word : program->flatbuf) {
                txt << std::format(" {:08x}", word);
            }
            txt << '\n' << stage << " flatbuf reads:";
            for (const u64 read : program->flatbuf_reads) {
                txt << std::format(" {:#x}", read - hle::kernel::GuestMirrorOffset());
            }
            txt << '\n';
        }
        for (size_t i = 0; i < program->buffers.size(); i++) {
            const auto& b = program->buffers[i];
            txt << std::format("{} buffer {}: {} {:#x} size {:#x}{}{}\n", stage, i,
                               kinds[static_cast<int>(b.kind)], b.address, b.size,
                               b.written ? " written" : "",
                               b.address && b.size && gpu::IsGpuModified(b.address, b.size)
                                   ? " (GPU-modified in the GPU layer)"
                                   : "");
        }
        for (size_t i = 0; i < program->sampler_bindings.size(); i++) {
            const auto& s = program->sampler_bindings[i];
            txt << std::format("{} sampler {}: filter {}/{} mip {} address {}/{}/{} lod bias {} "
                               "lod {}..{} aniso {} ({}) compare {} ({}) border {}\n",
                               stage, i, s.mag_filter, s.min_filter, s.mipmap_mode, s.address_u,
                               s.address_v, s.address_w, s.lod_bias, s.min_lod, s.max_lod,
                               s.anisotropy, s.max_anisotropy, s.compare, s.compare_op,
                               s.border);
        }
        for (size_t i = 0; i < program->image_bindings.size(); i++) {
            const auto& b = program->image_bindings[i];
            txt << std::format("{} image {}: {:#x} {}x{}x{} vk format {} view {}, {} of {} mips "
                               "from the GPU layer\n",
                               stage, i, b.address, b.width, b.height, b.depth, b.vk_format,
                               b.view_format, layer_mips(), b.levels);
            txt << gpu::DescribeLayerImages(b);
        }
    }
    // result.images: the shader images (all of them: dumps compare the inputs too), then the
    // targets; the comparison appends a line per image below ("image N ...").
    const size_t shader_images = d.ps.image_bindings.size() + d.vs.image_bindings.size();
    for (size_t i = 0; i < result.images.size(); i++) {
        const auto& image = result.images[i];
        if (i < shader_images) {
            txt << std::format("image {}: shader image at {:#x}{}\n", i, image.binding.address,
                               image.input ? "" : " (written)");
            continue;
        }
        txt << std::format("image {}: target at {:#x}, {} mips from the GPU layer\n", i,
                           image.binding.address, layer_mips());
        txt << gpu::DescribeLayerImages(image.binding);
    }
    for (size_t i = 0; i < result.images.size(); i++) {
        std::ofstream bin{base.string() + std::format(".image{}.native.bin", i), std::ios::binary};
        bin.write(reinterpret_cast<const char*>(result.images[i].bytes.data()),
                  static_cast<std::streamsize>(result.images[i].bytes.size()));
        if (!result.images[i].before.empty()) {
            std::ofstream before{base.string() + std::format(".image{}.before.bin", i),
                                 std::ios::binary};
            before.write(reinterpret_cast<const char*>(result.images[i].before.data()),
                         static_cast<std::streamsize>(result.images[i].before.size()));
        }
    }
}

struct ShaderStats {
    u64 changed = 0; // bytes the native dispatches changed (0: the check proves little)
    u32 attempts = 0;
    u32 matched = 0;
    u32 mismatched = 0;
    std::string skipped; // the last reason
    std::string error;
    std::string first_mismatch;
    bool clamped = false;
    u32 images_compared = 0;  // written images compared with the GPU layer's (all attempts)
    u32 layer_images = 0;     // input images taken from the GPU layer (all attempts)
    u32 dma_pages = 0;        // most pages given to dynamic loads in one attempt
    u32 dma_missing = 0;      // most pages dynamic loads still lacked at the end
    u32 reordered = 0;        // attempts whose only differences were records in another order
    u32 record_size = 0;      // the record size of the last of those
    std::string image_not_compared; // the last reason a written image could not be compared
    u32 skips = 0; // draws: skipped attempts
    u32 unchanged_retries = 0; // matches that changed nothing, not counted (MaxUnchangedRetries)
    std::chrono::steady_clock::time_point next_try{}; // no attempt before (after such a match)
};

// How many matching runs that changed nothing a shader may discard in favor of later runs, and
// how long it waits before the next one: runs in a row see the same (often empty) scene.
constexpr u32 MaxUnchangedRetries = 30;
constexpr auto UnchangedRetryDelay = std::chrono::seconds(2);

std::mutex g_mutex;
std::map<u64, ShaderStats> g_stats;
// Draws, by (vertex shader, pixel shader).
std::map<std::pair<u64, u64>, ShaderStats> g_draw_stats;
u32 g_draw_model_registers = 0; // registers the last draw took from the model (the rest from the layer)
u64 g_bytes_compared = 0;
u64 g_bytes_changed = 0; // of those, bytes the dispatches changed

struct Pending {
    u64 hash = 0;
    bool draw = false;
    u64 vs_hash = 0, ps_hash = 0; // draws
    u32 attempt = 0; // for the dump file names
    std::vector<render::WrittenRange> written;
    std::vector<render::WrittenImage> images;
};

// When `a` and `b` hold the same records in another order, the record size in bytes (0 if not):
// what atomics that hand out slots (a counter, an append) produce when the GPU runs the threads
// in another order. Records of 4 bytes (values, offsets), 6 (triangles of 16-bit indices), 12
// (of 32-bit ones), 8 and 16 are tried; bytes after the last whole record must be equal.
u32 ReorderedRecordSize(std::span<const u8> a, std::span<const u8> b) {
    if (a.size() != b.size()) {
        return 0;
    }
    for (const u32 size : {4u, 6u, 8u, 12u, 16u}) {
        const size_t whole = a.size() / size * size;
        if (!whole || !std::equal(a.begin() + whole, a.end(), b.begin() + whole)) {
            continue;
        }
        std::vector<std::span<const u8>> x, y;
        for (size_t at = 0; at < whole; at += size) {
            x.push_back(a.subspan(at, size));
            y.push_back(b.subspan(at, size));
        }
        const auto less = [](std::span<const u8> l, std::span<const u8> r) {
            return std::memcmp(l.data(), r.data(), l.size()) < 0;
        };
        std::ranges::sort(x, less);
        std::ranges::sort(y, less);
        if (std::ranges::equal(x, y, [](std::span<const u8> l, std::span<const u8> r) {
                return std::memcmp(l.data(), r.data(), l.size()) == 0;
            })) {
            return size;
        }
    }
    return 0;
}

// Where a byte of a linear image buffer is: "mip m slice s (x, y)" (x, y in elements: texels, or
// 4x4 blocks).
std::string DescribeImageOffset(const render::translate::ImageBinding& b, u64 offset) {
    const u64 dim = b.block ? 4 : 1;
    for (size_t m = b.copies.size(); m-- > 0;) {
        const auto& c = b.copies[m];
        if (c.buffer_offset > offset) {
            continue;
        }
        const u64 row_bytes =
            ((c.row_length ? c.row_length : c.width) + dim - 1) / dim * b.element_bytes;
        const u64 slice_bytes =
            row_bytes * (((c.image_height ? c.image_height : c.height) + dim - 1) / dim);
        const u64 at = offset - c.buffer_offset;
        if (!row_bytes || !slice_bytes) {
            break;
        }
        return std::format("mip {} slice {} ({}, {})", m, at / slice_bytes,
                           at % slice_bytes % row_bytes / std::max(b.element_bytes, 1u),
                           at % slice_bytes / row_bytes);
    }
    return std::format("+{:#x}", offset);
}
std::optional<Pending> g_pending;

// Value of a modelled register in the snapshot (nullopt if the model does not know it).
// A pointer to the pre-dispatch's copy of the LCUE resource table is known to the model only as
// "the table + offset" (its content is proven to be the model's table, CheckTable in the
// recorder); its address, chosen per dispatch, is taken from the GPU layer's register.
std::optional<u32> Value(const render::RegisterSnapshot& snapshot, u32 index) {
    static const auto slots = [] {
        std::unordered_map<u32, size_t> map;
        for (size_t i = 0; i < render::ModelledCount; i++) {
            map[render::ModelledRegisters[i].index] = i;
        }
        return map;
    }();
    const auto it = slots.find(index);
    if (it == slots.end() || !snapshot.known.test(it->second)) {
        return std::nullopt;
    }
    if (snapshot.table_pointer.test(it->second)) {
        return gpu::ReadComputeRegister(index);
    }
    return snapshot.values[it->second];
}

} // namespace

bool NativeValidationEnabled() {
    return GetSettings().compute.on || GetSettings().draws.on;
}

void ValidateDispatchBefore(const render::RegisterSnapshot& snapshot, const u32* packet,
                            bool indirect, bool compute_queue) {
    const Settings& settings = GetSettings();
    constexpr u32 Sh = render::ShBase;
    const auto pgm = Value(snapshot, Sh + 0x20c);
    if (!pgm) {
        return;
    }
    const u64 code = u64{*pgm} << 8;
    const u64 hash = ShaderHashAt(code);
    if (!settings.compute.on || !settings.compute.Selected(hash)) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (now < settings.start) {
        return; // (before the shader gets an entry: the report lists the shaders checked)
    }
    std::scoped_lock lock{g_mutex};
    ShaderStats& stats = g_stats[hash];
    if (stats.attempts >= settings.per_shader || now < stats.next_try) {
        return;
    }
    stats.attempts++;

    render::ComputeDispatch dispatch;
    dispatch.program.code_address = code;
    const auto rsrc1 = Value(snapshot, Sh + 0x212), rsrc2 = Value(snapshot, Sh + 0x213);
    const auto x = Value(snapshot, Sh + 0x207), y = Value(snapshot, Sh + 0x208),
               z = Value(snapshot, Sh + 0x209);
    if (!rsrc1 || !rsrc2 || !x || !y || !z) {
        stats.skipped = "program registers unknown to the model";
        return;
    }
    dispatch.program.rsrc1 = *rsrc1;
    dispatch.program.rsrc2 = *rsrc2;
    dispatch.program.num_threads = {*x, *y, *z};
    // The user data the shader is given must all be known (the state proof says they are).
    const u32 given = (*rsrc2 >> 1) & 0x1f;
    for (u32 i = 0; i < 16; i++) {
        const auto value = Value(snapshot, render::UserDataBase[render::StageCompute] + i);
        if (!value && i < given) {
            stats.skipped = "user data unknown to the model";
            return;
        }
        dispatch.program.user_data[i] = value.value_or(0);
    }
    if (!indirect) {
        // DISPATCH_DIRECT: header, dim_x, dim_y, dim_z, initiator.
        dispatch.groups = {packet[1], packet[2], packet[3]};
    } else {
        // The workgroup counts are in memory, often written by an earlier dispatch.
        u64 args = 0;
        if (compute_queue) {
            args = packet[1] | u64{packet[2] & 0xffff} << 32; // header, address lo, address hi
        } else {
            const auto lo = Value(snapshot, render::IndirectArgsBase);
            const auto hi = Value(snapshot, render::IndirectArgsBase + 1);
            if (!lo || !hi) {
                stats.skipped = "indirect arguments address unknown to the model";
                return;
            }
            args = (u64{*lo} | u64{*hi} << 32) + packet[1]; // header, data offset
        }
        gpu::SyncGuestRange(args, 12);
        if (!ReadMirror(args, dispatch.groups.data(), 12)) {
            stats.error = std::format("could not read the indirect arguments at {:#x}", args);
            return;
        }
        if (!dispatch.groups[0] || !dispatch.groups[1] || !dispatch.groups[2]) {
            stats.attempts--; // an empty dispatch checks nothing: try another one
            return;
        }
    }

    render::NativeComputeResult result;
    result.compare_inputs = settings.dump;
    const auto read = [](u64 address, void* out, u64 size) {
        gpu::SyncGuestRange(address, size); // the GPU layer's results so far
        return ReadMirror(address, out, size);
    };
    // Whenever the GPU layer holds the image, its contents (brought up to date as binding does)
    // are what the layer's run uses: game memory lacks its GPU writes, and some of its own
    // operations (fast clear eliminations) do not even mark the image GPU-written. Our detiling
    // runs for images the layer does not hold yet.
    const auto read_image = [](const render::translate::ImageBinding& binding, u8* out) {
        std::string why;
        return gpu::ReadLayerImage(binding, /*newer_only=*/false, out, why);
    };
    render::RunNativeCompute(dispatch, read, read_image, result);
    if (settings.dump && !result.program.spirv.empty()) {
        // Marks the place in the log: what the GPU layer logs next is about this dispatch.
        LOG_INFO("Validate", "native dispatch {:016x} attempt {} dumped", hash, stats.attempts);
        Dump(dispatch, result, stats.attempts);
        // Is the native run deterministic? The same dispatch once more, with the same inputs.
        render::NativeComputeResult second;
        second.compare_inputs = true;
        render::RunNativeCompute(dispatch, read, read_image, second);
        std::ofstream txt{platform::DataDir() / "native" /
                              std::format("{:016x}_{}.txt", hash, stats.attempts),
                          std::ios::app};
        for (size_t i = 0; i < std::min(result.images.size(), second.images.size()); i++) {
            size_t differing = 0;
            const auto& a = result.images[i].bytes;
            const auto& b = second.images[i].bytes;
            for (size_t j = 0; j < std::min(a.size(), b.size()); j++) {
                differing += a[j] != b[j];
            }
            if (differing) {
                txt << std::format("second native run: image {} differs in {} bytes\n", i,
                                   differing);
            }
        }
        for (size_t i = 0; i < std::min(result.written.size(), second.written.size()); i++) {
            if (result.written[i].bytes != second.written[i].bytes) {
                txt << std::format("second native run: buffer {:#x} differs\n",
                                   result.written[i].address);
            }
        }
        txt << "second native run compared\n";
    }
    stats.clamped |= result.clamped;
    stats.layer_images += result.layer_images;
    stats.dma_pages = std::max(stats.dma_pages, result.dma_pages);
    stats.dma_missing = std::max(stats.dma_missing, result.dma_missing);
    if (!result.skipped.empty()) {
        stats.skipped = result.skipped;
        return;
    }
    if (!result.error.empty()) {
        stats.error = result.error;
        return;
    }
    g_pending = Pending{.hash = hash,
                        .attempt = stats.attempts,
                        .written = std::move(result.written),
                        .images = std::move(result.images)};
}

void ValidateDrawBefore(const render::RegisterSnapshot& snapshot, const u32* packet) {
    const Settings& settings = GetSettings();
    if (!settings.draws.on) {
        return;
    }
    // The registers the draw sees: the GPU layer's register file with the model's values where
    // it knows them (the recorder proves them equal); the rest (not modelled yet) are the
    // layer's.
    render::translate::DrawState state;
    state.regs.resize(render::translate::RegisterCount);
    gpu::ReadRegisterFile(state.regs.data());
    u32 from_model = 0;
    for (size_t i = 0; i < render::ModelledCount; i++) {
        if (snapshot.known.test(i) && !snapshot.excluded.test(i) &&
            !snapshot.table_pointer.test(i)) {
            state.regs[render::ModelledRegisters[i].index] = snapshot.values[i];
            from_model++;
        }
    }
    gpu::ReadRenderTargetExtents(state.cb_extent.data(), state.db_extent);
    // The shaders, by the hash in their OrbShdr footer (SPI_SHADER_PGM_LO/HI_PS and _VS).
    constexpr u32 Sh = render::ShBase;
    const auto program = [&](u32 lo) {
        return u64{state.regs[Sh + lo]} << 8 | u64{state.regs[Sh + lo + 1] & 0xff} << 40;
    };
    const u64 ps_hash = ShaderHashAt(program(0x08));
    const u64 vs_hash = ShaderHashAt(program(0x48));
    if (!settings.draws.Selected(ps_hash) && !settings.draws.Selected(vs_hash)) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (now < settings.start) {
        return; // (before the shader gets an entry: the report lists the shaders checked)
    }
    std::scoped_lock lock{g_mutex};
    ShaderStats& stats = g_draw_stats[{vs_hash, ps_hash}];
    if (stats.attempts >= settings.per_shader || now < stats.next_try) {
        return;
    }
    stats.attempts++;
    g_draw_model_registers = from_model;

    render::NativeDrawResult result;
    result.compare_inputs = settings.dump;
    const auto read = [](u64 address, void* out, u64 size) {
        gpu::SyncGuestRange(address, size); // the GPU layer's results so far
        return ReadMirror(address, out, size);
    };
    // Whenever the GPU layer holds the image, its contents (brought up to date as binding does)
    // are what the layer's run uses: game memory lacks its GPU writes, and some of its own
    // operations (fast clear eliminations) do not even mark the image GPU-written. Our detiling
    // runs for images the layer does not hold yet.
    const auto read_image = [](const render::translate::ImageBinding& binding, u8* out) {
        std::string why;
        return gpu::ReadLayerImage(binding, /*newer_only=*/false, out, why);
    };
    const auto fast_cleared = [](u64 meta_address, u32 slice, bool tracked) {
        return gpu::IsFastCleared(meta_address, slice, tracked);
    };
    const auto read_stencil = [](const render::translate::ImageBinding& binding,
                                 std::vector<u8>& out) {
        std::string why;
        return gpu::ReadLayerStencil(binding, out.data(), why);
    };
    render::RunNativeDraw(state, packet, read, read_image, read_stencil, fast_cleared, result);
    if (settings.dump && !result.draw.ps.spirv.empty() && result.skipped.empty()) {
        // Marks the place in the log: what the GPU layer logs next is about this draw.
        LOG_INFO("Validate", "native draw vs {:016x} ps {:016x} attempt {} dumped", vs_hash,
                 ps_hash, stats.attempts);
        DumpDraw(result, stats.attempts);
        // Is the native run deterministic? The same draw once more, with the same inputs.
        {
            render::NativeDrawResult second;
            second.compare_inputs = true;
            render::RunNativeDraw(state, packet, read, read_image, read_stencil, fast_cleared,
                                  second);
            std::ofstream txt{platform::DataDir() / "native" /
                                  std::format("{:016x}_{}.txt", ps_hash, stats.attempts),
                              std::ios::app};
            for (size_t i = 0; i < std::min(result.images.size(), second.images.size()); i++) {
                const auto& a = result.images[i].bytes;
                const auto& b = second.images[i].bytes;
                size_t differing = 0;
                for (size_t j = 0; j < std::min(a.size(), b.size()); j++) {
                    differing += a[j] != b[j];
                }
                if (differing) {
                    txt << std::format("second native run: image {} differs in {} bytes\n", i,
                                       differing);
                }
            }
            txt << "second native run compared\n";
        }
        // Did the memory the shaders' user data points at (their constants) change while the
        // native draw ran? The GPU layer reads it only afterwards, when it binds the draw.
        render::translate::TranslatedDraw again;
        std::string error;
        if (render::translate::TranslateDraw(state, packet, again, error)) {
            std::ofstream txt{platform::DataDir() / "native" /
                                  std::format("{:016x}_{}.txt", ps_hash, stats.attempts),
                              std::ios::app};
            for (const auto& [stage, before, after] :
                 {std::tuple{"ps", &result.draw.ps.flatbuf, &again.ps.flatbuf},
                  {"vs", &result.draw.vs.flatbuf, &again.vs.flatbuf}}) {
                size_t changed = 0, first = 0;
                for (size_t i = 0; i < std::min(before->size(), after->size()); i++) {
                    if ((*before)[i] != (*after)[i] && !changed++) {
                        first = i;
                    }
                }
                txt << std::format("{} flatbuf after the native run: {}\n", stage,
                                   changed ? std::format("{} words changed, first {}", changed,
                                                         first)
                                           : std::string{"unchanged"});
            }
        }
    }
    stats.clamped |= result.clamped;
    stats.layer_images += result.layer_images;
    stats.dma_pages = std::max(stats.dma_pages, result.dma_pages);
    stats.dma_missing = std::max(stats.dma_missing, result.dma_missing);
    if (!result.skipped.empty()) {
        stats.skipped = result.skipped;
        stats.attempts--; // a skipped draw checks nothing: let another one of the pair try
        if (stats.skips++ > 20) {
            stats.attempts = settings.per_shader; // the pair never runs natively: stop trying
        }
        return;
    }
    if (!result.error.empty()) {
        stats.error = result.error;
        return;
    }
    g_pending = Pending{.hash = ps_hash,
                        .draw = true,
                        .vs_hash = vs_hash,
                        .ps_hash = ps_hash,
                        .attempt = stats.attempts,
                        .written = std::move(result.written),
                        .images = std::move(result.images)};
}

void ValidateAfterPacket() {
    std::scoped_lock lock{g_mutex};
    if (!g_pending) {
        return;
    }
    Pending pending = std::move(*g_pending);
    g_pending.reset();
    ShaderStats& stats =
        pending.draw ? g_draw_stats[{pending.vs_hash, pending.ps_hash}] : g_stats[pending.hash];
    bool matched = true;
    u32 reordered = 0; // record size
    u64 attempt_changed = 0;
    // Dump mode: what the comparison found per buffer and image, appended to the check's .txt.
    std::ofstream notes;
    if (GetSettings().dump) {
        notes.open(platform::DataDir() / "native" /
                       std::format("{:016x}_{}.txt", pending.hash, pending.attempt),
                   std::ios::app);
    }
    const auto note = [&](const std::string& line) {
        if (notes) {
            notes << line << '\n';
        }
    };
    const auto index = [&](const render::WrittenImage& image) {
        return &image - pending.images.data();
    };
    for (const auto& range : pending.written) {
        std::vector<u8> layer(range.bytes.size());
        gpu::SyncGuestRange(range.address, layer.size());
        if (!ReadMirror(range.address, layer.data(), layer.size())) {
            stats.error = std::format("could not read the GPU layer's result at {:#x}",
                                      range.address);
            return;
        }
        if (range.input) {
            // A buffer checked only for the dump: did game memory change while the packet ran
            // (another queue, the CPU)? Then the GPU layer may have read other bytes than we did.
            const auto [n, l] = std::ranges::mismatch(range.bytes, layer);
            size_t differing = 0;
            for (size_t i = 0; i < layer.size(); i++) {
                differing += range.bytes[i] != layer[i];
            }
            // And what the GPU layer's arena holds there (its copy, if the buffer went through
            // one: read-only buffers up to 8 KiB come straight from memory instead).
            std::vector<u8> arena(range.bytes.size());
            if (gpu::PeekLayerBuffer(range.address, arena.size(), arena.data())) {
                size_t arena_differing = 0;
                for (size_t i = 0; i < arena.size(); i++) {
                    arena_differing += arena[i] != range.bytes[i];
                }
                note(std::format("input buffer at {:#x}: the GPU layer's arena copy {}",
                                 range.address,
                                 arena_differing ? std::format("differs in {} bytes",
                                                               arena_differing)
                                                 : std::string{"is the same"}));
                if (arena_differing && GetSettings().dump) {
                    std::ofstream bin{platform::DataDir() / "native" /
                                          std::format("{:016x}_{}.buffer_{:x}.arena.bin",
                                                      pending.hash, pending.attempt,
                                                      range.address),
                                      std::ios::binary};
                    bin.write(reinterpret_cast<const char*>(arena.data()),
                              static_cast<std::streamsize>(arena.size()));
                    std::ofstream used{platform::DataDir() / "native" /
                                           std::format("{:016x}_{}.buffer_{:x}.native.bin",
                                                       pending.hash, pending.attempt,
                                                       range.address),
                                       std::ios::binary};
                    used.write(reinterpret_cast<const char*>(range.bytes.data()),
                               static_cast<std::streamsize>(range.bytes.size()));
                }
            }
            note(n == range.bytes.end()
                     ? std::format("input buffer at {:#x} ({:#x} bytes): unchanged after the "
                                   "packet",
                                   range.address, layer.size())
                     : std::format("input buffer at {:#x} ({:#x} bytes): changed by the time the "
                                   "packet ran, first at +{:#x}; {} bytes differ",
                                   range.address, layer.size(), n - range.bytes.begin(),
                                   differing));
            continue;
        }
        if (GetSettings().dump) {
            const auto path = platform::DataDir() / "native" /
                              std::format("{:016x}_{}.buffer{}.layer.bin", pending.hash,
                                          pending.attempt, &range - pending.written.data());
            std::ofstream bin{path, std::ios::binary};
            bin.write(reinterpret_cast<const char*>(layer.data()),
                      static_cast<std::streamsize>(layer.size()));
        }
        g_bytes_compared += layer.size();
        g_bytes_changed += range.changed;
        stats.changed += range.changed;
        attempt_changed += range.changed;
        const auto [native_it, layer_it] =
            std::ranges::mismatch(range.bytes, layer);
        const u32 record = native_it != range.bytes.end()
                               ? ReorderedRecordSize(range.bytes, layer)
                               : 0;
        if (record) {
            reordered = record;
        } else if (native_it != range.bytes.end()) {
            matched = false;
            if (stats.first_mismatch.empty()) {
                const size_t offset = (native_it - range.bytes.begin()) & ~size_t{3};
                u32 native = 0, theirs = 0;
                std::memcpy(&native, range.bytes.data() + offset,
                            std::min<size_t>(4, range.bytes.size() - offset));
                std::memcpy(&theirs, layer.data() + offset,
                            std::min<size_t>(4, layer.size() - offset));
                size_t differing = 0;
                for (size_t i = 0; i < layer.size(); i++) {
                    differing += range.bytes[i] != layer[i];
                }
                stats.first_mismatch = std::format(
                    "buffer {:#x} ({} bytes): first difference at +{:#x}, native {:08x} / GPU "
                    "layer {:08x}; {} bytes differ",
                    range.address, layer.size(), offset, native, theirs, differing);
            }
        }
    }
    for (const auto& image : pending.images) {
        const auto& b = image.binding;
        std::vector<u8> layer(b.guest_size, 0);
        std::string why;
        const u32 levels = gpu::ReadLayerImage(b, /*newer_only=*/false, layer.data(), why);
        if (image.input) {
            // An input checked only for the dump: what the GPU layer held for it, if anything.
            if (levels < b.copies.size()) {
                note(std::format("input image {} at {:#x}: not compared ({})", index(image),
                                 b.address, levels ? "the GPU layer has fewer mips" : why));
                continue;
            }
            const auto [n, l] = std::ranges::mismatch(image.bytes, layer);
            if (n == image.bytes.end()) {
                note(std::format("input image {} at {:#x}: same as the GPU layer's", index(image),
                                 b.address));
            } else {
                size_t differing = 0;
                for (size_t i = 0; i < layer.size(); i++) {
                    differing += image.bytes[i] != layer[i];
                }
                note(std::format("input image {} at {:#x}: differs from the GPU layer's at {}; {} "
                                 "bytes differ",
                                 index(image), b.address,
                                 DescribeImageOffset(b, n - image.bytes.begin()), differing));
            }
            continue;
        }
        if (levels < b.copies.size()) {
            stats.image_not_compared =
                levels ? std::format("the GPU layer's image has {} of its {} mips", levels,
                                     b.copies.size())
                       : why;
            continue;
        }
        stats.images_compared++;
        if (!image.stencil.empty()) {
            // The stencil plane, against the GPU layer's.
            std::vector<u8> layer_stencil(image.stencil.size(), 0);
            std::string stencil_why;
            if (!gpu::ReadLayerStencil(b, layer_stencil.data(), stencil_why)) {
                stats.image_not_compared = "stencil: " + stencil_why;
            } else {
                g_bytes_compared += layer_stencil.size();
                stats.changed += image.stencil_changed;
                attempt_changed += image.stencil_changed;
                if (GetSettings().dump) {
                    const auto base = platform::DataDir() / "native" /
                                      std::format("{:016x}_{}.image{}", pending.hash,
                                                  pending.attempt, index(image));
                    for (const auto& [suffix, plane] :
                         {std::pair{".stencil.native.bin", &image.stencil},
                          {".stencil.layer.bin", &layer_stencil}}) {
                        std::ofstream bin{base.string() + suffix, std::ios::binary};
                        bin.write(reinterpret_cast<const char*>(plane->data()),
                                  static_cast<std::streamsize>(plane->size()));
                    }
                }
                const auto [n, l] = std::ranges::mismatch(image.stencil, layer_stencil);
                note(std::format("image {} at {:#x}: stencil {} the GPU layer's ({} bytes changed)",
                                 index(image), b.address,
                                 n == image.stencil.end() ? "same as" : "differs from",
                                 image.stencil_changed));
                if (n != image.stencil.end()) {
                    matched = false;
                    if (stats.first_mismatch.empty()) {
                        const size_t at = n - image.stencil.begin();
                        size_t differing = 0;
                        for (size_t i = 0; i < layer_stencil.size(); i++) {
                            differing += image.stencil[i] != layer_stencil[i];
                        }
                        stats.first_mismatch = std::format(
                            "stencil of {:#x} ({}x{}): first difference at ({}, {}), native {} / "
                            "GPU layer {}; {} bytes differ",
                            b.address, b.extent[0], b.extent[1], at % b.extent[0],
                            at / b.extent[0], *n, *l, differing);
                    }
                }
            }
        }
        if (GetSettings().dump) {
            const auto path = platform::DataDir() / "native" /
                              std::format("{:016x}_{}.image{}.layer.bin", pending.hash,
                                          pending.attempt, &image - pending.images.data());
            std::ofstream bin{path, std::ios::binary};
            bin.write(reinterpret_cast<const char*>(layer.data()),
                      static_cast<std::streamsize>(layer.size()));
        }
        g_bytes_compared += layer.size();
        g_bytes_changed += image.changed;
        stats.changed += image.changed;
        attempt_changed += image.changed;
        const auto [native_it, layer_it] = std::ranges::mismatch(image.bytes, layer);
        if (native_it == image.bytes.end()) {
            note(std::format("image {} at {:#x}: same as the GPU layer's ({} bytes changed)",
                             index(image), b.address, image.changed));
            continue;
        }
        if (const u32 record = ReorderedRecordSize(image.bytes, layer)) {
            reordered = record;
            continue;
        }
        matched = false;
        note(std::format("image {} at {:#x}: differs from the GPU layer's at {} ({} bytes changed)",
                         index(image), b.address,
                         DescribeImageOffset(b, native_it - image.bytes.begin()), image.changed));
        if (!stats.first_mismatch.empty()) {
            continue;
        }
        const size_t offset = native_it - image.bytes.begin();
        const size_t element = offset - offset % std::max(b.element_bytes, 1u);
        std::string native, theirs;
        for (size_t i = element; i < element + std::min(b.element_bytes, 16u) && i < layer.size();
             i++) {
            native += std::format("{:02x}", image.bytes[i]);
            theirs += std::format("{:02x}", layer[i]);
        }
        size_t differing = 0;
        for (size_t i = 0; i < layer.size(); i++) {
            differing += image.bytes[i] != layer[i];
        }
        stats.first_mismatch = std::format(
            "image {:#x} ({}x{}x{} vk format {}): first difference at {}, native {} / GPU layer "
            "{}; {} bytes differ",
            b.address, b.extent[0], b.extent[1], b.extent[2], b.vk_format,
            DescribeImageOffset(b, element), native, theirs, differing);
    }
    if (matched && !attempt_changed && stats.unchanged_retries < MaxUnchangedRetries) {
        // A match where the run changed nothing proves little (a pass masked off by the
        // stencil, an empty frame): let a later run of the shader take the attempt.
        stats.unchanged_retries++;
        stats.attempts--;
        stats.next_try = std::chrono::steady_clock::now() + UnchangedRetryDelay;
        return;
    }
    (matched ? stats.matched : stats.mismatched)++;
    if (matched && reordered) {
        stats.reordered++;
        stats.record_size = reordered;
    }
}

void WriteNativeValidationReport(std::FILE* file) {
    std::scoped_lock lock{g_mutex};
    u32 matched = 0, mismatched = 0, skipped = 0, failed = 0;
    u32 unchanged = 0; // only matching runs that changed nothing so far
    for (const auto& [hash, stats] : g_stats) {
        if (stats.matched && !stats.mismatched) matched++;
        if (stats.mismatched) mismatched++;
        if (!stats.matched && !stats.mismatched && !stats.skipped.empty()) skipped++;
        if (!stats.error.empty()) failed++;
        if (!stats.matched && !stats.mismatched && stats.unchanged_retries) unchanged++;
    }
    std::fprintf(file,
                 "\n== native renderer vs the GPU layer (SOTC_VALIDATE_NATIVE), compute: %zu "
                 "shaders tried; %u always matched, %u differed, %u only matched where they "
                 "changed nothing, %u skipped, %u with errors; %llu bytes compared, %llu of them "
                 "changed by the dispatches\n",
                 g_stats.size(), matched, mismatched, unchanged, skipped, failed,
                 static_cast<unsigned long long>(g_bytes_compared),
                 static_cast<unsigned long long>(g_bytes_changed));
    const auto& names = ShaderNames();
    for (const auto& [hash, stats] : g_stats) {
        const auto name = names.find(hash);
        std::string line = std::format("  {:016x} {:<60} matched {} differed {} (changed {} bytes)",
                                       hash, name != names.end() ? name->second : "?",
                                       stats.matched, stats.mismatched, stats.changed);
        if (stats.clamped) line += " (buffers cut at 64 MiB)";
        if (stats.unchanged_retries) {
            line += std::format(" ({} matching runs that changed nothing not counted)",
                                stats.unchanged_retries);
        }
        if (stats.images_compared) {
            line += std::format(" ({} written images compared)", stats.images_compared);
        }
        if (stats.layer_images) {
            line += std::format(" ({} input images from the GPU layer)", stats.layer_images);
        }
        if (stats.reordered) {
            line += std::format(" ({} of the matches only in another order: atomics, records "
                                "of {} bytes)",
                                stats.reordered, stats.record_size);
        }
        if (stats.dma_pages || stats.dma_missing) {
            line += std::format(" (dynamic loads: {} pages, {} not readable)", stats.dma_pages,
                                stats.dma_missing);
        }
        if (!stats.image_not_compared.empty()) {
            line += " (a written image not compared: " + stats.image_not_compared + ")";
        }
        if (!stats.skipped.empty()) line += "  skipped: " + stats.skipped;
        if (!stats.error.empty()) line += "  error: " + stats.error;
        if (!stats.first_mismatch.empty()) line += "\n      " + stats.first_mismatch;
        std::fprintf(file, "%s\n", line.c_str());
    }

    if (g_draw_stats.empty()) {
        return;
    }
    matched = mismatched = skipped = failed = unchanged = 0;
    for (const auto& [hashes, stats] : g_draw_stats) {
        if (stats.matched && !stats.mismatched) matched++;
        if (stats.mismatched) mismatched++;
        if (!stats.matched && !stats.mismatched && !stats.skipped.empty()) skipped++;
        if (!stats.error.empty()) failed++;
        if (!stats.matched && !stats.mismatched && stats.unchanged_retries) unchanged++;
    }
    std::fprintf(file,
                 "\n== native renderer vs the GPU layer (SOTC_VALIDATE_NATIVE_DRAWS), draws: %zu "
                 "shader pairs tried; %u always matched, %u differed, %u only matched where they "
                 "changed nothing, %u skipped, %u with errors (registers from the model: %u; the "
                 "others from the GPU layer's register file)\n",
                 g_draw_stats.size(), matched, mismatched, unchanged, skipped, failed,
                 g_draw_model_registers);
    for (const auto& [hashes, stats] : g_draw_stats) {
        const auto vs = names.find(hashes.first);
        const auto ps = names.find(hashes.second);
        std::string line = std::format(
            "  vs {:016x} ps {:016x} {} + {} matched {} differed {} (changed {} bytes)",
            hashes.first, hashes.second, vs != names.end() ? vs->second : "?",
            ps != names.end() ? ps->second : "?", stats.matched, stats.mismatched, stats.changed);
        if (stats.clamped) line += " (buffers cut at 64 MiB)";
        if (stats.unchanged_retries) {
            line += std::format(" ({} matching runs that changed nothing not counted)",
                                stats.unchanged_retries);
        }
        if (stats.images_compared) {
            line += std::format(" ({} targets compared)", stats.images_compared);
        }
        if (stats.layer_images) {
            line += std::format(" ({} images from the GPU layer)", stats.layer_images);
        }
        if (stats.reordered) {
            line += std::format(" ({} of the matches only in another order)", stats.reordered);
        }
        if (!stats.image_not_compared.empty()) {
            line += " (a target not compared: " + stats.image_not_compared + ")";
        }
        if (!stats.skipped.empty()) line += "  skipped: " + stats.skipped;
        if (!stats.error.empty()) line += "  error: " + stats.error;
        if (!stats.first_mismatch.empty()) line += "\n      " + stats.first_mismatch;
        std::fprintf(file, "%s\n", line.c_str());
    }
}

} // namespace diag
