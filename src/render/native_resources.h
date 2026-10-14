#pragma once

#include <array>
#include <deque>
#include <functional>
#include <span>
#include <string>
#include <vector>

#include "common/types.h"
#include "render/shader_translate.h"
#include "render/vulkan/device.h"

// The resources the translated shaders of one native run use (a dispatch, a draw), prepared on
// the native device from game memory: buffers, the flattened user data, LDS, the page table of
// dynamic loads, images (uploaded, detiled with the GPU layer's own tiling shader) and samplers,
// in one descriptor set laid out as the recompiler expects (set 0; per stage in binding order:
// every buffer, then images, then samplers; Shader::PushData's buffer offsets numbered across
// the stages). Shared by render/native_compute.cpp and render/native_draw.cpp.

namespace render {

// Reads game memory as the GPU sees it right now (the caller syncs it first).
using GuestReader = std::function<bool(u64 address, void* out, u64 size)>;
// Reads the GPU layer's own image for a binding when it holds GPU writes that game memory lacks
// (render targets and compute outputs stay in its VkImages): linear, in the layout of the
// binding's copies (gpu::ReadLayerImage). Returns how many mips (from 0) it read; the others
// come from game memory.
using LayerImageReader = std::function<u32(const translate::ImageBinding& binding, u8* out)>;

struct WrittenRange {
    u64 address = 0;
    std::vector<u8> bytes; // the range after the native run
    u64 changed = 0;       // bytes the run changed (a check that compares nothing is weak)
    // A buffer the run only reads, as it was given (ResourceReport::compare_inputs).
    bool input = false;
};

// An image the run may have written, read back linear: the rows ImageBinding::ForEachRow lists
// hold the image after the run, every other byte is 0.
struct WrittenImage {
    translate::ImageBinding binding;
    std::vector<u8> bytes;
    u64 changed = 0; // bytes the run changed
    // An image the run only reads, read back as it was given (ResourceReport::compare_inputs).
    bool input = false;
    std::vector<u8> before; // with compare_inputs: the image before the run (diagnostics)
    // The stencil plane (ImageBinding::StencilBytes), when the run used it.
    std::vector<u8> stencil;
    u64 stencil_changed = 0;
};

// What the preparation and the run found out.
struct ResourceReport {
    // Set by the caller: also read back the images the run only reads (WrittenImage::input),
    // to check them against the GPU layer's (diagnostics: a wrong input explains a wrong output).
    bool compare_inputs = false;
    bool clamped = false;  // a buffer was larger than the native renderer copies
    u32 layer_images = 0;  // images taken (at least in part) from the GPU layer
    std::vector<u32> image_layer_levels; // per image binding: mips from the GPU layer
    std::vector<WrittenRange> written;
    std::vector<WrittenImage> images;
    // Dynamic loads (BufferKind::PageTable): the pages of game memory given to them, and how
    // many they still looked for in vain at the end (unreadable pages).
    u32 dma_pages = 0;
    u32 dma_missing = 0;
};

// Shader::PushData: x/y offset and scale (graphics), then one byte per buffer binding.
struct PushData {
    std::array<float, 4> viewport{};
    std::array<u8, 56> buf_offsets{};
};
static_assert(sizeof(PushData) <= 128);

// The Vulkan objects of one run, destroyed together.
struct Objects {
    explicit Objects(vk::Device& device_) : device{device_} {}
    Objects(const Objects&) = delete;
    Objects& operator=(const Objects&) = delete;
    ~Objects();
    vk::Device& device;
    std::vector<vk::HostBuffer> buffers;
    std::vector<vk::Image> images;
    std::vector<VkImageView> views;
    std::vector<VkSampler> samplers;
    std::vector<VkShaderModule> modules;
    std::vector<VkDescriptorSetLayout> set_layouts;
    std::vector<VkPipelineLayout> layouts;
    std::vector<VkPipeline> pipelines;
    std::vector<VkDescriptorPool> pools;
};

// An image on the native device holding what the binding describes: from game memory (detiled)
// or from the GPU layer (see LayerImageReader), in GENERAL layout once uploaded.
struct PreparedImage {
    const translate::ImageBinding* binding = nullptr;
    bool read_back = false; // read back linear before and after the run (written images)
    vk::HostBuffer raw, linear, params;
    vk::HostBuffer before, after;
    vk::HostBuffer layer; // the GPU layer's copy of its first layer_levels mips
    u32 layer_levels = 0;
    bool from_layer = false; // all mips from the GPU layer
    vk::Image image;
    VkImageView view = VK_NULL_HANDLE; // the view the shader binding expects (if any)
    VkPipeline detile = VK_NULL_HANDLE;
    VkPipelineLayout detile_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout detile_set_layout = VK_NULL_HANDLE;
    VkDescriptorSet detile_set = VK_NULL_HANDLE;
    u32 layers = 1;
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT; // what is uploaded and read back
    VkImageAspectFlags all_aspects = VK_IMAGE_ASPECT_COLOR_BIT; // every plane (layouts)
    // The stencil plane, uploaded from `stencil_in` and read back into `stencil_out` (a depth
    // target whose stencil the run tests).
    bool stencil = false;
    vk::HostBuffer stencil_in, stencil_out;
    // The copies between the image and its linear buffer, one per mip (all layers at once).
    std::vector<VkBufferImageCopy> Copies() const;
};

class ResourceSet {
public:
    struct Stage {
        const translate::TranslatedProgram* program = nullptr;
        VkShaderStageFlags stage = 0;
    };

    ResourceSet(vk::Device& device, Objects& objects, const GuestReader& read,
                const LayerImageReader& read_image)
        : device_{device}, objects_{objects}, read_{read}, read_image_{read_image} {}
    ResourceSet(const ResourceSet&) = delete;
    ResourceSet& operator=(const ResourceSet&) = delete;
    // Clears the entries it set in the shared page table.
    ~ResourceSet();

    // A reason the stages cannot run natively yet (an image or special buffer not supported),
    // or empty.
    static std::string Unsupported(std::span<const Stage> stages);

    // Prepares every resource of `stages`; `lds_bytes`: the size of a SharedMemory buffer;
    // `pages`: the pages the page table gives (unreadable ones are taken out). False with
    // `error` set.
    bool Prepare(std::span<const Stage> stages, u64 lds_bytes, std::vector<u64>& pages,
                 ResourceReport& report, std::string& error);
    // An image used outside the descriptor set (a render target), prepared the same way and
    // read back if `read_back`; `stencil`: its stencil plane's contents (StencilBytes), to
    // upload and read back too. Before Finish.
    PreparedImage* AddImage(const translate::ImageBinding& binding, VkImageUsageFlags usage,
                            bool read_back, const std::vector<u8>* stencil,
                            ResourceReport& report, std::string& error);
    // Creates the descriptor set and the detiling sets. False with `error` set.
    bool Finish(std::string& error);

    // Records the uploads: images detiled and copied in (then GENERAL, ready for any stage),
    // and the "before" readbacks.
    void RecordUploads(VkCommandBuffer cmd) const;
    // Records the readbacks after the run (`stages`: where it wrote).
    void RecordReadbacks(VkCommandBuffer cmd, VkPipelineStageFlags stages) const;
    // After the run: the written buffers and images, and the pages dynamic loads missed.
    void Collect(ResourceReport& report, std::vector<u64>& faults) const;

    VkDescriptorSetLayout SetLayout() const { return set_layout_; }
    VkDescriptorSet Set() const { return set_; }
    PushData& Push() { return push_; }

private:
    struct Arena {
        u64 start = 0;
        u64 end = 0;
        vk::HostBuffer buffer;
    };
    struct BufferSlot {
        const translate::BufferBinding* binding = nullptr;
        u64 size = 0; // bytes bound (game memory buffers)
        VkDescriptorBufferInfo info{};
        std::vector<u8> before; // written game memory buffers: the bytes before the run
    };
    const Arena* ArenaOf(u64 address) const;
    bool PrepareImage(PreparedImage& p, VkImageUsageFlags usage, ResourceReport& report,
                      std::string& error);

    vk::Device& device_;
    Objects& objects_;
    const GuestReader& read_;
    const LayerImageReader& read_image_;
    std::vector<Arena> arenas_;
    std::vector<BufferSlot> buffers_;
    std::vector<PreparedImage> images_;     // the shader images, in binding order
    std::deque<PreparedImage> attachments_; // AddImage (a deque: the pointers it returns stay valid)
    std::vector<VkSampler> samplers_;
    std::vector<VkDescriptorSetLayoutBinding> layout_bindings_;
    std::vector<u32> image_slot_; // per layout binding: index into images_ (images only)
    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkDescriptorSet set_ = VK_NULL_HANDLE;
    PushData push_{};
    vk::HostBuffer dma_pages_;
    bool uses_page_table_ = false;
    std::vector<u64> table_entries_; // pages whose page table entry this run set
};

// Runs `once` (one complete run that fills `faults` with the pages dynamic loads looked for in
// vain) again with those pages given, until nothing new is asked for: like the GPU layer, whose
// buffer cache keeps the pages a shader faulted on for its next runs. `done()` stops early (an
// error or a skip). Fills report.dma_pages / dma_missing; returns a skip reason or "".
std::string RunWithDynamicPages(
    const std::function<void(std::vector<u64>& pages, std::vector<u64>& faults)>& once,
    const std::function<bool()>& done, ResourceReport& report);

} // namespace render
