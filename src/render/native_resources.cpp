#include "render/native_resources.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <format>
#include <set>

#include "hle/kernel/gpu_memory.h"

namespace render {

namespace {

// Buffers are copied whole; the game often describes buffers with an "endless" size (num_records
// of ~4 GiB). Larger ones are cut here (the report says so).
constexpr u64 MaxBufferBytes = 64ull << 20;

// The page table and fault buffer of dynamic scalar loads (translate::BufferKind::PageTable):
// 2^24 entries (128 MiB) and 2 MiB of bits, made once and reused; between runs every entry is 0
// (each run clears the ones it set).
struct DmaTables {
    vk::HostBuffer table;
    vk::HostBuffer faults;
};

DmaTables* GetDmaTables(vk::Device& device) {
    static DmaTables tables;
    static bool tried = false;
    if (!std::exchange(tried, true)) {
        if (device.CreateHostBuffer(translate::DmaPageCount * 8, tables.table) &&
            device.CreateHostBuffer(translate::DmaPageCount / 8, tables.faults)) {
            std::memset(tables.table.data, 0, tables.table.size);
        } else {
            device.Destroy(tables.table);
            device.Destroy(tables.faults);
        }
    }
    return tables.table.buffer ? &tables : nullptr;
}

// At most this many pages (64 KiB each) are given to dynamic loads in one run.
constexpr size_t MaxDmaPages = 1024;

bool IsDepthFormat(VkFormat format) {
    return format == VK_FORMAT_D16_UNORM || format == VK_FORMAT_D32_SFLOAT ||
           format == VK_FORMAT_X8_D24_UNORM_PACK32 || format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
           format == VK_FORMAT_D24_UNORM_S8_UINT || format == VK_FORMAT_D16_UNORM_S8_UINT;
}

bool HasStencil(VkFormat format) {
    return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT ||
           format == VK_FORMAT_D16_UNORM_S8_UINT || format == VK_FORMAT_S8_UINT;
}

// The bytes of the first `levels` mips in the linear buffer: from each mip's offset to the next
// mip's (or the end), every layer of the mip included.
std::vector<VkBufferCopy> MipRanges(const translate::ImageBinding& b, u32 levels) {
    std::vector<VkBufferCopy> ranges;
    for (u32 m = 0; m < levels && m < b.copies.size(); m++) {
        const u64 start = b.copies[m].buffer_offset;
        u64 end = b.guest_size;
        for (const auto& c : b.copies) {
            if (c.buffer_offset > start) {
                end = std::min(end, c.buffer_offset);
            }
        }
        ranges.push_back({start, start, end - start});
    }
    return ranges;
}

} // namespace

Objects::~Objects() {
    const auto& api = device.api();
    const VkDevice handle = device.handle();
    for (VkPipeline p : pipelines) api.vkDestroyPipeline(handle, p, nullptr);
    for (VkPipelineLayout l : layouts) api.vkDestroyPipelineLayout(handle, l, nullptr);
    for (VkDescriptorPool p : pools) api.vkDestroyDescriptorPool(handle, p, nullptr);
    for (VkDescriptorSetLayout l : set_layouts) api.vkDestroyDescriptorSetLayout(handle, l, nullptr);
    for (VkShaderModule m : modules) api.vkDestroyShaderModule(handle, m, nullptr);
    for (VkSampler sampler : samplers) api.vkDestroySampler(handle, sampler, nullptr);
    for (VkImageView view : views) api.vkDestroyImageView(handle, view, nullptr);
    for (auto& image : images) device.Destroy(image);
    for (auto& buffer : buffers) device.Destroy(buffer);
}

std::vector<VkBufferImageCopy> PreparedImage::Copies() const {
    std::vector<VkBufferImageCopy> copies;
    for (u32 m = 0; m < binding->copies.size(); m++) {
        const auto& c = binding->copies[m];
        copies.push_back({
            .bufferOffset = c.buffer_offset,
            .bufferRowLength = c.row_length,
            .bufferImageHeight = c.image_height,
            .imageSubresource = {aspect, m, 0, layers},
            .imageOffset = {0, 0, 0},
            .imageExtent = {c.width, c.height, c.depth},
        });
    }
    return copies;
}

std::string ResourceSet::Unsupported(std::span<const Stage> stages) {
    for (const Stage& stage : stages) {
        for (const auto& image : stage.program->image_bindings) {
            // Supported: every image the GPU layer itself can detile. The reason describes the
            // image (type WxHxD data/number format, tile mode).
            if (!image.vk_format || (image.tiled && !image.tiling_spirv) || !image.guest_size ||
                image.per_mip_descriptors) {
                return std::format("image t{} {}x{}x{} f{}/{} tile{} not supported", image.type,
                                   image.width, image.height, image.depth, image.data_format,
                                   image.number_format, image.tile_mode);
            }
        }
        for (const auto& binding : stage.program->buffers) {
            if (binding.kind == translate::BufferKind::Unsupported) {
                return std::format("special buffer {}", binding.name);
            }
        }
    }
    return {};
}

ResourceSet::~ResourceSet() {
    if (DmaTables* tables = table_entries_.empty() ? nullptr : GetDmaTables(device_)) {
        for (u64 page : table_entries_) {
            reinterpret_cast<u64*>(tables->table.data)[page] = 0;
        }
    }
}

const ResourceSet::Arena* ResourceSet::ArenaOf(u64 address) const {
    for (const Arena& arena : arenas_) {
        if (address >= arena.start && address < arena.end) {
            return &arena;
        }
    }
    return nullptr;
}

bool ResourceSet::Prepare(std::span<const Stage> stages, u64 lds_bytes, std::vector<u64>& pages,
                          ResourceReport& report, std::string& error) {
    const auto& api = device_.api();
    const VkDevice handle = device_.handle();

    // --- game memory into arenas ----------------------------------------------------------
    // Ranges of game memory that overlap or touch are copied into one native buffer (an
    // "arena"), so that bindings aliasing the same memory see each other's writes as on the PS4.
    // An arena starts at a 256-byte boundary; each binding is bound at an aligned offset inside
    // it and the shader adds the rest (PushData::buf_offsets, which holds less than 256).
    std::vector<std::pair<u64, u64>> ranges;
    std::vector<VkShaderStageFlags> buffer_stages;
    for (const Stage& stage : stages) {
        for (const auto& binding : stage.program->buffers) {
            BufferSlot& slot = buffers_.emplace_back();
            slot.binding = &binding;
            buffer_stages.push_back(stage.stage);
            if (binding.kind != translate::BufferKind::Guest || !binding.address ||
                !binding.size) {
                continue;
            }
            slot.size = std::min(binding.size, MaxBufferBytes);
            report.clamped |= slot.size != binding.size;
            // Like the GPU layer (ClampRangeSize): no further than the mapped memory goes.
            slot.size = hle::kernel::ClampToMapped(binding.address, slot.size);
            if (slot.size) {
                ranges.emplace_back(binding.address & ~255ull, binding.address + slot.size);
            }
        }
    }
    if (buffers_.size() > PushData{}.buf_offsets.size()) {
        error = "too many buffers";
        return false;
    }
    std::ranges::sort(ranges);
    for (const auto& [start, end] : ranges) {
        if (!arenas_.empty() && start <= arenas_.back().end) {
            arenas_.back().end = std::max(arenas_.back().end, end);
        } else {
            arenas_.push_back({.start = start, .end = end, .buffer = {}});
        }
    }
    for (Arena& arena : arenas_) {
        if (!device_.CreateHostBuffer(arena.end - arena.start, arena.buffer)) {
            error = "could not allocate a buffer";
            return false;
        }
        objects_.buffers.push_back(arena.buffer);
        if (!read_(arena.start, arena.buffer.data, arena.end - arena.start)) {
            error = std::format("could not read game memory at {:#x}", arena.start);
            return false;
        }
    }

    // --- buffer descriptors ----------------------------------------------------------------
    const u64 alignment = device_.StorageAlignment();
    vk::HostBuffer lds, dummy;
    size_t slot_index = 0;
    for (const Stage& stage : stages) {
        vk::HostBuffer flatbuf; // per stage: each has its own flattened user data
        const auto& program = *stage.program;
        for (size_t b = 0; b < program.buffers.size(); b++, slot_index++) {
            BufferSlot& slot = buffers_[slot_index];
            const auto& binding = *slot.binding;
            switch (binding.kind) {
            case translate::BufferKind::Guest: {
                const Arena* arena = binding.address && slot.size ? ArenaOf(binding.address)
                                                                  : nullptr;
                if (!arena) { // null descriptor: an empty buffer
                    if (!dummy.buffer && device_.CreateHostBuffer(16, dummy)) {
                        std::memset(dummy.data, 0, 16);
                        objects_.buffers.push_back(dummy);
                    }
                    slot.info = {dummy.buffer, 0, 16};
                    break;
                }
                const u64 offset = binding.address - arena->start;
                const u64 aligned = offset / alignment * alignment;
                push_.buf_offsets[slot_index] = static_cast<u8>(offset - aligned);
                slot.info = {arena->buffer.buffer, aligned, slot.size + (offset - aligned)};
                if (binding.written) {
                    const u8* begin = arena->buffer.data + offset;
                    slot.before.assign(begin, begin + slot.size);
                }
                break;
            }
            case translate::BufferKind::Flatbuf: {
                const u64 bytes = std::max<u64>(program.flatbuf.size() * 4, 16);
                if (!flatbuf.buffer) {
                    if (!device_.CreateHostBuffer(bytes, flatbuf)) {
                        error = "could not allocate the flattened user data";
                        return false;
                    }
                    std::memset(flatbuf.data, 0, bytes);
                    std::memcpy(flatbuf.data, program.flatbuf.data(), program.flatbuf.size() * 4);
                    objects_.buffers.push_back(flatbuf);
                }
                slot.info = {flatbuf.buffer, 0, bytes};
                break;
            }
            case translate::BufferKind::SharedMemory: {
                const u64 bytes = std::max<u64>(lds_bytes, 16);
                if (!lds.buffer) {
                    if (!device_.CreateHostBuffer(bytes, lds)) {
                        error = "could not allocate the LDS buffer";
                        return false;
                    }
                    std::memset(lds.data, 0, bytes);
                    objects_.buffers.push_back(lds);
                }
                slot.info = {lds.buffer, 0, bytes};
                break;
            }
            case translate::BufferKind::PageTable: {
                DmaTables* tables = GetDmaTables(device_);
                if (!tables) {
                    error = "could not allocate the page table";
                    return false;
                }
                if (!pages.empty() && !dma_pages_.buffer) {
                    // Each page copied from game memory; its entry gets the copy's address.
                    constexpr u64 page_size = u64{1} << translate::DmaPageBits;
                    if (!device_.CreateHostBuffer(pages.size() * page_size, dma_pages_, true)) {
                        error = "could not allocate the pages of dynamic loads";
                        return false;
                    }
                    objects_.buffers.push_back(dma_pages_);
                    const u64 base = device_.DeviceAddress(dma_pages_);
                    std::vector<u64> readable;
                    for (u64 page : pages) {
                        u8* copy = dma_pages_.data + readable.size() * page_size;
                        if (read_(page << translate::DmaPageBits, copy, page_size)) {
                            reinterpret_cast<u64*>(tables->table.data)[page] =
                                base + readable.size() * page_size;
                            table_entries_.push_back(page);
                            readable.push_back(page);
                        }
                    }
                    pages = std::move(readable);
                }
                if (!uses_page_table_) {
                    std::memset(tables->faults.data, 0, tables->faults.size);
                }
                uses_page_table_ = true;
                slot.info = {tables->table.buffer, 0, tables->table.size};
                break;
            }
            case translate::BufferKind::Inline: {
                vk::HostBuffer inline_data;
                const u64 bytes = std::max<u64>(binding.data.size() * 4, 16);
                if (!device_.CreateHostBuffer(bytes, inline_data)) {
                    error = "could not allocate an inline buffer";
                    return false;
                }
                objects_.buffers.push_back(inline_data);
                std::memset(inline_data.data, 0, bytes);
                std::memcpy(inline_data.data, binding.data.data(), binding.data.size() * 4);
                slot.info = {inline_data.buffer, 0, bytes};
                break;
            }
            case translate::BufferKind::FaultBuffer: {
                DmaTables* tables = GetDmaTables(device_);
                if (!tables) {
                    error = "could not allocate the page table";
                    return false;
                }
                slot.info = {tables->faults.buffer, 0, tables->faults.size};
                break;
            }
            case translate::BufferKind::Unsupported:
                break;
            }
        }
    }

    // --- images, samplers and the layout ---------------------------------------------------
    size_t total_images = 0;
    for (const Stage& stage : stages) {
        total_images += stage.program->image_bindings.size();
    }
    images_.reserve(total_images); // PreparedImage pointers below must stay valid
    slot_index = 0;
    for (const Stage& stage : stages) {
        const auto& program = *stage.program;
        for (size_t b = 0; b < program.buffers.size(); b++, slot_index++) {
            layout_bindings_.push_back({
                .binding = static_cast<u32>(layout_bindings_.size()),
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = stage.stage,
            });
            image_slot_.push_back(~0u);
        }
        for (const auto& b : program.image_bindings) {
            PreparedImage& p = images_.emplace_back();
            p.binding = &b;
            p.read_back = b.written || report.compare_inputs;
            const VkImageUsageFlags usage =
                VK_IMAGE_USAGE_SAMPLED_BIT | (b.storage ? VK_IMAGE_USAGE_STORAGE_BIT : 0u);
            if (!PrepareImage(p, usage, report, error)) {
                return false;
            }
            const VkImageViewMinLodCreateInfoEXT min_lod{
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_MIN_LOD_CREATE_INFO_EXT,
                .minLod = b.view_min_lod,
            };
            const VkImageViewCreateInfo view_info{
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .pNext = b.view_min_lod != 0 && device_.HasMinLod() ? &min_lod : nullptr,
                .image = p.image.image,
                .viewType = static_cast<VkImageViewType>(b.view_type),
                .format = static_cast<VkFormat>(b.view_format ? b.view_format : b.vk_format),
                .components = {static_cast<VkComponentSwizzle>(b.swizzle[0]),
                               static_cast<VkComponentSwizzle>(b.swizzle[1]),
                               static_cast<VkComponentSwizzle>(b.swizzle[2]),
                               static_cast<VkComponentSwizzle>(b.swizzle[3])},
                .subresourceRange = {p.aspect, b.view_base_level, b.view_levels,
                                     b.volume ? 0u : b.view_base_layer,
                                     b.volume ? 1u : b.view_layers},
            };
            if (api.vkCreateImageView(handle, &view_info, nullptr, &p.view) != VK_SUCCESS) {
                error = std::format("could not create the view of image {:#x}", b.address);
                return false;
            }
            objects_.views.push_back(p.view);
            layout_bindings_.push_back({
                .binding = static_cast<u32>(layout_bindings_.size()),
                .descriptorType = b.storage ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                            : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                .descriptorCount = 1,
                .stageFlags = stage.stage,
            });
            image_slot_.push_back(static_cast<u32>(images_.size() - 1));
        }
        for (const translate::SamplerBinding& b : program.sampler_bindings) {
            const VkSamplerCreateInfo sampler_info{
                .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                .magFilter = static_cast<VkFilter>(b.mag_filter),
                .minFilter = static_cast<VkFilter>(b.min_filter),
                .mipmapMode = static_cast<VkSamplerMipmapMode>(b.mipmap_mode),
                .addressModeU = static_cast<VkSamplerAddressMode>(b.address_u),
                .addressModeV = static_cast<VkSamplerAddressMode>(b.address_v),
                .addressModeW = static_cast<VkSamplerAddressMode>(b.address_w),
                .mipLodBias = b.lod_bias,
                .anisotropyEnable = b.anisotropy && device_.MaxAnisotropy() > 1.0f,
                .maxAnisotropy = std::clamp(b.max_anisotropy, 1.0f, device_.MaxAnisotropy()),
                .compareEnable = b.compare,
                .compareOp = static_cast<VkCompareOp>(b.compare_op),
                .minLod = b.min_lod,
                .maxLod = b.max_lod,
                .borderColor = static_cast<VkBorderColor>(b.border),
            };
            VkSampler sampler = VK_NULL_HANDLE;
            if (api.vkCreateSampler(handle, &sampler_info, nullptr, &sampler) != VK_SUCCESS) {
                error = "could not create a sampler";
                return false;
            }
            objects_.samplers.push_back(sampler);
            samplers_.push_back(sampler);
            layout_bindings_.push_back({
                .binding = static_cast<u32>(layout_bindings_.size()),
                .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
                .descriptorCount = 1,
                .stageFlags = stage.stage,
            });
            image_slot_.push_back(~0u);
        }
    }
    return true;
}

PreparedImage* ResourceSet::AddImage(const translate::ImageBinding& binding,
                                     VkImageUsageFlags usage, bool read_back,
                                     const std::vector<u8>* stencil, ResourceReport& report,
                                     std::string& error) {
    PreparedImage& p = attachments_.emplace_back();
    p.binding = &binding;
    p.read_back = read_back;
    if (!PrepareImage(p, usage, report, error)) {
        return nullptr;
    }
    if (stencil && (p.all_aspects & VK_IMAGE_ASPECT_STENCIL_BIT)) {
        const u64 size = binding.StencilBytes();
        if (!device_.CreateHostBuffer(size, p.stencil_in) ||
            !device_.CreateHostBuffer(size, p.stencil_out)) {
            error = "could not allocate a stencil buffer";
            return nullptr;
        }
        objects_.buffers.push_back(p.stencil_in);
        objects_.buffers.push_back(p.stencil_out);
        std::memset(p.stencil_in.data, 0, size);
        std::memcpy(p.stencil_in.data, stencil->data(), std::min<u64>(size, stencil->size()));
        p.stencil = true;
    }
    return &p;
}

namespace {

// The copy of a stencil plane: mip 0, every layer, tightly packed.
VkBufferImageCopy StencilCopy(const PreparedImage& p) {
    return {
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = {VK_IMAGE_ASPECT_STENCIL_BIT, 0, 0, p.layers},
        .imageOffset = {0, 0, 0},
        .imageExtent = {p.binding->extent[0], p.binding->extent[1], 1},
    };
}

} // namespace

// Each image: its bytes from game memory into a buffer, detiled by the GPU layer's own tiling
// shader into a second buffer when tiled (TileManager::DetileImage), copied into the image mip
// by mip (TextureCache::RefreshImage). When the GPU layer holds GPU writes to the image that game
// memory lacks, its linear copy replaces those mips (all of them: no detiling).
bool ResourceSet::PrepareImage(PreparedImage& p, VkImageUsageFlags usage, ResourceReport& report,
                               std::string& error) {
    const auto& api = device_.api();
    const VkDevice handle = device_.handle();
    const translate::ImageBinding& b = *p.binding;
    if (!device_.CreateHostBuffer(b.guest_size, p.raw)) {
        error = "could not allocate an image upload buffer";
        return false;
    }
    objects_.buffers.push_back(p.raw);
    if (read_image_ && b.address) {
        if (!device_.CreateHostBuffer(b.guest_size, p.layer)) {
            error = "could not allocate an image upload buffer";
            return false;
        }
        objects_.buffers.push_back(p.layer);
        std::memset(p.layer.data, 0, b.guest_size);
        p.layer_levels = std::min(read_image_(b, p.layer.data), b.levels);
        p.from_layer = p.layer_levels == b.levels;
        report.layer_images += p.layer_levels > 0;
    }
    report.image_layer_levels.push_back(p.layer_levels);
    if (p.from_layer) {
        std::memcpy(p.raw.data, p.layer.data, b.guest_size); // linear already
    } else if (!b.address) {
        // A null T# (the recompiler still describes an image): reads see zeros.
        std::memset(p.raw.data, 0, b.guest_size);
    } else {
        // What is mapped of the image's memory; the holes read as 0, as the GPU layer sees them.
        // Streamed textures have holes: the game maps the pages of their large mips only when
        // it needs them, while the small mips at the end stay mapped. Images may also reach past
        // the end of the mapped memory, or start where nothing is mapped (a descriptor the game
        // left bound, that the shader does not read).
        constexpr u64 PageBytes = 16 * 1024; // the PS4's page
        std::memset(p.raw.data, 0, b.guest_size);
        u64 mapped_total = 0;
        for (u64 at = 0; at < b.guest_size;) {
            const u64 mapped = hle::kernel::ClampToMapped(b.address + at, b.guest_size - at);
            if (!mapped) {
                at = std::min<u64>(b.guest_size, ((b.address + at) / PageBytes + 1) * PageBytes -
                                                b.address); // the next page
                continue;
            }
            if (!read_(b.address + at, p.raw.data + at, mapped)) {
                error = std::format("could not read the image at {:#x}", b.address + at);
                return false;
            }
            mapped_total += mapped;
            at += mapped;
        }
        report.clamped |= mapped_total < b.guest_size;
    }
    if (p.read_back) {
        if (!device_.CreateHostBuffer(b.guest_size, p.before) ||
            !device_.CreateHostBuffer(b.guest_size, p.after)) {
            error = "could not allocate an image readback buffer";
            return false;
        }
        objects_.buffers.push_back(p.before);
        objects_.buffers.push_back(p.after);
        std::memset(p.before.data, 0, b.guest_size);
        std::memset(p.after.data, 0, b.guest_size);
    }
    const auto format = static_cast<VkFormat>(b.vk_format);
    if (IsDepthFormat(format)) {
        p.aspect = VK_IMAGE_ASPECT_DEPTH_BIT; // a stencil plane is not uploaded
        p.all_aspects = VK_IMAGE_ASPECT_DEPTH_BIT |
                        (HasStencil(format) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0u);
    }
    p.layers = b.volume ? 1 : b.layers;
    const VkImageCreateInfo image_info{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        // The view's format may differ (sRGB, storage): mutable, with extended usage.
        .flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT,
        .imageType = b.volume ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D,
        .format = format,
        .extent = {b.extent[0], b.extent[1], b.volume ? b.extent[2] : 1u},
        .mipLevels = b.levels,
        .arrayLayers = p.layers,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = usage | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    if (!device_.CreateImage(image_info, p.image)) {
        error = std::format("could not create the image at {:#x} (format {})", b.address,
                            b.vk_format);
        return false;
    }
    objects_.images.push_back(p.image);
    if (!b.tiled || p.from_layer) {
        return true;
    }
    if (!device_.CreateHostBuffer(b.guest_size, p.linear) ||
        !device_.CreateHostBuffer(std::max<u64>(b.tiling_info.size(), 16), p.params)) {
        error = "could not allocate a detiling buffer";
        return false;
    }
    objects_.buffers.push_back(p.linear);
    objects_.buffers.push_back(p.params);
    std::memcpy(p.params.data, b.tiling_info.data(), b.tiling_info.size());
    // tiling.comp: binding 0 the tiled bytes, 1 the linear output, 2 the TilingInfo block.
    const VkDescriptorSetLayoutBinding detile_bindings[3] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    const VkDescriptorSetLayoutCreateInfo detile_set_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = detile_bindings,
    };
    const VkShaderModuleCreateInfo detile_module_info{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = b.tiling_spirv_words * 4,
        .pCode = b.tiling_spirv,
    };
    VkShaderModule detile_module = VK_NULL_HANDLE;
    if (api.vkCreateDescriptorSetLayout(handle, &detile_set_info, nullptr,
                                        &p.detile_set_layout) != VK_SUCCESS ||
        api.vkCreateShaderModule(handle, &detile_module_info, nullptr, &detile_module) !=
            VK_SUCCESS) {
        error = "could not create the detiling shader";
        return false;
    }
    objects_.set_layouts.push_back(p.detile_set_layout);
    objects_.modules.push_back(detile_module);
    const VkPipelineLayoutCreateInfo detile_layout_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &p.detile_set_layout,
    };
    if (api.vkCreatePipelineLayout(handle, &detile_layout_info, nullptr, &p.detile_layout) !=
        VK_SUCCESS) {
        error = "could not create the detiling layout";
        return false;
    }
    objects_.layouts.push_back(p.detile_layout);
    std::array<VkSpecializationMapEntry, 12> entries{};
    for (u32 e = 0; e < entries.size(); e++) {
        entries[e] = {e, e * 4, 4};
    }
    const VkSpecializationInfo specialization{
        .mapEntryCount = b.tiling_spec_count,
        .pMapEntries = entries.data(),
        .dataSize = b.tiling_spec_count * 4,
        .pData = b.tiling_spec.data(),
    };
    const VkComputePipelineCreateInfo detile_info{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage =
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = detile_module,
                .pName = "main",
                .pSpecializationInfo = &specialization,
            },
        .layout = p.detile_layout,
    };
    if (api.vkCreateComputePipelines(handle, VK_NULL_HANDLE, 1, &detile_info, nullptr,
                                     &p.detile) != VK_SUCCESS) {
        error = "could not create the detiling pipeline";
        return false;
    }
    objects_.pipelines.push_back(p.detile);
    return true;
}

bool ResourceSet::Finish(std::string& error) {
    const auto& api = device_.api();
    const VkDevice handle = device_.handle();
    u32 tiled = 0;
    for (const PreparedImage& p : images_) tiled += p.detile != VK_NULL_HANDLE;
    for (const PreparedImage& p : attachments_) tiled += p.detile != VK_NULL_HANDLE;
    const u32 count = static_cast<u32>(layout_bindings_.size());
    const u32 images = static_cast<u32>(images_.size());

    const VkDescriptorSetLayoutCreateInfo set_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = count,
        .pBindings = layout_bindings_.data(),
    };
    if (api.vkCreateDescriptorSetLayout(handle, &set_info, nullptr, &set_layout_) != VK_SUCCESS) {
        error = "could not create the descriptor layout";
        return false;
    }
    objects_.set_layouts.push_back(set_layout_);
    const VkDescriptorPoolSize pool_sizes[5] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         std::max(static_cast<u32>(buffers_.size()) + 2 * tiled, 1u)},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, std::max(tiled, 1u)},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, std::max(images, 1u)},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, std::max(images, 1u)},
        {VK_DESCRIPTOR_TYPE_SAMPLER, std::max(static_cast<u32>(samplers_.size()), 1u)},
    };
    const VkDescriptorPoolCreateInfo pool_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1 + tiled,
        .poolSizeCount = 5,
        .pPoolSizes = pool_sizes,
    };
    VkDescriptorPool pool = VK_NULL_HANDLE;
    if (api.vkCreateDescriptorPool(handle, &pool_info, nullptr, &pool) != VK_SUCCESS) {
        error = "could not create the descriptor pool";
        return false;
    }
    objects_.pools.push_back(pool);
    const VkDescriptorSetAllocateInfo allocate{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &set_layout_,
    };
    if (api.vkAllocateDescriptorSets(handle, &allocate, &set_) != VK_SUCCESS) {
        error = "could not allocate the descriptor set";
        return false;
    }
    std::vector<VkDescriptorImageInfo> image_infos(count);
    std::vector<VkWriteDescriptorSet> writes(count);
    size_t buffer = 0, sampler = 0;
    for (u32 i = 0; i < count; i++) {
        writes[i] = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set_,
            .dstBinding = i,
            .descriptorCount = 1,
            .descriptorType = layout_bindings_[i].descriptorType,
        };
        switch (layout_bindings_[i].descriptorType) {
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
            writes[i].pBufferInfo = &buffers_[buffer++].info;
            break;
        case VK_DESCRIPTOR_TYPE_SAMPLER:
            image_infos[i] = {samplers_[sampler++], VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
            writes[i].pImageInfo = &image_infos[i];
            break;
        default:
            image_infos[i] = {VK_NULL_HANDLE, images_[image_slot_[i]].view,
                              VK_IMAGE_LAYOUT_GENERAL};
            writes[i].pImageInfo = &image_infos[i];
            break;
        }
    }
    api.vkUpdateDescriptorSets(handle, count, writes.data(), 0, nullptr);

    // The detiling sets.
    const auto detile_set = [&](PreparedImage& p) {
        if (!p.detile) {
            return true;
        }
        const VkDescriptorSetAllocateInfo detile_allocate{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &p.detile_set_layout,
        };
        if (api.vkAllocateDescriptorSets(handle, &detile_allocate, &p.detile_set) != VK_SUCCESS) {
            return false;
        }
        const VkDescriptorBufferInfo infos[3] = {
            {p.raw.buffer, 0, p.binding->guest_size},
            {p.linear.buffer, 0, p.binding->guest_size},
            {p.params.buffer, 0, p.binding->tiling_info.size()},
        };
        VkWriteDescriptorSet detile_writes[3];
        for (u32 w = 0; w < 3; w++) {
            detile_writes[w] = {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = p.detile_set,
                .dstBinding = w,
                .descriptorCount = 1,
                .descriptorType =
                    w == 2 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &infos[w],
            };
        }
        api.vkUpdateDescriptorSets(handle, 3, detile_writes, 0, nullptr);
        return true;
    };
    for (PreparedImage& p : images_) {
        if (!detile_set(p)) {
            error = "could not allocate a detiling set";
            return false;
        }
    }
    for (PreparedImage& p : attachments_) {
        if (!detile_set(p)) {
            error = "could not allocate a detiling set";
            return false;
        }
    }
    return true;
}

void ResourceSet::RecordUploads(VkCommandBuffer cmd) const {
    const auto& api = device_.api();
    const auto upload = [&](const PreparedImage& p) {
        const translate::ImageBinding& b = *p.binding;
        VkBuffer source = p.raw.buffer;
        if (p.detile) {
            api.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.detile);
            api.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.detile_layout, 0,
                                        1, &p.detile_set, 0, nullptr);
            api.vkCmdDispatch(cmd, std::max(b.tiling_groups, 1u), 1, 1);
            const VkMemoryBarrier detiled{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
            };
            api.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &detiled, 0, nullptr,
                                     0, nullptr);
            source = p.linear.buffer;
        }
        if (p.layer_levels && !p.from_layer) {
            // The mips the GPU layer has replace those from game memory (the rest are mips it
            // has not made yet; it fills them from game memory when it does).
            const auto ranges = MipRanges(b, p.layer_levels);
            api.vkCmdCopyBuffer(cmd, p.layer.buffer, source, static_cast<u32>(ranges.size()),
                               ranges.data());
            const VkMemoryBarrier copied{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
            };
            api.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                     VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &copied, 0, nullptr, 0,
                                     nullptr);
        }
        const VkImageSubresourceRange whole{p.all_aspects, 0, b.levels, 0, p.layers};
        VkImageMemoryBarrier to_transfer{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = p.image.image,
            .subresourceRange = whole,
        };
        api.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                                 &to_transfer);
        const std::vector<VkBufferImageCopy> copies = p.Copies();
        api.vkCmdCopyBufferToImage(cmd, source, p.image.image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                   static_cast<u32>(copies.size()), copies.data());
        if (p.stencil) {
            const VkBufferImageCopy stencil = StencilCopy(p);
            api.vkCmdCopyBufferToImage(cmd, p.stencil_in.buffer, p.image.image,
                                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &stencil);
        }
        // GENERAL for whatever uses it next (shaders, attachments, the readback below).
        VkImageMemoryBarrier to_general = to_transfer;
        to_general.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_general.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                                   VK_ACCESS_TRANSFER_READ_BIT |
                                   VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                                   VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                   VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                   VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        to_general.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_general.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        api.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1,
                                 &to_general);
        if (p.read_back) {
            // What the image holds before the run (to count what it changes); the run waits for
            // this read (write after read: an execution barrier).
            api.vkCmdCopyImageToBuffer(cmd, p.image.image, VK_IMAGE_LAYOUT_GENERAL,
                                       p.before.buffer, static_cast<u32>(copies.size()),
                                       copies.data());
            api.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                     VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr,
                                     0, nullptr);
        }
    };
    for (const PreparedImage& p : images_) upload(p);
    for (const PreparedImage& p : attachments_) upload(p);
}

void ResourceSet::RecordReadbacks(VkCommandBuffer cmd, VkPipelineStageFlags stages) const {
    const auto& api = device_.api();
    // The writes of the stages the run used (an access flag of a stage not in `stages` is
    // invalid).
    VkAccessFlags written = 0;
    if (stages & (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT)) {
        written |= VK_ACCESS_SHADER_WRITE_BIT;
    }
    if (stages & VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT) {
        written |= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    }
    if (stages & VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT) {
        written |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    }
    const VkMemoryBarrier barrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = written,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT,
    };
    api.vkCmdPipelineBarrier(cmd, stages, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 1, &barrier, 0, nullptr, 0, nullptr);
    // Written images back into their linear buffers (the layout stays GENERAL).
    bool read_back = false;
    const auto read = [&](const PreparedImage& p) {
        if (!p.read_back) {
            return;
        }
        const std::vector<VkBufferImageCopy> copies = p.Copies();
        api.vkCmdCopyImageToBuffer(cmd, p.image.image, VK_IMAGE_LAYOUT_GENERAL, p.after.buffer,
                                   static_cast<u32>(copies.size()), copies.data());
        if (p.stencil) {
            const VkBufferImageCopy stencil = StencilCopy(p);
            api.vkCmdCopyImageToBuffer(cmd, p.image.image, VK_IMAGE_LAYOUT_GENERAL,
                                       p.stencil_out.buffer, 1, &stencil);
        }
        read_back = true;
    };
    for (const PreparedImage& p : images_) read(p);
    for (const PreparedImage& p : attachments_) read(p);
    if (read_back) {
        const VkMemoryBarrier to_host{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
        };
        api.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                                 0, 1, &to_host, 0, nullptr, 0, nullptr);
    }
}

void ResourceSet::Collect(ResourceReport& report, std::vector<u64>& faults) const {
    if (uses_page_table_) {
        // Bit p of the fault buffer: page p was looked for in vain.
        DmaTables* tables = GetDmaTables(device_);
        const auto* words = reinterpret_cast<const u32*>(tables->faults.data);
        for (u64 w = 0; w < tables->faults.size / 4; w++) {
            for (u32 bits = words[w]; bits; bits &= bits - 1) {
                faults.push_back(w * 32 + std::countr_zero(bits));
            }
        }
    }
    for (const BufferSlot& slot : buffers_) {
        const auto& binding = *slot.binding;
        if (binding.kind != translate::BufferKind::Guest || !slot.size ||
            !(binding.written || report.compare_inputs)) {
            continue;
        }
        const Arena* arena = ArenaOf(binding.address);
        if (!arena) {
            continue;
        }
        const u8* begin = arena->buffer.data + (binding.address - arena->start);
        WrittenRange& range = report.written.emplace_back();
        range.address = binding.address;
        range.input = !binding.written;
        range.bytes.assign(begin, begin + slot.size);
        for (size_t b = 0; b < slot.before.size(); b++) {
            range.changed += slot.before[b] != range.bytes[b];
        }
    }
    const auto collect = [&](const PreparedImage& p, bool attachment) {
        if (!p.read_back) {
            return;
        }
        WrittenImage& image = report.images.emplace_back();
        image.binding = *p.binding;
        image.input = !p.binding->written && !attachment;
        if (report.compare_inputs && !image.input) {
            image.before.assign(p.before.data, p.before.data + p.binding->guest_size);
        }
        image.bytes.assign(p.after.data, p.after.data + p.binding->guest_size);
        for (u64 b = 0; b < p.binding->guest_size; b++) {
            image.changed += p.before.data[b] != p.after.data[b];
        }
        if (p.stencil) {
            const u64 size = p.binding->StencilBytes();
            image.stencil.assign(p.stencil_out.data, p.stencil_out.data + size);
            for (u64 b = 0; b < size; b++) {
                image.stencil_changed += p.stencil_in.data[b] != p.stencil_out.data[b];
            }
        }
    };
    for (const PreparedImage& p : images_) collect(p, false);
    for (const PreparedImage& p : attachments_) collect(p, true);
}

std::string RunWithDynamicPages(
    const std::function<void(std::vector<u64>& pages, std::vector<u64>& faults)>& once,
    const std::function<bool()>& done, ResourceReport& report) {
    std::vector<u64> pages;
    std::set<u64> tried;
    std::string skipped;
    u32 missing = 0;
    for (u32 round = 0;; round++) {
        std::vector<u64> faults;
        once(pages, faults);
        if (done() || faults.empty()) {
            break;
        }
        bool added = false;
        for (u64 page : faults) {
            if (tried.insert(page).second) {
                pages.push_back(page);
                added = true;
            }
        }
        if (!added || round == 8) {
            missing = static_cast<u32>(faults.size());
            break;
        }
        if (pages.size() > MaxDmaPages) {
            skipped = std::format("dynamic loads read more than {} pages", MaxDmaPages);
            break;
        }
    }
    report.dma_pages = static_cast<u32>(pages.size());
    report.dma_missing = missing;
    return skipped;
}

} // namespace render
