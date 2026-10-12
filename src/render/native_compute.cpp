#include "render/native_compute.h"

#include <algorithm>
#include <cstring>
#include <format>

#include "hle/kernel/gpu_memory.h"
#include "render/vulkan/device.h"

namespace render {

namespace {

// Buffers are copied whole; the game often describes buffers with an "endless" size (num_records
// of ~4 GiB). Larger ones are cut here (the result says so).
constexpr u64 MaxBufferBytes = 64ull << 20;

// Ranges of game memory that overlap or touch are copied into one native buffer (an "arena"), so
// that bindings aliasing the same memory see each other's writes as on the PS4. An arena starts
// at a 256-byte boundary; each binding is bound at an aligned offset inside it and the shader
// adds the rest (PushData::buf_offsets, which holds less than 256).
struct Arena {
    u64 start = 0;
    u64 end = 0;
    vk::HostBuffer buffer;
};

// Shader::PushData: x/y offset and scale (graphics only), then one byte per buffer binding.
struct PushData {
    std::array<float, 4> viewport{};
    std::array<u8, 56> buf_offsets{};
};
static_assert(sizeof(PushData) <= 128);

// The Vulkan objects of one dispatch, destroyed together.
struct Objects {
    explicit Objects(vk::Device& device_) : device{device_} {}
    vk::Device& device;
    std::vector<vk::HostBuffer> buffers;
    VkShaderModule module = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;

    ~Objects() {
        const auto& api = device.api();
        const VkDevice handle = device.handle();
        if (pipeline) api.vkDestroyPipeline(handle, pipeline, nullptr);
        if (layout) api.vkDestroyPipelineLayout(handle, layout, nullptr);
        if (set_layout) api.vkDestroyDescriptorSetLayout(handle, set_layout, nullptr);
        if (module) api.vkDestroyShaderModule(handle, module, nullptr);
        if (pool) api.vkDestroyDescriptorPool(handle, pool, nullptr);
        for (auto& buffer : buffers) {
            device.Destroy(buffer);
        }
    }
};

} // namespace

void RunNativeCompute(const ComputeDispatch& dispatch, const GuestReader& read,
                      NativeComputeResult& result) {
    result = NativeComputeResult{};
    vk::Device* device = vk::Device::Get();
    if (!device) {
        result.error = "no native device";
        return;
    }
    const auto& api = device->api();

    // --- the shader ------------------------------------------------------------------------
    translate::TranslatedProgram& program = result.program;
    if (!translate::TranslateCompute(dispatch.program, program, result.error)) {
        return;
    }
    result.hash = program.hash;
    if (program.images || program.samplers) {
        result.skipped = std::format("{} images, {} samplers", program.images, program.samplers);
        // What the images are, for planning their support: type WxHxD format/number tile, w(ritten).
        for (const auto& image : program.image_bindings) {
            result.skipped += std::format(" | t{} {}x{}x{} f{}/{} tile{} mips{}-{}{}{}", image.type,
                                          image.width, image.height, image.depth,
                                          image.data_format, image.number_format,
                                          image.tile_mode, image.base_level, image.last_level,
                                          image.written ? " w" : "", image.atomic ? " atomic" : "");
        }
        return;
    }
    for (const auto& binding : program.buffers) {
        if (binding.kind == translate::BufferKind::Unsupported) {
            result.skipped = std::format("special buffer {}", binding.name);
            return;
        }
    }

    // --- game memory into arenas -------------------------------------------------------------
    std::vector<std::pair<u64, u64>> ranges; // [start, end) of each game memory binding
    std::vector<u64> sizes(program.buffers.size());
    for (size_t i = 0; i < program.buffers.size(); i++) {
        const auto& binding = program.buffers[i];
        if (binding.kind != translate::BufferKind::Guest || !binding.address || !binding.size) {
            continue;
        }
        sizes[i] = std::min(binding.size, MaxBufferBytes);
        result.clamped |= sizes[i] != binding.size;
        // Like the GPU layer (ClampRangeSize): no further than the mapped memory goes.
        sizes[i] = hle::kernel::ClampToMapped(binding.address, sizes[i]);
        if (!sizes[i]) {
            continue;
        }
        ranges.emplace_back(binding.address & ~255ull, binding.address + sizes[i]);
    }
    std::ranges::sort(ranges);
    std::vector<Arena> arenas;
    for (const auto& [start, end] : ranges) {
        if (!arenas.empty() && start <= arenas.back().end) {
            arenas.back().end = std::max(arenas.back().end, end);
        } else {
            arenas.push_back({.start = start, .end = end, .buffer = {}});
        }
    }
    Objects objects{*device};
    for (Arena& arena : arenas) {
        if (!device->CreateHostBuffer(arena.end - arena.start, arena.buffer)) {
            result.error = "could not allocate a buffer";
            return;
        }
        objects.buffers.push_back(arena.buffer);
        if (!read(arena.start, arena.buffer.data, arena.end - arena.start)) {
            result.error = std::format("could not read game memory at {:#x}", arena.start);
            return;
        }
    }
    auto arena_of = [&](u64 address) -> const Arena* {
        for (const Arena& arena : arenas) {
            if (address >= arena.start && address < arena.end) {
                return &arena;
            }
        }
        return nullptr;
    };

    // --- descriptors ---------------------------------------------------------------------
    const u32 count = static_cast<u32>(program.buffers.size());
    std::vector<VkDescriptorBufferInfo> infos(count);
    PushData push{};
    const u64 alignment = device->StorageAlignment();
    vk::HostBuffer flatbuf, lds, dummy;
    for (u32 i = 0; i < count; i++) {
        const auto& binding = program.buffers[i];
        switch (binding.kind) {
        case translate::BufferKind::Guest: {
            const Arena* arena = binding.address ? arena_of(binding.address) : nullptr;
            if (!arena) { // null descriptor: an empty buffer
                if (!dummy.buffer && device->CreateHostBuffer(16, dummy)) {
                    std::memset(dummy.data, 0, 16);
                    objects.buffers.push_back(dummy);
                }
                infos[i] = {dummy.buffer, 0, 16};
                break;
            }
            const u64 offset = binding.address - arena->start;
            const u64 aligned = offset / alignment * alignment;
            push.buf_offsets[i] = static_cast<u8>(offset - aligned);
            infos[i] = {arena->buffer.buffer, aligned, sizes[i] + (offset - aligned)};
            break;
        }
        case translate::BufferKind::Flatbuf: {
            const u64 bytes = std::max<u64>(program.flatbuf.size() * 4, 16);
            if (!flatbuf.buffer) {
                if (!device->CreateHostBuffer(bytes, flatbuf)) {
                    result.error = "could not allocate the flattened user data";
                    return;
                }
                std::memset(flatbuf.data, 0, bytes);
                std::memcpy(flatbuf.data, program.flatbuf.data(), program.flatbuf.size() * 4);
                objects.buffers.push_back(flatbuf);
            }
            infos[i] = {flatbuf.buffer, 0, bytes};
            break;
        }
        case translate::BufferKind::SharedMemory: {
            const u64 bytes = std::max<u64>(u64{program.shared_memory_bytes} * dispatch.groups[0] *
                                                dispatch.groups[1] * dispatch.groups[2],
                                            16);
            if (!lds.buffer) {
                if (!device->CreateHostBuffer(bytes, lds)) {
                    result.error = "could not allocate the LDS buffer";
                    return;
                }
                std::memset(lds.data, 0, bytes);
                objects.buffers.push_back(lds);
            }
            infos[i] = {lds.buffer, 0, bytes};
            break;
        }
        case translate::BufferKind::Unsupported:
            break;
        }
    }

    // --- pipeline (the recompiler's layout: set 0, one storage buffer per binding) ----------
    const VkDevice handle = device->handle();
    const VkShaderModuleCreateInfo module_info{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = program.spirv.size() * 4,
        .pCode = program.spirv.data(),
    };
    std::vector<VkDescriptorSetLayoutBinding> layout_bindings(count);
    for (u32 i = 0; i < count; i++) {
        layout_bindings[i] = {
            .binding = i,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        };
    }
    const VkDescriptorSetLayoutCreateInfo set_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = count,
        .pBindings = layout_bindings.data(),
    };
    const VkPushConstantRange push_range{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 128,
    };
    if (api.vkCreateShaderModule(handle, &module_info, nullptr, &objects.module) != VK_SUCCESS ||
        api.vkCreateDescriptorSetLayout(handle, &set_info, nullptr, &objects.set_layout) !=
            VK_SUCCESS) {
        result.error = "could not create the shader module or descriptor layout";
        return;
    }
    const VkPipelineLayoutCreateInfo layout_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &objects.set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };
    // The GCN code assumes 64-wide waves (the recompiler's subgroup operations do too).
    const VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
        .requiredSubgroupSize = 64,
    };
    const VkComputePipelineCreateInfo pipeline_info{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage =
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext = &subgroup,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = objects.module,
                .pName = "main",
            },
    };
    if (api.vkCreatePipelineLayout(handle, &layout_info, nullptr, &objects.layout) != VK_SUCCESS) {
        result.error = "could not create the pipeline layout";
        return;
    }
    VkComputePipelineCreateInfo pipeline_with_layout = pipeline_info;
    pipeline_with_layout.layout = objects.layout;
    if (api.vkCreateComputePipelines(handle, VK_NULL_HANDLE, 1, &pipeline_with_layout, nullptr,
                                     &objects.pipeline) != VK_SUCCESS) {
        result.error = "could not create the compute pipeline";
        return;
    }

    const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, std::max(count, 1u)};
    const VkDescriptorPoolCreateInfo pool_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size,
    };
    VkDescriptorSet set = VK_NULL_HANDLE;
    const VkDescriptorSetAllocateInfo set_allocate{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorSetCount = 1,
        .pSetLayouts = &objects.set_layout,
    };
    if (api.vkCreateDescriptorPool(handle, &pool_info, nullptr, &objects.pool) != VK_SUCCESS) {
        result.error = "could not create the descriptor pool";
        return;
    }
    VkDescriptorSetAllocateInfo allocate = set_allocate;
    allocate.descriptorPool = objects.pool;
    if (api.vkAllocateDescriptorSets(handle, &allocate, &set) != VK_SUCCESS) {
        result.error = "could not allocate the descriptor set";
        return;
    }
    std::vector<VkWriteDescriptorSet> writes(count);
    for (u32 i = 0; i < count; i++) {
        writes[i] = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = i,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[i],
        };
    }
    api.vkUpdateDescriptorSets(handle, count, writes.data(), 0, nullptr);

    // The written ranges before the dispatch, to count what it changed.
    std::vector<std::vector<u8>> before(count);
    for (u32 i = 0; i < count; i++) {
        const auto& binding = program.buffers[i];
        if (binding.kind == translate::BufferKind::Guest && binding.written && sizes[i]) {
            if (const Arena* arena = arena_of(binding.address)) {
                const u8* begin = arena->buffer.data + (binding.address - arena->start);
                before[i].assign(begin, begin + sizes[i]);
            }
        }
    }

    // --- run --------------------------------------------------------------------------------
    const bool ran = device->Run([&](VkCommandBuffer cmd) {
        api.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, objects.pipeline);
        api.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, objects.layout, 0, 1,
                                    &set, 0, nullptr);
        api.vkCmdPushConstants(cmd, objects.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push),
                               &push);
        api.vkCmdDispatch(cmd, dispatch.groups[0], dispatch.groups[1], dispatch.groups[2]);
        const VkMemoryBarrier barrier{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
        };
        api.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0,
                                 nullptr);
    });
    if (!ran) {
        result.error = "the dispatch failed";
        return;
    }

    // --- what the shader may have written ---------------------------------------------------
    for (u32 i = 0; i < count; i++) {
        const auto& binding = program.buffers[i];
        if (binding.kind != translate::BufferKind::Guest || !binding.written || !sizes[i]) {
            continue;
        }
        const Arena* arena = arena_of(binding.address);
        if (!arena) {
            continue;
        }
        const u8* begin = arena->buffer.data + (binding.address - arena->start);
        WrittenRange& range = result.written.emplace_back();
        range.address = binding.address;
        range.bytes.assign(begin, begin + sizes[i]);
        for (size_t b = 0; b < before[i].size(); b++) {
            range.changed += before[i][b] != range.bytes[b];
        }
    }
}

} // namespace render
