#include "render/native_compute.h"

#include <array>
#include <span>

#include "render/vulkan/device.h"

namespace render {

namespace {

// One run of the dispatch; `pages`: the game memory pages its page table gives, `faults`: the
// pages its dynamic loads looked for in vain (see RunWithDynamicPages).
void RunOnce(const ComputeDispatch& dispatch, const GuestReader& read,
             const LayerImageReader& read_image, std::vector<u64>& pages,
             std::vector<u64>& faults, NativeComputeResult& result) {
    const bool compare_inputs = result.compare_inputs; // the caller's request
    result = NativeComputeResult{};
    result.compare_inputs = compare_inputs;
    vk::Device* device = vk::Device::Get();
    if (!device) {
        result.error = "no native device";
        return;
    }
    const auto& api = device->api();
    const VkDevice handle = device->handle();

    // --- the shader ------------------------------------------------------------------------
    translate::TranslatedProgram& program = result.program;
    if (!translate::TranslateCompute(dispatch.program, program, result.error)) {
        return;
    }
    result.hash = program.hash;
    const std::array<ResourceSet::Stage, 1> stages = {{{&program, VK_SHADER_STAGE_COMPUTE_BIT}}};
    result.skipped = ResourceSet::Unsupported(stages);
    if (!result.skipped.empty()) {
        return;
    }

    // --- resources -------------------------------------------------------------------------
    Objects objects{*device};
    ResourceSet resources{*device, objects, read, read_image};
    const u64 lds_bytes = u64{program.shared_memory_bytes} * dispatch.groups[0] *
                          dispatch.groups[1] * dispatch.groups[2];
    if (!resources.Prepare(stages, lds_bytes, pages, result, result.error) ||
        !resources.Finish(result.error)) {
        return;
    }

    // --- pipeline (the recompiler's layout: set 0, a 128-byte push constant block) ---------
    const VkShaderModuleCreateInfo module_info{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = program.spirv.size() * 4,
        .pCode = program.spirv.data(),
    };
    VkShaderModule module = VK_NULL_HANDLE;
    if (api.vkCreateShaderModule(handle, &module_info, nullptr, &module) != VK_SUCCESS) {
        result.error = "could not create the shader module";
        return;
    }
    objects.modules.push_back(module);
    const VkPushConstantRange push_range{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 128,
    };
    const VkDescriptorSetLayout set_layout = resources.SetLayout();
    const VkPipelineLayoutCreateInfo layout_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };
    VkPipelineLayout layout = VK_NULL_HANDLE;
    if (api.vkCreatePipelineLayout(handle, &layout_info, nullptr, &layout) != VK_SUCCESS) {
        result.error = "could not create the pipeline layout";
        return;
    }
    objects.layouts.push_back(layout);
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
                .module = module,
                .pName = "main",
            },
        .layout = layout,
    };
    VkPipeline pipeline = VK_NULL_HANDLE;
    if (api.vkCreateComputePipelines(handle, VK_NULL_HANDLE, 1, &pipeline_info, nullptr,
                                     &pipeline) != VK_SUCCESS) {
        result.error = "could not create the compute pipeline";
        return;
    }
    objects.pipelines.push_back(pipeline);

    // --- run --------------------------------------------------------------------------------
    const VkDescriptorSet set = resources.Set();
    const PushData push = resources.Push();
    const bool ran = device->Run([&](VkCommandBuffer cmd) {
        resources.RecordUploads(cmd);
        api.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        api.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0,
                                    nullptr);
        api.vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        api.vkCmdDispatch(cmd, dispatch.groups[0], dispatch.groups[1], dispatch.groups[2]);
        resources.RecordReadbacks(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    });
    if (!ran) {
        result.error = "the dispatch failed";
        return;
    }
    resources.Collect(result, faults);
}

} // namespace

void RunNativeCompute(const ComputeDispatch& dispatch, const GuestReader& read,
                      const LayerImageReader& read_image, NativeComputeResult& result) {
    const std::string skipped = RunWithDynamicPages(
        [&](std::vector<u64>& pages, std::vector<u64>& faults) {
            RunOnce(dispatch, read, read_image, pages, faults, result);
        },
        [&] { return !result.error.empty() || !result.skipped.empty(); }, result);
    if (!skipped.empty()) {
        result.skipped = skipped;
    }
}

} // namespace render
