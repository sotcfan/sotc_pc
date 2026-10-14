#include "render/native_draw.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <format>
#include <span>
#include <vector>

#include "hle/kernel/gpu_memory.h"
#include "render/vulkan/device.h"

namespace render {

namespace {

// Vertex buffers are copied whole, up to this size (their V# often says "endless").
constexpr u64 MaxVertexBytes = 64ull << 20;

// A copy of game memory in a buffer of the native device (vertex and index data).
bool CopyGuest(vk::Device& device, Objects& objects, const GuestReader& read, u64 address,
               u64 size, vk::HostBuffer& out, std::string& error) {
    if (!device.CreateHostBuffer(std::max<u64>(size, 16), out)) {
        error = "could not allocate a vertex or index buffer";
        return false;
    }
    objects.buffers.push_back(out);
    std::memset(out.data, 0, std::max<u64>(size, 16));
    if (size && !read(address, out.data, size)) {
        error = std::format("could not read game memory at {:#x}", address);
        return false;
    }
    return true;
}

VkShaderModule Module(vk::Device& device, Objects& objects, const std::vector<u32>& spirv) {
    const VkShaderModuleCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spirv.size() * 4,
        .pCode = spirv.data(),
    };
    VkShaderModule module = VK_NULL_HANDLE;
    if (device.api().vkCreateShaderModule(device.handle(), &info, nullptr, &module) !=
        VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    objects.modules.push_back(module);
    return module;
}

void RunOnce(const translate::DrawState& state, const u32* packet, const GuestReader& read,
             const LayerImageReader& read_image, const LayerStencilReader& read_stencil,
             const FastClearQuery& fast_cleared, std::vector<u64>& pages,
             std::vector<u64>& faults, NativeDrawResult& result) {
    const bool compare_inputs = result.compare_inputs; // the caller's request
    result = NativeDrawResult{};
    result.compare_inputs = compare_inputs;
    vk::Device* device = vk::Device::Get();
    if (!device) {
        result.error = "no native device";
        return;
    }
    const auto& api = device->api();
    const VkDevice handle = device->handle();

    // --- shaders and state -----------------------------------------------------------------
    translate::TranslatedDraw& draw = result.draw;
    if (!translate::TranslateDraw(state, packet, draw, result.error)) {
        return;
    }
    result.vs_hash = draw.vs.hash;
    result.ps_hash = draw.ps.hash;
    if (!draw.skipped.empty()) {
        result.skipped = draw.skipped;
        return;
    }
    for (const auto& input : draw.vertex_inputs) {
        if (input.per_instance && input.divisor != 1) {
            result.skipped = "instance step rate";
            return;
        }
    }
    // The pixel shader's bindings come first (PipelineCache::RefreshGraphicsStages).
    const std::array<ResourceSet::Stage, 2> stages = {{
        {&draw.ps, VK_SHADER_STAGE_FRAGMENT_BIT},
        {&draw.vs, VK_SHADER_STAGE_VERTEX_BIT},
    }};
    result.skipped = ResourceSet::Unsupported(stages);
    if (!result.skipped.empty()) {
        return;
    }

    // --- resources, vertex and index data ------------------------------------------------------
    Objects objects{*device};
    ResourceSet resources{*device, objects, read, read_image};
    if (!resources.Prepare(stages, 0, pages, result, result.error)) {
        return;
    }
    std::vector<vk::HostBuffer> vertex_buffers(draw.vertex_inputs.size());
    for (size_t i = 0; i < draw.vertex_inputs.size(); i++) {
        const auto& input = draw.vertex_inputs[i];
        u64 size = std::min(input.size, MaxVertexBytes);
        result.clamped |= size != input.size;
        size = input.address ? hle::kernel::ClampToMapped(input.address, size) : 0;
        if (!CopyGuest(*device, objects, read, input.address, size, vertex_buffers[i],
                       result.error)) {
            return;
        }
    }
    vk::HostBuffer index_buffer;
    if (draw.indexed && !CopyGuest(*device, objects, read, draw.index_address,
                                   u64{draw.count} * (draw.index32 ? 4 : 2), index_buffer,
                                   result.error)) {
        return;
    }

    // --- color targets: as the GPU layer holds them, read back after the draw ------------------
    std::vector<VkRenderingAttachmentInfo> attachments(draw.colors.size());
    std::vector<VkFormat> formats(draw.colors.size(), VK_FORMAT_UNDEFINED);
    std::vector<VkPipelineColorBlendAttachmentState> blends(draw.colors.size());
    u32 layers = 1;
    for (size_t i = 0; i < draw.colors.size(); i++) {
        const translate::ColorTarget& target = draw.colors[i];
        attachments[i] = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        blends[i] = {};
        if (!target.used) {
            continue;
        }
        PreparedImage* image = resources.AddImage(
            target.image, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, true,
            nullptr, result, result.error);
        if (!image) {
            return;
        }
        const VkImageViewCreateInfo view_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = image->image.image,
            .viewType =
                target.view_layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D,
            .format = static_cast<VkFormat>(target.view_format),
            .components = {static_cast<VkComponentSwizzle>(target.swizzle[0]),
                           static_cast<VkComponentSwizzle>(target.swizzle[1]),
                           static_cast<VkComponentSwizzle>(target.swizzle[2]),
                           static_cast<VkComponentSwizzle>(target.swizzle[3])},
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, target.view_level, 1,
                                 target.view_layer, target.view_layers},
        };
        VkImageView view = VK_NULL_HANDLE;
        if (api.vkCreateImageView(handle, &view_info, nullptr, &view) != VK_SUCCESS) {
            result.error = std::format("could not create the view of color target {}", i);
            return;
        }
        objects.views.push_back(view);
        layers = target.view_layers;
        const bool clear = fast_cleared && fast_cleared(target.cmask_address, target.view_layer, target.meta_tracked);
        attachments[i].imageView = view;
        attachments[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        attachments[i].loadOp = clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
        attachments[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        std::memcpy(attachments[i].clearValue.color.uint32, target.clear_value.data(), 16);
        formats[i] = static_cast<VkFormat>(target.view_format);
        blends[i] = {
            .blendEnable = target.blend,
            .srcColorBlendFactor = static_cast<VkBlendFactor>(target.src_color),
            .dstColorBlendFactor = static_cast<VkBlendFactor>(target.dst_color),
            .colorBlendOp = static_cast<VkBlendOp>(target.color_op),
            .srcAlphaBlendFactor = static_cast<VkBlendFactor>(target.src_alpha),
            .dstAlphaBlendFactor = static_cast<VkBlendFactor>(target.dst_alpha),
            .alphaBlendOp = static_cast<VkBlendOp>(target.alpha_op),
            .colorWriteMask = target.write_mask,
        };
    }
    // The depth target (its depth plane: stencil tests are not supported yet).
    VkRenderingAttachmentInfo depth_attachment{.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    VkRenderingAttachmentInfo stencil_attachment = depth_attachment;
    VkFormat depth_format = VK_FORMAT_UNDEFINED;
    std::vector<u8> stencil;
    if (draw.depth.used) {
        const translate::DepthTarget& depth = draw.depth;
        if (depth.stencil) {
            // The stencil plane as the GPU layer holds it (0 when it holds none).
            stencil.assign(depth.image.StencilBytes(), 0);
            result.stencil_from_layer = read_stencil && read_stencil(depth.image, stencil);
        }
        PreparedImage* image = resources.AddImage(
            depth.image, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            true, depth.stencil ? &stencil : nullptr, result, result.error);
        if (!image) {
            return;
        }
        depth_format = static_cast<VkFormat>(depth.image.vk_format);
        const VkImageViewCreateInfo view_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = image->image.image,
            .viewType = depth.view_layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D,
            .format = depth_format,
            .subresourceRange = {image->all_aspects, 0, 1, depth.view_layer, depth.view_layers},
        };
        VkImageView view = VK_NULL_HANDLE;
        if (api.vkCreateImageView(handle, &view_info, nullptr, &view) != VK_SUCCESS) {
            result.error = "could not create the view of the depth target";
            return;
        }
        objects.views.push_back(view);
        layers = std::min(layers, depth.view_layers);
        const bool clear =
            depth.clear_pass ||
            (fast_cleared && fast_cleared(depth.htile_address, depth.view_layer, depth.meta_tracked));
        depth_attachment.imageView = view;
        depth_attachment.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        depth_attachment.loadOp = clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
        depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depth_attachment.clearValue.depthStencil = {depth.clear_value, 0};
        if (depth.stencil) {
            stencil_attachment = depth_attachment;
            stencil_attachment.loadOp =
                depth.stencil_clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
            stencil_attachment.clearValue.depthStencil = {0, depth.stencil_clear_value};
        }
    }
    if (!resources.Finish(result.error)) {
        return;
    }

    // --- pipeline (GraphicsPipeline's, with the draw's dynamic state baked in) ------------------
    const VkShaderModule vs = Module(*device, objects, draw.vs.spirv);
    const VkShaderModule ps = Module(*device, objects, draw.ps.spirv);
    const VkShaderModule tcs =
        draw.patch_points ? Module(*device, objects, draw.tcs_spirv) : VK_NULL_HANDLE;
    const VkShaderModule tes =
        draw.patch_points ? Module(*device, objects, draw.tes_spirv) : VK_NULL_HANDLE;
    if (!vs || !ps || (draw.patch_points && (!tcs || !tes))) {
        result.error = "could not create a shader module";
        return;
    }
    // The GCN code assumes 64-wide waves (the recompiler's subgroup operations do too).
    const VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
        .requiredSubgroupSize = 64,
    };
    std::vector<VkPipelineShaderStageCreateInfo> shader_stages;
    const auto add_stage = [&](VkShaderStageFlagBits stage, VkShaderModule module) {
        shader_stages.push_back({
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = &subgroup,
            .stage = stage,
            .module = module,
            .pName = "main",
        });
    };
    add_stage(VK_SHADER_STAGE_VERTEX_BIT, vs);
    if (draw.patch_points) {
        add_stage(VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, tcs);
        add_stage(VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, tes);
    }
    add_stage(VK_SHADER_STAGE_FRAGMENT_BIT, ps);

    const VkPushConstantRange push_range{
        .stageFlags = VK_SHADER_STAGE_ALL_GRAPHICS,
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

    // One binding per attribute (GraphicsPipeline::GetVertexInputs).
    std::vector<VkVertexInputBindingDescription> vertex_bindings;
    std::vector<VkVertexInputAttributeDescription> vertex_attributes;
    for (u32 i = 0; i < draw.vertex_inputs.size(); i++) {
        const auto& input = draw.vertex_inputs[i];
        vertex_bindings.push_back({
            .binding = i,
            .stride = input.stride,
            .inputRate =
                input.per_instance ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX,
        });
        vertex_attributes.push_back({
            .location = i,
            .binding = i,
            .format = static_cast<VkFormat>(input.vk_format),
            .offset = 0,
        });
    }
    const VkPipelineVertexInputStateCreateInfo vertex_input{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = static_cast<u32>(vertex_bindings.size()),
        .pVertexBindingDescriptions = vertex_bindings.data(),
        .vertexAttributeDescriptionCount = static_cast<u32>(vertex_attributes.size()),
        .pVertexAttributeDescriptions = vertex_attributes.data(),
    };
    const VkPipelineInputAssemblyStateCreateInfo input_assembly{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = static_cast<VkPrimitiveTopology>(draw.topology),
        .primitiveRestartEnable = draw.primitive_restart,
    };
    const VkPipelineTessellationStateCreateInfo tessellation{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO,
        .patchControlPoints = draw.patch_points,
    };
    // The scissor stays inside the render area.
    const s32 x0 = std::clamp<s32>(draw.scissor_offset[0], 0, s32(draw.width));
    const s32 y0 = std::clamp<s32>(draw.scissor_offset[1], 0, s32(draw.height));
    const s32 x1 = std::clamp<s32>(draw.scissor_offset[0] + s32(draw.scissor_extent[0]), x0,
                                   s32(draw.width));
    const s32 y1 = std::clamp<s32>(draw.scissor_offset[1] + s32(draw.scissor_extent[1]), y0,
                                   s32(draw.height));
    const VkViewport viewport{draw.viewport[0], draw.viewport[1], draw.viewport[2],
                              draw.viewport[3], draw.viewport[4], draw.viewport[5]};
    const VkRect2D scissor{{x0, y0}, {u32(x1 - x0), u32(y1 - y0)}};
    const VkPipelineViewportDepthClipControlCreateInfoEXT clip_control{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT,
        .negativeOneToOne = draw.negative_one_to_one,
    };
    const VkPipelineViewportStateCreateInfo viewport_state{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .pNext = &clip_control,
        .viewportCount = 1,
        .pViewports = &viewport,
        .scissorCount = 1,
        .pScissors = &scissor,
    };
    const VkPipelineRasterizationDepthClipStateCreateInfoEXT depth_clip{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_DEPTH_CLIP_STATE_CREATE_INFO_EXT,
        .depthClipEnable = draw.depth_clip,
    };
    const VkPipelineRasterizationProvokingVertexStateCreateInfoEXT provoking{
        .sType =
            VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_PROVOKING_VERTEX_STATE_CREATE_INFO_EXT,
        .pNext = &depth_clip,
        .provokingVertexMode = draw.provoking_last ? VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT
                                                   : VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT,
    };
    const VkPipelineRasterizationStateCreateInfo rasterization{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .pNext = &provoking,
        .depthClampEnable = draw.depth_clamp,
        .rasterizerDiscardEnable = draw.rasterizer_discard,
        .polygonMode = static_cast<VkPolygonMode>(draw.polygon_mode),
        .cullMode = draw.cull_mode,
        .frontFace = static_cast<VkFrontFace>(draw.front_face),
        .depthBiasEnable = draw.depth_bias,
        .depthBiasConstantFactor = draw.bias_constant,
        .depthBiasClamp = draw.bias_clamp,
        .depthBiasSlopeFactor = draw.bias_slope,
        .lineWidth = 1.0f,
    };
    const VkPipelineMultisampleStateCreateInfo multisample{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };
    const auto stencil_state = [](const translate::StencilFace& face) {
        return VkStencilOpState{
            .failOp = static_cast<VkStencilOp>(face.fail_op),
            .passOp = static_cast<VkStencilOp>(face.pass_op),
            .depthFailOp = static_cast<VkStencilOp>(face.depth_fail_op),
            .compareOp = static_cast<VkCompareOp>(face.compare_op),
            .compareMask = face.compare_mask,
            .writeMask = face.write_mask,
            .reference = face.reference,
        };
    };
    const VkPipelineDepthStencilStateCreateInfo depth_stencil{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = draw.depth.test,
        .depthWriteEnable = draw.depth.write,
        .depthCompareOp = static_cast<VkCompareOp>(draw.depth.compare_op),
        .depthBoundsTestEnable = draw.depth.bounds,
        .stencilTestEnable = draw.depth.stencil,
        .front = stencil_state(draw.depth.front),
        .back = stencil_state(draw.depth.back),
        .minDepthBounds = draw.depth.bounds_min,
        .maxDepthBounds = draw.depth.bounds_max,
    };
    VkPipelineColorBlendStateCreateInfo color_blend{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .logicOpEnable = draw.logic_op_enable,
        .logicOp = static_cast<VkLogicOp>(draw.logic_op),
        .attachmentCount = static_cast<u32>(blends.size()),
        .pAttachments = blends.data(),
    };
    std::memcpy(color_blend.blendConstants, draw.blend_constants.data(), 16);
    const VkPipelineRenderingCreateInfo rendering_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount = static_cast<u32>(formats.size()),
        .pColorAttachmentFormats = formats.data(),
        .depthAttachmentFormat = depth_format,
        .stencilAttachmentFormat = draw.depth.stencil ? depth_format : VK_FORMAT_UNDEFINED,
    };
    // The color write masks as dynamic state, as the GPU layer sets them (see device.cpp: it
    // decides how RADV exports the colors, and so their rounding).
    const bool dynamic_mask = device->HasDynamicWriteMask() && !formats.empty();
    const VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_COLOR_WRITE_MASK_EXT};
    const VkPipelineDynamicStateCreateInfo dynamic_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 1,
        .pDynamicStates = dynamic_states,
    };
    const VkGraphicsPipelineCreateInfo pipeline_info{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &rendering_info,
        .stageCount = static_cast<u32>(shader_stages.size()),
        .pStages = shader_stages.data(),
        .pVertexInputState = &vertex_input,
        .pInputAssemblyState = &input_assembly,
        .pTessellationState = draw.patch_points ? &tessellation : nullptr,
        .pViewportState = &viewport_state,
        .pRasterizationState = &rasterization,
        .pMultisampleState = &multisample,
        .pDepthStencilState = &depth_stencil,
        .pColorBlendState = &color_blend,
        .pDynamicState = dynamic_mask ? &dynamic_info : nullptr,
        .layout = layout,
    };
    VkPipeline pipeline = VK_NULL_HANDLE;
    if (api.vkCreateGraphicsPipelines(handle, VK_NULL_HANDLE, 1, &pipeline_info, nullptr,
                                      &pipeline) != VK_SUCCESS) {
        result.error = "could not create the graphics pipeline";
        return;
    }
    objects.pipelines.push_back(pipeline);

    // --- run --------------------------------------------------------------------------------
    const VkDescriptorSet set = resources.Set();
    PushData push = resources.Push();
    push.viewport = draw.push_viewport;
    const VkRenderingInfo rendering{
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = {{0, 0}, {draw.width, draw.height}},
        .layerCount = layers,
        .colorAttachmentCount = static_cast<u32>(attachments.size()),
        .pColorAttachments = attachments.data(),
        .pDepthAttachment = draw.depth.used ? &depth_attachment : nullptr,
        .pStencilAttachment = draw.depth.stencil ? &stencil_attachment : nullptr,
    };
    std::vector<VkBuffer> vertex_handles;
    std::vector<VkDeviceSize> vertex_offsets(vertex_buffers.size(), 0);
    for (const auto& buffer : vertex_buffers) {
        vertex_handles.push_back(buffer.buffer);
    }
    const bool ran = device->Run([&](VkCommandBuffer cmd) {
        resources.RecordUploads(cmd);
        api.vkCmdBeginRendering(cmd, &rendering);
        api.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        if (dynamic_mask) {
            std::vector<VkColorComponentFlags> masks;
            for (const auto& blend : blends) {
                masks.push_back(blend.colorWriteMask);
            }
            api.vkCmdSetColorWriteMaskEXT(cmd, 0, static_cast<u32>(masks.size()), masks.data());
        }
        api.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &set, 0,
                                    nullptr);
        api.vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_ALL_GRAPHICS, 0, sizeof(push),
                               &push);
        if (!vertex_handles.empty()) {
            api.vkCmdBindVertexBuffers(cmd, 0, static_cast<u32>(vertex_handles.size()),
                                       vertex_handles.data(), vertex_offsets.data());
        }
        if (draw.indexed) {
            api.vkCmdBindIndexBuffer(cmd, index_buffer.buffer, 0,
                                     draw.index32 ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_UINT16);
            api.vkCmdDrawIndexed(cmd, draw.count, draw.instances, 0, s32(draw.vertex_offset),
                                 draw.instance_offset);
        } else {
            api.vkCmdDraw(cmd, draw.count, draw.instances, draw.vertex_offset,
                          draw.instance_offset);
        }
        api.vkCmdEndRendering(cmd);
        resources.RecordReadbacks(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                           VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
                                           VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    });
    if (!ran) {
        result.error = "the draw failed";
        return;
    }
    resources.Collect(result, faults);
}

} // namespace

void RunNativeDraw(const translate::DrawState& state, const u32* packet, const GuestReader& read,
                   const LayerImageReader& read_image, const LayerStencilReader& read_stencil,
                   const FastClearQuery& fast_cleared, NativeDrawResult& result) {
    const std::string skipped = RunWithDynamicPages(
        [&](std::vector<u64>& pages, std::vector<u64>& faults) {
            RunOnce(state, packet, read, read_image, read_stencil, fast_cleared, pages, faults,
                    result);
        },
        [&] { return !result.error.empty() || !result.skipped.empty(); }, result);
    if (!skipped.empty()) {
        result.skipped = skipped;
    }
}

} // namespace render
