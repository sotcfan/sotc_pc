// SPDX-FileCopyrightText: Copyright 2026 sotcport
// SPDX-License-Identifier: GPL-2.0-or-later

// sotcport: FSR 3.1 temporal upscaling of the game's scene (see vk_fsr.h). FSR itself is
// FireBurn/FSR-Vulkan (third_party/externals/fsr-vulkan, MIT): a native Vulkan build of AMD's
// FidelityFX SDK 1.1.4 FSR 3.1 upscaler that records into our command buffer.

#include "video_core/renderer_vulkan/vk_fsr.h"

#include <ffx_vk_portable.h>

#include "common/logging/log.h"
#include "gpu/gpu.h"
#include "gpu/upscaler.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"

namespace Vulkan {

namespace {

FfxVkPortableImage Describe(const VideoCore::Image& image, vk::ImageAspectFlags aspect,
                            FfxVkPortableResourceState state) {
    FfxVkPortableImage described{};
    described.structSize = sizeof(described);
    described.image = image.GetImage();
    described.format = static_cast<VkFormat>(image.info.pixel_format);
    described.extent = {image.info.size.width, image.info.size.height};
    described.mipCount = 1;
    described.arrayLayers = 1;
    described.usage = static_cast<VkImageUsageFlags>(image.backing->image.image_ci.usage);
    described.aspect = static_cast<VkImageAspectFlags>(aspect);
    described.state = state;
    return described;
}

} // namespace

FsrPass::FsrPass(const Instance& instance_, Scheduler& scheduler_, Runtime& runtime_,
                 VideoCore::TextureCache& texture_cache_)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_},
      texture_cache{texture_cache_} {}

void FsrPass::LoadConfig() {
    const auto& config = gpu::upscaler::GetConfig();
    if (!config.enabled) {
        // Not set up yet (or off): asked again on the next use, until the game starts drawing.
        return;
    }
    configured = true;
    enabled = true;
    render_width = config.render_width;
    render_height = config.render_height;
    output_width = config.output_width;
    output_height = config.output_height;
    sharpness = config.sharpness;
}

FsrPass::~FsrPass() {
    if (context) {
        ffxVkPortableUpscaleContextDestroy(context);
    }
}

bool FsrPass::CreateContext() {
    FfxVkPortableDeviceInfo device{};
    device.structSize = sizeof(device);
    device.physicalDevice = static_cast<VkPhysicalDevice>(instance.GetPhysicalDevice());
    device.device = static_cast<VkDevice>(instance.GetDevice());
    device.getDeviceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;
    device.shaderFloat16Enabled = instance.IsShaderFloat16Supported();
    device.subgroupSizeControlEnabled = instance.IsSubgroupSize64Supported();
    device.synchronization2Enabled = VK_TRUE;
    // sotcport: enabled for FSR in Instance::CreateDevice.
    device.shaderStorageImageWriteWithoutFormatEnabled = VK_TRUE;

    FfxVkPortableUpscaleCreateInfo create{};
    create.structSize = sizeof(create);
    // HDR scene color; reversed-Z depth with no far plane; FSR works out the exposure.
    create.flags = FFX_VK_PORTABLE_CONTEXT_HDR_COLOR_INPUT | FFX_VK_PORTABLE_CONTEXT_DEPTH_INVERTED |
                   FFX_VK_PORTABLE_CONTEXT_DEPTH_INFINITE | FFX_VK_PORTABLE_CONTEXT_AUTO_EXPOSURE;
    create.maxRenderSize = {render_width, render_height};
    create.maxOutputSize = {output_width, output_height};
    const auto result = ffxVkPortableUpscaleContextCreate(&device, &create, &context);
    if (result != FFX_VK_PORTABLE_OK || !context) {
        LOG_ERROR(Render_Vulkan, "FSR: could not create the upscaler ({}); the game's own "
                                 "enlargement stays",
                  static_cast<int>(result));
        context = nullptr;
        return false;
    }
    LOG_INFO(Render_Vulkan, "FSR 3.1 upscaler created: {}x{} -> {}x{}", render_width,
             render_height, output_width, output_height);
    return true;
}

void FsrPass::CreateOutput() {
    output = VideoCore::UniqueImage{instance.GetDevice(), instance.GetAllocator()};
    const vk::ImageCreateInfo image_ci{
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR16G16B16A16Sfloat,
        .extent = {output_width, output_height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled |
                 vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst,
        .initialLayout = vk::ImageLayout::eUndefined,
    };
    output.Create(image_ci);
    output_view = instance.GetDevice()
                      .createImageViewUnique(vk::ImageViewCreateInfo{
                          .image = output,
                          .viewType = vk::ImageViewType::e2D,
                          .format = vk::Format::eR16G16B16A16Sfloat,
                          .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
                      })
                      .value;
}

void FsrPass::NoteTarget(VideoCore::ImageId id, const VideoCore::Image& image) {
    if (!Enabled() || image.info.size.width != render_width ||
        image.info.size.height != render_height) {
        return;
    }
    if (image.info.props.is_depth) {
        depth_id = id;
    } else if (image.info.pixel_format == vk::Format::eR16G16Sfloat) {
        motion_id = id;
    }
}

bool FsrPass::IsScene(const VideoCore::Image& image) {
    return Enabled() && !failed && image.info.size.width == render_width &&
           image.info.size.height == render_height &&
           image.info.pixel_format == vk::Format::eR16G16B16A16Sfloat;
}

bool FsrPass::IsOutputSized(const VideoCore::Image& image) {
    return Enabled() && image.info.size.width == output_width &&
           image.info.size.height == output_height;
}

bool FsrPass::ReplaceScene(VideoCore::ImageId scene_id, vk::ImageView& view,
                           vk::ImageLayout& layout) {
    // Until the scene has drawn its depth and motion vectors (logos, menus before the game),
    // the game's own image stays.
    if (failed || !depth_id || !motion_id) {
        return false;
    }
    if (!context) {
        if (!CreateContext()) {
            failed = true;
            return false;
        }
        CreateOutput();
    }

    // Once per frame: the final draw may sample the scene more than once.
    const auto presented = gpu::FramesPresented();
    if (presented != dispatched_for) {
        dispatched_for = presented;
        auto& scene = texture_cache.GetImage(scene_id);
        auto& depth = texture_cache.GetImage(depth_id);
        auto& motion = texture_cache.GetImage(motion_id);

        scheduler.EndRendering();
        constexpr auto read_stage = vk::PipelineStageFlagBits2::eComputeShader;
        constexpr auto read_access = vk::AccessFlagBits2::eShaderRead;
        runtime.Transit(&scene, vk::ImageLayout::eShaderReadOnlyOptimal, read_stage, read_access);
        runtime.Transit(&depth, vk::ImageLayout::eShaderReadOnlyOptimal, read_stage, read_access);
        runtime.Transit(&motion, vk::ImageLayout::eShaderReadOnlyOptimal, read_stage, read_access);
        runtime.FlushBarriers();

        const auto jitter = gpu::upscaler::GetJitter();
        const auto now = std::chrono::steady_clock::now();
        const float frame_ms =
            last_dispatch.time_since_epoch().count()
                ? std::chrono::duration<float, std::milli>(now - last_dispatch).count()
                : 16.7f;
        last_dispatch = now;

        FfxVkPortableUpscaleDispatchInfo info{};
        info.structSize = sizeof(info);
        info.color = Describe(scene, vk::ImageAspectFlagBits::eColor,
                              FFX_VK_PORTABLE_RESOURCE_STATE_COMPUTE_READ);
        info.depth = Describe(depth, vk::ImageAspectFlagBits::eDepth,
                              FFX_VK_PORTABLE_RESOURCE_STATE_COMPUTE_READ);
        info.motionVectors = Describe(motion, vk::ImageAspectFlagBits::eColor,
                                      FFX_VK_PORTABLE_RESOURCE_STATE_COMPUTE_READ);
        // Not given (FSR works out the exposure; no reactive or composition masks yet), but
        // described: the API checks every image's struct size.
        info.exposure.structSize = sizeof(info.exposure);
        info.reactiveMask.structSize = sizeof(info.reactiveMask);
        info.transparencyAndCompositionMask.structSize = sizeof(info.transparencyAndCompositionMask);
        info.output.structSize = sizeof(info.output);
        info.output.image = static_cast<VkImage>(vk::Image{output});
        info.output.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        info.output.extent = {output_width, output_height};
        info.output.mipCount = 1;
        info.output.arrayLayers = 1;
        info.output.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                            VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        info.output.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
        info.output.state = FFX_VK_PORTABLE_RESOURCE_STATE_UNORDERED_ACCESS;
        info.jitterOffset = {jitter.x, jitter.y};
        // uv(now) - uv(previous) in texture units; FSR wants the offset to the previous frame
        // in render pixels.
        info.motionVectorScale = {-float(render_width), -float(render_height)};
        info.renderSize = {render_width, render_height};
        info.outputSize = {output_width, output_height};
        info.frameTimeMilliseconds = frame_ms;
        info.preExposure = 1.0f;
        info.cameraNear = 0.1f;
        info.cameraFar = 100000.0f;
        info.cameraVerticalFovRadians = 0.9f;
        info.viewSpaceToMeters = 1.0f;
        info.sharpness = sharpness;
        info.enableSharpening = sharpness > 0.0f;
        info.reset = frame_id == 0;
        info.frameId = ++frame_id;

        const bool first = !output_initialized;
        output_initialized = true;
        scheduler.Record([context = context, info, first, output_image = vk::Image{output}](
                             vk::CommandBuffer cmdbuf) mutable {
            const vk::ImageSubresourceRange range{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
            if (first) {
                // The output starts undefined; FSR expects it in GENERAL (UNORDERED_ACCESS).
                const vk::ImageMemoryBarrier2 to_general{
                    .srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe,
                    .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                    .dstAccessMask = vk::AccessFlagBits2::eShaderWrite,
                    .oldLayout = vk::ImageLayout::eUndefined,
                    .newLayout = vk::ImageLayout::eGeneral,
                    .image = output_image,
                    .subresourceRange = range,
                };
                cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                    .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_general});
            } else {
                // The previous frame's final draw read it.
                const vk::ImageMemoryBarrier2 reuse{
                    .srcStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
                    .srcAccessMask = vk::AccessFlagBits2::eShaderRead,
                    .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                    .dstAccessMask = vk::AccessFlagBits2::eShaderWrite,
                    .oldLayout = vk::ImageLayout::eGeneral,
                    .newLayout = vk::ImageLayout::eGeneral,
                    .image = output_image,
                    .subresourceRange = range,
                };
                cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                    .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &reuse});
            }
            info.commandBuffer = static_cast<VkCommandBuffer>(cmdbuf);
            const auto result = ffxVkPortableUpscaleContextRecordDispatch(context, &info);
            if (result != FFX_VK_PORTABLE_OK) {
                static bool reported = false;
                if (!reported) {
                    reported = true;
                    LOG_ERROR(Render_Vulkan, "FSR: dispatch refused ({})",
                              static_cast<int>(result));
                }
            }
            // Its result is read by the final draw's pixel shader.
            const vk::ImageMemoryBarrier2 to_read{
                .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
                .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                .oldLayout = vk::ImageLayout::eGeneral,
                .newLayout = vk::ImageLayout::eGeneral,
                .image = output_image,
                .subresourceRange = range,
            };
            cmdbuf.pipelineBarrier2(
                vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_read});
        });
    }
    view = *output_view;
    layout = vk::ImageLayout::eGeneral;
    return true;
}

} // namespace Vulkan
