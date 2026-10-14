#pragma once

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>

// Vulkan structures are filled with designated initializers naming only the fields that matter;
// the rest (pNext, flags...) are meant to be zero.
#pragma clang diagnostic ignored "-Wmissing-designated-field-initializers"

#include <functional>
#include <span>

#include "common/types.h"

// The native renderer's own Vulkan device: a second VkDevice on the same GPU, next to the GPU
// layer's (shadPS4's), sharing nothing with it. While the native renderer is being built it runs
// work in parallel to check it against the GPU layer; later it replaces the layer.
//
// The Vulkan entry points come from our own table (loaded from libvulkan with dlopen), not from
// the GPU layer's global dispatcher, so the two devices never mix function pointers.
//
// SOTC_NATIVE_VK_VALIDATION=1 turns on the Khronos validation layer for this device.

namespace render::vk {

// The entry points the native renderer uses.
#define RENDER_VK_INSTANCE_FUNCTIONS(X)                                                           \
    X(vkDestroyInstance)                                                                          \
    X(vkEnumeratePhysicalDevices)                                                                 \
    X(vkGetPhysicalDeviceProperties2)                                                             \
    X(vkGetPhysicalDeviceFeatures2)                                                               \
    X(vkGetPhysicalDeviceMemoryProperties)                                                        \
    X(vkGetPhysicalDeviceQueueFamilyProperties)                                                   \
    X(vkEnumerateDeviceExtensionProperties)                                                       \
    X(vkCreateDevice)                                                                             \
    X(vkGetDeviceProcAddr)

#define RENDER_VK_DEVICE_FUNCTIONS(X)                                                             \
    X(vkDestroyDevice)                                                                            \
    X(vkGetDeviceQueue)                                                                           \
    X(vkDeviceWaitIdle)                                                                           \
    X(vkQueueSubmit)                                                                              \
    X(vkQueueWaitIdle)                                                                            \
    X(vkCreateBuffer)                                                                             \
    X(vkDestroyBuffer)                                                                            \
    X(vkGetBufferMemoryRequirements)                                                              \
    X(vkAllocateMemory)                                                                           \
    X(vkFreeMemory)                                                                               \
    X(vkBindBufferMemory)                                                                         \
    X(vkMapMemory)                                                                                \
    X(vkCreateShaderModule)                                                                       \
    X(vkDestroyShaderModule)                                                                      \
    X(vkCreateDescriptorSetLayout)                                                                \
    X(vkDestroyDescriptorSetLayout)                                                               \
    X(vkCreatePipelineLayout)                                                                     \
    X(vkDestroyPipelineLayout)                                                                    \
    X(vkCreateComputePipelines)                                                                   \
    X(vkDestroyPipeline)                                                                          \
    X(vkCreateDescriptorPool)                                                                     \
    X(vkDestroyDescriptorPool)                                                                    \
    X(vkAllocateDescriptorSets)                                                                   \
    X(vkUpdateDescriptorSets)                                                                     \
    X(vkCreateCommandPool)                                                                        \
    X(vkDestroyCommandPool)                                                                       \
    X(vkAllocateCommandBuffers)                                                                   \
    X(vkResetCommandPool)                                                                         \
    X(vkBeginCommandBuffer)                                                                       \
    X(vkEndCommandBuffer)                                                                         \
    X(vkCmdBindPipeline)                                                                          \
    X(vkCmdBindDescriptorSets)                                                                    \
    X(vkCmdPushConstants)                                                                         \
    X(vkCmdDispatch)                                                                              \
    X(vkCmdPipelineBarrier)                                                                       \
    X(vkCreateImage)                                                                              \
    X(vkDestroyImage)                                                                             \
    X(vkGetImageMemoryRequirements)                                                               \
    X(vkBindImageMemory)                                                                          \
    X(vkCreateImageView)                                                                          \
    X(vkDestroyImageView)                                                                         \
    X(vkCreateSampler)                                                                            \
    X(vkDestroySampler)                                                                           \
    X(vkCmdCopyBufferToImage)                                                                     \
    X(vkCmdCopyImageToBuffer)                                                                     \
    X(vkCmdCopyBuffer)                                                                            \
    X(vkGetBufferDeviceAddress)                                                                   \
    X(vkCreateGraphicsPipelines)                                                                  \
    X(vkCmdBeginRendering)                                                                        \
    X(vkCmdEndRendering)                                                                          \
    X(vkCmdBindVertexBuffers)                                                                     \
    X(vkCmdBindIndexBuffer)                                                                       \
    X(vkCmdDraw)                                                                                  \
    X(vkCmdDrawIndexed)                                                                           \
    X(vkCmdSetColorWriteMaskEXT)                                                                  \
    X(vkCreateFence)                                                                              \
    X(vkDestroyFence)                                                                             \
    X(vkWaitForFences)                                                                            \
    X(vkResetFences)

struct Api {
#define RENDER_VK_DECLARE(name) PFN_##name name = nullptr;
    RENDER_VK_INSTANCE_FUNCTIONS(RENDER_VK_DECLARE)
    RENDER_VK_DEVICE_FUNCTIONS(RENDER_VK_DECLARE)
#undef RENDER_VK_DECLARE
};

// A buffer in memory both the CPU and the GPU see (host-visible, coherent): the native renderer
// fills its inputs and reads its outputs through `data`.
struct HostBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    u8* data = nullptr;
    u64 size = 0;
};

// An image in device-local memory.
struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
};

class Device {
public:
    // The device, created on first use; null if Vulkan or a suitable GPU is missing (logged).
    static Device* Get();

    const Api& api() const { return api_; }
    VkDevice handle() const { return device_; }
    // minStorageBufferOffsetAlignment: storage buffers are bound at multiples of this.
    u64 StorageAlignment() const { return storage_alignment_; }

    // `device_address`: also usable through its device address (vkGetBufferDeviceAddress).
    bool CreateHostBuffer(u64 size, HostBuffer& out, bool device_address = false);
    u64 DeviceAddress(const HostBuffer& buffer);
    void Destroy(HostBuffer& buffer);
    bool CreateImage(const VkImageCreateInfo& info, Image& out);
    void Destroy(Image& image);
    // maxSamplerAnisotropy.
    float MaxAnisotropy() const { return max_anisotropy_; }
    // VK_EXT_image_view_min_lod is on (views clamped to a minimum LOD).
    bool HasMinLod() const { return has_min_lod_; }
    // The color write mask can be dynamic state (VK_EXT_extended_dynamic_state3).
    bool HasDynamicWriteMask() const { return has_dynamic_write_mask_; }

    // Records commands with `record`, submits them to the compute queue and waits for them.
    bool Run(const std::function<void(VkCommandBuffer)>& record);

private:
    bool Initialize();

    Api api_;
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    u32 queue_family_ = 0;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    u32 host_memory_type_ = ~0u;
    VkPhysicalDeviceMemoryProperties memory_properties_{};
    float max_anisotropy_ = 1.0f;
    bool has_min_lod_ = false;
    bool has_dynamic_write_mask_ = false;
    u64 storage_alignment_ = 256;
};

} // namespace render::vk
