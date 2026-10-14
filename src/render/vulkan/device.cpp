#include "render/vulkan/device.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string_view>
#include <vector>

#include "common/log.h"

namespace render::vk {

namespace {

// Device extensions the recompiler's SPIR-V may need (SPV_* extensions it emits, see
// shader_recompiler/backend/spirv), enabled when the GPU has them.
constexpr const char* WantedExtensions[] = {
    "VK_KHR_workgroup_memory_explicit_layout",
    "VK_KHR_shader_clock",
    "VK_KHR_fragment_shader_barycentric",
    "VK_EXT_shader_atomic_float",
    "VK_EXT_shader_atomic_float2",
    "VK_EXT_shader_stencil_export",
    "VK_AMD_shader_explicit_vertex_parameter",
    "VK_AMD_shader_image_load_store_lod",
    "VK_AMD_shader_trinary_minmax",
    "VK_AMD_gcn_shader",
    // Draws, as the GPU layer renders them: clip space z in [-w, w], no depth clipping, the
    // last vertex provoking, depth ranges outside [0, 1].
    "VK_EXT_depth_clip_control",
    "VK_EXT_depth_clip_enable",
    "VK_EXT_provoking_vertex",
    "VK_EXT_depth_range_unrestricted",
    // Out-of-bounds buffer and image accesses read 0, as the GCN hardware's bounds checks
    // (num_records, image extents) make them, and as the GPU layer has them.
    "VK_EXT_robustness2",
    // Views clamped to a minimum LOD (the T#'s MIN_LOD), as the GPU layer's.
    "VK_EXT_image_view_min_lod",
    // The color write mask as dynamic state, as the GPU layer sets it: RADV then exports pixel
    // shader colors at 32 bits and the color blocks round them (a static mask lets it convert
    // to 16 bits in the shader, rounding toward zero: 1 ULP apart from the layer's).
    "VK_EXT_extended_dynamic_state3",
};

const char* ResultName(VkResult result) {
    switch (result) {
    case VK_SUCCESS: return "success";
    case VK_ERROR_OUT_OF_HOST_MEMORY: return "out of host memory";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "out of device memory";
    case VK_ERROR_INITIALIZATION_FAILED: return "initialization failed";
    case VK_ERROR_DEVICE_LOST: return "device lost";
    case VK_ERROR_EXTENSION_NOT_PRESENT: return "extension not present";
    case VK_ERROR_FEATURE_NOT_PRESENT: return "feature not present";
    case VK_ERROR_LAYER_NOT_PRESENT: return "layer not present";
    default: return "error";
    }
}

} // namespace

Device* Device::Get() {
    static Device* device = [] {
        auto* created = new Device;
        if (!created->Initialize()) {
            LOG_ERROR("NativeVk", "the native renderer's Vulkan device is not available");
            return static_cast<Device*>(nullptr);
        }
        return created;
    }();
    return device;
}

bool Device::Initialize() {
#ifdef _WIN32
    HMODULE library = LoadLibraryA("vulkan-1.dll");
    if (!library) {
        LOG_ERROR("NativeVk", "vulkan-1.dll could not be loaded (error {})", GetLastError());
        return false;
    }
    const auto get_instance_proc = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
        reinterpret_cast<void*>(GetProcAddress(library, "vkGetInstanceProcAddr")));
#else
    void* library = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!library) {
        LOG_ERROR("NativeVk", "libvulkan.so.1: {}", dlerror());
        return false;
    }
    const auto get_instance_proc =
        reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(library, "vkGetInstanceProcAddr"));
#endif
    const auto create_instance = reinterpret_cast<PFN_vkCreateInstance>(
        get_instance_proc(VK_NULL_HANDLE, "vkCreateInstance"));
    if (!create_instance) {
        return false;
    }

    // --- instance --------------------------------------------------------------------------
    const VkApplicationInfo app{
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "sotcport native renderer",
        .apiVersion = VK_API_VERSION_1_3,
    };
    const char* validation = "VK_LAYER_KHRONOS_validation";
    const char* env = std::getenv("SOTC_NATIVE_VK_VALIDATION");
    const bool validate = env && *env == '1';
    const VkInstanceCreateInfo instance_info{
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app,
        .enabledLayerCount = validate ? 1u : 0u,
        .ppEnabledLayerNames = validate ? &validation : nullptr,
    };
    if (const VkResult result = create_instance(&instance_info, nullptr, &instance_);
        result != VK_SUCCESS) {
        LOG_ERROR("NativeVk", "vkCreateInstance: {}", ResultName(result));
        return false;
    }
#define RENDER_VK_LOAD_INSTANCE(name)                                                            \
    api_.name = reinterpret_cast<PFN_##name>(get_instance_proc(instance_, #name));
    RENDER_VK_INSTANCE_FUNCTIONS(RENDER_VK_LOAD_INSTANCE)
#undef RENDER_VK_LOAD_INSTANCE

    // --- physical device: the discrete GPU (the user has one AMD card) -----------------------
    u32 count = 0;
    api_.vkEnumeratePhysicalDevices(instance_, &count, nullptr);
    std::vector<VkPhysicalDevice> physicals(count);
    api_.vkEnumeratePhysicalDevices(instance_, &count, physicals.data());
    VkPhysicalDeviceProperties2 properties{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    for (VkPhysicalDevice candidate : physicals) {
        api_.vkGetPhysicalDeviceProperties2(candidate, &properties);
        if (properties.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ||
            physical_ == VK_NULL_HANDLE) {
            physical_ = candidate;
            if (properties.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                break;
            }
        }
    }
    if (!physical_) {
        LOG_ERROR("NativeVk", "no Vulkan GPU");
        return false;
    }
    api_.vkGetPhysicalDeviceProperties2(physical_, &properties);
    storage_alignment_ = properties.properties.limits.minStorageBufferOffsetAlignment;
    max_anisotropy_ = properties.properties.limits.maxSamplerAnisotropy;

    // A queue family with compute (the graphics family has it too; one queue is enough here).
    api_.vkGetPhysicalDeviceQueueFamilyProperties(physical_, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    api_.vkGetPhysicalDeviceQueueFamilyProperties(physical_, &count, families.data());
    queue_family_ = ~0u;
    for (u32 i = 0; i < count; i++) {
        if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            queue_family_ = i;
            break;
        }
    }
    if (queue_family_ == ~0u) {
        return false;
    }

    // --- extensions and features: whatever the GPU offers of what the SPIR-V may use ---------
    api_.vkEnumerateDeviceExtensionProperties(physical_, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> available(count);
    api_.vkEnumerateDeviceExtensionProperties(physical_, nullptr, &count, available.data());
    auto has = [&](std::string_view name) {
        for (const auto& extension : available) {
            if (name == extension.extensionName) {
                return true;
            }
        }
        return false;
    };
    std::vector<const char*> extensions;
    for (const char* name : WantedExtensions) {
        if (has(name)) {
            extensions.push_back(name);
        }
    }

    // Query the features (core 1.1-1.3 plus the extensions'), then enable all of them: the
    // translated shaders declare what they use. Robust accesses included (robustness2 needs
    // robustBufferAccess too): out of bounds reads give 0, as on the PS4's GPU.
    VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR workgroup_layout{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR};
    VkPhysicalDeviceShaderClockFeaturesKHR clock{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR};
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
    VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT atomic_float2{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_2_FEATURES_EXT};
    VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR barycentric{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR};
    VkPhysicalDeviceDepthClipControlFeaturesEXT clip_control{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_CONTROL_FEATURES_EXT};
    VkPhysicalDeviceDepthClipEnableFeaturesEXT clip_enable{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_ENABLE_FEATURES_EXT};
    VkPhysicalDeviceProvokingVertexFeaturesEXT provoking{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROVOKING_VERTEX_FEATURES_EXT};
    VkPhysicalDeviceRobustness2FeaturesEXT robustness2{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT};
    VkPhysicalDeviceImageViewMinLodFeaturesEXT min_lod{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_VIEW_MIN_LOD_FEATURES_EXT};
    VkPhysicalDeviceExtendedDynamicState3FeaturesEXT dynamic_state3{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT};
    VkPhysicalDeviceVulkan13Features features13{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceVulkan12Features features12{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, .pNext = &features13};
    VkPhysicalDeviceVulkan11Features features11{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, .pNext = &features12};
    VkPhysicalDeviceFeatures2 features{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                                       .pNext = &features11};
    void** tail = &features13.pNext;
    auto chain = [&](bool wanted, auto& feature) {
        if (wanted) {
            *tail = &feature;
            tail = &feature.pNext;
        }
    };
    chain(has("VK_KHR_workgroup_memory_explicit_layout"), workgroup_layout);
    chain(has("VK_KHR_shader_clock"), clock);
    chain(has("VK_EXT_shader_atomic_float"), atomic_float);
    chain(has("VK_EXT_shader_atomic_float2"), atomic_float2);
    chain(has("VK_KHR_fragment_shader_barycentric"), barycentric);
    chain(has("VK_EXT_depth_clip_control"), clip_control);
    chain(has("VK_EXT_depth_clip_enable"), clip_enable);
    chain(has("VK_EXT_provoking_vertex"), provoking);
    chain(has("VK_EXT_robustness2"), robustness2);
    chain(has("VK_EXT_image_view_min_lod"), min_lod);
    chain(has("VK_EXT_extended_dynamic_state3"), dynamic_state3);
    api_.vkGetPhysicalDeviceFeatures2(physical_, &features);
    has_min_lod_ = has("VK_EXT_image_view_min_lod") && min_lod.minLod;
    has_dynamic_write_mask_ = has("VK_EXT_extended_dynamic_state3") &&
                              dynamic_state3.extendedDynamicState3ColorWriteMask;

    const float priority = 1.0f;
    const VkDeviceQueueCreateInfo queue_info{
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = queue_family_,
        .queueCount = 1,
        .pQueuePriorities = &priority,
    };
    const VkDeviceCreateInfo device_info{
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &features,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_info,
        .enabledExtensionCount = static_cast<u32>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data(),
    };
    if (const VkResult result = api_.vkCreateDevice(physical_, &device_info, nullptr, &device_);
        result != VK_SUCCESS) {
        LOG_ERROR("NativeVk", "vkCreateDevice: {}", ResultName(result));
        return false;
    }
#define RENDER_VK_LOAD_DEVICE(name)                                                              \
    api_.name = reinterpret_cast<PFN_##name>(api_.vkGetDeviceProcAddr(device_, #name));
    RENDER_VK_DEVICE_FUNCTIONS(RENDER_VK_LOAD_DEVICE)
#undef RENDER_VK_LOAD_DEVICE
    api_.vkGetDeviceQueue(device_, queue_family_, 0, &queue_);

    // Host-visible, coherent memory: on this GPU (resizable BAR) it can also be device-local.
    VkPhysicalDeviceMemoryProperties memory{};
    api_.vkGetPhysicalDeviceMemoryProperties(physical_, &memory);
    memory_properties_ = memory;
    constexpr VkMemoryPropertyFlags host =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (u32 i = 0; i < memory.memoryTypeCount; i++) {
        if ((memory.memoryTypes[i].propertyFlags & host) == host) {
            host_memory_type_ = i;
            break;
        }
    }

    const VkCommandPoolCreateInfo pool_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = queue_family_,
    };
    api_.vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_);
    const VkCommandBufferAllocateInfo command_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = command_pool_,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    api_.vkAllocateCommandBuffers(device_, &command_info, &command_buffer_);
    const VkFenceCreateInfo fence_info{.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    api_.vkCreateFence(device_, &fence_info, nullptr, &fence_);

    LOG_INFO("NativeVk", "native device on {} ({} extensions{})", properties.properties.deviceName,
             extensions.size(), validate ? ", validation layer on" : "");
    return host_memory_type_ != ~0u;
}

bool Device::CreateHostBuffer(u64 size, HostBuffer& out, bool device_address) {
    out = {};
    const VkBufferCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size ? size : 4,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                 (device_address ? VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT : 0u),
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    if (api_.vkCreateBuffer(device_, &info, nullptr, &out.buffer) != VK_SUCCESS) {
        return false;
    }
    VkMemoryRequirements requirements{};
    api_.vkGetBufferMemoryRequirements(device_, out.buffer, &requirements);
    const VkMemoryAllocateFlagsInfo flags{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
        .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
    };
    const VkMemoryAllocateInfo allocate{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = device_address ? &flags : nullptr,
        .allocationSize = requirements.size,
        .memoryTypeIndex = host_memory_type_,
    };
    void* data = nullptr;
    if (api_.vkAllocateMemory(device_, &allocate, nullptr, &out.memory) != VK_SUCCESS ||
        api_.vkBindBufferMemory(device_, out.buffer, out.memory, 0) != VK_SUCCESS ||
        api_.vkMapMemory(device_, out.memory, 0, VK_WHOLE_SIZE, 0, &data) != VK_SUCCESS) {
        Destroy(out);
        return false;
    }
    out.data = static_cast<u8*>(data);
    out.size = info.size;
    return true;
}

bool Device::CreateImage(const VkImageCreateInfo& info, Image& out) {
    out = {};
    if (api_.vkCreateImage(device_, &info, nullptr, &out.image) != VK_SUCCESS) {
        return false;
    }
    VkMemoryRequirements requirements{};
    api_.vkGetImageMemoryRequirements(device_, out.image, &requirements);
    // The first device-local type the image accepts (any accepted type otherwise).
    u32 type = ~0u;
    for (u32 i = 0; i < memory_properties_.memoryTypeCount; i++) {
        if (!(requirements.memoryTypeBits & (1u << i))) {
            continue;
        }
        if (type == ~0u) {
            type = i;
        }
        if (memory_properties_.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
            type = i;
            break;
        }
    }
    const VkMemoryAllocateInfo allocate{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = requirements.size,
        .memoryTypeIndex = type,
    };
    if (type == ~0u || api_.vkAllocateMemory(device_, &allocate, nullptr, &out.memory) != VK_SUCCESS ||
        api_.vkBindImageMemory(device_, out.image, out.memory, 0) != VK_SUCCESS) {
        Destroy(out);
        return false;
    }
    return true;
}

void Device::Destroy(Image& image) {
    if (image.image) {
        api_.vkDestroyImage(device_, image.image, nullptr);
    }
    if (image.memory) {
        api_.vkFreeMemory(device_, image.memory, nullptr);
    }
    image = {};
}

u64 Device::DeviceAddress(const HostBuffer& buffer) {
    const VkBufferDeviceAddressInfo info{
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .buffer = buffer.buffer,
    };
    return api_.vkGetBufferDeviceAddress(device_, &info);
}

void Device::Destroy(HostBuffer& buffer) {
    if (buffer.buffer) {
        api_.vkDestroyBuffer(device_, buffer.buffer, nullptr);
    }
    if (buffer.memory) {
        api_.vkFreeMemory(device_, buffer.memory, nullptr); // also unmaps
    }
    buffer = {};
}

bool Device::Run(const std::function<void(VkCommandBuffer)>& record) {
    static std::mutex mutex;
    std::scoped_lock lock{mutex};
    api_.vkResetCommandPool(device_, command_pool_, 0);
    const VkCommandBufferBeginInfo begin{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    api_.vkBeginCommandBuffer(command_buffer_, &begin);
    record(command_buffer_);
    api_.vkEndCommandBuffer(command_buffer_);
    const VkSubmitInfo submit{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &command_buffer_,
    };
    api_.vkResetFences(device_, 1, &fence_);
    if (const VkResult result = api_.vkQueueSubmit(queue_, 1, &submit, fence_);
        result != VK_SUCCESS) {
        LOG_ERROR("NativeVk", "vkQueueSubmit: {}", ResultName(result));
        return false;
    }
    if (const VkResult result = api_.vkWaitForFences(device_, 1, &fence_, VK_TRUE, ~0ull);
        result != VK_SUCCESS) {
        LOG_ERROR("NativeVk", "vkWaitForFences: {}", ResultName(result));
        return false;
    }
    return true;
}

} // namespace render::vk
