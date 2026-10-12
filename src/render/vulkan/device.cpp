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
    // translated shaders declare what they use. robustBufferAccess stays off, as on the PS4.
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
    api_.vkGetPhysicalDeviceFeatures2(physical_, &features);
    features.features.robustBufferAccess = VK_FALSE;

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

bool Device::CreateHostBuffer(u64 size, HostBuffer& out) {
    out = {};
    const VkBufferCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size ? size : 4,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    if (api_.vkCreateBuffer(device_, &info, nullptr, &out.buffer) != VK_SUCCESS) {
        return false;
    }
    VkMemoryRequirements requirements{};
    api_.vkGetBufferMemoryRequirements(device_, out.buffer, &requirements);
    const VkMemoryAllocateInfo allocate{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
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
