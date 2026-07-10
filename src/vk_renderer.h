#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>

#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "linalg.h"
using namespace linalg::aliases;

#ifndef SHADER_DIR
#define SHADER_DIR "shaders"
#endif

// OS detection macros
#if defined(__APPLE__)
  #define VKR_OS_MACOS 1
  #define VKR_OS_LINUX 0
  #define VKR_OS_WINDOWS 0
#elif defined(__linux__)
  #define VKR_OS_MACOS 0
  #define VKR_OS_LINUX 1
  #define VKR_OS_WINDOWS 0
#elif defined(_WIN32)
  #define VKR_OS_MACOS 0
  #define VKR_OS_LINUX 0
  #define VKR_OS_WINDOWS 1
#else
  #define VKR_OS_MACOS 0
  #define VKR_OS_LINUX 0
  #define VKR_OS_WINDOWS 0
#endif

namespace vkr {

// Per-instance vertex attributes
struct InstanceData {
  float3 pos;
  float speed;
};

// Push-constant block
// has to match particle.vert
struct PushConstants {
  float4x4 viewProj;
  float4 camRight;
  float4 camUp;
  float4 params; // (x = particle radius)
};

inline void vkCheck(VkResult r, const char *what) {
  if (r != VK_SUCCESS) {
    throw std::runtime_error(std::string("Vulkan error in ") + what + ": " +
                             std::to_string((int)r));
  }
}

class VulkanRenderer {
public:
  static constexpr int kMaxFramesInFlight = 2;

  void init(int width, int height, const char *title, uint32_t maxInstances) {
    maxInstances_ = maxInstances;
    initWindow(width, height, title);
    createInstance(title);
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapchain();
    createImageViews();
    createRenderPass();
    createDepthResources();
    createFramebuffers();
    createGraphicsPipeline();
    createQuadVertexBuffer();
    createInstanceBuffers();
    createCommandPoolAndBuffers();
    createSyncObjects();
  }

  GLFWwindow *window() const { return window_; }
  bool shouldClose() const { return glfwWindowShouldClose(window_); }
  void pollEvents() const { glfwPollEvents(); }
  void waitIdle() const { vkDeviceWaitIdle(device_); }

  // Upload the instances and draw one frame.
  void drawFrame(const std::vector<InstanceData> &instances,
                 const float4x4 &viewProj, const float3 &camRight,
                 const float3 &camUp, float radius) {
    vkWaitForFences(device_, 1, &inFlightFences_[currentFrame_], VK_TRUE,
                    UINT64_MAX);

    uint32_t imageIndex = 0;
    VkResult acquire = vkAcquireNextImageKHR(
        device_, swapchain_, UINT64_MAX,
        imageAvailableSemaphores_[currentFrame_], VK_NULL_HANDLE, &imageIndex);
    if (acquire == VK_ERROR_OUT_OF_DATE_KHR) {
      recreateSwapchain();
      return;
    } else if (acquire != VK_SUCCESS && acquire != VK_SUBOPTIMAL_KHR) {
      vkCheck(acquire, "vkAcquireNextImageKHR");
    }

    vkResetFences(device_, 1, &inFlightFences_[currentFrame_]);

    // Upload instance data for this frame.
    uint32_t count = std::min<uint32_t>((uint32_t)instances.size(), maxInstances_);
    if (count > 0) {
      std::memcpy(instanceMapped_[currentFrame_], instances.data(),
                  count * sizeof(InstanceData));
    }
    instanceCount_[currentFrame_] = count;

    PushConstants pc{};
    pc.viewProj = viewProj;
    pc.camRight = float4(camRight, 0.0f);
    pc.camUp = float4(camUp, 0.0f);
    pc.params = float4(radius, 0.0f, 0.0f, 0.0f);

    VkCommandBuffer cmd = commandBuffers_[currentFrame_];
    vkResetCommandBuffer(cmd, 0);
    recordCommandBuffer(cmd, imageIndex, pc);

    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    VkSemaphore waitSemaphores[] = {imageAvailableSemaphores_[currentFrame_]};
    VkPipelineStageFlags waitStages[] = {
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = waitSemaphores;
    submit.pWaitDstStageMask = waitStages;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    VkSemaphore signalSemaphores[] = {
        renderFinishedSemaphores_[currentFrame_]};
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = signalSemaphores;
    vkCheck(vkQueueSubmit(graphicsQueue_, 1, &submit,
                          inFlightFences_[currentFrame_]),
            "vkQueueSubmit");

    VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = signalSemaphores;
    present.swapchainCount = 1;
    present.pSwapchains = &swapchain_;
    present.pImageIndices = &imageIndex;
    VkResult pres = vkQueuePresentKHR(presentQueue_, &present);
    if (pres == VK_ERROR_OUT_OF_DATE_KHR || pres == VK_SUBOPTIMAL_KHR ||
        framebufferResized_) {
      framebufferResized_ = false;
      recreateSwapchain();
    } else if (pres != VK_SUCCESS) {
      vkCheck(pres, "vkQueuePresentKHR");
    }

    currentFrame_ = (currentFrame_ + 1) % kMaxFramesInFlight;
  }

  void cleanup() {
    if (device_ == VK_NULL_HANDLE)
      return;
    vkDeviceWaitIdle(device_);
    cleanupSwapchain();

    if (pipeline_) vkDestroyPipeline(device_, pipeline_, nullptr);
    if (pipelineLayout_) vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
    if (renderPass_) vkDestroyRenderPass(device_, renderPass_, nullptr);

    for (int i = 0; i < kMaxFramesInFlight; i++) {
      if (instanceBuffers_[i]) {
        vkUnmapMemory(device_, instanceMemories_[i]);
        vkDestroyBuffer(device_, instanceBuffers_[i], nullptr);
        vkFreeMemory(device_, instanceMemories_[i], nullptr);
      }
    }
    if (quadBuffer_) {
      vkDestroyBuffer(device_, quadBuffer_, nullptr);
      vkFreeMemory(device_, quadMemory_, nullptr);
    }

    for (int i = 0; i < kMaxFramesInFlight; i++) {
      if (renderFinishedSemaphores_[i])
        vkDestroySemaphore(device_, renderFinishedSemaphores_[i], nullptr);
      if (imageAvailableSemaphores_[i])
        vkDestroySemaphore(device_, imageAvailableSemaphores_[i], nullptr);
      if (inFlightFences_[i])
        vkDestroyFence(device_, inFlightFences_[i], nullptr);
    }
    if (commandPool_) vkDestroyCommandPool(device_, commandPool_, nullptr);
    vkDestroyDevice(device_, nullptr);
    if (surface_) vkDestroySurfaceKHR(instance_, surface_, nullptr);
    if (instance_) vkDestroyInstance(instance_, nullptr);
    if (window_) glfwDestroyWindow(window_);
    glfwTerminate();
    device_ = VK_NULL_HANDLE;
  }

private:
  GLFWwindow *window_ = nullptr;
  bool framebufferResized_ = false;

  VkInstance instance_ = VK_NULL_HANDLE;
  VkSurfaceKHR surface_ = VK_NULL_HANDLE;
  VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue graphicsQueue_ = VK_NULL_HANDLE;
  VkQueue presentQueue_ = VK_NULL_HANDLE;
  uint32_t graphicsFamily_ = 0, presentFamily_ = 0;

  VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
  std::vector<VkImage> swapchainImages_;
  std::vector<VkImageView> swapchainImageViews_;
  VkFormat swapchainFormat_ = VK_FORMAT_UNDEFINED;
  VkExtent2D swapchainExtent_{};
  std::vector<VkFramebuffer> framebuffers_;

  VkImage depthImage_ = VK_NULL_HANDLE;
  VkDeviceMemory depthMemory_ = VK_NULL_HANDLE;
  VkImageView depthView_ = VK_NULL_HANDLE;
  VkFormat depthFormat_ = VK_FORMAT_D32_SFLOAT;

  VkRenderPass renderPass_ = VK_NULL_HANDLE;
  VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
  VkPipeline pipeline_ = VK_NULL_HANDLE;

  VkBuffer quadBuffer_ = VK_NULL_HANDLE;
  VkDeviceMemory quadMemory_ = VK_NULL_HANDLE;
  uint32_t maxInstances_ = 0;
  std::array<VkBuffer, kMaxFramesInFlight> instanceBuffers_{};
  std::array<VkDeviceMemory, kMaxFramesInFlight> instanceMemories_{};
  std::array<void *, kMaxFramesInFlight> instanceMapped_{};
  std::array<uint32_t, kMaxFramesInFlight> instanceCount_{};

  VkCommandPool commandPool_ = VK_NULL_HANDLE;
  std::array<VkCommandBuffer, kMaxFramesInFlight> commandBuffers_{};
  std::array<VkSemaphore, kMaxFramesInFlight> imageAvailableSemaphores_{};
  std::array<VkSemaphore, kMaxFramesInFlight> renderFinishedSemaphores_{};
  std::array<VkFence, kMaxFramesInFlight> inFlightFences_{};
  uint32_t currentFrame_ = 0;

  static void framebufferResizeCallback(GLFWwindow *win, int, int) {
    auto self = reinterpret_cast<VulkanRenderer *>(glfwGetWindowUserPointer(win));
    self->framebufferResized_ = true;
  }

  void initWindow(int width, int height, const char *title) {
    if (!glfwInit())
      throw std::runtime_error("Failed to initialize GLFW");
    if (!glfwVulkanSupported())
      throw std::runtime_error("GLFW: Vulkan not supported");
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window_ = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!window_)
      throw std::runtime_error("Failed to create GLFW window");
    glfwSetWindowUserPointer(window_, this);
    glfwSetFramebufferSizeCallback(window_, framebufferResizeCallback);
  }

  void createInstance(const char *title) {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = title;
    app.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app.pEngineName = "sph-vk";
    app.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app.apiVersion = VK_API_VERSION_1_2;

    uint32_t glfwExtCount = 0;
    const char **glfwExts = glfwGetRequiredInstanceExtensions(&glfwExtCount);
    std::vector<const char *> extensions(glfwExts, glfwExts + glfwExtCount);
    
    // MoltenVK / portability extensions (macOS only)
    #if VKR_OS_MACOS
    extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    #endif

    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    #if VKR_OS_MACOS
    ci.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    #endif
    ci.pApplicationInfo = &app;
    ci.enabledExtensionCount = (uint32_t)extensions.size();
    ci.ppEnabledExtensionNames = extensions.data();
    vkCheck(vkCreateInstance(&ci, nullptr, &instance_), "vkCreateInstance");
  }

  void createSurface() {
    vkCheck(glfwCreateWindowSurface(instance_, window_, nullptr, &surface_),
            "glfwCreateWindowSurface");
  }

  struct QueueFamilies {
    std::optional<uint32_t> graphics;
    std::optional<uint32_t> present;
    bool complete() const { return graphics && present; }
  };

  QueueFamilies findQueueFamilies(VkPhysicalDevice dev) {
    QueueFamilies q;
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, nullptr);
    std::vector<VkQueueFamilyProperties> props(count);
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, props.data());
    for (uint32_t i = 0; i < count; i++) {
      if (props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
        q.graphics = i;
      VkBool32 present = VK_FALSE;
      vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, surface_, &present);
      if (present)
        q.present = i;
      if (q.complete())
        break;
    }
    return q;
  }

  void pickPhysicalDevice() {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance_, &count, nullptr);
    if (count == 0)
      throw std::runtime_error("No Vulkan physical devices");
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance_, &count, devices.data());
    for (auto dev : devices) {
      QueueFamilies q = findQueueFamilies(dev);
      if (q.complete()) {
        physicalDevice_ = dev;
        graphicsFamily_ = *q.graphics;
        presentFamily_ = *q.present;
        break;
      }
    }
    if (physicalDevice_ == VK_NULL_HANDLE)
      throw std::runtime_error("No suitable GPU found");
  }

  void createLogicalDevice() {
    std::set<uint32_t> uniqueFamilies = {graphicsFamily_, presentFamily_};
    std::vector<VkDeviceQueueCreateInfo> queueInfos;
    float priority = 1.0f;
    for (uint32_t fam : uniqueFamilies) {
      VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
      qi.queueFamilyIndex = fam;
      qi.queueCount = 1;
      qi.pQueuePriorities = &priority;
      queueInfos.push_back(qi);
    }

    // Required device extensions. Add portability_subset if the device
    // advertises it (mandatory to enable on MoltenVK when present).
    std::vector<const char *> deviceExts = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    #if VKR_OS_MACOS
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice_, nullptr, &extCount,
                                         nullptr);
    std::vector<VkExtensionProperties> available(extCount);
    vkEnumerateDeviceExtensionProperties(physicalDevice_, nullptr, &extCount,
                                         available.data());
    for (const auto &e : available) {
      if (std::strcmp(e.extensionName, "VK_KHR_portability_subset") == 0) {
        deviceExts.push_back("VK_KHR_portability_subset");
        break;
      }
    }
    #endif

    VkPhysicalDeviceFeatures features{};

    VkDeviceCreateInfo ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    ci.queueCreateInfoCount = (uint32_t)queueInfos.size();
    ci.pQueueCreateInfos = queueInfos.data();
    ci.pEnabledFeatures = &features;
    ci.enabledExtensionCount = (uint32_t)deviceExts.size();
    ci.ppEnabledExtensionNames = deviceExts.data();
    vkCheck(vkCreateDevice(physicalDevice_, &ci, nullptr, &device_),
            "vkCreateDevice");

    vkGetDeviceQueue(device_, graphicsFamily_, 0, &graphicsQueue_);
    vkGetDeviceQueue(device_, presentFamily_, 0, &presentQueue_);
  }

  VkSurfaceFormatKHR chooseSurfaceFormat() {
    uint32_t count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice_, surface_, &count,
                                         nullptr);
    std::vector<VkSurfaceFormatKHR> formats(count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice_, surface_, &count,
                                         formats.data());
    for (const auto &f : formats) {
      if (f.format == VK_FORMAT_B8G8R8A8_SRGB &&
          f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        return f;
    }
    return formats[0];
  }

  VkPresentModeKHR choosePresentMode() {
    uint32_t count = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice_, surface_, &count,
                                              nullptr);
    std::vector<VkPresentModeKHR> modes(count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice_, surface_, &count,
                                              modes.data());
    for (auto m : modes)
      if (m == VK_PRESENT_MODE_MAILBOX_KHR)
        return m;
    return VK_PRESENT_MODE_FIFO_KHR;
  }

  VkExtent2D chooseExtent(const VkSurfaceCapabilitiesKHR &caps) {
    if (caps.currentExtent.width != UINT32_MAX)
      return caps.currentExtent;
    int w, h;
    glfwGetFramebufferSize(window_, &w, &h);
    VkExtent2D e{(uint32_t)w, (uint32_t)h};
    e.width = std::max(caps.minImageExtent.width,
                       std::min(caps.maxImageExtent.width, e.width));
    e.height = std::max(caps.minImageExtent.height,
                        std::min(caps.maxImageExtent.height, e.height));
    return e;
  }

  void createSwapchain() {
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice_, surface_, &caps);
    VkSurfaceFormatKHR surfaceFormat = chooseSurfaceFormat();
    VkPresentModeKHR presentMode = choosePresentMode();
    VkExtent2D extent = chooseExtent(caps);

    uint32_t imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount)
      imageCount = caps.maxImageCount;

    VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    ci.surface = surface_;
    ci.minImageCount = imageCount;
    ci.imageFormat = surfaceFormat.format;
    ci.imageColorSpace = surfaceFormat.colorSpace;
    ci.imageExtent = extent;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    uint32_t families[] = {graphicsFamily_, presentFamily_};
    if (graphicsFamily_ != presentFamily_) {
      ci.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
      ci.queueFamilyIndexCount = 2;
      ci.pQueueFamilyIndices = families;
    } else {
      ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }
    ci.preTransform = caps.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode = presentMode;
    ci.clipped = VK_TRUE;
    ci.oldSwapchain = VK_NULL_HANDLE;
    vkCheck(vkCreateSwapchainKHR(device_, &ci, nullptr, &swapchain_),
            "vkCreateSwapchainKHR");

    uint32_t count = 0;
    vkGetSwapchainImagesKHR(device_, swapchain_, &count, nullptr);
    swapchainImages_.resize(count);
    vkGetSwapchainImagesKHR(device_, swapchain_, &count,
                            swapchainImages_.data());
    swapchainFormat_ = surfaceFormat.format;
    swapchainExtent_ = extent;
  }

  void createImageViews() {
    swapchainImageViews_.resize(swapchainImages_.size());
    for (size_t i = 0; i < swapchainImages_.size(); i++) {
      VkImageViewCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
      ci.image = swapchainImages_[i];
      ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
      ci.format = swapchainFormat_;
      ci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      ci.subresourceRange.levelCount = 1;
      ci.subresourceRange.layerCount = 1;
      vkCheck(vkCreateImageView(device_, &ci, nullptr,
                                &swapchainImageViews_[i]),
              "vkCreateImageView(swapchain)");
    }
  }

  void createRenderPass() {
    VkAttachmentDescription color{};
    color.format = swapchainFormat_;
    color.samples = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentDescription depth{};
    depth.format = depthFormat_;
    depth.samples = VK_SAMPLE_COUNT_1_BIT;
    depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depth.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorRef{
        0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depthRef{
        1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;
    subpass.pDepthStencilAttachment = &depthRef;

    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                       VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.srcAccessMask = 0;
    dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                       VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    std::array<VkAttachmentDescription, 2> attachments = {color, depth};
    VkRenderPassCreateInfo ci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    ci.attachmentCount = (uint32_t)attachments.size();
    ci.pAttachments = attachments.data();
    ci.subpassCount = 1;
    ci.pSubpasses = &subpass;
    ci.dependencyCount = 1;
    ci.pDependencies = &dep;
    vkCheck(vkCreateRenderPass(device_, &ci, nullptr, &renderPass_),
            "vkCreateRenderPass");
  }

  uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
      if ((typeFilter & (1 << i)) &&
          (memProps.memoryTypes[i].propertyFlags & props) == props)
        return i;
    }
    throw std::runtime_error("No suitable memory type");
  }

  void createDepthResources() {
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.extent = {swapchainExtent_.width, swapchainExtent_.height, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.format = depthFormat_;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ci.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCheck(vkCreateImage(device_, &ci, nullptr, &depthImage_),
            "vkCreateImage(depth)");

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(device_, depthImage_, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex =
        findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkCheck(vkAllocateMemory(device_, &ai, nullptr, &depthMemory_),
            "vkAllocateMemory(depth)");
    vkBindImageMemory(device_, depthImage_, depthMemory_, 0);

    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = depthImage_;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = depthFormat_;
    vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    vi.subresourceRange.levelCount = 1;
    vi.subresourceRange.layerCount = 1;
    vkCheck(vkCreateImageView(device_, &vi, nullptr, &depthView_),
            "vkCreateImageView(depth)");
  }

  void createFramebuffers() {
    framebuffers_.resize(swapchainImageViews_.size());
    for (size_t i = 0; i < swapchainImageViews_.size(); i++) {
      std::array<VkImageView, 2> attachments = {swapchainImageViews_[i],
                                                depthView_};
      VkFramebufferCreateInfo ci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
      ci.renderPass = renderPass_;
      ci.attachmentCount = (uint32_t)attachments.size();
      ci.pAttachments = attachments.data();
      ci.width = swapchainExtent_.width;
      ci.height = swapchainExtent_.height;
      ci.layers = 1;
      vkCheck(vkCreateFramebuffer(device_, &ci, nullptr, &framebuffers_[i]),
              "vkCreateFramebuffer");
    }
  }

  static std::vector<char> readFile(const std::string &path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open())
      throw std::runtime_error("Failed to open shader file: " + path);
    size_t size = (size_t)file.tellg();
    std::vector<char> buffer(size);
    file.seekg(0);
    file.read(buffer.data(), size);
    return buffer;
  }

  VkShaderModule createShaderModule(const std::vector<char> &code) {
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = code.size();
    ci.pCode = reinterpret_cast<const uint32_t *>(code.data());
    VkShaderModule module;
    vkCheck(vkCreateShaderModule(device_, &ci, nullptr, &module),
            "vkCreateShaderModule");
    return module;
  }

  void createGraphicsPipeline() {
    auto vertCode = readFile(std::string(SHADER_DIR) + "/particle.vert.spv");
    auto fragCode = readFile(std::string(SHADER_DIR) + "/particle.frag.spv");
    VkShaderModule vert = createShaderModule(vertCode);
    VkShaderModule frag = createShaderModule(fragCode);

    VkPipelineShaderStageCreateInfo vs{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    vs.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vs.module = vert;
    vs.pName = "main";
    VkPipelineShaderStageCreateInfo fs{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    fs.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fs.module = frag;
    fs.pName = "main";
    VkPipelineShaderStageCreateInfo stages[] = {vs, fs};

    // Binding 0: per-vertex quad corner. Binding 1: per-instance data.
    std::array<VkVertexInputBindingDescription, 2> bindings{};
    bindings[0].binding = 0;
    bindings[0].stride = sizeof(float) * 2;
    bindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    bindings[1].binding = 1;
    bindings[1].stride = sizeof(InstanceData);
    bindings[1].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

    std::array<VkVertexInputAttributeDescription, 3> attrs{};
    attrs[0] = {0, 0, VK_FORMAT_R32G32_SFLOAT, 0};
    attrs[1] = {1, 1, VK_FORMAT_R32G32B32_SFLOAT,
                (uint32_t)offsetof(InstanceData, pos)};
    attrs[2] = {2, 1, VK_FORMAT_R32_SFLOAT,
                (uint32_t)offsetof(InstanceData, speed)};

    VkPipelineVertexInputStateCreateInfo vi{
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = (uint32_t)bindings.size();
    vi.pVertexBindingDescriptions = bindings.data();
    vi.vertexAttributeDescriptionCount = (uint32_t)attrs.size();
    vi.pVertexAttributeDescriptions = attrs.data();

    VkPipelineInputAssemblyStateCreateInfo ia{
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1;
    vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = VK_TRUE;
    ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState cba{};
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    cba.blendEnable = VK_FALSE;
    VkPipelineColorBlendStateCreateInfo cb{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1;
    cb.pAttachments = &cba;

    std::array<VkDynamicState, 2> dynStates = {VK_DYNAMIC_STATE_VIEWPORT,
                                               VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{
        VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dyn.dynamicStateCount = (uint32_t)dynStates.size();
    dyn.pDynamicStates = dynStates.data();

    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pcRange.offset = 0;
    pcRange.size = sizeof(PushConstants);

    VkPipelineLayoutCreateInfo pl{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl.pushConstantRangeCount = 1;
    pl.pPushConstantRanges = &pcRange;
    vkCheck(vkCreatePipelineLayout(device_, &pl, nullptr, &pipelineLayout_),
            "vkCreatePipelineLayout");

    VkGraphicsPipelineCreateInfo ci{
        VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    ci.stageCount = 2;
    ci.pStages = stages;
    ci.pVertexInputState = &vi;
    ci.pInputAssemblyState = &ia;
    ci.pViewportState = &vp;
    ci.pRasterizationState = &rs;
    ci.pMultisampleState = &ms;
    ci.pDepthStencilState = &ds;
    ci.pColorBlendState = &cb;
    ci.pDynamicState = &dyn;
    ci.layout = pipelineLayout_;
    ci.renderPass = renderPass_;
    ci.subpass = 0;
    vkCheck(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &ci, nullptr,
                                      &pipeline_),
            "vkCreateGraphicsPipelines");

    vkDestroyShaderModule(device_, vert, nullptr);
    vkDestroyShaderModule(device_, frag, nullptr);
  }

  void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags props, VkBuffer &buffer,
                    VkDeviceMemory &memory) {
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size = size;
    ci.usage = usage;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCheck(vkCreateBuffer(device_, &ci, nullptr, &buffer), "vkCreateBuffer");
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device_, buffer, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, props);
    vkCheck(vkAllocateMemory(device_, &ai, nullptr, &memory),
            "vkAllocateMemory");
    vkBindBufferMemory(device_, buffer, memory, 0);
  }

  void createQuadVertexBuffer() {
    // Two triangles covering [-1,1]^2.
    const float quad[] = {
        -1.f, -1.f, 1.f, -1.f, 1.f, 1.f,
        -1.f, -1.f, 1.f,  1.f, -1.f, 1.f,
    };
    VkDeviceSize size = sizeof(quad);
    createBuffer(size,
                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 quadBuffer_, quadMemory_);
    void *data = nullptr;
    vkMapMemory(device_, quadMemory_, 0, size, 0, &data);
    std::memcpy(data, quad, (size_t)size);
    vkUnmapMemory(device_, quadMemory_);
  }

  void createInstanceBuffers() {
    VkDeviceSize size = sizeof(InstanceData) * maxInstances_;
    for (int i = 0; i < kMaxFramesInFlight; i++) {
      createBuffer(size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                   instanceBuffers_[i], instanceMemories_[i]);
      vkMapMemory(device_, instanceMemories_[i], 0, size, 0,
                  &instanceMapped_[i]);
      instanceCount_[i] = 0;
    }
  }

  void createCommandPoolAndBuffers() {
    VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    ci.queueFamilyIndex = graphicsFamily_;
    vkCheck(vkCreateCommandPool(device_, &ci, nullptr, &commandPool_),
            "vkCreateCommandPool");

    VkCommandBufferAllocateInfo ai{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = commandPool_;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = kMaxFramesInFlight;
    vkCheck(vkAllocateCommandBuffers(device_, &ai, commandBuffers_.data()),
            "vkAllocateCommandBuffers");
  }

  void createSyncObjects() {
    VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (int i = 0; i < kMaxFramesInFlight; i++) {
      vkCheck(vkCreateSemaphore(device_, &si, nullptr,
                                &imageAvailableSemaphores_[i]),
              "vkCreateSemaphore");
      vkCheck(vkCreateSemaphore(device_, &si, nullptr,
                                &renderFinishedSemaphores_[i]),
              "vkCreateSemaphore");
      vkCheck(vkCreateFence(device_, &fi, nullptr, &inFlightFences_[i]),
              "vkCreateFence");
    }
  }

  void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex,
                           const PushConstants &pc) {
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkCheck(vkBeginCommandBuffer(cmd, &bi), "vkBeginCommandBuffer");

    std::array<VkClearValue, 2> clears{};
    clears[0].color = {{0.02f, 0.03f, 0.06f, 1.0f}};
    clears[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rp.renderPass = renderPass_;
    rp.framebuffer = framebuffers_[imageIndex];
    rp.renderArea.offset = {0, 0};
    rp.renderArea.extent = swapchainExtent_;
    rp.clearValueCount = (uint32_t)clears.size();
    rp.pClearValues = clears.data();
    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)swapchainExtent_.width;
    viewport.height = (float)swapchainExtent_.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    VkRect2D scissor{{0, 0}, swapchainExtent_};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_VERTEX_BIT, 0,
                       sizeof(PushConstants), &pc);

    VkBuffer vbs[] = {quadBuffer_, instanceBuffers_[currentFrame_]};
    VkDeviceSize offsets[] = {0, 0};
    vkCmdBindVertexBuffers(cmd, 0, 2, vbs, offsets);

    uint32_t count = instanceCount_[currentFrame_];
    if (count > 0)
      vkCmdDraw(cmd, 6, count, 0, 0);

    vkCmdEndRenderPass(cmd);
    vkCheck(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
  }

  void cleanupSwapchain() {
    if (depthView_) vkDestroyImageView(device_, depthView_, nullptr);
    if (depthImage_) vkDestroyImage(device_, depthImage_, nullptr);
    if (depthMemory_) vkFreeMemory(device_, depthMemory_, nullptr);
    depthView_ = VK_NULL_HANDLE;
    depthImage_ = VK_NULL_HANDLE;
    depthMemory_ = VK_NULL_HANDLE;
    for (auto fb : framebuffers_)
      vkDestroyFramebuffer(device_, fb, nullptr);
    framebuffers_.clear();
    for (auto view : swapchainImageViews_)
      vkDestroyImageView(device_, view, nullptr);
    swapchainImageViews_.clear();
    if (swapchain_) {
      vkDestroySwapchainKHR(device_, swapchain_, nullptr);
      swapchain_ = VK_NULL_HANDLE;
    }
  }

  void recreateSwapchain() {
    int w = 0, h = 0;
    glfwGetFramebufferSize(window_, &w, &h);
    while (w == 0 || h == 0) {
      glfwGetFramebufferSize(window_, &w, &h);
      glfwWaitEvents();
    }
    vkDeviceWaitIdle(device_);
    cleanupSwapchain();
    createSwapchain();
    createImageViews();
    createDepthResources();
    createFramebuffers();
  }
};

} // namespace vkr
