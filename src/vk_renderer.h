#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>

#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
#include <imgui.h>

#include <array>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "linalg.h"
using namespace linalg::aliases;

#include "vk_radix_sort.h"

float *stbi_loadf(const char *filename, int *x, int *y, int *comp,
                  int req_comp);
void stbi_image_free(void *retval_from_stbi_load);

#ifndef SHADER_DIR
#define SHADER_DIR "shaders"
#endif
#ifndef ASSET_DIR
#define ASSET_DIR "."
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

// SPH parameter block
struct SphParams {
  float gravity[4];    // xyz used
  float boundsSize[4]; // xyz used
  float deltaT;
  float smoothingRadius;
  float particleMass;
  float targetDensity;
  float pressureMultiplier;
  float viscosityStrength;
  float collisionDamping;
  float spikyPow2Scale;
  float spikyPow2GradScale;
  float poly6Scale;
  uint32_t numParticles;
  float epsilon;
  uint32_t grid[4]; // xyz = exact grid dimensions, w = total cells
};
static_assert(sizeof(SphParams) == 96,
              "SphParams must match the std140 shader block");

// Push-constant block
// has to match particle.vert
struct PushConstants {
  float4x4 viewProj;
  float4 camRight;
  float4 camUp;
  float4 params; // (x = particle radius)
};

struct InteractionParams {
  float point[4];
  float radius;
  float strength; // > 0 attract, < 0 repel
  float _pad0, _pad1;
};

struct DepthPush {
  float4x4 view;
  float4x4 proj;
  float4 camRight;
  float4 camUp;
  float4 params; // x = particle radius
};

struct BlurPush {
  float dir[2];
  float depthDifferenceStrength;
  float maxScreenSpaceRadius;
  float strength;
  float particleRadius;
  float fillSilhouette;
};

struct WaterPush {
  float4 camRight;
  float4 camUp;
  float4 camForward;
  // x = refraction scale, y = absorption scale, z = base reflectance
  float4 material;
  // RGB Beer-Lambert extinction coefficients.
  float4 extinction;
};

struct ScenePush {
  float4x4 view;
  float4x4 proj;
};

struct SceneLightingParams {
  float4x4 lightVP;
  float4x4 lightView;
  float4 sunDirection;
  float4 extinction;
  // x = fraction of unshadowed ambient light
  float4 shadowParams;
};

struct FluidRenderSettings {
  float renderScale = 0.75f;
  float maxBlurRadius = 12.0f;
  float blurStrength = 1.25f;
  float depthDifferenceStrength = 20.0f;
  int blurIterations = 1;
  float refractionStrength = 0.10f;
  float absorptionScale = 1.1f;
  float extinctionRed = 0.55f;
  float extinctionGreen = 0.25f;
  float extinctionBlue = 0.05f;
  float baseReflectance = 0.02f;
  float sunAzimuthDegrees = 36.0f;
  float sunElevationDegrees = 53.0f;
  float shadowAmbientLight = 0.17f;
  int shadowUpdateInterval = 2;
  int simulationSubsteps = 1;
  float simulationRate = 10.0f;
};

inline void vkCheck(VkResult r, const char *what) {
  if (r != VK_SUCCESS) {
    throw std::runtime_error(std::string("Vulkan error in ") + what + ": " +
                             std::to_string((int)r));
  }
}

class VulkanRenderer {
public:
  // A single simulation state lives in the SSBOs, so we serialize to one
  // frame in flight to avoid two frames racing on those buffers.
  static constexpr int kMaxFramesInFlight = 1;
  // costly 1024x1024 particle raster passes every frame.
  static constexpr uint32_t kWaterShadowMapSize = 512;

  void init(int width, int height, const char *title, uint32_t numParticles) {
    numParticles_ = numParticles;
    initWindow(width, height, title);
    createInstance(title);
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapchain();
    createImageViews();
    createRenderPass();
    createDepthResources();
    createSceneTargets();
    createFramebuffers();
    createComputeBuffers();
    createSceneLightingBuffer();
    createSorter();
    createDescriptorSetLayout();
    createDescriptorPoolAndSet();
    createGraphicsPipeline();
    createComputePipelines();
    createQuadVertexBuffer();
    createEnvironmentTexture();
    createFluidTargets();
    createFluidPipelines();
    createCommandPoolAndBuffers();
    createSyncObjects();
  }

  GLFWwindow *window() const { return window_; }
  void loadSceneMesh(const std::string &path, float scale = 1.0f) {
    std::string resolvedPath = path;
    std::ifstream direct(path);
    if (!direct)
      resolvedPath = std::string(ASSET_DIR) + "/" + path;
    std::vector<float> vertices = loadObj(resolvedPath, scale);
    if (vertices.empty())
      throw std::runtime_error("OBJ contains no renderable triangles: " + path);
    vkDeviceWaitIdle(device_);
    destroyBuffer(sceneMeshBuffer_, sceneMeshMemory_);
    VkDeviceSize size = vertices.size() * sizeof(float);
    createBuffer(size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 sceneMeshBuffer_, sceneMeshMemory_);
    void *data = nullptr;
    vkMapMemory(device_, sceneMeshMemory_, 0, size, 0, &data);
    std::memcpy(data, vertices.data(), (size_t)size);
    vkUnmapMemory(device_, sceneMeshMemory_);
    sceneMeshVertexCount_ = (uint32_t)(vertices.size() / 6);
    std::cout << "Loaded scene mesh " << resolvedPath << " (" << sceneMeshVertexCount_
              << " vertices)" << std::endl;
  }
  bool shouldClose() const { return glfwWindowShouldClose(window_); }
  void pollEvents() const { glfwPollEvents(); }
  void waitIdle() const { vkDeviceWaitIdle(device_); }

  void initImGui() {
    if (imguiInitialized_)
      return;

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    if (!ImGui_ImplGlfw_InitForVulkan(window_, true))
      throw std::runtime_error("Failed to initialize ImGui GLFW backend");

    createImGuiDescriptorPool();
    ImGui_ImplVulkan_InitInfo info{};
    info.ApiVersion = VK_API_VERSION_1_4;
    info.Instance = instance_;
    info.PhysicalDevice = physicalDevice_;
    info.Device = device_;
    info.QueueFamily = graphicsFamily_;
    info.Queue = graphicsQueue_;
    info.DescriptorPool = imguiDescriptorPool_;
    info.RenderPass = renderPass_;
    info.MinImageCount = (uint32_t)swapchainImages_.size();
    info.ImageCount = (uint32_t)swapchainImages_.size();
    info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    info.CheckVkResultFn = [](VkResult result) {
      if (result != VK_SUCCESS)
        std::cerr << "ImGui Vulkan error: " << static_cast<int>(result) << '\n';
    };
    if (!ImGui_ImplVulkan_Init(&info))
      throw std::runtime_error("Failed to initialize ImGui Vulkan backend");
    imguiInitialized_ = true;
  }

  void beginImGuiFrame() {
    if (!imguiInitialized_)
      return;

    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("Fluid controls");
    ImGui::Text("Particles: %u", numParticles_);
    ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
    ImGui::SliderInt("Simulation substeps", &fluidSettings_.simulationSubsteps,
                     1, 8);
    ImGui::SliderFloat("Simulation rate", &fluidSettings_.simulationRate, 0.25f,
                       10.0f, "%.2fx", ImGuiSliderFlags_Logarithmic);
    const int activeSubsteps = simulationSubsteps();
    float frameDt = baseDeltaT_ * fluidSettings_.simulationRate;
    ImGui::Text("Simulated dt/frame: %.4f", frameDt);
    ImGui::Text("Active substeps: %d (dt %.5f)", activeSubsteps,
                frameDt / float(activeSubsteps));
    if (paramsMapped_) {
      auto *physics = reinterpret_cast<SphParams *>(paramsMapped_);
      ImGui::Separator();
      ImGui::TextUnformatted("Fluid physics");
      ImGui::SliderFloat("Gravity", &physics->gravity[1], -20.0f, 0.0f,
                         "%.2f");
      ImGui::SliderFloat("Target density", &physics->targetDensity, 500.0f,
                         4000.0f, "%.0f");
      ImGui::SliderFloat("Pressure stiffness", &physics->pressureMultiplier,
                         1.0f, 500.0f, "%.1f",
                         ImGuiSliderFlags_Logarithmic);
      ImGui::SliderFloat("Surface attraction", &physics->gravity[3], 0.0f,
                         0.25f, "%.3f");
      ImGui::SliderFloat("Viscosity", &physics->viscosityStrength, 0.0f,
                         0.02f, "%.4f");
      ImGui::SliderFloat("Collision damping", &physics->collisionDamping,
                         0.0f, 1.0f, "%.2f");
      if (ImGui::Button("Reset water preset")) {
        physics->gravity[1] = defaultSphParams_.gravity[1];
        physics->gravity[3] = defaultSphParams_.gravity[3];
        physics->targetDensity = defaultSphParams_.targetDensity;
        physics->pressureMultiplier = defaultSphParams_.pressureMultiplier;
        physics->viscosityStrength = defaultSphParams_.viscosityStrength;
        physics->collisionDamping = defaultSphParams_.collisionDamping;
        fluidSettings_.simulationRate = 10.0f;
      }
    }
    ImGui::Separator();
    ImGui::TextUnformatted("Screen-space reconstruction");
    ImGui::SliderFloat("Render scale", &fluidSettings_.renderScale, 0.5f,
                       1.0f, "%.2fx");
    ImGui::SliderFloat("Max blur radius (px)",
                       &fluidSettings_.maxBlurRadius, 1.0f, 32.0f, "%.1f");
    ImGui::SliderFloat("Blur strength", &fluidSettings_.blurStrength, 0.05f,
                       2.0f, "%.2f");
    ImGui::SliderFloat("Depth difference strength",
                       &fluidSettings_.depthDifferenceStrength, 0.1f, 250.0f,
                       "%.1f", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderInt("Blur iterations", &fluidSettings_.blurIterations, 1, 6);
    ImGui::Separator();
    ImGui::TextUnformatted("Water material");
    ImGui::SliderFloat("Refraction strength",
                       &fluidSettings_.refractionStrength, 0.0f, 0.30f,
                       "%.3f");
    ImGui::SliderFloat("Absorption scale", &fluidSettings_.absorptionScale,
                       0.0f, 12.0f, "%.2f");
    ImGui::ColorEdit3("Extinction colour", &fluidSettings_.extinctionRed,
                      ImGuiColorEditFlags_Float);
    ImGui::SliderFloat("Base reflectance", &fluidSettings_.baseReflectance,
                       0.0f, 0.15f, "%.3f");
    ImGui::Separator();
    ImGui::TextUnformatted("Sun / shadows");
    ImGui::SliderFloat("Sun azimuth", &fluidSettings_.sunAzimuthDegrees,
                       -180.0f, 180.0f, "%.0f deg");
    ImGui::SliderFloat("Sun elevation", &fluidSettings_.sunElevationDegrees,
                       10.0f, 85.0f, "%.0f deg");
    ImGui::SliderFloat("Shadow ambient light",
                       &fluidSettings_.shadowAmbientLight, 0.0f, 0.75f,
                       "%.2f");
    ImGui::SliderInt("Shadow update interval",
                     &fluidSettings_.shadowUpdateInterval, 1, 4,
                     "%d frames");
    ImGui::TextUnformatted("Click and drag outside this panel to orbit.");
    ImGui::End();
  }

  int simulationSubsteps() const {
    return std::max(fluidSettings_.simulationSubsteps, 1);
  }

  void setParams(const SphParams &params) {
    if (params.grid[3] > startCapacity_)
      throw std::runtime_error("SPH grid exceeds start-index buffer capacity");
    gridCellCount_ = params.grid[3];
    baseDeltaT_ = params.deltaT;
    defaultSphParams_ = params;
    std::memcpy(paramsMapped_, &params, sizeof(SphParams));
  }

  void setInteraction(const float3 &worldPoint, float radius, float strength) {
    pendingInteraction_ = {};
    pendingInteraction_.point[0] = worldPoint.x;
    pendingInteraction_.point[1] = worldPoint.y;
    pendingInteraction_.point[2] = worldPoint.z;
    pendingInteraction_.radius = radius;
    pendingInteraction_.strength = strength;
  }

  void uploadInitialState(const std::vector<float4> &positions,
                          const std::vector<float4> &velocities) {
    uint32_t n = std::min<uint32_t>(numParticles_, (uint32_t)positions.size());
    VkDeviceSize size = n * sizeof(float4);
    uploadViaStaging(positionsBuf_, positions.data(), size);
    uploadViaStaging(velocitiesBuf_, velocities.data(), size);
  }

  // Advance the simulation `substeps` times on the GPU, then draw one frame.
  void drawFrame(const float4x4 &view, const float4x4 &proj,
                 const float3 &camRight, const float3 &camUp,
                 const float3 &camForward, float radius, int substeps) {
    vkWaitForFences(device_, 1, &inFlightFences_[currentFrame_], VK_TRUE,
                    UINT64_MAX);

    if (std::abs(fluidSettings_.renderScale - activeRenderScale_) > 0.001f) {
      // This only stalls when the user changes the quality control.
      vkDeviceWaitIdle(device_);
      destroyFluidSizedTargets();
      createFluidTargets();
      updateFluidDescriptors();
    }

    auto *liveParams = reinterpret_cast<SphParams *>(paramsMapped_);
    liveParams->deltaT = baseDeltaT_ * fluidSettings_.simulationRate /
                         float(std::max(substeps, 1));
    std::memcpy(interactionMapped_, &pendingInteraction_,
                sizeof(InteractionParams));

    uint32_t imageIndex = 0;
    VkResult acquire = vkAcquireNextImageKHR(
        device_, swapchain_, UINT64_MAX,
        imageAvailableSemaphores_[currentFrame_], VK_NULL_HANDLE, &imageIndex);
    if (acquire == VK_ERROR_OUT_OF_DATE_KHR) {
      if (imguiInitialized_)
        ImGui::EndFrame();
      recreateSwapchain();
      return;
    } else if (acquire != VK_SUCCESS && acquire != VK_SUBOPTIMAL_KHR) {
      vkCheck(acquire, "vkAcquireNextImageKHR");
    }

    vkResetFences(device_, 1, &inFlightFences_[currentFrame_]);

    DepthPush dp{};
    dp.view = view;
    dp.proj = proj;
    dp.camRight = float4(camRight, 0.0f);
    dp.camUp = float4(camUp, 0.0f);
    dp.params = float4(radius, 0.0f, 0.0f, 0.0f);

    WaterPush wp{};
    wp.camRight = float4(camRight, 0.0f);
    wp.camUp = float4(camUp, 0.0f);
    wp.camForward = float4(camForward, 0.0f);
    wp.material = float4(fluidSettings_.refractionStrength,
                         fluidSettings_.absorptionScale,
                         fluidSettings_.baseReflectance, 0.0f);
    wp.extinction = float4(fluidSettings_.extinctionRed,
                           fluidSettings_.extinctionGreen,
                           fluidSettings_.extinctionBlue, 0.0f);

    if (imguiInitialized_)
      ImGui::Render();

    VkCommandBuffer cmd = commandBuffers_[currentFrame_];
    vkResetCommandBuffer(cmd, 0);
    recordCommandBuffer(cmd, imageIndex, dp, wp, substeps);
    ++renderedFrameCount_;

    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    VkSemaphore waitSemaphores[] = {imageAvailableSemaphores_[currentFrame_]};
    VkPipelineStageFlags waitStages[] = {
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = waitSemaphores;
    submit.pWaitDstStageMask = waitStages;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    VkSemaphore signalSemaphores[] = {renderFinishedSemaphores_[imageIndex]};
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

    if (imguiInitialized_) {
      ImGui_ImplVulkan_Shutdown();
      ImGui_ImplGlfw_Shutdown();
      ImGui::DestroyContext();
      imguiInitialized_ = false;
    }
    if (imguiDescriptorPool_) {
      vkDestroyDescriptorPool(device_, imguiDescriptorPool_, nullptr);
      imguiDescriptorPool_ = VK_NULL_HANDLE;
    }

    cleanupSwapchain();

    for (VkPipeline p : computePipelines_)
      if (p)
        vkDestroyPipeline(device_, p, nullptr);
    if (computePipelineLayout_)
      vkDestroyPipelineLayout(device_, computePipelineLayout_, nullptr);
    if (pipeline_)
      vkDestroyPipeline(device_, pipeline_, nullptr);
    if (pipelineLayout_)
      vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);

    // fluid rendering
    if (compositePipeline_)
      vkDestroyPipeline(device_, compositePipeline_, nullptr);
    if (backgroundPipeline_)
      vkDestroyPipeline(device_, backgroundPipeline_, nullptr);
    if (sceneMeshPipeline_)
      vkDestroyPipeline(device_, sceneMeshPipeline_, nullptr);
    if (sceneMeshPipelineLayout_)
      vkDestroyPipelineLayout(device_, sceneMeshPipelineLayout_, nullptr);
    if (compositePipelineLayout_)
      vkDestroyPipelineLayout(device_, compositePipelineLayout_, nullptr);
    if (blurPipeline_)
      vkDestroyPipeline(device_, blurPipeline_, nullptr);
    if (thicknessBlurPipeline_)
      vkDestroyPipeline(device_, thicknessBlurPipeline_, nullptr);
    if (thicknessPipeline_)
      vkDestroyPipeline(device_, thicknessPipeline_, nullptr);
    if (blurPipelineLayout_)
      vkDestroyPipelineLayout(device_, blurPipelineLayout_, nullptr);
    if (blurPass_)
      vkDestroyRenderPass(device_, blurPass_, nullptr);
    if (compositePool_)
      vkDestroyDescriptorPool(device_, compositePool_, nullptr);
    if (compositeSetLayout_)
      vkDestroyDescriptorSetLayout(device_, compositeSetLayout_, nullptr);
    if (depthPipeline_)
      vkDestroyPipeline(device_, depthPipeline_, nullptr);
    if (depthPipelineLayout_)
      vkDestroyPipelineLayout(device_, depthPipelineLayout_, nullptr);
    if (fluidSampler_)
      vkDestroySampler(device_, fluidSampler_, nullptr);
    if (fluidDepthPass_)
      vkDestroyRenderPass(device_, fluidDepthPass_, nullptr);
    if (sceneSampler_)
      vkDestroySampler(device_, sceneSampler_, nullptr);
    if (envSampler_)
      vkDestroySampler(device_, envSampler_, nullptr);
    if (environmentView_)
      vkDestroyImageView(device_, environmentView_, nullptr);
    if (environmentImage_)
      vkDestroyImage(device_, environmentImage_, nullptr);
    if (environmentMem_)
      vkFreeMemory(device_, environmentMem_, nullptr);
    if (scenePass_)
      vkDestroyRenderPass(device_, scenePass_, nullptr);

    if (renderPass_)
      vkDestroyRenderPass(device_, renderPass_, nullptr);

    if (descriptorPool_)
      vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
    if (descriptorSetLayout_)
      vkDestroyDescriptorSetLayout(device_, descriptorSetLayout_, nullptr);

    destroyBuffer(positionsBuf_, positionsMem_);
    destroyBuffer(velocitiesBuf_, velocitiesMem_);
    destroyBuffer(predictedBuf_, predictedMem_);
    destroyBuffer(densitiesBuf_, densitiesMem_);
    destroyBuffer(keysBuf_, keysMem_);
    destroyBuffer(indicesBuf_, indicesMem_);
    destroyBuffer(startBuf_, startMem_);
    destroyBuffer(endBuf_, endMem_);
    destroyBuffer(sortedPosBuf_, sortedPosMem_);
    destroyBuffer(sortedVelBuf_, sortedVelMem_);
    destroyBuffer(sortedPredBuf_, sortedPredMem_);
    destroyBuffer(sortStorageBuf_, sortStorageMem_);
    if (sorter_) {
      vrdxDestroySorter(sorter_);
      sorter_ = VK_NULL_HANDLE;
    }
    destroyBuffer(paramsBuf_, paramsMem_);
    destroyBuffer(interactionBuf_, interactionMem_);
    if (sceneLightingMapped_)
      vkUnmapMemory(device_, sceneLightingMem_);
    destroyBuffer(sceneLightingBuf_, sceneLightingMem_);

    if (quadBuffer_) {
      vkDestroyBuffer(device_, quadBuffer_, nullptr);
      vkFreeMemory(device_, quadMemory_, nullptr);
    }
    destroyBuffer(sceneMeshBuffer_, sceneMeshMemory_);

    for (auto sem : renderFinishedSemaphores_)
      if (sem)
        vkDestroySemaphore(device_, sem, nullptr);
    renderFinishedSemaphores_.clear();
    for (int i = 0; i < kMaxFramesInFlight; i++) {
      if (imageAvailableSemaphores_[i])
        vkDestroySemaphore(device_, imageAvailableSemaphores_[i], nullptr);
      if (inFlightFences_[i])
        vkDestroyFence(device_, inFlightFences_[i], nullptr);
    }
    if (commandPool_)
      vkDestroyCommandPool(device_, commandPool_, nullptr);
    vkDestroyDevice(device_, nullptr);
    if (surface_)
      vkDestroySurfaceKHR(instance_, surface_, nullptr);
    if (instance_)
      vkDestroyInstance(instance_, nullptr);
    if (window_)
      glfwDestroyWindow(window_);
    glfwTerminate();
    device_ = VK_NULL_HANDLE;
  }

  void destroyBuffer(VkBuffer &buf, VkDeviceMemory &mem) {
    if (buf) {
      vkDestroyBuffer(device_, buf, nullptr);
      buf = VK_NULL_HANDLE;
    }
    if (mem) {
      vkFreeMemory(device_, mem, nullptr);
      mem = VK_NULL_HANDLE;
    }
  }

  void uploadViaStaging(VkBuffer dst, const void *src, VkDeviceSize size) {
    if (size == 0)
      return;
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory stagingMem = VK_NULL_HANDLE;
    createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 staging, stagingMem);
    void *mapped = nullptr;
    vkMapMemory(device_, stagingMem, 0, size, 0, &mapped);
    std::memcpy(mapped, src, (size_t)size);
    vkUnmapMemory(device_, stagingMem);

    VkCommandBufferAllocateInfo ai{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = commandPool_;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device_, &ai, &cmd);

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    VkBufferCopy region{0, 0, size};
    vkCmdCopyBuffer(cmd, staging, dst, 1, &region);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    vkQueueSubmit(graphicsQueue_, 1, &submit, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue_);

    vkFreeCommandBuffers(device_, commandPool_, 1, &cmd);
    destroyBuffer(staging, stagingMem);
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

  VkFormat fluidDepthFormat_ = VK_FORMAT_R32_SFLOAT;
  VkExtent2D fluidExtent_{};
  float activeRenderScale_ = 0.0f;
  uint64_t renderedFrameCount_ = 0;
  bool waterShadowValid_ = false;
  float shadowSunAzimuth_ = 1.0e9f;
  float shadowSunElevation_ = 1.0e9f;
  VkFormat sceneColorFormat_ = VK_FORMAT_UNDEFINED;
  VkFormat sceneDepthFormat_ = VK_FORMAT_R32_SFLOAT;
  VkImage fluidDepthImage_ = VK_NULL_HANDLE;
  VkDeviceMemory fluidDepthMem_ = VK_NULL_HANDLE;
  VkImageView fluidDepthView_ = VK_NULL_HANDLE;
  VkImage fluidZImage_ = VK_NULL_HANDLE;
  VkDeviceMemory fluidZMem_ = VK_NULL_HANDLE;
  VkImageView fluidZView_ = VK_NULL_HANDLE; // offscreen depth buffer
  VkImage sceneColorImage_ = VK_NULL_HANDLE;
  VkDeviceMemory sceneColorMem_ = VK_NULL_HANDLE;
  VkImageView sceneColorView_ = VK_NULL_HANDLE;
  VkImage sceneDepthImage_ = VK_NULL_HANDLE;
  VkDeviceMemory sceneDepthMem_ = VK_NULL_HANDLE;
  VkImageView sceneDepthView_ = VK_NULL_HANDLE;
  VkImage sceneZImage_ = VK_NULL_HANDLE;
  VkDeviceMemory sceneZMem_ = VK_NULL_HANDLE;
  VkImageView sceneZView_ = VK_NULL_HANDLE;
  VkRenderPass scenePass_ = VK_NULL_HANDLE;
  VkFramebuffer sceneFB_ = VK_NULL_HANDLE;
  VkRenderPass fluidDepthPass_ = VK_NULL_HANDLE;
  VkFramebuffer fluidDepthFB_ = VK_NULL_HANDLE;
  VkSampler fluidSampler_ = VK_NULL_HANDLE;
  VkSampler sceneSampler_ = VK_NULL_HANDLE;
  VkImage environmentImage_ = VK_NULL_HANDLE;
  VkDeviceMemory environmentMem_ = VK_NULL_HANDLE;
  VkImageView environmentView_ = VK_NULL_HANDLE;
  VkSampler envSampler_ = VK_NULL_HANDLE;
  VkPipelineLayout depthPipelineLayout_ = VK_NULL_HANDLE;
  VkPipeline depthPipeline_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout compositeSetLayout_ = VK_NULL_HANDLE;
  VkDescriptorPool compositePool_ = VK_NULL_HANDLE;
  VkDescriptorSet compositeSet_ = VK_NULL_HANDLE;
  VkPipelineLayout compositePipelineLayout_ = VK_NULL_HANDLE;
  VkPipeline backgroundPipeline_ = VK_NULL_HANDLE;
  VkPipelineLayout sceneMeshPipelineLayout_ = VK_NULL_HANDLE;
  VkPipeline sceneMeshPipeline_ = VK_NULL_HANDLE;
  VkPipeline compositePipeline_ = VK_NULL_HANDLE;

  FluidRenderSettings fluidSettings_{};
  VkDescriptorPool imguiDescriptorPool_ = VK_NULL_HANDLE;
  bool imguiInitialized_ = false;

  // bilateral blur
  VkImage blurTempImage_ = VK_NULL_HANDLE;
  VkDeviceMemory blurTempMem_ = VK_NULL_HANDLE;
  VkImageView blurTempView_ = VK_NULL_HANDLE;
  VkImage blurSmoothImage_ = VK_NULL_HANDLE;
  VkDeviceMemory blurSmoothMem_ = VK_NULL_HANDLE;
  VkImageView blurSmoothView_ = VK_NULL_HANDLE;
  VkImage fluidThicknessImage_ = VK_NULL_HANDLE;
  VkDeviceMemory fluidThicknessMem_ = VK_NULL_HANDLE;
  VkImageView fluidThicknessView_ = VK_NULL_HANDLE;
  VkImage thicknessBlurTempImage_ = VK_NULL_HANDLE;
  VkDeviceMemory thicknessBlurTempMem_ = VK_NULL_HANDLE;
  VkImageView thicknessBlurTempView_ = VK_NULL_HANDLE;
  VkImage waterShadowImage_ = VK_NULL_HANDLE;
  VkDeviceMemory waterShadowMem_ = VK_NULL_HANDLE;
  VkImageView waterShadowView_ = VK_NULL_HANDLE;
  VkImage waterShadowDepthImage_ = VK_NULL_HANDLE;
  VkDeviceMemory waterShadowDepthMem_ = VK_NULL_HANDLE;
  VkImageView waterShadowDepthView_ = VK_NULL_HANDLE;
  VkImage waterShadowZImage_ = VK_NULL_HANDLE;
  VkDeviceMemory waterShadowZMem_ = VK_NULL_HANDLE;
  VkImageView waterShadowZView_ = VK_NULL_HANDLE;
  VkRenderPass blurPass_ = VK_NULL_HANDLE;
  VkFramebuffer blurTempFB_ = VK_NULL_HANDLE;
  VkFramebuffer blurSmoothFB_ = VK_NULL_HANDLE;
  VkFramebuffer fluidThicknessFB_ = VK_NULL_HANDLE;
  VkFramebuffer thicknessBlurTempFB_ = VK_NULL_HANDLE;
  VkFramebuffer waterShadowFB_ = VK_NULL_HANDLE;
  VkFramebuffer waterShadowDepthFB_ = VK_NULL_HANDLE;
  VkPipelineLayout blurPipelineLayout_ = VK_NULL_HANDLE;
  VkPipeline blurPipeline_ = VK_NULL_HANDLE;
  VkPipeline thicknessBlurPipeline_ = VK_NULL_HANDLE;
  VkPipeline thicknessPipeline_ = VK_NULL_HANDLE;
  VkDescriptorSet blurSetH_ = VK_NULL_HANDLE;
  VkDescriptorSet blurSetV_ = VK_NULL_HANDLE;
  VkDescriptorSet blurSetSmooth_ = VK_NULL_HANDLE;
  VkDescriptorSet thicknessBlurSet_ = VK_NULL_HANDLE;
  VkDescriptorSet thicknessBlurSetV_ = VK_NULL_HANDLE;

  VkBuffer quadBuffer_ = VK_NULL_HANDLE;
  VkDeviceMemory quadMemory_ = VK_NULL_HANDLE;
  VkBuffer sceneMeshBuffer_ = VK_NULL_HANDLE;
  VkDeviceMemory sceneMeshMemory_ = VK_NULL_HANDLE;
  uint32_t sceneMeshVertexCount_ = 0;

  uint32_t numParticles_ = 0;
  uint32_t gridCellCount_ = 0;
  uint32_t startCapacity_ = 0;
  float baseDeltaT_ = 0.002f;

  VkBuffer positionsBuf_ = VK_NULL_HANDLE;
  VkDeviceMemory positionsMem_ = VK_NULL_HANDLE;
  VkBuffer velocitiesBuf_ = VK_NULL_HANDLE;
  VkDeviceMemory velocitiesMem_ = VK_NULL_HANDLE;
  VkBuffer predictedBuf_ = VK_NULL_HANDLE;
  VkDeviceMemory predictedMem_ = VK_NULL_HANDLE;
  VkBuffer densitiesBuf_ = VK_NULL_HANDLE;
  VkDeviceMemory densitiesMem_ = VK_NULL_HANDLE;
  VkBuffer paramsBuf_ = VK_NULL_HANDLE;
  VkDeviceMemory paramsMem_ = VK_NULL_HANDLE;
  void *paramsMapped_ = nullptr;
  SphParams defaultSphParams_{};
  VkBuffer interactionBuf_ = VK_NULL_HANDLE;
  VkDeviceMemory interactionMem_ = VK_NULL_HANDLE;
  void *interactionMapped_ = nullptr;
  InteractionParams pendingInteraction_{};
  VkBuffer sceneLightingBuf_ = VK_NULL_HANDLE;
  VkDeviceMemory sceneLightingMem_ = VK_NULL_HANDLE;
  void *sceneLightingMapped_ = nullptr;
  // spatial-hash buffers
  VkBuffer keysBuf_ = VK_NULL_HANDLE;
  VkDeviceMemory keysMem_ = VK_NULL_HANDLE;
  VkBuffer indicesBuf_ = VK_NULL_HANDLE;
  VkDeviceMemory indicesMem_ = VK_NULL_HANDLE;
  VkBuffer startBuf_ = VK_NULL_HANDLE;
  VkDeviceMemory startMem_ = VK_NULL_HANDLE;
  VkBuffer endBuf_ = VK_NULL_HANDLE;
  VkDeviceMemory endMem_ = VK_NULL_HANDLE;

  // sorted-order scratch buffers
  VkBuffer sortedPosBuf_ = VK_NULL_HANDLE;
  VkDeviceMemory sortedPosMem_ = VK_NULL_HANDLE;
  VkBuffer sortedVelBuf_ = VK_NULL_HANDLE;
  VkDeviceMemory sortedVelMem_ = VK_NULL_HANDLE;
  VkBuffer sortedPredBuf_ = VK_NULL_HANDLE;
  VkDeviceMemory sortedPredMem_ = VK_NULL_HANDLE;

  VrdxSorter sorter_ = VK_NULL_HANDLE;
  VkBuffer sortStorageBuf_ = VK_NULL_HANDLE;
  VkDeviceMemory sortStorageMem_ = VK_NULL_HANDLE;

  VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
  VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
  VkDescriptorSet descriptorSet_ = VK_NULL_HANDLE;

  VkPipelineLayout computePipelineLayout_ = VK_NULL_HANDLE;

  // 0:external+keys 1:reorder 2:startReset 3:startIndices 4:density
  // 5:pressure 6:viscosity+integrate
  std::array<VkPipeline, 7> computePipelines_{};

  VkCommandPool commandPool_ = VK_NULL_HANDLE;
  std::array<VkCommandBuffer, kMaxFramesInFlight> commandBuffers_{};
  std::array<VkSemaphore, kMaxFramesInFlight> imageAvailableSemaphores_{};
  std::vector<VkSemaphore> renderFinishedSemaphores_; // one per swapchain image
  std::array<VkFence, kMaxFramesInFlight> inFlightFences_{};
  uint32_t currentFrame_ = 0;

  static void framebufferResizeCallback(GLFWwindow *win, int, int) {
    auto self =
        reinterpret_cast<VulkanRenderer *>(glfwGetWindowUserPointer(win));
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
    app.apiVersion = VK_API_VERSION_1_4;

    uint32_t glfwExtCount = 0;
    const char **glfwExts = glfwGetRequiredInstanceExtensions(&glfwExtCount);
    std::vector<const char *> extensions(glfwExts, glfwExts + glfwExtCount);

// MoltenVK / portability extensions (macOS only)
#if VKR_OS_MACOS
    extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    extensions.push_back(
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
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

    std::vector<const char *> deviceExts = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME};
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

    VkPhysicalDeviceVulkan13Features features13{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    features13.synchronization2 = VK_TRUE;
    VkPhysicalDeviceVulkan14Features features14{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES};
    features14.pNext = &features13;
    features14.pushDescriptor = VK_TRUE;

    VkDeviceCreateInfo ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    ci.pNext = &features14;
    ci.queueCreateInfoCount = (uint32_t)queueInfos.size();
    ci.pQueueCreateInfos = queueInfos.data();
    ci.pEnabledFeatures = nullptr; // features supplied via pNext
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
      vkCheck(
          vkCreateImageView(device_, &ci, nullptr, &swapchainImageViews_[i]),
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

    VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
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

  void createSceneTargets() {
    sceneColorFormat_ = swapchainFormat_;
    createImage2D(sceneColorFormat_,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                      VK_IMAGE_USAGE_SAMPLED_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT, sceneColorImage_, sceneColorMem_,
                  sceneColorView_, swapchainExtent_.width,
                  swapchainExtent_.height);
    createImage2D(sceneDepthFormat_,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
              VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT, sceneDepthImage_, sceneDepthMem_,
            sceneDepthView_, swapchainExtent_.width,
            swapchainExtent_.height);
    createImage2D(depthFormat_, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                  VK_IMAGE_ASPECT_DEPTH_BIT, sceneZImage_, sceneZMem_,
                  sceneZView_, swapchainExtent_.width, swapchainExtent_.height);

    if (!sceneSampler_) {
      VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
      si.magFilter = VK_FILTER_LINEAR;
      si.minFilter = VK_FILTER_LINEAR;
      si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
      si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      vkCheck(vkCreateSampler(device_, &si, nullptr, &sceneSampler_),
              "vkCreateSampler(scene)");
    }

    if (!scenePass_) {
      VkAttachmentDescription color{};
      color.format = sceneColorFormat_;
      color.samples = VK_SAMPLE_COUNT_1_BIT;
      color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
      color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
      color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
      color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
      color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      color.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      VkAttachmentDescription depth{};
      depth.format = sceneDepthFormat_;
      depth.samples = VK_SAMPLE_COUNT_1_BIT;
      depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
      depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
      depth.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
      depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
      depth.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      depth.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      VkAttachmentDescription sceneZ{};
      sceneZ.format = depthFormat_;
      sceneZ.samples = VK_SAMPLE_COUNT_1_BIT;
      sceneZ.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
      sceneZ.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
      sceneZ.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
      sceneZ.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
      sceneZ.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      sceneZ.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
      VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
      VkAttachmentReference depthRef{1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
      VkAttachmentReference sceneZRef{2, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
      std::array<VkAttachmentReference, 2> colorRefs = {colorRef, depthRef};
      VkSubpassDescription subpass{};
      subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
      subpass.colorAttachmentCount = 2;
      subpass.pColorAttachments = colorRefs.data();
      subpass.pDepthStencilAttachment = &sceneZRef;
      VkSubpassDependency dep{};
      dep.srcSubpass = 0;
      dep.dstSubpass = VK_SUBPASS_EXTERNAL;
      dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
      dep.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
      dep.dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
      dep.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      VkRenderPassCreateInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
      std::array<VkAttachmentDescription, 3> atts = {color, depth, sceneZ};
      rp.attachmentCount = (uint32_t)atts.size();
      rp.pAttachments = atts.data();
      rp.subpassCount = 1;
      rp.pSubpasses = &subpass;
      rp.dependencyCount = 1;
      rp.pDependencies = &dep;
      vkCheck(vkCreateRenderPass(device_, &rp, nullptr, &scenePass_),
              "vkCreateRenderPass(scene)");
    }

    VkFramebufferCreateInfo fb{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fb.renderPass = scenePass_;
    std::array<VkImageView, 3> fbAtt = {sceneColorView_, sceneDepthView_,
                                        sceneZView_};
    fb.attachmentCount = (uint32_t)fbAtt.size();
    fb.pAttachments = fbAtt.data();
    fb.width = swapchainExtent_.width;
    fb.height = swapchainExtent_.height;
    fb.layers = 1;
    vkCheck(vkCreateFramebuffer(device_, &fb, nullptr, &sceneFB_),
            "vkCreateFramebuffer(scene)");
  }

  void createEnvironmentTexture() {
    if (environmentImage_)
      return;

    std::vector<std::string> candidates = {
        "media/uffizi_probe.hdr", "../media/uffizi_probe.hdr",
        "../../media/uffizi_probe.hdr"};
    int width = 0;
    int height = 0;
    int comp = 0;
    float *pixels = nullptr;
    for (const auto &path : candidates) {
      pixels = stbi_loadf(path.c_str(), &width, &height, &comp, 4);
      if (pixels)
        break;
    }

    std::vector<float> fallback;
    const float *src = pixels;
    if (!src) {
      width = 1024;
      height = 512;
      fallback.resize((size_t)width * (size_t)height * 4u);
      for (int y = 0; y < height; ++y) {
        float v = float(y) / float(height - 1);
        for (int x = 0; x < width; ++x) {
          float u = float(x) / float(width - 1);
          float horizon = 1.0f - std::abs(v * 2.0f - 1.0f);
          horizon = std::clamp(horizon, 0.0f, 1.0f);
          float3 top = float3(0.18f, 0.34f, 0.62f);
          float3 bottom = float3(0.02f, 0.05f, 0.10f);
          float3 c = bottom * (1.0f - horizon) + top * horizon;
          float sun = std::pow(std::max(0.0f, 1.0f - std::abs(u - 0.35f) * 12.0f), 4.0f) *
                      std::pow(std::max(0.0f, 1.0f - std::abs(v - 0.22f) * 20.0f), 4.0f);
          c += float3(1.0f, 0.92f, 0.78f) * sun * 2.0f;
          size_t idx = (size_t(y) * (size_t)width + size_t(x)) * 4u;
          fallback[idx + 0] = c.x;
          fallback[idx + 1] = c.y;
          fallback[idx + 2] = c.z;
          fallback[idx + 3] = 1.0f;
        }
      }
      src = fallback.data();
    }

    VkDeviceSize imageSize = VkDeviceSize(width) * VkDeviceSize(height) * 4u * sizeof(float);
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory stagingMem = VK_NULL_HANDLE;
    createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 staging, stagingMem);
    void *mapped = nullptr;
    vkMapMemory(device_, stagingMem, 0, imageSize, 0, &mapped);
    std::memcpy(mapped, src, (size_t)imageSize);
    vkUnmapMemory(device_, stagingMem);

    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.extent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ci.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCheck(vkCreateImage(device_, &ci, nullptr, &environmentImage_),
            "vkCreateImage(environment)");

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(device_, environmentImage_, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex =
        findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkCheck(vkAllocateMemory(device_, &ai, nullptr, &environmentMem_),
            "vkAllocateMemory(environment)");
    vkBindImageMemory(device_, environmentImage_, environmentMem_, 0);

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pci.queueFamilyIndex = graphicsFamily_;
    VkCommandPool pool = VK_NULL_HANDLE;
    vkCheck(vkCreateCommandPool(device_, &pci, nullptr, &pool),
            "vkCreateCommandPool(environment)");

    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkCheck(vkAllocateCommandBuffers(device_, &cai, &cmd),
            "vkAllocateCommandBuffers(environment)");

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkCheck(vkBeginCommandBuffer(cmd, &bi), "vkBeginCommandBuffer(environment)");

    transitionImageLayout(cmd, environmentImage_, VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_ASPECT_COLOR_BIT);
    copyBufferToImage(cmd, staging, environmentImage_,
                      static_cast<uint32_t>(width),
                      static_cast<uint32_t>(height));
    transitionImageLayout(cmd, environmentImage_,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                          VK_IMAGE_ASPECT_COLOR_BIT);

    vkCheck(vkEndCommandBuffer(cmd), "vkEndCommandBuffer(environment)");
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    vkCheck(vkQueueSubmit(graphicsQueue_, 1, &submit, VK_NULL_HANDLE),
            "vkQueueSubmit(environment)");
    vkCheck(vkQueueWaitIdle(graphicsQueue_), "vkQueueWaitIdle(environment)");

    vkFreeCommandBuffers(device_, pool, 1, &cmd);
    vkDestroyCommandPool(device_, pool, nullptr);
    destroyBuffer(staging, stagingMem);

    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = environmentImage_;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vi.subresourceRange.levelCount = 1;
    vi.subresourceRange.layerCount = 1;
    vkCheck(vkCreateImageView(device_, &vi, nullptr, &environmentView_),
            "vkCreateImageView(environment)");

    if (!envSampler_) {
      VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
      si.magFilter = VK_FILTER_LINEAR;
      si.minFilter = VK_FILTER_LINEAR;
      si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
      si.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
      si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      vkCheck(vkCreateSampler(device_, &si, nullptr, &envSampler_),
              "vkCreateSampler(environment)");
    }

    if (pixels)
      stbi_image_free(pixels);
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

    std::array<VkVertexInputBindingDescription, 1> bindings{};
    bindings[0].binding = 0;
    bindings[0].stride = sizeof(float) * 2;
    bindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    std::array<VkVertexInputAttributeDescription, 1> attrs{};
    attrs[0] = {0, 0, VK_FORMAT_R32G32_SFLOAT, 0};

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
    pl.setLayoutCount = 1;
    pl.pSetLayouts = &descriptorSetLayout_;
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
    const float quad[] = {
        -1.f, -1.f, 1.f, -1.f, -1.f, 1.f, 1.f, 1.f,
    };
    VkDeviceSize size = sizeof(quad);
    createBuffer(size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 quadBuffer_, quadMemory_);
    void *data = nullptr;
    vkMapMemory(device_, quadMemory_, 0, size, 0, &data);
    std::memcpy(data, quad, (size_t)size);
    vkUnmapMemory(device_, quadMemory_);
  }

  static std::vector<float> loadObj(const std::string &path, float scale) {
    std::ifstream file(path);
    if (!file)
      throw std::runtime_error("Failed to open OBJ: " + path);
    std::vector<float3> positions, normals;
    std::vector<float> out;
    std::string line;
    auto index = [](int i, size_t n) { return i > 0 ? i - 1 : int(n) + i; };
    while (std::getline(file, line)) {
      std::istringstream s(line);
      std::string tag;
      s >> tag;
      if (tag == "v") {
        float3 p; s >> p.x >> p.y >> p.z; positions.push_back(p);
      } else if (tag == "vn") {
        float3 n; s >> n.x >> n.y >> n.z; normals.push_back(normalize(n));
      } else if (tag == "f") {
        struct Ref { int p = 0, n = 0; };
        std::vector<Ref> face;
        std::string token;
        while (s >> token) {
          Ref r{}; size_t a = token.find('/'); size_t b = token.rfind('/');
          r.p = std::stoi(token.substr(0, a));
          if (a != std::string::npos && b != a && b + 1 < token.size())
            r.n = std::stoi(token.substr(b + 1));
          face.push_back(r);
        }
        for (size_t i = 1; i + 1 < face.size(); ++i) {
          Ref tri[3] = {face[0], face[i], face[i + 1]};
          float3 p[3];
          for (int k = 0; k < 3; ++k) p[k] = positions.at(index(tri[k].p, positions.size())) * scale;
          float3 fallback = normalize(cross(p[1] - p[0], p[2] - p[0]));
          for (int k = 0; k < 3; ++k) {
            float3 n = tri[k].n ? normals.at(index(tri[k].n, normals.size())) : fallback;
            out.insert(out.end(), {p[k].x, p[k].y, p[k].z, n.x, n.y, n.z});
          }
        }
      }
    }
    return out;
  }

  void createComputeBuffers() {
    const VkMemoryPropertyFlags devLocal = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    const VkBufferUsageFlags storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    const VkBufferUsageFlags storageDst =
        storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkDeviceSize vec4Size = sizeof(float) * 4 * numParticles_;
    VkDeviceSize fltSize = sizeof(float) * numParticles_;
    VkDeviceSize uintSize = sizeof(uint32_t) * numParticles_;
    startCapacity_ = numParticles_ * 2u;
    VkDeviceSize startSize = sizeof(uint32_t) * startCapacity_;

    createBuffer(vec4Size, storageDst, devLocal, positionsBuf_, positionsMem_);
    createBuffer(vec4Size, storageDst, devLocal, velocitiesBuf_,
                 velocitiesMem_);
    createBuffer(vec4Size, storage, devLocal, predictedBuf_, predictedMem_);
    createBuffer(fltSize, storage, devLocal, densitiesBuf_, densitiesMem_);
    createBuffer(uintSize, storage, devLocal, keysBuf_, keysMem_);
    createBuffer(uintSize, storage, devLocal, indicesBuf_, indicesMem_);
    createBuffer(startSize, storage, devLocal, startBuf_, startMem_);
    createBuffer(startSize, storage, devLocal, endBuf_, endMem_);

    createBuffer(vec4Size, storage, devLocal, sortedPosBuf_, sortedPosMem_);
    createBuffer(vec4Size, storage, devLocal, sortedVelBuf_, sortedVelMem_);
    createBuffer(vec4Size, storage, devLocal, sortedPredBuf_, sortedPredMem_);

    createBuffer(sizeof(SphParams), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 paramsBuf_, paramsMem_);
    vkMapMemory(device_, paramsMem_, 0, sizeof(SphParams), 0, &paramsMapped_);

    createBuffer(sizeof(InteractionParams), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 interactionBuf_, interactionMem_);
    vkMapMemory(device_, interactionMem_, 0, sizeof(InteractionParams), 0,
                &interactionMapped_);
  }

  void createSceneLightingBuffer() {
    createBuffer(sizeof(SceneLightingParams), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 sceneLightingBuf_, sceneLightingMem_);
    vkCheck(vkMapMemory(device_, sceneLightingMem_, 0,
                        sizeof(SceneLightingParams), 0,
                        &sceneLightingMapped_),
            "vkMapMemory(scene lighting)");
  }

  void createSorter() {
    VrdxSorterCreateInfo ci{};
    ci.physicalDevice = physicalDevice_;
    ci.device = device_;
    ci.pipelineCache = VK_NULL_HANDLE;
    vkCheck(vrdxCreateSorter(&ci, &sorter_), "vrdxCreateSorter");

    VrdxSorterStorageRequirements req{};
    vrdxGetSorterStorageRequirements(sorter_, numParticles_,
                                     VRDX_SORT_MODE_KEY_VALUE, &req);
    createBuffer(req.size, req.usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 sortStorageBuf_, sortStorageMem_);
  }

  void createDescriptorSetLayout() {
    constexpr std::array<uint32_t, 13> bindings = {
        0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13};
    std::array<VkDescriptorSetLayoutBinding, bindings.size()> layoutBindings{};
    for (size_t i = 0; i < bindings.size(); ++i) {
      auto &binding = layoutBindings[i];
      binding.binding = bindings[i];
      binding.descriptorType = (bindings[i] == 5 || bindings[i] == 12)
                                   ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                   : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      binding.descriptorCount = 1;
      binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      if (bindings[i] == 0 || bindings[i] == 1)
        binding.stageFlags |= VK_SHADER_STAGE_VERTEX_BIT;
    }

    VkDescriptorSetLayoutCreateInfo ci{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    ci.bindingCount = (uint32_t)layoutBindings.size();
    ci.pBindings = layoutBindings.data();
    vkCheck(vkCreateDescriptorSetLayout(device_, &ci, nullptr,
                                        &descriptorSetLayout_),
            "vkCreateDescriptorSetLayout");
  }

  void createDescriptorPoolAndSet() {
    std::array<VkDescriptorPoolSize, 2> sizes{};
    sizes[0] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 11};
    sizes[1] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2};
    VkDescriptorPoolCreateInfo pi{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.maxSets = 1;
    pi.poolSizeCount = (uint32_t)sizes.size();
    pi.pPoolSizes = sizes.data();
    vkCheck(vkCreateDescriptorPool(device_, &pi, nullptr, &descriptorPool_),
            "vkCreateDescriptorPool");

    VkDescriptorSetAllocateInfo ai{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = descriptorPool_;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &descriptorSetLayout_;
    vkCheck(vkAllocateDescriptorSets(device_, &ai, &descriptorSet_),
            "vkAllocateDescriptorSets");

    constexpr std::array<uint32_t, 13> bindings = {
        0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13};
    std::array<VkBuffer, 13> buffers = {
        positionsBuf_, velocitiesBuf_, predictedBuf_,  densitiesBuf_,
        paramsBuf_,    keysBuf_,       indicesBuf_,    startBuf_,
        sortedPosBuf_, sortedVelBuf_,  sortedPredBuf_, interactionBuf_,
        endBuf_};
    std::array<VkDescriptorBufferInfo, 13> infos{};
    std::array<VkWriteDescriptorSet, 13> writes{};
    for (size_t i = 0; i < bindings.size(); ++i) {
      infos[i] = {buffers[i], 0, VK_WHOLE_SIZE};
      writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      writes[i].dstSet = descriptorSet_;
      writes[i].dstBinding = bindings[i];
      writes[i].descriptorCount = 1;
      writes[i].descriptorType = (bindings[i] == 5 || bindings[i] == 12)
                                     ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                     : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(device_, (uint32_t)writes.size(), writes.data(), 0,
                           nullptr);
  }

  VkPipeline createComputePipeline(const std::string &spvName) {
    auto code = readFile(std::string(SHADER_DIR) + "/" + spvName);
    VkShaderModule module = createShaderModule(code);
    VkPipelineShaderStageCreateInfo stage{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = module;
    stage.pName = "main";
    VkComputePipelineCreateInfo ci{
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = stage;
    ci.layout = computePipelineLayout_;
    VkPipeline pipeline;
    vkCheck(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &ci, nullptr,
                                     &pipeline),
            "vkCreateComputePipelines");
    vkDestroyShaderModule(device_, module, nullptr);
    return pipeline;
  }

  void createComputePipelines() {
    VkPipelineLayoutCreateInfo pl{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl.setLayoutCount = 1;
    pl.pSetLayouts = &descriptorSetLayout_;
    vkCheck(
        vkCreatePipelineLayout(device_, &pl, nullptr, &computePipelineLayout_),
        "vkCreatePipelineLayout(compute)");

    computePipelines_[0] = createComputePipeline("sph_external.comp.spv");
    computePipelines_[1] = createComputePipeline("sph_reorder.comp.spv");
    computePipelines_[2] = createComputePipeline("sph_start_reset.comp.spv");
    computePipelines_[3] = createComputePipeline("sph_start_indices.comp.spv");
    computePipelines_[4] = createComputePipeline("sph_density.comp.spv");
    computePipelines_[5] = createComputePipeline("sph_pressure.comp.spv");
    computePipelines_[6] =
        createComputePipeline("sph_viscosity_integrate.comp.spv");
  }

  void createImage2D(VkFormat format, VkImageUsageFlags usage,
                     VkImageAspectFlags aspect, VkImage &image,
                     VkDeviceMemory &mem, VkImageView &view) {
    createImage2D(format, usage, aspect, image, mem, view,
                  swapchainExtent_.width, swapchainExtent_.height);
  }

  void createImage2D(VkFormat format, VkImageUsageFlags usage,
                     VkImageAspectFlags aspect, VkImage &image,
                     VkDeviceMemory &mem, VkImageView &view, uint32_t width,
                     uint32_t height) {
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.extent = {width, height, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.format = format;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ci.usage = usage;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCheck(vkCreateImage(device_, &ci, nullptr, &image), "vkCreateImage");
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(device_, image, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex =
        findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkCheck(vkAllocateMemory(device_, &ai, nullptr, &mem), "vkAllocateMemory");
    vkBindImageMemory(device_, image, mem, 0);
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = format;
    vi.subresourceRange.aspectMask = aspect;
    vi.subresourceRange.levelCount = 1;
    vi.subresourceRange.layerCount = 1;
    vkCheck(vkCreateImageView(device_, &vi, nullptr, &view),
            "vkCreateImageView");
  }

  void transitionImageLayout(VkCommandBuffer cmd, VkImage image,
                             VkImageLayout oldLayout,
                             VkImageLayout newLayout,
                             VkImageAspectFlags aspect) {
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = aspect;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    VkPipelineStageFlags dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
        newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
      barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
               newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
      barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }

    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1,
                         &barrier);
  }

  void copyBufferToImage(VkCommandBuffer cmd, VkBuffer buffer, VkImage image,
                         uint32_t width, uint32_t height) {
    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {width, height, 1};
    vkCmdCopyBufferToImage(cmd, buffer, image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
  }

  void createImGuiDescriptorPool() {
    VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 32};
    VkDescriptorPoolCreateInfo ci{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    ci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    ci.maxSets = 32;
    ci.poolSizeCount = 1;
    ci.pPoolSizes = &size;
    vkCheck(
        vkCreateDescriptorPool(device_, &ci, nullptr, &imguiDescriptorPool_),
        "vkCreateDescriptorPool(imgui)");
  }

  void createFluidTargets() {
    activeRenderScale_ = std::clamp(fluidSettings_.renderScale, 0.5f, 1.0f);
    fluidExtent_.width = std::max(
        1u, static_cast<uint32_t>(std::lround(swapchainExtent_.width *
                                              activeRenderScale_)));
    fluidExtent_.height = std::max(
        1u, static_cast<uint32_t>(std::lround(swapchainExtent_.height *
                                              activeRenderScale_)));
    waterShadowValid_ = false;
    createImage2D(fluidDepthFormat_,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                      VK_IMAGE_USAGE_SAMPLED_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT, fluidDepthImage_, fluidDepthMem_,
                  fluidDepthView_, fluidExtent_.width, fluidExtent_.height);
    createImage2D(depthFormat_, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                  VK_IMAGE_ASPECT_DEPTH_BIT, fluidZImage_, fluidZMem_,
                  fluidZView_, fluidExtent_.width, fluidExtent_.height);

    VkAttachmentDescription color{};
    color.format = fluidDepthFormat_;
    color.samples = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkAttachmentDescription depth{};
    depth.format = depthFormat_;
    depth.samples = VK_SAMPLE_COUNT_1_BIT;
    depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depth.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depthRef{
        1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;
    subpass.pDepthStencilAttachment = &depthRef;
    // make the depth-write result visible to the composite pass's sampling
    VkSubpassDependency dep{};
    dep.srcSubpass = 0;
    dep.dstSubpass = VK_SUBPASS_EXTERNAL;
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dep.dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dep.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    std::array<VkAttachmentDescription, 2> atts = {color, depth};
    VkRenderPassCreateInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rp.attachmentCount = 2;
    rp.pAttachments = atts.data();
    rp.subpassCount = 1;
    rp.pSubpasses = &subpass;
    rp.dependencyCount = 1;
    rp.pDependencies = &dep;
    if (!fluidDepthPass_) // format-invariant; create once, reuse across resizes
      vkCheck(vkCreateRenderPass(device_, &rp, nullptr, &fluidDepthPass_),
              "vkCreateRenderPass(fluid)");

    std::array<VkImageView, 2> fbatt = {fluidDepthView_, fluidZView_};
    VkFramebufferCreateInfo fb{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fb.renderPass = fluidDepthPass_;
    fb.attachmentCount = 2;
    fb.pAttachments = fbatt.data();
    fb.width = fluidExtent_.width;
    fb.height = fluidExtent_.height;
    fb.layers = 1;
    vkCheck(vkCreateFramebuffer(device_, &fb, nullptr, &fluidDepthFB_),
            "vkCreateFramebuffer(fluid)");

    if (!fluidSampler_) {
      VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
      si.magFilter = VK_FILTER_NEAREST;
      si.minFilter = VK_FILTER_NEAREST;
      si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      vkCheck(vkCreateSampler(device_, &si, nullptr, &fluidSampler_),
              "vkCreateSampler");
    }

    // blur target
    createImage2D(
        fluidDepthFormat_,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT, blurTempImage_, blurTempMem_, blurTempView_,
        fluidExtent_.width, fluidExtent_.height);
    createImage2D(fluidDepthFormat_,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                      VK_IMAGE_USAGE_SAMPLED_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT, blurSmoothImage_, blurSmoothMem_,
                  blurSmoothView_, fluidExtent_.width, fluidExtent_.height);
    createImage2D(fluidDepthFormat_,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                      VK_IMAGE_USAGE_SAMPLED_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT, fluidThicknessImage_,
                  fluidThicknessMem_, fluidThicknessView_, fluidExtent_.width,
                  fluidExtent_.height);
    createImage2D(fluidDepthFormat_,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                      VK_IMAGE_USAGE_SAMPLED_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT, thicknessBlurTempImage_,
                  thicknessBlurTempMem_, thicknessBlurTempView_,
                  fluidExtent_.width, fluidExtent_.height);
    createImage2D(fluidDepthFormat_,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                      VK_IMAGE_USAGE_SAMPLED_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT, waterShadowImage_,
                  waterShadowMem_, waterShadowView_, kWaterShadowMapSize,
                  kWaterShadowMapSize);
    createImage2D(fluidDepthFormat_,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                      VK_IMAGE_USAGE_SAMPLED_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT, waterShadowDepthImage_,
                  waterShadowDepthMem_, waterShadowDepthView_,
                  kWaterShadowMapSize, kWaterShadowMapSize);
    createImage2D(depthFormat_, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                  VK_IMAGE_ASPECT_DEPTH_BIT, waterShadowZImage_,
                  waterShadowZMem_, waterShadowZView_, kWaterShadowMapSize,
                  kWaterShadowMapSize);

    if (!blurPass_) { // color-only pass, reused across resizes
      VkAttachmentDescription bc{};
      bc.format = fluidDepthFormat_;
      bc.samples = VK_SAMPLE_COUNT_1_BIT;
      bc.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
      bc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
      bc.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
      bc.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
      bc.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      bc.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      VkAttachmentReference bref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
      VkSubpassDescription bsub{};
      bsub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
      bsub.colorAttachmentCount = 1;
      bsub.pColorAttachments = &bref;
      VkSubpassDependency bdep{};
      bdep.srcSubpass = 0;
      bdep.dstSubpass = VK_SUBPASS_EXTERNAL;
      bdep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
      bdep.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
      bdep.dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
      bdep.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      VkRenderPassCreateInfo brp{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
      brp.attachmentCount = 1;
      brp.pAttachments = &bc;
      brp.subpassCount = 1;
      brp.pSubpasses = &bsub;
      brp.dependencyCount = 1;
      brp.pDependencies = &bdep;
      vkCheck(vkCreateRenderPass(device_, &brp, nullptr, &blurPass_),
              "vkCreateRenderPass(blur)");
    }

    VkFramebufferCreateInfo bfb{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    bfb.renderPass = blurPass_;
    bfb.attachmentCount = 1;
    bfb.width = fluidExtent_.width;
    bfb.height = fluidExtent_.height;
    bfb.layers = 1;
    bfb.pAttachments = &blurTempView_;
    vkCheck(vkCreateFramebuffer(device_, &bfb, nullptr, &blurTempFB_),
            "vkCreateFramebuffer(blurTemp)");
    bfb.pAttachments = &blurSmoothView_;
    vkCheck(vkCreateFramebuffer(device_, &bfb, nullptr, &blurSmoothFB_),
            "vkCreateFramebuffer(blurSmooth)");
    bfb.pAttachments = &fluidThicknessView_;
    vkCheck(vkCreateFramebuffer(device_, &bfb, nullptr, &fluidThicknessFB_),
            "vkCreateFramebuffer(fluidThickness)");
    bfb.pAttachments = &thicknessBlurTempView_;
    vkCheck(vkCreateFramebuffer(device_, &bfb, nullptr, &thicknessBlurTempFB_),
            "vkCreateFramebuffer(thicknessBlurTemp)");
    bfb.width = kWaterShadowMapSize;
    bfb.height = kWaterShadowMapSize;
    bfb.pAttachments = &waterShadowView_;
    vkCheck(vkCreateFramebuffer(device_, &bfb, nullptr, &waterShadowFB_),
            "vkCreateFramebuffer(waterShadow)");

    std::array<VkImageView, 2> shadowDepthAttachments = {
        waterShadowDepthView_, waterShadowZView_};
    VkFramebufferCreateInfo sdfb{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    sdfb.renderPass = fluidDepthPass_;
    sdfb.attachmentCount = (uint32_t)shadowDepthAttachments.size();
    sdfb.pAttachments = shadowDepthAttachments.data();
    sdfb.width = kWaterShadowMapSize;
    sdfb.height = kWaterShadowMapSize;
    sdfb.layers = 1;
    vkCheck(vkCreateFramebuffer(device_, &sdfb, nullptr,
                                &waterShadowDepthFB_),
            "vkCreateFramebuffer(waterShadowDepth)");
  }

  void createFluidPipelines() {
    // depth target
    VkPushConstantRange depthPC{};
    depthPC.stageFlags =
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    depthPC.offset = 0;
    depthPC.size = sizeof(DepthPush);
    VkPipelineLayoutCreateInfo dpl{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    dpl.setLayoutCount = 1;
    dpl.pSetLayouts = &descriptorSetLayout_;
    dpl.pushConstantRangeCount = 1;
    dpl.pPushConstantRanges = &depthPC;
    vkCheck(
        vkCreatePipelineLayout(device_, &dpl, nullptr, &depthPipelineLayout_),
        "vkCreatePipelineLayout(depth)");

    VkVertexInputBindingDescription bind{0, sizeof(float) * 2,
                                         VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription attr{0, 0, VK_FORMAT_R32G32_SFLOAT, 0};
    depthPipeline_ =
        buildGraphicsPipeline("fluid_depth.vert.spv", "fluid_depth.frag.spv",
                              fluidDepthPass_, depthPipelineLayout_, &bind,
                              &attr, /*attrCount*/ 1, /*depthTest*/ true, 1,
                              false, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP);

    std::array<VkDescriptorSetLayoutBinding, 8> sb{};
    for (uint32_t i = 0; i < 6; ++i) {
      sb[i].binding = i;
      sb[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      sb[i].descriptorCount = 1;
      sb[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    sb[6].binding = 6;
    sb[6].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sb[6].descriptorCount = 1;
    sb[6].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    sb[7].binding = 7;
    sb[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sb[7].descriptorCount = 1;
    sb[7].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo sl{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    sl.bindingCount = (uint32_t)sb.size();
    sl.pBindings = sb.data();
    vkCheck(vkCreateDescriptorSetLayout(device_, &sl, nullptr,
                                        &compositeSetLayout_),
            "vkCreateDescriptorSetLayout(composite)");

    std::array<VkDescriptorPoolSize, 2> ps = {{
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 42},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 6},
    }};
    VkDescriptorPoolCreateInfo pp{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pp.maxSets = 6; // composite + three depth blur + two thickness blur sets
    pp.poolSizeCount = (uint32_t)ps.size();
    pp.pPoolSizes = ps.data();
    vkCheck(vkCreateDescriptorPool(device_, &pp, nullptr, &compositePool_),
            "vkCreateDescriptorPool(composite)");
    VkDescriptorSetAllocateInfo sa{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    sa.descriptorPool = compositePool_;
    sa.descriptorSetCount = 1;
    sa.pSetLayouts = &compositeSetLayout_;
    vkCheck(vkAllocateDescriptorSets(device_, &sa, &compositeSet_),
            "vkAllocateDescriptorSets(composite)");
    vkCheck(vkAllocateDescriptorSets(device_, &sa, &blurSetH_),
            "vkAllocateDescriptorSets(blurH)");
    vkCheck(vkAllocateDescriptorSets(device_, &sa, &blurSetV_),
            "vkAllocateDescriptorSets(blurV)");
    vkCheck(vkAllocateDescriptorSets(device_, &sa, &blurSetSmooth_),
            "vkAllocateDescriptorSets(blurSmooth)");
    vkCheck(vkAllocateDescriptorSets(device_, &sa, &thicknessBlurSet_),
            "vkAllocateDescriptorSets(thicknessBlur)");
    vkCheck(vkAllocateDescriptorSets(device_, &sa, &thicknessBlurSetV_),
            "vkAllocateDescriptorSets(thicknessBlurV)");
    updateFluidDescriptors();

    VkPushConstantRange waterPC{};
    waterPC.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    waterPC.offset = 0;
    waterPC.size = sizeof(WaterPush);
    VkPipelineLayoutCreateInfo bpl{
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    bpl.setLayoutCount = 1;
    bpl.pSetLayouts = &compositeSetLayout_;
    bpl.pushConstantRangeCount = 1;
    bpl.pPushConstantRanges = &waterPC;
    vkCheck(vkCreatePipelineLayout(device_, &bpl, nullptr,
                     &compositePipelineLayout_),
        "vkCreatePipelineLayout(composite)");
    vkCheck(
      vkCreatePipelineLayout(device_, &bpl, nullptr, &blurPipelineLayout_),
      "vkCreatePipelineLayout(blur)");
    blurPipeline_ = buildGraphicsPipeline(
        "fullscreen.vert.spv", "fluid_blur.frag.spv", blurPass_,
        blurPipelineLayout_, nullptr, nullptr, 0, false);
    thicknessBlurPipeline_ = buildGraphicsPipeline(
        "fullscreen.vert.spv", "fluid_thickness_blur.frag.spv", blurPass_,
        blurPipelineLayout_, nullptr, nullptr, 0, false);
    thicknessPipeline_ = buildGraphicsPipeline(
        "fluid_depth.vert.spv", "fluid_thickness.frag.spv", blurPass_,
        depthPipelineLayout_, &bind, &attr, /*attrCount*/ 1,
        /*depthTest*/ false, 1, /*additiveBlend*/ true,
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP);

    backgroundPipeline_ = buildGraphicsPipeline(
      "fullscreen.vert.spv", "scene_background.frag.spv", scenePass_,
      compositePipelineLayout_, nullptr, nullptr, 0, /*depthTest*/ false, 2);

    VkPushConstantRange scenePC{};
    scenePC.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    scenePC.size = sizeof(ScenePush);
    VkPipelineLayoutCreateInfo spl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    spl.setLayoutCount = 1;
    spl.pSetLayouts = &compositeSetLayout_;
    spl.pushConstantRangeCount = 1;
    spl.pPushConstantRanges = &scenePC;
    vkCheck(vkCreatePipelineLayout(device_, &spl, nullptr, &sceneMeshPipelineLayout_),
            "vkCreatePipelineLayout(scene mesh)");
    VkVertexInputBindingDescription sceneBind{0, sizeof(float) * 6,
                                               VK_VERTEX_INPUT_RATE_VERTEX};
    std::array<VkVertexInputAttributeDescription, 2> sceneAttr = {{
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT, sizeof(float) * 3},
    }};
    sceneMeshPipeline_ = buildGraphicsPipeline(
        "scene_mesh.vert.spv", "scene_mesh.frag.spv", scenePass_,
        sceneMeshPipelineLayout_, &sceneBind, sceneAttr.data(),
        (uint32_t)sceneAttr.size(), /*depthTest*/ true, 2);

    compositePipeline_ = buildGraphicsPipeline(
        "fullscreen.vert.spv", "fluid_composite.frag.spv", renderPass_,
        compositePipelineLayout_, nullptr, nullptr, 0, /*depthTest*/ false);
  }

  void updateFluidDescriptors() {
    auto writeSampler = [&](VkDescriptorSet set, uint32_t binding,
                VkImageView view, VkSampler sampler) {
      VkDescriptorImageInfo img{};
      img.sampler = sampler;
      img.imageView = view;
      img.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      w.dstSet = set;
      w.dstBinding = binding;
      w.descriptorCount = 1;
      w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      w.pImageInfo = &img;
      vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
    };
    writeSampler(blurSetH_, 0, fluidDepthView_, fluidSampler_);
    writeSampler(blurSetV_, 0, blurTempView_, fluidSampler_);
    writeSampler(blurSetSmooth_, 0, blurSmoothView_, fluidSampler_);
    writeSampler(thicknessBlurSet_, 0, fluidThicknessView_, fluidSampler_);
    writeSampler(thicknessBlurSetV_, 0, thicknessBlurTempView_, fluidSampler_);
    writeSampler(compositeSet_, 0, blurSmoothView_, fluidSampler_);
    writeSampler(compositeSet_, 1, sceneColorView_, sceneSampler_);
    writeSampler(compositeSet_, 2, environmentView_, envSampler_);
    writeSampler(compositeSet_, 3, sceneDepthView_, sceneSampler_);
    writeSampler(compositeSet_, 4, blurTempView_, fluidSampler_);
    writeSampler(compositeSet_, 5, waterShadowView_, sceneSampler_);
    writeSampler(compositeSet_, 7, waterShadowDepthView_, sceneSampler_);
    VkDescriptorBufferInfo lightingInfo{};
    lightingInfo.buffer = sceneLightingBuf_;
    lightingInfo.range = sizeof(SceneLightingParams);
    VkWriteDescriptorSet lightingWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    lightingWrite.dstSet = compositeSet_;
    lightingWrite.dstBinding = 6;
    lightingWrite.descriptorCount = 1;
    lightingWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    lightingWrite.pBufferInfo = &lightingInfo;
    vkUpdateDescriptorSets(device_, 1, &lightingWrite, 0, nullptr);
  }

  VkPipeline
  buildGraphicsPipeline(const char *vertSpv, const char *fragSpv,
                        VkRenderPass pass, VkPipelineLayout layout,
                        const VkVertexInputBindingDescription *bind,
                        const VkVertexInputAttributeDescription *attr,
                        uint32_t attrCount, bool depthTest,
                        uint32_t colorAttachmentCount = 1,
                        bool additiveBlend = false,
                        VkPrimitiveTopology topology =
                            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST) {
    VkShaderModule vert =
        createShaderModule(readFile(std::string(SHADER_DIR) + "/" + vertSpv));
    VkShaderModule frag =
        createShaderModule(readFile(std::string(SHADER_DIR) + "/" + fragSpv));
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

    VkPipelineVertexInputStateCreateInfo vi{
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    if (bind) {
      vi.vertexBindingDescriptionCount = 1;
      vi.pVertexBindingDescriptions = bind;
      vi.vertexAttributeDescriptionCount = attrCount;
      vi.pVertexAttributeDescriptions = attr;
    }

    VkPipelineInputAssemblyStateCreateInfo ia{
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = topology;

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
    ds.depthTestEnable = depthTest ? VK_TRUE : VK_FALSE;
    ds.depthWriteEnable = depthTest ? VK_TRUE : VK_FALSE;
    ds.depthCompareOp = VK_COMPARE_OP_LESS;

    std::vector<VkPipelineColorBlendAttachmentState> cba(colorAttachmentCount);
    for (auto &attachment : cba) {
      attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                                  VK_COLOR_COMPONENT_G_BIT |
                                  VK_COLOR_COMPONENT_B_BIT |
                                  VK_COLOR_COMPONENT_A_BIT;
      attachment.blendEnable = additiveBlend ? VK_TRUE : VK_FALSE;
      if (additiveBlend) {
        attachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        attachment.colorBlendOp = VK_BLEND_OP_ADD;
        attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        attachment.alphaBlendOp = VK_BLEND_OP_ADD;
      }
    }
    VkPipelineColorBlendStateCreateInfo cb{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = colorAttachmentCount;
    cb.pAttachments = cba.data();

    std::array<VkDynamicState, 2> dynStates = {VK_DYNAMIC_STATE_VIEWPORT,
                                               VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{
        VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dyn.dynamicStateCount = (uint32_t)dynStates.size();
    dyn.pDynamicStates = dynStates.data();

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
    ci.layout = layout;
    ci.renderPass = pass;
    ci.subpass = 0;
    VkPipeline pipeline;
    vkCheck(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &ci, nullptr,
                                      &pipeline),
            "vkCreateGraphicsPipelines(fluid)");
    vkDestroyShaderModule(device_, vert, nullptr);
    vkDestroyShaderModule(device_, frag, nullptr);
    return pipeline;
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
      vkCheck(vkCreateFence(device_, &fi, nullptr, &inFlightFences_[i]),
              "vkCreateFence");
    }

    renderFinishedSemaphores_.resize(swapchainImages_.size());
    for (auto &sem : renderFinishedSemaphores_)
      vkCheck(vkCreateSemaphore(device_, &si, nullptr, &sem),
              "vkCreateSemaphore(renderFinished)");
  }

  // Barrier so one compute pass's SSBO writes are visible to the next.
  void ssboBarrier(VkCommandBuffer cmd) {
    VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b, 0,
                         nullptr, 0, nullptr);
  }

  // Explicit barrier between two render passes
  void colorToSampledBarrier(VkCommandBuffer cmd, VkImage image) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr,
                         0, nullptr, 1, &b);
  }

  void dispatchStage(VkCommandBuffer cmd, VkPipeline pipeline, uint32_t groups,
                     bool barrierAfter = true) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdDispatch(cmd, groups, 1, 1);
    if (barrierAfter)
      ssboBarrier(cmd);
  }

  void radixSort(VkCommandBuffer cmd) {
    VrdxSortInfo info{};
    info.elementCount = numParticles_;
    info.keysBuffer = keysBuf_;
    info.valuesBuffer = indicesBuf_;
    info.storageBuffer = sortStorageBuf_;
    info.queryPool = VK_NULL_HANDLE;
    vrdxCmdSort(cmd, sorter_, &info);
    ssboBarrier(cmd);
  }

  void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex,
                           const DepthPush &dp, const WaterPush &wp,
                           int substeps) {
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkCheck(vkBeginCommandBuffer(cmd, &bi), "vkBeginCommandBuffer");

    const uint32_t groupsN = (numParticles_ + 255u) / 256u;
    const uint32_t groupsGrid = (gridCellCount_ + 255u) / 256u;
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            computePipelineLayout_, 0, 1, &descriptorSet_, 0,
                            nullptr);
    for (int s = 0; s < substeps; ++s) {
      dispatchStage(cmd, computePipelines_[0],
                    groupsN); // external + exact keys
      radixSort(cmd);         // sort keys + indices
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              computePipelineLayout_, 0, 1, &descriptorSet_, 0,
                              nullptr);
      dispatchStage(cmd, computePipelines_[1], groupsN); // gather sorted state
      dispatchStage(cmd, computePipelines_[2], groupsGrid); // reset exact grid
      dispatchStage(cmd, computePipelines_[3], groupsN);    // bucket starts
      dispatchStage(cmd, computePipelines_[4], groupsN);    // density
      dispatchStage(cmd, computePipelines_[5], groupsN);    // pressure
      dispatchStage(cmd, computePipelines_[6], groupsN,
                    s + 1 < substeps); // viscosity + integrate
    }

    VkMemoryBarrier toVertex{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    toVertex.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    toVertex.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, 0, 1, &toVertex,
                         0, nullptr, 0, nullptr);

    VkViewport viewport{0.0f,
                        0.0f,
                        (float)swapchainExtent_.width,
                        (float)swapchainExtent_.height,
                        0.0f,
                        1.0f};
    VkRect2D scissor{{0, 0}, swapchainExtent_};
    VkViewport fluidViewport{0.0f, 0.0f, (float)fluidExtent_.width,
                             (float)fluidExtent_.height, 0.0f, 1.0f};
    VkRect2D fluidScissor{{0, 0}, fluidExtent_};

    // Render water thickness from the lights view.
    const float azimuth = fluidSettings_.sunAzimuthDegrees * 0.01745329252f;
    const float elevation = fluidSettings_.sunElevationDegrees * 0.01745329252f;
    const float3 sunDir = normalize(float3(std::cos(elevation) * std::cos(azimuth),
                                           std::sin(elevation),
                                           std::cos(elevation) * std::sin(azimuth)));
    const float3 lightForward = -sunDir;
    const float3 lightRight = normalize(cross(lightForward, float3(0, 1, 0)));
    const float3 lightUp = cross(lightRight, lightForward);
    const float3 lightEye = sunDir * 12.0f;
    const float4x4 lightView(
        float4(lightRight.x, lightUp.x, -lightForward.x, 0),
        float4(lightRight.y, lightUp.y, -lightForward.y, 0),
        float4(lightRight.z, lightUp.z, -lightForward.z, 0),
        float4(-dot(lightRight, lightEye), -dot(lightUp, lightEye),
               dot(lightForward, lightEye), 1));
    constexpr float lightNear = 0.1f;
    constexpr float lightFar = 32.0f;
    const float4x4 lightProj(
        float4(1.0f / 8.0f, 0, 0, 0), float4(0, -1.0f / 8.0f, 0, 0),
        float4(0, 0, 1.0f / (lightNear - lightFar), 0),
        float4(0, 0, lightNear / (lightNear - lightFar), 1));
    SceneLightingParams lighting{};
    lighting.lightVP = linalg::mul(lightProj, lightView);
    lighting.lightView = lightView;
    lighting.sunDirection = float4(sunDir, 0.0f);
    lighting.extinction = float4(fluidSettings_.extinctionRed,
                                 fluidSettings_.extinctionGreen,
                                 fluidSettings_.extinctionBlue, 0.0f) *
                          fluidSettings_.absorptionScale;
    lighting.shadowParams = float4(fluidSettings_.shadowAmbientLight, 0, 0, 0);
    std::memcpy(sceneLightingMapped_, &lighting, sizeof(lighting));
    const float shadowSize = static_cast<float>(kWaterShadowMapSize);
    VkViewport shadowViewport{0.0f, 0.0f, shadowSize, shadowSize, 0.0f, 1.0f};
    VkRect2D shadowScissor{{0, 0},
                           {kWaterShadowMapSize, kWaterShadowMapSize}};
    DepthPush shadowDp{};
    shadowDp.view = lightView;
    shadowDp.proj = lightProj;
    shadowDp.camRight = float4(lightRight, 0);
    shadowDp.camUp = float4(lightUp, 0);
    shadowDp.params = dp.params;

    const int shadowInterval =
        std::max(fluidSettings_.shadowUpdateInterval, 1);
    const bool updateWaterShadow =
        !waterShadowValid_ || (renderedFrameCount_ % shadowInterval == 0) ||
        std::abs(fluidSettings_.sunAzimuthDegrees - shadowSunAzimuth_) > 0.01f ||
        std::abs(fluidSettings_.sunElevationDegrees - shadowSunElevation_) > 0.01f;
    if (updateWaterShadow) {
    std::array<VkClearValue, 2> shadowDepthClears{};
    shadowDepthClears[0].color = {{1.0e4f, 0, 0, 0}};
    shadowDepthClears[1].depthStencil = {1.0f, 0};
    VkRenderPassBeginInfo shadowDepthRp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    shadowDepthRp.renderPass = fluidDepthPass_;
    shadowDepthRp.framebuffer = waterShadowDepthFB_;
    shadowDepthRp.renderArea.extent = {kWaterShadowMapSize,
                                       kWaterShadowMapSize};
    shadowDepthRp.clearValueCount = (uint32_t)shadowDepthClears.size();
    shadowDepthRp.pClearValues = shadowDepthClears.data();
    vkCmdBeginRenderPass(cmd, &shadowDepthRp, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, depthPipeline_);
    vkCmdSetViewport(cmd, 0, 1, &shadowViewport);
    vkCmdSetScissor(cmd, 0, 1, &shadowScissor);
    vkCmdPushConstants(cmd, depthPipelineLayout_,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(DepthPush), &shadowDp);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            depthPipelineLayout_, 0, 1, &descriptorSet_, 0,
                            nullptr);
    VkBuffer shadowDepthVB[] = {quadBuffer_};
    VkDeviceSize shadowDepthOffsets[] = {0};
    vkCmdBindVertexBuffers(cmd, 0, 1, shadowDepthVB, shadowDepthOffsets);
    if (numParticles_ > 0)
      vkCmdDraw(cmd, 4, numParticles_, 0, 0);
    vkCmdEndRenderPass(cmd);
    colorToSampledBarrier(cmd, waterShadowDepthImage_);

    VkClearValue shadowClear{};
    shadowClear.color = {{0, 0, 0, 0}};
    VkRenderPassBeginInfo shadowRp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    shadowRp.renderPass = blurPass_;
    shadowRp.framebuffer = waterShadowFB_;
    shadowRp.renderArea.extent = {kWaterShadowMapSize, kWaterShadowMapSize};
    shadowRp.clearValueCount = 1;
    shadowRp.pClearValues = &shadowClear;
    vkCmdBeginRenderPass(cmd, &shadowRp, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, thicknessPipeline_);
    vkCmdSetViewport(cmd, 0, 1, &shadowViewport);
    vkCmdSetScissor(cmd, 0, 1, &shadowScissor);
    vkCmdPushConstants(cmd, depthPipelineLayout_,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(DepthPush), &shadowDp);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            depthPipelineLayout_, 0, 1, &descriptorSet_, 0,
                            nullptr);
    VkBuffer shadowVB[] = {quadBuffer_};
    VkDeviceSize shadowOffsets[] = {0};
    vkCmdBindVertexBuffers(cmd, 0, 1, shadowVB, shadowOffsets);
    if (numParticles_ > 0)
      vkCmdDraw(cmd, 4, numParticles_, 0, 0);
    vkCmdEndRenderPass(cmd);
    colorToSampledBarrier(cmd, waterShadowImage_);
    waterShadowValid_ = true;
    shadowSunAzimuth_ = fluidSettings_.sunAzimuthDegrees;
    shadowSunElevation_ = fluidSettings_.sunElevationDegrees;
    }

    // Render the environment background first so the water shader can
    // refract against it in screen space.
    VkClearValue sceneClear{};
    sceneClear.color = {{0.02f, 0.03f, 0.06f, 1.0f}};
    VkClearValue sceneDepthClear{};
    sceneDepthClear.color = {{1.0e4f, 0.0f, 0.0f, 0.0f}};
    VkRenderPassBeginInfo srp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    srp.renderPass = scenePass_;
    srp.framebuffer = sceneFB_;
    srp.renderArea.offset = {0, 0};
    srp.renderArea.extent = swapchainExtent_;
    VkClearValue sceneZClear{};
    sceneZClear.depthStencil = {1.0f, 0};
    std::array<VkClearValue, 3> sceneClears = {sceneClear, sceneDepthClear,
                                                sceneZClear};
    srp.clearValueCount = (uint32_t)sceneClears.size();
    srp.pClearValues = sceneClears.data();
    vkCmdBeginRenderPass(cmd, &srp, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, backgroundPipeline_);
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                compositePipelineLayout_, 0, 1, &compositeSet_, 0,
                nullptr);
    vkCmdPushConstants(cmd, compositePipelineLayout_,
               VK_SHADER_STAGE_VERTEX_BIT |
                 VK_SHADER_STAGE_FRAGMENT_BIT,
               0, sizeof(WaterPush), &wp);
    vkCmdDraw(cmd, 3, 1, 0, 0);

    // Opaque geometry writes both regular depth (for visibility) and linear
    // view-space depth (for the later fluid thickness/refraction pass).
    ScenePush scenePush{};
    scenePush.view = dp.view;
    scenePush.proj = dp.proj;
    if (sceneMeshVertexCount_ > 0) {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, sceneMeshPipeline_);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              sceneMeshPipelineLayout_, 0, 1, &compositeSet_,
                              0, nullptr);
      vkCmdPushConstants(cmd, sceneMeshPipelineLayout_, VK_SHADER_STAGE_VERTEX_BIT,
                         0, sizeof(ScenePush), &scenePush);
      VkBuffer sceneVB[] = {sceneMeshBuffer_};
      VkDeviceSize sceneOffsets[] = {0};
      vkCmdBindVertexBuffers(cmd, 0, 1, sceneVB, sceneOffsets);
      vkCmdDraw(cmd, sceneMeshVertexCount_, 1, 0, 0);
    }
    vkCmdEndRenderPass(cmd);
    colorToSampledBarrier(cmd, sceneColorImage_);
    colorToSampledBarrier(cmd, sceneDepthImage_);

    // Accumulate particle discs into a thickness texture.  This is separate
    // from surface depth: overlapping particles add optical path length.
    VkClearValue thicknessClear{};
    thicknessClear.color = {{0.0f, 0.0f, 0.0f, 0.0f}};
    VkRenderPassBeginInfo trp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    trp.renderPass = blurPass_;
    trp.framebuffer = fluidThicknessFB_;
    trp.renderArea.offset = {0, 0};
    trp.renderArea.extent = fluidExtent_;
    trp.clearValueCount = 1;
    trp.pClearValues = &thicknessClear;
    vkCmdBeginRenderPass(cmd, &trp, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, thicknessPipeline_);
    vkCmdSetViewport(cmd, 0, 1, &fluidViewport);
    vkCmdSetScissor(cmd, 0, 1, &fluidScissor);
    vkCmdPushConstants(cmd, depthPipelineLayout_,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(DepthPush), &dp);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            depthPipelineLayout_, 0, 1, &descriptorSet_, 0,
                            nullptr);
    VkBuffer thicknessVB[] = {quadBuffer_};
    VkDeviceSize thicknessOffsets[] = {0};
    vkCmdBindVertexBuffers(cmd, 0, 1, thicknessVB, thicknessOffsets);
    if (numParticles_ > 0)
      vkCmdDraw(cmd, 4, numParticles_, 0, 0);
    vkCmdEndRenderPass(cmd);
    colorToSampledBarrier(cmd, fluidThicknessImage_);

    DepthPush surfaceDp = dp;
    surfaceDp.params.x *= 1.35f;

    // Render spheres to depth target
    std::array<VkClearValue, 2> depthClears{};
    depthClears[0].color = {{1.0e4f, 0.0f, 0.0f, 0.0f}}; // "empty" depth
    depthClears[1].depthStencil = {1.0f, 0};
    VkRenderPassBeginInfo drp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    drp.renderPass = fluidDepthPass_;
    drp.framebuffer = fluidDepthFB_;
    drp.renderArea.offset = {0, 0};
    drp.renderArea.extent = fluidExtent_;
    drp.clearValueCount = (uint32_t)depthClears.size();
    drp.pClearValues = depthClears.data();
    vkCmdBeginRenderPass(cmd, &drp, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, depthPipeline_);
    vkCmdSetViewport(cmd, 0, 1, &fluidViewport);
    vkCmdSetScissor(cmd, 0, 1, &fluidScissor);
    vkCmdPushConstants(cmd, depthPipelineLayout_,
                       VK_SHADER_STAGE_VERTEX_BIT |
                           VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(DepthPush), &surfaceDp);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            depthPipelineLayout_, 0, 1, &descriptorSet_, 0,
                            nullptr);
    VkBuffer vbs[] = {quadBuffer_};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(cmd, 0, 1, vbs, offsets);
    if (numParticles_ > 0)
      vkCmdDraw(cmd, 4, numParticles_, 0, 0);
    vkCmdEndRenderPass(cmd);
    colorToSampledBarrier(cmd, fluidDepthImage_);

    // Bilateral blur using two passes
    auto blurPass = [&](VkFramebuffer fb, VkDescriptorSet src, float dx,
                        float dy, bool fillSilhouette) {
      VkRenderPassBeginInfo brp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
      brp.renderPass = blurPass_;
      brp.framebuffer = fb;
      brp.renderArea.offset = {0, 0};
      brp.renderArea.extent = fluidExtent_;
      brp.clearValueCount = 0;
      vkCmdBeginRenderPass(cmd, &brp, VK_SUBPASS_CONTENTS_INLINE);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, blurPipeline_);
      vkCmdSetViewport(cmd, 0, 1, &fluidViewport);
      vkCmdSetScissor(cmd, 0, 1, &fluidScissor);
      BlurPush bpc{{dx, dy},
                   fluidSettings_.depthDifferenceStrength,
                   fluidSettings_.maxBlurRadius * activeRenderScale_,
                   fluidSettings_.blurStrength,
                   surfaceDp.params.x,
                   fillSilhouette ? 1.0f : 0.0f};
      vkCmdPushConstants(cmd, blurPipelineLayout_, VK_SHADER_STAGE_FRAGMENT_BIT,
                         0, sizeof(bpc), &bpc);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              blurPipelineLayout_, 0, 1, &src, 0, nullptr);
      vkCmdDraw(cmd, 3, 1, 0, 0);
      vkCmdEndRenderPass(cmd);
    };
    float invW = 1.0f / (float)fluidExtent_.width;
    float invH = 1.0f / (float)fluidExtent_.height;
    for (int i = 0; i < fluidSettings_.blurIterations; ++i) {
      blurPass(blurTempFB_, i == 0 ? blurSetH_ : blurSetSmooth_, invW,
               0.0f, i == 0); // horizontal
      colorToSampledBarrier(cmd, blurTempImage_);
      blurPass(blurSmoothFB_, blurSetV_, 0.0f, invH,
               i == 0); // vertical
      colorToSampledBarrier(cmd, blurSmoothImage_);
    }

    // two pass thickness blur
    auto thicknessBlurPass = [&](VkFramebuffer fb, VkDescriptorSet src,
                                 float dx, float dy) {
      VkRenderPassBeginInfo brp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
      brp.renderPass = blurPass_;
      brp.framebuffer = fb;
      brp.renderArea.extent = fluidExtent_;
      vkCmdBeginRenderPass(cmd, &brp, VK_SUBPASS_CONTENTS_INLINE);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        thicknessBlurPipeline_);
      vkCmdSetViewport(cmd, 0, 1, &fluidViewport);
      vkCmdSetScissor(cmd, 0, 1, &fluidScissor);
      float thicknessBlurPC[5] = {dx, dy, 0.0f,
                                  fluidSettings_.maxBlurRadius * activeRenderScale_,
                                  fluidSettings_.blurStrength};
      vkCmdPushConstants(cmd, blurPipelineLayout_, VK_SHADER_STAGE_FRAGMENT_BIT,
                         0, sizeof(thicknessBlurPC), thicknessBlurPC);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              blurPipelineLayout_, 0, 1, &src, 0, nullptr);
      vkCmdDraw(cmd, 3, 1, 0, 0);
      vkCmdEndRenderPass(cmd);
    };
    thicknessBlurPass(thicknessBlurTempFB_, thicknessBlurSet_, 1.0f, 0.0f);
    colorToSampledBarrier(cmd, thicknessBlurTempImage_);
    thicknessBlurPass(blurTempFB_, thicknessBlurSetV_, 0.0f, 1.0f);
    colorToSampledBarrier(cmd, blurTempImage_);

    // composite into swapchain image
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
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, compositePipeline_);
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            compositePipelineLayout_, 0, 1, &compositeSet_, 0,
                            nullptr);
    vkCmdPushConstants(cmd, compositePipelineLayout_,
               VK_SHADER_STAGE_VERTEX_BIT |
                 VK_SHADER_STAGE_FRAGMENT_BIT,
               0, sizeof(WaterPush), &wp);
    vkCmdDraw(cmd, 3, 1, 0, 0); // fullscreen triangle
    if (imguiInitialized_)
      ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    vkCmdEndRenderPass(cmd);

    vkCheck(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
  }

  void destroyFluidSizedTargets() {
    if (waterShadowDepthFB_) {
      vkDestroyFramebuffer(device_, waterShadowDepthFB_, nullptr);
      waterShadowDepthFB_ = VK_NULL_HANDLE;
    }
    if (waterShadowZView_) {
      vkDestroyImageView(device_, waterShadowZView_, nullptr);
      waterShadowZView_ = VK_NULL_HANDLE;
    }
    if (waterShadowZImage_) {
      vkDestroyImage(device_, waterShadowZImage_, nullptr);
      waterShadowZImage_ = VK_NULL_HANDLE;
    }
    if (waterShadowZMem_) {
      vkFreeMemory(device_, waterShadowZMem_, nullptr);
      waterShadowZMem_ = VK_NULL_HANDLE;
    }
    if (waterShadowDepthView_) {
      vkDestroyImageView(device_, waterShadowDepthView_, nullptr);
      waterShadowDepthView_ = VK_NULL_HANDLE;
    }
    if (waterShadowDepthImage_) {
      vkDestroyImage(device_, waterShadowDepthImage_, nullptr);
      waterShadowDepthImage_ = VK_NULL_HANDLE;
    }
    if (waterShadowDepthMem_) {
      vkFreeMemory(device_, waterShadowDepthMem_, nullptr);
      waterShadowDepthMem_ = VK_NULL_HANDLE;
    }
    if (waterShadowFB_) {
      vkDestroyFramebuffer(device_, waterShadowFB_, nullptr);
      waterShadowFB_ = VK_NULL_HANDLE;
    }
    if (waterShadowView_) {
      vkDestroyImageView(device_, waterShadowView_, nullptr);
      waterShadowView_ = VK_NULL_HANDLE;
    }
    if (waterShadowImage_) {
      vkDestroyImage(device_, waterShadowImage_, nullptr);
      waterShadowImage_ = VK_NULL_HANDLE;
    }
    if (waterShadowMem_) {
      vkFreeMemory(device_, waterShadowMem_, nullptr);
      waterShadowMem_ = VK_NULL_HANDLE;
    }
    if (fluidThicknessFB_) {
      vkDestroyFramebuffer(device_, fluidThicknessFB_, nullptr);
      fluidThicknessFB_ = VK_NULL_HANDLE;
    }
    if (fluidThicknessView_) {
      vkDestroyImageView(device_, fluidThicknessView_, nullptr);
      fluidThicknessView_ = VK_NULL_HANDLE;
    }
    if (fluidThicknessImage_) {
      vkDestroyImage(device_, fluidThicknessImage_, nullptr);
      fluidThicknessImage_ = VK_NULL_HANDLE;
    }
    if (fluidThicknessMem_) {
      vkFreeMemory(device_, fluidThicknessMem_, nullptr);
      fluidThicknessMem_ = VK_NULL_HANDLE;
    }
    if (fluidDepthFB_) {
      vkDestroyFramebuffer(device_, fluidDepthFB_, nullptr);
      fluidDepthFB_ = VK_NULL_HANDLE;
    }
    if (fluidDepthView_) {
      vkDestroyImageView(device_, fluidDepthView_, nullptr);
      fluidDepthView_ = VK_NULL_HANDLE;
    }
    if (fluidDepthImage_) {
      vkDestroyImage(device_, fluidDepthImage_, nullptr);
      fluidDepthImage_ = VK_NULL_HANDLE;
    }
    if (fluidDepthMem_) {
      vkFreeMemory(device_, fluidDepthMem_, nullptr);
      fluidDepthMem_ = VK_NULL_HANDLE;
    }
    if (fluidZView_) {
      vkDestroyImageView(device_, fluidZView_, nullptr);
      fluidZView_ = VK_NULL_HANDLE;
    }
    if (fluidZImage_) {
      vkDestroyImage(device_, fluidZImage_, nullptr);
      fluidZImage_ = VK_NULL_HANDLE;
    }
    if (fluidZMem_) {
      vkFreeMemory(device_, fluidZMem_, nullptr);
      fluidZMem_ = VK_NULL_HANDLE;
    }
    if (blurTempFB_) {
      vkDestroyFramebuffer(device_, blurTempFB_, nullptr);
      blurTempFB_ = VK_NULL_HANDLE;
    }
    if (blurSmoothFB_) {
      vkDestroyFramebuffer(device_, blurSmoothFB_, nullptr);
      blurSmoothFB_ = VK_NULL_HANDLE;
    }
    if (thicknessBlurTempFB_) {
      vkDestroyFramebuffer(device_, thicknessBlurTempFB_, nullptr);
      thicknessBlurTempFB_ = VK_NULL_HANDLE;
    }
    if (blurTempView_) {
      vkDestroyImageView(device_, blurTempView_, nullptr);
      blurTempView_ = VK_NULL_HANDLE;
    }
    if (blurTempImage_) {
      vkDestroyImage(device_, blurTempImage_, nullptr);
      blurTempImage_ = VK_NULL_HANDLE;
    }
    if (blurTempMem_) {
      vkFreeMemory(device_, blurTempMem_, nullptr);
      blurTempMem_ = VK_NULL_HANDLE;
    }
    if (blurSmoothView_) {
      vkDestroyImageView(device_, blurSmoothView_, nullptr);
      blurSmoothView_ = VK_NULL_HANDLE;
    }
    if (blurSmoothImage_) {
      vkDestroyImage(device_, blurSmoothImage_, nullptr);
      blurSmoothImage_ = VK_NULL_HANDLE;
    }
    if (blurSmoothMem_) {
      vkFreeMemory(device_, blurSmoothMem_, nullptr);
      blurSmoothMem_ = VK_NULL_HANDLE;
    }
    if (thicknessBlurTempView_) {
      vkDestroyImageView(device_, thicknessBlurTempView_, nullptr);
      thicknessBlurTempView_ = VK_NULL_HANDLE;
    }
    if (thicknessBlurTempImage_) {
      vkDestroyImage(device_, thicknessBlurTempImage_, nullptr);
      thicknessBlurTempImage_ = VK_NULL_HANDLE;
    }
    if (thicknessBlurTempMem_) {
      vkFreeMemory(device_, thicknessBlurTempMem_, nullptr);
      thicknessBlurTempMem_ = VK_NULL_HANDLE;
    }
  }

  void destroySceneSizedTargets() {
    if (sceneFB_) {
      vkDestroyFramebuffer(device_, sceneFB_, nullptr);
      sceneFB_ = VK_NULL_HANDLE;
    }
    if (sceneColorView_) {
      vkDestroyImageView(device_, sceneColorView_, nullptr);
      sceneColorView_ = VK_NULL_HANDLE;
    }
    if (sceneColorImage_) {
      vkDestroyImage(device_, sceneColorImage_, nullptr);
      sceneColorImage_ = VK_NULL_HANDLE;
    }
    if (sceneColorMem_) {
      vkFreeMemory(device_, sceneColorMem_, nullptr);
      sceneColorMem_ = VK_NULL_HANDLE;
    }
    if (sceneDepthView_) {
      vkDestroyImageView(device_, sceneDepthView_, nullptr);
      sceneDepthView_ = VK_NULL_HANDLE;
    }
    if (sceneDepthImage_) {
      vkDestroyImage(device_, sceneDepthImage_, nullptr);
      sceneDepthImage_ = VK_NULL_HANDLE;
    }
    if (sceneDepthMem_) {
      vkFreeMemory(device_, sceneDepthMem_, nullptr);
      sceneDepthMem_ = VK_NULL_HANDLE;
    }
    if (sceneZView_) {
      vkDestroyImageView(device_, sceneZView_, nullptr);
      sceneZView_ = VK_NULL_HANDLE;
    }
    if (sceneZImage_) {
      vkDestroyImage(device_, sceneZImage_, nullptr);
      sceneZImage_ = VK_NULL_HANDLE;
    }
    if (sceneZMem_) {
      vkFreeMemory(device_, sceneZMem_, nullptr);
      sceneZMem_ = VK_NULL_HANDLE;
    }
  }

  void cleanupSwapchain() {
    destroySceneSizedTargets();
    destroyFluidSizedTargets();
    if (depthView_)
      vkDestroyImageView(device_, depthView_, nullptr);
    if (depthImage_)
      vkDestroyImage(device_, depthImage_, nullptr);
    if (depthMemory_)
      vkFreeMemory(device_, depthMemory_, nullptr);
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
    for (auto sem : renderFinishedSemaphores_)
      if (sem)
        vkDestroySemaphore(device_, sem, nullptr);
    renderFinishedSemaphores_.clear();
    cleanupSwapchain();
    createSwapchain();
    createImageViews();
    createDepthResources();
    createSceneTargets();
    createFramebuffers();
    createFluidTargets();
    updateFluidDescriptors();
    if (imguiInitialized_)
      ImGui_ImplVulkan_SetMinImageCount((uint32_t)swapchainImages_.size());
    // swapchain image count may have changed; rebuild the per-image semaphores
    VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    renderFinishedSemaphores_.resize(swapchainImages_.size());
    for (auto &sem : renderFinishedSemaphores_)
      vkCheck(vkCreateSemaphore(device_, &si, nullptr, &sem),
              "vkCreateSemaphore(renderFinished)");
  }
};

} // namespace vkr
