#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>

#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
#include <imgui.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "linalg.h"
using namespace linalg::aliases;

#include "obj_loader.h"
#include "utils.h"
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

namespace vkr {

struct SphParams {
  float gravity[4];    // xyz = gravity, w = near-pressure multiplier
  float boundsSize[4]; // xyz = tank size
  float deltaT;
  float smoothingRadius;
  float particleMass;
  float targetDensity;
  float pressureMultiplier;
  float viscosityStrength;
  float collisionDamping;
  float densityKernelScale;
  float pressureGradientKernelScale;
  float viscosityKernelScale;
  uint32_t numParticles;
  float epsilon;
  uint32_t grid[4]; // xyz = dimensions, w = total cells
};
static_assert(sizeof(SphParams) == 96, "SphParams layout changed");

struct InteractionParams {
  float point[4];
  float radius;
  float strength; // > 0 attract, < 0 repel
};
static_assert(sizeof(InteractionParams) == 24,
              "InteractionParams layout changed");

struct alignas(16) WhitewaterParams {
  float generationRate;
  float trappedAirMin;
  float trappedAirMax;
  float kineticEnergyMin;
  float kineticEnergyMax;
  float bubbleBuoyancy;
  float sprayDrag;
  float fluidFollow;
  float lifetimeMin;
  float lifetimeMax;
  float bubbleScale;
  float renderScale;
  uint32_t capacity;
  uint32_t sprayMaxNeighbours;
  uint32_t bubbleMinNeighbours;
  uint32_t enabled;
  uint32_t debugClassification;
  uint32_t activeCount;
};
static_assert(sizeof(WhitewaterParams) == 80,
              "WhitewaterParams layout changed");

struct WhiteParticle {
  float positionLife[4];
  float velocityScale[4];
};
static_assert(sizeof(WhiteParticle) == 32, "WhiteParticle layout changed");

struct WhitewaterAllocatorHeader {
  uint32_t spawnCursor;
  uint32_t activeCount;
};

struct DepthPush {
  float4x4 view;
  float4x4 proj;
  float4 camRight;
  float4 camUp;
  float4 params; // x: radius, y: debug mode
};
static_assert(sizeof(DepthPush) == 176, "DepthPush layout changed");

struct BlurPush {
  float dir[2]; // Pixel direction.
  float depthDifferenceStrength;
  float maxScreenSpaceRadius;
  float strength;
  float particleRadius;
  float fillSilhouette;
  float tanHalfFov;
};

struct ThicknessBlurPush {
  float dir[2];
  float maxScreenSpaceRadius;
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
  float4 camForward; // w = tan half fov
  float4 material;
  float4 extinction;
};
static_assert(sizeof(WaterPush) == 80, "WaterPush layout changed");

struct ScenePush {
  float4x4 view;
  float4x4 proj;
};

struct SceneLightingParams {
  float4x4 lightVP;
  float4x4 lightView;
  float4 sunDirection;
  float4 extinction;
  float4 shadowParams; // x = ambient light left inside the shadow
};
static_assert(sizeof(SceneLightingParams) == 176,
              "SceneLightingParams layout changed");

struct FluidRenderSettings {
  float renderScale = 0.75f;
  float maxBlurRadius = 12.0f;
  float blurStrength = 1.25f;
  float depthDifferenceStrength = 20.0f;
  int blurIterations = 1;
  float refractionStrength = 0.10f;
  float absorptionScale = 0.7f;
  float extinctionRed = 0.55f;
  float extinctionGreen = 0.15f;
  float extinctionBlue = 0.1f;
  float baseReflectance = 0.02f;
  float sunAngleDegrees = 36.0f;
  float sunElevationDegrees = 53.0f;
  float shadowAmbientLight = 0.17f;
  int shadowUpdateInterval = 2;
};

struct WhitewaterSettings {
  bool enabled = true;
  float spawnRate = 120.0f;
  float trappedAirMin = 10.0f;
  float trappedAirMax = 20.0f;
  float kineticEnergyMin = 15.0f;
  float kineticEnergyMax = 30.0f;
  float lifetimeMin = 2.0f;
  float lifetimeMax = 12.0f;
  float bubbleBuoyancy = 1.4f;
  float sprayDrag = 0.04f;
  float fluidFollow = 3.0f;
  float bubbleScale = 0.3f;
  float renderScale = 0.25f;
  int sprayMaxNeighbours = 5;
  int bubbleMinNeighbours = 20;
};

enum DebugVisualizationMode : int {
  kDebugNone = 0,
  kDebugWhitewaterClassification = 1,
  kDebugFluidVelocity = 2,
  kDebugFluidPressure = 3,
};

struct RendererConfig {
  int width;
  int height;
  const char *title;
  uint32_t particleCount;
  uint32_t whitewaterCapacity;
  uint32_t gridCellCount;
};

inline void vkCheck(VkResult r, const char *what) {
  if (r != VK_SUCCESS) {
    throw std::runtime_error(std::string("Vulkan error in ") + what + ": " +
                             std::to_string(static_cast<int>(r)));
  }
}

class VulkanRenderer {
  struct AllocatedBuffer {
    VkBuffer handle = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void *mapped = nullptr;

    AllocatedBuffer() = default;
    AllocatedBuffer(const AllocatedBuffer &) = delete;
    AllocatedBuffer &operator=(const AllocatedBuffer &) = delete;
    operator VkBuffer() const { return handle; }
  };

public:
  static constexpr int kMaxFramesInFlight = 2;
  static constexpr uint32_t kWaterShadowMapSize = 512u;
  static constexpr uint32_t kComputeWorkgroupSize = 256u;
  static constexpr float kNoSurfaceDepth = 1.0e4f;

  void init(const RendererConfig &config) {
    numParticles_ = config.particleCount;
    maxWhitewaterParticles_ = config.whitewaterCapacity;
    gridCellCount_ = config.gridCellCount;
    initWindow(config.width, config.height, config.title);
    createInstance(config.title);
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapchain();
    createImageViews();
    createRenderPass();
    createDepthResources();
    createSceneTargets();
    createFramebuffers();
    createCommandPoolAndBuffers();
    createComputeBuffers();
    createSceneLightingBuffer();
    createSorter();
    createComputeDescriptors();
    createComputePipelines();
    createQuadVertexBuffer();
    createEnvironmentTexture();
    createFluidTargets();
    createFluidPipelines();
    clearWhitewaterState();
    createSyncObjects();
  }

  GLFWwindow *window() const { return window_; }
  void loadSceneMesh(const std::string &path, float scale) {
    std::string resolvedPath = path;
    std::ifstream direct(path);
    if (!direct)
      resolvedPath = std::string(ASSET_DIR) + "/" + path;
    std::vector<SceneVertex> vertices =
        loadObjWithMaterials(resolvedPath, scale);
    if (vertices.empty())
      throw std::runtime_error("OBJ contains no renderable triangles: " + path);
    vkDeviceWaitIdle(device_);
    destroyBuffer(sceneMeshBuffer_);
    VkDeviceSize size = vertices.size() * sizeof(SceneVertex);
    createHostBuffer(size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertices.data(),
                     sceneMeshBuffer_, "scene mesh");
    sceneMeshVertexCount_ = static_cast<uint32_t>(vertices.size());
    std::cout << "Loaded scene mesh " << resolvedPath << " ("
              << sceneMeshVertexCount_ << " vertices)" << std::endl;
  }

  bool shouldClose() const { return glfwWindowShouldClose(window_); }
  void pollEvents() const { glfwPollEvents(); }

  void initImGui() {
    if (imguiInitialized_)
      return;

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui::GetStyle().ScaleAllSizes(0.8f);
    ImGui::GetIO().FontGlobalScale = 0.85f;
    ImGui::GetIO().IniFilename = nullptr;
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
    info.MinImageCount = static_cast<uint32_t>(swapchainImages_.size());
    info.ImageCount = static_cast<uint32_t>(swapchainImages_.size());
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

    vkWaitForFences(device_, 1, &inFlightFences_[currentFrame_], VK_TRUE,
                    UINT64_MAX);
    std::memcpy(&activeWhitewaterParticleCount_,
                whitewaterCountReadbackBuffers_[currentFrame_].mapped,
                sizeof(activeWhitewaterParticleCount_));
    const uint32_t totalParticleCount =
        numParticles_ + activeWhitewaterParticleCount_;

    constexpr float panelMargin = 10.0f;
    constexpr float panelWidth = 270.0f;
    constexpr float panelGap = 10.0f;
    constexpr float debugPanelWidth = 200.0f;
    const float panelColumnHeight = std::max(
        1.0f, ImGui::GetIO().DisplaySize.y - 2.0f * panelMargin - panelGap);
    const float simulationPanelHeight = panelColumnHeight * 0.45f;
    ImGui::SetNextWindowPos(ImVec2(panelMargin, panelMargin), ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2(panelWidth, simulationPanelHeight),
                             ImGuiCond_Once);
    ImGui::SetNextWindowBgAlpha(0.6f);
    ImGui::Begin("Simulation controls");
    ImGui::PushItemWidth(100.0f);
    ImGui::Text("Particles: %u \nTotal (%u fluid + %u whitewater)",
                totalParticleCount, numParticles_,
                activeWhitewaterParticleCount_);
    ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
    if (ImGui::Button("Restart simulation"))
      restartRequested_ = true;
    {
      SphParams &physics = liveSphParams_;
      ImGui::Separator();
      ImGui::TextUnformatted("Fluid physics");
      ImGui::SliderFloat("Gravity", &physics.gravity[1], -20.0f, 0.0f, "%.2f");
      ImGui::SliderFloat("Target density", &physics.targetDensity, 500.0f,
                         4000.0f, "%.0f");
      ImGui::SliderFloat("Pressure stiffness", &physics.pressureMultiplier,
                         1.0f, 500.0f, "%.1f", ImGuiSliderFlags_Logarithmic);
      ImGui::SliderFloat("Near pressure", &physics.gravity[3], 0.0f, 2.0f,
                         "%.3f", ImGuiSliderFlags_Logarithmic);
      ImGui::SliderFloat("Viscosity", &physics.viscosityStrength, 0.0f, 0.02f,
                         "%.4f");
      ImGui::SliderFloat("Collision damping", &physics.collisionDamping, 0.0f,
                         1.0f, "%.2f");
      ImGui::SliderInt("Simulation substeps", &simulationSubsteps_, 1, 8);
      if (ImGui::Button("Reset water preset")) {
        physics = defaultSphParams_;
        simulationSubsteps_ = 1;
      }
    }
    ImGui::Separator();
    ImGui::TextUnformatted("Whitewater");
    ImGui::Checkbox("Enable whitewater", &whitewaterSettings_.enabled);
    ImGui::SliderFloat("Spawn rate", &whitewaterSettings_.spawnRate, 0.0f,
                       300.0f, "%.1f");
    ImGui::SliderFloat("Whitewater size", &whitewaterSettings_.renderScale,
                       0.05f, 0.75f, "%.2fx");
    if (ImGui::Button("Reset whitewater preset")) {
      whitewaterSettings_ = WhitewaterSettings{};
      whitewaterSimulationTime_ = 0.0f;
    }
    ImGui::PopItemWidth();
    ImGui::End();

    ImGui::SetNextWindowPos(
        ImVec2(panelMargin, panelMargin + simulationPanelHeight + panelGap),
        ImGuiCond_Once);
    ImGui::SetNextWindowSize(
        ImVec2(panelWidth, panelColumnHeight - simulationPanelHeight),
        ImGuiCond_Once);
    ImGui::SetNextWindowBgAlpha(0.78f);
    ImGui::Begin("Rendering controls");
    ImGui::PushItemWidth(100.0f);
    ImGui::TextUnformatted("Screen-space reconstruction");
    ImGui::SliderFloat("Render scale", &fluidSettings_.renderScale, 0.5f, 1.0f,
                       "%.2fx");
    ImGui::SliderFloat("Max blur radius (px)", &fluidSettings_.maxBlurRadius,
                       1.0f, 32.0f, "%.1f");
    ImGui::SliderFloat("Blur strength", &fluidSettings_.blurStrength, 0.05f,
                       2.0f, "%.2f");
    ImGui::SliderFloat("Depth difference strength",
                       &fluidSettings_.depthDifferenceStrength, 0.1f, 250.0f,
                       "%.1f", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderInt("Blur iterations", &fluidSettings_.blurIterations, 1, 6);
    ImGui::Separator();
    ImGui::TextUnformatted("Water material");
    ImGui::SliderFloat("Refraction strength",
                       &fluidSettings_.refractionStrength, 0.0f, 0.30f, "%.3f");
    ImGui::SliderFloat("Absorption scale", &fluidSettings_.absorptionScale,
                       0.0f, 12.0f, "%.2f");
    ImGui::ColorEdit3("Extinction color", &fluidSettings_.extinctionRed,
                      ImGuiColorEditFlags_Float);
    ImGui::SliderFloat("Base reflectance", &fluidSettings_.baseReflectance,
                       0.0f, 0.15f, "%.3f");
    ImGui::Separator();
    ImGui::TextUnformatted("Sun / shadows");
    ImGui::SliderFloat("Sun angle", &fluidSettings_.sunAngleDegrees,
                       -180.0f, 180.0f, "%.0f deg");
    ImGui::SliderFloat("Sun elevation", &fluidSettings_.sunElevationDegrees,
                       10.0f, 85.0f, "%.0f deg");
    ImGui::SliderFloat("Shadow ambient light",
                       &fluidSettings_.shadowAmbientLight, 0.0f, 0.75f, "%.2f");
    ImGui::SliderInt("Shadow update interval",
                     &fluidSettings_.shadowUpdateInterval, 1, 4, "%d frames");
    ImGui::PopItemWidth();
    ImGui::End();

    const float debugX =
        std::max(panelMargin, ImGui::GetIO().DisplaySize.x - debugPanelWidth -
                                  panelMargin);
    ImGui::SetNextWindowPos(ImVec2(debugX, panelMargin), ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2(debugPanelWidth, 100.0f), ImGuiCond_Once);
    ImGui::SetNextWindowBgAlpha(0.78f);
    ImGui::Begin("Debug visualization");
    constexpr const char *debugModes[] = {
        "Off", "Whitewater", "Particle velocity",
        "Particle pressure"};
    ImGui::Combo("View", &debugMode_, debugModes,
                 static_cast<int>(std::size(debugModes)));
    if (debugMode_ == kDebugWhitewaterClassification) {
      ImGui::TextColored(ImVec4(1, 0, 0, 1), "Red: spray");
      ImGui::TextColored(ImVec4(0, 1, 0, 1), "Green: foam");
      ImGui::TextColored(ImVec4(0, 0.45f, 1, 1), "Blue: bubbles");
    } else if (debugMode_ == kDebugFluidVelocity) {
      ImGui::TextColored(ImVec4(1, 0, 0, 1), "Red: fast");
      ImGui::TextColored(ImVec4(0, 1, 0, 1), "Green: medium");
      ImGui::TextColored(ImVec4(0, 0.45f, 1, 1), "Blue: still");
    } else if (debugMode_ == kDebugFluidPressure) {
      ImGui::TextColored(ImVec4(0, 0.45f, 1, 1), "Blue: negative");
      ImGui::TextColored(ImVec4(1,1,1,1), "White: near zero");
      ImGui::TextColored(ImVec4(1,0,0,1), "Red: positive");

    } else {
      ImGui::TextUnformatted("Normal scene rendering.");
    }
    ImGui::End();
  }

  void setParams(const SphParams &params) {
    baseDeltaT_ = params.deltaT;
    defaultSphParams_ = params;
    liveSphParams_ = params;
    for (auto &buffer : paramsBuffers_)
      std::memcpy(buffer.mapped, &params, sizeof(SphParams));
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
    const std::size_t expectedCount =
        static_cast<std::size_t>(numParticles_);
    if (positions.size() != expectedCount ||
        velocities.size() != expectedCount) {
      throw std::runtime_error(
          "Initial position and velocity counts must match the renderer "
          "particle count");
    }

    const VkDeviceSize uploadSize =
        static_cast<VkDeviceSize>(numParticles_) * sizeof(float4);
    initialPositions_ = positions;
    initialVelocities_ = velocities;
    uploadViaStaging(positionsBuf_, positions.data(), uploadSize);
    uploadViaStaging(velocitiesBuf_, velocities.data(), uploadSize);
  }

  void drawFrame(const float4x4 &view, const float4x4 &proj,
                 const float3 &camRight, const float3 &camUp,
                 const float3 &camForward, float radius, int physicsSteps) {
    vkWaitForFences(device_, 1, &inFlightFences_[currentFrame_], VK_TRUE,
                    UINT64_MAX);
    if (restartRequested_) {
      restartSimulation();
      restartRequested_ = false;
    }
    if (std::abs(fluidSettings_.renderScale - activeRenderScale_) > 0.001f) {
      vkDeviceWaitIdle(device_);
      destroyFluidSizedTargets();
      createFluidTargets();
      updateFluidDescriptors();
    }

    const int simulationSubsteps = simulationSubsteps_;
    const int simulationSteps = physicsSteps * simulationSubsteps;
    SphParams frameParams = liveSphParams_;
    frameParams.deltaT = baseDeltaT_ / static_cast<float>(simulationSubsteps);
    std::memcpy(paramsBuffers_[currentFrame_].mapped, &frameParams,
                sizeof(frameParams));
    std::memcpy(interactionBuffers_[currentFrame_].mapped, &pendingInteraction_,
                sizeof(InteractionParams));
    whitewaterSimulationTime_ +=
        frameParams.deltaT * static_cast<float>(simulationSteps);
    const float fadeT =
        std::clamp((whitewaterSimulationTime_ - 0.2f) / 0.35f, 0.0f, 1.0f);
    WhitewaterParams whitewater{};
    whitewater.generationRate =
        whitewaterSettings_.spawnRate * fadeT * fadeT;
    whitewater.trappedAirMin = whitewaterSettings_.trappedAirMin;
    whitewater.trappedAirMax = whitewaterSettings_.trappedAirMax;
    whitewater.kineticEnergyMin = whitewaterSettings_.kineticEnergyMin;
    whitewater.kineticEnergyMax = whitewaterSettings_.kineticEnergyMax;
    whitewater.bubbleBuoyancy = whitewaterSettings_.bubbleBuoyancy;
    whitewater.sprayDrag = whitewaterSettings_.sprayDrag;
    whitewater.fluidFollow = whitewaterSettings_.fluidFollow;
    whitewater.lifetimeMin = whitewaterSettings_.lifetimeMin;
    whitewater.lifetimeMax = whitewaterSettings_.lifetimeMax;
    whitewater.bubbleScale = whitewaterSettings_.bubbleScale;
    whitewater.renderScale = whitewaterSettings_.renderScale;
    whitewater.capacity = maxWhitewaterParticles_;
    whitewater.sprayMaxNeighbours =
        static_cast<uint32_t>(whitewaterSettings_.sprayMaxNeighbours);
    whitewater.bubbleMinNeighbours =
        static_cast<uint32_t>(whitewaterSettings_.bubbleMinNeighbours);
    whitewater.enabled = whitewaterSettings_.enabled ? 1u : 0u;
    whitewater.debugClassification =
        debugMode_ == kDebugWhitewaterClassification ? 1u : 0u;
    whitewater.activeCount = activeWhitewaterParticleCount_;
    std::memcpy(whitewaterParamsBuffers_[currentFrame_].mapped, &whitewater,
                sizeof(whitewater));

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
    wp.camForward = float4(camForward, -1.0f / proj[1].y);
    wp.material =
        float4(fluidSettings_.refractionStrength,
               fluidSettings_.absorptionScale,
               fluidSettings_.baseReflectance,
               static_cast<float>(debugMode_));
    wp.extinction =
        float4(fluidSettings_.extinctionRed, fluidSettings_.extinctionGreen,
               fluidSettings_.extinctionBlue, 0.0f);

    if (imguiInitialized_)
      ImGui::Render();

    VkCommandBuffer cmd = commandBuffers_[currentFrame_];
    vkResetCommandBuffer(cmd, 0);
    recordCommandBuffer(cmd, imageIndex, dp, wp, simulationSteps);
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
    if (whitewaterPipeline_)
      vkDestroyPipeline(device_, whitewaterPipeline_, nullptr);
    if (blurPipelineLayout_)
      vkDestroyPipelineLayout(device_, blurPipelineLayout_, nullptr);
    if (blurPass_)
      vkDestroyRenderPass(device_, blurPass_, nullptr);
    if (blurLoadPass_)
      vkDestroyRenderPass(device_, blurLoadPass_, nullptr);
    if (compositePool_)
      vkDestroyDescriptorPool(device_, compositePool_, nullptr);
    if (compositeSetLayout_)
      vkDestroyDescriptorSetLayout(device_, compositeSetLayout_, nullptr);
    if (depthPipeline_)
      vkDestroyPipeline(device_, depthPipeline_, nullptr);
    if (fluidDebugPipeline_)
      vkDestroyPipeline(device_, fluidDebugPipeline_, nullptr);
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
    destroyImage(environmentImage_, environmentMem_, environmentView_);
    if (scenePass_)
      vkDestroyRenderPass(device_, scenePass_, nullptr);

    if (renderPass_)
      vkDestroyRenderPass(device_, renderPass_, nullptr);

    if (descriptorPool_)
      vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
    if (descriptorSetLayout_)
      vkDestroyDescriptorSetLayout(device_, descriptorSetLayout_, nullptr);

    if (sorter_) {
      vrdxDestroySorter(sorter_);
      sorter_ = VK_NULL_HANDLE;
    }
    destroyBuffers({&positionsBuf_, &velocitiesBuf_, &predictedBuf_,
                    &densitiesBuf_, &keysBuf_, &indicesBuf_, &startBuf_,
                    &endBuf_, &sortedPosBuf_, &sortedVelBuf_, &sortedPredBuf_,
                    &sortStorageBuf_, &whitewaterParticlesBuf_,
                    &whitewaterAllocatorBuf_, &quadBuffer_, &sceneMeshBuffer_});
    for (int i = 0; i < kMaxFramesInFlight; ++i) {
      destroyBuffers({&paramsBuffers_[i], &interactionBuffers_[i],
                      &whitewaterParamsBuffers_[i],
                      &whitewaterCountReadbackBuffers_[i],
                      &sceneLightingBuffers_[i]});
    }

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

private:
  void destroyBuffer(AllocatedBuffer &buffer) {
    if (buffer.mapped) {
      vkUnmapMemory(device_, buffer.memory);
      buffer.mapped = nullptr;
    }
    if (buffer.handle) {
      vkDestroyBuffer(device_, buffer.handle, nullptr);
      buffer.handle = VK_NULL_HANDLE;
    }
    if (buffer.memory) {
      vkFreeMemory(device_, buffer.memory, nullptr);
      buffer.memory = VK_NULL_HANDLE;
    }
  }

  void destroyBuffers(std::initializer_list<AllocatedBuffer *> buffers) {
    for (AllocatedBuffer *buffer : buffers)
      destroyBuffer(*buffer);
  }

  void uploadViaStaging(VkBuffer dst, const void *src, VkDeviceSize size) {
    if (size == 0)
      return;
    AllocatedBuffer staging;
    createHostBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, src, staging,
                     "staging upload");
    submitOneTimeCommands(
        commandPool_,
        [&](VkCommandBuffer commandBuffer) {
          VkBufferCopy region{0, 0, size};
          vkCmdCopyBuffer(commandBuffer, staging, dst, 1, &region);
        },
        "staging buffer upload");
    destroyBuffer(staging);
  }

  void restartSimulation() {
    const VkDeviceSize uploadSize =
        static_cast<VkDeviceSize>(numParticles_) * sizeof(float4);
    uploadViaStaging(positionsBuf_, initialPositions_.data(), uploadSize);
    uploadViaStaging(velocitiesBuf_, initialVelocities_.data(), uploadSize);
    clearWhitewaterState();

    for (auto &buffer : whitewaterCountReadbackBuffers_)
      std::memset(buffer.mapped, 0, sizeof(uint32_t));
    whitewaterSimulationTime_ = 0.0f;
    activeWhitewaterParticleCount_ = 0;
    renderedFrameCount_ = 0;
    waterShadowValid_ = false;
  }

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
  VkFormat fluidDepthFormat_ = VK_FORMAT_R32_SFLOAT;
  VkExtent2D fluidExtent_{};
  float activeRenderScale_ = 0.0f;
  uint64_t renderedFrameCount_ = 0;
  bool waterShadowValid_ = false;
  float shadowSunAngle_ = 1.0e9f;
  float shadowSunElevation_ = 1.0e9f;
  VkFormat sceneColorFormat_ = VK_FORMAT_UNDEFINED;
  VkFormat sceneDepthFormat_ = VK_FORMAT_R32_SFLOAT;
  VkImage fluidDepthImage_ = VK_NULL_HANDLE;
  VkDeviceMemory fluidDepthMem_ = VK_NULL_HANDLE;
  VkImageView fluidDepthView_ = VK_NULL_HANDLE;
  VkImage fluidZImage_ = VK_NULL_HANDLE;
  VkDeviceMemory fluidZMem_ = VK_NULL_HANDLE;
  VkImageView fluidZView_ = VK_NULL_HANDLE;
  VkImage whitewaterDepthImage_ = VK_NULL_HANDLE;
  VkDeviceMemory whitewaterDepthMem_ = VK_NULL_HANDLE;
  VkImageView whitewaterDepthView_ = VK_NULL_HANDLE;
  VkImage whitewaterZImage_ = VK_NULL_HANDLE;
  VkDeviceMemory whitewaterZMem_ = VK_NULL_HANDLE;
  VkImageView whitewaterZView_ = VK_NULL_HANDLE;
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
  VkFramebuffer whitewaterDepthFB_ = VK_NULL_HANDLE;
  VkSampler fluidSampler_ = VK_NULL_HANDLE;
  VkSampler sceneSampler_ = VK_NULL_HANDLE;
  VkImage environmentImage_ = VK_NULL_HANDLE;
  VkDeviceMemory environmentMem_ = VK_NULL_HANDLE;
  VkImageView environmentView_ = VK_NULL_HANDLE;
  VkSampler envSampler_ = VK_NULL_HANDLE;
  VkPipelineLayout depthPipelineLayout_ = VK_NULL_HANDLE;
  VkPipeline depthPipeline_ = VK_NULL_HANDLE;
  VkPipeline fluidDebugPipeline_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout compositeSetLayout_ = VK_NULL_HANDLE;
  VkDescriptorPool compositePool_ = VK_NULL_HANDLE;
  std::array<VkDescriptorSet, kMaxFramesInFlight> compositeSets_{};
  VkPipelineLayout compositePipelineLayout_ = VK_NULL_HANDLE;
  VkPipeline backgroundPipeline_ = VK_NULL_HANDLE;
  VkPipelineLayout sceneMeshPipelineLayout_ = VK_NULL_HANDLE;
  VkPipeline sceneMeshPipeline_ = VK_NULL_HANDLE;
  VkPipeline compositePipeline_ = VK_NULL_HANDLE;

  FluidRenderSettings fluidSettings_{};
  VkDescriptorPool imguiDescriptorPool_ = VK_NULL_HANDLE;
  bool imguiInitialized_ = false;

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
  VkRenderPass blurLoadPass_ = VK_NULL_HANDLE;
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
  VkPipeline whitewaterPipeline_ = VK_NULL_HANDLE;
  VkDescriptorSet blurSetH_ = VK_NULL_HANDLE;
  VkDescriptorSet blurSetV_ = VK_NULL_HANDLE;
  VkDescriptorSet blurSetSmooth_ = VK_NULL_HANDLE;
  VkDescriptorSet thicknessBlurSet_ = VK_NULL_HANDLE;
  VkDescriptorSet thicknessBlurSetV_ = VK_NULL_HANDLE;

  AllocatedBuffer quadBuffer_;
  AllocatedBuffer sceneMeshBuffer_;
  uint32_t sceneMeshVertexCount_ = 0;

  uint32_t numParticles_ = 0;
  uint32_t maxWhitewaterParticles_ = 0;
  uint32_t gridCellCount_ = 0;
  float baseDeltaT_ = 0.002f;

  AllocatedBuffer positionsBuf_;
  AllocatedBuffer velocitiesBuf_;
  std::vector<float4> initialPositions_;
  std::vector<float4> initialVelocities_;
  bool restartRequested_ = false;
  AllocatedBuffer predictedBuf_;
  AllocatedBuffer densitiesBuf_;
  std::array<AllocatedBuffer, kMaxFramesInFlight> paramsBuffers_;
  SphParams liveSphParams_{};
  SphParams defaultSphParams_{};
  std::array<AllocatedBuffer, kMaxFramesInFlight> interactionBuffers_;
  InteractionParams pendingInteraction_{};
  AllocatedBuffer whitewaterParticlesBuf_;
  AllocatedBuffer whitewaterAllocatorBuf_;
  std::array<AllocatedBuffer, kMaxFramesInFlight> whitewaterParamsBuffers_;
  std::array<AllocatedBuffer, kMaxFramesInFlight>
      whitewaterCountReadbackBuffers_;
  WhitewaterSettings whitewaterSettings_{};
  int debugMode_ = kDebugNone;
  int simulationSubsteps_ = 1;
  float whitewaterSimulationTime_ = 0.0f;
  uint32_t activeWhitewaterParticleCount_ = 0;
  std::array<AllocatedBuffer, kMaxFramesInFlight> sceneLightingBuffers_;
  AllocatedBuffer keysBuf_;
  AllocatedBuffer indicesBuf_;
  AllocatedBuffer startBuf_;
  AllocatedBuffer endBuf_;

  AllocatedBuffer sortedPosBuf_;
  AllocatedBuffer sortedVelBuf_;
  AllocatedBuffer sortedPredBuf_;

  VrdxSorter sorter_ = VK_NULL_HANDLE;
  AllocatedBuffer sortStorageBuf_;

  VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
  VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
  std::array<VkDescriptorSet, kMaxFramesInFlight> descriptorSets_{};

  VkPipelineLayout computePipelineLayout_ = VK_NULL_HANDLE;

  std::array<VkPipeline, 8> computePipelines_{};

  VkCommandPool commandPool_ = VK_NULL_HANDLE;
  std::array<VkCommandBuffer, kMaxFramesInFlight> commandBuffers_{};
  std::array<VkSemaphore, kMaxFramesInFlight> imageAvailableSemaphores_{};
  std::vector<VkSemaphore> renderFinishedSemaphores_;
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

#if defined(__APPLE__)
    extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    extensions.push_back(
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
#endif

    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
#if defined(__APPLE__)
    ci.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif
    ci.pApplicationInfo = &app;
    ci.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
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

  static bool supportsComputeWorkgroupSize(VkPhysicalDevice device) {
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(device, &properties);
    return properties.limits.maxComputeWorkGroupInvocations >=
               kComputeWorkgroupSize &&
           properties.limits.maxComputeWorkGroupSize[0] >=
               kComputeWorkgroupSize;
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
      if (q.complete() && supportsComputeWorkgroupSize(dev)) {
        physicalDevice_ = dev;
        graphicsFamily_ = *q.graphics;
        presentFamily_ = *q.present;
        break;
      }
    }
    if (physicalDevice_ == VK_NULL_HANDLE)
      throw std::runtime_error(
          "No suitable GPU found with support for " +
          std::to_string(kComputeWorkgroupSize) +
          "-thread compute workgroups");
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
#if defined(__APPLE__)
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
    ci.queueCreateInfoCount = static_cast<uint32_t>(queueInfos.size());
    ci.pQueueCreateInfos = queueInfos.data();
    ci.pEnabledFeatures = nullptr;
    ci.enabledExtensionCount = static_cast<uint32_t>(deviceExts.size());
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
      if (m == VK_PRESENT_MODE_IMMEDIATE_KHR)
        return m;
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
    VkExtent2D e{static_cast<uint32_t>(w), static_cast<uint32_t>(h)};
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
    for (std::size_t i = 0; i < swapchainImages_.size(); i++) {
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
    ci.attachmentCount = static_cast<uint32_t>(attachments.size());
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
    createImage2D(
        sceneColorFormat_,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT, sceneColorImage_, sceneColorMem_,
        sceneColorView_, swapchainExtent_.width, swapchainExtent_.height);
    createImage2D(
        sceneDepthFormat_,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT, sceneDepthImage_, sceneDepthMem_,
        sceneDepthView_, swapchainExtent_.width, swapchainExtent_.height);
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
      VkAttachmentReference colorRef{0,
                                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
      VkAttachmentReference depthRef{1,
                                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
      VkAttachmentReference sceneZRef{
          2, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
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
      rp.attachmentCount = static_cast<uint32_t>(atts.size());
      rp.pAttachments = atts.data();
      rp.subpassCount = 1;
      rp.pSubpasses = &subpass;
      rp.dependencyCount = 1;
      rp.pDependencies = &dep;
      vkCheck(vkCreateRenderPass(device_, &rp, nullptr, &scenePass_),
              "vkCreateRenderPass(scene)");
    }

    createFramebuffer(scenePass_,
                      {sceneColorView_, sceneDepthView_, sceneZView_},
                      swapchainExtent_, sceneFB_, "create scene framebuffer");
  }

  void createEnvironmentTexture() {
    const std::string path =
        std::string(ASSET_DIR) + "/media/uffizi_probe.hdr";
    int width = 0;
    int height = 0;
    int channels = 0;
    float *pixels = stbi_loadf(path.c_str(), &width, &height, &channels, 4);
    if (!pixels)
      throw std::runtime_error("Failed to load environment map: " + path);

    const VkDeviceSize imageSize = static_cast<VkDeviceSize>(width) * height *
                                   4u * sizeof(float);
    AllocatedBuffer staging;
    createHostBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, pixels,
                     staging, "environment staging buffer");
    stbi_image_free(pixels);

    createImage2D(VK_FORMAT_R32G32B32A32_SFLOAT,
                  VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT, environmentImage_,
                  environmentMem_, environmentView_,
                  static_cast<uint32_t>(width),
                  static_cast<uint32_t>(height));

    submitOneTimeCommands(
        commandPool_,
        [&](VkCommandBuffer cmd) {
          VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
          barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
          barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
          barrier.image = environmentImage_;
          barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
          barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
          barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
          barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
          vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                               VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                               nullptr, 1, &barrier);

          VkBufferImageCopy region{};
          region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
          region.imageSubresource.layerCount = 1;
          region.imageExtent = {static_cast<uint32_t>(width),
                                static_cast<uint32_t>(height), 1};
          vkCmdCopyBufferToImage(cmd, staging, environmentImage_,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                                 &region);

          barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
          barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
          barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
          barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
          vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                               nullptr, 0, nullptr, 1, &barrier);
        },
        "upload environment texture");
    destroyBuffer(staging);

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

  void createFramebuffer(VkRenderPass renderPass,
                         std::initializer_list<VkImageView> attachments,
                         VkExtent2D extent, VkFramebuffer &framebuffer,
                         const char *label) {
    VkFramebufferCreateInfo info{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    info.renderPass = renderPass;
    info.attachmentCount = static_cast<uint32_t>(attachments.size());
    info.pAttachments = attachments.begin();
    info.width = extent.width;
    info.height = extent.height;
    info.layers = 1;
    vkCheck(vkCreateFramebuffer(device_, &info, nullptr, &framebuffer), label);
  }

  void createFramebuffers() {
    framebuffers_.resize(swapchainImageViews_.size());
    for (std::size_t i = 0; i < swapchainImageViews_.size(); i++) {
      createFramebuffer(renderPass_, {swapchainImageViews_[i], depthView_},
                        swapchainExtent_, framebuffers_[i],
                        "create swapchain framebuffer");
    }
  }

  static std::vector<char> readFile(const std::string &path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open())
      throw std::runtime_error("Failed to open shader file: " + path);
    const std::size_t size =
        static_cast<std::size_t>(file.tellg());
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

  void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags props, AllocatedBuffer &buffer) {
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size = size;
    ci.usage = usage;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCheck(vkCreateBuffer(device_, &ci, nullptr, &buffer.handle),
            "vkCreateBuffer");
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device_, buffer.handle, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, props);
    vkCheck(vkAllocateMemory(device_, &ai, nullptr, &buffer.memory),
            "vkAllocateMemory");
    vkCheck(vkBindBufferMemory(device_, buffer.handle, buffer.memory, 0),
            "vkBindBufferMemory");
  }

  void createDeviceBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                          AllocatedBuffer &buffer) {
    createBuffer(size, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer);
  }

  void createStorageBuffer(VkDeviceSize size, AllocatedBuffer &buffer,
                           bool allowStagingUpload = false) {
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if (allowStagingUpload)
      usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    createDeviceBuffer(size, usage, buffer);
  }

  void createHostBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                        const void *initialData, AllocatedBuffer &buffer,
                        const char *label) {
    createBuffer(size, usage,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 buffer);
    if (!initialData)
      return;

    void *mapped = nullptr;
    vkCheck(vkMapMemory(device_, buffer.memory, 0, size, 0, &mapped), label);
    std::memcpy(mapped, initialData, static_cast<std::size_t>(size));
    vkUnmapMemory(device_, buffer.memory);
  }

  template <typename T>
  void createMappedUniformBuffer(AllocatedBuffer &buffer, const char *label) {
    createHostBuffer(sizeof(T), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, nullptr,
                     buffer, label);
    vkCheck(
        vkMapMemory(device_, buffer.memory, 0, sizeof(T), 0, &buffer.mapped),
        label);
  }

  void createPipelineLayout(VkDescriptorSetLayout descriptorLayout,
                            VkPipelineLayout &pipelineLayout, const char *label,
                            VkShaderStageFlags pushConstantStages = 0,
                            uint32_t pushConstantSize = 0) {
    VkPushConstantRange pushConstants{};
    pushConstants.stageFlags = pushConstantStages;
    pushConstants.size = pushConstantSize;

    VkPipelineLayoutCreateInfo info{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    info.setLayoutCount = 1;
    info.pSetLayouts = &descriptorLayout;
    if (pushConstantSize > 0) {
      info.pushConstantRangeCount = 1;
      info.pPushConstantRanges = &pushConstants;
    }
    vkCheck(vkCreatePipelineLayout(device_, &info, nullptr, &pipelineLayout),
            label);
  }

  template <typename Recorder>
  void submitOneTimeCommands(VkCommandPool commandPool, Recorder record,
                             const char *label) {
    VkCommandBufferAllocateInfo allocateInfo{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocateInfo.commandPool = commandPool;
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocateInfo.commandBufferCount = 1;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    vkCheck(vkAllocateCommandBuffers(device_, &allocateInfo, &commandBuffer),
            label);

    VkCommandBufferBeginInfo beginInfo{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkCheck(vkBeginCommandBuffer(commandBuffer, &beginInfo), label);
    record(commandBuffer);
    vkCheck(vkEndCommandBuffer(commandBuffer), label);

    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    vkCheck(vkQueueSubmit(graphicsQueue_, 1, &submitInfo, VK_NULL_HANDLE),
            label);
    vkCheck(vkQueueWaitIdle(graphicsQueue_), label);
    vkFreeCommandBuffers(device_, commandPool, 1, &commandBuffer);
  }

  void createQuadVertexBuffer() {
    const float quad[] = {
        -1.f, -1.f, 1.f, -1.f, -1.f, 1.f, 1.f, 1.f,
    };
    VkDeviceSize size = sizeof(quad);
    createHostBuffer(size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, quad, quadBuffer_,
                     "quad vertex buffer");
  }

  void createComputeBuffers() {
    const VkDeviceSize vec4Size = sizeof(float) * 4 * numParticles_;
    const VkDeviceSize densitySize = sizeof(float) * 2 * numParticles_;
    const VkDeviceSize uintSize = sizeof(uint32_t) * numParticles_;
    const VkDeviceSize cellSize = sizeof(uint32_t) * gridCellCount_;

    createStorageBuffer(vec4Size, positionsBuf_, true);
    createStorageBuffer(vec4Size, velocitiesBuf_, true);
    createStorageBuffer(vec4Size, predictedBuf_);
    createStorageBuffer(densitySize, densitiesBuf_);
    createStorageBuffer(uintSize, keysBuf_);
    createStorageBuffer(uintSize, indicesBuf_);
    createStorageBuffer(cellSize, startBuf_);
    createStorageBuffer(cellSize, endBuf_);
    createStorageBuffer(vec4Size, sortedPosBuf_);
    createStorageBuffer(vec4Size, sortedVelBuf_);
    createStorageBuffer(vec4Size, sortedPredBuf_);
    createStorageBuffer(sizeof(WhiteParticle) * maxWhitewaterParticles_,
                        whitewaterParticlesBuf_, true);
    const VkDeviceSize allocatorSize =
        sizeof(WhitewaterAllocatorHeader) +
        sizeof(uint32_t) * VkDeviceSize(maxWhitewaterParticles_);
    createDeviceBuffer(allocatorSize,
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                       whitewaterAllocatorBuf_);

    for (int i = 0; i < kMaxFramesInFlight; ++i) {
      createMappedUniformBuffer<SphParams>(paramsBuffers_[i],
                                           "SPH params buffer");
      createMappedUniformBuffer<InteractionParams>(interactionBuffers_[i],
                                                   "interaction params buffer");
      createMappedUniformBuffer<WhitewaterParams>(
          whitewaterParamsBuffers_[i], "whitewater params buffer");
      createHostBuffer(sizeof(uint32_t), VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                       nullptr, whitewaterCountReadbackBuffers_[i],
                       "whitewater count readback buffer");
      vkCheck(vkMapMemory(device_,
                          whitewaterCountReadbackBuffers_[i].memory, 0,
                          sizeof(uint32_t), 0,
                          &whitewaterCountReadbackBuffers_[i].mapped),
              "map whitewater count readback buffer");
      std::memset(whitewaterCountReadbackBuffers_[i].mapped, 0,
                  sizeof(uint32_t));
    }
  }

  void clearWhitewaterState() {
    submitOneTimeCommands(
        commandPool_,
        [&](VkCommandBuffer commandBuffer) {
          vkCmdFillBuffer(commandBuffer, whitewaterParticlesBuf_, 0,
                          VK_WHOLE_SIZE, 0u);
          vkCmdFillBuffer(commandBuffer, whitewaterAllocatorBuf_, 0,
                          VK_WHOLE_SIZE, 0u);
        },
        "clear whitewater buffers");
  }

  void createSceneLightingBuffer() {
    for (auto &buffer : sceneLightingBuffers_)
      createMappedUniformBuffer<SceneLightingParams>(buffer,
                                                     "scene lighting buffer");
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
    createDeviceBuffer(req.size, req.usage, sortStorageBuf_);
  }

  struct DescriptorBinding {
    uint32_t binding;
    VkDescriptorType type;
    VkShaderStageFlags stages;
  };

  static DescriptorBinding storageBinding(uint32_t binding,
                                          VkShaderStageFlags stages) {
    return {binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, stages};
  }

  static DescriptorBinding uniformBinding(uint32_t binding,
                                          VkShaderStageFlags stages) {
    return {binding, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, stages};
  }

  static DescriptorBinding samplerBinding(uint32_t binding,
                                          VkShaderStageFlags stages) {
    return {binding, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stages};
  }

  template <std::size_t N>
  std::array<VkDescriptorSet, N>
  createDescriptorSets(std::initializer_list<DescriptorBinding> bindings,
                       VkDescriptorSetLayout &layout, VkDescriptorPool &pool,
                       const char *label) {
    std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
    std::vector<VkDescriptorPoolSize> poolSizes;
    layoutBindings.reserve(bindings.size());
    for (const auto &binding : bindings) {
      layoutBindings.push_back(
          {binding.binding, binding.type, 1, binding.stages, nullptr});
      auto poolSize =
          std::find_if(poolSizes.begin(), poolSizes.end(),
                       [&](const auto &p) { return p.type == binding.type; });
      if (poolSize == poolSizes.end())
        poolSizes.push_back({binding.type, static_cast<uint32_t>(N)});
      else
        poolSize->descriptorCount += static_cast<uint32_t>(N);
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = static_cast<uint32_t>(layoutBindings.size());
    layoutInfo.pBindings = layoutBindings.data();
    vkCheck(vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &layout),
            label);

    VkDescriptorPoolCreateInfo poolInfo{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = static_cast<uint32_t>(N);
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    vkCheck(vkCreateDescriptorPool(device_, &poolInfo, nullptr, &pool), label);

    std::array<VkDescriptorSetLayout, N> layouts{};
    layouts.fill(layout);
    std::array<VkDescriptorSet, N> sets{};
    VkDescriptorSetAllocateInfo allocateInfo{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocateInfo.descriptorPool = pool;
    allocateInfo.descriptorSetCount = static_cast<uint32_t>(N);
    allocateInfo.pSetLayouts = layouts.data();
    vkCheck(vkAllocateDescriptorSets(device_, &allocateInfo, sets.data()),
            label);
    return sets;
  }

  struct BufferDescriptorBinding {
    uint32_t binding;
    VkDescriptorType type;
    VkBuffer buffer;
    VkDeviceSize range = VK_WHOLE_SIZE;
    VkDeviceSize offset = 0;
  };

  static BufferDescriptorBinding storageDescriptor(uint32_t binding,
                                                   VkBuffer buffer) {
    return {binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, buffer};
  }

  static BufferDescriptorBinding
  uniformDescriptor(uint32_t binding, VkBuffer buffer,
                    VkDeviceSize range = VK_WHOLE_SIZE) {
    return {binding, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, buffer, range};
  }

  void writeBufferDescriptors(
      VkDescriptorSet set,
      std::initializer_list<BufferDescriptorBinding> descriptors) {
    std::vector<VkDescriptorBufferInfo> infos(descriptors.size());
    std::vector<VkWriteDescriptorSet> writes(descriptors.size());
    std::size_t i = 0;
    for (const auto &descriptor : descriptors) {
      infos[i] = {descriptor.buffer, descriptor.offset, descriptor.range};
      writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      writes[i].dstSet = set;
      writes[i].dstBinding = descriptor.binding;
      writes[i].descriptorCount = 1;
      writes[i].descriptorType = descriptor.type;
      writes[i].pBufferInfo = &infos[i];
      ++i;
    }
    vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()),
                           writes.data(), 0, nullptr);
  }

  struct ImageDescriptorWrite {
    VkDescriptorSet set;
    uint32_t binding;
    VkImageView view;
    VkSampler sampler;
    VkImageLayout layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  };

  void writeImageDescriptors(
      std::initializer_list<ImageDescriptorWrite> descriptors) {
    std::vector<VkDescriptorImageInfo> infos(descriptors.size());
    std::vector<VkWriteDescriptorSet> writes(descriptors.size());
    std::size_t i = 0;
    for (const auto &descriptor : descriptors) {
      infos[i] = {descriptor.sampler, descriptor.view, descriptor.layout};
      writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      writes[i].dstSet = descriptor.set;
      writes[i].dstBinding = descriptor.binding;
      writes[i].descriptorCount = 1;
      writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      writes[i].pImageInfo = &infos[i];
      ++i;
    }
    vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()),
                           writes.data(), 0, nullptr);
  }

  void createComputeDescriptors() {
    constexpr VkShaderStageFlags compute = VK_SHADER_STAGE_COMPUTE_BIT;
    constexpr VkShaderStageFlags computeAndVertex =
        VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT;
    constexpr VkShaderStageFlags computeVertexAndFragment =
        VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT |
        VK_SHADER_STAGE_FRAGMENT_BIT;
    descriptorSets_ = createDescriptorSets<kMaxFramesInFlight>(
        {storageBinding(0, computeAndVertex),
         storageBinding(1, computeAndVertex), storageBinding(2, compute),
         storageBinding(3, computeAndVertex),
         uniformBinding(5, computeAndVertex),
         storageBinding(6, compute), storageBinding(7, compute),
         storageBinding(8, compute), storageBinding(9, computeAndVertex),
         storageBinding(10, computeAndVertex), storageBinding(11, compute),
         uniformBinding(12, compute), storageBinding(13, compute),
         uniformBinding(14, computeVertexAndFragment),
         storageBinding(15, computeAndVertex), storageBinding(16, compute)},
        descriptorSetLayout_, descriptorPool_,
        "create compute descriptor sets");

    for (int i = 0; i < kMaxFramesInFlight; ++i) {
      writeBufferDescriptors(descriptorSets_[i],
                             {
                                 storageDescriptor(0, positionsBuf_),
                                 storageDescriptor(1, velocitiesBuf_),
                                 storageDescriptor(2, predictedBuf_),
                                 storageDescriptor(3, densitiesBuf_),
                                 uniformDescriptor(5, paramsBuffers_[i]),
                                 storageDescriptor(6, keysBuf_),
                                 storageDescriptor(7, indicesBuf_),
                                 storageDescriptor(8, startBuf_),
                                 storageDescriptor(9, sortedPosBuf_),
                                 storageDescriptor(10, sortedVelBuf_),
                                 storageDescriptor(11, sortedPredBuf_),
                                 uniformDescriptor(12, interactionBuffers_[i]),
                                 storageDescriptor(13, endBuf_),
                                 uniformDescriptor(
                                     14, whitewaterParamsBuffers_[i]),
                                 storageDescriptor(15,
                                                   whitewaterParticlesBuf_),
                                 storageDescriptor(16,
                                                   whitewaterAllocatorBuf_),
                             });
    }
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
    createPipelineLayout(descriptorSetLayout_, computePipelineLayout_,
                         "create compute pipeline layout");

    computePipelines_[0] = createComputePipeline("sph_external.comp.spv");
    computePipelines_[1] = createComputePipeline("sph_reorder.comp.spv");
    computePipelines_[2] = createComputePipeline("sph_start_reset.comp.spv");
    computePipelines_[3] = createComputePipeline("sph_start_indices.comp.spv");
    computePipelines_[4] = createComputePipeline("sph_density.comp.spv");
    computePipelines_[5] = createComputePipeline("sph_pressure.comp.spv");
    computePipelines_[6] =
        createComputePipeline("sph_viscosity_integrate.comp.spv");
    computePipelines_[7] =
        createComputePipeline("whitewater_update.comp.spv");
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

  void createImGuiDescriptorPool() {
    VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 32};
    VkDescriptorPoolCreateInfo info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    info.maxSets = 32;
    info.poolSizeCount = 1;
    info.pPoolSizes = &size;
    vkCheck(vkCreateDescriptorPool(device_, &info, nullptr,
                                   &imguiDescriptorPool_),
            "vkCreateDescriptorPool(imgui)");
  }

  VkRenderPass createColorOnlyPass(VkAttachmentLoadOp loadOp) {
    VkAttachmentDescription color{};
    color.format = fluidDepthFormat_;
    color.samples = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp = loadOp;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;
    VkSubpassDependency dep{};
    dep.srcSubpass = 0;
    dep.dstSubpass = VK_SUBPASS_EXTERNAL;
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dep.dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dep.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    VkRenderPassCreateInfo info{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    info.attachmentCount = 1;
    info.pAttachments = &color;
    info.subpassCount = 1;
    info.pSubpasses = &subpass;
    info.dependencyCount = 1;
    info.pDependencies = &dep;
    VkRenderPass pass = VK_NULL_HANDLE;
    vkCheck(vkCreateRenderPass(device_, &info, nullptr, &pass),
            "vkCreateRenderPass(color only)");
    return pass;
  }

  void createFluidTargets() {
    activeRenderScale_ = std::clamp(fluidSettings_.renderScale, 0.5f, 1.0f);
    fluidExtent_.width =
        std::max(1u, static_cast<uint32_t>(std::lround(swapchainExtent_.width *
                                                       activeRenderScale_)));
    fluidExtent_.height =
        std::max(1u, static_cast<uint32_t>(std::lround(swapchainExtent_.height *
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
    if (!fluidDepthPass_)
      vkCheck(vkCreateRenderPass(device_, &rp, nullptr, &fluidDepthPass_),
              "vkCreateRenderPass(fluid)");

    createFramebuffer(fluidDepthPass_, {fluidDepthView_, fluidZView_},
                      fluidExtent_, fluidDepthFB_,
                      "create fluid depth framebuffer");
    createImage2D(
        fluidDepthFormat_,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT, whitewaterDepthImage_, whitewaterDepthMem_,
        whitewaterDepthView_, swapchainExtent_.width, swapchainExtent_.height);
    createImage2D(depthFormat_, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                  VK_IMAGE_ASPECT_DEPTH_BIT, whitewaterZImage_,
                  whitewaterZMem_, whitewaterZView_, swapchainExtent_.width,
                  swapchainExtent_.height);
    createFramebuffer(
        fluidDepthPass_, {whitewaterDepthView_, whitewaterZView_},
        swapchainExtent_, whitewaterDepthFB_,
        "create whitewater depth framebuffer");

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

    createImage2D(fluidDepthFormat_,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                      VK_IMAGE_USAGE_SAMPLED_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT, blurTempImage_, blurTempMem_,
                  blurTempView_, fluidExtent_.width, fluidExtent_.height);
    createImage2D(fluidDepthFormat_,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                      VK_IMAGE_USAGE_SAMPLED_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT, blurSmoothImage_, blurSmoothMem_,
                  blurSmoothView_, fluidExtent_.width, fluidExtent_.height);
    createImage2D(
        fluidDepthFormat_,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT, fluidThicknessImage_, fluidThicknessMem_,
        fluidThicknessView_, fluidExtent_.width, fluidExtent_.height);
    createImage2D(fluidDepthFormat_,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                      VK_IMAGE_USAGE_SAMPLED_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT, thicknessBlurTempImage_,
                  thicknessBlurTempMem_, thicknessBlurTempView_,
                  fluidExtent_.width, fluidExtent_.height);
    createImage2D(fluidDepthFormat_,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                      VK_IMAGE_USAGE_SAMPLED_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT, waterShadowImage_, waterShadowMem_,
                  waterShadowView_, kWaterShadowMapSize, kWaterShadowMapSize);
    createImage2D(
        fluidDepthFormat_,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT, waterShadowDepthImage_, waterShadowDepthMem_,
        waterShadowDepthView_, kWaterShadowMapSize, kWaterShadowMapSize);
    createImage2D(depthFormat_, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                  VK_IMAGE_ASPECT_DEPTH_BIT, waterShadowZImage_,
                  waterShadowZMem_, waterShadowZView_, kWaterShadowMapSize,
                  kWaterShadowMapSize);

    if (!blurPass_) {
      blurPass_ = createColorOnlyPass(VK_ATTACHMENT_LOAD_OP_CLEAR);
      blurLoadPass_ = createColorOnlyPass(VK_ATTACHMENT_LOAD_OP_DONT_CARE);
    }

    createFramebuffer(blurPass_, {blurTempView_}, fluidExtent_, blurTempFB_,
                      "create horizontal blur framebuffer");
    createFramebuffer(blurPass_, {blurSmoothView_}, fluidExtent_, blurSmoothFB_,
                      "create vertical blur framebuffer");
    createFramebuffer(blurPass_, {fluidThicknessView_}, fluidExtent_,
                      fluidThicknessFB_, "create thickness framebuffer");
    createFramebuffer(blurPass_, {thicknessBlurTempView_}, fluidExtent_,
                      thicknessBlurTempFB_,
                      "create thickness blur framebuffer");
    constexpr VkExtent2D shadowExtent{kWaterShadowMapSize, kWaterShadowMapSize};
    createFramebuffer(blurPass_, {waterShadowView_}, shadowExtent,
                      waterShadowFB_, "create water shadow framebuffer");
    createFramebuffer(fluidDepthPass_,
                      {waterShadowDepthView_, waterShadowZView_}, shadowExtent,
                      waterShadowDepthFB_,
                      "create water shadow depth framebuffer");
  }

  void createFluidPipelines() {
    constexpr VkShaderStageFlags vertexAndFragment =
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    createPipelineLayout(descriptorSetLayout_, depthPipelineLayout_,
                         "create depth pipeline layout", vertexAndFragment,
                         sizeof(DepthPush));

    depthPipeline_ = buildParticlePipeline("fluid_depth.vert.spv",
                                           "fluid_depth.frag.spv",
                                           fluidDepthPass_, true, false);
    fluidDebugPipeline_ = buildParticlePipeline(
        "fluid_debug.vert.spv", "fluid_debug.frag.spv", renderPass_, true,
        false);
    whitewaterPipeline_ = buildParticlePipeline("whitewater.vert.spv",
                                                "whitewater.frag.spv",
                                                fluidDepthPass_, true, false);
    thicknessPipeline_ = buildParticlePipeline("fluid_depth.vert.spv",
                                                "fluid_thickness.frag.spv",
                                                blurPass_, false, true);

    constexpr VkShaderStageFlags fragment = VK_SHADER_STAGE_FRAGMENT_BIT;
    const auto sets = createDescriptorSets<kMaxFramesInFlight + 5>(
        {samplerBinding(0, fragment), samplerBinding(1, fragment),
         samplerBinding(2, fragment), samplerBinding(3, fragment),
         samplerBinding(4, fragment), samplerBinding(5, fragment),
         uniformBinding(6, fragment), samplerBinding(7, fragment),
         samplerBinding(8, fragment)},
        compositeSetLayout_, compositePool_, "create fluid descriptor sets");
    for (int i = 0; i < kMaxFramesInFlight; ++i)
      compositeSets_[i] = sets[i];
    blurSetH_ = sets[kMaxFramesInFlight + 0];
    blurSetV_ = sets[kMaxFramesInFlight + 1];
    blurSetSmooth_ = sets[kMaxFramesInFlight + 2];
    thicknessBlurSet_ = sets[kMaxFramesInFlight + 3];
    thicknessBlurSetV_ = sets[kMaxFramesInFlight + 4];
    updateFluidDescriptors();

    createPipelineLayout(compositeSetLayout_, compositePipelineLayout_,
                         "create composite pipeline layout", vertexAndFragment,
                         sizeof(WaterPush));
    createPipelineLayout(compositeSetLayout_, blurPipelineLayout_,
                         "create blur pipeline layout", fragment,
                         sizeof(BlurPush));
    blurPipeline_ = buildFullscreenPipeline("fluid_blur.frag.spv",
                                            blurLoadPass_, blurPipelineLayout_);
    thicknessBlurPipeline_ = buildFullscreenPipeline(
        "fluid_thickness_blur.frag.spv", blurLoadPass_, blurPipelineLayout_);
    compositePipeline_ = buildFullscreenPipeline(
        "fluid_composite.frag.spv", renderPass_, compositePipelineLayout_);
    backgroundPipeline_ = buildFullscreenPipeline(
        "scene_background.frag.spv", scenePass_, compositePipelineLayout_, 2);

    createPipelineLayout(compositeSetLayout_, sceneMeshPipelineLayout_,
                         "create scene mesh pipeline layout",
                         VK_SHADER_STAGE_VERTEX_BIT, sizeof(ScenePush));
    VkVertexInputBindingDescription sceneBind{0, sizeof(SceneVertex),
                                              VK_VERTEX_INPUT_RATE_VERTEX};
    std::array<VkVertexInputAttributeDescription, 6> sceneAttr = {{
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(SceneVertex, position)},
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(SceneVertex, normal)},
        {2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(SceneVertex, ambient)},
        {3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(SceneVertex, diffuse)},
        {4, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(SceneVertex, specular)},
        {5, 0, VK_FORMAT_R32_SFLOAT, offsetof(SceneVertex, shininess)},
    }};
    sceneMeshPipeline_ = buildGraphicsPipeline(
        "scene_mesh.vert.spv", "scene_mesh.frag.spv", scenePass_,
        sceneMeshPipelineLayout_, &sceneBind, sceneAttr.data(),
        static_cast<uint32_t>(sceneAttr.size()), true, 2);
  }

  VkPipeline buildParticlePipeline(const char *vertexShader,
                                   const char *fragmentShader,
                                   VkRenderPass pass, bool depthTest,
                                   bool additiveBlend) {
    VkVertexInputBindingDescription bind{0, sizeof(float) * 2,
                                         VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription attr{0, 0, VK_FORMAT_R32G32_SFLOAT, 0};
    return buildGraphicsPipeline(vertexShader, fragmentShader, pass,
                                 depthPipelineLayout_, &bind, &attr, 1,
                                 depthTest, 1, additiveBlend,
                                 VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP);
  }

  VkPipeline buildFullscreenPipeline(const char *fragmentShader,
                                     VkRenderPass pass,
                                     VkPipelineLayout layout,
                                     uint32_t colorAttachmentCount = 1) {
    return buildGraphicsPipeline("fullscreen.vert.spv", fragmentShader, pass,
                                 layout, nullptr, nullptr, 0, false,
                                 colorAttachmentCount);
  }

  void updateFluidDescriptors() {
    writeImageDescriptors({
        {blurSetH_, 0, fluidDepthView_, fluidSampler_},
        {blurSetV_, 0, blurTempView_, fluidSampler_},
        {blurSetSmooth_, 0, blurSmoothView_, fluidSampler_},
        {thicknessBlurSet_, 0, fluidThicknessView_, fluidSampler_},
        {thicknessBlurSetV_, 0, thicknessBlurTempView_, fluidSampler_},
    });
    for (int i = 0; i < kMaxFramesInFlight; ++i) {
      writeBufferDescriptors(compositeSets_[i],
                             {
                                 uniformDescriptor(6, sceneLightingBuffers_[i],
                                                   sizeof(SceneLightingParams)),
                             });
      writeImageDescriptors({
          {compositeSets_[i], 0, blurSmoothView_, fluidSampler_},
          {compositeSets_[i], 1, sceneColorView_, sceneSampler_},
          {compositeSets_[i], 2, environmentView_, envSampler_},
          {compositeSets_[i], 3, sceneDepthView_, sceneSampler_},
          {compositeSets_[i], 4, blurTempView_, fluidSampler_},
          {compositeSets_[i], 5, waterShadowView_, sceneSampler_},
          {compositeSets_[i], 7, waterShadowDepthView_, sceneSampler_},
          {compositeSets_[i], 8, whitewaterDepthView_, fluidSampler_},
      });
    }
  }

  VkPipeline buildGraphicsPipeline(
      const char *vertSpv, const char *fragSpv, VkRenderPass pass,
      VkPipelineLayout layout, const VkVertexInputBindingDescription *bind,
      const VkVertexInputAttributeDescription *attr, uint32_t attrCount,
      bool depthTest, uint32_t colorAttachmentCount = 1,
      bool additiveBlend = false,
      VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST) {
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
      attachment.colorWriteMask =
          VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
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
    dyn.dynamicStateCount = static_cast<uint32_t>(dynStates.size());
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

  void ssboBarrier(VkCommandBuffer cmd) {
    VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b, 0,
                         nullptr, 0, nullptr);
  }

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

  static constexpr uint32_t dispatchGroupCount(uint32_t elementCount) {
    return (elementCount + kComputeWorkgroupSize - 1u) /
           kComputeWorkgroupSize;
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

  static VkClearValue clearColor(float r, float g, float b, float a = 1.0f) {
    VkClearValue value{};
    value.color = {{r, g, b, a}};
    return value;
  }

  static VkClearValue clearDepth() {
    VkClearValue value{};
    value.depthStencil = {1.0f, 0};
    return value;
  }

  void beginPass(VkCommandBuffer cmd, VkRenderPass pass,
                 VkFramebuffer framebuffer, VkExtent2D extent,
                 std::initializer_list<VkClearValue> clears = {}) {
    VkRenderPassBeginInfo info{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    info.renderPass = pass;
    info.framebuffer = framebuffer;
    info.renderArea.extent = extent;
    info.clearValueCount = static_cast<uint32_t>(clears.size());
    info.pClearValues = clears.begin();
    vkCmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport{0.0f,
                        0.0f,
                        static_cast<float>(extent.width),
                        static_cast<float>(extent.height),
                        0.0f,
                        1.0f};
    VkRect2D scissor{{0, 0}, extent};
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);
  }

  void bindParticleState(VkCommandBuffer cmd, VkDescriptorSet computeSet,
                         const DepthPush &push) {
    vkCmdPushConstants(cmd, depthPipelineLayout_,
                       VK_SHADER_STAGE_VERTEX_BIT |
                           VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(DepthPush), &push);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            depthPipelineLayout_, 0, 1, &computeSet, 0,
                            nullptr);
  }

  void drawParticleQuads(VkCommandBuffer cmd, uint32_t instanceCount) {
    VkBuffer buffer = quadBuffer_;
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &buffer, &offset);
    vkCmdDraw(cmd, 4, instanceCount, 0, 0);
  }

  void recordSimulation(VkCommandBuffer cmd, VkDescriptorSet computeSet,
                        int physicsSteps) {
    const uint32_t groupsN = dispatchGroupCount(numParticles_);
    const uint32_t groupsGrid = dispatchGroupCount(gridCellCount_);
    const uint32_t groupsWhitewater =
        dispatchGroupCount(maxWhitewaterParticles_);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            computePipelineLayout_, 0, 1, &computeSet, 0,
                            nullptr);
    for (int step = 0; step < physicsSteps; ++step) {
      dispatchStage(cmd, computePipelines_[0], groupsN);
      radixSort(cmd);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              computePipelineLayout_, 0, 1, &computeSet, 0,
                              nullptr);
      dispatchStage(cmd, computePipelines_[1], groupsN);
      dispatchStage(cmd, computePipelines_[2], groupsGrid);
      dispatchStage(cmd, computePipelines_[3], groupsN);
      dispatchStage(cmd, computePipelines_[4], groupsN);
      dispatchStage(cmd, computePipelines_[5], groupsN);
      dispatchStage(cmd, computePipelines_[6], groupsN);
      dispatchStage(cmd, computePipelines_[7], groupsWhitewater,
                    step + 1 < physicsSteps);
    }

    VkBufferMemoryBarrier toTransfer{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    toTransfer.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    toTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransfer.buffer = whitewaterAllocatorBuf_;
    toTransfer.offset = offsetof(WhitewaterAllocatorHeader, activeCount);
    toTransfer.size = sizeof(uint32_t);
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1,
                         &toTransfer, 0, nullptr);

    VkBufferCopy countCopy{offsetof(WhitewaterAllocatorHeader, activeCount), 0,
                           sizeof(uint32_t)};
    vkCmdCopyBuffer(cmd, whitewaterAllocatorBuf_,
                    whitewaterCountReadbackBuffers_[currentFrame_], 1,
                    &countCopy);

    VkBufferMemoryBarrier toHost{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    toHost.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toHost.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    toHost.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toHost.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toHost.buffer = whitewaterCountReadbackBuffers_[currentFrame_];
    toHost.offset = 0;
    toHost.size = sizeof(uint32_t);
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &toHost,
                         0, nullptr);

    VkMemoryBarrier toVertex{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    toVertex.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    toVertex.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, 0, 1, &toVertex,
                         0, nullptr, 0, nullptr);
  }

  void recordWhitewaterMask(VkCommandBuffer cmd, VkDescriptorSet computeSet,
                            const DepthPush &dp) {
    const float clear = debugMode_ == kDebugWhitewaterClassification
                            ? -100.0f
                            : kNoSurfaceDepth;
    beginPass(cmd, fluidDepthPass_, whitewaterDepthFB_, swapchainExtent_,
              {clearColor(clear, 0, 0, 0), clearDepth()});
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      whitewaterPipeline_);
    bindParticleState(cmd, computeSet, dp);
    drawParticleQuads(cmd, maxWhitewaterParticles_);
    vkCmdEndRenderPass(cmd);
    colorToSampledBarrier(cmd, whitewaterDepthImage_);
  }

  void recordWaterShadow(VkCommandBuffer cmd, VkDescriptorSet computeSet,
                         const DepthPush &dp) {
    const float sunAngle =
        fluidSettings_.sunAngleDegrees * utils::degreesToRadians;
    const float elevation =
        fluidSettings_.sunElevationDegrees * utils::degreesToRadians;
    const float3 sunDir = normalize(
        float3(std::cos(elevation) * std::cos(sunAngle), std::sin(elevation),
               std::cos(elevation) * std::sin(sunAngle)));
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
    lighting.extinction =
        float4(fluidSettings_.extinctionRed, fluidSettings_.extinctionGreen,
               fluidSettings_.extinctionBlue, 0.0f) *
        fluidSettings_.absorptionScale;
    lighting.shadowParams = float4(fluidSettings_.shadowAmbientLight, 0, 0, 0);
    std::memcpy(sceneLightingBuffers_[currentFrame_].mapped, &lighting,
                sizeof(lighting));

    const bool sunMoved =
        std::abs(fluidSettings_.sunAngleDegrees - shadowSunAngle_) > 0.01f ||
        std::abs(fluidSettings_.sunElevationDegrees - shadowSunElevation_) >
            0.01f;
    const bool dueThisFrame =
        renderedFrameCount_ % fluidSettings_.shadowUpdateInterval == 0;
    if (waterShadowValid_ && !sunMoved && !dueThisFrame)
      return;

    DepthPush shadowDp{};
    shadowDp.view = lightView;
    shadowDp.proj = lightProj;
    shadowDp.camRight = float4(lightRight, 0);
    shadowDp.camUp = float4(lightUp, 0);
    shadowDp.params = dp.params;
    constexpr VkExtent2D extent{kWaterShadowMapSize, kWaterShadowMapSize};

    beginPass(cmd, fluidDepthPass_, waterShadowDepthFB_, extent,
              {clearColor(kNoSurfaceDepth, 0, 0, 0), clearDepth()});
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, depthPipeline_);
    bindParticleState(cmd, computeSet, shadowDp);
    drawParticleQuads(cmd, numParticles_);
    vkCmdEndRenderPass(cmd);
    colorToSampledBarrier(cmd, waterShadowDepthImage_);

    beginPass(cmd, blurPass_, waterShadowFB_, extent,
              {clearColor(0, 0, 0, 0)});
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, thicknessPipeline_);
    bindParticleState(cmd, computeSet, shadowDp);
    drawParticleQuads(cmd, numParticles_);
    vkCmdEndRenderPass(cmd);
    colorToSampledBarrier(cmd, waterShadowImage_);

    waterShadowValid_ = true;
    shadowSunAngle_ = fluidSettings_.sunAngleDegrees;
    shadowSunElevation_ = fluidSettings_.sunElevationDegrees;
  }

  void recordScene(VkCommandBuffer cmd, VkDescriptorSet compositeSet,
                   const DepthPush &dp, const WaterPush &wp) {
    beginPass(cmd, scenePass_, sceneFB_, swapchainExtent_,
              {clearColor(0.02f, 0.03f, 0.06f),
               clearColor(kNoSurfaceDepth, 0, 0, 0), clearDepth()});
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      backgroundPipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            compositePipelineLayout_, 0, 1, &compositeSet, 0,
                            nullptr);
    vkCmdPushConstants(cmd, compositePipelineLayout_,
                       VK_SHADER_STAGE_VERTEX_BIT |
                           VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(WaterPush), &wp);
    vkCmdDraw(cmd, 3, 1, 0, 0);

    if (sceneMeshVertexCount_ > 0) {
      ScenePush scenePush{};
      scenePush.view = dp.view;
      scenePush.proj = dp.proj;
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        sceneMeshPipeline_);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              sceneMeshPipelineLayout_, 0, 1, &compositeSet, 0,
                              nullptr);
      vkCmdPushConstants(cmd, sceneMeshPipelineLayout_,
                         VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ScenePush),
                         &scenePush);
      VkBuffer buffer = sceneMeshBuffer_;
      VkDeviceSize offset = 0;
      vkCmdBindVertexBuffers(cmd, 0, 1, &buffer, &offset);
      vkCmdDraw(cmd, sceneMeshVertexCount_, 1, 0, 0);
    }
    vkCmdEndRenderPass(cmd);
    colorToSampledBarrier(cmd, sceneColorImage_);
    colorToSampledBarrier(cmd, sceneDepthImage_);
  }

  void recordFluidThickness(VkCommandBuffer cmd, VkDescriptorSet computeSet,
                            const DepthPush &dp) {
    beginPass(cmd, blurPass_, fluidThicknessFB_, fluidExtent_,
              {clearColor(0, 0, 0, 0)});
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, thicknessPipeline_);
    bindParticleState(cmd, computeSet, dp);
    drawParticleQuads(cmd, numParticles_);
    vkCmdEndRenderPass(cmd);
    colorToSampledBarrier(cmd, fluidThicknessImage_);
  }

  void recordFluidDepth(VkCommandBuffer cmd, VkDescriptorSet computeSet,
                        const DepthPush &surfaceDp) {
    beginPass(cmd, fluidDepthPass_, fluidDepthFB_, fluidExtent_,
              {clearColor(kNoSurfaceDepth, 0, 0, 0), clearDepth()});
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, depthPipeline_);
    bindParticleState(cmd, computeSet, surfaceDp);
    drawParticleQuads(cmd, numParticles_);
    vkCmdEndRenderPass(cmd);
    colorToSampledBarrier(cmd, fluidDepthImage_);
  }

  void recordSurfaceBlur(VkCommandBuffer cmd, const DepthPush &surfaceDp) {
    auto pass = [&](VkFramebuffer framebuffer, VkDescriptorSet src, float dx,
                    float dy, bool fillSilhouette) {
      beginPass(cmd, blurLoadPass_, framebuffer, fluidExtent_);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, blurPipeline_);
      BlurPush push{{dx, dy},
                    fluidSettings_.depthDifferenceStrength,
                    fluidSettings_.maxBlurRadius * activeRenderScale_,
                    fluidSettings_.blurStrength,
                    surfaceDp.params.x,
                    fillSilhouette ? 1.0f : 0.0f,
                    -1.0f / surfaceDp.proj[1].y};
      vkCmdPushConstants(cmd, blurPipelineLayout_, VK_SHADER_STAGE_FRAGMENT_BIT,
                         0, sizeof(push), &push);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              blurPipelineLayout_, 0, 1, &src, 0, nullptr);
      vkCmdDraw(cmd, 3, 1, 0, 0);
      vkCmdEndRenderPass(cmd);
    };

    for (int i = 0; i < fluidSettings_.blurIterations; ++i) {
      pass(blurTempFB_, i == 0 ? blurSetH_ : blurSetSmooth_, 1.0f, 0.0f,
           i == 0);
      colorToSampledBarrier(cmd, blurTempImage_);
      pass(blurSmoothFB_, blurSetV_, 0.0f, 1.0f, i == 0);
      colorToSampledBarrier(cmd, blurSmoothImage_);
    }
  }

  void recordThicknessBlur(VkCommandBuffer cmd) {
    auto pass = [&](VkFramebuffer framebuffer, VkDescriptorSet src, float dx,
                    float dy) {
      beginPass(cmd, blurLoadPass_, framebuffer, fluidExtent_);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        thicknessBlurPipeline_);
      ThicknessBlurPush push{
          {dx, dy}, fluidSettings_.maxBlurRadius * activeRenderScale_};
      vkCmdPushConstants(cmd, blurPipelineLayout_, VK_SHADER_STAGE_FRAGMENT_BIT,
                         0, sizeof(push), &push);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              blurPipelineLayout_, 0, 1, &src, 0, nullptr);
      vkCmdDraw(cmd, 3, 1, 0, 0);
      vkCmdEndRenderPass(cmd);
    };

    pass(thicknessBlurTempFB_, thicknessBlurSet_, 1.0f, 0.0f);
    colorToSampledBarrier(cmd, thicknessBlurTempImage_);
    pass(blurTempFB_, thicknessBlurSetV_, 0.0f, 1.0f);
    colorToSampledBarrier(cmd, blurTempImage_);
  }

  void recordComposite(VkCommandBuffer cmd, uint32_t imageIndex,
                       VkDescriptorSet computeSet,
                       VkDescriptorSet compositeSet, const DepthPush &dp,
                       const WaterPush &wp) {
    beginPass(cmd, renderPass_, framebuffers_[imageIndex], swapchainExtent_,
              {clearColor(0.02f, 0.03f, 0.06f), clearDepth()});
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, compositePipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            compositePipelineLayout_, 0, 1, &compositeSet, 0,
                            nullptr);
    vkCmdPushConstants(cmd, compositePipelineLayout_,
                       VK_SHADER_STAGE_VERTEX_BIT |
                           VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(WaterPush), &wp);
    vkCmdDraw(cmd, 3, 1, 0, 0);

    if (debugMode_ == kDebugFluidVelocity ||
        debugMode_ == kDebugFluidPressure) {
      DepthPush debugPush = dp;
      debugPush.params = float4(0.5f * dp.params.x,
                                static_cast<float>(debugMode_), 0.0f, 0.0f);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        fluidDebugPipeline_);
      bindParticleState(cmd, computeSet, debugPush);
      drawParticleQuads(cmd, numParticles_);
    }

    if (imguiInitialized_)
      ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    vkCmdEndRenderPass(cmd);
  }

  void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex,
                           const DepthPush &dp, const WaterPush &wp,
                           int physicsSteps) {
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkCheck(vkBeginCommandBuffer(cmd, &bi), "vkBeginCommandBuffer");

    VkDescriptorSet computeSet = descriptorSets_[currentFrame_];
    VkDescriptorSet compositeSet = compositeSets_[currentFrame_];

    DepthPush surfaceDp = dp;
    surfaceDp.params.x *= 1.35f;

    recordSimulation(cmd, computeSet, physicsSteps);
    recordWhitewaterMask(cmd, computeSet, dp);
    recordWaterShadow(cmd, computeSet, dp);
    recordScene(cmd, compositeSet, dp, wp);
    recordFluidThickness(cmd, computeSet, dp);
    recordFluidDepth(cmd, computeSet, surfaceDp);
    recordSurfaceBlur(cmd, surfaceDp);
    recordThicknessBlur(cmd);
    recordComposite(cmd, imageIndex, computeSet, compositeSet, dp, wp);

    vkCheck(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
  }

  void destroyFramebuffer(VkFramebuffer &framebuffer) {
    if (!framebuffer)
      return;
    vkDestroyFramebuffer(device_, framebuffer, nullptr);
    framebuffer = VK_NULL_HANDLE;
  }

  void destroyImage(VkImage &image, VkDeviceMemory &memory, VkImageView &view) {
    if (view)
      vkDestroyImageView(device_, view, nullptr);
    if (image)
      vkDestroyImage(device_, image, nullptr);
    if (memory)
      vkFreeMemory(device_, memory, nullptr);
    view = VK_NULL_HANDLE;
    image = VK_NULL_HANDLE;
    memory = VK_NULL_HANDLE;
  }

  void destroyFluidSizedTargets() {
    destroyFramebuffer(whitewaterDepthFB_);
    destroyFramebuffer(waterShadowDepthFB_);
    destroyFramebuffer(waterShadowFB_);
    destroyFramebuffer(fluidThicknessFB_);
    destroyFramebuffer(fluidDepthFB_);
    destroyFramebuffer(blurTempFB_);
    destroyFramebuffer(blurSmoothFB_);
    destroyFramebuffer(thicknessBlurTempFB_);

    destroyImage(waterShadowZImage_, waterShadowZMem_, waterShadowZView_);
    destroyImage(waterShadowDepthImage_, waterShadowDepthMem_,
                 waterShadowDepthView_);
    destroyImage(waterShadowImage_, waterShadowMem_, waterShadowView_);
    destroyImage(fluidThicknessImage_, fluidThicknessMem_, fluidThicknessView_);
    destroyImage(whitewaterDepthImage_, whitewaterDepthMem_,
                 whitewaterDepthView_);
    destroyImage(whitewaterZImage_, whitewaterZMem_, whitewaterZView_);
    destroyImage(fluidDepthImage_, fluidDepthMem_, fluidDepthView_);
    destroyImage(fluidZImage_, fluidZMem_, fluidZView_);
    destroyImage(blurTempImage_, blurTempMem_, blurTempView_);
    destroyImage(blurSmoothImage_, blurSmoothMem_, blurSmoothView_);
    destroyImage(thicknessBlurTempImage_, thicknessBlurTempMem_,
                 thicknessBlurTempView_);
  }

  void destroySceneSizedTargets() {
    destroyFramebuffer(sceneFB_);
    destroyImage(sceneColorImage_, sceneColorMem_, sceneColorView_);
    destroyImage(sceneDepthImage_, sceneDepthMem_, sceneDepthView_);
    destroyImage(sceneZImage_, sceneZMem_, sceneZView_);
  }

  void cleanupSwapchain() {
    destroySceneSizedTargets();
    destroyFluidSizedTargets();
    for (auto &framebuffer : framebuffers_)
      destroyFramebuffer(framebuffer);
    framebuffers_.clear();
    destroyImage(depthImage_, depthMemory_, depthView_);
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
      ImGui_ImplVulkan_SetMinImageCount(
          static_cast<uint32_t>(swapchainImages_.size()));
    VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    renderFinishedSemaphores_.resize(swapchainImages_.size());
    for (auto &sem : renderFinishedSemaphores_)
      vkCheck(vkCreateSemaphore(device_, &si, nullptr, &sem),
              "vkCreateSemaphore(renderFinished)");
  }
};

} // namespace vkr
