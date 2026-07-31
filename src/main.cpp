#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include <imgui.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "linalg.h"
#include "sph.h"
#include "utils.h"
#include "vk_renderer.h"

using namespace linalg::aliases;

constexpr float PI = 3.14159265358979f;
constexpr float mouseLookSpeed = 0.2f; // mouse look sensitivity
constexpr float moveSpeed = 1.5f;      // keyboard movement
constexpr float interactionRadius = 1.5f;
constexpr float interactionStrength = 40.0f;
constexpr float physicsTimeStep = 1 / 75.0f;

// width, height, length
const float3 tankSize(8.77f, 4.20f, 2.92f);

const float3 waterMin(-4.27f, -1.98f, -1.34f);
const float3 waterMax(-2.19f, 1.69f, 1.34f);

static float3 globalEye = float3(0.0f, 6.0f, 17.0f);
static float3 globalLookat = float3(0.0f, 0.0f, 0.0f);
static float3 globalUp = normalize(float3(0.0f, 1.0f, 0.0f));
static float3 globalViewDir = normalize(globalLookat - globalEye);
static float3 globalRight = normalize(cross(globalViewDir, globalUp));

static bool mouseLeftPressed = false;
static bool mouseRightPressed = false;
static double m_mouseX = 0.0;
static double m_mouseY = 0.0;

static void keyFunc(GLFWwindow *window, int key, int, int action, int) {
  if (action == GLFW_PRESS && key == GLFW_KEY_ESCAPE)
    glfwSetWindowShouldClose(window, GLFW_TRUE);
}

static void processKeyboard(GLFWwindow *w, float dt) {
  if (ImGui::GetIO().WantCaptureKeyboard)
    return;
  const float d = moveSpeed * dt;
  float3 move = float3(0.0f);
  if (glfwGetKey(w, GLFW_KEY_W) == GLFW_PRESS)
    move += globalViewDir;
  if (glfwGetKey(w, GLFW_KEY_S) == GLFW_PRESS)
    move -= globalViewDir;
  if (glfwGetKey(w, GLFW_KEY_D) == GLFW_PRESS)
    move += globalRight;
  if (glfwGetKey(w, GLFW_KEY_A) == GLFW_PRESS)
    move -= globalRight;
  if (glfwGetKey(w, GLFW_KEY_Q) == GLFW_PRESS)
    move += globalUp;
  if (glfwGetKey(w, GLFW_KEY_Z) == GLFW_PRESS)
    move -= globalUp;
  if (dot(move, move) > 0.0f) {
    move = d * move;
    globalEye += move;
    globalLookat += move;
  }
}

static void mouseButtonFunc(GLFWwindow *, int button, int action, int) {
  if (ImGui::GetIO().WantCaptureMouse) {
    if (button == GLFW_MOUSE_BUTTON_LEFT)
      mouseLeftPressed = false;
    if (button == GLFW_MOUSE_BUTTON_RIGHT)
      mouseRightPressed = false;
    return;
  }
  if (button == GLFW_MOUSE_BUTTON_LEFT) {
    mouseLeftPressed = (action == GLFW_PRESS);
  } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
    mouseRightPressed = (action == GLFW_PRESS);
  }
}

static void cursorPosFunc(GLFWwindow *window, double mouse_x, double mouse_y) {
  if (ImGui::GetIO().WantCaptureMouse) {
    m_mouseX = mouse_x;
    m_mouseY = mouse_y;
    return;
  }
  const bool shiftHeld =
      glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
  if (mouseLeftPressed && !shiftHeld) {
    const float xfact = -mouseLookSpeed * float(mouse_y - m_mouseY);
    const float yfact = -mouseLookSpeed * float(mouse_x - m_mouseX);
    float3 v = globalViewDir;

    v = utils::rotateVector(xfact * utils::degreesToRadians, v, globalRight);
    v = utils::rotateVector(yfact * utils::degreesToRadians, v, globalUp);
    globalViewDir = v;
    globalLookat = globalEye + globalViewDir;
    globalRight = cross(globalViewDir, globalUp);
  }
  m_mouseX = mouse_x;
  m_mouseY = mouse_y;
}

static bool parseParticleCount(std::string_view text,
                               uint32_t &particleCount) {
  uint32_t parsed = 0;
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), parsed);
  if (error != std::errc{} || end != text.data() + text.size() || parsed == 0 ||
      parsed > sph::maxParticles)
    return false;
  particleCount = parsed;
  return true;
}

static std::vector<float4>
generateParticlePositions(const float3 &waterMin, const float3 &waterMax,
                          uint32_t particleCount) {
  std::vector<float4> positions(particleCount);
  const float3 waterSize = waterMax - waterMin;
  const float nominalSpacing = std::cbrt(
      (waterSize.x * waterSize.y * waterSize.z) / float(particleCount));
  const uint32_t countX = std::max(
      1u, static_cast<uint32_t>(std::ceil(waterSize.x / nominalSpacing)));
  const uint32_t countZ = std::max(
      1u, static_cast<uint32_t>(std::ceil(waterSize.z / nominalSpacing)));
  const uint32_t particlesPerLayer = countX * countZ;
  const uint32_t countY = std::max(
      1u, (particleCount + particlesPerLayer - 1u) / particlesPerLayer);
  const float3 spacing(waterSize.x / float(countX), waterSize.y / float(countY),
                       waterSize.z / float(countZ));

  for (uint32_t i = 0; i < particleCount; ++i) {
    const uint32_t xIndex = i % countX;
    const uint32_t zIndex = (i / countX) % countZ;
    const uint32_t yIndex = i / particlesPerLayer;
    const float3 position =
        waterMin +
        spacing * (float3(float(xIndex), float(yIndex), float(zIndex)) + 0.5f);
    positions[i] = float4(position.x, position.y, position.z, 0.0f);
  }
  return positions;
}

int main(int argc, char **argv) {
  constexpr int width = 1024;
  constexpr int height = 768;

  uint32_t particleCount = sph::defaultFluidParticleCount;
  std::string scenePath = "media/fluid_container.obj";
  float sceneScale = 1.0f;
  for (int arg = 1; arg < argc; ++arg) {
    std::string_view flag = argv[arg];
    if (flag == "--particles" && arg + 1 < argc) {
      ++arg;
      if (!parseParticleCount(argv[arg], particleCount)) {
        std::fprintf(stderr, "Particle count must be between 1 and %u\n",
                     sph::maxParticles);
        return 1;
      }
    } else if (flag == "--scene" && arg + 1 < argc) {
      scenePath = argv[++arg];
    } else if (flag == "--scene-scale" && arg + 1 < argc) {
      try {
        sceneScale = std::stof(argv[++arg]);
        if (sceneScale <= 0.0f)
          throw std::out_of_range("scene scale");
      } catch (const std::exception &) {
        std::fprintf(stderr, "Scene scale must be positive\n");
        return 1;
      }
    } else {
      std::fprintf(stderr,
                   "Usage: %s [--particles 1..%u] [--scene mesh.obj] "
                   "[--scene-scale positive]\n",
                   argv[0], sph::maxParticles);
      return 1;
    }
  }

  const float densityRatio =
      float(particleCount) / float(sph::referenceParticleCount);
  const float particleScale = std::cbrt(1.0f / densityRatio);
  const float smoothingRadius = 0.12f * particleScale;
  const float particleMass = 1.0f / densityRatio;
  const float simulationDeltaT = physicsTimeStep;
  const float renderRadius = 0.068f * particleScale;

  const uint32_t gridX =
      static_cast<uint32_t>(std::ceil(tankSize.x / smoothingRadius)) +
      2u;
  const uint32_t gridY =
      static_cast<uint32_t>(std::ceil(tankSize.y / smoothingRadius)) +
      2u;
  const uint32_t gridZ =
      static_cast<uint32_t>(std::ceil(tankSize.z / smoothingRadius)) +
      2u;
  const uint32_t gridCellCount = gridX * gridY * gridZ;
  const uint32_t maxGridCellCount = particleCount * 2u;
  if (gridCellCount > maxGridCellCount) {
    std::fprintf(stderr,
                 "Tank grid needs %u cells, but the current particle count "
                 "supports at most %u. Increase the particle count or use a "
                 "smaller tank.\n",
                 gridCellCount, maxGridCellCount);
    return 1;
  }

  std::cout << "Starting " << particleCount
            << " particles; h=" << smoothingRadius << ", grid=" << gridX << 'x'
            << gridY << 'x' << gridZ << std::endl;

  vkr::VulkanRenderer renderer;
  try {
    renderer.init(width, height, "CS488 Final Project", particleCount,
                  sph::maxWhitewaterParticleCount);
    if (!scenePath.empty())
      renderer.loadSceneMesh(scenePath, sceneScale);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "Renderer init failed: %s\n", e.what());
    return 1;
  }

  GLFWwindow *win = renderer.window();

  glfwSetKeyCallback(win, keyFunc);
  glfwSetMouseButtonCallback(win, mouseButtonFunc);
  glfwSetCursorPosCallback(win, cursorPosFunc);

  try {
    renderer.initImGui();
  } catch (const std::exception &e) {
    std::fprintf(stderr, "ImGui init failed: %s\n", e.what());
    renderer.cleanup();
    return 1;
  }

  // Push SPH parameters
  vkr::SphParams params{};
  params.gravity[0] = sph::gravity.x;
  params.gravity[1] = sph::gravity.y;
  params.gravity[2] = sph::gravity.z;
  params.gravity[3] = sph::nearPressureMultiplier;
  params.boundsSize[0] = tankSize.x;
  params.boundsSize[1] = tankSize.y;
  params.boundsSize[2] = tankSize.z;
  params.deltaT = simulationDeltaT;
  params.smoothingRadius = smoothingRadius;
  params.particleMass = particleMass;
  params.targetDensity = sph::targetDensity;
  params.pressureMultiplier = sph::pressureMultiplier;
  params.viscosityStrength = sph::viscosityStrength;
  params.collisionDamping = sph::collisionDamping;
  const float r2 = smoothingRadius * smoothingRadius;
  const float r5 = r2 * r2 * smoothingRadius;
  const float r9 = r2 * r2 * r2 * r2 * smoothingRadius;
  params.densityKernelScale = 15.0f / (2.0f * PI * r5);
  params.pressureGradientKernelScale = 15.0f / (PI * r5);
  params.viscosityKernelScale = 315.0f / (64.0f * PI * r9);
  params.numParticles = particleCount;
  params.epsilon = sph::Epsilon;
  params.grid[0] = gridX;
  params.grid[1] = gridY;
  params.grid[2] = gridZ;
  params.grid[3] = gridCellCount;
  renderer.setParams(params);

  // Upload initial particle positions and velocities.
  // Scoped to ensure the large vectors are destroyed after upload.
  {
    std::vector<float4> initialPositions =
        generateParticlePositions(waterMin, waterMax, particleCount);
    std::vector<float4> initialVelocities(particleCount, float4(0.0f));
    renderer.uploadInitialState(initialPositions, initialVelocities);
  }

  double lastTime = glfwGetTime();
  double physicsAccumulator = 0.0;

  try {
    while (!renderer.shouldClose()) {
      renderer.pollEvents();
      renderer.beginImGuiFrame();

      const double now = glfwGetTime();
      const double elapsedTime = std::min(now - lastTime, 0.1);
      lastTime = now;
      const float dt = static_cast<float>(elapsedTime);
      physicsAccumulator += elapsedTime;

      globalViewDir = normalize(globalLookat - globalEye);
      globalRight = normalize(cross(globalViewDir, globalUp));

      processKeyboard(win, dt);

      const float3 camUp = cross(globalRight, globalViewDir);

      int windowWidth = 0, windowHeight = 0;
      int framebufferWidth = 0, framebufferHeight = 0;
      glfwGetWindowSize(win, &windowWidth, &windowHeight);
      glfwGetFramebufferSize(win, &framebufferWidth, &framebufferHeight);
      windowWidth = std::max(windowWidth, 1);
      windowHeight = std::max(windowHeight, 1);
      framebufferWidth = std::max(framebufferWidth, 1);
      framebufferHeight = std::max(framebufferHeight, 1);
      const float aspect = float(framebufferWidth) / float(framebufferHeight);
      const float4x4 proj =
          utils::perspectiveVK(45.0f, aspect, 0.01f, 100.0f);
      const float4x4 view = utils::lookAt(globalEye, globalLookat, globalUp);

      // Shift+Left click attracts the fluid, Shift+Right click repels it.
      const bool shiftHeld =
          glfwGetKey(win, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
          glfwGetKey(win, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
      float interactionStrengthSigned = 0.0f;
      float3 interactionPoint = float3(0.0f);
      if (shiftHeld && (mouseLeftPressed || mouseRightPressed)) {
        float3 rayDir = utils::mouseRayDirection(
            m_mouseX, m_mouseY, windowWidth, windowHeight, 45.0f, aspect,
            globalViewDir, globalRight);
        float3 planeNormal = -globalViewDir; // faces the camera
        float3 planePoint = float3(0.0f);    // box centre
        float denom = dot(rayDir, planeNormal);
        if (std::abs(denom) > 1e-6f) {
          float t = dot(planePoint - globalEye, planeNormal) / denom;
          if (t > 0.0f) {
            interactionPoint = globalEye + rayDir * t;
            interactionStrengthSigned =
                mouseLeftPressed ? interactionStrength : -interactionStrength;
          }
        }
      }
      renderer.setInteraction(interactionPoint, interactionRadius,
                              interactionStrengthSigned);

      const int physicsSteps =
          physicsAccumulator >= static_cast<double>(physicsTimeStep) ? 1 : 0;
      if (physicsSteps != 0)
        physicsAccumulator =
            std::fmod(physicsAccumulator, static_cast<double>(physicsTimeStep));
      renderer.drawFrame(view, proj, globalRight, camUp, globalViewDir,
                         renderRadius, physicsSteps);
    }
  } catch (const std::exception &e) {
    std::fprintf(stderr, "Runtime error: %s\n", e.what());
    renderer.cleanup();
    return 1;
  }

  renderer.cleanup();
  return 0;
}
