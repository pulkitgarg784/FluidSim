#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <imgui.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "linalg.h"
#include "sph.h"
#include "vk_renderer.h"

using namespace linalg::aliases;

constexpr float PI = 3.14159265358979f;
constexpr float DegToRad = PI / 180.0f;
constexpr float ANGFACT = 0.2f;    // mouse look sensitivity
constexpr float moveSpeed = 1.5f;  // keyboard movement
constexpr float interactionRadius = 1.5f;
constexpr float interactionStrength = 40.0f;

static float3 globalEye = float3(0.0f, 5.0f, 14.0f);
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
  if (ImGui::GetIO().WantCaptureKeyboard)
    return;
}

static void processKeyboard(GLFWwindow *w, float dt) {
  if (ImGui::GetIO().WantCaptureKeyboard)
    return;
  const float d = moveSpeed * dt;
  auto down = [&](int k) { return glfwGetKey(w, k) == GLFW_PRESS; };
  float3 move = float3(0.0f);
  if (down(GLFW_KEY_W))
    move += globalViewDir;
  if (down(GLFW_KEY_S))
    move -= globalViewDir;
  if (down(GLFW_KEY_D))
    move += globalRight;
  if (down(GLFW_KEY_A))
    move -= globalRight;
  if (down(GLFW_KEY_Q))
    move += globalUp;
  if (down(GLFW_KEY_Z))
    move -= globalUp;
  if (dot(move, move) > 0.0f) {
    move = d * move;
    globalEye += move;
    globalLookat += move;
  }
}

static void mouseButtonFunc(GLFWwindow *window, int button, int action, int) {
  if (ImGui::GetIO().WantCaptureMouse) {
    if (button == GLFW_MOUSE_BUTTON_LEFT)
      mouseLeftPressed = false;
    if (button == GLFW_MOUSE_BUTTON_RIGHT)
      mouseRightPressed = false;
    return;
  }

  const bool shiftHeld = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                        glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
  if (button == GLFW_MOUSE_BUTTON_LEFT) {
    if (action == GLFW_PRESS) {
      if (!shiftHeld)
        mouseLeftPressed = true;
    } else if (action == GLFW_RELEASE) {
      mouseLeftPressed = false;
    }
  } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
    mouseRightPressed = (action == GLFW_PRESS);
  }
}

static void cursorPosFunc(GLFWwindow *, double mouse_x, double mouse_y) {
  if (ImGui::GetIO().WantCaptureMouse) {
    m_mouseX = mouse_x;
    m_mouseY = mouse_y;
    return;
  }
  if (mouseLeftPressed) {
    const float xfact = -ANGFACT * float(mouse_y - m_mouseY);
    const float yfact = -ANGFACT * float(mouse_x - m_mouseX);
    float3 v = globalViewDir;

    struct {
      float3 operator()(float theta, const float3 &v, const float3 &w) {
        const float c = cosf(theta);
        const float s = sinf(theta);
        const float3 v0 = dot(v, w) * w;
        const float3 v1 = v - v0;
        const float3 v2 = cross(w, v1);
        return v0 + c * v1 + s * v2;
      }
    } rotateVector;

    v = rotateVector(xfact * DegToRad, v, globalRight);
    v = rotateVector(yfact * DegToRad, v, globalUp);
    globalViewDir = v;
    globalLookat = globalEye + globalViewDir;
    globalRight = cross(globalViewDir, globalUp);
  }
  m_mouseX = mouse_x;
  m_mouseY = mouse_y;
}

static float4x4 perspectiveVK(float fovyDeg, float aspect, float zNear,
                              float zFar) {
  const float f = 1.0f / std::tan(fovyDeg * DegToRad * 0.5f);
  float4x4 m(float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f));
  m[0] = float4(f / aspect, 0.0f, 0.0f, 0.0f);
  m[1] = float4(0.0f, -f, 0.0f, 0.0f);
  m[2] = float4(0.0f, 0.0f, zFar / (zNear - zFar), -1.0f);
  m[3] = float4(0.0f, 0.0f, (zFar * zNear) / (zNear - zFar), 0.0f);
  return m;
}

static float4x4 lookAt(const float3 &eye, const float3 &center,
                       const float3 &up) {
  const float3 f = normalize(center - eye);
  const float3 s = normalize(cross(f, normalize(up)));
  const float3 u = cross(s, f);
  float4x4 m = linalg::identity;
  m[0] = float4(s.x, u.x, -f.x, 0.0f);
  m[1] = float4(s.y, u.y, -f.y, 0.0f);
  m[2] = float4(s.z, u.z, -f.z, 0.0f);
  m[3] = float4(-dot(s, eye), -dot(u, eye), dot(f, eye), 1.0f);
  return m;
}

static float3 mouseRayDirection(double mouseX, double mouseY, int width,
                                int height, float fovyDeg, float aspect) {
  float ndcX = (2.0f * float(mouseX) / float(width)) - 1.0f;
  float ndcY = 1.0f - (2.0f * float(mouseY) / float(height));
  float tanHalfFov = std::tan(fovyDeg * DegToRad * 0.5f);
  float3 dir = globalViewDir + globalRight * (ndcX * tanHalfFov * aspect) +
              cross(globalRight, globalViewDir) * (ndcY * tanHalfFov);
  return normalize(dir);
}

int main(int argc, char **argv) {
  constexpr int width = 1024;
  constexpr int height = 768;

  uint32_t particleCount = sph::maxParticles;
  std::string scenePath = "media/fluid_container.obj";
  float sceneScale = 1.0f;
  for (int arg = 1; arg < argc; ++arg) {
    std::string flag = argv[arg];
    if (flag == "--particles" && arg + 1 < argc) {
      ++arg;
    try {
      unsigned long requested = std::stoul(argv[arg]);
      if (requested == 0 || requested > (unsigned long)sph::maxParticles)
        throw std::out_of_range("particle count");
      particleCount = static_cast<uint32_t>(requested);
    } catch (const std::exception &) {
      std::fprintf(stderr, "Particle count must be between 1 and %d\n",
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
      std::fprintf(stderr, "Usage: %s [--particles 1..%d] [--scene mesh.obj] [--scene-scale positive]\n", argv[0], sph::maxParticles);
      return 1;
    }
  }

  const float densityRatio =
      float(particleCount) / float(sph::referenceParticleCount);
  const float particleScale = std::cbrt(1.0f / densityRatio);
  const float smoothingRadius = 0.12f * particleScale;
  const float particleMass = 1.0f / densityRatio;
  const float simulationDeltaT = 0.002f * particleScale;
  const float renderRadius = 0.068f * particleScale;

  uint32_t gridX = uint32_t(std::ceil(sph::boundsSize.x / smoothingRadius)) + 2u;
  uint32_t gridY = uint32_t(std::ceil(sph::boundsSize.y / smoothingRadius)) + 2u;
  uint32_t gridZ = uint32_t(std::ceil(sph::boundsSize.z / smoothingRadius)) + 2u;
  uint64_t gridCells64 = uint64_t(gridX) * gridY * gridZ;
  if (gridCells64 > uint64_t(particleCount) * 2u) {
    return 1;
  }

  std::cout << "Starting " << particleCount << " particles; h="
            << smoothingRadius << ", grid=" << gridX << 'x' << gridY << 'x'
            << gridZ << std::endl;

  vkr::VulkanRenderer renderer;
  try {
    renderer.init(width, height, "CS488 Final Project", particleCount);
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
    renderer.cleanup();
    return 1;
  }

  // Push SPH parameters
  vkr::SphParams params{};
  params.gravity[0] = sph::gravity.x;
  params.gravity[1] = sph::gravity.y;
  params.gravity[2] = sph::gravity.z;
  params.gravity[3] = sph::tensilePressureScale;
  params.boundsSize[0] = sph::boundsSize.x;
  params.boundsSize[1] = sph::boundsSize.y;
  params.boundsSize[2] = sph::boundsSize.z;
  params.deltaT = simulationDeltaT;
  params.smoothingRadius = smoothingRadius;
  params.particleMass = particleMass;
  params.targetDensity = sph::targetDensity;
  params.pressureMultiplier = sph::pressureMultiplier;
  params.viscosityStrength = sph::viscosityStrength;
  params.collisionDamping = sph::collisionDamping;
  const float h2 = smoothingRadius * smoothingRadius;
  const float h5 = h2 * h2 * smoothingRadius;
  const float h9 = h2 * h2 * h2 * h2 * smoothingRadius;
  params.spikyPow2Scale = 15.0f / (2.0f * PI * h5);
  params.spikyPow2GradScale = 15.0f / (PI * h5);
  params.poly6Scale = 315.0f / (64.0f * PI * h9);
  params.numParticles = particleCount;
  params.epsilon = sph::Epsilon;
  params.grid[0] = gridX;
  params.grid[1] = gridY;
  params.grid[2] = gridZ;
  params.grid[3] = static_cast<uint32_t>(gridCells64);
  renderer.setParams(params);

  std::vector<float4> initPos(particleCount), initVel(particleCount);
  const float3 half = sph::boundsSize * 0.5f;
  const float slab = sph::boundsSize.x * 0.2f;
  for (uint32_t i = 0; i < particleCount; ++i) {
    bool leftEdge = i < particleCount / 2u;
    float x = leftEdge ? (-half.x + sph::PCG32::rand() * slab)
                       : (half.x - sph::PCG32::rand() * slab);
    float y = (sph::PCG32::rand() - 0.5f) * sph::boundsSize.y;
    float z = (sph::PCG32::rand() - 0.5f) * sph::boundsSize.z;
    initPos[i] = float4(x, y, z, 0.0f);
    initVel[i] = float4(0.0f);
  }
  renderer.uploadInitialState(initPos, initVel);
  initPos.clear();
  initVel.clear();
  initPos.shrink_to_fit();
  initVel.shrink_to_fit();

  double lastTime = glfwGetTime();

  // FPS counter variables
  auto fpsStartTime = std::chrono::high_resolution_clock::now();
  uint64_t frameCount = 0;
  double fps = 0.0;

  try {
    while (!renderer.shouldClose()) {
      renderer.pollEvents();
      renderer.beginImGuiFrame();

      const double now = glfwGetTime();
      float dt = float(now - lastTime);
      lastTime = now;
      dt = std::min(dt, 0.1f);

      globalViewDir = normalize(globalLookat - globalEye);
      globalRight = normalize(cross(globalViewDir, globalUp));

      processKeyboard(win, dt);
      globalViewDir = normalize(globalLookat - globalEye);
      globalRight = normalize(cross(globalViewDir, globalUp));

      const float3 camUp = cross(globalRight, globalViewDir);

      int windowWidth = 0, windowHeight = 0;
      int framebufferWidth = 0, framebufferHeight = 0;
      glfwGetWindowSize(win, &windowWidth, &windowHeight);
      glfwGetFramebufferSize(win, &framebufferWidth, &framebufferHeight);
      windowWidth = std::max(windowWidth, 1);
      windowHeight = std::max(windowHeight, 1);
      framebufferWidth = std::max(framebufferWidth, 1);
      framebufferHeight = std::max(framebufferHeight, 1);
      const float aspect =
          float(framebufferWidth) / float(framebufferHeight);
      const float4x4 proj = perspectiveVK(45.0f, aspect, 0.01f, 100.0f);
      const float4x4 view = lookAt(globalEye, globalLookat, globalUp);

      // Shift+Left click attracts the fluid, Shift+Right click repels it.
      const bool shiftHeld =
          glfwGetKey(win, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
          glfwGetKey(win, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
      float interactionStrengthSigned = 0.0f;
      float3 interactionPoint = float3(0.0f);
      if (shiftHeld && (mouseLeftPressed || mouseRightPressed)) {
        float3 rayDir = mouseRayDirection(m_mouseX, m_mouseY, windowWidth,
                                          windowHeight, 45.0f, aspect);
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

      // the simulation now runs entirely on the GPU inside drawFrame
      renderer.drawFrame(view, proj, globalRight, camUp, globalViewDir,
             renderRadius, renderer.simulationSubsteps());

      // Update FPS counter
      frameCount++;
      auto currentTime = std::chrono::high_resolution_clock::now();
      auto elapsed = std::chrono::duration<double>(currentTime - fpsStartTime).count();
      
      // Update and print FPS every second
      if (elapsed >= 1.0) {
        fps = frameCount / elapsed;
        frameCount = 0;
        fpsStartTime = currentTime;
        std::cout << "FPS: " << fps << " (" << particleCount
                  << " particles, " << renderer.simulationSubsteps()
                  << " substeps)" << std::endl;
      }
    }
    renderer.waitIdle();
  } catch (const std::exception &e) {
    std::fprintf(stderr, "Runtime error: %s\n", e.what());
    renderer.cleanup();
    return 1;
  }

  renderer.cleanup();
  return 0;
}
