#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

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

static float3 globalEye = float3(0.0f, 0.0f, 14.0f);
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

int main() {
  constexpr int width = 1024;
  constexpr int height = 768;

  sph::FluidSim sim;
  sim.initialize();

  vkr::VulkanRenderer renderer;
  try {
    renderer.init(width, height, "CS488 Final Project",
                  (uint32_t)sph::numParticles);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "Renderer init failed: %s\n", e.what());
    return 1;
  }

  GLFWwindow *win = renderer.window();
  glfwSetKeyCallback(win, keyFunc);
  glfwSetMouseButtonCallback(win, mouseButtonFunc);
  glfwSetCursorPosCallback(win, cursorPosFunc);

  // Push SPH parameters
  vkr::SphParams params{};
  params.gravity[0] = sph::gravity.x;
  params.gravity[1] = sph::gravity.y;
  params.gravity[2] = sph::gravity.z;
  params.boundsSize[0] = sph::boundsSize.x;
  params.boundsSize[1] = sph::boundsSize.y;
  params.boundsSize[2] = sph::boundsSize.z;
  params.deltaT = sph::deltaT;
  params.smoothingRadius = sph::smoothingRadius;
  params.particleMass = sph::particleMass;
  params.targetDensity = sph::targetDensity;
  params.pressureMultiplier = sph::pressureMultiplier;
  params.viscosityStrength = sph::viscosityStrength;
  params.collisionDamping = sph::collisionDamping;
  params.spikyPow2Scale = sph::Pow2Scale;
  params.spikyPow2GradScale = sph::Pow2GradScale;
  params.poly6Scale = sph::viscocityScale;
  params.numParticles = (uint32_t)sph::numParticles;
  params.epsilon = sph::Epsilon;
  renderer.setParams(params);

  // copy initial particle positions and velocities into GPU buffers
  std::vector<float4> initPos(sph::numParticles), initVel(sph::numParticles);
  for (int i = 0; i < sph::numParticles; i++) {
    initPos[i] = float4(sim.particles[i].position, 0.0f);
    initVel[i] = float4(sim.particles[i].velocity, 0.0f);
  }
  renderer.uploadInitialState(initPos, initVel);

  const float renderRadius = 0.02f;

  double lastTime = glfwGetTime();

  // FPS counter variables
  auto fpsStartTime = std::chrono::high_resolution_clock::now();
  uint64_t frameCount = 0;
  double fps = 0.0;

  try {
    while (!renderer.shouldClose()) {
      renderer.pollEvents();

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

      const float aspect = float(width) / float(height);
      const float4x4 proj = perspectiveVK(45.0f, aspect, 0.01f, 100.0f);
      const float4x4 view = lookAt(globalEye, globalLookat, globalUp);
      const float4x4 viewProj = mul(proj, view);

      // Shift+Left click attracts the fluid, Shift+Right click repels it.
      const bool shiftHeld =
          glfwGetKey(win, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
          glfwGetKey(win, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
      float interactionStrengthSigned = 0.0f;
      float3 interactionPoint = float3(0.0f);
      if (shiftHeld && (mouseLeftPressed || mouseRightPressed)) {
        float3 rayDir = mouseRayDirection(m_mouseX, m_mouseY, width, height,
                                          45.0f, aspect);
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
      renderer.drawFrame(viewProj, globalRight, camUp, renderRadius,
                         sph::simIterationsPerFrame);

      // Update FPS counter
      frameCount++;
      auto currentTime = std::chrono::high_resolution_clock::now();
      auto elapsed = std::chrono::duration<double>(currentTime - fpsStartTime).count();
      
      // Update and print FPS every second
      if (elapsed >= 1.0) {
        fps = frameCount / elapsed;
        frameCount = 0;
        fpsStartTime = currentTime;
        std::cout << "FPS: " << fps << std::endl;
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
