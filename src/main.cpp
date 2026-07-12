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

static float3 globalEye = float3(0.0f, 0.0f, 4.0f);
static float3 globalLookat = float3(0.0f, 0.0f, 0.0f);
static float3 globalUp = normalize(float3(0.0f, 1.0f, 0.0f));
static float3 globalViewDir = normalize(globalLookat - globalEye);
static float3 globalRight = normalize(cross(globalViewDir, globalUp));

static bool mouseLeftPressed = false;
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

static void mouseButtonFunc(GLFWwindow *, int button, int action, int) {
  if (button == GLFW_MOUSE_BUTTON_LEFT) {
    if (action == GLFW_PRESS)
      mouseLeftPressed = true;
    else if (action == GLFW_RELEASE)
      mouseLeftPressed = false;
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

int main() {
  constexpr int width = 1024;
  constexpr int height = 768;

  sph::FluidSim sim;
  sim.initialize();

  vkr::VulkanRenderer renderer;
  try {
    renderer.init(width, height, "CS488 - SPH Fluid (Vulkan)",
                  (uint32_t)sph::numParticles);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "Renderer init failed: %s\n", e.what());
    return 1;
  }

  GLFWwindow *win = renderer.window();
  glfwSetKeyCallback(win, keyFunc);
  glfwSetMouseButtonCallback(win, mouseButtonFunc);
  glfwSetCursorPosCallback(win, cursorPosFunc);

  std::vector<vkr::InstanceData> instances(sph::numParticles);

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

      sim.step();

      const float3 camUp = cross(globalRight, globalViewDir);

      const float aspect = float(width) / float(height);
      const float4x4 proj = perspectiveVK(45.0f, aspect, 0.01f, 100.0f);
      const float4x4 view = lookAt(globalEye, globalLookat, globalUp);
      const float4x4 viewProj = mul(proj, view);

      // pack instance data
      for (int i = 0; i < sph::numParticles; i++) {
        instances[i].pos = sim.particles[i].position;
        instances[i].speed = length(sim.particles[i].velocity);
      }

      renderer.drawFrame(instances, viewProj, globalRight, camUp,
                         renderRadius);

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
