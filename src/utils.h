#pragma once

#include <cmath>

#include "linalg.h"

namespace utils {

using linalg::aliases::float3;
using linalg::aliases::float4;
using linalg::aliases::float4x4;

constexpr float pi = 3.14159265358979323846f;
constexpr float degreesToRadians = pi / 180.0f;

inline float3 rotateVector(float theta, const float3 &vector,
                           const float3 &axis) {
  const float c = std::cos(theta);
  const float s = std::sin(theta);
  const float3 parallel = linalg::dot(vector, axis) * axis;
  const float3 perpendicular = vector - parallel;
  return parallel + c * perpendicular +
         s * linalg::cross(axis, perpendicular);
}

inline float4x4 perspectiveVK(float fovyDeg, float aspect, float zNear,
                              float zFar) {
  const float f = 1.0f / std::tan(fovyDeg * degreesToRadians * 0.5f);
  float4x4 matrix(float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f));
  matrix[0] = float4(f / aspect, 0.0f, 0.0f, 0.0f);
  matrix[1] = float4(0.0f, -f, 0.0f, 0.0f);
  matrix[2] = float4(0.0f, 0.0f, zFar / (zNear - zFar), -1.0f);
  matrix[3] = float4(0.0f, 0.0f, (zFar * zNear) / (zNear - zFar), 0.0f);
  return matrix;
}

inline float4x4 lookAt(const float3 &eye, const float3 &center,
                       const float3 &up) {
  const float3 forward = linalg::normalize(center - eye);
  const float3 right =
      linalg::normalize(linalg::cross(forward, linalg::normalize(up)));
  const float3 cameraUp = linalg::cross(right, forward);
  float4x4 matrix = linalg::identity;
  matrix[0] = float4(right.x, cameraUp.x, -forward.x, 0.0f);
  matrix[1] = float4(right.y, cameraUp.y, -forward.y, 0.0f);
  matrix[2] = float4(right.z, cameraUp.z, -forward.z, 0.0f);
  matrix[3] = float4(-linalg::dot(right, eye), -linalg::dot(cameraUp, eye),
                     linalg::dot(forward, eye), 1.0f);
  return matrix;
}

inline float3 mouseRayDirection(double mouseX, double mouseY, int width,
                                int height, float fovyDeg, float aspect,
                                const float3 &viewDir, const float3 &right) {
  const float ndcX = (2.0f * float(mouseX) / float(width)) - 1.0f;
  const float ndcY = 1.0f - (2.0f * float(mouseY) / float(height));
  const float tanHalfFov = std::tan(fovyDeg * degreesToRadians * 0.5f);
  const float3 direction =
      viewDir + right * (ndcX * tanHalfFov * aspect) +
      linalg::cross(right, viewDir) * (ndcY * tanHalfFov);
  return linalg::normalize(direction);
}

} // namespace utils
