#version 450
#extension GL_GOOGLE_include_directive : require

#include "particle_push.glsl"
#include "sph_params.glsl"

layout(location = 0) in vec2 inCorner;

layout(std430, binding = 0) readonly buffer Positions { vec4 positions[]; };
layout(std430, binding = 1) readonly buffer Velocities { vec4 velocities[]; };
layout(std430, binding = 3) readonly buffer Densities { vec2 densities[]; };

layout(location = 0) out vec2 vUV;
layout(location = 1) out vec3 vViewCenter;
layout(location = 2) flat out float vValue;

void main() {
  uint index = gl_InstanceIndex;
  vec3 center = positions[index].xyz;
  vec3 world = billboardPosition(center, inCorner, pc.params.x);
  gl_Position = pc.proj * pc.view * vec4(world, 1.0);

  bool pressureMode = int(pc.params.y + 0.5) == 3;
  vValue = pressureMode
               ? (densities[index].x - P.targetDensity) * P.pressureMultiplier
               : length(velocities[index].xyz);
  vUV = inCorner;
  vViewCenter = (pc.view * vec4(center, 1.0)).xyz;
}
