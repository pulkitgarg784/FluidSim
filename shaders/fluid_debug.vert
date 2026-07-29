#version 450

layout(location = 0) in vec2 inCorner;

layout(std430, binding = 3) readonly buffer Densities {
  vec2 densities[];
};
layout(std140, binding = 5) uniform Params {
  vec4 gravity;
  vec4 boundsSize;
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
  uint numParticles;
  float epsilon;
  uvec4 grid;
} P;
layout(std430, binding = 9) readonly buffer SortedPositions {
  vec4 sortedPositions[];
};
layout(std430, binding = 10) readonly buffer SortedVelocities {
  vec4 sortedVelocities[];
};

layout(push_constant) uniform PC {
  mat4 view;
  mat4 proj;
  vec4 camRight;
  vec4 camUp;
  int debugMode;
} pc;

layout(location = 0) out vec2 vUV;
layout(location = 1) out vec3 vViewCenter;
layout(location = 2) flat out float vValue;
layout(location = 3) flat out float vRadius;

const float DEBUG_PARTICLE_RADIUS_SCALE = 0.2833333333;

void main() {
  uint index = gl_InstanceIndex;
  vec3 center = sortedPositions[index].xyz;
  float radius = P.smoothingRadius * DEBUG_PARTICLE_RADIUS_SCALE;
  vec3 world =
      center +
      (inCorner.x * pc.camRight.xyz + inCorner.y * pc.camUp.xyz) * radius;
  vec4 viewPos = pc.view * vec4(world, 1.0);
  gl_Position = pc.proj * viewPos;

  bool pressureMode = pc.debugMode == 3;
  vValue = pressureMode
               ? (densities[index].x - P.targetDensity) * P.pressureMultiplier
               : length(sortedVelocities[index].xyz);
  vUV = inCorner;
  vViewCenter = (pc.view * vec4(center, 1.0)).xyz;
  vRadius = radius;
}
