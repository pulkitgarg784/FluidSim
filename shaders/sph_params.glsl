#ifndef SPH_PARAMS_GLSL
#define SPH_PARAMS_GLSL

layout(std140, binding = 5) uniform Params {
  vec4 gravity;     // xyz = gravity, w = near-pressure multiplier
  vec4 boundsSize;  // xyz = tank size
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
  uvec4 grid; // xyz = dimensions, w = total cells
} P;

const uint EMPTY_CELL = 0xffffffffu;

const float TWO_PI = 6.28318530718;

ivec3 gridCell(vec3 p) {
  ivec3 c = ivec3(floor((p + 0.5 * P.boundsSize.xyz) / P.smoothingRadius)) + 1;
  return clamp(c, ivec3(0), ivec3(P.grid.xyz) - 1);
}

uint cellKey(ivec3 c) {
  return (uint(c.z) * P.grid.y + uint(c.y)) * P.grid.x + uint(c.x);
}

ivec3 neighbourMin(ivec3 c) { return max(ivec3(-1), -c); }
ivec3 neighbourMax(ivec3 c) { return min(ivec3(1), ivec3(P.grid.xyz) - 1 - c); }

uint neighbourKey(int centerKey, ivec3 offset) {
  int planar = (offset.z * int(P.grid.y) + offset.y) * int(P.grid.x);
  return uint(centerKey + planar + offset.x);
}

#endif
