#version 450

layout(location = 0) in vec2 vUV;
layout(location = 1) in vec3 vViewCenter;
layout(location = 2) flat in float vActive;
layout(location = 3) flat in float vType;

layout(std140, binding = 14) uniform WhitewaterParams {
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
  uint capacity;
  uint sprayMaxNeighbours;
  uint bubbleMinNeighbours;
  uint enabled;
  uint debugClassification;
  uint activeCount;
} W;
layout(push_constant) uniform PC {
  mat4 view;
  mat4 proj;
  vec4 camRight;
  vec4 camUp;
  vec4 params;
} pc;

layout(location = 0) out float outLinearDepth;

void main() {
  float radiusSquared = dot(vUV, vUV);
  if (vActive < 0.5 || radiusSquared > 1.0)
    discard;

  // -1 spray, -2 foam, -3 bubbles for debug
  outLinearDepth =
      W.debugClassification != 0u ? -(floor(vType + 0.5) + 1.0)
                                  : -vViewCenter.z;

  vec4 clip = pc.proj * vec4(vViewCenter, 1.0);
  gl_FragDepth = clip.z / clip.w;
}
