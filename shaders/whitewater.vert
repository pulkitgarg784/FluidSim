#version 450

layout(location = 0) in vec2 inCorner;

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
} W;
struct WhiteParticle {
  vec4 positionLife;
  vec4 velocityScale;
};
layout(std430, binding = 15) readonly buffer WhiteParticles {
  WhiteParticle whiteParticles[];
};

layout(push_constant) uniform PC {
  mat4 view;
  mat4 proj;
  vec4 camRight;
  vec4 camUp;
  vec4 params;
} pc;

layout(location = 0) out vec2 vUV;
layout(location = 1) out vec3 vViewCenter;
layout(location = 2) flat out float vActive;
layout(location = 3) flat out float vType;

void main() {
  WhiteParticle particle = whiteParticles[gl_InstanceIndex];
  bool isAlive = W.enabled != 0u && particle.positionLife.w > 0.0;
  float dissolveScale = clamp(particle.positionLife.w / 3.0, 0.0, 1.0);
  float speedScale =
      mix(0.6, 1.0, clamp((length(particle.velocityScale.xyz) - 1.0) / 2.0,
                          0.0, 1.0));
  float renderType = floor(particle.velocityScale.w * 0.5);
  float visualScale = particle.velocityScale.w - renderType * 2.0;
  float radius = pc.params.x * W.renderScale * visualScale *
                 dissolveScale * speedScale;
  vec3 center = particle.positionLife.xyz;
  vec3 world =
      center + (inCorner.x * pc.camRight.xyz + inCorner.y * pc.camUp.xyz) *
                   radius;
  vec4 viewPos = pc.view * vec4(world, 1.0);
  gl_Position = pc.proj * viewPos;
  if (!isAlive)
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
  vUV = inCorner;
  vViewCenter = (pc.view * vec4(center, 1.0)).xyz;
  vActive = isAlive ? 1.0 : 0.0;
  vType = renderType;
}
