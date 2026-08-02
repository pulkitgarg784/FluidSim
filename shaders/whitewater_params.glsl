#ifndef WHITEWATER_PARAMS_GLSL
#define WHITEWATER_PARAMS_GLSL

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

struct WhiteParticle {
  vec4 positionLife;
  vec4 velocityScale;
};

const float SPRAY = 0.0;
const float FOAM = 1.0;
const float BUBBLE = 2.0;

float particleType(WhiteParticle p) { return floor(p.velocityScale.w * 0.5); }
float particleScale(WhiteParticle p) {
  return p.velocityScale.w - particleType(p) * 2.0;
}

#endif
