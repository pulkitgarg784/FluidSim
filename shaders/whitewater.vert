#version 450
#extension GL_GOOGLE_include_directive : require

#include "particle_push.glsl"
#include "whitewater_params.glsl"

layout(location = 0) in vec2 inCorner;

layout(std430, binding = 15) readonly buffer WhiteParticles {
  WhiteParticle whiteParticles[];
};

layout(location = 0) out vec2 vUV;
layout(location = 1) out vec3 vViewCenter;
layout(location = 2) flat out float vType;

void main() {
  WhiteParticle particle = whiteParticles[gl_InstanceIndex];
  vec3 center = particle.positionLife.xyz;

  // Shrink over lifetime to fade out
  float fade = clamp(particle.positionLife.w / 3.0, 0.0, 1.0);
  float speedFade = mix(
      0.6, 1.0,
      clamp((length(particle.velocityScale.xyz) - 1.0) / 2.0, 0.0, 1.0));
  float radius = pc.params.x * W.renderScale * particleScale(particle) * fade *
                 speedFade;

  vec3 world = billboardPosition(center, inCorner, radius);
  gl_Position = pc.proj * pc.view * vec4(world, 1.0);
  // Push retired slots ou
  if (W.enabled == 0u || particle.positionLife.w <= 0.0)
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);

  vUV = inCorner;
  vViewCenter = (pc.view * vec4(center, 1.0)).xyz;
  vType = particleType(particle);
}
