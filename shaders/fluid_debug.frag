#version 450
#extension GL_GOOGLE_include_directive : require

#include "particle_push.glsl"

layout(location = 0) in vec2 vUV;
layout(location = 1) in vec3 vViewCenter;
layout(location = 2) flat in float vValue;

layout(location = 0) out vec4 outColor;

const float VELOCITY_MAX = 5.0;
const float PRESSURE_MAX = 5000.0;

vec3 velocityRamp(float t) {
  t = clamp(t, 0.0, 1.0) * 4.0;
  if (t < 1.0)
    return mix(vec3(0.02, 0.05, 0.35), vec3(0.0, 0.75, 1.0), t);
  if (t < 2.0)
    return mix(vec3(0.0, 0.75, 1.0), vec3(0.1, 1.0, 0.25), t - 1.0);
  if (t < 3.0)
    return mix(vec3(0.1, 1.0, 0.25), vec3(1.0, 0.9, 0.0), t - 2.0);
  return mix(vec3(1.0, 0.9, 0.0), vec3(1.0, 0.0, 0.0), t - 3.0);
}

vec3 pressureRamp(float pressure) {
  float t = clamp(pressure / PRESSURE_MAX, -1.0, 1.0);
  vec3 neutral = vec3(0.95);
  return t < 0.0 ? mix(neutral, vec3(0.05, 0.2, 1.0), -t)
                 : mix(neutral, vec3(1.0, 0.05, 0.0), t);
}

void main() {
  float r2 = dot(vUV, vUV);
  if (r2 > 1.0)
    discard;

  vec3 viewPos =
      vViewCenter + pc.params.x * vec3(vUV.x, vUV.y, sqrt(1.0 - r2));
  vec4 clip = pc.proj * vec4(viewPos, 1.0);
  gl_FragDepth = clip.z / clip.w;

  // params.y is debug mode
  bool pressureMode = int(pc.params.y + 0.5) == 3;
  outColor = vec4(pressureMode ? pressureRamp(vValue)
                               : velocityRamp(vValue / VELOCITY_MAX),
                  1.0);
}
