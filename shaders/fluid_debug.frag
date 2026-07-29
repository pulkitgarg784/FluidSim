#version 450

layout(location = 0) in vec2 vUV;
layout(location = 1) in vec3 vViewCenter;
layout(location = 2) flat in float vValue;
layout(location = 3) flat in float vRadius;

layout(push_constant) uniform PC {
  mat4 view;
  mat4 proj;
  vec4 camRight;
  vec4 camUp;
  int debugMode;
} pc;

layout(location = 0) out vec4 outColor;

const float VELOCITY_MAX = 5.0;
const float PRESSURE_MAX = 5000.0;

vec3 velocityRamp(float t) {
  t = clamp(t, 0.0, 1.0);
  if (t < 0.25)
    return mix(vec3(0.02, 0.05, 0.35), vec3(0.0, 0.75, 1.0), t * 4.0);
  if (t < 0.5)
    return mix(vec3(0.0, 0.75, 1.0), vec3(0.1, 1.0, 0.25),
               (t - 0.25) * 4.0);
  if (t < 0.75)
    return mix(vec3(0.1, 1.0, 0.25), vec3(1.0, 0.9, 0.0),
               (t - 0.5) * 4.0);
  return mix(vec3(1.0, 0.9, 0.0), vec3(1.0, 0.0, 0.0),
             (t - 0.75) * 4.0);
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
      vViewCenter + vRadius * vec3(vUV.x, vUV.y, sqrt(1.0 - r2));
  vec4 clip = pc.proj * vec4(viewPos, 1.0);
  gl_FragDepth = clip.z / clip.w;

  vec3 color = pc.debugMode == 3
                   ? pressureRamp(vValue)
                   : velocityRamp(vValue / VELOCITY_MAX);
  outColor = vec4(color, 1.0);
}
