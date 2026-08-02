#version 450
#extension GL_GOOGLE_include_directive : require

#include "particle_push.glsl"
#include "whitewater_params.glsl"

layout(location = 0) in vec2 vUV;
layout(location = 1) in vec3 vViewCenter;
layout(location = 2) flat in float vType;

layout(location = 0) out float outLinearDepth;

void main() {
  float r2 = dot(vUV, vUV);
  if (r2 > 1.0)
    discard;

  outLinearDepth = W.debugClassification != 0u ? -(floor(vType + 0.5) + 1.0)
                                               : -vViewCenter.z;

  vec4 clip = pc.proj * vec4(vViewCenter, 1.0);
  gl_FragDepth = clip.z / clip.w;
}
