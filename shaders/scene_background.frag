#version 450
#extension GL_GOOGLE_include_directive : require

#include "screen.glsl"

layout(location = 0) in vec2 vUV;
layout(binding = 2) uniform sampler2D environmentMap;

layout(push_constant) uniform PC {
    vec4 camRight;
    vec4 camUp;
    vec4 camForward; // w = tan half fov
} pc;

layout(location = 0) out vec4 outColor;
layout(location = 1) out float outDepth;

void main() {
    vec2 ndc = vUV * 2.0 - 1.0;
    ndc.y = -ndc.y;
    float tanHalfFov = pc.camForward.w;
    vec3 rayDir = normalize(pc.camForward.xyz +
                            ndc.x * pc.camRight.xyz * tanHalfFov +
                            ndc.y * pc.camUp.xyz * tanHalfFov);
    outColor = vec4(texture(environmentMap, directionToEnvUV(rayDir)).rgb, 1.0);
    outDepth = NO_SURFACE_DEPTH / max(dot(rayDir, pc.camForward.xyz), 0.2);
}
