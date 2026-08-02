#version 450
#extension GL_GOOGLE_include_directive : require

#include "particle_push.glsl"

layout(location = 0) in vec2 inCorner;

layout(std430, binding = 0) readonly buffer Positions { vec4 positions[]; };

layout(location = 0) out vec2 vUV;
layout(location = 1) out vec3 vViewCenter;

// One billboard quad per particle
void main() {
    vec3 center = positions[gl_InstanceIndex].xyz;
    vec3 world = billboardPosition(center, inCorner, pc.params.x);
    gl_Position = pc.proj * pc.view * vec4(world, 1.0);
    vUV = inCorner;
    vViewCenter = (pc.view * vec4(center, 1.0)).xyz;
}
