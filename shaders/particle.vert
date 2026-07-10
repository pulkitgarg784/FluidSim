#version 450

layout(location = 0) in vec2 inCorner;
layout(location = 1) in vec3 inPos;
layout(location = 2) in float inSpeed;

layout(push_constant) uniform PushConstants {
    mat4 viewProj;
    vec4 camRight;
    vec4 camUp;
    vec4 params;   // x: particle radius
} pc;

layout(location = 0) out vec2 vUV;
layout(location = 1) out float vSpeed;

void main() {
    vec3 world = inPos +
                 (inCorner.x * pc.camRight.xyz + inCorner.y * pc.camUp.xyz) *
                     pc.params.x;
    gl_Position = pc.viewProj * vec4(world, 1.0);
    vUV = inCorner;
    vSpeed = inSpeed;
}
