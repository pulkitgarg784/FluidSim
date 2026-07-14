#version 450

layout(location = 0) in vec2 inCorner;

layout(std430, binding = 0) readonly buffer Positions { vec4 positions[]; };

layout(push_constant) uniform PC {
    mat4 view;
    mat4 proj;
    vec4 camRight;
    vec4 camUp;
    vec4 params; // x = radius
} pc;

layout(location = 0) out vec2 vUV;
layout(location = 1) out vec3 vViewCenter;

void main() {
    vec3 center = positions[gl_InstanceIndex].xyz;
    vec3 world = center +
                 (inCorner.x * pc.camRight.xyz + inCorner.y * pc.camUp.xyz) *
                     pc.params.x;
    vec4 viewPos = pc.view * vec4(world, 1.0);
    gl_Position = pc.proj * viewPos;
    vUV = inCorner;
    vViewCenter = (pc.view * vec4(center, 1.0)).xyz;
}
