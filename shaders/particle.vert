#version 450

layout(location = 0) in vec2 inCorner;

layout(std430, binding = 0) readonly buffer Positions  { vec4 positions[];  };
layout(std430, binding = 1) readonly buffer Velocities { vec4 velocities[]; };

layout(push_constant) uniform PushConstants {
    mat4 viewProj;
    vec4 camRight;
    vec4 camUp;
    vec4 params; // x: particle radius
} pc;

layout(location = 0) out vec2 vUV;
layout(location = 1) out float vSpeed;

void main() {
    vec3 center = positions[gl_InstanceIndex].xyz;
    vec3 world = center +
                 (inCorner.x * pc.camRight.xyz + inCorner.y * pc.camUp.xyz) *
                     pc.params.x;
    gl_Position = pc.viewProj * vec4(world, 1.0);
    vUV = inCorner;
    vSpeed = length(velocities[gl_InstanceIndex].xyz);
}
