#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(push_constant) uniform PC {
    mat4 view;
    mat4 proj;
} pc;

layout(location = 0) out vec3 vNormal;
layout(location = 1) out vec3 vWorldPos;
layout(location = 2) out float vLinearDepth;

void main() {
    vec4 viewPos = pc.view * vec4(inPosition, 1.0);
    gl_Position = pc.proj * viewPos;
    vNormal = mat3(pc.view) * inNormal;
    vWorldPos = inPosition;
    vLinearDepth = -viewPos.z;
}
