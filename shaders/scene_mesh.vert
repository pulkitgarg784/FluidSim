#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inAmbient;
layout(location = 3) in vec3 inDiffuse;
layout(location = 4) in vec3 inSpecular;
layout(location = 5) in float inShininess;

layout(push_constant) uniform PC {
    mat4 view;
    mat4 proj;
} pc;

layout(location = 0) out vec3 vNormal;
layout(location = 1) out vec3 vWorldPos;
layout(location = 2) out float vLinearDepth;
layout(location = 3) out vec3 vAmbient;
layout(location = 4) out vec3 vDiffuse;
layout(location = 5) out vec3 vSpecular;
layout(location = 6) out float vShininess;
layout(location = 7) out vec3 vViewDirection;

void main() {
    vec4 viewPos = pc.view * vec4(inPosition, 1.0);
    gl_Position = pc.proj * viewPos;
    vNormal = inNormal;
    vWorldPos = inPosition;
    vLinearDepth = -viewPos.z;
    vAmbient = inAmbient;
    vDiffuse = inDiffuse;
    vSpecular = inSpecular;
    vShininess = inShininess;
    vViewDirection = inverse(pc.view)[3].xyz - inPosition;
}
