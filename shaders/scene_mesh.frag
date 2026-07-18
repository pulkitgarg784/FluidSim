#version 450

layout(location = 0) in vec3 vNormal;
layout(location = 1) in vec3 vWorldPos;
layout(location = 2) in float vLinearDepth;

layout(location = 0) out vec4 outColor;
layout(location = 1) out float outLinearDepth;

void main() {
    vec3 N = normalize(vNormal);
    vec3 L = normalize(vec3(0.4, 0.8, 0.3));
    float light = 0.25 + 0.75 * max(dot(N, L), 0.0);
    float checker = mod(floor(vWorldPos.x) + floor(vWorldPos.z), 2.0);
    vec3 floorColor = mix(vec3(0.10, 0.12, 0.14), vec3(0.19, 0.21, 0.23), checker);
    bool isFloor = vNormal.y > 0.9;
    vec3 base = isFloor ? floorColor : vec3(0.42, 0.20, 0.08);
    outColor = vec4(base * light, 1.0);
    outLinearDepth = vLinearDepth;
}
