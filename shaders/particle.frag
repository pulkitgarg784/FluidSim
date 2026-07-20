#version 450

layout(location = 0) in vec2 vUV;
layout(location = 1) in float vSpeed;

layout(location = 0) out vec4 outColor;

void main() {
    float r2 = dot(vUV, vUV);
    if (r2 > 1.0)
        discard; // draw circle

    // sphere normals
    float nz = sqrt(max(0.0, 1.0 - r2));
    vec3 N = normalize(vec3(vUV, nz));
    vec3 L = normalize(vec3(0.4, 0.7, 0.6));
    float diff = max(dot(N, L), 0.0) * 0.8 + 0.2;

    // Color based on speed 
    float t = clamp(vSpeed * 0.5, 0.0, 1.0);
    vec3 slow = vec3(0.10, 0.30, 0.90);
    vec3 fast = vec3(0.90, 0.30, 0.10);
    vec3 col = mix(slow, fast, t) * diff;

    outColor = vec4(col, 1.0);
}
