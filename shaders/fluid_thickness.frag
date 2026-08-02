#version 450

layout(location = 0) in vec2 vUV;
layout(location = 0) out float outThickness;

void main() {
    if (dot(vUV, vUV) > 1.0)
        discard;
    // contribute to scene depth
    outThickness = 0.1;
}
