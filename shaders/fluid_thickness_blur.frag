#version 450

layout(location = 0) in vec2 vUV;
layout(binding = 0) uniform sampler2D src;

layout(push_constant) uniform PC {
    vec2 dir;
    float depthDifferenceStrength;
    float maxScreenSpaceRadius;
    float strength;
} pc;

layout(location = 0) out float outThickness;

void main() {
    ivec2 size = textureSize(src, 0);
    vec2 texel = 1.0 / vec2(size);
    int radius = clamp(int(round(pc.maxScreenSpaceRadius * 0.35)), 1, 5);
    float sigma = max(float(radius) * 0.65, 0.75);
    float sum = 0.0;
    float weights = 0.0;
    for (int y = -radius; y <= radius; ++y) {
        for (int x = -radius; x <= radius; ++x) {
            float w = exp(-float(x * x + y * y) / (2.0 * sigma * sigma));
            sum += texture(src, vUV + vec2(x, y) * texel).r * w;
            weights += w;
        }
    }
    outThickness = sum / weights;
}
