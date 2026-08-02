#version 450

layout(location = 0) in vec2 vUV;
layout(binding = 0) uniform sampler2D src;

layout(push_constant) uniform PC {
    vec2 dir;
    float maxScreenSpaceRadius;
} pc;

layout(location = 0) out float outThickness;

// Gaussian blur thcikness
void main() {
    float center = texture(src, vUV).r;
    if (center <= 1.0e-6) {
        outThickness = 0.0;
        return;
    }

    vec2 step = pc.dir / vec2(textureSize(src, 0));
    int radius = clamp(int(round(pc.maxScreenSpaceRadius * 0.35)), 1, 5);
    float sigma = max(float(radius) * 0.65, 0.75);
    float twoSigmaSquared = 2.0 * sigma * sigma;

    float sum = 0.0;
    float weights = 0.0;
    for (int i = -radius; i <= radius; ++i) {
        float weight = exp(-float(i * i) / twoSigmaSquared);
        sum += texture(src, vUV + step * float(i)).r * weight;
        weights += weight;
    }
    outThickness = sum / weights;
}
