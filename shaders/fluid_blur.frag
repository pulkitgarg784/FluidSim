#version 450

layout(location = 0) in vec2 vUV;
layout(binding = 0) uniform sampler2D src;

layout(push_constant) uniform PC {
    vec2 dir;
    float depthDifferenceStrength;
    float maxScreenSpaceRadius;
    float strength;
} pc;

layout(location = 0) out float outDepth;

void main() {
    float centerD = texture(src, vUV).r;
    if (centerD >= 1.0e4 * 0.5) {
        outDepth = centerD; // no water
        return;
    }

    float sum = 0.0;
    float wsum = 0.0;
    float radius = pc.maxScreenSpaceRadius;
    int R = int(ceil(radius));
    float sigma = max(radius * pc.strength, 1e-4);
    float sigma2 = sigma * sigma * 0.5;
    for (int i = -R; i <= R; i++) {
        vec2 uv = vUV + pc.dir * float(i);
        float d = texture(src, uv).r;
        if (d >= 1.0e4 * 0.5)
            continue; // no water
        float wSpatial = exp(-float(i * i) / sigma2);
        float dd = d - centerD;
        float wDepth = exp(-(dd * dd) * pc.depthDifferenceStrength);
        float w = wSpatial * wDepth;
        sum += d * w;
        wsum += w;
    }
    outDepth = (wsum > 0.0) ? sum / wsum : centerD;
}
