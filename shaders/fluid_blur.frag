#version 450
#extension GL_GOOGLE_include_directive : require

#include "screen.glsl"

layout(location = 0) in vec2 vUV;
layout(binding = 0) uniform sampler2D src;

layout(push_constant) uniform PC {
    vec2 dir;
    float depthDifferenceStrength;
    float maxScreenSpaceRadius;
    float strength;
    float particleRadius;
    float fillSilhouette;
    float tanHalfFov;
} pc;

layout(location = 0) out float outDepth;

// Bilateral blur along one axis
void main() {
    vec2 step = pc.dir / vec2(textureSize(src, 0));
    float center = texture(src, vUV).r;

    if (!hasSurface(center) && pc.fillSilhouette > 0.5) {
        for (int distance = 1; distance <= 2; ++distance) {
            float back = texture(src, vUV - step * float(distance)).r;
            float front = texture(src, vUV + step * float(distance)).r;
            if (hasSurface(back) || hasSurface(front)) {
                center = min(hasSurface(back) ? back : NO_SURFACE_DEPTH,
                             hasSurface(front) ? front : NO_SURFACE_DEPTH);
                break;
            }
        }
    }
    if (!hasSurface(center)) {
        outDepth = NO_SURFACE_DEPTH;
        return;
    }

    // Use world space size for blur
    float focalLengthPixels =
        0.5 * float(textureSize(src, 0).y) / pc.tanHalfFov;
    float projected =
        pc.particleRadius * focalLengthPixels / max(center, 1.0e-4);
    float radius = clamp(projected, 1.0, pc.maxScreenSpaceRadius);
    float sigma = max(radius * pc.strength * 0.5, 0.5);
    float twoSigmaSquared = 2.0 * sigma * sigma;

    float sum = 0.0;
    float weights = 0.0;
    int taps = int(ceil(radius));
    for (int i = -taps; i <= taps; ++i) {
        float depth = texture(src, vUV + step * float(i)).r;
        if (!hasSurface(depth))
            continue;
        float difference = depth - center;
        float weight =
            exp(-float(i * i) / twoSigmaSquared) *
            exp(-difference * difference * pc.depthDifferenceStrength);
        sum += depth * weight;
        weights += weight;
    }
    outDepth = weights > 0.0 ? sum / weights : center;
}
