#version 450

layout(location = 0) in vec2 vUV;
layout(binding = 0) uniform sampler2D src;

layout(push_constant) uniform PC {
    vec2 dir;
    float depthDifferenceStrength;
    float maxScreenSpaceRadius;
    float strength;
    float particleRadius;
    float fillSilhouette;
} pc;

layout(location = 0) out float outDepth;

const float EMPTY = 1.0e4;
const float TAN_HALF_FOV = 0.41421356237;

void main() {
    float centerD = texture(src, vUV).r;
    if (centerD >= EMPTY * 0.5 && pc.fillSilhouette > 0.5) {
        for (int distance = 1; distance <= 2; ++distance) {
            float negativeD = texture(src, vUV - pc.dir * float(distance)).r;
            float positiveD = texture(src, vUV + pc.dir * float(distance)).r;
            bool negativeValid = negativeD < EMPTY * 0.5;
            bool positiveValid = positiveD < EMPTY * 0.5;
            if (negativeValid || positiveValid) {
                centerD = negativeValid && positiveValid
                              ? min(negativeD, positiveD)
                              : (negativeValid ? negativeD : positiveD);
                break;
            }
        }
    }
    if (centerD >= EMPTY * 0.5) {
        outDepth = EMPTY;
        return;
    }

    float sum = 0.0;
    float wsum = 0.0;
    float focalLengthPixels = 0.5 * float(textureSize(src, 0).y) / TAN_HALF_FOV;
    float projectedRadius = pc.particleRadius * focalLengthPixels /
                            max(centerD, 1.0e-4);
    float radius = clamp(projectedRadius, 1.0, pc.maxScreenSpaceRadius);
    int R = int(ceil(radius));
    float sigma = max(radius * pc.strength * 0.5, 0.5);
    float twoSigma2 = 2.0 * sigma * sigma;
    for (int i = -R; i <= R; i++) {
        vec2 uv = vUV + pc.dir * float(i);
        float d = texture(src, uv).r;
        if (d >= EMPTY * 0.5)
            continue; // no water
        float wSpatial = exp(-float(i * i) / twoSigma2);
        float dd = d - centerD;
        float wDepth = exp(-(dd * dd) * pc.depthDifferenceStrength);
        float w = wSpatial * wDepth;
        sum += d * w;
        wsum += w;
    }
    outDepth = (wsum > 0.0) ? sum / wsum : centerD;
}
