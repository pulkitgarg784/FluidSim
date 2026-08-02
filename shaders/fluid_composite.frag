#version 450
#extension GL_GOOGLE_include_directive : require

#include "screen.glsl"

layout(location = 0) in vec2 vUV;

layout(binding = 0) uniform sampler2D fluidDepth;
layout(binding = 1) uniform sampler2D sceneColor;
layout(binding = 2) uniform sampler2D environmentMap;
layout(binding = 3) uniform sampler2D sceneDepth;
layout(binding = 4) uniform sampler2D fluidThickness;
layout(binding = 8) uniform sampler2D whitewaterDepth;

layout(push_constant) uniform PC {
    vec4 camRight;
    vec4 camUp;
    vec4 camForward;
    // x = refraction scale, y = absorption scale, z = base reflectance, w = tan half pov
    vec4 material;
    vec4 extinction;
} pc;

layout(location = 0) out vec4 outColor;

const float WATER_IOR = 1.333;

vec3 viewPos(vec2 uv, float depth) {
    vec2 size = vec2(textureSize(fluidDepth, 0));
    float tanHalfFov = pc.camForward.w;
    vec2 ndc = uv * 2.0 - 1.0;
    return vec3(ndc.x * (size.x / size.y) * tanHalfFov * depth,
                -ndc.y * tanHalfFov * depth, -depth);
}

vec3 toWorld(vec3 v) {
    mat3 basis = mat3(pc.camRight.xyz, pc.camUp.xyz, -pc.camForward.xyz);
    return normalize(basis * v);
}

// Pick the better one-sided derivative
vec3 axisDeriv(vec2 uv, float d, vec3 P, vec2 step) {
    float dp = texture(fluidDepth, uv + step).r;
    float dn = texture(fluidDepth, uv - step).r;
    bool forward = hasSurface(dp);
    bool backward = hasSurface(dn);

    if (!forward && !backward)
        return vec3(0.0);
    if (forward && !backward)
        return viewPos(uv + step, dp) - P;
    if (backward && !forward)
        return P - viewPos(uv - step, dn);
    return abs(dp - d) < abs(dn - d) ? viewPos(uv + step, dp) - P
                                      : P - viewPos(uv - step, dn);
}

void main() {
    // Skip water if in debug view
    int debugMode = int(floor(pc.material.w + 0.5));
    if (debugMode == 2 || debugMode == 3) {
        outColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    float depth = texture(fluidDepth, vUV).r;
    vec3 sceneCol = texture(sceneColor, vUV).rgb;
    float sceneDepthValue = texture(sceneDepth, vUV).r;
    bool sceneVisible = hasSurface(sceneDepthValue);
    float whitewaterZ = texture(whitewaterDepth, vUV).r;

    // -1 spray, -2 foam, -3 bubbles, -100 background.
    if (whitewaterZ < 0.0) {
        int type = int(floor(-whitewaterZ + 0.5)) - 1;
        vec3 debugColor = type == 0 ? vec3(1.0, 0.0, 0.0)
                        : type == 1 ? vec3(0.0, 1.0, 0.0)
                        : type == 2 ? vec3(0.0, 0.0, 1.0)
                                    : vec3(0.0);
        outColor = vec4(debugColor, 1.0);
        return;
    }

    bool whitewaterVisible = hasSurface(whitewaterZ) &&
                             !(sceneVisible && sceneDepthValue <= whitewaterZ);
    float whitewater = whitewaterVisible ? 1.0 : 0.0;

    if (!hasSurface(depth) || (sceneVisible && sceneDepthValue <= depth)) {
        outColor = vec4(mix(sceneCol, vec3(1.0), whitewater), 1.0);
        return;
    }

    vec2 texel = 1.0 / vec2(textureSize(fluidDepth, 0));
    vec3 surface = viewPos(vUV, depth);
    vec3 normal = cross(axisDeriv(vUV, depth, surface, vec2(texel.x, 0.0)),
                        axisDeriv(vUV, depth, surface, vec2(0.0, texel.y)));
    vec3 N = dot(normal, normal) > 1.0e-10 ? normalize(normal)
                                           : vec3(0.0, 0.0, 1.0);
    vec3 V = normalize(-surface);
    if (dot(N, V) < 0.0)
        N = -N;

    vec3 reflection =
        texture(environmentMap, directionToEnvUV(toWorld(reflect(-V, N)))).rgb;

    // Screen space refraction
    vec3 refracted = refract(-V, N, 1.0 / WATER_IOR);
    vec2 refractUV = clamp(vUV + refracted.xy / max(abs(refracted.z), 0.18) *
                                     pc.material.x,
                           vec2(0.0), vec2(1.0));
    float refractSceneDepth = texture(sceneDepth, refractUV).r;
    if (hasSurface(refractSceneDepth) && refractSceneDepth <= depth)
        refractUV = vUV;
    vec3 behind =
        mix(texture(sceneColor, refractUV).rgb, vec3(1.0), whitewater);
    if (whitewaterVisible && whitewaterZ < depth)
        reflection = vec3(1.0);

    // Fresnel
    float fresnel = mix(pc.material.z, 1.0,
                        pow(1.0 - clamp(dot(N, V), 0.0, 1.0), 5.0));

    // Beer Lambert over the accumulated particle thickness. 
    // extinction factory controls body color of the water.
    float thickness = texture(fluidThickness, vUV).r;
    vec3 transmittance = exp(-pc.extinction.rgb * thickness * pc.material.y);
    const vec3 waterBody = vec3(0.015, 0.045, 0.085);
    vec3 through = behind * transmittance + waterBody * (1.0 - transmittance);

    outColor = vec4(mix(through, reflection, fresnel), 1.0);
}
