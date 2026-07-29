#version 450

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
    // x = refraction scale, y = absorption scale, z = base reflectance
    vec4 material;
    vec4 extinction;
} pc;

layout(location = 0) out vec4 outColor;

const float NO_SURFACE_DEPTH = 1.0e4;
const float TAN_HALF_FOV = 0.41421356237;
const float PI = 3.14159265358979323846;

vec3 viewPos(vec2 uv, float d) {
    ivec2 size = textureSize(fluidDepth, 0);
    float aspect = float(size.x) / float(size.y);
    float ndcX = uv.x * 2.0 - 1.0;
    float ndcY = uv.y * 2.0 - 1.0;
    float vx = ndcX * aspect * TAN_HALF_FOV * d;
    float vy = -ndcY * TAN_HALF_FOV * d;
    return vec3(vx, vy, -d);
}

vec2 directionToEnvUV(vec3 dir) {
    dir = normalize(dir);
    float denom = length(dir.xy);
    if (denom < 1.0e-6)
        return vec2(0.5, 0.5);

    float radius = acos(clamp(-dir.z, -1.0, 1.0)) / (PI * denom);
    return clamp(vec2(dir.x, -dir.y) * radius * 0.5 + 0.5, 0.0, 1.0);
}

vec3 toWorld(vec3 v) {
    mat3 basis = mat3(pc.camRight.xyz, pc.camUp.xyz, -pc.camForward.xyz);
    return normalize(basis * v);
}

// Pick the better one-sided derivative
vec3 axisDeriv(vec2 uv, float d, vec3 P, vec2 step) {
    float dp = texture(fluidDepth, uv + step).r;
    float dn = texture(fluidDepth, uv - step).r;
    bool vp = dp < NO_SURFACE_DEPTH * 0.5;
    bool vn = dn < NO_SURFACE_DEPTH * 0.5;

    if (vp && !vn)
        return viewPos(uv + step, dp) - P;
    if (!vp && vn)
        return P - viewPos(uv - step, dn);
    if (!vp && !vn)
        return vec3(0.0);

    vec3 fwd = viewPos(uv + step, dp) - P;
    vec3 bwd = P - viewPos(uv - step, dn);
    return abs(dp - d) < abs(dn - d) ? fwd : bwd;
}

void main() {
    int debugMode = int(floor(pc.material.w + 0.5));
    if (debugMode == 2 || debugMode == 3) {
        outColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    float d = texture(fluidDepth, vUV).r;
    vec3 sceneCol = texture(sceneColor, vUV).rgb;
    float sceneD = texture(sceneDepth, vUV).r;
    bool hasSceneSurface = sceneD < NO_SURFACE_DEPTH * 0.5;
    float whitewaterSample = texture(whitewaterDepth, vUV).r;

    // -1 spray, -2 foam, -3 bubbles, -100 background.
    if (whitewaterSample < 0.0) {
        int particleType = int(floor(-whitewaterSample + 0.5)) - 1;
        vec3 debugColor = particleType == 0
                              ? vec3(1.0, 0.0, 0.0)
                              : (particleType == 1
                                     ? vec3(0.0, 1.0, 0.0)
                                     : (particleType == 2
                                            ? vec3(0.0, 0.0, 1.0)
                                            : vec3(0.0)));
        outColor = vec4(debugColor, 1.0);
        return;
    }

    float foamDepth = whitewaterSample;
    bool hasWhitewater = foamDepth < NO_SURFACE_DEPTH * 0.5;
    if (hasWhitewater && hasSceneSurface && sceneD <= foamDepth)
        hasWhitewater = false;
    float whitewater = hasWhitewater ? 1.0 : 0.0;

    if (d >= NO_SURFACE_DEPTH * 0.5) {
        outColor = vec4(mix(sceneCol, vec3(1.0), whitewater), 1.0);
        return;
    }

    if (hasSceneSurface && sceneD <= d) {
        outColor = vec4(mix(sceneCol, vec3(1.0), whitewater), 1.0);
        return;
    }

    vec2 invTexel = 1.0 / vec2(textureSize(fluidDepth, 0));
    vec3 P = viewPos(vUV, d);
    vec3 ddx = axisDeriv(vUV, d, P, vec2(invTexel.x, 0.0));
    vec3 ddy = axisDeriv(vUV, d, P, vec2(0.0, invTexel.y));
    vec3 normal = cross(ddx, ddy);
    vec3 N = dot(normal, normal) > 1.0e-10
                 ? normalize(normal)
                 : vec3(0.0, 0.0, 1.0);
    vec3 V = normalize(-P);
    if (dot(N, V) < 0.0)
        N = -N;

    vec3 reflView = reflect(-V, N);
    vec3 reflWorld = toWorld(reflView);
    vec3 reflection = texture(environmentMap, directionToEnvUV(reflWorld)).rgb;

    vec3 refrView = refract(-V, N, 1.0 / 1.333);
    vec2 refrOffset = refrView.xy / max(abs(refrView.z), 0.18);
    vec2 refrUV = clamp(vUV + refrOffset * pc.material.x, vec2(0.0), vec2(1.0));

    // avoid refracting across a foreground object
    float refrSceneD = texture(sceneDepth, refrUV).r;
    bool refractsIntoScene = refrSceneD < NO_SURFACE_DEPTH * 0.5;
    if (refractsIntoScene && refrSceneD <= d)
        refrUV = vUV;
    vec3 refraction = texture(sceneColor, refrUV).rgb;
    refraction = mix(refraction, vec3(1.0), whitewater);
    if (hasWhitewater && foamDepth < d)
        reflection = vec3(1.0);

    float fresnel = pow(1.0 - clamp(dot(N, V), 0.0, 1.0), 5.0);
    float F0 = pc.material.z;
    fresnel = mix(F0, 1.0, fresnel);

    // accumulated particle depth
    float thickness = texture(fluidThickness, vUV).r;

    // beer lambert
    vec3 transmittance = exp(-pc.extinction.rgb * thickness * pc.material.y);

    // Dark, blue-leaning water body color for deeper regions.
    vec3 waterBody = vec3(0.015, 0.045, 0.085);
    vec3 refracted = refraction * transmittance + waterBody * (1.0 - transmittance);

    vec3 col = mix(refracted, reflection, fresnel);
    outColor = vec4(col, 1.0);
}
