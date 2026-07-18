#version 450

layout(location = 0) in vec2 vUV;
layout(binding = 0) uniform sampler2D fluidDepth;
layout(binding = 1) uniform sampler2D sceneColor;
layout(binding = 2) uniform sampler2D environmentMap;

layout(push_constant) uniform PC {
    vec4 camRight;
    vec4 camUp;
    vec4 camForward;
} pc;

layout(location = 0) out vec4 outColor;

const float EMPTY = 1.0e4;
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
    bool vp = dp < EMPTY * 0.5;
    bool vn = dn < EMPTY * 0.5;

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
    float d = texture(fluidDepth, vUV).r;
    if (d >= EMPTY * 0.5) {
        outColor = texture(sceneColor, vUV);
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
    vec2 refrUV = clamp(vUV + refrOffset * 0.10, vec2(0.0), vec2(1.0));
    vec3 refraction = texture(sceneColor, refrUV).rgb;

    float fresnel = pow(1.0 - clamp(dot(N, V), 0.0, 1.0), 5.0);
    float F0 = 0.02;
    fresnel = mix(F0, 1.0, fresnel);

    float thickness = clamp(d * 0.04, 0.0, 1.0);
    vec3 absorption = exp(-vec3(0.08, 0.03, 0.015) * thickness * 6.0);
    vec3 waterTint = vec3(0.03, 0.18, 0.26);
    vec3 refracted = mix(refraction * absorption, waterTint, 0.12 * thickness);

    vec3 col = mix(refracted, reflection, fresnel);
    outColor = vec4(col, 1.0);
}
