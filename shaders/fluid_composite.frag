#version 450

layout(location = 0) in vec2 vUV;
layout(binding = 0) uniform sampler2D fluidDepth;

layout(push_constant) uniform PC {
    vec2 invTexel;
    float tanHalfFov;
    float aspect;
} pc;

layout(location = 0) out vec4 outColor;

const float EMPTY = 1.0e4;

vec3 viewPos(vec2 uv, float d) {
    float ndcX = uv.x * 2.0 - 1.0;
    float ndcY = uv.y * 2.0 - 1.0;
    float vx = ndcX * pc.aspect * pc.tanHalfFov * d;
    float vy = -ndcY * pc.tanHalfFov * d;
    return vec3(vx, vy, -d);
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
        outColor = vec4(0.02, 0.03, 0.06, 1.0); // background
        return;
    }

    vec3 P = viewPos(vUV, d);
    vec3 ddx = axisDeriv(vUV, d, P, vec2(pc.invTexel.x, 0.0));
    vec3 ddy = axisDeriv(vUV, d, P, vec2(0.0, pc.invTexel.y));
    vec3 normal = cross(ddx, ddy);
    vec3 N = dot(normal, normal) > 1.0e-10
                 ? normalize(normal)
                 : vec3(0.0, 0.0, 1.0);
    if (N.z < 0.0)
        N = -N;

    vec3 L = normalize(vec3(0.5, 0.8, 0.6));
    vec3 V = normalize(-P);
    vec3 H = normalize(L + V);
    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), 56.0);

    vec3 waterColor = vec3(0.10, 0.40, 0.85);
    vec3 col = waterColor * (0.25 + 0.75 * diff) + vec3(1.0) * spec * 0.7;
    outColor = vec4(col, 1.0);
}
