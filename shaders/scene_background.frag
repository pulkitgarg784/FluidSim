#version 450

layout(location = 0) in vec2 vUV;

layout(binding = 2) uniform sampler2D environmentMap;

layout(push_constant) uniform PC {
    vec4 camRight;
    vec4 camUp;
    vec4 camForward;
} pc;

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265358979323846;
const float TAN_HALF_FOV = 0.41421356237;

vec2 directionToEnvUV(vec3 dir) {
    dir = normalize(dir);
    float denom = length(dir.xy);
    if (denom < 1.0e-6)
        return vec2(0.5, 0.5);

    float radius = acos(clamp(-dir.z, -1.0, 1.0)) / (PI * denom);
    return clamp(vec2(dir.x, -dir.y) * radius * 0.5 + 0.5, 0.0, 1.0);
}

void main() {
    vec2 ndc = vUV * 2.0 - 1.0;
    ndc.y = -ndc.y;

    vec3 rayDir = normalize(pc.camForward.xyz +
                            ndc.x * pc.camRight.xyz * TAN_HALF_FOV +
                            ndc.y * pc.camUp.xyz * TAN_HALF_FOV);
    vec3 env = texture(environmentMap, directionToEnvUV(rayDir)).rgb;
    outColor = vec4(env, 1.0);
}
