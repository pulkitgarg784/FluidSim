#version 450
#extension GL_GOOGLE_include_directive : require

#include "screen.glsl"

layout(location = 0) in vec3 vNormal;
layout(location = 1) in vec3 vWorldPos;
layout(location = 2) in float vLinearDepth;
layout(location = 3) in vec3 vAmbient;
layout(location = 4) in vec3 vDiffuse;
layout(location = 5) in vec3 vSpecular;
layout(location = 6) in float vShininess;
layout(location = 7) in vec3 vViewDirection;

layout(binding = 2) uniform sampler2D environmentMap;
layout(binding = 5) uniform sampler2D waterShadow;
layout(binding = 7) uniform sampler2D waterShadowFrontDepth;
layout(std140, binding = 6) uniform SceneLighting {
    mat4 lightVP;
    mat4 lightView;
    vec4 sunDirection;
    vec4 extinction;
    vec4 shadowParams; // x = ambient light left inside the shadow
} lighting;

layout(location = 0) out vec4 outColor;
layout(location = 1) out float outLinearDepth;

const vec3 SUN_COLOR = vec3(1.0, 0.96, 0.88);

// How much sunlight survives the water above this point. The shadow map holds
// fluid thickness from the light's view.
vec3 waterTransmission() {
    vec4 shadowClip = lighting.lightVP * vec4(vWorldPos, 1.0);
    vec2 shadowUV = shadowClip.xy * 0.5 + 0.5;
    if (any(lessThan(shadowUV, vec2(0.0))) ||
        any(greaterThan(shadowUV, vec2(1.0))))
        return vec3(1.0);

    float frontDepth = texture(waterShadowFrontDepth, shadowUV).r;
    float receiverDepth = -(lighting.lightView * vec4(vWorldPos, 1.0)).z;
    if (!hasSurface(frontDepth) || frontDepth >= receiverDepth - 0.02)
        return vec3(1.0);

    float thickness = texture(waterShadow, shadowUV).r;
    float ambient = lighting.shadowParams.x;
    return exp(-thickness * lighting.extinction.rgb) * (1.0 - ambient) +
           ambient;
}

void main() {
    vec3 N = normalize(vNormal);
    vec3 L = normalize(lighting.sunDirection.xyz);
    vec3 V = normalize(vViewDirection);
    vec3 H = normalize(L + V);

    vec3 diffuseEnvironment = texture(environmentMap, directionToEnvUV(N)).rgb;
    vec3 specularEnvironment =
        texture(environmentMap, directionToEnvUV(reflect(-V, N))).rgb;
    diffuseEnvironment /= vec3(1.0) + diffuseEnvironment;
    specularEnvironment /= vec3(1.0) + specularEnvironment;

    float specularLight =
        pow(max(dot(N, H), 0.0), clamp(vShininess, 1.0, 1024.0));
    vec3 shadow = waterTransmission();
    vec3 ambient = vAmbient * 0.35 + vDiffuse * diffuseEnvironment * 0.15;
    vec3 direct = vDiffuse * max(dot(N, L), 0.0) * SUN_COLOR * shadow;
    vec3 specular = vSpecular *
                    (specularEnvironment * 0.25 + specularLight * SUN_COLOR) *
                    shadow;

    outColor = vec4(ambient + direct + specular, 1.0);
    outLinearDepth = vLinearDepth;
}
