#version 450

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
    vec4 shadowParams;
} lighting;

layout(location = 0) out vec4 outColor;
layout(location = 1) out float outLinearDepth;

const float PI = 3.14159265358979323846;

vec2 directionToEnvUV(vec3 dir) {
    dir = normalize(dir);
    float xy = length(dir.xy);
    if (xy < 1.0e-6)
        return vec2(0.5);
    float r = acos(clamp(-dir.z, -1.0, 1.0)) / (PI * xy);
    return clamp(vec2(dir.x, -dir.y) * r * 0.5 + 0.5, 0.0, 1.0);
}

void main() {
    vec3 N = normalize(vNormal);
    vec3 L = normalize(lighting.sunDirection.xyz);
    vec3 V = normalize(vViewDirection);
    vec3 H = normalize(L + V);
    float diffuseLight = max(dot(N, L), 0.0);
    float specularLight = pow(max(dot(N, H), 0.0),
                              clamp(vShininess, 1.0, 1024.0));
    vec3 reflected = reflect(-V, N);
    vec3 diffuseEnvironment = texture(environmentMap, directionToEnvUV(N)).rgb;
    vec3 specularEnvironment =
        texture(environmentMap, directionToEnvUV(reflected)).rgb;
    diffuseEnvironment /= vec3(1.0) + diffuseEnvironment;
    specularEnvironment /= vec3(1.0) + specularEnvironment;
    vec4 shadowClip = lighting.lightVP * vec4(vWorldPos, 1.0);
    vec2 shadowUV = shadowClip.xy * 0.5 + 0.5;
    bool insideShadowMap = all(greaterThanEqual(shadowUV, vec2(0.0))) &&
                           all(lessThanEqual(shadowUV, vec2(1.0)));
    float waterThickness = insideShadowMap ? texture(waterShadow, shadowUV).r : 0.0;
    float waterFrontDepth = insideShadowMap
        ? texture(waterShadowFrontDepth, shadowUV).r : 1.0e4;
    float receiverLightDepth = -(lighting.lightView * vec4(vWorldPos, 1.0)).z;

    bool waterIsAboveReceiver = waterFrontDepth < 9.0e3 &&
                                waterFrontDepth < receiverLightDepth - 0.02;
    vec3 waterTransmission = vec3(1.0);
    if (waterIsAboveReceiver) {
        waterTransmission = exp(-waterThickness * lighting.extinction.rgb);
        float ambientLight = lighting.shadowParams.x;
        waterTransmission = waterTransmission * (1.0 - ambientLight) + ambientLight;
    }
    vec3 sunColor = vec3(1.0, 0.96, 0.88);
    vec3 ambient = vAmbient * 0.35 + vDiffuse * diffuseEnvironment * 0.15;
    vec3 direct = vDiffuse * diffuseLight * sunColor * waterTransmission;
    vec3 specular = vSpecular *
                    (specularEnvironment * 0.25 + specularLight * sunColor) *
                    waterTransmission;
    outColor = vec4(ambient + direct + specular, 1.0);
    outLinearDepth = vLinearDepth;
}
