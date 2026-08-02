#version 450
#extension GL_GOOGLE_include_directive : require

#include "particle_push.glsl"

layout(location = 0) in vec2 vUV;
layout(location = 1) in vec3 vViewCenter;

layout(location = 0) out float outDepth;

void main() {
    float r2 = dot(vUV, vUV);
    if (r2 > 1.0)
        discard; // outside the sphere

    float radius = pc.params.x;
    vec3 viewPos = vViewCenter + radius * vec3(vUV.x, vUV.y, sqrt(1.0 - r2));
    outDepth = -viewPos.z;

    // use depth to put the nearest particle on top
    vec4 clip = pc.proj * vec4(viewPos, 1.0);
    gl_FragDepth = clip.z / clip.w;
}
