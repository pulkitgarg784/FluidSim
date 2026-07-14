#version 450

layout(location = 0) in vec2 vUV;
layout(binding = 0) uniform sampler2D fluidDepth;

layout(push_constant) uniform PC {
    vec4 params; // x = near, y = far
} pc;

layout(location = 0) out vec4 outColor;

void main() {
    float d = texture(fluidDepth, vUV).r;
    if (d >= 1.0e4 * 0.5) { // no fluid
        outColor = vec4(0.02, 0.03, 0.06, 1.0);
        return;
    }
    // near = white, far = black
    float g = clamp(1.0 - (d - pc.params.x) / (pc.params.y - pc.params.x), 0.0, 1.0);
    outColor = vec4(vec3(g), 1.0);
}
