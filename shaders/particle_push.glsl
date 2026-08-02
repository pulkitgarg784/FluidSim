#ifndef PARTICLE_PUSH_GLSL
#define PARTICLE_PUSH_GLSL

layout(push_constant) uniform PC {
  mat4 view;
  mat4 proj;
  vec4 camRight;
  vec4 camUp;
  vec4 params; // x = particle radius, y = debug mode
} pc;

vec3 billboardPosition(vec3 center, vec2 corner, float radius) {
  return center +
         (corner.x * pc.camRight.xyz + corner.y * pc.camUp.xyz) * radius;
}

#endif
