#ifndef SCREEN_GLSL
#define SCREEN_GLSL

const float PI = 3.14159265358979323846;

const float NO_SURFACE_DEPTH = 1.0e4;

bool hasSurface(float depth) { return depth < NO_SURFACE_DEPTH * 0.5; }

// angluar light probe map
vec2 directionToEnvUV(vec3 dir) {
  dir = normalize(dir);
  float xy = length(dir.xy);
  if (xy < 1.0e-6)
    return vec2(0.5);
  float r = acos(clamp(-dir.z, -1.0, 1.0)) / (PI * xy);
  return clamp(vec2(dir.x, -dir.y) * r * 0.5 + 0.5, 0.0, 1.0);
}

#endif
