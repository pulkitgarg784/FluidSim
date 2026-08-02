#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace vkr {

struct SceneVertex {
  float position[3];
  float normal[3];
  float ambient[3];
  float diffuse[3];
  float specular[3];
  float shininess;
};
static_assert(sizeof(SceneVertex) == 64 &&
                  offsetof(SceneVertex, position) == 0 &&
                  offsetof(SceneVertex, normal) == 12 &&
                  offsetof(SceneVertex, ambient) == 24 &&
                  offsetof(SceneVertex, diffuse) == 36 &&
                  offsetof(SceneVertex, specular) == 48 &&
                  offsetof(SceneVertex, shininess) == 60,
              "SceneVertex must match the graphics vertex input layout");

std::vector<SceneVertex> loadObjWithMaterials(const std::string &path,
                                              float scale);

} // namespace vkr
