#pragma once

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
static_assert(sizeof(SceneVertex) == sizeof(float) * 16);

std::vector<SceneVertex> loadObjWithMaterials(const std::string &path,
                                              float scale);

} // namespace vkr
