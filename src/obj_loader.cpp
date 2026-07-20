#include "obj_loader.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "linalg.h"

using namespace linalg::aliases;

namespace vkr {
namespace {

struct ObjMaterial {
  float3 ambient = float3(0.1f, 0.1f, 0.1f);
  float3 diffuse = float3(0.8f, 0.8f, 0.8f);
  float3 specular = float3(0.0f, 0.0f, 0.0f);
  float shininess = 1.0f;
};

std::string trim(std::string value) {
  auto notSpace = [](unsigned char c) { return !std::isspace(c); };
  value.erase(value.begin(),
              std::find_if(value.begin(), value.end(), notSpace));
  value.erase(std::find_if(value.rbegin(), value.rend(), notSpace).base(),
              value.end());
  return value;
}

std::unordered_map<std::string, ObjMaterial>
loadMtl(const std::filesystem::path &path) {
  std::ifstream file(path);
  if (!file)
    throw std::runtime_error("Failed to open MTL: " + path.string());

  std::unordered_map<std::string, ObjMaterial> materials;
  ObjMaterial *current = nullptr;
  std::string line;
  while (std::getline(file, line)) {
    std::istringstream stream(line);
    std::string tag;
    stream >> tag;
    if (tag == "newmtl") {
      std::string name;
      std::getline(stream, name);
      name = trim(name);
      if (!name.empty())
        current = &materials[name];
    } else if (current && tag == "Ka") {
      stream >> current->ambient.x >> current->ambient.y >> current->ambient.z;
    } else if (current && tag == "Kd") {
      stream >> current->diffuse.x >> current->diffuse.y >> current->diffuse.z;
    } else if (current && tag == "Ks") {
      stream >> current->specular.x >> current->specular.y >> current->specular.z;
    } else if (current && tag == "Ns") {
      stream >> current->shininess;
    }
  }
  return materials;
}

} // namespace

std::vector<SceneVertex> loadObjWithMaterials(const std::string &path,
                                              float scale) {
  std::ifstream file(path);
  if (!file)
    throw std::runtime_error("Failed to open OBJ: " + path);

  std::vector<float3> positions;
  std::vector<float3> normals;
  std::vector<SceneVertex> vertices;
  std::unordered_map<std::string, ObjMaterial> materials;
  const ObjMaterial defaultMaterial{};
  std::string currentMaterial;
  const std::filesystem::path objDirectory =
      std::filesystem::path(path).parent_path();

  auto index = [](int i, size_t count) {
    return i > 0 ? i - 1 : static_cast<int>(count) + i;
  };
  auto store = [](float (&destination)[3], const float3 &value) {
    destination[0] = value.x;
    destination[1] = value.y;
    destination[2] = value.z;
  };

  std::string line;
  while (std::getline(file, line)) {
    std::istringstream stream(line);
    std::string tag;
    stream >> tag;
    if (tag == "mtllib") {
      std::string filename;
      std::getline(stream, filename);
      auto loaded = loadMtl(objDirectory / trim(filename));
      materials.insert(loaded.begin(), loaded.end());
    } else if (tag == "usemtl") {
      std::getline(stream, currentMaterial);
      currentMaterial = trim(currentMaterial);
    } else if (tag == "v") {
      float3 position;
      stream >> position.x >> position.y >> position.z;
      positions.push_back(position);
    } else if (tag == "vn") {
      float3 normal;
      stream >> normal.x >> normal.y >> normal.z;
      normals.push_back(normalize(normal));
    } else if (tag == "f") {
      struct FaceVertex {
        int position = 0;
        int normal = 0;
      };
      std::vector<FaceVertex> face;
      std::string token;
      while (stream >> token) {
        FaceVertex vertex{};
        size_t firstSlash = token.find('/');
        size_t lastSlash = token.rfind('/');
        vertex.position = std::stoi(token.substr(0, firstSlash));
        if (firstSlash != std::string::npos && lastSlash != firstSlash &&
            lastSlash + 1 < token.size()) {
          vertex.normal = std::stoi(token.substr(lastSlash + 1));
        }
        face.push_back(vertex);
      }

      auto material = materials.find(currentMaterial);
      const ObjMaterial &mtl =
          material != materials.end() ? material->second : defaultMaterial;
      for (size_t i = 1; i + 1 < face.size(); ++i) {
        FaceVertex triangle[3] = {face[0], face[i], face[i + 1]};
        float3 trianglePositions[3];
        for (int corner = 0; corner < 3; ++corner) {
          trianglePositions[corner] =
              positions.at(index(triangle[corner].position, positions.size())) *
              scale;
        }
        float3 fallbackNormal =
            normalize(cross(trianglePositions[1] - trianglePositions[0],
                            trianglePositions[2] - trianglePositions[0]));
        for (int corner = 0; corner < 3; ++corner) {
          float3 normal = triangle[corner].normal
                              ? normals.at(index(triangle[corner].normal,
                                                 normals.size()))
                              : fallbackNormal;
          SceneVertex vertex{};
          store(vertex.position, trianglePositions[corner]);
          store(vertex.normal, normal);
          store(vertex.ambient, mtl.ambient);
          store(vertex.diffuse, mtl.diffuse);
          store(vertex.specular, mtl.specular);
          vertex.shininess = std::max(mtl.shininess, 1.0f);
          vertices.push_back(vertex);
        }
      }
    }
  }
  return vertices;
}

} // namespace vkr
