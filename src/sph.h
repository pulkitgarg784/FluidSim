#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "linalg.h"
using namespace linalg::aliases;

namespace sph {

constexpr float PI = 3.14159265358979f;

// integration
constexpr float deltaT = 0.002f;
constexpr float3 gravity = float3(0.0f, -9.8f, 0.0f);
constexpr int numParticles = 10000;

// SPH parameters
constexpr float smoothingRadius = 0.2f;      // kernel support radius
constexpr float particleMass = 1.0f;         // mass used for density accumulation
constexpr float collisionDamping = 0.9f;     // velocity kept on a wall bounce
constexpr float3 boundsSize = float3(3.0f, 1.0f, 1.0f); // simulation box extents
constexpr float targetDensity = 1193.0f;     // rest density the fluid relaxes to
constexpr float pressureMultiplier = 30.0f;  // stiffness (pressure per density error)
constexpr float viscosityStrength = 0.001f;  // neighbour velocity averaging
constexpr int simIterationsPerFrame = 3;     // physics substeps per rendered frame

constexpr float Epsilon = 5e-5f;

// Precomputed kernel normalization factors 
constexpr float sr2 = smoothingRadius * smoothingRadius;
constexpr float sr5 = sr2 * sr2 * smoothingRadius;
constexpr float sr9 = sr2 * sr2 * sr2 * sr2 * smoothingRadius;
constexpr float Pow2Scale = 15.0f / (2.0f * PI * sr5);
constexpr float Pow2GradScale = 15.0f / (PI * sr5);
constexpr float viscocityScale = 315.0f / (64.0f * PI * sr9);

namespace PCG32 {
static uint64_t mcg_state = 0xcafef00dd15ea5e5u; // must be odd
static uint64_t const multiplier = 6364136223846793005u;
inline uint32_t pcg32_fast() {
  uint64_t x = mcg_state;
  const unsigned count = (unsigned)(x >> 61);
  mcg_state = x * multiplier;
  x ^= x >> 22;
  return (uint32_t)(x >> (22 + count));
}
inline float rand() { return float(double(pcg32_fast()) / 4294967296.0); }
} // namespace PCG32

struct Particle {
  float3 position = float3(0.0f);
  float3 velocity = float3(0.0f);
  float3 predictedPosition = float3(0.0f); // lookahead used for SPH sampling
  float3 force = float3(0.0f);

  void reset() {
    position = float3(PCG32::rand() - 0.5f, PCG32::rand() - 0.5f,
                      PCG32::rand() - 0.5f);
    velocity = float3(0.0f);
    predictedPosition = position;
    force = float3(0.0f);
  }

  // reflect the velocity component when a wall is crossed
  void resolveCollisions() {
    const float3 halfBounds = boundsSize * 0.5f;
    for (int axis = 0; axis < 3; axis++) {
      if (std::abs(position[axis]) > halfBounds[axis]) {
        position[axis] =
            halfBounds[axis] * ((position[axis] > 0) ? 1.0f : -1.0f);
        velocity[axis] *= -collisionDamping;
      }
    }
  }
};

class FluidSim {
public:
  std::vector<Particle> particles;
  std::vector<float> densities; // cached density per particle
  std::vector<float3> viscosityForces;

  // Spatial hash for neighbour search. Each particle is assigned a cell keyS
  struct SpatialHash {
    uint32_t index;
    uint32_t key; // hashed cell key
  };
  std::vector<SpatialHash> spatialEntries; // one per particle, sorted by key
  std::vector<uint32_t> startIndices; // first entry index for each key, or UINT32_MAX

  void initialize() {
    particles.resize(numParticles);
    densities.assign(numParticles, 0.0f);
    viscosityForces.assign(numParticles, float3(0.0f));
    spatialEntries.assign(numParticles, SpatialHash{0u, 0u});
    startIndices.assign(numParticles, UINT32_MAX);

    const float3 half = boundsSize * 0.5f;
    const float slab = boundsSize.x * 0.2f; // slab thickness at each end
    for (int i = 0; i < numParticles; i++) {
      Particle &p = particles[i];
      const bool leftEdge = (i < numParticles / 2);
      const float x = leftEdge ? (-half.x + PCG32::rand() * slab)
                               : (half.x - PCG32::rand() * slab);
      const float y = (PCG32::rand() - 0.5f) * boundsSize.y;
      const float z = (PCG32::rand() - 0.5f) * boundsSize.z;
      p.position = float3(x, y, z);
      p.velocity = float3(0.0f);
      p.predictedPosition = p.position;
      p.force = float3(0.0f);
    }
  }

  // advance a few stable substeps per rendered frame
  void step() {
    for (int iter = 0; iter < simIterationsPerFrame; iter++)
      simulationStep();
  }

  void simulationStep() {
    // gravity + lookahead predictedPosition 
    for (int i = 0; i < numParticles; i++) {
      particles[i].velocity += gravity * deltaT;
      particles[i].predictedPosition =
          particles[i].position + particles[i].velocity * deltaT;
    }

    // rebuild the neighbour grid on the predicted positions
    updateSpatialHash();
    updateDensities();

    // pressure + acceleration + velocity 
    for (int i = 0; i < numParticles; i++) {
      float3 pressureForce = calculatePressureForce(i);
      float3 pressureAcceleration = pressureForce / densities[i];
      particles[i].velocity += pressureAcceleration * deltaT;
    }

    // viscosity. pull towards neighbour avergae
    for (int i = 0; i < numParticles; i++)
      viscosityForces[i] = calculateViscosityForce(i);
    for (int i = 0; i < numParticles; i++)
      particles[i].velocity += viscosityForces[i] * deltaT;

    // set position and check collision
    for (int i = 0; i < numParticles; i++) {
      particles[i].position += particles[i].velocity * deltaT;
      particles[i].resolveCollisions();
    }
  }

  void updateDensities() {
    for (int i = 0; i < numParticles; i++)
      densities[i] = calculateDensity(particles[i].predictedPosition);
  }

  // cell size = smoothingRadius
  static int3 cellCoord(const float3 &p) {
    return int3((int)std::floor(p.x / smoothingRadius),
                (int)std::floor(p.y / smoothingRadius),
                (int)std::floor(p.z / smoothingRadius));
  }

  static uint32_t keyFromCell(int cx, int cy, int cz) {
    uint32_t a = (uint32_t)(int32_t)cx * 15823u;
    uint32_t b = (uint32_t)(int32_t)cy * 9737333u;
    uint32_t c = (uint32_t)(int32_t)cz * 440817757u;
    return (a + b + c) % (uint32_t)numParticles;
  }

  // Assign each particle a cell key, sort by key, and record bucket starts.
  void updateSpatialHash() {
    for (int i = 0; i < numParticles; i++) {
      int3 c = cellCoord(particles[i].predictedPosition);
      spatialEntries[i].index = (uint32_t)i;
      spatialEntries[i].key = keyFromCell(c.x, c.y, c.z);
    }
    std::sort(spatialEntries.begin(), spatialEntries.end(),
              [](const SpatialHash &a, const SpatialHash &b) {
                return a.key < b.key;
              });
    std::fill(startIndices.begin(), startIndices.end(), UINT32_MAX);
    for (int i = 0; i < numParticles; i++) {
      uint32_t key = spatialEntries[i].key;
      uint32_t prev = (i == 0) ? UINT32_MAX : spatialEntries[i - 1].key;
      if (key != prev)
        startIndices[key] = (uint32_t)i;
    }
  }

  template <typename Fn>
  void forEachNeighbour(const float3 &point, Fn &&fn) const {
    const int3 center = cellCoord(point);
    // can be 27 at most surroudning cells
    uint32_t keys[27];
    int keyCount = 0;
    for (int dz = -1; dz <= 1; dz++)
      for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) {
          uint32_t key = keyFromCell(center.x + dx, center.y + dy, center.z + dz);
          bool seen = false;
          for (int k = 0; k < keyCount; k++)
            if (keys[k] == key) {
              seen = true;
              break;
            }
          if (!seen)
            keys[keyCount++] = key;
        }

    for (int k = 0; k < keyCount; k++) {
      uint32_t key = keys[k];
      uint32_t start = startIndices[key];
      if (start == UINT32_MAX) // invalid cell
        continue;
      for (uint32_t e = start; e < (uint32_t)numParticles; e++) {
        if (spatialEntries[e].key != key)
          break;
        fn((int)spatialEntries[e].index);
      }
    }
  }

  static float smoothingKernel(float radius, float dst) {
    if (dst >= radius)
      return 0.0f;
    float value = radius - dst;
    return value * value * Pow2Scale;
  }

  float calculateDensity(const float3 &samplePoint) const {
    float density = 0.0f;
    forEachNeighbour(samplePoint, [&](int j) {
      float dst = length(particles[j].predictedPosition - samplePoint);
      density += particleMass * smoothingKernel(smoothingRadius, dst);
    });
    return density;
  }

  static float smoothingKernelDerivative(float dst, float radius) {
    if (dst >= radius)
      return 0.0f;
    return (dst - radius) * Pow2GradScale; // negative inside the radius
  }

  static float convertDensityToPressure(float density) {
    float densityError = density - targetDensity;
    return densityError * pressureMultiplier;
  }

  static float calculateSharedPressure(float densityA, float densityB) {
    float pressureA = convertDensityToPressure(densityA);
    float pressureB = convertDensityToPressure(densityB);
    return (pressureA + pressureB) * 0.5f;
  }

  float3 calculatePressureForce(int particleIndex) const {
    float3 pressureForce = float3(0.0f);
    const float3 samplePoint = particles[particleIndex].predictedPosition;
    forEachNeighbour(samplePoint, [&](int i) {
      if (i == particleIndex)
        return;
      float3 offset = particles[i].predictedPosition - samplePoint;
      float dst = length(offset);
      float3 dir;
      if (dst < Epsilon) {
        // push overlapping particles in random direction on unit spehere
        float z = PCG32::rand() * 2.0f - 1.0f;
        float a = PCG32::rand() * 2.0f * PI;
        float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
        dir = float3(r * std::cos(a), r * std::sin(a), z);
      } else {
        dir = offset / dst;
      }
      float slope = smoothingKernelDerivative(dst, smoothingRadius);
      float density = densities[i];
      float sharedPressure =
          calculateSharedPressure(density, densities[particleIndex]);
      pressureForce += sharedPressure * dir * slope * particleMass / density;
    });
    return pressureForce;
  }

  static float viscosityKernel(float radius, float dst) {
    if (dst >= radius)
      return 0.0f;
    float v = radius * radius - dst * dst;
    return v * v * v * viscocityScale;
  }

  float3 calculateViscosityForce(int particleIndex) const {
    float3 viscosityForce = float3(0.0f);
    const float3 pos = particles[particleIndex].predictedPosition;
    const float3 vel = particles[particleIndex].velocity;
    forEachNeighbour(pos, [&](int i) {
      if (i == particleIndex)
        return;
      float dst = length(particles[i].predictedPosition - pos);
      float influence = viscosityKernel(smoothingRadius, dst);
      viscosityForce += (particles[i].velocity - vel) * influence;
    });
    return viscosityForce * viscosityStrength;
  }
};

} // namespace sph
