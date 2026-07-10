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
constexpr int numParticles = 2000;

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

// ---------------------------------------------------------------------------
// Fluid simulation
// ---------------------------------------------------------------------------
class FluidSim {
public:
  std::vector<Particle> particles;
  std::vector<float> densities; // cached density per particle
  std::vector<float3> viscosityForces;

  void initialize() {
    particles.resize(numParticles);
    densities.assign(numParticles, 0.0f);
    viscosityForces.assign(numParticles, float3(0.0f));

    // Two-column demo: place half the particles in a slab at the -X edge and
    // half at the +X edge, filling the full height/depth. Gravity pulls both
    // columns down; the fluid then flows inward and meets in the middle.
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
    // 1. apply gravity, predict lookahead positions used for SPH sampling
    for (int i = 0; i < numParticles; i++) {
      particles[i].velocity += gravity * deltaT;
      particles[i].predictedPosition =
          particles[i].position + particles[i].velocity * deltaT;
    }
    updateDensities();

    // 2. pressure force -> acceleration -> velocity
    for (int i = 0; i < numParticles; i++) {
      float3 pressureForce = calculatePressureForce(i);
      float3 pressureAcceleration = pressureForce / densities[i];
      particles[i].velocity += pressureAcceleration * deltaT;
    }

    // 2b. viscosity: pull each velocity toward the neighbourhood average
    for (int i = 0; i < numParticles; i++)
      viscosityForces[i] = calculateViscosityForce(i);
    for (int i = 0; i < numParticles; i++)
      particles[i].velocity += viscosityForces[i] * deltaT;

    // 3. advance positions and resolve wall collisions
    for (int i = 0; i < numParticles; i++) {
      particles[i].position += particles[i].velocity * deltaT;
      particles[i].resolveCollisions();
    }
  }

  void updateDensities() {
    for (int i = 0; i < numParticles; i++)
      densities[i] = calculateDensity(particles[i].predictedPosition);
  }

  // 3D "spiky" smoothing kernel: (r - d)^2 normalized over the support sphere.
  static float smoothingKernel(float radius, float dst) {
    if (dst >= radius)
      return 0.0f;
    float volume = (2.0f * PI * std::pow(radius, 5)) / 15.0f;
    float value = radius - dst;
    return (value * value) / volume;
  }

  float calculateDensity(const float3 &samplePoint) const {
    float density = 0.0f;
    for (const Particle &p : particles) {
      float dst = length(p.predictedPosition - samplePoint);
      float influence = smoothingKernel(smoothingRadius, dst);
      density += particleMass * influence;
    }
    return density;
  }

  // derivative of the spiky kernel wrt distance
  static float smoothingKernelDerivative(float dst, float radius) {
    if (dst >= radius)
      return 0.0f;
    float scale = 15.0f / (PI * std::pow(radius, 5));
    return (dst - radius) * scale; // negative inside the radius
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
    for (int i = 0; i < numParticles; i++) {
      if (i == particleIndex)
        continue;
      float3 offset = particles[i].predictedPosition -
                      particles[particleIndex].predictedPosition;
      float dst = length(offset);
      float3 dir;
      if (dst < Epsilon) {
        // coincident particles: push in a random direction on the unit sphere
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
    }
    return pressureForce;
  }

  // poly6-style kernel used for viscosity
  static float viscosityKernel(float radius, float dst) {
    if (dst >= radius)
      return 0.0f;
    float volume = (64.0f * PI * std::pow(radius, 9)) / 315.0f;
    float v = radius * radius - dst * dst;
    return (v * v * v) / volume;
  }

  float3 calculateViscosityForce(int particleIndex) const {
    float3 viscosityForce = float3(0.0f);
    const float3 pos = particles[particleIndex].predictedPosition;
    const float3 vel = particles[particleIndex].velocity;
    for (int i = 0; i < numParticles; i++) {
      if (i == particleIndex)
        continue;
      float dst = length(particles[i].predictedPosition - pos);
      float influence = viscosityKernel(smoothingRadius, dst);
      viscosityForce += (particles[i].velocity - vel) * influence;
    }
    return viscosityForce * viscosityStrength;
  }
};

} // namespace sph
