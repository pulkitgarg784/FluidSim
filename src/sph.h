#pragma once

#include <cstdint>

#include "linalg.h"

namespace sph {

constexpr uint32_t referenceParticleCount = 75000u;
constexpr uint32_t defaultFluidParticleCount = 250000u;
constexpr uint32_t maxWhitewaterParticleCount = 150000u;
constexpr uint32_t maxParticles = 400000u;
constexpr linalg::aliases::float3 gravity(0.0f, -9.8f, 0.0f);
constexpr float collisionDamping = 0.2f;
constexpr float targetDensity = 2315.0f;
constexpr float pressureMultiplier = 16.0f;
constexpr float nearPressureMultiplier = 0.12f;
constexpr float viscosityStrength = 0.001f;
constexpr float epsilon = 5e-5f;

} // namespace sph
