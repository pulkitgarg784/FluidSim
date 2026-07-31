# Real-Time GPU Fluid Simulation

A Vulkan application that simulates and renders water with Smoothed Particle Hydrodynamics (SPH). The fluid simulation, spatial neighbor search, and whitewater updates run in GPU compute shaders; the renderer reconstructs a smooth water surface from particles and shades it with reflection, refraction, depth absorption, and shadows.

This project implements the main goals in the [proposal](proposal.md): GPU particle simulation, spatial hashing, screen-space surface reconstruction, Fresnel reflection/refraction, Beer–Lambert absorption, and secondary whitewater particles.

## Features

- SPH fluid simulation with density, pressure, near-pressure, viscosity, gravity, and fixed-step integration.
- GPU spatial hash grid with radix sorting, reducing neighbor queries from all-pairs work to the particle's grid cell and its 26 neighbors.
- Up to 400,000 primary particles; the default scene uses 250,000. Whitewater has a separate capacity of 150,000 particles.
- Screen-space fluid surface reconstruction: particle depth, bilateral depth blur, and normals derived from reconstructed depth.
- Fresnel-blended environment reflection and refraction, plus Beer–Lambert depth absorption.
- Spray, foam, and bubble particles spawned from turbulence, surface exposure, and impacts.
- A scene mesh, fluid thickness shadowing, live material/physics controls, debugging views, and mouse-driven fluid interaction.

## Requirements

- CMake 3.16 or newer
- A C++17 compiler
- Vulkan SDK, including `glslc`
- A Vulkan 1.4-capable GPU/driver with synchronization2, push descriptors, and support for 256 compute invocations per workgroup

GLFW, Dear ImGui, and the Vulkan radix-sort library are included under `external/`.

## Build and run

From the repository root:

```sh
cmake -S . -B build
cmake --build build -j
./build/CS488
```

The build compiles the GLSL shaders into `build/shaders/`; run the executable through CMake rather than copying it elsewhere so its generated shaders remain available.

### Command-line options

```text
./build/CS488 [--particles 1..400000] [--scene mesh.obj] [--scene-scale positive]
```

Examples:

```sh
# Default: 250,000 fluid particles and media/fluid_container.obj
./build/CS488

# A faster, lower-particle run (the selected count must still accommodate the grid)
./build/CS488 --particles 100000

# Render another OBJ scene mesh
./build/CS488 --scene media/cornellbox.obj --scene-scale 1.0
```

The scene mesh is rendered as the environment. Fluid collision currently uses the axis-aligned simulation tank, not triangle-level collision against the OBJ mesh.

## Controls

| Input | Action |
| --- | --- |
| Left mouse drag | Orbit the camera |
| `W` / `S` | Move camera forward / backward |
| `A` / `D` | Move camera left / right |
| `Q` / `Z` | Move camera up / down |
| Shift + left mouse | Attract nearby fluid at the cursor ray |
| Shift + right mouse | Repel nearby fluid at the cursor ray |
| Escape | Close the application |

The ImGui panels expose restart controls, SPH parameters, whitewater generation settings, reconstruction quality, water material, sunlight/shadow controls, and debug views. The debug panel can show whitewater classes (red spray, green foam, blue bubbles), particle velocity, or pressure.

## Simulation pipeline

Each fixed physics step is executed entirely on the GPU:

1. Apply gravity and optional cursor force; predict particle positions and compute spatial hash keys.
2. Radix-sort particle keys and indices, then gather sorted particle state.
3. Build per-cell start/end indices for the hash grid.
4. Estimate density and near density from particles in the current cell and its surrounding 26 cells.
5. Compute pressure forces and spawn whitewater from turbulence, kinetic energy, impacts, and wave crests.
6. Apply viscosity, integrate velocity and position, and resolve collisions with the tank boundaries.
7. Advect, classify, and retire secondary spray, foam, and bubble particles.

The simulation uses a 75 Hz fixed timestep. Its smoothing radius and mass scale with the requested particle count so the initial water volume retains roughly consistent behavior.

## Rendering pipeline

1. Render the scene mesh and environment background.
2. Render particles as camera-facing sphere impostors into a fluid depth buffer.
3. Bilaterally blur depth to reconstruct a continuous surface; derive normals from depth derivatives.
4. Accumulate and blur a thickness buffer for optical depth and fluid shadows.
5. Composite refraction, environment reflection, Schlick-style Fresnel blending, and Beer–Lambert absorption.
6. Render whitewater in a separate depth mask so it can appear as foam/spray over the reconstructed water.

## Project layout

| Path | Purpose |
| --- | --- |
| `src/main.cpp` | Application entry point, camera/input handling, CLI parsing, and initial fluid setup |
| `src/vk_renderer.h` | Vulkan renderer, compute/render passes, ImGui UI, scene loading, and resource management |
| `src/sph.h` | Default SPH constants and particle limits |
| `shaders/sph_*.comp` | SPH, spatial-grid, sorting support, and integration compute stages |
| `shaders/whitewater_*` | Secondary-particle update and rendering shaders |
| `shaders/fluid_*` | Particle depth, reconstruction blur, thickness, and compositing shaders |
| `media/` | OBJ scene assets, materials, textures, and HDR environment map |
| `proposal.md` | Original project proposal and technical objectives |

## Notes

- This is a real-time visual simulation. Default physical constants are tuned for stable, convincing motion rather than strict real-world scale.
- The available scene mesh changes the rendered environment; the fluid container remains the configured box used by the simulation.
