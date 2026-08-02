# Real-Time GPU Fluid Simulation

A Vulkan application that simulates and renders water with Smoothed Particle Hydrodynamics (SPH). The fluid simulation, spatial neighbor search, and whitewater updates run in GPU compute shaders; the renderer reconstructs a smooth water surface from particles and shades it with reflection, refraction, depth absorption, and shadows.

This project implements the main goals in the [proposal](proposal.md): GPU particle simulation, spatial hashing, screen-space surface reconstruction, Fresnel reflection/refraction, Beer–Lambert absorption, and secondary whitewater particles.

## Features

- SPH fluid simulation with density, pressure, near-pressure, viscosity, gravity, and fixed-step integration.
- GPU spatial hash grid, reducing neighbor queries from all-pairs work to the particle's grid cell and its 26 neighbors.
- The default simulation uses 250,000 fluid particles and supports up to 150,000 whitewater particles; the primary-particle limit is 400,000.
- Screen-space fluid surface reconstruction: particle depth, bilateral depth blur, and normals derived from reconstructed depth.
- Fresnel blended environment reflection and refraction, plus Beer–Lambert depth absorption.
- Spray, foam, and bubble particles spawned from turbulence, surface exposure, and impacts.
- A scene mesh, fluid thickness shadowing, live material/physics controls, debugging views, and mouse-driven fluid interaction.

## Proposal objectives

1. **SPH fluid simulation:** [`shaders/sph_density.comp`](shaders/sph_density.comp) estimates density and near density, [`shaders/sph_pressure.comp`](shaders/sph_pressure.comp) computes pressure forces, and [`shaders/sph_viscosity_integrate.comp`](shaders/sph_viscosity_integrate.comp) applies viscosity and integrates position and velocity. Default physical parameters are defined in [`src/sph.h`](src/sph.h).
2. **Spatial hashing acceleration:** [`shaders/sph_external.comp`](shaders/sph_external.comp) assigns each predicted particle position to a grid cell. [`src/vk_renderer.h`](src/vk_renderer.h) radix-sorts the cell keys, then `sph_reorder.comp`, `sph_start_reset.comp`, and `sph_start_indices.comp` build contiguous per-cell particle ranges so each particle only searches its cell and the 26 adjacent cells.
3. **GPU compute simulation:** [`src/vk_renderer.h`](src/vk_renderer.h) creates and dispatches eight compute pipelines for external forces, sorting support, grid construction, SPH forces, integration, and whitewater. Particle state remains in Vulkan storage buffers throughout each simulation step.
4. **Fluid surface reconstruction:** [`shaders/fluid_depth.vert`](shaders/fluid_depth.vert) and [`shaders/fluid_depth.frag`](shaders/fluid_depth.frag) render particles as sphere impostors into a depth texture. [`shaders/fluid_blur.frag`](shaders/fluid_blur.frag) performs separable bilateral filtering, and the composite shader reconstructs surface normals from neighboring depth samples.
5. **Reflection and refraction:** [`shaders/fluid_composite.frag`](shaders/fluid_composite.frag) refracts the rendered scene, samples the HDR environment for reflections, and blends them using a Fresnel term. [`shaders/scene_background.frag`](shaders/scene_background.frag) renders the same environment map behind the scene.
6. **Depth-based light attenuation:** [`shaders/fluid_thickness.frag`](shaders/fluid_thickness.frag) accumulates particle thickness, which is smoothed by [`shaders/fluid_thickness_blur.frag`](shaders/fluid_thickness_blur.frag). The composite shader applies Beer–Lambert transmittance, while [`src/vk_renderer.h`](src/vk_renderer.h) also renders thickness from the light's view for fluid shadows.
7. **Whitewater particles:** [`shaders/sph_pressure.comp`](shaders/sph_pressure.comp) spawns secondary particles from trapped air, kinetic energy, impacts, and exposed wave crests. [`shaders/whitewater_update.comp`](shaders/whitewater_update.comp) classifies and advances spray, foam, and bubbles; `whitewater.vert` and `whitewater.frag` render them.
8. **Mouse-based fluid interaction:** [`src/main.cpp`](src/main.cpp) casts a ray from the cursor and sends an interaction point, radius, and signed strength to the renderer. [`shaders/sph_external.comp`](shaders/sph_external.comp) applies the resulting distance-weighted force to nearby particles: Shift + left mouse attracts water and Shift + right mouse repels it.

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
cmake --build build
./build/CS488
```

### Runtime Command line options

```text
./build/CS488 [--particles 1..400000] [--scene mesh.obj] [--scene-scale positive]
```

Examples:

```sh
# Default is with 250000 fluid particles and media/fluid_container.obj
./build/CS488

# Run with less particles for faster simulation
./build/CS488 --particles 100000

# Use a different scene mesh and scale it to fit the simulation tank
./build/CS488 --scene media/cornellbox.obj --scene-scale 10
```

## Notes

- The scene mesh is rendered as the environment. Fluid collision currently only uses the tank. The scene is only for visualization and lighting. Also running with different particle count might require adjusting the SPH parameters in the ImGui panel for stable simulation. The default parameters are tuned for 250,000 particles, and while they do adjust automatically for different particle counts, they may not be optimal for very low or very high particle counts.

- The simulation uses 1 physics step per frame, with a fixed timestep of 1/75 seconds. However, if the computer can support it, I recommend setting it to 2 or more. That greatly improves the stability of the simulation, especially with higher particle counts. You can change this in the ImGui panel.

## Controls

| Input | Action |
| --- | --- |
| Left mouse drag | Orbit the camera |
| `W` / `S` | Move camera forward / backward |
| `A` / `D` | Move camera left / right |
| `Q` / `Z` | Move camera up / down |
| Shift + left mouse | Attract nearby fluid towards the cursor |
| Shift + right mouse | Repel nearby fluid away from the cursor |
| Escape | Quit |

## ImGui panels

The semi-transparent panels begin in a resolution-aware layout, with Simulation and Rendering stacked on the left and Debug visualization in the upper-right. Their positions and sizes reset when the application starts, but they remain movable and resizable during the run.

### Simulation controls

- **Particles / FPS:** Displays the current fluid, whitewater, and total particle counts and the UI framerate.
- **Restart simulation:** Restores the initial fluid positions and velocities and clears all whitewater.
- **Gravity:** Sets downward acceleration.
- **Target density:** Sets the rest density that the pressure solver tries to maintain.
- **Pressure stiffness:** Controls the strength of the density-error pressure response.
- **Near pressure:** Adds short-range repulsion to reduce particle clumping.
- **Viscosity:** Controls how strongly neighboring particle velocities are smoothed.
- **Collision damping:** Controls velocity retained after collision with a tank boundary.
- **Simulation substeps:** Splits each 1/75-second physics step into smaller steps for stability. Higher values increase simulation accuracy but reduce framerate.
- **Reset water preset:** Restores the default SPH values and one simulation substep.
- **Enable whitewater:** Enables or disables whitewater generation and simulation.
- **Spawn rate:** Scales the number of secondary particles created by qualifying fluid events.
- **Whitewater size:** Changes the rendered size of spray, foam, and bubble particles.
- **Reset whitewater preset:** Restores the default whitewater settings and restarts its spawn ramp.

### Rendering controls

- **Render scale:** Sets the resolution of the screen-space fluid buffers relative to the window.
- **Max blur radius:** Caps the projected bilateral-filter radius in pixels.
- **Blur strength:** Controls the spatial smoothing used to join particle depths into a surface.
- **Depth difference strength:** Prevents blur across large depth discontinuities; higher values preserve sharper depth edges.
- **Blur iterations:** Repeats the horizontal and vertical surface-smoothing passes.
- **Refraction strength:** Controls the screen-space offset used to sample the scene through water.
- **Absorption scale:** Scales Beer–Lambert attenuation with fluid thickness.
- **Extinction colour:** Sets the per-channel absorption coefficient of the water.
- **Base reflectance:** Sets the minimum Fresnel reflectance when viewing the surface head-on.
- **Sun angle / elevation:** Sets the direction used for scene lighting and fluid shadows.
- **Shadow ambient light:** Sets the minimum light remaining inside fluid shadows.
- **Shadow update interval:** Updates the fluid shadow map every N frames; a larger interval reduces rendering work. Default set to every 2 frames.

### Debug visualization

- **Off:** Uses the normal water renderer.
- **Whitewater:** Shows spray in red, foam in green, and bubbles in blue.
- **Particle velocity:** Shows still particles in blue, medium speeds in green, and fast particles in red.
- **Particle pressure:** Shows negative pressure in blue, near-zero pressure in white, and positive pressure in red.

## Simulation pipeline

Each fixed physics step is executed entirely on the GPU:

1. Apply gravity and optional cursor force. Predict particle positions and compute spatial hash keys.
2. Radix sort particle keys and indices, then gather sorted particle state.
3. Build per cell start/end indices for the hash grid.
4. Estimate density and near density from particles in the current cell and its surrounding 26 cells.
5. Compute pressure forces and spawn whitewater from turbulence, kinetic energy, impacts, and wave crests.
6. Apply viscosity, integrate velocity and position, and resolve collisions with the tank boundaries.
7. Simulate, classify, and retire secondary spray, foam, and bubble particles.

The simulation uses a 75 Hz fixed timestep, so that the fluid motion is consistent regardless of the rendering framerate.

## Rendering pipeline

1. Render the scene mesh and environment background.
2. Render particles as camera-facing sphere impostors into a fluid depth buffer.
3. Bilaterally blur depth to reconstruct a continuous surface; derive normals from depth derivatives.
4. Accumulate and blur a thickness buffer for optical depth and fluid shadows.
5. Composite refraction, environment reflection, Fresnel blending, and Beer–Lambert absorption.
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
