# Real-Time GPU Fluid Simulation

A Vulkan application that simulates and renders water with Smoothed Particle Hydrodynamics (SPH). The fluid simulation, spatial neighbor search, and whitewater updates run in GPU compute shaders; the renderer reconstructs a smooth water surface from particles and shades it with reflection, refraction, depth absorption, and shadows.

This project implements the main goals in the [proposal](proposal.md): GPU particle simulation, spatial hashing, screen-space surface reconstruction, Fresnel reflection/refraction, Beer–Lambert absorption, and secondary whitewater particles.

## Motivation

Assignment 3 was the first time I worked with particles, and after handing it in I kept going with it and optimized the particle code for fun. Once it could handle a large enough number of spheres, the result stopped reading as a collection of spheres and started to look like a liquid. That is where this project came from: I wanted to see how far that impression could be pushed, so I chose water simulation for the final project.

Building it on Vulkan with compute shaders came out of my earlier work with Vulkan and real-time GPU programming at SideFX Software. That experience is what made me think a real-time simulation was actually within reach rather than something to precompute, so the simulation, the neighbor search, and the whitewater all run on the GPU.

## Features

- SPH fluid simulation with density, pressure, near-pressure, viscosity, gravity, and fixed-step integration.
- GPU spatial hash grid, reducing neighbor queries from all-pairs work to the particle's grid cell and its 26 neighbors.
- The default simulation uses 250,000 fluid particles and supports up to 150,000 whitewater particles; the primary-particle limit is 400,000.
- Screen-space fluid surface reconstruction: particle depth, bilateral depth blur, and normals derived from reconstructed depth.
- Fresnel blended environment reflection and refraction, plus Beer–Lambert depth absorption.
- Spray, foam, and bubble particles spawned from turbulence, surface exposure, and impacts.
- A scene mesh, fluid thickness shadowing, live material/physics controls, debugging views, and mouse-driven fluid interaction.

## How the objectives are met

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
- **Extinction color:** Sets the per-channel absorption coefficient of the water.
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

## Implementation notes

### Algorithms, data structures, and complexity

Particle state lives in device-local storage buffers, one array per attribute rather than an array of structs: positions, velocities, predicted positions, densities (a `vec2` of density and near density), and the sorted copies used by the neighbor search. Everything is 16-byte aligned so the std430 layout in the shaders matches the `float4` arrays on the host.

A step runs the eight compute pipelines plus the library sort, in this order, with a barrier between each because every stage reads what the previous one wrote:

1. `sph_external.comp`: apply gravity and the cursor force, then predict a position with `x + v * dt`. Density and pressure are evaluated at the predicted position rather than the current one, which is what keeps the solver stable at this timestep.
2. Radix sort of the cell keys with particle indices as values, four passes over 8-bit digits.
3. `sph_reorder.comp`: gather position, velocity, and predicted position into cell order so the neighbor loops read contiguous memory.
4. `sph_start_reset.comp` and `sph_start_indices.comp`: because the keys are sorted, a run of equal keys is one cell, so each cell's `[start, end)` range is found by comparing each key with its neighbors.
5. `sph_density.comp`: density from the spiky kernel `15/(2*pi*h^5) * (h-r)^2` and near density from `15/(pi*h^6) * (h-r)^3`. The near-density term is Clavet et al.'s double density relaxation, and it is what prevents clumping without a large pressure stiffness.
6. `sph_pressure.comp`: symmetrized pressure gradient plus short-range near-pressure repulsion, and the whitewater spawn cues computed from the same neighbor loop.
7. `sph_viscosity_integrate.comp`: poly6-weighted (`315/(64*pi*h^9) * (h^2-r^2)^3`) velocity smoothing, integration, and tank collisions.
8. `whitewater_update.comp`: classify and advance the secondary particles.

All kernel normalization constants are computed once on the host in `main.cpp`, so the shaders only multiply.

The acceleration structure is a uniform grid with cell size equal to the smoothing radius, indexed directly as `(z * gy + y) * gx + x` rather than hashed, so distinct cells can never collide. The grid is padded by one cell on every side, which means the 27-cell neighborhood of any in-bounds particle is always inside the grid. At the default 250,000 particles it is 112x55x39, or 240,240 cells.

Per step this gives O(n) for prediction, reordering, and range building, O(n) for the sort (four fixed passes), O(c) for the cell reset where c is the cell count, and O(n*k) for the three neighbor passes, where k is the average number of particles within the smoothing radius. All-pairs neighbor search would be O(n^2): at 250,000 particles that is 6.25x10^10 pairs per step instead of roughly 10^7.

Whitewater uses a fixed-capacity pool with an atomic bump cursor and one state word per slot. A spawning thread claims a slot with `atomicCompSwap(state, 0, 1)` and skips it if it is already alive, so there is no lock and no compaction pass; retirement is an `atomicExchange` back to zero with a matching decrement of the live count. The cost is O(capacity) rather than O(alive), because both the update dispatch and the draw cover all 150,000 slots, and dead slots exit early or are clipped away.

On the rendering side, the impostor and thickness passes are O(n) instanced quads with overdraw, the bilateral blur is O(W*H*R) per axis per iteration where R is the projected particle radius clamped by **Max blur radius**, and the composite is O(W*H).

At the default settings the simulation holds about 35 MB of storage buffers, 24 MB of which is the six `vec4` particle arrays, plus roughly 66 KB of radix-sort scratch. The screen-space targets are separate and scale with the window size and **Render scale**.

### Platform dependence, constants, and configurability

- The application needs a Vulkan 1.4 device with `synchronization2`, push descriptors (required by the radix-sort library, not used directly by this code), and at least 256 invocations per compute workgroup. Only the workgroup limit is checked when picking a physical device.
- macOS is the only platform-specific path: portability enumeration and `VK_KHR_portability_subset` are enabled under `__APPLE__`, and CMake bakes the Vulkan library directory into the rpath so MoltenVK is found. Nothing else in the code is OS-specific.
- Some formats are assumed rather than queried: `D32_SFLOAT` for depth attachments, and `R32_SFLOAT` as a color attachment with linear filtering for the scene depth and shadow maps. The particle passes also push 176 bytes of push constants, while the Vulkan minimum is 128, so a device at exactly the guaranteed minimum would not work.
- Presentation prefers `IMMEDIATE`, then `MAILBOX`, then `FIFO`, so by default the frame rate is unsynchronized. Two frames are in flight.
- Constants are grouped by what owns them. [`src/sph.h`](src/sph.h) has the particle limits and default physical parameters; [`src/main.cpp`](src/main.cpp) has the tank size, the initial water block, the camera, the 1/75 s timestep, the field of view, and the cursor force; [`src/vk_renderer.h`](src/vk_renderer.h) has the rendering and whitewater defaults, two frames in flight, the 512x512 shadow map, the 256-thread workgroup size, and the sentinel depth written where no particle was drawn.
- A few values have to be stated on both sides of the C++/GLSL boundary: the workgroup size (`local_size_x`), the no-surface sentinel (`NO_SURFACE_DEPTH` in [`shaders/screen.glsl`](shaders/screen.glsl)), and the debug view numbering. The shared `.glsl` headers carry the uniform block layouts, and `static_assert`s on the host structs catch a layout mismatch at compile time.
- CMake passes `SHADER_DIR` and `ASSET_DIR` as compile definitions and copies `media/` next to the executable, so the binary does not depend on the working directory.
- Everything on the ImGui panels is live, including all SPH parameters and substeps. Particle count, scene mesh, and scene scale are start-up only.

### Input, output, and processing

- Input is the command line (`--particles`, `--scene`, `--scene-scale`), keyboard and mouse, an OBJ scene with its MTL file, and the radiance environment map at `media/uffizi_probe.hdr`.
- Pre-processing happens at build time and at start-up rather than per frame. `glslc` compiles every shader to SPIR-V during the build. The OBJ loader triangulates n-gons as a fan, resolves negative (relative) indices, generates face normals when the file has none, and flattens each material into per-vertex ambient/diffuse/specular/shininess, since the scene is drawn as one non-indexed vertex buffer. The initial water block is generated as an even lattice sized to the requested particle count.
- Output is on-screen only; there is no image or file export. The only GPU to CPU transfer is the live whitewater count, copied into a mapped buffer for the panel readout, so that number can lag by up to two frames.
- The screen-space chain is the post-processing: bilateral depth blur, thickness blur, then compositing. The environment map is HDR and is tone mapped only where the scene shader samples it (`x / (1 + x)`); final encoding is handled by the sRGB swapchain format.

### Sources, re-use, and acknowledgements

Third-party code, all under `external/` or clearly marked in `src/`:

- GLFW for windowing and input, and Dear ImGui for the panels, both as submodules.
- [`external/vulkan_radix_sort`](external/vulkan_radix_sort) provides the GPU radix sort used in step 2. It is the only step of the simulation pipeline I did not write; the eight compute stages around it are mine.
- [`src/linalg.h`](src/linalg.h), Sterling Orsten's public-domain single-header linear algebra library, as provided with the course.
- [`src/stb_image.h`](src/stb_image.h), Sean Barrett's public-domain image loader, used only to read the `.hdr` environment map.

Assets: the course scenes in `media/` (`cornellbox`, `teapot`, `bunny`, `sponza`, `testObj`) are unchanged, `fluid_container.obj` was made for this project, and `uffizi_probe.hdr` is Paul Debevec's Uffizi Gallery light probe.

The algorithms follow published descriptions, which I implemented rather than copied:

- SPH formulation and kernels: Koschier, Bender, Solenthaler and Teschner, *SPH Techniques for the Physics Based Simulation of Fluids and Solids*.
- Near pressure and double density relaxation: Clavet, Beaudoin and Poulin, *Particle-based Viscoelastic Fluid Simulation* (2005).
- Screen-space surface reconstruction: Green, *Screen Space Fluid Rendering* ([NVIDIA GDC 2010](https://developer.download.nvidia.com/presentations/2010/gdc/Direct3D_Effects.pdf)).
- Spray, foam, and bubble classification and spawn cues: Ihmsen, Akinci, Akinci and Teschner, [*Unified spray, foam and bubbles for particle-based fluids*](https://cg.informatik.uni-freiburg.de/publications/2012_CGI_sprayFoamBubbles.pdf) (2012).
- Fresnel: Schlick's approximation.

### Caveats, cautions, and assumptions

- The fluid only collides with the tank. The scene mesh is there for lighting and visual context, so water passes through it.
- At most one physics step is taken per frame. Below 75 fps the simulation runs in slow motion instead of trying to catch up, which keeps it stable but makes motion frame-rate dependent below that point. Substeps improve accuracy, not simulation speed.
- The SPH parameters are tuned for 250,000 particles. Smoothing radius and particle mass scale with the count automatically, but very low or very high counts can still need the panel sliders to stay stable.
- Whitewater costs the same whether zero or 150,000 particles are alive, because the update and the draw both cover the whole pool.
- The surface is reconstructed in screen space, with the usual consequences: normals are view-dependent, there is no internal refraction or caustics, and the thickness buffer counts overlapping particle discs rather than measuring a true path length.
- Fluid shadows come from thickness in light space and are refreshed every N frames, so fast motion can briefly show a stale shadow. There is no shadowing of the fluid by the scene mesh.
- `media/uffizi_probe.hdr` must be present; the renderer reports an error and exits if it cannot be loaded.
- The default present mode is unsynchronized, so tearing is expected unless the driver overrides it.
- The particle debug views replace the water rather than blending with it, since the composite pass outputs black in those modes and the particles are drawn over it.
- The 400,000 particle ceiling is a limit chosen for memory and frame time, not a hard technical one.
- Cursor interaction acts on the point where the cursor ray meets a camera-facing plane through the center of the tank, so it is only as accurate as that plane is to the water you are aiming at.

## Project layout

| Path | Purpose |
| --- | --- |
| `src/main.cpp` | Application entry point, camera/input handling, CLI parsing, and initial fluid setup |
| `src/vk_renderer.h` | Vulkan renderer, compute/render passes, ImGui UI, scene loading, and resource management |
| `src/sph.h` | Default SPH constants and particle limits |
| `shaders/sph_*.comp` | SPH, spatial-grid, sorting support, and integration compute stages |
| `shaders/whitewater_*` | Secondary-particle update and rendering shaders |
| `shaders/fluid_*` | Particle depth, reconstruction blur, thickness, and compositing shaders |
| `shaders/*.glsl` | Declarations shared between shaders, including the parameter block and grid addressing, the whitewater parameters and classification, the particle push constants, and the screen-space helpers |
| `media/` | OBJ scene assets, materials, textures, and HDR environment map |
| `proposal.md` | Original project proposal and technical objectives |

## Objectives

These are the objectives from the [proposal](proposal.md). [How the objectives are met](#how-the-objectives-are-met) lists the file that implements each one.

1. Implement a Smoothed Particle Hydrodynamics fluid simulation. It should calculate particle density, pressure, forces, and time integration to simulate fluid behaviour.
2. Implement a spatial hashing structure to improve efficiency for particle search, so that we can support around 100,000 particles for our simulation. This should aim to reduce complexity from O(n^2) to O(nk).
3. Use a GPU compute shader for the core physics simulation pass, so that we can run the simulation in parallel to achieve real-time performance.
4. Create a fluid surface from the particles by generating a depth map from the particles, and use bilateral filtering to get a smooth surface.
5. Implement physically based water surface rendering using environment reflection, screen-space refraction, and Fresnel blending.
6. Implement volumetric light attenuation to simulate depth based color using the Beer-Lambert Law.
7. Implement a whitewater particle system that can generate and render foam and bubbles.
8. Add user interactions where users can grab or push water away using physical forces based on mouse clicks.

Two of these ended up going further than the proposal asked. Objective 2 targeted roughly 100,000 particles, and the simulation runs 250,000 by default with a 400,000 ceiling. Objective 7 asked for foam and bubbles, and the whitewater system also classifies and simulates spray, with each of the three types getting its own motion.
