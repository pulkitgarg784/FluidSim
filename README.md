# Real-Time GPU Fluid Sim

A Vulkan application that simulates and renders water with Smoothed Particle Hydrodynamics (SPH). The fluid simulation, spatial neighbor search, and whitewater updates run in GPU compute shaders; the renderer reconstructs a smooth water surface from particles and shades it with reflection, refraction, depth absorption, and shadows.

## Features

- SPH fluid simulation with density, pressure, near-pressure, viscosity, gravity, and fixed-step integration.
- GPU spatial hash grid, reducing neighbor queries from all-pairs work to the particle's grid cell and its 26 neighbors.
- The default simulation uses 250,000 fluid particles and supports up to 150,000 whitewater particles; the primary-particle limit is 400,000.
- Screen-space fluid surface reconstruction: particle depth, bilateral depth blur, and normals derived from reconstructed depth.
- Fresnel blended environment reflection and refraction, plus Beer–Lambert depth absorption.
- Spray, foam, and bubble particles spawned from turbulence, surface exposure, and impacts.
- A scene mesh, fluid thickness shadowing, live material/physics controls, debugging views, and mouse-driven fluid interaction.


## Requirements

- CMake 3.16 or newer
- A C++17 compiler
- [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) , including `glslc`
- A Vulkan 1.4-capable GPU/driver with synchronization2, push descriptors, and support for 256 compute invocations per workgroup

GLFW, Dear ImGui, and the Vulkan radix-sort library are included under `external/`.

## Build and run

From the repository root:

```sh
cmake -S . -B build
cmake --build build
./build/FluidSim
```

### Runtime Command line options

```text
./build/FluidSim [--particles 1..400000] [--scene mesh.obj] [--scene-scale positive]
```

Examples:

```sh
# Default is with 250000 fluid particles and media/fluid_container.obj
./build/FluidSim

# Run with less particles for faster simulation
./build/FluidSim --particles 100000

# Use a different scene mesh and scale it to fit the simulation tank
./build/FluidSim --scene media/cornellbox.obj --scene-scale 10
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