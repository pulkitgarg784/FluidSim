# Project Proposal

## Summary
The goal of this project is to develop a real-time water simulation and rendering system capable of simulating and visualizing fluids using particles.  In addition to physically plausible fluid motion, the project will also generate visually convincing water  including reflection, refraction, foam generation, and bubble effects.
The simulation will also support interactions between fluid and solid meshes, allowing objects to interact with the simulation. 

The final application will implement most of the physics calculations, neighbor search sorting algorithms, and water surface reconstruction on the GPU. Visually, the system will feature continuous water surfaces, environment reflections/refractions, secondary white water particle systems (spray, foam, and bubbles), and volumetric light attenuation.

## Technical Outline
To achieve the performance and visual goals, most of the core data structures and algorithms will be on the GPU using SSBO (Shader storage buffer objects) and use Compute Shaders.

### Particles
Particles will be switched over from C++ structs to a GLSL buffer which will probably store postion, velocity, force, density, and pressure for each.

### SPH simulation
The physics will be implemented using Smoothed Particle Hydrodynamics (D. Koschier, J. Bender, B. Solenthaler & M. Teschner / SPH Techniques for the Physics Based Simulation of Fluids and Solids). Each particle is interpolated using a smoothing kernel over a given radius to create the simulation.
The local density $\rho_i$ is calculated based on its neighbouring particle $j$ based on its mass and distance from our current particle $i$. Then using the ideal gas equation, we can calculate the pressure for each particle as the density difference times the gas constant. Finally we can use the pressure gradients, viscosity, and gravity to calculate the total Force acting on each particle which can then be applied using a time intergration pass to update position and velocity similar to A3.

### Acceleration Structure
Since we will need each particle to look at every other particle, it will have an $O(n^2)$ complexity which will slow our simulation. Instead, we will use a spatial hashing acceleration structure to map particle coordinate to a 1D grid cell index, and use a GPU compute shader radix fort to rearrange the particle identifiers sequentially, along with a bucket to represent each grid cell. Neighbour lookups will be in the cell/bucket in which the particle is and the surrounding cells, hence 27 cells, which will give average complexity of $O(nk)$.

### Surface Reconstruction
To generate a smooth surface from the particles, we can first render them as camera facing quads, project those onto a sphere, then implement a bilateral gaussian filter on the depth texture to get a smooth surface. Then for each pixel, we can also calculate screen space normals from the derivatives of the smoothed depth buffer. (https://developer.download.nvidia.com/presentations/2010/gdc/Direct3D_Effects.pdf)

### Fresnal Effects
For the environment reflection and refractions, we will use an environment map and sample it to get reflection and refraction, and then use Fresnal laws to blend them dynamically.

### Water Depth Visuals
To render light absorption, we can first create a thickness FBO by adding particle depth along the view rate. Then using the Beer-Lambert Law ($I = I_0 \cdot e^{-\mu \cdot d}$), we can calculate the light absorbtion and color based on fluid depth, where $I$ is the intensity after passing through the liquid, $\mu$ is the absorption coefficient and $d$ is the thickness/depth. We can also project the thickness map from the light's view to get realistic shadows.

### Whitewater Foam
Secondary particles will be spawned dynamically based on water turbulence and velocity to represent foam and bubbles. This is based on the paper at https://cg.informatik.uni-freiburg.de/publications/2012_CGI_sprayFoamBubbles.pdf

## Objectives
1. Implement a Smoothed Particle Hydrodynamics fluid simulation. It should calculate particle density, pressure, forces, and time intergration to simulate fluid behaviour.
2. Implement a spatial hashing structure to improve efficiency for particle search, so that we can support around 100,000 particles for our simulation. This should aim to reduce completixity from $O(n^2)$ to $O(nk)$.
3. Use a GPU Compute shader for the core physics simulation pass, so that we can run the simulation in parallel to achieve realtime performance.
4. Create a fluid surface from the particles by generating a depth map from the particles, and use bilateral filtering to get a smooth surface.
5. Implement physically based water surface rendering using environment reflection, screen-space refraction, and Fresnal blending. 
6. Implement volumetric light attenuation to simulate depth based color using the Beer-Lambert Law.
7. Implement a whitewater particle system that can generate and render foam and bubbles.
8. Implement fluid-solid interaction with collision detection and have it interact with the simulation.
