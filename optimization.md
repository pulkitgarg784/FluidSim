```
To scale your Vulkan SPH fluid simulation to 100k to millions of particles, you must eliminate CPU bottlenecks and optimize GPU memory layout. At this scale, the simulation loop should run completely on the GPU, utilizing cache-friendly data structures and tight synchronization.Here is the technical architecture required to handle 100k+ particles smoothly in Vulkan:1. The GPU Particle Memory LayoutDo not use an array of structs (Particle { vec3 pos; vec3 vel; ... }). That layout triggers horrible cache thrashing during sorting and density evaluations.Instead, use a Structure of Arrays (SoA) layout split across separate Storage Buffers (SSBOs). This ensures that when a compute shader only needs to read positions to compute density, it does not waste memory bandwidth loading velocities or densities into the GPU cache.cpp// Allocate separate SSBOs for maximum cache efficiency
VkBuffer particlePositionsBuffer;  // vec4 (xyz = pos, w = density/mass)
VkBuffer particleVelocitiesBuffer; // vec4 (xyz = vel, w = unused)
VkBuffer particleForcesBuffer;     // vec4 (xyz = force, w = pressure)
Use code with caution.2. High-Performance Spatial HashingTo find neighboring particles without an \(\mathcal{O}(n^2)\) check, map your 3D space into a uniform grid where the grid cell size equals your SPH smoothing length (h).Calculate 3D Grid Coordinates:\(\text{grid\_coord}=\lfloor (\text{pos}-\text{grid\_min})/h\rfloor \)Hash into a 1D Key: Convert the 3D coordinate into a 1D integer. A simple spatial hash or a Morton Code (Z-curve) works best. Morton codes are superior because they preserve spatial locality, meaning particles close together in 3D space end up close together in your sorted memory buffer.Output Key-Value Pairs: Your hashing compute shader will read particlePositionsBuffer and write to two separate scratch buffers managed by your Radix Sort library:Keys Buffer (uint32_t[]): Array of calculated Grid Cell IDs.Values Buffer (uint32_t[]): Array of original Particle Indices (0 to N-1).3. Build a "Cell Index" Look-Up TableOnce the Radix Sorter finishes reordering your keys and values, the particles are sorted by grid cell. However, your SPH shader needs to know exactly where a specific grid cell's particle list starts and ends.Create a Cell Range SSBO sized to your total number of grid cells:glslstruct CellRange {
    uint start;
    uint end;
};
layout(std430, set = 0, binding = 3) buffer CellRangeBuffer { CellRange cells[]; };
Use code with caution.Run a quick Index Compute Shader with a workgroup size matching your particle count. Each thread checks its sorted key against the neighbor's sorted key to find boundaries:glsl// Run with 1 thread per particle (index 'i')
uint currentCell = SortedKeys[i];
uint previousCell = (i > 0) ? SortedKeys[i - 1] : 0xFFFFFFFF;

if (i == 0) {
    cells[currentCell].start = i;
} else if (currentCell != previousCell) {
    cells[currentCell].end = i;       // End previous cell
    cells[currentCell].start = i;      // Start current cell
}

if (i == totalParticles - 1) {
    cells[currentCell].end = i + 1;
}
Use code with caution.4. Cache-Local Neighbor Search (The SPH Pass)When calculating density or pressure forces for a target particle, calculate its 3D grid cell coordinate. Because an SPH sphere can overlap adjacent cells, you must sample a 3 × 3 × 3 block of 27 neighboring grid cells.glsl// Inside SPH Density / Force Compute Shader
for (int z = -1; z <= 1; z++) {
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            uint targetCellHash = CalculateHash(currentCellCoord + ivec3(x, y, z));
            CellRange range = cells[targetCellHash];
            
            // Loop through all particles inside this neighboring cell
            for (uint i = range.start; i < range.end; i++) {
                uint neighborParticleID = SortedValues[i];
                vec3 neighborPos = ParticlePositions[neighborParticleID].xyz;
                
                // Perform standard SPH math (Kernel function evaluation)
                float dist = distance(myPos, neighborPos);
                if (dist < smoothingLength) {
                    // Accumulate fluid density/force
                }
            }
        }
    }
}
Use code with caution.5. Vulkan Pipeline Barriers & SynchronizationBecause your compute shaders execute back-to-back and write to buffers that the next shader immediately reads, you must place precise execution and memory barriers using vkCmdPipelineBarrier2.For example, between the Radix Sort Pass and the Index Pass, you must tell Vulkan to wait for the sort writes to finish before allowing index shader reads:cppVkBufferMemoryBarrier2 barrier = {
    .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
    .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,          // Radix sort finished writing
    .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,           // Index shader ready to read
    .buffer = sortedKeysBuffer,
    .size = VK_WHOLE_SIZE
};

VkDependencyInfo dependencyInfo = {
    .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
    .bufferMemoryBarrierCount = 1,
    .pBufferMemoryBarriers = &barrier
};

vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
Use code with caution.Summary of Best Practices for 100k+Zero Host-Mapping: Never call vkMapMemory or read particle data back to the CPU during the physics loop. Keep everything in device-local memory (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT).Pre-allocate Everything: Allocate your particles, keys, values, and cell ranges up front. Resizing Vulkan buffers at runtime causes catastrophic frame drops.Asynchronous Compute: If your simulation runs alongside a graphics renderer, dispatch the SPH physics loop on a dedicated Compute Queue to overlap physics math calculations with the main render passes.
```
