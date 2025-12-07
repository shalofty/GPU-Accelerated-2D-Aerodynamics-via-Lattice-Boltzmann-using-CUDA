# Presentation Outline

## Slide 1: Title
- GPU-Accelerated 2D Aerodynamics via Lattice Boltzmann using CUDA
- Author, Date

## Slide 2: Objectives
- Implement CUDA-based LBM solver
- Achieve >20x speedup
- Validate against reference cases

## Slide 3: LBM Overview
- D2Q9 model diagram
- Collision + Streaming
- Macroscopic variables

## Slide 4: Implementation Architecture
- CPU vs CUDA backends
- Strategy pattern
- Memory layout

## Slide 5: CUDA Optimizations
- Shared-memory tiling
- CUDA streams
- Kernel design

## Slide 6: Validation - Cavity Flow
- Velocity field visualization
- Convergence plot
- Comparison with reference

## Slide 7: Validation - Cylinder Flow
- Flow around cylinder
- Drag coefficient comparison
- Vortex shedding (if applicable)

## Slide 8: Performance Results
- Speedup chart (grid size vs speedup)
- Bandwidth utilization
- Scalability

## Slide 9: Key Findings
- Achieved speedups
- Accuracy maintained
- Optimization impact

## Slide 10: Conclusion
- Objectives met
- Future work
- Questions

## Visual Assets Needed
- Flow field visualizations (ParaView screenshots)
- Performance charts (speedup vs grid size)
- Architecture diagram
- Kernel execution timeline

