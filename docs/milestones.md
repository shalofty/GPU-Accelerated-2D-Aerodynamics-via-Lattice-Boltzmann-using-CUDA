# Project Milestones

## Milestone 1: CPU Reference Solver
- Finalize simulation configuration schema and parsing.
- Introduce build system (CMake) wiring CPU backend and core modules.
- Implement CPU D2Q9 LBM core with cavity and cylinder cases.
- Establish unit tests and baseline validation metrics.
- Document implementation progress and testing outcomes.

## Milestone 2: CUDA Parity Implementation
- Port lattice update loop to CUDA with struct-of-arrays layout.
- Match CPU accuracy within tolerance on benchmark problems.
- Set up automated comparison harness between CPU and CUDA.

## Milestone 3: Validation & Diagnostics
- Automate regression suite with golden data and reporting.
- Integrate diagnostic observers for residuals and aerodynamic coefficients.
- Produce initial visualization outputs (VTK/CSV) for flow fields.

## Milestone 4: Performance Optimization
- Profile kernels and remove bottlenecks via shared-memory tiling and stream overlap.
- Document speedup metrics and scalability studies.
- Capture profiling runs with scripts and archive results.

## Milestone 5: Reporting & Presentation Assets
- Compile technical report with methodology, validation, and performance sections.
- Prepare presentation visuals highlighting flow features and acceleration.
- Review documentation (developer guide, milestones, READMEs) for completeness.
