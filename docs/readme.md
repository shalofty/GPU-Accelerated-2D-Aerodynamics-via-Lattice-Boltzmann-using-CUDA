# GPU-Accelerated 2D Aerodynamics via Lattice Boltzmann using CUDA

## Objective

Implement a CUDA-based 2D Lattice Boltzmann solver (D2Q9) to simulate incompressible
flow around obstacles, demonstrating correct flow features and substantial acceleration
over a CPU baseline.

## Framework

CUDA (primary), with a single-threaded CPU version to be used for comparison.

## Hypothesis

By exploiting massive data parallelism (one thread per cell), struct-of-arrays layout,
shared-memory tiling, and asynchronous host-device transfers with CUDA streams/events,
the solver will achieve a >20x speedup over a single-thread CPU implementation on
1024x1024 grids, while reproducing reference aerodynamic metrics within 5-10% literature
values.

## Deliverables

- **Code (20%)**: CUDA LBM solver with unit tests and profiling scripts.
- **Report (10%)**: Methods, validation (cavity, cylinder), performance results.
- **Presentation (10%)**: Visualization of vortices/pressure fields and speedup charts.

## Development Plan Overview

### Simulation Core

- Build a modular D2Q9 LBM solver with distinct components for lattice updates, boundary handling, and macroscopic variable computation (`computeMacros`, `streamCollide`, `applyBCs`).
- Share algorithmic steps between CPU and CUDA backends via a strategy pattern while reusing common data layout utilities.
- Store distribution fields in struct-of-arrays form with double buffering to ensure coalesced memory access; wrap device memory with an RAII-style `LbDeviceGrid`.
- Implement boundary condition policies (no-slip, inflow, outflow) to avoid scattering conditionals through kernels.

### CUDA Acceleration

- Use tiled kernels with shared-memory staging for collision/stream steps and overlap host-device transfers with computation through CUDA streams.
- Add configuration-driven grid/block sizing with an auto-tuning harness and queue profiling runs using a command pattern.
- Instrument with CUDA event timers and CUPTI/Nsight-compatible markers; wrap CUDA API calls in checked helpers.

### Validation & Testing

- Create regression tests for lid-driven cavity and cylinder flows with reference coefficients, using Python/NumPy to generate golden data.
- Provide a CPU baseline implementation sharing interfaces with the CUDA backend to enable small-grid bitwise comparisons and tolerance-based checks at scale.
- Automate testing with `ctest`/`pytest`, storing performance metrics in JSON for historical tracking.

### Tooling & Architecture

- Organize sources into `src/core`, `src/backend/cpu`, `src/backend/cuda`, `src/io`, and `src/analysis`, with a façade-style `SimulationRunner` coordinating setup, timestep loops, and diagnostics.
- Use a builder pattern to construct typed `SimulationConfig` instances from YAML/TOML inputs specifying grid size, relaxation parameters, and obstacle meshes.
- Adopt an observer pattern for diagnostics (vorticity snapshots, convergence logs) so new outputs attach to the simulation loop without invasive changes.

### Documentation & Deliverables

- Maintain developer documentation covering memory layout, kernel launch strategies, and validation methodology; catalogue any logged datasets to satisfy documentation preferences.
- Provide reproducible profiling scripts (`scripts/profile_cuda.py`, `scripts/profile_cpu.py`) that capture runtime, throughput, and bandwidth metrics.
- Prepare visualization workflows (e.g., ParaView-ready VTK output) and Jupyter notebooks for report figures and presentation assets.

### Current Implementation Status

- **Build System**: CMake-based project with optional CUDA backend target and Catch2 test discovery enabled via `ctest`.
- **Core CPU Solver**: BGK D2Q9 collide–stream loop with lid-driven cavity boundary, macroscopic field updates, and diagnostic snapshots (`drag`, `lift`, `residual`).
- **Configuration Handling**: `SimulationConfig` expanded with lid velocity/residual tolerance; builder returns sensible defaults pending YAML/TOML parsing.
- **Testing**: Catch2 unit tests covering configuration defaults/error cases and CPU backend initialization/convergence run via `ctest --test-dir build`.

### Project Management

- Define milestones: CPU reference solver, CUDA parity implementation, validation suite, performance optimization/profiling, and report/presentation preparation.
- Track tasks via issues/roadmap, including a risk register for stability, memory limits, and validation data availability alongside mitigation strategies.

## Repository Skeleton & Interfaces

### Directory Structure

- `src/core`: Lattice data structures (`DistributionField`, `GridGeometry`), core update logic (`streamCollide`, `computeMacros`).
- `src/backend/cpu`: CPU reference backend (`CpuLbBackend`) sharing interfaces with CUDA implementation.
- `src/backend/cuda`: CUDA kernels, launch utilities, device buffer management (`LbDeviceGrid`).
- `src/io`: Configuration parsing (`SimulationConfigBuilder`), snapshot output (VTK/CSV), restart I/O.
- `src/analysis`: Validation metrics (`DragLiftCalculator`), performance logging, regression tooling.
- `tests`: Unit tests and regression harness (Catch2/GTest/PyTest).
- `scripts`: Profiling, validation comparison, visualization pipelines.
- `docs`: Developer guide, validation notes, profiling reports.

### Key Interfaces & Classes

- `SimulationConfig`: Immutable simulation settings built via builder pattern from YAML/TOML.
- `SimulationBackend`: Strategy interface with `initialize()`, `step()`, `fetchDiagnostics()`; implemented by `CpuLbBackend` and `CudaLbBackend`.
- `BoundaryCondition`: Policy interface implemented by `NoSlipBC`, `InflowBC`, `OutflowBC`, etc.
- `SimulationRunner`: Façade coordinating configuration loading, backend selection, timestep loop, and diagnostic observers.
- `DiagnosticObserver`: Observer interface for residual logs, VTK snapshots, performance metrics.
- `PerformanceLogger`: Encapsulates CUDA events and host timers, writes documented JSON logs.
- `ValidationSuite`: Automates benchmark cases, compares against golden data, reports deviations.

## Architectural & Software Patterns

- **Strategy Pattern**: Swap CPU and CUDA backends without changing simulation flow.
- **Abstract Factory / Builder**: Construct complex `SimulationConfig` objects from external configuration files.
- **Facade Pattern**: `SimulationRunner` presents a single entry point for running simulations.
- **Observer Pattern**: Diagnostics subscribe to timestep events for extensible outputs.
- **Policy Objects**: Boundary conditions encapsulated to avoid conditional sprawl in kernels.
- **Command Pattern**: Batch profiling or validation runs as queued commands.
- **Template Method**: Shared timestep skeleton with backend-specific hooks.
- **Repository Pattern**: Optional abstraction for persisting simulation metadata/results.
- **Pipes and Filters**: Chainable post-processing stages for field data transformations.
- **Layered / Hexagonal Architecture**: Separates domain logic from infrastructure and presentation layers.
- **Component-Based Modular Design**: Distinct modules (`core`, `backend`, `analysis`, `io`) with controlled dependencies.
- **Data-Oriented Design**: Struct-of-arrays layouts and cache/coalescing-friendly memory management, especially for CUDA kernels.