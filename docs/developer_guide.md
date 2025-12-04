# Developer Guide

## Overview

This document provides technical details for developers working on the GPU-Accelerated Lattice Boltzmann Method (LBM) solver.

## Architecture

### Component Structure

```
src/
├── core/              # Core interfaces and data structures
│   ├── D2Q9.hpp      # Lattice constants (D2Q9 model)
│   ├── SimulationBackend.hpp  # Abstract backend interface
│   ├── SimulationConfig.hpp   # Configuration structure
│   └── SimulationRunner.hpp   # Facade for simulation execution
├── backend/
│   ├── cpu/          # CPU reference implementation
│   └── cuda/         # CUDA accelerated implementation
├── io/               # Input/output handling
│   ├── SimulationConfigBuilder.cpp  # YAML config parsing
│   └── VtkWriter.cpp                # VTK visualization output
└── analysis/         # Analysis and diagnostics
    ├── DragLiftCalculator.cpp       # Aerodynamic force computation
    ├── ValidationSuite.cpp           # Validation against reference data
    ├── CudaProfiler.cpp             # CUDA performance profiling
    └── VtkObserver.cpp              # VTK output observer
```

## Memory Layout

### CPU Backend

- **Distribution Functions**: Array-of-structures layout
  - `f_curr_[cell * 9 + q]` where `cell = y * nx + x`
  - Double buffering: `f_curr_` and `f_next_` swapped each timestep

### CUDA Backend

- **Distribution Functions**: Same layout as CPU for compatibility
- **Device Memory**: Allocated via `cudaMalloc`, managed with RAII
- **Shared Memory Tiling**: Optional optimization using `__shared__` memory
  - Tile size: 16x16 cells with 1-cell halo
  - Reduces global memory accesses for collision computation

## Kernel Design

### Collide-and-Stream Kernel

**Standard Version:**
- One thread per cell
- Direct global memory access
- Suitable for all grid sizes

**Tiled Version:**
- Uses shared memory to cache distribution functions
- Reduces global memory traffic
- Best for larger grids (>256x256)

### Boundary Conditions

1. **Lid-Driven Cavity**: Top wall with prescribed velocity
2. **Inflow/Outflow**: Left/right boundaries for channel flow
3. **No-Slip Walls**: Top/bottom boundaries
4. **Obstacles**: Bounce-back at obstacle cells

## Performance Optimization

### Shared Memory Tiling

The tiled kernel loads a tile of cells into shared memory, processes them, then writes results. This reduces global memory bandwidth requirements.

**Usage:**
- Enabled by default (`use_tiled_kernel_ = true`)
- Automatically selected based on grid size

### CUDA Streams

Two streams are used:
- `compute_stream_`: Kernel execution
- `transfer_stream_`: Host-device data transfers

This allows overlapping computation and data transfers.

### Profiling

Use `CudaProfiler` to measure kernel execution times:

```cpp
CudaProfiler profiler;
profiler.record_kernel_start();
// ... launch kernel ...
profiler.record_kernel_end("collide_stream");
profiler.write_report("profile.txt");
```

## Validation

### Test Cases

1. **Lid-Driven Cavity**
   - Grid: 64x64
   - Validation: Convergence to steady state

2. **Cylinder Flow**
   - Grid: 200x100
   - Validation: Drag coefficient within 15% of reference

### Running Validation

```cpp
ValidationSuite suite;
auto results = suite.run_validation(backend.get(), "all");
for (const auto& result : results) {
    std::cout << result.test_name << ": " 
              << (result.passed ? "PASS" : "FAIL") << std::endl;
}
```

## Building and Testing

### Build Configuration

```bash
mkdir build && cd build
cmake .. -DENABLE_CUDA=ON
make
```

### Running Tests

```bash
ctest --test-dir build
```

### Profiling

```bash
# CUDA profiling
python3 scripts/profile_cuda.py

# CPU profiling
python3 scripts/profile_cpu.py
```

## Configuration Files

YAML configuration format:

```yaml
nx: 256
ny: 256
relaxation_time: 0.6
max_timesteps: 1000
output_interval: 100
lid_velocity: 0.1
residual_tolerance: 1e-6
backend_id: "cuda"
obstacles:
  - id: "cylinder1"
    type: "cylinder"
    parameters: [50.0, 50.0, 10.0]  # cx, cy, radius
```

## Visualization

VTK files are written at specified intervals. Open in ParaView:

```bash
paraview output/field_*.vtk
```

Fields included:
- `density`: Fluid density
- `velocity`: Velocity vector field
- `velocity_magnitude`: Speed

## Performance Expectations

### Speedup Targets

- **1024x1024 grid**: >20x speedup over CPU
- **512x512 grid**: >15x speedup
- **256x256 grid**: >10x speedup

### Memory Requirements

- **CPU**: ~200 MB for 512x512 grid
- **CUDA**: ~400 MB (includes device memory)

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce grid size or use CPU backend
2. **Slow performance**: Enable tiled kernel, check block size
3. **Validation failures**: Check relaxation time, increase max_timesteps

### Debugging

Enable CUDA error checking:
```cpp
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
        } \
    } while(0)
```

## Contributing

When adding new features:

1. Maintain interface compatibility with `SimulationBackend`
2. Add tests for new functionality
3. Update this documentation
4. Profile performance impact

