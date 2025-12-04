# Testing Guide

## Quick Test Summary

The project structure has been verified. All required components are in place:

**Core Components**
- D2Q9 lattice model
- Simulation backend interfaces
- CPU and CUDA implementations

**Features**
- Collision and streaming kernels (standard + tiled)
- Boundary conditions (lid-driven, inflow/outflow, obstacles)
- Obstacle handling (cylinder flow)
- Drag/lift calculation
- VTK visualization output
- Validation suite

**Tests**
- CPU backend unit tests
- CPU/CUDA comparison tests
- Configuration builder tests

## Building and Testing

### Prerequisites
- CMake >= 3.22
- C++20 compatible compiler (GCC 10+, Clang 12+)
- CUDA toolkit (for CUDA backend)
- Catch2 (will be downloaded automatically)

### Build Steps

```bash
# Create build directory
mkdir build && cd build

# Configure (CUDA enabled by default)
cmake ..

# Build
make -j$(nproc)

# Run tests
ctest --output-on-failure

# Or run tests directly
./tests/lbm_tests
```

### Test Cases

1. **CPU Backend Tests** (`CpuLbBackendTests.cpp`)
   - Initialization
   - Timestep execution
   - Convergence checking

2. **CPU/CUDA Comparison** (`CpuCudaComparisonTests.cpp`)
   - Result matching (within tolerance)
   - Convergence similarity

3. **Configuration Tests** (`SimulationConfigBuilderTests.cpp`)
   - Default values
   - YAML parsing

## Manual Testing

### Test 1: CPU Lid-Driven Cavity

Create `test_cavity.yaml`:
```yaml
nx: 64
ny: 64
relaxation_time: 0.6
max_timesteps: 500
output_interval: 50
lid_velocity: 0.1
residual_tolerance: 1e-4
backend_id: "cpu"
```

Run:
```bash
./build/lbm_sim --config test_cavity.yaml --output-dir output_cavity
```

Expected: VTK files in `output_cavity/` showing flow development

### Test 2: CUDA Cylinder Flow

Create `test_cylinder.yaml`:
```yaml
nx: 200
ny: 100
relaxation_time: 0.6
max_timesteps: 1000
output_interval: 100
residual_tolerance: 1e-4
backend_id: "cuda"
obstacles:
  - id: "cylinder1"
    type: "cylinder"
    parameters: [50.0, 50.0, 10.0]
```

Run:
```bash
./build/lbm_sim --config test_cylinder.yaml --output-dir output_cylinder
```

Expected: Flow around cylinder with drag/lift coefficients

### Test 3: Validation Suite

```cpp
#include "ValidationSuite.hpp"
#include "backend/cpu/CpuLbBackend.hpp"

lbm::CpuLbBackend backend;
lbm::ValidationSuite suite;
auto results = suite.run_validation(&backend, "all");

for (const auto& result : results) {
    std::cout << result.test_name << ": " 
              << (result.passed ? "PASS" : "FAIL") << std::endl;
}
```

## Performance Testing

### CPU Profiling
```bash
python3 scripts/profile_cpu.py
```

### CUDA Profiling
```bash
python3 scripts/profile_cuda.py
```

Results saved to JSON files for analysis.

## Expected Results

### Lid-Driven Cavity
- Converges within tolerance
- Velocity field shows recirculation
- Residual decreases monotonically

### Cylinder Flow
- Drag coefficient: ~2.0 (Re=20)
- Flow separation behind cylinder
- Vortex formation

### Performance
- CUDA speedup: >10x for 256x256, >20x for 1024x1024
- Memory usage: ~200MB CPU, ~400MB CUDA (512x512)

## Troubleshooting

### Build Issues
- **CMake not found**: Install CMake 3.22+
- **CUDA not found**: Set `CUDA_PATH` or disable with `-DENABLE_CUDA=OFF`
- **Catch2 download fails**: Check internet connection

### Runtime Issues
- **CUDA out of memory**: Reduce grid size
- **Tests fail**: Check CUDA device availability
- **Slow performance**: Enable tiled kernel (default)

### Validation Failures
- **Drag coefficient mismatch**: Check Reynolds number, relaxation time
- **Convergence issues**: Increase max_timesteps or adjust tolerance

## Next Steps

After successful build:
1. Run all unit tests: `ctest`
2. Run validation suite
3. Generate performance profiles
4. Visualize results in ParaView
5. Compare CPU vs CUDA results

