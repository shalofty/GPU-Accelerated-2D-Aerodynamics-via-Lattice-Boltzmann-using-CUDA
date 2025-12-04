#!/bin/bash
# Test Verification Script
# Checks code structure and prepares for testing

set -e

echo "=== LBM Project Test Verification ==="
echo ""

# Check for required files
echo "Checking project structure..."
REQUIRED_FILES=(
    "src/core/D2Q9.hpp"
    "src/core/SimulationBackend.hpp"
    "src/backend/cpu/CpuLbBackend.cpp"
    "src/backend/cuda/CudaLbBackend.cu"
    "src/analysis/DragLiftCalculator.cpp"
    "src/analysis/ValidationSuite.cpp"
    "tests/CpuLbBackendTests.cpp"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ Missing: $file"
        exit 1
    fi
done

echo ""
echo "Checking for common compilation issues..."

# Check for missing includes
echo "Checking includes..."
if grep -q "#include.*<limits>" src/backend/cpu/CpuLbBackend.cpp; then
    echo "✓ limits header included"
else
    echo "⚠ May need <limits> header"
fi

# Check CUDA backend structure
echo ""
echo "Checking CUDA backend..."
if grep -q "collide_and_stream_kernel" src/backend/cuda/CudaLbBackend.cu; then
    echo "✓ Collide-stream kernel found"
fi

if grep -q "collide_and_stream_tiled_kernel" src/backend/cuda/CudaLbBackend.cu; then
    echo "✓ Tiled kernel found"
fi

if grep -q "cudaStreamCreate" src/backend/cuda/CudaLbBackend.cu; then
    echo "✓ CUDA streams implemented"
fi

# Check test files
echo ""
echo "Checking test files..."
if [ -f "tests/CpuLbBackendTests.cpp" ]; then
    echo "✓ CPU backend tests found"
fi

if [ -f "tests/CpuCudaComparisonTests.cpp" ]; then
    echo "✓ CPU/CUDA comparison tests found"
fi

echo ""
echo "=== Verification Complete ==="
echo ""
echo "To build and test:"
echo "  1. Install CMake (>=3.22) and CUDA toolkit"
echo "  2. mkdir build && cd build"
echo "  3. cmake .."
echo "  4. make"
echo "  5. ctest"
echo ""

