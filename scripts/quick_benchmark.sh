#!/bin/bash
# Quick Performance Benchmark (smaller grids for faster testing)

set -e

echo "=== Quick LBM Performance Benchmark ==="
echo ""

# Smaller grid sizes for quick test
GRID_SIZES=(
    "128 128"
    "256 256"
)

TIMESTEPS=50

for grid in "${GRID_SIZES[@]}"; do
    read -r nx ny <<< "$grid"
    
    echo "Testing ${nx}x${ny}..."
    
    # Create config
    CONFIG_FILE="/tmp/bench_${nx}x${ny}.yaml"
    cat > "$CONFIG_FILE" <<EOF
nx: $nx
ny: $ny
relaxation_time: 0.6
max_timesteps: $TIMESTEPS
output_interval: $TIMESTEPS
lid_velocity: 0.1
residual_tolerance: 1e-4
backend_id: "cpu"
EOF
    
    # CPU
    echo -n "  CPU: "
    CPU_START=$(date +%s.%N)
    ./build/src/lbm_sim --config "$CONFIG_FILE" --backend cpu --output-dir /tmp/cpu_${nx}x${ny} > /dev/null 2>&1
    CPU_END=$(date +%s.%N)
    CPU_TIME=$(echo "$CPU_END - $CPU_START" | bc)
    echo -n "${CPU_TIME}s  "
    
    # CUDA
    echo -n "CUDA: "
    CUDA_START=$(date +%s.%N)
    ./build/src/lbm_sim --config "$CONFIG_FILE" --backend cuda --output-dir /tmp/cuda_${nx}x${ny} > /dev/null 2>&1
    CUDA_END=$(date +%s.%N)
    CUDA_TIME=$(echo "$CUDA_END - $CUDA_START" | bc)
    echo -n "${CUDA_TIME}s  "
    
    # Speedup
    SPEEDUP=$(echo "scale=2; $CPU_TIME / $CUDA_TIME" | bc)
    echo "Speedup: ${SPEEDUP}x"
    echo ""
done

echo "Done!"

