#!/bin/bash
# Performance Benchmarking Script

set -e

echo "=== LBM Performance Benchmarks ==="
echo ""

RESULTS_FILE="benchmark_results.json"
RESULTS_DIR="benchmarks"

mkdir -p "$RESULTS_DIR"

# Grid sizes to test
GRID_SIZES=(
    "128 128"
    "256 256"
    "512 512"
    "1024 1024"
)

TIMESTEPS=100

echo "[" > "$RESULTS_FILE"

FIRST=true

for grid in "${GRID_SIZES[@]}"; do
    read -r nx ny <<< "$grid"
    
    echo "Testing grid size: ${nx}x${ny}"
    
    # Create temporary config
    CONFIG_FILE="${RESULTS_DIR}/config_${nx}x${ny}.yaml"
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
    
    # CPU benchmark
    echo "  Running CPU benchmark..."
    CPU_OUTPUT="${RESULTS_DIR}/cpu_${nx}x${ny}"
    CPU_START=$(date +%s.%N)
    ./build/src/lbm_sim --config "$CONFIG_FILE" --backend cpu --output-dir "$CPU_OUTPUT" > /dev/null 2>&1
    CPU_END=$(date +%s.%N)
    CPU_TIME=$(echo "$CPU_END - $CPU_START" | bc)
    
    # CUDA benchmark
    echo "  Running CUDA benchmark..."
    CUDA_OUTPUT="${RESULTS_DIR}/cuda_${nx}x${ny}"
    CUDA_START=$(date +%s.%N)
    ./build/src/lbm_sim --config "$CONFIG_FILE" --backend cuda --output-dir "$CUDA_OUTPUT" > /dev/null 2>&1
    CUDA_END=$(date +%s.%N)
    CUDA_TIME=$(echo "$CUDA_END - $CUDA_START" | bc)
    
    # Calculate speedup
    SPEEDUP=$(echo "scale=2; $CPU_TIME / $CUDA_TIME" | bc)
    
    # Write result
    if [ "$FIRST" = false ]; then
        echo "," >> "$RESULTS_FILE"
    fi
    FIRST=false
    
    cat >> "$RESULTS_FILE" <<EOF
  {
    "grid_size": "${nx}x${ny}",
    "nx": $nx,
    "ny": $ny,
    "timesteps": $TIMESTEPS,
    "cpu_time_seconds": $CPU_TIME,
    "cuda_time_seconds": $CUDA_TIME,
    "speedup": $SPEEDUP,
    "cpu_time_per_step_ms": $(echo "scale=3; $CPU_TIME * 1000 / $TIMESTEPS" | bc),
    "cuda_time_per_step_ms": $(echo "scale=3; $CUDA_TIME * 1000 / $TIMESTEPS" | bc)
  }
EOF
    
    echo "  CPU: ${CPU_TIME}s, CUDA: ${CUDA_TIME}s, Speedup: ${SPEEDUP}x"
    echo ""
done

echo "]" >> "$RESULTS_FILE"

echo "=== Benchmark Complete ==="
echo "Results saved to: $RESULTS_FILE"
echo ""

# Print summary table
echo "Performance Summary:"
echo "Grid Size    | CPU (s)  | CUDA (s) | Speedup"
echo "-------------|----------|----------|--------"
python3 <<PYTHON
import json
with open('$RESULTS_FILE') as f:
    results = json.load(f)
for r in results:
    print(f"{r['grid_size']:12} | {r['cpu_time_seconds']:8.3f} | {r['cuda_time_seconds']:8.3f} | {r['speedup']:6.2f}x")
PYTHON

