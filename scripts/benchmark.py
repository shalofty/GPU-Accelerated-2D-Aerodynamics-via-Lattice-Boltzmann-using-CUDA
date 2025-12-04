#!/usr/bin/env python3
"""
Performance Benchmarking Script
Runs CPU and CUDA simulations and compares performance
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


def run_simulation(config_path: str, backend: str, output_dir: str) -> Dict:
    """Run simulation and return timing information."""
    cmd = ["./build/src/lbm_sim", "--config", config_path, "--backend", backend, "--output-dir", output_dir]
    
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None
    
    # Parse output for timestep info
    timesteps = 0
    for line in result.stdout.split('\n'):
        if 'Final timestep:' in line:
            try:
                timesteps = int(line.split(':')[1].strip())
            except:
                pass
    
    return {
        "total_time": elapsed,
        "timesteps": timesteps,
        "time_per_step_ms": (elapsed / timesteps * 1000) if timesteps > 0 else 0,
        "success": True
    }


def benchmark_grid_size(nx: int, ny: int, timesteps: int = 100) -> Dict:
    """Benchmark a specific grid size."""
    print(f"\nBenchmarking {nx}x{ny} grid...")
    
    # Create config
    config = {
        "nx": nx,
        "ny": ny,
        "relaxation_time": 0.6,
        "max_timesteps": timesteps,
        "output_interval": timesteps,
        "lid_velocity": 0.1,
        "residual_tolerance": 1e-4,
        "backend_id": "cpu"
    }
    
    config_path = f"/tmp/bench_{nx}x{ny}.yaml"
    with open(config_path, 'w') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    
    # CPU benchmark
    print(f"  Running CPU...", end=" ", flush=True)
    cpu_result = run_simulation(config_path, "cpu", f"/tmp/cpu_{nx}x{ny}")
    if cpu_result:
        print(f"{cpu_result['total_time']:.3f}s ({cpu_result['time_per_step_ms']:.3f} ms/step)")
    else:
        return None
    
    # CUDA benchmark
    print(f"  Running CUDA...", end=" ", flush=True)
    cuda_result = run_simulation(config_path, "cuda", f"/tmp/cuda_{nx}x{ny}")
    if cuda_result:
        print(f"{cuda_result['total_time']:.3f}s ({cuda_result['time_per_step_ms']:.3f} ms/step)")
    else:
        return None
    
    speedup = cpu_result['total_time'] / cuda_result['total_time'] if cuda_result['total_time'] > 0 else 0
    
    return {
        "grid_size": f"{nx}x{ny}",
        "nx": nx,
        "ny": ny,
        "timesteps": timesteps,
        "cpu": cpu_result,
        "cuda": cuda_result,
        "speedup": speedup
    }


def main():
    """Main benchmarking function."""
    print("=== LBM Performance Benchmark ===\n")
    
    # Grid sizes to test
    grid_sizes = [
        (128, 128),
        (256, 256),
        (512, 512),
    ]
    
    results = []
    
    for nx, ny in grid_sizes:
        result = benchmark_grid_size(nx, ny, timesteps=100)
        if result:
            results.append(result)
    
    # Save results
    output_file = "benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Performance Summary")
    print("=" * 60)
    print(f"{'Grid Size':<12} {'CPU (s)':<10} {'CUDA (s)':<10} {'Speedup':<10} {'CPU ms/step':<12} {'CUDA ms/step':<12}")
    print("-" * 60)
    
    for r in results:
        cpu_time = r['cpu']['total_time']
        cuda_time = r['cuda']['total_time']
        speedup = r['speedup']
        cpu_ms = r['cpu']['time_per_step_ms']
        cuda_ms = r['cuda']['time_per_step_ms']
        print(f"{r['grid_size']:<12} {cpu_time:<10.3f} {cuda_time:<10.3f} {speedup:<10.2f}x {cpu_ms:<12.3f} {cuda_ms:<12.3f}")
    
    print(f"\nResults saved to: {output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

