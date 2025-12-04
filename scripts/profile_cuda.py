#!/usr/bin/env python3
"""
CUDA Performance Profiling Script

This script runs CUDA LBM simulations and collects performance metrics.
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


def run_cuda_simulation(config_path: str, output_dir: str) -> Dict:
    """Run CUDA simulation and collect metrics."""
    print(f"Running CUDA simulation with config: {config_path}")
    
    # Build command
    cmd = [
        "./build/lbm_sim",
        "--config", config_path,
        "--backend", "cuda",
        "--output-dir", output_dir
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed_time = time.time() - start_time
    
    if result.returncode != 0:
        print(f"Error running simulation: {result.stderr}")
        return None
    
    # Parse output for metrics
    metrics = {
        "total_time_seconds": elapsed_time,
        "return_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr
    }
    
    # Try to parse performance log if it exists
    perf_log = Path(output_dir) / "performance.csv"
    if perf_log.exists():
        metrics["performance_log"] = str(perf_log)
    
    return metrics


def profile_grid_sizes() -> List[Dict]:
    """Profile different grid sizes."""
    results = []
    
    grid_sizes = [
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024)
    ]
    
    for nx, ny in grid_sizes:
        print(f"\nProfiling grid size: {nx}x{ny}")
        
        # Create temporary config
        config = {
            "nx": nx,
            "ny": ny,
            "relaxation_time": 0.6,
            "max_timesteps": 100,
            "output_interval": 10,
            "lid_velocity": 0.1,
            "residual_tolerance": 1e-4,
            "backend_id": "cuda"
        }
        
        config_path = f"/tmp/lbm_config_{nx}x{ny}.yaml"
        with open(config_path, 'w') as f:
            for key, value in config.items():
                f.write(f"{key}: {value}\n")
        
        output_dir = f"/tmp/lbm_output_{nx}x{ny}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        metrics = run_cuda_simulation(config_path, output_dir)
        if metrics:
            metrics["grid_size"] = f"{nx}x{ny}"
            metrics["nx"] = nx
            metrics["ny"] = ny
            results.append(metrics)
    
    return results


def main():
    """Main profiling function."""
    output_file = "cuda_profile_results.json"
    
    print("CUDA Performance Profiling")
    print("=" * 50)
    
    results = profile_grid_sizes()
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Print summary
    print("\nPerformance Summary:")
    print("-" * 50)
    for result in results:
        grid = result.get("grid_size", "unknown")
        time_sec = result.get("total_time_seconds", 0.0)
        print(f"Grid {grid}: {time_sec:.3f} seconds")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

