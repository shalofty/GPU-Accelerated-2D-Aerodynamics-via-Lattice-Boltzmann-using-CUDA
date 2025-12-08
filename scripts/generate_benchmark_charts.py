#!/usr/bin/env python3
"""
Generate performance benchmark charts from benchmark_results.json
Requires: matplotlib, numpy
Install with: pip install matplotlib numpy
"""

import json
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install with: pip install matplotlib numpy")
    sys.exit(1)

# Set style for academic figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'docs' / 'images'
BENCHMARK_FILE = PROJECT_ROOT / 'benchmark_results.json'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_benchmark_data():
    """Load benchmark data from JSON file"""
    if not BENCHMARK_FILE.exists():
        print(f"Error: Benchmark file not found: {BENCHMARK_FILE}")
        sys.exit(1)
    
    with open(BENCHMARK_FILE, 'r') as f:
        return json.load(f)


def generate_performance_scaling(benchmark_data):
    """Generate performance scaling charts"""
    # Extract data
    grid_sizes = [entry['grid_size'] for entry in benchmark_data]
    cpu_times = [entry['cpu']['total_time'] for entry in benchmark_data]
    cuda_times = [entry['cuda']['total_time'] for entry in benchmark_data]
    speedups = [entry['speedup'] for entry in benchmark_data]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Execution time comparison (log scale)
    x_pos = np.arange(len(grid_sizes))
    ax1.semilogy(x_pos, cpu_times, 'o-', label='CPU', linewidth=2.5, 
                markersize=12, color='#d62728', markerfacecolor='white', 
                markeredgewidth=2)
    ax1.semilogy(x_pos, cuda_times, 's-', label='CUDA', linewidth=2.5, 
                markersize=12, color='#2ca02c', markerfacecolor='white',
                markeredgewidth=2)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(grid_sizes)
    ax1.set_xlabel('Grid Size', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Execution Time (s)', fontsize=12, fontweight='bold')
    ax1.set_title('Execution Time vs Grid Size', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_ylim([0.3, 35])
    
    # Right: Speedup
    bars = ax2.bar(x_pos, speedups, color='#1f77b4', alpha=0.8, width=0.6,
                   edgecolor='navy', linewidth=1.5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(grid_sizes)
    ax2.set_xlabel('Grid Size', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Speedup (×)', fontsize=12, fontweight='bold')
    ax2.set_title('GPU Speedup vs Grid Size', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 16])
    
    # Add value labels on bars
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.2f}×', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'performance_scaling.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"✓ Generated: {output_path}")
    plt.close()


def generate_time_per_step_comparison(benchmark_data):
    """Generate time per timestep comparison chart"""
    grid_sizes = [entry['grid_size'] for entry in benchmark_data]
    cpu_ms_per_step = [entry['cpu']['time_per_step_ms'] for entry in benchmark_data]
    cuda_ms_per_step = [entry['cuda']['time_per_step_ms'] for entry in benchmark_data]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(grid_sizes))
    width = 0.35
    
    bars1 = ax.bar(x_pos - width/2, cpu_ms_per_step, width, label='CPU', 
                   color='#d62728', alpha=0.8, edgecolor='darkred', linewidth=1.5)
    bars2 = ax.bar(x_pos + width/2, cuda_ms_per_step, width, label='CUDA', 
                   color='#2ca02c', alpha=0.8, edgecolor='darkgreen', linewidth=1.5)
    
    ax.set_xlabel('Grid Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time per Timestep (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Time per Timestep: CPU vs CUDA', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(grid_sizes)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'time_per_step_comparison.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"✓ Generated: {output_path}")
    plt.close()


def generate_speedup_chart(benchmark_data):
    """Generate standalone speedup chart"""
    grid_sizes = [entry['grid_size'] for entry in benchmark_data]
    speedups = [entry['speedup'] for entry in benchmark_data]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(grid_sizes, speedups, color='#1f77b4', alpha=0.8, width=0.6,
                  edgecolor='navy', linewidth=2)
    ax.set_xlabel('Grid Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup (×)', fontsize=12, fontweight='bold')
    ax.set_title('GPU Speedup vs Grid Size', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 16])
    
    # Add value labels on bars
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{speedup:.2f}×', ha='center', va='bottom', 
               fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'speedup_chart.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"✓ Generated: {output_path}")
    plt.close()


def main():
    print("Generating benchmark performance charts...")
    print(f"Reading from: {BENCHMARK_FILE}")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    try:
        benchmark_data = load_benchmark_data()
        
        generate_performance_scaling(benchmark_data)
        generate_time_per_step_comparison(benchmark_data)
        generate_speedup_chart(benchmark_data)
        
        print(f"\n✓ All charts generated successfully!")
        print(f"  Images saved to: {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"\n✗ Error generating charts: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

