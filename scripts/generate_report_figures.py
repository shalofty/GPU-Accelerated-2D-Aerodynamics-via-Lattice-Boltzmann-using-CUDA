#!/usr/bin/env python3
"""
Generate figures for the academic report
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
import sys

# Check for required packages
try:
    import matplotlib
    import numpy
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

OUTPUT_DIR = Path(__file__).parent.parent / 'docs' / 'images'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_d2q9_lattice():
    """Generate D2Q9 lattice model diagram"""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    
    # Center point
    center = (0, 0)
    
    # Velocity vectors: (dx, dy, label, weight, color)
    vectors = [
        (0, 0, 'c₀', 4/9, 'black'),      # Rest
        (1, 0, 'c₁', 1/9, 'red'),        # Right
        (0, 1, 'c₂', 1/9, 'red'),        # Up
        (-1, 0, 'c₃', 1/9, 'red'),       # Left
        (0, -1, 'c₄', 1/9, 'red'),       # Down
        (1, 1, 'c₅', 1/36, 'blue'),      # Up-right
        (-1, 1, 'c₆', 1/36, 'blue'),     # Up-left
        (-1, -1, 'c₇', 1/36, 'blue'),    # Down-left
        (1, -1, 'c₈', 1/36, 'blue'),     # Down-right
    ]
    
    # Draw grid cell
    cell = plt.Rectangle((-0.5, -0.5), 1, 1, fill=False, 
                        edgecolor='gray', linewidth=2, linestyle='--')
    ax.add_patch(cell)
    
    # Draw vectors
    for dx, dy, label, weight, color in vectors:
        if dx == 0 and dy == 0:
            # Rest particle - draw as circle
            circle = plt.Circle(center, 0.15, color=color, fill=True)
            ax.add_patch(circle)
            ax.text(0, -0.3, f'{label}\nw={weight:.3f}', 
                   ha='center', va='top', fontsize=9, color=color)
        else:
            # Draw arrow
            ax.arrow(0, 0, dx*0.7, dy*0.7, head_width=0.1, head_length=0.1,
                    fc=color, ec=color, linewidth=2, length_includes_head=True)
            # Label
            label_x = dx * 0.9
            label_y = dy * 0.9
            ax.text(label_x, label_y, f'{label}\nw={weight:.3f}', 
                   ha='center', va='center', fontsize=8, color=color,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.scatter(*center, s=100, c='black', zorder=5)
    ax.set_title('D2Q9 Lattice Model', fontsize=14, fontweight='bold')
    ax.set_xlabel('x direction', fontsize=11)
    ax.set_ylabel('y direction', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'd2q9_lattice.png', bbox_inches='tight')
    print(f"Generated: {OUTPUT_DIR / 'd2q9_lattice.png'}")


def generate_performance_scaling():
    """Generate performance scaling charts from benchmark_results.json"""
    # Load data from JSON file
    json_path = Path(__file__).parent.parent / 'benchmark_results.json'
    with open(json_path, 'r') as f:
        benchmark_data = json.load(f)
    
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
    plt.savefig(OUTPUT_DIR / 'performance_scaling.png', bbox_inches='tight', 
                dpi=300, facecolor='white')
    print(f"Generated: {OUTPUT_DIR / 'performance_scaling.png'}")
    plt.close()


def generate_memory_layout():
    """Generate memory layout diagram"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Grid cells (3x2 example)
    cell_size = 1.0
    spacing = 0.15
    
    nx, ny = 3, 2
    colors = plt.cm.viridis(np.linspace(0, 1, 9))
    
    # Draw grid cells
    for y in range(ny):
        for x in range(nx):
            # Cell position
            cell_x = x * (cell_size + spacing)
            cell_y = (ny - 1 - y) * (cell_size + spacing)  # Flip y for top-to-bottom
            
            # Draw cell border
            rect = plt.Rectangle((cell_x, cell_y), cell_size, cell_size,
                               fill=False, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            
            # Draw distribution functions (9 small squares inside)
            dx = cell_size / 3
            dy = cell_size / 3
            for i in range(9):
                fx = cell_x + (i % 3) * dx
                fy = cell_y + (2 - i // 3) * dy
                small_rect = plt.Rectangle((fx + 0.01, fy + 0.01), dx-0.02, dy-0.02,
                                         facecolor=colors[i], edgecolor='gray', linewidth=0.3)
                ax.add_patch(small_rect)
            
            # Label cell coordinates
            ax.text(cell_x + cell_size/2, cell_y - 0.2, f'({x},{y})',
                   ha='center', va='top', fontsize=10, fontweight='bold')
    
    # Arrow pointing to memory array
    arrow_x = nx * (cell_size + spacing) + 0.3
    arrow_y = ny * (cell_size + spacing) / 2
    ax.annotate('', xy=(arrow_x + 0.5, arrow_y), xytext=(arrow_x, arrow_y),
               arrowprops=dict(arrowstyle='->', lw=3, color='black'))
    
    # Linear memory array
    mem_start_x = arrow_x + 1.2
    mem_block_width = 0.25
    mem_block_height = 0.15
    mem_spacing = 0.02
    
    total_cells = nx * ny
    total_blocks = total_cells * 9
    
    # Draw memory blocks in a vertical stack
    mem_y_start = 0.1
    for i in range(total_blocks):
        block_x = mem_start_x
        block_y = mem_y_start + i * (mem_block_height + mem_spacing)
        color_idx = i % 9
        
        # Draw memory block
        rect = plt.Rectangle((block_x, block_y), mem_block_width, mem_block_height,
                           facecolor=colors[color_idx], edgecolor='black', linewidth=0.5)
        ax.add_patch(rect)
        
        # Add index label
        cell_idx = i // 9
        dir_idx = i % 9
        x_coord = cell_idx % nx
        y_coord = cell_idx // nx
        label = f'[{i}]'
        ax.text(block_x + mem_block_width/2, block_y + mem_block_height/2, 
               label, ha='center', va='center', fontsize=6, fontweight='bold')
        
        # Add coordinate label on the right
        if dir_idx == 0:  # First direction of each cell
            coord_label = f'({x_coord},{y_coord})'
            ax.text(block_x + mem_block_width + 0.05, block_y + mem_block_height/2,
                   coord_label, ha='left', va='center', fontsize=7, 
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.3))
    
    # Add title for memory array
    ax.text(mem_start_x + mem_block_width/2, mem_y_start + total_blocks * (mem_block_height + mem_spacing) + 0.3,
           'Linear Memory Array\n(SoA Layout)', ha='center', va='bottom', 
           fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.5))
    
    # Add formula annotation
    formula_text = 'Index: (y × nx + x) × 9 + q'
    ax.text((nx * (cell_size + spacing) + mem_start_x) / 2, -0.5,
           formula_text, ha='center', va='top', fontsize=10, fontstyle='italic',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black'))
    
    ax.set_xlim(-0.3, mem_start_x + mem_block_width + 1.5)
    ax.set_ylim(-0.8, ny * (cell_size + spacing) + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Struct-of-Arrays Memory Layout\n(Grid Cells → Linear Array)', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'memory_layout.png', bbox_inches='tight', dpi=300, facecolor='white')
    print(f"Generated: {OUTPUT_DIR / 'memory_layout.png'}")
    plt.close()


def main():
    print("Generating report figures...")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    generate_d2q9_lattice()
    generate_performance_scaling()
    generate_memory_layout()
    
    print("\n✓ Done! Figures saved to docs/images/")
    print("  Note: performance_scaling.png was generated from benchmark_results.json")


if __name__ == '__main__':
    main()

