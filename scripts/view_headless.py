#!/usr/bin/env python3
"""
Headless VTK visualization - generates PNG images without display
Works in Docker/headless environments
"""

import sys
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
except ImportError:
    print("Error: VTK not installed. Install with: pip install vtk")
    sys.exit(1)


def read_vtk_file(filename):
    """Read VTK file and extract field data."""
    reader = vtk.vtkXMLRectilinearGridReader()
    if not os.path.exists(filename):
        # Try legacy VTK format
        reader = vtk.vtkStructuredPointsReader()
    
    reader.SetFileName(filename)
    reader.Update()
    
    data = reader.GetOutput()
    dims = data.GetDimensions()
    nx, ny = dims[0], dims[1]
    
    # Get available arrays
    point_data = data.GetPointData()
    arrays = {}
    for i in range(point_data.GetNumberOfArrays()):
        name = point_data.GetArrayName(i)
        array = point_data.GetArray(i)
        values = vtk_to_numpy(array)
        arrays[name] = values.reshape(ny, nx)
    
    return arrays, nx, ny


def read_legacy_vtk(filename):
    """Read legacy VTK format (ASCII)."""
    arrays = {}
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith('DIMENSIONS'):
            parts = line.split()
            nx, ny = int(parts[1]), int(parts[2])
            i += 1
            continue
        
        if line.startswith('POINT_DATA'):
            num_points = int(line.split()[1])
            i += 1
            continue
        
        if line.startswith('SCALARS'):
            field_name = line.split()[1]
            i += 2  # Skip LOOKUP_TABLE line
            values = []
            while i < len(lines) and len(values) < num_points:
                line = lines[i].strip()
                if line and not line.startswith('VECTORS') and not line.startswith('SCALARS'):
                    try:
                        values.extend([float(x) for x in line.split()])
                    except:
                        pass
                i += 1
            if len(values) >= num_points:
                arrays[field_name] = np.array(values[:num_points]).reshape(ny, nx)
            continue
        
        if line.startswith('VECTORS'):
            field_name = line.split()[1]
            i += 1
            values = []
            while i < len(lines) and len(values) < num_points * 3:
                line = lines[i].strip()
                if line and not line.startswith('VECTORS') and not line.startswith('SCALARS'):
                    try:
                        values.extend([float(x) for x in line.split()])
                    except:
                        pass
                i += 1
            if len(values) >= num_points * 3:
                vec_data = np.array(values[:num_points * 3]).reshape(num_points, 3)
                arrays[f'{field_name}_x'] = vec_data[:, 0].reshape(ny, nx)
                arrays[f'{field_name}_y'] = vec_data[:, 1].reshape(ny, nx)
                arrays[f'{field_name}_magnitude'] = np.sqrt(vec_data[:, 0]**2 + vec_data[:, 1]**2).reshape(ny, nx)
            continue
        
        i += 1
    
    return arrays, nx, ny


def plot_field(arrays, field_name, output_file, title_suffix=""):
    """Plot a 2D field and save to file."""
    if field_name not in arrays:
        print(f"Field '{field_name}' not found. Available: {list(arrays.keys())}")
        return False
    
    data = arrays[field_name]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(data, origin='lower', cmap='viridis', interpolation='bilinear')
    plt.colorbar(im, ax=ax, label=field_name.replace('_', ' ').title())
    ax.set_title(f'{field_name.replace("_", " ").title()}{title_suffix}')
    ax.set_xlabel('X (lattice units)')
    ax.set_ylabel('Y (lattice units)')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    return True


def visualize_directory(directory, field_name='velocity_magnitude', output_dir=None):
    """Visualize all VTK files in a directory."""
    vtk_files = sorted(glob.glob(os.path.join(directory, 'field_*.vtk')))
    
    if not vtk_files:
        print(f"No VTK files found in {directory}")
        return
    
    print(f"Found {len(vtk_files)} VTK files in {directory}")
    
    if output_dir is None:
        output_dir = Path(directory) / 'plots'
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    for vtk_file in vtk_files:
        print(f"Processing: {os.path.basename(vtk_file)}")
        try:
            arrays, nx, ny = read_vtk_file(vtk_file)
        except:
            try:
                arrays, nx, ny = read_legacy_vtk(vtk_file)
            except Exception as e:
                print(f"  Error reading {vtk_file}: {e}")
                continue
        
        # Extract timestep from filename
        timestep = os.path.basename(vtk_file).split('_')[-1].split('.')[0]
        title_suffix = f" (Timestep {timestep})"
        
        output_file = output_dir / f'{field_name}_t{timestep}.png'
        if plot_field(arrays, field_name, str(output_file), title_suffix):
            print(f"  Saved: {output_file}")
        
        # Also plot other interesting fields
        if 'density' in arrays and field_name != 'density':
            output_file = output_dir / f'density_t{timestep}.png'
            plot_field(arrays, 'density', str(output_file), title_suffix)
    
    print(f"\nAll images saved to: {output_dir}")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python view_headless.py <vtk_file>                    # View single file")
        print("  python view_headless.py <directory> [field_name]    # View all files")
        print("")
        print("Examples:")
        print("  python view_headless.py output/cavity_cpu/field_000050.vtk")
        print("  python view_headless.py output/cavity_cpu/ velocity_magnitude")
        print("")
        print("Available fields: velocity_magnitude, density, velocity_x, velocity_y")
        sys.exit(1)
    
    path = sys.argv[1]
    field_name = sys.argv[2] if len(sys.argv) > 2 else 'velocity_magnitude'
    
    if os.path.isfile(path):
        # Single file
        try:
            arrays, nx, ny = read_vtk_file(path)
        except:
            arrays, nx, ny = read_legacy_vtk(path)
        
        print(f"Grid size: {nx}x{ny}")
        print(f"Available fields: {list(arrays.keys())}")
        
        output_file = path.replace('.vtk', f'_{field_name}.png')
        if plot_field(arrays, field_name, output_file):
            print(f"Saved: {output_file}")
    elif os.path.isdir(path):
        # Directory
        visualize_directory(path, field_name)
    else:
        print(f"Error: {path} not found")
        sys.exit(1)


if __name__ == "__main__":
    main()

