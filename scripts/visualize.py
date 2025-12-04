#!/usr/bin/env python3
"""
Simple VTK visualization script using matplotlib
Creates 2D plots of simulation fields
"""

import sys
import os
import glob
import numpy as np
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


def plot_field(arrays, field_name, output_file=None):
    """Plot a 2D field."""
    if field_name not in arrays:
        print(f"Field '{field_name}' not found. Available fields: {list(arrays.keys())}")
        return
    
    data = arrays[field_name]
    
    plt.figure(figsize=(10, 8))
    im = plt.imshow(data, origin='lower', cmap='viridis', interpolation='bilinear')
    plt.colorbar(im, label=field_name)
    plt.title(f'{field_name.replace("_", " ").title()}')
    plt.xlabel('X (lattice units)')
    plt.ylabel('Y (lattice units)')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")
    else:
        plt.show()


def visualize_directory(directory, field_name='velocity_magnitude', save_images=True):
    """Visualize all VTK files in a directory."""
    vtk_files = sorted(glob.glob(os.path.join(directory, 'field_*.vtk')))
    
    if not vtk_files:
        print(f"No VTK files found in {directory}")
        return
    
    print(f"Found {len(vtk_files)} VTK files in {directory}")
    
    output_dir = Path(directory) / 'plots'
    if save_images:
        output_dir.mkdir(exist_ok=True)
    
    for vtk_file in vtk_files:
        print(f"Processing: {os.path.basename(vtk_file)}")
        arrays, nx, ny = read_vtk_file(vtk_file)
        
        # Extract timestep from filename
        timestep = os.path.basename(vtk_file).split('_')[-1].split('.')[0]
        
        if save_images:
            output_file = output_dir / f'{field_name}_t{timestep}.png'
            plot_field(arrays, field_name, str(output_file))
            plt.close()
        else:
            plot_field(arrays, field_name)
            break  # Only show first one if not saving


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python visualize.py <vtk_file>                    # View single file")
        print("  python visualize.py <directory> [field_name]      # View all files in directory")
        print("")
        print("Examples:")
        print("  python visualize.py output/cavity_cpu/field_000050.vtk")
        print("  python visualize.py output/cavity_cpu/ velocity_magnitude")
        print("")
        print("Available fields: velocity_magnitude, density, velocity_x, velocity_y")
        sys.exit(1)
    
    path = sys.argv[1]
    field_name = sys.argv[2] if len(sys.argv) > 2 else 'velocity_magnitude'
    
    if os.path.isfile(path):
        # Single file
        arrays, nx, ny = read_vtk_file(path)
        print(f"Grid size: {nx}x{ny}")
        print(f"Available fields: {list(arrays.keys())}")
        plot_field(arrays, field_name)
    elif os.path.isdir(path):
        # Directory
        visualize_directory(path, field_name, save_images=True)
    else:
        print(f"Error: {path} not found")
        sys.exit(1)


if __name__ == "__main__":
    main()