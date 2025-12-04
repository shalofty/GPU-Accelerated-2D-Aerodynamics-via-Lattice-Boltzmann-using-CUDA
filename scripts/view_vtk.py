#!/usr/bin/env python3
"""
Simple VTK file viewer using VTK's offscreen rendering
Works in headless/Docker environments without X11
"""

import sys
import os
import glob
import numpy as np

try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
except ImportError:
    print("Error: VTK not installed. Install with: pip install vtk")
    sys.exit(1)

# Enable offscreen rendering (no display needed)
vtk.vtkRenderWindow.GlobalWarningDisplayOff()


def read_legacy_vtk_ascii(filename):
    """Read legacy ASCII VTK format."""
    arrays = {}
    nx, ny = 0, 0
    num_points = 0
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
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
                        vals = [float(x) for x in line.split()]
                        values.extend(vals)
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
                        vals = [float(x) for x in line.split()]
                        values.extend(vals)
                    except:
                        pass
                i += 1
            if len(values) >= num_points * 3:
                vec_data = np.array(values[:num_points * 3]).reshape(num_points, 3)
                arrays[f'{field_name}_x'] = vec_data[:, 0].reshape(ny, nx)
                arrays[f'{field_name}_y'] = vec_data[:, 1].reshape(ny, nx)
                mag = np.sqrt(vec_data[:, 0]**2 + vec_data[:, 1]**2)
                arrays[f'{field_name}_magnitude'] = mag.reshape(ny, nx)
            continue
        
        i += 1
    
    return arrays, nx, ny


def save_field_as_text(arrays, field_name, output_file):
    """Save field data as text file for inspection."""
    if field_name not in arrays:
        return False
    
    data = arrays[field_name]
    np.savetxt(output_file, data, fmt='%.6e')
    return True


def print_field_info(arrays, field_name):
    """Print field statistics."""
    if field_name not in arrays:
        print(f"Field '{field_name}' not found.")
        print(f"Available fields: {list(arrays.keys())}")
        return
    
    data = arrays[field_name]
    print(f"\n{field_name}:")
    print(f"  Shape: {data.shape}")
    print(f"  Min: {np.min(data):.6e}")
    print(f"  Max: {np.max(data):.6e}")
    print(f"  Mean: {np.mean(data):.6e}")
    print(f"  Std: {np.std(data):.6e}")


def visualize_directory(directory, field_name='velocity_magnitude', output_dir=None):
    """Process all VTK files in a directory."""
    vtk_files = sorted(glob.glob(os.path.join(directory, 'field_*.vtk')))
    
    if not vtk_files:
        print(f"No VTK files found in {directory}")
        return
    
    print(f"Found {len(vtk_files)} VTK files in {directory}")
    
    if output_dir is None:
        output_dir = os.path.join(directory, 'data')
    os.makedirs(output_dir, exist_ok=True)
    
    for vtk_file in vtk_files:
        print(f"\nProcessing: {os.path.basename(vtk_file)}")
        arrays, nx, ny = read_legacy_vtk_ascii(vtk_file)
        
        print(f"  Grid size: {nx}x{ny}")
        print(f"  Available fields: {list(arrays.keys())}")
        
        # Print info for requested field
        print_field_info(arrays, field_name)
        
        # Save as text file
        timestep = os.path.basename(vtk_file).split('_')[-1].split('.')[0]
        output_file = os.path.join(output_dir, f'{field_name}_t{timestep}.txt')
        if save_field_as_text(arrays, field_name, output_file):
            print(f"  Saved data: {output_file}")
    
    print(f"\nAll data files saved to: {output_dir}")
    print("\nTo visualize:")
    print("  1. Copy files to a machine with ParaView/matplotlib")
    print("  2. Or use: python3 -c \"import numpy as np; import matplotlib.pyplot as plt; data=np.loadtxt('{output_file}'); plt.imshow(data); plt.savefig('plot.png')\"")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python view_vtk.py <vtk_file>                    # Inspect single file")
        print("  python view_vtk.py <directory> [field_name]      # Process all files")
        print("")
        print("Examples:")
        print("  python view_vtk.py output/cavity_cpu/field_000050.vtk")
        print("  python view_vtk.py output/cavity_cpu/ velocity_magnitude")
        print("")
        print("Available fields: velocity_magnitude, density, velocity_x, velocity_y")
        sys.exit(1)
    
    path = sys.argv[1]
    field_name = sys.argv[2] if len(sys.argv) > 2 else 'velocity_magnitude'
    
    if os.path.isfile(path):
        # Single file
        arrays, nx, ny = read_legacy_vtk_ascii(path)
        print(f"Grid size: {nx}x{ny}")
        print(f"Available fields: {list(arrays.keys())}")
        print_field_info(arrays, field_name)
        
        # Save as text
        output_file = path.replace('.vtk', f'_{field_name}.txt')
        if save_field_as_text(arrays, field_name, output_file):
            print(f"\nData saved to: {output_file}")
    elif os.path.isdir(path):
        # Directory
        visualize_directory(path, field_name)
    else:
        print(f"Error: {path} not found")
        sys.exit(1)


if __name__ == "__main__":
    main()

