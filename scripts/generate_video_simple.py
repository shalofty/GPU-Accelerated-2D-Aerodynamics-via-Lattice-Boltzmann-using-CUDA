#!/usr/bin/env python3
"""
Simple video generator for 2D VTK files using numpy and matplotlib.
Better for 2D structured points data.
"""

import argparse
import glob
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
import sys


def read_vtk_structured_points(filename):
    """Read 2D structured points VTK file and return data as numpy array."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    nx, ny = 0, 0
    data_arrays = {}
    current_array = None
    current_data = []
    
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
            # Save previous array if exists
            if current_array and current_data:
                data_arrays[current_array] = np.array(current_data).reshape(ny, nx)
            
            # Start new array
            current_array = line.split()[1]
            current_data = []
            i += 2  # Skip LOOKUP_TABLE line
            continue
        
        if line.startswith('VECTORS'):
            # Save previous array if exists
            if current_array and current_data:
                data_arrays[current_array] = np.array(current_data).reshape(ny, nx, -1)
            
            # Start new vector array
            current_array = line.split()[1]
            current_data = []
            i += 1
            continue
        
        # Parse data lines
        if current_array and line and not line.startswith('#'):
            try:
                values = [float(x) for x in line.split()]
                current_data.extend(values)
            except ValueError:
                pass
        
        i += 1
    
    # Save last array
    if current_array and current_data:
        if 'velocity' in current_array.lower():
            # Vector data - reshape appropriately
            arr = np.array(current_data)
            if len(arr) == nx * ny * 2:
                data_arrays[current_array] = arr.reshape(ny, nx, 2)
            else:
                data_arrays[current_array] = arr.reshape(ny, nx, -1)
        else:
            data_arrays[current_array] = np.array(current_data).reshape(ny, nx)
    
    return nx, ny, data_arrays


def find_vtk_files(directory):
    """Find all VTK files in directory, sorted by timestep."""
    vtk_files = glob.glob(str(Path(directory) / "*.vtk"))
    
    def extract_timestep(filename):
        match = re.search(r'(\d+)', Path(filename).name)
        return int(match.group(1)) if match else 0
    
    return sorted(vtk_files, key=extract_timestep)


def generate_video(vtk_files, output_file, field='velocity_magnitude', fps=30, 
                   width=1920, height=1080, colormap='viridis'):
    """Generate video from VTK files."""
    
    if not vtk_files:
        print("Error: No VTK files found")
        return False
    
    # Create temp directory
    frames_dir = Path(output_file).parent / "temp_frames"
    frames_dir.mkdir(exist_ok=True)
    
    print(f"Processing {len(vtk_files)} VTK files...")
    
    # Process first file to get dimensions
    nx, ny, data = read_vtk_structured_points(vtk_files[0])
    print(f"Data dimensions: {nx}x{ny}")
    print(f"Available fields: {list(data.keys())}")
    
    # Determine aspect ratio
    aspect = nx / ny
    
    # Render each frame
    for i, vtk_file in enumerate(vtk_files):
        nx, ny, data_arrays = read_vtk_structured_points(vtk_file)
        
        # Get the field to display
        if field == 'velocity_magnitude':
            if 'velocity_magnitude' in data_arrays:
                field_data = data_arrays['velocity_magnitude']
            elif 'velocity' in data_arrays:
                vel = data_arrays['velocity']
                if len(vel.shape) == 3 and vel.shape[2] >= 2:
                    field_data = np.sqrt(vel[:, :, 0]**2 + vel[:, :, 1]**2)
                else:
                    field_data = np.abs(vel)
            else:
                print(f"Warning: velocity data not found in {vtk_file}")
                continue
        elif field in data_arrays:
            field_data = data_arrays[field]
        else:
            print(f"Warning: field '{field}' not found in {vtk_file}")
            continue
        
        # Create figure
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        
        # Plot the field
        im = ax.imshow(field_data, cmap=colormap, origin='lower', 
                      aspect='auto', interpolation='bilinear')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label=field)
        
        ax.set_title(f'Timestep {i*50}', fontsize=14)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Save frame
        frame_path = frames_dir / f"frame_{i:06d}.png"
        plt.savefig(frame_path, bbox_inches='tight', dpi=100)
        plt.close()
        
        if (i + 1) % 10 == 0:
            print(f"Rendered {i+1}/{len(vtk_files)} frames")
    
    print(f"All frames rendered to {frames_dir}")
    
    # Create video with ffmpeg
    print(f"Creating video: {output_file}")
    cmd = [
        'ffmpeg', '-y', '-framerate', str(fps),
        '-i', str(frames_dir / 'frame_%06d.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        str(output_file)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Video created: {output_file}")
        # Cleanup
        import shutil
        shutil.rmtree(frames_dir)
        return True
    else:
        print(f"Error creating video: {result.stderr}")
        print(f"Frames saved to: {frames_dir}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate video from 2D VTK files")
    parser.add_argument("vtk_directory", help="Directory containing VTK files")
    parser.add_argument("--field", default="velocity_magnitude",
                       choices=["velocity_magnitude", "density"],
                       help="Field to visualize")
    parser.add_argument("--output", default="animation.mp4",
                       help="Output video filename")
    parser.add_argument("--fps", type=int, default=30,
                       help="Frames per second")
    parser.add_argument("--width", type=int, default=1920,
                       help="Video width")
    parser.add_argument("--height", type=int, default=1080,
                       help="Video height")
    parser.add_argument("--colormap", default="viridis",
                       help="Colormap name (viridis, plasma, jet, etc.)")
    
    args = parser.parse_args()
    
    vtk_files = find_vtk_files(args.vtk_directory)
    if not vtk_files:
        print(f"Error: No VTK files found in {args.vtk_directory}")
        return 1
    
    print(f"Found {len(vtk_files)} VTK files")
    
    success = generate_video(
        vtk_files, args.output, args.field, args.fps,
        args.width, args.height, args.colormap
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
