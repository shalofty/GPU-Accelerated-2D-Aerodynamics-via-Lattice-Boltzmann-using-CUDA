#!/usr/bin/env python3
"""
Simple video generation from VTK files - extracts data and creates frames.

This version works in headless environments without X11 or complex rendering.

Usage:
    python3 generate_video_simple.py <vtk_directory> [options]
"""

import argparse
import os
import sys
import glob
import re
import subprocess
from pathlib import Path
import numpy as np


def find_vtk_files(directory):
    """Find all VTK files in directory, sorted by timestep."""
    vtk_files = glob.glob(os.path.join(directory, "*.vtk"))
    
    def extract_timestep(filename):
        match = re.search(r'(\d+)', os.path.basename(filename))
        return int(match.group(1)) if match else 0
    
    return sorted(vtk_files, key=extract_timestep)


def read_legacy_vtk_ascii(filename):
    """Read legacy ASCII VTK format - same parser as view_vtk.py."""
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


def read_vtk_field(vtk_file, field_name):
    """Read a field from a VTK file and return as 2D numpy array."""
    try:
        arrays, nx, ny = read_legacy_vtk_ascii(vtk_file)
    except Exception as e:
        return None
    
    # Map field names
    if field_name == "velocity_magnitude":
        if "velocity_magnitude" in arrays:
            return arrays["velocity_magnitude"]
        elif "velocity_magnitude" in arrays:  # From VECTORS
            return arrays["velocity_magnitude"]
    elif field_name == "density":
        if "density" in arrays:
            return arrays["density"]
    
    # Fallback: try velocity_magnitude
    if "velocity_magnitude" in arrays:
        return arrays["velocity_magnitude"]
    
    return None


def create_frame_simple(values_2d, output_path, field_name, frame_num, total_frames):
    """Create a simple frame using PIL/Pillow with a simple colormap."""
    try:
        from PIL import Image
    except ImportError:
        # Fallback: save raw data
        np.save(output_path.replace('.png', '.npy'), values_2d)
        print(f"Note: PIL not available. Saved raw data to {output_path.replace('.png', '.npy')}")
        print("Install with: pip install pillow")
        return False
    
    # Normalize values
    vmin, vmax = np.nanmin(values_2d), np.nanmax(values_2d)
    if vmax > vmin:
        normalized = (values_2d - vmin) / (vmax - vmin)
    else:
        normalized = np.zeros_like(values_2d)
    
    # Handle NaN/Inf
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
    normalized = np.clip(normalized, 0.0, 1.0)
    
    # Simple colormap: blue (low) -> cyan -> yellow -> red (high)
    # This approximates viridis/plasma without matplotlib
    r = np.zeros_like(normalized)
    g = np.zeros_like(normalized)
    b = np.zeros_like(normalized)
    
    # Blue to cyan (0.0 - 0.33)
    mask1 = normalized < 0.33
    t1 = normalized[mask1] / 0.33
    r[mask1] = 0.0
    g[mask1] = t1
    b[mask1] = 1.0
    
    # Cyan to yellow (0.33 - 0.67)
    mask2 = (normalized >= 0.33) & (normalized < 0.67)
    t2 = (normalized[mask2] - 0.33) / 0.34
    r[mask2] = t2
    g[mask2] = 1.0
    b[mask2] = 1.0 - t2
    
    # Yellow to red (0.67 - 1.0)
    mask3 = normalized >= 0.67
    t3 = (normalized[mask3] - 0.67) / 0.33
    r[mask3] = 1.0
    g[mask3] = 1.0 - t3
    b[mask3] = 0.0
    
    # Convert to uint8
    img_array = np.stack([r, g, b], axis=2)
    img_array = (img_array * 255).astype(np.uint8)
    
    # Create image
    img = Image.fromarray(img_array)
    img.save(output_path)
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate video from VTK files (simple version)")
    parser.add_argument("vtk_directory", help="Directory containing VTK files")
    parser.add_argument("--field", default="velocity_magnitude",
                       choices=["velocity_magnitude", "density"],
                       help="Field to visualize")
    parser.add_argument("--output", default="animation.mp4",
                       help="Output video filename")
    parser.add_argument("--fps", type=int, default=30,
                       help="Frames per second")
    parser.add_argument("--start", type=int, default=0,
                       help="Start frame number")
    parser.add_argument("--end", type=int, default=None,
                       help="End frame number")
    
    args = parser.parse_args()
    
    # Find VTK files
    vtk_files = find_vtk_files(args.vtk_directory)
    if not vtk_files:
        print(f"Error: No VTK files found in {args.vtk_directory}")
        return 1
    
    # Filter by range
    if args.end is None:
        args.end = len(vtk_files)
    vtk_files = vtk_files[args.start:args.end]
    
    if not vtk_files:
        print("Error: No VTK files in specified range")
        return 1
    
    print(f"Processing {len(vtk_files)} VTK files...")
    
    # Create frames directory
    output_path = Path(args.output)
    frames_dir = output_path.parent / "temp_frames"
    frames_dir.mkdir(exist_ok=True)
    
    # Extract data and create frames
    print("Extracting data and creating frames...")
    for i, vtk_file in enumerate(vtk_files):
        values_2d = read_vtk_field(vtk_file, args.field)
        if values_2d is None:
            print(f"Warning: Could not read {vtk_file}, skipping")
            continue
        
        frame_path = frames_dir / f"frame_{i:06d}.png"
        if not create_frame_simple(values_2d, str(frame_path), args.field, i, len(vtk_files)):
            continue
        
        if (i + 1) % 10 == 0:
            print(f"Created {i+1}/{len(vtk_files)} frames")
    
    print(f"Frames created in: {frames_dir}")
    
    # Try to create video with imageio first (works without system dependencies)
    try:
        import imageio.v2 as imageio  # Use v2 API to avoid deprecation warnings
        print(f"Creating video: {args.output}")
        images = sorted(glob.glob(str(frames_dir / "*.png")))
        if not images:
            print("Error: No frames found")
            return 1
        imageio.mimsave(args.output, [imageio.imread(img) for img in images], fps=args.fps)
        print(f"Video created: {args.output}")
        
        # Cleanup
        import shutil
        shutil.rmtree(frames_dir)
        return 0
    except ImportError:
        pass  # Fall through to ffmpeg
    except Exception as e:
        print(f"Error creating video with imageio: {e}")
        print("Trying ffmpeg...")
    
    # Fallback to ffmpeg
    frame_pattern = str(frames_dir / "frame_%06d.png")
    try:
        subprocess.run(["which", "ffmpeg"], check=True, capture_output=True)
        print(f"Creating video: {args.output}")
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(args.fps),
            "-i", frame_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            str(args.output)
        ]
        subprocess.run(cmd, check=True)
        print(f"Video created: {args.output}")
        
        # Cleanup
        import shutil
        shutil.rmtree(frames_dir)
        return 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"\nFrames are in: {frames_dir}")
        print(f"To create video, install imageio:")
        print(f"  pip install imageio")
        print(f"  python3 -c \"import imageio, glob; images=sorted(glob.glob('{frames_dir}/*.png')); imageio.mimsave('{args.output}', [imageio.imread(img) for img in images], fps={args.fps})\"")
        print(f"\nOr install ffmpeg and run:")
        print(f"  ffmpeg -framerate {args.fps} -i {frame_pattern} -c:v libx264 -pix_fmt yuv420p {args.output}")
        return 0


if __name__ == "__main__":
    sys.exit(main())

