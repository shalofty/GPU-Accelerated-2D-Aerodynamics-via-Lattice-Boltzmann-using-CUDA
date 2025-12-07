#!/usr/bin/env python3
"""
Generate video from VTK files using ParaView batch mode or Python fallback.

Usage:
    python3 generate_video.py <vtk_directory> [options]

Options:
    --field <name>          Field to visualize (velocity_magnitude, density, velocity) [default: velocity_magnitude]
    --output <file>         Output video filename [default: animation.mp4]
    --fps <number>          Frames per second [default: 30]
    --width <pixels>        Video width [default: 1920]
    --height <pixels>       Video height [default: 1080]
    --colormap <name>       Colormap name (viridis, plasma, jet, etc.) [default: viridis]
    --method <pvpython|python>  Rendering method [default: pvpython]
    --start <frame>         Start frame number [default: 0]
    --end <frame>           End frame number [default: all]
"""

import argparse
import os
import sys
import subprocess
import glob
import re
from pathlib import Path


def find_vtk_files(directory):
    """Find all VTK files in directory, sorted by timestep."""
    vtk_files = glob.glob(os.path.join(directory, "*.vtk"))
    
    def extract_timestep(filename):
        # Extract number from filename like "field_000300.vtk"
        match = re.search(r'(\d+)', os.path.basename(filename))
        return int(match.group(1)) if match else 0
    
    return sorted(vtk_files, key=extract_timestep)


def generate_video_pvpython(vtk_files, output_file, field, fps, width, height, colormap, start_frame, end_frame):
    """Generate video using ParaView's pvpython."""
    
    # Filter files by frame range
    if end_frame is None:
        end_frame = len(vtk_files)
    vtk_files = vtk_files[start_frame:end_frame]
    
    if not vtk_files:
        print("Error: No VTK files found in specified range")
        return False
    
    # Create temporary directory for frames
    frames_dir = Path(output_file).parent / "temp_frames"
    frames_dir.mkdir(exist_ok=True)
    
    # ParaView Python script
    pvpython_script = f"""
import paraview.simple as pv
import os
import sys

# Configuration
vtk_files = {repr(vtk_files)}
frames_dir = {repr(str(frames_dir))}
field_name = {repr(field)}
colormap = {repr(colormap)}
width = {width}
height = {height}

# Create reader
reader = pv.LegacyVTKReader(FileNames=vtk_files)

# Create render view
view = pv.CreateRenderView()
view.ViewSize = [width, height]
view.Background = [1.0, 1.0, 1.0]

# Create representation
rep = pv.Show(reader, view)

# Set field to display
if field_name == "velocity_magnitude":
    rep.ColorArrayName = ["POINTS", "velocity_magnitude"]
elif field_name == "density":
    rep.ColorArrayName = ["POINTS", "density"]
elif field_name == "velocity":
    rep.ColorArrayName = ["POINTS", "velocity"]
    rep.SetRepresentationType("Surface With Edges")
else:
    rep.ColorArrayName = ["POINTS", "velocity_magnitude"]

# Set colormap
lut = rep.LookupTable
if colormap == "viridis":
    lut.ApplyPreset("Viridis (matplotlib)", True)
elif colormap == "plasma":
    lut.ApplyPreset("Plasma (matplotlib)", True)
elif colormap == "jet":
    lut.ApplyPreset("Jet", True)
else:
    lut.ApplyPreset("Viridis (matplotlib)", True)

# Set scalar bar
scalarBar = pv.GetScalarBar(lut, view)
scalarBar.Visibility = 1
scalarBar.Title = field_name.replace("_", " ").title()

# Camera setup
view.CameraPosition = [0, 0, 1000]
view.CameraFocalPoint = [0, 0, 0]
view.CameraViewUp = [0, 1, 0]
view.CameraParallelProjection = 1

# Render each timestep
for i, vtk_file in enumerate(vtk_files):
    reader.FileName = vtk_file
    reader.UpdatePipeline()
    
    # Update view
    view.Update()
    
    # Save frame
    frame_path = os.path.join(frames_dir, f"frame_{{i:06d}}.png")
    pv.SaveScreenshot(frame_path, view, ImageResolution=[width, height])
    
    if (i + 1) % 10 == 0:
        print(f"Rendered {{i+1}}/{{len(vtk_files)}} frames")

print(f"All frames rendered to {{frames_dir}}")
"""
    
    # Write script to temp file
    script_path = frames_dir / "render_script.py"
    with open(script_path, 'w') as f:
        f.write(pvpython_script)
    
    # Run pvpython
    print(f"Rendering {len(vtk_files)} frames using ParaView...")
    try:
        result = subprocess.run(
            ["pvpython", str(script_path)],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running pvpython: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: pvpython not found. Install ParaView or use --method python")
        return False
    
    # Combine frames into video using ffmpeg (if available) or provide instructions
    frame_pattern = str(frames_dir / "frame_%06d.png")
    
    # Check for ffmpeg
    try:
        subprocess.run(["which", "ffmpeg"], check=True, capture_output=True)
        ffmpeg_available = True
    except:
        ffmpeg_available = False
    
    if ffmpeg_available:
        print(f"Combining frames into video: {output_file}")
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", frame_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            str(output_file)
        ]
        try:
            subprocess.run(cmd, check=True)
            print(f"Video created: {output_file}")
            
            # Cleanup frames
            import shutil
            shutil.rmtree(frames_dir)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error creating video: {e}")
            print(f"Frames are available in: {frames_dir}")
            print(f"To create video manually, run:")
            print(f"  ffmpeg -framerate {fps} -i {frame_pattern} -c:v libx264 -pix_fmt yuv420p {output_file}")
            return False
    else:
        print(f"\nFrames rendered to: {frames_dir}")
        print(f"To create video, install ffmpeg and run:")
        print(f"  ffmpeg -framerate {fps} -i {frame_pattern} -c:v libx264 -pix_fmt yuv420p {output_file}")
        print(f"\nOr use Python with imageio:")
        print(f"  python3 -c \"import imageio; import glob; images = sorted(glob.glob('{frames_dir}/*.png')); imageio.mimsave('{output_file}', [imageio.imread(img) for img in images], fps={fps})\"")
        return True


def generate_video_python(vtk_files, output_file, field, fps, width, height, colormap, start_frame, end_frame):
    """Generate video using Python with VTK (fallback method)."""
    
    try:
        import vtk
        import numpy as np
    except ImportError:
        print("Error: VTK Python bindings not available. Install with: pip install vtk")
        return False
    
    # Filter files by frame range
    if end_frame is None:
        end_frame = len(vtk_files)
    vtk_files = vtk_files[start_frame:end_frame]
    
    if not vtk_files:
        print("Error: No VTK files found in specified range")
        return False
    
    # Create temporary directory for frames
    frames_dir = Path(output_file).parent / "temp_frames"
    frames_dir.mkdir(exist_ok=True)
    
    print(f"Rendering {len(vtk_files)} frames using Python+VTK...")
    
    # Setup VTK rendering pipeline
    reader = vtk.vtkStructuredPointsReader()
    
    # For 2D data, convert to image data for better visualization
    # Create a filter to convert structured points to image data
    image_data_filter = vtk.vtkImageDataGeometryFilter()
    image_data_filter.SetInputConnection(reader.GetOutputPort())
    
    # Create mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(image_data_filter.GetOutputPort())
    mapper.ScalarVisibilityOn()
    
    # Create actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    # Use surface representation for 2D data
    actor.GetProperty().SetRepresentationToSurface()
    
    # Setup lookup table for colormap
    lut = vtk.vtkLookupTable()
    if colormap == "viridis" or colormap == "plasma":
        # Use a blue-to-yellow colormap as approximation
        lut.SetHueRange(0.6, 0.0)  # Blue to yellow
        lut.SetSaturationRange(1.0, 1.0)
        lut.SetValueRange(1.0, 1.0)
    elif colormap == "jet":
        lut.SetHueRange(0.0, 0.67)  # Red to blue
        lut.SetSaturationRange(1.0, 1.0)
        lut.SetValueRange(1.0, 1.0)
    else:
        lut.SetHueRange(0.6, 0.0)
        lut.SetSaturationRange(1.0, 1.0)
        lut.SetValueRange(1.0, 1.0)
    lut.Build()
    mapper.SetLookupTable(lut)
    mapper.SetUseLookupTableScalarRange(True)
    
    # Create renderer
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(1.0, 1.0, 1.0)  # White background
    
    # Create render window (offscreen)
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetOffScreenRendering(1)
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(width, height)
    
    # Create window-to-image filter
    windowToImage = vtk.vtkWindowToImageFilter()
    windowToImage.SetInput(renderWindow)
    windowToImage.SetInputBufferTypeToRGB()
    windowToImage.ReadFrontBufferOff()
    
    # Create PNG writer
    writer = vtk.vtkPNGWriter()
    writer.SetInputConnection(windowToImage.GetOutputPort())
    
    # Process each file
    for i, vtk_file in enumerate(vtk_files):
        reader.SetFileName(vtk_file)
        reader.Update()
        
        # Set the field to display
        data = reader.GetOutput()
        point_data = data.GetPointData()
        
        # Get the array to display
        if field == "velocity_magnitude":
            # Calculate velocity magnitude if not present
            vel_array = point_data.GetArray("velocity")
            if vel_array and vel_array.GetNumberOfComponents() >= 2:
                # Create velocity magnitude array
                num_points = vel_array.GetNumberOfTuples()
                vel_mag = vtk.vtkFloatArray()
                vel_mag.SetName("velocity_magnitude")
                vel_mag.SetNumberOfTuples(num_points)
                for j in range(num_points):
                    vx = vel_array.GetComponent(j, 0)
                    vy = vel_array.GetComponent(j, 1)
                    mag = (vx*vx + vy*vy)**0.5
                    vel_mag.SetValue(j, mag)
                point_data.AddArray(vel_mag)
                array = vel_mag
            else:
                array = point_data.GetArray("velocity_magnitude")
        elif field == "density":
            array = point_data.GetArray("density")
        else:
            array = point_data.GetArray("velocity_magnitude")
        
        if array:
            point_data.SetActiveScalars(array.GetName())
            mapper.SetScalarModeToUsePointData()
            mapper.SetColorModeToMapScalars()
            mapper.SetScalarRange(array.GetRange())
        
        # Setup camera based on data bounds (do this for each frame in case bounds change)
        bounds = data.GetBounds()
        center_x = (bounds[0] + bounds[1]) / 2.0
        center_y = (bounds[2] + bounds[3]) / 2.0
        center_z = (bounds[4] + bounds[5]) / 2.0
        
        data_width = bounds[1] - bounds[0]
        data_height = bounds[3] - bounds[2]
        
        # Setup camera to view XY plane
        camera = renderer.GetActiveCamera()
        camera.SetPosition(center_x, center_y, center_z + max(data_width, data_height))
        camera.SetFocalPoint(center_x, center_y, center_z)
        camera.SetViewUp(0, 1, 0)
        camera.ParallelProjectionOn()
        
        # Set parallel scale to fit the data
        if data_width / data_height > width / height:
            camera.SetParallelScale(data_width / 2.0 * height / width)
        else:
            camera.SetParallelScale(data_height / 2.0)
        
        # Reset camera to show all data
        renderer.ResetCamera()
        
        # Render
        renderWindow.Render()
        windowToImage.Modified()
        
        # Save frame
        frame_path = str(frames_dir / f"frame_{i:06d}.png")
        writer.SetFileName(frame_path)
        writer.Write()
        
        if (i + 1) % 10 == 0:
            print(f"Rendered {i+1}/{len(vtk_files)} frames")
    
    print(f"All frames rendered to {frames_dir}")
    
    # Try to create video with imageio
    try:
        import imageio
        print(f"Creating video: {output_file}")
        images = sorted(glob.glob(str(frames_dir / "*.png")))
        imageio.mimsave(output_file, [imageio.imread(img) for img in images], fps=fps)
        print(f"Video created: {output_file}")
        
        # Cleanup
        import shutil
        shutil.rmtree(frames_dir)
        return True
    except ImportError:
        print(f"\nFrames rendered to: {frames_dir}")
        print(f"Install imageio to create video automatically:")
        print(f"  pip install imageio")
        print(f"Or use ffmpeg:")
        print(f"  ffmpeg -framerate {fps} -i {frames_dir}/frame_%06d.png -c:v libx264 -pix_fmt yuv420p {output_file}")
        return True


def main():
    parser = argparse.ArgumentParser(description="Generate video from VTK files")
    parser.add_argument("vtk_directory", help="Directory containing VTK files")
    parser.add_argument("--field", default="velocity_magnitude",
                       choices=["velocity_magnitude", "density", "velocity"],
                       help="Field to visualize")
    parser.add_argument("--output", default="animation.mp4",
                       help="Output video filename")
    parser.add_argument("--fps", type=int, default=30,
                       help="Frames per second")
    parser.add_argument("--width", type=int, default=1920,
                       help="Video width in pixels")
    parser.add_argument("--height", type=int, default=1080,
                       help="Video height in pixels")
    parser.add_argument("--colormap", default="viridis",
                       choices=["viridis", "plasma", "jet"],
                       help="Colormap name")
    parser.add_argument("--method", default="pvpython",
                       choices=["pvpython", "python"],
                       help="Rendering method")
    parser.add_argument("--start", type=int, default=0,
                       help="Start frame number (0-indexed)")
    parser.add_argument("--end", type=int, default=None,
                       help="End frame number (exclusive, None for all)")
    
    args = parser.parse_args()
    
    # Find VTK files
    vtk_files = find_vtk_files(args.vtk_directory)
    if not vtk_files:
        print(f"Error: No VTK files found in {args.vtk_directory}")
        return 1
    
    print(f"Found {len(vtk_files)} VTK files")
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate video
    if args.method == "pvpython":
        success = generate_video_pvpython(
            vtk_files, args.output, args.field, args.fps,
            args.width, args.height, args.colormap,
            args.start, args.end
        )
    else:
        success = generate_video_python(
            vtk_files, args.output, args.field, args.fps,
            args.width, args.height, args.colormap,
            args.start, args.end
        )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

