# Video Generation from VTK Files

Generate videos directly from VTK simulation outputs without needing to download files locally.

## Quick Start

```bash
# Generate video from VTK files
python3 scripts/generate_video.py output/cylinder_test/ --output cylinder_flow.mp4

# With custom settings
python3 scripts/generate_video.py output/cylinder_test/ \
    --field velocity_magnitude \
    --fps 30 \
    --colormap viridis \
    --output cylinder_flow.mp4
```

## Methods

### Method 1: ParaView (Recommended - Best Quality)

Uses `pvpython` (ParaView Python) for high-quality rendering with LIC support:

```bash
python3 scripts/generate_video.py output/cylinder_test/ \
    --method pvpython \
    --output video.mp4
```

**Requirements:**
- ParaView installed (pvpython available)
- ffmpeg (optional, for video encoding)

### Method 2: Python Fallback

Uses Python with VTK and matplotlib (works without ParaView):

```bash
python3 scripts/generate_video.py output/cylinder_test/ \
    --method python \
    --output video.mp4
```

**Requirements:**
- Python VTK bindings: `pip install vtk`
- matplotlib: `pip install matplotlib`
- imageio (for video creation): `pip install imageio`

## Options

- `--field`: Field to visualize (`velocity_magnitude`, `density`, `velocity`)
- `--fps`: Frames per second (default: 30)
- `--width/--height`: Video resolution (default: 1920x1080)
- `--colormap`: Color scheme (`viridis`, `plasma`, `jet`)
- `--start/--end`: Frame range (for partial videos)
- `--output`: Output filename

## Examples

```bash
# High-quality cylinder flow video
python3 scripts/generate_video.py output/cylinder_test/ \
    --field velocity_magnitude \
    --colormap plasma \
    --fps 30 \
    --output cylinder_flow.mp4

# Density visualization
python3 scripts/generate_video.py output/cavity_cuda/ \
    --field density \
    --colormap viridis \
    --output cavity_density.mp4

# First 100 frames only
python3 scripts/generate_video.py output/cylinder_test/ \
    --start 0 \
    --end 100 \
    --output cylinder_flow_short.mp4
```

## Installing Dependencies

### Install ffmpeg (Recommended - for video encoding):
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

### For Python method (if needed):
```bash
pip install pillow matplotlib  # For frame generation
pip install imageio            # Alternative to ffmpeg for video creation
```

## Simple Method (Recommended for Headless)

If you have issues with ParaView or complex rendering, use the simple version:

```bash
python3 scripts/generate_video_simple.py output/cylinder_test/ --output video.mp4
```

This version:
- Extracts data directly from VTK files
- Creates frames using PIL/matplotlib
- Works in headless environments
- Only requires: VTK, PIL/Pillow, matplotlib, and ffmpeg (or imageio)

## Output

The script will:
1. Render frames from VTK files
2. Combine them into an MP4 video
3. Save to the specified output file

If ffmpeg/imageio isn't available, frames will be saved to `temp_frames/` and instructions provided for manual video creation.

