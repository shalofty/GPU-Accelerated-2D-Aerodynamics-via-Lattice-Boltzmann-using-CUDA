# Quick Video Generation Guide

Generate videos from VTK files directly on the server - no need to download files!

## Quick Start

```bash
# Install ffmpeg (one-time setup)
sudo apt-get update && sudo apt-get install -y ffmpeg

# Generate video from your simulation
python3 scripts/generate_video_simple.py output/cylinder_test/ --output cylinder_flow.mp4
```

## Examples

```bash
# Basic usage
python3 scripts/generate_video_simple.py output/cylinder_test/ --output video.mp4

# Custom field and FPS
python3 scripts/generate_video_simple.py output/cylinder_test/ \
    --field velocity_magnitude \
    --fps 30 \
    --output cylinder_flow.mp4

# First 100 frames only
python3 scripts/generate_video_simple.py output/cylinder_test/ \
    --start 0 --end 100 \
    --output short_video.mp4
```

## What It Does

1. Reads all VTK files from the directory
2. Extracts the field data (velocity_magnitude, density, etc.)
3. Creates PNG frames
4. Combines frames into MP4 video using ffmpeg

## Output

- Video file: `cylinder_flow.mp4` (or your specified name)
- Temporary frames: `temp_frames/` (automatically cleaned up)

## Requirements

- VTK Python bindings (already installed)
- PIL/Pillow (already installed)
- ffmpeg (install with command above)

That's it! No need to push/pull files to GitHub anymore.
