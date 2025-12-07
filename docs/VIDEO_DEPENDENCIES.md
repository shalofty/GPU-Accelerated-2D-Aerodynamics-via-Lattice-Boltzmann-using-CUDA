# Video Generation Dependencies

This document lists all dependencies required to generate videos from VTK simulation output files using the `generate_video.py` script.

## System Dependencies

These are system-level packages that need to be installed via the package manager (apt-get on Ubuntu/Debian).

### Build Dependencies
- **cmake** - Build system for compiling the simulation
- **build-essential** - Essential build tools (gcc, g++, make, etc.)

### Graphics/Rendering Dependencies
Required for VTK's offscreen rendering capabilities:

- **libxrender1** - X11 Render extension library
- **libxext6** - X11 miscellaneous extension library
- **libgl1-mesa-glx** - Mesa OpenGL runtime
- **libglu1-mesa** - Mesa GLU library
- **libosmesa6-dev** - Offscreen Mesa library (for headless rendering)

### Video Encoding
- **ffmpeg** - Video encoding/decoding tool for creating MP4 files

### Installation Command (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y \
    cmake \
    build-essential \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    libglu1-mesa \
    libosmesa6-dev \
    ffmpeg
```

## Python Dependencies

These are Python packages installed in a virtual environment.

### Core Dependencies
- **vtk** (version 9.5.2+) - Visualization Toolkit for reading and rendering VTK files
- **imageio** (version 2.37.2+) - Image I/O library (optional, for video creation via Python)

### Automatic Dependencies
The following are automatically installed as dependencies of the above packages:

- **numpy** - Numerical computing (required by vtk, imageio)
- **matplotlib** - Plotting library (required by vtk)
- **pillow** - Image processing (required by imageio)
- **contourpy**, **cycler**, **fonttools**, **kiwisolver**, **packaging**, **pyparsing**, **python-dateutil**, **six** - Matplotlib dependencies

### Installation via Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
pip install vtk imageio
```

### Installation via pip (System-wide, not recommended)

```bash
pip install vtk imageio
```

## Environment Variables

For headless rendering, you may need to set:

```bash
export MESA_GL_VERSION_OVERRIDE=3.3
```

## Complete Setup Script

Here's a complete setup script for Ubuntu/Debian systems:

```bash
#!/bin/bash
set -e

echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    cmake \
    build-essential \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    libglu1-mesa \
    libosmesa6-dev \
    ffmpeg

echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install vtk imageio

echo "Setup complete!"
echo "To use the virtual environment, run: source venv/bin/activate"
```

## Usage

After installing all dependencies:

```bash
# Activate virtual environment
source venv/bin/activate

# Set environment variable for headless rendering (if needed)
export MESA_GL_VERSION_OVERRIDE=3.3

# Generate video
python3 scripts/generate_video.py output/cylinder_columns/ \
    --method python \
    --output cylinder_columns.mp4 \
    --field velocity_magnitude
```

## Alternative: Using ffmpeg Directly

If imageio has issues creating videos, you can use ffmpeg directly after frames are rendered:

```bash
ffmpeg -framerate 30 \
    -i temp_frames/frame_%06d.png \
    -c:v libx264 \
    -pix_fmt yuv420p \
    -y output.mp4
```

## Troubleshooting

### VTK Import Errors
- **Error**: `libXrender.so.1: cannot open shared object file`
  - **Solution**: Install `libxrender1`

- **Error**: `Failed to load EGL` or `libOSMesa not found`
  - **Solution**: Install `libosmesa6-dev`

### Video Creation Errors
- **Error**: `Could not find a backend to open ... with iomode wI`
  - **Solution**: Use ffmpeg directly instead of imageio, or install `imageio[ffmpeg]`

### Rendering Issues
- **Issue**: Video shows only a red rectangle or blank frames
  - **Solution**: Ensure camera/view setup is correct for 2D data (see `generate_video.py`)

## Version Information

Tested with:
- **Ubuntu**: 22.04.5 LTS
- **Python**: 3.11
- **VTK**: 9.5.2
- **CUDA**: 12.4.131
- **ffmpeg**: 4.4.2

## Notes

- The virtual environment approach is recommended to avoid conflicts with system Python packages
- Some dependencies (like OSMesa) are specifically needed for headless/off-screen rendering in Docker or server environments
- The `generate_video.py` script supports two methods:
  - `pvpython`: Requires ParaView installation (not covered here)
  - `python`: Uses VTK Python bindings (covered in this document)

