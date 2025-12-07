# Visualization Guide

Complete guide for visualizing LBM simulation results.

## Quick Start: ParaView (Recommended)

### Installation

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install paraview
```

**macOS:**
```bash
brew install paraview
```

**Windows:**
Download from: https://www.paraview.org/download/

### Opening Files

1. **Launch ParaView**
   ```bash
   paraview
   ```

2. **Open VTK Files**
   - File → Open
   - Navigate to `output/cavity_cpu/` or `output/cylinder_cpu/`
   - Select one or more VTK files (e.g., `field_000050.vtk`)
   - **Check "Group files into time series"** (for animation)
   - Click "OK"
   - Click "Apply" in Properties panel

3. **Visualize**
   - In "Coloring" dropdown, select "velocity_magnitude"
   - Click the ▶️ play button to animate through timesteps

## Understanding Your Visualization

### What You're Seeing

**Lid-Driven Cavity Flow:**
- **Top Boundary (Lid)**: Moving at specified velocity, creating shear
- **Recirculation Zones**: Large rotating vortices
- **Primary Vortex**: Main recirculation zone (typically upper-right quadrant)
- **Flow Development**: Pattern stabilizes over time

**Cylinder Flow:**
- **Wake Formation**: Flow separation behind cylinder
- **Vortex Shedding**: Periodic vortex formation (at higher Re)
- **Flow Separation**: Boundary layer separation points

### Color Coding

Colors represent **velocity magnitude** (speed):
- **Dark Blue**: Low velocity (~0.0) - vortex centers, corners
- **Light Blue/Green**: Moderate velocity (~0.05) - transition regions
- **Yellow/Orange**: High velocity (~0.1) - near moving boundaries
- **Red**: Highest velocity (~0.14+) - strongest flow regions

### Available Fields

Each VTK file contains:
- **density**: Fluid density field
- **velocity_x**: X-component of velocity
- **velocity_y**: Y-component of velocity  
- **velocity_magnitude**: Speed (√(vx² + vy²))

## Recommended Visualizations

### 1. Velocity Magnitude (Color-coded)
- Default view after loading
- Select "velocity_magnitude" in Coloring dropdown
- Shows flow speed distribution

### 2. Velocity Vectors (Glyphs)
- Filters → Glyphs
- Glyph Type: "Arrow"
- Orientation: "velocity"
- Scale: "velocity_magnitude"
- Click "Apply"

### 3. Streamlines
- Filters → Stream Tracer
- Seed Type: "Line" or "Point Cloud"
- Shows flow paths
- Click "Apply"

### 4. Contour Lines
- Filters → Contour
- Contour By: "velocity_magnitude" or "density"
- Isosurfaces: Add values (e.g., 0.05, 0.1, 0.15)
- Click "Apply"

### 5. Slice View
- Filters → Slice
- Plane: "XY Plane" (for 2D)
- Click "Apply"

## Animation Controls

- Use play/pause buttons at top to animate through timesteps
- Adjust playback speed with speed slider
- Step through frames manually with arrow buttons
- Loop animation for continuous playback

## Alternative: Python Visualization

### Installation
```bash
pip install vtk numpy matplotlib
```

### Quick View Script
```bash
# View single file
python3 scripts/view_vtk.py output/cavity_cpu/

# Generate images for all files
python3 scripts/view_headless.py output/cavity_cpu/ --output-dir plots/
```

## Generating Videos

See [VIDEO_DEPENDENCIES.md](VIDEO_DEPENDENCIES.md) for complete setup.

**Quick command:**
```bash
source venv/bin/activate
python3 scripts/generate_video.py output/cylinder_columns/ \
    --method python \
    --output cylinder_columns.mp4 \
    --field velocity_magnitude
```

## File Locations

- **Cavity Flow (CPU)**: `output/cavity_cpu/field_*.vtk`
- **Cavity Flow (CUDA)**: `output/cavity_cuda/field_*.vtk`
- **Cylinder Flow (CPU)**: `output/cylinder_cpu/field_*.vtk`
- **Cylinder Flow (CUDA)**: `output/cylinder_cuda/field_*.vtk`
- **Showcase Simulations**: `output/cylinder_*/field_*.vtk`

## Tips

1. **Start with one file** to test, then load the time series
2. **Use "velocity_magnitude"** for color coding - shows flow speed
3. **Add streamlines** to see flow paths
4. **Export images** (File → Save Screenshot) for reports
5. **Compare CPU vs CUDA** - results should be identical!

## Troubleshooting

**ParaView won't start?**
- Try: `paraview --mesa` (software rendering)

**No data visible?**
- Make sure you clicked "Apply"
- Check "Coloring" dropdown has a field selected
- Try different fields (velocity_magnitude, density)

**Animation not working?**
- Make sure "Group files into time series" was checked when opening
- Files must be in the same directory
- Check file naming is sequential

## Next Steps

1. **Explore different visualizations**: Streamlines, contours, slices
2. **Compare CPU vs CUDA**: Load both outputs - should be identical
3. **Export for presentations**: Save screenshots or animations
4. **Quantitative analysis**: Measure vortex positions, calculate coefficients

