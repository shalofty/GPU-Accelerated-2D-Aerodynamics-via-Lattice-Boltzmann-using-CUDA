# Quick Visualization Guide

## Method 1: ParaView (Easiest - Recommended)

### Step 1: Launch ParaView
```bash
paraview
```

### Step 2: Open Your Files

**For Cavity Flow:**
1. File → Open
2. Navigate to: `output/cavity_cpu/`
3. Select **all** VTK files (or just one to start)
4. **Check "Group files into time series"** (for animation)
5. Click "OK"
6. Click "Apply" in the Properties panel

**For Cylinder Flow:**
- Same steps, but use: `output/cylinder_cpu/`

### Step 3: Visualize

1. **Color by velocity:**
   - In the "Coloring" dropdown, select "velocity_magnitude"
   - Adjust color scale if needed

2. **Add velocity vectors:**
   - Filters → Glyphs
   - Glyph Type: "Arrow"
   - Orientation: "velocity"
   - Scale: "velocity_magnitude"
   - Click "Apply"

3. **Animate:**
   - Click the ▶️ play button at the top
   - Watch the flow evolve over time!

### Quick Commands

```bash
# Open ParaView
paraview

# Or open directly with a file
paraview output/cavity_cpu/field_000050.vtk
```

## Method 2: Python Script (If ParaView Not Available)

### Install Dependencies
```bash
pip install vtk numpy matplotlib
```

### Run Visualization
```bash
# View single file
python3 scripts/visualize.py output/cavity_cpu/field_000050.vtk

# Generate images for all files in directory
python3 scripts/visualize.py output/cavity_cpu/ velocity_magnitude
```

This will create PNG images in `output/cavity_cpu/plots/`

## Available Fields

Each VTK file contains:
- **density**: Fluid density field
- **velocity_x**: X-component of velocity
- **velocity_y**: Y-component of velocity  
- **velocity_magnitude**: Speed (√(vx² + vy²))

## File Locations

- **Cavity Flow**: `output/cavity_cpu/field_*.vtk` (10 files)
- **Cylinder Flow**: `output/cylinder_cpu/field_*.vtk` (10 files)

## Tips

1. **Start with one file** to test, then load the time series
2. **Use "velocity_magnitude"** for color coding - shows flow speed
3. **Add streamlines** (Filters → Stream Tracer) to see flow paths
4. **Export images** (File → Save Screenshot) for reports

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

## Next Steps

1. **Explore different visualizations:**
   - Streamlines for flow paths
   - Contours for pressure/density
   - Slices for cross-sections

2. **Compare CPU vs CUDA:**
   - Load both `output/cavity_cpu/` and `output/cavity_cuda/`
   - Results should be identical!

3. **Export for presentations:**
   - File → Save Screenshot (PNG/JPEG)
   - File → Save Animation (for videos)

Enjoy visualizing your LBM simulations!

