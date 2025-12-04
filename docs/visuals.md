# Visualization Guide

## Quick Start

The simulation outputs VTK (Visualization Toolkit) files that can be viewed in ParaView or other visualization tools.

## Option 1: ParaView (Recommended)

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
   - Click "OK"

3. **Load Time Series (Animation)**
   - Select all VTK files in a directory
   - Check "Group files into time series"
   - ParaView will automatically create an animation

### Recommended Visualizations

#### 1. Velocity Magnitude (Color-coded)
- After loading, click "Apply"
- In the "Coloring" dropdown, select "velocity_magnitude"
- Adjust color scale as needed

#### 2. Velocity Vectors (Glyphs)
- Filters → Glyphs
- Glyph Type: "Arrow"
- Orientation: "velocity"
- Scale: "velocity_magnitude"
- Click "Apply"

#### 3. Streamlines
- Filters → Stream Tracer
- Seed Type: "Line" or "Point Cloud"
- Click "Apply"
- Adjust seed points to see flow paths

#### 4. Contour Lines
- Filters → Contour
- Contour By: "velocity_magnitude" or "density"
- Isosurfaces: Add values (e.g., 0.05, 0.1, 0.15)
- Click "Apply"

#### 5. Slice View
- Filters → Slice
- Plane: "XY Plane" (for 2D)
- Click "Apply"

### Animation Controls

- Use the play/pause buttons at the top to animate through timesteps
- Adjust playback speed with the speed slider
- Step through frames manually with arrow buttons

## Option 2: Python with VTK/Matplotlib

### Installation
```bash
pip install vtk numpy matplotlib
```

### Quick Visualization Script

Create `visualize.py`:

```python
#!/usr/bin/env python3
import vtk
import sys
import os

def visualize_vtk(filename):
    # Read VTK file
    reader = vtk.vtkXMLRectilinearGridReader()
    reader.SetFileName(filename)
    reader.Update()
    
    # Create mapper
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputConnection(reader.GetOutputPort())
    
    # Create actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    
    # Create renderer
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.1, 0.1)
    
    # Create render window
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 600)
    render_window.SetWindowName("LBM Visualization")
    
    # Create interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.Initialize()
    interactor.Start()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize.py <vtk_file>")
        sys.exit(1)
    visualize_vtk(sys.argv[1])
```

Run:
```bash
python3 visualize.py output/cavity_cpu/field_000050.vtk
```

## Option 3: Python with Matplotlib (2D Plot)

For simple 2D field visualization:

```python
#!/usr/bin/env python3
import vtk
import numpy as np
import matplotlib.pyplot as plt

def plot_field(vtk_file, field_name='velocity_magnitude'):
    reader = vtk.vtkXMLRectilinearGridReader()
    reader.SetFileName(vtk_file)
    reader.Update()
    
    data = reader.GetOutput()
    field = data.GetPointData().GetArray(field_name)
    
    # Get grid dimensions
    dims = data.GetDimensions()
    nx, ny = dims[0], dims[1]
    
    # Convert to numpy array
    values = np.array([field.GetValue(i) for i in range(field.GetNumberOfValues())])
    values = values.reshape(ny, nx)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(values, origin='lower', cmap='viridis')
    plt.colorbar(label=field_name)
    plt.title(f'{field_name} at timestep {vtk_file.split("_")[-1].split(".")[0]}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout()
    plt.savefig(f'{field_name}_plot.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    import sys
    plot_field(sys.argv[1] if len(sys.argv) > 1 else 'output/cavity_cpu/field_000050.vtk')
```

## Option 4: Online VTK Viewer

If you can't install ParaView locally:

1. Upload VTK files to a cloud service
2. Use online VTK viewers (limited functionality)
3. Convert to more common formats (see below)

## Converting to Other Formats

### Convert to PNG/JPEG (for reports)

Using ParaView:
1. Load VTK file
2. Apply visualization
3. File → Save Screenshot
4. Choose format (PNG, JPEG, etc.)

### Convert to CSV (for data analysis)

```python
import vtk
import numpy as np
import pandas as pd

reader = vtk.vtkXMLRectilinearGridReader()
reader.SetFileName('output/cavity_cpu/field_000050.vtk')
reader.Update()

data = reader.GetOutput()
# Extract arrays and save to CSV
# (implementation depends on what data you need)
```

## Quick Command-Line Viewing

### List available fields
```bash
# VTK files contain: density, velocity_x, velocity_y, velocity_magnitude
```

### Check file info
```bash
head -20 output/cavity_cpu/field_000050.vtk
```

## Recommended Workflow

1. **Start with ParaView** - Best for interactive exploration
2. **Load time series** - See flow evolution over time
3. **Create multiple views** - Velocity, streamlines, contours
4. **Export images** - For reports/presentations
5. **Use Python scripts** - For batch processing or custom analysis

## Example: Viewing Cavity Flow

```bash
# 1. Open ParaView
paraview

# 2. In ParaView:
#    - File → Open → output/cavity_cpu/field_*.vtk
#    - Check "Group files into time series"
#    - Click OK
#    - Click "Apply"
#    - Coloring: "velocity_magnitude"
#    - Click play button to animate
```

## Example: Viewing Cylinder Flow

```bash
# Same process, but use:
# output/cylinder_cpu/field_*.vtk

# Recommended filters:
# - Stream Tracer (to see flow around cylinder)
# - Contour (to see pressure/density)
# - Glyphs (to see velocity vectors)
```

## Troubleshooting

### ParaView won't open files
- Check file path is correct
- Verify VTK files are not corrupted
- Try opening a single file first

### No data visible
- Check "Coloring" dropdown has a field selected
- Try different fields (velocity_magnitude, density, etc.)
- Adjust color scale range

### Animation not working
- Make sure "Group files into time series" was checked
- Check that files are named sequentially
- Verify all files are in the same directory

## File Locations

- **Cavity Flow (CPU)**: `output/cavity_cpu/field_*.vtk`
- **Cavity Flow (CUDA)**: `output/cavity_cuda/field_*.vtk`
- **Cylinder Flow (CPU)**: `output/cylinder_cpu/field_*.vtk`
- **Cylinder Flow (CUDA)**: `output/cylinder_cuda/field_*.vtk`

Each directory contains 10 VTK files at different timesteps.

