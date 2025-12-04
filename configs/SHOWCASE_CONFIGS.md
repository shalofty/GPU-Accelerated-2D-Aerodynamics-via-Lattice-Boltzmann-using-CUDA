# Showcase Configurations for Stunning LIC Visualizations

## Overview

These configurations are optimized to produce the most visually impressive and exciting LIC (Line Integral Convolution) visualizations, showcasing the full capabilities of the LBM simulation.

## Key Parameters for Visual Appeal

### 1. **Reynolds Number (via relaxation_time)**
- **Lower relaxation_time = Higher Re = More Complex Flow**
- `0.5-0.55`: High Re, multiple vortices, complex patterns
- `0.6`: Moderate Re, smooth vortices (current default)
- `0.7+`: Low Re, very smooth, simple patterns

### 2. **Grid Resolution**
- **Higher = More Detail**
- `512×512`: Maximum detail, shows fine structures
- `256×256`: Good balance of detail and performance
- `128×128`: Faster, but less detail

### 3. **Lid Velocity**
- **Higher = Stronger Forcing = More Dramatic**
- `0.2-0.25`: Strong forcing, multiple vortices
- `0.15`: Moderate (current)
- `0.1`: Gentle flow

### 4. **Output Frequency**
- **More frequent = Smoother Animation**
- Every 20-30 timesteps: Smooth, detailed animation
- Every 50 timesteps: Standard
- Every 100+: Choppy animation

## Recommended Configurations

### 1. **Cavity Showcase** (Most Impressive Overall)
**File**: `configs/cavity_showcase.yaml`

```yaml
nx: 512
ny: 512
relaxation_time: 0.55
max_timesteps: 10000
output_interval: 25
lid_velocity: 0.2
```

**Why it's impressive:**
- **512×512 resolution**: Captures fine vortex structures
- **High Re (τ=0.55)**: Multiple vortices, complex patterns
- **Strong forcing (u=0.2)**: Dramatic flow development
- **400 output files**: Smooth, detailed animation
- **10,000 timesteps**: Fully developed, stable flow

**Expected visuals:**
- Multiple recirculation zones
- Fine-scale vortex structures
- High velocity gradients
- Complex streamline patterns
- Secondary vortices in corners

**Runtime**: ~15-20 minutes (CUDA)

### 2. **Cylinder Showcase** (Most Dramatic Wake)
**File**: `configs/cylinder_showcase.yaml`

```yaml
nx: 400
ny: 200
relaxation_time: 0.55
max_timesteps: 8000
output_interval: 20
obstacles:
  - type: "cylinder"
    parameters: [100.0, 100.0, 20.0]
```

**Why it's impressive:**
- **Large cylinder**: More dramatic wake
- **High Re**: Complex wake patterns, vortex shedding
- **400 output files**: Smooth wake development
- **Wake visualization**: Very appealing in LIC

**Expected visuals:**
- Kármán vortex street (at higher Re)
- Complex wake patterns
- Flow separation
- Vortex shedding
- High contrast around cylinder

**Runtime**: ~10-15 minutes (CUDA)

### 3. **Turbulent Cavity** (Maximum Complexity)
**File**: `configs/cavity_turbulent.yaml`

```yaml
nx: 256
ny: 256
relaxation_time: 0.5
max_timesteps: 12000
output_interval: 30
lid_velocity: 0.25
```

**Why it's impressive:**
- **Very high Re (τ=0.5)**: Maximum complexity
- **Strong forcing (u=0.25)**: Very dramatic
- **400 output files**: Smooth animation
- **Long simulation**: Fully developed turbulence

**Expected visuals:**
- Multiple interacting vortices
- Complex, chaotic patterns
- High-frequency structures
- Very dynamic flow
- Maximum visual complexity

**Runtime**: ~8-12 minutes (CUDA)

## ParaView Visualization Tips for Maximum Impact

### 1. **LIC Settings**
- **LIC Type**: Surface LIC (what you're using)
- **Noise Texture**: Increase for finer detail
- **LIC Contrast Enhancement**: Enable for better visibility
- **LIC Integration Steps**: 20-50 for smooth streamlines

### 2. **Color Mapping**
- **Colormap**: Use "Rainbow" or "Cool to Warm" for high contrast
- **Color Scale**: Logarithmic for better detail in low-velocity regions
- **Range**: Auto-adjust for each timestep (dynamic range)

### 3. **Additional Filters**
- **Stream Tracer**: Add for explicit streamlines
- **Glyphs**: Velocity vectors (sparse, for clarity)
- **Contour**: Velocity magnitude isosurfaces
- **Slice**: Cross-sections at different Y positions

### 4. **Animation**
- **Frame Rate**: 30 fps for smooth playback
- **Loop**: Enable for continuous animation
- **Speed**: Adjust to show flow development clearly

## Performance vs Visual Quality

| Configuration | Resolution | Runtime | Visual Quality | Best For |
|--------------|------------|---------|----------------|----------|
| cavity_showcase | 512×512 | ~20 min | ⭐⭐⭐⭐⭐ | Maximum detail |
| cylinder_showcase | 400×200 | ~15 min | ⭐⭐⭐⭐⭐ | Wake patterns |
| cavity_turbulent | 256×256 | ~12 min | ⭐⭐⭐⭐ | Complexity |
| cavity_10sec | 128×128 | ~2.5 min | ⭐⭐⭐ | Quick preview |

## Running the Showcase Simulations

```bash
# Most impressive (high resolution, complex flow)
./build/src/lbm_sim --config configs/cavity_showcase.yaml --output-dir output/showcase_cavity

# Most dramatic (cylinder wake)
./build/src/lbm_sim --config configs/cylinder_showcase.yaml --output-dir output/showcase_cylinder

# Maximum complexity (turbulent)
./build/src/lbm_sim --config configs/cavity_turbulent.yaml --output-dir output/showcase_turbulent
```

## What Makes These Visually Appealing

1. **High Resolution**: Captures fine details, smooth curves
2. **High Reynolds Number**: Multiple vortices, complex interactions
3. **Strong Forcing**: Dramatic flow development, high contrast
4. **Smooth Animation**: Many output points for fluid motion
5. **Complex Patterns**: Multiple scales, interacting structures

## Expected Visual Features

### Cavity Showcase:
- **Primary vortex**: Large, well-defined
- **Secondary vortices**: In corners (at high Re)
- **Fine structures**: Small-scale features
- **High contrast**: Clear velocity gradients
- **Smooth streamlines**: Well-resolved flow paths

### Cylinder Showcase:
- **Wake formation**: Behind cylinder
- **Vortex shedding**: Periodic pattern (at higher Re)
- **Flow separation**: At cylinder sides
- **High velocity gradients**: Around obstacle
- **Complex wake**: Multiple vortices downstream

## Tips for Best Results

1. **Start with cavity_showcase**: Best overall visual quality
2. **Use CUDA backend**: Much faster for high-resolution
3. **Let simulation complete**: Fully developed flow looks best
4. **Adjust ParaView settings**: Fine-tune LIC parameters
5. **Export high-res images**: For presentations/reports

These configurations will produce publication-quality visualizations! 🎨

