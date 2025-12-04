# Improved Simulation Configurations

## Problem Identified
The current simulations show minimal change between timesteps because:
1. **Too few output points**: Only 10 VTK files for 500 timesteps
2. **Flow hasn't fully developed**: Lid-driven cavity needs more time
3. **Conservative parameters**: Low Reynolds number = less dramatic effects

## Recommended Configurations

### 1. Extended Cavity (More Timesteps)
**File**: `configs/cavity_extended.yaml`
- 5000 timesteps (vs 500)
- Output every 50 timesteps = 100 VTK files
- Better resolution of flow development

```bash
./build/src/lbm_sim --config configs/cavity_extended.yaml --output-dir output/cavity_extended
```

### 2. High Reynolds Number (More Dramatic)
**File**: `configs/cavity_high_re.yaml`
- Lower relaxation time (0.5) = higher Reynolds number
- Higher lid velocity (0.2) = stronger forcing
- 256x256 grid for better detail
- 10,000 timesteps for full development

```bash
./build/src/lbm_sim --config configs/cavity_high_re.yaml --output-dir output/cavity_high_re
```

### 3. Flow Development Focus
**File**: `configs/cavity_developing.yaml`
- Output every 10 timesteps = 200 files
- Captures rapid initial development
- Perfect for seeing flow evolution

```bash
./build/src/lbm_sim --config configs/cavity_developing.yaml --output-dir output/cavity_developing
```

### 4. Detailed Cylinder Flow
**File**: `configs/cylinder_detailed.yaml`
- Larger cylinder, more dramatic wake
- More frequent output (every 50 steps)
- Better resolution (300x150)

```bash
./build/src/lbm_sim --config configs/cylinder_detailed.yaml --output-dir output/cylinder_detailed
```

## Key Parameters to Adjust

### For More Dramatic Effects:
1. **Lower relaxation time** (τ): 0.5-0.55 (higher Reynolds number)
2. **Higher lid velocity**: 0.15-0.2 (stronger forcing)
3. **More timesteps**: 2000-10000 (full flow development)
4. **Frequent output**: Every 10-50 timesteps (see evolution)

### For Better Visualization:
1. **Higher resolution**: 256x256 or 512x512
2. **More output points**: 50-200 VTK files
3. **Longer simulation**: Let flow fully develop

## Expected Results

### Lid-Driven Cavity:
- **Early timesteps**: Flow starts near lid, spreads downward
- **Mid timesteps**: Recirculation zones form
- **Late timesteps**: Fully developed vortices, stable pattern

### Cylinder Flow:
- **Early**: Flow approaches cylinder
- **Mid**: Separation occurs, wake forms
- **Late**: Fully developed wake with vortices

## Running Multiple Simulations

```bash
# Run all improved simulations
for config in cavity_extended cavity_high_re cavity_developing cylinder_detailed; do
    echo "Running $config..."
    ./build/src/lbm_sim --config configs/${config}.yaml --output-dir output/${config}
done
```

## Visualization Tips

1. **Load time series**: In ParaView, select all VTK files and check "Group files into time series"
2. **Animate**: Use play button to see flow evolution
3. **Add filters**:
   - **Stream Tracer**: See flow paths
   - **Glyphs**: Velocity vectors
   - **Contour**: Isosurfaces of velocity/density

## Performance Notes

- Extended simulations will take longer but show better results
- CUDA backend recommended for large grids/long simulations
- Typical runtime: 5-30 minutes depending on configuration

