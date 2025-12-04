# Configuration Parameters Guide

## How Each Parameter Affects Your Visualization

This guide explains how to manipulate configuration values to achieve different visual effects in your LBM simulations.

---

## Core Parameters

### 1. **Grid Resolution** (`nx`, `ny`)

**What it controls**: Level of detail in the visualization

```yaml
nx: 64   # Width (X direction)
ny: 64   # Height (Y direction)
```

**Effects on Visualization**:
- **Low (64×64)**: 
  - Fast computation
  - Coarse details, pixelated streamlines
  - Good for quick previews
  - Less smooth curves

- **Medium (128×128, 256×256)**:
  - Good balance of detail and speed
  - Smooth streamlines
  - Captures main flow features
  - **Recommended for most cases**

- **High (512×512, 1024×1024)**:
  - Maximum detail, publication quality
  - Very smooth, fine structures visible
  - Shows small-scale vortices
  - Slower computation

**Visual Impact**: Higher = smoother, more detailed streamlines

**Example**:
```yaml
nx: 128   # Smooth curves
ny: 128   # Good detail
```

---

### 2. **Relaxation Time** (`relaxation_time`)

**What it controls**: Reynolds number → Flow complexity

```yaml
relaxation_time: 0.6
```

**Physical Meaning**:
- Controls viscosity in LBM
- **Lower value = Higher Reynolds number = More complex flow**
- **Higher value = Lower Reynolds number = Smoother flow**

**Effects on Visualization**:

| Value | Reynolds Number | Visual Effect |
|-------|----------------|---------------|
| 0.5 | Very High | Multiple vortices, complex patterns, turbulent-like |
| 0.55 | High | Several vortices, rich detail |
| 0.6 | Moderate | 1-2 main vortices, smooth patterns |
| 0.7+ | Low | Single vortex, very smooth, simple |

**Visual Impact**:
- **Low (0.5-0.55)**: 
  - Multiple interacting vortices
  - Complex, chaotic patterns
  - High visual interest
  - More "exciting" to watch
  
- **Medium (0.6)**:
  - Well-defined vortices
  - Smooth, predictable patterns
  - Classic lid-driven cavity look
  
- **High (0.7-1.0)**:
  - Single large vortex
  - Very smooth, simple
  - Less visually interesting

**Example for Complex Flow**:
```yaml
relaxation_time: 0.55  # High Re, multiple vortices
```

**Example for Smooth Flow**:
```yaml
relaxation_time: 0.7   # Low Re, simple pattern
```

**⚠️ Warning**: Values below 0.5 can cause numerical instability!

---

### 3. **Lid Velocity** (`lid_velocity`)

**What it controls**: Strength of forcing → Flow intensity

```yaml
lid_velocity: 0.1
```

**Effects on Visualization**:

| Value | Visual Effect |
|-------|---------------|
| 0.05 | Gentle flow, subtle patterns |
| 0.1 | Moderate flow, standard cavity |
| 0.15-0.2 | Strong flow, dramatic patterns |
| 0.25+ | Very strong, multiple vortices |

**Visual Impact**:
- **Low (0.05-0.1)**:
  - Gentle, subtle flow
  - Lower velocity gradients
  - Less color contrast
  - Softer patterns

- **Medium (0.1-0.15)**:
  - Balanced flow
  - Good velocity range
  - Nice color contrast
  - **Recommended default**

- **High (0.2-0.3)**:
  - Strong forcing
  - High velocity gradients
  - Bright colors (red/orange)
  - More dramatic, exciting
  - Can create multiple vortices

**Example for Dramatic Effect**:
```yaml
lid_velocity: 0.2  # Strong forcing, high contrast
```

**⚠️ Warning**: Values > 0.3 can cause numerical instability in LBM!

---

### 4. **Max Timesteps** (`max_timesteps`)

**What it controls**: How long the simulation runs

```yaml
max_timesteps: 500
```

**Effects on Visualization**:
- **Low (100-500)**:
  - Flow still developing
  - Patterns not fully formed
  - Shows initial development
  
- **Medium (500-2000)**:
  - Flow mostly developed
  - Main patterns visible
  - Good for most cases

- **High (2000-10000)**:
  - Fully developed flow
  - Stable, mature patterns
  - Best for final visualizations

**Visual Impact**:
- More timesteps = More developed, stable flow patterns
- Early timesteps show flow development
- Late timesteps show final, stable state

**Example**:
```yaml
max_timesteps: 2000  # Fully developed flow
```

---

### 5. **Output Interval** (`output_interval`)

**What it controls**: How many VTK files are generated

```yaml
output_interval: 50
```

**Calculation**: Number of VTK files = `max_timesteps / output_interval`

**Effects on Visualization**:

| Interval | Files (500 steps) | Animation Quality |
|----------|-------------------|-------------------|
| 10 | 50 files | Very smooth, detailed |
| 25 | 20 files | Smooth |
| 50 | 10 files | Standard |
| 100 | 5 files | Choppy |

**Visual Impact**:
- **Low interval (10-25)**: 
  - Many VTK files
  - Smooth animation
  - See gradual changes
  - **Best for presentations**

- **Medium (50)**:
  - Moderate number of files
  - Reasonable smoothness
  - Good balance

- **High (100+)**: 
  - Few files
  - Choppy animation
  - Large jumps between frames

**Example for Smooth Animation**:
```yaml
max_timesteps: 3000
output_interval: 10  # 300 files, very smooth
```

**Example for Quick Preview**:
```yaml
max_timesteps: 500
output_interval: 100  # 5 files, quick check
```

---

### 6. **Residual Tolerance** (`residual_tolerance`)

**What it controls**: When simulation stops (convergence)

```yaml
residual_tolerance: 1e-4
```

**Effects on Visualization**:
- **Strict (1e-6)**: 
  - Simulation runs longer
  - More accurate, stable
  - Fully converged
  
- **Moderate (1e-4)**:
  - Standard stopping criterion
  - Good balance
  - **Recommended**

- **Loose (1e-3)**:
  - Stops earlier
  - May not be fully developed
  - Faster but less accurate

**Visual Impact**: Usually doesn't affect visuals much, but stricter = more developed flow

---

## Parameter Combinations for Different Effects

### For Maximum Visual Appeal (LIC Showcase)

```yaml
nx: 512
ny: 512
relaxation_time: 0.55      # High Re, complex patterns
max_timesteps: 10000       # Fully developed
output_interval: 25        # 400 files, smooth
lid_velocity: 0.2          # Strong forcing
residual_tolerance: 1e-6
backend_id: "cuda"
```

**Result**: Multiple vortices, fine details, high contrast, smooth animation

---

### For Smooth, Elegant Flow

```yaml
nx: 256
ny: 256
relaxation_time: 0.7       # Low Re, smooth
max_timesteps: 5000
output_interval: 20        # 250 files
lid_velocity: 0.1          # Gentle
residual_tolerance: 1e-6
backend_id: "cuda"
```

**Result**: Single large vortex, smooth streamlines, elegant patterns

---

### For Complex, Turbulent-Like Patterns

```yaml
nx: 256
ny: 256
relaxation_time: 0.5       # Very high Re
max_timesteps: 8000
output_interval: 20        # 400 files
lid_velocity: 0.25         # Strong forcing
residual_tolerance: 1e-6
backend_id: "cuda"
```

**Result**: Multiple interacting vortices, complex patterns, high visual interest

---

### For Quick Preview

```yaml
nx: 128
ny: 128
relaxation_time: 0.6
max_timesteps: 1000
output_interval: 50        # 20 files
lid_velocity: 0.1
residual_tolerance: 1e-4
backend_id: "cuda"
```

**Result**: Fast computation, quick visualization check

---

## Understanding the Relationships

### Reynolds Number Effect

**Reynolds Number ≈ (1/relaxation_time - 0.5) × velocity × length**

- **Low Re (τ=0.7)**: Viscous dominates → Smooth, simple
- **Medium Re (τ=0.6)**: Balanced → Classic patterns
- **High Re (τ=0.5)**: Inertial dominates → Complex, multiple vortices

### Velocity Scaling

- **Lid velocity** directly affects maximum velocity in domain
- Higher velocity = More color range (blue to red)
- Higher velocity = Stronger vortices

### Resolution vs Performance

- **512×512**: ~16x slower than 128×128, but 4x more detail
- **256×256**: Good compromise (4x slower, 2x more detail)

---

## Quick Reference: What to Change For...

### More Vortices
```yaml
relaxation_time: 0.55  # Lower = higher Re
lid_velocity: 0.2      # Higher forcing
```

### Smoother Patterns
```yaml
relaxation_time: 0.7   # Higher = lower Re
lid_velocity: 0.1      # Lower forcing
```

### More Detail
```yaml
nx: 512
ny: 512
```

### Smoother Animation
```yaml
output_interval: 10    # More frequent output
max_timesteps: 3000    # Longer simulation
```

### Higher Contrast (More Color Range)
```yaml
lid_velocity: 0.2      # Higher velocity
relaxation_time: 0.55  # Higher Re
```

### Faster Computation
```yaml
nx: 128
ny: 128
max_timesteps: 500
```

---

## Safety Limits

**⚠️ Avoid These Values** (causes instability):

- `relaxation_time < 0.5`: Numerical instability
- `relaxation_time > 1.0`: Very slow convergence
- `lid_velocity > 0.3`: Mach number too high for incompressible LBM
- `lid_velocity < 0.01`: Too slow, no visible flow

**✅ Safe Ranges**:

- `relaxation_time`: 0.5 - 1.0
- `lid_velocity`: 0.05 - 0.25
- `nx, ny`: 64 - 1024 (higher = slower)

---

## Experimentation Tips

1. **Start with defaults**, then adjust one parameter at a time
2. **For LIC visuals**: Focus on `relaxation_time` and `lid_velocity`
3. **For smoothness**: Increase `nx/ny` and decrease `output_interval`
4. **For complexity**: Lower `relaxation_time`, raise `lid_velocity`
5. **Test small changes first** before going to extremes

---

## Example: Creating Your Own Showcase Config

```yaml
# My Custom Showcase
nx: 256              # Good detail, reasonable speed
ny: 256
relaxation_time: 0.55  # High Re for complexity
max_timesteps: 5000    # Fully developed
output_interval: 12    # ~400 files, very smooth
lid_velocity: 0.18     # Strong but stable
residual_tolerance: 1e-6
backend_id: "cuda"
```

This gives you: Complex flow, smooth animation, good detail, reasonable runtime!

