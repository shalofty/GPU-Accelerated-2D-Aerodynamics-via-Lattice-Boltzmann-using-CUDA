# Available Simulation Scenarios

## Current Implementation Capabilities

Your LBM solver is a **single-phase, isothermal (constant temperature) incompressible flow solver**. This means it simulates fluid flow without:
- Phase changes (liquid ↔ vapor)
- Temperature variations
- Multiple fluid phases
- Chemical reactions

## What You CAN Simulate (With Current Code)

### 1. **Lid-Driven Cavity Flow** ✅
**What it is**: Square box with moving top wall

**Config**:
```yaml
nx: 128
ny: 128
relaxation_time: 0.6
lid_velocity: 0.1
# No obstacles
```

**Visual**: Recirculation vortices, classic benchmark case

---

### 2. **Flow Around Cylinder** ✅
**What it is**: Flow past a circular obstacle

**Config**:
```yaml
nx: 200
ny: 100
obstacles:
  - type: "cylinder"
    parameters: [50.0, 50.0, 10.0]  # [cx, cy, radius]
```

**Visual**: Wake formation, vortex shedding, flow separation

**Variations**:
- Multiple cylinders
- Different sizes
- Different positions

---

### 3. **Channel Flow** ✅ (Inflow/Outflow)
**What it is**: Flow through a channel with inlet/outlet

**Config**:
```yaml
nx: 200
ny: 100
obstacles:
  - type: "cylinder"  # Creates channel flow around obstacle
    parameters: [50.0, 50.0, 10.0]
# Automatically uses inflow/outflow BCs
```

**Visual**: Flow through channel, boundary layers, wake patterns

---

### 4. **Flow Around Multiple Obstacles** ✅
**What it is**: Multiple cylinders in flow

**Config**:
```yaml
nx: 300
ny: 150
obstacles:
  - type: "cylinder"
    parameters: [75.0, 50.0, 15.0]
  - type: "cylinder"
    parameters: [150.0, 100.0, 10.0]
  - type: "cylinder"
    parameters: [225.0, 75.0, 12.0]
```

**Visual**: Complex wake interactions, flow interference

---

## What You CANNOT Simulate (Requires Extensions)

### ❌ Boiling
**Why not**: Requires:
- Multi-phase LBM (liquid + vapor)
- Energy equation (temperature)
- Phase change model (evaporation/condensation)
- Surface tension

**What's needed**: 
- Add temperature field
- Implement multi-phase model (Shan-Chen, color-gradient, etc.)
- Add phase change physics

---

### ❌ Droplet Dynamics
**Why not**: Requires:
- Multi-phase LBM (liquid + gas)
- Surface tension model
- Contact angle (wetting)
- Interface tracking

**What's needed**:
- Multi-phase LBM implementation
- Surface tension forces
- Interface tracking/capturing

---

### ❌ Temperature-Driven Flow
**Why not**: Requires:
- Energy equation
- Temperature field
- Buoyancy (Boussinesq approximation)

**What's needed**:
- Add energy equation to LBM
- Temperature-dependent density
- Buoyancy force

---

## Other Scenarios Possible With Extensions

### Easy to Add (Minor Code Changes)

#### 1. **Different Obstacle Shapes**
Currently only cylinder is implemented. Could add:
- **Rectangle/Box obstacles**
- **Ellipse obstacles**
- **Custom shapes** (from image/coordinates)

**Effort**: Medium (need to add shape detection)

#### 2. **Backward-Facing Step**
Flow over a step - classic benchmark

**Effort**: Low (just geometry setup)

#### 3. **Flow in Porous Media**
Multiple small obstacles

**Effort**: Low (just add many small cylinders)

#### 4. **Periodic Boundary Conditions**
Flow in periodic domain

**Effort**: Medium (need to implement periodic BCs)

---

### Moderate Extensions

#### 5. **Moving Obstacles**
Cylinder that moves/rotates

**Effort**: Medium (update obstacle mask each timestep)

#### 6. **Time-Varying Boundary Conditions**
Lid velocity that changes with time

**Effort**: Low (modify boundary condition application)

#### 7. **Pressure-Driven Flow**
Pressure gradient instead of velocity

**Effort**: Medium (different boundary condition)

---

### Major Extensions (Significant Work)

#### 8. **Multi-Phase Flow** (Droplets, Bubbles)
- Shan-Chen model
- Color-gradient model
- Free energy model

**Effort**: High (new physics, new kernels)

#### 9. **Thermal LBM** (Boiling, Convection)
- Energy equation
- Temperature field
- Buoyancy

**Effort**: High (new field, new equations)

#### 10. **Reactive Flow**
- Chemical reactions
- Species transport

**Effort**: Very High (multiple species, reactions)

---

## Why Your Visualization Looks Similar

Even with different parameters, you're still seeing **lid-driven cavity flow** because:

1. **Same geometry**: Square box with moving top
2. **Same physics**: Single-phase, isothermal flow
3. **Parameter changes affect intensity, not type**:
   - Higher Re → More vortices (but still cavity flow)
   - Higher velocity → Stronger flow (but still cavity flow)
   - More resolution → More detail (but same pattern)

**To see different patterns**, you need:
- Different geometries (obstacles)
- Different boundary conditions
- Different flow scenarios

---

## Recommended Next Steps

### Option 1: Explore Current Capabilities

**Try cylinder flow** - Very different visual:
```yaml
nx: 400
ny: 200
relaxation_time: 0.55
max_timesteps: 5000
output_interval: 25
obstacles:
  - type: "cylinder"
    parameters: [100.0, 100.0, 25.0]
backend_id: "cuda"
```

**Visual**: Dramatic wake, vortex shedding, very different from cavity!

### Option 2: Add New Obstacle Shapes

I can help you add:
- Rectangle obstacles
- Multiple obstacles
- Custom geometries

### Option 3: Extend to Multi-Phase (For Droplets/Boiling)

This would require:
- Implementing multi-phase LBM model
- Adding surface tension
- Adding phase change (for boiling)

**Effort**: Significant, but doable

---

## Quick Comparison

| Scenario | Current Code | Visual Appeal | Complexity |
|----------|--------------|---------------|------------|
| Lid-driven cavity | ✅ | ⭐⭐⭐ | Low |
| Cylinder flow | ✅ | ⭐⭐⭐⭐⭐ | Low |
| Multiple obstacles | ✅ | ⭐⭐⭐⭐ | Low |
| Channel flow | ✅ | ⭐⭐⭐ | Low |
| Droplets | ❌ | ⭐⭐⭐⭐⭐ | High |
| Boiling | ❌ | ⭐⭐⭐⭐⭐ | Very High |
| Thermal convection | ❌ | ⭐⭐⭐⭐ | High |

---

## My Recommendation

**For immediate visual variety**: Try **cylinder flow** - it produces dramatically different, more exciting visuals than cavity flow!

Would you like me to:
1. Create a cylinder flow showcase config?
2. Add support for rectangle obstacles?
3. Start implementing multi-phase LBM for droplets?

