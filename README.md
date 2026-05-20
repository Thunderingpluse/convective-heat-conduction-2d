# 2D Steady-State Heat Conduction (Convective Boundaries)

## Aim
To solve 2D steady-state heat conduction on a square plate with convective boundary conditions on one or more faces using Finite Difference Method (FDM) and ghost node substitutions.

## Theory
This solver implements two distinct convective configurations:
1. **Case 1: 1 Convective Wall (Right Side)**: Bottom, Left, and Top walls are maintained at fixed temperatures, while the Right boundary transfers heat to a surrounding fluid via convection.
2. **Case 2: 3 Convective Walls (Bottom, Top, Right)**: The Left boundary is kept at a fixed temperature, while the Bottom, Top, and Right walls are exposed to convection.

### Convective Boundary Discretization (Robin BC)
At a convective boundary (e.g., Right Wall $x = L$):
$$-k \frac{\partial T}{\partial x} = h (T - T_\infty)$$

Using central differences introduces virtual "ghost nodes" outside the grid ($T_{i, j+1}$):
$$\frac{T_{i,j+1} - T_{i,j-1}}{2\Delta x} \approx \frac{\partial T}{\partial x} = -\frac{h}{k}(T_{i,j} - T_\infty)$$

Substituting the ghost node expression into the main 2D conduction equation gives the boundary node equation:
$$-(4 + 2Bi)T_{i,j} + 2T_{i,j-1} + T_{i-1,j} + T_{i+1,j} = -2Bi T_\infty$$
Where $Bi = \frac{h \Delta x}{k}$ is the grid Biot number.

## File Structure
- `convective_combined.py` - Combined script containing solvers for both 1-Side and 3-Side convective boundaries with comparison plots.
- `v0_1side.py` & `v1_3sides.py` - Separate standalone codes for the individual cases.
- `output combined.txt` - Sample output containing calculations of nodal values at the convective boundaries.
- `v0_output_1 side.txt` & `v1_output_3 sides.txt` - Standalone outputs for case validation.
- `Graph convective combined.png`, `Graph convective 1 side.png`, `Graph convective 3 sides.png` - Contour temperature profiles.

## How to Run
Ensure you have the required dependencies:
```bash
pip install numpy matplotlib
python convective_combined.py
```
