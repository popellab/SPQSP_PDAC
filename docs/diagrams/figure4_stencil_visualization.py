#!/usr/bin/env python3
"""
Figure 4b: 7-Point Stencil Visualization for PDE Solver
Creates a visual representation of the 7-point stencil used in the Laplacian operator.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ========== LEFT PLOT: 2D Stencil (X-Y plane) ==========
ax = axes[0]
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.set_title('7-Point Stencil (Cross-Section: X-Y Plane)', fontsize=14, fontweight='bold')
ax.set_xlabel('X direction', fontsize=11)
ax.set_ylabel('Y direction', fontsize=11)

# Draw grid
for i in range(-2, 3):
    ax.axvline(i, color='lightgray', linewidth=0.5, linestyle='--')
    ax.axhline(i, color='lightgray', linewidth=0.5, linestyle='--')

# Draw voxels
grid_positions = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
colors = ['#FF6B6B', '#FF6B6B', '#FF6B6B', '#FF6B6B', '#4ECDC4']
sizes = [600, 600, 600, 600, 1200]
labels = ['C[-1,0]', 'C[+1,0]', 'C[0,-1]', 'C[0,+1]', 'C[0,0]\n(center)']

for (x, y), color, size, label in zip(grid_positions, colors, sizes, labels):
    ax.scatter(x, y, s=size, c=color, edgecolors='black', linewidth=2.5, zorder=5)
    offset_x = 0.35 if x != 0 else 0
    offset_y = 0.4 if y != 0 else -0.5
    ax.text(x + offset_x, y + offset_y, label, fontsize=10, ha='center',
            fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Draw connections
for x, y in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
    ax.plot([0, x], [0, y], 'k--', linewidth=1.5, alpha=0.5)

ax.set_xticks(range(-2, 3))
ax.set_yticks(range(-2, 3))
ax.grid(True, alpha=0.3)

# Add text annotation
textstr = 'Stencil weights (in X-Y plane):\nNeighbor: -1\nCenter: +6\n\n(Same pattern in Z direction)'
ax.text(0, -2.7, textstr, fontsize=10, ha='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))

# ========== RIGHT PLOT: 3D Stencil visualization ==========
ax = axes[1]
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.set_title('3D 7-Point Stencil (Coordinate System)', fontsize=14, fontweight='bold')

# Draw center voxel (large)
center = patches.Circle((0, 0), 0.35, color='#4ECDC4', edgecolor='black', linewidth=2.5)
ax.add_patch(center)
ax.text(0, -0.5, 'C[i,j,k]\n(center)', fontsize=10, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Draw neighbors with labels
neighbors = [
    (-1, 0, 'X-'),
    (1, 0, 'X+'),
    (0, -1, 'Y-'),
    (0, 1, 'Y+'),
]

for x, y, label in neighbors:
    circle = patches.Circle((x, y), 0.25, color='#FF6B6B', edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    ax.plot([0, x], [0, y], 'k--', linewidth=1.5, alpha=0.5)

# Add Z direction indication
ax.text(0, 1.3, 'Z+ & Z- neighbors\n(perpendicular to page)', fontsize=9, ha='center',
        style='italic', bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

# Equation annotation
eq_text = r'$\nabla^2 C_{i,j,k} = \frac{C_{i-1,j,k} + C_{i+1,j,k} + C_{i,j-1,k} + C_{i,j+1,k} + C_{i,j,k-1} + C_{i,j,k+1} - 6 \cdot C_{i,j,k}}{(\Delta x)^2}$'
ax.text(0, -1.35, eq_text, fontsize=11, ha='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9))

ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

plt.tight_layout()
plt.savefig('/home/chase/SPQSP/SPQSP_PDAC-main/docs/diagrams/figure4_stencil_visualization.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: figure4_stencil_visualization.png")
plt.close()

# ========== Create a second figure: Solver workflow ==========
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Title
ax.text(5, 11.5, 'PDE Implicit Solver: Algorithm Flow', fontsize=16, fontweight='bold', ha='center')

# Steps
steps = [
    (5, 10.5, 'Input: C^n (current concentrations)', '#E8F4F8'),
    (5, 9.5, 'Compute RHS: b = C^n + dt·S', '#E8F4F8'),
    (5, 8.5, 'Initialize: x₀ = C^n, r = b - A·x₀', '#FFE5E5'),
    (5, 7.5, 'CG Loop: i = 0 to max_iter', '#F0E5FF'),
    (3.5, 6.5, 'Compute: x_new = A·x (matvec)', '#F0E5FF'),
    (6.5, 6.5, 'β = (r^T·r) / (p^T·A·p)', '#F0E5FF'),
    (3.5, 5.5, 'Update: x += β·p', '#F0E5FF'),
    (6.5, 5.5, 'Update: r -= β·A·p', '#F0E5FF'),
    (5, 4.5, 'Check convergence: ||r|| < ε?', '#FFE5CC'),
    (3, 3.3, 'No: continue loop', '#FFFFCC'),
    (7, 3.3, 'Yes: exit loop', '#90EE90'),
    (5, 2.2, 'Output: C^(n+1) = x (solved concentrations)', '#E8F8E8'),
    (5, 1.0, 'Properties: Unconditionally stable, 1 solve/timestep', '#C8E6C9'),
]

for x, y, text, color in steps:
    if y > 8:
        width, height = 4, 0.6
    elif y > 4.5:
        width, height = 3.2, 0.6
    elif y > 3:
        width, height = 2.2, 0.6
    else:
        width, height = 4.5, 0.6

    rect = patches.FancyBboxPatch((x - width/2, y - height/2), width, height,
                                   boxstyle="round,pad=0.1",
                                   facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x, y, text, fontsize=9.5, ha='center', va='center', fontweight='bold')

# Arrows
arrow_pairs = [
    ((5, 10.2), (5, 9.8)),
    ((5, 9.2), (5, 8.8)),
    ((5, 8.2), (5, 7.8)),
    ((5, 7.2), (4, 6.8)),
    ((5, 7.2), (6, 6.8)),
    ((3.5, 6.2), (3.5, 5.8)),
    ((6.5, 6.2), (6.5, 5.8)),
    ((3.5, 5.2), (4.2, 4.8)),
    ((6.5, 5.2), (5.8, 4.8)),
    ((5, 4.2), (3.8, 3.6)),
    ((5, 4.2), (6.2, 3.6)),
    ((3, 3.0), (4.2, 2.5)),  # loop back
    ((7, 3.0), (5, 2.5)),
    ((5, 1.9), (5, 1.3)),
]

for start, end in arrow_pairs:
    if start[1] > end[1]:  # downward
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    else:  # loopback
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='red',
                                 connectionstyle="arc3,rad=0.5"))

plt.tight_layout()
plt.savefig('/home/chase/SPQSP/SPQSP_PDAC-main/docs/diagrams/figure4_solver_workflow.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: figure4_solver_workflow.png")

print("\nDone! Generated:")
print("  • figure4_stencil_visualization.png")
print("  • figure4_solver_workflow.png")
