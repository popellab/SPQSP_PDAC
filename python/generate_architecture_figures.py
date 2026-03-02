#!/usr/bin/env python3
"""
Generate architecture figures for SPQSP PDAC simulator slides.
Creates 6 professional diagrams as PNG files.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
import os

# Ensure output directory exists
os.makedirs('outputs/architecture', exist_ok=True)

# Color scheme
COLOR_GPU = '#FF6B6B'      # Red
COLOR_CPU = '#4ECDC4'      # Teal
COLOR_PDE = '#95E1D3'      # Light teal
COLOR_ABM = '#FFE66D'      # Yellow
COLOR_QSP = '#A8E6CF'      # Light green
COLOR_MEMORY = '#FF8B94'   # Pink
COLOR_DATA = '#FFB8C6'     # Light pink

def fig1_system_architecture():
    """System architecture: GPU vs CPU split"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.5, 'SPQSP PDAC: System Architecture',
            fontsize=20, fontweight='bold', ha='center')

    # GPU Section
    gpu_box = FancyBboxPatch((0.5, 4), 4.5, 5, boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor=COLOR_GPU, alpha=0.3, linewidth=2)
    ax.add_patch(gpu_box)
    ax.text(2.75, 8.5, 'GPU (FLAME GPU 2)', fontsize=14, fontweight='bold', ha='center')

    # GPU components
    y_pos = 7.8
    gpu_components = [
        ('Agent-Based Model', '7 agent types (cancer, T cells, MAC, FIB, VASC, TREG, MDSC)'),
        ('PDE Solver', '10 chemicals: implicit CG + operator splitting (36 substeps)'),
        ('Occupancy Grid', 'CAS-based exclusive voxel access (cancer, MAC)'),
        ('ECM Density Field', 'Gaussian fibroblast density → movement restriction'),
        ('Memory', '~8GB for 320³ voxel grid + agent data'),
    ]

    for title, desc in gpu_components:
        ax.text(0.8, y_pos, f'• {title}', fontsize=11, fontweight='bold')
        ax.text(1.0, y_pos-0.35, desc, fontsize=9, style='italic', wrap=True)
        y_pos -= 0.8

    # CPU Section
    cpu_box = FancyBboxPatch((5.5, 4), 4, 5, boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor=COLOR_CPU, alpha=0.3, linewidth=2)
    ax.add_patch(cpu_box)
    ax.text(7.5, 8.5, 'CPU (SUNDIALS CVODE)', fontsize=14, fontweight='bold', ha='center')

    # CPU components
    y_pos = 7.8
    cpu_components = [
        ('QSP Model', 'LymphCentral: 153 species ODE system'),
        ('Recruitment Logic', 'Host functions for immune cell marking/creation'),
        ('Event Bookkeeping', 'Cancer deaths, immune recruitment counts'),
        ('Parameter System', 'XML-based, 164+ parameters, type-safe enums'),
    ]

    for title, desc in cpu_components:
        ax.text(5.8, y_pos, f'• {title}', fontsize=11, fontweight='bold')
        ax.text(6.0, y_pos-0.35, desc, fontsize=9, style='italic')
        y_pos -= 0.9

    # Integration arrows
    arrow1 = FancyArrowPatch((4.5, 6.5), (5.5, 6.5),
                             arrowstyle='<->', mutation_scale=25, lw=2, color='black')
    ax.add_patch(arrow1)
    ax.text(5, 6.8, 'Synchronized\n(dt=600s)', fontsize=9, ha='center', fontweight='bold')

    # Data flow details
    ax.text(0.5, 3.5, 'GPU→CPU Data:', fontsize=11, fontweight='bold')
    ax.text(0.7, 3.1, '• Agent counts (cancer, T cells, etc.)', fontsize=9)
    ax.text(0.7, 2.7, '• Cancer death events', fontsize=9)
    ax.text(0.7, 2.3, '• Immune recruitment requests', fontsize=9)

    ax.text(5.5, 3.5, 'CPU→GPU Data:', fontsize=11, fontweight='bold')
    ax.text(5.7, 3.1, '• Initial populations (presim option)', fontsize=9)
    ax.text(5.7, 2.7, '• Drug concentrations (planned)', fontsize=9)
    ax.text(5.7, 2.3, '• Immune state from QSP (planned)', fontsize=9)

    # Bottom: Timescale
    ax.text(5, 1.5, 'Timescale: ABM step = 600s (10 min) = 1 QSP step | PDE substep = 16.67s (36 per ABM)',
            fontsize=10, ha='center', style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('outputs/architecture/01_system_architecture.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: 01_system_architecture.png")
    plt.close()

def fig2_simulation_loop():
    """Simulation loop: layer execution order"""
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')

    # Title
    ax.text(5, 13.5, 'Simulation Execution Loop (One ABM Timestep)',
            fontsize=18, fontweight='bold', ha='center')
    ax.text(5, 13, '~38 layers, 600s physical time',
            fontsize=11, ha='center', style='italic')

    phases = [
        {
            'name': 'Phase 0: Population & Recruitment Setup',
            'color': '#FFE5E5',
            'layers': [
                '1. Update agent counts',
                '2. Reset ABM event counters',
                '3. Reset recruitment sources (CCL2 map)',
                '4. Update vasculature count (scaler)',
                '5. Mark vascular T sources (IFN-γ dep)',
                '6-10. Recruit immune cells (T, MAC, MDSC)',
            ]
        },
        {
            'name': 'Phase 1: Broadcast & Neighbor Scanning',
            'color': '#FFF5E5',
            'layers': [
                '11-17. Broadcast (x,y,z) positions (all types)',
                '18. Scan neighbors (26-voxel Moore)',
            ]
        },
        {
            'name': 'Phase 2: PDE & Gradient Computation',
            'color': '#E5F5FF',
            'layers': [
                '19. Reset PDE buffers (src/upt arrays)',
                '20. State transitions (division intents, death)',
                '21. Compute chemical sources (atomicAdd)',
                '22. Solve PDE (36 substeps, implicit CG)',
                '23. Compute gradients (∇C for chemotaxis)',
            ]
        },
        {
            'name': 'Phase 3: Fibroblast ECM',
            'color': '#E5FFE5',
            'layers': [
                '24. Zero ECM density field',
                '25. Build ECM (atomicAdd Gaussians)',
                '26. Update ECM grid (per-voxel smoothing)',
            ]
        },
        {
            'name': 'Phase 4: Occupancy & Movement',
            'color': '#FFF0E5',
            'layers': [
                '27. Zero occupancy grid',
                '28. Write to occ grid (CAS exclusive access)',
                '29-35. Movement substeps (6 iterations)',
                '   - Random walk + chemotaxis',
                '   - ECM restriction for cancer',
            ]
        },
        {
            'name': 'Phase 5: Division (Two-Phase)',
            'color': '#F5E5FF',
            'layers': [
                '36. Mark division intents',
                '37. Execute division (create offspring)',
            ]
        },
        {
            'name': 'Phase 6: QSP Integration',
            'color': '#E5FFE5',
            'layers': [
                '38. Solve QSP (CVODE forward dt)',
                '39. Export QSP data to CSV',
            ]
        },
    ]

    y_pos = 12.3
    for phase in phases:
        # Phase header
        phase_box = FancyBboxPatch((0.3, y_pos - 0.4), 9.4, 0.5,
                                    boxstyle="round,pad=0.05",
                                    edgecolor='black', facecolor=phase['color'],
                                    linewidth=2)
        ax.add_patch(phase_box)
        ax.text(0.5, y_pos - 0.15, phase['name'], fontsize=11, fontweight='bold', va='center')

        y_pos -= 0.7
        for layer in phase['layers']:
            ax.text(0.6, y_pos, layer, fontsize=9, va='top')
            y_pos -= 0.35

        y_pos -= 0.2


    plt.tight_layout()
    plt.savefig('outputs/architecture/02_simulation_loop.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: 02_simulation_loop.png")
    plt.close()

def fig3_flamegpu_overview():
    """FLAMEGPU GPU execution model with parallelism and messaging"""
    fig = plt.figure(figsize=(18, 14))

    # Create grid layout with more space
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.35)

    # Title
    fig.suptitle('FLAME GPU 2: GPU Execution Model & Parallelism',
                 fontsize=20, fontweight='bold', y=0.97)

    # ===== TOP LEFT: Agent Function Execution Flow =====
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('Agent Function Execution (Per Layer)', fontsize=13, fontweight='bold', pad=15)

    # Show how one layer executes
    y_exec = 9.2

    # Layer header
    ax1.text(5, y_exec, 'Layer: compute_chemical_sources()', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#FFE5CC', alpha=0.7))

    y_exec -= 1.0
    # GPU grid structure
    ax1.text(0.5, y_exec, 'GPU Grid Structure:', fontsize=9, fontweight='bold', color='#FF6B6B')
    y_exec -= 0.5
    ax1.text(0.8, y_exec, '• Blocks: ceil(N_agents / 256)', fontsize=8.5, family='monospace')
    y_exec -= 0.45
    ax1.text(0.8, y_exec, '• Threads/Block: 256 (warp-aligned)', fontsize=8.5, family='monospace')
    y_exec -= 0.45
    ax1.text(0.8, y_exec, '• 1 agent → 1 thread (fully parallel)', fontsize=8.5, family='monospace')

    y_exec -= 0.8
    # Execution model
    ax1.text(0.5, y_exec, 'What Each Thread Does:', fontsize=9, fontweight='bold', color='#FF6B6B')
    y_exec -= 0.5
    ax1.text(0.8, y_exec, '1. Read agent properties', fontsize=8.5, family='monospace')
    y_exec -= 0.45
    ax1.text(0.8, y_exec, '2. Compute release_rate', fontsize=8.5, family='monospace')
    y_exec -= 0.45
    ax1.text(0.8, y_exec, '3. atomicAdd to pde_source[voxel]', fontsize=8.5, family='monospace')
    y_exec -= 0.45
    ax1.text(0.8, y_exec, '4. Synchronize at kernel boundary', fontsize=8.5, family='monospace')

    y_exec -= 0.8
    ax1.text(0.5, y_exec, 'Memory Pattern:', fontsize=9, fontweight='bold', color='#FF6B6B')
    y_exec -= 0.5
    ax1.text(0.8, y_exec, '✓ Coalesced reads: agent data', fontsize=8.5, style='italic')
    y_exec -= 0.45
    ax1.text(0.8, y_exec, '✓ Atomic ops: serialize voxel writes', fontsize=8.5, style='italic')

    # ===== TOP RIGHT: Spatial Messaging =====
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('Spatial Messaging: broadcast_location()', fontsize=13, fontweight='bold', pad=15)

    # Draw a 3x3 voxel grid (simplified 2D view)
    voxel_size = 1.1
    grid_x_start = 2.8
    grid_y_start = 7.0

    # 3x3 grid
    colors = ['#FFE5E5', '#FFFFCC', '#E5FFE5']
    for i in range(3):
        for j in range(3):
            x = grid_x_start + i * voxel_size
            y = grid_y_start - j * voxel_size
            color = colors[(i + j) % 3]
            rect = Rectangle((x, y), voxel_size, voxel_size,
                            facecolor=color, edgecolor='black', linewidth=1.5)
            ax2.add_patch(rect)
            ax2.text(x + 0.6, y + 0.6, f'({i},{j})', fontsize=7, ha='center', va='center')

    # Agent in center
    center_x = grid_x_start + 1.2
    center_y = grid_y_start - 1.2
    circle = plt.Circle((center_x + 0.6, center_y + 0.6), 0.3, color='red', alpha=0.7, zorder=10)
    ax2.add_patch(circle)
    ax2.text(center_x + 0.6, center_y + 0.6, 'A', fontsize=8, ha='center', va='center',
            color='white', fontweight='bold', zorder=11)

    # Arrows to neighbors
    ax2.annotate('', xy=(grid_x_start + 0.6, grid_y_start + 0.6),
                xytext=(center_x + 0.6, center_y + 0.6),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='green', alpha=0.6))
    ax2.text(2, 8, 'Broadcast', fontsize=8, color='green', fontweight='bold')

    # Explanation
    y_msg = 5.8
    ax2.text(0.5, y_msg, 'Two-Phase Pattern:', fontsize=9, fontweight='bold', color='#4ECDC4')

    y_msg -= 0.65
    ax2.text(0.5, y_msg, '① Broadcast Phase', fontsize=8.5, fontweight='bold', color='green')
    y_msg -= 0.45
    ax2.text(0.8, y_msg, 'Each agent sends (id, x, y, z)', fontsize=8, family='monospace')
    y_msg -= 0.40
    ax2.text(0.8, y_msg, 'to 26-voxel neighbors', fontsize=8, family='monospace')

    y_msg -= 0.55
    ax2.text(0.5, y_msg, '② Global Barrier', fontsize=8.5, fontweight='bold', color='purple')
    y_msg -= 0.40
    ax2.text(0.8, y_msg, 'All agents complete', fontsize=8, family='monospace')

    y_msg -= 0.55
    ax2.text(0.5, y_msg, '③ Receive Phase', fontsize=8.5, fontweight='bold', color='blue')
    y_msg -= 0.45
    ax2.text(0.8, y_msg, 'Agents read messages,', fontsize=8, family='monospace')
    y_msg -= 0.40
    ax2.text(0.8, y_msg, 'count neighbors by type', fontsize=8, family='monospace')

    # ===== MIDDLE LEFT: Synchronization Points =====
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis('off')
    ax3.set_title('Layer Synchronization Model', fontsize=13, fontweight='bold', pad=15)

    y_sync = 9.3
    layers_to_show = [
        ('broadcast_location()', 'Send positions', '#FFE5CC'),
        ('GLOBAL BARRIER', 'All agents complete', '#FFCCCC'),
        ('scan_neighbors()', 'Count neighbors', '#FFE5CC'),
        ('GLOBAL BARRIER', 'All agents complete', '#FFCCCC'),
        ('state_step()', 'Update states', '#FFE5CC'),
        ('compute_sources()', 'AtomicAdd to PDE', '#FFE5CC'),
        ('GLOBAL BARRIER', 'Memory fence', '#FFCCCC'),
        ('solve_pde()', 'Host: PDE solver (sequential)', '#E5F5FF'),
    ]

    for layer_name, desc, color in layers_to_show:
        if 'BARRIER' in layer_name:
            # Barrier box
            barrier_box = FancyBboxPatch((0.3, y_sync - 0.38), 9.4, 0.35,
                                        boxstyle="round,pad=0.02",
                                        edgecolor='red', facecolor=color,
                                        linewidth=2, linestyle='--')
            ax3.add_patch(barrier_box)
            ax3.text(0.6, y_sync - 0.15, f'{layer_name}', fontsize=8, va='center', fontweight='bold')
            ax3.text(4.5, y_sync - 0.15, desc, fontsize=7.5, va='center', style='italic')
        else:
            # Layer box
            layer_box = FancyBboxPatch((0.3, y_sync - 0.38), 9.4, 0.35,
                                      boxstyle="round,pad=0.02",
                                      edgecolor='black', facecolor=color,
                                      linewidth=1)
            ax3.add_patch(layer_box)
            ax3.text(0.6, y_sync - 0.15, layer_name, fontsize=8, va='center', fontweight='bold')
            ax3.text(4.5, y_sync - 0.15, desc, fontsize=7.5, va='center', style='italic')

        y_sync -= 0.60

    # ===== MIDDLE RIGHT: AtomicAdd Contention =====
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    ax4.axis('off')
    ax4.set_title('Parallel AtomicAdd to Shared Memory', fontsize=13, fontweight='bold', pad=15)

    y_atomic = 9.2
    ax4.text(0.5, y_atomic, 'Layer: compute_chemical_sources()', fontsize=9, fontweight='bold', color='#4ECDC4')

    y_atomic -= 1.0
    # Show voxel with multiple agents
    ax4.add_patch(Rectangle((1.5, y_atomic-1.3), 2.5, 1.3, facecolor='#E5F5FF',
                            edgecolor='black', linewidth=2))
    ax4.text(2.75, y_atomic - 0.4, 'Voxel [x,y,z]', fontsize=9, fontweight='bold', ha='center')

    # Agents writing to same voxel
    agents_data = [
        ('T0\n(Cancer)', 0.8, y_atomic - 0.9),
        ('T5\n(T cell)', 2.75, y_atomic - 0.9),
        ('T42\n(MAC)', 4.7, y_atomic - 0.9),
    ]

    for agent_label, x, y in agents_data:
        ax4.text(x, y, agent_label, fontsize=7.5, ha='center',
                bbox=dict(boxstyle='round', facecolor='#FFE5CC', alpha=0.6))

    # Arrows to shared memory
    for agent_label, x, y in agents_data:
        ax4.annotate('', xy=(2.75, y_atomic - 1.6), xytext=(x, y - 0.25),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='red', alpha=0.6))

    ax4.text(2.75, y_atomic - 2.0, 'pde_source[voxel] += rate', fontsize=8, ha='center',
            bbox=dict(boxstyle='round', facecolor='#FFE5E5', alpha=0.7), family='monospace', fontweight='bold')

    y_atomic -= 3.0
    ax4.text(0.5, y_atomic, 'The Problem:', fontsize=9, fontweight='bold', color='#FF6B6B')
    y_atomic -= 0.55
    ax4.text(0.8, y_atomic, 'Multiple threads write same voxel', fontsize=8, family='monospace')
    y_atomic -= 0.5
    ax4.text(0.8, y_atomic, 'Race condition without locking', fontsize=8, family='monospace')

    y_atomic -= 0.65
    ax4.text(0.5, y_atomic, 'The Solution:', fontsize=9, fontweight='bold', color='#4ECDC4')
    y_atomic -= 0.55
    ax4.text(0.8, y_atomic, 'atomicAdd() ensures correctness', fontsize=8, family='monospace')
    y_atomic -= 0.50
    ax4.text(0.8, y_atomic, 'Hardware serializes writes', fontsize=8, family='monospace')

    y_atomic -= 0.65
    ax4.text(0.5, y_atomic, 'Performance Impact:', fontsize=9, fontweight='bold', color='green')
    y_atomic -= 0.55
    ax4.text(0.8, y_atomic, '✓ Minimal: grid large, sparse agents', fontsize=8, style='italic')

    # ===== BOTTOM: Agent Types & Execution =====
    ax5 = fig.add_subplot(gs[2, :])
    ax5.set_xlim(0, 10)
    ax5.set_ylim(0, 5)
    ax5.axis('off')
    ax5.set_title('All Agent Types Execute Functions in Parallel', fontsize=13, fontweight='bold', pad=15)

    # Simple listing
    agent_info = [
        ('Cancer Cell', 'broadcast → count_neighbors → state_step → compute_sources → move', '#FFE5E5'),
        ('T Cell / TReg', 'broadcast → scan_neighbors → state_step → compute_sources → move', '#FFF5E5'),
        ('Macrophage', 'broadcast → scan_neighbors → state_step → compute_sources → move', '#E5F5FF'),
        ('Fibroblast', 'broadcast → build_density → state_step → compute_sources → move', '#E5FFE5'),
        ('Vascular', 'broadcast → mark_sources → state_step → compute_sources → move', '#FFF0E5'),
        ('MDSC', 'broadcast → scan_neighbors → state_step → compute_sources → move', '#F5E5FF'),
    ]

    y_agent = 4.2
    for agent_name, functions, color in agent_info:
        # Agent box
        agent_box = FancyBboxPatch((0.3, y_agent - 0.38), 9.4, 0.45,
                                    boxstyle="round,pad=0.05",
                                    edgecolor='black', facecolor=color,
                                    linewidth=1, alpha=0.7)
        ax5.add_patch(agent_box)

        # Name and functions
        ax5.text(0.6, y_agent - 0.08, agent_name, fontsize=9, fontweight='bold', ha='left', va='center')
        ax5.text(2.5, y_agent - 0.08, functions, fontsize=8, ha='left', va='center', family='monospace')

        y_agent -= 0.65

    # Bottom note
    ax5.text(5, 0.5, 'Key: GPU assigns threads to agents. Each layer synchronizes all agents.',
            fontsize=9, ha='center', style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.6))

    plt.savefig('outputs/architecture/03_flamegpu_overview.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: 03_flamegpu_overview.png (enhanced GPU execution model)")
    plt.close()

def fig4_pde_solver():
    """PDE solver architecture"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.5, 'PDE Solver: Operator Splitting Architecture',
            fontsize=18, fontweight='bold', ha='center')

    # Problem statement
    problem_box = FancyBboxPatch((0.5, 8.3), 9, 0.9,
                                  boxstyle="round,pad=0.08",
                                  edgecolor='black', facecolor=COLOR_PDE, alpha=0.4,
                                  linewidth=2)
    ax.add_patch(problem_box)
    ax.text(5, 8.75, '∂C/∂t = D∇²C - λC + S(agents)',
            fontsize=13, fontweight='bold', ha='center', family='monospace')
    ax.text(5, 8.35, 'Backward Euler (implicit) | Unconditionally stable | 10 substrates',
            fontsize=10, ha='center', style='italic')

    # Per-ABM-step flow
    y = 7.8
    ax.text(0.5, y, 'Per ABM Timestep (dt = 600s):', fontsize=12, fontweight='bold')

    y -= 0.5
    steps = [
        ('1. Reset Buffers', 'Zero PDE_sources[] and PDE_uptakes[]', COLOR_DATA),
        ('2. Agent Phase', 'All agents atomicAdd to PDE source/uptake arrays', COLOR_ABM),
        ('3. Loop: 36 substeps (dt_sub = 16.67s)', '', 'white'),
        ('   a. LOD Diffusion', 'Solve: dC/dt = D∇²C via Conjugate Gradient (7-point stencil)', COLOR_PDE),
        ('   b. Exact Decay', 'Apply: C(t) = C(t-dt)·exp(-λ·dt_sub)', COLOR_PDE),
        ('4. Compute Gradients', '∇C for all 10 chemicals (used by chemotaxis next step)', COLOR_DATA),
    ]

    for i, (step, desc, color) in enumerate(steps):
        if color == 'white':
            ax.text(0.7, y, step, fontsize=11, fontweight='bold', style='italic')
        else:
            step_box = FancyBboxPatch((0.5, y - 0.25), 9, 0.35,
                                      boxstyle="round,pad=0.02",
                                      edgecolor='gray', facecolor=color, alpha=0.3,
                                      linewidth=1)
            ax.add_patch(step_box)
            ax.text(0.7, y - 0.05, f'{step}: {desc}', fontsize=10)
        y -= 0.45

    # Algorithm details box
    y -= 0.3
    algo_box = FancyBboxPatch((0.5, y - 2.5), 4.3, 2.5,
                               boxstyle="round,pad=0.08",
                               edgecolor='black', facecolor=COLOR_PDE, alpha=0.3,
                               linewidth=2)
    ax.add_patch(algo_box)
    ax.text(2.65, y - 0.15, 'LOD Diffusion Step', fontsize=11, fontweight='bold', ha='center')

    algo_details = [
        'Matrix-free conjugate gradient',
        'Operator: A·x = (I + α·∇²)·x',
        '  α = dt_sub · D / dx²',
        '7-point stencil (6 neighbors + center)',
        'Neumann BCs (no-flux boundaries)',
        'Tolerance: 1e-4 relative',
        'Max iterations: 100',
    ]

    y_algo = y - 0.5
    for detail in algo_details:
        ax.text(0.7, y_algo, detail, fontsize=8, family='monospace')
        y_algo -= 0.3

    # Decay details box
    decay_box = FancyBboxPatch((5.2, y - 2.5), 4.3, 2.5,
                                boxstyle="round,pad=0.08",
                                edgecolor='black', facecolor=COLOR_PDE, alpha=0.3,
                                linewidth=2)
    ax.add_patch(decay_box)
    ax.text(7.35, y - 0.15, 'Exact Decay Step', fontsize=11, fontweight='bold', ha='center')

    decay_details = [
        'Exponential decay + sources',
        'C_new = C_old·exp(-λ·dt) + S·(1-exp(-λ·dt))/λ',
        'Handles stiff decay (e.g., NO)',
        'Exact solution (no error)',
        'Fast: element-wise operation',
        '',
        'Stability: unconditional (any dt!)',
    ]

    y_decay = y - 0.5
    for detail in decay_details:
        ax.text(5.4, y_decay, detail, fontsize=8, family='monospace')
        y_decay -= 0.3

    # Bottom: Performance notes
    perf_box = FancyBboxPatch((0.5, 0.2), 9, 0.8,
                               boxstyle="round,pad=0.08",
                               edgecolor='black', facecolor='lightyellow',
                               linewidth=1, linestyle='--')
    ax.add_patch(perf_box)
    ax.text(5, 0.8, 'Performance: ~36 ms per ABM step on modern GPU (A100/RTX)',
            fontsize=10, ha='center', fontweight='bold')
    ax.text(5, 0.4, 'Key innovation: Operator splitting avoids large coupled system; LOD avoids matrix inversion',
            fontsize=9, ha='center', style='italic')

    plt.tight_layout()
    plt.savefig('outputs/architecture/04_pde_solver.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: 04_pde_solver.png")
    plt.close()

def fig5_coupling():
    """PDE-ABM and QSP-ABM coupling"""
    fig, ax = plt.subplots(figsize=(14, 11))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11)
    ax.axis('off')

    # Title
    ax.text(5, 10.5, 'Coupling Mechanisms: PDE-ABM & QSP-ABM',
            fontsize=18, fontweight='bold', ha='center')

    # PDE-ABM Coupling (left)
    ax.text(2.5, 10, 'PDE-ABM Coupling (Bidirectional)',
            fontsize=13, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor=COLOR_PDE, alpha=0.4))

    # ABM reads from PDE
    read_box = FancyBboxPatch((0.3, 8.5), 4.2, 1.2,
                               boxstyle="round,pad=0.08",
                               edgecolor='black', facecolor=COLOR_ABM, alpha=0.3,
                                linewidth=2)
    ax.add_patch(read_box)
    ax.text(2.4, 9.5, 'ABM Reads from PDE', fontsize=11, fontweight='bold', ha='center')
    read_details = [
        'Layer: compute_chemical_sources',
        'Pointer-based: float* ptr from environment',
        'C[voxel_idx] = pde_conc_ptr[voxel_idx]',
        'All agents read in parallel',
    ]
    for i, detail in enumerate(read_details):
        ax.text(0.5, 9.15 - i*0.3, detail, fontsize=8, family='monospace')

    # Arrow down
    arrow_read = FancyArrowPatch((2.4, 8.4), (2.4, 7.9),
                                 arrowstyle='->', mutation_scale=20, lw=2, color='green')
    ax.add_patch(arrow_read)

    # ABM writes to PDE
    write_box = FancyBboxPatch((0.3, 6.7), 4.2, 1.0,
                                boxstyle="round,pad=0.08",
                                edgecolor='black', facecolor=COLOR_ABM, alpha=0.3,
                                linewidth=2)
    ax.add_patch(write_box)
    ax.text(2.4, 7.5, 'ABM Writes to PDE', fontsize=11, fontweight='bold', ha='center')
    write_details = [
        'atomicAdd to pde_source_ptr & pde_uptake_ptr',
        'Source: release_rate / voxel_volume',
        'Uptake: uptake_rate [1/s] direct',
    ]
    for i, detail in enumerate(write_details):
        ax.text(0.5, 7.25 - i*0.3, detail, fontsize=8, family='monospace')

    # Arrow down
    arrow_write = FancyArrowPatch((2.4, 6.6), (2.4, 6.1),
                                  arrowstyle='->', mutation_scale=20, lw=2, color='blue')
    ax.add_patch(arrow_write)

    # PDE solve
    pde_box = FancyBboxPatch((0.3, 4.9), 4.2, 1.0,
                              boxstyle="round,pad=0.08",
                              edgecolor='black', facecolor=COLOR_PDE, alpha=0.3,
                              linewidth=2)
    ax.add_patch(pde_box)
    ax.text(2.4, 5.7, 'PDE Solve', fontsize=11, fontweight='bold', ha='center')
    pde_details = [
        'Layer: solve_pde (host function)',
        '36 substeps with operator splitting',
        'Updates C[voxel] concentrations',
    ]
    for i, detail in enumerate(pde_details):
        ax.text(0.5, 5.45 - i*0.3, detail, fontsize=8, family='monospace')

    # Arrow down
    arrow_pde = FancyArrowPatch((2.4, 4.8), (2.4, 4.3),
                                arrowstyle='->', mutation_scale=20, lw=2, color='purple')
    ax.add_patch(arrow_pde)

    # Gradient computation
    grad_box = FancyBboxPatch((0.3, 3.1), 4.2, 1.0,
                               boxstyle="round,pad=0.08",
                               edgecolor='black', facecolor=COLOR_PDE, alpha=0.3,
                               linewidth=2)
    ax.add_patch(grad_box)
    ax.text(2.4, 3.9, 'Compute Gradients', fontsize=11, fontweight='bold', ha='center')
    grad_details = [
        'Layer: compute_pde_gradients (host)',
        '∇C for all 10 chemicals',
        'Used by chemotaxis next step',
    ]
    for i, detail in enumerate(grad_details):
        ax.text(0.5, 3.65 - i*0.3, detail, fontsize=8, family='monospace')

    # QSP-ABM Coupling (right)
    ax.text(7.5, 10, 'QSP-ABM Coupling (Partial)',
            fontsize=13, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor=COLOR_QSP, alpha=0.4))

    # QSP forward
    qsp_box = FancyBboxPatch((5.3, 8.7), 4.2, 1.3,
                              boxstyle="round,pad=0.08",
                              edgecolor='black', facecolor=COLOR_QSP, alpha=0.3,
                              linewidth=2)
    ax.add_patch(qsp_box)
    ax.text(7.4, 9.8, 'QSP ODE Integration', fontsize=11, fontweight='bold', ha='center')
    qsp_details = [
        'Layer: solve_qsp (host)',
        'CVODE integration: y_new = solve(dt=600s)',
        '153 species, synchronized with ABM',
        'Currently decoupled from ABM feedback',
    ]
    for i, detail in enumerate(qsp_details):
        ax.text(5.5, 9.55 - i*0.3, detail, fontsize=8, family='monospace')

    # Bidirectional annotations
    ax.text(7.4, 8.2, 'ABM Events → QSP (Bookkeeping):', fontsize=10, fontweight='bold')
    event_details = [
        '✓ Cancer deaths (stem/prog/senescent)',
        '✓ T cell recruitment count',
        '✓ MAC recruitment count',
        '✗ Not yet: Drug effects on cells',
        '✗ Not yet: Immune state feedback',
    ]
    for i, detail in enumerate(event_details):
        color = 'green' if detail.startswith('✓') else 'red'
        ax.text(5.5, 7.85 - i*0.35, detail, fontsize=8, color=color, family='monospace')

    ax.text(7.4, 6.2, 'QSP State → ABM (Planned):', fontsize=10, fontweight='bold')
    planned_details = [
        '✗ Drug concentrations (NIVO, CABO)',
        '✗ Immune lymphocyte populations',
        '✗ Cytokine effects on cells',
        '✗ Checkpoint blockade (PD1-PDL1)',
    ]
    for i, detail in enumerate(planned_details):
        ax.text(5.5, 5.85 - i*0.35, detail, fontsize=8, color='red', family='monospace')

    # Key points box
    key_box = FancyBboxPatch((0.3, 0.3), 9.4, 2.4,
                              boxstyle="round,pad=0.08",
                              edgecolor='black', facecolor='lightyellow',
                              linewidth=2, linestyle='--')
    ax.add_patch(key_box)
    ax.text(5, 2.5, 'Key Architectural Insights', fontsize=12, fontweight='bold', ha='center')

    key_points = [
        ('PDE-ABM:', 'Tight coupling (every layer). Agents read/write directly via pointers. AtomicAdds handle parallelism.'),
        ('QSP-ABM:', 'Loose coupling (presently). ODE steps forward each ABM step but doesn\'t drive recruitment. Bookkeeping ready for feedback.'),
        ('Synchronization:', 'All three components (ABM, PDE, QSP) step together: dt_ABM = 600s = 1 QSP step = 36 PDE substeps.'),
        ('Memory:', 'Pointer-based design (uint64_t env props) avoids extra copies. Device arrays accessed directly by agents.'),
    ]

    y_key = 2.2
    for title, detail in key_points:
        ax.text(0.6, y_key, f'{title} {detail}', fontsize=8.5)
        y_key -= 0.4

    plt.tight_layout()
    plt.savefig('outputs/architecture/05_coupling.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: 05_coupling.png")
    plt.close()

def fig6_memory_passing():
    """Memory usage and data passing"""
    fig, ax = plt.subplots(figsize=(14, 11))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11)
    ax.axis('off')

    # Title
    ax.text(5, 10.5, 'Memory Usage & Data Passing Architecture',
            fontsize=18, fontweight='bold', ha='center')

    # GPU Memory allocation
    ax.text(2.5, 10, 'GPU Memory (VRAM)', fontsize=12, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor=COLOR_GPU, alpha=0.3))

    # Memory breakdown
    gpu_mem_items = [
        ('PDE Concentration Arrays (pde_conc)', 10 * 4, 'Grid³ × 10 chemicals × 4B/float'),
        ('PDE Source/Uptake Buffers', 10 * 4 * 2, 'Grid³ × 10 × 2 (src, upt) × 4B'),
        ('PDE Gradient Arrays (∇x,∇y,∇z)', 10 * 4 * 3, 'Grid³ × 10 × 3 (gradients) × 4B'),
        ('Occupancy Grid (d_cancer_occ)', 8, 'Grid³ × 8B/uint64'),
        ('ECM Density Field', 4, 'Grid³ × 4B/float'),
        ('Agent Data (positions, states, etc.)', 200, 'N_agents × ~500B/agent'),
        ('FLAMEGPU Internals', 500, 'CUDA runtime, message buffers, etc.'),
    ]

    y_mem = 9.5
    total_for_50 = 0
    for name, bytes_per_grid, desc in gpu_mem_items:
        if 'Agents' in name or 'FLAMEGPU' in name:
            ax.text(0.5, y_mem, f'• {name}: ~{bytes_per_grid}MB (fixed)', fontsize=9)
            ax.text(0.7, y_mem - 0.25, desc, fontsize=7.5, style='italic')
        else:
            mem_50 = bytes_per_grid * 50**3 / 1e9  # Convert to GB for 50³
            mem_320 = bytes_per_grid * 320**3 / 1e9  # Convert to GB for 320³
            ax.text(0.5, y_mem, f'• {name}: 50³→{mem_50:.2f}GB, 320³→{mem_320:.2f}GB', fontsize=9)
            ax.text(0.7, y_mem - 0.25, desc, fontsize=7.5, style='italic')
            total_for_50 += mem_50
        y_mem -= 0.55

    # Memory totals box
    totals_box = FancyBboxPatch((0.3, y_mem - 0.8), 4.4, 0.8,
                                 boxstyle="round,pad=0.08",
                                 edgecolor='black', facecolor='lightcyan',
                                 linewidth=2)
    ax.add_patch(totals_box)
    ax.text(2.5, y_mem - 0.2, f'Total (50³ grid): ~0.8-1.2 GB', fontsize=10, fontweight='bold', ha='center')
    ax.text(2.5, y_mem - 0.5, f'Max (320³ grid): ~8-10 GB on 12GB card', fontsize=9, ha='center', style='italic')

    # Data passing mechanisms
    ax.text(7.5, 10, 'Data Passing Mechanisms', fontsize=12, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor=COLOR_MEMORY, alpha=0.3))

    # Method 1: Environment Properties
    method1_box = FancyBboxPatch((5.3, 8.5), 4.2, 1.5,
                                  boxstyle="round,pad=0.08",
                                  edgecolor='black', facecolor=COLOR_DATA, alpha=0.3,
                                  linewidth=1.5)
    ax.add_patch(method1_box)
    ax.text(7.4, 9.7, '1. Environment Properties (Pointers)', fontsize=10, fontweight='bold', ha='center')
    method1_details = [
        'Type: uint64_t env properties',
        'Value: GPU array pointers',
        'Usage: PDE concentrations, sources,',
        '       gradients, ECM field, occ_grid',
        'Accessed: reinterpret_cast<float*>(...)',
        'Parallel: All agents read/write in parallel',
    ]
    for i, detail in enumerate(method1_details):
        ax.text(5.5, 9.45 - i*0.25, detail, fontsize=7.5, family='monospace')

    # Method 2: Messages
    method2_box = FancyBboxPatch((5.3, 6.5), 4.2, 1.8,
                                  boxstyle="round,pad=0.08",
                                  edgecolor='black', facecolor=COLOR_DATA, alpha=0.3,
                                  linewidth=1.5)
    ax.add_patch(method2_box)
    ax.text(7.4, 7.95, '2. FLAMEGPU Messages', fontsize=10, fontweight='bold', ha='center')
    method2_details = [
        'broadcast_location(): Send (x,y,z) to neighbors',
        'scan_neighbors(): Count agent types in Moore neighborhood',
        'Built-in FLAMEGPU messaging (efficient)',
        'One message type: broadcast (x,y,z,cell_id)',
        'Scope: 26-voxel neighborhood + center',
    ]
    for i, detail in enumerate(method2_details):
        ax.text(5.5, 7.7 - i*0.28, detail, fontsize=7.5, family='monospace')

    # Method 3: Host-GPU boundary
    method3_box = FancyBboxPatch((5.3, 4.2), 4.2, 2.0,
                                  boxstyle="round,pad=0.08",
                                  edgecolor='black', facecolor=COLOR_DATA, alpha=0.3,
                                  linewidth=1.5)
    ax.add_patch(method3_box)
    ax.text(7.4, 6.0, '3. Host-GPU Data Transfer', fontsize=10, fontweight='bold', ha='center')
    method3_details = [
        'Host reads: Agent counts (cudaMemcpy D2H)',
        '→ Used for QSP event bookkeeping',
        'Host writes: Recruitment markers (cudaMemcpy H2D)',
        '→ T sources (IFN-γ map), MAC sources (CCL2 map)',
        'CSV output: Agent positions & states (pinned mem)',
        '→ Async transfer during next step (overlap compute)',
    ]
    for i, detail in enumerate(method3_details):
        ax.text(5.5, 5.75 - i*0.28, detail, fontsize=7.5, family='monospace')

    # Data flow diagram at bottom
    ax.text(5, 3.8, 'Data Flow Hierarchy (per ABM step)', fontsize=11, fontweight='bold', ha='center')

    # Layer sequence
    y_flow = 3.4
    flow_items = [
        ('GPU Layer: compute_chemical_sources', 'Agents atomicAdd → PDE src/upt', COLOR_ABM),
        ('GPU Layer: solve_pde', 'Update C[voxel] → PDE concentrations', COLOR_PDE),
        ('GPU Layer: compute_pde_gradients', 'Compute ∇C from concentrations', COLOR_PDE),
        ('Host Layer: update_agent_counts', 'cudaMemcpy: agent counts D2H', COLOR_MEMORY),
        ('Host Function: recruit_t_cells', 'cudaMemcpy: mark sources H2D', COLOR_MEMORY),
        ('GPU Layer: move', 'Agents read ∇C for chemotaxis', COLOR_ABM),
    ]

    for i, (layer, action, color) in enumerate(flow_items):
        flow_box = FancyBboxPatch((0.3, y_flow - 0.3), 9.4, 0.35,
                                   boxstyle="round,pad=0.02",
                                   edgecolor='gray', facecolor=color, alpha=0.25,
                                   linewidth=1)
        ax.add_patch(flow_box)
        ax.text(0.5, y_flow - 0.05, f'{layer}: {action}', fontsize=8.5, family='monospace')
        y_flow -= 0.5

    # Key points
    key_box = FancyBboxPatch((0.3, 0.1), 9.4, 0.8,
                              boxstyle="round,pad=0.08",
                              edgecolor='black', facecolor='lightyellow',
                              linewidth=1, linestyle='--')
    ax.add_patch(key_box)
    ax.text(5, 0.75, 'Key: Pointer-based design minimizes copies. Most data stays on GPU. Host transfers only summaries (agent counts, events).',
            fontsize=9, ha='center', fontweight='bold')
    ax.text(5, 0.35, 'Performance: Async H2D/D2H overlap with GPU compute. No blocking on host.',
            fontsize=8, ha='center', style='italic')

    plt.tight_layout()
    plt.savefig('outputs/architecture/06_memory_passing.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: 06_memory_passing.png")
    plt.close()

def fig2b_simulation_phases():
    """Simplified simulation loop: just the 7 phases"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(0.5, 9.5, 'ABM Execution: 7 Phases per Timestep',
            fontsize=20, fontweight='bold', ha='left')
    ax.text(0.5, 9, '600 seconds (10 min) physical time',
            fontsize=12, ha='left', style='italic')

    # Color pattern: red, orange, blue, green, repeat
    colors = ['#FFE5E5', '#FFF5E5', '#E5F5FF', '#E5FFE5']

    phases = [
        {
            'num': '0',
            'name': 'Population & Recruitment',
            'desc': 'Count agents, mark immune sources, create new T cells/MACs/MDSCs',
            'y': 8.0
        },
        {
            'num': '1',
            'name': 'Broadcast & Neighbors',
            'desc': 'Agents send positions, count local neighbors (26-voxel Moore)',
            'y': 7.0
        },
        {
            'num': '2',
            'name': 'PDE & Gradients',
            'desc': 'Agents compute chemical sources → PDE solves (36 substeps) → compute ∇C',
            'y': 6.0
        },
        {
            'num': '3',
            'name': 'Fibroblast ECM',
            'desc': 'Zero field → build density → smooth (Gaussian per-voxel)',
            'y': 5.0
        },
        {
            'num': '4',
            'name': 'Occupancy & Movement',
            'desc': 'Write voxel occupancy → 6 movement substeps with chemotaxis',
            'y': 4.0
        },
        {
            'num': '5',
            'name': 'Division',
            'desc': 'Mark division intents → execute divisions (create offspring)',
            'y': 3.0
        },
        {
            'num': '6',
            'name': 'QSP Integration',
            'desc': 'Solve ODE system (CVODE) → advance by 600s → export state',
            'y': 2.0
        },
    ]

    for idx, phase in enumerate(phases):
        y = phase['y']
        color = colors[idx % len(colors)]

        # Calculate box width based on longest text (description)
        # Using approximate character width: ~0.045 per char at fontsize 10
        desc_width = len(phase['desc']) * 0.045
        box_width = max(desc_width, 5.5)  # Minimum width

        # Phase box (sized to text)
        phase_box = FancyBboxPatch((0.5, y - 0.5), box_width, 0.75,
                                    boxstyle="round,pad=0.08",
                                    edgecolor='black', facecolor=color,
                                    linewidth=2, alpha=0.7)
        ax.add_patch(phase_box)

        # Phase number and name (left justified)
        ax.text(0.7, y + 0.05, f"Phase {phase['num']}", fontsize=14, fontweight='bold', ha='left')
        ax.text(1.8, y + 0.05, phase['name'], fontsize=13, fontweight='bold', ha='left')

        # Description (left justified)
        ax.text(0.7, y - 0.25, phase['desc'], fontsize=10, style='italic', ha='left')

    # Bottom legend
    y_legend = 1.1
    ax.text(0.5, y_legend, 'GPU Layers:', fontsize=11, fontweight='bold')
    ax.text(0.5, y_legend - 0.35, '• Phase 0: Host functions (recruitment)', fontsize=9)
    ax.text(0.5, y_legend - 0.65, '• Phase 1: Agent message passing (broadcast → receive)', fontsize=9)
    ax.text(0.5, y_legend - 0.95, '• Phases 2-5: Agent functions (parallel) + host functions (coordinate)', fontsize=9)
    ax.text(0.5, y_legend - 1.25, '• Phase 6: Host function (synchronous QSP ODE solve)', fontsize=9)

    ax.text(5.5, y_legend, 'Synchronization:', fontsize=11, fontweight='bold')
    ax.text(5.5, y_legend - 0.35, '• Global barrier after each GPU phase', fontsize=9)
    ax.text(5.5, y_legend - 0.65, '• All agents complete before next phase', fontsize=9)
    ax.text(5.5, y_legend - 0.95, '• Memory fence after atomic operations', fontsize=9)
    ax.text(5.5, y_legend - 1.25, '• Ensures correct chemical/state updates', fontsize=9)

    plt.tight_layout()
    plt.savefig('outputs/architecture/02b_simulation_phases.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: 02b_simulation_phases.png")
    plt.close()

if __name__ == '__main__':
    print("Generating architecture figures...")
    print()

    fig1_system_architecture()
    fig2_simulation_loop()
    fig2b_simulation_phases()
    fig3_flamegpu_overview()
    fig4_pde_solver()
    fig5_coupling()
    fig6_memory_passing()

    print()
    print("=" * 60)
    print("✓ All figures generated successfully!")
    print("=" * 60)
    print()
    print("Output files:")
    print("  - outputs/architecture/01_system_architecture.png")
    print("  - outputs/architecture/02_simulation_loop.png")
    print("  - outputs/architecture/03_flamegpu_overview.png")
    print("  - outputs/architecture/04_pde_solver.png")
    print("  - outputs/architecture/05_coupling.png")
    print("  - outputs/architecture/06_memory_passing.png")
    print()
