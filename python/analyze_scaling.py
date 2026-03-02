#!/usr/bin/env python3
"""
Scaling analysis for SPQSP PDAC model.

Reads timing.csv, init_timing.csv, and memory.txt from each grid size directory,
analyzes wall time vs grid size, GPU memory usage, component breakdown, and
produces scaling plots.
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Configuration
SCALING_DIR = Path("outputs/scaling")
WARMUP_STEPS = 3  # Discard first N steps (CUDA JIT compilation)
OUTPUT_DIR = Path("outputs/scaling")

def extract_grid_size(dirname):
    """Extract grid size from directory name 'grid_N'."""
    try:
        return int(dirname.split('_')[-1])
    except:
        return None

def read_timing_csv(filepath, warmup=WARMUP_STEPS):
    """Read timing.csv, discard warmup steps, return DataFrame with stats."""
    if not filepath.exists():
        return None
    try:
        df = pd.read_csv(filepath)
        # Discard warmup
        df = df.iloc[warmup:]
        return df
    except:
        return None

def read_init_timing_csv(filepath):
    """Read init_timing.csv, return dict of phase->ms."""
    if not filepath.exists():
        return {}
    try:
        df = pd.read_csv(filepath)
        return dict(zip(df['phase'], df['ms']))
    except:
        return {}

def parse_memory_file(filepath):
    """Parse memory.txt, extract last (peak) memory usage in MB."""
    if not filepath.exists():
        return None
    try:
        with open(filepath) as f:
            line = f.readlines()[-1].strip()
            # Parse "[MEM] After agent init: XXXX MB used / YYYY MB total"
            parts = line.split()
            used_idx = [i for i, p in enumerate(parts) if p == "used"][0]
            used_mb = int(parts[used_idx - 1])
            return used_mb
    except:
        return None

def main():
    # Discover all grid_* directories
    grid_dirs = sorted(
        [(extract_grid_size(d), d) for d in os.listdir(SCALING_DIR) if d.startswith('grid_')],
        key=lambda x: x[0] if x[0] else 999
    )

    if not grid_dirs:
        print(f"Error: No grid_* directories found in {SCALING_DIR}")
        return

    grid_sizes = []
    timing_stats = []  # List of (grid, mean_step_ms, std_step_ms, pde_ms, qsp_ms, abm_ms)
    memory_usage = []  # List of (grid, memory_mb)
    init_timings = {}  # Dict[grid] -> dict of phase->ms

    print(f"Analyzing {len(grid_dirs)} grid sizes...\n")

    for grid_size, dirname in grid_dirs:
        if grid_size is None:
            continue

        griddir = SCALING_DIR / dirname
        print(f"Grid {grid_size}³:")

        # Read timing
        timing_df = read_timing_csv(griddir / "timing.csv", warmup=WARMUP_STEPS)
        if timing_df is not None and len(timing_df) > 0:
            mean_step_ms = timing_df['total_ms'].mean()
            std_step_ms = timing_df['total_ms'].std()
            mean_pde_ms = timing_df['pde_ms'].mean()
            mean_qsp_ms = timing_df['qsp_ms'].mean()
            mean_abm_ms = timing_df['abm_ms'].mean()

            grid_sizes.append(grid_size)
            timing_stats.append((grid_size, mean_step_ms, std_step_ms, mean_pde_ms, mean_qsp_ms, mean_abm_ms))
            print(f"  Mean step time: {mean_step_ms:.2f} ± {std_step_ms:.2f} ms")
            print(f"    PDE: {mean_pde_ms:.2f} ms, QSP: {mean_qsp_ms:.2f} ms, ABM: {mean_abm_ms:.2f} ms")
        else:
            print(f"  No timing data found")

        # Read memory
        mem_mb = parse_memory_file(griddir / "memory.txt")
        if mem_mb is not None:
            memory_usage.append((grid_size, mem_mb))
            print(f"  GPU memory: {mem_mb} MB")

        # Read init timing
        init_times = read_init_timing_csv(griddir / "init_timing.csv")
        if init_times:
            init_timings[grid_size] = init_times
            print(f"  Init phases: {', '.join([f'{k}={v:.1f}ms' for k,v in list(init_times.items())[:3]])}")

        print()

    if not timing_stats:
        print("Error: No timing data available")
        return

    # =========================================================================
    # Plotting
    # =========================================================================
    print("Generating plots...")
    fig = plt.figure(figsize=(16, 12))

    # Convert to arrays for plotting
    grids = np.array([t[0] for t in timing_stats])
    voxel_counts = grids ** 3
    step_times = np.array([t[1] for t in timing_stats])
    pde_times = np.array([t[3] for t in timing_stats])
    qsp_times = np.array([t[4] for t in timing_stats])
    abm_times = np.array([t[5] for t in timing_stats])

    # 1. Scaling plot (linear scale)
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(voxel_counts, step_times, 'o-', linewidth=2, markersize=8, label='Total')
    ax1.plot(voxel_counts, pde_times, 's-', linewidth=2, markersize=6, label='PDE')
    ax1.plot(voxel_counts, qsp_times, '^-', linewidth=2, markersize=6, label='QSP')
    ax1.plot(voxel_counts, abm_times, 'v-', linewidth=2, markersize=6, label='ABM')
    ax1.set_xlabel('Voxel Count (log scale)')
    ax1.set_ylabel('Time per Step (ms)')
    ax1.set_title('Scaling: Step Time vs Voxel Count')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Scaling plot (log-log with fit)
    ax2 = plt.subplot(2, 3, 2)
    ax2.loglog(voxel_counts, step_times, 'o-', linewidth=2, markersize=8, label='Measured')

    # Fit power law: t ~ N^p
    coeffs = np.polyfit(np.log(voxel_counts), np.log(step_times), 1)
    power = coeffs[0]
    fit_times = np.exp(coeffs[1]) * voxel_counts ** power
    ax2.loglog(voxel_counts, fit_times, '--', linewidth=2, label=f'Fit: t ∝ N^{power:.2f}')
    ax2.set_xlabel('Voxel Count')
    ax2.set_ylabel('Time per Step (ms)')
    ax2.set_title('Scaling Exponent')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Component breakdown (stacked bar)
    ax3 = plt.subplot(2, 3, 3)
    x_pos = np.arange(len(grids))
    width = 0.6
    ax3.bar(x_pos, pde_times, width, label='PDE', color='#FF6B6B')
    ax3.bar(x_pos, qsp_times, width, bottom=pde_times, label='QSP', color='#4ECDC4')
    ax3.bar(x_pos, abm_times, width, bottom=pde_times+qsp_times, label='ABM', color='#45B7D1')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'{int(g)}³' for g in grids], rotation=45)
    ax3.set_ylabel('Time per Step (ms)')
    ax3.set_title('Component Time Breakdown')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. GPU memory usage
    if memory_usage:
        ax4 = plt.subplot(2, 3, 4)
        mem_grids = np.array([m[0] for m in memory_usage])
        mem_mbs = np.array([m[1] for m in memory_usage])
        mem_voxels = mem_grids ** 3
        ax4.plot(mem_voxels, mem_mbs, 'o-', linewidth=2, markersize=8, color='#95E1D3')
        ax4.set_xlabel('Voxel Count (log scale)')
        ax4.set_ylabel('GPU Memory (MB)')
        ax4.set_title('GPU Memory Usage')
        ax4.set_xscale('log')
        ax4.grid(True, alpha=0.3)
        # Add 12GB limit line
        ax4.axhline(y=12000, color='red', linestyle='--', label='12 GB limit')
        ax4.legend()

    # 5. Throughput (voxels/second)
    ax5 = plt.subplot(2, 3, 5)
    throughput = (voxel_counts / step_times) * 1000  # voxels/second
    ax5.loglog(voxel_counts, throughput, 'o-', linewidth=2, markersize=8, color='#F38181')
    ax5.set_xlabel('Voxel Count')
    ax5.set_ylabel('Throughput (voxels/sec)')
    ax5.set_title('GPU Efficiency')
    ax5.grid(True, alpha=0.3)

    # 6. Initialization breakdown (if available)
    if init_timings:
        ax6 = plt.subplot(2, 3, 6)
        phases_to_plot = ['build_model', 'init_pde', 'init_qsp', 'cuda_sim_create', 'init_agents']
        init_grids = sorted(init_timings.keys())

        phase_data = {phase: [] for phase in phases_to_plot}
        for g in init_grids:
            for phase in phases_to_plot:
                phase_data[phase].append(init_timings[g].get(phase, 0))

        x_pos = np.arange(len(init_grids))
        width = 0.6
        bottom = np.zeros(len(init_grids))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#95E1D3', '#F38181']

        for i, phase in enumerate(phases_to_plot):
            ax6.bar(x_pos, phase_data[phase], width, bottom=bottom, label=phase, color=colors[i % len(colors)])
            bottom += np.array(phase_data[phase])

        ax6.set_xticks(x_pos)
        ax6.set_xticklabels([f'{int(g)}³' for g in init_grids], rotation=45)
        ax6.set_ylabel('Time (ms)')
        ax6.set_title('Initialization Breakdown')
        ax6.legend(loc='upper left', fontsize=8)
        ax6.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save plot
    plot_file = OUTPUT_DIR / "scaling_analysis.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_file}")

    # =========================================================================
    # Summary statistics
    # =========================================================================
    summary_file = OUTPUT_DIR / "scaling_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("SPQSP PDAC Scaling Study Summary\n")
        f.write("=" * 60 + "\n\n")

        f.write("Grid Size | Voxels   | Step Time (ms) | PDE | QSP | ABM\n")
        f.write("-" * 60 + "\n")
        for i, (g, t, _, pde, qsp, abm) in enumerate(timing_stats):
            vc = g ** 3
            f.write(f"{g:3d}³    | {vc:8d} | {t:14.2f} | {pde:7.2f} | {qsp:7.2f} | {abm:7.2f}\n")

        if memory_usage:
            f.write("\n\nGPU Memory Usage\n")
            f.write("-" * 60 + "\n")
            f.write("Grid Size | Memory (MB)\n")
            f.write("-" * 30 + "\n")
            for g, mem in memory_usage:
                f.write(f"{g:3d}³    | {mem:10d}\n")

        f.write("\n\nScaling Analysis\n")
        f.write("-" * 60 + "\n")
        if len(timing_stats) > 1:
            power = np.polyfit(np.log(voxel_counts), np.log(step_times), 1)[0]
            f.write(f"Scaling exponent (t ∝ N^p): {power:.2f}\n")
            f.write(f"Expected for O(N): 1.0\n")
            f.write(f"Expected for O(N log N): ~1.0-1.1\n")
            f.write(f"Observed: {power:.2f}\n")

    print(f"Saved: {summary_file}")
    print("\nAnalysis complete!")

if __name__ == '__main__':
    main()
