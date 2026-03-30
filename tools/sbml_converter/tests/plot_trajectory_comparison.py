"""Plot relative error between C++ and SimBiology trajectories.

Reads directly from the simulation outputs — no hardcoded parameters.

Usage:
    python plot_trajectory_comparison.py <cpp_csv> <sb_csv> <output_png>
"""
import csv
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read_trajectory(path):
    """Read trajectory CSV into {time: {species: value}}."""
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            t = round(float(row['Time']))
            if t not in data:
                data[t] = {k: float(v) for k, v in row.items() if k != 'Time'}
    return data


def main():
    cpp_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/cpp_1000d.csv'
    sb_path = sys.argv[2] if len(sys.argv) > 2 else '/tmp/sb_1000d.csv'
    out_path = sys.argv[3] if len(sys.argv) > 3 else 'tests/trajectory_relerr.png'

    cpp = read_trajectory(cpp_path)
    sb_raw = read_trajectory(sb_path)

    # SimBiology uses Compartment_Species format, C++ uses Compartment.Species
    # Normalize SB keys: replace '.' with '_' so V_T.ArgI -> V_T_ArgI
    sb = {}
    for t, species in sb_raw.items():
        sb[t] = {k.replace('.', '_'): v for k, v in species.items()}

    # All V_T species to compare
    vt_species = [
        'C1', 'C_x', 'Treg', 'CD8', 'Th', 'CD8_exh', 'Th_exh',
        'Mac_M1', 'Mac_M2', 'MDSC', 'qPSC', 'iCAF', 'myCAF', 'apCAF',
        'collagen', 'VEGF', 'ArgI', 'TGFb', 'NO', 'K',
    ]

    times = sorted(set(cpp.keys()) & set(sb.keys()))

    # Compute relative errors
    errors = {}  # species -> [(t, err%)]
    for name in vt_species:
        cpp_key = f'V_T.{name}'
        sb_key = f'V_T_{name}'
        errs = []
        for t in times:
            cv = cpp[t].get(cpp_key, 0)
            sv = sb[t].get(sb_key, 0)
            if abs(sv) > 1e-20:
                errs.append((t, (cv / sv - 1) * 100))
        errors[name] = errs

    # Plot
    fig, axes = plt.subplots(5, 4, figsize=(20, 22))
    axes = axes.flatten()

    for idx, name in enumerate(vt_species):
        ax = axes[idx]
        errs = errors[name]
        if errs:
            tt, ee = zip(*errs)
            ax.plot(tt, ee, 'k-', lw=1.2)
        ax.axhline(y=0, color='gray', alpha=0.3)
        ax.axhline(y=5, color='red', ls=':', alpha=0.4)
        ax.axhline(y=-5, color='red', ls=':', alpha=0.4)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 500)
        ax.set_ylim(-30, 30)
        ax.set_ylabel('%')
        if idx >= 16:
            ax.set_xlabel('days')

    plt.suptitle(
        'Relative error (C++/SimBiology - 1) per V_T species\n'
        f'C++: {cpp_path}\nSB: {sb_path}',
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    print(f'Saved to {out_path}')


if __name__ == '__main__':
    main()
