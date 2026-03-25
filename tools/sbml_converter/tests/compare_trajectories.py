#!/usr/bin/env python3
"""Compare MATLAB and C++ ODE trajectories.

Usage:
    python compare_trajectories.py <matlab_csv> <cpp_csv> [--rtol 1e-3] [--atol 1e-9]
"""

import argparse
import sys

import numpy as np


def load_csv(path):
    """Load trajectory CSV, return (times, data, names)."""
    with open(path) as f:
        header = f.readline().strip().split(",")
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    times = data[:, 0]
    values = data[:, 1:]
    names = header[1:]
    return times, values, names


def compare(matlab_csv, cpp_csv, rtol=1e-3, atol=1e-9):
    """Compare two trajectory files. Returns (pass, report_string)."""
    t_m, v_m, names_m = load_csv(matlab_csv)
    t_c, v_c, names_c = load_csv(cpp_csv)

    lines = []
    lines.append(f"MATLAB: {len(t_m)} time points, {len(names_m)} species")
    lines.append(f"C++:    {len(t_c)} time points, {len(names_c)} species")

    # Match species by name
    # MATLAB outputs "V_T_CD8" (underscores), C++ outputs "V_T.CD8" (dots)
    # Normalize both to underscores for matching
    def normalize(name):
        return name.replace(".", "_")

    common = []
    m_norm = {normalize(n): (n, i) for i, n in enumerate(names_m)}
    c_idx = {n: i for i, n in enumerate(names_c)}

    for name in names_c:
        key = normalize(name)
        if key in m_norm:
            m_name, mi = m_norm[key]
            common.append((name, mi, c_idx[name]))

    lines.append(f"Matched species: {len(common)} / {len(names_c)}")

    if len(common) == 0:
        lines.append("ERROR: No species matched between files!")
        return False, "\n".join(lines)

    # Interpolate to common time points
    t_common = np.intersect1d(np.round(t_m, 6), np.round(t_c, 6))
    if len(t_common) == 0:
        # Fall back: use C++ time points, interpolate MATLAB
        t_common = t_c
        lines.append(f"No exact time matches; interpolating MATLAB to C++ times ({len(t_common)} points)")

    # Compare at common times
    n_fail = 0
    max_rdiff = 0.0
    worst_species = ""
    worst_time = 0.0

    for name, mi, ci in common:
        for ti, t in enumerate(t_common):
            # Find closest time index in each
            mi_t = np.argmin(np.abs(t_m - t))
            ci_t = np.argmin(np.abs(t_c - t))

            vm = v_m[mi_t, mi]
            vc = v_c[ci_t, ci]

            # Relative or absolute comparison
            denom = max(abs(vm), abs(vc), atol)
            rdiff = abs(vm - vc) / denom

            if rdiff > max_rdiff:
                max_rdiff = rdiff
                worst_species = name
                worst_time = t

            if rdiff > rtol and abs(vm - vc) > atol:
                n_fail += 1
                if n_fail <= 10:
                    lines.append(f"  FAIL: {name} at t={t:.2f}: MATLAB={vm:.6e}, C++={vc:.6e}, rdiff={rdiff:.2e}")

    total_comparisons = len(common) * len(t_common)
    lines.append(f"Comparisons: {total_comparisons}")
    lines.append(f"Failures: {n_fail} (rtol={rtol}, atol={atol})")
    lines.append(f"Worst relative diff: {max_rdiff:.2e} ({worst_species} at t={worst_time:.2f})")

    passed = n_fail == 0
    lines.append(f"Result: {'PASS' if passed else 'FAIL'}")

    return passed, "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare MATLAB vs C++ ODE trajectories")
    parser.add_argument("matlab_csv", help="MATLAB trajectory CSV")
    parser.add_argument("cpp_csv", help="C++ trajectory CSV")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance")
    parser.add_argument("--atol", type=float, default=1e-9, help="Absolute tolerance")
    args = parser.parse_args()

    passed, report = compare(args.matlab_csv, args.cpp_csv, args.rtol, args.atol)
    print(report)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()