#!/usr/bin/env python3
"""Run all PDAC sync guards and exit nonzero on any failure.

Use before running a C++ simulation to confirm:
- SBML is up to date with the live MATLAB model script
- The generated C++ ODE is up to date with the SBML
- The qsp_sim binary is up to date with the codegen
- param_all.xml contains every name the codegen expects

Usage:
    python PDAC/codegen/check_sync.py
"""
from __future__ import annotations

import sys

from sync_checks import run_all


def main() -> int:
    results = run_all()
    n_fail = 0
    for name, ok, msg in results:
        marker = "OK  " if ok else "FAIL"
        print(f"[{marker}] {name}: {msg}")
        if not ok:
            n_fail += 1
    if n_fail:
        print(f"\n{n_fail}/{len(results)} sync check(s) failed.")
        return 1
    print(f"\nAll {len(results)} sync checks passed.")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    sys.exit(main())