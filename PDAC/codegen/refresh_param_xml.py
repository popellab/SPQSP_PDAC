#!/usr/bin/env python3
"""Refresh the <QSP>...</QSP> block of param_all.xml from qsp_params_xml_snippet.xml.

The snippet is regenerated alongside the C++ ODE every time qsp_codegen.py
runs. Without this merge, param renames in the SBML cause silent default-to-0
fallbacks in QSPParam.cpp, producing diverged trajectories.

Usage:
    python PDAC/codegen/refresh_param_xml.py
    python PDAC/codegen/refresh_param_xml.py --xml PDAC/sim/resource/param_all_test.xml
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SNIPPET = REPO_ROOT / "PDAC" / "qsp" / "ode" / "qsp_params_xml_snippet.xml"
DEFAULT_TARGETS = [
    REPO_ROOT / "PDAC" / "sim" / "resource" / "param_all.xml",
    REPO_ROOT / "PDAC" / "sim" / "resource" / "param_all_test.xml",
]

QSP_BLOCK = re.compile(r"<QSP>.*?</QSP>", re.DOTALL)


def refresh(target: Path, snippet: Path) -> bool:
    """Replace the <QSP>...</QSP> block in target with snippet contents.

    Returns True on success, False if no <QSP> block was found.
    """
    new_block = snippet.read_text().rstrip()
    text = target.read_text()
    new_text, n = QSP_BLOCK.subn(new_block.replace("\\", r"\\"), text, count=1)
    if n == 0:
        return False
    if new_text == text:
        print(f"  {target.name}: already up to date")
        return True
    target.write_text(new_text)
    print(f"  {target.name}: refreshed <QSP> block ({len(new_block)} bytes)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snippet", type=Path, default=DEFAULT_SNIPPET)
    parser.add_argument(
        "--xml",
        type=Path,
        action="append",
        help="param_all.xml path (repeatable). Defaults to param_all{,_test}.xml.",
    )
    args = parser.parse_args()

    targets = args.xml or DEFAULT_TARGETS
    if not args.snippet.exists():
        print(f"ERROR: snippet not found: {args.snippet}", file=sys.stderr)
        return 1

    n_fail = 0
    for t in targets:
        if not t.exists():
            print(f"  {t}: not found, skipping")
            continue
        if not refresh(t, args.snippet):
            print(f"ERROR: no <QSP>...</QSP> block found in {t}", file=sys.stderr)
            n_fail += 1
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())