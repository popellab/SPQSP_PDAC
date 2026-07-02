#!/usr/bin/env python3
"""PDAC parameter coverage audit."""

import sys
from pathlib import Path

from qsp_inference.audit.report import AuditConfig, run_audit

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "notes" / "calibration" / "parameter_audit_report.md"
DEFAULT_STAGE2_CACHE = ROOT / "figures" / "sbi_stage2_current"

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PDAC parameter coverage audit")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--invalidate",
        nargs="+",
        metavar="PARAM",
        help="Invalidate cached components containing these parameters",
    )
    parser.add_argument(
        "--sbi-run-path",
        type=Path,
        default=None,
        help=(
            "Stage-2 SBI run dir (enables Stage 2 section). "
            "Defaults to figures/sbi_stage2_current if present "
            "(managed by scripts/promote_sbi_stage2.py)."
        ),
    )
    args = parser.parse_args()

    sbi_run_path = args.sbi_run_path
    if sbi_run_path is None and DEFAULT_STAGE2_CACHE.exists():
        sbi_run_path = DEFAULT_STAGE2_CACHE

    config = AuditConfig(project_root=ROOT, sbi_run_path=sbi_run_path)
    report = run_audit(config, output=args.output, invalidate_params=args.invalidate)

    # Prepend Quarto front matter for dark mode toggle in VS Code preview
    quarto_header = """\
---
format:
  html:
    theme:
      light: flatly
      dark: darkly
---

"""
    output_path = args.output or DEFAULT_OUTPUT
    output_path.write_text(quarto_header + output_path.read_text())

    if not args.output:
        print(report)