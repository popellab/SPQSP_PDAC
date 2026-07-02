#!/usr/bin/env python3
"""Thin entry point for the multi-scenario SBI pipeline.

The pipeline itself lives in `workflows.sbi_runner` as `run_inference(cfg)`.
This file stays for backward compatibility — `python workflows/sbi_workflow.py`
runs the pipeline with the default RunConfig.

For sweeps, prefer `python -m workflows.sbi_runner --seed N --hidden-features H ...`
or import `RunConfig`/`run_inference` directly.
"""
import sys
from pathlib import Path

# Allow running directly as `python workflows/sbi_workflow.py` from project root.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from workflows.sbi_runner import (  # noqa: E402
    run_inference,
    _build_cli_parser,
    _cfg_from_cli,
)


if __name__ == '__main__':
    args = _build_cli_parser().parse_args()
    run_inference(_cfg_from_cli(args))