#!/usr/bin/env python3
"""Single-job entry point for SBI sweep cells.

Usage:
    python -m workflows.sweep_run_one path/to/config.json

The config JSON is a RunConfig field dict (produced by sweep_submit.py).
"""
import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from workflows.sbi_runner import RunConfig, run_inference  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single SBI sweep cell")
    parser.add_argument("config_json", type=Path)
    args = parser.parse_args()

    data = json.loads(args.config_json.read_text())
    if data.get("save_dir"):
        data["save_dir"] = Path(data["save_dir"])
    cfg = RunConfig(**data)

    print(f"[sweep_run_one] cfg={args.config_json}")
    print(f"[sweep_run_one] save_dir={cfg.save_dir}")
    metrics = run_inference(cfg)
    print(
        f"[sweep_run_one] DONE best_val_loss={metrics['best_val_loss']:.3f} "
        f"n_train={metrics['n_train']} n_epochs={metrics['n_epochs']}"
    )


if __name__ == "__main__":
    main()