#!/usr/bin/env python3
"""Promote an SBI run as the current stage-2 reference for `parameter_audit.py`.

Creates a symlink at ``figures/sbi_stage2_current`` → <chosen run dir> and
writes ``figures/sbi_stage2_current.json`` with provenance + selector stats.

Two entry modes:

* ``promote_sbi_stage2.py <run_dir>`` — promote a single run directory.
* ``promote_sbi_stage2.py --scan <parent_dir>`` — scan subdirectories for
  valid runs and auto-pick the best by the configured selector.

A "valid run directory" has ``posterior_samples.csv``,
``local_calibration.csv``, and ``diagnostics/summary.json`` (the inputs that
``qsp_inference.audit.report.load_sbi_run`` reads).

Default selector: maximise ``n_identified_local`` (count of parameters with
``contraction_local ≥ 0.5`` in local_calibration.csv), tie-broken by lower
``best_val_loss``.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE_LINK = ROOT / "figures" / "sbi_stage2_current"
CACHE_REGISTRY = ROOT / "figures" / "sbi_stage2_current.json"

REQUIRED_FILES = (
    "posterior_samples.csv",
    "local_calibration.csv",
    "diagnostics/summary.json",
)

IDENTIFIED_CONTRACTION_THRESHOLD = 0.5


@dataclass
class RunStats:
    path: Path
    tag: str
    n_identified_local: int
    best_val_loss: float | None
    n_params: int | None
    n_train: int | None
    scenarios: list[str]

    def as_dict(self) -> dict:
        return {
            "path": str(self.path),
            "tag": self.tag,
            "n_identified_local": self.n_identified_local,
            "best_val_loss": self.best_val_loss,
            "n_params": self.n_params,
            "n_train": self.n_train,
            "scenarios": self.scenarios,
        }


def _is_valid_run_dir(p: Path) -> bool:
    return all((p / rel).exists() for rel in REQUIRED_FILES)


def _read_stats(run_dir: Path) -> RunStats:
    local = pd.read_csv(run_dir / "local_calibration.csv")
    n_identified = int((local["contraction_local"] >= IDENTIFIED_CONTRACTION_THRESHOLD).sum())

    summary = json.loads((run_dir / "diagnostics" / "summary.json").read_text())
    training = summary.get("training", {})

    return RunStats(
        path=run_dir.resolve(),
        tag=run_dir.name,
        n_identified_local=n_identified,
        best_val_loss=training.get("best_val_loss"),
        n_params=training.get("n_params"),
        n_train=training.get("n_train"),
        scenarios=list(training.get("scenarios", [])),
    )


def _scan(parent: Path) -> list[RunStats]:
    runs: list[RunStats] = []
    for sub in sorted(parent.iterdir()):
        if not sub.is_dir():
            continue
        if not _is_valid_run_dir(sub):
            continue
        try:
            runs.append(_read_stats(sub))
        except Exception as exc:
            print(f"[skip] {sub.name}: {exc}", file=sys.stderr)
    return runs


def _select_best(runs: list[RunStats]) -> RunStats:
    def key(r: RunStats):
        # Higher n_identified_local wins; tie-break by lower best_val_loss
        # (fall back to +inf so None sorts worst).
        return (
            -r.n_identified_local,
            r.best_val_loss if r.best_val_loss is not None else float("inf"),
        )

    return sorted(runs, key=key)[0]


def _print_ranking(runs: list[RunStats], best: RunStats) -> None:
    ordered = sorted(
        runs,
        key=lambda r: (
            -r.n_identified_local,
            r.best_val_loss if r.best_val_loss is not None else float("inf"),
        ),
    )
    print(f"{'tag':<22} {'n_id_local':>10} {'best_val':>10} {'n_params':>8} {'n_train':>8}")
    print("-" * 62)
    for r in ordered:
        marker = " *" if r.path == best.path else "  "
        bv = f"{r.best_val_loss:.2f}" if r.best_val_loss is not None else "n/a"
        print(
            f"{r.tag:<22} {r.n_identified_local:>10} {bv:>10} "
            f"{(r.n_params or 0):>8} {(r.n_train or 0):>8}{marker}"
        )


def _promote(run: RunStats) -> None:
    CACHE_LINK.parent.mkdir(parents=True, exist_ok=True)

    # Replace any existing symlink or directory at the cache path.
    if CACHE_LINK.is_symlink() or CACHE_LINK.exists():
        if CACHE_LINK.is_symlink():
            CACHE_LINK.unlink()
        else:
            raise SystemExit(
                f"{CACHE_LINK} exists and is not a symlink — refusing to replace. "
                "Move or delete it manually."
            )

    CACHE_LINK.symlink_to(run.path, target_is_directory=True)

    registry = {
        "promoted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selector": {
            "primary": "n_identified_local (contraction_local ≥ 0.5)",
            "tiebreaker": "best_val_loss (lower wins)",
        },
        "run": run.as_dict(),
    }
    CACHE_REGISTRY.write_text(json.dumps(registry, indent=2) + "\n")

    print(f"[promoted] {CACHE_LINK} → {run.path}")
    print(f"[registry] {CACHE_REGISTRY}")


def _show() -> int:
    if not CACHE_LINK.exists():
        print("No stage-2 run promoted yet.")
        return 1
    if CACHE_LINK.is_symlink():
        print(f"symlink: {CACHE_LINK} → {CACHE_LINK.resolve()}")
    else:
        print(f"path:    {CACHE_LINK} (not a symlink)")
    if CACHE_REGISTRY.exists():
        print(CACHE_REGISTRY.read_text())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        help="Single run directory to promote (ignored if --scan is given).",
    )
    ap.add_argument(
        "--scan",
        type=Path,
        metavar="PARENT",
        help="Scan subdirectories for valid runs and auto-pick the best.",
    )
    ap.add_argument(
        "--show",
        action="store_true",
        help="Print the currently promoted run and exit.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print ranking / selection but do not update the cache.",
    )
    args = ap.parse_args()

    if args.show:
        return _show()

    if args.scan is not None:
        parent = args.scan.resolve()
        if not parent.is_dir():
            raise SystemExit(f"--scan target does not exist: {parent}")
        runs = _scan(parent)
        if not runs:
            raise SystemExit(f"No valid run directories found under {parent}.")
        best = _select_best(runs)
        _print_ranking(runs, best)
        if args.dry_run:
            print("[dry-run] no promotion")
            return 0
        _promote(best)
        return 0

    if args.run_dir is None:
        ap.print_usage()
        print("\nProvide either <run_dir> or --scan <parent>.", file=sys.stderr)
        return 2

    run_dir = args.run_dir.resolve()
    if not _is_valid_run_dir(run_dir):
        missing = [rel for rel in REQUIRED_FILES if not (run_dir / rel).exists()]
        raise SystemExit(f"{run_dir} is missing: {', '.join(missing)}")
    stats = _read_stats(run_dir)
    print(f"[candidate] {stats.as_dict()}")
    if args.dry_run:
        print("[dry-run] no promotion")
        return 0
    _promote(stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
