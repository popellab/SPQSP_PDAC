#!/usr/bin/env python3
"""Walk a finished sweep and aggregate diagnostics into a tidy long-form CSV.

Usage:
    python workflows/sweep_collect.py sweep_runs/seed_flow_nsims_27cell
    python workflows/sweep_collect.py sweep_runs/... --out results.csv

Output schema (one row per parameter per run):
    sweep_name, run_id, job_hash, seed, prcc_tier, hidden_features,
    num_transforms, num_total_simulations, n_train_subsample, best_val_loss,
    n_epochs, n_nn_params, n_train, parameter, r2_global, r2_local,
    contraction_global, contraction_local, ks_pval_global, ks_pval_local,
    sbc_z_abs_mean, sbc_z_mean

Also emits a per-run summary CSV alongside with one row per run (aggregates
across parameters: median/mean contraction, shell-count thresholds).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def collect_long(sweep_dir: Path) -> pd.DataFrame:
    runs_dir = sweep_dir / "runs"
    rows: list[dict] = []
    for run_dir in sorted(runs_dir.iterdir()) if runs_dir.exists() else []:
        if not run_dir.is_dir():
            continue
        metrics = _load_json(run_dir / "run_metrics.json")
        if not metrics:
            print(f"[skip] {run_dir.name}: no run_metrics.json")
            continue
        cfg = _load_json(run_dir / "run_config.json") or {}

        cfg_cols = {
            "seed": cfg.get("seed"),
            "prcc_tier": cfg.get("prcc_tier"),
            "hidden_features": cfg.get("npe_hidden_features"),
            "num_transforms": cfg.get("npe_num_transforms"),
            "num_total_simulations": cfg.get("num_total_simulations"),
            "n_train_subsample": cfg.get("n_train_subsample"),
        }
        run_cols = {
            "run_id": run_dir.name,
            "job_hash": run_dir.name,
            "best_val_loss": metrics.get("best_val_loss"),
            "n_epochs": metrics.get("n_epochs"),
            "n_nn_params": metrics.get("n_nn_params"),
            "n_train": metrics.get("n_train"),
        }

        local_cal_path = run_dir / "local_calibration.csv"
        zsc_path = run_dir / "diagnostics" / "z_score_contraction.csv"

        if not local_cal_path.exists():
            print(f"[skip] {run_dir.name}: no local_calibration.csv")
            continue

        lc = pd.read_csv(local_cal_path)
        if zsc_path.exists():
            zsc = pd.read_csv(zsc_path)[
                ["parameter", "z_score_abs_mean", "z_score_mean"]
            ]
            lc = lc.merge(zsc, on="parameter", how="left")
        else:
            lc["z_score_abs_mean"] = None
            lc["z_score_mean"] = None

        for _, r in lc.iterrows():
            rows.append(
                {
                    **cfg_cols,
                    **run_cols,
                    "parameter": r["parameter"],
                    "r2_global": r.get("r2_global"),
                    "r2_local": r.get("r2_local"),
                    "contraction_global": r.get("contraction_global"),
                    "contraction_local": r.get("contraction_local"),
                    "ks_pval_global": r.get("ks_pval_global"),
                    "ks_pval_local": r.get("ks_pval_local"),
                    "sbc_z_abs_mean": r.get("z_score_abs_mean"),
                    "sbc_z_mean": r.get("z_score_mean"),
                }
            )
    return pd.DataFrame(rows)


def per_run_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return long_df

    cfg_cols = [
        "run_id",
        "seed",
        "prcc_tier",
        "hidden_features",
        "num_transforms",
        "num_total_simulations",
        "n_train_subsample",
        "best_val_loss",
        "n_epochs",
        "n_nn_params",
        "n_train",
    ]

    def _agg(g: pd.DataFrame) -> pd.Series:
        shell = g["contraction_local"]
        return pd.Series(
            {
                "n_params": len(g),
                "median_r2_global": g["r2_global"].median(),
                "median_r2_local": g["r2_local"].median(),
                "median_contr_global": g["contraction_global"].median(),
                "median_contr_local": shell.median(),
                "shell_ge50": int((shell >= 0.5).sum()),
                "shell_lt10": int((shell < 0.1).sum()),
                "shell_lt0": int((shell < 0.0).sum()),
                "ks_local_lt05": int((g["ks_pval_local"] < 0.05).sum()),
                "sbc_z_abs_gt1": int((g["sbc_z_abs_mean"] > 1.0).sum()),
            }
        )

    summary = (
        long_df.groupby(cfg_cols, dropna=False)
        .apply(_agg, include_groups=False)
        .reset_index()
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate sweep outputs")
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument(
        "--out-long",
        type=Path,
        default=None,
        help="Long-form per-parameter CSV (default: <sweep_dir>/sweep_results_long.csv)",
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=None,
        help="Per-run summary CSV (default: <sweep_dir>/sweep_results_summary.csv)",
    )
    args = parser.parse_args()

    long_df = collect_long(args.sweep_dir)
    out_long = args.out_long or args.sweep_dir / "sweep_results_long.csv"
    long_df.to_csv(out_long, index=False)
    print(f"Wrote {len(long_df)} rows to {out_long}")

    summary = per_run_summary(long_df)
    out_summary = args.out_summary or args.sweep_dir / "sweep_results_summary.csv"
    summary.to_csv(out_summary, index=False)
    print(f"Wrote {len(summary)} runs to {out_summary}")


if __name__ == "__main__":
    main()