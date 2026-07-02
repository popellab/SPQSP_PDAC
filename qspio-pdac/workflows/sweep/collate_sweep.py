#!/usr/bin/env python
"""Collate per-cell sweep outputs into long-form + per-cell summary tables.

Reads:
    {sweep_root}/runs/{tag}/local_calibration.csv
    {sweep_root}/runs/{tag}/run_metrics.json
    {configs_dir}/{tag}.json

Writes:
    {sweep_root}/sweep_by_param.csv    one row per (cell, parameter)
    {sweep_root}/sweep_summary.csv     one row per cell (aggregate counts)

Per-param view — the primary deliverable — carries:
    cell_tag, hidden_features, num_transforms, n_train_subsample,
    parameter,
    r2_global, r2_local, contraction_global, contraction_local,
    ks_pval_global, ks_pval_local,
    zscore_mean_local, zscore_std_local, coverage95_local, post_sd_log_local,
    is_miscal_local (ks < 0.05), is_identified_local (contraction_local > 0.5)

Per-cell summary compares each cell to a reference cell (smallest
hidden_features × smallest num_transforms × smallest n_train_subsample by
default) to count params that gained/lost identifiability.
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def _load_cell(runs_dir: Path, configs_dir: Path, tag: str) -> pd.DataFrame | None:
    cal = runs_dir / tag / "local_calibration.csv"
    if not cal.exists():
        print(f"[collate] {tag}: SKIP (no local_calibration.csv)")
        return None
    cfg = json.loads((configs_dir / f"{tag}.json").read_text())
    df = pd.read_csv(cal)
    df.insert(0, "cell_tag", tag)
    for k in ("hidden_features", "num_transforms", "n_train_subsample",
              "num_bins", "batch_size", "max_epochs"):
        df[k] = cfg.get(k)
    return df


def _cell_metrics(runs_dir: Path, tag: str) -> dict:
    mf = runs_dir / tag / "run_metrics.json"
    if mf.exists():
        return json.loads(mf.read_text())
    return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-root", type=Path, required=True)
    ap.add_argument("--configs-dir", type=Path, required=True)
    ap.add_argument("--ref-tag", default=None,
                    help="Reference cell tag for gain/loss counts. "
                         "Default: first line of configs/_tags.txt.")
    args = ap.parse_args()

    runs_dir = args.sweep_root / "runs"
    tags = [
        line.strip() for line in (args.configs_dir / "_tags.txt").read_text().splitlines()
        if line.strip()
    ]
    if not tags:
        raise SystemExit("No tags in _tags.txt")

    ref_tag = args.ref_tag or tags[0]

    # --- Long-form per-param ---
    per_param_frames = []
    for tag in tags:
        df = _load_cell(runs_dir, args.configs_dir, tag)
        if df is None:
            continue
        per_param_frames.append(df)

    if not per_param_frames:
        raise SystemExit("No cells had local_calibration.csv — nothing to collate.")

    by_param = pd.concat(per_param_frames, ignore_index=True)
    by_param["is_miscal_local"] = by_param["ks_pval_local"] < 0.05
    by_param["is_identified_local"] = by_param["contraction_local"] > 0.5

    out_param = args.sweep_root / "sweep_by_param.csv"
    by_param.to_csv(out_param, index=False)
    print(f"[collate] wrote {out_param}  ({len(by_param)} rows, "
          f"{by_param['cell_tag'].nunique()} cells)")

    # --- Per-cell summary ---
    if ref_tag not in set(by_param["cell_tag"]):
        print(f"[collate] WARNING: ref-tag {ref_tag} has no rows; gain/loss "
              f"columns will be omitted")
        ref_df = None
    else:
        ref_df = by_param[by_param["cell_tag"] == ref_tag].set_index("parameter")

    summary_rows = []
    for tag in tags:
        cell = by_param[by_param["cell_tag"] == tag]
        if cell.empty:
            continue
        row = {
            "cell_tag": tag,
            "hidden_features": cell["hidden_features"].iloc[0],
            "num_transforms": cell["num_transforms"].iloc[0],
            "n_train_subsample": cell["n_train_subsample"].iloc[0],
            "n_params": len(cell),
            "n_identified_local": int(cell["is_identified_local"].sum()),
            "n_calibrated_local": int((~cell["is_miscal_local"]).sum()),
            "median_contraction_local": float(cell["contraction_local"].median()),
            "median_coverage95_local": float(cell["coverage95_local"].median()),
            "median_zscore_std_local": float(cell["zscore_std_local"].median()),
            "median_post_sd_log_local": float(cell["post_sd_log_local"].median()),
        }
        metrics = _cell_metrics(runs_dir, tag)
        for k in ("n_epochs", "best_val_loss", "n_nn_params"):
            if k in metrics:
                row[k] = metrics[k]

        if ref_df is not None and tag != ref_tag:
            c = cell.set_index("parameter")
            shared = c.index.intersection(ref_df.index)
            c_s = c.loc[shared]
            r_s = ref_df.loc[shared]
            row["n_newly_identified_vs_ref"] = int(
                ((c_s["contraction_local"] > 0.5) & (r_s["contraction_local"] <= 0.5)).sum()
            )
            row["n_lost_identified_vs_ref"] = int(
                ((c_s["contraction_local"] <= 0.5) & (r_s["contraction_local"] > 0.5)).sum()
            )
            row["n_newly_calibrated_vs_ref"] = int(
                ((c_s["ks_pval_local"] >= 0.05) & (r_s["ks_pval_local"] < 0.05)).sum()
            )
            row["n_broke_calibration_vs_ref"] = int(
                ((c_s["ks_pval_local"] < 0.05) & (r_s["ks_pval_local"] >= 0.05)).sum()
            )
            row["n_contraction_gain_0p1_vs_ref"] = int(
                (c_s["contraction_local"] - r_s["contraction_local"] > 0.1).sum()
            )
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    # Sort by identified count (primary), then calibration quality
    summary = summary.sort_values(
        by=["n_identified_local", "n_calibrated_local", "median_contraction_local"],
        ascending=[False, False, False],
    )
    out_summary = args.sweep_root / "sweep_summary.csv"
    summary.to_csv(out_summary, index=False)
    print(f"[collate] wrote {out_summary}  ({len(summary)} cells, ref={ref_tag})")
    print()
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
