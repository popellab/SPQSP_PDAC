"""Analyze and visualize Morris SA results for ABM phi observables.

Reads the consolidated observables and Morris results from run_abm_sa.py,
generates ranked bar plots (mu* vs sigma) and heatmaps of top influential
parameters per phi pair.

Usage::

    python -I scripts/analyze_abm_sa.py \
        --sa-dir results/sa_morris \
        --top-k 20
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sa-dir", required=True)
    p.add_argument("--top-k", type=int, default=20,
                   help="Number of top params to show per plot (default 20)")
    args = p.parse_args(argv)

    sa_dir = Path(args.sa_dir)
    results_dir = sa_dir / "results"
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    morris_csv = results_dir / "morris_results.csv"
    if not morris_csv.exists():
        log.error(f"Morris results not found: {morris_csv}")
        log.error("Run `run_abm_sa.py consolidate` first")
        return

    df = pd.read_csv(morris_csv)
    log.info(f"Loaded {len(df)} Morris results "
             f"({df['observable'].nunique()} observables × "
             f"{df['parameter'].nunique()} parameters)")

    obs_data = np.load(sa_dir / "observables.npz", allow_pickle=True)
    sa_categories = list(obs_data["sa_categories"])
    n_cat = len(sa_categories)

    # ── 1. Aggregate mu* across all phi pairs → overall parameter ranking ──
    agg = (df.groupby("parameter")
             .agg(mu_star_mean=("mu_star", "mean"),
                  mu_star_max=("mu_star", "max"),
                  sigma_mean=("sigma", "mean"),
                  n_obs=("observable", "count"))
             .sort_values("mu_star_mean", ascending=False))

    agg.to_csv(results_dir / "parameter_ranking.csv")
    log.info(f"Parameter ranking saved ({len(agg)} params)")

    # Bar plot: top-K by mean mu*
    top = agg.head(args.top_k)
    fig, ax = plt.subplots(figsize=(10, max(6, args.top_k * 0.35)))
    y_pos = np.arange(len(top))
    ax.barh(y_pos, top["mu_star_mean"], color="#2196F3", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean µ* (across all φ pairs)")
    ax.set_title(f"Top {args.top_k} ABM Parameters by Morris µ*")
    fig.tight_layout()
    fig.savefig(fig_dir / "top_params_mu_star.png", dpi=150)
    plt.close(fig)
    log.info(f"  Saved top_params_mu_star.png")

    # ── 2. mu* vs sigma scatter (screening plot) ──
    fig, ax = plt.subplots(figsize=(8, 8))
    top_names = set(top.index[:10])
    for _, row in agg.iterrows():
        c = "#E53935" if row.name in top_names else "#90A4AE"
        ax.scatter(row["mu_star_mean"], row["sigma_mean"],
                   c=c, s=30, alpha=0.7, edgecolors="none")
        if row.name in top_names:
            ax.annotate(row.name, (row["mu_star_mean"], row["sigma_mean"]),
                        fontsize=6, alpha=0.8, xytext=(4, 4),
                        textcoords="offset points")
    ax.set_xlabel("µ* (mean effect)")
    ax.set_ylabel("σ (interaction / nonlinearity)")
    ax.plot([0, ax.get_xlim()[1]], [0, ax.get_xlim()[1]],
            "k--", alpha=0.3, label="σ = µ*")
    ax.legend(fontsize=8)
    ax.set_title("Morris Screening: µ* vs σ")
    fig.tight_layout()
    fig.savefig(fig_dir / "morris_screening.png", dpi=150)
    plt.close(fig)
    log.info(f"  Saved morris_screening.png")

    # ── 3. Heatmap: mu* per (parameter, phi-pair) for top params ──
    top_params = list(agg.head(args.top_k).index)
    diag_obs = [f"dphi_{c}__{c}" for c in sa_categories]

    # Diagonal (self-clustering) heatmap
    diag_df = df[df["observable"].isin(diag_obs) & df["parameter"].isin(top_params)]
    if not diag_df.empty:
        pivot = diag_df.pivot_table(
            index="parameter", columns="observable", values="mu_star")
        pivot = pivot.reindex(index=top_params)
        pivot.columns = [c.replace("dphi_", "").replace("__", " ↔ ") for c in pivot.columns]

        fig, ax = plt.subplots(figsize=(max(8, n_cat * 0.8), max(6, args.top_k * 0.35)))
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns, fontsize=7, rotation=45, ha="right")
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels(pivot.index, fontsize=7)
        fig.colorbar(im, ax=ax, label="µ*", shrink=0.8)
        ax.set_title("Morris µ* — Self-Clustering φ(A,A)")
        fig.tight_layout()
        fig.savefig(fig_dir / "heatmap_diag_mu_star.png", dpi=150)
        plt.close(fig)
        log.info(f"  Saved heatmap_diag_mu_star.png")

    # ── 4. Per-category pair plots for top-3 most sensitive pairs ──
    pair_sensitivity = (df.groupby("observable")["mu_star"]
                          .mean()
                          .sort_values(ascending=False))

    for obs_name in pair_sensitivity.head(6).index:
        sub = df[df["observable"] == obs_name].sort_values("mu_star", ascending=False)
        top_sub = sub.head(args.top_k)

        fig, ax = plt.subplots(figsize=(8, max(4, args.top_k * 0.3)))
        y_pos = np.arange(len(top_sub))
        ax.barh(y_pos, top_sub["mu_star"].values, xerr=top_sub["mu_star_conf"].values,
                color="#4CAF50", alpha=0.85, capsize=2)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_sub["parameter"].values, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("µ*")
        pair_label = obs_name.replace("dphi_", "Δφ(").replace("__", ", ") + ")"
        ax.set_title(f"{pair_label}")
        fig.tight_layout()
        safe_name = obs_name.replace("__", "_v_")
        fig.savefig(fig_dir / f"bar_{safe_name}.png", dpi=150)
        plt.close(fig)

    log.info(f"  Saved per-pair bar plots")

    # ── 5. Population drift summary ──
    pop_avg = obs_data["pop_avg"]
    status = obs_data["status"]
    if pop_avg.shape[0] > 0:
        valid_mask = np.any(~np.isnan(pop_avg), axis=1)
        if valid_mask.any():
            pop_mean = np.nanmean(pop_avg[valid_mask], axis=0)
            pop_std = np.nanstd(pop_avg[valid_mask], axis=0)
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(n_cat)
            ax.bar(x, pop_mean, yerr=pop_std, capsize=3, color="#78909C", alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(sa_categories, fontsize=7, rotation=45, ha="right")
            ax.set_ylabel("Mean cell count (across SA sets)")
            ax.set_title("Population Composition at ABM Endpoint")
            fig.tight_layout()
            fig.savefig(fig_dir / "population_drift.png", dpi=150)
            plt.close(fig)
            log.info(f"  Saved population_drift.png")

    # ── Summary ──
    log.info("=" * 60)
    log.info("ANALYSIS COMPLETE")
    log.info(f"  Figures: {fig_dir}")
    log.info(f"  Parameter ranking: {results_dir / 'parameter_ranking.csv'}")
    log.info(f"  Top 5 parameters (by mean µ*):")
    for i, (name, row) in enumerate(agg.head(5).iterrows()):
        log.info(f"    {i+1}. {name}: µ*={row['mu_star_mean']:.4f}, "
                 f"σ={row['sigma_mean']:.4f}")


if __name__ == "__main__":
    main()
