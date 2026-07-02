#!/usr/bin/env python3
"""Compare simulations before vs after submodel target updates.

Runs N samples from the old (last committed) and new (working copy) priors
through clinical_progression, then plots spider + prior shift comparisons.

Simulation results are cached in cache/prior_comparison/ so that re-runs
(e.g. to fix plotting) don't re-run MATLAB.

Usage:
    python scripts/compare_prior_update.py
    python scripts/compare_prior_update.py --n-samples 10 --git-ref HEAD~3
    python scripts/compare_prior_update.py --replot  # reuse cached sims
"""

import argparse
import hashlib
import json
import subprocess
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import lognorm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PRIORS = PROJECT_ROOT / "parameters" / "pdac_priors.csv"
DEFAULT_SCENARIO = PROJECT_ROOT / "scenarios" / "clinical_progression.yaml"
OUT_DIR = PROJECT_ROOT / "figures" / "prior_comparison"
CACHE_DIR = PROJECT_ROOT / "cache" / "prior_comparison"

OBSERVABLES = [
    {"name": "Tumor Volume",   "species": ["V_T"],                       "transform": "identity", "log": True,  "units": "mL"},
    {"name": "CD8 (Tumor)",    "species": ["V_T.CD8"],                   "transform": "identity", "log": True,  "units": "cell/mL"},
    {"name": "Treg (Tumor)",   "species": ["V_T.Treg"],                  "transform": "identity", "log": True,  "units": "cell/mL"},
    {"name": "CD8/Treg Ratio", "species": ["V_T.CD8", "V_T.Treg"],      "transform": "ratio",    "log": True,  "units": ""},
    {"name": "ECM",            "species": ["V_T.ECM"],                   "transform": "identity", "log": True,  "units": "mg"},
    {"name": "mAPC (LN)",      "species": ["V_LN.mcDC1", "V_LN.mcDC2"], "transform": "sum",      "log": True,  "units": "cell/mL"},
]


# ---------------------------------------------------------------------------
# Prior loading
# ---------------------------------------------------------------------------

def load_priors_from_git(git_ref: str, priors_path: Path) -> pd.DataFrame:
    """Retrieve priors CSV from a git ref, following renames."""
    rel = str(priors_path.relative_to(PROJECT_ROOT))
    # Try current path first, then known previous locations
    candidates = [rel]
    if rel.startswith("parameters/"):
        candidates.append(rel.replace("parameters/", "scenarios/", 1))
    elif rel.startswith("scenarios/"):
        candidates.append(rel.replace("scenarios/", "parameters/", 1))

    for path in candidates:
        result = subprocess.run(
            ["git", "show", f"{git_ref}:{path}"],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        if result.returncode == 0:
            print(f"  (resolved to {git_ref}:{path})")
            return pd.read_csv(StringIO(result.stdout))
    raise RuntimeError(
        f"Could not find priors at {git_ref} in any of: {candidates}"
    )


def find_changed_params(old: pd.DataFrame, new: pd.DataFrame) -> list[dict]:
    """Identify parameters whose priors differ (>5% median shift or >0.01 sigma change)."""
    changes = []
    for _, row in new.iterrows():
        name = row["name"]
        old_row = old[old["name"] == name]
        if old_row.empty:
            changes.append({
                "name": name,
                "old_median": np.nan, "new_median": np.exp(row["dist_param1"]),
                "old_sigma": np.nan, "new_sigma": row["dist_param2"],
            })
            continue
        old_mu = old_row["dist_param1"].values[0]
        old_sig = old_row["dist_param2"].values[0]
        new_mu = row["dist_param1"]
        new_sig = row["dist_param2"]
        if abs(new_mu - old_mu) > 0.05 or abs(new_sig - old_sig) > 0.01:
            changes.append({
                "name": name,
                "old_median": np.exp(old_mu), "new_median": np.exp(new_mu),
                "old_sigma": old_sig, "new_sigma": new_sig,
            })
    return changes


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_from_priors(priors: pd.DataFrame, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n samples from lognormal priors. Returns (n, n_params)."""
    theta = np.zeros((n, len(priors)))
    for i, (_, row) in enumerate(priors.iterrows()):
        mu, sigma = row["dist_param1"], row["dist_param2"]
        theta[:, i] = np.exp(mu + sigma * rng.standard_normal(n))
        lb = row.get("lower_bound", np.nan)
        ub = row.get("upper_bound", np.nan)
        if pd.notna(lb):
            theta[:, i] = np.maximum(theta[:, i], lb)
        if pd.notna(ub):
            theta[:, i] = np.minimum(theta[:, i], ub)
    return theta


def build_controlled_theta(
    old_priors: pd.DataFrame,
    new_priors: pd.DataFrame,
    changed: list[dict],
    n: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample theta so unchanged params are identical between conditions."""
    rng_shared = np.random.default_rng(seed)
    theta_new = sample_from_priors(new_priors, n, rng_shared)

    # Start from new draws, overwrite only changed columns with old-prior draws
    theta_old = theta_new.copy()
    rng_changed = np.random.default_rng(seed + 1)

    for ch in changed:
        cname = ch["name"]
        new_idx = new_priors.index[new_priors["name"] == cname]
        old_row = old_priors[old_priors["name"] == cname]
        if new_idx.empty or old_row.empty:
            continue
        ni = new_idx[0]
        mu = old_row["dist_param1"].values[0]
        sigma = old_row["dist_param2"].values[0]
        theta_old[:, ni] = np.exp(mu + sigma * rng_changed.standard_normal(n))
        lb = old_row["lower_bound"].values[0]
        ub = old_row["upper_bound"].values[0]
        if pd.notna(lb):
            theta_old[:, ni] = np.maximum(theta_old[:, ni], lb)
        if pd.notna(ub):
            theta_old[:, ni] = np.minimum(theta_old[:, ni], ub)

    return theta_old, theta_new


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def _cache_key(theta: np.ndarray, scenario_yaml: Path, label: str) -> str:
    """Deterministic hash for a simulation run."""
    h = hashlib.sha256()
    h.update(theta.tobytes())
    h.update(scenario_yaml.read_bytes())
    h.update(label.encode())
    return h.hexdigest()[:12]


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.parquet"


# ---------------------------------------------------------------------------
# Simulation (delegates to qsp-hpc batch_runner)
# ---------------------------------------------------------------------------

def run_simulations(
    theta: np.ndarray,
    priors: pd.DataFrame,
    scenario_yaml: Path,
    label: str,
    matlab_path: str = "/Applications/MATLAB_R2024b.app/bin/matlab",
) -> pd.DataFrame:
    """Run batch simulations, caching results for reuse."""
    import yaml
    from qsp_hpc.simulation.batch_runner import run_batch_worker

    # Check cache
    key = _cache_key(theta, scenario_yaml, label)
    cached = _cache_path(key)
    if cached.exists():
        print(f"  {label}: using cached results ({cached.name})")
        return pd.read_parquet(cached)

    with open(scenario_yaml) as f:
        scenario = yaml.safe_load(f)

    sim_config = scenario.get("sim_config", {})
    dosing = scenario.get("dosing", None)

    # Use a persistent pool dir so parquet survives
    pool_path = CACHE_DIR / "pools"
    pool_path.mkdir(parents=True, exist_ok=True)

    print(f"  Launching MATLAB for {label} ({theta.shape[0]} sims)...")
    parquet_path = run_batch_worker(
        params=theta,
        param_names=priors["name"].tolist(),
        model_script="immune_oncology_model_PDAC",
        project_root=PROJECT_ROOT,
        seed=42,
        dosing=dosing,
        sim_config=sim_config,
        matlab_path=matlab_path,
        simulation_pool_path=pool_path,
        simulation_pool_id=label,
        verbose=True,
    )
    print(f"  {label} done -> {parquet_path}")

    # Copy to cache with content-addressed name
    df = pd.read_parquet(parquet_path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cached, engine="pyarrow", compression="snappy")
    print(f"  Cached as {cached.name}")
    return df


# ---------------------------------------------------------------------------
# Observable extraction
# ---------------------------------------------------------------------------

def extract_observable(df: pd.DataFrame, obs: dict) -> tuple[list, list]:
    """Extract observable time series per simulation.

    The parquet stores each simulation as one row with list-valued columns:
      simulation_id | status | time | V_T | V_T.CD8 | ...

    Returns (succeeded, failed) where each is a list of (time, values) tuples.
    Failed-IC sims have empty time arrays but we still return them if they
    have any data; otherwise they're in the failed list as None.
    """
    succeeded = []
    failed = []
    for _, row in df.iterrows():
        time = np.asarray(row.get("time", []))
        if len(time) == 0:
            failed.append(None)
            continue
        try:
            if obs["transform"] == "identity":
                vals = np.asarray(row[obs["species"][0]])
            elif obs["transform"] == "ratio":
                num = np.asarray(row[obs["species"][0]])
                den = np.asarray(row[obs["species"][1]])
                vals = num / np.maximum(den, 1e-30)
            elif obs["transform"] == "sum":
                vals = sum(np.asarray(row[s]) for s in obs["species"])
            else:
                continue
            succeeded.append((time, vals))
        except (KeyError, TypeError):
            failed.append(None)
    return succeeded, failed


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_spider(df_old: pd.DataFrame, df_new: pd.DataFrame, changed: list[dict]):
    """Overlaid spider plot: old (red) and new (blue) on the same axes.

    Each observable gets one panel in a grid. Individual trajectories are
    thin/transparent; medians are thick. Failed-IC counts annotated.
    """
    n_obs = len(OBSERVABLES)
    ncols = min(n_obs, 3)
    nrows = int(np.ceil(n_obs / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.5 * nrows), squeeze=False)
    fig.patch.set_facecolor("white")

    c_old = (0.7, 0.3, 0.3)
    c_new = (0.2, 0.5, 0.7)

    for k, obs in enumerate(OBSERVABLES):
        ax = axes[k // ncols, k % ncols]
        ok_old, fail_old = extract_observable(df_old, obs)
        ok_new, fail_new = extract_observable(df_new, obs)

        if not ok_old and not ok_new:
            ax.set_title(f"{obs['name']} (not found)")
            continue

        # Plot old prior traces
        all_y_old = []
        t_old = None
        for t, y in ok_old:
            ax.plot(t, y, "-", color=(*c_old, 0.12), linewidth=0.5)
            all_y_old.append(y)
            t_old = t
        if all_y_old:
            med = np.nanmedian(np.array(all_y_old), axis=0)
            ax.plot(t_old, med, "-", color=c_old, linewidth=2.5, label="Old prior")

        # Plot new prior traces
        all_y_new = []
        t_new = None
        for t, y in ok_new:
            ax.plot(t, y, "-", color=(*c_new, 0.12), linewidth=0.5)
            all_y_new.append(y)
            t_new = t
        if all_y_new:
            med = np.nanmedian(np.array(all_y_new), axis=0)
            ax.plot(t_new, med, "-", color=c_new, linewidth=2.5, label="New prior")

        if obs["log"]:
            ax.set_yscale("log")
        ax.set_xlabel("Day")
        unit_str = f" ({obs['units']})" if obs["units"] else ""
        ax.set_ylabel(f"{obs['name']}{unit_str}")
        ax.grid(True, alpha=0.3)

        # Annotate counts
        n_ok_old, n_fail_old = len(ok_old), len(fail_old)
        n_ok_new, n_fail_new = len(ok_new), len(fail_new)
        annot = f"old: {n_ok_old} ok"
        if n_fail_old:
            annot += f", {n_fail_old} fail"
        annot += f"  |  new: {n_ok_new} ok"
        if n_fail_new:
            annot += f", {n_fail_new} fail"
        ax.text(0.98, 0.02, annot, transform=ax.transAxes, ha="right",
                va="bottom", fontsize=7, color="gray", style="italic")

        if k == 0:
            ax.legend(fontsize=9, loc="upper left")

    # Hide unused axes
    for j in range(n_obs, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)

    changed_names = [c["name"] for c in changed]
    fig.suptitle(
        f"Prior Update: {', '.join(changed_names)}",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_prior_shifts(
    old_priors: pd.DataFrame,
    new_priors: pd.DataFrame,
    changed: list[dict],
):
    """Overlay old vs new prior densities for changed parameters."""
    n = len(changed)
    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    fig.patch.set_facecolor("white")

    c_old = (0.7, 0.3, 0.3)
    c_new = (0.2, 0.5, 0.7)

    for i, ch in enumerate(changed):
        ax = axes[i // ncols, i % ncols]
        name = ch["name"]

        new_row = new_priors[new_priors["name"] == name].iloc[0]
        new_mu, new_sig = new_row["dist_param1"], new_row["dist_param2"]

        old_row = old_priors[old_priors["name"] == name]
        has_old = not old_row.empty and np.isfinite(ch["old_median"])

        # x range
        x_lo = np.exp(new_mu - 3.5 * new_sig)
        x_hi = np.exp(new_mu + 3.5 * new_sig)
        if has_old:
            old_mu = old_row["dist_param1"].values[0]
            old_sig = old_row["dist_param2"].values[0]
            x_lo = min(x_lo, np.exp(old_mu - 3.5 * old_sig))
            x_hi = max(x_hi, np.exp(old_mu + 3.5 * old_sig))

        x = np.linspace(x_lo, x_hi, 400)

        if has_old:
            y_old = lognorm.pdf(x, s=old_sig, scale=np.exp(old_mu))
            ax.fill_between(x, y_old, alpha=0.3, color=c_old, edgecolor=c_old, linewidth=1.5)
            ax.text(0.02, 0.95, f"Old: {ch['old_median']:.3g} (σ={ch['old_sigma']:.2f})",
                    transform=ax.transAxes, color=c_old, fontsize=9, va="top")

        y_new = lognorm.pdf(x, s=new_sig, scale=np.exp(new_mu))
        ax.fill_between(x, y_new, alpha=0.3, color=c_new, edgecolor=c_new, linewidth=1.5)
        ax.text(0.02, 0.82, f"New: {ch['new_median']:.3g} (σ={ch['new_sigma']:.2f})",
                transform=ax.transAxes, color=c_new, fontsize=9, va="top")

        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")

    # Hide unused axes
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)

    fig.suptitle("Prior Distributions: Old vs New", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare prior update impact on simulations")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--scenario", type=str, default=str(DEFAULT_SCENARIO))
    parser.add_argument("--priors", type=str, default=str(DEFAULT_PRIORS))
    parser.add_argument("--git-ref", type=str, default="HEAD")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--replot", action="store_true",
                        help="Skip simulations, reuse cached results (error if no cache)")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Clear simulation cache before running")
    parser.add_argument("--matlab-path", type=str,
                        default="/Applications/MATLAB_R2024b.app/bin/matlab")
    args = parser.parse_args()

    priors_path = Path(args.priors)
    scenario_path = Path(args.scenario)

    if args.clear_cache and CACHE_DIR.exists():
        import shutil
        shutil.rmtree(CACHE_DIR)
        print(f"Cleared cache: {CACHE_DIR}")

    print("=" * 50)
    print("Prior Update Comparison")
    print("=" * 50)

    # Load priors
    new_priors = pd.read_csv(priors_path)
    print(f"New priors: {len(new_priors)} params from {priors_path.name}")

    old_priors = load_priors_from_git(args.git_ref, priors_path)
    print(f"Old priors: {len(old_priors)} params from {args.git_ref}:{priors_path.name}")

    # Detect changes
    changed = find_changed_params(old_priors, new_priors)
    print(f"\nChanged parameters ({len(changed)}):")
    for c in changed:
        print(f"  {c['name']}: median {c['old_median']:.4g} -> {c['new_median']:.4g} "
              f"(sigma {c['old_sigma']:.3f} -> {c['new_sigma']:.3f})")

    if not changed:
        print("\nNo parameter changes detected. Nothing to compare.")
        return

    # Sample (controlled: only changed params differ)
    print(f"\nSampling {args.n_samples} parameter sets (controlled comparison)...")
    theta_old, theta_new = build_controlled_theta(
        old_priors, new_priors, changed, args.n_samples, args.seed,
    )
    n_shared = len(new_priors) - len(changed)
    print(f"  {n_shared} params shared, {len(changed)} params resampled")

    # Run simulations (or load from cache)
    print(f"\nRunning simulations ({scenario_path.name})...")
    df_old = run_simulations(theta_old, new_priors, scenario_path, "old_prior", args.matlab_path)
    df_new = run_simulations(theta_new, new_priors, scenario_path, "new_prior", args.matlab_path)

    # Summary
    n_old_ok = sum(1 for _, r in df_old.iterrows() if len(np.asarray(r.get("time", []))) > 0)
    n_new_ok = sum(1 for _, r in df_new.iterrows() if len(np.asarray(r.get("time", []))) > 0)
    print(f"\n  Old prior: {n_old_ok}/{len(df_old)} reached diagnosis")
    print(f"  New prior: {n_new_ok}/{len(df_new)} reached diagnosis")

    # Plot
    print("\nGenerating plots...")
    fig_spider = plot_spider(df_old, df_new, changed)
    fig_prior = plot_prior_shifts(old_priors, new_priors, changed)

    if not args.no_save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        fig_spider.savefig(OUT_DIR / "spider_comparison.png", dpi=150, bbox_inches="tight")
        fig_prior.savefig(OUT_DIR / "prior_shifts.png", dpi=150, bbox_inches="tight")
        print(f"\nFigures saved to {OUT_DIR}/")
    else:
        plt.show()

    print("Done.")


if __name__ == "__main__":
    main()
