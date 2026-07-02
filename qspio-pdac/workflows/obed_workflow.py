"""
OBED Workflow: Optimal Bayesian Experimental Design for Literature Prioritization

Identifies which additional observables or tighter parameter priors would most
reduce uncertainty in treatment response prediction (log tumor volume ratio).

Prerequisites:
    - Full simulations on HPC (trajectory grids extracted automatically)
    - PRCC sensitivity analysis (prcc_sensitivity.py)

Steps:
    1. Config
    2. Load trajectory grids (from cache or HPC)
    3. MI sweep — rank species×timepoints by mutual information with outcome
    4. Observable value of information — conditional variance reduction from measuring species X
    5. Prior tightening — variance reduction from narrowing parameter θ_k
    6. Literature target prioritization — aggregate rankings into search list
    7. Summary

Outputs (to figures/obed/):
    - mi_heatmap_{scenario}.png/csv: MI(species × timepoint, log V ratio) heatmap
    - observable_augmentation_{scenario}.csv: Variance reduction from adding measurement X
    - prior_tightening_{scenario}.csv: Variance reduction from tightening prior θ_k
    - literature_target_priorities.csv: Combined prioritized search list
    - obed_summary.json: Summary statistics

See notes/obed_plan.md for the full analysis plan.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qsp_hpc.simulation.qsp_simulator import QSPSimulator
from qsp_inference.inference.obed import (
    mi_ksg,
    mi_sweep_continuous,
)


# ============================================================================
# 1. CONFIG
# ============================================================================

# Scenarios to analyze
SCENARIOS = [
    # "baseline_no_treatment",
    # "gvax_neoadjuvant_zheng2022",
    "gvax_nivo_neoadjuvant_zheng2022",
    # "clinical_progression",
]

# Trajectory grid settings
SPECIES_LIST = "all"  # or list of species names
TIME_GRID = "daily"  # or list of floats

# Must match sbi_workflow.py for correct pool path resolution
PRIORS_CSV = "parameters/pdac_priors.csv"
MODEL_VERSION = "multi_scenario_v4_all_scenarios"
MODEL_SCRIPT = "immune_oncology_model_PDAC"

# Outcome variable: log tumor volume ratio (continuous)
TUMOR_VOLUME_SPECIES = "V_T.C1"

# Species to exclude from MI rankings (PK, tumor volume itself, synapses, compartment volumes)
EXCLUDE_PREFIXES = ("syn_",)
EXCLUDE_CONTAINS = (".aPD1", ".GVAX")  # PK species in any compartment
EXCLUDE_EXACT = {"V_T", "V_LN", "V_C", "V_P", "V_T.C1"}  # compartment volumes, cancer cells


def _is_excluded(species_name):
    """Check if a species should be excluded from rankings."""
    if species_name in EXCLUDE_EXACT:
        return True
    if species_name.startswith(EXCLUDE_PREFIXES):
        return True
    return any(pat in species_name for pat in EXCLUDE_CONTAINS)

def iqr_dispersion(x):
    """IQR-based dispersion measure, robust to heavy tails.

    Returns (IQR/1.349)^2, which equals variance for Gaussian data
    but is insensitive to outliers that dominate np.var in heavy-tailed
    outcome distributions (e.g., log tumor volume ratio with kurtosis ~50).
    """
    q75, q25 = np.percentile(x, [75, 25])
    return ((q75 - q25) / 1.349) ** 2


# Composite feature discovery
RUN_COMPOSITE_DISCOVERY = False  # Set True for targeted ratios on top TME species
COMPOSITE_TOP_N_SPECIES = 20  # Only use top N species from univariate MI for ratios

# Observable augmentation (step 4)
N_AUGMENTATION_CANDIDATES = 20  # top MI species to test
N_BINS = 10  # bins for conditional variance estimation

# Prior tightening (step 5)
TIGHTENING_QUANTILE = 0.25  # keep inner 50% (25th–75th percentile)

# Output
SAVE_DIR = Path("figures/obed")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 2. LOAD TRAJECTORY GRIDS
# ============================================================================
print("=" * 80)
print("OBED WORKFLOW")
print("=" * 80)

print("\n" + "=" * 80)
print("2. LOAD TRAJECTORY GRIDS")
print("=" * 80)

mi_results = {}
composite_results = {}
trajectory_data = {}  # scenario → (grid_df, meta)

for scen_name in SCENARIOS:
    print(f"\n--- {scen_name} ---")

    # Check for pre-extracted local grid first
    local_grid_path = Path(f"cache/trajectory_grids/{scen_name}/trajectory_grid.parquet")
    local_meta_path = local_grid_path.parent / "trajectory_meta.json"

    if local_grid_path.exists() and local_meta_path.exists():
        print(f"  Using cached grid: {local_grid_path}")
        grid_df = pd.read_parquet(local_grid_path)
        with open(local_meta_path) as f:
            meta = json.load(f)
    else:
        # Extract from HPC — reuses SBI pool path via model_version + priors hash
        simulator = QSPSimulator(
            priors_csv=PRIORS_CSV,
            model_version=MODEL_VERSION,
            model_script=MODEL_SCRIPT,
            model_description=f"OBED trajectory grid extraction: {scen_name}",
            scenario=scen_name,
            project_root=".",
        )

        print("  Extracting trajectory grid from HPC...")
        grid_df, meta = simulator.extract_trajectory_grid(
            species_list=SPECIES_LIST,
            time_grid=TIME_GRID,
            local_dest=local_grid_path.parent,
        )

    trajectory_data[scen_name] = (grid_df, meta)

    print(f"  Grid: {grid_df.shape[0]} sims × {grid_df.shape[1]} columns")
    print(f"  Species: {len(meta['species_list'])}, Timepoints: {len(meta['time_grid'])}")

# ============================================================================
# 3. MI SWEEP — rank species×timepoints by information about outcome
# ============================================================================
print("\n" + "=" * 80)
print("3. MI SWEEP")
print("=" * 80)

for scen_name in SCENARIOS:
    if scen_name not in trajectory_data:
        continue

    print(f"\n--- {scen_name} ---")
    grid_df, meta = trajectory_data[scen_name]

    # Compute continuous outcome: log(tumor_final / tumor_baseline)
    time_grid = np.array(meta["time_grid"])
    t_baseline_col = f"{TUMOR_VOLUME_SPECIES}__t{time_grid[0]:.1f}"
    t_final_col = None
    for t in reversed(time_grid):
        candidate = f"{TUMOR_VOLUME_SPECIES}__t{t:.1f}"
        if candidate in grid_df.columns and grid_df[candidate].notna().sum() > 0:
            t_final_col = candidate
            break

    if t_baseline_col not in grid_df.columns or t_final_col is None:
        print(f"  ERROR: Tumor volume columns not found for {TUMOR_VOLUME_SPECIES}")
        print(f"  Available species (first 10): {meta['species_list'][:10]}")
        continue

    print(f"  Tumor outcome: {t_baseline_col} → {t_final_col}")
    v_baseline = grid_df[t_baseline_col].values
    v_final = grid_df[t_final_col].values

    with np.errstate(divide="ignore", invalid="ignore"):
        outcome = np.log(v_final / v_baseline)
    valid_outcome = np.isfinite(outcome)
    n_valid = int(valid_outcome.sum())
    n_total = len(outcome)

    print(f"  Outcome: log(V_final/V_baseline), {n_valid}/{n_total} valid")
    if n_valid > 0:
        print(f"    median={np.median(outcome[valid_outcome]):.3f}, "
              f"IQR=[{np.percentile(outcome[valid_outcome], 25):.3f}, "
              f"{np.percentile(outcome[valid_outcome], 75):.3f}]")

    if n_valid < 50:
        print("  WARNING: Too few valid outcomes — skipping")
        continue

    # MI sweep over all species × timepoints (continuous-continuous)
    mi_cache_path = SAVE_DIR / f"mi_heatmap_{scen_name}.csv"
    if mi_cache_path.exists():
        print(f"  Using cached MI sweep: {mi_cache_path}")
        mi_parsed = pd.read_csv(mi_cache_path)
    else:
        feature_cols = [c for c in grid_df.columns if not c.startswith("param:")]
        features = grid_df[feature_cols].values
        mi_df = mi_sweep_continuous(
            features[valid_outcome], outcome[valid_outcome], feature_names=feature_cols
        )

        # Parse species and timepoint from column names
        parsed = []
        for _, row in mi_df.iterrows():
            if "__t" in row["feature"]:
                species, t_str = row["feature"].rsplit("__t", 1)
                parsed.append({"species": species, "timepoint": float(t_str), "mi": row["mi"]})
        mi_parsed = pd.DataFrame(parsed)
        mi_parsed.to_csv(mi_cache_path, index=False)

    mi_results[scen_name] = mi_parsed

    print(f"  Top 5 univariate features:")
    for _, row in mi_parsed.head(5).iterrows():
        print(f"    {row['species']} @ day {row['timepoint']:.0f}: MI = {row['mi']:.4f}")

    # Plot heatmap
    if len(mi_parsed) > 0:
        pivot = mi_parsed.pivot_table(
            index="species", columns="timepoint", values="mi", aggfunc="first"
        )
        pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=False).index]
        n_show = min(30, len(pivot))

        fig, ax = plt.subplots(figsize=(14, max(6, n_show * 0.3)))
        im = ax.imshow(pivot.iloc[:n_show].values, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        ax.set_yticks(range(n_show))
        ax.set_yticklabels(pivot.index[:n_show], fontsize=7)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"d{int(t)}" for t in pivot.columns], fontsize=8)
        ax.set_xlabel("Timepoint (days)")
        ax.set_title(f"MI(species×time, log V ratio) — {scen_name}")
        plt.colorbar(im, ax=ax, label="MI (nats)")
        fig.tight_layout()
        fig.savefig(SAVE_DIR / f"mi_heatmap_{scen_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ---- Composite features: ratios + fold changes ----
    comp_rows = []

    if not RUN_COMPOSITE_DISCOVERY:
        print("\n  Composite feature discovery: SKIPPED (RUN_COMPOSITE_DISCOVERY=False)")
    else:
        top_species = mi_parsed.groupby("species")["mi"].max().nlargest(COMPOSITE_TOP_N_SPECIES).index.tolist()
        print(f"\n  Composite feature discovery (top {len(top_species)} species)...")

        key_idx = np.linspace(0, len(time_grid) - 1, min(5, len(time_grid)), dtype=int)
        eval_times = time_grid[key_idx]
        outcome_valid = outcome[valid_outcome]

        n_pairs = len(top_species) * (len(top_species) - 1) // 2 * len(eval_times)
        print(f"  {n_pairs} ratio pairs to evaluate")

        for t in eval_times:
            species_vals = {}
            for s in top_species:
                c = f"{s}__t{t:.1f}"
                if c in grid_df.columns:
                    v = grid_df[c].values[valid_outcome]
                    if np.nanstd(v) > 1e-15:
                        species_vals[s] = v

            snames = list(species_vals.keys())
            for i, s1 in enumerate(snames):
                for s2 in snames[i + 1:]:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ratio = np.where(np.abs(species_vals[s2]) > 1e-15,
                                         species_vals[s1] / species_vals[s2], np.nan)
                    valid = ~np.isnan(ratio) & ~np.isinf(ratio)
                    if valid.sum() > 50:
                        mi = mi_ksg(ratio[valid], outcome_valid[valid])
                        mi = max(0.0, mi)
                        if mi > 0.01:
                            comp_rows.append({
                                "feature": f"({s1}/{s2})@d{t:.0f}",
                                "type": "ratio", "timepoint": t, "mi": mi,
                            })

        if len(time_grid) >= 2:
            t0, t_end = time_grid[0], time_grid[-1]
            for s in top_species:
                c0 = f"{s}__t{t0:.1f}"
                ce = f"{s}__t{t_end:.1f}"
                if c0 in grid_df.columns and ce in grid_df.columns:
                    v0 = grid_df[c0].values[valid_outcome]
                    ve = grid_df[ce].values[valid_outcome]
                    with np.errstate(divide="ignore", invalid="ignore"):
                        fc = np.where(np.abs(v0) > 1e-15, ve / v0, np.nan)
                    valid = ~np.isnan(fc) & ~np.isinf(fc)
                    if valid.sum() > 50:
                        mi = mi_ksg(fc[valid], outcome_valid[valid])
                        mi = max(0.0, mi)
                        if mi > 0.01:
                            comp_rows.append({
                                "feature": f"FC({s})_d{t0:.0f}_to_d{t_end:.0f}",
                                "type": "fold_change", "timepoint": t_end, "mi": mi,
                            })

    comp_df = pd.DataFrame(comp_rows)
    if len(comp_df) > 0:
        comp_df = comp_df.sort_values("mi", ascending=False).reset_index(drop=True)
    composite_results[scen_name] = comp_df
    comp_df.to_csv(SAVE_DIR / f"mi_composite_{scen_name}.csv", index=False)

    print(f"  Found {len(comp_df)} composite features with MI > 0.01")
    if len(comp_df) > 0:
        print("  Top 5:")
        for _, row in comp_df.head(5).iterrows():
            print(f"    {row['feature']}: MI = {row['mi']:.4f}")

# ============================================================================
# 4. OBSERVABLE VALUE OF INFORMATION — "What if we measured species X?"
# ============================================================================
# For each top MI species, estimate how much outcome variance would shrink
# if we could perfectly measure that species. Uses conditional variance:
# E[Var(outcome | species_X)] / Var(outcome). Computed by binning species
# values and averaging within-bin outcome variance. No new sims needed.
print("\n" + "=" * 80)
print("4. OBSERVABLE VALUE OF INFORMATION")
print("=" * 80)

augmentation_results = {}

if len(trajectory_data) > 0 and len(mi_results) > 0:
    for scen_name in SCENARIOS:
        if scen_name not in mi_results or scen_name not in trajectory_data:
            continue

        mi_df = mi_results[scen_name]
        grid_df, meta = trajectory_data[scen_name]

        # For each species, find peak MI and the timepoint where it occurs
        peak_mi_idx = mi_df.loc[mi_df.groupby("species")["mi"].idxmax()]
        peak_mi = peak_mi_idx.set_index("species")[["mi", "timepoint"]].rename(
            columns={"mi": "peak_mi", "timepoint": "best_day"}
        ).sort_values("peak_mi", ascending=False)

        peak_mi = peak_mi[~peak_mi.index.map(_is_excluded)]
        candidates = peak_mi.head(N_AUGMENTATION_CANDIDATES)

        print(f"\n--- {scen_name}: {len(candidates)} candidate observables ---")

        # Compute outcome
        time_grid = np.array(meta["time_grid"])
        t_baseline_col = f"{TUMOR_VOLUME_SPECIES}__t{time_grid[0]:.1f}"
        t_final_col = None
        for t in reversed(time_grid):
            col = f"{TUMOR_VOLUME_SPECIES}__t{t:.1f}"
            if col in grid_df.columns and grid_df[col].notna().sum() > 0:
                t_final_col = col
                break

        if t_final_col is None:
            print("  No valid tumor volume columns — skipping")
            continue

        v_baseline = grid_df[t_baseline_col].values
        v_final = grid_df[t_final_col].values
        with np.errstate(divide="ignore", invalid="ignore"):
            grid_outcome = np.log(v_final / v_baseline)
        valid_mask = np.isfinite(grid_outcome)
        outcome_valid = grid_outcome[valid_mask]
        total_disp = iqr_dispersion(outcome_valid)

        print(f"  {int(valid_mask.sum())} valid sims, "
              f"baseline IQR-dispersion = {total_disp:.4f}")

        scen_aug_results = []
        for species_name, row in candidates.iterrows():
            best_day = row["best_day"]
            col_name = f"{species_name}__t{best_day:.1f}"

            if col_name not in grid_df.columns:
                continue

            species_vals = grid_df[col_name].values[valid_mask]
            finite = np.isfinite(species_vals)
            if finite.sum() < 100 or np.nanstd(species_vals) < 1e-15:
                continue

            # Conditional dispersion: bin species, compute within-bin IQR dispersion
            species_binned = pd.qcut(
                species_vals[finite], q=N_BINS, duplicates="drop"
            )
            bin_df = pd.DataFrame({
                "outcome": outcome_valid[finite], "bin": species_binned
            })

            bin_disps, bin_counts = [], []
            for _, grp in bin_df.groupby("bin")["outcome"]:
                if len(grp) >= 5:
                    bin_disps.append(iqr_dispersion(grp.values))
                    bin_counts.append(len(grp))

            if not bin_disps:
                continue

            conditional_disp = np.average(bin_disps, weights=bin_counts)
            disp_reduction = 1.0 - conditional_disp / total_disp if total_disp > 0 else 0.0

            scen_aug_results.append({
                "species": species_name,
                "best_day": best_day,
                "peak_mi": row["peak_mi"],
                "dispersion_reduction": disp_reduction,
                "conditional_disp": conditional_disp,
                "total_disp": total_disp,
                "n_sims": int(finite.sum()),
            })

            print(f"    {species_name:<30s} @d{best_day:<3.0f}  "
                  f"MI={row['peak_mi']:.3f}  "
                  f"ΔDisp={disp_reduction:+.1%}")

        if scen_aug_results:
            aug_df = pd.DataFrame(scen_aug_results).sort_values(
                "dispersion_reduction", ascending=False
            )
            aug_df.to_csv(
                SAVE_DIR / f"observable_augmentation_{scen_name}.csv", index=False
            )
            augmentation_results[scen_name] = aug_df
            print(f"\n  Saved: {SAVE_DIR}/observable_augmentation_{scen_name}.csv")
else:
    reason = []
    if not trajectory_data:
        reason.append("no trajectory data")
    if not mi_results:
        reason.append("no MI results")
    print(f"  Skipped — {', '.join(reason)}")

# ============================================================================
# 5. PRIOR TIGHTENING — "What if we knew parameter θ_k better?"
# ============================================================================
# For each parameter, filter existing sims to those where θ_k is in the inner
# quantile range, then compute how much outcome variance shrinks.
# Pure filtering on trajectory grid — no NPE, no new sims.
print("\n" + "=" * 80)
print("5. PRIOR TIGHTENING")
print("=" * 80)

tightening_results = {}

# Load prior CV for each parameter
# For lognormal: CV = sqrt(exp(sigma^2) - 1)
# Inner 50% of lognormal ≈ ±0.6745σ in log-space → effective σ_new = 0.6745 * σ
prior_csv = pd.read_csv(PRIORS_CSV)
_prior_cv = {}
_prior_cv_tight = {}
for _, prow in prior_csv.iterrows():
    if prow["distribution"] == "lognormal":
        sigma = prow["dist_param2"]
        _prior_cv[prow["name"]] = np.sqrt(np.exp(sigma**2) - 1)
        sigma_tight = 0.6745 * sigma  # IQR of normal ≈ ±0.6745σ
        _prior_cv_tight[prow["name"]] = np.sqrt(np.exp(sigma_tight**2) - 1)

for scen_name in SCENARIOS:
    if scen_name not in trajectory_data:
        continue

    grid_df, meta = trajectory_data[scen_name]
    param_cols = [c for c in grid_df.columns if c.startswith("param:")]

    if not param_cols:
        print(f"  {scen_name}: no param: columns — skipping")
        continue

    # Compute outcome
    time_grid = np.array(meta["time_grid"])
    t_baseline_col = f"{TUMOR_VOLUME_SPECIES}__t{time_grid[0]:.1f}"
    t_final_col = None
    for t in reversed(time_grid):
        candidate_col = f"{TUMOR_VOLUME_SPECIES}__t{t:.1f}"
        if candidate_col in grid_df.columns and grid_df[candidate_col].notna().sum() > 0:
            t_final_col = candidate_col
            break

    if t_final_col is None:
        continue

    v_baseline = grid_df[t_baseline_col].values
    v_final = grid_df[t_final_col].values
    with np.errstate(divide="ignore", invalid="ignore"):
        outcome = np.log(v_final / v_baseline)
    valid = np.isfinite(outcome)
    outcome_valid = outcome[valid]
    total_disp = iqr_dispersion(outcome_valid)

    print(f"\n--- {scen_name}: {int(valid.sum())} valid sims, "
          f"baseline IQR-dispersion = {total_disp:.4f} ---")

    tightening_rows = []
    for pcol in param_cols:
        pname = pcol.replace("param:", "")
        pvals = grid_df[pcol].values[valid]

        lo = np.percentile(pvals, TIGHTENING_QUANTILE * 100)
        hi = np.percentile(pvals, (1 - TIGHTENING_QUANTILE) * 100)
        in_range = (pvals >= lo) & (pvals <= hi)
        n_in = int(in_range.sum())

        if n_in < 30:
            continue

        filtered_disp = iqr_dispersion(outcome_valid[in_range])
        disp_reduction = 1.0 - filtered_disp / total_disp if total_disp > 0 else 0.0

        tightening_rows.append({
            "parameter": pname,
            "dispersion_reduction": disp_reduction,
            "filtered_disp": filtered_disp,
            "total_disp": total_disp,
            "n_filtered": n_in,
            "n_total": len(outcome_valid),
            "quantile_range": f"{TIGHTENING_QUANTILE:.0%}–{1-TIGHTENING_QUANTILE:.0%}",
            "prior_cv": _prior_cv.get(pname, np.nan),
            "tightened_cv": _prior_cv_tight.get(pname, np.nan),
        })

    if tightening_rows:
        tight_df = pd.DataFrame(tightening_rows).sort_values(
            "dispersion_reduction", ascending=False
        ).reset_index(drop=True)

        # Add delta CV column
        tight_df["delta_cv"] = tight_df["tightened_cv"] - tight_df["prior_cv"]

        tight_df.to_csv(SAVE_DIR / f"prior_tightening_{scen_name}.csv", index=False)
        tightening_results[scen_name] = tight_df

        print(f"  Top 20 parameters by outcome IQR-dispersion reduction "
              f"(inner {1-2*TIGHTENING_QUANTILE:.0%} of prior):")
        print(f"  {'Parameter':<30s} {'Prior CV':>9s} {'Tight CV':>9s} "
              f"{'ΔCV':>9s}  {'ΔDisp':>7s}")
        print(f"  {'-' * 70}")
        for _, row in tight_df.head(20).iterrows():
            cv = row["prior_cv"]
            cv_t = row["tightened_cv"]
            dcv = row["delta_cv"]
            dr = row["dispersion_reduction"]
            if np.isfinite(cv) and np.isfinite(cv_t):
                print(f"    {row['parameter']:<30s} {cv:>8.1%} {cv_t:>8.1%} "
                      f"{dcv:>+8.1%}  {dr:>+.1%}")
            else:
                print(f"    {row['parameter']:<30s}      N/A      N/A "
                      f"     N/A  {dr:>+.1%}")

        print(f"\n  Saved: {SAVE_DIR}/prior_tightening_{scen_name}.csv")

# ============================================================================
# 6. LITERATURE TARGET PRIORITIZATION
# ============================================================================
print("\n" + "=" * 80)
print("6. LITERATURE TARGET PRIORITIZATION")
print("=" * 80)

priority_rows = []

for scen_name, mi_df in mi_results.items():
    # Filter out PK, tumor volume, synapses — same as step 4
    mi_filtered = mi_df[~mi_df["species"].map(_is_excluded)]
    # Deduplicate to peak MI per species
    peak_idx = mi_filtered.groupby("species")["mi"].idxmax()
    mi_top = mi_filtered.loc[peak_idx].sort_values("mi", ascending=False).head(20)
    for _, row in mi_top.iterrows():
        priority_rows.append({
            "source": "mi_univariate", "scenario": scen_name,
            "feature": f"{row['species']}@d{row['timepoint']:.0f}",
            "species": row["species"], "timepoint": row["timepoint"],
            "mi": row["mi"], "type": "single_species",
        })

for scen_name, comp_df in composite_results.items():
    for _, row in comp_df.head(20).iterrows():
        priority_rows.append({
            "source": "mi_composite", "scenario": scen_name,
            "feature": row["feature"], "species": "",
            "timepoint": row.get("timepoint", np.nan),
            "mi": row["mi"], "type": row.get("type", "composite"),
        })

priority_df = pd.DataFrame(priority_rows)
if len(priority_df) > 0:
    priority_df = priority_df.sort_values("mi", ascending=False).reset_index(drop=True)

priority_df.to_csv(SAVE_DIR / "literature_target_priorities.csv", index=False)

if len(priority_df) > 0:
    print("\n  Top 10 literature targets to search for:")
    for _, row in priority_df.head(10).iterrows():
        print(f"    [{row['scenario']:<35s}] {row['feature']:<40s} MI={row['mi']:.4f}")
else:
    print("  No MI results available — priority list empty.")

# ============================================================================
# 7. SUMMARY
# ============================================================================

summary = {
    "n_scenarios_mi": len(mi_results),
    "n_univariate_features": sum(len(df) for df in mi_results.values()),
    "n_composite_features": sum(len(df) for df in composite_results.values()),
    "n_augmentation_candidates": sum(len(df) for df in augmentation_results.values()),
    "n_tightening_params": sum(len(df) for df in tightening_results.values()),
    "outcome": "log_tumor_volume_ratio",
}
with open(SAVE_DIR / "obed_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n{'=' * 80}")
print("OBED WORKFLOW COMPLETE")
print(f"{'=' * 80}")
print(f"  Results saved to {SAVE_DIR}/")
