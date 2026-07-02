#!/usr/bin/env python3
"""Short-and-sweet misspecification audit for a promoted SBI run.

Reads artifacts already on disk (no NPE retraining, no simulations) and emits a
focused diagnosis + recommended action per outside-CI target. Designed for fast
iteration cycles between model edits — complements the longer
``scripts/parameter_audit.py`` rather than replacing it.

Single-run usage:
    python scripts/misspec_audit.py                       # uses figures/sbi_stage2_current
    python scripts/misspec_audit.py --run figures/<run>

Comparison usage:
    python scripts/misspec_audit.py --run <new> --baseline <old> \
        --intended-fix cd8_density,m1_m2_ratio

Reseed stability usage (NPE-variance check):
    # First produce one or more reseed runs with sbi_runner, varying --seed but
    # keeping --split-seed and --cache-sampling-seed fixed so train/test split
    # and sim draws are identical. Then:
    python scripts/misspec_audit.py --run <primary> \
        --reseed-runs <reseed1>,<reseed2>,<reseed3>

    Targets whose diagnosis flips across seeds are flagged as NPE artifacts —
    don't act on those without retraining or more sims.

Inputs expected in the run directory (all produced by ``workflows/sbi_runner``):
    predictive_summary.csv               — prior/posterior CI per observable (raw)
    posterior_samples.csv                — posterior θ samples (raw, linear space)
    artifacts/theta_train.pt             — prior θ samples (log-space tensor)
    artifacts/x_train.pt                 — prior x samples (transformed/z-space)
    artifacts/obs_transformed.npy        — target in transformed/z-space
    artifacts/observable_names.json
    artifacts/param_names.json
    jacobian/jacobian.csv                — elasticity matrix
    jacobian/misspec_verdicts.csv        — sbi_runner's per-target verdicts
    jacobian/target_conflicts.csv        — cosine conflict pairs (may be absent)

Outputs (written to ``<run_dir>/misspec_audit/``):
    report.md          — markdown report (headline + tables)
    diagnosis.csv      — per-target diagnosis + action
    rail_pinning.csv   — per-param rail-pinning flags
    joint_reachability.csv — joint-reachability for conflict pairs
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN = ROOT / "figures" / "sbi_stage2_current"


# Diagnosis labels and the action each one implies.
DIAGNOSIS_ACTIONS = {
    "MECHANISM_MISSING": "Fix forward model — target unreachable under prior.",
    "TOPOLOGY_CONFLICT": "Fix model coupling — paired targets jointly unreachable.",
    "DEGENERACY": "Extract a discriminating submodel target to break the ridge.",
    "PRIOR_TIGHT": "Widen prior or question the submodel target on the named driver.",
    "PRIOR_MEDIAN_OR_SOFT": "Check target source; if primary PDAC, revisit data / wider prior.",
    "NPE_UNDER_TRAINED": "More sims / NPE capacity / sequential rounds.",
    "OK_INSIDE_CI": "No action.",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@dataclass
class RunArtifacts:
    run_dir: Path
    predictive: pd.DataFrame
    verdicts: pd.DataFrame
    conflicts: pd.DataFrame
    jacobian: pd.DataFrame
    posterior_samples: pd.DataFrame
    theta_train: np.ndarray  # log-space
    x_train: np.ndarray  # transformed/z-space
    obs_transformed: np.ndarray  # transformed/z-space
    observable_names: list[str]
    param_names: list[str]
    run_config: dict


def _load_torch(path: Path) -> np.ndarray:
    import torch
    t = torch.load(path, map_location="cpu", weights_only=False)
    if hasattr(t, "numpy"):
        return t.numpy()
    return np.asarray(t)


def load_run(run_dir: Path) -> RunArtifacts:
    run_dir = run_dir.resolve()
    predictive = pd.read_csv(run_dir / "predictive_summary.csv")
    verdicts = pd.read_csv(run_dir / "jacobian" / "misspec_verdicts.csv")
    conflicts_path = run_dir / "jacobian" / "target_conflicts.csv"
    conflicts = pd.read_csv(conflicts_path) if conflicts_path.exists() else pd.DataFrame(
        columns=["target_a", "target_b", "cosine_similarity", "z_a", "z_b"]
    )
    jacobian = pd.read_csv(run_dir / "jacobian" / "jacobian.csv", index_col=0)
    posterior_samples = pd.read_csv(run_dir / "posterior_samples.csv")
    theta_train = _load_torch(run_dir / "artifacts" / "theta_train.pt")
    x_train = _load_torch(run_dir / "artifacts" / "x_train.pt")
    obs_transformed = np.load(run_dir / "artifacts" / "obs_transformed.npy")
    with open(run_dir / "artifacts" / "observable_names.json") as f:
        observable_names = json.load(f)
    with open(run_dir / "artifacts" / "param_names.json") as f:
        param_names = json.load(f)
    run_config_path = run_dir / "run_config.json"
    run_config = json.loads(run_config_path.read_text()) if run_config_path.exists() else {}
    return RunArtifacts(
        run_dir=run_dir,
        predictive=predictive,
        verdicts=verdicts,
        conflicts=conflicts,
        jacobian=jacobian,
        posterior_samples=posterior_samples,
        theta_train=theta_train,
        x_train=x_train,
        obs_transformed=obs_transformed.ravel(),
        observable_names=observable_names,
        param_names=param_names,
        run_config=run_config,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_reachability(run: RunArtifacts) -> pd.DataFrame:
    """Per-target prior + posterior reachability flags (target inside CI95 raw)."""
    df = run.predictive.copy()
    # Reachable = target within [q05, q95] of the predictive distribution
    df["prior_reachable"] = (
        (df["obs_raw"] >= df["prior_q05_raw"]) & (df["obs_raw"] <= df["prior_q95_raw"])
    )
    df["posterior_reachable"] = (
        (df["obs_raw"] >= df["post_q05_raw"]) & (df["obs_raw"] <= df["post_q95_raw"])
    )
    # Distance to nearest CI edge in log-space (positive = inside, negative = outside)
    def _log_margin(obs, lo, hi):
        if obs <= 0 or lo <= 0 or hi <= 0:
            return np.nan
        return min(np.log(obs) - np.log(lo), np.log(hi) - np.log(obs))

    df["prior_log_margin"] = [
        _log_margin(o, lo, hi)
        for o, lo, hi in zip(df["obs_raw"], df["prior_q05_raw"], df["prior_q95_raw"])
    ]
    df["post_log_margin"] = [
        _log_margin(o, lo, hi)
        for o, lo, hi in zip(df["obs_raw"], df["post_q05_raw"], df["post_q95_raw"])
    ]
    return df[
        [
            "observable",
            "obs_raw",
            "prior_med_raw",
            "post_med_raw",
            "prior_reachable",
            "posterior_reachable",
            "prior_log_margin",
            "post_log_margin",
            "shift_sigma",
        ]
    ]


def compute_rail_pinning(
    run: RunArtifacts,
    edge_frac: float = 0.15,
    max_post_width_frac: float = 0.40,
) -> pd.DataFrame:
    """Per-param rail-pinning flag.

    A param is pinned against a prior edge if the posterior is *narrow*
    (posterior 5–95% span ≤ ``max_post_width_frac`` of the prior 5–95% span)
    *and* concentrated against that edge (posterior 95th-pctl below the
    prior's lower ``edge_frac`` band, or posterior 5th-pctl above the prior's
    upper ``edge_frac`` band). This separates "posterior wanted to push past
    the prior" from "NPE didn't update" (posterior ≈ prior).
    """
    theta_log = run.theta_train  # already log-space
    post = run.posterior_samples[run.param_names].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        post_log = np.where(post > 0, np.log(post), np.nan)

    prior_q05 = np.nanpercentile(theta_log, 5, axis=0)
    prior_q95 = np.nanpercentile(theta_log, 95, axis=0)
    post_q05 = np.nanpercentile(post_log, 5, axis=0)
    post_q95 = np.nanpercentile(post_log, 95, axis=0)
    prior_span = prior_q95 - prior_q05
    post_span = post_q95 - post_q05

    width_ratio = np.where(prior_span > 0, post_span / prior_span, np.inf)
    narrow = width_ratio < max_post_width_frac

    # Pinned_low: entire posterior 5-95% sits in the bottom edge_frac of the prior CI
    low_band = prior_q05 + edge_frac * prior_span
    high_band = prior_q95 - edge_frac * prior_span
    pinned_low = narrow & (post_q95 < low_band)
    pinned_high = narrow & (post_q05 > high_band)

    return pd.DataFrame({
        "parameter": run.param_names,
        "prior_log_q05": prior_q05,
        "prior_log_q95": prior_q95,
        "post_log_q05": post_q05,
        "post_log_q95": post_q95,
        "post_width_frac": width_ratio,
        "pinned_low": pinned_low,
        "pinned_high": pinned_high,
        "rail_pinned": pinned_low | pinned_high,
    })


# Minimum "alone reach" for a target to participate in a *true* topology
# conflict diagnosis. If alone-reach falls below this, the joint-reach=0%
# is just a co-occurrence of single-target unreachability, not coupling.
TOPOLOGY_ALONE_REACH_MIN = 0.05

# Threshold on jacobian-reported gap_log (log-distance from target to nearest
# posterior-predictive CI edge). Beyond this, the modal posterior is too far
# from the target for a discriminating submodel target (DEGENERACY action) to
# close the gap by tightening the param ridge — escalate to MECHANISM_MISSING.
# Empirical motivation: gap_log ≈ 0.7 corresponds to a 2× factor in raw space,
# already past what one extra anchor on a top driver is likely to shift the
# modal posterior in a multi-target NPE fit.
MISSPEC_GAP_LOG_MIN = 0.5


def compute_joint_reachability(
    run: RunArtifacts, conflicts: pd.DataFrame, z_window: float = 1.96
) -> pd.DataFrame:
    """For each conflict pair, fraction of prior draws that simultaneously land
    both observables within ``z_window`` (default 1.96 ≈ 95%) of the target in
    transformed space.
    """
    if conflicts.empty:
        return pd.DataFrame(columns=[
            "target_a", "target_b", "cosine_similarity",
            "frac_joint_reachable", "frac_a_alone", "frac_b_alone",
        ])
    x = run.x_train  # (n_prior, n_obs) in z-space
    obs = run.obs_transformed  # (n_obs,) in z-space
    name_to_idx = {n: i for i, n in enumerate(run.observable_names)}

    rows = []
    for _, c in conflicts.iterrows():
        if c["target_a"] not in name_to_idx or c["target_b"] not in name_to_idx:
            continue
        i = name_to_idx[c["target_a"]]
        j = name_to_idx[c["target_b"]]
        # Use only rows finite in both columns
        col_i, col_j = x[:, i], x[:, j]
        ok = np.isfinite(col_i) & np.isfinite(col_j)
        if not ok.any():
            continue
        ci, cj = col_i[ok], col_j[ok]
        within_i = np.abs(ci - obs[i]) <= z_window
        within_j = np.abs(cj - obs[j]) <= z_window
        rows.append({
            "target_a": c["target_a"],
            "target_b": c["target_b"],
            "cosine_similarity": float(c["cosine_similarity"]),
            "frac_a_alone": float(within_i.mean()),
            "frac_b_alone": float(within_j.mean()),
            "frac_joint_reachable": float((within_i & within_j).mean()),
        })
    return pd.DataFrame(rows)


def _d2_summary(run: RunArtifacts) -> Optional[dict]:
    """Pull the posterior-predictive D² + empirical p-value if available."""
    diag = run.run_dir / "diagnostics"
    summary_path = diag / "summary.json"
    if summary_path.exists():
        try:
            data = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            return None
        # Try a few likely keys; bail silently if absent.
        d2 = data.get("d2_post") or data.get("posterior_d2")
        p = data.get("d2_post_p_value") or data.get("posterior_d2_p_value")
        if d2 is not None:
            return {"d2_post": float(d2), "p_value": float(p) if p is not None else None}
    return None


# ---------------------------------------------------------------------------
# Diagnosis tree
# ---------------------------------------------------------------------------

def compute_partial_correlations(
    run: RunArtifacts, ridge: float = 1e-3
) -> pd.DataFrame:
    """Per-observable rank-partial correlations against every parameter,
    computed over the prior cloud (theta_train, x_train).

    For each (observable, parameter) pair, returns the Spearman partial
    correlation controlling for all *other* parameters — i.e., the
    parameter's *unique* monotonic contribution to that observable with the
    variance shared with correlated parameters projected out. This is the
    same PRCC statistic produced by ``workflows/prcc_sensitivity.py``,
    re-derived here against the audit's cal-target-granularity observables.

    Replaces ``|Jacobian|.idxmax`` for driver attribution: a parameter with
    a large raw |dY/dX| at the posterior mean can be a Jacobian-aliasing
    red herring when several correlated parameters share a common effect
    channel (e.g. ``k_PSC_to_apCAF`` topping six unrelated misspec rows in
    the cdc1 / CAF / IL-1 cluster). Partial correlation breaks the alias.

    Returns:
        DataFrame indexed by observable name with parameter-name columns.
        Same shape as ``run.jacobian``; signs follow the same convention
        (positive → param↑ tends to increase obs).
    """
    from scipy.stats import rankdata

    theta = np.asarray(run.theta_train, dtype=np.float64)
    x = np.asarray(run.x_train, dtype=np.float64)
    p = theta.shape[1]
    m = x.shape[1]

    # Rank-transform each column (mid-rank handles ties).
    theta_r = np.apply_along_axis(rankdata, 0, theta)
    x_r = np.apply_along_axis(rankdata, 0, x)

    # Joint correlation matrix over (theta_ranks | x_ranks). With finite N
    # and >100 parameters that share latent structure, the raw Spearman C is
    # near-singular; ridge-regularize before inverting.
    Z = np.concatenate([theta_r, x_r], axis=1)
    C = np.corrcoef(Z, rowvar=False)
    C = C + ridge * np.eye(p + m)
    P = np.linalg.inv(C)
    diag = np.diag(P)

    # Partial corr between param k and obs j: -P[k, p+j] / sqrt(P[k,k]*P[p+j,p+j])
    block = P[:p, p : p + m]  # (p, m)
    denom = np.sqrt(np.outer(diag[:p], diag[p : p + m]))
    partial = -block / denom  # (p, m) → params × obs

    return pd.DataFrame(
        partial.T,  # (m, p) → obs × params, matching run.jacobian shape
        index=run.observable_names,
        columns=run.param_names,
    )


def _top_param_pinned_in_correction_dir(
    obs_name: str,
    driver_matrix: pd.DataFrame,
    rail_df: pd.DataFrame,
    z_residual: float,
) -> tuple[Optional[str], bool]:
    """Identify the top driver (max |partial correlation|, or |elasticity|
    under the legacy Jacobian scorer) and whether it's rail-pinned in the
    direction that would correct the residual.

    ``driver_matrix`` carries either rank-partial correlations
    (:func:`compute_partial_correlations`, default) or raw Jacobian
    elasticities. Sign convention matches across both: positive value means
    ↑param tends to ↑obs.

    Sign convention: if z_residual > 0 (sim > target), the driver must move
    *against* its score sign. So a positive-score driver needs to decrease
    — check pinned_low. A negative-score driver needs to increase — check
    pinned_high. Vice versa for z_residual < 0.
    """
    if obs_name not in driver_matrix.index:
        return None, False
    row = driver_matrix.loc[obs_name]
    top_param = row.abs().idxmax()
    score = float(row.loc[top_param])

    rail = rail_df.set_index("parameter").loc[top_param]
    # Direction the driver must move to close the gap
    if z_residual > 0:
        # sim too high — driver must move opposite to its score sign
        need_pinned_low = score > 0
    else:
        need_pinned_low = score < 0
    pinned_in_dir = bool(rail["pinned_low"] if need_pinned_low else rail["pinned_high"])
    return top_param, pinned_in_dir


def diagnose(
    run: RunArtifacts,
    reachability: pd.DataFrame,
    rail: pd.DataFrame,
    joint: pd.DataFrame,
    conflicts: pd.DataFrame,
    driver_matrix: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Apply the decision tree per observable.

    ``driver_matrix`` carries the obs × param score table used for the
    ``top_driver`` column. When ``None``, falls back to ``run.jacobian``
    (legacy raw-elasticity scorer); pass :func:`compute_partial_correlations`
    output for partial-correlation scoring (default in :func:`run_audit`).
    """
    if driver_matrix is None:
        driver_matrix = run.jacobian
    reach_by_name = reachability.set_index("observable")
    verdict_by_name = run.verdicts.set_index("observable")

    # Per-observable conflict membership with joint-reachability
    joint_by_pair = {
        (r["target_a"], r["target_b"]): r for _, r in joint.iterrows()
    }
    # Per-observable conflict membership with joint-reachability AND alone-reach
    # for both partners (so we can tell single-target unreachability from real
    # coupling). joint_by_pair already carries frac_a_alone / frac_b_alone.
    obs_conflict_info: dict[str, list[tuple[str, float, float, float, float]]] = {}
    for (a, b), r in joint_by_pair.items():
        a_alone = float(r["frac_a_alone"])
        b_alone = float(r["frac_b_alone"])
        jr = float(r["frac_joint_reachable"])
        cos = float(r["cosine_similarity"])
        # entry: (partner, cosine, joint_reach, self_alone, partner_alone)
        obs_conflict_info.setdefault(a, []).append((b, cos, jr, a_alone, b_alone))
        obs_conflict_info.setdefault(b, []).append((a, cos, jr, b_alone, a_alone))

    rows = []
    for obs_name in reachability["observable"]:
        rch = reach_by_name.loc[obs_name]
        prior_ok = bool(rch["prior_reachable"])
        post_ok = bool(rch["posterior_reachable"])

        if post_ok:
            rows.append({
                "observable": obs_name,
                "diagnosis": "OK_INSIDE_CI",
                "action": DIAGNOSIS_ACTIONS["OK_INSIDE_CI"],
                "prior_reachable": prior_ok,
                "posterior_reachable": post_ok,
                "z_residual": float(
                    verdict_by_name.loc[obs_name, "z_residual"]
                    if obs_name in verdict_by_name.index else np.nan
                ),
                "top_driver": "",
                "driver_pinned_for_fix": False,
                "joint_unreachable_with": "",
                "joint_reach_frac": np.nan,
                "note": "",
            })
            continue

        v = verdict_by_name.loc[obs_name] if obs_name in verdict_by_name.index else None
        z_res = float(v["z_residual"]) if v is not None else np.nan
        target_unreachable = bool(v["target_unreachable"]) if v is not None and "target_unreachable" in v.index else False
        gap_log = float(v["gap_log"]) if v is not None and "gap_log" in v.index else np.nan

        diagnosis = None
        note = ""
        joint_partner = ""
        joint_frac = np.nan

        # Single-target unreachability dominates: if THIS target is unreachable
        # under the prior on its own, the diagnosis is mechanism/prior, not a
        # conflict — any joint-reach=0% with another outside-CI partner is a
        # co-occurrence of independent failures, not coupling.
        if not prior_ok:
            diagnosis = "MECHANISM_MISSING"
            note = "target outside prior 90% predictive CI"
        # Posterior-predictive gap is too large for partner-tightening to close:
        # even if there's a conflict pair, a discriminating submodel target
        # won't shift the modal posterior far enough. The jacobian step
        # already flagged `target_unreachable` (target outside posterior
        # predictive support) and computed `gap_log` (log distance to nearest
        # CI edge); if gap_log > MISSPEC_GAP_LOG_MIN, escalate to MECHANISM_MISSING
        # regardless of conflict-pair existence. See
        # notes/calibration/cd8fc_vegf_trace_2026_05_12.md for the motivating
        # case (cd8_fc_nivo: posterior median 2.4, target 5.98, gap_log≈0.93;
        # 25× sweep on top driver moves it only 13%, so partner-tightening
        # can't close the gap).
        elif target_unreachable and np.isfinite(gap_log) and gap_log > MISSPEC_GAP_LOG_MIN:
            diagnosis = "MECHANISM_MISSING"
            note = f"target outside posterior predictive (gap_log={gap_log:.2f} > {MISSPEC_GAP_LOG_MIN}); structural fix likely required"
        else:
            # Target IS individually reachable in prior. Now check for *true*
            # topology / degeneracy conflicts: pairings where BOTH targets are
            # individually reachable (alone_reach ≥ TOPOLOGY_ALONE_REACH_MIN)
            # but jointly are not. Excludes "fake" conflicts where the partner
            # is just independently unreachable.
            conflicts_for_obs = obs_conflict_info.get(obs_name, [])
            real_conflict_partners = [
                (other, cos, jr, self_alone, partner_alone)
                for (other, cos, jr, self_alone, partner_alone) in conflicts_for_obs
                if other in reach_by_name.index
                and not bool(reach_by_name.loc[other, "posterior_reachable"])
                and self_alone >= TOPOLOGY_ALONE_REACH_MIN
                and partner_alone >= TOPOLOGY_ALONE_REACH_MIN
            ]

            topology_partners = [t for t in real_conflict_partners if t[2] <= 0.001]
            if topology_partners:
                other, cos, jr, _, _ = sorted(topology_partners, key=lambda r: r[1])[0]
                diagnosis = "TOPOLOGY_CONFLICT"
                joint_partner = other
                joint_frac = jr
                note = f"jointly unreachable with {other} (cos={cos:.2f}); both individually reachable"
            elif real_conflict_partners:
                other, cos, jr, _, _ = sorted(real_conflict_partners, key=lambda r: r[2])[0]
                diagnosis = "DEGENERACY"
                joint_partner = other
                joint_frac = jr
                note = f"degenerate with {other} (cos={cos:.2f}, joint_reach={jr:.2%})"

        # No-conflict path: target is individually reachable in prior but
        # outside posterior CI — distinguish prior-tight (rail-pinned driver)
        # from NPE-under-trained vs prior-median / soft-data mismatch.
        if diagnosis is None:
            top_param, pinned_for_fix = _top_param_pinned_in_correction_dir(
                obs_name, driver_matrix, rail, z_res
            )
            if pinned_for_fix:
                diagnosis = "PRIOR_TIGHT"
                note = f"driver {top_param} pinned against the prior edge in correction direction"
            else:
                post_margin = float(rch["post_log_margin"])
                if not np.isfinite(post_margin) or post_margin < -1.0:
                    diagnosis = "PRIOR_MEDIAN_OR_SOFT"
                    note = "outside posterior CI by >1 log unit; check target source / widen prior"
                else:
                    diagnosis = "NPE_UNDER_TRAINED"
                    note = "small posterior gap, driver not pinned; more sims may suffice"

        top_param, pinned_for_fix = _top_param_pinned_in_correction_dir(
            obs_name, driver_matrix, rail, z_res
        )
        rows.append({
            "observable": obs_name,
            "diagnosis": diagnosis,
            "action": DIAGNOSIS_ACTIONS[diagnosis],
            "prior_reachable": prior_ok,
            "posterior_reachable": post_ok,
            "z_residual": z_res,
            "top_driver": top_param or "",
            "driver_pinned_for_fix": bool(pinned_for_fix),
            "joint_unreachable_with": joint_partner,
            "joint_reach_frac": joint_frac,
            "note": note,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _headline_counts(diagnosis_df: pd.DataFrame) -> dict[str, int]:
    counts = diagnosis_df["diagnosis"].value_counts().to_dict()
    return {k: int(counts.get(k, 0)) for k in DIAGNOSIS_ACTIONS}


def render_report(
    run: RunArtifacts,
    diagnosis: pd.DataFrame,
    rail: pd.DataFrame,
    joint: pd.DataFrame,
    d2: Optional[dict],
    baseline: Optional[dict] = None,
    intended_fix: Optional[list[str]] = None,
    reseed_stability: Optional[pd.DataFrame] = None,
    sweep_df: Optional[pd.DataFrame] = None,
) -> str:
    counts = _headline_counts(diagnosis)
    outside = diagnosis[diagnosis["diagnosis"] != "OK_INSIDE_CI"]

    lines: list[str] = []
    lines.append(f"# Misspecification audit — `{run.run_dir.name}`\n")

    # Headline
    total = sum(counts.values())
    ok = counts.get("OK_INSIDE_CI", 0)
    bad_counts = ", ".join(
        f"{k}: {v}" for k, v in counts.items() if v > 0 and k != "OK_INSIDE_CI"
    )
    short_counts = (
        f"OK: {ok}/{total}" + (f"  •  {bad_counts}" if bad_counts else "  •  all inside posterior 90% CI")
    )
    d2_line = ""
    if d2 is not None:
        p = d2.get("p_value")
        d2_line = (
            f"  •  D²_post = {d2['d2_post']:.1f}"
            + (f" (p={p:.3f})" if p is not None else "")
        )
    lines.append(f"**Headline:** {short_counts}{d2_line}\n")

    # Intended-fix tracker
    if intended_fix:
        lines.append("\n## Intended-fix tracker\n")
        lines.append("| observable | diagnosis | posterior_reachable | z_residual |")
        lines.append("|---|---|---|---|")
        for name in intended_fix:
            match = diagnosis[diagnosis["observable"].str.contains(name, case=False)]
            if match.empty:
                lines.append(f"| {name} | _not found_ | — | — |")
                continue
            for _, r in match.iterrows():
                arrow = "✅" if r["posterior_reachable"] else "❌"
                lines.append(
                    f"| {r['observable']} | {r['diagnosis']} | "
                    f"{arrow} {r['posterior_reachable']} | {r['z_residual']:+.2f} |"
                )

    # Outside-CI table
    lines.append("\n## Targets outside posterior 90% CI\n")
    if outside.empty:
        lines.append("_None._\n")
    else:
        lines.append(
            "| observable | diagnosis | action | prior_reach | top_driver | pinned_for_fix | z | note |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, r in outside.sort_values(["diagnosis", "z_residual"], key=lambda s: s.abs() if s.name == "z_residual" else s, ascending=[True, False]).iterrows():
            lines.append(
                f"| `{r['observable']}` | **{r['diagnosis']}** | {r['action']} | "
                f"{'✓' if r['prior_reachable'] else '✗'} | "
                f"{r['top_driver']} | "
                f"{'✓' if r['driver_pinned_for_fix'] else ' '} | "
                f"{r['z_residual']:+.2f} | {r['note']} |"
            )

    # Reach-sweep evidence
    if sweep_df is not None and not sweep_df.empty:
        lines.append("\n## Top-driver reach sweep (forward-sim probe)\n")
        lines.append(
            "Each outside-CI observable's top driver swept at ±3σ_prior around the\n"
            "posterior median; `✓` means at least one sweep point lands in target CI95.\n"
        )
        lines.append("| observable | top_driver | reaches target | range vs target | note |")
        lines.append("|---|---|---|---|---|")
        for _, r in sweep_df.iterrows():
            reaches = "✅" if r["sweep_reaches_target"] else "❌"
            obs_values = [v for v in (r["sweep_obs_values"] or []) if v == v]
            if obs_values:
                rng = f"[{min(obs_values):.3g}, {max(obs_values):.3g}]"
            else:
                rng = "—"
            tgt = (
                f"[{r['target_ci95_lo']:.3g}, {r['target_ci95_hi']:.3g}]"
                if r["target_ci95_lo"] == r["target_ci95_lo"]
                else "—"
            )
            lines.append(
                f"| `{r['observable']}` | {r['top_driver']} | {reaches} | {rng} vs {tgt} | {r['note']} |"
            )

    # Joint reachability for conflict pairs
    if not joint.empty:
        lines.append("\n## Conflict-pair joint reachability (prior predictive)\n")
        lines.append("| target_a | target_b | cosine | a_alone | b_alone | joint |")
        lines.append("|---|---|---|---|---|---|")
        for _, r in joint.sort_values("frac_joint_reachable").head(15).iterrows():
            lines.append(
                f"| `{r['target_a']}` | `{r['target_b']}` | "
                f"{r['cosine_similarity']:+.2f} | "
                f"{r['frac_a_alone']:.2%} | {r['frac_b_alone']:.2%} | "
                f"**{r['frac_joint_reachable']:.2%}** |"
            )

    # Rail-pinned params — only report those that are top drivers of outside-CI targets
    driver_params = set(
        p for p in diagnosis.loc[
            diagnosis["diagnosis"] != "OK_INSIDE_CI", "top_driver"
        ].unique() if p
    )
    pinned = rail[rail["rail_pinned"] & rail["parameter"].isin(driver_params)]
    if not pinned.empty:
        lines.append("\n## Rail-pinned drivers (top-driver params with narrow posterior at a prior edge)\n")
        lines.append("| parameter | edge | post log range (5-95%) | post / prior width |")
        lines.append("|---|---|---|---|")
        for _, r in pinned.iterrows():
            edge = "low" if r["pinned_low"] else "high"
            lines.append(
                f"| `{r['parameter']}` | {edge} | "
                f"[{r['post_log_q05']:.2f}, {r['post_log_q95']:.2f}] "
                f"(prior [{r['prior_log_q05']:.2f}, {r['prior_log_q95']:.2f}]) | "
                f"{r['post_width_frac']:.2f} |"
            )

    # Baseline diff
    if baseline is not None:
        lines.append(f"\n## Diff vs baseline (`{baseline['run_name']}`)\n")
        diff = baseline["diff"]
        if diff.empty:
            lines.append("_No diagnoses changed._\n")
        else:
            lines.append("| observable | baseline | current | change |")
            lines.append("|---|---|---|---|")
            for _, r in diff.iterrows():
                marker = (
                    "🟢 improved"
                    if r["current"] == "OK_INSIDE_CI"
                    else "🔴 regressed"
                    if r["baseline"] == "OK_INSIDE_CI"
                    else "↔ shifted"
                )
                lines.append(
                    f"| `{r['observable']}` | {r['baseline']} | {r['current']} | {marker} |"
                )
        if baseline.get("d2_delta") is not None:
            lines.append(f"\nΔD²_post = {baseline['d2_delta']:+.2f}\n")

    # Reseed stability
    if reseed_stability is not None and not reseed_stability.empty:
        unstable = reseed_stability[~reseed_stability["stable"]]
        n_total = len(reseed_stability)
        n_unstable = len(unstable)
        lines.append(
            f"\n## Reseed stability ({n_total - n_unstable}/{n_total} stable across reseeds)\n"
        )
        if unstable.empty:
            lines.append("_All diagnoses stable across reseeds — verdicts are trustworthy._\n")
        else:
            lines.append(
                "⚠️  Targets below flipped diagnosis under different NPE seeds — "
                "treat as NPE variance, not model evidence.\n"
            )
            lines.append("| observable | primary | reseed diagnoses | flipped |")
            lines.append("|---|---|---|---|")
            for _, r in unstable.sort_values("n_flipped", ascending=False).iterrows():
                lines.append(
                    f"| `{r['observable']}` | {r['primary']} | "
                    f"{r['reseed_diagnoses']} | {r['n_flipped']} |"
                )

    return "\n".join(lines) + "\n"


def compute_reseed_stability(
    primary: pd.DataFrame, reseed_diagnoses: list[pd.DataFrame], reseed_names: list[str]
) -> pd.DataFrame:
    """Per-observable stability: majority diagnosis, flip count, alternatives seen.

    A diagnosis is "stable" if it matches the primary across all reseed runs.
    Any disagreement is treated as NPE variance, not model evidence.
    """
    all_runs = [primary] + reseed_diagnoses
    all_names = ["primary"] + reseed_names
    obs_sets = [set(df["observable"]) for df in all_runs]
    common = set.intersection(*obs_sets) if obs_sets else set()
    rows = []
    for obs in sorted(common):
        diags = [df.set_index("observable").loc[obs, "diagnosis"] for df in all_runs]
        primary_diag = diags[0]
        flips = sum(1 for d in diags[1:] if d != primary_diag)
        unique_diags = sorted(set(diags))
        rows.append({
            "observable": obs,
            "primary": primary_diag,
            "reseed_diagnoses": "; ".join(
                f"{n}={d}" for n, d in zip(all_names[1:], diags[1:])
            ),
            "n_flipped": flips,
            "n_unique_diagnoses": len(unique_diags),
            "stable": flips == 0,
        })
    return pd.DataFrame(rows)


def compute_baseline_diff(
    current: pd.DataFrame, baseline: pd.DataFrame
) -> pd.DataFrame:
    cur = current.set_index("observable")["diagnosis"]
    base = baseline.set_index("observable")["diagnosis"]
    common = cur.index.intersection(base.index)
    diff_rows = []
    for obs in common:
        if cur[obs] != base[obs]:
            diff_rows.append({
                "observable": obs,
                "baseline": base[obs],
                "current": cur[obs],
            })
    return pd.DataFrame(diff_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_audit(
    run_dir: Path,
    baseline_dir: Optional[Path] = None,
    intended_fix: Optional[list[str]] = None,
    reseed_dirs: Optional[list[Path]] = None,
    driver_method: str = "partial_corr",
    reach_sweep: bool = False,
) -> Path:
    run = load_run(run_dir)
    reachability = compute_reachability(run)
    rail = compute_rail_pinning(run)
    joint = compute_joint_reachability(run, run.conflicts)
    driver_matrix = (
        compute_partial_correlations(run)
        if driver_method == "partial_corr"
        else run.jacobian
    )
    diagnosis = diagnose(
        run, reachability, rail, joint, run.conflicts, driver_matrix
    )
    d2 = _d2_summary(run)

    sweep_df: Optional[pd.DataFrame] = None
    if reach_sweep:
        # Local import so the audit doesn't require yaml / qsp_sim binary
        # in the common no-sweep path.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from misspec_reach_sweep import run_reach_sweep
        sweep_df = run_reach_sweep(diagnosis, run.posterior_samples)
        # Annotate diagnosis with sweep flags (merge by observable; keep
        # the original verdict — the sweep is corroborating evidence, not
        # a re-classification gate).
        if not sweep_df.empty:
            sweep_cols = sweep_df[[
                "observable", "sweep_reaches_target", "note",
            ]].rename(columns={"note": "sweep_note"})
            diagnosis = diagnosis.merge(sweep_cols, on="observable", how="left")

    baseline_block = None
    if baseline_dir is not None:
        base_run = load_run(baseline_dir)
        base_reach = compute_reachability(base_run)
        base_rail = compute_rail_pinning(base_run)
        base_joint = compute_joint_reachability(base_run, base_run.conflicts)
        base_driver = (
            compute_partial_correlations(base_run)
            if driver_method == "partial_corr"
            else base_run.jacobian
        )
        base_diag = diagnose(
            base_run, base_reach, base_rail, base_joint, base_run.conflicts, base_driver
        )
        diff = compute_baseline_diff(diagnosis, base_diag)
        base_d2 = _d2_summary(base_run)
        d2_delta = None
        if d2 is not None and base_d2 is not None:
            d2_delta = d2["d2_post"] - base_d2["d2_post"]
        baseline_block = {
            "run_name": base_run.run_dir.name,
            "diff": diff,
            "d2_delta": d2_delta,
        }

    reseed_stability = None
    if reseed_dirs:
        reseed_diags = []
        reseed_names = []
        for rd in reseed_dirs:
            rr = load_run(rd)
            rrch = compute_reachability(rr)
            rrail = compute_rail_pinning(rr)
            rjoint = compute_joint_reachability(rr, rr.conflicts)
            rdriver = (
                compute_partial_correlations(rr)
                if driver_method == "partial_corr"
                else rr.jacobian
            )
            reseed_diags.append(
                diagnose(rr, rrch, rrail, rjoint, rr.conflicts, rdriver)
            )
            reseed_names.append(rr.run_dir.name)
        reseed_stability = compute_reseed_stability(diagnosis, reseed_diags, reseed_names)

    out_dir = run.run_dir / "misspec_audit"
    out_dir.mkdir(exist_ok=True)
    diagnosis.to_csv(out_dir / "diagnosis.csv", index=False)
    rail.to_csv(out_dir / "rail_pinning.csv", index=False)
    joint.to_csv(out_dir / "joint_reachability.csv", index=False)
    if driver_method == "partial_corr":
        driver_matrix.to_csv(out_dir / "partial_correlations.csv")
    if reseed_stability is not None:
        reseed_stability.to_csv(out_dir / "reseed_stability.csv", index=False)
    if sweep_df is not None:
        sweep_df.to_csv(out_dir / "reach_sweep.csv", index=False)
    report = render_report(
        run, diagnosis, rail, joint, d2, baseline_block, intended_fix, reseed_stability,
        sweep_df=sweep_df,
    )
    (out_dir / "report.md").write_text(report)
    return out_dir / "report.md"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run", type=Path, default=DEFAULT_RUN,
        help=f"Run directory (default: {DEFAULT_RUN})",
    )
    p.add_argument(
        "--baseline", type=Path, default=None,
        help="Comparison run directory; emits a diff section.",
    )
    p.add_argument(
        "--intended-fix", type=str, default=None,
        help="Comma-separated observable substrings to track in a 'did the fix land?' section.",
    )
    p.add_argument(
        "--driver-method",
        choices=["partial_corr", "jacobian"],
        default="partial_corr",
        help=(
            "Driver-attribution scorer. 'partial_corr' (default) uses "
            "rank-partial Spearman correlations over the prior cloud — "
            "decorrelates Jacobian-aliasing red herrings (e.g. k_PSC_to_apCAF "
            "topping unrelated misspec rows). 'jacobian' uses the raw "
            "|elasticity|.idxmax at the posterior mean (legacy)."
        ),
    )
    p.add_argument(
        "--reach-sweep", action="store_true",
        help=(
            "After classifying, for each outside-CI observable sweep its top "
            "driver at ±3σ_prior around the posterior median (5 points) via "
            "local qsp_sim. ❌ in the resulting table = the top driver alone "
            "cannot lift the observable into target CI95, corroborating a "
            "structural MECHANISM_MISSING verdict. ~0.2s per sim; opt-in."
        ),
    )
    p.add_argument(
        "--reseed-runs", type=str, default=None,
        help=(
            "Comma-separated paths to NPE-reseed run dirs (same SPLIT_SEED + "
            "CACHE_SAMPLING_SEED, different SEED). Diagnoses are computed for "
            "each and compared with the primary; per-target flips are flagged."
        ),
    )
    args = p.parse_args()

    if not args.run.exists():
        print(f"error: run dir does not exist: {args.run}", file=sys.stderr)
        return 1

    intended = (
        [s.strip() for s in args.intended_fix.split(",") if s.strip()]
        if args.intended_fix else None
    )
    reseed_dirs = None
    if args.reseed_runs:
        reseed_dirs = [Path(s.strip()) for s in args.reseed_runs.split(",") if s.strip()]
        for rd in reseed_dirs:
            if not rd.exists():
                print(f"error: reseed run dir does not exist: {rd}", file=sys.stderr)
                return 1
    report_path = run_audit(
        args.run, args.baseline, intended, reseed_dirs,
        driver_method=args.driver_method,
        reach_sweep=args.reach_sweep,
    )
    print(f"Wrote {report_path}")
    print(report_path.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
