"""Cohort random-effect prototype for the SBI runner.

Per-cohort additive shift in copula-Gaussian space, applied at training-data
assembly time. Each calibration target carries a `source_tag` from its
`primary_data_source`; observables sharing a tag share a single δ_c per
training row. Across rows, δ_c ~ Normal(0, τ_cohort) iid.

Why this exists: multiple human-PDAC primary clinical sources can rate
`indication_match: exact` under the rubric yet still report systematically
different cohort medians (e.g. Golesworthy CD8/Treg=3.69 vs Liudahl-implied
~10). Translation σ from the rubric correctly stays small in that case, but
the joint likelihood then forces a single θ to compromise across cohorts —
which is exactly the structural-misspec signature the φ pairwise-tension
diagnostic reports. Adding a cohort random effect tells the NPE: "any single
realization of cohort medians is plausible; widen θ accordingly." It absorbs
inter-cohort heterogeneity into a nuisance, leaving structural misfit (cdc1↔
Th-exhaustion, the apCAF/stroma/M1:M2 tetrahedron) cleanly separable.

Path A only: τ_cohort is a fixed scale, not inferred. δ_c is iid noise per
training row; the NPE never sees it as a parameter. Posterior is over θ_QSP
alone. To infer τ_cohort itself, augment θ with τ + δ as nuisance latents
(Path B); not done here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import yaml


def attach_cohort_ids(
    target_df: pd.DataFrame,
    calibration_target_dirs: Sequence[str | Path],
) -> pd.DataFrame:
    """Read `primary_data_source.source_tag` from each target YAML and attach
    it as a `cohort_id` column on `target_df`. Targets without a source_tag
    fall back to a synthetic `_solo_<test_statistic_id>` cohort so they get
    their own δ_c (degenerate to per-observable iid noise — fine, just won't
    correlate with anything).
    """
    tag_by_id: dict[str, str] = {}
    for d in calibration_target_dirs:
        for yp in Path(d).glob("*.yaml"):
            try:
                data = yaml.safe_load(yp.read_text()) or {}
            except yaml.YAMLError:
                continue
            # CalibrationTarget YAMLs use `calibration_target_id`; older
            # SubmodelTarget-style files used `test_statistic_id`. Match
            # whichever the runner ends up keying on (target_df column is
            # `test_statistic_id` regardless — qsp-inference normalizes).
            tid = (
                data.get("calibration_target_id")
                or data.get("test_statistic_id")
                or yp.stem
            )
            pds = data.get("primary_data_source") or {}
            tag = pds.get("source_tag") if isinstance(pds, dict) else None
            if tag:
                tag_by_id[tid] = str(tag)

    out = target_df.copy()
    out["cohort_id"] = out["test_statistic_id"].map(tag_by_id)
    missing_mask = out["cohort_id"].isna()
    if missing_mask.any():
        # Fallback: solo cohort per missing target. Logs the names so a
        # YAML drift doesn't silently merge them under one phantom cohort.
        missing_names = out.loc[missing_mask, "test_statistic_id"].tolist()
        print(
            f"  [cohort-RE] {len(missing_names)} target(s) missing source_tag "
            f"— assigning solo cohorts: {missing_names}"
        )
        out.loc[missing_mask, "cohort_id"] = (
            "_solo_" + out.loc[missing_mask, "test_statistic_id"].astype(str)
        )
    return out


def apply_cohort_shift(
    x: np.ndarray,
    cohort_ids: Sequence[str],
    tau_cohort: float,
    seed: int,
) -> np.ndarray:
    """Add per-(row, cohort) Gaussian shift to `x`. Shape (n_rows, n_obs).

    Within a row, observables sharing a cohort_id receive the same δ_c.
    Across rows, δ_c is drawn iid from Normal(0, τ_cohort).

    Designed to run in copula-Gaussian space (post copula-transform) so the
    shift is unit-agnostic: standard-normal observables get a standard-normal
    nuisance shift. τ_cohort=0 is a no-op.
    """
    if tau_cohort <= 0:
        return x
    cohort_arr = np.asarray(cohort_ids)
    if len(cohort_arr) != x.shape[1]:
        raise ValueError(
            f"cohort_ids length {len(cohort_arr)} != x.shape[1] {x.shape[1]}"
        )
    _, cohort_idx = np.unique(cohort_arr, return_inverse=True)
    n_cohorts = int(cohort_idx.max()) + 1
    rng = np.random.default_rng(seed)
    deltas = rng.normal(0.0, tau_cohort, size=(x.shape[0], n_cohorts))
    return x + deltas[:, cohort_idx]


def print_cohort_table(target_df: pd.DataFrame, label: str = "") -> None:
    """One-shot diagnostic: cohorts × member targets, sorted by member count."""
    if "cohort_id" not in target_df.columns:
        print(f"  [cohort-RE] {label}: no cohort_id column — skipping table")
        return
    grp = (
        target_df.groupby("cohort_id")["test_statistic_id"]
        .agg(list)
        .reset_index()
    )
    grp["n_targets"] = grp["test_statistic_id"].apply(len)
    grp = grp.sort_values(["n_targets", "cohort_id"], ascending=[False, True])
    header = f"  [cohort-RE] cohort table {('(' + label + ')') if label else ''}"
    print(header)
    print(f"  {'cohort_id':<48} {'n':>3}  targets")
    print(f"  {'-' * 48} {'-' * 3}  {'-' * 40}")
    for _, row in grp.iterrows():
        names = ", ".join(row["test_statistic_id"])
        if len(names) > 80:
            names = names[:77] + "..."
        print(f"  {row['cohort_id']:<48} {row['n_targets']:>3}  {names}")
    n_solo = int((grp["n_targets"] == 1).sum())
    n_multi = int((grp["n_targets"] > 1).sum())
    print(
        f"  [cohort-RE] {len(grp)} cohorts total "
        f"({n_multi} multi-target, {n_solo} solo)"
    )
