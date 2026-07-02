#!/usr/bin/env python3
"""
SBI Workflow for PDAC QSP Model — Multi-Scenario Joint Inference

Clean pipeline: simulate → prior predictive → train NPE → posterior → posterior predictive.
Supports N scenarios with shared parameters and per-scenario observables.

Sections:
1. Config + setup
2. Load & process simulations (per-scenario simulate, joint NaN filter, copula, concatenate)
3. Prior predictive check (quick sanity gate)
4. Train NPE
5. Evaluate NPE (θ-space)
   5b. Local calibration (near observed data)
6. Posterior from observed data
7. Posterior predictive check (actual simulations)
   7e. Posterior Jacobian + misspecification assessment
8. Save

Usage:
    python workflows/sbi_workflow.py
"""

import json
import threading
import time
from pathlib import Path
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

from sbi.inference import NPE
from sbi.neural_nets import posterior_nn

from qsp_hpc.batch.hpc_job_manager import HPCJobManager
from qsp_hpc.simulation.cpp_simulator import CppSimulator
from qsp_hpc.simulation.multi_scenario_runner import MultiScenarioRunner
from qsp_inference.inference import (
    get_observed_data,
    compute_per_param_calibration,
    convert_posterior_samples_to_original_space,
    plot_posterior_vs_prior_marginals,
    sbi_recovery,
    sbi_z_score_contraction,
    sbi_calibration_ecdf,
    sbi_loo_predictive_check,
    sbi_self_reference_null,
    sbi_posterior_predictive_coverage,
    save_diagnostics,
)
from qsp_inference.inference.data_processing import add_observation_noise
from qsp_inference.inference.gaussian_copula_transform import compute_quantiles_from_array, transform_to_normal_from_array
from qsp_inference.priors.copula_prior import load_composite_prior_log
from qsp_inference.auxiliary import (
    HierarchicalAuxiliaryPrior,
    discover_auxiliary_members,
    load_auxiliary_config,
)

from workflows.cohort_random_effect import (
    apply_cohort_shift,
    attach_cohort_ids,
    print_cohort_table,
)


class _JointBlockPrior(torch.distributions.Distribution):
    """Concatenate two log-space priors along the parameter axis.

    Both blocks must operate in **log θ space** (samples are log values,
    log_prob expects log values). For QSP that's the existing
    GaussianCopulaPrior with log-marginals; for aux that's the
    lognormal-base HierarchicalAuxiliaryPrior's underlying MVN, which
    is itself a log-space MVN — see ``_aux_log_block_mvn`` below.

    Used to give NPE a single prior over the full ``[θ_qsp, θ_aux]``
    vector. The prior contributes structure during training (sbi
    leashes posterior support to it) and is the sampling distribution
    for diagnostic plots.
    """

    arg_constraints: dict = {}
    has_rsample = False

    def __init__(self, blocks):
        # blocks: list of torch.distributions.Distribution, each with
        # 1-D event_shape.
        self._blocks = list(blocks)
        sizes = [int(b.event_shape[0]) for b in self._blocks]
        self._sizes = sizes
        super().__init__(
            batch_shape=torch.Size(),
            event_shape=torch.Size([sum(sizes)]),
            validate_args=False,
        )

    def sample(self, sample_shape=torch.Size()):
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        parts = [b.sample(sample_shape) for b in self._blocks]
        return torch.cat([p.float() for p in parts], dim=-1)

    def log_prob(self, value):
        offsets = np.cumsum([0] + self._sizes).tolist()
        lp = None
        for b, lo, hi in zip(self._blocks, offsets[:-1], offsets[1:]):
            sub = value[..., lo:hi]
            sub_lp = b.log_prob(sub.double()).float()
            lp = sub_lp if lp is None else lp + sub_lp
        return lp


def _aux_log_block_mvn(aux_prior):
    """Return a torch Distribution over ``log θ_aux`` from a HierarchicalAuxiliaryPrior.

    For lognormal-base aux blocks, the underlying ``_GroupBlock.mvn`` is
    already a log-space MVN. We expose it as a single block-diagonal
    MVN over the concatenated log-aux vector so :class:`_JointBlockPrior`
    can compose it with the QSP log prior without an exp/log round-trip.
    Validated upstream: every block must be ``transform == "exp"``.
    """
    blocks = aux_prior._blocks
    n = sum(int(b.mvn.event_shape[0]) for b in blocks)
    mu = torch.zeros(n, dtype=torch.float64)
    cov = torch.zeros((n, n), dtype=torch.float64)
    offset = 0
    for b in blocks:
        k = int(b.mvn.event_shape[0])
        mu[offset:offset + k] = b.mvn.loc.to(torch.float64)
        cov[offset:offset + k, offset:offset + k] = b.mvn.covariance_matrix.to(
            torch.float64
        )
        offset += k
    return torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=cov)


def _joint_reachability_check(sim, target_df, observable_names, save_dir, label):
    """Joint reachability: per-sample fraction of targets where the simulated
    observable falls inside the target's empirical 95% CI.

    Per-target reachability is misleading for joint inference: a wide prior can
    cover any single target in its tail while no single draw simultaneously
    satisfies a meaningful subset of targets. This check reports the per-sample
    fraction of targets reached, the maximum joint reach across draws (a hard
    ceiling on what the posterior can achieve), and the least-jointly-reachable
    targets — the cleanest signal for structural misspec.

    sim: (n_samples, n_obs) raw simulator outputs aligned with target_df rows.
    target_df: must have columns ci95_lower, ci95_upper aligned with observable_names.
    label: 'prior' or 'posterior' — used for output filename + console header.
    """
    num_obs = len(observable_names)
    if len(target_df) != num_obs:
        print(f"  [joint reach] target_df rows ({len(target_df)}) != n_obs ({num_obs}); skip")
        return None
    ci_low = pd.to_numeric(target_df['ci95_lower'], errors='coerce').values.astype(float)
    ci_high = pd.to_numeric(target_df['ci95_upper'], errors='coerce').values.astype(float)
    sim = np.asarray(sim, dtype=float)
    target_valid = np.isfinite(ci_low) & np.isfinite(ci_high)
    sim_valid = np.isfinite(sim)
    valid = sim_valid & target_valid[None, :]
    lo_b = np.broadcast_to(ci_low[None, :], sim.shape)
    hi_b = np.broadcast_to(ci_high[None, :], sim.shape)
    reachable = np.zeros_like(sim, dtype=bool)
    reachable[valid] = (sim[valid] >= lo_b[valid]) & (sim[valid] <= hi_b[valid])
    per_sample_n = reachable.sum(axis=1)
    per_sample_d = valid.sum(axis=1)
    per_target_n = reachable.sum(axis=0)
    per_target_d = valid.sum(axis=0)
    per_target_rate = np.divide(
        per_target_n, np.maximum(per_target_d, 1), dtype=float
    )
    n_eval = int(target_valid.sum())

    print(f"\n  Joint reachability ({label}, sim within target empirical CI95):")
    print(f"    Targets evaluated: {n_eval}/{num_obs} (others have NaN CI)")
    if per_sample_n.size > 0 and n_eval > 0:
        ps_frac = per_sample_n / np.maximum(per_sample_d, 1)
        print(f"    Per-sample count: median={int(np.median(per_sample_n))}/{n_eval}, "
              f"max={int(per_sample_n.max())}, "
              f"p95={int(np.percentile(per_sample_n,95))}, "
              f"p99={int(np.percentile(per_sample_n,99))}")
        print(f"    Per-sample fraction: median={np.median(ps_frac):.1%}, max={ps_frac.max():.1%}")
        order = np.argsort(per_target_rate)
        order = [i for i in order if target_valid[i]]
        print(f"    Least-jointly-reachable targets (bottom 10):")
        for i in order[:10]:
            print(f"      {per_target_rate[i]:6.2%} "
                  f"({per_target_n[i]:4d}/{per_target_d[i]:4d})  {observable_names[i]}")

    diag_dir = save_dir / 'diagnostics'
    diag_dir.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({
        'observable': observable_names,
        'ci95_lower': ci_low,
        'ci95_upper': ci_high,
        'reach_n': per_target_n,
        'reach_d': per_target_d,
        'reach_rate': per_target_rate,
    })
    out.to_csv(diag_dir / f'{label}_joint_reachability.csv', index=False)

    # ---- pairwise tension (phi coefficient on per-sample reach matrix) ----
    # Targets that are jointly reachable individually but rarely co-reachable
    # (negative phi) reveal structural conflicts: parameter regions that
    # satisfy one target systematically miss the other. Restrict to rows
    # where every target is finite (the joint training set) so phi is not
    # confounded by NaN survival.
    joint_mask = np.all(sim_valid & target_valid[None, :], axis=1)
    if joint_mask.sum() < 100 or n_eval < 2:
        return out
    R = reachable[joint_mask].astype(np.float64)  # (N_joint, n_obs)
    Rsel = R[:, target_valid]                     # drop NaN-CI targets
    sel_names = [observable_names[i] for i in range(num_obs) if target_valid[i]]
    N = Rsel.shape[0]
    P1 = Rsel.mean(axis=0)
    P11 = (Rsel.T @ Rsel) / N
    P00 = ((1 - Rsel).T @ (1 - Rsel)) / N
    P10 = (Rsel.T @ (1 - Rsel)) / N
    P01 = ((1 - Rsel).T @ Rsel) / N
    denom = np.sqrt(P1[:, None] * P1[None, :] * (1 - P1)[:, None] * (1 - P1)[None, :])
    denom = np.where(denom == 0, np.nan, denom)
    phi = (P11 * P00 - P10 * P01) / denom
    np.fill_diagonal(phi, np.nan)

    rows = []
    n_sel = len(sel_names)
    for i in range(n_sel):
        for j in range(i + 1, n_sel):
            if not np.isfinite(phi[i, j]):
                continue
            rows.append({
                'target_i': sel_names[i],
                'target_j': sel_names[j],
                'phi': phi[i, j],
                'P_i': P1[i],
                'P_j': P1[j],
                'P_ij': P11[i, j],
                'expected_indep': P1[i] * P1[j],
                'co_reach_deficit': (
                    (P1[i] * P1[j] - P11[i, j]) / max(P1[i] * P1[j], 1e-12)
                ),
            })
    tension_df = pd.DataFrame(rows)
    tension_df.to_csv(diag_dir / f'{label}_pairwise_tension.csv', index=False)

    qual = tension_df[(tension_df['P_i'] >= 0.05) & (tension_df['P_j'] >= 0.05)]
    qual = qual.sort_values('phi').head(15)
    if not qual.empty:
        print(f"\n  Pairwise tension ({label}, top 15 by negative phi, both marginals ≥ 5%):")
        print(f"      phi    P_i    P_j   P_ij  target_i / target_j")
        for _, r in qual.iterrows():
            print(f"    {r['phi']:+6.3f} {r['P_i']:6.1%} {r['P_j']:6.1%} {r['P_ij']:6.1%}  "
                  f"{r['target_i']}")
            print(f"    {'':>6} {'':>6} {'':>6} {'':>6}     {r['target_j']}")
    return out


def load_llm_elicited_targets(csv_path, observable_names):
    """Load LLM-elicited targets and remap generic names to model observable names.

    The elicitation prompt uses generic names (no author suffixes) to avoid
    hinting at specific publications. This function maps them back.
    """
    # Mapping from elicited (generic) names → model observable names
    NAME_REMAP = {
        'cd8_exhausted_fraction': 'cd8_exhausted_fraction_shen2017',
        'cd8_treg_ratio_alt': 'cd8_treg_ratio_golesworthy2022',
        'treg_fraction_cd4_alt': 'treg_fraction_cd4_hiraoka2006',
        'apcaf_fraction_of_caf_alt': 'apcaf_fraction_of_caf_elyada2019',
    }
    df = pd.read_csv(csv_path)
    df['observable'] = df['observable'].replace(NAME_REMAP)

    # Build dict keyed by observable name → estimate value
    elicited = {}
    for _, row in df.iterrows():
        name = row['observable']
        if name in observable_names:
            elicited[name] = float(row['estimate'])

    # Warn about missing observables
    missing = set(observable_names) - set(elicited.keys())
    if missing:
        print(f"  WARNING: {len(missing)} observables not in LLM-elicited CSV: {missing}")

    # Return as array in same order as observable_names
    return np.array([elicited.get(name, np.nan) for name in observable_names])


def participation_ratio(X):
    """Participation ratio of covariance eigenvalue spectrum.

    k_eff = (sum lambda_i)^2 / sum(lambda_i^2)

    Equals n if all eigenvalues equal, <n if correlated. Estimates effective
    degrees of freedom for chi^2 reference distribution of Mahalanobis D^2.
    """
    cov = np.cov(X, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 0]
    k_eff = float(np.sum(eigenvalues)**2 / np.sum(eigenvalues**2))
    return k_eff, eigenvalues

def _processed_cache_path(cfg) -> Path:
    """Path of the processed-tensors cache bundle for the given config.

    Keyed by the fields that determine the step-2 output: model version,
    scenarios, PRCC tier, train-subsample, both seeds that affect theta
    draws and the train/test split, AND the restriction classifier
    fingerprint when set (so restricted + unrestricted bundles don't
    collide at the same n_train_subsample). Priors CSV content is
    intentionally NOT hashed here — use ``model_version`` to bump.
    """
    scen_key = '-'.join(sorted(cfg.scenarios.keys()))
    tier_key = str(cfg.prcc_tier) if cfg.prcc_tier is not None else 'none'
    sub_key = str(cfg.n_train_subsample) if cfg.n_train_subsample else 'all'
    if cfg.restriction_classifier_dir is not None:
        import hashlib
        cdir = Path(cfg.restriction_classifier_dir)
        h = hashlib.sha256()
        for fname in ('classifier.pkl', 'metadata.json'):
            f = cdir / fname
            if f.exists():
                h.update(f.read_bytes())
        h.update(f"|tau={cfg.restriction_threshold:.6f}".encode())
        restrict_key = f"__restr{h.hexdigest()[:10]}"
    else:
        restrict_key = ""
    tag = (
        f"{cfg.model_version}__{scen_key}__tier{tier_key}__sub{sub_key}"
        f"__split{cfg.split_seed}__s{cfg.cache_sampling_seed}{restrict_key}"
    )
    return Path('cache/sbi_processed') / f"{tag}.pkl"


def _save_processed_bundle(path: Path, state: dict) -> None:
    import pickle
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(state, f)


def _load_processed_bundle(path: Path) -> dict:
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)


def _analyze_on_delta(inputs: dict, stage_dir: Path, tag: str) -> Path:
    """Run NPE evaluation + posterior analysis (steps 5/5b/6) fully on Delta.

    `inputs` is a dict of file-name → payload:
      - '*.pt'   → torch.Tensor
      - '*.npy'  → np.ndarray
      - '*.json' → jsonable object
      - '*.csv'  → source Path (file is copied)
      - other files → source Path (copied)

    Writes the bundle into `stage_dir/input/`, invokes
    workflows/delta/run_delta_analysis.sh, and returns the local path to the
    downloaded `output/` directory (CSVs, PNGs, run_metrics.json).
    """
    import shutil
    import subprocess

    stage_dir = Path(stage_dir)
    in_dir = stage_dir / 'input'
    in_dir.mkdir(parents=True, exist_ok=True)

    for name, payload in inputs.items():
        dst = in_dir / name
        if name.endswith('.pt'):
            torch.save(payload, dst)
        elif name.endswith('.npy'):
            np.save(dst, payload)
        elif name.endswith('.json'):
            dst.write_text(json.dumps(payload, indent=2))
        else:
            shutil.copy(payload, dst)

    script = Path(__file__).parent / 'delta' / 'run_delta_analysis.sh'
    print(f"  [delta] invoking {script.name} (stage={stage_dir})", flush=True)
    subprocess.run([str(script), str(stage_dir), tag], check=True)
    return stage_dir / 'output'


# ============================================================================
# CSV-ONLY PRIOR HELPER
# ============================================================================


def _load_csv_only_prior_log(csv_path):
    """CSV-only equivalent of load_composite_prior_log: independent normals
    in log-space for every parameter in the CSV, no copula correlations.

    Used when ``cfg.skip_submodel_sync=True`` — pdac_priors.csv as-is.
    """
    import csv as csv_mod
    from qsp_inference.priors.copula_prior import GaussianCopulaPrior
    from scipy import stats

    csv_params = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            csv_params.append(
                {
                    "name": row["name"],
                    "mu": float(row["dist_param1"]),
                    "sigma": float(row["dist_param2"]),
                }
            )
    param_names = [p["name"] for p in csv_params]
    marginals = [stats.norm(loc=p["mu"], scale=p["sigma"]) for p in csv_params]
    R = np.eye(len(param_names))
    prior = GaussianCopulaPrior(marginals=marginals, correlation=R, param_names=param_names)
    return prior, param_names


# ============================================================================
# RUN CONFIG
# ============================================================================

from dataclasses import dataclass, field, asdict
from typing import Optional


# RunConfig + scenario/tier defaults moved to workflows/run_config.py
# so the dataclass can be imported without paying for torch / sbi /
# matplotlib / etc. Re-exported here for backward compatibility.
from workflows.run_config import (
    DEFAULT_SCENARIOS,
    DEFAULT_TIER_CUTOFFS,
    RunConfig,
    _PDAC_BUILD_ROOT,
)


def _check_submodel_priors_freshness(
    *,
    yaml_path: Path,
    submodel_dir: Path,
    priors_csv: Path,
    strict: bool,
) -> None:
    """Warn (or abort) if `submodel_priors.yaml` is stale.

    Compares the per-component fingerprint manifest stored in the YAML
    header against a freshly-recomputed manifest from the current tree.
    Drift means the static composite prior the runner is about to consume
    no longer reflects the current submodel target / prior CSV / config.
    """
    try:
        from qsp_inference.submodel.freshness import (
            check_submodel_priors_freshness,
            load_stored_freshness,
        )
    except ImportError:
        # Older qsp-inference without freshness support — skip silently
        # rather than gate launches on a version bump.
        return

    if not yaml_path.exists():
        return  # downstream loader will raise its own error

    stored = load_stored_freshness(yaml_path)
    if not stored:
        msg = (
            f"  [submodel-freshness] {yaml_path} has no freshness header "
            f"(pre-fingerprint vintage). Cannot verify staleness — "
            f"regenerate with `python scripts/regen_submodel_priors.py "
            f"--rebuild-all` to add fingerprints."
        )
        if strict:
            raise SystemExit(msg + "\n  --strict-submodel-freshness set; aborting.")
        print(msg)
        return

    report = check_submodel_priors_freshness(yaml_path, submodel_dir, priors_csv)
    if report.is_fresh():
        n = len(stored)
        print(f"  [submodel-freshness] OK ({n} component fingerprints match current tree).")
        return

    print("  [submodel-freshness] STALE:")
    for line in report.format().splitlines():
        print(f"    {line}")
    print(
        "    Regenerate with `python scripts/regen_submodel_priors.py "
        "--invalidate <PARAM> [<PARAM> ...]` for the affected parameters, "
        "or `--rebuild-all` for a clean slate."
    )
    if strict:
        raise SystemExit(
            "  --strict-submodel-freshness set; aborting before consuming a stale composite prior."
        )


def run_inference(cfg: RunConfig) -> dict:
    """Run the full multi-scenario SBI pipeline for a single config.

    Returns a metrics dict with training + diagnostic summary statistics.
    All intermediate artifacts (diagnostics CSVs, figures, posterior samples)
    are written to cfg.save_dir (auto-derived from scenarios + prcc_tier if None).
    """
    # --- unpack config into local aliases matching the original script ---
    MODEL_VERSION = cfg.model_version
    MODEL_DESCRIPTION = cfg.model_description
    MODEL_SCRIPT = cfg.model_script
    SCENARIOS = cfg.scenarios
    PRIORS_CSV = cfg.priors_csv
    MODEL_STRUCTURE_FILE = cfg.model_structure_file
    PRCC_RANKING_CSV = cfg.prcc_ranking_csv
    PRCC_TIER = cfg.prcc_tier
    TIER_CUTOFFS = cfg.tier_cutoffs
    NUM_TOTAL_SIMULATIONS = cfg.num_total_simulations
    MAX_TASKS = cfg.max_tasks
    SEED = cfg.seed
    CACHE_SAMPLING_SEED = cfg.cache_sampling_seed
    SPLIT_SEED = cfg.split_seed
    # Independently-seeded RNG sources — vary one to isolate that source's
    # contribution to run-to-run variance (e.g., NPE-init reseed audits).
    NPE_SEED = cfg.npe_seed
    NOISE_SEED = cfg.noise_seed
    COHORT_SEED = cfg.cohort_seed
    RUN_POSTERIOR_PREDICTIVE = cfg.run_posterior_predictive
    SAVE_EVOLVE_TRAJECTORIES = cfg.save_evolve_trajectories
    EVOLVE_TRAJECTORY_DT_DAYS = cfg.evolve_trajectory_dt_days
    MISSPEC_DETECTION_MODE = cfg.misspec_detection_mode
    LLM_ELICITED_TARGETS_CSV = cfg.llm_elicited_targets_csv
    N_PLOT_PARAMS = cfg.n_plot_params
    N_POSTERIOR_SIMS = cfg.n_posterior_sims
    NPE_HIDDEN_FEATURES = cfg.npe_hidden_features
    NPE_NUM_TRANSFORMS = cfg.npe_num_transforms
    NPE_NUM_BINS = cfg.npe_num_bins
    NPE_TRAINING_BATCH_SIZE = cfg.npe_training_batch_size
    NPE_MAX_EPOCHS = cfg.npe_max_epochs
    PRETRAINED_DIR = cfg.pretrained_dir
    SAMPLE_ON_DELTA = cfg.sample_on_delta
    EVAL_NPE_THETA_SPACE = cfg.eval_npe_theta_space
    if SAMPLE_ON_DELTA and not EVAL_NPE_THETA_SPACE:
        raise ValueError(
            "sample_on_delta=True is incompatible with eval_npe_theta_space=False "
            "(the Delta path bundles step 5 with step 6). Pass --eval-npe-theta-space "
            "or drop --sample-on-delta."
        )
    CACHE_PROCESSED_TENSORS = cfg.cache_processed_tensors
    STAGE_ARTIFACTS_ONLY = cfg.stage_artifacts_only
    SUBMODEL_PRIORS_YAML = cfg.submodel_priors_yaml
    N_TRAIN_SUBSAMPLE = cfg.n_train_subsample
    # C++ backend config
    SCENARIOS_DIR = cfg.scenarios_dir
    CPP_BINARY_PATH = cfg.cpp_binary_path
    CPP_TEMPLATE_XML = cfg.cpp_template_xml
    CPP_SUBTREE = cfg.cpp_subtree
    CPP_DRUG_METADATA_YAML = cfg.cpp_drug_metadata_yaml
    CPP_HEALTHY_STATE_YAML = cfg.cpp_healthy_state_yaml
    CPP_T_END_DAYS = cfg.cpp_t_end_days
    CPP_DT_DAYS = cfg.cpp_dt_days
    RESTRICTION_CLASSIFIER_DIR = cfg.restriction_classifier_dir
    RESTRICTION_THRESHOLD = cfg.restriction_threshold
    CLASSIFIER_FEATURE_FILLS = cfg.classifier_feature_fills or None
    # NPE init: torch RNG. Other np.random sites use their own seeds
    # (NOISE_SEED, COHORT_SEED, SPLIT_SEED) — the global np.random.seed(SEED)
    # is a catch-all for any incidental np.random uses downstream.
    torch.manual_seed(NPE_SEED)
    np.random.seed(SEED)

    scenario_names = list(SCENARIOS.keys())
    n_scenarios = len(scenario_names)
    is_multi = n_scenarios > 1

    tier_suffix = f'_tier{PRCC_TIER}' if PRCC_TIER is not None else ''
    null_suffix = '_null' if cfg.null_shuffle else ''
    if cfg.save_dir is not None:
        SAVE_DIR = Path(cfg.save_dir)
    elif is_multi:
        SAVE_DIR = Path(f'figures/sbi_multi{tier_suffix}{null_suffix}')
    else:
        SAVE_DIR = Path(f'figures/sbi_{scenario_names[0]}{tier_suffix}{null_suffix}')
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    tier_label = f"PRCC TIER {PRCC_TIER} ({TIER_CUTOFFS[PRCC_TIER]} PARAMS)" if PRCC_TIER is not None else "ALL PARAMETERS"
    scenario_label = ' + '.join(scenario_names)
    print(f"SBI WORKFLOW: {scenario_label} — {tier_label}")
    print("=" * 80)

    # ============================================================================
    # 1b. LOAD PRIOR (composite from submodel YAML + CSV, OR CSV-only)
    # ============================================================================
    if cfg.skip_submodel_sync:
        print("  [skip_submodel_sync=True] CSV-only prior; submodel_priors.yaml ignored.")
        prior_log_full, prior_param_names = _load_csv_only_prior_log(PRIORS_CSV)
        # Drop submodel_priors_yaml from the simulator pool key so a CSV-only
        # run doesn't share a pool with composite-prior runs.
        SUBMODEL_PRIORS_YAML = None
    else:
        SUBMODEL_PRIORS_YAML = cfg.submodel_priors_yaml
        _check_submodel_priors_freshness(
            yaml_path=Path(SUBMODEL_PRIORS_YAML),
            submodel_dir=Path('calibration_targets/submodel_targets'),
            priors_csv=Path(PRIORS_CSV),
            strict=cfg.strict_submodel_freshness,
        )
        prior_log_full, prior_param_names = load_composite_prior_log(SUBMODEL_PRIORS_YAML, PRIORS_CSV)
    print(f"  Prior: {len(prior_param_names)} params (QSP)")

    # --- 1c. Auxiliary parameters (measurement-bridging) ---
    # Walk every scenario's calibration_target dirs (plus cross-scenario
    # dirs) for observable.auxiliary_parameters declarations and validate
    # them against the structural prior in auxiliary_config.yaml. An empty
    # registry is a no-op — preserves pre-aux runner semantics so this
    # change is byte-equivalent for runs that don't declare any aux
    # parameters.
    aux_config = load_auxiliary_config(cfg.auxiliary_config_path)
    _aux_roots: list[Path] = []
    for _scen_cfg in SCENARIOS.values():
        for _d in _scen_cfg.get('calibration_targets', []):
            _aux_roots.append(Path(_d))
    for _d in (cfg.cross_scenario_targets or []):
        _aux_roots.append(Path(_d))
    aux_registry = discover_auxiliary_members(_aux_roots, aux_config)
    if not aux_registry.is_empty:
        # Build the hierarchical aux prior, draw one record per sample_index
        # (deterministic on cache_sampling_seed so reruns produce identical
        # aux draws — matches the trajectory pool's sample_index alignment),
        # and write a sidecar CSV that the MSR uploads to HPC. The derive
        # worker joins the CSV into species_dict per sim before evaluating
        # each cal target's observable.code; aux is purely a derive-stage
        # concern and the trajectory pool stays aux-independent.
        aux_prior = HierarchicalAuxiliaryPrior(aux_registry)
        aux_member_names = list(aux_prior.param_names)
        n_groups = len(aux_registry.group_names)
        n_members = len(aux_member_names)
        print(
            f"  Auxiliary parameters: {n_members} member(s) across {n_groups} group(s)"
        )
        for _g in aux_registry.group_names:
            for _m in aux_registry.members_in(_g):
                print(f"      - {_m.name} (group={_m.group}, units={_m.units})")

        aux_records = aux_prior.sample_as_records(
            NUM_TOTAL_SIMULATIONS, seed=CACHE_SAMPLING_SEED
        )
        aux_units_by_name = {m.name: m.units for g in aux_registry.group_names
                             for m in aux_registry.members_in(g)}
        aux_df = pd.DataFrame(
            {
                'sample_index': np.arange(NUM_TOTAL_SIMULATIONS, dtype=np.int64),
                **{
                    name: np.array([rec[name] for rec in aux_records], dtype=np.float64)
                    for name in aux_member_names
                },
            }
        )
        aux_csv_path = Path('cache') / (
            f'aux_samples_n{NUM_TOTAL_SIMULATIONS}_s{CACHE_SAMPLING_SEED}.csv'
        )
        aux_csv_path.parent.mkdir(parents=True, exist_ok=True)
        aux_df.to_csv(aux_csv_path, index=False)
        print(f"  Aux samples CSV: {aux_csv_path} ({len(aux_df)} rows)")
        # Numpy view of aux draws keyed by sample_index (row i = sim i).
        # NPE training θ_full[i] = [θ_qsp[i], θ_aux[i]] aligned by sample_index.
        aux_array = aux_df[aux_member_names].to_numpy(dtype=np.float64)
        # Validate every aux block is lognormal — Phase B currently
        # log-transforms the aux block alongside QSP for NPE training, which
        # is exact for lognormal blocks (block.mvn already lives in log
        # space) but invalid for normal-base blocks (negative aux values
        # would break np.log). When a normal-base aux is needed, switch
        # the joint prior to a mixed-space form below.
        for _block in aux_prior._blocks:
            if _block.transform != "exp":
                raise NotImplementedError(
                    f"Aux group {_block.group_name!r} has non-lognormal base "
                    f"({_block.transform}); Phase B joint NPE currently "
                    "supports lognormal aux only. Either restrict the registry "
                    "to lognormal groups or extend the JointPrior wrapper."
                )
    else:
        aux_prior = None
        aux_member_names = []
        aux_records = []
        aux_units_by_name = {}
        aux_csv_path = None
        aux_array = None
        print(f"  Auxiliary parameters: 0 members (registry empty; no joint composition)")

    # ============================================================================
    # 2. LOAD & PROCESS SIMULATIONS
    # ============================================================================
    print("\n" + "=" * 80)
    print("2. LOAD & PROCESS SIMULATIONS")
    print("=" * 80)

    # --- Load cached processed tensors if available ---
    _step2_cache_path = _processed_cache_path(cfg) if CACHE_PROCESSED_TENSORS else None
    _step2_loaded = False
    if CACHE_PROCESSED_TENSORS:
        if MISSPEC_DETECTION_MODE or RUN_POSTERIOR_PREDICTIVE:
            raise ValueError(
                "--cache-processed-tensors is incompatible with "
                "misspec_detection_mode or run_posterior_predictive "
                "(scenario_meta['simulator'] is not cached)."
            )
        if _step2_cache_path.exists():
            print(f"  Loading processed bundle from {_step2_cache_path}")
            _bundle = _load_processed_bundle(_step2_cache_path)
            theta_train = _bundle['theta_train']
            theta_test = _bundle['theta_test']
            x_train = _bundle['x_train']
            x_test = _bundle['x_test']
            x_train_raw = _bundle['x_train_raw']
            obs_transformed = _bundle['obs_transformed']
            obs_transformed_tensor = _bundle['obs_transformed_tensor']
            obs_original = _bundle['obs_original']
            obs_values_cat = _bundle['obs_values_cat']
            observable_names = _bundle['observable_names']
            num_observables = _bundle['num_observables']
            target_df = _bundle['target_df']
            param_names = _bundle['param_names']
            param_names_full = _bundle['param_names_full']
            num_params = _bundle['num_params']
            num_params_full = _bundle['num_params_full']
            param_indices = _bundle['param_indices']
            scenario_meta = _bundle['scenario_meta']
            obs_diag_transformed = _bundle['obs_diag_transformed']
            obs_diag_raw = _bundle['obs_diag_raw']
            obs_diag_original = _bundle['obs_diag_original']
            priors_df = _bundle['priors_df']
            _step2_loaded = True

    if not _step2_loaded:
        # --- 2a. Create per-scenario simulators and run simulations ---
        # Each scenario gets its own CppSimulator (same priors + binary, different
        # calibration targets + scenario YAML). One shared HPCJobManager so
        # scenarios reuse the SSH transport and venv-bootstrap probe.
        # cache_sampling_seed isn't directly threaded (CppSimulator's theta
        # pool is deterministic on `seed` alone), but identical theta draws
        # across scenarios still hold because the same seed produces the
        # same first N draws from theta_pool.
        scenario_meta = {}  # stores per-scenario data for downstream steps
        Path('cache').mkdir(exist_ok=True)

        job_manager = HPCJobManager(verbose=True)

        # Branch overrides applied in-place on the loaded credentials. Same
        # pattern as scripts/smoke_sbi_hpc.py in qsp-hpc-tools.
        if cfg.cpp_branch:
            job_manager.config.cpp_branch = cfg.cpp_branch
        if cfg.tools_branch:
            src = job_manager.config.qsp_hpc_tools_source
            if '.git' in src:
                head, _sep, _tail = src.partition('.git')
                job_manager.config.qsp_hpc_tools_source = f'{head}.git@{cfg.tools_branch}'
            else:
                job_manager.config.qsp_hpc_tools_source = f'{src}@{cfg.tools_branch}'
        if cfg.codegen_branch:
            src = job_manager.config.cpp_codegen_source
            if '.git' in src:
                head, _sep, _tail = src.partition('.git')
                job_manager.config.cpp_codegen_source = f'{head}.git@{cfg.codegen_branch}'
            else:
                job_manager.config.cpp_codegen_source = f'{src}@{cfg.codegen_branch}'
        if cfg.cpp_branch or cfg.tools_branch or cfg.codegen_branch:
            print(
                f'  Overriding refs: cpp_branch={job_manager.config.cpp_branch}, '
                f'qsp_hpc_tools_source={job_manager.config.qsp_hpc_tools_source}, '
                f'cpp_codegen_source={job_manager.config.cpp_codegen_source}'
            )
        if cfg.cpp_jobs_per_chunk is not None:
            job_manager.config.jobs_per_chunk = cfg.cpp_jobs_per_chunk
            print(f'  jobs_per_chunk override: {cfg.cpp_jobs_per_chunk}')

        # Phase 1: build all CppSimulators (fast — no HPC calls).
        sims = {}
        for scen_name in scenario_names:
            scen_cfg = SCENARIOS[scen_name]
            scenario_yaml = scen_cfg.get(
                'scenario_yaml', f'{SCENARIOS_DIR}/{scen_name}.yaml'
            )
            sims[scen_name] = CppSimulator(
                priors_csv=PRIORS_CSV,
                binary_path=CPP_BINARY_PATH,
                template_xml=CPP_TEMPLATE_XML,
                subtree=CPP_SUBTREE,
                model_version=MODEL_VERSION,
                scenario=scen_name,
                submodel_priors_yaml=SUBMODEL_PRIORS_YAML,
                scenario_yaml=scenario_yaml,
                drug_metadata_yaml=CPP_DRUG_METADATA_YAML,
                healthy_state_yaml=CPP_HEALTHY_STATE_YAML,
                calibration_targets=scen_cfg['calibration_targets'],
                model_structure_file=MODEL_STRUCTURE_FILE,
                job_manager=job_manager,
                t_end_days=CPP_T_END_DAYS,
                min_cadence_hours=4.0,
                seed=SEED,
                # Size the theta pool to the request — CppSimulator defaults
                # to 100k which mis-matches any NUM_TOTAL_SIMULATIONS > 100k
                # and blows up inside theta_for_indices with IndexError.
                theta_pool_size=max(NUM_TOTAL_SIMULATIONS, 100_000),
                restriction_classifier_dir=RESTRICTION_CLASSIFIER_DIR,
                restriction_threshold=RESTRICTION_THRESHOLD,
                classifier_feature_fills=CLASSIFIER_FEATURE_FILLS,
                # LMDB-on-NFS write-lock contention deadlocks the per-task
                # workers at fan-out >~50; the cache only buys ~10-15%
                # wall time at this scale, not worth the recurring hangs.
                # Re-enable once cache moves off NFS or fan-out drops.
                evolve_cache=False,
                verbose=True,
            )

        # Param-order invariant: simulator.param_names, prior_param_names,
        # and priors_df['name'] must agree (all read the same priors CSV in
        # row order). A divergence here scrambles every NPE training column
        # against its label silently — guard the run before sbatch gets
        # involved. Cheap (just list equality), runs once per workflow.
        _csv_names = list(pd.read_csv(PRIORS_CSV)['name'])
        for _scen, _sim in sims.items():
            if list(_sim.param_names) != _csv_names:
                raise RuntimeError(
                    f"param_names order mismatch: sims['{_scen}'].param_names "
                    f"diverges from {PRIORS_CSV} row order. Likely cause: "
                    f"simulator and runner reading different priors files."
                )
        if list(prior_param_names) != _csv_names:
            raise RuntimeError(
                f"param_names order mismatch: prior_param_names diverges from "
                f"{PRIORS_CSV} row order. Likely cause: composite-prior loader "
                f"received a different CSV than the simulator did."
            )
        del _csv_names

        # Phase 2: HPC fan-out via MultiScenarioRunner. The runner hoists
        # one upload of the shared (sample_index, theta) CSV, then submits
        # each scenario with samples_csv_remote= pointing at it and
        # skip_setup=True after the first. CppSimulator's existing 3-tier
        # cache + top-up logic stays intact per scenario.
        msr = MultiScenarioRunner(sims)
        if aux_csv_path is not None:
            msr.attach_auxiliary_samples(aux_csv_path, aux_units_by_name)
        msr_results = msr.run_all(NUM_TOTAL_SIMULATIONS)
        run_results = {name: (r.theta, r.x) for name, r in msr_results.items()}

        # Phase 3: post-process (fast, serial). Extract sample_index, load
        # target_df, replace inf with NaN, and populate scenario_meta in
        # the canonical scenario_names order.
        for scen_name in scenario_names:
            sim = sims[scen_name]
            theta_scen, x_scen = run_results[scen_name]
            if getattr(sim, 'last_sample_index', None) is not None and len(sim.last_sample_index) == len(theta_scen):
                sample_index_scen = sim.last_sample_index.astype(np.int64)
            else:
                print(f"  WARNING: {scen_name} pool has no sample_index column — "
                      f"falling back to positional index (cross-scenario alignment "
                      f"may fail)")
                sample_index_scen = np.arange(len(theta_scen), dtype=np.int64)
            print(f"  {scen_name}: theta {theta_scen.shape}, x {x_scen.shape}")

            target_df_scen = pd.read_csv(sim.test_stats_csv)
            obs_names_scen = target_df_scen['test_statistic_id'].tolist()

            # Cohort random-effect: attach source_tag → cohort_id from YAMLs.
            # Always attached (cheap), but only consumed when cfg.tau_cohort>0.
            scen_cfg = SCENARIOS.get(scen_name, {})
            cal_dirs = scen_cfg.get('calibration_targets', [])
            target_df_scen = attach_cohort_ids(target_df_scen, cal_dirs)
            if cfg.tau_cohort > 0:
                print_cohort_table(target_df_scen, label=scen_name)

            csv_path = f'cache/{scen_name}_calibration_targets_summary.csv'
            target_df_scen.to_csv(csv_path, index=False)

            n_inf = np.isinf(x_scen).sum()
            if n_inf > 0:
                print(f"  Replacing {n_inf} inf values with NaN in {scen_name}")
                x_scen[np.isinf(x_scen)] = np.nan

            scenario_meta[scen_name] = {
                'simulator': sim,
                'theta': theta_scen,
                'x_raw': x_scen,
                'sample_index': sample_index_scen,
                'target_df': target_df_scen,
                'observable_names': obs_names_scen,
                'summary_csv': csv_path,
            }

        # --- 2b. Per-observable NaN audit (diagnostic) ---
        print("\n  Per-observable NaN rates (obs with >5% NaN shown):")
        nan_audit_rows = []
        for scen_name in scenario_names:
            x_scen = scenario_meta[scen_name]['x_raw']
            obs_names = scenario_meta[scen_name]['observable_names']
            nan_frac = np.mean(~np.isfinite(x_scen), axis=0)
            any_nan = np.mean(~np.all(np.isfinite(x_scen), axis=1))
            print(f"  [{scen_name}] row-level NaN-any rate: {any_nan:.1%}")
            ranked = sorted(enumerate(nan_frac), key=lambda t: -t[1])
            for i, frac in ranked:
                nan_audit_rows.append({'scenario': scen_name, 'observable': obs_names[i], 'nan_frac': frac})
                if frac > 0.05:
                    print(f"    {obs_names[i]}: {frac:.1%}")
        pd.DataFrame(nan_audit_rows).to_csv('cache/nan_audit.csv', index=False)
        print("  Wrote cache/nan_audit.csv")

        # --- 2c. Align scenarios by sample_index before joint NaN filter ---
        # Scenarios may have different on-disk row orderings even when they share
        # the same theta pool. Intersect on sample_index so each retained row is
        # the same theta across all scenarios (structurally required for joint
        # inference). Rebases scenario_meta[*]['x_raw'] and ['theta'] to the
        # shared-index view.
        shared_idx = None
        for scen_name in scenario_names:
            idx = scenario_meta[scen_name]['sample_index']
            s = set(idx.tolist())
            shared_idx = s if shared_idx is None else (shared_idx & s)
        shared_arr = np.array(sorted(shared_idx), dtype=np.int64)
        print(f"\n  Shared sample_indices across scenarios: {len(shared_arr)}")

        theta_ref = None
        for scen_name in scenario_names:
            idx = scenario_meta[scen_name]['sample_index']
            # Pool may carry duplicate sample_indices (e.g. a top-up batch that
            # overlapped with existing chunks, or a smoke batch left in the pool).
            # The set-based shared_arr already collapses within-scenario dupes;
            # mirror that by keeping the first occurrence here.
            pos = pd.Series(np.arange(len(idx)), index=idx)
            if pos.index.has_duplicates:
                n_dup = int(pos.index.duplicated().sum())
                print(f"  [{scen_name}] dropping {n_dup} duplicate sample_index rows (keeping first)")
                pos = pos[~pos.index.duplicated(keep='first')]
            order = pos.reindex(shared_arr).to_numpy().astype(np.int64)
            theta_aligned = scenario_meta[scen_name]['theta'][order]
            x_aligned = scenario_meta[scen_name]['x_raw'][order]
            if theta_ref is None:
                theta_ref = theta_aligned
            elif not np.allclose(theta_ref, theta_aligned, equal_nan=True):
                raise RuntimeError(
                    f"Theta mismatch between scenarios at shared sample_indices "
                    f"(scenario={scen_name}). Pools disagree on what theta a given "
                    f"sample_index represents — likely a stale pre-theta-pool parquet."
                )
            scenario_meta[scen_name]['theta'] = theta_aligned
            scenario_meta[scen_name]['x_raw'] = x_aligned
            scenario_meta[scen_name]['sample_index'] = shared_arr

        # --- 2c.bis. Cross-scenario compose (Phase D-2 / Option D3+) ---
        # Runs only when cfg.cross_scenario_targets is non-empty. Loads
        # CrossScenarioCalibrationTarget YAMLs, composes them over the
        # aligned scenario_meta, and registers the result as a virtual
        # "_cross_scenario" entry that the downstream joint NaN filter +
        # per-scenario noise/copula loop processes the same way as a real
        # scenario. The dispatch is gated, so existing runs without
        # cross-scenario YAMLs are byte-identical to pre-D3+ behavior.
        cross_scenario_dirs = list(cfg.cross_scenario_targets or [])
        if cross_scenario_dirs:
            from qsp_hpc.batch.cross_scenario_worker import compute_cross_scenario_statistics
            from qsp_hpc.calibration import load_cross_scenario_targets
            from qsp_hpc.utils.model_structure_units import load_units_from_model_structure

            cross_dfs = [load_cross_scenario_targets(d) for d in cross_scenario_dirs]
            cross_df = pd.concat(cross_dfs, ignore_index=True)
            print(f"\n  Cross-scenario targets: {len(cross_df)} from {len(cross_scenario_dirs)} dir(s)")

            # Refuse timeseries inputs until run_hpc() returns sim_dfs as
            # well as x. The cross_scenario_worker fully supports the
            # timeseries path; the bridge from HPC parquets to a
            # scenario_meta[*]['sim_df'] is the missing piece. Fail loudly
            # rather than silently NaN every row.
            import json as _json
            for _, row in cross_df.iterrows():
                inputs_spec = _json.loads(row['inputs_json'])
                ts_inputs = [inp for inp in inputs_spec if inp['input_kind'] == 'timeseries']
                if ts_inputs:
                    raise NotImplementedError(
                        f"Cross-scenario target '{row['cross_scenario_target_id']}' "
                        f"declares timeseries inputs (roles: "
                        f"{[i['role'] for i in ts_inputs]}). The composer worker "
                        "supports them, but sbi_runner doesn't yet ferry full "
                        "sim_dfs back from HPC — see notes/calibration/"
                        "cross_scenario_observable_spike.md follow-ups."
                    )

            species_units = load_units_from_model_structure(MODEL_STRUCTURE_FILE)
            cross_x = compute_cross_scenario_statistics(scenario_meta, cross_df, species_units)

            # Build a target_df + summary_csv shaped like a per-scenario
            # CalibrationTarget summary so the downstream loop body
            # (line ~792 onward) can consume it without special-casing.
            cross_target_df = pd.DataFrame({
                'test_statistic_id': cross_df['cross_scenario_target_id'],
                'required_species': '',  # cross-scenario observables don't expose required_species
                'model_output_code': '',  # already executed; no per-row wrapper code
                'median': cross_df['median'],
                'ci95_lower': cross_df['ci95_lower'],
                'ci95_upper': cross_df['ci95_upper'],
                'units': cross_df['units'],
                'sample_size': cross_df['sample_size'],
            })
            cross_csv_path = 'cache/_cross_scenario_calibration_targets_summary.csv'
            cross_target_df.to_csv(cross_csv_path, index=False)

            scenario_meta['_cross_scenario'] = {
                'simulator': None,  # no per-scenario simulator; cross is purely derived
                'theta': theta_ref,  # aligned theta; the joint-NaN block uses theta_ref directly
                'x_raw': cross_x,
                'sample_index': shared_arr,
                'target_df': cross_target_df,
                'observable_names': cross_df['cross_scenario_target_id'].tolist(),
                'summary_csv': cross_csv_path,
            }
            scenario_names.append('_cross_scenario')
            # Force scenario-name prefixing on observables: with a cross
            # block, `<real_scen>/<id>` and `_cross_scenario/<id>` need to
            # be distinguishable downstream even if the user only listed
            # one real scenario.
            is_multi = True
            print(f"  Cross-scenario x: shape {cross_x.shape}, "
                  f"finite rows {int(np.all(np.isfinite(cross_x), axis=1).sum())}/{cross_x.shape[0]}")

        # --- 2d. Joint NaN filter across all scenarios ---
        theta_raw = theta_ref
        n_sims = theta_raw.shape[0]

        valid_mask_all = np.ones(n_sims, dtype=bool)
        for scen_name in scenario_names:
            x_scen = scenario_meta[scen_name]['x_raw']
            valid_mask_all &= np.all(np.isfinite(x_scen), axis=1)

        theta_clean = theta_raw[valid_mask_all]
        n_survived = len(theta_clean)
        print(f"\n  Joint NaN filter: {n_survived}/{n_sims} survived ({n_survived/n_sims:.0%})")

        for scen_name in scenario_names:
            scenario_meta[scen_name]['x_clean'] = scenario_meta[scen_name]['x_raw'][valid_mask_all]

        # --- 2c. Parameter setup + PRCC tier filtering ---
        priors_df = pd.read_csv(PRIORS_CSV)
        param_names = priors_df['name'].tolist()
        num_params_full = len(param_names)

        if PRCC_TIER is not None:
            ranking_df = pd.read_csv(PRCC_RANKING_CSV)
            n_keep = TIER_CUTOFFS[PRCC_TIER]
            selected_params = set(ranking_df[ranking_df['rank_prcc'] <= n_keep]['parameter'].tolist())
            param_indices = [i for i, n in enumerate(param_names) if n in selected_params]
            param_names_full = param_names
            param_names = [param_names[i] for i in param_indices]
            num_params = len(param_names)
            theta_clean = theta_clean[:, param_indices]
            print(f"  PRCC tier {PRCC_TIER}: {num_params}/{num_params_full} parameters (rank <= {n_keep})")
            print(f"  Top 5: {', '.join(param_names[:5])}")
        else:
            param_indices = None
            param_names_full = param_names
            num_params = num_params_full

        # Phase B: append aux block to θ. Aux dimensions are always kept
        # (no PRCC tier filtering of aux — measurement bridges have no
        # PRCC ranking in the QSP-trained ranker, and dropping them would
        # un-do the joint inference). theta_clean rows are aligned to
        # scenario_meta[*]['sample_index'] = shared_arr after the
        # cross-scenario alignment block; aux_array rows are indexed
        # 0..N-1 by sample_index, so slice by shared_arr first then
        # apply valid_mask_all to match theta_clean.
        if aux_member_names:
            shared_arr = scenario_meta[scenario_names[0]]['sample_index']
            aux_clean = aux_array[shared_arr][valid_mask_all]
            theta_clean = np.concatenate([theta_clean, aux_clean], axis=1)
            param_names = list(param_names) + list(aux_member_names)
            param_names_full = list(param_names_full) + list(aux_member_names)
            if param_indices is not None:
                # Aux dims occupy positions [num_params_full_qsp ..
                # num_params_full_qsp + n_aux) of the full vector. The
                # subset operation below only filters the QSP block, so
                # extend param_indices to keep aux positions.
                _n_qsp_full = num_params_full
                param_indices = list(param_indices) + list(
                    range(_n_qsp_full, _n_qsp_full + len(aux_member_names))
                )
            num_params = len(param_names)
            num_params_full = len(param_names_full)
            print(
                f"  Joint θ: {num_params} params total "
                f"({num_params - len(aux_member_names)} QSP + {len(aux_member_names)} aux)"
            )

        # --- 2d. Shared train/test split ---
        # Replicate _split_data's shuffling (np.random.seed + np.random.permutation) to get indices
        np.random.seed(SPLIT_SEED)
        n_total = len(theta_clean)
        split_indices = np.random.permutation(n_total)
        n_test = int(n_total * 0.1)
        n_train = n_total - n_test
        train_idx = split_indices[:n_train]
        test_idx = split_indices[n_train:]
        print(f"  Train/test split: {n_train} train (90%), {n_test} test (10%)")

        if N_TRAIN_SUBSAMPLE is not None and N_TRAIN_SUBSAMPLE < n_train:
            rng_sub = np.random.default_rng(SPLIT_SEED + 1)
            keep = rng_sub.choice(n_train, size=N_TRAIN_SUBSAMPLE, replace=False)
            train_idx = train_idx[keep]
            n_train = N_TRAIN_SUBSAMPLE
            print(f'  Train subsample: {n_train} rows (seed {SPLIT_SEED + 1})')

        theta_train_np = theta_clean[train_idx]
        theta_test_np = theta_clean[test_idx]

        # --- 2e. Per-scenario: noise injection, copula transform, obs data ---
        # Each scenario gets its own copula quantiles (different distributions).
        # Final training data is concatenated across scenarios in copula space.
        x_train_parts = []
        x_test_parts = []
        x_train_raw_parts = []
        observable_names_all = []
        target_df_parts = []
        obs_original_all = {}
        obs_values_parts = []

        for scen_name in scenario_names:
            meta = scenario_meta[scen_name]
            x_clean_scen = meta['x_clean']
            target_df_scen = meta['target_df']
            obs_names_scen = meta['observable_names']

            x_train_scen = x_clean_scen[train_idx]
            x_test_scen = x_clean_scen[test_idx]
            x_train_raw_scen = x_train_scen.copy()

            # Add observation noise
            x_train_noisy = add_observation_noise(
                x_train_scen,
                ci95_lower=target_df_scen['ci95_lower'].values,
                ci95_upper=target_df_scen['ci95_upper'].values,
                medians=target_df_scen['median'].values,
                seed=NOISE_SEED,
                sample_sizes=target_df_scen['sample_size'].values,
            )

            # Compute per-scenario copula quantiles from noisy training data
            quantiles_scen = compute_quantiles_from_array(x_train_noisy, n_quantiles=1000)
            x_train_t = transform_to_normal_from_array(x_train_noisy, quantiles_scen)
            x_test_t = transform_to_normal_from_array(x_test_scen, quantiles_scen)

            # Cohort random-effect: per-cohort additive shift in copula-Gauss
            # space, applied after the copula transform so the shift is
            # unit-agnostic. Same draw applied to train + test so they stay
            # in the same conditional distribution. x_obs is NOT shifted —
            # it's already one realization of cohort offsets.
            if cfg.tau_cohort > 0:
                cohort_ids_scen = target_df_scen['cohort_id'].values
                x_train_t = apply_cohort_shift(
                    x_train_t, cohort_ids_scen,
                    cfg.tau_cohort, seed=COHORT_SEED,
                )
                x_test_t = apply_cohort_shift(
                    x_test_t, cohort_ids_scen,
                    cfg.tau_cohort, seed=COHORT_SEED + 1,
                )

            # Load observed data for this scenario
            obs_scen = get_observed_data(meta['summary_csv'], value_column='median')
            obs_vals_scen = np.array([np.squeeze(obs_scen[name]) for name in obs_names_scen])
            obs_transformed_scen = transform_to_normal_from_array(
                obs_vals_scen[np.newaxis, :], quantiles_scen
            )[0]

            # Prefix observable names for multi-scenario to avoid collisions
            if is_multi:
                prefixed_names = [f'{scen_name}/{name}' for name in obs_names_scen]
            else:
                prefixed_names = obs_names_scen

            # Store
            meta['quantiles'] = quantiles_scen
            meta['x_train_raw'] = x_train_raw_scen
            meta['x_train_transformed'] = x_train_t
            meta['x_test_transformed'] = x_test_t
            meta['obs_original'] = obs_scen
            meta['obs_values'] = obs_vals_scen
            meta['obs_transformed'] = obs_transformed_scen
            meta['prefixed_names'] = prefixed_names

            x_train_parts.append(x_train_t)
            x_test_parts.append(x_test_t)
            x_train_raw_parts.append(x_train_raw_scen)
            observable_names_all.extend(prefixed_names)
            obs_values_parts.append(obs_vals_scen)
            obs_original_all.update({
                (f'{scen_name}/{k}' if is_multi else k): v
                for k, v in obs_scen.items()
            })

            # Build combined target_df with scenario prefix
            tdf = target_df_scen.copy()
            if is_multi:
                tdf['test_statistic_id'] = tdf['test_statistic_id'].apply(lambda n: f'{scen_name}/{n}')
            target_df_parts.append(tdf)

            print(f"  {scen_name}: {len(obs_names_scen)} observables, "
                  f"train {x_train_t.shape}, test {x_test_t.shape}")

        # Concatenate across scenarios
        x_train_cat = np.concatenate(x_train_parts, axis=1)
        x_test_cat = np.concatenate(x_test_parts, axis=1)
        x_train_raw_cat = np.concatenate(x_train_raw_parts, axis=1)
        obs_values_cat = np.concatenate(obs_values_parts)
        obs_transformed_cat = np.concatenate([meta['obs_transformed'] for meta in scenario_meta.values()])

        # Combined target_df
        target_df = pd.concat(target_df_parts, ignore_index=True)
        observable_names = observable_names_all
        num_observables = len(observable_names)

        # Convert to tensors
        theta_train = torch.tensor(np.log(theta_train_np), dtype=torch.float32)
        theta_test = torch.tensor(np.log(theta_test_np), dtype=torch.float32)
        x_train = torch.tensor(x_train_cat, dtype=torch.float32)
        x_test = torch.tensor(x_test_cat, dtype=torch.float32)
        obs_transformed = obs_transformed_cat
        obs_transformed_tensor = torch.tensor(obs_transformed, dtype=torch.float32).unsqueeze(0)

        # Convenience aliases
        obs_original = obs_original_all
        x_train_raw = x_train_raw_cat

        # --- Misspecification detection mode: swap diagnostic reference to LLM-elicited targets ---
        if MISSPEC_DETECTION_MODE:
            print(f"\n  MISSPEC DETECTION MODE: using LLM-elicited targets for diagnostics")
            # Strip scenario prefixes for lookup (multi-scenario uses 'scenario/observable')
            obs_names_stripped = [n.split('/')[-1] for n in observable_names]
            obs_llm_raw = load_llm_elicited_targets(LLM_ELICITED_TARGETS_CSV, obs_names_stripped)

            # Copula-transform LLM values using the same per-scenario quantiles
            obs_llm_transformed_parts = []
            offset = 0
            for scen_name in scenario_names:
                meta = scenario_meta[scen_name]
                n_obs_scen = len(meta['observable_names'])
                llm_vals_scen = obs_llm_raw[offset:offset + n_obs_scen]
                llm_transformed_scen = transform_to_normal_from_array(
                    llm_vals_scen[np.newaxis, :], meta['quantiles']
                )[0]
                obs_llm_transformed_parts.append(llm_transformed_scen)
                offset += n_obs_scen
            obs_diag_transformed = np.concatenate(obs_llm_transformed_parts)
            obs_diag_raw = obs_llm_raw
            obs_diag_original = {name: obs_llm_raw[i] for i, name in enumerate(observable_names)}

            # In misspec mode, NPE conditioning also uses LLM targets (no real data at all)
            obs_transformed = obs_diag_transformed
            obs_transformed_tensor = torch.tensor(obs_diag_transformed, dtype=torch.float32).unsqueeze(0)

            # Show comparison
            print(f"  LLM-elicited vs real targets (first 5):")
            for i in range(min(5, num_observables)):
                print(f"    {observable_names[i]}: LLM={obs_llm_raw[i]:.3g}, real={obs_values_cat[i]:.3g}")
        else:
            obs_diag_transformed = obs_transformed
            obs_diag_raw = obs_values_cat
            obs_diag_original = obs_original

        # --- Save processed bundle for reuse across sweep cells ---
        if CACHE_PROCESSED_TENSORS:
            _bundle = {
                'theta_train': theta_train,
                'theta_test': theta_test,
                'x_train': x_train,
                'x_test': x_test,
                'x_train_raw': x_train_raw,
                'obs_transformed': obs_transformed,
                'obs_transformed_tensor': obs_transformed_tensor,
                'obs_original': obs_original,
                'obs_values_cat': obs_values_cat,
                'observable_names': observable_names,
                'num_observables': num_observables,
                'target_df': target_df,
                'param_names': param_names,
                'param_names_full': param_names_full,
                'num_params': num_params,
                'num_params_full': num_params_full,
                'param_indices': param_indices,
                'obs_diag_transformed': obs_diag_transformed,
                'obs_diag_raw': obs_diag_raw,
                'obs_diag_original': obs_diag_original,
                'priors_df': priors_df,
                # Reduced scenario_meta: drop simulator (unpicklable) and
                # intermediate arrays; keep fields downstream non-PP /
                # non-misspec code paths read.
                'scenario_meta': {
                    name: {
                        'quantiles': meta['quantiles'],
                        'x_train_raw': meta['x_train_raw'],
                        'obs_transformed': meta['obs_transformed'],
                        'observable_names': meta['observable_names'],
                        'prefixed_names': meta['prefixed_names'],
                    }
                    for name, meta in scenario_meta.items()
                },
            }
            _save_processed_bundle(_step2_cache_path, _bundle)
            print(f"  Saved processed bundle to {_step2_cache_path}")

    print(f"\n  Concatenated: {num_observables} observables across {n_scenarios} scenarios")
    print(f"  Parameters: {num_params}, Observables: {num_observables}")
    print(f"  Train: theta {theta_train.shape}, x {x_train.shape}")
    print(f"  Test:  theta {theta_test.shape}, x {x_test.shape}")

    # --- Write training artifacts early (enables --stage-artifacts-only and
    # Delta GPU training before the local process moves on to steps 3-7). ---
    artifacts_dir = SAVE_DIR / 'artifacts'
    artifacts_dir.mkdir(exist_ok=True)
    torch.save(theta_train, artifacts_dir / 'theta_train.pt')
    torch.save(x_train, artifacts_dir / 'x_train.pt')
    np.save(artifacts_dir / 'obs_transformed.npy', obs_diag_transformed)
    with open(artifacts_dir / 'observable_names.json', 'w') as _f:
        json.dump(observable_names, _f)
    with open(artifacts_dir / 'param_names.json', 'w') as _f:
        json.dump(param_names, _f)
    with open(artifacts_dir / 'npe_config.json', 'w') as _f:
        json.dump({
            'hidden_features': NPE_HIDDEN_FEATURES,
            'num_transforms': NPE_NUM_TRANSFORMS,
            'num_bins': NPE_NUM_BINS,
            'batch_size': NPE_TRAINING_BATCH_SIZE,
            'max_epochs': NPE_MAX_EPOCHS,
        }, _f)
    print(f"  Saved training artifacts to {artifacts_dir}/")

    if STAGE_ARTIFACTS_ONLY:
        print("\n" + "=" * 80)
        print("STAGE-ARTIFACTS-ONLY: exiting after step 2 (training + diagnostics skipped)")
        print("=" * 80)
        return {
            'save_dir': str(SAVE_DIR),
            'artifacts_dir': str(artifacts_dir),
            'n_train': int(theta_train.shape[0]),
            'n_test': int(theta_test.shape[0]),
            'n_params': num_params,
            'n_observables': num_observables,
            'stage_artifacts_only': True,
        }

    # ============================================================================
    # 3. PRIOR PREDICTIVE CHECK (quick sanity gate)
    # ============================================================================
    print("\n" + "=" * 80)
    print("3. PRIOR PREDICTIVE CHECK")
    print("=" * 80)
    print("  LOO decomposition of Mahalanobis D² — quick sanity check before training.")
    print("  (Jacobian + misspec assessment uses posterior samples in step 7e.)\n")

    x_train_copula = x_train.numpy() if torch.is_tensor(x_train) else x_train

    loo_prior_df, fig_loo_prior = sbi_loo_predictive_check(
        x_train_copula, obs_diag_transformed, observable_names,
    )
    fig_loo_prior.savefig(SAVE_DIR / 'loo_prior_predictive.png', dpi=150, bbox_inches='tight')
    plt.close(fig_loo_prior)

    k_eff_prior, _ = participation_ratio(x_train_copula)
    d2_prior = loo_prior_df['d_full'].iloc[0]
    chi2_p_prior = 1 - sp_stats.chi2.cdf(d2_prior, df=k_eff_prior)
    d2_z_prior = (d2_prior - k_eff_prior) / np.sqrt(2 * k_eff_prior)

    print(f"  D² = {d2_prior:.1f}, k_eff = {k_eff_prior:.1f}/{num_observables}, "
          f"χ²(k_eff) p = {chi2_p_prior:.3f} ({d2_z_prior:+.1f}σ)")

    print(f"\n  Top 5 structural misspecification drivers:")
    for _, row in loo_prior_df.head(5).iterrows():
        print(f"    {row['observable']}: influence = {row['influence']:.1f}")

    # Per-observable prior predictive histograms (raw simulator outputs vs observed)
    print("\n  Per-observable prior predictive coverage:")
    prior_pred_df, fig_prior_pred = sbi_posterior_predictive_coverage(
        x_train_raw, obs_diag_original, observable_names, log_scale=True
    )
    fig_prior_pred.suptitle('Prior predictive coverage', fontsize=14)
    fig_prior_pred.savefig(SAVE_DIR / 'prior_predictive_coverage.png', dpi=150, bbox_inches='tight')
    plt.close(fig_prior_pred)

    n_prior_in_ci = int(prior_pred_df['in_90ci'].sum())
    print(f"  {n_prior_in_ci}/{num_observables} observables within 90% prior predictive CI")
    for _, row in prior_pred_df[~prior_pred_df['in_90ci']].iterrows():
        print(f"    OUTSIDE: {row['observable']}: obs={row['obs_value']:.3g}, "
              f"90%CI=[{row['pp_q05']:.3g}, {row['pp_q95']:.3g}], p={row['p_value']:.4f}")

    _joint_reachability_check(
        x_train_raw, target_df, observable_names, SAVE_DIR, 'prior'
    )

    # Self-reference null for prior predictive D²
    (SAVE_DIR / 'diagnostics').mkdir(parents=True, exist_ok=True)
    print("\n  Self-reference null (no distributional assumptions):")
    sr_prior, fig_sr_prior = sbi_self_reference_null(
        x_train_copula, obs_diag_transformed, observable_names, loo_bias_correction=True
    )
    fig_sr_prior.savefig(SAVE_DIR / 'diagnostics' / 'self_reference_null_prior.png',
                         dpi=150, bbox_inches='tight')
    plt.close(fig_sr_prior)
    print(f"  D²_obs = {sr_prior['d2_obs']:.1f}, "
          f"null median = {np.median(sr_prior['d2_null']):.1f}, "
          f"empirical p = {sr_prior['p_value']:.3f}")
    n_flagged = int((sr_prior['influence_p_values']['p_value'] < 0.05).sum())
    if n_flagged > 0:
        print(f"  {n_flagged} observables with influence exceeding null 95th pctl:")
        for _, row in sr_prior['influence_p_values'].iterrows():
            if row['p_value'] < 0.05:
                print(f"    {row['observable']}: influence p = {row['p_value']:.3f}")
    sr_prior['influence_null'].to_csv(
        SAVE_DIR / 'diagnostics' / 'self_reference_null_prior.csv', index=False
    )

    # ============================================================================
    # 4. TRAIN NPE
    # ============================================================================
    print("\n" + "=" * 80)
    print("4. TRAIN NPE")
    print("=" * 80)

    if param_indices is not None:
        # When aux dims were appended, param_indices includes positions
        # past the QSP block — drop those before subsetting the QSP-only
        # prior_log_full, then re-attach the aux block as a separate
        # prior block via _JointBlockPrior below.
        n_qsp_full = prior_log_full.event_shape[0]
        qsp_indices = [i for i in param_indices if i < n_qsp_full]
        prior_qsp = prior_log_full.subset(qsp_indices)
    else:
        prior_qsp = prior_log_full

    if aux_member_names:
        # Aux samples in log space (block-diagonal MVN). Compose with QSP
        # log prior so the SBI prior matches theta_train = log([qsp ⊕ aux]).
        aux_log_mvn = _aux_log_block_mvn(aux_prior)
        prior_normal = _JointBlockPrior([prior_qsp, aux_log_mvn])
    else:
        prior_normal = prior_qsp
    prior_var = torch.var(prior_normal.sample((2000,)), dim=0).numpy()
    print(f"  Prior: {prior_normal.event_shape[0]}D (log-Normal space)")

    density_estimator = posterior_nn(
        model='nsf',
        hidden_features=NPE_HIDDEN_FEATURES,
        num_transforms=NPE_NUM_TRANSFORMS,
        num_bins=NPE_NUM_BINS,
    )
    print(f"  NSF: {NPE_NUM_TRANSFORMS} transforms, {NPE_HIDDEN_FEATURES} hidden, {NPE_NUM_BINS} bins")

    inference = NPE(prior=prior_normal, density_estimator=density_estimator)

    if cfg.null_shuffle:
        # Null-shuffle diagnostic: permute x_train rows relative to theta_train
        # to destroy all true (theta, x) coupling. The resulting trained NPE
        # has zero real signal; per-param contractions in local_calibration.csv
        # are the artifact floor for this architecture / N / hyperparams. Pair
        # with a matched real run and subtract: real - null = real signal.
        n_rows = x_train.shape[0]
        g = torch.Generator().manual_seed(cfg.null_shuffle_seed)
        perm = torch.randperm(n_rows, generator=g)
        x_train = x_train[perm].contiguous()
        print("=" * 80)
        print(f"  NULL-SHUFFLE diagnostic mode active (seed={cfg.null_shuffle_seed})")
        print(f"  Permuted {n_rows} x_train rows; (theta, x) coupling destroyed.")
        print("  All downstream contraction is artifact floor — do not interpret as signal.")
        print("=" * 80)

    inference = inference.append_simulations(theta_train, x_train)

    if PRETRAINED_DIR is not None:
        pre = Path(PRETRAINED_DIR)
        print(f"  Loading pretrained density_estimator from {pre}/")
        estimator = torch.load(pre / 'density_estimator.pt', weights_only=False)
        posterior = inference.build_posterior(density_estimator=estimator)
        lc_df_loaded = pd.read_csv(pre / 'learning_curve.csv')
        train_losses = lc_df_loaded['train_loss'].tolist()
        val_losses = lc_df_loaded['val_loss'].tolist()
        n_epochs = len(train_losses)
        pre_metrics = json.loads((pre / 'metrics.json').read_text())
        best_val_loss = float(pre_metrics.get('best_val_loss', min(val_losses)))
        n_nn_params = sum(p.numel() for p in estimator.parameters())
        print(f"  Loaded pretrained: {n_epochs} epochs, best val loss = {best_val_loss:.4f} "
              f"(trained on {pre_metrics.get('device', '?')} / {pre_metrics.get('gpu_name') or 'cpu'})")
        print(f"  NN parameters: {n_nn_params:,}")
    else:
        stop_event = threading.Event()
        def _progress_poll():
            t0 = time.time()
            last_epoch = 0
            while not stop_event.wait(10.0):
                tl = inference._summary.get('training_loss', [])
                vl = inference._summary.get('validation_loss', [])
                n = len(tl)
                if n > last_epoch:
                    last_epoch = n
                    tloss = tl[-1] if tl else float('nan')
                    vloss = vl[-1] if vl else float('nan')
                    print(f"  [train] epoch {n}/{NPE_MAX_EPOCHS}  train={tloss:.4f}  val={vloss:.4f}  elapsed={time.time()-t0:.1f}s", flush=True)
        progress_thread = threading.Thread(target=_progress_poll, daemon=True)
        progress_thread.start()
        try:
            estimator = inference.train(
                training_batch_size=NPE_TRAINING_BATCH_SIZE,
                max_num_epochs=NPE_MAX_EPOCHS,
                show_train_summary=True,
            )
        finally:
            stop_event.set()
        posterior = inference.build_posterior()

        train_losses = inference._summary['training_loss']
        val_losses = inference._summary['validation_loss']
        n_epochs = len(train_losses)
        best_val_loss = inference._summary['best_validation_loss'][-1]
        n_nn_params = sum(p.numel() for p in estimator.parameters())
        print(f"  Training complete: {n_epochs} epochs, best val loss = {best_val_loss:.4f}")
        print(f"  NN parameters: {n_nn_params:,}")

        if n_epochs >= NPE_MAX_EPOCHS:
            print(f"  WARNING: hit max_epochs ({NPE_MAX_EPOCHS}) — training may not have converged")
            recent_val = val_losses[-5:]
            if len(recent_val) >= 5 and recent_val[-1] < recent_val[0]:
                print(f"  WARNING: val loss still decreasing ({recent_val[0]:.4f} → {recent_val[-1]:.4f})")

    # Save learning curve
    lc_df = pd.DataFrame({
        'epoch': range(1, n_epochs + 1),
        'train_loss': train_losses,
        'val_loss': val_losses,
    })
    lc_df.to_csv(SAVE_DIR / 'learning_curve.csv', index=False)

    # Plot learning curve
    fig_lc, ax_lc = plt.subplots(figsize=(8, 4))
    ax_lc.plot(lc_df['epoch'], lc_df['train_loss'], label='Train', alpha=0.8)
    ax_lc.plot(lc_df['epoch'], lc_df['val_loss'], label='Validation', alpha=0.8)
    ax_lc.axhline(best_val_loss, color='gray', linestyle='--', linewidth=0.8, label=f'Best val: {best_val_loss:.4f}')
    ax_lc.set_xlabel('Epoch')
    ax_lc.set_ylabel('Loss')
    ax_lc.set_title(f'NPE training ({num_params} params, {n_nn_params:,} NN params)')
    ax_lc.legend()
    fig_lc.tight_layout()
    fig_lc.savefig(SAVE_DIR / 'learning_curve.png', dpi=150, bbox_inches='tight')
    plt.close(fig_lc)

    # Artifacts were already written before training (moved to end of step 2
    # so the staging path can invoke --stage-artifacts-only without training).

    if SAMPLE_ON_DELTA:
        if PRETRAINED_DIR is None:
            raise ValueError("--sample-on-delta requires --pretrained-dir")
        print("\n" + "=" * 80)
        print("5 + 5b + 6.  EVALUATE NPE + POSTERIOR (offloaded to Delta GPU)")
        print("=" * 80)
        import shutil
        output_dir = _analyze_on_delta(
            inputs={
                'density_estimator.pt': torch.load(Path(PRETRAINED_DIR) / 'density_estimator.pt', weights_only=False),
                'theta_test.pt': theta_test,
                'x_test.pt': x_test,
                'obs_transformed_tensor.pt': obs_transformed_tensor,
                'obs_transformed.npy': obs_transformed.astype(np.float32),
                'prior_var.npy': prior_var.astype(np.float32),
                'param_names.json': param_names,
                'pdac_priors.csv': Path(PRIORS_CSV),
                'config.json': {
                    'seed': SEED,
                    'n_plot_params': N_PLOT_PARAMS,
                    'n_test_samples': 200,
                    'n_post_samples': 2000,
                    'local_fraction': 0.2,
                    'chunk_size': 256,
                },
            },
            stage_dir=SAVE_DIR / 'delta_analysis',
            tag=f"sbi_{int(time.time())}",
        )
        # Copy Delta outputs into SAVE_DIR
        for fname in ['recovery.png', 'sbc_ecdf_global.png', 'sbc_ecdf_local.png',
                      'local_calibration.csv', 'posterior_samples.csv',
                      'posterior_summary.csv', 'posterior_vs_prior.png']:
            src = output_dir / fname
            if src.exists():
                shutil.copy(src, SAVE_DIR / fname)
        delta_metrics = json.loads((output_dir / 'run_metrics.json').read_text())
        r2_array = np.array([delta_metrics['r2_per_param'][n] for n in param_names])
        r2_ranking = np.argsort(r2_array)[::-1]
        plot_indices = r2_ranking[:N_PLOT_PARAMS].tolist()
        local_cal_df = pd.read_csv(SAVE_DIR / 'local_calibration.csv').set_index('parameter')
        r2_local_array = local_cal_df.loc[param_names, 'r2_local'].values
        ks_pvals_global = local_cal_df.loc[param_names, 'ks_pval_global'].values
        ks_pvals_local = local_cal_df.loc[param_names, 'ks_pval_local'].values
        median_contraction = float(delta_metrics['contraction_global_median'])
        median_contraction_local = float(delta_metrics['contraction_local_median'])
        n_miscal_global = int(delta_metrics['n_miscal_global'])
        n_miscal_local = int(delta_metrics['n_miscal_local'])
        # Per-condition z-score / contraction / rank matrices aren't round-tripped from Delta
        # (they'd add ~20 MB). save_diagnostics skips their CSVs when these are None; the
        # per-param medians are already in local_calibration.csv.
        z_scores = None
        contractions = None
        z_scores_local = None
        contractions_local = None
        ranks_global = None
        ranks_local = None
        # Reconstruct post_samples (log-space tensor) + post_samples_dict from the CSV
        post_df_orig = pd.read_csv(SAVE_DIR / 'posterior_samples.csv')
        post_samples = torch.tensor(np.log(post_df_orig[param_names].values), dtype=torch.float32)
        post_samples_dict = {name: post_df_orig[name].values for name in param_names}
        r2_values = {n: float(r2_array[i]) for i, n in enumerate(param_names)}
        r2_local = {n: float(r2_local_array[i]) for i, n in enumerate(param_names)}
        ks_global = {n: {'ks_p_value': float(ks_pvals_global[i]),
                         'ks_statistic': float('nan')} for i, n in enumerate(param_names)}
        ks_local = {n: {'ks_p_value': float(ks_pvals_local[i]),
                        'ks_statistic': float('nan')} for i, n in enumerate(param_names)}
        print(f"  R² recovery — top 5:")
        for i in r2_ranking[:5]:
            print(f"    {param_names[i]}: R² = {r2_array[i]:.3f}")
        print(f"  Median contraction: global={median_contraction:.3f}  local={median_contraction_local:.3f}")
        print(f"  SBC miscalibrated: global={n_miscal_global}/{num_params}  local={n_miscal_local}/{num_params}")
        print(f"  Delta artifacts copied to {SAVE_DIR}/")
    elif not EVAL_NPE_THETA_SPACE:
        print("\n" + "=" * 80)
        print("5 + 5b. EVALUATE NPE (θ-SPACE) — SKIPPED (eval_npe_theta_space=False)")
        print("=" * 80)
        print("  Misspec audit only needs step 6 + 7 outputs. Pass --eval-npe-theta-space")
        print("  to restore recovery.png, sbc_ecdf_*.png, local_calibration.csv.")
        # Safe defaults so step 6 + 7 + save_diagnostics + run_metrics still work.
        r2_array = np.full(num_params, np.nan)
        r2_values = {n: float('nan') for n in param_names}
        r2_local_array = np.full(num_params, np.nan)
        r2_local = dict(r2_values)
        r2_ranking = np.arange(num_params)
        plot_indices = list(range(min(N_PLOT_PARAMS, num_params)))
        z_scores = contractions = None
        z_scores_local = contractions_local = None
        ranks_global = ranks_local = None
        ks_global = {n: {'ks_p_value': float('nan'), 'ks_statistic': float('nan')}
                     for n in param_names}
        ks_local = dict(ks_global)
        ks_pvals_global = np.full(num_params, np.nan)
        ks_pvals_local = np.full(num_params, np.nan)
        median_contraction = float('nan')
        median_contraction_local = float('nan')
        n_miscal_global = 0
        n_miscal_local = 0

        # ========================================================================
        # 6. POSTERIOR FROM OBSERVED DATA
        # ========================================================================
        print("\n" + "=" * 80)
        print("6. POSTERIOR FROM OBSERVED DATA")
        print("=" * 80)
        print(f"  Sampling 2000 posterior draws conditioned on observed data...")
        torch.manual_seed(NPE_SEED)
        with torch.no_grad():
            post_samples = posterior.posterior_estimator.sample(
                torch.Size([2000]), condition=obs_transformed_tensor
            ).squeeze(1)
        post_samples_dict = convert_posterior_samples_to_original_space(post_samples, param_names)
        print(f"  Posterior samples: {post_samples.shape}")

    else:
        # ========================================================================
        # 5. EVALUATE NPE (θ-space)
        # ========================================================================
        print("\n" + "=" * 80)
        print("5. EVALUATE NPE (θ-SPACE)")
        print("=" * 80)

        n_test = x_test.shape[0]
        n_test_post_samples = 100
        print(
            f"  Sampling posterior for {n_test} test observations "
            f"({n_test_post_samples} samples each → {n_test * n_test_post_samples:,} total)..."
        )
        _t_test = time.time()
        with torch.no_grad():
            samples = posterior.posterior_estimator.sample(
                torch.Size([n_test_post_samples]), condition=x_test
            )
        print(f"  Test-set posterior sampling done in {time.time() - _t_test:.1f}s "
              f"(samples shape: {tuple(samples.shape)})")

        # R² recovery
        _, _, r2_values = sbi_recovery(samples, theta_test, param_names, plot=False)
        r2_array = np.array([r2_values[name] for name in param_names])
        r2_ranking = np.argsort(r2_array)[::-1]
        plot_indices = r2_ranking[:N_PLOT_PARAMS].tolist()

        print(f"\n  R² recovery — top 5:")
        for i in r2_ranking[:5]:
            print(f"    {param_names[i]}: R² = {r2_array[i]:.3f}")
        print(f"  R² recovery — bottom 5:")
        for i in r2_ranking[-5:]:
            print(f"    {param_names[i]}: R² = {r2_array[i]:.3f}")

        fig_recovery, _, _ = sbi_recovery(samples, theta_test, param_names, param_indices=plot_indices)
        fig_recovery.savefig(SAVE_DIR / 'recovery.png', dpi=150, bbox_inches='tight')
        plt.close(fig_recovery)

        # Z-score contraction (all params for save_diagnostics)
        _, _, z_scores, contractions = sbi_z_score_contraction(
            samples, theta_test, prior_normal, param_names, plot=False, prior_var=prior_var
        )
        # Plot subset
        sbi_z_score_contraction(
            samples, theta_test, prior_normal, param_names, param_indices=plot_indices, prior_var=prior_var
        )
        median_contraction = float(np.median(contractions))
        print(f"\n  Median z-score contraction: {median_contraction:.3f}")
        print(f"  (1.0 = no learning, 0.0 = perfect recovery)")

        # SBC calibration ECDF (global) — compute for all params, plot top N
        _, _, ranks_global, ks_global = sbi_calibration_ecdf(
            samples, theta_test, param_names, plot=False,
        )
        fig_ecdf, _, _, _ = sbi_calibration_ecdf(
            samples, theta_test, param_names, param_indices=plot_indices, difference=True,
        )
        fig_ecdf.suptitle('SBC calibration ECDF (global)', fontsize=14)
        fig_ecdf.savefig(SAVE_DIR / 'sbc_ecdf_global.png', dpi=150, bbox_inches='tight')
        plt.close(fig_ecdf)

        ks_pvals_global = np.array([ks_global[name]['ks_p_value'] for name in param_names])
        n_miscal_global = int((ks_pvals_global < 0.05).sum())
        print(f"\n  SBC ECDF: {n_miscal_global}/{num_params} params with KS p < 0.05 (miscalibrated)")
        if n_miscal_global > 0:
            worst = np.argsort(ks_pvals_global)[:5]
            for i in worst:
                if ks_pvals_global[i] < 0.05:
                    print(f"    {param_names[i]}: KS p = {ks_pvals_global[i]:.4f}")

        # --- 5b. LOCAL CALIBRATION (near observed data) ---
        # Standard SBC averages over the prior. Local SBC restricts to test observations
        # near the actual data, testing NPE calibration where it matters most.
        print("\n  " + "-" * 70)
        print("  5b. LOCAL CALIBRATION (near observed data)")
        print("  " + "-" * 70)

        LOCAL_FRACTION = 0.2  # closest 20% of test observations

        x_test_np = x_test.numpy() if torch.is_tensor(x_test) else x_test
        obs_ref = obs_transformed[np.newaxis, :]  # (1, n_obs)
        dists = np.linalg.norm(x_test_np - obs_ref, axis=1)
        n_local = max(10, int(LOCAL_FRACTION * len(dists)))
        local_idx = np.argsort(dists)[:n_local]

        print(f"  {n_local}/{len(dists)} test observations within distance "
              f"<= {dists[local_idx[-1]]:.2f} of observed data (copula space)")

        # Local R² recovery
        samples_local = samples[:, local_idx]
        theta_test_local = theta_test[local_idx]

        _, _, r2_local = sbi_recovery(samples_local, theta_test_local, param_names, plot=False)
        r2_local_array = np.array([r2_local[name] for name in param_names])

        # Local z-score contraction
        _, _, z_scores_local, contractions_local = sbi_z_score_contraction(
            samples_local, theta_test_local, prior_normal, param_names, plot=False, prior_var=prior_var
        )
        median_contraction_local = float(np.median(contractions_local))

        # Local SBC calibration ECDF — compute for all params, plot top N
        _, _, ranks_local, ks_local = sbi_calibration_ecdf(
            samples_local, theta_test_local, param_names, plot=False,
        )
        fig_ecdf_local, _, _, _ = sbi_calibration_ecdf(
            samples_local, theta_test_local, param_names, param_indices=plot_indices, difference=True,
        )
        fig_ecdf_local.suptitle('SBC calibration ECDF (local — near observed data)', fontsize=14)
        fig_ecdf_local.savefig(SAVE_DIR / 'sbc_ecdf_local.png', dpi=150, bbox_inches='tight')
        plt.close(fig_ecdf_local)

        ks_pvals_local = np.array([ks_local[name]['ks_p_value'] for name in param_names])
        n_miscal_local = int((ks_pvals_local < 0.05).sum())

        # Compare global vs local
        r2_diff = r2_local_array - r2_array
        print(f"\n  Global vs local calibration:")
        print(f"  {'Metric':<30s} {'Global':>8s} {'Local':>8s} {'Δ':>8s}")
        print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8}")
        print(f"  {'Median R²':<30s} {np.median(r2_array):8.3f} {np.median(r2_local_array):8.3f} "
              f"{np.median(r2_local_array) - np.median(r2_array):+8.3f}")
        print(f"  {'Median contraction':<30s} {median_contraction:8.3f} {median_contraction_local:8.3f} "
              f"{median_contraction_local - median_contraction:+8.3f}")
        print(f"  {'SBC miscalibrated (KS p<.05)':<30s} {n_miscal_global:8d} {n_miscal_local:8d} "
              f"{n_miscal_local - n_miscal_global:+8d}")

        # Flag parameters with big local-vs-global R² drops (unreliable near your data)
        degraded = np.where(r2_diff < -0.1)[0]
        if len(degraded) > 0:
            print(f"\n  WARNING: {len(degraded)} params with local R² drop > 0.1 (unreliable near observed data):")
            for i in sorted(degraded, key=lambda i: r2_diff[i]):
                print(f"    {param_names[i]}: R² {r2_array[i]:.3f} → {r2_local_array[i]:.3f} ({r2_diff[i]:+.3f})")

        improved = np.where(r2_diff > 0.1)[0]
        if len(improved) > 0:
            print(f"\n  {len(improved)} params with local R² gain > 0.1 (better identified near observed data):")
            for i in sorted(improved, key=lambda i: -r2_diff[i]):
                print(f"    {param_names[i]}: R² {r2_array[i]:.3f} → {r2_local_array[i]:.3f} ({r2_diff[i]:+.3f})")

        # Per-param metrics beyond r2/ks/contraction (local subset). samples_local
        # is log-space (theta_train was log-transformed upstream).
        pp = compute_per_param_calibration(samples_local, theta_test_local, z_scores=z_scores_local)

        # Save local calibration summary
        local_cal_df = pd.DataFrame({
            'parameter': param_names,
            'r2_global': r2_array,
            'r2_local': r2_local_array,
            'r2_diff': r2_diff,
            'contraction_global': np.median(contractions, axis=0),
            'contraction_local': np.median(contractions_local, axis=0),
            'ks_pval_global': ks_pvals_global,
            'ks_pval_local': ks_pvals_local,
            'zscore_mean_local': pp['zscore_mean'],
            'zscore_std_local': pp['zscore_std'],
            'coverage95_local': pp['coverage95'],
            'post_sd_log_local': pp['post_sd'],
        }).sort_values('r2_diff')
        local_cal_df.to_csv(SAVE_DIR / 'local_calibration.csv', index=False)
        print(f"\n  Saved to {SAVE_DIR}/local_calibration.csv")

        # ========================================================================
        # 6. POSTERIOR FROM OBSERVED DATA
        # ========================================================================
        print("\n" + "=" * 80)
        print("6. POSTERIOR FROM OBSERVED DATA")
        print("=" * 80)

        print(f"  Sampling 2000 posterior draws conditioned on observed data...")
        torch.manual_seed(NPE_SEED)  # Fix seed so posterior samples are deterministic (enables cache hits)
        with torch.no_grad():
            post_samples = posterior.posterior_estimator.sample(
                torch.Size([2000]), condition=obs_transformed_tensor
            ).squeeze(1)

        post_samples_dict = convert_posterior_samples_to_original_space(post_samples, param_names)
        print(f"  Posterior samples: {post_samples.shape}")

        # Quick summary
        post_df_summary = pd.DataFrame(post_samples_dict).describe(percentiles=[0.05, 0.5, 0.95]).T
        print(f"  Posterior summary (top 5 by R²):")
        for i in r2_ranking[:5]:
            name = param_names[i]
            row = post_df_summary.loc[name]
            print(f"    {name}: median={row['50%']:.3g}, 90%CI=[{row['5%']:.3g}, {row['95%']:.3g}]")

        # Prior vs posterior marginals (top N by R²)
        # The helper reads prior marginals from PRIORS_CSV (QSP only) — aux
        # names aren't in the CSV, so we restrict plot_indices to QSP
        # positions for this particular plot. Aux posteriors can be read
        # from posterior_samples.csv directly.
        n_qsp = len(param_names) - len(aux_member_names)
        plot_indices_qsp = [i for i in plot_indices if i < n_qsp]
        if not plot_indices_qsp:
            print("\n  Skipping posterior-vs-prior plot (top-R² selection contains only aux dims)")
        else:
            print(f"\n  Plotting posterior vs prior marginals (top {len(plot_indices_qsp)} QSP params by R²)...")
            fig_vs_prior, _ = plot_posterior_vs_prior_marginals(
                post_samples_dict,
                priors_csv=PRIORS_CSV,
                param_names=param_names,
                n_prior_samples=5000,
                log_transform=True,
                figsize=(20, 16),
                param_indices=plot_indices_qsp,
            )
            fig_vs_prior.savefig(SAVE_DIR / 'posterior_vs_prior.png', dpi=150, bbox_inches='tight')
            plt.close(fig_vs_prior)
            print(f"  Saved to {SAVE_DIR}/posterior_vs_prior.png")

    # ============================================================================
    # 7. POSTERIOR PREDICTIVE CHECK (actual simulations)
    # ============================================================================
    if not RUN_POSTERIOR_PREDICTIVE:
        print("\n" + "=" * 80)
        print("7. POSTERIOR PREDICTIVE CHECK — SKIPPED (RUN_POSTERIOR_PREDICTIVE = False)")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("7. POSTERIOR PREDICTIVE CHECK (ACTUAL SIMULATIONS)")
        print("=" * 80)

        # --- 7a. Run posterior predictive simulations (per-scenario, joint NaN filter) ---
        print(f"\n  Subsampling {N_POSTERIOR_SIMS} posterior draws for simulation...")
        rng = np.random.default_rng(NPE_SEED)
        idx = rng.choice(post_samples.shape[0], size=N_POSTERIOR_SIMS, replace=False)
        theta_posterior = post_samples[idx].numpy() if torch.is_tensor(post_samples) else post_samples[idx]

        # Convert from log-space to original parameter space for simulation
        # If using tier subset, embed posterior draws back into full param space
        if param_indices is not None:
            n_post = theta_posterior.shape[0]
            torch.manual_seed(NPE_SEED + 1)
            prior_fill = prior_log_full.sample((n_post,)).numpy()
            theta_posterior_full = prior_fill.copy()
            for j_sub, j_full in enumerate(param_indices):
                theta_posterior_full[:, j_full] = theta_posterior[:, j_sub]
            theta_posterior_orig = np.exp(theta_posterior_full)
            print(f"  Embedded {num_params} posterior params into {num_params_full}-dim space (rest from prior)")
        else:
            theta_posterior_orig = np.exp(theta_posterior)

        # Split joint theta into QSP and aux blocks. Joint theta layout is
        # [θ_qsp_full, θ_aux] (aux always appended after the QSP block —
        # see section 2c). The C++ simulator only consumes the QSP slice;
        # aux values are merged into species_dict at derive time via
        # aux_by_sample_index. PPC sims thus see *posterior* aux draws,
        # not the prior aux samples cached for training.
        if aux_member_names:
            n_qsp_full = theta_posterior_orig.shape[1] - len(aux_member_names)
            theta_qsp_orig = theta_posterior_orig[:, :n_qsp_full]
            theta_aux_orig = theta_posterior_orig[:, n_qsp_full:]
            ppc_aux_by_sample_index: dict[int, dict[str, float]] = {
                i: {
                    name: float(theta_aux_orig[i, j])
                    for j, name in enumerate(aux_member_names)
                }
                for i in range(theta_aux_orig.shape[0])
            }
            ppc_auxiliary_units = dict(aux_units_by_name)
            print(
                f"  PPC aux: {len(aux_member_names)} member(s) drawn from "
                f"posterior ({theta_aux_orig.shape[0]} sims × "
                f"{theta_aux_orig.shape[1]} aux dims)"
            )
        else:
            theta_qsp_orig = theta_posterior_orig
            ppc_aux_by_sample_index = None
            ppc_auxiliary_units = None

        # Simulate each scenario and collect per-scenario posterior predictive outputs
        print(f"  Running {N_POSTERIOR_SIMS} posterior predictive simulations across {n_scenarios} scenarios...")
        t0 = time.time()
        pp_raw_per_scenario = {}
        for scen_name in scenario_names:
            sim_scen = scenario_meta[scen_name]['simulator']
            # CppSimulator doesn't yet expose simulate_with_parameters; the
            # equivalent would need a theta-hashed suffix pool + chained
            # derivation. Surface a clear message instead of an
            # AttributeError-into-traceback when RUN_POSTERIOR_PREDICTIVE is
            # turned on.
            if not hasattr(sim_scen, 'simulate_with_parameters'):
                raise NotImplementedError(
                    "RUN_POSTERIOR_PREDICTIVE is not supported on the C++ "
                    "backend yet — CppSimulator needs a "
                    "simulate_with_parameters(theta) method (theta-hashed "
                    "suffix pool + chained derivation, mirroring "
                    "QSPSimulator.simulate_with_parameters). Re-run with "
                    "run_posterior_predictive=False to skip section 7."
                )
            ppc_kwargs: dict = {}
            if SAVE_EVOLVE_TRAJECTORIES:
                # Per-scenario per-sim trajectory binaries land here for
                # the orchestration pipeline below to assemble + plot.
                evolve_traj_dir = SAVE_DIR / 'evolve_trajectories' / scen_name
                evolve_traj_dir.mkdir(parents=True, exist_ok=True)
                ppc_kwargs['evolve_trajectory_dir'] = evolve_traj_dir
                if EVOLVE_TRAJECTORY_DT_DAYS is not None:
                    ppc_kwargs['evolve_trajectory_dt_days'] = EVOLVE_TRAJECTORY_DT_DAYS
            if ppc_aux_by_sample_index is not None:
                ppc_kwargs['aux_by_sample_index'] = ppc_aux_by_sample_index
                ppc_kwargs['auxiliary_units'] = ppc_auxiliary_units
            pp_result = sim_scen.simulate_with_parameters(theta_qsp_orig, **ppc_kwargs)
            # CppSimulator returns (theta_out, pyarrow.Table) with ts:<id>
            # columns; legacy QSPSimulator returns just an ndarray.
            if isinstance(pp_result, tuple):
                _, pp_table = pp_result
                obs_names_scen = scenario_meta[scen_name]['observable_names']
                pp_raw_per_scenario[scen_name] = np.column_stack([
                    pp_table.column(f"ts:{name}").to_numpy() for name in obs_names_scen
                ])
            else:
                pp_raw_per_scenario[scen_name] = pp_result
            print(f"    {scen_name}: {pp_raw_per_scenario[scen_name].shape}")
        elapsed = time.time() - t0
        print(f"  All scenarios complete in {elapsed:.0f}s")

        # Joint NaN filter across all scenarios
        valid_mask = np.ones(N_POSTERIOR_SIMS, dtype=bool)
        for scen_name in scenario_names:
            valid_mask &= np.all(np.isfinite(pp_raw_per_scenario[scen_name]), axis=1)

        for scen_name in scenario_names:
            scenario_meta[scen_name]['pp_raw_clean'] = pp_raw_per_scenario[scen_name][valid_mask]

        # Concatenate raw posterior predictive across scenarios
        x_posterior_pred_clean = np.concatenate(
            [scenario_meta[s]['pp_raw_clean'] for s in scenario_names], axis=1
        )
        print(f"  Valid posterior predictive sims: {valid_mask.sum()}/{N_POSTERIOR_SIMS}")

        # --- 7b. Per-observable coverage ---
        print("\n  Per-observable posterior predictive coverage:")
        coverage_df, fig_coverage = sbi_posterior_predictive_coverage(
            x_posterior_pred_clean, obs_diag_original, observable_names, log_scale=True
        )
        fig_coverage.savefig(SAVE_DIR / 'posterior_predictive_coverage.png', dpi=150, bbox_inches='tight')
        plt.close(fig_coverage)

        n_in_ci = int(coverage_df['in_90ci'].sum())
        print(f"  {n_in_ci}/{num_observables} observables within 90% posterior predictive CI")
        for _, row in coverage_df[~coverage_df['in_90ci']].iterrows():
            print(f"    OUTSIDE: {row['observable']}: obs={row['obs_value']:.3g}, "
                  f"90%CI=[{row['pp_q05']:.3g}, {row['pp_q95']:.3g}], p={row['p_value']:.4f}")

        _joint_reachability_check(
            x_posterior_pred_clean, target_df, observable_names, SAVE_DIR, 'posterior'
        )

        # --- 7b'. Prior vs posterior predictive histograms ---
        print("\n  Plotting prior vs posterior predictive histograms...")
        n_obs = len(observable_names)
        ncols = 6
        nrows = int(np.ceil(n_obs / ncols))
        fig_pp, axes_pp = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.5))
        axes_pp = axes_pp.flatten()

        for k in range(n_obs):
            ax = axes_pp[k]
            prior_vals = x_train_raw[:, k]
            post_vals = x_posterior_pred_clean[:, k]
            obs_val = float(np.squeeze(obs_original[observable_names[k]]))

            # Use log scale if values span > 2 orders of magnitude
            all_vals = np.concatenate([prior_vals[np.isfinite(prior_vals)],
                                       post_vals[np.isfinite(post_vals)]])
            use_log = (all_vals.max() / max(all_vals.min(), 1e-30)) > 100

            if use_log:
                prior_vals = np.log10(np.clip(prior_vals, 1e-30, None))
                post_vals = np.log10(np.clip(post_vals, 1e-30, None))
                obs_val = np.log10(max(obs_val, 1e-30))

            bins = np.linspace(
                min(np.nanquantile(prior_vals, 0.01), np.nanquantile(post_vals, 0.01)),
                max(np.nanquantile(prior_vals, 0.99), np.nanquantile(post_vals, 0.99)),
                40,
            )

            ax.hist(prior_vals, bins=bins, alpha=0.4, density=True, color='steelblue', label='Prior')
            ax.hist(post_vals, bins=bins, alpha=0.6, density=True, color='coral', label='Posterior')
            ax.axvline(obs_val, color='black', linewidth=1.5, linestyle='--', label='Observed')
            name = observable_names[k]
            ax.set_title(name[:30], fontsize=12)
            ax.tick_params(labelsize=10)
            if use_log:
                ax.set_xlabel('log\u2081\u2080', fontsize=10)

        # Hide unused axes
        for k in range(n_obs, len(axes_pp)):
            axes_pp[k].set_visible(False)

        # Single legend
        handles, labels = axes_pp[0].get_legend_handles_labels()
        fig_pp.legend(handles, labels, loc='upper right', fontsize=13)
        fig_pp.suptitle('Prior vs Posterior Predictive', fontsize=18)
        fig_pp.tight_layout(rect=[0, 0, 1, 0.96])
        fig_pp.savefig(SAVE_DIR / 'prior_vs_posterior_predictive.png', dpi=150, bbox_inches='tight')
        plt.close(fig_pp)
        print(f"  Saved to {SAVE_DIR}/prior_vs_posterior_predictive.png")

        # --- 7b''. Prior vs posterior predictive in TRANSFORMED (copula) space ---
        print("\n  Plotting prior vs posterior predictive in transformed space...")
        # Per-scenario copula transform using stored quantiles, then concatenate
        x_prior_t_parts, x_post_t_parts = [], []
        for scen_name in scenario_names:
            meta = scenario_meta[scen_name]
            q = meta['quantiles']
            x_prior_t_parts.append(transform_to_normal_from_array(meta['x_train_raw'], q))
            x_post_t_parts.append(transform_to_normal_from_array(meta['pp_raw_clean'], q))
        x_prior_transformed = np.concatenate(x_prior_t_parts, axis=1)
        x_post_transformed = np.concatenate(x_post_t_parts, axis=1)

        fig_pp_t, axes_pp_t = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.5))
        axes_pp_t = axes_pp_t.flatten()

        for k in range(n_obs):
            ax = axes_pp_t[k]
            prior_t = x_prior_transformed[:, k]
            post_t = x_post_transformed[:, k]
            obs_t = obs_diag_transformed[k]

            # Filter non-finite
            prior_t = prior_t[np.isfinite(prior_t)]
            post_t = post_t[np.isfinite(post_t)]

            if len(prior_t) == 0:
                ax.set_title(observable_names[k][:30] + ' (empty)', fontsize=12)
                continue

            bins_t = np.linspace(
                min(np.quantile(prior_t, 0.01), np.quantile(post_t, 0.01) if len(post_t) > 0 else 0),
                max(np.quantile(prior_t, 0.99), np.quantile(post_t, 0.99) if len(post_t) > 0 else 0),
                40,
            )

            ax.hist(prior_t, bins=bins_t, alpha=0.4, density=True, color='steelblue', label='Prior')
            if len(post_t) > 0:
                ax.hist(post_t, bins=bins_t, alpha=0.6, density=True, color='coral', label='Posterior')
            ax.axvline(obs_t, color='black', linewidth=1.5, linestyle='--', label='Observed')
            ax.set_title(observable_names[k][:30], fontsize=12)
            ax.tick_params(labelsize=10)
            ax.set_xlabel('z-score', fontsize=10)

        for k in range(n_obs, len(axes_pp_t)):
            axes_pp_t[k].set_visible(False)

        handles_t, labels_t = axes_pp_t[0].get_legend_handles_labels()
        fig_pp_t.legend(handles_t, labels_t, loc='upper right', fontsize=13)
        fig_pp_t.suptitle('Prior vs Posterior Predictive (Copula-Transformed Space)', fontsize=18)
        fig_pp_t.tight_layout(rect=[0, 0, 1, 0.96])
        fig_pp_t.savefig(SAVE_DIR / 'prior_vs_posterior_predictive_transformed.png', dpi=150, bbox_inches='tight')
        plt.close(fig_pp_t)
        print(f"  Saved to {SAVE_DIR}/prior_vs_posterior_predictive_transformed.png")

        # --- 7b'''. Numerical summary: per-observable predictive stats in both spaces ---
        print("\n  Per-observable predictive summary (raw + transformed spaces):")
        pred_summary_rows = []
        for k in range(n_obs):
            name = observable_names[k]
            obs_val = float(np.squeeze(obs_original[name]))

            # Raw space
            prior_raw_k = x_train_raw[:, k]
            post_raw_k = x_posterior_pred_clean[:, k]
            prior_raw_k = prior_raw_k[np.isfinite(prior_raw_k)]
            post_raw_k = post_raw_k[np.isfinite(post_raw_k)]

            # Transformed space
            prior_t_k = x_prior_transformed[:, k]
            post_t_k = x_post_transformed[:, k]
            obs_t_k = obs_diag_transformed[k]
            prior_t_k = prior_t_k[np.isfinite(prior_t_k)]
            post_t_k = post_t_k[np.isfinite(post_t_k)]

            # KL-like shift: difference in medians (transformed space, in sigma units)
            prior_t_med = float(np.median(prior_t_k)) if len(prior_t_k) > 0 else np.nan
            post_t_med = float(np.median(post_t_k)) if len(post_t_k) > 0 else np.nan
            shift_sigma = post_t_med - prior_t_med

            # Posterior narrowing: ratio of IQRs
            prior_t_iqr = float(np.subtract(*np.percentile(prior_t_k, [75, 25]))) if len(prior_t_k) > 1 else np.nan
            post_t_iqr = float(np.subtract(*np.percentile(post_t_k, [75, 25]))) if len(post_t_k) > 1 else np.nan
            iqr_ratio = post_t_iqr / prior_t_iqr if prior_t_iqr > 0 else np.nan

            pred_summary_rows.append({
                'observable': name,
                'obs_raw': obs_val,
                'prior_med_raw': float(np.median(prior_raw_k)) if len(prior_raw_k) > 0 else np.nan,
                'post_med_raw': float(np.median(post_raw_k)) if len(post_raw_k) > 0 else np.nan,
                'prior_q05_raw': float(np.percentile(prior_raw_k, 5)) if len(prior_raw_k) > 0 else np.nan,
                'prior_q95_raw': float(np.percentile(prior_raw_k, 95)) if len(prior_raw_k) > 0 else np.nan,
                'post_q05_raw': float(np.percentile(post_raw_k, 5)) if len(post_raw_k) > 0 else np.nan,
                'post_q95_raw': float(np.percentile(post_raw_k, 95)) if len(post_raw_k) > 0 else np.nan,
                'obs_z': obs_t_k,
                'prior_med_z': prior_t_med,
                'post_med_z': post_t_med,
                'shift_sigma': shift_sigma,
                'iqr_ratio': iqr_ratio,
                'n_prior_valid': len(prior_raw_k),
                'n_post_valid': len(post_raw_k),
            })

        pred_summary_df = pd.DataFrame(pred_summary_rows)
        pred_summary_df.to_csv(SAVE_DIR / 'predictive_summary.csv', index=False)

        # Print compact table. Backslash escapes in f-string expressions
        # are a Python 3.12+ feature (PEP 701) \u2014 extract the unicode chars
        # to locals so 3.11 CI still parses this module.
        iqr_col = 'IQR\u2193'
        rule = '\u2500'
        print(f"\n  {'Observable':<40s} {'obs_z':>6s} {'shift':>6s} {iqr_col:>6s} {'obs_raw':>12s} {'post_med':>12s}")
        print(f"  {rule*40} {rule*6} {rule*6} {rule*6} {rule*12} {rule*12}")
        for _, r in pred_summary_df.sort_values('obs_z', key=abs, ascending=False).iterrows():
            print(f"  {r['observable']:<40s} {r['obs_z']:>6.2f} {r['shift_sigma']:>+6.2f} "
                  f"{r['iqr_ratio']:>6.2f} "
                  f"{r['obs_raw']:>12.3g} {r['post_med_raw']:>12.3g}")

        extreme_obs = pred_summary_df[pred_summary_df['obs_z'].abs() > 2.0]
        if len(extreme_obs) > 0:
            print(f"\n  WARNING: {len(extreme_obs)} observables have observed z-score > 2\u03c3 from prior median:")
            for _, r in extreme_obs.iterrows():
                print(f"    {r['observable']}: obs_z = {r['obs_z']:+.2f}")

        print(f"\n  Full table saved to {SAVE_DIR}/predictive_summary.csv")

        # --- 7c. Posterior predictive LOO (calibration quality) ---
        print("\n  Posterior predictive LOO decomposition...")
        # Per-scenario copula transform, then concatenate
        x_pp_copula_parts = []
        for scen_name in scenario_names:
            meta = scenario_meta[scen_name]
            x_pp_copula_parts.append(
                transform_to_normal_from_array(meta['pp_raw_clean'], meta['quantiles'])
            )
        x_pp_copula = np.concatenate(x_pp_copula_parts, axis=1)
        valid_copula = np.all(np.isfinite(x_pp_copula), axis=1)
        x_pp_copula = x_pp_copula[valid_copula]
        print(f"  Copula-transformed posterior predictive: {x_pp_copula.shape[0]} valid samples")

        loo_post_df, fig_loo_post = sbi_loo_predictive_check(
            x_pp_copula, obs_diag_transformed, observable_names,
        )
        fig_loo_post.savefig(SAVE_DIR / 'loo_posterior_predictive.png', dpi=150, bbox_inches='tight')
        plt.close(fig_loo_post)

        k_eff_post, _ = participation_ratio(x_pp_copula)
        d2_post = loo_post_df['d_full'].iloc[0]
        chi2_p_post = 1 - sp_stats.chi2.cdf(d2_post, df=k_eff_post)

        print(f"\n  Top 5 posterior misfit drivers:")
        for _, row in loo_post_df.head(5).iterrows():
            print(f"    {row['observable']}: influence = {row['influence']:.1f}")

        # Self-reference null for posterior predictive D²
        print("\n  Self-reference null (posterior predictive):")
        sr_post, fig_sr_post = sbi_self_reference_null(
            x_pp_copula, obs_diag_transformed, observable_names, loo_bias_correction=True
        )
        fig_sr_post.savefig(SAVE_DIR / 'diagnostics' / 'self_reference_null_posterior.png',
                            dpi=150, bbox_inches='tight')
        plt.close(fig_sr_post)
        print(f"  D²_obs = {sr_post['d2_obs']:.1f}, "
              f"null median = {np.median(sr_post['d2_null']):.1f}, "
              f"empirical p = {sr_post['p_value']:.3f}")

        # --- 7d. Summary table ---
        print(f"\n  {'Distribution':<35s} {'k_eff':>6s} {'D²':>7s} {'χ² p':>8s} {'emp. p':>8s}")
        print(f"  {'-'*35} {'-'*6} {'-'*7} {'-'*8} {'-'*8}")
        print(f"  {'Prior predictive':<35s} {k_eff_prior:6.1f} {d2_prior:7.1f} {chi2_p_prior:8.3f} {sr_prior['p_value']:8.3f}")
        print(f"  {'Posterior predictive (actual sims)':<35s} {k_eff_post:6.1f} {d2_post:7.1f} {chi2_p_post:8.3f} {sr_post['p_value']:8.3f}")

        # D² ratio: structural fraction of prior misfit
        d2_ratio = d2_post / d2_prior if d2_prior > 0 else np.nan
        print(f"  D²_post/D²_prior = {d2_ratio:.3f} "
              f"({'mostly structural' if d2_ratio > 0.5 else 'mostly parametric'})")

        n_flagged_post = int((sr_post['influence_p_values']['p_value'] < 0.05).sum())
        if n_flagged_post > 0:
            print(f"  {n_flagged_post} observables with influence exceeding null 95th pctl:")
            for _, row in sr_post['influence_p_values'].iterrows():
                if row['p_value'] < 0.05:
                    print(f"    {row['observable']}: influence p = {row['p_value']:.3f}")
        else:
            print("  No observables exceed null 95th pctl — no structural misspecification detected")
        sr_post['influence_null'].to_csv(
            SAVE_DIR / 'diagnostics' / 'self_reference_null_posterior.csv', index=False
        )

        # --- 7e. POSTERIOR JACOBIAN + MISSPECIFICATION ASSESSMENT ---
        print("\n  " + "-" * 70)
        print("  7e. POSTERIOR JACOBIAN + MISSPECIFICATION ASSESSMENT")
        print("  " + "-" * 70)
        print("  Estimating Jacobian d(log x)/d(log theta) from posterior samples...")

        # Use posterior theta (log-space) and posterior predictive x, filtered to valid rows
        theta_post_log = theta_posterior[valid_mask]  # (n_valid, num_params), already log-space
        log_x_post = np.log(np.clip(x_posterior_pred_clean, 1e-30, None))

        # Center for regression
        log_theta_c = theta_post_log - theta_post_log.mean(axis=0)
        log_x_c = log_x_post - log_x_post.mean(axis=0)

        # OLS Jacobian: J = (X^T X)^{-1} X^T Y
        XtX = log_theta_c.T @ log_theta_c
        try:
            XtX_inv = np.linalg.inv(XtX + 1e-8 * np.eye(XtX.shape[0]))
            jacobian = (XtX_inv @ (log_theta_c.T @ log_x_c)).T  # (n_obs, n_params)
        except np.linalg.LinAlgError:
            print("  WARNING: Jacobian estimation failed (singular matrix). Skipping.")
            jacobian = None

        if jacobian is not None:
            from matplotlib.colors import TwoSlopeNorm

            jac_dir = SAVE_DIR / 'jacobian'
            jac_dir.mkdir(parents=True, exist_ok=True)

            jac_df = pd.DataFrame(jacobian, index=observable_names, columns=param_names)
            jac_df.to_csv(jac_dir / 'jacobian.csv')

            # Per-observable sensitivity summary
            max_sens = np.max(np.abs(jacobian), axis=1)
            sens_df = pd.DataFrame({
                'observable': observable_names,
                'max_abs_elasticity': max_sens,
                'top_param': [
                    param_names[np.argmax(np.abs(jacobian[i]))]
                    for i in range(num_observables)
                ],
                'top_param_elasticity': [
                    float(jacobian[i, np.argmax(np.abs(jacobian[i]))])
                    for i in range(num_observables)
                ],
                'n_params_above_01': [
                    int(np.sum(np.abs(jacobian[i]) > 0.1))
                    for i in range(num_observables)
                ],
            }).sort_values('max_abs_elasticity', ascending=True)
            sens_df.to_csv(jac_dir / 'sensitivity_summary.csv', index=False)

            n_dead = int((max_sens < 0.05).sum())
            if n_dead > 0:
                print(f"  WARNING: {n_dead} targets insensitive to all params (max |elasticity| < 0.05)")
            print(f"  Top 5 most sensitive observables:")
            for _, row in sens_df.tail(5).iloc[::-1].iterrows():
                print(f"    {row['observable']}: |elasticity|={row['max_abs_elasticity']:.2f} "
                      f"(top: {row['top_param']})")

            # SVD of CI-weighted Jacobian
            target_log_scale = np.zeros(num_observables)
            for i in range(num_observables):
                lo = target_df['ci95_lower'].values[i]
                hi = target_df['ci95_upper'].values[i]
                if lo > 0 and hi > 0:
                    target_log_scale[i] = (np.log(hi) - np.log(lo)) / (2 * 1.96)
                else:
                    target_log_scale[i] = 1.0

            J_weighted = jacobian / target_log_scale[:, np.newaxis]
            U, S, Vt = np.linalg.svd(J_weighted, full_matrices=False)
            n_eff_jac = float(np.sum(S) ** 2 / np.sum(S ** 2))

            print(f"\n  Jacobian SVD: effective rank = {n_eff_jac:.1f}/{min(num_params, num_observables)}")
            print(f"  Top 5 singular values: {', '.join(f'{s:.2f}' for s in S[:5])}")

            svd_df = pd.DataFrame({
                'index': range(len(S)),
                'singular_value': S,
                'cumulative_variance_frac': np.cumsum(S ** 2) / np.sum(S ** 2),
            })
            svd_df.to_csv(jac_dir / 'svd_spectrum.csv', index=False)

            # Sloppy params
            sloppy_threshold = S[0] * 0.01
            n_sloppy = int(np.sum(S < sloppy_threshold))
            if n_sloppy > 0:
                sloppy_directions = Vt[-n_sloppy:]
                sloppy_loading = np.sum(sloppy_directions ** 2, axis=0)
                sloppy_df = pd.DataFrame({
                    'parameter': param_names,
                    'sloppy_loading': sloppy_loading,
                }).sort_values('sloppy_loading', ascending=False)
                sloppy_df.to_csv(jac_dir / 'sloppy_params.csv', index=False)
                print(f"  {n_sloppy}/{num_params} sloppy directions (SV < {sloppy_threshold:.3g})")
                print(f"  Top 3 sloppy params: {', '.join(sloppy_df.head(3)['parameter'])}")

            # --- Residual-direction-aware misspecification verdict ---
            # Compute per-observable residuals: posterior median vs target
            target_medians = target_df['median'].values
            post_medians = np.median(x_posterior_pred_clean, axis=0)
            residuals = np.zeros(num_observables)
            for i in range(num_observables):
                s, t = post_medians[i], target_medians[i]
                if s > 0 and t > 0:
                    residuals[i] = (np.log(s) - np.log(t)) / target_log_scale[i]

            # Desired correction direction per target (sign-flip Jacobian row by residual)
            desired_direction = np.zeros_like(jacobian)
            for i in range(num_observables):
                if residuals[i] > 0:  # sim > target, want decrease
                    desired_direction[i] = -jacobian[i]
                else:  # sim < target, want increase
                    desired_direction[i] = jacobian[i]

            dd_norms = np.linalg.norm(desired_direction, axis=1, keepdims=True)
            dd_norms = np.clip(dd_norms, 1e-10, None)
            desired_normed = desired_direction / dd_norms

            # Coupling matrix from desired correction directions
            coupling_matrix = desired_normed @ desired_normed.T

            # Identify targets outside posterior predictive 90% CI
            outside_ci_mask = ~coverage_df.set_index('observable').loc[observable_names, 'in_90ci'].values
            outside_idx = np.where(outside_ci_mask)[0]

            # Check conflicts among outside-CI targets
            conflict_pairs = []
            if len(outside_idx) >= 2:
                for ii, i in enumerate(outside_idx):
                    for jj in range(ii + 1, len(outside_idx)):
                        j = outside_idx[jj]
                        cos_sim = float(desired_normed[i] @ desired_normed[j])
                        if cos_sim < -0.3:
                            conflict_pairs.append({
                                'target_a': observable_names[i],
                                'target_b': observable_names[j],
                                'cosine_similarity': cos_sim,
                                'z_a': float(residuals[i]),
                                'z_b': float(residuals[j]),
                            })

            if conflict_pairs:
                cp_df = pd.DataFrame(conflict_pairs).sort_values('cosine_similarity')
                cp_df.to_csv(jac_dir / 'target_conflicts.csv', index=False)

            if len(outside_idx) >= 1:
                outside_ci_names = {observable_names[i] for i in outside_idx}
                print(f"\n  MISSPECIFICATION ASSESSMENT (posterior-based):")
                print(f"  {len(outside_idx)} targets outside 90% posterior predictive CI: "
                      f"{', '.join(outside_ci_names)}")

                if conflict_pairs:
                    print(f"\n  {len(conflict_pairs)} conflicting target pairs:")
                    for _, row in cp_df.head(5).iterrows():
                        print(f"    {row['target_a']} vs {row['target_b']}: "
                              f"cos={row['cosine_similarity']:.2f}")

                # Per-observable predictive span (posterior-predictive reachability check).
                # Detects the "MARGINAL with named driver but unreachable" failure mode:
                # elasticity attribution can name a top driver even when the parameter has
                # near-zero absolute leverage — in that case the target sits outside the
                # 90% posterior-predictive interval and no posterior-supported θ can hit
                # it without an extreme tail draw. Flag this as STRUCTURAL (not driver-fixable).
                obs_pp = x_posterior_pred_clean  # (n_valid, n_obs), positive
                obs_pp_log = np.log(np.clip(obs_pp, 1e-30, None))
                pred_log_q05 = np.percentile(obs_pp_log, 5, axis=0)
                pred_log_q95 = np.percentile(obs_pp_log, 95, axis=0)
                target_log_med = np.log(np.clip(target_medians, 1e-30, None))
                # Gap (log) and predictive 90% span (log)
                gap_log = np.abs(target_log_med - np.log(np.clip(post_medians, 1e-30, None)))
                span_log = pred_log_q95 - pred_log_q05
                # Leverage = how much of the gap can the posterior 90% predictive span cover?
                # <1 means even the bulk of the posterior can't reach the target.
                leverage_ratio = np.where(gap_log > 1e-9, span_log / gap_log, np.inf)
                target_unreachable = (
                    (target_log_med < pred_log_q05) | (target_log_med > pred_log_q95)
                )

                misspec_verdicts = []
                for i in outside_idx:
                    obs_name = observable_names[i]
                    reasons = []

                    # Check if dead (insensitive — relative-elasticity floor)
                    if max_sens[i] < 0.05:
                        reasons.append(
                            f"insensitive (max |elasticity|={max_sens[i]:.4f})"
                        )

                    # Posterior-predictive reachability: target outside the span of
                    # obs values across posterior samples means no posterior-supported θ
                    # can hit it. The named "top driver" attribution is then an artifact
                    # of relative-elasticity ranking under low absolute leverage.
                    if target_unreachable[i]:
                        reasons.append(
                            f"target unreachable in posterior predictive "
                            f"(span={span_log[i]:.2f}, gap={gap_log[i]:.2f}, "
                            f"leverage={leverage_ratio[i]:.2f}; structural fix required)"
                        )

                    # Check if in conflict with another outside-CI target
                    for pair in conflict_pairs:
                        other = None
                        if pair['target_a'] == obs_name:
                            other = pair['target_b']
                        elif pair['target_b'] == obs_name:
                            other = pair['target_a']
                        if other:
                            reasons.append(
                                f"conflicts with {other} (cos={pair['cosine_similarity']:.2f})"
                            )

                    p_val = float(
                        coverage_df.set_index('observable').loc[obs_name, 'p_value']
                    )

                    if reasons:
                        verdict = 'LIKELY MISSPECIFIED'
                        reason_str = '; '.join(reasons)
                    else:
                        top_p = param_names[np.argmax(np.abs(jacobian[i]))]
                        verdict = 'MARGINAL — sensitive, no conflicts'
                        reason_str = (
                            f'sensitive to params (top: {top_p}, leverage={leverage_ratio[i]:.2f}), '
                            'no competing targets outside CI'
                        )

                    misspec_verdicts.append({
                        'observable': obs_name,
                        'verdict': verdict,
                        'reasons': reason_str,
                        'p_value': p_val,
                        'z_residual': float(residuals[i]),
                        'max_abs_elasticity': float(max_sens[i]),
                        'top_param': param_names[np.argmax(np.abs(jacobian[i]))],
                        'pred_log_span': float(span_log[i]),
                        'gap_log': float(gap_log[i]),
                        'leverage_ratio': float(leverage_ratio[i]),
                        'target_unreachable': bool(target_unreachable[i]),
                    })
                    print(f"    {obs_name}: {verdict} (p={p_val:.4f}, z={residuals[i]:+.2f})")
                    print(f"      {reason_str}")

                misspec_df = pd.DataFrame(misspec_verdicts)
                misspec_df.to_csv(jac_dir / 'misspec_verdicts.csv', index=False)
            else:
                print("\n  All targets within 90% posterior predictive CI — no misspecification flags.")

            # --- Figures ---
            # Jacobian heatmap
            fig_jac, ax_jac = plt.subplots(
                figsize=(max(10, num_params * 0.4), max(6, num_observables * 0.35))
            )
            vmax = np.percentile(np.abs(jacobian), 95)
            if vmax < 1e-10:
                vmax = 1.0
            jac_norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            im = ax_jac.imshow(jacobian, aspect='auto', cmap='RdBu_r', norm=jac_norm)
            ax_jac.set_xticks(range(num_params))
            ax_jac.set_xticklabels(param_names, rotation=90, fontsize=6)
            ax_jac.set_yticks(range(num_observables))
            ax_jac.set_yticklabels(observable_names, fontsize=7)
            ax_jac.set_title('Posterior Jacobian: d(log observable) / d(log parameter)')
            fig_jac.colorbar(im, ax=ax_jac, shrink=0.8, label='Elasticity')
            fig_jac.tight_layout()
            fig_jac.savefig(SAVE_DIR / 'jacobian_heatmap.png', dpi=150, bbox_inches='tight')
            plt.close(fig_jac)

            # SVD spectrum
            fig_svd, ax_svd = plt.subplots(figsize=(8, 4))
            ax_svd.semilogy(range(len(S)), S, 'o-', markersize=4)
            ax_svd.axhline(sloppy_threshold, color='red', linestyle='--', alpha=0.5,
                           label=f'1% of max ({sloppy_threshold:.2g})')
            ax_svd.set_xlabel('Singular value index')
            ax_svd.set_ylabel('Singular value')
            ax_svd.set_title(f'Posterior Jacobian SVD (eff. rank = {n_eff_jac:.1f})')
            ax_svd.legend()
            fig_svd.tight_layout()
            fig_svd.savefig(SAVE_DIR / 'svd_spectrum.png', dpi=150, bbox_inches='tight')
            plt.close(fig_svd)

            # Target coupling heatmap (desired correction directions)
            fig_coup, ax_coup = plt.subplots(
                figsize=(max(8, num_observables * 0.4), max(8, num_observables * 0.4))
            )
            coup_norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
            im2 = ax_coup.imshow(coupling_matrix, aspect='auto', cmap='RdBu_r',
                                 norm=coup_norm)
            ax_coup.set_xticks(range(num_observables))
            ax_coup.set_xticklabels(observable_names, rotation=90, fontsize=7)
            ax_coup.set_yticks(range(num_observables))
            ax_coup.set_yticklabels(observable_names, fontsize=7)
            ax_coup.set_title('Target coupling (desired correction cosine similarity)')
            fig_coup.colorbar(im2, ax=ax_coup, shrink=0.8, label='Cosine similarity')
            fig_coup.tight_layout()
            fig_coup.savefig(SAVE_DIR / 'target_coupling_heatmap.png', dpi=150,
                             bbox_inches='tight')
            plt.close(fig_coup)

            print(f"\n  Saved: jacobian_heatmap.png, svd_spectrum.png, target_coupling_heatmap.png")

        # --- 7f. CAL-TARGET POSTERIOR TRAJECTORY PLOT ---
        # Only when SAVE_EVOLVE_TRAJECTORIES — the per-sim binaries the
        # plot depends on were dumped during step 7a's
        # simulate_with_parameters call. Skipped silently otherwise.
        if SAVE_EVOLVE_TRAJECTORIES:
            print("\n  ----------------------------------------------------------------------")
            print("  7f. CAL-TARGET POSTERIOR TRAJECTORY PLOT (burn-in)")
            print("  ----------------------------------------------------------------------")
            try:
                from workflows.calibration_trajectory_pipeline import (
                    collect_required_trajectory_columns,
                    load_cal_targets_from_dirs,
                    run_calibration_trajectory_pipeline,
                )
                import pyarrow.parquet as _pq

                from qsp_hpc.cpp.runner import read_binary_trajectory

                for scen_name in scenario_names:
                    sim_scen = scenario_meta[scen_name]['simulator']
                    traj_dir = SAVE_DIR / 'evolve_trajectories' / scen_name
                    if not traj_dir.is_dir() or not list(traj_dir.glob('*.bin')):
                        print(f"    {scen_name}: no trajectory binaries in {traj_dir}; "
                              f"skipping (was the PPC step run with this flag?)")
                        continue

                    # Resolve species / compartment / rule column names
                    # from the species_parquet schema + a sample binary
                    # header. CppBatchRunner doesn't cache names on the
                    # instance — they only land in BatchResult — so we
                    # read them off the suffix-pool parquet emitted by
                    # the PPC step.
                    suffix_glob = (
                        sim_scen.pool_dir.parent
                        / f"{sim_scen.pool_dir.name}_posterior_predictive_*"
                    )
                    species_parquets = sorted(suffix_glob.parent.glob(
                        f"{suffix_glob.name}/species_*.parquet"
                    ))
                    if not species_parquets:
                        print(f"    {scen_name}: no PPC species parquet; skipping")
                        continue
                    schema = _pq.read_schema(str(species_parquets[-1]))
                    meta_cols = {"sample_index", "simulation_id", "status", "time"}
                    data_cols = [
                        c for c in schema.names
                        if c not in meta_cols and not c.startswith("param:")
                    ]
                    sample_traj = next(traj_dir.glob('*.bin'))
                    _, hdr = read_binary_trajectory(sample_traj)
                    sp_names = data_cols[: hdr.n_species]
                    cp_names = data_cols[hdr.n_species : hdr.n_species + hdr.n_compartments]
                    rl_names = data_cols[hdr.n_species + hdr.n_compartments :]

                    # Filter to successful sims only — the C++ side writes a
                    # binary even for rejected sims (too-fast / never-reached
                    # max_days), and mixing those into the fraction-normalized
                    # band produces a visible 'transient' at the right edge
                    # because rejected sims have qualitatively different
                    # t_diag values that distort normalization.
                    status_tbl = _pq.read_table(
                        str(species_parquets[-1]),
                        columns=["sample_index", "status"],
                    ).to_pandas()
                    success_indices = status_tbl.loc[
                        status_tbl["status"] == 0, "sample_index"
                    ].tolist()
                    if not success_indices:
                        print(f"    {scen_name}: no successful PPC sims; skipping")
                        continue

                    cal_dirs = SCENARIOS.get(scen_name, {}).get('calibration_targets', [])
                    cal_targets = load_cal_targets_from_dirs(
                        cal_dirs, model_structure=MODEL_STRUCTURE_FILE,
                    )
                    if not cal_targets:
                        print(f"    {scen_name}: no calibration target YAMLs; skipping")
                        continue
                    columns_needed = collect_required_trajectory_columns(cal_targets)
                    # Drop columns that aren't in the trajectory (e.g.
                    # rho_collagen lands as a model parameter, not a
                    # species/compartment/rule, so it isn't dumped). The
                    # plotter recovers these via parameter_records below.
                    available = set(sp_names) | set(cp_names) | set(rl_names)
                    columns_needed = [c for c in columns_needed if c in available]

                    # Per-sim parameter + auxiliary records for cal targets
                    # that read varied prior params (rho_collagen) or
                    # nuisance auxiliaries (f_serum_to_tumor_X) inside
                    # compute_observable. Without these, the underlying
                    # evaluator KeyErrors → NaN row → log_y='auto' wipes
                    # the panel. post_samples_dict carries every column
                    # (QSP params + aux + varied priors) but for the FULL
                    # 2k-ish posterior; the PPC sims used the subsampled
                    # ``idx`` rows (line 1813). So PPC sample_index k ↔
                    # post_samples_dict row idx[k].
                    _ppc_records_all = [
                        {name: float(post_samples_dict[name][idx[k]])
                         for name in post_samples_dict}
                        for k in range(N_POSTERIOR_SIMS)
                    ]
                    _records = [_ppc_records_all[i] for i in success_indices]

                    # Bifurcate on the scenario YAML's sim_config.stop_time:
                    # baseline-snapshot scenarios (stop_time <= 1) only have
                    # diagnosis-state targets — the burn-in evolve trajectories
                    # are what's informative. Treatment / clinical scenarios
                    # (stop_time > 1) evaluate targets at d=21/30/365 — the
                    # post-scenario dynamics are what's informative, not the
                    # pre-treatment burn-in. Reading
                    # ``sim_scen.t_end_days`` is unreliable: the field is set
                    # from cfg.cpp_t_end_days (defaults to 1.0) and isn't
                    # overridden per-scenario, while actual simulation duration
                    # comes from the scenario yaml's sim_config.stop_time.
                    import yaml as _yaml
                    try:
                        with open(sim_scen.scenario_yaml) as _sf:
                            _scen_yaml = _yaml.safe_load(_sf)
                        stop_time = float(
                            (_scen_yaml or {}).get('sim_config', {}).get('stop_time', 1.0)
                        )
                    except Exception:
                        stop_time = 1.0

                    if stop_time <= 1.0:
                        out_path = SAVE_DIR / f'cal_target_trajectories_{scen_name}.png'
                        run_calibration_trajectory_pipeline(
                            phase='burn_in',
                            traj_dir=traj_dir,
                            species_names=sp_names,
                            compartment_names=cp_names,
                            rule_names=rl_names,
                            calibration_targets=cal_targets,
                            output_path=out_path,
                            title=f'{scen_name} — posterior burn-in trajectories',
                            scenario_name=scen_name,
                            columns_needed=columns_needed or None,
                            sample_indices=success_indices,
                            parameter_records=_records,
                            auxiliary_records=_records,
                            # Default to fraction-of-evolve x-axis (same-length
                            # trajectories across draws with variable t_diag)
                            # and auto log-y for lognormal-ish targets (CI95
                            # ratio >= 10×). Both are documented opt-outs of
                            # the underlying plotter for callers that prefer
                            # absolute time / linear y.
                            time_axis='fraction',
                            log_y='auto',
                        )
                        print(f"    {scen_name}: wrote {out_path.name} (burn-in)")
                    else:
                        out_path = SAVE_DIR / f'cal_target_trajectories_{scen_name}.png'
                        # post-scenario assembler reads list-typed cols, so
                        # restrict to species/comp/rule names actually in
                        # the parquet schema (no rho_collagen-style param
                        # injection yet — same caveat as burn-in).
                        run_calibration_trajectory_pipeline(
                            phase='post_scenario',
                            species_parquet=species_parquets[-1],
                            calibration_targets=cal_targets,
                            output_path=out_path,
                            title=f'{scen_name} — posterior post-scenario trajectories',
                            scenario_name=scen_name,
                            columns_needed=columns_needed or None,
                            sample_indices=success_indices,
                            parameter_records=_records,
                            auxiliary_records=_records,
                            log_y='auto',
                        )
                        print(f"    {scen_name}: wrote {out_path.name} (post-scenario)")
            except Exception as e:
                # Don't let a plot-step failure poison the rest of the run.
                # Log loudly and move on; the PPC sims + diagnostics already
                # landed by this point.
                print(f"    [WARNING] cal-target trajectory plot failed: {e}")
                import traceback
                traceback.print_exc()

    # ============================================================================
    # 8. SAVE
    # ============================================================================
    print("\n" + "=" * 80)
    print("8. SAVE RESULTS")
    print("=" * 80)

    # Posterior samples
    posterior_df = pd.DataFrame(post_samples_dict)
    posterior_df.to_csv(SAVE_DIR / 'posterior_samples.csv', index=False)
    print(f"  Posterior samples: {SAVE_DIR}/posterior_samples.csv")

    # Posterior summary
    post_summary = posterior_df.describe(percentiles=[0.025, 0.25, 0.5, 0.75, 0.975]).T
    post_summary.to_csv(SAVE_DIR / 'posterior_summary.csv')

    # Diagnostics
    diagnostics_dir = SAVE_DIR / 'diagnostics'
    written = save_diagnostics(
        diagnostics_dir,
        r2_values=r2_values,
        z_scores=z_scores,
        contractions=contractions,
        param_names=param_names,
        loo_df=loo_prior_df,
        ppc_df=coverage_df if RUN_POSTERIOR_PREDICTIVE else None,
        training_metadata={
            'n_train': int(theta_train.shape[0]),
            'n_test': int(theta_test.shape[0]),
            'n_params': num_params,
            'n_params_full': num_params_full,
            'n_observables': num_observables,
            'n_raw_sims': NUM_TOTAL_SIMULATIONS,
            'n_posterior_sims': N_POSTERIOR_SIMS if RUN_POSTERIOR_PREDICTIVE else 0,
            'model_version': MODEL_VERSION,
            'scenarios': scenario_names,
            'npe_hidden_features': NPE_HIDDEN_FEATURES,
            'npe_num_transforms': NPE_NUM_TRANSFORMS,
            'npe_num_bins': NPE_NUM_BINS,
            'n_nn_params': n_nn_params,
            'n_epochs': n_epochs,
            'best_val_loss': float(best_val_loss),
            'prcc_tier': PRCC_TIER,
            'prcc_tier_cutoff': TIER_CUTOFFS.get(PRCC_TIER) if PRCC_TIER is not None else None,
        },
    )

    prior_pred_df.to_csv(diagnostics_dir / 'prior_predictive_coverage.csv', index=False)

    if RUN_POSTERIOR_PREDICTIVE:
        loo_post_df.to_csv(diagnostics_dir / 'loo_posterior_predictive.csv', index=False)
        coverage_df.to_csv(diagnostics_dir / 'posterior_predictive_coverage.csv', index=False)

        effective_dof_df = pd.DataFrame([
            {'distribution': 'prior', 'k_eff': k_eff_prior, 'd2': d2_prior, 'chi2_p': chi2_p_prior},
            {'distribution': 'posterior_actual', 'k_eff': k_eff_post, 'd2': d2_post, 'chi2_p': chi2_p_post},
        ])
    else:
        effective_dof_df = pd.DataFrame([
            {'distribution': 'prior', 'k_eff': k_eff_prior, 'd2': d2_prior, 'chi2_p': chi2_p_prior},
        ])
    effective_dof_df.to_csv(diagnostics_dir / 'effective_dof.csv', index=False)

    print(f"  Diagnostics: {diagnostics_dir}/")
    for key, path in written.items():
        print(f"    {key}")

    print("\n" + "=" * 80)
    print("SBI WORKFLOW COMPLETE")
    print("=" * 80)

    metrics = {
        'save_dir': str(SAVE_DIR),
        'n_train': int(theta_train.shape[0]),
        'n_test': int(theta_test.shape[0]),
        'n_params': num_params,
        'n_params_full': num_params_full,
        'n_observables': num_observables,
        'n_raw_sims': NUM_TOTAL_SIMULATIONS,
        'n_nn_params': int(n_nn_params),
        'n_epochs': int(n_epochs),
        'best_val_loss': float(best_val_loss),
        'median_r2_global': float(np.nanmedian(r2_array)) if np.isfinite(r2_array).any() else None,
        'median_r2_local': float(np.nanmedian(r2_local_array)) if np.isfinite(r2_local_array).any() else None,
        'median_contraction_global': float(median_contraction) if np.isfinite(median_contraction) else None,
        'median_contraction_local': float(median_contraction_local) if np.isfinite(median_contraction_local) else None,
        'n_miscal_global': int(n_miscal_global),
        'n_miscal_local': int(n_miscal_local),
        'seed': SEED,
        'prcc_tier': PRCC_TIER,
        'npe_hidden_features': NPE_HIDDEN_FEATURES,
        'npe_num_transforms': NPE_NUM_TRANSFORMS,
        'n_train_subsample': N_TRAIN_SUBSAMPLE,
    }
    with open(SAVE_DIR / 'run_metrics.json', 'w') as _f:
        json.dump(metrics, _f, indent=2)
    with open(SAVE_DIR / 'run_config.json', 'w') as _f:
        cfg_dict = asdict(cfg)
        cfg_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg_dict.items()}
        json.dump(cfg_dict, _f, indent=2)
    return metrics


# ============================================================================
# CLI
# ============================================================================

def _build_cli_parser():
    import argparse
    p = argparse.ArgumentParser(description='Multi-scenario SBI runner for PDAC')
    p.add_argument('--config-json', type=Path, default=None,
                   help='Path to JSON file with RunConfig fields (overrides CLI flags below)')
    p.add_argument('--scenarios', type=str, default=None,
                   help='Comma-separated subset of scenario names from DEFAULT_SCENARIOS '
                        '(e.g. "baseline_no_treatment,clinical_progression"). Defaults to '
                        'all four scenarios. Changes the pool key (save dir is auto-derived '
                        'from scenarios), so a subset run gets its own figures dir.')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--npe-seed', type=int, default=None,
                   help='NPE training init + posterior sampling + PPC subsample RNG. '
                        'Vary this alone for reseed-stability audits.')
    p.add_argument('--noise-seed', type=int, default=None,
                   help='Observation-noise injection RNG.')
    p.add_argument('--cohort-seed', type=int, default=None,
                   help='Cohort RE shift RNG (uses this and this+1 for train/test).')
    p.add_argument('--eval-npe-theta-space', action=argparse.BooleanOptionalAction,
                   default=None,
                   help='Run step 5+5b (test-set posterior, R², SBC, local cal). '
                        'Default False — skipped for fast misspec iteration. '
                        'Use --eval-npe-theta-space to enable.')
    p.add_argument('--prcc-tier', type=int, default=None)
    p.add_argument('--no-prcc-tier', action='store_true',
                   help='Force prcc_tier=None (use all params)')
    p.add_argument('--hidden-features', type=int, default=None)
    p.add_argument('--num-transforms', type=int, default=None)
    p.add_argument('--num-bins', type=int, default=None)
    p.add_argument('--num-total-simulations', type=int, default=None)
    p.add_argument('--model-version', type=str, default=None,
                   help='Override model_version (changes config_hash, invalidates caches)')
    p.add_argument('--n-train-subsample', type=int, default=None)
    p.add_argument('--save-dir', type=Path, default=None)
    p.add_argument('--run-posterior-predictive', action='store_true',
                   help='Run section 7 (posterior predictive sims + misspec '
                        'verdict). Default since 2026-05-04 — pass '
                        '--no-posterior-predictive to skip.')
    p.add_argument('--no-posterior-predictive', action='store_true',
                   help='Skip section 7 (PPC + misspec assessment). Use for '
                        'fast smoke checks where only training-time '
                        'diagnostics are needed.')
    p.add_argument('--save-evolve-trajectories', action=argparse.BooleanOptionalAction,
                   default=None,
                   help='During PPC, dump per-sim burn-in trajectories under '
                        'SAVE_DIR/evolve_trajectories/<scenario>/ and render '
                        'cal_target_trajectories_<scenario>.png per scenario '
                        '(observable posterior band + observed CI95 at t=0). '
                        'Forces fresh sims for the PPC step (suffix-pool '
                        'cache is bypassed for trajectory-enabled calls). '
                        'Default: enabled. Use --no-save-evolve-trajectories '
                        'to opt out.')
    p.add_argument('--evolve-trajectory-dt-days', type=float, default=None,
                   help='Sampling interval (model-time days) for the burn-in '
                        'trajectory dump. Default 14 (biweekly). Pass 0 or '
                        'omit-with-default to use the evolve spec\'s step_days.')
    p.add_argument('--cpp-branch', type=str, default=None,
                   help='SPQSP_PDAC branch for qsp_sim build on HPC (overrides cpp.branch credential default)')
    p.add_argument('--tools-branch', type=str, default=None,
                   help='qsp-hpc-tools pip ref (overrides the @<ref> suffix of package.qsp_hpc_tools_source)')
    p.add_argument('--codegen-branch', type=str, default=None,
                   help='qsp-codegen pip ref (overrides the @<ref> suffix of cpp.codegen_source). '
                        'Useful when an unmerged qsp-codegen branch ships headers/sources that '
                        'the HPC qsp_sim build needs.')
    p.add_argument('--cpp-jobs-per-chunk', type=int, default=None,
                   help='Sims per SLURM array task (overrides credentials slurm.jobs_per_chunk; '
                        '100 is a good starting point for N=10k runs)')
    p.add_argument('--pretrained-dir', type=Path, default=None,
                   help='Load a pretrained density_estimator.pt from this dir (e.g. Delta GPU '
                        'output) instead of training locally. Expects learning_curve.csv + '
                        'metrics.json alongside it.')
    p.add_argument('--sample-on-delta', action='store_true',
                   help='Offload NPE sampling to Delta GPU in one batched round-trip '
                        '(requires --pretrained-dir).')
    p.add_argument('--cache-processed-tensors', action='store_true',
                   help='Cache post-step-2 processed tensors to cache/sbi_processed/ '
                        'so subsequent runs skip ~3-4 min sim rehydration. Incompatible '
                        'with run_posterior_predictive or misspec_detection_mode — '
                        'pass --no-posterior-predictive alongside this flag, since '
                        'PPC is on by default.')
    p.add_argument('--stage-artifacts-only', action='store_true',
                   help='Stop after step 2 + artifact write (theta_train.pt, '
                        'x_train.pt, npe_config.json, etc.). Skips training + '
                        'diagnostics. Intended for sweep drivers that train on Delta.')
    p.add_argument('--restriction-classifier-dir', type=str, default=None,
                   help='Path to a RestrictionClassifier dir (classifier.pkl + '
                        'metadata.json) produced by workflows/restrict_prior_train.py. '
                        'When set, the theta pool is rejection-sampled against the '
                        'classifier (new pool cache key — does not overwrite the '
                        'unrestricted pool). Defaults to '
                        'tests/fixtures/restriction_classifier.')
    p.add_argument('--no-restriction-classifier', action='store_true',
                   help='Force restriction_classifier_dir=None (unfiltered prior).')
    p.add_argument('--restriction-threshold', type=float, default=None,
                   help='p(valid) threshold for classifier acceptance (default 0.5).')
    p.add_argument('--skip-submodel-sync', action='store_true',
                   help='Use pdac_priors.csv as-is (independent normals per param). '
                        'Skips the static composite prior loaded from '
                        'submodel_priors.yaml. Drops submodel_priors_yaml from the '
                        'simulator pool key so a CSV-only run does not share a pool '
                        'with composite-prior runs. CLI opt-out flag — composite '
                        'prior is the default since 2026-05-04.')
    p.add_argument('--with-submodel-sync', action='store_true',
                   help='Explicitly load the static composite prior from '
                        'submodel_priors.yaml. Default since 2026-05-04 — kept '
                        'as a no-op flag for self-documenting launch lines.')
    p.add_argument('--strict-submodel-freshness', action='store_true',
                   help='Abort if submodel_priors.yaml has stale per-component '
                        'fingerprints (or no fingerprint header). Default: warn '
                        'and continue. Has no effect with --skip-submodel-sync.')
    p.add_argument('--null-shuffle', action='store_true',
                   help='Diagnostic: permute x_train rows relative to theta_train '
                        'before NPE training, destroying all true (theta,x) coupling. '
                        'Auto-suffixes save_dir with _null. Pair with a matched real '
                        'run on the same pool; per-param artifact floor = '
                        'contraction_local from this run.')
    p.add_argument('--null-shuffle-seed', type=int, default=None,
                   help='Seed for the null-shuffle permutation (default 9999). Vary '
                        'across runs to estimate floor variance.')
    p.add_argument('--tau-cohort', type=float, default=None,
                   help='Cohort random-effect SD in copula-Gaussian space. Adds '
                        'a per-(row, cohort) Gaussian shift to training targets, '
                        'where cohort_id = primary_data_source.source_tag. 0=off. '
                        'Try 0.3, 0.5, 0.7, 1.0 and compare prior-PC pairwise φ.')
    return p


def _cfg_from_cli(args) -> RunConfig:
    if args.config_json is not None:
        with open(args.config_json) as f:
            data = json.load(f)
        if data.get('save_dir'):
            data['save_dir'] = Path(data['save_dir'])
        return RunConfig(**data)
    kwargs = {}
    if args.seed is not None:
        kwargs['seed'] = args.seed
    if args.npe_seed is not None:
        kwargs['npe_seed'] = args.npe_seed
    if args.noise_seed is not None:
        kwargs['noise_seed'] = args.noise_seed
    if args.cohort_seed is not None:
        kwargs['cohort_seed'] = args.cohort_seed
    if args.eval_npe_theta_space is not None:
        kwargs['eval_npe_theta_space'] = args.eval_npe_theta_space
    if args.no_prcc_tier:
        kwargs['prcc_tier'] = None
    elif args.prcc_tier is not None:
        kwargs['prcc_tier'] = args.prcc_tier
    if args.hidden_features is not None:
        kwargs['npe_hidden_features'] = args.hidden_features
    if args.num_transforms is not None:
        kwargs['npe_num_transforms'] = args.num_transforms
    if args.num_bins is not None:
        kwargs['npe_num_bins'] = args.num_bins
    if args.num_total_simulations is not None:
        kwargs['num_total_simulations'] = args.num_total_simulations
    if args.model_version is not None:
        kwargs['model_version'] = args.model_version
    if args.n_train_subsample is not None:
        kwargs['n_train_subsample'] = args.n_train_subsample
    if args.save_dir is not None:
        kwargs['save_dir'] = args.save_dir
    if args.no_posterior_predictive and args.run_posterior_predictive:
        raise SystemExit(
            "error: --no-posterior-predictive and --run-posterior-predictive "
            "are mutually exclusive."
        )
    if args.no_posterior_predictive:
        kwargs['run_posterior_predictive'] = False
    elif args.run_posterior_predictive:
        kwargs['run_posterior_predictive'] = True
    if args.save_evolve_trajectories is not None:
        kwargs['save_evolve_trajectories'] = args.save_evolve_trajectories
    if args.evolve_trajectory_dt_days is not None:
        kwargs['evolve_trajectory_dt_days'] = args.evolve_trajectory_dt_days
    if args.cpp_branch is not None:
        kwargs['cpp_branch'] = args.cpp_branch
    if args.tools_branch is not None:
        kwargs['tools_branch'] = args.tools_branch
    if args.codegen_branch is not None:
        kwargs['codegen_branch'] = args.codegen_branch
    if args.cpp_jobs_per_chunk is not None:
        kwargs['cpp_jobs_per_chunk'] = args.cpp_jobs_per_chunk
    if args.pretrained_dir is not None:
        kwargs['pretrained_dir'] = args.pretrained_dir
    if args.sample_on_delta:
        kwargs['sample_on_delta'] = True
    if args.cache_processed_tensors:
        kwargs['cache_processed_tensors'] = True
    if args.stage_artifacts_only:
        kwargs['stage_artifacts_only'] = True
    if args.no_restriction_classifier:
        kwargs['restriction_classifier_dir'] = None
    elif args.restriction_classifier_dir is not None:
        kwargs['restriction_classifier_dir'] = args.restriction_classifier_dir
    if args.restriction_threshold is not None:
        kwargs['restriction_threshold'] = args.restriction_threshold
    if args.with_submodel_sync:
        kwargs['skip_submodel_sync'] = False
    elif args.skip_submodel_sync:
        kwargs['skip_submodel_sync'] = True
    if args.strict_submodel_freshness:
        kwargs['strict_submodel_freshness'] = True
    if args.tau_cohort is not None:
        kwargs['tau_cohort'] = args.tau_cohort
    if args.null_shuffle:
        kwargs['null_shuffle'] = True
    if args.null_shuffle_seed is not None:
        kwargs['null_shuffle_seed'] = args.null_shuffle_seed
    if args.scenarios:
        wanted = [s.strip() for s in args.scenarios.split(',') if s.strip()]
        unknown = [s for s in wanted if s not in DEFAULT_SCENARIOS]
        if unknown:
            raise SystemExit(
                f"Unknown scenario(s): {unknown}. "
                f"Known: {list(DEFAULT_SCENARIOS.keys())}"
            )
        kwargs['scenarios'] = {s: DEFAULT_SCENARIOS[s] for s in wanted}
    return RunConfig(**kwargs)


if __name__ == '__main__':
    parser = _build_cli_parser()
    args = parser.parse_args()
    cfg = _cfg_from_cli(args)
    run_inference(cfg)
