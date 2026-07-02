#!/usr/bin/env python3
"""Sequential SNPE-C runner for the PDAC multi-scenario NPE.

Round 1 reuses the h256_t5 capacity-sweep run + the cached processed bundle
under ``cache/sbi_processed/``. Each subsequent round:

1. Sample N proposals from the current posterior at ``x_obs``.
2. Reject low-``p(valid)`` thetas via the trained restriction classifier
   (``cache/restriction_classifier/``).
3. Simulate the accepted thetas against all four scenarios via
   :meth:`CppSimulator.simulate_with_parameters` (local or HPC backend).
4. Joint-NaN-filter, copula-transform using the *round-1* quantiles
   (rounds must share a latent space for the density estimator), and
   stage tensors under the run dir.
5. Ship the cumulative round manifest to Delta GPU
   (``workflows/delta/run_delta_sequential_training.sh``), which trains
   atomic-loss SNPE-C across all rounds via
   :mod:`workflows.delta.train_npe_sequential`.
6. Download the new estimator, rebuild the posterior, log convergence
   (Wasserstein-1 of marginals vs the previous round's samples).

Usage::

    .venv/bin/python -u workflows/sequential_runner.py \
        --num-rounds 4 --n-per-round 500 --backend local \
        --run-tag seq_$(date +%Y%m%d_%H%M%S)

For the HPC backend, the run requires the same SSH/SLURM credentials
``sbi_runner.py`` uses. Each round submits its own theta-hash-keyed pool;
no theta-pool collisions across rounds.
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn

from qsp_hpc.batch.hpc_job_manager import HPCJobManager
from qsp_hpc.simulation.cpp_simulator import CppSimulator
from qsp_inference.inference.gaussian_copula_transform import transform_to_normal_from_array
from qsp_inference.inference.restriction import RestrictionClassifier


_REPO = Path(__file__).resolve().parent.parent
_SPQSP_ROOT = Path('/Users/joeleliason/Projects/SPQSP_PDAC')

DEFAULT_SCENARIOS = {
    'baseline_no_treatment':           'calibration_targets/baseline_no_treatment',
    'clinical_progression':            'calibration_targets/clinical_progression',
    'gvax_neoadjuvant_zheng2022':      'calibration_targets/gvax_neoadjuvant',
    'gvax_nivo_neoadjuvant_zheng2022': 'calibration_targets/gvax_nivo_neoadjuvant',
}

DEFAULT_BUNDLE = (
    _REPO / 'cache' / 'sbi_processed'
    / 'multi_scenario_v4_all_scenarios__'
      'baseline_no_treatment-clinical_progression-'
      'gvax_neoadjuvant_zheng2022-gvax_nivo_neoadjuvant_zheng2022__'
      'tiernone__suball__split2027__s2025__restr2adbb9b075.pkl'
)
DEFAULT_ROUND1_DIR = _REPO / 'figures' / 'sbi_multi' / 'sweep_capacity' / 'delta_out' / 'h256_t5'


@dataclass
class SeqConfig:
    num_rounds: int = 4
    n_per_round: int = 500
    backend: str = 'local'  # 'local' or 'hpc'
    classifier_threshold: float = 0.5
    run_tag: str = ''  # auto-derived if empty
    bundle_path: Path = DEFAULT_BUNDLE
    round1_dir: Path = DEFAULT_ROUND1_DIR
    classifier_dir: Path = _REPO / 'cache' / 'restriction_classifier'
    scenarios: dict = field(default_factory=lambda: dict(DEFAULT_SCENARIOS))
    scenarios_dir: Path = _REPO / 'scenarios'
    priors_csv: Path = _REPO / 'parameters' / 'pdac_priors.csv'
    submodel_priors_yaml: Path = (
        _REPO / 'calibration_targets' / 'submodel_targets' / 'submodel_priors.yaml'
    )
    model_structure_file: Path = _REPO / 'model_structure.json'
    cpp_binary_path: Path = _SPQSP_ROOT / 'PDAC' / 'qsp' / 'sim' / 'build' / 'qsp_sim'
    cpp_template_xml: Path = _SPQSP_ROOT / 'PDAC' / 'sim' / 'resource' / 'param_all.xml'
    cpp_drug_metadata_yaml: Path = _SPQSP_ROOT / 'PDAC' / 'sim' / 'resource' / 'drug_metadata.yaml'
    cpp_healthy_state_yaml: Path = _SPQSP_ROOT / 'PDAC' / 'sim' / 'resource' / 'healthy_state.yaml'
    cpp_subtree: str = 'QSP'
    cpp_t_end_days: float = 1.0
    cpp_dt_days: float = 0.5
    model_version: str = 'multi_scenario_v4_all_scenarios'
    seed: int = 2025
    cpp_branch: Optional[str] = None
    tools_branch: Optional[str] = None
    cpp_jobs_per_chunk: Optional[int] = None
    out_root: Path = _REPO / 'figures' / 'sbi_sequential'
    npe_hidden_features: int = 256
    npe_num_transforms: int = 5
    npe_num_bins: int = 10
    npe_batch_size: int = 2048
    npe_max_epochs: int = 80
    npe_num_atoms: int = 5  # SNPE-C atomic-loss inner-loop size (sbi default 10)
    sample_chunk: int = 200  # how many posterior draws per .sample() call
    # Warm-start: skip retraining round 1 from scratch on Delta and use the
    # cached h256_t5 estimator as the network's initial weights for all
    # subsequent rounds. Round 1's data is still recorded in the manifest
    # (for the throwaway MVN prior fit) but no append_simulations / train
    # pass runs for it. Each later round trains with retrain_from_scratch
    # =False on its own (small) data only — atomic loss runs on ~300 rows
    # × num_atoms instead of ~91k × num_atoms, ~30x speedup.
    warm_start_round1: bool = True


def _build_simulators(cfg: SeqConfig, job_manager: Optional[HPCJobManager]) -> dict:
    """One CppSimulator per scenario, configured for either backend.

    For HPC mode the simulators share a single HPCJobManager (Duo / SSH
    multiplexing); for local mode they are independent.
    """
    sims = {}
    for scen, cal_dir in cfg.scenarios.items():
        sims[scen] = CppSimulator(
            priors_csv=str(cfg.priors_csv),
            binary_path=str(cfg.cpp_binary_path),
            template_xml=str(cfg.cpp_template_xml),
            subtree=cfg.cpp_subtree,
            model_version=cfg.model_version,
            scenario=scen,
            submodel_priors_yaml=str(cfg.submodel_priors_yaml),
            scenario_yaml=str(cfg.scenarios_dir / f'{scen}.yaml'),
            drug_metadata_yaml=str(cfg.cpp_drug_metadata_yaml),
            healthy_state_yaml=str(cfg.cpp_healthy_state_yaml),
            calibration_targets=str(_REPO / cal_dir),
            model_structure_file=str(cfg.model_structure_file),
            job_manager=job_manager,
            t_end_days=cfg.cpp_t_end_days,
            dt_days=cfg.cpp_dt_days,
            seed=cfg.seed,
            verbose=True,
        )
    return sims


def _ts_columns_in_order(table_columns: list[str], target_ids: list[str]) -> list[str]:
    """Return ``ts:<id>`` column names in the order ``target_ids`` requests.

    simulate_with_parameters now emits named ``ts:<test_statistic_id>``
    columns (PR #51 onward) so this is a straight rename, not positional.
    """
    cols = []
    missing = []
    for tid in target_ids:
        col = f'ts:{tid}'
        if col in table_columns:
            cols.append(col)
        else:
            missing.append(tid)
    if missing:
        raise RuntimeError(f"simulator output missing ts columns: {missing[:5]}")
    return cols


def _simulate_one_scenario(
    scen: str, sim: CppSimulator, theta_orig: np.ndarray, target_ids: list[str], backend: str,
    pool_suffix: str,
) -> np.ndarray:
    """Run ``sim`` at the given theta and return ``(n, len(target_ids))`` x_raw."""
    print(f"  [{scen}] simulating {len(theta_orig)} thetas (backend={backend})", flush=True)
    t0 = time.time()
    _, table = sim.simulate_with_parameters(
        theta_orig, backend=backend, pool_suffix=pool_suffix,
    )
    cols = _ts_columns_in_order(table.column_names, target_ids)
    x_raw = np.column_stack([table.column(c).to_numpy() for c in cols])
    print(f"  [{scen}] done in {time.time()-t0:.1f}s  x={x_raw.shape}", flush=True)
    return x_raw


def _stage_round_tensors(
    round_idx: int, theta_log: np.ndarray, x_copula: np.ndarray, staging: Path,
) -> tuple[Path, Path]:
    """Save (theta_rN.pt, x_rN.pt) under staging/. Returns the two paths."""
    staging.mkdir(parents=True, exist_ok=True)
    theta_path = staging / f'theta_r{round_idx}.pt'
    x_path = staging / f'x_r{round_idx}.pt'
    torch.save(torch.tensor(theta_log, dtype=torch.float32), theta_path)
    torch.save(torch.tensor(x_copula, dtype=torch.float32), x_path)
    return theta_path, x_path


def _build_posterior_from_round_dir(
    estimator_dir: Path, theta_train_r1: torch.Tensor, x_train_r1: torch.Tensor,
    obs_tensor: torch.Tensor,
):
    """Reconstruct a sbi DirectPosterior from a downloaded ``density_estimator.pt``.

    Matches train_npe_sequential's setup: throwaway MVN prior moment-matched
    to the round-1 theta. Uses one append_simulations call so sbi's NPE
    object has a populated ``_neural_net``; we then plug the loaded estimator
    in and ask for the posterior. The training data we hand in is *only*
    used by sbi to read shapes — the real weights came off Delta.
    """
    estimator = torch.load(estimator_dir / 'density_estimator.pt', weights_only=False)
    prior = torch.distributions.MultivariateNormal(
        theta_train_r1.mean(0),
        torch.diag(theta_train_r1.var(0) + 1e-6),
    )
    de_factory = posterior_nn(model='nsf', hidden_features=256, num_transforms=5, num_bins=10)
    inference = NPE(prior=prior, density_estimator=de_factory)
    inference = inference.append_simulations(theta_train_r1, x_train_r1)
    posterior = inference.build_posterior(density_estimator=estimator)
    return posterior, estimator


def _wasserstein1_marginals(samples_a: np.ndarray, samples_b: np.ndarray) -> np.ndarray:
    """Per-parameter 1-Wasserstein distance via sorted-quantile diff.

    Both arrays are ``(n, n_params)``. Returns ``(n_params,)``.
    """
    if samples_a.shape[1] != samples_b.shape[1]:
        raise ValueError("param dim mismatch")
    n = min(len(samples_a), len(samples_b))
    a = np.sort(samples_a[:n], axis=0)
    b = np.sort(samples_b[:n], axis=0)
    return np.mean(np.abs(a - b), axis=0)


def _ship_to_delta(staging: Path, manifest: dict, run_tag: str, output_dir: Path) -> Path:
    """Upload staging dir + manifest, sbatch sequential trainer, download.

    Mirrors the env-var contract of ``run_delta_training.sh``:
    ``ARTIFACTS`` (input dir) and ``LOCAL_OUT`` (where results land).
    """
    (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    npe_cfg = manifest['npe_config']
    cfg_path = staging / 'npe_config.json'
    cfg_path.write_text(json.dumps(npe_cfg, indent=2))

    script = _REPO / 'workflows' / 'delta' / 'run_delta_sequential_training.sh'
    output_dir.mkdir(parents=True, exist_ok=True)
    env = {
        'ARTIFACTS': str(staging.resolve()),
        'LOCAL_OUT': str(output_dir.resolve()),
        'PATH': str(Path.home() / 'bin') + ':/usr/bin:/bin:/usr/local/bin',
    }
    print(f"  [delta] launching sequential trainer (tag={run_tag})", flush=True)
    subprocess.run(
        [str(script), run_tag],
        cwd=str(_REPO), env={**__import__('os').environ, **env}, check=True,
    )
    return output_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--num-rounds', type=int, default=4,
                    help="Total rounds including round 1 (>= 2).")
    ap.add_argument('--n-per-round', type=int, default=500,
                    help="Posterior draws sampled per round (before classifier rejection).")
    ap.add_argument('--backend', choices=['local', 'hpc'], default='local')
    ap.add_argument('--classifier-threshold', type=float, default=0.5)
    ap.add_argument('--run-tag', default='', help="If empty, auto-tagged with timestamp.")
    ap.add_argument('--bundle-path', type=Path, default=DEFAULT_BUNDLE)
    ap.add_argument('--round1-dir', type=Path, default=DEFAULT_ROUND1_DIR)
    ap.add_argument('--cpp-branch', default=None)
    ap.add_argument('--tools-branch', default=None)
    ap.add_argument('--cpp-jobs-per-chunk', type=int, default=None)
    ap.add_argument('--dry-run', action='store_true',
                    help="Stop after sampling+classifier-reject for round 2; skip sims and Delta.")
    args = ap.parse_args()

    if args.num_rounds < 2:
        raise ValueError("--num-rounds must be >= 2 (round 1 is reused from cache)")

    cfg = SeqConfig(
        num_rounds=args.num_rounds,
        n_per_round=args.n_per_round,
        backend=args.backend,
        classifier_threshold=args.classifier_threshold,
        run_tag=args.run_tag or f"seq_{time.strftime('%Y%m%d_%H%M%S')}",
        bundle_path=args.bundle_path,
        round1_dir=args.round1_dir,
        cpp_branch=args.cpp_branch,
        tools_branch=args.tools_branch,
        cpp_jobs_per_chunk=args.cpp_jobs_per_chunk,
    )

    print("=" * 80)
    print(f"SEQUENTIAL SNPE-C  tag={cfg.run_tag}  rounds={cfg.num_rounds}  "
          f"n_per_round={cfg.n_per_round}  backend={cfg.backend}")
    print("=" * 80)

    run_dir = cfg.out_root / 'runs' / cfg.run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    staging = run_dir / 'staging'
    rounds_dir = run_dir / 'rounds'
    rounds_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ------------------------------------------------------------
    # 1. Load round-1 state (bundle + estimator)
    # ------------------------------------------------------------
    print(f"\n[round 1] loading processed bundle  {cfg.bundle_path.name}")
    if not cfg.bundle_path.exists():
        raise FileNotFoundError(f"bundle missing: {cfg.bundle_path}")
    with open(cfg.bundle_path, 'rb') as f:
        bundle = pickle.load(f)

    theta_train_r1 = bundle['theta_train']  # log-space float32 tensor
    x_train_r1 = bundle['x_train']  # copula float32 tensor
    obs_transformed = bundle['obs_transformed']  # (n_obs,)
    obs_tensor = torch.tensor(obs_transformed, dtype=torch.float32).unsqueeze(0)  # (1, n_obs)
    param_names = list(bundle['param_names'])
    observable_names = list(bundle['observable_names'])
    scenario_meta = bundle['scenario_meta']  # per-scenario quantiles + obs names

    print(f"  theta_train_r1 {tuple(theta_train_r1.shape)}  "
          f"x_train_r1 {tuple(x_train_r1.shape)}  "
          f"obs {tuple(obs_tensor.shape)}")
    print(f"  scenarios in bundle: {list(scenario_meta.keys())}")

    # Verify scenarios match cfg
    if set(scenario_meta.keys()) != set(cfg.scenarios.keys()):
        raise RuntimeError(
            f"scenario mismatch — bundle has {sorted(scenario_meta.keys())}, "
            f"cfg has {sorted(cfg.scenarios.keys())}"
        )

    # Stage round-1 tensors so train_npe_sequential can replay it
    print(f"\n[round 1] staging tensors -> {staging}")
    _stage_round_tensors(1, theta_train_r1.numpy(), x_train_r1.numpy(), staging)

    # Save the round-1 estimator into the run dir for posterior reconstruction.
    print(f"[round 1] copying h256_t5 estimator from {cfg.round1_dir.name}/")
    r1_dir = rounds_dir / 'r1'
    r1_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg.round1_dir / 'density_estimator.pt', r1_dir / 'density_estimator.pt')
    shutil.copy(cfg.round1_dir / 'metrics.json', r1_dir / 'metrics.json')

    posterior_prev, _ = _build_posterior_from_round_dir(
        r1_dir, theta_train_r1, x_train_r1, obs_tensor,
    )

    # Cache prev round's posterior samples for convergence tracking
    print("[round 1] sampling baseline posterior for convergence reference (1000 draws)")
    with torch.no_grad():
        prev_post_log = posterior_prev.posterior_estimator.sample(
            torch.Size([1000]), condition=obs_tensor,
        ).squeeze(1).cpu().numpy()
    pd.DataFrame(prev_post_log, columns=param_names).to_csv(
        r1_dir / 'posterior_samples_log.csv', index=False,
    )

    # ------------------------------------------------------------
    # 2. Load classifier + simulators
    # ------------------------------------------------------------
    classifier = RestrictionClassifier.load(cfg.classifier_dir)
    print(f"\nclassifier: feature_order n={len(classifier.feature_order)}  "
          f"transform={classifier.input_transform}  default_tau={classifier.default_threshold}")

    job_manager = None
    if cfg.backend == 'hpc':
        job_manager = HPCJobManager(verbose=True)
        if cfg.cpp_branch:
            job_manager.config.cpp_branch = cfg.cpp_branch
        if cfg.tools_branch:
            src = job_manager.config.qsp_hpc_tools_source
            if '.git' in src:
                head, _sep, _tail = src.partition('.git')
                job_manager.config.qsp_hpc_tools_source = f'{head}.git@{cfg.tools_branch}'
        if cfg.cpp_jobs_per_chunk is not None:
            job_manager.config.jobs_per_chunk = cfg.cpp_jobs_per_chunk

    sims = _build_simulators(cfg, job_manager)

    # ------------------------------------------------------------
    # 3. Round loop
    # ------------------------------------------------------------
    convergence_rows: list[dict] = []
    for r in range(2, cfg.num_rounds + 1):
        round_dir = rounds_dir / f'r{r}'
        round_dir.mkdir(parents=True, exist_ok=True)
        print("\n" + "=" * 80)
        print(f"ROUND {r}/{cfg.num_rounds}")
        print("=" * 80)

        # 3a. Sample N proposals from the previous posterior (in chunks to
        # cap GPU memory; sbi's posterior_estimator.sample expects (n, 1, d)).
        print(f"  sampling {cfg.n_per_round} proposals from round-{r-1} posterior...")
        proposals_log: list[np.ndarray] = []
        with torch.no_grad():
            n_left = cfg.n_per_round
            while n_left > 0:
                k = min(cfg.sample_chunk, n_left)
                batch = posterior_prev.posterior_estimator.sample(
                    torch.Size([k]), condition=obs_tensor,
                ).squeeze(1).cpu().numpy()
                proposals_log.append(batch)
                n_left -= k
        theta_log = np.concatenate(proposals_log, axis=0)
        theta_orig = np.exp(theta_log)
        print(f"  sampled theta_log {theta_log.shape}, theta_orig range "
              f"{theta_orig.min():.2g}..{theta_orig.max():.2g}")

        # 3b. Reject low-p(valid) thetas.
        # Classifier was trained on the priors-CSV column order; bundle's
        # param_names is the same order (no PRCC tier in the snapshot).
        if classifier.feature_order != param_names:
            raise RuntimeError(
                f"classifier feature_order disagrees with bundle param_names "
                f"(diff={set(classifier.feature_order) ^ set(param_names)})"
            )
        p_valid = classifier.score(theta_orig)
        keep_mask = p_valid >= cfg.classifier_threshold
        n_keep = int(keep_mask.sum())
        accept_frac = n_keep / len(theta_orig)
        print(f"  classifier (tau={cfg.classifier_threshold}): kept {n_keep}/{len(theta_orig)} "
              f"({accept_frac:.0%}), p_valid mean={p_valid.mean():.3f}")
        theta_log_keep = theta_log[keep_mask]
        theta_orig_keep = theta_orig[keep_mask]
        if n_keep < 10:
            raise RuntimeError(
                f"classifier rejected almost everything ({n_keep} kept). "
                f"Threshold={cfg.classifier_threshold} is too tight for the current "
                f"posterior, or the proposal collapsed."
            )

        if args.dry_run and r == 2:
            print("\n--dry-run: stopping after classifier rejection in round 2.")
            return

        # 3c. Per-scenario simulation. Local backend is parallel-safe across
        # scenarios; HPC backend serializes to avoid sshd rate-limit storms
        # (same reason sbi_runner does).
        pool_suffix = f"sequential_round{r}_{cfg.run_tag}"
        scen_x: dict[str, np.ndarray] = {}
        scen_target_ids: dict[str, list[str]] = {
            scen: list(scenario_meta[scen]['observable_names'])
            for scen in cfg.scenarios
        }
        if cfg.backend == 'local':
            with ThreadPoolExecutor(max_workers=len(cfg.scenarios)) as exe:
                futures = {
                    scen: exe.submit(
                        _simulate_one_scenario,
                        scen, sims[scen], theta_orig_keep,
                        scen_target_ids[scen], cfg.backend, pool_suffix,
                    )
                    for scen in cfg.scenarios
                }
                for scen, fut in futures.items():
                    scen_x[scen] = fut.result()
        else:
            for scen in cfg.scenarios:
                scen_x[scen] = _simulate_one_scenario(
                    scen, sims[scen], theta_orig_keep,
                    scen_target_ids[scen], cfg.backend, pool_suffix,
                )

        # 3d. Joint NaN filter across scenarios. Every row that has any NaN
        # in any scenario is dropped — same convention as sbi_runner step 2d.
        valid_mask = np.ones(n_keep, dtype=bool)
        for scen, x in scen_x.items():
            valid_mask &= np.all(np.isfinite(x), axis=1)
        n_valid = int(valid_mask.sum())
        survival = n_valid / n_keep
        print(f"  joint NaN filter: {n_valid}/{n_keep} survived ({survival:.0%})")
        if n_valid < 10:
            raise RuntimeError(
                f"too few rows survived NaN filter ({n_valid}). "
                "Posterior is proposing thetas that fail simulation despite "
                "the classifier — investigate."
            )

        theta_log_clean = theta_log_keep[valid_mask]

        # 3e. Copula transform with the round-1 quantiles, per scenario.
        # observable order MUST match scenario_meta[scen]['observable_names'].
        x_copula_parts: list[np.ndarray] = []
        for scen in cfg.scenarios:  # iterate in cfg order = bundle scenario_meta order
            quantiles = scenario_meta[scen]['quantiles']  # (1000, n_obs_scen)
            x_clean = scen_x[scen][valid_mask]
            x_t = transform_to_normal_from_array(x_clean, quantiles)
            x_copula_parts.append(x_t)
        x_copula = np.concatenate(x_copula_parts, axis=1)
        if x_copula.shape[1] != obs_transformed.shape[0]:
            raise RuntimeError(
                f"x_copula obs dim {x_copula.shape[1]} != bundle obs dim {obs_transformed.shape[0]}"
            )

        # Save round-r tensors next to the round dir + into staging for Delta
        np.save(round_dir / 'theta_log.npy', theta_log_clean)
        np.save(round_dir / 'x_copula.npy', x_copula)
        _stage_round_tensors(r, theta_log_clean, x_copula, staging)

        # 3f. Build cumulative manifest + ship to Delta
        manifest = {
            'num_rounds': r,
            'obs_transformed': obs_transformed.astype(np.float64).tolist(),
            'npe_config': {
                'hidden_features': cfg.npe_hidden_features,
                'num_transforms': cfg.npe_num_transforms,
                'num_bins': cfg.npe_num_bins,
                'batch_size': cfg.npe_batch_size,
                'max_epochs': cfg.npe_max_epochs,
                'num_atoms': cfg.npe_num_atoms,
            },
            'rounds': [
                {'theta_file': f'theta_r{k}.pt', 'x_file': f'x_r{k}.pt',
                 'proposal': 'prior' if k == 1 else f'posterior_from_round_{k-1}'}
                for k in range(1, r + 1)
            ],
        }
        if cfg.warm_start_round1:
            # Stage the round-1 estimator alongside the tensors so the Delta
            # trainer can load it as the initial weights instead of training
            # round 1 from scratch. Filename is referenced by manifest so
            # the run-driver shell uploads it without special-casing.
            shutil.copy(
                cfg.round1_dir / 'density_estimator.pt',
                staging / 'density_estimator_round1.pt',
            )
            # Carry the round-1 metrics too so the trainer can report
            # best_val_loss for round 1 in metrics.json without retraining.
            shutil.copy(
                cfg.round1_dir / 'metrics.json',
                staging / 'metrics.json',
            )
            manifest['warm_start_from'] = 'density_estimator_round1.pt'
        delta_out = round_dir / 'delta_out'
        run_tag_round = f"{cfg.run_tag}_r{r}"
        _ship_to_delta(staging, manifest, run_tag_round, delta_out)

        # 3g. Reload trained estimator + rebuild posterior for next round
        print(f"  loading round-{r} estimator from {delta_out}/")
        posterior_prev, _ = _build_posterior_from_round_dir(
            delta_out, theta_train_r1, x_train_r1, obs_tensor,
        )

        # 3h. Convergence vs previous-round posterior samples
        with torch.no_grad():
            curr_post_log = posterior_prev.posterior_estimator.sample(
                torch.Size([1000]), condition=obs_tensor,
            ).squeeze(1).cpu().numpy()
        pd.DataFrame(curr_post_log, columns=param_names).to_csv(
            round_dir / 'posterior_samples_log.csv', index=False,
        )
        w1 = _wasserstein1_marginals(prev_post_log, curr_post_log)
        prev_post_log = curr_post_log

        # Per-round convergence row + write running CSV
        convergence_rows.append({
            'round': r,
            'n_proposed': cfg.n_per_round,
            'n_kept_classifier': n_keep,
            'classifier_accept_frac': accept_frac,
            'n_valid_after_sim': n_valid,
            'sim_survival_frac': survival,
            'w1_max': float(w1.max()),
            'w1_median': float(np.median(w1)),
            'w1_mean': float(w1.mean()),
            'best_val_loss': float(json.loads((delta_out / 'metrics.json').read_text()).get(
                'best_val_loss', float('nan'))),
        })
        pd.DataFrame(convergence_rows).to_csv(
            run_dir / 'convergence.csv', index=False,
        )
        print(f"  W1(round {r-1} vs round {r}): max={w1.max():.4f}  "
              f"median={np.median(w1):.4f}")

    # ------------------------------------------------------------
    # 4. Final summary
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SEQUENTIAL RUN COMPLETE")
    print("=" * 80)
    print(f"  run dir: {run_dir}")
    print(f"  convergence: {run_dir / 'convergence.csv'}")
    if convergence_rows:
        last = convergence_rows[-1]
        print(f"  final round {last['round']}: W1_max={last['w1_max']:.4f}  "
              f"best_val={last['best_val_loss']:.4f}")


if __name__ == '__main__':
    main()
