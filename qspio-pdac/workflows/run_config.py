"""SBI run configuration — the heavy-import-free half of sbi_runner.

Split out so importers that only need ``RunConfig`` (e.g. CI smoke tests
checking field defaults) don't have to pay for ``torch``, ``sbi``,
``matplotlib``, etc. ``workflows.sbi_runner`` re-exports the symbols
defined here for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


DEFAULT_SCENARIOS = {
    'baseline_no_treatment': {
        'calibration_targets': ['calibration_targets/baseline_no_treatment'],
        # scenario_yaml auto-derives to f'{cfg.scenarios_dir}/{scen_name}.yaml'
        # if omitted (which is the convention in pdac-build/scenarios/).
    },
    'clinical_progression': {
        # Literature dir + mechanistic-prior dir (qsp-hpc-tools >= a97c2c0
        # accepts a list of cal-target dirs and unions YAMLs by basename
        # with collision detection).
        'calibration_targets': [
            'calibration_targets/clinical_progression',
            'calibration_targets/mechanistic/clinical_progression',
        ],
    },
    'gvax_neoadjuvant_zheng2022': {
        'calibration_targets': [
            'calibration_targets/gvax_neoadjuvant',
            'calibration_targets/mechanistic/gvax_neoadjuvant',
        ],
    },
    'gvax_nivo_neoadjuvant_zheng2022': {
        'calibration_targets': [
            'calibration_targets/gvax_nivo_neoadjuvant',
            'calibration_targets/mechanistic/gvax_nivo_neoadjuvant',
        ],
    },
}

# Default paths into pdac-build's in-repo C++ tree (post-PR-#46). The local
# binary + template feed the binary-aware config_hash CppSimulator computes;
# the actual sims run on HPC via job_manager. Build the local binary with
# `make refresh-cpp` (writes cpp/sim/build/qsp_sim).
# Override on a per-environment basis via RunConfig.cpp_* fields or
# --config-json. The legacy --SPQSP_PDAC sibling-checkout layout is gone.
_PDAC_BUILD_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_TIER_CUTOFFS = {1: 30, 2: 70, 3: 117}


@dataclass
class RunConfig:
    """Configuration for a single SBI inference run.

    All fields default to the historical sbi_workflow.py values, so
    `run_inference(RunConfig())` reproduces the original script's behavior.

    Sweep axes (most common overrides):
        seed, prcc_tier, npe_hidden_features, npe_num_transforms,
        num_total_simulations, n_train_subsample.
    """
    # --- sweep axes ---
    seed: int = 2025
    prcc_tier: Optional[int] = None
    npe_hidden_features: int = 192
    npe_num_transforms: int = 9
    num_total_simulations: int = 5000
    n_train_subsample: Optional[int] = None  # None = use all valid train rows

    # --- seeds (don't change casually) ---
    # Each RNG source is independently seeded so reseed-stability audits and
    # noise/cohort ablations can vary one source at a time. Set distinct
    # defaults so different sources don't accidentally share an RNG stream.
    #   seed                  — CppSimulator (theta pool / aux draws) and
    #                            catch-all np.random global seed
    #   cache_sampling_seed   — sim-pool subsampling for training rows
    #   split_seed            — train/test split shuffling
    #   npe_seed              — NPE training init + posterior sampling +
    #                            PPC subsample
    #   noise_seed            — observation-noise injection
    #   cohort_seed           — cohort RE shift (uses cohort_seed for train,
    #                            cohort_seed+1 for test)
    cache_sampling_seed: int = 2025
    split_seed: int = 2027
    npe_seed: int = 2031
    noise_seed: int = 2029
    cohort_seed: int = 2033

    # --- NPE training ---
    npe_num_bins: int = 10
    npe_training_batch_size: int = 128
    npe_max_epochs: int = 500
    # If set, skip training and load a pretrained density_estimator.pt from
    # this directory. Expects density_estimator.pt + learning_curve.csv +
    # metrics.json (written by workflows/delta/train_npe_delta.py).
    pretrained_dir: Optional[Path] = None
    # If True, offload NPE posterior sampling (test-obs + observed-data) to
    # Delta GPU in a single batched round-trip via workflows/delta/
    # run_delta_sampling.sh. Requires pretrained_dir to be set.
    sample_on_delta: bool = False
    # If True, run step 5 + 5b (test-set posterior sampling, R² recovery, SBC
    # ECDF, local calibration). Produces recovery.png, sbc_ecdf_*.png,
    # local_calibration.csv, posterior_summary.csv — used by parameter_audit /
    # promote_sbi_stage2 selectors but NOT by misspec_audit. Default False
    # (skipped) for fast misspec-focused iteration. Incompatible with
    # sample_on_delta=True (delta path bundles steps 5 + 6).
    eval_npe_theta_space: bool = False
    # If True, cache the post–step-2 processed tensors (theta/x train+test,
    # copula-transformed observed, etc.) to cache/sbi_processed/<key>.pkl
    # so subsequent runs with the same model_version / scenarios / tier /
    # subsample / seeds skip the ~3-4 min sim rehydration. Incompatible
    # with misspec_detection_mode and run_posterior_predictive (those
    # paths need scenario_meta['simulator'] which isn't cached).
    cache_processed_tensors: bool = False
    # If True, stop after step 2 + artifact writing; skip training, diagnostics,
    # posterior sampling, etc. Intended for the sweep driver to stage
    # artifacts for Delta GPU training. Exits with a small metrics dict.
    stage_artifacts_only: bool = False

    # --- null-shuffle diagnostic ---
    # When True, permute x_train rows relative to theta_train just before
    # NPE.append_simulations(). Destroys all true (theta, x) coupling, so the
    # trained NPE has zero likelihood signal — any contraction or shift in the
    # resulting local_calibration.csv is a per-param artifact floor for this
    # architecture / N / hyperparameter set. Pair with a real run on the same
    # pool to subtract the floor: real_contraction - null_contraction. See
    # 2026-05-01 GMCSF phantom-contraction investigation. CLI: --null-shuffle.
    null_shuffle: bool = False
    null_shuffle_seed: int = 9999

    # --- model + priors ---
    model_version: str = 'multi_scenario_v5_qual_migration'  # bumped 2026-04-27 after qual→cal migration + new mechanistic targets — fresh pool desired.
    model_description: str = 'PDAC multi-scenario SBI: joint inference across N scenarios'
    model_script: str = 'immune_oncology_model_PDAC'  # legacy MATLAB tag; kept for run metadata only
    priors_csv: str = 'parameters/pdac_priors.csv'
    model_structure_file: str = 'model_structure.json'
    submodel_priors_yaml: str = 'notes/calibration/submodel_priors.yaml'
    # If True, build the prior from priors_csv only (independent normals per
    # parameter, no copula correlations from submodel_priors.yaml). The
    # simulator pool key drops submodel_priors_yaml in this mode, so the pool
    # cache is keyed solely on the CSV. Useful when iterating on stage-2
    # wiring or running smoke tests against deliberately wide priors that
    # shouldn't be tightened by the static composite. CLI: --skip-submodel-sync.
    # Default flipped to False on 2026-05-04 — composite prior is the
    # standard debugging mode; CSV-only is opt-in.
    skip_submodel_sync: bool = False
    # When True, abort the run if `submodel_priors.yaml` has stale
    # per-component fingerprints (or no fingerprint header at all).
    # Default False: warn loudly and continue, since freshness drift is
    # not always intentional but also shouldn't block exploratory runs.
    # CLI: --strict-submodel-freshness. Has no effect when
    # `skip_submodel_sync=True` since the composite prior is unused then.
    strict_submodel_freshness: bool = False

    # --- scenarios ---
    scenarios: dict = field(default_factory=lambda: dict(DEFAULT_SCENARIOS))
    # Where per-scenario YAMLs live (used to auto-derive the scenario_yaml
    # path when a scenario dict in `scenarios` doesn't set one explicitly).
    scenarios_dir: str = 'scenarios'

    # --- auxiliary parameters (measurement-bridging, jointly inferred) ---
    # Path to auxiliary_config.yaml declaring hierarchical groups for
    # measurement-bridging parameters (serum:tumor ratios, cross-species
    # translation factors, IHC-to-concentration conversions). Members are
    # auto-discovered by walking each scenario's calibration_target dirs at
    # workflow start. Empty / missing config + no member declarations is a
    # no-op (preserves pre-aux runner behavior).
    auxiliary_config_path: str = 'auxiliary_config.yaml'

    # --- cross-scenario calibration targets (Phase D-2 / Option D3+) ---
    # List of directories holding CrossScenarioCalibrationTarget YAMLs
    # (sibling schema in maple). Empty by default — when set, the runner
    # loads + composes after per-scenario alignment, surfacing one virtual
    # "_cross_scenario" block that flows through joint NaN filter, noise
    # injection, copula transform, and concatenation alongside real
    # scenarios. Currently only `input_kind=test_statistic` (scalar
    # composition over per-scenario test_stats matrices) is wired
    # end-to-end — `input_kind=timeseries` requires HPC sim_df plumbing
    # not yet in run_hpc(), and any cross-scenario YAML that references
    # a timeseries input will raise at compose time.
    cross_scenario_targets: list = field(default_factory=list)

    # --- C++ backend (qsp_sim) ---
    # qsp_sim binary + template XML on the local filesystem. Used for the
    # binary-aware config_hash CppSimulator computes; the actual sims run
    # on HPC via job_manager. Build with `make refresh-cpp` (writes
    # cpp/sim/build/qsp_sim from the in-repo cpp/qsp/ode/ codegen + the
    # in-repo resources/cpp/param_all.xml template).
    cpp_binary_path: str = str(_PDAC_BUILD_ROOT / 'cpp/sim/build/qsp_sim')
    cpp_template_xml: str = str(_PDAC_BUILD_ROOT / 'resources/cpp/param_all.xml')
    cpp_subtree: str = 'QSP'
    # Resource YAMLs that are constant across scenarios. Live alongside the
    # in-repo C++ template under resources/cpp/.
    cpp_drug_metadata_yaml: str = str(_PDAC_BUILD_ROOT / 'resources/cpp/drug_metadata.yaml')
    cpp_healthy_state_yaml: str = str(_PDAC_BUILD_ROOT / 'resources/cpp/healthy_state.yaml')
    # qsp_sim reads stop_time from the scenario YAML when present and
    # overrides --t-end-days, so this is a fallback only. dt_days controls
    # the output sampling step (not the integrator step).
    cpp_t_end_days: float = 1.0
    cpp_dt_days: float = 0.5

    # --- Classifier-based prior restriction ---
    # Path to a RestrictionClassifier serialization dir (classifier.pkl +
    # metadata.json). When set, the theta pool is rejection-sampled against
    # the classifier so only thetas with p(valid) >= restriction_threshold
    # are drawn. Produced by workflows/restrict_prior_train.py. Yields a
    # different pool cache key → does NOT collide with unrestricted pools.
    restriction_classifier_dir: Optional[str] = 'tests/fixtures/restriction_classifier'
    restriction_threshold: float = 0.7
    # Fills for classifier features missing from the live priors CSV (e.g. a
    # parameter the classifier was trained on but has since been retired).
    # Threaded into theta_pool via accept_named so a drifted prior can still
    # use the existing classifier. ``IFNg_50: 1000.0`` covers the 2026-04
    # retirement of the IFNg_50 Hill EC50. ``phi_col_ref: 0.12`` and
    # ``rho_collagen: 0.025`` cover the 2026-04 move of those structural
    # anchors out of the priors (issue #93) — values match
    # parameters_PDAC.m defaults.
    classifier_feature_fills: dict = field(default_factory=lambda: {
        "IFNg_50": 1000.0,
        "phi_col_ref": 0.12,
        "rho_collagen": 0.025,
        # alpha_Treg_cap → rho_Treg_apCAF_extra → retired entirely
        # (2026-05-01): the apCAF→Treg-cap pathway was removed after a
        # counterfactual SBI showed the data prefers it off. The
        # restriction classifier was trained at α=2 (cap-pathway
        # active), so it expects this feature in its input vector.
        # Fill at the legacy α=2.0 to keep the classifier usable
        # without retraining; the value is structurally inert now
        # (the parameter doesn't exist in the model) and only
        # threads through the classifier's prior-rejection step.
        "alpha_Treg_cap": 2.0,
        # apCAF_50_Tcap retired alongside the cap pathway in #128.
        # Same logic — classifier was trained with it as a feature;
        # fill at legacy default (5e6) keeps the classifier usable
        # without retraining.
        "apCAF_50_Tcap": 5e6,
        # APC0_cDC{1,2}_T renamed → cDC{1,2}_recruit_target_T on
        # 2026-05-02 (semantic rename: parameter is a recruitment-target
        # density, not a realized density — see notes/model/
        # cdc1_homeostasis_vs_recruitment.md). Classifier was trained
        # against the old names; fill at legacy default values keeps
        # it usable until next regen.
        "APC0_cDC1_T": 5e4,
        "APC0_cDC2_T": 2e5,
    })

    # --- PRCC tiers ---
    prcc_ranking_csv: str = 'results/prcc_sensitivity/aggregate_parameter_ranking.csv'
    tier_cutoffs: dict = field(default_factory=lambda: dict(DEFAULT_TIER_CUTOFFS))

    # --- execution ---
    max_tasks: int = 35
    # Default flipped to True on 2026-05-04 — section 7 (PPC + misspec
    # verdict) is the canonical debugging output; CLI opt-out is
    # --no-posterior-predictive.
    run_posterior_predictive: bool = True
    # When True, the PPC step also dumps per-sim burn-in trajectories
    # under ``SAVE_DIR/evolve_trajectories/<scenario>/`` and renders one
    # ``cal_target_trajectories_<scenario>.png`` per scenario showing
    # each observable's posterior path from healthy → diagnosis with
    # the observed value (95% CI) overlaid at t=0. Costs an extra full
    # PPC round of fresh sims (the suffix-pool cache is bypassed for
    # trajectory-enabled calls) — opt-in.
    save_evolve_trajectories: bool = True
    # Sampling interval (model-time days) for the burn-in trajectory
    # dump. ``None`` lets the C++ side use the evolve spec's step_days
    # (1 day for PDAC, ~7300 rows per sim — chunky). 14 is a reasonable
    # biweekly default.
    evolve_trajectory_dt_days: float | None = 14.0
    misspec_detection_mode: bool = False
    llm_elicited_targets_csv: str = 'notes/inference/llm_elicitation/llm_elicited_consensus.csv'

    # --- cohort random-effect (workflows/cohort_random_effect.py) ---
    # Per-cohort additive Gaussian shift in copula-Gaussian space, applied
    # at training-data assembly time. Each calibration target carries a
    # primary_data_source.source_tag which becomes its cohort_id; observables
    # in the same cohort share δ_c per training row, with δ_c ~ N(0, τ).
    # Absorbs inter-cohort heterogeneity (e.g., Liudahl vs Golesworthy
    # systematic offsets in PDAC TIL/Treg measurements) as a nuisance, so
    # the QSP θ posterior doesn't have to compromise across cohorts.
    # Default 0.5 (per project_cohort_random_effect.md): the standard
    # inference setting. Use --tau-cohort 0.0 for diagnostic mode (surface
    # cohort spread as misspec) and ≥0.7 to crank latitude further.
    # Empirical 2026-05-05: at τ=0.5 the misfit driver list strips out
    # cohort-spread red herrings (apc_maturation_ratio, stromal_fraction,
    # cdc2_density_baseline) and surfaces the genuine structural tensions
    # (cd8_density_baseline + golesworthy, mycaf/apcaf/icaf composition).
    tau_cohort: float = 0.5

    # --- diagnostics ---
    n_plot_params: int = 20
    n_posterior_sims: int = 500

    # --- output ---
    save_dir: Optional[Path] = None  # auto-derived from scenarios + tier if None

    # --- HPC branch overrides (laptop-side, applied after HPCJobManager
    # loads credentials.yaml; None = use whatever credentials set) ---
    # cpp_branch pins the SPQSP_PDAC branch qsp_sim is built from on HPC
    # (cred default: cpp-sweep-binary-io). tools_branch pins the
    # qsp-hpc-tools pip install ref (cred default whatever the
    # @<ref> suffix on package.qsp_hpc_tools_source is). codegen_branch
    # pins the qsp-codegen pip install ref (cred default whatever the
    # @<ref> suffix on cpp.codegen_source is) — the runner pip-upgrades
    # qsp-codegen into the HPC venv on every cpp build via
    # ensure_cpp_binary, so this lets you point at an unmerged
    # qsp-codegen feature branch without rebuilding credentials.yaml.
    # Useful when the feature branches aren't yet merged to credentials
    # defaults.
    cpp_branch: Optional[str] = None
    tools_branch: Optional[str] = None
    codegen_branch: Optional[str] = None
    # Sims per SLURM array task. None = credentials default (currently
    # 20). For N=10k runs, 100 is about right: 100 array tasks × ~100s
    # compute/task works well within Rockfish shared's ~48-concurrent
    # cap while keeping per-task overhead (venv activate, upload, etc)
    # amortized. Too small (<20) spends more time on overhead than work;
    # too large (>500) underfills the partition.
    cpp_jobs_per_chunk: Optional[int] = None
