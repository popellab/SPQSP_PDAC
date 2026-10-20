"""MCMC fit of the PDAC vpop model using trained surrogate arms.

Loads:
  - Prior from parameters/vpop_prior.npz
  - Trained arms from results/vpop_pool/arms/
  - Corpus from results/vpop_pool/corpus.yaml + cohorts.yaml

Assembles a vpop Problem and runs NUTS (ungated) or WALNUTS (gated).

Usage
-----
    .venv/bin/python -I scripts/run_vpop_fit.py
    .venv/bin/python -I scripts/run_vpop_fit.py --chains 4 --warmup 500 --samples 500
    .venv/bin/python -I scripts/run_vpop_fit.py --gated   # use eligibility model
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vpop.corpus import GammaSpec, load_corpus
from vpop.model import (FunctionSurrogate, Measurement, Mechanism,
                        assemble_V, block_draw_plan, check_per_patient)
from vpop.model.basis import Basis, row_sensitivity, truncated_basis
from vpop.model.fit import Problem
from vpop.prior import read_prior
from vpop.surrogate.emulator import Arm, arm_p_ok, arm_species, load_arm
from vpop.emit import Labels

from vpop_readouts import (
    SCENARIOS, READOUTS, LOGIT_READOUT, N_SPECIES, readouts,
    LOG_R_0, SIGMA_R, N_AUX,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent


def load_arms(arms_dir: Path) -> dict[str, Arm]:
    """Load all trained arms from the arms directory."""
    arms = {}
    for scenario in SCENARIOS:
        path = arms_dir / scenario / "arm.npz"
        if not path.exists():
            path = arms_dir / f"{scenario}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"No arm found for scenario {scenario!r} at {arms_dir}")
        arms[scenario] = load_arm(path)
        log.info("Loaded arm %s: %d params, %d targets, status codes %s",
                 scenario, len(arms[scenario].param_names),
                 len(arms[scenario].target_names),
                 arms[scenario].status_codes)
    return arms


def build_surrogate(arms: dict[str, Arm], *, gated: bool
                    ) -> FunctionSurrogate:
    """Wrap trained arms into a FunctionSurrogate."""

    def species_fn(log_theta, scenario: int):
        arm = arms[SCENARIOS[scenario]]
        return arm_species(arm, log_theta)

    w_fn = None
    if gated:
        gate_arm = arms[SCENARIOS[0]]
        def w_fn(log_theta):
            return arm_p_ok(gate_arm, log_theta)

    return FunctionSurrogate(
        scenarios=SCENARIOS,
        readouts=READOUTS,
        n_species=N_SPECIES,
        g_fn=species_fn,
        h_fn=readouts,
        logit_readout=LOGIT_READOUT,
        w_fn=w_fn,
    )


def main():
    parser = argparse.ArgumentParser(description="MCMC fit of PDAC vpop model")
    parser.add_argument("--pool-dir", type=str,
                        default=str(REPO / "results" / "vpop_pool"))
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=400)
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--n-cloud", type=int, default=800)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sampler", type=str, default="nuts",
                        choices=["nuts", "walnuts"],
                        help="MCMC sampler (default: nuts)")
    parser.add_argument("--gated", action="store_true",
                        help="Use eligibility model")
    parser.add_argument("--k-mu", type=int, default=None,
                        help="Basis rank for mu (centre). None = auto from SVD")
    parser.add_argument("--k-omega", type=int, default=None,
                        help="Basis rank for omega (widths). None = no reduction")
    parser.add_argument("--sv-threshold", type=float, default=0.95,
                        help="Cumulative variance threshold for auto k selection")
    parser.add_argument("--metric-file", type=str, default=None,
                        help="Path to precomputed Laplace metric (.npy). "
                             "Computed and saved to out_dir/laplace_metric.npy "
                             "if not provided.")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    pool_dir = Path(args.pool_dir)
    out_dir = Path(args.out_dir) if args.out_dir else pool_dir / "fit"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load prior ───────────────────────────────────────────────────
    prior_path = REPO / "parameters" / "vpop_prior.npz"
    log.info("Loading prior from %s", prior_path)
    pi = read_prior(prior_path)
    log.info("Prior: %d params, asserted=%s", pi.n_params, pi.asserted)

    # ── Load trained arms ────────────────────────────────────────────
    arms_dir = pool_dir / "arms"
    log.info("Loading trained arms from %s", arms_dir)
    arms = load_arms(arms_dir)

    # ── Build surrogate ──────────────────────────────────────────────
    log.info("Building surrogate (gated=%s)", args.gated)
    surrogate = build_surrogate(arms, gated=args.gated)

    # ── Check readouts function ──────────────────────────────────────
    log.info("Checking per-patient readouts...")
    check_per_patient(readouts, READOUTS,
                      widths=[N_SPECIES] * len(SCENARIOS),
                      n_aux=N_AUX)
    log.info("  readouts function OK")

    # ── Load corpus ──────────────────────────────────────────────────
    corpus_path = pool_dir / "corpus.yaml"
    cohorts_path = pool_dir / "cohorts.yaml"
    log.info("Loading corpus from %s", corpus_path)
    corpus = load_corpus(corpus_path, cohorts=cohorts_path)
    log.info("Corpus: %d targets, %d cohorts, %d rows",
             len(corpus.targets), len(corpus.cohorts.cohorts), len(corpus.rows))

    # ── Build measurement ────────────────────────────────────────────
    gamma_spec = GammaSpec(
        active_fields=("assay_modality",),
        references={"assay_modality": "IHC"},
        weights={"assay_modality": 1.0},
    )
    meas = Measurement.of_corpus(corpus, surrogate, gamma_spec=gamma_spec)
    log.info("Measurement: %d targets, %d gamma columns",
             meas.n_targets, meas.X_gamma.shape[1])

    # ── Build mechanism ──────────────────────────────────────────────
    population = pi.cloud(args.n_cloud, seed=args.seed + 1)
    mech = Mechanism(
        surrogate=surrogate,
        population=population,
        measurement=meas,
        population_target="diagnosed",
    )

    # ── Block draw plans ─────────────────────────────────────────────
    plans = block_draw_plan(corpus.cohorts, corpus.row_cohorts)
    log.info("Block draw plans: %d blocks", len(plans))

    # ── Assemble problem ─────────────────────────────────────────────
    log.info("Assembling problem (n_cloud=%d, n_boot=1000)...", args.n_cloud)
    t0 = time.time()
    problem, v_boot, observed = Problem.assemble(
        mech, corpus.by_cohort(), plans,
        mu_0=pi.mu_0, omega_0=pi.omega_0,
        log_R_0=LOG_R_0,
        n_cloud=args.n_cloud, n_boot=1000, seed=args.seed + 2,
    )
    log.info("Problem assembled in %.1fs", time.time() - t0)

    # ── Floor v_boot diagonal ───────────────────────────────────────
    # Density readouts (cells/mm²) produce near-zero bootstrapped variance
    # at the lower quantiles because the prior-predictive cloud clusters
    # tightly. On log scale, sd < 0.01 means <1% relative spread — claiming
    # measurement precision no instrument achieves. Floor at sd=0.05 (~5%
    # relative on natural scale), applied to the diagonal only.
    VAR_FLOOR = 0.05**2  # sd = 0.05 in log/logit space
    n_floored = 0
    for V in v_boot:
        diag = np.diag(V)
        low = diag < VAR_FLOOR
        if low.any():
            n_floored += int(low.sum())
            np.fill_diagonal(V, np.maximum(diag, VAR_FLOOR))
    if n_floored:
        log.info("Floored %d v_boot diagonal entries (var < %.1e)", n_floored, VAR_FLOOR)

    # ── Build preliminary fit spec (full rank, for Jacobian) ────────
    spec_full = pi.for_fit(
        meas,
        tau_s=0.3,
        tau_u=0.3,
        tau_gamma=0.25,
        tau_beta=0.1,
        n_beta=0,
        log_R_0=LOG_R_0,
        sigma_R=SIGMA_R,
        pin_discrepancy=False,
        pin_aux=False,
    )

    # ── Basis reduction (centre + widths) ───────────────────────────
    from vpop.model.basis import truncation_blocks

    cov_full = assemble_V(plans, v_boot)
    V_chol_full = [np.asarray(L) for L in cov_full.chol]
    sizes = [L.shape[0] for L in V_chol_full]

    def _auto_k(cum, threshold, floor=5):
        k = int(np.searchsorted(cum, threshold) + 1)
        return max(k, floor)

    def _log_spectrum(label, sv, cum, k, P, threshold):
        log.info("%s: k=%d / %d params (%.1f%% variance at threshold %.2f)",
                 label, k, P, float(cum[min(k, len(cum)) - 1]) * 100, threshold)
        log.info("  Top 10 singular values: %s",
                 np.round(sv[:min(10, len(sv))], 4))
        log.info("  Cumulative at k=5,10,20,30,50: %s",
                 [f"{float(cum[min(i, len(cum))-1]):.3f}"
                  for i in [5, 10, 20, 30, 50] if i <= len(cum)])

    # --- Centre (mu) ---
    A_mu = row_sensitivity(spec_full, problem, site="mu_raw")
    _, sv_mu, cum_mu = truncated_basis(
        A_mu, V_chol_full, k=min(A_mu.shape), hold=())

    k_mu = args.k_mu if args.k_mu is not None else _auto_k(cum_mu, args.sv_threshold)
    k_mu = min(k_mu, min(A_mu.shape))
    _log_spectrum("mu basis", sv_mu, cum_mu, k_mu, pi.n_params, args.sv_threshold)

    B_mu, _, _ = truncated_basis(A_mu, V_chol_full, k=k_mu, hold=())
    mu_basis_obj = Basis(B=B_mu, A=A_mu, singular_values=sv_mu,
                         cumulative=cum_mu, names=tuple(pi.names),
                         held=(), site="mu_raw")

    # --- Widths (omega) ---
    A_u = row_sensitivity(spec_full, problem, site="u_raw")
    _, sv_u, cum_u = truncated_basis(
        A_u, V_chol_full, k=min(A_u.shape), hold=())

    k_omega = args.k_omega if args.k_omega is not None else _auto_k(cum_u, args.sv_threshold)
    k_omega = min(k_omega, min(A_u.shape))
    _log_spectrum("omega basis", sv_u, cum_u, k_omega, pi.n_params, args.sv_threshold)

    B_u, _, _ = truncated_basis(A_u, V_chol_full, k=k_omega, hold=())
    u_basis_obj = Basis(B=B_u, A=A_u, singular_values=sv_u,
                        cumulative=cum_u, names=tuple(pi.names),
                        held=(), site="u_raw")

    # --- Truncation covariance ---
    trunc_V = truncation_blocks(sizes, centre=mu_basis_obj, widths=u_basis_obj)

    cov = assemble_V(plans, v_boot, trunc=trunc_V)
    V_chol = [jnp.asarray(L) for L in cov.chol]
    V0_chol = ([jnp.asarray(L) for L in cov.chol_untruncated]
               if hasattr(cov, 'chol_untruncated') and cov.chol_untruncated is not None
               else None)
    log.info("V reassembled with truncation: %d blocks, %d rows (V0_chol=%s)",
             len(V_chol), sum(L.shape[0] for L in V_chol),
             "yes" if V0_chol is not None else "no")

    # ── Build reduced fit spec ───────────────────────────────────────
    spec = pi.for_fit(
        meas,
        tau_s=0.3,
        tau_u=0.3,
        tau_gamma=0.25,
        tau_beta=0.1,
        n_beta=0,
        log_R_0=LOG_R_0,
        sigma_R=SIGMA_R,
        pin_discrepancy=False,
        pin_aux=False,
        mu_basis=B_mu,
        u_basis=B_u,
    )

    # Report effective sampled dimensions
    from vpop.model.fit import site_spec
    sampled_dims = sum(z.shape[0] if z.ndim else 1 for _, z, _ in site_spec(spec))
    log.info("FitSpec: %d sampled dims (mu_c=%d + u_c=%d + aux), "
             "reduced from %d+%d+aux",
             sampled_dims, k_mu, k_omega, pi.n_params, pi.n_params)

    # ── Compute or load Laplace metric ─────────────────────────────
    from vpop.metric import laplace_inverse_mass

    metric_path = (Path(args.metric_file) if args.metric_file
                   else out_dir / "laplace_metric.npy")
    if metric_path.exists():
        log.info("Loading cached Laplace metric from %s", metric_path)
        inv_metric = np.load(metric_path)
        log.info("  shape=%s", inv_metric.shape)
    else:
        log.info("Computing Laplace inverse mass matrix...")
        t0_metric = time.time()
        inv_mass_dict = laplace_inverse_mass(spec, problem, V_chol)
        inv_metric = np.asarray(next(iter(inv_mass_dict.values())))
        elapsed_metric = time.time() - t0_metric
        log.info("Laplace metric computed in %.1fs (%.1f min), shape=%s",
                 elapsed_metric, elapsed_metric / 60, inv_metric.shape)
        np.save(metric_path, inv_metric)
        log.info("Saved metric to %s", metric_path)

    # ── Run sampler ──────────────────────────────────────────────────
    param_names = list(pi.names)
    labels = Labels(param=param_names)

    log.info("=" * 60)
    use_walnuts = args.sampler == "walnuts"
    if use_walnuts:
        log.info("Running WALNUTS: %d chains, %d warmup, %d samples",
                 args.chains, args.warmup, args.samples)
        from vpop.samplers import walnuts
        t0 = time.time()
        idata = walnuts(
            spec, problem, V_chol, observed,
            chains=args.chains, warmup=args.warmup, samples=args.samples,
            seed=args.seed, labels=labels, V0_chol=V0_chol,
            inv_metric=inv_metric,
        )
    else:
        log.info("Running NUTS: %d chains, %d warmup, %d samples",
                 args.chains, args.warmup, args.samples)
        from vpop.samplers import nuts
        t0 = time.time()
        idata = nuts(
            spec, problem, V_chol, observed,
            chains=args.chains, warmup=args.warmup, samples=args.samples,
            seed=args.seed, labels=labels, V0_chol=V0_chol,
            inv_metric=inv_metric,
        )
    elapsed = time.time() - t0
    log.info("Sampling completed in %.1fs (%.1f min)", elapsed, elapsed / 60)

    # ── Save results ─────────────────────────────────────────────────
    idata_path = out_dir / "idata.nc"
    idata.to_netcdf(str(idata_path))
    log.info("InferenceData saved to %s", idata_path)

    # ── Print summary ────────────────────────────────────────────────
    import arviz as az

    log.info("=" * 60)
    log.info("Posterior summary:")
    summary = az.summary(idata, var_names=["mu", "omega"], ci_prob=0.9)
    print(summary.to_string())

    if "alpha_gamma" in idata.posterior:
        log.info("\nGamma (measurement correction):")
        gamma_summary = az.summary(idata, var_names=["alpha_gamma"],
                                   ci_prob=0.9)
        print(gamma_summary.to_string())

    if "log_R" in idata.posterior:
        log.info("\nAuxiliary (log_R):")
        aux_summary = az.summary(idata, var_names=["log_R"], ci_prob=0.9)
        print(aux_summary.to_string())

    # Diagnostics
    log.info("\nDiagnostics:")
    try:
        rhat = az.rhat(idata, var_names=["mu", "omega", "s"])
        for var in ["mu", "omega", "s"]:
            if var in rhat:
                vals = rhat[var].values
                log.info("  %s R-hat: max=%.3f, mean=%.3f",
                         var, float(np.max(vals)), float(np.mean(vals)))
    except Exception as e:
        log.warning("  R-hat failed (need >=2 chains): %s", e)

    try:
        ess = az.ess(idata, var_names=["mu", "omega", "s"])
        for var in ["mu", "omega", "s"]:
            if var in ess:
                vals = ess[var].values
                log.info("  %s ESS: min=%.0f, median=%.0f",
                         var, float(np.min(vals)), float(np.median(vals)))
    except Exception as e:
        log.warning("  ESS failed: %s", e)

    log.info("=" * 60)
    log.info("Fit complete. Results at %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
