#!/usr/bin/env python3
"""Prototype: full joint NPE inference across all submodel targets.

No components, no JAX, no phases — just:
1. Load all targets
2. Sample from hierarchical prior (scipy)
3. Simulate all observables (scipy)
4. Train one NPE
5. Condition on observed data
6. SBC + PPC diagnostics
"""

import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
SUBMODEL_DIR = ROOT / "calibration_targets" / "submodel_targets"
PRIORS_CSV = ROOT / "parameters" / "pdac_priors.csv"


def main():
    from maple.core.calibration.parameter_groups import load_parameter_groups
    from maple.core.calibration.submodel_inference import (
        build_numpy_forward_fns,
        build_target_likelihoods,
        load_priors_from_csv,
    )
    from maple.core.calibration.submodel_target import SubmodelTarget

    # ── Load everything ──
    print("Loading targets...")
    csv_priors = load_priors_from_csv(PRIORS_CSV)
    groups_config = load_parameter_groups(SUBMODEL_DIR / "submodel_config.yaml")

    targets = []
    for yf in sorted(SUBMODEL_DIR.glob("*_deriv*.yaml")):
        try:
            data = yaml.safe_load(open(yf))
            if not data or "calibration" not in data:
                continue
            target = SubmodelTarget.model_validate(data)
            targets.append(target)
        except Exception as e:
            print(f"  Skip {yf.name}: {e}")

    n_ode = sum(1 for t in targets if t.calibration.forward_model.type == "custom_ode")
    n_alg = sum(1 for t in targets if t.calibration.forward_model.type == "algebraic")
    n_other = len(targets) - n_ode - n_alg
    print(f"  {len(targets)} targets loaded ({n_alg} algebraic, {n_ode} ODE, {n_other} other)")

    # Collect QSP params
    all_param_names = set()
    for t in targets:
        for p in t.calibration.parameters:
            if not p.nuisance:
                all_param_names.add(p.name)
    if groups_config:
        all_param_names |= groups_config.all_grouped_params
    qsp_params = sorted(all_param_names)
    comp_prior_specs = {k: v for k, v in csv_priors.items() if k in all_param_names}

    # Nuisance specs
    nuisance_specs = {}
    for t in targets:
        for p in t.calibration.parameters:
            if p.nuisance and p.prior and p.name not in nuisance_specs:
                nuisance_specs[p.name] = {"mu": p.prior.mu, "sigma": p.prior.sigma}

    print(f"  {len(qsp_params)} QSP params, {len(nuisance_specs)} nuisance params")

    # ── Build forward functions ──
    print("\nBuilding forward functions (numpy/scipy)...")
    t0 = time.perf_counter()

    # Bootstrap medians from maple pipeline
    target_likelihoods = build_target_likelihoods(targets, comp_prior_specs)

    forward_fns = []
    obs_names = []
    obs_medians = []
    for target, tl in zip(targets, target_likelihoods):
        fns = build_numpy_forward_fns(target)
        for fn, yaml_entry, likelihood_entry in zip(
            fns, target.calibration.error_model, tl.entries
        ):
            forward_fns.append(fn)
            obs_names.append(f"{target.target_id}__{yaml_entry.name}")
            obs_medians.append(float(likelihood_entry.fit.median))

    n_obs = len(forward_fns)
    t_build = time.perf_counter() - t0
    n_obs_ode = sum(1 for t in targets for _ in t.calibration.error_model if t.calibration.forward_model.type == "custom_ode")
    n_obs_alg = n_obs - n_obs_ode
    print(f"  {n_obs} observables built in {t_build:.1f}s ({n_obs_alg} algebraic, {n_obs_ode} ODE)")
    print(f"  Observed medians range: [{min(obs_medians):.4g}, {max(obs_medians):.4g}]")

    # ── Hierarchical prior sampling ──
    print("\nSetting up hierarchical prior...")
    group_sampling = []
    param_in_group = set()
    if groups_config:
        for group in groups_config.groups:
            member_indices = []
            delta_priors = []
            for member in group.members:
                if member.name in qsp_params:
                    idx = qsp_params.index(member.name)
                    member_indices.append(idx)
                    delta_priors.append(
                        (member.delta_prior.mu, member.delta_prior.sigma)
                        if member.delta_prior else None
                    )
                    param_in_group.add(member.name)
            if member_indices:
                bp = group.resolve_base_prior(comp_prior_specs)
                tau = group.between_member_sd.sigma
                group_sampling.append((group.group_id, bp, member_indices, tau, delta_priors))
        print(f"  {len(group_sampling)} hierarchical groups")

    # ── Simulate ──
    N = 10000
    rng = np.random.default_rng(42)

    t0 = time.perf_counter()
    print(f"\nStep 1: Sampling {N} prior draws ({len(qsp_params)} params)...")
    theta = np.zeros((N, len(qsp_params)))

    # Independent params
    for j, pname in enumerate(qsp_params):
        if pname in param_in_group:
            continue
        spec = comp_prior_specs.get(pname)
        if spec:
            theta[:, j] = rng.lognormal(spec.mu, spec.sigma, N)
        else:
            theta[:, j] = rng.lognormal(0, 1, N)

    # Grouped params
    for gid, bp, member_indices, tau, delta_priors in group_sampling:
        base = rng.lognormal(bp.mu, bp.sigma, N)
        for idx, dp in zip(member_indices, delta_priors):
            if dp is not None:
                delta = rng.normal(dp[0], dp[1], N)
            else:
                delta = rng.normal(0, tau, N)
            theta[:, idx] = base * np.exp(delta)

    t_prior = time.perf_counter() - t0
    print(f"  Prior sampling done in {t_prior:.1f}s")
    print(f"  Theta range: [{theta.min():.4g}, {theta.max():.4g}]")
    print(f"\nStep 2: Simulating {N} draws x {n_obs} observables...")
    x = np.full((N, n_obs), np.nan)

    try:
        from tqdm import trange
        sim_iter = trange(N, desc="Simulating", unit="sim")
    except ImportError:
        sim_iter = range(N)

    for i in sim_iter:
        param_dict = {pname: float(theta[i, j]) for j, pname in enumerate(qsp_params)}
        for nname, nspec in nuisance_specs.items():
            param_dict[nname] = float(rng.lognormal(nspec["mu"], nspec["sigma"]))
        for obs_idx, fn in enumerate(forward_fns):
            try:
                x[i, obs_idx] = float(fn(param_dict))
            except Exception:
                pass

    # Check per-observable validity
    print(f"\n  Per-observable validity:")
    good_obs = []
    for obs_idx in range(n_obs):
        col = x[:, obs_idx]
        n_ok = int(np.sum(np.isfinite(col) & (col > 0)))
        if n_ok > N * 0.5:
            good_obs.append(obs_idx)
        else:
            print(f"    SKIP {obs_names[obs_idx]:<55} {n_ok}/{N} ({100*n_ok/N:.0f}%)")

    print(f"  Usable: {len(good_obs)}/{n_obs} observables")

    x_good = x[:, good_obs]
    valid = np.all(np.isfinite(x_good) & (x_good > 0), axis=1)
    n_valid = int(np.sum(valid))
    theta_valid = theta[valid]
    x_valid = x_good[valid]
    t_sim = time.perf_counter() - t0
    print(f"\n  Simulation summary:")
    print(f"    Time: {t_sim:.1f}s ({t_sim/N*1000:.2f} ms/sim)")
    print(f"    Valid rows: {n_valid}/{N} ({100*n_valid/N:.0f}%)")
    print(f"    Usable observables: {len(good_obs)}/{n_obs}")

    if n_valid < 500:
        print("ERROR: Not enough valid simulations")
        return

    # ── Train/test split ──
    n_test = min(2000, n_valid // 10)
    n_train = n_valid - n_test
    idx = rng.permutation(n_valid)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    # ── Train NPE ──
    print(f"\nStep 3: Training NPE ({len(qsp_params)} params, {len(good_obs)} obs, {n_train} train, {n_test} test)...")

    import torch
    from sbi.inference import NPE

    log_theta_train = np.log(theta_valid[train_idx]).astype(np.float32)
    log_x_train = np.log(x_valid[train_idx]).astype(np.float32)
    log_theta_test = np.log(theta_valid[test_idx]).astype(np.float32)
    log_x_test = np.log(x_valid[test_idx]).astype(np.float32)

    # Standardize x (fit on train)
    x_mean = np.mean(log_x_train, axis=0)
    x_std = np.std(log_x_train, axis=0)
    x_std = np.where(x_std < 1e-6, 1.0, x_std)
    x_train_normed = (log_x_train - x_mean) / x_std
    x_test_normed = (log_x_test - x_mean) / x_std

    # Prior in log-space (unbounded Gaussian)
    prior_mus = []
    prior_sigmas = []
    for pname in qsp_params:
        spec = comp_prior_specs.get(pname)
        if spec:
            prior_mus.append(spec.mu)
            prior_sigmas.append(spec.sigma)
        else:
            prior_mus.append(0.0)
            prior_sigmas.append(1.0)
    prior_dist = torch.distributions.Independent(
        torch.distributions.Normal(
            torch.tensor(prior_mus, dtype=torch.float32),
            torch.tensor(prior_sigmas, dtype=torch.float32),
        ),
        1,
    )

    t0 = time.perf_counter()
    inference = NPE(prior=prior_dist)
    inference.append_simulations(
        torch.as_tensor(log_theta_train),
        torch.as_tensor(x_train_normed),
    )
    density_estimator = inference.train(
        training_batch_size=min(512, n_train // 4),
        show_train_summary=True,
    )
    posterior = inference.build_posterior(density_estimator)
    t_train = time.perf_counter() - t0
    print(f"  NPE trained in {t_train:.1f}s")

    # ── Condition on observed data ──
    print("\nStep 4: Sampling posterior...")
    obs_arr = np.array([obs_medians[j] for j in good_obs], dtype=np.float64)
    log_obs = np.log(obs_arr).astype(np.float32)
    obs_normed = (log_obs - x_mean) / x_std
    obs_tensor = torch.as_tensor(obs_normed)
    print(f"  x_obs (log, normed): min={obs_normed.min():.2f}, max={obs_normed.max():.2f}, mean={obs_normed.mean():.2f}")

    n_post = 4000
    t0 = time.perf_counter()
    log_post = posterior.sample((n_post,), x=obs_tensor).numpy()
    t_sample = time.perf_counter() - t0
    log_post = np.clip(log_post, -50, 50)
    post_samples = np.exp(log_post)
    print(f"  Sampled {n_post} posterior draws in {t_sample:.1f}s")

    # ── Results ──
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"{'Parameter':<30} {'Prior Med':>10} {'Post Med':>10} {'Post CV':>8} {'Contr':>8}")
    print("-" * 70)
    for j, pname in enumerate(qsp_params):
        spec = comp_prior_specs.get(pname)
        prior_med = np.exp(spec.mu) if spec else float("nan")
        prior_sig = spec.sigma if spec else 1.0
        post = post_samples[:, j]
        post_med = np.median(post)
        post_cv = np.std(post) / np.mean(post) if np.mean(post) > 0 else float("nan")
        log_post_j = np.log(post[post > 0]) if np.any(post > 0) else np.array([0])
        post_std_log = np.std(log_post_j)
        contraction = 1.0 - post_std_log / prior_sig if prior_sig > 0 else 0
        print(
            f"{pname:<30} {prior_med:>10.4g} {post_med:>10.4g} "
            f"{post_cv:>8.2f} {contraction:>7.0%}"
        )

    # ── SBC ──
    print(f"\n{'='*70}")
    print(f"SBC ({n_test} test simulations)")
    print(f"{'='*70}")

    n_sbc_posterior = 200
    n_sbc = min(n_test, 500)
    ranks = np.zeros((n_sbc, len(qsp_params)), dtype=int)

    try:
        from tqdm import trange
        sbc_iter = trange(n_sbc, desc="SBC", unit="test")
    except ImportError:
        sbc_iter = range(n_sbc)

    for k in sbc_iter:
        x_k = torch.as_tensor(x_test_normed[k:k+1])
        theta_true_k = log_theta_test[k]
        post_k = posterior.sample((n_sbc_posterior,), x=x_k).numpy()
        for j in range(len(qsp_params)):
            ranks[k, j] = int(np.sum(post_k[:, j] < theta_true_k[j]))

    from scipy.stats import kstest

    print(f"\n{'Parameter':<30} {'KS p-val':>10} {'Mean Rank':>10} {'Status':>8}")
    print("-" * 62)
    n_ok = 0
    for j, pname in enumerate(qsp_params):
        normalized_ranks = ranks[:, j] / n_sbc_posterior
        ks_stat, ks_p = kstest(normalized_ranks, "uniform")
        mean_rank = np.mean(ranks[:, j])
        status = "OK" if ks_p > 0.01 else "FAIL"
        if ks_p > 0.01:
            n_ok += 1
        print(f"{pname:<30} {ks_p:>10.4f} {mean_rank:>10.1f} {status:>8}")
    print(f"\nSBC: {n_ok}/{len(qsp_params)} well-calibrated (KS p > 0.01)")

    # ── PPC ──
    print(f"\n{'='*70}")
    print("POSTERIOR PREDICTIVE CHECK")
    print(f"{'='*70}")

    n_ppc = 200
    obs_names_good = [obs_names[j] for j in good_obs]

    try:
        from tqdm import trange
        ppc_iter = trange(n_ppc, desc="PPC", unit="sim")
    except ImportError:
        ppc_iter = range(n_ppc)

    ppc_x = np.full((n_ppc, len(good_obs)), np.nan)
    for i in ppc_iter:
        param_dict = {pname: float(post_samples[i, j]) for j, pname in enumerate(qsp_params)}
        for nname, nspec in nuisance_specs.items():
            param_dict[nname] = float(rng.lognormal(nspec["mu"], nspec["sigma"]))
        for k, obs_idx in enumerate(good_obs):
            try:
                ppc_x[i, k] = float(forward_fns[obs_idx](param_dict))
            except Exception:
                pass

    print(f"\n{'Observable':<55} {'Obs':>10} {'PPC Med':>10} {'PPC 95%':>20} {'Cover?':>7}")
    print("-" * 105)
    n_covered = 0
    for k in range(len(good_obs)):
        col = ppc_x[:, k]
        valid_col = col[np.isfinite(col)]
        if len(valid_col) < 10:
            continue
        ppc_med = np.median(valid_col)
        ppc_lo = np.percentile(valid_col, 2.5)
        ppc_hi = np.percentile(valid_col, 97.5)
        obs_val = obs_arr[k]
        covered = "YES" if ppc_lo <= obs_val <= ppc_hi else "no"
        if covered == "YES":
            n_covered += 1
        name = obs_names_good[k][-50:]
        print(
            f"{name:<55} {obs_val:>10.4g} {ppc_med:>10.4g} "
            f"[{ppc_lo:>8.4g}, {ppc_hi:>8.4g}] {covered:>7}"
        )
    print(f"\nPPC coverage: {n_covered}/{len(good_obs)} observables covered by 95% CI")


if __name__ == "__main__":
    main()