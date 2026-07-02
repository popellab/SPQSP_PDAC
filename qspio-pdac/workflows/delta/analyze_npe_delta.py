#!/usr/bin/env python
"""Full NPE evaluation + posterior analysis on Delta GPU.

Replicates sbi_runner.py steps 5 (global + local calibration) and 6 (posterior
from observed data) in a self-contained form so only summaries + plots need to
be downloaded locally.

Reads bundled inputs from --input-dir:
    density_estimator.pt
    theta_test.pt, x_test.pt, obs_transformed_tensor.pt
    obs_transformed.npy
    param_names.json, pdac_priors.csv, prior_var.npy
    config.json   (seed, n_plot_params, n_test_samples=200, n_post_samples=2000,
                   local_fraction=0.2, chunk_size=256)

Writes to --output-dir:
    recovery.png, sbc_ecdf_{global,local}.png, posterior_vs_prior.png
    local_calibration.csv, posterior_samples.csv, posterior_summary.csv
    run_metrics.json
"""
import argparse
import gc
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from qsp_inference.inference import (
    compute_per_param_calibration,
    convert_posterior_samples_to_original_space,
    plot_posterior_vs_prior_marginals,
    sbi_recovery,
    sbi_z_score_contraction,
    sbi_calibration_ecdf,
)


def _sample_chunks(estimator, x, n_samples, chunk_size, device, label=""):
    """Sample n_samples thetas per row of x, chunking conditions to bound VRAM."""
    if x.ndim == 1:
        x = x.unsqueeze(0)
    n_cond = x.shape[0]
    x = x.to(device)
    chunks = []
    n_chunks = -(-n_cond // chunk_size)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n_cond, chunk_size):
            xb = x[i:i + chunk_size]
            sb = estimator.sample(torch.Size([n_samples]), condition=xb)
            chunks.append(sb.cpu())
            if device == "cuda":
                torch.cuda.synchronize()
            print(
                f"[analyze] {label}: chunk {i // chunk_size + 1}/{n_chunks} "
                f"({xb.shape[0]} conds) @ {time.time() - t0:.1f}s",
                flush=True,
            )
    samples = torch.cat(chunks, dim=1)
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    in_dir = Path(args.input_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((in_dir / "config.json").read_text())
    seed = int(cfg.get("seed", 2025))
    n_plot_params = int(cfg.get("n_plot_params", 12))
    n_test_samples = int(cfg.get("n_test_samples", 200))
    n_post_samples = int(cfg.get("n_post_samples", 2000))
    local_fraction = float(cfg.get("local_fraction", 0.2))
    chunk_size = int(cfg.get("chunk_size", 256))

    torch.manual_seed(seed)

    # --- Load inputs ---
    param_names = json.loads((in_dir / "param_names.json").read_text())
    theta_test = torch.load(in_dir / "theta_test.pt", weights_only=False).float()
    x_test = torch.load(in_dir / "x_test.pt", weights_only=False).float()
    obs_tensor = torch.load(in_dir / "obs_transformed_tensor.pt", weights_only=False).float()
    obs_np = np.load(in_dir / "obs_transformed.npy")
    prior_var = np.load(in_dir / "prior_var.npy")
    priors_csv = str(in_dir / "pdac_priors.csv")

    num_params = len(param_names)
    print(f"[analyze] device={device}  theta_test={tuple(theta_test.shape)}  "
          f"x_test={tuple(x_test.shape)}  obs={tuple(obs_tensor.shape)}", flush=True)
    if device == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"[analyze] gpu={p.name}  vram={p.total_memory/1e9:.1f}GB", flush=True)

    estimator = torch.load(in_dir / "density_estimator.pt", weights_only=False).to(device)
    estimator.eval()
    print(f"[analyze] estimator loaded ({sum(p.numel() for p in estimator.parameters()):,} NN params)",
          flush=True)

    # ===== Step 5: EVALUATE NPE (θ-space) =====
    print("[analyze] sampling test posteriors "
          f"({n_test_samples} samples x {x_test.shape[0]} conds)", flush=True)
    samples = _sample_chunks(estimator, x_test, n_test_samples, chunk_size, device, "test")

    # Global R² recovery
    _, _, r2_values = sbi_recovery(samples, theta_test, param_names, plot=False)
    r2_array = np.array([r2_values[n] for n in param_names])
    r2_ranking = np.argsort(r2_array)[::-1]
    plot_indices = r2_ranking[:n_plot_params].tolist()

    print(f"[analyze] R² — top 5:", flush=True)
    for i in r2_ranking[:5]:
        print(f"  {param_names[i]}: R² = {r2_array[i]:.3f}", flush=True)

    fig_rec, _, _ = sbi_recovery(samples, theta_test, param_names, param_indices=plot_indices)
    fig_rec.savefig(out / "recovery.png", dpi=150, bbox_inches="tight")
    plt.close(fig_rec)

    # Z-score contraction (global)
    _, _, z_scores, contractions = sbi_z_score_contraction(
        samples, theta_test, None, param_names, plot=False, prior_var=prior_var
    )
    median_contraction = float(np.median(contractions))

    # SBC (global)
    _, _, ranks_global, ks_global = sbi_calibration_ecdf(
        samples, theta_test, param_names, plot=False,
    )
    fig_ecdf, _, _, _ = sbi_calibration_ecdf(
        samples, theta_test, param_names, param_indices=plot_indices, difference=True,
    )
    fig_ecdf.suptitle("SBC calibration ECDF (global)", fontsize=14)
    fig_ecdf.savefig(out / "sbc_ecdf_global.png", dpi=150, bbox_inches="tight")
    plt.close(fig_ecdf)
    ks_pvals_global = np.array([ks_global[n]["ks_p_value"] for n in param_names])
    n_miscal_global = int((ks_pvals_global < 0.05).sum())
    print(f"[analyze] SBC (global): {n_miscal_global}/{num_params} miscalibrated", flush=True)

    # ===== Step 5b: LOCAL CALIBRATION =====
    x_test_np = x_test.numpy()
    obs_ref = obs_np[np.newaxis, :]
    dists = np.linalg.norm(x_test_np - obs_ref, axis=1)
    n_local = max(10, int(local_fraction * len(dists)))
    local_idx = np.argsort(dists)[:n_local]
    print(f"[analyze] local: {n_local}/{len(dists)} test obs "
          f"(dist ≤ {dists[local_idx[-1]]:.2f})", flush=True)

    samples_local = samples[:, local_idx]
    theta_test_local = theta_test[local_idx]

    _, _, r2_local = sbi_recovery(samples_local, theta_test_local, param_names, plot=False)
    r2_local_array = np.array([r2_local[n] for n in param_names])

    _, _, z_scores_local, contractions_local = sbi_z_score_contraction(
        samples_local, theta_test_local, None, param_names, plot=False, prior_var=prior_var
    )
    median_contraction_local = float(np.median(contractions_local))

    _, _, ranks_local, ks_local = sbi_calibration_ecdf(
        samples_local, theta_test_local, param_names, plot=False,
    )
    fig_ecdf_l, _, _, _ = sbi_calibration_ecdf(
        samples_local, theta_test_local, param_names, param_indices=plot_indices, difference=True,
    )
    fig_ecdf_l.suptitle("SBC calibration ECDF (local — near observed data)", fontsize=14)
    fig_ecdf_l.savefig(out / "sbc_ecdf_local.png", dpi=150, bbox_inches="tight")
    plt.close(fig_ecdf_l)
    ks_pvals_local = np.array([ks_local[n]["ks_p_value"] for n in param_names])
    n_miscal_local = int((ks_pvals_local < 0.05).sum())

    # Per-param metrics beyond r2/ks/contraction (local subset).
    # samples_local is log-space (theta_train was log-transformed upstream).
    pp = compute_per_param_calibration(samples_local, theta_test_local, z_scores=z_scores_local)

    local_cal_df = pd.DataFrame({
        "parameter": param_names,
        "r2_global": r2_array,
        "r2_local": r2_local_array,
        "r2_diff": r2_local_array - r2_array,
        "contraction_global": np.median(contractions, axis=0),
        "contraction_local": np.median(contractions_local, axis=0),
        "ks_pval_global": ks_pvals_global,
        "ks_pval_local": ks_pvals_local,
        "zscore_mean_local": pp["zscore_mean"],
        "zscore_std_local": pp["zscore_std"],
        "coverage95_local": pp["coverage95"],
        "post_sd_log_local": pp["post_sd"],
    }).sort_values("r2_diff")
    local_cal_df.to_csv(out / "local_calibration.csv", index=False)

    # Free the big test-sample tensor before step 6
    del samples, samples_local
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # ===== Step 6: POSTERIOR FROM OBSERVED DATA =====
    print(f"[analyze] sampling {n_post_samples} posterior draws from observed", flush=True)
    torch.manual_seed(seed)  # deterministic posterior draws
    post_samples = _sample_chunks(
        estimator, obs_tensor, n_post_samples, chunk_size, device, "obs"
    ).squeeze(1)

    post_samples_dict = convert_posterior_samples_to_original_space(post_samples, param_names)
    post_df = pd.DataFrame(post_samples_dict)
    post_df.to_csv(out / "posterior_samples.csv", index=False)
    post_summary = post_df.describe(percentiles=[0.05, 0.5, 0.95]).T
    post_summary.to_csv(out / "posterior_summary.csv")

    fig_vp, _ = plot_posterior_vs_prior_marginals(
        post_samples_dict,
        priors_csv=priors_csv,
        param_names=param_names,
        n_prior_samples=5000,
        log_transform=True,
        figsize=(20, 16),
        param_indices=plot_indices,
        show=False,
    )
    fig_vp.savefig(out / "posterior_vs_prior.png", dpi=150, bbox_inches="tight")
    plt.close(fig_vp)

    # ===== Run metrics =====
    metrics = {
        "r2_global_median": float(np.median(r2_array)),
        "r2_local_median": float(np.median(r2_local_array)),
        "contraction_global_median": median_contraction,
        "contraction_local_median": median_contraction_local,
        "n_miscal_global": n_miscal_global,
        "n_miscal_local": n_miscal_local,
        "n_params": int(num_params),
        "n_test": int(x_test.shape[0]),
        "n_test_samples": n_test_samples,
        "n_post_samples": n_post_samples,
        "plot_indices": plot_indices,
        "r2_per_param": {param_names[i]: float(r2_array[i]) for i in range(num_params)},
        "r2_ranking_top5": [param_names[i] for i in r2_ranking[:5]],
        "r2_ranking_bottom5": [param_names[i] for i in r2_ranking[-5:]],
    }
    (out / "run_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[analyze] done → {out}", flush=True)


if __name__ == "__main__":
    main()
