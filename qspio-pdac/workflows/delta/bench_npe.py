#!/usr/bin/env python
"""NPE training micro-benchmark.

Isolates where time goes in NSF training by measuring:
  1. pure forward pass  (no autograd)
  2. forward + backward  (training step)
  3. full sbi train-loop iteration  (includes sbi bookkeeping)
at multiple batch sizes on the specified device.

Usage: bench_npe.py --theta theta.pt --x x.pt --device cuda|cpu \
         [--batch-sizes 128,512,2048] [--iters 50]
"""
import argparse
import json
import time
from pathlib import Path

import torch
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn


def build_density_estimator(theta, x, cfg):
    de_builder = posterior_nn(
        model="nsf",
        hidden_features=int(cfg["hidden_features"]),
        num_transforms=int(cfg["num_transforms"]),
        num_bins=int(cfg["num_bins"]),
    )
    return de_builder(theta[:10], x[:10])


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def bench_forward(model, theta, x, batch_size, n_iters, device):
    model.eval()
    idx = torch.randperm(theta.shape[0])[:batch_size]
    t = theta[idx].to(device)
    X = x[idx].to(device)
    _sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iters):
            _ = model.loss(t, X)
    _sync(device)
    return (time.perf_counter() - t0) / n_iters


def bench_train_step(model, theta, x, batch_size, n_iters, device):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    idx = torch.randperm(theta.shape[0])[:batch_size]
    t = theta[idx].to(device)
    X = x[idx].to(device)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        opt.zero_grad(set_to_none=True)
        loss = model.loss(t, X).mean()
        loss.backward()
        opt.step()
    _sync(device)
    return (time.perf_counter() - t0) / n_iters


def bench_sbi_epoch(theta, x, cfg, batch_size, device, n_epochs=2):
    """Run sbi's NPE.train for a few epochs with a stop condition, measure avg time."""
    prior = torch.distributions.MultivariateNormal(
        theta.mean(0).to(device), torch.diag(theta.var(0).to(device) + 1e-6)
    )
    de = posterior_nn(
        model="nsf",
        hidden_features=int(cfg["hidden_features"]),
        num_transforms=int(cfg["num_transforms"]),
        num_bins=int(cfg["num_bins"]),
    )
    inference = NPE(prior=prior, density_estimator=de, device=device)
    inference = inference.append_simulations(theta, x)
    t0 = time.perf_counter()
    inference.train(
        training_batch_size=batch_size,
        max_num_epochs=n_epochs,
        show_train_summary=False,
        stop_after_epochs=10_000,  # disable early stop so we hit max
    )
    total = time.perf_counter() - t0
    n_completed = len(inference._summary["training_loss"])
    return total / max(n_completed, 1), n_completed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", required=True)
    ap.add_argument("--x", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-sizes", default="128,512,2048")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--sbi-epochs", type=int, default=2)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = json.loads(Path(args.config).read_text())
    theta = torch.load(args.theta, weights_only=False).float()
    x = torch.load(args.x, weights_only=False).float()
    n_total = theta.shape[0]

    print(f"[bench] device={device}  theta={tuple(theta.shape)}  x={tuple(x.shape)}", flush=True)
    if device == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"[bench] gpu={p.name}  vram={p.total_memory/1e9:.1f}GB  sm={p.major}.{p.minor}", flush=True)
    nth = torch.get_num_threads()
    print(f"[bench] torch.get_num_threads()={nth}", flush=True)

    # build one model for the forward/backward micro-benches
    model = build_density_estimator(theta, x, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[bench] model params: {n_params:,}", flush=True)

    results = {"device": device, "n_train": n_total, "n_params_model": n_params, "bench": []}
    for bs_s in args.batch_sizes.split(","):
        bs = int(bs_s)
        # warmup
        _ = bench_forward(model, theta, x, bs, 5, device)
        _ = bench_train_step(model, theta, x, bs, 5, device)

        fwd = bench_forward(model, theta, x, bs, args.iters, device)
        tr = bench_train_step(model, theta, x, bs, args.iters, device)
        ep_time, ep_n = bench_sbi_epoch(theta, x, cfg, bs, device, n_epochs=args.sbi_epochs)
        batches_per_epoch = n_total * 0.9 // bs  # sbi uses 10% val split by default

        print(
            f"[bs={bs:>5}]  fwd={fwd*1000:.2f}ms  train={tr*1000:.2f}ms  "
            f"sbi_epoch={ep_time:.2f}s  n_batches_full={int(batches_per_epoch)}  "
            f"train*nbatch={tr*batches_per_epoch:.2f}s",
            flush=True,
        )
        results["bench"].append({
            "batch_size": bs,
            "fwd_ms": fwd * 1000,
            "train_step_ms": tr * 1000,
            "sbi_epoch_s": ep_time,
            "batches_per_epoch_full": int(batches_per_epoch),
            "theoretical_epoch_s": tr * batches_per_epoch,
        })

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
