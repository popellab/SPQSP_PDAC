#!/usr/bin/env python
"""Standalone NPE trainer for Delta GPU.

Loads pre-transformed (theta, x) tensors + NPE config, trains the neural spline
flow density estimator, saves the trained estimator + learning curve + metrics.
A throwaway MVN prior is used (sbi API requires one); the real CopulaPrior is
reattached locally for posterior sampling / diagnostics.
"""
import argparse
import json
import threading
import time
from pathlib import Path

import torch
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", required=True)
    ap.add_argument("--x", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    theta = torch.load(args.theta, weights_only=False)
    x = torch.load(args.x, weights_only=False)
    max_epochs = int(cfg.get("max_epochs", 500))
    print(f"[train] device={device}  theta={tuple(theta.shape)}  x={tuple(x.shape)}", flush=True)
    if device == "cuda":
        print(f"[train] gpu: {torch.cuda.get_device_name(0)}  vram={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB", flush=True)

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

    stop = threading.Event()

    def _poll():
        t0 = time.time()
        last = 0
        while not stop.wait(5.0):
            tl = inference._summary.get("training_loss", [])
            vl = inference._summary.get("validation_loss", [])
            n = len(tl)
            if n > last:
                last = n
                print(
                    f"[train] epoch {n}/{max_epochs}  train={tl[-1]:.4f}  val={vl[-1]:.4f}  elapsed={time.time()-t0:.1f}s",
                    flush=True,
                )

    t = threading.Thread(target=_poll, daemon=True)
    t.start()
    t_start = time.time()
    try:
        estimator = inference.train(
            training_batch_size=int(cfg["batch_size"]),
            max_num_epochs=max_epochs,
            show_train_summary=True,
        )
    finally:
        stop.set()
    wall = time.time() - t_start

    estimator = estimator.to("cpu")
    torch.save(estimator, out / "density_estimator.pt")

    tl = inference._summary["training_loss"]
    vl = inference._summary["validation_loss"]
    with open(out / "learning_curve.csv", "w") as f:
        f.write("epoch,train_loss,val_loss\n")
        for i, (tr, va) in enumerate(zip(tl, vl)):
            f.write(f"{i+1},{tr},{va}\n")

    metrics = {
        "n_epochs": len(tl),
        "best_val_loss": float(inference._summary["best_validation_loss"][-1]),
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "n_params": int(theta.shape[1]),
        "n_obs": int(x.shape[1]),
        "n_train": int(theta.shape[0]),
        "wall_seconds": round(wall, 1),
        "sec_per_epoch": round(wall / max(len(tl), 1), 2),
        "max_epochs": max_epochs,
        "hidden_features": int(cfg["hidden_features"]),
        "num_transforms": int(cfg["num_transforms"]),
        "num_bins": int(cfg["num_bins"]),
        "batch_size": int(cfg["batch_size"]),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(
        f"[train] done  epochs={metrics['n_epochs']}  best_val={metrics['best_val_loss']:.4f}  wall={wall:.1f}s  sec/epoch={metrics['sec_per_epoch']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
