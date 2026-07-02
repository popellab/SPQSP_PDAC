#!/usr/bin/env python
"""Sample from a trained NPE on Delta GPU for a batch of jobs.

Manifest format (jobs.json):
    {"jobs": [{"name": str, "x_path": str, "n_samples": int}, ...]}

For each job: loads x (1D or 2D), iterates rows, samples n_samples thetas per
row using the density estimator's native .sample(condition=...) call (matches
what sbi's DirectPosterior does internally for a full-support prior — no
rejection), writes {name}_samples.pt. Output shape matches posterior_estimator
.sample(Size([n]), condition=x): (n_samples, n_conditions, theta_dim) when x
is 2D, or (n_samples, theta_dim) when x was 1D on input.
"""
import argparse
import json
import time
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimator", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--chunk-size", type=int, default=256,
                    help="Max conditions processed per GPU call (controls peak VRAM).")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[sample] device={device}", flush=True)
    if device == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"[sample] gpu={p.name}  vram={p.total_memory/1e9:.1f}GB", flush=True)

    estimator = torch.load(args.estimator, weights_only=False).to(device)
    estimator.eval()
    n_params_nn = sum(p.numel() for p in estimator.parameters())
    print(f"[sample] estimator loaded ({n_params_nn:,} NN params)", flush=True)

    manifest = json.loads(Path(args.jobs).read_text())
    for job in manifest["jobs"]:
        name = job["name"]
        n_samples = int(job["n_samples"])
        x_path = job["x_path"]
        x = torch.load(x_path, weights_only=False).float()
        input_was_1d = x.ndim == 1
        if input_was_1d:
            x = x.unsqueeze(0)
        n_cond, x_dim = x.shape
        x = x.to(device)
        print(
            f"[sample] {name}: n_conditions={n_cond} x_dim={x_dim} n_samples={n_samples}",
            flush=True,
        )

        t0 = time.time()
        chunk = max(1, int(args.chunk_size))
        chunks = []
        with torch.no_grad():
            for i in range(0, n_cond, chunk):
                xb = x[i:i + chunk]
                sb = estimator.sample(torch.Size([n_samples]), condition=xb)
                chunks.append(sb.cpu())
                if device == "cuda":
                    torch.cuda.synchronize()
                print(
                    f"[sample] {name}: chunk {i//chunk + 1}/{-(-n_cond // chunk)} "
                    f"({xb.shape[0]} conditions) @ {time.time() - t0:.1f}s",
                    flush=True,
                )
        samples = torch.cat(chunks, dim=1)  # cat along condition dim
        elapsed = time.time() - t0
        if input_was_1d:
            samples = samples.squeeze(1)
        out_path = out / f"{name}_samples.pt"
        torch.save(samples, out_path)
        print(
            f"[sample] {name}: {tuple(samples.shape)} in {elapsed:.2f}s -> {out_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
