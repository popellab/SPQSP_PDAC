#!/usr/bin/env python
"""Standalone SNPE-C trainer for Delta GPU — multi-round atomic loss.

Reads a ``manifest.json`` listing N rounds, each with its own ``theta_rK.pt``,
``x_rK.pt``, and a ``proposal`` tag. Round 1 proposes from the (throwaway MVN)
prior; rounds 2+ propose from ``DirectPosterior(prev_estimator, prior, x=obs)``.

Manifest schema::

    {
      "num_rounds": 4,
      "obs_transformed": [...],                 # (n_obs,) list[float]
      "npe_config": {"hidden_features": 256, ...},
      "rounds": [
        {"theta_file": "theta_r1.pt", "x_file": "x_r1.pt", "proposal": "prior"},
        {"theta_file": "theta_r2.pt", "x_file": "x_r2.pt",
         "proposal": "posterior_from_round_1"},
        ...
      ]
    }

All per-round ``theta`` tensors are in log-space, ``x`` tensors in copula-space,
same as :mod:`train_npe_delta` expects. The throwaway MVN prior is fit from
the round-1 theta (so rounds 2+ share the same prior footprint for proposal
conditioning — sbi needs the same prior object across append_simulations calls).
"""
import argparse
import json
import threading
import time
from pathlib import Path

import torch
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn


def _fit_mvn_prior(theta: torch.Tensor, device: str) -> torch.distributions.MultivariateNormal:
    """Moment-match an MVN to log-theta. Same recipe as ``train_npe_delta``."""
    return torch.distributions.MultivariateNormal(
        theta.mean(0).to(device),
        torch.diag(theta.var(0).to(device) + 1e-6),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--input-dir", required=True,
                    help="Directory containing per-round theta/x tensors.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument(
        "--warm-start-from", default=None,
        help=(
            "Path to a pretrained density_estimator.pt to use as the round-1 "
            "weights (skips re-training round 1 from scratch). Round 1's "
            "theta/x are still loaded — only to moment-match the throwaway "
            "MVN prior — but no append_simulations / train pass runs for "
            "round 1. Subsequent rounds train with retrain_from_scratch=False, "
            "so their atomic loss only operates on the new round's data and "
            "fine-tunes the existing weights. Drops per-round wall from "
            "~50 min (cumulative retrain at 11x atomic-loss cost) to ~1-2 min "
            "for typical sequential refinement (300-row rounds vs 90k-row r1)."
        ),
    )
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    in_dir = Path(args.input_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(args.manifest).read_text())
    rounds = manifest["rounds"]
    num_rounds = manifest.get("num_rounds", len(rounds))
    if num_rounds != len(rounds):
        raise ValueError(f"manifest num_rounds={num_rounds} disagrees with rounds list length {len(rounds)}")

    cfg = manifest["npe_config"]
    max_epochs = int(cfg.get("max_epochs", 80))
    # Atomic-loss inner-loop size. sbi default is 10; lowering to 5 halves
    # per-epoch cost in rounds 2+ at a modest precision tradeoff. Plumb through
    # the manifest so callers can tune without editing this script.
    num_atoms = int(cfg.get("num_atoms", 10))
    obs_vec = torch.tensor(manifest["obs_transformed"], dtype=torch.float32).to(device)
    obs_for_posterior = obs_vec.unsqueeze(0)  # (1, n_obs) for set_default_x

    warm_start_path = args.warm_start_from
    if warm_start_path is None and manifest.get("warm_start_from"):
        warm_start_path = str(in_dir / manifest["warm_start_from"])
    warm_start = warm_start_path is not None

    print(f"[train] device={device}  rounds={num_rounds}  num_atoms={num_atoms}  "
          f"warm_start={warm_start}", flush=True)
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"[train] gpu: {torch.cuda.get_device_name(0)}  vram={props.total_memory/1e9:.1f}GB",
              flush=True)

    # Fit throwaway MVN prior from round-1 theta. Kept constant across all rounds
    # so proposal objects all share the same prior footprint — required by sbi
    # for multi-round append_simulations to compose a valid atomic loss.
    theta_r1 = torch.load(in_dir / rounds[0]["theta_file"], weights_only=False)
    prior = _fit_mvn_prior(theta_r1, device)

    if warm_start:
        # Skip the round-1 retrain entirely: build the network shape via
        # sbi's posterior_nn factory (using a 2-row dummy from round-1 data
        # to determine in/out dims), then overwrite the weights with the
        # pretrained estimator's state_dict. Round-1 data is NOT appended
        # to the inference object — that's the whole point of warm-start.
        # Each subsequent round trains atomic-loss on its own (small) data
        # with retrain_from_scratch=False, fine-tuning the loaded weights.
        print(f"[train] warm-start: loading {warm_start_path}", flush=True)
        loaded_estimator = torch.load(warm_start_path, weights_only=False)
        de_factory = posterior_nn(
            model="nsf",
            hidden_features=int(cfg["hidden_features"]),
            num_transforms=int(cfg["num_transforms"]),
            num_bins=int(cfg["num_bins"]),
        )
        inference = NPE(prior=prior, density_estimator=de_factory, device=device)
        # Build the net shape with a 2-row dummy from round-1 — sbi's
        # _build_neural_net needs a (theta, x) batch to infer dims.
        x_r1_dummy = torch.load(in_dir / rounds[0]["x_file"], weights_only=False)[:2]
        built_net = inference._build_neural_net(theta_r1[:2].to("cpu"), x_r1_dummy.to("cpu"))
        built_net.load_state_dict(loaded_estimator.state_dict())
        inference._neural_net = built_net.to(device)
        prev_posterior = inference.build_posterior(density_estimator=inference._neural_net)
        # Snapshot the warm-start as round-1 outputs so downstream tooling
        # (per-round estimator browser, sequential_runner) doesn't have to
        # special-case the warm-start path.
        cpu_est = inference._neural_net.to("cpu")
        torch.save(cpu_est, out / "density_estimator.pt")
        torch.save(cpu_est, out / "density_estimator_r1.pt")
        inference._neural_net.to(device)
        # Round 1 isn't trained here; we still need a placeholder for the
        # metrics block. Read best_val_loss from the warm-start metrics.json
        # if it sits next to the .pt; otherwise NaN.
        warm_metrics_path = Path(warm_start_path).with_name("metrics.json")
        warm_best_val = float("nan")
        if warm_metrics_path.exists():
            try:
                warm_best_val = float(json.loads(warm_metrics_path.read_text()).get(
                    "best_val_loss", float("nan")))
            except Exception:
                pass
        per_round_epochs_init: list[int] = [0]
        per_round_best_val_init: list[float] = [warm_best_val]
        round_wall_init: list[float] = [0.0]
        rounds_to_iter = list(enumerate(rounds, start=1))[1:]  # skip round 1
    else:
        de = posterior_nn(
            model="nsf",
            hidden_features=int(cfg["hidden_features"]),
            num_transforms=int(cfg["num_transforms"]),
            num_bins=int(cfg["num_bins"]),
        )
        inference = NPE(prior=prior, density_estimator=de, device=device)
        prev_posterior = None
        per_round_epochs_init = []
        per_round_best_val_init = []
        round_wall_init = []
        rounds_to_iter = list(enumerate(rounds, start=1))

    round_wall: list[float] = list(round_wall_init)
    per_round_epochs: list[int] = list(per_round_epochs_init)
    per_round_best_val: list[float] = list(per_round_best_val_init)

    for k, rd in rounds_to_iter:
        theta_k = torch.load(in_dir / rd["theta_file"], weights_only=False)
        x_k = torch.load(in_dir / rd["x_file"], weights_only=False)
        prop_tag = rd.get("proposal", "prior")
        print(f"[train] round {k}/{num_rounds}: theta={tuple(theta_k.shape)}  "
              f"x={tuple(x_k.shape)}  proposal={prop_tag}", flush=True)

        if prop_tag == "prior" or prev_posterior is None:
            proposal = prior
        elif prop_tag.startswith("posterior_from_round_"):
            # Use the most recent round's posterior as the proposal, regardless
            # of which specific round the tag points to. Sequential SNPE-C only
            # needs the *immediately* preceding proposal — earlier rounds'
            # samples are already accounted for via the data in the inference
            # object's round accumulator.
            proposal = prev_posterior.set_default_x(obs_for_posterior)
        else:
            raise ValueError(f"round {k}: unknown proposal tag {prop_tag!r}")

        inference = inference.append_simulations(theta_k, x_k, proposal=proposal)

        # Progress poll across all rounds. sbi restarts its summary on train(),
        # so per-round losses overwrite the accumulator; we snapshot at the end.
        stop = threading.Event()

        def _poll() -> None:
            t0 = time.time()
            last = 0
            while not stop.wait(5.0):
                tl = inference._summary.get("training_loss", [])
                vl = inference._summary.get("validation_loss", [])
                n = len(tl)
                if n > last:
                    last = n
                    print(
                        f"[train r{k}] epoch {n}/{max_epochs}  "
                        f"train={tl[-1]:.4f}  val={vl[-1]:.4f}  "
                        f"elapsed={time.time()-t0:.1f}s",
                        flush=True,
                    )

        t = threading.Thread(target=_poll, daemon=True)
        t.start()
        t_start = time.time()
        try:
            train_kwargs = dict(
                training_batch_size=int(cfg["batch_size"]),
                max_num_epochs=max_epochs,
                show_train_summary=True,
            )
            # Atomic loss only runs when proposal != prior — only forward
            # num_atoms then to avoid sbi warning about unused kwarg in NPE-A
            # round-1-from-scratch mode.
            if prop_tag.startswith("posterior_from_round_"):
                train_kwargs["num_atoms"] = num_atoms
            # Warm-start mode: every post-warm round fine-tunes the existing
            # weights instead of reinitializing. retrain_from_scratch=False
            # keeps the loaded estimator across train() calls; True (default)
            # would discard our warm start on the first call.
            if warm_start:
                train_kwargs["retrain_from_scratch"] = False
            estimator = inference.train(**train_kwargs)
        finally:
            stop.set()
        wall = time.time() - t_start
        round_wall.append(wall)

        tl = list(inference._summary["training_loss"])
        vl = list(inference._summary["validation_loss"])
        best_val = float(inference._summary["best_validation_loss"][-1])
        per_round_epochs.append(len(tl))
        per_round_best_val.append(best_val)

        # Save per-round learning curve so convergence plots can reassemble
        # the training trajectory across rounds.
        lc_path = out / f"learning_curve_r{k}.csv"
        with open(lc_path, "w") as f:
            f.write("epoch,train_loss,val_loss\n")
            for i, (tr, va) in enumerate(zip(tl, vl)):
                f.write(f"{i+1},{tr},{va}\n")

        print(f"[train r{k}] done  epochs={len(tl)}  best_val={best_val:.4f}  wall={wall:.1f}s",
              flush=True)

        # Build the posterior for this round; next iteration will use it as
        # the proposal. Also save the estimator to disk so laptop-side can
        # reconstruct a DirectPosterior for sampling / diagnostics.
        posterior = inference.build_posterior()
        prev_posterior = posterior

        # Always write the latest estimator; overwrite as we go so crashes
        # leave the most recent one on disk. Also snapshot per-round copies.
        cpu_est = estimator.to("cpu")
        torch.save(cpu_est, out / "density_estimator.pt")
        torch.save(cpu_est, out / f"density_estimator_r{k}.pt")
        estimator = estimator.to(device)

    metrics = {
        "num_rounds": num_rounds,
        "best_val_loss_per_round": per_round_best_val,
        "best_val_loss": per_round_best_val[-1],
        "epochs_per_round": per_round_epochs,
        "wall_seconds_per_round": [round(w, 1) for w in round_wall],
        "wall_seconds": round(sum(round_wall), 1),
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "n_params": int(theta_r1.shape[1]),
        "n_obs": int(obs_vec.numel()),
        "hidden_features": int(cfg["hidden_features"]),
        "num_transforms": int(cfg["num_transforms"]),
        "num_bins": int(cfg["num_bins"]),
        "batch_size": int(cfg["batch_size"]),
        "max_epochs": max_epochs,
        "n_train_per_round": [
            int(torch.load(in_dir / rd["theta_file"], weights_only=False).shape[0])
            for rd in rounds
        ],
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[train] all rounds done  total_wall={sum(round_wall):.1f}s  "
          f"final_best_val={per_round_best_val[-1]:.4f}", flush=True)


if __name__ == "__main__":
    main()
