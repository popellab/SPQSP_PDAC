"""Build a vpop.Prior from pdac_priors.csv.

Reads the QSP prior CSV (name, median, distribution, dist_param1, dist_param2,
lower_bound, upper_bound) and produces the five arrays vpop.Prior needs:

    mu_0        log-scale centres (or logit for bounded margins)
    L_sigma_1   (P,P) Cholesky of epistemic covariance over mu
    omega_0     (P,) between-patient log-scale widths
    L_R         (P,P) Cholesky of between-patient correlation
    logit       (P,) boolean mask for (0,1)-bounded margins

The total marginal sigma from the CSV is split into epistemic and population
components via a configurable fraction (default: 30% epistemic, 70% population).
Without elicited factor loadings, L_R starts as identity.

Usage:
    python scripts/build_vpop_prior.py                           # defaults
    python scripts/build_vpop_prior.py --epistemic-frac 0.4      # 40% epistemic
    python scripts/build_vpop_prior.py -o vpop_prior.npz         # custom output
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# vpop is a sibling repo; add it to the path if not installed
_vpop_src = Path(__file__).resolve().parents[1] / "vpop" / "src"
if _vpop_src.is_dir():
    sys.path.insert(0, str(_vpop_src))

from vpop.prior import Prior, save_prior


def load_priors_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def build_prior(
    rows: list[dict],
    epistemic_frac: float = 0.30,
    min_omega: float = 0.05,
) -> Prior:
    """Build a vpop.Prior from parsed CSV rows.

    Parameters
    ----------
    rows : list of dicts from DictReader
    epistemic_frac : fraction of total log-sigma allocated to epistemic
        uncertainty (L_sigma_1 diagonal). The rest goes to omega_0.
    min_omega : floor on omega_0 to avoid degenerate zero-width populations.
    """
    P = len(rows)
    names = []
    medians = np.zeros(P)
    sigma_epistemic = np.zeros(P)
    omega = np.zeros(P)
    logit_mask = np.zeros(P, dtype=bool)

    for j, row in enumerate(rows):
        name = row["name"]
        names.append(name)
        med = float(row["median"])
        dist = row["distribution"]
        sigma_total = float(row["dist_param2"])
        lb = row.get("lower_bound", "").strip()
        ub = row.get("upper_bound", "").strip()

        if dist == "beta":
            # Beta(a,b) on (0,1): treat as logit-normal in vpop
            logit_mask[j] = True
            medians[j] = med
            # Delta-method logit-space SD for Beta(a,b):
            #   Var(logit(X)) ≈ 1 / [p * q * (a + b + 1)]
            #   where p = a/(a+b), q = 1-p
            alpha = float(row["dist_param1"])
            beta_p = float(row["dist_param2"])
            p = alpha / (alpha + beta_p)
            q = 1.0 - p
            logit_sd = 1.0 / np.sqrt(p * q * (alpha + beta_p + 1.0))
            sigma_epistemic[j] = epistemic_frac * logit_sd
            omega[j] = max((1.0 - epistemic_frac) * logit_sd, min_omega)

        elif dist == "lognormal":
            medians[j] = med
            if lb and ub:
                # Bounded lognormal on (lb, ub) — treat as logit-normal
                lb_f, ub_f = float(lb), float(ub)
                logit_mask[j] = True
                # Rescale median to (0,1) fraction within bounds
                medians[j] = (med - lb_f) / (ub_f - lb_f)
                # Approximate logit-space sigma from log-space sigma
                # (crude but sufficient for a prior)
                logit_sd = sigma_total / (medians[j] * (1.0 - medians[j]))
                logit_sd = min(logit_sd, 5.0)  # cap extreme values
                sigma_epistemic[j] = epistemic_frac * logit_sd
                omega[j] = max((1.0 - epistemic_frac) * logit_sd, min_omega)
            else:
                # Standard unbounded lognormal
                sigma_epistemic[j] = epistemic_frac * sigma_total
                omega[j] = max((1.0 - epistemic_frac) * sigma_total, min_omega)
        else:
            raise ValueError(f"Unknown distribution {dist!r} for {name}")

    L_sigma_1 = np.diag(sigma_epistemic)
    L_R = np.eye(P)

    prior = Prior.of_medians(
        names=names,
        medians=medians,
        L_sigma_1=L_sigma_1,
        omega_0=omega,
        L_R=L_R,
        logit=logit_mask if logit_mask.any() else None,
    )
    return prior


def main():
    parser = argparse.ArgumentParser(
        description="Build vpop.Prior from pdac_priors.csv")
    parser.add_argument(
        "--priors", type=Path,
        default=Path(__file__).resolve().parent.parent
        / "parameters" / "pdac_priors.csv",
        help="Path to QSP priors CSV")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output .npz path (default: parameters/vpop_prior.npz)")
    parser.add_argument(
        "--epistemic-frac", type=float, default=0.30,
        help="Fraction of total sigma for epistemic uncertainty (default: 0.30)")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.priors.parent / "vpop_prior.npz"

    rows = load_priors_csv(args.priors)
    print(f"Loaded {len(rows)} parameters from {args.priors}")

    prior = build_prior(rows, epistemic_frac=args.epistemic_frac)

    n_logit = 0 if prior.logit is None else int(prior.logit.sum())
    print(f"Prior: {prior.n_params} parameters, {n_logit} bounded (logit)")
    print(f"  omega_0 range: [{prior.omega_0.min():.3f}, {prior.omega_0.max():.3f}]")
    diag = np.diag(prior.L_sigma_1)
    print(f"  L_sigma_1 diag range: [{diag.min():.3f}, {diag.max():.3f}]")
    print(f"  L_R: identity ({prior.n_params}x{prior.n_params})")

    save_prior(args.output, prior)
    print(f"Saved to {args.output}")

    # Verify round-trip
    from vpop.prior import read_prior
    rt = read_prior(args.output)
    assert rt.names == prior.names
    assert np.allclose(rt.mu_0, prior.mu_0)
    assert np.allclose(rt.omega_0, prior.omega_0)
    print("Round-trip verified.")


if __name__ == "__main__":
    main()
