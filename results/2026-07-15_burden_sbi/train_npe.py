#!/usr/bin/env python
"""
Train the amortized burden NPE on the simulation bank.

theta is inferred in LOG-space (prior = independent Normal(MU,SIG) = the emcee
lognormals). Features from features.py; per-channel observation noise added so the
NPE learns that x is a noisy measurement. Saves the posterior estimator for
instant, per-patient amortized inference.
"""
import os, time, pickle
from pathlib import Path
import numpy as np, torch
from sbi.inference import SNPE
import features as F

OUT = Path("/home/chase/SPQSP/SPQSP_PDAC-main/results/burden_sbi_7-15-26")
NAMES = ["k_C1_growth","k_C_nabp","IC50_nabp","Kc_nabp","r_nabp","k_cell_clear",
         "k_ca199_shed","k_ca199_clear","k_sec_suppress","k_resist","r_resist","f_ca199_C2"]
MU  = np.array([np.log(x) for x in [0.02,0.06,92.0,8e7,1.5,0.1,1e-3,0.5,20.0,1e-4,100.0,10.0]])
SIG = np.array([0.6,0.9,1.2,0.6,0.4,0.7,1.2,0.6,1.5,1.5,1.0,1.5])

def main():
    d = np.load(OUT/"bank.npz")
    v = d["valid"]; tl = d["theta_log"][v]; sld = d["sld_rel"][v]; ca = d["ca_rel"][v]; grid = d["grid"]
    print(f"bank: {v.sum()} valid / {len(v)}")
    X = np.stack([F.features_from_grid(sld[i], ca[i], grid) for i in range(len(tl))])
    # drop any non-finite feature rows (safety)
    good = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(tl), axis=1)
    X, tl = X[good], tl[good]
    print(f"features: X {X.shape}  theta {tl.shape}  (dropped {np.sum(~good)} non-finite)")
    rng = np.random.default_rng(0)
    Xn = F.add_noise(X, rng=rng)

    theta = torch.as_tensor(tl, dtype=torch.float32)
    x     = torch.as_tensor(Xn, dtype=torch.float32)
    prior = torch.distributions.Independent(
        torch.distributions.Normal(torch.as_tensor(MU,dtype=torch.float32),
                                    torch.as_tensor(SIG,dtype=torch.float32)), 1)
    t0=time.time()
    inf = SNPE(prior=prior, density_estimator="nsf")
    inf.append_simulations(theta, x)
    de = inf.train(show_train_summary=True)
    post = inf.build_posterior(de)
    print(f"trained in {(time.time()-t0)/60:.1f} min")
    with open(OUT/"npe_posterior.pkl","wb") as f:
        pickle.dump({"posterior":post,"names":NAMES,"MU":MU,"SIG":SIG,
                     "OUT_GRID":F.OUT_GRID}, f)
    # normalization stats for feature sanity
    np.savez(OUT/"train_stats.npz", X_mean=X.mean(0), X_std=X.std(0), names=np.array(NAMES))
    print(f"saved -> {OUT/'npe_posterior.pkl'}")

if __name__=="__main__":
    main()
