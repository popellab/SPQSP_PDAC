#!/usr/bin/env python
"""
Train the v2 amortized burden NPE: window-relative features + W context, so the
same estimator serves patients of ANY 1st-line window length.

For each valid bank theta (full 1050d trajectory), draw ONE window
W ~ Uniform(W_LO, W_MAX) and compute the window-relative features on [0,W]. The
13k draws span the whole window range -> the NPE amortizes over W. theta in
log-space; prior = the emcee lognormals.
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
W_LO = 400.0   # min plausible 1st-line window (days)

def main():
    d = np.load(OUT/"bank.npz")
    v = d["valid"]; tl = d["theta_log"][v]; sld = d["sld_rel"][v]; ca = d["ca_rel"][v]; grid = d["grid"]
    n = len(tl); rng = np.random.default_rng(3)
    W = rng.uniform(W_LO, F.W_MAX, size=n)                     # one window per theta
    print(f"bank: {n} valid  | W ~ U({W_LO:.0f},{F.W_MAX:.0f})")
    X = np.stack([F.features_v2_from_grid(sld[i], ca[i], grid, W[i]) for i in range(n)])
    good = np.all(np.isfinite(X),axis=1) & np.all(np.isfinite(tl),axis=1)
    X, tl = X[good], tl[good]
    print(f"features v2: X {X.shape} (2*{F.NG2}+1)  theta {tl.shape}  dropped {np.sum(~good)}")
    Xn = F.add_noise_v2(X, rng=rng)

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
    with open(OUT/"npe_posterior_v2.pkl","wb") as f:
        pickle.dump({"posterior":post,"names":NAMES,"MU":MU,"SIG":SIG,
                     "NG2":F.NG2,"W_MAX":F.W_MAX}, f)
    print(f"saved -> {OUT/'npe_posterior_v2.pkl'}")

if __name__=="__main__":
    main()
