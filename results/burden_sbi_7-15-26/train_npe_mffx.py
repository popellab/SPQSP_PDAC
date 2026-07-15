#!/usr/bin/env python
"""Train the mFFX amortized burden NPE (window-aware v2 features). Parallel to
train_npe_v2.py but on bank_mffx.npz with the 14 mFFX params. W_LO=90 covers all
patient 1st-line windows."""
import os, time, pickle
from pathlib import Path
import numpy as np, torch
from sbi.inference import SNPE
import features as F

OUT = Path("/home/chase/SPQSP/SPQSP_PDAC-main/results/burden_sbi_7-15-26")
NAMES = ["k_C1_growth","k_kill_fu5","IC50_fu5","k_kill_oxa","IC50_oxa","k_kill_iri","IC50_iri",
         "k_cell_clear","k_ca199_shed","k_ca199_clear","k_sec_suppress","k_resist","r_resist","f_ca199_C2"]
MU  = np.array([np.log(x) for x in [0.02,1.0,2.0,1.0,1.0,1.0,0.02,0.1,1e-3,0.5,20.0,1e-4,100.0,10.0]])
SIG = np.array([0.6,1.0,1.2,1.0,1.2,1.0,1.2,0.7,1.2,0.6,1.5,1.5,1.0,1.5])
W_LO = 90.0

def main():
    d = np.load(OUT/"bank_mffx.npz")
    v=d["valid"]; tl=d["theta_log"][v]; sld=d["sld_rel"][v]; ca=d["ca_rel"][v]; grid=d["grid"]
    n=len(tl); rng=np.random.default_rng(3)
    W=rng.uniform(W_LO, F.W_MAX, size=n)
    print(f"mFFX bank: {n} valid | W~U({W_LO:.0f},{F.W_MAX:.0f}) | {len(NAMES)} params")
    X=np.stack([F.features_v2_from_grid(sld[i],ca[i],grid,W[i]) for i in range(n)])
    good=np.all(np.isfinite(X),axis=1)&np.all(np.isfinite(tl),axis=1); X,tl=X[good],tl[good]
    Xn=F.add_noise_v2(X, rng=rng)
    theta=torch.as_tensor(tl,dtype=torch.float32); x=torch.as_tensor(Xn,dtype=torch.float32)
    prior=torch.distributions.Independent(torch.distributions.Normal(
        torch.as_tensor(MU,dtype=torch.float32), torch.as_tensor(SIG,dtype=torch.float32)),1)
    t0=time.time(); inf=SNPE(prior=prior, density_estimator="nsf")
    inf.append_simulations(theta,x); de=inf.train(show_train_summary=True); post=inf.build_posterior(de)
    print(f"trained {(time.time()-t0)/60:.1f}min")
    with open(OUT/"npe_posterior_mffx.pkl","wb") as f:
        pickle.dump({"posterior":post,"names":NAMES,"MU":MU,"SIG":SIG,"NG2":F.NG2,"W_MAX":F.W_MAX}, f)
    print(f"saved -> {OUT/'npe_posterior_mffx.pkl'}")

if __name__=="__main__": main()
