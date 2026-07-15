#!/usr/bin/env python
"""
Simulation-Based Calibration (SBC) for the v2 amortized burden NPE.

Generative check (held-out, NOT training data):
  for many i:  theta_i ~ prior;  x_i = features_v2(sim(theta_i), W_i) + noise
               rank_i[d] = #{posterior samples[d] < theta_i[d]}
If the NPE is calibrated, ranks are Uniform(0,L) per dimension. Deviations:
  U-shaped  -> OVER-confident (posteriors too narrow)
  Bell      -> UNDER-confident (too wide)
  Sloped    -> biased
Reports per-param KS p-value vs uniform + a coverage metric, and a rank-hist plot.
"""
import os, pickle, time
from pathlib import Path
import numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy.stats import kstest
import gen_bank as G          # simulate(theta_log), MU, SIG, NAMES, GRID
import features as F

OUT = Path("/home/chase/SPQSP/SPQSP_PDAC-main/results/burden_sbi_7-15-26")
NAMES = ["k_C1_growth","k_C_nabp","IC50_nabp","Kc_nabp","r_nabp","k_cell_clear",
         "k_ca199_shed","k_ca199_clear","k_sec_suppress","k_resist","r_resist","f_ca199_C2"]
MU  = np.array([np.log(x) for x in [0.02,0.06,92.0,8e7,1.5,0.1,1e-3,0.5,20.0,1e-4,100.0,10.0]])
SIG = np.array([0.6,0.9,1.2,0.6,0.4,0.7,1.2,0.6,1.5,1.5,1.0,1.5])

def main():
    N  = int(os.environ.get("NSBC","700"))
    L  = int(os.environ.get("LPOST","1000"))
    rng = np.random.default_rng(99)
    theta_log = MU + SIG*rng.standard_normal((N, 12))
    t0=time.time()
    with Pool(16) as p:
        res = p.map(G.simulate, list(theta_log))
    keep=[i for i,r in enumerate(res) if r is not None]
    theta_log=theta_log[keep]; trajs=[res[i] for i in keep]
    n=len(keep); print(f"SBC set: {n}/{N} valid sims ({(time.time()-t0):.0f}s)")
    W = rng.uniform(400.0, F.W_MAX, size=n)
    X = np.stack([F.features_v2_from_grid(trajs[i][0],trajs[i][1],G.GRID,W[i]) for i in range(n)])
    X = F.add_noise_v2(X, rng=rng)

    with open(OUT/"npe_posterior_v2.pkl","rb") as f: post = pickle.load(f)["posterior"]
    ranks=np.zeros((n,12),dtype=int)
    t1=time.time()
    for i in range(n):
        s=post.sample((L,), x=torch.as_tensor(X[i],dtype=torch.float32), show_progress_bars=False).numpy()
        ranks[i]=(s < theta_log[i]).sum(0)
        if (i+1)%50==0:
            el=time.time()-t1; print(f"  ranked {i+1}/{n}  ({el:.0f}s, {el/(i+1):.2f}s/pt, eta {(n-i-1)*el/(i+1):.0f}s)",flush=True)
    print(f"posterior sampling {(time.time()-t1):.0f}s")

    # per-param diagnostics
    print(f"\n{'param':16s} {'KS_p':>7s} {'rank_mean':>10s} {'central90_cov':>14s}  verdict")
    print("-"*66)
    r01 = ranks/float(L)
    verdicts={}
    for d,nm in enumerate(NAMES):
        ksp = kstest(r01[:,d], 'uniform').pvalue
        rmean = r01[:,d].mean()               # ~0.5 if unbiased
        # central 90% coverage: fraction with rank in [0.05,0.95]*L  (should be ~0.90)
        cov = np.mean((r01[:,d]>=0.05)&(r01[:,d]<=0.95))
        if ksp>0.05: v="calibrated"
        elif cov<0.85: v="OVER-confident"
        elif cov>0.95: v="under-confident"
        else: v="biased/other"
        verdicts[nm]=v
        print(f"{nm:16s} {ksp:7.3f} {rmean:10.3f} {cov:14.2f}  {v}")

    # rank-hist figure
    fig,ax=plt.subplots(3,4,figsize=(15,9))
    for d,nm in enumerate(NAMES):
        a=ax.flat[d]; a.hist(ranks[:,d],bins=20,range=(0,L),color='C0',alpha=.8)
        a.axhline(n/20,color='k',ls='--',lw=1); a.set_title(f"{nm}\n{verdicts[nm]}",fontsize=9)
        a.set_xticks([]); a.set_yticks([])
    plt.suptitle(f"SBC rank histograms (v2 NPE, n={n}, L={L}) — flat=calibrated, U=over-confident, ∩=under-confident")
    plt.tight_layout(); plt.savefig(OUT/"sbc_ranks.png",dpi=110); plt.close()
    np.savez(OUT/"sbc.npz", ranks=ranks, theta_log=theta_log, names=np.array(NAMES), L=L)
    ncal=sum(v=="calibrated" for v in verdicts.values())
    print(f"\nSBC summary: {ncal}/12 params calibrated (KS p>0.05). fig -> sbc_ranks.png")
    print("SBC_DONE")

if __name__=="__main__":
    main()
