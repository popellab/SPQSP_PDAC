#!/usr/bin/env python
"""
Cohort inference + posterior-predictive misspecification check (GnP, 1st-line).

For each usable patient: v2 window-aware features -> amortized NPE posterior
(instant). Then draw NPOST posterior samples, simulate qsp_sim for each, and
check whether the posterior-predictive band covers the patient's SLD + CA19-9
(the misspecification guard SBC cannot provide). Writes cohort_posteriors.csv +
population figures.
"""
import os, pickle, time
from pathlib import Path
import numpy as np, pandas as pd, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from multiprocessing import Pool
import features as F

ARM  = os.environ.get("ARM","GnP")           # 'GnP' or 'mFFX'
SUF  = "" if ARM=="GnP" else f"_{ARM}"
if ARM=="GnP":  import gen_bank as G          # simulate(theta_log)->(sld_rel76,ca_rel76) on G.GRID
else:           import gen_bank_mffx as G
PKL  = "npe_posterior_v2.pkl" if ARM=="GnP" else "npe_posterior_mffx.pkl"

OUT  = Path("/home/chase/SPQSP/SPQSP_PDAC-main/results/burden_sbi_7-15-26")
PDIR = OUT/f"cohort{SUF}"
NPOST = int(os.environ.get("NPOST","40"))    # postpred draws per patient

man = pd.read_csv(OUT/f"cohort{SUF}_manifest.csv"); man = man[man.usable].reset_index(drop=True)
with open(OUT/PKL,"rb") as f: _P=pickle.load(f); post=_P["posterior"]; NAMES=list(_P["names"])

def load(tag):
    s=pd.read_csv(PDIR/f"{tag}_SLD.csv"); c=pd.read_csv(PDIR/f"{tag}_CA.csv")
    return s, c

# ---- Phase 1: amortized posteriors ----
print(f"Phase 1: NPE posteriors for {len(man)} patients")
POST={}   # tag -> (physical samples 2000x12)
xobs={}
for _,r in man.iterrows():
    s,c=load(r.tag); W=float(r.W)
    sld_rel=(s.SLD/s.SLD.iloc[0]).values; ca_rel=(c.CA199/c.CA199.iloc[0]).values
    x=F.features_v2_from_patient(s.day.values.astype(float),sld_rel,c.day.values.astype(float),ca_rel,W)
    xobs[r.tag]=(s,c,sld_rel,ca_rel)
    samp=post.sample((2000,), x=torch.as_tensor(x,dtype=torch.float32), show_progress_bars=False).numpy()
    POST[r.tag]=np.exp(samp)

# ---- Phase 2: posterior-predictive (parallel over patient x draw) ----
print(f"Phase 2: posterior-predictive ({NPOST} draws/patient, {len(man)*NPOST} sims)")
rng=np.random.default_rng(7)
jobs=[]; jidx=[]
for _,r in man.iterrows():
    ph=POST[r.tag]; idx=rng.choice(len(ph),NPOST,replace=False)
    for i in idx:
        jobs.append(np.log(ph[i])); jidx.append(r.tag)
t0=time.time()
with Pool(16) as p:
    sims=p.map(G.simulate, jobs)
print(f"  sims done {(time.time()-t0):.0f}s")
# collect per patient
PP={tag:[] for tag in man.tag}
for tag,out in zip(jidx,sims):
    if out is not None: PP[tag].append(out)   # (sld_rel76, ca_rel76)

# ---- coverage + assemble table ----
rows=[]
for _,r in man.iterrows():
    tag=r.tag; s,c,sld_rel,ca_rel=xobs[tag]; ph=POST[tag]
    row={k:r[k] for k in ["subject","tag","arm","W","nSLD","nCA","SLD_fin","CA_nadir","CA_fin"]}
    for j,nm in enumerate(NAMES):
        q=np.percentile(ph[:,j],[16,50,84]); row[f"{nm}_med"]=q[1]; row[f"{nm}_lo"]=q[0]; row[f"{nm}_hi"]=q[2]
    sims_t=PP[tag]
    if len(sims_t)>=5:
        SLD=np.array([np.interp(s.day.values, G.GRID, st[0]) for st in sims_t])       # rel-diam at SLD days
        CA =np.array([np.interp(c.day.values, G.GRID, st[1]) for st in sims_t])       # rel-CA at CA days
        # posterior-PREDICTIVE band = model spread + observation noise (the levels the
        # NPE was trained with: SLD ~10% additive, CA19-9 ~0.45 on log). Without this
        # the noiseless band is too tight and over-flags misspecification.
        rng2=np.random.default_rng(11); NK=15
        SLDn=(SLD[:,None,:]+rng2.normal(0,0.10,(SLD.shape[0],NK,SLD.shape[1]))).reshape(-1,SLD.shape[1])
        CAn =(CA[:,None,:]*np.exp(rng2.normal(0,0.45,(CA.shape[0],NK,CA.shape[1])))).reshape(-1,CA.shape[1])
        slo,shi=np.percentile(SLDn,[5,95],axis=0); clo,chi=np.percentile(CAn,[5,95],axis=0)
        scov=np.mean((sld_rel>=slo)&(sld_rel<=shi)); ccov=np.mean((ca_rel>=clo)&(ca_rel<=chi))
    else:
        scov=ccov=np.nan
    row["sld_cov"]=round(scov,2); row["ca_cov"]=round(ccov,2); row["n_pp"]=len(sims_t)
    row["misspec"]= bool((scov<0.5) or (ccov<0.5)) if np.isfinite(scov) else True
    rows.append(row)
res=pd.DataFrame(rows)
res.to_csv(OUT/f"cohort{SUF}_posteriors.csv",index=False)
nflag=res.misspec.sum()
print(f"\n[{ARM}] cohort{SUF}_posteriors.csv written. misspec-flagged: {nflag}/{len(res)}")
print(res[["tag","arm","W","CA_fin","f_ca199_C2_med","r_resist_med","k_resist_med","sld_cov","ca_cov","misspec"]].to_string(index=False))

# ---- population figures (SHARED params -> comparable across regimens) ----
key=["k_C1_growth","k_cell_clear","k_resist","r_resist","f_ca199_C2","k_sec_suppress"]
fig,ax=plt.subplots(2,3,figsize=(14,8))
ok=~res.misspec
for a,nm in zip(ax.flat,key):
    y=res[f"{nm}_med"].values
    a.scatter(res.CA_fin[ok], y[ok.values], c='C0', s=28, label='ok')
    a.scatter(res.CA_fin[~ok], y[~ok.values], c='crimson', marker='x', s=40, label='misspec')
    a.set_yscale('log'); a.set_xscale('log'); a.set_xlabel("CA19-9 final / baseline"); a.set_ylabel(f"{nm} (post median)")
    a.set_title(nm); a.grid(alpha=.3,which='both')
ax.flat[0].legend(fontsize=8)
plt.suptitle(f"{ARM} cohort (n={len(res)}) posterior medians vs response (CA19-9 final)")
plt.tight_layout(); plt.savefig(OUT/f"cohort{SUF}_params.png",dpi=110); plt.close()

# postpred coverage summary
fig,ax=plt.subplots(1,2,figsize=(11,4))
ax[0].hist(res.sld_cov.dropna(),bins=np.linspace(0,1,11),color='C0',alpha=.8); ax[0].set_title("SLD postpred coverage"); ax[0].axvline(.9,color='k',ls='--')
ax[1].hist(res.ca_cov.dropna(),bins=np.linspace(0,1,11),color='C1',alpha=.8); ax[1].set_title("CA19-9 postpred coverage"); ax[1].axvline(.9,color='k',ls='--')
for a in ax: a.set_xlabel("fraction of points in 90% band"); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig(OUT/f"cohort{SUF}_coverage.png",dpi=110); plt.close()
print(f"figures: cohort{SUF}_params.png, cohort{SUF}_coverage.png")
print("COHORT_DONE")
