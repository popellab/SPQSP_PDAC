#!/usr/bin/env python
"""
v2 validation: apply the window-aware amortized NPE to all three patients using
their 1ST-LINE-truncated data + window W. PM-036 (full window) is checked against
its emcee gold-standard; PM031/PM001 (short windows) are sanity-checked (responders
-> low resistance, good response params) since they lack a clean 1st-line MCMC.
"""
import pickle, json
from pathlib import Path
import numpy as np, pandas as pd, torch
import features as F

OUT = Path("/home/chase/SPQSP/SPQSP_PDAC-main/results/2026-07-15_burden_sbi")
PDIR= OUT/"patients"
meta= json.load(open(PDIR/"meta.json"))
with open(OUT/"npe_posterior_v2.pkl","rb") as f: P = pickle.load(f)
post, NAMES = P["posterior"], P["names"]

def patient_x(tag, W):
    s=pd.read_csv(PDIR/f"{tag}_SLD_1L.csv"); c=pd.read_csv(PDIR/f"{tag}_CA199_1L.csv")
    c=c[c.day>=0]
    sld_rel=(s.SLD/s.SLD.iloc[0]).values; ca_rel=(c.CA199/c.CA199.iloc[0]).values
    return F.features_v2_from_patient(s.day.values.astype(float), sld_rel,
                                      c.day.values.astype(float), ca_rel, W)

def sample(tag, W):
    x=patient_x(tag, W)
    return np.exp(post.sample((5000,), x=torch.as_tensor(x,dtype=torch.float32)).numpy())

for tag in ["PM036","PM031","PM001"]:
    W=meta[tag]["W"]; phys=sample(tag, W)
    print(f"\n===== {tag}  (W={W}d, {meta[tag]['subject']}, SLD_fin={meta[tag]['SLD_final_rel']}, CA nadir={meta[tag]['CA_nadir_rel']} final={meta[tag]['CA_final_rel']}) =====")
    for i,n in enumerate(NAMES):
        q=np.percentile(phys[:,i],[16,50,84]); print(f"  {n:16s} {q[1]:.3g}  [{q[0]:.3g}, {q[2]:.3g}]")
    np.save(OUT/f"{tag}_npe_v2_samples.npy", phys)

# PM-036 gold-standard check
V4=Path("/home/chase/SPQSP/SPQSP_PDAC-main/results/2026-07-14_emcee_singlepatient_v4")
em=pd.read_csv(V4/"posterior.csv"); phys=np.load(OUT/"PM036_npe_v2_samples.npy")
agree=0;tot=0
print("\n=== PM-036 v2 NPE vs emcee gold-standard ===")
for i,n in enumerate(NAMES):
    if n in em.columns:
        e=np.percentile(em[n],[16,84]); m=np.percentile(phys[:,i],50)
        ok=e[0]<=m<=e[1]; agree+=ok; tot+=1
        print(f"  {n:16s} NPE {m:.3g}  emcee[{e[0]:.3g},{e[1]:.3g}]  {'OK' if ok else 'x'}")
print(f"PM-036 v2: NPE median inside emcee[16,84] for {agree}/{tot} params")
