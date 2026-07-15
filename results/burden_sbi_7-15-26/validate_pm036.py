#!/usr/bin/env python
"""
Acceptance test: does the amortized NPE recover the PM-036 emcee gold-standard?

PM-036 is a full-1st-line-window patient (1015 d, no line switch) so v1 fixed-grid
features are valid. We compute its features, sample the amortized NPE posterior,
and compare to the emcee v4 posterior (same 12 params incl f_ca199_C2).
"""
import pickle
from pathlib import Path
import numpy as np, pandas as pd
import features as F

OUT = Path("/home/chase/SPQSP/SPQSP_PDAC-main/results/burden_sbi_7-15-26")
V4  = Path("/home/chase/SPQSP/SPQSP_PDAC-main/results/emcee_singlepatient_v4_7-14-26")

# PM-036 observed trajectory (same source as the emcee fits)
sld = pd.read_csv(V4/"PM036_SLD.csv");   sld = sld[sld.day<=1050]
ca  = pd.read_csv(V4/"PM036_CA199.csv"); ca  = ca[(ca.day>=0)&(ca.day<=1050)]
sld_rel = (sld.SLD/sld.SLD.iloc[0]).values; sld_day = sld.day.values.astype(float)
ca_rel  = (ca.CA199/ca.CA199.iloc[0]).values; ca_day = ca.day.values.astype(float)
x_obs = F.features_from_patient(sld_day, sld_rel, ca_day, ca_rel)

import torch
with open(OUT/"npe_posterior.pkl","rb") as f: P = pickle.load(f)
post, NAMES = P["posterior"], P["names"]
samp = post.sample((5000,), x=torch.as_tensor(x_obs,dtype=torch.float32)).numpy()
phys = np.exp(samp)   # log -> physical

emcee = pd.read_csv(V4/"posterior.csv")   # v4: 12 phys params + 2 noise
print(f"{'param':16s} {'NPE median[16,84]':32s}  {'emcee v4 median[16,84]':32s}")
print("-"*84)
for i,n in enumerate(NAMES):
    q=np.percentile(phys[:,i],[16,50,84])
    npe=f"{q[1]:.3g} [{q[0]:.3g},{q[2]:.3g}]"
    if n in emcee.columns:
        e=np.percentile(emcee[n],[16,50,84]); em=f"{e[1]:.3g} [{e[0]:.3g},{e[2]:.3g}]"
    else: em="(n/a)"
    print(f"{n:16s} {npe:32s}  {em:32s}")

# posterior-predictive check via the NPE median could go here; for now report
# rough agreement metric: fraction of params whose NPE median is within emcee 16-84
agree=0; tot=0
for i,n in enumerate(NAMES):
    if n in emcee.columns:
        e=np.percentile(emcee[n],[16,84]); m=np.percentile(phys[:,i],50)
        agree+= (e[0]<=m<=e[1]); tot+=1
print(f"\nNPE median inside emcee[16,84] for {agree}/{tot} params")
np.save(OUT/"pm036_npe_samples.npy", phys)
print(f"saved NPE samples -> {OUT/'pm036_npe_samples.npy'}")
