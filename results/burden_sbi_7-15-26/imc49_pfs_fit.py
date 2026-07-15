#!/usr/bin/env python
"""
Patient-specific burden fits for the 49 IMC-matched patients (both SLD+CA19-9),
calibration window = PFS (progression) clamped to end-of-1st-line-treatment so we
never fit off-drug time. Applies the arm-appropriate amortized NPE (window-aware,
so no re-banking) -> per-patient posteriors + posterior-predictive check. NO
population/cross-regimen machinery -- purely per-patient characterization to pair
with the IMC spatial data.
"""
import os, pickle
from pathlib import Path
import numpy as np, pandas as pd, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from multiprocessing import Pool
import features as F
import gen_bank as GG, gen_bank_mffx as GM

OUT=Path("/home/chase/SPQSP/SPQSP_PDAC-main/results/burden_sbi_7-15-26"); PDIR=OUT/"cohort_imc49"; PDIR.mkdir(exist_ok=True)
XL=Path("/home/chase/SPQSP/SPQSP_PDAC-main/data/PASS01/metadata/ClinicalTable_filtered_DeID.xlsx")
xl=pd.ExcelFile(XL); tl=xl.parse('Target Lesions'); ca=xl.parse('All CA 19.9')
ev=xl.parse('135 events').rename(columns={'PFS Eligible':'Subject','1st Line Treatment':'arm','Date to be used in PFS Calc':'pfs'})
arm=dict(zip(ev.Subject,ev.arm)); pfsd=dict(zip(ev.Subject, pd.to_datetime(ev.pfs)))
imc=pd.read_csv('/tmp/claude-1000/-home-chase-SPQSP-SPQSP-PDAC-main/af6d9afd-f107-484d-8206-af5c92e347a3/scratchpad/imc_longitudinal.csv')
subs=imc[imc.both].subject.tolist()

with open(OUT/"npe_posterior_v2.pkl","rb") as f: PG=pickle.load(f)
with open(OUT/"npe_posterior_mffx.pkl","rb") as f: PM=pickle.load(f)

def wendtx(subj,base):
    d=tl[tl.Subject==subj].copy(); d['dt']=pd.to_datetime(d['LES_TAR_TEST_DT']); inst=d.groupby('InstanceName')['dt'].first()
    end=[(v-base).days for k,v in inst.items() if '1st Line - End' in str(k)]; sec=[(v-base).days for k,v in inst.items() if '2nd Line' in str(k)]
    return end[0] if end else (min(sec) if sec else (d['dt'].max()-base).days)

def extract(subj):
    d=tl[tl.Subject==subj].copy(); d['sum']=pd.to_numeric(d['LES_TAR_SUM'],errors='coerce'); d=d.dropna(subset=['LES_TAR_TEST_DT','sum'])
    d['dt']=pd.to_datetime(d['LES_TAR_TEST_DT']); base=d['dt'].min()
    Wtx=wendtx(subj,base); Wp=(pfsd[subj]-base).days if pd.notna(pfsd.get(subj)) else Wtx
    W=int(min(Wp,Wtx))                                     # PFS clamped to end-of-treatment
    g=d.groupby('dt')['sum'].first().sort_index(); s=pd.DataFrame({'day':(g.index-base).days,'SLD':g.values.astype(int)}); s=s[(s.day>=0)&(s.day<=W)]
    c=ca[ca.Subject==subj].copy(); c['val']=pd.to_numeric(c['CA199_VAL'],errors='coerce'); c=c.dropna(subset=['CA199_DATE','val'])
    c['dt']=pd.to_datetime(c['CA199_DATE']); cc=pd.DataFrame({'day':(c['dt']-base).dt.days.values,'CA199':c['val'].values}).groupby('day',as_index=False)['CA199'].first()
    cc=cc[(cc.day>=0)&(cc.day<=W)].sort_values('day')
    return s,cc,W

# ---- fit each patient ----
rows=[]; postpred_jobs=[]; jmeta=[]; store={}
rng=np.random.default_rng(7)
for subj in subs:
    a='GnP' if 'GnP' in str(arm.get(subj,'')) else 'mFFX'
    s,c,W=extract(subj)
    if len(s)<3 or len(c)<3: continue
    tag=subj.replace('100-','')
    s.to_csv(PDIR/f"{tag}_SLD.csv",index=False); c.to_csv(PDIR/f"{tag}_CA.csv",index=False)
    P = PG if a=='GnP' else PM; NAMES=list(P["names"])
    sld_rel=(s.SLD/s.SLD.iloc[0]).values; ca_rel=(c.CA199/c.CA199.iloc[0]).values
    x=F.features_v2_from_patient(s.day.values.astype(float),sld_rel,c.day.values.astype(float),ca_rel,float(W))
    ph=np.exp(P["posterior"].sample((2000,),x=torch.as_tensor(x,dtype=torch.float32),show_progress_bars=False).numpy())
    store[subj]=(a,s,c,sld_rel,ca_rel,W,NAMES,ph)
    row=dict(subject=subj,tag=tag,arm=a,W=W,nSLD=len(s),nCA=len(c),
             SLD_fin=round(sld_rel[-1],3),CA_nadir=round(ca_rel.min(),3),CA_fin=round(ca_rel[-1],3))
    for j,nm in enumerate(NAMES):
        q=np.percentile(ph[:,j],[16,50,84]); row[f"{nm}_med"]=q[1]; row[f"{nm}_lo"]=q[0]; row[f"{nm}_hi"]=q[2]
    rows.append(row)
    idx=rng.choice(len(ph),40,replace=False)
    for i in idx: postpred_jobs.append((a,np.log(ph[i]))); jmeta.append(subj)

# ---- posterior-predictive (parallel; route to correct sim) ----
def sim_route(job):
    a,tl_=job; return (GG.simulate(tl_) if a=='GnP' else GM.simulate(tl_))
with Pool(16) as p: sims=p.map(sim_route, postpred_jobs)
PP={s:[] for s in store}
for subj,o in zip(jmeta,sims):
    if o is not None: PP[subj].append(o)

for subj in list(store):
    a,s,c,sld_rel,ca_rel,W,NAMES,ph=store[subj]; sims_t=PP[subj]; G=GG if a=='GnP' else GM
    row=[r for r in rows if r['subject']==subj][0]
    if len(sims_t)>=5:
        SLD=np.array([np.interp(s.day.values,G.GRID,o[0]) for o in sims_t]); CA=np.array([np.interp(c.day.values,G.GRID,o[1]) for o in sims_t])
        r2=np.random.default_rng(11); NK=15
        SLDn=(SLD[:,None,:]+r2.normal(0,0.10,(SLD.shape[0],NK,SLD.shape[1]))).reshape(-1,SLD.shape[1])
        CAn=(CA[:,None,:]*np.exp(r2.normal(0,0.45,(CA.shape[0],NK,CA.shape[1])))).reshape(-1,CA.shape[1])
        slo,shi=np.percentile(SLDn,[5,95],0); clo,chi=np.percentile(CAn,[5,95],0)
        row['sld_cov']=round(np.mean((sld_rel>=slo)&(sld_rel<=shi)),2); row['ca_cov']=round(np.mean((ca_rel>=clo)&(ca_rel<=chi)),2)
    else: row['sld_cov']=row['ca_cov']=np.nan
    row['misspec']=bool((row.get('sld_cov',0)<0.5) or (row.get('ca_cov',0)<0.5)) if np.isfinite(row.get('sld_cov',np.nan)) else True

res=pd.DataFrame(rows)
res.to_csv(OUT/"imc49_pfs_posteriors.csv",index=False)
print(f"IMC-49 PFS-window fits: {len(res)} patients ({(res.arm=='GnP').sum()} GnP, {(res.arm=='mFFX').sum()} mFFX)")
print(f"misspec-flagged: {int(res.misspec.sum())}/{len(res)}")
print(f"median window: {res.W.median():.0f}d (PFS-clamped)")
print(res[['tag','arm','W','nSLD','nCA','SLD_fin','CA_fin','f_ca199_C2_med','k_resist_med','sld_cov','ca_cov','misspec']].to_string(index=False))

# ---- QC fits figures: all 49 (SLD grid + CA19-9 grid), PFS window ----
import math
mis={r['subject']:r['misspec'] for r in rows}
subs_ok=[r['subject'] for r in rows]; n=len(subs_ok); ncol=7; nrow=math.ceil(n/ncol)
for chan,logy in [('SLD',False),('CA',True)]:
    fig,ax=plt.subplots(nrow,ncol,figsize=(ncol*2.4,nrow*1.9))
    for k,subj in enumerate(subs_ok):
        a=ax.flat[k]; arm,s,c,sld_rel,ca_rel,W,NAMES,ph=store[subj]; sims_t=PP[subj]; G=GG if arm=='GnP' else GM
        tg=np.linspace(0,W,60)
        if chan=='SLD': Y=np.array([np.interp(tg,G.GRID,o[0]) for o in sims_t]); dy=s.day.values; dat=sld_rel
        else:           Y=np.array([np.interp(tg,G.GRID,o[1]) for o in sims_t]); dy=c.day.values; dat=ca_rel
        if len(Y):
            lo,md,hi=np.percentile(Y,[5,50,95],0); a.fill_between(tg,lo,hi,alpha=.25,color='C0'); a.plot(tg,md,'C0',lw=1)
        a.plot(dy,dat,'ko',ms=3)
        if logy: a.set_yscale('log')
        a.set_title(f"{subj.replace('100-','')} {arm}",fontsize=7,color='crimson' if mis[subj] else 'k'); a.tick_params(labelsize=5)
    for k in range(n,nrow*ncol): ax.flat[k].axis('off')
    plt.suptitle(f"IMC-49 {chan} posterior-predictive fits (PFS window; band=model+obs-noise 5-95%) — red=misspec",fontsize=11)
    plt.tight_layout(); plt.savefig(OUT/f"imc49_fits_{chan}_7-15-26.png",dpi=110); plt.close()
print("figures: imc49_fits_SLD_7-15-26.png, imc49_fits_CA_7-15-26.png")
print("IMC49_DONE")
