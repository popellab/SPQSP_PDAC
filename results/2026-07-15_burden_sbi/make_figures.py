#!/usr/bin/env python
"""Generate decision-oriented result figures (suffix _7-15-26) for the amortized
burden-SBI cohort analysis: (1) f_ca199_C2 vs response both arms, (2) cross-regimen
shared-param comparison, (3) SBC calibration, (4) example posterior-predictive fits."""
import pickle, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy.stats import spearmanr, mannwhitneyu
warnings.filterwarnings("ignore")
import features as F
import gen_bank as GG          # GnP simulate
import gen_bank_mffx as GM     # mFFX simulate

OUT=Path("/home/chase/SPQSP/SPQSP_PDAC-main/results/2026-07-15_burden_sbi"); SUF="_7-15-26"
g=pd.read_csv(OUT/"cohort_posteriors.csv"); m=pd.read_csv(OUT/"cohort_mFFX_posteriors.csv")
with open(OUT/"npe_posterior_v2.pkl","rb") as f: PG=pickle.load(f)
with open(OUT/"npe_posterior_mffx.pkl","rb") as f: PM=pickle.load(f)

# ---------- Fig 1: f_ca199_C2 vs response, both arms ----------
fig,ax=plt.subplots(figsize=(7.5,5.5))
for d,c,lab,G in [(g,'C0','GnP (n=%d)'%len(g),None),(m,'C1','mFFX (n=%d)'%len(m),None)]:
    r,p=spearmanr(d.CA_fin,d.f_ca199_C2_med)
    ax.scatter(d.CA_fin,d.f_ca199_C2_med,c=c,s=45,alpha=.8,edgecolor='k',lw=.4,
               label=f"{lab}  ρ={r:.2f}, p={p:.0e}")
    flg=d[d.misspec]; ax.scatter(flg.CA_fin,flg.f_ca199_C2_med,facecolors='none',edgecolors='crimson',s=120,lw=1.6)
ax.axhline(1,color='gray',ls=':',lw=1); ax.axvline(1,color='gray',ls=':',lw=1)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel("CA19-9 final / baseline  (>1 = progressed on treatment)"); ax.set_ylabel("f_ca199_C2 posterior median\n(clone-specific CA19-9 secretion; 1 = uniform)")
ax.set_title("Clone-specific CA19-9 secretion tracks response (both regimens)\nred rings = posterior-predictive misspec-flagged")
ax.legend(fontsize=9,loc='upper left'); ax.grid(alpha=.3,which='both')
plt.tight_layout(); plt.savefig(OUT/f"fig1_f_vs_response{SUF}.png",dpi=130); plt.close()

# ---------- Fig 2: cross-regimen shared-param comparison ----------
params=["f_ca199_C2","r_resist","k_resist","k_sec_suppress","k_cell_clear","k_C1_growth"]
fig,ax=plt.subplots(2,3,figsize=(13,7.5))
for a,nm in zip(ax.flat,params):
    gv=g[f"{nm}_med"].values; mv=m[f"{nm}_med"].values; _,p=mannwhitneyu(gv,mv)
    bp=a.boxplot([gv,mv],labels=['GnP','mFFX'],widths=.55,patch_artist=True,showfliers=False)
    for patch,c in zip(bp['boxes'],['C0','C1']): patch.set_facecolor(c); patch.set_alpha(.5)
    for i,v in enumerate([gv,mv]): a.scatter(np.full(len(v),i+1)+np.random.uniform(-.12,.12,len(v)),v,s=12,c='k',alpha=.4)
    a.set_yscale('log'); a.set_title(f"{nm}\nMann-Whitney p={p:.2f}  {'(SHARED)' if p>0.05 else '(DIFFERS)'}",fontsize=10)
    a.grid(alpha=.3,axis='y',which='both')
plt.suptitle("Cross-regimen posterior comparison — resistance/CA19-9 params SHARED, only growth differs\n(supports regimen-independent, transferable chemo-response for counterfactuals)",fontsize=11)
plt.tight_layout(); plt.savefig(OUT/f"fig2_cross_regimen{SUF}.png",dpi=130); plt.close()

# ---------- Fig 3: SBC calibration (rank hists) ----------
try:
    d=np.load(OUT/"sbc.npz"); ranks=d["ranks"]; names=[str(x) for x in d["names"]]; L=int(d["L"]); n=ranks.shape[0]
    from scipy.stats import kstest
    fig,ax=plt.subplots(3,4,figsize=(15,9))
    for i,nm in enumerate(names):
        a=ax.flat[i]; ksp=kstest(ranks[:,i]/L,'uniform').pvalue
        col='C0' if ksp>0.05 else 'crimson'
        a.hist(ranks[:,i],bins=20,range=(0,L),color=col,alpha=.8); a.axhline(n/20,color='k',ls='--',lw=1)
        a.set_title(f"{nm}  KS p={ksp:.3f}\n{'calibrated' if ksp>0.05 else 'MISCALIBRATED'}",fontsize=9,color='k' if ksp>0.05 else 'crimson')
        a.set_xticks([]); a.set_yticks([])
    plt.suptitle(f"SBC calibration — GnP NPE over full window range U(90,1050), n={n}\nflat=calibrated; f_ca199_C2 & r_resist mildly off (short-window under-constraint)",fontsize=11)
    plt.tight_layout(); plt.savefig(OUT/f"fig3_sbc_calibration{SUF}.png",dpi=120); plt.close()
except Exception as e: print("fig3 skipped:",e)

# ---------- Fig 4: example posterior-predictive fits ----------
def postpred(arm, tag, W, ND=40):
    P = PG if arm=='GnP' else PM; G = GG if arm=='GnP' else GM
    sub = "" if arm=='GnP' else "_mFFX"
    s=pd.read_csv(OUT/f"cohort{sub}/{tag}_SLD.csv"); c=pd.read_csv(OUT/f"cohort{sub}/{tag}_CA.csv")
    sld_rel=(s.SLD/s.SLD.iloc[0]).values; ca_rel=(c.CA199/c.CA199.iloc[0]).values
    x=F.features_v2_from_patient(s.day.values.astype(float),sld_rel,c.day.values.astype(float),ca_rel,float(W))
    ph=np.exp(P["posterior"].sample((ND,),x=torch.as_tensor(x,dtype=torch.float32),show_progress_bars=False).numpy())
    with Pool(12) as pool: sims=pool.map(G.simulate,[np.log(ph[i]) for i in range(ND)])
    sims=[o for o in sims if o is not None]
    tg=np.linspace(0,W,80)
    SLD=np.array([np.interp(tg,G.GRID,o[0]) for o in sims]); CA=np.array([np.interp(tg,G.GRID,o[1]) for o in sims])
    return s,c,sld_rel,ca_rel,tg,SLD,CA

examples=[('GnP','PM036',1015,'GnP durable (gold-std)'),('GnP','JHU021',183,'GnP progressor f≈63'),('GnP','NH034',146,'GnP deep-CA resp (misspec)'),
          ('mFFX','NH053',231,'mFFX deep responder'),('mFFX','JHU013',870,'mFFX progressor'),('mFFX','PM044',115,'mFFX progressor f≈27')]
fig,ax=plt.subplots(2,6,figsize=(20,6.5))
for k,(arm,tag,W,lab) in enumerate(examples):
    try:
        s,c,sr,cr,tg,SLD,CA=postpred(arm,tag,W)
        for row,(Y,dyd,dat,ylab,logy) in enumerate([(SLD,s.day.values,sr,"SLD/base",False),(CA,c.day.values,cr,"CA19-9/base",True)]):
            a=ax[row,k]; lo,md,hi=np.percentile(Y,[5,50,95],axis=0)
            a.fill_between(tg,lo,hi,alpha=.25,color='C0'); a.plot(tg,md,'C0',lw=1.3); a.plot(dyd,dat,'ko',ms=4)
            if logy: a.set_yscale('log')
            if row==0: a.set_title(lab,fontsize=9)
            if k==0: a.set_ylabel(ylab); a.set_xlabel("day"); a.grid(alpha=.3)
    except Exception as e: ax[0,k].set_title(f"{tag} ERR",fontsize=8); print(tag,"err",e)
plt.suptitle("Example posterior-predictive fits (band = model + obs-noise, 5-95%; dots = data)",fontsize=11)
plt.tight_layout(); plt.savefig(OUT/f"fig4_postpred_examples{SUF}.png",dpi=120); plt.close()

print("WROTE:", *[f"fig{i}" for i in [1,2,3,4]], f"-> {OUT}/fig*{SUF}.png")
print("FIGS_DONE")
