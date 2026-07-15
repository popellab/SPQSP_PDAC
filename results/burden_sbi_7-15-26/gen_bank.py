#!/usr/bin/env python
"""
Amortized burden-SBI: simulation-bank generator (feature-AGNOSTIC).

Samples theta from the burden priors, runs the FULL qsp_sim on a GnP template
(1050 d window, protocol dosing), and stores the rel-SLD and rel-CA19-9 TRAJECTORIES
on a canonical 14-day grid. One long-window GnP bank serves every GnP patient:
each patient's features are read downstream from the portion <= their 1st-line
window (causally correct -- dosing is identical protocol, post-window doses don't
affect earlier values).

12 params (v4 free set, INCLUDING f_ca199_C2 -- amortizing across patients is how
it becomes identifiable). Priors = the validated emcee lognormals (log-space).

Output: bank.npz  { theta_log (N,12), theta (N,12), sld_rel (N,G), ca_rel (N,G),
                    grid (G,), valid (N,), names (12,) }.
Chunked incremental save (survives interruption); re-run resumes.
"""
import os, re, sys, time, tempfile, subprocess
from pathlib import Path
import numpy as np, pandas as pd
from multiprocessing import Pool

Q    = Path("/home/chase/SPQSP/SPQSP_PDAC-main/qspio-pdac")
OUT  = Path("/home/chase/SPQSP/SPQSP_PDAC-main/results/burden_sbi_7-15-26")
SIM  = Q/"cpp/sim/build/qsp_sim"
BASE = (Q/"resources/cpp/param_all.xml").read_text()
DRUG = Q/"resources/cpp/drug_metadata.yaml"
HEAL = Q/"resources/cpp/healthy_state.yaml"
SCEN = OUT/"GnP_template.yaml"

# param, prior log-mean, prior log-sigma  (lognormal; identical to emcee v4)
PRIOR = [
    ("k_C1_growth",  np.log(0.02), 0.6),   # raised from log(0.01),0.7: prior median 0.01 sat on the
                                           # tumor-eradication viability cliff (~50% sims failed C_total->0).
                                           # 0.02 median covers the validated emcee posterior range 0.011-0.055.
    ("k_C_nabp",     np.log(0.06), 0.9),
    ("IC50_nabp",    np.log(92.0), 1.2),
    ("Kc_nabp",      np.log(8e7),  0.6),
    ("r_nabp",       np.log(1.5),  0.4),
    ("k_cell_clear", np.log(0.1),  0.7),
    ("k_ca199_shed", np.log(1e-3), 1.2),
    ("k_ca199_clear",np.log(0.5),  0.6),
    ("k_sec_suppress",np.log(20.0),1.5),
    ("k_resist",     np.log(1e-4), 1.5),
    ("r_resist",     np.log(100.0),1.0),
    ("f_ca199_C2",   np.log(10.0), 1.5),
]
NAMES = [p[0] for p in PRIOR]
MU    = np.array([p[1] for p in PRIOR])
SIG   = np.array([p[2] for p in PRIOR])
tag   = {n: re.compile(rf"(<{re.escape(n)}>)([^<]*)(</{re.escape(n)}>)") for n in NAMES}
GRID  = np.arange(0.0, 1050.0+1e-6, 14.0)   # canonical 14-day grid, 76 pts
G     = len(GRID)

def simulate(theta_log):
    phys = np.exp(theta_log)
    txt = BASE
    for n, v in zip(NAMES, phys):
        txt = tag[n].sub(rf"\g<1>{v:.8g}\g<3>", txt)
    with tempfile.TemporaryDirectory() as td:
        xml=os.path.join(td,"p.xml"); Path(xml).write_text(txt)
        csv=os.path.join(td,"o.csv")
        try:
            r=subprocess.run([str(SIM),"--param",xml,"--scenario",str(SCEN),
                "--drug-metadata",str(DRUG),"--evolve-to-diagnosis",str(HEAL),
                "--csv-out",csv,"--min-cadence-hours","336"],
                capture_output=True,text=True,timeout=180)
        except Exception:
            return None
        if r.returncode!=0 or "did not advance" in r.stderr or not os.path.exists(csv):
            return None
        try:
            df=pd.read_csv(csv); df.columns=[c.strip() for c in df.columns]
            t=df['Time'].values; C=df['C_total'].values; CA=df['V_C.CA199'].values
        except Exception:
            return None
        if len(t)<5 or C[0]<=0 or CA[0]<=0 or not np.all(np.isfinite(C)) or not np.all(np.isfinite(CA)):
            return None
        sld_rel = np.interp(GRID, t, (np.clip(C/C[0],1e-30,None))**(1/3.0))
        ca_rel  = np.interp(GRID, t, np.clip(CA/CA[0],1e-30,None))
        return sld_rel.astype(np.float32), ca_rel.astype(np.float32)

def worker(theta_log):
    out = simulate(theta_log)
    if out is None: return None
    return out

if __name__=="__main__":
    N  = int(os.environ.get("NBANK","10000"))
    NP = int(os.environ.get("NPROC","16"))
    CH = int(os.environ.get("CHUNK","1000"))
    rng = np.random.default_rng(2025)
    theta_log = MU + SIG*rng.standard_normal((N, len(NAMES)))   # (N,12) prior draws
    bankf = OUT/"bank.npz"
    done = 0
    if bankf.exists():
        d=np.load(bankf); done=int(d["done"]);
        theta_log[:done]=d["theta_log"][:done]  # keep prior draws consistent on resume
        sld=list(d["sld_rel"]); ca=list(d["ca_rel"]); valid=list(d["valid"])
        print(f"RESUME: {done}/{N} already done, {int(np.sum(valid))} valid")
    else:
        sld=[]; ca=[]; valid=[]
    t0=time.time()
    with Pool(NP) as pool:
        i=done
        while i<N:
            j=min(i+CH, N)
            res=pool.map(worker, list(theta_log[i:j]))
            for out in res:
                if out is None:
                    sld.append(np.full(G,np.nan,np.float32)); ca.append(np.full(G,np.nan,np.float32)); valid.append(False)
                else:
                    sld.append(out[0]); ca.append(out[1]); valid.append(True)
            i=j
            nv=int(np.sum(valid)); rate=(i)/(time.time()-t0)
            np.savez(bankf, theta_log=theta_log[:i], theta=np.exp(theta_log[:i]),
                     sld_rel=np.array(sld), ca_rel=np.array(ca), valid=np.array(valid),
                     grid=GRID, names=np.array(NAMES), done=i)
            print(f"  {i}/{N}  valid={nv} ({100*nv/i:.0f}%)  {rate:.1f} sim/s  eta {(N-i)/rate/60:.0f}min", flush=True)
    print(f"BANK_DONE N={N} valid={int(np.sum(valid))} in {(time.time()-t0)/60:.1f}min -> {bankf}")
