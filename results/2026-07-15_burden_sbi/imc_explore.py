"""EXPLORATORY IMC x burden sweep + niche archetypes. NOT confirmatory.

Per PREREGISTRATION_imc_join.md: anything surviving here is a hypothesis for a future
cohort, not a result. Reported with BH-FDR across the ENTIRE sweep.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.dirname(__file__))
from imc_features import cells_files, nbhd_cols, subject_of  # noqa: E402
from imc_join import bh_fdr, load  # noqa: E402

RES = "results/2026-07-15_burden_sbi"
SHARED = ["k_C1_growth_med", "k_cell_clear_med", "k_ca199_shed_med",
          "k_ca199_clear_med", "k_sec_suppress_med", "k_resist_med",
          "r_resist_med", "f_ca199_C2_med"]


def sweep(df, feats, tag):
    rows = []
    for p in SHARED:
        for f in feats:
            d = df[[p, f]].dropna()
            if len(d) < 10 or d[f].nunique() < 5:
                continue
            rho, pv = spearmanr(d[p], d[f])
            rows.append(dict(param=p, feature=f, n=len(d), rho=rho, p=pv))
    out = pd.DataFrame(rows)
    out["q"] = bh_fdr(out.p.values)
    out = out.sort_values("p")
    print(f"\n=== EXPLORATORY SWEEP [{tag}] : {len(out)} tests ===")
    print(f"  survive BH-FDR q<0.05      : {(out.q < 0.05).sum()}")
    print(f"  survive q<0.10             : {(out.q < 0.10).sum()}")
    print(f"  raw p<0.05 (expect ~{0.05*len(out):.0f} by chance): {(out.p < 0.05).sum()}")
    print("\n  top 12 by raw p:")
    for _, r in out.head(12).iterrows():
        print(f"    {r.param:22s} {r.feature:28s} n={r.n:3d} "
              f"rho={r.rho:+.3f} p={r.p:.4f} q={r.q:.3f}")
    return out


def niche_archetypes(subjects, K_list=(4, 6, 8)):
    """Cluster per-cell nbhd_* vectors -> per-patient archetype abundance."""
    files = [f for f in cells_files() if subject_of(f) in subjects]
    print(f"\nloading {len(files)} cores for niche clustering...")
    samples, cols = [], None
    rng = np.random.default_rng(0)
    per_core = {}
    for f in files:
        d = pd.read_csv(f)
        d = d[d.lineage.isin(["cancer", "immune", "fibroblast", "vasculature"])]
        if cols is None:
            cols = nbhd_cols(d)
        X = d[cols].values.astype(np.float32)
        per_core[subject_of(f)] = X
        idx = rng.choice(len(X), size=min(2000, len(X)), replace=False)
        samples.append(X[idx])
    Xfit = np.vstack(samples)
    print(f"  fit matrix {Xfit.shape}")
    out = {}
    for K in K_list:
        km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(Xfit)
        sil = silhouette_score(Xfit[rng.choice(len(Xfit), 5000, replace=False)],
                              km.predict(Xfit[rng.choice(len(Xfit), 5000, replace=False)]))
        rows = []
        for s, X in per_core.items():
            lab = km.predict(X)
            r = {"subject": s}
            for k in range(K):
                r[f"niche{K}_{k}"] = float((lab == k).mean())
            rows.append(r)
        out[K] = pd.DataFrame(rows)
        print(f"  K={K}: silhouette={sil:.3f}")
    return out


if __name__ == "__main__":
    df = load()
    clean = df[~df.misspec].copy()
    featnames = [c for c in pd.read_csv(f"{RES}/imc_patient_features.csv").columns
                 if c not in ("subject",) and not c.startswith("n_")]
    sw = sweep(clean, featnames, "composition+markers, misspec excluded")
    sw.to_csv(f"{RES}/imc_explore_sweep.csv", index=False)

    niches = niche_archetypes(set(clean.subject))
    alln = []
    for K, nd in niches.items():
        m = clean.merge(nd, on="subject", how="inner")
        ncols = [c for c in nd.columns if c != "subject"]
        alln.append(sweep(m, ncols, f"NICHE K={K}"))
    pd.concat(alln).to_csv(f"{RES}/imc_explore_niches.csv", index=False)
    print(f"\nwrote {RES}/imc_explore_sweep.csv, {RES}/imc_explore_niches.csv")
