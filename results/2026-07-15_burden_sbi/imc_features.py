"""Per-patient IMC spatial/phenotype features from PASS01 neighborhood cells files.

Cell-level -> one row per patient. Feeds imc_join.py.
See PREREGISTRATION_imc_join.md for which of these are confirmatory vs exploratory.
"""
import glob
import os

import numpy as np
import pandas as pd

CELLS_DIR = "data/PASS01/neighborhoods"
MARKERS = ["Ki67", "PD1", "FOXP3", "MHC2", "MHC1", "GATA6", "SMAD4",
           "CLDN18.2", "CD44", "CD86", "CD40"]
# artifact/unclassified are segmentation leftovers, not biology -> excluded from
# composition denominators so fractions mean "of real cells".
REAL_LINEAGES = ["cancer", "immune", "fibroblast", "vasculature"]


def cells_files():
    return sorted(f for f in glob.glob(os.path.join(CELLS_DIR, "*_cells.csv"))
                  if not f.endswith("Zone.Identifier"))


def subject_of(path):
    return os.path.basename(path).replace("_cells.csv", "")


def nbhd_cols(df):
    return [c for c in df.columns
            if c.startswith("nbhd_") and c not in ("nbhd_total_K", "nbhd_n_neighbors")]


def patient_features(df):
    """Composition + marker + tumor-subtype features for one patient."""
    f = {}
    real = df[df.lineage.isin(REAL_LINEAGES)]
    f["n_cells_total"] = len(df)
    f["n_cells_real"] = len(real)
    f["frac_unclassified"] = float((df.lineage == "unclassified").mean())
    f["frac_artifact"] = float((df.lineage == "artifact").mean())

    # lineage fractions (of real cells)
    for lin in REAL_LINEAGES:
        f[f"lin_{lin}"] = float((real.lineage == lin).mean()) if len(real) else np.nan

    # cell_type fractions (of real cells)
    for ct in ["cancer", "fibroblast", "macrophage", "mdsc", "endothelial",
               "dendritic", "cd4_t", "cd8_t", "b_cell", "nk", "neutrophil",
               "other_immune", "t_other"]:
        f[f"ct_{ct}"] = float((real.cell_type == ct).mean()) if len(real) else np.nan

    ca = df[df.lineage == "cancer"]
    fib = df[df.lineage == "fibroblast"]
    mac = df[df.cell_type == "macrophage"]
    f["n_cancer"] = len(ca)
    f["n_fib"] = len(fib)

    # cancer state (NB: subtype=='proliferating' is identical to Ki67 -- see prereg)
    f["cancer_proliferating"] = float((ca.subtype == "proliferating").mean()) if len(ca) else np.nan
    f["cancer_quiescent"] = float((ca.subtype == "quiescent").mean()) if len(ca) else np.nan

    # tumor subtype (of cancer cells) -- H2/H3 features
    for ts in ["classical", "basal", "intermediate", "mixed"]:
        f[f"tumor_{ts}"] = float((ca.tumor_subtype == ts).mean()) if len(ca) else np.nan
    if len(ca):
        f["tumor_basal_intermediate"] = float(
            ca.tumor_subtype.isin(["basal", "intermediate"]).mean())
    else:
        f["tumor_basal_intermediate"] = np.nan

    # CAF classes (of fibroblasts) -- H4 features
    for st in ["myCAF", "iCAF", "activated_CAF", "fibroblast_other"]:
        f[f"caf_{st}"] = float((fib.subtype == st).mean()) if len(fib) else np.nan
    # myCAF as fraction of ALL real cells = stromal density proxy, not just CAF mix
    f["myCAF_of_real"] = (float((fib.subtype == "myCAF").sum() / len(real))
                          if len(real) else np.nan)

    # macrophage polarization (of macrophages)
    for st in ["m1", "m2", "macrophage_other"]:
        f[f"mac_{st}"] = float((mac.subtype == st).mean()) if len(mac) else np.nan

    # markers: mean positivity within cancer, and within all real cells
    for m in MARKERS:
        f[f"ca_{m}"] = float(ca[m].mean()) if len(ca) else np.nan
        f[f"all_{m}"] = float(real[m].mean()) if len(real) else np.nan

    # mean cancer-cell neighborhood composition = "what surrounds a cancer cell"
    if len(ca):
        for c in nbhd_cols(df):
            f[f"canbhd_{c[5:]}"] = float(ca[c].mean())
    return f


def build(out_csv="results/2026-07-15_burden_sbi/imc_patient_features.csv"):
    rows = []
    files = cells_files()
    for i, path in enumerate(files, 1):
        df = pd.read_csv(path)
        f = {"subject": subject_of(path)}
        f.update(patient_features(df))
        rows.append(f)
        if i % 20 == 0:
            print(f"  {i}/{len(files)}")
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}: {out.shape[0]} patients x {out.shape[1]-1} features")
    return out


if __name__ == "__main__":
    build()
