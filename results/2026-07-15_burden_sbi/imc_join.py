"""Join IMC spatial phenotypes to burden posteriors. Confirmatory tests only.

Runs EXACTLY the hypotheses fixed in PREREGISTRATION_imc_join.md. Exploratory sweep
lives in imc_explore.py so it cannot contaminate this file's output.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kruskal

RES = "results/2026-07-15_burden_sbi"
RHO_MIN = 0.3  # prereg: no claim below this regardless of p


def bh_fdr(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1.0)
    return out


def load():
    post = pd.read_csv(f"{RES}/imc49_pfs_posteriors.csv")
    feat = pd.read_csv(f"{RES}/imc_patient_features.csv")
    df = post.merge(feat, on="subject", how="inner")
    df["institution"] = df.subject.str.split("-").str[1]
    return df


def test(df, param, feature, direction, family, label):
    d = df[[param, feature]].dropna()
    if len(d) < 8:
        return None
    rho, p = spearmanr(d[param], d[feature])
    return dict(family=family, label=label, param=param, feature=feature,
                n=len(d), rho=rho, p=p, direction=direction)


# ---- pre-registered hypothesis families -------------------------------------
H2_FEATS = ["tumor_classical", "ca_GATA6", "ca_CLDN18.2"]
H3_FEATS = ["tumor_basal_intermediate", "ca_SMAD4"]
H4_FEATS = ["lin_fibroblast", "myCAF_of_real"]
H4_PARAMS = {"GnP": ["k_C_nabp_med", "IC50_nabp_med"],
             "mFFX": ["k_kill_fu5_med", "k_kill_oxa_med", "k_kill_iri_med",
                      "IC50_fu5_med", "IC50_oxa_med", "IC50_iri_med"]}


def run_primary(df, tag):
    rows = []
    # H1 -- orthogonal validation, directional (+)
    r = test(df, "k_C1_growth_med", "cancer_proliferating", "+", "H1",
             "growth <-> Ki67+ cancer frac")
    if r:
        rows.append(r)
    # H2 -- secretory phenotype, two-sided
    for f in H2_FEATS:
        r = test(df, "f_ca199_C2_med", f, "two-sided", "H2", f"f_ca199_C2 <-> {f}")
        if r:
            rows.append(r)
    # H3 -- resistance, two-sided
    for p_ in ["r_resist_med", "k_resist_med"]:
        for f in H3_FEATS:
            r = test(df, p_, f, "two-sided", "H3", f"{p_} <-> {f}")
            if r:
                rows.append(r)
    out = pd.DataFrame(rows)
    # BH within family
    out["q"] = np.nan
    for fam, g in out.groupby("family"):
        out.loc[g.index, "q"] = bh_fdr(g.p.values)

    # H4 -- arm-stratified, underpowered
    h4 = []
    for arm, params in H4_PARAMS.items():
        sub = df[df.arm == arm]
        for p_ in params:
            if p_ not in sub.columns:
                continue
            for f in H4_FEATS:
                d = sub[[p_, f]].dropna()
                if len(d) < 8:
                    continue
                rho, p = spearmanr(d[p_], d[f])
                exp = "-" if p_.startswith("k_kill") or p_.startswith("k_C_") else "+"
                h4.append(dict(family="H4", arm=arm, label=f"{arm}: {p_} <-> {f}",
                               param=p_, feature=f, n=len(d), rho=rho, p=p,
                               direction=exp))
    h4 = pd.DataFrame(h4)
    if len(h4):
        h4["q"] = np.nan
        for arm, g in h4.groupby("arm"):
            h4.loc[g.index, "q"] = bh_fdr(g.p.values)

    def show(t, frame):
        print(f"\n=== {t} [{tag}] ===")
        if not len(frame):
            print("  (none)")
            return
        for _, r in frame.iterrows():
            sig = "**" if (r.q < 0.05 and abs(r.rho) >= RHO_MIN) else "  "
            dirok = ""
            if r.direction in ("+", "-"):
                ok = (r.rho > 0) if r.direction == "+" else (r.rho < 0)
                dirok = " sign:OK" if ok else " sign:WRONG(pred %s)" % r.direction
            print(f"{sig} {r.label:52s} n={r.n:3d} rho={r.rho:+.3f} "
                  f"p={r.p:.4f} q={r.q:.4f}{dirok}")

    for fam in ["H1", "H2", "H3"]:
        show(fam, out[out.family == fam])
    show("H4 (ARM-STRATIFIED, UNDERPOWERED)", h4)
    return out, h4


def confounder_check(df, feats):
    print("\n=== CONFOUNDER: feature ~ institution (Kruskal-Wallis) ===")
    for f in feats:
        groups = [g[f].dropna().values for _, g in df.groupby("institution")
                  if g[f].notna().sum() >= 3]
        if len(groups) < 2:
            continue
        try:
            h, p = kruskal(*groups)
            flag = "  <-- BATCH RISK" if p < 0.05 else ""
            print(f"  {f:28s} p={p:.4f}{flag}")
        except ValueError:
            pass


if __name__ == "__main__":
    df = load()
    print(f"joined: {len(df)} patients | arms: {df.arm.value_counts().to_dict()}")
    print(f"misspec: {int(df.misspec.sum())}")
    clean = df[~df.misspec].copy()
    print(f"PRIMARY set (misspec excluded): n={len(clean)} "
          f"| arms {clean.arm.value_counts().to_dict()}")

    out, h4 = run_primary(clean, "PRIMARY, misspec excluded")
    out_s, h4_s = run_primary(df, "SENSITIVITY, all patients")

    confounder_check(df, ["cancer_proliferating", "tumor_classical",
                          "tumor_basal_intermediate", "ca_SMAD4",
                          "lin_fibroblast", "myCAF_of_real", "frac_unclassified"])

    pd.concat([out, h4]).to_csv(f"{RES}/imc_join_primary.csv", index=False)
    print(f"\nwrote {RES}/imc_join_primary.csv")
