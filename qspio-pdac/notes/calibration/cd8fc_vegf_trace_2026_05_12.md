# cd8_fc ↔ VEGF audit-degeneracy: not coupling, it's a CD8-expansion ceiling

**Date:** 2026-05-12
**Audit run:** `figures/multi_v2_nucleated_total/misspec_audit/`
**Branch:** main (post-#180)

## What the audit said

After the `nucleated_total_T` _pct_-target fix (PR #180), the dominant tensions
became:

- `gvax_nivo_neoadjuvant/cd8_fc_nonLA_gvax_nivo_d21` — **DEGENERACY**, z=−3.06,
  top driver `k_CD8_basal_exh`, paired with `baseline/vegf_tissue_concentration`
  (cosine −0.34, joint reach 44.33%).
- `gvax_neoadjuvant/cd8_fc_nonLA_gvax_d21` — DEGENERACY, z=−1.04, same VEGF pair
  (cosine −0.35, joint reach 45.43%).
- `baseline/vegf_tissue_concentration` regressed OK → MECHANISM_MISSING, z=+0.70.

Audit recommendation for DEGENERACY: "Extract a discriminating submodel target
to break the ridge."

## What the trace shows

Two single-knob sweeps against the gvax_nivo scenario, evaluated at t=21d, with
all other parameters held at submodel-prior medians (overlay of
`notes/calibration/submodel_priors.yaml` onto `resources/cpp/param_all.xml`;
median underflow for 14 tiny lognormals recovered from the `mu` field).

### k_vas_Msec sweep (M2 VEGF secretion, top positive VEGF driver)

| k_vas_Msec | V_T.VEGF(t=0) | V_T.VEGF(t=21) | CD8tot(t=0) | CD8tot(t=21) | **cd8_fc** | C1(t=21) |
|-----------:|--------------:|---------------:|------------:|-------------:|-----------:|---------:|
| 0.013      | 2.72e+04      | 2.73e+04       | 9.54e+04    | 1.71e+05     | **1.79**   | 8.92e+08 |
| 0.05       | 5.04e+04      | 5.11e+04       | 9.35e+04    | 1.71e+05     | **1.82**   | 8.94e+08 |
| 0.20       | 1.44e+05      | 1.47e+05       | 9.46e+04    | 1.75e+05     | **1.85**   | 9.08e+08 |

VEGF varies 5.3× across the sweep; **cd8_fc moves by 3% (1.79→1.85)**. No
mechanistic VEGF→CD8-expansion coupling.

### k_CD8_basal_exh sweep (top SBI partial correlate of cd8_fc)

| k_CD8_basal_exh | CD8(t=0) | CD8_exh(t=0) | CD8(t=21) | CD8_exh(t=21) | CD8tot(t=0) | CD8tot(t=21) | **cd8_fc** |
|----------------:|---------:|-------------:|----------:|--------------:|------------:|-------------:|-----------:|
| 0.01            | 7.82e+04 | 2.18e+04     | 1.52e+05  | 3.93e+04      | 1.00e+05    | 1.91e+05     | **1.91**   |
| 0.05            | 6.94e+04 | 2.41e+04     | 1.23e+05  | 4.71e+04      | 9.35e+04    | 1.71e+05     | **1.82**   |
| 0.25            | 4.39e+04 | 3.08e+04     | 6.29e+04  | 6.30e+04      | 7.47e+04    | 1.26e+05     | **1.69**   |

25× swing in `k_CD8_basal_exh` moves cd8_fc by 13%, in the *opposite* direction
of the SBI partial correlation (+0.168). Sign of the SBI marginal is noise; the
mechanistic direction is clearly negative (more basal exhaustion → fewer total
CD8 by t=21, because k_CD8_exh_death > k_CD8_death in the death-rate ratio).

## What's actually wrong

The cd8_fc target median is **5.98** (CI95 [2.7, 8.85]). The model produces
cd8_fc in the band **[1.69, 1.91]** across both probed axes. This is a
*structural ceiling* on CD8 fold-change in V_T over 21 days under GVAX+nivo —
not a parameter-ridge degeneracy with VEGF.

Adding a submodel target for `k_CD8_basal_exh` (or any other top driver) will
**not close this gap**. The audit's "DEGENERACY" label is misleading here:
joint reachability of 44% reflects that some prior draws *do* land near
cd8_fc=6, but the modal posterior θ cannot reach it.

## Likely real causes (untested)

In rough priority order:

1. **GVAX-driven CD8 priming too weak.** GVAX dose is `8.30e-16` (mol of
   irradiated tumor cells, scaled). The amplification chain GVAX → APC
   maturation → naive CD8 priming → effector CD8 expansion → trafficking to V_T
   has multiple Hill-saturating steps; if any is at its half-max plateau under
   the modeled dose, the expansion saturates regardless of upstream input.
   Worth checking: `H_mAPC`, `N_IL2_CD8`, `n_CD8_clones`, `EC50_GMCSF`,
   `k_GVAX_clearance`.

2. **Nivolumab → CD8_exh reversal mechanism.** Model's `H_PD1_C1` blocks killing
   but does not directly reverse CD8_exh → CD8. Pan-CD8 IHC counts both pools,
   so what nivolumab needs to do for cd8_fc is *prevent further exhaustion* +
   *survive longer*. If `k_CD8_exh_death` is large relative to `k_cell_clear`,
   the CD8_exh pool drains under nivolumab regardless.

3. **Trafficking ceiling.** `q_CD8_T_in` (transendothelial migration) and
   `H_pore_T` (pore-size exclusion) gate V_T arrival. Worth a sweep.

## Action items

- **Don't extract a `k_CD8_basal_exh` discriminating target.** That's what the
  audit DEGENERACY recommendation suggests but the trace says it won't close
  the gap.
- **Re-classify cd8_fc as MECHANISM_MISSING in our heads** even though the
  audit labels it DEGENERACY (joint reach 44% is misleadingly high for *target*
  feasibility — most of that reach is from cd8_fc-easy + VEGF-easy θs that
  don't co-occur with realistic post-treatment exhaustion ratios).
- **Free VEGF for direct calibration.** Baseline `vegf_tissue_concentration` is
  not actually entangled with cd8_fc at the parameter level; the VEGF gap can
  be addressed independently via `k_vas_Msec` / `k_vas_deg` widening.
- **Next investigative trace:** GVAX→CD8 priming chain. Sweep `EC50_GMCSF`,
  `N_IL2_CD8`, `n_CD8_clones`, and `q_CD8_T_in` to find which (if any) lifts
  cd8_fc above 2. If none does, the model lacks a sufficient CD8-expansion
  mechanism and needs structural work (e.g., more aggressive GVAX-induced
  priming, or a TCR-clonal-expansion burst term).

## Reproducing

```bash
TRACE=/tmp/cd8fc_vegf_trace
# Submodel-median overlay (recovers underflowed medians from mu)
.venv/bin/python <<'PY'
import yaml, pathlib, re, math
sm = yaml.safe_load(open("notes/calibration/submodel_priors.yaml"))
xml = pathlib.Path("resources/cpp/param_all.xml").read_text()
for entry in sm["parameters"]:
    name = entry["name"]; med = entry["marginal"]["median"]
    if med == 0.0 and "mu" in entry["marginal"]:
        med = math.exp(entry["marginal"]["mu"])
    xml = re.sub(rf"<{name}>.*?</{name}>", f"<{name}>{med}</{name}>", xml)
pathlib.Path(f"{TRACE}/param_base.xml").write_text(xml)
PY

# Sweep + run (see commit log)
```
