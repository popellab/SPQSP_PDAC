# Structural Model Failures: Root Cause Analysis

**Date:** 2026-02-27
**Context:** The ablation study identified 3 observables that fail the prior predictive check (p<0.05) invariantly across all training configurations (noise on/off, 96/193 params). These are structural model problems, not calibration issues. This report documents the root cause of each failure, why qualitative bounds didn't catch them, and what the LOO predictive check reveals about joint misspecification.

---

## Why Qualitative Bounds Didn't Catch These

| Observable | Qual bound exists? | Why it missed |
|---|---|---|
| collagen1_protein_level | Yes — `ecm_fold_change_range_diagnosis` = [5, 20] | Bound is too loose. Both use Xu 2017 but the bound assumes wide uncertainty while the calibration target uses paired-specimen statistics giving CI95 [6.07, 16.00]. Model satisfies [5, 20] while failing the tighter target. |
| tumor_glucose_ratio | No — only absolute glucose bounds exist | Missing entirely. Bounds check absolute µM concentration and QSS algebraic consistency, not the tumor/normal ratio that matters clinically. |
| apcaf_fraction_of_caf | Yes — `apcaf_fraction_max` = 0.15 | Bound predates transcript-vs-protein distinction. Set from mouse flow cytometry (Elyada 2019), but the human calibration target spans [0.02, 0.21] after accounting for transcript-protein gap (MHC-II transcripts ~21% vs functional protein-level apCAFs ~2%, Ogawa 2021). |

### Qualitative bounds recommendations

1. **collagen1_protein_level**: Tighten bound to [6, 16] with updated rationale citing paired-specimen correlation
2. **tumor_glucose_ratio**: Add new bound `glucose_tumor_normal_ratio_range` = [0.15, 5.0] — currently missing
3. **apcaf_fraction_of_caf**: Update bound to [0.02, 0.21] to encompass transcript-vs-protein uncertainty

---

## LOO Predictive Check: The Joint Misspecification Story

The prior predictive p-values test each observable marginally. The LOO predictive check decomposes the joint Mahalanobis distance to identify which observables drive multivariate misspecification. These tell different stories:

| LOO rank | Observable | LOO influence (ΔD²) | Prior p-value | Interpretation |
|----------|-----------|---------------------|---------------|----------------|
| 1 | cdc2_density_baseline | 4.95 | 0.206 | Fine marginally, worst joint driver |
| 2 | cdc1_density_baseline | 4.06 | 0.354 | Fine marginally, 2nd worst joint driver |
| 3 | collagen1_protein_level | 3.59 | 0.000 | Bad both ways |
| 4 | apcaf_fraction_elyada2019 | 2.16 | 0.042 | Marginal failure + joint |
| 5 | icaf_fraction_elyada2019 | 1.87 | 0.587 | Fine marginally, bad jointly |
| 6 | cd4_helper_density | 1.77 | 0.143 | Fine marginally, bad jointly |
| 7 | treg_fraction_cd4 | 1.67 | 0.072 | Borderline marginal, bad jointly |
| 8 | tumor_glucose_ratio | 1.32 | 0.022 | Marginal failure + joint |
| 9 | th_exhausted_fraction | 1.05 | 0.170 | Fine marginally, moderate joint |

The top two LOO drivers — cdc2_density and cdc1_density — have perfectly acceptable prior predictive p-values (0.21 and 0.35). The model can produce each value individually but can't produce both simultaneously at the right values. The DC subtype ratio is structurally constrained: the APC module's cDC1/cDC2 split is governed by `f_cDC1`, so matching one density forces the other. This is a joint covariance problem invisible to marginal checks.

The top 3 LOO observables (cdc2, cdc1, collagen) account for 41.5% of total joint misspecification. The 3 prior predictive failures (collagen, glucose, apCAF) account for 23.3%. The difference represents the joint structure that marginal tests miss entirely.

---

## Failure 1: collagen1_protein_level

### Observable definition

Computed as ECM fold-change: `ECM_level / normal_ecm_conc`, where ECM_level = V_T.ECM / V_T (mg/mL) and normal_ecm_conc = 1.14 mg/mL (from Imamura 1995: 5 µg/mg dry × 0.22 dry fraction × 1.04 g/mL density).

**Calibration target:** Median 9.84, CI95 [6.07, 16.00] (Xu 2017, paired tumor/normal specimens).

**Simulation range:** [0.001, 9.43]. The observed 9.84 exceeds the simulation maximum — a hard structural ceiling.

### Equation structure

ECM production (fibroblast_module_enhanced.m, lines 482–496):

```matlab
% myCAF ECM secretion (dominant source):
rate = 'k_ECM_myCAF_sec * ((ECM_max - ECM_level) / ECM_max) * V_T.myCAF * (0.3 + 0.7*TGFb_Hill)'

% qPSC ECM secretion (minor):
rate = 'k_ECM_qpsc_sec * ((ECM_max - ECM_level) / ECM_max) * V_T.qPSC * (0.1 + 0.9*TGFb_Hill)'

% Degradation:
rate = 'k_ECM_deg * V_T.ECM'
```

### Root cause: logistic saturation ceiling

Both production reactions include `(ECM_max - ECM_level) / ECM_max`. This logistic saturation drives production to zero as ECM approaches ECM_max (100 mg/mL). At steady state the fold-change is bounded by the balance between saturating production and linear degradation. The model reaches equilibrium below the observed fold-change of 9.84.

At low ECM levels, the theoretical steady-state fold-change is approximately `k_ECM_myCAF_sec * myCAF / (k_ECM_deg * V_T * normal_ecm_conc)`. But as ECM accumulates, the saturation factor progressively throttles production, preventing the model from reaching this theoretical maximum.

### Key parameters

| Parameter | Value | Role |
|-----------|-------|------|
| k_ECM_myCAF_sec | 5e-9 mg/(cell·day) | Dominant production rate |
| k_ECM_qpsc_sec | 1e-9 mg/(cell·day) | Minor production rate |
| k_ECM_deg | 0.035 /day | Degradation rate |
| ECM_max | 100 mg/mL | Carrying capacity |
| ECM_base | 3 mg/mL | Normal pancreas reference |

### Fix options

1. **Widen ECM_max prior range** — the parameter may need to be higher, or the saturation could be relaxed. With untruncated priors this may resolve naturally.
2. **Reduce k_ECM_deg** — slower degradation allows higher steady-state. Currently at 0.035/day; the boundary piling data shows 15.4% lower-bound piling, suggesting the posterior wants lower values.
3. **Increase k_ECM_myCAF_sec** — the boundary piling data shows 20.5% upper-bound piling, confirming the model wants more ECM production.
4. **Structural change**: Replace `(ECM_max - ECM_level)/ECM_max` with a softer Hill-type saturation, e.g., `ECM_max^n / (ECM_max^n + ECM_level^n)` with n<1, allowing production to taper more gradually.

**Assessment:** Likely a parameter problem. The boundary piling on both k_ECM_myCAF_sec (upper) and k_ECM_deg (lower) indicates the model is trying to push toward higher collagen but is hitting prior bounds. With truncation removed, the next run should test whether the model can now reach the target.

---

## Failure 2: tumor_glucose_ratio

### Observable definition

Computed as `glc / glc_normal`, where glc is the tumor glucose concentration (µM, quasi-steady-state algebraic parameter) and glc_normal = 1225 µM (normal pancreas, from Battini 2017).

**Calibration target:** Median 0.924, CI95 [0.167, 5.05] (Battini 2017 HR-MAS NMR, chemo-naive PDAC).

**Simulation median:** 0.33. The model depletes glucose to ~1/3 of normal, but the observed value is ~92% of normal (barely depleted).

### Equation structure

Glucose is a quasi-steady-state algebraic parameter (metabolism_module.m, lines 98–100):

```matlab
glc = glc_normal * V_T.K / (V_T.K + demand * (1 + V_T.VEGF / VEGF_norm_50))

where:
  demand = k_glc_tumor * C_total + k_glc_immune * (T_total + M_total + aPSC_total + MDSC + NK)
```

This is a supply/demand balance:
- **Supply** (numerator): `V_T.K` — vasculature carrying capacity
- **Demand** (denominator): `V_T.K + (cell consumption) × (VEGF vascular dysfunction factor)`

### Root cause: demand overwhelms supply at diagnosis

At diagnosis, C_total is large (~10⁸ cells). The demand term `k_glc_tumor * C_total` dominates the denominator, driving glucose well below normal. The VEGF-dependent factor `(1 + VEGF/VEGF_norm_50)` further amplifies demand by modeling aberrant vasculature.

The glucose ratio simplifies to approximately:

```
glc/glc_normal ≈ K / (K + k_glc_tumor * C_total * VEGF_factor)
```

For this ratio to be ~0.92 (barely depleted), we need `K >> k_glc_tumor * C_total * VEGF_factor`. But at diagnosis with a large tumor, the demand term is comparable to or larger than K.

### Key parameters

| Parameter | Value | Role |
|-----------|-------|------|
| k_glc_tumor | 0.15 µM/(cell·day) | Cancer cell glucose consumption |
| k_glc_immune | 0.02 µM/(cell·day) | Immune cell glucose consumption |
| glc_normal | 1225 µM | Normal pancreas baseline |
| VEGF_norm_50 | 500 pg/mL | VEGF half-saturation for vascular dysfunction |

### Boundary piling evidence

VEGF_norm_50 shows 33.6% upper-bound piling (the worst of all parameters). The model wants VEGF_norm_50 to be much higher, which would reduce the VEGF amplification factor `(1 + VEGF/VEGF_norm_50)` toward 1, reducing effective glucose demand. This is consistent with the glucose over-depletion — the VEGF-driven vascular dysfunction term is too strong.

### Fix options

1. **Widen VEGF_norm_50 prior** — with truncation removed, the posterior can now explore higher values, reducing the VEGF amplification effect. This was the most boundary-piled parameter.
2. **Reduce k_glc_tumor** — lower per-cell glucose consumption. The current value may be too high for the cell densities at diagnosis.
3. **Structural change**: The QSS assumes all glucose consumption scales linearly with cell count. In reality, not all tumor cells are metabolically active (quiescent cores, necrotic regions). A correction factor for metabolically active fraction could help.
4. **Structural change**: Add a glucose supply term that scales with vasculature quality (pO2), not just carrying capacity K. Currently supply is purely `K`, but K represents total tumor volume capacity, not vascular delivery.

**Assessment:** Likely a combined parameter + structural problem. The VEGF_norm_50 boundary piling strongly suggests the VEGF amplification is too aggressive. Removing truncation may partially resolve this. But the QSS formula's linear scaling with C_total may also need structural revision for large tumors.

---

## Failure 3: apcaf_fraction_of_caf_elyada2019

### Observable definition

Computed as `V_T.apCAF / (V_T.apCAF + V_T.iCAF + V_T.myCAF)` (dimensionless fraction).

**Calibration target (Elyada 2019):** Median 0.103, CI95 [0.080, 0.126] (mouse KPC flow cytometry).
**Calibration target (human, deriv001):** Median 0.064, CI95 [0.020, 0.210] (scRNA-seq + IHC composite).

**Simulation median:** ~0.014 (1.4%). The model produces far fewer apCAFs than either target.

### Equation structure

apCAF dynamics (fibroblast_module_enhanced.m):

```matlab
% Production — constitutive qPSC activation (lines 405-408):
rate = 'k_PSC_to_apCAF * V_T.qPSC'

% Death — constitutive (lines 474-477):
rate = 'k_apCAF_death * V_T.apCAF'
```

Compare to iCAF and myCAF, which have regulated activation:

```matlab
% iCAF: TGFb-inhibited, PDGF-dependent, IL-6 feedback
rate = 'k_PSC_to_iCAF * V_T.qPSC * (1 - TGFb_Hill) * PDGF_Hill * IL6_feedback'

% myCAF: TGFb-dependent, PDGF-dependent
rate = 'k_PSC_to_myCAF * V_T.qPSC * TGFb_Hill * PDGF_Hill'
```

### Root cause: constitutive low activation with no regulatory feedback

Three problems compound:

1. **No regulation**: apCAF activation is purely constitutive (`k_PSC_to_apCAF * qPSC`), unlike iCAF/myCAF which are TGFb/PDGF/IL-6 regulated. There is no mechanism for the TME to upregulate apCAF production.

2. **High death:activation ratio**: k_apCAF_death (0.025/day) / k_PSC_to_apCAF (0.01/day) = 2.5. At steady state, apCAF = 0.4 × qPSC. But qPSC is being rapidly drained by iCAF/myCAF activation (which have higher regulated rates in the TGFb/PDGF-rich TME), leaving little substrate.

3. **No plasticity**: iCAF and myCAF interconvert (k_iCAF_to_myCAF, k_myCAF_to_iCAF), but there is no pathway from iCAF or myCAF to apCAF. The one-way qPSC→apCAF path is the only source.

### Key parameters

| Parameter | Value | Role |
|-----------|-------|------|
| k_PSC_to_apCAF | 0.01 /day | Only source of apCAF production |
| k_apCAF_death | 0.025 /day | Constitutive death |
| k_PSC_to_iCAF | 0.2 /day | Competing qPSC drain (20x faster) |
| k_PSC_to_myCAF | 0.4 /day | Competing qPSC drain (40x faster) |

### Boundary piling evidence

k_PSC_to_apCAF shows 22.4% upper-bound piling — the model wants faster apCAF activation than the prior allows. With truncation removed, this may partially resolve.

### Fix options

1. **Parameter adjustment**: Increase k_PSC_to_apCAF and/or decrease k_apCAF_death. With truncation removed, the posterior can explore this direction.
2. **Structural change — add regulation**: Make apCAF activation depend on antigen availability or IFNg (evidence suggests MHC-II expression is IFNg-inducible). E.g., `k_PSC_to_apCAF * qPSC * H_IFNg_apCAF`.
3. **Structural change — add plasticity**: Add iCAF→apCAF or myCAF→apCAF conversion pathway. There is some evidence (Hutton 2021) that mesenchymal-to-immune CAF transitions occur.
4. **Simplest structural fix**: Just increase the constitutive rate. The parameter is currently 0.01/day vs 0.2–0.4/day for iCAF/myCAF. Increasing to 0.05/day and reducing k_apCAF_death to 0.015/day would give apCAF_ss ≈ 3.3 × qPSC, likely bringing the fraction into the target range.

**Assessment:** Primarily a parameter problem, but the lack of regulation is a structural limitation that prevents the TME from driving apCAF production in response to immune signals. Parameter adjustment (removing truncation on k_PSC_to_apCAF) may be sufficient in the short term, but adding IFNg-dependent regulation would be more mechanistically grounded.

---

## Cross-Cutting Observations

### The LOO top drivers (cDC1/cDC2 densities) are a separate structural issue

The two worst LOO observables — cdc1_density and cdc2_density — don't fail the prior predictive individually, but they drive 30% of joint misspecification. This is because the APC module's cDC1/cDC2 ratio is governed by a single parameter (`f_cDC1`), creating a hard constraint: matching cDC1 density forces a specific cDC2 density. If both calibration targets demand values that violate this ratio constraint, no parameter combination can satisfy both simultaneously. This is a separate structural issue from the 3 prior predictive failures, and may require independent cDC1/cDC2 recruitment or maturation pathways.

### Removing prior truncation may resolve 2 of 3 failures

The boundary piling evidence shows the model was already trying to push in the right direction for collagen (k_ECM_myCAF_sec upper, k_ECM_deg lower) and apCAF fraction (k_PSC_to_apCAF upper). With truncation removed from pdac_priors.csv, the next SBI run should test whether the model can now reach these targets. The glucose ratio is more likely to need a structural fix (VEGF amplification term is too aggressive).

### All 3 failures are in newer modules

The collagen, glucose, and apCAF observables all come from modules added in the PDAC extension (fibroblast_module_enhanced.m, metabolism_module.m). The core immune modules (T cells, macrophages, checkpoint) don't have prior predictive failures. This suggests the newer modules need more parameter calibration rather than fundamental equation redesign.

---

## Files

| File | Contents |
|------|----------|
| `core/modules/fibroblast_module_enhanced.m` | ECM ODE (lines 482–496), apCAF dynamics (lines 405–408, 474–477) |
| `core/modules/metabolism_module.m` | Glucose QSS rule (lines 98–100) |
| `parameters/parameters_PDAC.m` | All parameter values referenced above |
| `calibration_targets/baseline_no_treatment/collagen1_protein_level_PDAC_deriv001.yaml` | Collagen calibration target |
| `calibration_targets/baseline_no_treatment/tumor_glucose_ratio_PDAC_deriv001.yaml` | Glucose ratio calibration target |
| `calibration_targets/baseline_no_treatment/apcaf_fraction_of_caf_PDAC_deriv001.yaml` | apCAF fraction calibration target |
| `tests/qualitative_bounds.yaml` | Qualitative bounds definitions |
| `figures/sbi_baseline_allparams/diagnostics/loo_predictive.csv` | LOO misspecification decomposition |