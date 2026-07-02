# Tumor Growth Parameter Analysis

**Date**: 2026-02-20
**Status**: Structural fixes implemented (2026-02-20). Awaiting smoke test validation.

## Implementation Summary (2026-02-20)

All five root causes identified below have been addressed:

| Change | Files Modified | Key Values |
|--------|---------------|------------|
| **Gompertzian growth** | `immune_oncology_model_PDAC.m:62` | `'model_type','Gompertzian'` |
| **K_initial fix** | `set_healthy_populations.m` | K = tumor_cells at init |
| **ECM-C_max moved + stromal support** | `gvax_module.m` (removed), `fibroblast_module_enhanced.m` (added) | `C_max = (K + k_stroma_support * aPSC_total) * (1 - ECM/(ECM + ECM_50_Kmax))` |
| **Glucose-dependent cancer proliferation** | `metabolism_module.m` | `H_glc_C = glc / (glc_50_C + glc)`, appended to growth reaction |
| **Parameter adjustments** | `parameters_PDAC.m`, `parameters/pdac_priors.csv` | k_C1_growth: 0.0066→0.012, f_ECM_50_Kmax: 0.2→0.5, new: glc_50_C=250 µM, k_stroma_support=1.0 |

**New files:**
- `scripts/smoke_test_growth.m` — Diagnostic script: applies median priors, runs evolve_to_diagnosis, plots 6 panels

**Prior CSV changes:**
- `k_C1_growth` median 0.012 (was 0.0066)
- `ECM_50_Kmax` → `f_ECM_50_Kmax` median 0.5 (was ECM_50_Kmax=300 mg/mL; now rule-derived so prior must be on the fraction)
- Added `glc_50_C` (250 µM, σ=0.5) and `k_stroma_support` (1.0, σ=0.7)

**Expected dynamics with median parameters (Gompertzian, k=0.012, k_death=0.003):**
- At C/C_max = 0.01: effective = 0.012 × 4.61 − 0.003 = 0.052/day (VDT ~13 days)
- At C/C_max = 0.50: effective = 0.012 × 0.69 − 0.003 = 0.005/day (VDT ~139 days)
- Late-stage VDT (~139 days) matches PDAC literature (100-200 days)

**Remaining work:**
- Run `scripts/smoke_test_growth.m` to validate dynamics
- Create pydantic CalibrationTarget/IsolatedSystemTarget YAMLs for new parameter values (literature validation)
- Bump MODEL_VERSION, clear simulation pool, re-run surrogate sensitivity

---

## Original Problem Statement

With median prior parameters, the model could not grow a tumor from the healthy state
(1e6 cells, ~1.7 mm diameter) to diagnosis size (3.2 cm) within the 10-year
simulation window. The `evolve_to_diagnosis` function failed (`success = false`)
for the median patient.

Diagnostic output (`scripts/debug_growth_rate.m`):
```
k_C1_growth = 6.60e-3 /day
k_C1_death  = 3.00e-3 /day
k_vas_growth       = 7.37 /day
net_growth   = 3.6e-3 /day  (doubling time = 193 days)
Max diameter reached in 10 years: 0.76 cm
```

---

## 1. Growth Model Architecture

### 1.1 Cancer Cell ODE (Gompertzian) — UPDATED

```
dC/dt = k_C1_growth * C * log(max(C_max/C, 1)) * H_glc_C - k_C1_death * C - immune_killing
```

Source: `core/modules/cancer_module.m:66` (Gompertzian selected via `'model_type','Gompertzian'`
in `immune_oncology_model_PDAC.m:62`). `H_glc_C` appended by `metabolism_module.m` at assembly time.

Previously used logistic growth `(1 - C/C_max)` which was too slow at small C/C_max.

### 1.2 Carrying Capacity Chain — UPDATED

```
C_max = (K + k_stroma_support * aPSC_total) * (1 - ECM_level / (ECM_level + ECM_50_Kmax))
                                                              [fibroblast_module_enhanced.m]
K     = vasculature species (dynamic ODE)                     [cancer_module.m:82]
aPSC_total = iCAF + myCAF + apCAF                             [fibroblast_module_enhanced.m]
ECM_50_Kmax = f_ECM_50_Kmax * ECM_max                         [fibroblast_module_enhanced.m]
```

Previously in `gvax_module.m` with `C_max = K * (1 - ECM/(ECM+ECM_50))`. Moved to
`fibroblast_module_enhanced.m` (where ECM dynamics are defined) and restructured to include
stromal support. CAFs provide nutrients/growth factors that partially compensate for poor
vasculature (Rhim 2014, Ozdemir 2014).

With f_ECM_50_Kmax = 0.5 and ECM_max = 1500: ECM_50_Kmax = 750 mg/mL.
At typical PDAC ECM_level (~300 mg/mL):
```
ECM penalty = 1 - 300/(300 + 750) = 0.71   (29% capacity reduction)
```

This is weaker than the old penalty (0.2→50% reduction at 300 mg/mL), consistent with
clinical evidence that stroma depletion worsens PDAC.

### 1.3 Vasculature Dynamics

```
dK/dt = k_vas_growth * C * VEGF/(VEGF + VEGF_50)           [growth]
      - k_vas_decay * K * (C * 2.57e-6)^(2/3)              [surface-area degradation]
```

VEGF at quasi-steady state:
```
VEGF_ss = k_vas_Csec * C / k_vas_deg = (0.004 / 16.6) * C ≈ 2.41e-4 * C  [pg/mL]
```

The degradation term uses `(C * vol_cell_mm3)^(2/3)` = tumor volume^(2/3),
i.e., proportional to tumor surface area. This is biologically motivated:
anti-angiogenic signals act at the tumor boundary.

### 1.4 V_T (Tumor Volume) Rule

After full module assembly, V_T includes contributions from:
- Cancer cells + dead cells (`vol_cell = 2572 um^3`)
- T cells, Th, CD8_exh (`vol_Tcell = 140 um^3`)
- Macrophages (`vol_Mcell`)
- NK cells (`vol_NK`)
- Fibroblasts: qPSC, iCAF, myCAF, apCAF (various vol)
- ECM (`ECM / ECM_density`)

Source: Rule 1 in `simbio_init.m:136`, modified by macrophage, fibroblast_enhanced,
and NK modules.

---

## 2. Biological Reality Check

### 2.1 What PDAC Cells Actually Do (Literature Values)

| Quantity | Value | Source |
|----------|-------|--------|
| **Apoptotic index (TUNEL)** | median 0.26%, mean 0.7% (range 0-1.8%) | Sharma 2005 (PMID 15841044); Virkajarvi 1998 (PMID 9839167) |
| **Ki-67 proliferative index** | median 30% (range 2-95%) | Pergolini 2019 (PMID 31592095, n=170) |
| **In vitro doubling time** | PANC-1: 52h; MiaPaCa-2: 40h; BxPC-3: 48-60h | Deer 2010 (PMC2860631) |
| **TUNEL-visible window** | 2-3 hours | Saraste & Pulkki 2000 (PMID 10728374) |
| **In vivo volume doubling time** | median 144 days (range 64-255); wider range 20-977 days | Aoki 2001 (PMID 11345136); Nakashima 2017 |
| **Microvessel density (MVD)** | PDAC 31.6/field vs. normal pancreas 112.8/field (ratio 0.28) | Ueda 2002 (PMID 11836587) |
| **HCC MVD for comparison** | 91.6/field (ratio ~1.4 vs. normal liver) | Ueda 2002 |

### 2.2 Deriving Death Rate from Apoptotic Index

The standard conversion (used in mathematical oncology) is:

```
k_death = AI / t_a
```

where `AI` = apoptotic index and `t_a` = duration of the TUNEL-detectable window.

With AI = 0.5% and t_a = 2-3 hours:
```
k_death_total ≈ 0.005 / (2.5/24 day) ≈ 0.048/day
```

This is the TOTAL death rate from ALL causes (intrinsic + immune + anoikis).
The model's `k_C1_death` represents only intrinsic death (immune killing is
computed separately). Even attributing just 10-25% of observed apoptosis to
intrinsic mechanisms gives k_C1_death = 0.005-0.012/day.

**The parameter audit's k_C1_death = 0.003/day is in the right ballpark —
and may even be conservative.** The old value of 0.0001/day (half-life 19 years)
implied essentially zero intrinsic death, which is not biologically defensible.

### 2.3 The Real Issue: k_C1_growth Is Too Low

If k_C1_death = 0.003/day is biologically reasonable (and 0.01-0.05/day is even
more defensible), then the question becomes: what should k_C1_growth be?

**Biological proliferation rate estimates:**

Ki-67 = 30% means 30% of cells are actively cycling. With a cell cycle time of
~24-48h in vivo:
```
k_birth ≈ Ki-67 / T_cycle = 0.30 / 1.5 day ≈ 0.20/day
```

This is the MAXIMAL in vivo birth rate — 30x higher than the current
k_C1_growth = 0.0066/day. But most of this proliferation is balanced by death
within the tumor, giving the observed slow net growth.

**The model uses "effective" rates:** k_C1_growth = 0.0066/day is NOT the actual
cell birth rate. It's an effective rate that already absorbs most of the
birth/death turnover into a single small number. This is a valid modeling
simplification — but only if the effective rates are internally consistent.

**The internal consistency problem:** When the parameter audit increased
k_C1_death from 0.0001 to 0.003/day, it consumed 83% of the available net
growth headroom (0.0036 remaining out of 0.0066). This leaves almost no margin
for:
- Logistic slowing near carrying capacity
- Immune-mediated killing (k_C_Tcell_eff)
- ECM carrying capacity penalty

The model essentially breaks because k_C1_growth was calibrated to work with
k_C1_death ≈ 0, and increasing death without increasing growth destroys the
balance.

### 2.4 Effective Rate Framework

For the logistic model to reproduce PDAC biology, we need the effective rates to
satisfy these constraints simultaneously:

**Constraint 1: Observed VDT at established tumors**
At diagnosis (C near C_max, immune pressure present):
```
k_growth * (1 - C/K * ecm_factor) - k_death - k_immune ≈ ln(2)/VDT
```
With VDT = 144 days → net ≈ 0.0048/day

**Constraint 2: Tumor reaches 3 cm from 1e6 cells within 10 years**
During exponential phase (C << K, immune minimal):
```
k_growth - k_death ≈ 0.005-0.010/day   (70-140 day doubling)
```

**Constraint 3: Immune killing has meaningful but not overwhelming effect**
The immune term should slow growth by ~10-30%, not dominate:
```
k_immune ≈ 0.1-0.3 * (k_growth - k_death)
```

**Constraint 4: Carrying capacity allows 3 cm tumors**
The vasculature K must support ~1e8-1e9 cancer cells at equilibrium.

Working backward from Constraint 2 with k_death = 0.003/day:
```
k_growth ≈ 0.008-0.013/day    (net = 0.005-0.010)
```

With k_death = 0.01/day (more biologically defensible):
```
k_growth ≈ 0.015-0.020/day    (net = 0.005-0.010)
```

---

## 3. Why Growth Stalls: Three Compounding Bottlenecks

### 3.1 Bottleneck 1: Initial Condition Mismatch (C >> K)

| Species | Initial Value | Source |
|---------|--------------|--------|
| C1 (cancer cells) | 1e6 cells | `set_healthy_populations` |
| K (vasculature) | 1.03e4 cells | K_initial in `parameters_PDAC.m:1262` |

At t=0, C/K ≈ 97. The logistic term `(1 - C/K) ≈ -96`, meaning the growth rate
is massively negative. Cancer cells crash rapidly until K catches up.

K grows fast (dK/dt ≈ 670k cells/day at t=0), so the crash is brief (~1-2 days),
but C may drop by an order of magnitude before recovering. This wastes simulation
time and sets the tumor back.

**This is unphysical**: if 1e6 cells exist, the vasculature must already support
them. K_initial should be at least as large as the initial cancer cell count, or the
initial C should be reduced to match K_initial.

### 3.2 Bottleneck 2: Insufficient Headroom in Net Growth

At small C relative to K (exponential phase):

```
dC/dt ≈ (k_C1_growth - k_C1_death - immune_kill_rate) * C
```

| Parameter | Old (pre-audit) | New (post-audit) | Need |
|-----------|----------------|-----------------|------|
| k_C1_growth | 0.0066/day | 0.0066/day | 0.008-0.020/day |
| k_C1_death | 0.0001/day | 0.003/day | 0.003-0.010/day |
| Net (no immune) | 0.0065/day | 0.0036/day | 0.005-0.010/day |
| Doubling time | 107 days | 193 days | 70-140 days |

The parameter audit increased k_C1_death 30x without compensating k_C1_growth.
Both the old and new parameterizations have problems:
- **Old**: k_C1_death = 0.0001 is biologically implausible (zero intrinsic death)
- **New**: net growth too slow to reach diagnosis within simulation window

### 3.3 Bottleneck 3: ECM Carrying Capacity Penalty

As the tumor grows, fibroblasts activate and ECM accumulates. The ECM constraint
`C_max = K * (1 - ECM/(ECM + ECM_50_Kmax))` progressively reduces the effective
carrying capacity.

If ECM_level reaches 300 mg/mL (= ECM_50_Kmax):
```
C_max = K * 0.5   (50% reduction)
```

The fraction of K filled by cancer cells becomes
`f * ecm_factor = 0.545 * 0.5 = 0.27`, meaning logistic slowing kicks in when
cancer cells are only 27% of vasculature capacity. This further reduces the
effective growth rate.

---

## 4. Parameter Provenance Audit

**None of the core growth parameters were calibrated to PDAC data.**

| Parameter | Value | Cited Source | Actual Cancer Type | Issue |
|-----------|-------|-------------|-------------------|-------|
| k_C1_growth | 0.0066/day | Ryu 2014 (PMID 24895040), Desai 2006 (PMID 16489089) | **Breast cancer** (Ryu); nab-paclitaxel PK in lung/breast/ovarian/prostate/colon xenografts (Desai) | Neither paper is PDAC-specific |
| k_C1_death | 0.003/day | Palsson 2013 (PMID 24074340) | **Generic** (FIRM model) | General immune response model, not cancer-type-specific |
| k_vas_growth | 7.37/day | Xiang 2014 (PMID 24700742) | **Hepatocellular carcinoma** | HCC is hyper-vascular (MVD ratio 1.4); PDAC is hypo-vascular (MVD ratio 0.28) |
| k_vas_decay | 4e-5/day | Xiang 2014 | HCC | Same concern |
| K_initial | 1.03e4 cells | Xiang 2014 | HCC | Calibrated for HCC xenografts |
| C_max | 2.7e4 cells | Desai 2006 | Mixed xenografts | Not PDAC |

**PDAC vs. HCC vasculature — opposite biology:**

| Feature | HCC | PDAC |
|---------|-----|------|
| Vascularity | Hyper-vascular | Hypo-vascular (MVD 0.28x normal) |
| Anti-VEGF response | Sorafenib/bevacizumab effective | No benefit from anti-angiogenics |
| Stroma | Minimal | Dense desmoplasia (60-80% stroma) |
| Growth mechanism | Angiogenesis-dependent | Metabolic adaptation (glycolysis, HIF-1α) |
| Drug delivery | Adequate | Poor perfusion, delivery barrier |

Using HCC vasculature parameters for PDAC is analogous to using lung compliance
parameters for liver — the organs have fundamentally different biology.

---

## 5. Candidate Fixes (Biological Assessment)

### Option A: Increase k_C1_growth to Match k_C1_death

**Change**: Increase k_C1_growth from 0.0066 to 0.010-0.015/day.

**Biological basis**: The current k_C1_growth was derived from breast cancer data.
PDAC-specific data suggests:
- In vivo VDT median 144 days → net rate 0.0048/day
- If k_C1_death = 0.003/day and immune killing adds ~0.001/day:
  - Exponential-phase net should be ~0.006-0.008/day (faster than observed VDT
    because observed VDT includes logistic slowing)
  - k_C1_growth = net + k_death = 0.009-0.011/day

This is a modest ~1.5-2x increase, well within uncertainty. The fact that the
existing k_C1_growth came from breast cancer (not PDAC) means there's no strong
reason to preserve the current value.

**Pro**: Directly addresses the headroom problem. Biologically motivated.
Preserves the biological realism of k_C1_death = 0.003/day.

**Con**: Changes a model parameter that has been used in prior work. Needs
re-calibration of everything downstream.

### Option B: Use PDAC-Derived Effective Parameters

**Change**: Re-derive both k_C1_growth and k_C1_death from PDAC-specific data.

From apoptotic index (intrinsic component only):
```
k_C1_death = 0.25 * AI_total / t_a = 0.25 * 0.005 / (2.5/24) ≈ 0.012/day
```
(assuming 25% of observed apoptosis is intrinsic; rest is immune/anoikis/hypoxia)

From observed VDT = 144 days (at established tumors where C ≈ 0.5*K):
```
net_at_diagnosis = 0.0048/day
k_growth * (1 - 0.5) - k_death - k_immune = 0.0048
k_growth * 0.5 = 0.0048 + 0.012 + 0.001 = 0.018
k_growth = 0.036/day
```

This gives much higher turnover: k_growth = 0.036, k_death = 0.012.
At small C (exponential phase): net = 0.036 - 0.012 = 0.024/day → VDT = 29 days.

That's fast, but remember: at small tumor sizes, growth IS faster (tumors are
well-vascularized, no hypoxia). The logistic slowing to ~144-day VDT at diagnosis
would be captured by the carrying capacity.

**Pro**: Most biologically faithful. High-turnover model correctly scales immune
killing relative to background death.
**Con**: Requires re-calibration of many coupled parameters. Immune killing
becomes a smaller fraction of total death, potentially reducing treatment effect
sizes. Need to verify stability of the ODE system with faster dynamics.

### Option C: Fix K_initial to Match Initial Cancer Cells

**Change**: Set K_initial to match initial C (currently 1e6 cells).

**Biological basis**: A small established tumor (1e6 cells) must have vascular
supply already supporting it. Setting K_initial = 1.03e4 while C0 = 1e6 is
self-contradictory — it implies the tumor is 97x above carrying capacity, which
would have already caused massive necrosis.

**Implementation**: In `set_healthy_populations.m`, set:
```
K_init = initial_tumor_cells * safety_factor  (e.g., 1.5x)
```
Or in `parameters_PDAC.m`, set `K_initial = 1e6`.

**Pro**: Necessary regardless of other changes. The initial state must be
self-consistent.
**Con**: Only fixes the initial transient, doesn't address long-term dynamics.

### Option D: Switch to Gompertzian Growth

**Change**: Set `model_type = 'Gompertzian'` in cancer_module call
(`scripts/immune_oncology_model_PDAC.m:62`).

Growth becomes:
```
dC/dt = k_C1_growth * C * log(max(C_max/C, 1))
```

**Biological basis**: Gompertzian growth is better-supported empirically for solid
tumors. The key difference from logistic:

| C/C_max | Logistic factor | Gompertz factor |
|---------|----------------|-----------------|
| 0.01 | 0.99 | 4.61 |
| 0.10 | 0.90 | 2.30 |
| 0.50 | 0.50 | 0.69 |
| 0.90 | 0.10 | 0.105 |
| 0.99 | 0.01 | 0.010 |

At low C/C_max (early tumor), Gompertz gives MUCH faster growth than logistic.
At C/C_max = 0.01, Gompertz is 4.7x faster. This dramatically accelerates the
early expansion phase when the tumor is small relative to carrying capacity.

At C/C_max = 0.5 (mid-growth), Gompertz is 1.4x faster. Near saturation, they
converge.

**Effect on the growth problem**: With Gompertzian growth and current parameters
(k_growth = 0.0066, k_death = 0.003):
- At C/C_max = 0.01: effective growth = 0.0066 * 4.61 - 0.003 = 0.027/day
  → VDT = 26 days (fast early growth)
- At C/C_max = 0.5: effective growth = 0.0066 * 0.69 - 0.003 = 0.0016/day
  → VDT = 433 days (slow late growth)

This matches PDAC natural history: fast early expansion, slowing at larger sizes.

**Pro**: Already implemented in code. Biologically better-supported. May solve
the growth timing problem without changing parameter values. Gives realistic
early-fast/late-slow dynamics.

**Con**: Changes the growth model type. Needs re-calibration of k_C1_growth
because the effective rate changes meaning. The prior for k_C1_growth would
need adjustment.

### Option E: Re-Calibrate Vasculature Parameters for PDAC

**Change**: Re-derive k_vas_growth, k_vas_decay, K_initial from PDAC-specific data.

**Biological basis**: Current values from HCC (Xiang 2014) are inappropriate.
PDAC-specific constraints:

- PDAC MVD is 0.28x normal pancreas (Ueda 2002)
- Anti-VEGF therapy does not benefit PDAC patients
- Tumor growth is supported by metabolic adaptation, not angiogenesis
- Solid stress from dense stroma is the primary physical growth constraint

**Approach**: Rather than fitting to vascular data (which is sparse for PDAC),
calibrate K dynamics so that K can support 1e8-1e9 cancer cells (needed for
3 cm tumor with 80% stroma). This is an output-constrained calibration.

**Pro**: Addresses the root cause (HCC parameters on PDAC model).
**Con**: Limited PDAC-specific vasculature growth rate data. May need to treat
as free SBI parameters.

### Option F: Decouple Carrying Capacity from Vasculature

**Change**: Replace vasculature-mediated C_max with a formulation appropriate
for PDAC's stroma-supported growth.

Possible formulations:
1. **Volume-based**: `C_max = C_max_density * V_T` (cancer cells per unit volume)
2. **Nutrient-diffusion**: `C_max = f(pO2, glucose)` (already partially in model
   via metabolism and hypoxia modules)
3. **Solid-stress**: `C_max = K_base * (1 - σ_solid/σ_yield)` (mechanical limit)

**Biological basis**: In PDAC, carrying capacity is limited by:
1. Nutrient diffusion through dense stroma (already in metabolism module)
2. Mechanical compression / solid stress
3. Metabolic waste (lactate, pH — already in metabolism module)
4. NOT vasculature (PDAC is hypo-vascular and anti-VEGF-resistant)

The model already has metabolism and hypoxia modules that impose growth penalties
(`H_glc_Teff`, `H_lac_Teff`, `HIF`). These may be double-counting with the
vasculature-mediated carrying capacity.

**Pro**: Most mechanistically honest for PDAC. Eliminates HCC-derived bottleneck.
**Con**: Largest structural change. Removes K dynamics that couple to VEGF and
anti-angiogenic therapy modules.

### Option G: Increase max_evolution_time

**Change**: Increase from 3650 to 7300 days (20 years) in `evolve_to_diagnosis.m`.

**Biological basis**: Genomic studies (Yachida 2010, PMID 20966255) estimate
~11.7 years from initiating mutation to metastatic capacity.

**Pro**: Biologically supported.
**Con**: Doesn't fix underlying parameter issues. Just extends runway.

---

## 6. Structural Assessment: Model Architecture for PDAC

The growth model was originally designed for a generic solid tumor (or possibly
HCC, given the parameter sources). Several structural elements are inappropriate
for PDAC biology.

### 6.1 Gap: Cancer Growth Has Zero Direct Metabolic Dependence

The metabolism module adds glucose depletion and lactate accumulation, but these
only affect **immune** function:

| Hill function | Mechanism | Affects cancer growth? |
|--------------|-----------|----------------------|
| `H_glc_Teff` | Glucose depletion → impaired T cell killing | No (affects immune killing, not proliferation) |
| `H_lac_Teff` | Lactate → impaired T/NK effector function | No |
| `H_lac_M2` | Lactate → M1→M2 polarization | No |
| `HIF` | Hypoxia → M1→M2, VEGF secretion | No |

Cancer cell proliferation has **no glucose or oxygen dependence whatsoever**. The
growth reaction rate is purely `k_growth * C * logistic_term` — cancer cells
grow at the same rate whether glucose is abundant or depleted to zero.

This is biologically wrong. PDAC cells are metabolically adapted (Warburg effect,
macropinocytosis, autophagy), but they still require nutrients for proliferation.
Ki-67 staining shows proliferation is lower in poorly perfused tumor regions.

**Consequence**: The model's ONLY direct limit on cancer proliferation is the
logistic/Gompertz term through C_max, which depends entirely on vasculature K.
This makes K the sole gatekeeper of tumor growth — a role that doesn't match
PDAC biology.

### 6.2 Problem: Vasculature-Mediated Carrying Capacity

The carrying capacity chain:
```
cancer_module.m:   C_max = K                              (default)
gvax_module.m:     C_max = K * (1 - ECM/(ECM + ECM_50))   (overrides default)
```

K is driven by VEGF-dependent angiogenesis:
```
dK/dt = k_vas_growth * C * VEGF/(VEGF+VEGF_50) - k_vas_decay * K * (C*vol_cell)^(2/3)
```

**This is a reasonable model for angiogenesis-dependent tumors** (HCC, RCC) where:
- More vasculature → more nutrients → more growth
- Anti-VEGF therapy works (sorafenib in HCC, bevacizumab in RCC)

**This is wrong for PDAC** where:
- Tumors are hypo-vascular (MVD 0.28x normal; Ueda 2002, PMID 11836587)
- Anti-VEGF therapy provides no survival benefit
- Growth depends on metabolic adaptation + stromal co-option, not angiogenesis
- The parameters (k_vas_growth, k_vas_decay, K_initial) come from HCC data (Xiang 2014)

However, K serves a necessary **mathematical** role: without it (or a substitute),
cancer growth is unbounded. The question is not whether to have a carrying
capacity, but what should drive it.

### 6.3 Problem: ECM Penalty on C_max — Direction and Magnitude

The GVAX module adds:
```
C_max = K * (1 - ECM_level / (ECM_level + ECM_50_Kmax))
```

With `f_ECM_50_Kmax = 0.2`, moderate desmoplasia (ECM at 20% of max) halves the
carrying capacity. This encodes the hypothesis: **more stroma → less growth**.

**The biology is more nuanced.** Stroma-depletion trials in PDAC showed the
opposite — removing stroma **accelerated** tumor growth:

- Rhim 2014 (PMID 25171127): SHH deletion in KPC mice → more aggressive PDAC,
  more metastasis, worse survival
- Ozdemir 2014 (PMID 25417103): αSMA+ myofibroblast depletion → worse survival,
  more aggressive tumors
- Clinical: Hedgehog inhibitors (vismodegib, saridegib) failed or worsened
  outcomes in PDAC trials

The stroma is **both barrier and support**:

| Barrier effects (current model captures) | Support effects (model ignores) |
|----------------------------------------|-------------------------------|
| Physical compression limits expansion | CAFs provide nutrients (glutamine, lipids) |
| Dense ECM limits perfusion | CAF-derived growth factors (HGF, FGF, IGF) |
| Poor drug delivery | Immune exclusion protects tumor from killing |
| | Metabolic symbiosis (CAF → lactate → cancer fuel) |

The net effect is context-dependent, but the clinical evidence strongly suggests
that in PDAC, stroma has a **net restraining effect that you would NOT want to
remove** (from the host perspective), while also providing survival support.

The current formulation (pure penalty, aggressive magnitude) is an
oversimplification that may prevent the model from reproducing PDAC's hallmark
phenotype: **high stromal fraction AND growing tumor**.

**Architectural issue**: This rule is in `gvax_module.m` (a therapy module),
not in `cancer_module.m` or `fibroblast_module_enhanced.m` where it logically
belongs. It only applies when the GVAX module is loaded.

### 6.4 Concern: Double-Counting of Growth Limitation

The model has multiple mechanisms that limit or slow tumor expansion:

1. **Logistic/Gompertz term** (C/C_max) — direct cancer growth limit
2. **ECM penalty on C_max** — reduces carrying capacity with desmoplasia
3. **Glucose depletion** (metabolism_module) — impairs immune killing
4. **Lactate accumulation** (metabolism_module) — impairs immune killing, M2 skewing
5. **HIF/hypoxia** (hypoxia_module) — M1→M2 polarization, VEGF feedback

Currently only #1-2 directly limit cancer proliferation, and #3-5 affect immune
function. If we add direct metabolic limitation of cancer growth (recommended
below), #3 would also limit cancer proliferation. This is not double-counting —
glucose limits BOTH cancer growth AND immune function through different
mechanisms — but the combined effect needs tuning to avoid over-constraining
growth.

### 6.5 What Should Drive Carrying Capacity in PDAC?

In PDAC, the factors that limit tumor size are:

1. **Nutrient availability** — glucose, glutamine, oxygen (diffusion-limited
   in dense stroma). Already partially modeled: glucose and pO2 exist as
   algebraic parameters. K determines glucose supply in the metabolism module.

2. **Mechanical compression** — solid stress from expanding tumor within
   confined tissue. Not currently modeled. Would require a solid mechanics
   formulation.

3. **Metabolic waste** — lactate, acidic pH. Already modeled as species
   (`V_T.lactate`) but only affects immune function, not cancer growth.

4. **Stromal co-option** — CAFs provide nutrients and growth signals that
   partially compensate for poor vasculature. Not captured in C_max.

5. **Vasculature** — still relevant (tumors need SOME blood supply) but
   much less limiting than in HCC/RCC. Over-weighted in current model.

---

## 7. Implementation Plan: Structural Changes — IMPLEMENTED

All changes below have been implemented (2026-02-20).

### Change 1: Switch to Gompertzian Growth ✅

**File**: `scripts/immune_oncology_model_PDAC.m:62`

```matlab
model = cancer_module(model, 'C1', params_out, 1, 'model_type', 'Gompertzian');
```

**Rationale**: Better empirical fit for solid tumors. Fast-early/slow-late
dynamics match PDAC natural history.

### Change 2: Add Direct Glucose Limitation of Cancer Proliferation ✅

**File**: `core/modules/metabolism_module.m`

Added `glc_50_C` parameter (250 µM, lower than glc_50_Teff=500 µM reflecting
Warburg adaptation), `H_glc_C` Hill function, and assembly-time append of
`* H_glc_C` to cancer growth reaction. Uses simple n=1 Hill (no cooperativity).

At normal glucose (1225 µM): H_glc_C = 0.83 (minimal limitation).
At severe depletion (100 µM): H_glc_C = 0.29 (strong limitation).

### Change 3: Restructure Carrying Capacity with Stromal Support ✅

**File**: `core/modules/fibroblast_module_enhanced.m` (added), `gvax_module.m` (removed)

Implemented Option A (K + stromal support):
```matlab
C_max = (V_T.K + k_stroma_support * aPSC_total) * (1 - ECM_level/(ECM_level + ECM_50_Kmax))
```

`k_stroma_support = 1.0`: each CAF adds 1 cell to carrying capacity. With ~1e6
CAFs at diagnosis, stromal support adds ~1e6 to C_max (comparable to K).

### Change 4: Move ECM-C_max Rule ✅

Moved from `gvax_module.m` (a therapy module) to `fibroblast_module_enhanced.m`
(where ECM dynamics and aPSC_total are defined). Also moved ECM_50_Kmax parameter
creation. ECM_50_Kmax is now rule-derived: `ECM_50_Kmax = f_ECM_50_Kmax * ECM_max`.

Weakened ECM penalty: f_ECM_50_Kmax 0.2→0.5 (ECM_50_Kmax = 750 vs. old 300 mg/mL).
Prior changed from direct `ECM_50_Kmax` to `f_ECM_50_Kmax` since the parameter is
now computed by a repeatedAssignment rule and can't be set via variant.

### Change 5: Fix K_initial Initial Condition ✅

**File**: `scripts/set_healthy_populations.m`

After setting cell populations, K is now set equal to tumor_cells:
```matlab
K_obj(1).InitialAmount = tumor_cells;  % K = C at initialization
```

Eliminates the unphysical C >> K initial transient (was C/K ≈ 97).

### Change 6: Adjust k_C1_growth ✅

**Files**: `parameters_PDAC.m`, `parameters/pdac_priors.csv`

k_C1_growth: 0.0066→0.012/day (also k_C2_growth updated to match).

With Gompertzian growth:
- At C/C_max = 0.01: effective = 0.012 × 4.61 − 0.003 = 0.052/day (VDT ~13 days)
- At C/C_max = 0.50: effective = 0.012 × 0.69 − 0.003 = 0.005/day (VDT ~139 days)

Late-stage VDT matches PDAC literature (100-200 days). Prior σ kept at 0.7.

---

## 8. Implementation Order — COMPLETED

1. ✅ **Parameters** — Added glc_50_C, k_stroma_support; updated f_ECM_50_Kmax, k_C1_growth
2. ✅ **Gompertzian growth** — Flag change in immune_oncology_model_PDAC.m
3. ✅ **K_initial fix** — set_healthy_populations.m
4. ✅ **Move ECM rule + stromal support** — gvax_module.m → fibroblast_module_enhanced.m
5. ✅ **Glucose-dependent cancer growth** — metabolism_module.m
6. ✅ **Update priors CSV** — pdac_priors.csv
7. ⬜ **Validate** — Run `scripts/smoke_test_growth.m` (evolve_to_diagnosis, VDT, densities)
8. ⬜ **Bump MODEL_VERSION, clear simulation pool, re-run SA**
9. ⬜ **Create CalibrationTarget/IsolatedSystemTarget YAMLs** for new parameter values

---

## 7. Impact on Existing Work

- **All cached simulations are invalid**: The 2000 simulations in the surrogate
  sensitivity pool were run with broken `evolve_to_diagnosis` (tumors reaching
  only ~0.2 cm instead of 3.2 cm). They must be discarded.

- **MODEL_VERSION must be bumped** after any parameter changes.

- **Surrogate sensitivity must be re-run** with corrected growth dynamics.

- **Calibration targets for tumor volume, cell densities, etc.** will become
  achievable once growth is fixed — they were likely showing R² = -1922 because
  simulated tumors were 15x too small.

---

## Appendix A: Equilibrium Analysis

At cancer cell equilibrium (no immune killing):
```
Logistic:   C/C_max = 1 - k_death/k_growth ≡ f
Gompertzian: C/C_max = exp(-k_death/k_growth)
```

| k_growth | k_death | f (logistic) | f (Gompertz) |
|----------|---------|-------------|-------------|
| 0.0066 | 0.003 | 0.545 | 0.636 |
| 0.0066 | 0.0001 | 0.985 | 0.985 |
| 0.010 | 0.003 | 0.700 | 0.741 |
| 0.015 | 0.003 | 0.800 | 0.819 |
| 0.036 | 0.012 | 0.667 | 0.716 |

Vasculature equilibrium values are very large (>1e12 cells) because the growth
term (linear in C) dominates degradation (C^(2/3)) at all scales. The practical
limit is NOT equilibrium but the TIME to reach diagnosis size, which depends on
the net growth rate during the exponential expansion phase.

## Appendix B: Gompertzian vs. Logistic Growth Dynamics

With current parameters (k_growth = 0.0066, k_death = 0.003, K_init = 1e6):

**Logistic**: At C = 1e4 (post-crash), C_max = 1e6:
```
growth_factor = 1 - 1e4/1e6 = 0.99
effective_growth = 0.0066 * 0.99 - 0.003 = 0.0035/day
VDT = 198 days
```

**Gompertzian**: At C = 1e4, C_max = 1e6:
```
growth_factor = log(1e6/1e4) = log(100) = 4.61
effective_growth = 0.0066 * 4.61 - 0.003 = 0.027/day
VDT = 26 days
```

The Gompertzian model gives 7.7x faster growth when C is far below C_max. This
dramatically accelerates the early expansion phase. As C approaches C_max, the
two models converge.

**Time to 3 cm (rough estimate, Gompertzian with current parameters):**
- Phase 1 (C = 1e4 → 1e6): ~9 doublings at ~30-day VDT = ~270 days
- Phase 2 (C = 1e6 → 1e8): ~7 doublings at ~50-100-day VDT = ~500 days
- Phase 3 (C = 1e8 → 1e9): ~3 doublings at ~200-day VDT = ~600 days
- Total: ~1,370 days ≈ 3.8 years (well within 10-year window)

This suggests Gompertzian growth may solve the timing problem without any
parameter value changes, by correctly modeling the fast-early/slow-late growth
dynamics of solid tumors.