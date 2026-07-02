# ECM Module Design

Date: 2025-03-14

## Motivation

ECM is the dominant immune evasion mechanism in PDAC — more important than checkpoint ligands. The current model treats ECM as a single bulk concentration feeding generic Hill functions. This document designs a more mechanistic ECM module that captures the distinct physical properties through which ECM affects tumor-immune dynamics, and that will couple cleanly to a future agent-based model (ABM).

---

## 1. Biological properties of ECM in PDAC

### 1.1 Composition

PDAC stroma is primarily:
- **Fibrillar collagen** (types I and III) — provides structural rigidity, ~60-80% of ECM dry mass
- **Hyaluronic acid (HA)** — glycosaminoglycan that attracts water, drives interstitial fluid pressure
- **Fibronectin, laminin** — adhesion/signaling substrates (lower mass contribution)

The two dominant components (collagen and HA) have distinct biophysical effects and are produced by different cell types, so they should be tracked separately.

### 1.2 Physical properties and what they affect

| Property | Depends on | Biological effect | Current model? |
|----------|-----------|-------------------|----------------|
| **Fiber density** | Collagen concentration | Pore size between fibers → immune cell exclusion | Partially (ECM_level as proxy) |
| **Pore size** | Fiber density, architecture | T cells need ~7 μm pores; DCs more deformable (~5 μm) | No (implicit in ECM_level Hill) |
| **Stiffness** (Young's modulus) | Collagen density + crosslinking | Fibroblast activation (YAP/TAZ), cancer EMT, PD-L1 upregulation | No |
| **Crosslinking** | LOX enzyme activity | Increases stiffness without changing mass; irreversible | No |
| **Interstitial fluid pressure** (IFP) | HA content, vascular leakiness | Drug delivery barrier, vascular collapse | No |
| **Cytokine sequestration** | Collagen + HA binding sites | TGFβ stored in latent form; released by MMPs or strain | No |

### 1.3 Why each property matters for immunotherapy prediction

**Pore size / fiber density** — This is the primary immune exclusion mechanism. T cells physically cannot squeeze through dense collagen. In the current model, `H_ECM_T_mot` and `H_ECM_APC_mig` are proxies for this, but they use bulk concentration rather than a pore-size-derived quantity. For ABM coupling, pore size is the natural variable for migration rules.

**Stiffness** — Creates a positive feedback loop: stiff ECM → activates more fibroblasts → more ECM → stiffer. This is the mechanotransduction loop missing from the current model (CAF carrying capacity is cancer-cell-coupled, not ECM-coupled). Stiffness also upregulates PD-L1 on cancer cells (Paszek 2005, Rubiano 2018) and drives T cell exhaustion via chronic mechanosensing.

**Crosslinking** — LOX-mediated crosslinking is elevated in PDAC and correlates with poor prognosis (Miller 2015). It increases stiffness disproportionately to collagen mass, meaning two tumors with the same collagen content can have very different mechanical properties. Important for the stiffness feedback loop.

**IFP** — PDAC has the highest IFP of any solid tumor (75-130 mmHg vs ~10-20 in most cancers). HA is the primary driver. High IFP prevents antibody penetration (checkpoint inhibitors, GM-CSF). This could explain why GVAX (local injection, no systemic delivery barrier) works better than systemic immunotherapy in PDAC.

**Cytokine sequestration** — ECM binds latent TGFβ, VEGF, PDGF, and chemokines. This means:
- Local TGFβ release depends on MMP activity, not just secretion rate
- Therapy-induced MMP activity can release sequestered TGFβ (immunosuppressive burst)
- Current model has TGFβ as freely diffusing — misses the ECM reservoir

---

## 2. Proposed state variables

### Primary (ODE tracked)

| Variable | Units | Source | Degradation |
|----------|-------|--------|-------------|
| `collagen` | mg | myCAF > qPSC secretion (TGFβ-dependent) | MMP degradation (constitutive + immune-dependent) |
| `HA` | mg | iCAF + cancer cell secretion (HAS2) | Hyaluronidase (constitutive) |
| `LOX` | nM | myCAF + cancer cell secretion (hypoxia-enhanced) | First-order degradation |
| `TGFb_latent` | nmol | Bound to ECM collagen; released by MMPs or mechanical strain | Release → active TGFβ pool |

### Derived (repeatedAssignment rules)

| Variable | Formula | Units | Range |
|----------|---------|-------|-------|
| `phi_collagen` | `V_T.collagen / (V_T × rho_collagen)` | dimensionless | 0–1 |
| `d_pore` | `d_pore_ref × (phi_col_ref / phi_collagen)^0.5` | μm | ~3–20 μm |
| `stiffness` | `E_ref × (phi_collagen / phi_col_ref)^n_stiff × (1 + α_LOX × LOX_crosslinked)` | kPa | ~1–15 kPa |
| `IFP` | `IFP_base + β_HA × HA_conc` | mmHg | ~10–130 mmHg |
| `f_ECM` | `(V_T.collagen/rho_collagen + V_T.HA/rho_HA) / V_T` | dimensionless | 0–1 |

### Hill functions (inputs matched to mechanism)

| Effect | Input | Hill function | Rationale |
|--------|-------|---------------|-----------|
| T cell exclusion | `d_pore` | `H_pore_T = 1 / (1 + (d_pore/d_Tcell_nuc)^n)` | Physical: pore < nuclear diameter → excluded |
| APC exclusion | `d_pore` | `H_pore_APC = 1 / (1 + (d_pore/d_APC_nuc)^n)` | DCs more deformable; d_APC < d_T |
| T cell exhaustion | `stiffness` | `H_stiff_Texh = stiffness / (E_50_Texh + stiffness)` | Chronic mechanosensing, PD-L1 |
| Fibroblast activation | `stiffness` | `H_stiff_fib = stiffness / (E_50_fib + stiffness)` | YAP/TAZ mechanotransduction |
| Carrying capacity | `f_ECM` | `1 - f_ECM / (f_ECM + f_ECM_50_Kmax)` | Volume competition |
| Drug delivery | `IFP` | `H_IFP = IFP / (IFP_50 + IFP)` | Convective transport barrier |
| Secretion saturation | `phi_collagen` | `1 - phi_collagen` | Physical packing limit |
| TGFβ sequestration | `phi_collagen` | Mass-action binding/release | Reservoir dynamics |

---

## 3. Coupled carrying capacities

Three logistic-type growth limits interact through collagen:

### 3.1 Cancer carrying capacity (C_max)

```
C_max = (V_T.K + k_stroma_support × aPSC_total) × (1 - phi_collagen/(phi_collagen + phi_col_50_Kmax))
```

Cancer growth is *supported* by CAFs (nutrients, growth factors → additive term) but *inhibited* by the collagen those same CAFs deposit (volume competition → multiplicative barrier). This is the stroma-depletion paradox: removing CAFs eliminates both the support and the barrier, and the net effect is worse outcomes (Rhim 2014, Ozdemir 2014).

### 3.2 CAF carrying capacity (K_stroma)

```
K_stroma = f_stroma_max × C_total + K_stiff_niche × H_stiff_fib
```

Two terms:
- **Cancer-dependent**: paracrine signaling (PDGF, etc.) — CAFs exist because cancer recruits them
- **Stiffness-dependent**: mechanotransduction niche — stiff ECM sustains CAFs via YAP/TAZ even without cancer cells. This is new and enables post-therapy fibrotic scar persistence.

### 3.3 Collagen secretion saturation

```
secretion_rate × (1 - phi_collagen)
```

Physical packing limit: production → 0 as collagen fills available volume. Unlike the current `(ECM_max - ECM_level)/ECM_max` which never drops below ~0.75, this actually engages because `phi_collagen` can approach 1.

### 3.4 Coupling diagram

```
Cancer ──PDGF──→ CAFs (K_stroma term 1: paracrine)
CAFs ──collagen secretion──→ phi_collagen ↑
  phi_collagen ↑ → stiffness ↑ → CAF niche ↑     (+ feedback, K_stroma term 2)
  phi_collagen ↑ → (1-phi_collagen) ↓ → secretion slows  (- feedback, bounds the system)
  phi_collagen ↑ → C_max ↓                        (- feedback, volume competition)
  phi_collagen ↑ → pore size ↓ → immune exclusion ↑
  phi_collagen ↑ → V_T ↑ → phi_collagen diluted   (- feedback, volume expansion)
CAFs ──k_stroma_support──→ C_max ↑                 (+ feedback, nutrient provision)
```

Two negative feedback loops (secretion saturation, volume dilution) bound the positive feedback loop (stiffness → CAF activation → more collagen). SBI resolves where the equilibrium lands for each virtual patient.

---

## 4. Additional feedback loops this enables

### 4.1 Stiffness-fibroblast positive feedback (currently missing)
```
myCAF → collagen + LOX → stiffness ↑ → H_stiff_fib ↑ → qPSC activation ↑ → more myCAF
```
This replaces the current cancer-cell-only CAF carrying capacity with a mechanotransduction-coupled feedback. CAFs persist in stiff ECM even if cancer cells are depleted (post-therapy fibrotic scar).

### 4.2 TGFβ sequestration-release (currently missing)
```
myCAF → TGFβ_active → binds collagen → TGFβ_latent (stored)
MMP activity (M1 mac, activated T cells) → TGFβ_latent → TGFβ_active (released)
```
This means therapy-induced inflammation releases sequestered TGFβ, creating an immunosuppressive rebound. Could explain why initial immune responses in PDAC often stall.

### 4.3 Immune-dependent ECM remodeling (currently missing)
```
GVAX → M1 polarization → MMP secretion → collagen degradation → pore size ↑ → T cell infiltration ↑
```
But also:
```
MMP activity → TGFβ release → myCAF activation → collagen re-deposition
```
This creates a tug-of-war: therapy opens the stroma, but the released TGFβ triggers re-fibrosis. The net effect depends on the balance, which SBI can resolve.

### 4.4 HA-IFP-drug delivery (currently missing)
```
iCAF + cancer → HA secretion → IFP ↑ → antibody penetration ↓
```
Relevant for systemic checkpoint inhibitors. Less relevant for GVAX (injected locally). But matters for GVAX + nivo combination scenarios.

---

## 5. What to include vs defer

### Include now (high impact, tractable)

| Feature | Why now | Complexity |
|---------|---------|------------|
| **Collagen as primary state variable** (rename ECM → collagen) | Direct replacement, cleaner semantics | Low |
| **phi_collagen as dimensionless fraction** | Fixes ECM_max/ECM_density redundancy | Low |
| **Pore-size-derived immune exclusion** | Mechanistic replacement for ad-hoc H_ECM_T_mot/H_ECM_APC_mig | Medium |
| **Stiffness** (collagen-density-derived, no crosslinking yet) | Enables fibroblast mechanotransduction feedback | Medium |
| **Immune-dependent ECM degradation** | M1 macrophage MMP term; needed for treatment scenarios | Low |
| **Secretion saturation via phi_collagen** | Fixes inert (ECM_max - ECM_level)/ECM_max term | Low |

### Defer (important but lower priority or data-limited)

| Feature | Why defer | When to add |
|---------|-----------|-------------|
| **HA as separate species** | Adds 3+ parameters; IFP mainly matters for systemic drug delivery | When modeling systemic checkpoint inhibitor delivery |
| **LOX / crosslinking** | Limited dose-response data for PDAC; stiffness tracks density reasonably without it | When stiffness-density relationship proves insufficient |
| **TGFβ sequestration** | Major mechanistic addition (latent/active pools, MMP-dependent release); high parameter uncertainty | After confirming TGFβ dynamics matter for treatment predictions |
| **IFP** | Requires HA; mainly affects drug PK, not immune cell dynamics directly | With HA, when modeling systemic delivery |

---

## 6. Proposed implementation

### 6.1 Rename ECM → collagen

- State variable: `V_T.collagen` (mg)
- `rho_collagen` — collagen protein packing density within extracellular space (material constant, 0.025 g/mL = 25 mg/mL). Replaces `ECM_density`.
- `phi_collagen` — collagen volume fraction (dimensionless, 0–1). Defined directly: `phi_collagen = V_T.collagen / (V_T × rho_collagen)`. This is the primary input for all ECM-dependent Hill functions.
- No intermediate "collagen_level" or "collagen concentration" variable — `phi_collagen` is computed directly from the state variable and the material constant.

### 6.2 Derived pore size

```matlab
% Pore size decreases with collagen density (Sabeh 2009, Wolf 2013)
% d_pore ~ 1/sqrt(collagen_concentration) relationship from fiber network theory
d_pore = addparameter(model, 'd_pore', 0, 'ValueUnits', 'micrometer', 'ConstantValue', false);
addrule(model, 'd_pore = d_pore_ref * (phi_col_ref / max(phi_collagen, 1e-6))^0.5', 'repeatedAssignment');
```

Reference values:
- Normal pancreas: collagen ~3 mg/mL → phi_collagen ~0.12 → d_pore ~15-20 μm (permissive)
- PDAC diagnosis: collagen ~10 mg/mL → phi_collagen ~0.40 → d_pore ~7-9 μm (marginal for T cells)
- Severe desmoplasia: collagen ~20 mg/mL → phi_collagen ~0.80 → d_pore ~4-5 μm (excludes T cells)

### 6.3 Pore-size-based immune exclusion

```matlab
% T cell exclusion: pore size relative to nuclear diameter (~7 μm)
% When d_pore >> d_nuc: H → 0 (no exclusion). When d_pore << d_nuc: H → 1 (full exclusion).
addrule(model, 'H_pore_T = 1 / (1 + (d_pore / d_Tcell_nuc)^n_pore)', 'repeatedAssignment');

% APC exclusion: DCs are more deformable (podosomes, MMP-dependent migration)
% Effective nuclear diameter ~5 μm (vs 7 μm for T cells)
addrule(model, 'H_pore_APC = 1 / (1 + (d_pore / d_APC_nuc)^n_pore)', 'repeatedAssignment');
```

These replace `H_ECM_T_mot` and `H_ECM_APC_mig`. The `n_pore` Hill coefficient controls transition sharpness (n=2-4: gradual; n=6+: sharp cutoff at nuclear diameter).

### 6.4 Stiffness (density-derived, no crosslinking)

```matlab
% Tissue stiffness scales as power law of collagen concentration
% E ~ c^n where n ≈ 1.5-2.0 for collagen gels (Latinovic 2010, Motte & Bhatt 2013)
% Normal pancreas ~1 kPa, PDAC ~2-10 kPa (Rubiano 2018)
addrule(model, 'stiffness = E_ref * (phi_collagen / phi_col_ref)^n_stiff', 'repeatedAssignment');
```

Used for:
- Fibroblast activation: `H_stiff_fib = stiffness / (E_50_fib + stiffness)` (replaces cancer-cell-only CAF carrying capacity)
- T cell exhaustion: `H_stiff_Texh = stiffness / (E_50_Texh + stiffness)` (replaces `H_ECM_T_exh`)

### 6.5 Immune-dependent ECM degradation

```matlab
% Current: k_ECM_deg * V_T.collagen (constitutive only)
% Proposed: add M1 macrophage MMP contribution
rate = 'k_ECM_deg * V_T.collagen + k_MMP_M1 * V_T.M1 * V_T.collagen / V_T';
```

This allows GVAX-induced M1 polarization to open the stroma, enabling T cell infiltration — a key therapeutic mechanism in PDAC.

### 6.6 CAF carrying capacity: add stiffness-dependent term

```matlab
% Current: K_stroma = f_stroma_max * C_total (cancer-dependent only)
% Proposed: add stiffness-dependent niche capacity
% CAFs persist in stiff ECM even if cancer cells are depleted
K_stroma = f_stroma_max * C_total + K_stiff_niche * H_stiff_fib
```

`K_stiff_niche` represents the maximum CAF population that stiff ECM alone can sustain (without cancer cell support). This allows fibrotic scar persistence post-therapy.

---

## 7. Parameter inventory

### New parameters needed

| Parameter | Estimated value | Units | Source |
|-----------|----------------|-------|--------|
| `d_pore_ref` | 15 | μm | Normal pancreas pore size (Wolf 2013) |
| `phi_col_ref` | 0.12 | dimensionless | Normal pancreas collagen fraction |
| `d_Tcell_nuc` | 7 | μm | T cell nuclear diameter (Wolf 2013) |
| `d_APC_nuc` | 5 | μm | DC effective diameter (deformable, podosomes) |
| `n_pore` | 4 | dimensionless | Hill coefficient for pore exclusion sharpness |
| `E_ref` | 1.0 | kPa | Normal pancreas stiffness (Rubiano 2018) |
| `n_stiff` | 1.8 | dimensionless | Stiffness-concentration power law exponent |
| `E_50_fib` | 5.0 | kPa | Half-max stiffness for fibroblast activation (Liu 2015) |
| `E_50_Texh` | 4.0 | kPa | Half-max stiffness for T cell exhaustion |
| `k_MMP_M1` | TBD | mL/(cell·day) | M1 macrophage MMP-mediated collagen degradation |
| `K_stiff_niche` | TBD | cell | Max CAF population sustained by stiff ECM alone |

### Parameters removed

| Parameter | Replacement |
|-----------|-------------|
| `ECM_max` | Removed — `phi_collagen` naturally bounded [0,1] by volume feedback |
| `ECM_50_T_exh` | → `E_50_Texh` (stiffness-based) |
| `ECM_50_T_mot` | → `d_Tcell_nuc` + `n_pore` (pore-size-based) |
| `ECM_50_APC_mig` | → `d_APC_nuc` + `n_pore` (pore-size-based) |
| `ECM_50_Kmax` | → `f_ECM_50_Kmax` (fraction of phi_collagen directly) |
| `f_ECM_50_T_exh` | → `E_50_Texh` (stiffness-based) |
| `f_ECM_50_T_mot` | → pore size parameters |
| `f_ECM_50_APC_mig` | → pore size parameters |

### Parameters retained (renamed)

| Old | New | Notes |
|-----|-----|-------|
| `ECM` (species) | `collagen` | Clearer semantics |
| `ECM_density` | `rho_collagen` | Collagen packing density, material constant (0.025 g/mL) |
| `ECM_base` | `collagen_base` | Normal pancreas baseline (3 mg/mL) |
| `ECM_level` | Removed | Replaced by `phi_collagen` (= mass / (V_T × rho_collagen)); no intermediate concentration variable |
| `k_ECM_qpsc_sec` | `k_col_qpsc_sec` | Same value |
| `k_ECM_myCAF_sec` | `k_col_myCAF_sec` | Same value |
| `k_ECM_deg` | `k_col_deg` | Constitutive component retained |
| `f_ECM_50_Kmax` | `phi_col_50_Kmax` | Recalibrate: now fraction of phi_collagen directly (not fraction of a separate ceiling) |

---

## 8. ABM coupling points

When coupling to a spatial ABM, the ODE model provides:

| ODE output | ABM usage |
|-----------|-----------|
| `phi_collagen` (bulk average) | Initialize spatial collagen field; ABM adds local heterogeneity |
| `d_pore` (bulk average) | ABM computes local pore size from local collagen; ODE provides population-level constraint |
| `stiffness` (bulk average) | ABM local mechanics for cell migration rules |
| Collagen production/degradation rates | ABM uses same rate constants for local dynamics |
| MMP secretion rates | ABM applies locally around M1 macrophages |

The key design principle: **ODE state variables should be quantities the ABM also tracks** (masses, concentrations), not derived quantities (stiffness, pore size). The ABM will compute its own spatially resolved versions of derived quantities using the same constitutive relationships.

---

## 9. Current model issues (for reference)

### Issue 1: ECM_max vs ECM_density redundancy
Resolved by this redesign — `phi_collagen` bounded [0,1] by volume feedback; single ceiling.

### Issue 2: ECM_level dilution during tumor growth
Partially resolved: pore-size-based exclusion depends on `phi_collagen`, which has the same dilution behavior as `ECM_level`. However, this is now physically correct — if cancer cells grow faster than ECM, the collagen volume fraction really does decrease, and pore size really does increase. The fix is ensuring secretion rates are calibrated so desmoplasia increases during tumor growth (as observed in PDAC), not changing the state variable definition.

### Issue 3: CAF carrying capacity cancer-cell-coupled only
Resolved by adding stiffness-dependent niche term (`K_stiff_niche * H_stiff_fib`).

### Issue 4: Constitutive-only ECM degradation
Resolved by adding immune-dependent MMP term (`k_MMP_M1 * M1 * collagen`).

---

## 10. Calibration strategy

The ECM/CAF submodel has tightly coupled parameters — secretion rates, proliferation rates, carrying capacities, and constitutive relationships all interact. Individual-parameter SubmodelTargets give marginal estimates that may be inconsistent when combined. Multi-parameter SubmodelTargets constrain the *joint* parameter space by fitting coupled ODE dynamics to multi-output data.

### 10.1 Calibration levels

| Level | What it constrains | Example |
|-------|-------------------|---------|
| **Single-parameter SubmodelTarget** | One EC50, rate constant, or material property | TGFb_50_APC from DC dose-response |
| **Multi-parameter SubmodelTarget** | Coupled submodel dynamics (2-5 params jointly) | CAF proliferation + collagen deposition time course |
| **Full-model CalibrationTarget** | Emergent in vivo observables | CD8 density at diagnosis, stromal fraction |

The multi-parameter level fills the gap between individual parameters and full-model calibration. The `SubmodelTarget` schema already supports this: the submodel can be a coupled ODE system with multiple outputs, and SBI learns correlated posteriors over the parameter set.

### 10.2 Priority submodel calibration targets

#### Target 1: Pore size → immune cell exclusion

**Status: EXTRACTED (partially).** Two SubmodelTargets from Wolf 2013:
- `d_pore_ref_phi_col_ref` — 4-point power-law fit (bovine dermal, Fig 1D). d_pore_ref contraction=0.48, phi_col_ref=-0.13 (degenerate).
- `d_crit_T_n_pore` — 3-point Hill fit (polycarbonate filters, Fig 7A). d_crit_T contraction=0.97, n_pore=0.86.

**NOT extracted:** T cell migration speed in 3D collagen gels (Fig 5D/6A). This was originally planned to jointly constrain d_pore_ref, phi_col_ref, n_pore, but analysis showed:
- T blast speed data is in **rat tail** collagen (Fig 5D), which has a different pore-size vs concentration relationship than the bovine dermal collagen used in the pore-size target
- Fig 6A consolidates speed vs **pore cross-section** (not concentration), which constrains d_crit_T/n_pore (already well-constrained) but bypasses d_pore_ref/phi_col_ref
- Bovine dermal speed data (Fig 2A) is for HT/MT1 tumor cells, not T cells

**phi_col_ref remains weakly constrained.** Breaking this degeneracy requires data linking collagen *concentration* to a functional outcome through the d_pore_ref/phi_col_ref power law in a single collagen type. Candidates:
- PDAC tissue measurements mapping collagen volume fraction to measured pore sizes
- Stiffness vs collagen data (Target 2 below) which shares phi_col_ref

#### Target 2: Stiffness vs collagen concentration (constitutive relationship)

**Status: EXTRACTED.** Two SubmodelTargets from Rubiano 2018 (DOI: 10.1016/j.actbio.2017.11.037):
- `stiffness_collagen_power_law` — power-law fit using collagen-only gel SSM (0.8 kPa at 3 mg/mL). Jointly constrains E_ref, n_stiff, phi_col_ref.
- `E_ref` — identity target from normal pancreas SSM (1.06 ± 0.25 kPa). Contraction=0.80.

Joint inference (12 targets, 14 params): phi_col_ref contraction improved from -0.13 to 0.69 by sharing across pore-size and stiffness targets. E_ref contraction=0.80, n_stiff=0.34.

#### Target 3: CAF + collagen coupled dynamics

**Status: LITERATURE REVIEWED, NOT YET EXTRACTABLE.**

Papers reviewed:
- **Biffi 2019** (DOI: 10.1158/2159-8290.CD-18-0710): Defines iCAF/myCAF biology and TGFb/IL-1 antagonism. Data is almost entirely qualitative (qPCR fold-changes, Western blots). No quantitative kinetics extractable for rate constants. Best used as mechanistic justification for model structure.
- **Shek 2002** (DOI: 10.1016/S0002-9440(10)61125-X): TGFb increases PSC collagen synthesis by 34%; neutralizing TGFb drops to 44% of baseline. Data in % of control (³H-proline incorporation), not absolute rates. Constrains TGFb-dependent fraction (model uses 30% basal / 70% TGFb; data suggests 44% basal / 56% TGFb).
- **Rolfs 2021** (DOI: 10.1038/s41467-021-26842-3): Protein turnover atlas — no pancreas tissue. Not useful for k_col_deg directly.

**Collagen secretion rate (k_col_myCAF_sec) and basal fraction (f_col_basal):**

**Status: EXTRACTED.** Two SubmodelTargets from Illsley 2000 (DOI: 10.1054/bjoc.2000.1321):
- `k_col_myCAF_sec`: Human dermal fibroblasts, HPLC OHpro, 39.78 pmol/1e5 cells/h basal rate. Joint inference median = 11.6 pg/cell/day (pulled up from prior 5 pg). Contraction=0.68.
- `f_col_basal`: TGFb fold-change 1.31-1.91x constrains 1/f_basal. Joint inference median = 0.60 (vs old hardcoded 0.30). Contraction=0.34.

Model changes: Removed qPSC secretion (negligible). Replaced hardcoded 0.3/0.7 with new parameter f_col_basal. Added f_col_basal to pdac_priors.csv.

Additional papers reviewed but not extracted:
- Schwarz 1985 (DOI: JBC 260:3045): Procollagen secretion t_half = 20 min (fast) / 120 min (slow). Mechanistic detail on secretion pathway but no absolute per-cell rates.
- Schwarz 2015 (DOI: 10.1016/j.bbrep.2015.07.007): Review — PAT cells produce >50% of protein as procollagen when fully induced, ~10% for average fibroblast. No absolute pg/cell/day.
- Frahs 2018 (DOI: 10.1007/s10439-018-2064-0): NIH-3T3 in 3D collagen, mass spec distinguishes new vs old collagen. Time course at 1/2/3 weeks. Could give absolute rate if Fig 7A digitized.
- Clark & Hilderbran 1984: ~1.7 pg/cell/day (fetal rat). LeRoy 1974: scleroderma 4x. Both superseded by Illsley.

**Collagen degradation rate (k_col_deg):**
- Decaris 2015 (DOI: 10.1371/journal.pone.0123311): Hepatic collagen FSR 0.2-0.6%/day → t_half 120-350 days → k_deg 0.002-0.006/day. Human, in vivo, stable isotope labeling. Best available. PDF in hand.
- Model value (0.007/day) is at the fast end of the liver range. Appropriate for PDAC with active remodeling.
- **NEXT PRIORITY: Extract from Decaris 2015.** Forward model: identity (FSR = k_col_deg at steady state). Translation sigma for liver→pancreas, normal→PDAC.

**CAF interconversion rates (k_iCAF_to_myCAF, k_myCAF_to_iCAF):**
- Biffi 2019: Qualitative only. No extractable kinetics.
- Elyada 2019 (DOI: 10.1158/2159-8290.CD-19-0094): KPC tumors iCAF 44.4±3.9%, myCAF 45.3±4.1%, apCAF 10.3±1.25%. Would constrain steady-state ratio.
- General fibroblast literature: myofibroblast differentiation timescale ~24-48h under TGFb.
- **Strategy:** Elyada for ratio + differentiation timescale for absolute rate. Need Elyada PDF.

**Key insight:** Reparameterizing from hardcoded splits to explicit parameters (f_col_basal) enabled direct calibration from literature. The same approach should be applied to other hardcoded values in the model.

#### Target 3a: Collagen secretion — DONE

See above. k_col_myCAF_sec and f_col_basal extracted from Illsley 2000.

#### Target 3b: Collagen degradation — DONE

**Data:** Decaris 2015 — hepatic collagen type I/III FSR in human chronic liver disease (n=6).
FSR 0.2-0.6%/day. Prior derivation: median=0.0038/day (t_half ~181 days), contraction=0.99.
Pulls model from 0.007 toward 0.004/day — collagen turns over slower than originally parameterized.

#### Target 3c: CAF subtype ratios — UNBLOCKED (2025-03-18)

**Data:** Elyada 2019 — flow cytometry of KPC tumors. iCAF:myCAF:apCAF = 44:45:10. PDF needed.

**Previously blocked** on `(1-H_TGFb)` proxy for iCAF formation, which forced near-zero
myCAF→iCAF conversion when TGFb is high. Now resolved:

**Model change implemented:** IL-1alpha added as explicit species (`V_T.IL1`). iCAF formation
driven by `H_IL1_eff = IL1 / (IL1_50 * (1 + TGFb/TGFb_50_IL1R1) + IL1)`, where TGFb
antagonizes by shifting EC50 rightward (IL-1R1 downregulation). Replaces `(1-H_TGFb)` in
qPSC→iCAF activation and myCAF→iCAF interconversion.

**TGFb_50 split completed.** Generic TGFb_50 fully replaced in fibroblast module:
- `TGFb_50_CAF_act`: PSC activation + interconversion
- `TGFb_50_col_sec`: Collagen secretion
- `TGFb_50_IL1R1`: TGFb antagonism of IL-1R1

**Next step:** Extract Elyada 2019 ratio target. Need PDF.

#### Target 4: Stromal fraction distribution at diagnosis (equilibrium constraint)

#### Target 4: Stromal fraction distribution at diagnosis (equilibrium constraint)

**Data:** Torphy et al. 2018 (n=66 autopsy, median stromal fraction 0.74), Vendittelli et al. 2024 (n=62, median 0.85), Ogawa et al. 2021 (combined ~72%).

**Submodel:** Full CAF + collagen dynamics run to equilibrium (or to diagnosis time point via evolve_to_diagnosis).

**Output:** `phi_collagen + cell_volume_fraction_CAF` compared to reported stromal fraction.

**Parameters constrained:** `f_stroma_max`, `phi_col_50_Kmax`, `k_stroma_support` (jointly)

**Why multi-parameter:** Stromal fraction at diagnosis is the equilibrium outcome of cancer growth, CAF carrying capacity, and collagen dynamics all interacting. No single parameter determines it.

#### Target 5: Immune-dependent ECM degradation (treatment response)

**Data:** Paired pre/post biopsies from neoadjuvant therapy showing delta in collagen content + immune infiltration. FOLFIRINOX data is most available; immunotherapy data from Zheng 2022 if stromal fraction was measured.

**Submodel:**
```
d_collagen/dt = secretion - (k_col_deg + k_MMP_M1 × M1/V_T) × collagen
```

**Parameters constrained:** `k_MMP_M1` (primary), ratio of immune-dependent to constitutive degradation

**Why submodel level:** The delta in collagen content depends on both the baseline degradation rate and the immune-dependent term — measuring the change under known M1 perturbation separates the two contributions.

### 10.3 Calibration sequence

```
1. Material constants — DONE
   └─ rho_collagen = 25 mg/mL (fixed conversion constant, not estimated)

2. Constitutive relationships — DONE (7 params, contraction 0.34-0.97)
   ├─ Pore size: Wolf 2013 → d_pore_ref (0.53), phi_col_ref (0.69), d_crit_T (0.97), n_pore (0.85)
   └─ Stiffness: Rubiano 2018 → E_ref (0.80), n_stiff (0.34), phi_col_ref (shared, 0.69)

3. Dynamic rates — IN PROGRESS
   ├─ k_col_myCAF_sec: DONE (Illsley 2000, 11.6 pg/cell/day, contraction 0.68)
   ├─ f_col_basal: DONE (Illsley 2000 TGFb fold-change, median 0.60, contraction 0.34)
   ├─ k_col_deg: NEXT (Decaris 2015 hepatic FSR, PDF in hand)
   └─ CAF interconversion: Elyada 2019 ratio + differentiation timescale (need PDF)

4. Coupled equilibrium — NOT STARTED
   └─ Stromal fraction: Torphy 2018 f_stroma_max already extracted (contraction 0.80)
   └─ Remaining: phi_col_50_Kmax, K_stiff_niche (need equilibrium target)

5. Treatment perturbation — NOT STARTED
   └─ k_MMP_M1: needs pre/post therapy collagen data

6. Full-model CalibrationTargets — existing (23 baseline + 10 treatment)
```

Each level uses posteriors from the previous level as priors. Joint inference across all submodel targets (currently 15 targets, 16 params) provides cross-parameter constraints through shared parameters like phi_col_ref.

### 10.4 Papers to obtain

| Priority | Paper | Status | Data type | Parameters |
|:---:|-------|:---:|-----------|------------|
| 1 | Wolf et al. 2013 (DOI: 10.1083/jcb.201210152) | DONE | Pore CS (Fig 1D), filter transmigration (Fig 7A) | d_pore_ref, phi_col_ref, d_crit_T, n_pore |
| 2 | Rubiano et al. 2018 (DOI: 10.1016/j.actbio.2017.11.037) | DONE | Normal pancreas SSM, collagen gel SSM | E_ref, n_stiff, phi_col_ref |
| 3 | Decaris et al. 2015 (DOI: 10.1371/journal.pone.0123311) | PDF IN HAND | Hepatic collagen FSR 0.2-0.6%/day | k_col_deg |
| 4 | Clark & Hilderbran 1984 (DOI: 10.1016/0003-2697(84)90197-0) | NEED PDF | Absolute collagen synthesis ~1.7 pg/cell/day | k_col_myCAF_sec |
| 5 | LeRoy 1974 (DOI: 10.1172/JCI107827) | NEED PDF | Scleroderma fibroblast 4× collagen | k_col_myCAF_sec (scaling) |
| 6 | Elyada et al. 2019 (DOI: 10.1158/2159-8290.CD-19-0094) | NEED PDF | KPC iCAF:myCAF:apCAF = 44:45:10 | CAF subtype ratios |
| 7 | Biffi et al. 2019 (DOI: 10.1158/2159-8290.CD-18-0710) | REVIEWED | Qualitative only — no extractable kinetics | Model topology validation |
| 8 | Shek et al. 2002 (DOI: 10.1016/S0002-9440(10)61125-X) | REVIEWED | TGFb fold-change on collagen synthesis | Basal/TGFb fraction (0.44 vs 0.30) |
| — | Sapudom et al. 2021 | DEFERRED | CTL migration in collagen | Functional validation |
| — | Salmon et al. 2019 | DEFERRED | TIL killing vs collagen | Functional validation |
| — | Erstad et al. 2020 | NOT FOUND | Could not locate DOI | k_col_myCAF_sec, k_col_deg |
| — | Helms et al. 2022 | DEFERRED | PSC-PDAC co-culture | Joint CAF + collagen |

---

## 11. Implementation sequence

1. **Rename ECM → collagen** throughout; remove ECM_max; use phi_collagen for saturation
2. **Add pore size** derived variable + replace H_ECM_T_mot, H_ECM_APC_mig with pore-based exclusion
3. **Add stiffness** derived variable + replace H_ECM_T_exh with stiffness-based exhaustion
4. **Add stiffness-dependent CAF niche** term to carrying capacity
5. **Add immune-dependent ECM degradation** (M1 MMP term)
6. **Validate** against qualitative bounds and calibration targets
7. **Recalibrate priors** for new parameters via literature + submodel targets
