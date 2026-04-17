# PDAC ABM Module Summary — Update Meeting Slides

## 1. O2 Release, Uptake & Hypoxia Cascade

### O2 Source (Vascular PHALANX cells)
- Krogh cylinder implicit formula: `dC/dt = KvLv/V * (C_blood - C_local)`
  - Split into constant source + proportional uptake for numerical stability
- `C_blood` = 0.51 mM (arterial pO2)
- ECM compression reduces O2 delivery: `KvLv *= (1 - density/(density + K_compress * (1 + maturity_resistance * maturity)))`
- Dysfunctional vessels (sprouted into hypoxic tissue) deliver at 40% capacity

### O2 Uptake
- Cancer cells: uptake rate 1e-3 /s (all states)
- PDE: diffusivity 2.8e-5 cm²/s, decay 1e-5 /s (~115 min half-life)

### Downstream Hypoxia Effects (threshold: 0.01 mM for all cell types)

**Cancer Cells:**
- HIF-1a binary switch → VEGF-A boosted 7.15x, CCL2 boosted 3.2x
- PD-L1 upregulated (+0.41 fraction of max)
- MHC-I downregulated (kill probability reduced 50%) — immune evasion

**T Cells (3-tier graded impairment with hysteresis):**
- Tier 1 (>10 steps hypoxic): kill/secrete factor = 0.7
- Tier 2 (>30 steps): factor = 0.4
- Terminal (>60 steps): forced SUPPRESSED state
- Impairs: killing, division, IFN-g/IL-2 secretion
- Recovery: 1 step per normoxic step (slow)

**Macrophages:**
- Hypoxia shifts polarization toward M2 (anti-inflammatory)
- Increases M1->M2 transition, decreases M2->M1 reversion

**Fibroblasts (myCAF):**
- HIF boosts TGF-b secretion (3x), ECM deposition (2.17x)
- PSC metabolic decline at severe hypoxia (<50% threshold) — self-limiting

**Vascular TIP cells:**
- Sprouting into hypoxic tissue → marked dysfunctional (permanent 40% O2 capacity)

---

## 2. B Cells

### States
| State | Description |
|-------|-------------|
| BCELL_NAIVE | Recruited, patrols via CXCL13 chemotaxis |
| BCELL_ACTIVATED | Clonal expansion, secretes IL-6 + CXCL13 |
| BCELL_PLASMA | Terminal, secretes antibody (ADCC) or IL-10 (Breg) |

### State Machine
- **Naive -> Activated**: Antigen capture (from persistent antigen grid, Hill probability) + T cell help (TH or TFH neighbor required)
- **Activated -> Plasma**: Time-dependent, accelerated by TLS B cell clustering + TFH germinal center help
- At activation: Breg fate rolled (10% fraction)
- **Division**: Activated cells only, cooldown 4 steps, max 6 divisions

### Cytokine Production
| State | Secretes |
|-------|----------|
| Activated | IL-6, CXCL13; +IL-10 if Breg |
| Plasma (normal) | Antibody (ADCC), CXCL13 |
| Plasma (Breg) | IL-10, CXCL13 |

### Key Interactions
- **Antigen capture**: Reads persistent antigen grid (spatial DAMP from dead cancer cells)
- **T cell help**: TH/TFH neighbors required for activation
- **TFH germinal center**: Speeds up Activated->Plasma differentiation
- **TLS clustering**: B cell neighbors >= threshold accelerates differentiation
- **ADCC**: Antibody enhances M1 MAC killing + DC cross-presentation
- **Chemotaxis**: CXCL13 gradient; plasma cells near-sessile
- **Recruitment**: Baseline probability at vascular sources + CXCL13 Hill boost

---

## 3. Dendritic Cells

### States & Subtypes
- **DC_IMMATURE / DC_MATURE**
- **Subtypes**: cDC1 (cross-present to CD8, secrete IL-12), cDC2 (MHC-II to CD4/Treg)

### Maturation Model (matches QSP equation)
```
p_mature = k_mat * H_signal * (1 - H_IL10) * (1 - H_TGFb)
H_signal = 1 - (1 - H_antigen_eff)(1 - H_IL12)
H_antigen_eff = H_antigen * (1 + antibody_boost * H_ab)
```
- Either antigen OR IL-12 drives maturation; IL-10 and TGF-b suppress it
- Immune complexes (antibody) enhance antigen uptake via Fc receptors

### Presentation & Exhaustion
- Mature DCs have limited presentation capacity (~25 contacts)
- Each neighboring T/B cell costs 1 capacity; death when depleted

### T Cell Priming (host-side, QSP-derived rates)
- cDC1 -> CD8 T cell priming (PARAM_DC_PRIME_K_CD8)
- cDC2 -> TH priming (PARAM_DC_PRIME_K_TH) and Treg priming (PARAM_DC_PRIME_K_TREG)
- Primed cells receive division burst on top of base limit

### Cytokine Production
| State | Secretes | Senses |
|-------|----------|--------|
| Immature | None | Antigen, IL-12, IL-10, TGF-b, Antibody |
| Mature cDC1 | IL-12 | (depletes via presentation) |
| Mature cDC2 | None | (depletes via presentation) |

### Movement
- Immature: CCL2 chemotaxis (toward tumor/inflammation)
- Mature: CCL21 chemotaxis (toward TLS T-zones, FRC-sourced gradient)

### Key Interactions
- **No division** — consumable APCs
- **Recruitment**: Homeostatic (not chemotaxis-gated), cDC1 and cDC2 independent rates
- **B cell cooperation**: Antibody from plasma B cells boosts DC maturation via immune complexes

---

## 4. ECM Module

### Density
- Per-voxel array in biological units (nmol/mL, cap = 1000)
- **Deposition** (myCAF only): `depo = k * (1 + H_TGFb) / 3 * (1 - saturation) * YAP * dt`
  - YAP nuclear translocation feedback at high mechanical stress (EC50 = 800 nmol/mL)
  - TGF-b Hill function enhances deposition (EC50 = 0.07 nM)
  - HIF boost under hypoxia (2.17x), with PSC metabolic decline at severe hypoxia
- **Decay**: 0.007/day (~8.1e-8/s), half-life ~99 days
- **MMP degradation**: `k_mmp * MMP * density / (1 + alpha * crosslink)`
  - MMP is a PDE substrate (diffusivity 1e-7 cm²/s, decay 5e-4/s)
  - Secreted by cancer cells and M1 macrophages

### Crosslinking
- LOX from myCAFs: `crosslink += k_lox * (1 - crosslink) * dt`
- Rate: 1e-6/s (half-life to 50% = 8 days)
- Crosslinked ECM resists MMP (1 + 5*crosslink protection factor)

### Porosity (Movement Gating)
- `porosity = max(0, 1 - density/cap * (1 + crosslink))`
- Per-cell-type thresholds (matches migration mode biology):
  - FIB (0.05) < CANCER (0.1) < VAS_TIP (0.2) < MAC/MDSC/DC (0.3) < TREG/TFH/BCELL (0.35) < TCELL (0.4)
- Key driver of immune exclusion in desmoplastic PDAC stroma

### Fiber Orientation & Anisotropy
- Per-voxel unit vector (orient_x/y/z) + stress field
- **Initialization**: Isotropic; region-specific densities (Septum=700, Margin=500, Stroma=100, Lobule=50, Tumor=100 nmol/mL)
- **Reorientation drivers**:
  - myCAF traction → TACS-2 alignment (weight 0.5)
  - Cancer cell movement stress → TACS-3 alignment (weight 0.3)
  - Crosslink resistance slows reorientation (factor 2.0)
- **Contact guidance**: Amplifies gradient parallel to fibers, suppresses perpendicular (per-cell-type strength)
- **Fiber barrier**: Penalizes perpendicular movement: `w = max(0, 1 - barrier * fiber_mag * sin²θ)`

---

## 5. Cytokine Mapping: QSP vs ABM

### Shared — In Both QSP (ODE) and ABM (PDE) — 12 chemicals

| Chemical | QSP Sources (ODE) | ABM Sources (PDE) | ABM Uptake/Sensing |
|----------|-------------------|-------------------|-------------------|
| IFN-g | CD8 T cells, TH cells | T cell (EFF/CYT), MAC M1 | Cancer (PDL1 upregulation), MAC (M1 polarization), DC (maturation) |
| TGF-b | Treg, MAC M2, myCAF; homeostatic baseline | Treg, MAC M2, myCAF, Cancer stem | FIB (CAF activation), DC (suppression), T cell (suppression) |
| IL-10 | MAC M2 | MAC M2, Breg B cells | DC (suppression), MAC (M2 polarization) |
| IL-12 | mature cDC1, MAC M1 | MAC M1, DC mature cDC1 | DC (maturation co-signal), MAC (M1 polarization) |
| CCL2 | Cancer (C_total) | Cancer (all states), iCAF, MAC | MAC/MDSC/DC recruitment + chemotaxis |
| VEGF-A | Cancer (C_total), MAC M2, myCAF | Cancer (all states), myCAF, MAC | Vascular TIP chemotaxis |
| NO | MDSC | MDSC | (effect not yet wired to T cells) |
| ArgI | MDSC | MDSC | (effect not yet wired to T cells) |
| CCL5 | Cancer (C_total), iCAF | Cancer (all states), iCAF | T cell chemotaxis (EFF/CYT/SUP) |
| CXCL12 | iCAF, Cancer (C_total) | Cancer, iCAF, FRC | Treg chemotaxis (CXCR4) |
| IL-1 | Cancer (C_total) | Cancer, MAC M1 | iCAF activation signal |
| IL-6 | iCAF | iCAF, B cell activated | TFH differentiation (TH->TFH EC50) |

**Key differences in shared chemicals:**
- ABM adds MAC M1 as IL-1 source (QSP has cancer only)
- ABM adds Breg B cells as IL-10 source (QSP has MAC M2 only)
- ABM adds Cancer stem TGF-b secretion (QSP has no cancer TGF-b)
- ABM adds FRC as CXCL12 source (QSP has iCAF + cancer only)
- ABM adds MAC as CCL2 source (QSP has cancer only)

### QSP-Only — Not Spatially Resolved in ABM

| Species | QSP Sources (ODE) | QSP Role |
|---------|-------------------|----------|
| aPD1 (nivolumab) | Exogenous dosing | PD-1 checkpoint blockade — binds PD-1 on CD8 and MAC synapses |
| aPDL1 | Exogenous dosing | PD-L1 checkpoint blockade |
| aCTLA4 (ipilimumab) | Exogenous dosing | CTLA-4 blockade — CD80/CD86 co-stimulation |
| P0, P1 (neoantigens) | Cancer cell death (lysis) | Antigen presentation to DCs (ABM uses persistent antigen grid instead) |
| GM-CSF | GVAX cells (injection depot) | Dendritic cell recruitment/maturation boost |
| IL-2 (LN compartment) | CD8 T cells, TH cells (in LN) | T cell proliferation + Treg homeostasis in lymph node |
| Collagen (V_T_collagen) | myCAF deposition, MMP degradation | ECM dynamics (ABM has separate ECM density/crosslink arrays) |

### ABM-Only — Not in QSP ODE

| Chemical | ABM Sources (PDE) | ABM Role |
|----------|-------------------|----------|
| O2 | Vascular PHALANX (Krogh model) | Vascular source, cancer uptake, hypoxia cascade for all cell types |
| IL-2 (tumor) | T cell (EFF/CYT), Treg | T cell autocrine proliferation signal (QSP has IL-2 only in LN) |
| CXCL13 | TFH, activated B cells, plasma B cells | TLS organizing chemokine (B cell/TFH attraction) |
| CCL21 | FRC fibroblasts | TLS T-zone homing (mature DC chemotaxis toward T-zone) |
| MMP | Cancer (all states), MAC M1 | ECM degradation — diffuses as PDE, degrades density field |
| Antibody | Plasma B cells (non-Breg) | ADCC boost to M1 MAC killing + DC cross-presentation enhancement |
