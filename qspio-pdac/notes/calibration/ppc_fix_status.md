# PPC Fix Status

**Current: 229/287 (80%)** posterior predictive observables covered (up from 186/284 = 65%)

## Fixes applied (branch: fix/submodel-inference-issues)

### PPC code bug fix (qsp-inference)
- `comparison.py` PPC filtered `v > 0`, silently dropping all negative observables (e.g., log-ratios). Fixed to filter only `np.isfinite(v)`. This affected k_IL2_cons (4 obs) which returns log(C_f/C_0) < 0. (+4 obs)

### Unit conversion bug
- `k_IL6_sec_iCAF`: forward model used `nM × MW_kDa` claiming pg/mL output, but `nM × MW_kDa = ng/mL`. Fixed by switching input to MW_Da (26000). (+1 obs)

### Forward model fixes
- `k_CD8_exh_death`: linear model `f = k*tau` replaced with saturating `f = 1 - exp(-k*tau)`. The linear approximation breaks when Annexin-V+ fraction is large (72% in terminal exhaustion). (+1 obs)

### Prior scale corrections (wrong OOM / wrong process / wrong cell type)
| Parameter | Old | New | Rationale |
|-----------|-----|-----|-----------|
| `k_CCL5_sec_iCAF` | 1e-12, σ=0.7 | 1e-10, σ=1.2 | Match CCL2_sec range (both NF-kB-driven chemokines) |
| `VEGF_norm_50` | 500 pg/mL, σ=1.0 | 20000 pg/mL, σ=1.0 | Literature VEGFR2 EC50 range 100-1000 pM |
| `k_IL2_sec` | 3e-5, σ=0.7 | 1e-11, σ=1.0 | Consistent with other secretion rates converted to per-hour |
| `IFNg_50` | 2.9 pM, σ=0.7 | 1000 pM, σ=1.0 | Literature IFNg EC50 range 100-10000 pM |
| `k_TGFb_sec_myCAF` | 2e-11, σ=1.0 | 1e-9, σ=1.5 | TGFb is major myCAF product, higher than other CAF cytokines |
| `k_T_activation_per_APC` | 5e-7, σ=1.0 | 1e-4, σ=1.0 | Original far below feasible range given typical DC counts |
| `k_IL2_cons` | 6e-6, σ=0.7 | 5e-11, σ=1.5 | Lotze1985 systemic PK conflated renal clearance with receptor-mediated consumption |
| `k_vas_myCAF_sec` | 8.5e-6, σ=1.0 | 0.005, σ=1.0 | Prior was from macrophage rate, not PSC-specific data; PSCs secrete ~400x more VEGF per cell |

### Prior sigma widenings
| Parameter | Old σ | New σ | Rationale |
|-----------|-------|-------|-----------|
| `k_vas_decay` | 0.7 | 1.0 | Data shows 8% decline at day 21; prior was too narrow |
| `rho_Th_max` | 0.307 | 0.7 | Areal→volumetric conversion uncertainty, Abercrombie overcounting |
| `k_IL10_sec` | 0.7 | 1.0 | Allow inference to reach AAMphi data (low tme_compat) |
| `k_myCAF_death` | 0.5 | 1.0 | Allow higher death rates for Kisseleva2012 recovery data |

### Nuisance prior fixes
| Target | Param | Old | New | Reason |
|--------|-------|-----|-----|--------|
| `IL6_50_MDSC` | `k_MDSC_gen_baseline` | mu=0, σ=1.0 | mu=10.49, σ=0.5 | Was predicting 1 cell vs observed 35000 |
| `k_Treg_CD8_exh` | `k_CD8_exh_basal_effective` | 0.1, σ=1.0 | 0.001, σ=1.5 | Basal exhaustion without Tregs should be very low |
| `k_IL2_cons` | `f_cd25hi` | σ=0.19 | σ=0.5 | Was effectively pinned |
| `k_CD8_exh_death` | `tau_CD8_exh_apop` | σ=0.2 | σ=1.5 | Was effectively pinned; widened for saturating model |
| `k_vas_myCAF_sec` | `V_well_mL` | σ=0.2 | σ=0.2 | Tightened (was 0.5) — 6-well plate volume is well-known |
| `caf_prolif_rates` | `K_well` | mu=14, σ=0.5 | mu=14.5, σ=1.0 | Carrying capacity too low for observed myCAF growth |
| `caf_interconversion` | `K_well` | mu=13.5, σ=0.5 | mu=14, σ=1.0 | Same issue |
| `k_C1_death` | `apoptosis_duration` | mu=-2.08, σ=0.4 | mu=0, σ=0.7 | 3h too short for PDAC; desmoplastic stroma extends TUNEL detectability to ~24h |

### Group config changes
| Group | Change | Rationale |
|-------|--------|-----------|
| `CAF_death_rates` | between_member_sd 0.4 → 0.7 | Allow ~2x fold differences between subtypes |

### NPE decomposition via cascade cuts
Dissolved `CAF_prolif_rates` group and added cascade cuts to break the 10-param NPE mega-component:

| Cascade cut | Upstream target | Method | Result |
|-------------|----------------|--------|--------|
| `k_myCAF_prolif` | caf_prolif_rates_biffi2019 | NUTS (algebraic) | 96% contraction, 12/12 PPC |
| `k_iCAF_prolif` | caf_prolif_rates_biffi2019 | NUTS (algebraic) | 65% contraction, 12/12 PPC |
| `k_myCAF_to_iCAF` | k_myCAF_to_iCAF (Zhang2025) | NUTS (algebraic) | 11% contraction, 2/3 PPC |
| `k_iCAF_to_myCAF` | k_myCAF_to_iCAF (Zhang2025) | NUTS (algebraic) | 4% contraction, 2/3 PPC |

NPE core improved: 24/48 (50%) → 33/44 (75%) → 11/12 (92%). SBC failures: 6 → 3 → 1.

Further decomposition: dissolved TGFb_EC50_fibroblast group, cascade-cut TGFb_50_CAF_act
from Bachem1992 dose-response, removed k_PSC_to_myCAF cascade cut (activation targets
became orphans). Result: interconversion NPE reduced to 1 free param (TGFb_50_IL1R1),
activation+apCAF forms new 3-param NPE (18/30 PPC). (+2 obs)

### Orphan forward model fixes
| Target | Fix | Result |
|--------|-----|--------|
| `IL1_50` (iCAF induction) | `max_iCAF_induction` prior mu: 0.0 → 3.0 (median 1→20) | Was predicting fold-change=1 at all doses |
| `k_myCAF_prolif` (Bachem BrdU) | Error model: all-FCS bootstrap → 10% FCS only | Forward model predicts saturating rate, was averaging sub-saturating conditions |

### qsp-inference bug fixes
- Orphan rescue: targets with all-cascade QSP params were silently dropped from inference
- Duplicate-edge in_degree: two cascade cuts sharing the same upstream→downstream pair caused false cycle detection

## Remaining failures (60 uncovered observables)

### NPE component (33/44) — 11 uncovered
Dominated by CAF time-course undershoot (structural — competing constraints from other observables):
- **myCAF proliferation (Biffi2019)**: days 2-5 are 2-9x under
- **CAF interconversion (Biffi2019 CMT)**: 1.6-3.5x under at days 2-5
- **PSC activation (Bachem1993)**: cell counts undershoot at later days

### Complete failures (0/N) — 7 components, 10 obs
| Component | PPC | Miss | Notes |
|-----------|-----|------|-------|
| `k_IL12_sec` | 0/1 | 2.81x above | Assumed well volume, supraphysiological TLR stimulation |
| `k_TGFb_Tsec` | 0/1 | 2.48x below | 3 assumed experimental params, indication mismatch (NHL) |
| `k_cell_clear` | 0/1 | 3.30x below | Thymic data, tme_compat=low; rate too high for PDAC |
| `CAF_death_rates` | 0/1 | 1.13x below | Group pooling pulls k_myCAF_death toward low base rate |
| `MDSC` | 0/2 | 1.11-1.63x above | tme_compat=low, ratio identifiability |
| `k_col_deg` group | 1/3 | — | f_M2_eff tension |
| `k_IL6_sec_iCAF` | 1/2 | 1.03x above | TNFa condition near-miss |

### Partial failures — ~34 obs
- **Dose-response extremes**: IFNg_50_CXCL9 (5/8), IL10_50_NK (5/8), E_50_Texh (4/6) — Hill tails
- **Multi-tissue tension**: k_APC_death (6/12) — spleen vs mesLN data pull in opposite directions
- **IL1_50 low-dose**: 5-70x below at lowest dose (Hill curve extrapolation)
- **Near-misses** (<1.5x): CCL5 late timepoints, d_crit_T, k_PSC_to_apCAF, k_APC_mig, TGFb_50_NK, k_CAF_encounter, CXCL12 secretion, K_M1_phago, E_50_fib