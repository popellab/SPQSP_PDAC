# CD8 Collapse — Prior / Submodel-Target Action List

**Date:** 2026-04-22
**Context:** Qualitative-bounds audit added two-sided bounds on `cd8_treg_ratio_max_diagnosis` and `treg_fraction_min` (both now xfailed). Mechanistic diagnosis pointed at CD8 priming + death rate + missing antigen-cascade calibration. See:

- `notes/calibration/qualitative_bounds_gap_audit.md`
- `notes/calibration/cd8_collapse_diagnosis.md`

## Parameter audit cross-reference (2026-04-22)

All numbers below are from `notes/calibration/parameter_audit_report.md` (fresh).

### Params we care about — audit status

| Param | In priors | Submodel target | S1 eff (contraction) | Audit action |
|---|---|---|---:|---|
| `k_CD8_death` | ✅ (prior median 0.1, σ=0.5) | ⚠ nuisance only (rides along in `q_CD8_T_in_Boissonnas2007`) | **0%** | "Improve existing" #9 — dedicated target needed |
| `k_CD8_basal_exh` | ✅ | ✅ Bucks2009 | 3% | Low contraction; target-audit question |
| `k_Treg_CD8_exh` | ✅ | ✅ Sawant2019 | 12% | Partial |
| `k_C_CD8_exh` | ✅ | ✅ Fang2026 | 75% | Healthy |
| `k_stiff_CD8_exh` | ✅ | ✅ Zhang2024 | 79% | Healthy |
| `k_T_activation_per_APC` | ✅ | ✅ Bousso2003 | 12% | Partial |
| `n_CD8_clones`, `n_T_expansion_burst` | ✅ | ✅ Butz1998 (expansion) | 87% / 1% | Expansion burst target weak |
| `q_CD8_T_in` | ✅ | ✅ Boissonnas2007 | 68% | Healthy |
| `k_Treg_pro_tumor`, `rho_Treg_max`, `k_Th_pro_tumor`, `rho_Th_max` | ✅ | ✅ various | 41–65% | Moderate |
| **`K_C_exh`** | ✅ | ❌ none | n/a | **Audit #3 new-target priority (score 0.44, 6 obs)** |
| **`k_Th_to_Treg`** | ✅ | ❌ none | n/a | **Audit #9 new-target priority (score 0.38, in 6/6 models)** |
| `k_CD4_death` | ✅ | ❌ none | n/a | Audit #12 (score 0.37) — same story as `k_CD8_death` on the CD4 side |
| **`p1_50`** (TCR-pMHC EC50) | ❌ **NOT in priors** | ❌ none | n/a | **The single highest structural lever (H_P1 = 0.010). Add to priors + extract.** |
| `k_xP1_deg` / `k_P1_deg` | ❌ NOT in priors | ❌ none | n/a | Antigen decay; adjunct to `p1_50` |
| `y_Ag_per_cell` | ✅ | ❌ none | n/a | Audit #20 |
| `P1_50_apCAF` | ✅ | ❌ none | n/a | Audit #6 (apCAF-Treg induction — different EC50 from `p1_50`) |

### Headline

1. **`k_CD8_death` has a submodel target only as a nuisance parameter in Boissonnas 2007's `q_CD8_T_in` composite fit.** No direct CD8 half-life target anchors it. S1 eff = 0%, so the posterior width is effectively the prior width. This is the easiest high-impact fix.
2. **`p1_50` is not in the priors CSV at all** — it's held at default during SBI. Since H_P1 = pTCR/(pTCR + p1_50) and pTCR/p1_50 = 0.01 at baseline, this is the dominant lever on the priming cascade, but we currently can't move it. Add to priors first, then extract.
3. **`K_C_exh` and `k_Th_to_Treg`** are high in the audit's "extract next" list and have no submodel target yet.

## Proposed targets for extraction (added to `notes/targets.csv`)

Four new rows. Order chosen by (leverage × extraction ease):

1. **`k_CD8_death`** — dedicated target (separate from Boissonnas composite fit)
2. **`k_Th_to_Treg`** — audit-ranked, TGFb-induced Treg conversion literature is clean
3. **`K_C_exh`** — audit-ranked, 6 PRCC-significant observables
4. **`p1_50`** — highest-leverage but requires adding to priors CSV first

## Recommended sequence

1. **Add `p1_50` (and optionally `k_P1_deg`, `k_xP1_deg`) to `parameters/pdac_priors.csv`** with wide priors so SBI can at least move them. (Prerequisite for (4) below.)
2. **Extract `k_CD8_death` target** — easy, high-impact. Candidate sources: Macallan 2004 (PMID 15067029, BrdU CD8 half-life); De Boer 2003 (PMID 12594297); Kaech 2002 memory-effector kinetics.
3. **Extract `k_Th_to_Treg` target** — Chen 2003 (PMID 14676299) TGFb-induced Treg; Zheng 2007 iTreg conversion kinetics; Davidson 2007.
4. **Extract `K_C_exh` target** — Hill-EC50 for chronic antigen exposure → exhaustion. Sources: Angelosanto 2012 (PMID 22337936); Wherry 2007.
5. **Extract `p1_50` target** — hardest; requires converting pMHC-per-APC to density (molecule/μm²) via known APC contact area. Sources: Irvine 2002 (PMID 12415262, single-pMHC activation); Manz 2011 (PMID 21998596); Gonzalez 2005.

## Structural questions to address independently of priors

- **Chemotaxis module is intentionally off.** Stripped 2026-03-02 (commit 94aaf6d) — ~50 unlearnable parameters removed. If CD8 recruitment amplification is needed, a minimal 2–3 param IFNg → CD8-recruitment additive flux would give back the positive feedback without the full no-learning cluster. Lives on branch `feature/extended-tme`.
- **No local CD8 proliferation in V_T.** Treg has `k_Treg_pro_tumor × (1 − cap)`; CD8 has none. Biological literature (Reiser 2016, Beltra 2020) supports in-tumor CD8 proliferation for some contexts, but including it would add another nuisance parameter.
- **Th → Treg conversion cap fix applied in this worktree** (`core/modules/Th_module.m:61-69`). Small effect alone (~1% on Treg) because V_C.Treg transport dominates; keep as consistency fix but don't expect it to solve CD8:Treg by itself.

## Related files in this worktree

- `tests/qualitative_bounds.yaml` — added `lower: 0.1` to `cd8_treg_ratio_max_diagnosis`, `upper: 0.40` to `treg_fraction_min` (both xfailed in `tests/scenarios/test_qualitative_behaviors.py:2118-2140`)
- `core/modules/Th_module.m:61-69` — Th→Treg cap fix (uncommitted)
- `notes/calibration/qualitative_bounds_gap_audit.md`
- `notes/calibration/qualitative_bounds_tier_a_drafts.yaml`
- `notes/calibration/cd8_collapse_diagnosis.md`
- `notes/calibration/cd8_collapse_priors_handoff.md` — this file
