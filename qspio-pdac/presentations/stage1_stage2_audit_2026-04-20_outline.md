# Integrating Stage-1 and Stage-2 Calibration — Outline

**Target length:** 20 minutes (~20–22 slides, ~1 min each on average)
**Draft date:** 2026-04-20

---

## I. Framing (3 min, ~3 slides)

### Slide 1 — Title
- Integrating Stage-1 and Stage-2 Calibration: A Parameter Audit of the PDAC QSP Model
- Date, venue, presenter.

### Slide 2 — The calibration problem
- 217 free parameters in `pdac_priors.csv`; ~40 scenario-level observables.
- No single experiment can constrain this — need a staged pipeline.
- **Goal of the talk:** show how Stage-1 (submodel targets) and Stage-2 (joint NPE) combine, and what the audit tells us to do next.

### Slide 3 — Two-stage pipeline, one pager
- **Stage 1:** per-parameter posteriors from in vitro / ex vivo submodel targets → `submodel_priors.yaml`.
- **Stage 2:** joint NPE across 4 scenarios, 38 observables → `posterior_samples.csv`.
- **Audit:** compares pre-prior → S1 → S2 along four diagnostic axes.
- (Diagram: targets → parameters → observables.)

---

## II. Stage 1 — what submodel targets buy us (4 min, ~4 slides)

### Slide 4 — Submodel target design
- Extract the **experiment**, not the parameter.
- ODE submodel + literature data + meta-analytic translation sigma.
- Handles context mismatch (mouse→human, in vitro→in vivo) formally via variance inflation.
- Composable: per-parameter posteriors become informative priors for full-model SBI.

### Slide 5 — Coverage snapshot
- 117/217 parameters covered (54%).
- Breakdown by type (rate / EC50 / secretion / death / trafficking).
- Gaps that matter: rate (59), EC50 (43), secretion (24), death (16).

### Slide 6 — Partial pooling (hierarchical groups)
- One extraction constrains siblings (CAF_death_rates 1/4, IL6_EC50s 2/3).
- Why this matters for extraction budget.
- Member with data pulls the group mean; data-less members shrink toward it.

### Slide 7 — Stage-1 effectiveness is not guaranteed
- ~40 params have targets but S1 contraction ≤ 30%.
- Root causes: wrong target / wrong forward-model code / prior too narrow.
- Weak-target table: TGFb_50*, IL6_50*, k_CD8_death, k_cell_clear, ArgI_50_Treg.
- **Key point:** "has a target" ≠ "is calibrated".

---

## III. Stage 2 — joint NPE (5 min, ~5 slides)

### Slide 8 — Why joint NPE
- Baseline targets pin operating point; treatment-response targets break degeneracies.
- Two parameterizations matching M1/M2 = 0.57 at diagnosis can diverge completely under anti-PD-1.
- Modular calibration ≠ joint calibration — joint captures cross-block correlations.

### Slide 9 — C++ backend + sim budget
- Switched `sbi_runner.py` from MATLAB `QSPSimulator` to `CppSimulator` (HPC array + chained derivation).
- 5000 raw → 2467 effective train sims after joint NaN filter.
- Simulation pooling accumulates across runs (config hash over priors + binary + templates).

### Slide 10 — Flow capacity sweep
- Table: (hidden × transforms) × (θ dim) → best val loss + shell ≥50%.
- Picked **192×9** for 217 params; 256×11 shows diminishing returns.
- Tier-3 collapse at small flow was **capacity**, not information.
- Capacity now adequate → non-informative params correctly marginalize to ~0 contraction.

### Slide 11 — Four diagnostic numbers per parameter
- **vs pre:** total calibration effect (pre-prior → S2 posterior).
- **vs S1:** marginal stage-2 effect on top of submodel priors.
- **Contraction global:** identifiability under the prior.
- **Contraction local-shell:** identifiability near x_obs (the trustworthy one).
- Why shell-local guards against single-point NPE artifact.

### Slide 12 — What got identified (and what didn't)
- **Genuine ID:** `k_M2_pol` local 62% / shell 51%, `f_stroma_max` 38%, `k_Th_pro_tumor` 64%.
- **Information-limited:** T-cell trafficking (`q_CD8_T_in`, `q_nCD8_LN_in`), IL-2 axis (`k_IL2_sec`, `N_IL2_CD8`), CAF interconversion, vascular (`k_vas_growth`).
- These do not move with more flow or more priors-from-pool sims. They need **new calibration targets**.

---

## IV. Stage 1 × Stage 2 integration — the audit (4 min, ~4 slides)

### Slide 13 — The inference DAG
- Figure from audit: targets → parameters → observables.
- Node coloring: S1 effectiveness, S2 shell-local contraction.
- Reading the DAG: where are the thick constraints vs the dangling parameters.

### Slide 14 — Posterior predictive coverage
- **240/282 observables** covered by 95% CI (85%).
- Remaining 42 are inherent — concentrated in treatment-response / ratio targets.
- Ready for Stage 2 follow-on (OBED) once PPC file lands.

### Slide 15 — Artifact vs genuine identification
- Concrete cases where local >> shell-local.
- `k_TGFb_sec_myCAF` local 79% / shell 4%; `k_col_myCAF_sec` local 75% / shell 7%; `d_pore_ref` local 58% / shell 8%.
- S1 already constrains these; whole-model data sees only one ridge of a larger degeneracy.
- **Lesson:** never trust local alone.

### Slide 16 — Extraction-priority ranking
- Score = sigma × PRCC × effective-coverage.
- Top of "new target" list: `q_nTreg_P_out`, `k_APC_mat_cDC2`, `K_C_exh`, `k_Treg_act`, `P1_50_apCAF`, `k_Cy_Treg_deplete`.
- Top of "fix existing" list (different axis): `k_CAF_encounter`, `k_CD8_exh_death`, `ArgI_50_Treg`, `k_cell_clear`.
- These are **not the same action** — extract-new vs fix-target-or-prior.

---

## V. What's next (3 min, ~3 slides)

### Slide 17 — Targeted extractions
- **Paired pre→post mIHC** (Heumann 2023 Fig 1F/1G): CD8 and M1/M2 ratios by arm → move T-cell and macrophage clusters.
- **KM survival at landmark times** (6, 12, 18, 24 mo): out-of-sample mechanistic check on `k_C1_growth` + kill/growth balance.
- Deprioritized: bare median OS/DFS (information-poor at n=14–16).

### Slide 18 — Option D (OBED loop)
- Predict clinical endpoints at the Stage-2 posterior → attribute endpoint CI95 back to parameters → feed into extraction priority.
- **Phases 1–2 merged** (endpoint functions + 36 prediction-only YAMLs across 3 long-horizon scenarios).
- **Phases 3–5 in flight:** `CppSimulator.simulate_with_parameters(theta)` → one-shot PPC run → audit OBED section.

### Slide 19 — Decision framing
- Audit produces **two distinct action queues**, not one:
  1. **Extract new targets** for under-identified, high-impact params.
  2. **Fix priors or forward-model code** for weak targets that aren't contracting.
- Conflating them wastes extraction effort.

---

## VI. Close (1 min, ~2 slides)

### Slide 20 — Takeaways
1. **Staged pipeline works** — S1 + S2 + OBED form a tight, audit-driven loop.
2. **Residual unidentifiability is data-limited**, with specific extractable literature targets.
3. We now have a **ranked work queue**, not a parameter soup.

### Slide 21 — Backup / Q&A
- Links:
  - `notes/calibration/parameter_audit_report.md`
  - `notes/calibration/calibration_strategy.md`
  - `notes/calibration/sbi_tier_flow_sweep.md`
  - `notes/calibration/obed_integration_handoff.md`

---

## Open questions (for editing pass)

- Audience — biweekly-update crowd or more calibration-methods-focused lab meeting? Affects NPE-concept depth.
- Keep four-numbers slide (11) or simplify to two (pre→S2 + shell-local)?
- Figures to pull in: inference DAG SVG, coverage summary, contraction plots — or placeholder boxes?
- Anything to cut or expand?
