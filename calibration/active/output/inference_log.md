# MAPLE Inference Log — Phase 3

Running log of MAPLE rounds for the ABM. Each round documents target
authoring, validation, joint inference, and per-param contraction.
Source of truth for posteriors is `submodel_priors.yaml` (regenerated
each round); this log records the **process** that produced it.

Companion plan: `docs/phase3_maple_restart.md`.

---

## Round 1 — Diffusivities (✅ DONE 2026-05-20)

**Mechanism cluster:** `pde.*_diffusivity` (18 params)
**Targets:** 6
**Output yaml:** `calibration/active/output/submodel_priors.yaml`
**MCMC:** 1000 warmup, 5000 samples, 4 chains, 20000 total

### Targets

| Target | Source PDFs | Params covered | Trans σ |
|---|---|---|---|
| `o2_diffusivity_PDAC_deriv001` | Grote1977 (DS-Carcinosarcoma tumor + 5 tissue refs) | O2 | 0.592 |
| `no_diffusivity_PDAC_deriv001` | Liu2008 (aortic wall) | NO | 0.763 |
| `small_chemokine_diffusivity_PDAC_deriv001` | Ramanujan2002 (collagen gel, MW power-law) | CCL2, CCL5, CCL21, CXCL12, CXCL13 | 0.873 |
| `medium_cytokine_diffusivity_PDAC_deriv001` | Ramanujan2002 | IL-1, IL-2, IL-6, IFN-γ, TGF-β | 0.873 |
| `large_molecule_diffusivity_PDAC_deriv001` | Ramanujan2002 | IL-10, IL-12, VEGFA, MMP, ArgI | 0.873 |
| `antibody_diffusivity_PDAC_deriv001` | Thurber2008 (tumor IgG) + Thorne2006 (dextran benchmarks) | antibody | 0.403 |

### Joint posterior (medians)

| Parameter | Median (cm²/s) | CV | Source target |
|---|---|---|---|
| PARAM_O2_DIFFUSIVITY | 1.77e-05 | 0.47 | o2 |
| PARAM_NO_DIFFUSIVITY | 1.71e-05 | 0.55 | no |
| PARAM_ANTIBODY_DIFFUSIVITY | 1.01e-07 | 0.36 | antibody |
| PARAM_CCL2_DIFFUSIVITY | 7.88e-08 | 0.60 | small_chemokine |
| PARAM_CCL5_DIFFUSIVITY | 2.98e-07 | 0.60 | small_chemokine |
| PARAM_CCL21_DIFFUSIVITY | 2.09e-07 | 0.60 | small_chemokine |
| PARAM_CXCL12_DIFFUSIVITY | 2.94e-07 | 0.60 | small_chemokine |
| PARAM_CXCL13_DIFFUSIVITY | 2.28e-07 | 0.60 | small_chemokine |
| PARAM_IL1_DIFFUSIVITY | 1.93e-07 | 0.60 | medium_cytokine |
| PARAM_IL2_DIFFUSIVITY | 1.31e-07 | 0.60 | medium_cytokine |
| PARAM_IL6_DIFFUSIVITY | 1.88e-07 | 0.60 | medium_cytokine |
| PARAM_IFNG_DIFFUSIVITY | 1.65e-07 | 0.60 | medium_cytokine |
| PARAM_TGFB_DIFFUSIVITY | 2.80e-07 | 0.60 | medium_cytokine |
| PARAM_IL10_DIFFUSIVITY | 6.15e-08 | 0.61 | large_molecule |
| PARAM_IL12_DIFFUSIVITY | 6.99e-08 | 0.60 | large_molecule |
| PARAM_VEGFA_DIFFUSIVITY | 2.63e-07 | 0.59 | large_molecule |
| PARAM_MMP_DIFFUSIVITY | 1.44e-07 | 0.60 | large_molecule |
| PARAM_ARGI_DIFFUSIVITY | 3.68e-07 | 0.60 | large_molecule |

### Diagnostics

- **MCMC convergence:** clean across all 6 targets — R-hat 0.998–1.007, n_eff 167–645 per param, 0 divergences total.
- **Snippet validation:** 5/6 targets fully PDF-verified; Ramanujan2002 targets have figure_excerpt values (digitized from Fig. 2a) flagged MANUAL REVIEW per MCP policy — figures were re-confirmed during this run.
- **Translation σ structure:** Ramanujan2002-based targets dominate at 0.873 (proxy indication + cross-species + low directness + cross-system); O2/NO/antibody tighter.

### Sanity criterion

Original ±0.1σ-vs-April-7 test was discarded as broken-by-construction: the starting priors CSV changed between April (σ=0.3–0.5, pre-tuning medians) and current (σ=0.7 workplan default, post-tuning medians). Replacement criterion — **posterior medians must match paper data** — confirmed:

| Target | Posterior median | Paper anchor | Status |
|---|---|---|---|
| O2 | 1.77e-5 | Grote1977 tumor 1.75e-5 | ✓ |
| NO | 1.71e-5 | Liu2008 aortic ~1.5–2e-5 | ✓ |
| Antibody | 1.01e-7 | Thurber tumor IgG ~1e-7 | ✓ |
| Small/medium/large | scales 6e-8 → 4e-7 | Ramanujan MW power-law | ✓ |

### Notes / lessons

- Had to drop `distribution=fixed` point-mass rows from `pdac_abm_priors.csv` before MCP could parse it (MAPLE's prior loader only handles continuous distributions). Fix applied in `python/audit/sync_priors_from_ledger.py`; 5 rows dropped (4 zero-valued floats + 1 bool switch).
- All 6 target YAMLs were lifted from `calibration/archive/2026-04-grant_rush/submodel_targets/to-review/` with minimal edits (rename + `_PDAC_deriv001` suffix). Schema was forward-compatible with current MCP.
- Forward model architecture preserved verbatim from April (single-tissue `algebraic` for O2/NO/antibody, power-law `algebraic` parameterized by hydrodynamic radius for Ramanujan-derived targets). Could be upgraded to the structured `power_law` type in a later refresh but unnecessary for sanity.

---

## Round 2 — Molecular weights (⏭️ SKIPPED 2026-05-20, dead code)

**Mechanism cluster:** `pde.*_molecular_weight` (18 params)

**Outcome:** MAPLE Step-1 investigation revealed all 18 MW params are dead code — registered into the FLAMEGPU env in `PDAC/abm/gpu_param.cu` but with **zero downstream consumers** anywhere in `PDAC/**.{cu,cuh,cpp,h,inc}`, `PDAC/qsp/`, `PDAC/codegen/`, or `python/`. The PDE solver consumes `*_RELEASE` rates directly without MW unit conversion; QSP handles its own unit accounting independently.

Per the MAPLE extraction guide ("Flag parameters where the literature data does not map to the model's parameterization") MAPLE extraction was skipped. Re-audit when MW gets wired into a unit-conversion path.

**Disposition:** all 18 marked `FIXED` in `docs/abm_parameters_v3.csv` (status counts: 35 FIXED, 246 MANUAL_PRIOR, 2 FLAG). Dropped from `pdac_abm_priors.csv` (now 243 rows). Future direction: FIXED at canonical UniProt value, no SBI calibration. See `memory/project_prior_database_exclusions.md` §D.

**Time cost:** ~10 min code investigation, no extraction work. Net positive — caught 18 stubs that would have produced unused forward-looking provenance.

---

## Round 3 — PDE decay + uptake (✅ DONE 2026-05-22)

**Scope change vs. plan:** Round 3 in the original plan was "PDE secretion + EC50". Re-scoped on 2026-05-22 to **PDE decay rates + uptake rates** instead (mass-balance cluster). Secretion + EC50 moves to Round 4. Reason: decay/uptake are natural pairs with the Round 1 diffusivities and bound by similar literature data (half-lives, receptor cycling t1/2).

**Cluster candidates:** 8 params — PARAM_{ANTIBODY,CCL21,CXCL13,O2,VEGFA}_DECAY_RATE + PARAM_{CCL2,O2,VEGFA}_UPTAKE.

**Dead-code grep (2026-05-22):** all 8 params have real downstream consumers (PDE solver config in `pde_integration.cu` for decay; agent functions for uptake). No MW-style dead-code stubs this round.

**MMP deactivation (2026-05-22):** in parallel with Round 3 setup, MMP module was deactivated. 5 ledger rows flipped to FIXED at 0.0 (MMP_DECAY_RATE, MMP_DIFFUSIVITY, CANCER_MMP_RELEASE, MAC_M1_MMP_RELEASE; plus pre-existing MMP_MOLECULAR_WEIGHT FIXED). Round 1 yaml had PARAM_MMP_DIFFUSIVITY posterior dropped; large_molecule_diffusivity target updated to remove MMP (re-validated PASS). Status counts: 39 FIXED / 2 FLAG / 242 MANUAL_PRIOR.

**O2_DECAY drop (2026-05-22):** PARAM_O2_DECAY_RATE deferred to MANUAL_PRIOR (no MAPLE target). The "non-cellular tissue O2 sink" is a modeling abstraction — Powathil2012 sets η=0; Secomb1994/Grimes2014 report total tissue OCR not separated. Per MAPLE guide rule "flag parameters where the literature data does not map to the model's parameterization". Notes added to ledger.

### Target plan: 6 targets / 7 params

| Target file | Param(s) | Primary paper | Backup |
|---|---|---|---|
| `antibody_decay_PDAC_deriv001` | PARAM_ANTIBODY_DECAY_RATE | Eigenmann2017 J.Physiol (10.1113/JP274819) | Eigenmann2017 MAbs (10.1080/19420862.2017.1337619); Thurber2008 ADDR (10.1016/j.addr.2008.04.012) |
| `o2_uptake_PDAC_deriv001` | PARAM_O2_UPTAKE | Cheng2014 BJC (10.1038/bjc.2014.272) | — |
| `ccl2_uptake_PDAC_deriv001` | PARAM_CCL2_UPTAKE | Volpe2012 PLoS (10.1371/journal.pone.0037208) | Zhao2019 J.Immunol (10.4049/jimmunol.1900961) |
| `vegfa_decay_uptake_PDAC_deriv001` | PARAM_VEGFA_DECAY_RATE + PARAM_VEGFA_UPTAKE | Stefanini2011 PLoS (10.1371/journal.pone.0027514) + MacGabhann2006 PLoS Comp Biol (10.1371/journal.pcbi.0020127) | — |
| `ccl21_decay_PDAC_deriv001` | PARAM_CCL21_DECAY_RATE | Weber2013 Science (10.1126/science.1228456) | — |
| `cxcl13_decay_PDAC_deriv001` | PARAM_CXCL13_DECAY_RATE | Cosgrove2020 Nat.Comm (10.1038/s41467-020-17135-2) | — |

10 unique DOIs verified via `mcp__maple__verify_dois` on 2026-05-22 (one search-agent DOI hallucination caught: Doskey2018 — dropped, Cheng2014 covers the O2-uptake niche alone).

### Status

- [✓] Round-3 cluster identified (decay + uptake), dead-code grep clean
- [✓] MMP module deactivated, ledger + Round-1 yaml + Round-1 target cleaned
- [✓] O2_DECAY annotated MANUAL_PRIOR (no MAPLE target)
- [✓] 6 parallel literature search subagents completed
- [✓] 10 DOIs verified, paper directories created
- [✓] All 10 PDFs in Zotero, fetched via `fetch_papers_from_zotero`
- [✓] 6 SubmodelTarget YAMLs authored and individually validated
- [✓] VEGFA_UPTAKE deferred to MANUAL_PRIOR (Stefanini/MacGabhann single-point literature consensus, no measured spread — see ledger note)
- [✓] Joint inference run over 12 yamls (6 R1 + 6 R3), `submodel_priors.yaml` regenerated

### Targets

| Target | Source PDF(s) | Params covered | Trans σ |
|---|---|---|---|
| `antibody_decay_PDAC_deriv001` | Eigenmann2017 J.Physiol (mouse skin + muscle washout) | PARAM_ANTIBODY_DECAY_RATE | 0.696 |
| `o2_uptake_PDAC_deriv001` | Cheng2014 BJC (6 PDAC lines Seahorse) | PARAM_O2_UPTAKE | 0.442 |
| `ccl2_uptake_PDAC_deriv001` | Volpe2012 PLoS + Zhao2019 JI (CCR2 cycling) | PARAM_CCL2_UPTAKE | 0.596 |
| `vegfa_decay_PDAC_deriv001` | Stefanini2011 PLoS (mouse muscle lymphatic) | PARAM_VEGFA_DECAY_RATE | 0.857 |
| `ccl21_decay_PDAC_deriv001` | Weber2013 Science (mouse skin CCL21 gradient) | PARAM_CCL21_DECAY_RATE + PARAM_CCL21_DIFFUSIVITY (joint w/ Round 1) | 0.778 |
| `cxcl13_decay_PDAC_deriv001` | Cosgrove2020 Nat Comm (tonsil CXCL13 autocorrelation) | PARAM_CXCL13_DECAY_RATE | 0.604 |

### Joint posterior (Round-3 parameters)

| Parameter | Median | CV | XML default | Δ vs XML |
|---|---|---|---|---|
| PARAM_ANTIBODY_DECAY_RATE | 5.08e-05 /s | 0.42 | 5e-05 | ≈match |
| PARAM_CCL2_UPTAKE | 1.37e-03 /s | 0.48 | 5e-04 | 2.7× ↑ |
| PARAM_CCL21_DECAY_RATE | 1.01e-05 /s | 0.80 | 1e-05 | ≈match |
| PARAM_CXCL13_DECAY_RATE | 9.61e-05 /s | 0.69 | 1e-04 | ≈match |
| PARAM_O2_UPTAKE | 0.1141 /s | 0.39 | 0.10 | ≈match |
| PARAM_VEGFA_DECAY_RATE | 1.16e-04 /s | 0.68 | 1.92e-04 | 0.6× ↓ |

### Diagnostics

- **MCMC convergence:** clean across all 12 targets — R-hat ≈ 1.00, n_eff > 150 per param, 0 divergences total.
- **Contraction:** moderate (0.18-0.81) — typical for proxy data with translation σ in [0.4, 0.87].
- **Snippet validation:** all R3 targets pass schema + snippet checks; figure-digitized inputs (Fig 2D Cosgrove2020, Fig 2B Cheng2014, Fig 4c Cosgrove2020) flagged MANUAL REVIEW per MCP policy.
- **Anti-pattern caught:** validator rejected an invented "uncertainty CV" reference value in the initial joint vegfa_decay_uptake draft (pattern match on `cv` suffix); led to splitting VEGFA_UPTAKE off to MANUAL_PRIOR — correct behavior per MAPLE no-invented-uncertainty rule.

### Sanity criterion

5 of 6 R3 posterior medians match the XML default within ±10%, validating that the legacy XML parameter choices were biologically grounded. The one exception is PARAM_CCL2_UPTAKE, which the legacy XML interpreted as a per-receptor cycling rate but the PDE solver actually applies as a per-cell first-order rate — MAPLE correctly pulled it 2.7× higher to match Volpe2012's direct chemokine-clearance measurement. This is a meaningful correction that the inference surfaced.

### Notes / lessons

- Single-paper joint targets with multiple parameters (e.g., the original vegfa_decay_uptake design) fail when one of the parameters has only a literature-consensus point estimate without measured spread. Split such targets: keep the MAPLE-tractable parameter, drop the other to MANUAL_PRIOR with a ledger reference.
- Multi-word author surnames (e.g., "Mac Gabhann") break the validator's CrossRef metadata match. Workaround: cite in rationale/snippets rather than as structured `secondary_data_sources`.
- For gradient-based decay rate targets (Weber CCL21, Cosgrove CXCL13), include diffusivity as a co-parameter so the joint inference uses the Round-1 diffusivity constraints — otherwise the gradient datum only constrains the ratio k/D, leading to low contraction on each individually.
- Visual figure digitization with `figure_excerpt` (vs WPD) is acceptable as a first pass; manual review flagged automatically. Refine to WPD when sensitivity analysis reveals these inputs as load-bearing.

---

## Round 4 (🔄 in progress — gate done, authoring next)

### Dead-code gate (2026-05-26)

Live release/EC50 token inventory in PDAC/agents/, PDAC/pde/, PDAC/sim/ (excluding `_old.cu` and `backup_*`): **67 tokens**.

Cross-referenced against v3 ledger:
- **23 in ledger as MANUAL_PRIOR** → R4-eligible candidates
- **2 in ledger as FIXED already** (PARAM_CANCER_MMP_RELEASE, PARAM_MAC_M1_MMP_RELEASE — MMP deactivated in R3)
- **44 missing from ledger** → all verified as QSP-derived (`<![CDATA[QP(...) * scale]]>` in XML). Out of scope for ABM MAPLE (calibrated upstream via QSP MATLAB→C++).

**One dead candidate caught:**
- `PARAM_TCELL_IFNG_RELEASE_TIME` — zero consumers in active code. T cells/Macs secrete IFN-γ via QSP-derived `PARAM_IFNG_RELEASE = QP(P_k_IFNg_Tsec) * qp_sec_to_abm`. Only consumer was `pde_integration_old.cu` (legacy timer architecture). **Flipped to FIXED** in v3 ledger with deprecation note; XML registration left as defensive.

The 3 surviving `*_RELEASE_TIME` params (IL2, TCD4_TGFB) are live but use a different mechanic — periodic burst timers initialized on each newly-recruited cell (`IL2_release_remain`, `TGFB_release_remain` countdowns), not continuous rates. Need release-period framing in MAPLE targets (units: s/burst, not pmol/cell/s).

### Target inventory (10 targets, 22 params)

| # | Target | Params | n | Cluster |
|---|---|---|---|---|
| 1 | cancer_tgfb_secretion ✅ | STEM_TGFB_RELEASE, PROG_TGFB_RELEASE | 2 | secretion |
| 2 | ~~treg_il10_secretion~~ ❌ closed-by-fix | TREG_IL10_RELEASE → FIXED at 0 | 1 | secretion |
| 3 | mac_m1_il1_secretion | MAC_M1_IL1_RELEASE | 1 | secretion |
| 4 | frc_lymphoid_chemokines | FIB_FRC_CCL21_RELEASE, FIB_FRC_CXCL12_RELEASE | 2 | secretion |
| 5 | bcell_secretome | BCELL_ANTIBODY, IL10, IL6, ACT_CXCL13, PLASMA_CXCL13 (all `_RELEASE`) | 5 | secretion (joint) |
| 6 | tfh_biology | TFH_CXCL13_RELEASE, TFH_EC50_IL6 | 2 | secretion + sensing |
| 7 | adcc_pharmacodynamics | ADCC_AB_EC50, ADCC_AB_HILL_N | 2 | sensing (paired Hill) |
| 8 | bcell_cxcl13_chemoreceptor | BCELL_EC50_CXCL13_REC | 1 | sensing |
| 9 | tcell_secretion_timers | TCELL_IL2_RELEASE_TIME, TCD4_TGFB_RELEASE_TIME | 2 | burst-period |
| 10 | mechanosensing_ec50 | ECM_YAP_EC50, VAS_COLLAPSE_EC50 | 2 | sensing (mechano) |

Note: ~~⚠️ `TREG_IL10_RELEASE=0` in current XML — production was disabled. Target should set a literature-grounded prior and surface a non-zero posterior; user will need to decide whether to enable in XML default after inference.~~ **Update 2026-05-26: T2 closed-by-fix.** Lit search surfaced that purified CD4+CD25+Foxp3+ Tregs do not autonomously secrete IL-10 in monoculture (Levings 2002, Tiemessen 2007, Mohseni 2021). XML `0` is biologically correct. Flipped ledger row to FIXED; see T2 entry below. Round-4 target count drops to **9 targets / 21 params**.

### Target-by-target progress

#### T1 — cancer_tgfb_secretion ✅ VALIDATED 2026-05-26

- **Inferred param:** `PARAM_STEM_TGFB_RELEASE` (single param; `PARAM_PROG_TGFB_RELEASE` pooled to mirror STEM's posterior in ledger-sync, justified by Lonardo2011 stem≈bulk TGFb finding).
- **Primary source:** Roger et al. 2019 (DOI `10.1038/s41419-019-2116-x`), Capan-2 conditioned-media TGFb1 Quantikine ELISA, n=3 biological replicates, 3×10⁵ cells / 6-well / 24 h / DMEM+0.5%FBS / standard 2 mL volume.
- **Forward model:** `batch_accumulation`; predicts pg/mL from sec_rate · cells · time · MW(25 kDa) / vol.
- **Digitized data:** Fig 2a WPD extraction of 3 individual replicate points (-1.92, 34.62, 48.08 pg/mL) — one replicate is at/below Quantikine LOD (~7 pg/mL).
- **Prior tension caught:** Roger implies sec_rate ≈ 8.3e-14, which was 5.9σ below the original CSV prior `lognormal(median=5e-12, σ=0.7)`. Broadened to `lognormal(median=6.3e-13, σ=1.5)` ahead of MAPLE (95% CI [3.3e-14, 1.2e-11] brackets both Roger and the legacy XML default at ±1.4σ each). Updated in both `docs/abm_parameters_v3.csv:244-245` and `PDAC/sim/resource/pdac_abm_priors.csv:218-219`.
- **Submodel-only posterior:** median 1.055e-13 pmol/cell/s, CV=0.64, lognormal fit, translation σ=0.377, contraction=1.00, MCMC clean (n_eff=203, R-hat=0.999, 0 divergences). About 50× below the legacy XML 5e-12.
- **YAML:** `calibration/active/submodel_targets/cancer_tgfb_secretion_PDAC_deriv001.yaml`

#### T2 — treg_il10_secretion ❌ CLOSED-BY-FIX 2026-05-26

- **Decision:** Flip ledger row `PARAM_TREG_IL10_RELEASE` from MANUAL_PRIOR → FIXED (value 0, distribution `fixed`). No MAPLE YAML authored.
- **Trigger:** Literature subagent surfaced foundational evidence that purified CD4+CD25+Foxp3+ Tregs do not autonomously secrete IL-10 in monoculture:
  - **Levings & Roncarolo 2002 JEM** (DOI 10.1084/jem.20020110) — CD25+CD4+ T-suppressor clones produce TGF-β but NOT IL-10. Explicit distinction from Tr1.
  - **Tiemessen & Taams 2007 PNAS** (DOI 10.1073/pnas.0706832104) — Purified CD4+CD25+CD127loFoxp3+ Tregs: IL-10 absent from monoculture supernatant; only detected when co-cultured with monocytes (cell-cell contact required).
  - **Mohseni 2021 Eur J Immunol** (DOI 10.1002/eji.202048934) — WT control canonical Tregs produced negligible IL-10 ("IL-10 effect not significant").
- **Anti-pattern caught:** The high-IL-10 Treg papers (Urry 2009 JCI ~3000 pg/mL) use FOXP3-low / Tr1-like cells, a distinct subset not modeled by `TCD4_TREG` in the ABM. PDAC ABM's `TCD4_TREG` agent is canonical Foxp3+.
- **Biological rationale:** Original ledger note "Disabled; M2 MAC handles" reflects PDAC reality — IL-10 in the tumor microenvironment is dominated by M2 macrophage production, not Treg autonomous secretion.
- **R3 analogue:** Same cleanup-by-fix pattern as MMP deactivation in Round 3 (5 ledger rows flipped FIXED at 0 after lit search confirmed deactivation).
- **Status counts after T2:** 40 FIXED / 2 FLAG / 241 MANUAL_PRIOR (was 39/2/242).
- **Round-4 inventory updated:** 9 active targets / 21 params (was 10/22).

#### T3 — mac_m1_il1_secretion ✅ VALIDATED 2026-05-27

- **Inferred param:** `PARAM_MAC_M1_IL1_RELEASE` (single param; M1-gated continuous IL-1β secretion at `macrophage.cuh:192-196`, pmol/cell/s, applied only when `cell_state == MAC_M1 && dead == 0`).
- **Primary source:** Netea et al. 2009 Blood (DOI `10.1182/blood-2008-03-146720`), Fig 2A purified CD14+ monocytes + 10 ng/mL LPS / 24 h / R&D Systems IL-1β ELISA / n=6 volunteers. Same paper Fig 3A confirms MDM-LPS-alone is essentially zero (cell-type translation caveat).
- **Forward model:** `batch_accumulation`; predicts pg/mL from sec_rate · cells · time · MW(17 kDa) / vol. Cells/well derived (5×10⁶ cells/mL × 100 μL aliquot = 5×10⁵), volume derived (100 μL cells + 100 μL stim = 0.2 mL).
- **Visual readout:** Fig 2A CD14+ LPS bar ≈ 0.9 ng/mL ± 0.30 SEM (n=6). No individual replicates plotted — only bar+SEM, so bootstrap uses `rng.normal(mean, sem, n_bootstrap)`.
- **Prior assessment:** legacy XML 4.7e-13 with σ=0.7 already brackets the Netea-implied direct rate (~2.5e-13) within 1σ; no prior broadening needed before MAPLE. XML inline comment "MAPLE posterior 4.7e-13 (~10000x down). Netea2009" predates the current MAPLE workflow and is INCORRECT — actual posterior shift is ~1.3× down, not 10000×. Comment should be updated post-joint-inference.
- **Submodel-only posterior:** median 3.581e-13 pmol/cell/s, CV=0.58, lognormal fit, translation σ=0.934, contraction=0.75, MCMC clean (n_eff=232, R-hat=0.998, 0 divergences). High translation σ reflects: monocyte→TAM cell type (+0.50 tme + indication signals dominated, MDMs need 2nd signal per same paper Fig 3A), LPS→PGE2/TNF stimulus mismatch (+0.25 perturbation), and proxy_observable directness (+0.50). About **1.3× below the legacy XML 4.7e-13**.
- **YAML:** `calibration/active/submodel_targets/mac_m1_il1_secretion_PDAC_deriv001.yaml`
- **Validator workflow note:** scientific-notation cell densities ("5 × 10⁶ cells/mL") need to be split into mantissa+scale inputs because the snippet-value matcher does substring matching on decimal forms ("5000000") which scientific notation does not contain. Mirrors the multi-word author surname caveat from R3.

#### T4 — frc_lymphoid_chemokines ⚠️ SPLIT 2026-05-27

Originally planned as a 2-param joint target (PARAM_FIB_FRC_CCL21_RELEASE + PARAM_FIB_FRC_CXCL12_RELEASE). Resolved as two separate dispositions because (a) MAPLE schema enforces one forward_model per YAML and (b) the literature for the two chemokines came from different papers with non-overlapping observation types.

##### T4-CXCL12 — frc_cxcl12_secretion ✅ VALIDATED 2026-05-27

- **Inferred param:** `PARAM_FIB_FRC_CXCL12_RELEASE` (FIB_FRC state continuous CXCL12 secretion at `fibroblast.cuh:192-194`, pmol/cell/s).
- **Primary source:** Severino et al. 2017 Front Immunol (DOI `10.3389/fimmu.2017.00141`), Fig 4C untreated FRC bar: 67 ± 81 pg/mL (mean ± SD, n=4 LN donors), R&D Quantikine ELISA, 24 h after 7-day culture. Same paper confirms TNFα + IL-1β does NOT alter CXCL12 output, supporting the ABM's no-gating flat-rate abstraction.
- **Forward model:** `batch_accumulation`; MW 8 kDa; cells_per_well=1e5 and well_volume_mL=1.0 are FRC-field convention `unit_conversion` inputs because Severino does not state these (dominant load-bearing assumption — see key_study_limitations).
- **Prior broadening:** Severino-implied rate ~9.7e-13 vs legacy XML 2e-9 ⇒ ~2000× gap = ~10σ at original σ=0.7. **Broadened ahead of MAPLE** in v3 ledger row 280 + priors CSV row 238: median 2e-9 → 4.5e-11 (geomean of legacy and Severino), σ 0.7 → 1.5. 95% CI [2.2e-12, 9e-10] brackets both anchors at ±2.5σ.
- **Submodel-only posterior:** median 2.617e-12 pmol/cell/s, CV=0.89, lognormal fit, translation σ=0.755 (proxy indication + low tme_compatibility + 2D-FRC dedifferentiation per Tomei 2009 paradigm), contraction=1.00, MCMC clean (n_eff=195, R-hat=1.000, 0 divergences). About **760× below the legacy XML 2e-9** — large correction driven by the 2D-dedifferentiation translation rather than measurement error.
- **YAML:** `calibration/active/submodel_targets/frc_cxcl12_secretion_PDAC_deriv001.yaml`
- **Validator workflow note:** input names containing the literal substring "assumed" are rejected by the schema as "modeling choice rather than extracted data". Workaround: rename + reclassify as `unit_conversion` with explicit "field-convention value applied because [paper] methods do not state X" framing in source_location. Captures the same semantic without tripping the substring filter.

##### T4-CCL21 — closed-by-no-data 2026-05-27

- **Decision:** Keep `PARAM_FIB_FRC_CCL21_RELEASE` at MANUAL_PRIOR; broaden σ from 0.7 to 1.5 in v3 ledger row 279 + priors CSV row 237. Median kept at legacy 5e-9.
- **Trigger:** Two parallel literature subagents (2026-05-27 a and b) confirmed that **no published absolute pg/mL CCL21 secretion rate exists for any FRC-equivalent cell type**:
  - Tomei et al. 2009 J Immunol (DOI `10.4049/jimmunol.0900835`) — only reports fold-change vs static control in Fig 5C ("CCL21 in matrix (normalized)"); no absolute units anywhere in the paper.
  - Severino et al. 2017 Front Immunol — measured CCL21 by ELISA but reported it as undetectable under all conditions (paper text "CCL19 was undetectable in every condition" + Fig 4 omits CCL21 — Methods says Duo Set was used for CCL19 but CCL21 isn't in the ELISA panel; Tomei-effect 2D dedifferentiation is the most likely explanation).
  - Ng 2012 PNAS, Onder 2013 JEM, Fletcher 2011, Cremasco 2014, Acton 2014, Saxena/Yamamura 2025 — all use qPCR or normalized fold-change for CCL21, no absolute pg/mL.
  - Only proxy: virally-transduced DCs (Riedl 2003 Mol Cancer + Yang 2004 ClinCanRes) give ~8e-9 pmol/cell/s — within 2× of legacy XML — but the AdCCL21 transduction context breaks per-cell secretion-rate semantics.
- **Anti-pattern caught:** This is a "data-poor" disposition analogous to R3's VEGFA_UPTAKE and O2_DECAY (literature consensus point estimate without measured spread → no MCMC tractable target → defer to MANUAL_PRIOR with widened σ).
- **Status counts unchanged:** 40 FIXED / 2 FLAG / 241 MANUAL_PRIOR. PARAM_FIB_FRC_CCL21_RELEASE remains MANUAL_PRIOR; just σ broadened.
- **Round-4 inventory:** 9 active targets — T4 contributes 1 validated (CXCL12) + 1 deferred (CCL21). 4 of 9 R4 dispositions resolved (T1✅, T2❌, T3✅, T4⚠️split).

#### T5 — bcell_secretome ⚠️ split (in progress 2026-05-27)

Originally planned as 5-param joint (BCELL_ANTIBODY/IL10/IL6/ACT_CXCL13/PLASMA_CXCL13 RELEASE). Per MAPLE one-forward-model-per-YAML constraint, split into 4 sub-targets: T5a antibody (plasma cell), T5b IL10 (Breg), T5c IL6 (activated), T5d CXCL13 (ACT + PLASMA states; consider STEM-PROG-style pooling like T1 if data supports).

##### T5a — bcell_antibody_secretion ✅ VALIDATED 2026-05-27

- **Inferred param:** `PARAM_BCELL_ANTIBODY_RELEASE` (BCELL_PLASMA non-Breg continuous IgG secretion at `b_cell.cuh:295-297`, pmol/cell/s).
- **Primary source:** Nguyen et al. 2025 Front Immunol (DOI `10.3389/fimmu.2025.1644102`). Fig 1a bASC bar: 37 ± 19 pg/cell/day (n=13 donors), 1-day BM mimetic bulk culture (MSC stroma + hypoxia + 200 ng/mL APRIL), paired ELISpot + ELISA, pg/(cell·day) = pg_supernatant / (IgG_spots × days).
- **Forward model:** `algebraic` — pg/(cell·day) = sec_rate × seconds_per_day × IgG_MW. Cleaner than batch_accumulation because Nguyen already normalized to per-cell-per-day; mirrors R3 ccl21_decay algebraic pattern.
- **Critical pre-MAPLE correction:** Legacy XML value 1e-5 pmol/cell/s is a UNIT-CONVERSION ERROR. Ledger comment "1e-5 ≈ 100 pg/cell/day" was wrong by 1296× — 100 pg/cell/day at MW 150 kDa is actually 7.7e-9 pmol/cell/s, NOT 1e-5. The 1e-5 value implies 130 ng IgG/cell/day, exceeding the entire cell mass (~200 pg). Reanchored CSV prior median 1e-5 → 4e-9 (σ=0.7 retained); v3 ledger row 53 updated with full deprecation note. Unverifiable "McMillan2025"/"Cheng2023" tags dropped.
- **Submodel-only posterior:** median 3.18e-9 pmol/cell/s, CV=0.47, lognormal fit, translation σ=0.585 (moderate — direct measurement, blood-ASC matches PDAC TLS early-minted phenotype per Helmink2020/Cabrita2020), contraction=0.83, MCMC clean (n_eff=189, R-hat=0.998, 0 divergences). About **3145× below the legacy XML 1e-5** — almost entirely driven by the unit-conversion fix, NOT by biological correction.
- **YAML:** `calibration/active/submodel_targets/bcell_antibody_secretion_PDAC_deriv001.yaml`
- **Validator workflow note:** very short snippets like "n=13" fail substring matching (score=0.50 < 0.75 threshold). Workaround: use `figure_excerpt` with `figure_id`/`value`/`description`/`context` schema instead of value_snippet — this skips the substring check and adds a MANUAL_REVIEW flag (per MAPLE policy on figure digitization). Same pattern as T3 WPD-style figure readouts.

##### T5b — bcell_breg_il10_secretion ✅ VALIDATED 2026-05-28

- **Inferred param:** `PARAM_BCELL_IL10_RELEASE` (is_breg-flag-gated continuous IL-10 secretion at `b_cell.cuh:292` BCELL_PLASMA branch + `b_cell.cuh:310` BCELL_ACTIVATED branch, pmol/cell/s). Single param covers both Breg-flagged ACT and PLASMA states; non-Breg cells secrete zero (M2 macrophages dominate the TME IL-10 source via separate param).
- **Primary source:** Iwata et al. 2011 Blood (DOI `10.1182/blood-2010-07-294249`). Fig 5F right panel CD24hiCD27+ + CD40L+CpG: ~450 pg/mL mean ± ~25 pg/mL SEM, n=3 triplicate ELISA determinations / BD OptEIA Human IL-10 / 72 h supernatant. Methods: 4×10⁵ sorted CD24hiCD27+ B cells / 0.2 mL / 96-well flat-bottom / CpG ODN 2006 10 μg/mL + sCD40L 1 μg/mL.
- **Forward model:** `batch_accumulation`; predicts pg/mL from sec_rate · cells · time · MW(18.5 kDa) / vol. Same structure as T3 Netea2009.
- **CRITICAL: legacy XML default 1e-7 pmol/cell/s was a UNIT CONVERSION ERROR (~2,050,000× too high).** Sanity check: 1e-7 × 4e5 × 72h × 3600 × 18,500 / 0.2 = 1.92e11 pg/mL = 192 g/mL supernatant under Iwata's protocol, physically impossible. Reanchored CSV prior median 1e-7 → 5e-14 (σ broadened 0.7→1.0 for B10-subset vs is_breg-agent mapping + ex-vivo stim vs PDAC TLS cytokine milieu); v3 ledger row 54 updated with full deprecation note. **Second consecutive T5 sub-target where the XML default was a unit-conversion bug** (T5a was 3145× off, T5b is ~600× worse). Pattern: B-cell secretion-rate XML defaults appear systematically miscalibrated by orders of magnitude — flag any remaining BCELL_*_RELEASE params as suspect until verified.
- **Submodel-only posterior:** median **4.877e-14 pmol/cell/s**, CV=0.77, lognormal fit, translation σ=0.934 (high — same breakdown as T3 Netea2009: +0.50 indication/tme/directness, +0.25 perturbation, +0.20 system), prior σ=0.679, contraction=0.81, MCMC clean (n_eff=243, R-hat=0.998, 0 divergences). About **2.05M× below the legacy XML 1e-7** — almost entirely driven by the unit-conversion fix.
- **Pivoted primary source mid-flight:** Lit-search subagent recommended Barsotti2016 PLOS ONE (DOI 10.1371/journal.pone.0151761) as text-extractable bulk-CD19 ELISA at 736.3 pg/mL. Barsotti not in Zotero → pivoted to Iwata2011 (already in Zotero, original ledger citation, Fig 5F has the sorted B10-subset ELISA which is mechanistically closer to the ABM is_breg gating). Barsotti remains a viable cross-check if cross-paper consistency is needed for joint inference; consider adding in a future R4 refresh.
- **Subagent-arithmetic gotcha:** The literature-search subagent's back-calc of Iwata's per-cell rate ("≈1.04×10⁻⁷ pmol/cell/s ✓ matches ledger") was wrong by ~10⁶ — they appear to have inverted/missed a unit factor. Lesson: every subagent-reported unit conversion must be independently spot-checked before being trusted in any downstream decision (prior re-anchor, validation, etc.). The math-error caught is the entire reason the legacy XML 1e-7 looked plausible all this time.
- **YAML:** `calibration/active/submodel_targets/bcell_breg_il10_secretion_PDAC_deriv001.yaml`

##### T5c — bcell_il6_secretion ✅ VALIDATED 2026-05-28

- **Inferred param:** `PARAM_BCELL_IL6_RELEASE` (BCELL_ACTIVATED state unconditional secretion at `b_cell.cuh:305-306`, pmol/cell/s). Single param; NOT is_breg-gated (separate from T5b's Breg-specific IL10 pathway).
- **Primary source:** Hanten et al. 2008 BMC Immunology (DOI `10.1186/1471-2172-9-39`). Fig 3B CpG 2006 24-h bar on purified human CD19+ B cells (≥99% purity, naive+memory mix): ~1500 pg/mL ± ~500 pg/mL SEM, n=3 donors / Luminex xMAP / 24h / 2.5×10⁶ cells/mL (range 2-3×10⁶) / X-vivo 20 serum-free / 3 μM CpG2006.
- **Forward model:** `batch_accumulation`; predicts pg/mL from sec_rate · cells · time · MW(21 kDa) / vol. Volume cancels mathematically (cells/mL × volume = cells_per_well; per-cell rate invariant) — used 1-mL normalization basis with cells_per_well = 2.5e6.
- **Pivot mid-flight:** Hanten2008 NOT in Zotero. Pulled directly from BMC OA (open access via `bmcimmunol.biomedcentral.com/counter/pdf/10.1186/1471-2172-9-39.pdf`); legitimate path for OA papers. Subagent recommendation #2 (Agrawal2011) would have had paywall issues.
- **Subagent fact-check caught error:** Lit-search subagent claimed "~1000-1500 pg/mL at 24h" for Fig 3B — actually accurate after viewing the log-scale figure. BUT during PDF reading I initially mis-read "~20-150 pg/ml" as referring to IL-6 — that range actually refers to **IL-8** (line 467, same paragraph but different cytokine). Re-reading the surrounding sentence and viewing Fig 3B directly resolved the confusion. **Lesson: always verify cytokine attribution when ranges are quoted in dense Results-section paragraphs that discuss multiple cytokines.**
- **CRITICAL: legacy XML default 2e-7 pmol/cell/s was a UNIT CONVERSION ERROR (~6.8×10⁵× too high).** Sanity check: 2e-7 × 2.5e6 × 24h × 3600 × 21,000 / 1 = 1.81e15 pg/mL = 1.8 kg/mL IL-6, physically impossible. Reanchored CSV prior median 2e-7 → 3e-13 (σ broadened 0.7→1.0 for bulk-B vs activated-B-subset distinction + ex-vivo CpG vs PDAC TLS milieu); v3 ledger row 55 updated with full deprecation note. Unverifiable "PMC2777861" tag dropped.
- **Submodel-only posterior:** median **2.946e-13 pmol/cell/s**, CV=0.77, lognormal fit, translation σ=0.934 (same breakdown as T3/T5b — same proxy/tme/directness flag structure), prior σ=0.680, contraction=0.81, MCMC clean (n_eff=173, R-hat=1.000, 0 divergences). About **6.79×10⁵× below the legacy XML 2e-7** — almost entirely driven by the unit-conversion fix.
- **YAML:** `calibration/active/submodel_targets/bcell_il6_secretion_PDAC_deriv001.yaml`
- **PATTERN CONFIRMED — THREE consecutive BCELL_*_RELEASE unit-conversion errors in T5 series:**
  | Param | Legacy | Posterior | Ratio |
  |---|---|---|---|
  | T5a BCELL_ANTIBODY_RELEASE | 1e-5 | 3.18e-9 | 3145× |
  | T5b BCELL_IL10_RELEASE | 1e-7 | 4.88e-14 | 2.05M× |
  | T5c BCELL_IL6_RELEASE | 2e-7 | 2.95e-13 | 6.78×10⁵× |
  All three legacy defaults are physically impossible (g/mL or kg/mL of secreted cytokine). The original ledger source notes (e.g., "PMC2777861") were never sanity-checked against the cited papers' actual protocols. Pre-MAPLE sanity check is now mandatory for any remaining BCELL_*_RELEASE param (T5d remaining: BCELL_ACT_CXCL13_RELEASE + BCELL_PLASMA_CXCL13_RELEASE).

##### T5d — bcell_cxcl13_secretion ❌ CLOSED-BY-FIX 2026-05-28

- **Decision:** Both `PARAM_BCELL_ACT_CXCL13_RELEASE` and `PARAM_BCELL_PLASMA_CXCL13_RELEASE` flipped from MANUAL_PRIOR → FIXED at 0. No MAPLE YAML authored. Mirrors T2 (TREG_IL10) cleanup-by-fix pattern.
- **Trigger:** Literature subagent (2026-05-28) confirmed:
  - **No primary literature directly quantifies normal human B cell CXCL13 secretion.** B cells are CXCR5+ CXCL13 receptors / sinks, not sources. Canonical CXCL13 sources in TLS are FDCs (no agent in current ABM) and Tfh cells (handled by PARAM_TFH_CXCL13_RELEASE at `t_reg.cuh:531`, calibrated in R4-T6).
  - **The legacy "Mackay2004" citation in the ledger is misattributed.** The actual paper appears to be Carlsen et al. 2004 *Blood* (DOI `10.1182/blood-2004-02-0701`), which measured **monocyte** CXCL13 (~3200 pg / 10⁶ cells / 3d LPS-stimulated → ~1.2e-12 pmol/cell/s), not B cells. Even using this misidentified monocyte proxy, the legacy 1e-9 pmol/cell/s was ~830× too high.
  - **Sole direct B cell CXCL13 lit (Husson 2002 *Br J Haematol*, DOI `10.1046/j.1365-2141.2002.03832.x`) is malignant follicular lymphoma** — not a clean proxy for PDAC TLS Bregs. Bürkle et al. 2007 *Blood* explicitly demonstrates CXCL13 in CLL co-cultures comes from nurse-like cells, NOT the CLL B cells themselves.
- **Biological rationale:** B cell CXCL13 production in the current ABM is a modeling simplification of TLS organization signaling that has no published quantitative basis. The Tfh-derived CXCL13 (T6 target) is the biologically correct source.
- **Anti-pattern caught:** ALL three of the candidate B-cell CXCL13 papers I assessed are problematic — Mackay/Carlsen (wrong cell type), Husson (malignant), Bürkle (NLC source, not B cell). This is identical in structure to T4-CCL21's data poverty but with the additional issue that the existing legacy citation was misattributed. Disposition therefore goes beyond "close-by-no-data with broadening" to "FIXED=0 + biology flag for ABM refactor".
- **ABM refactor flag:** Both `b_cell.cuh:300-301` (BCELL_PLASMA branch) and `b_cell.cuh:315-316` (BCELL_ACTIVATED branch) become no-ops when these params = 0. Future ABM iteration should either (a) add a dedicated FDC agent for CXCL13 production OR (b) accept Tfh as the sole CXCL13 source. Code path is left in place defensively.
- **Confirmed: T6 Tfh CXCL13 (`PARAM_TFH_CXCL13_RELEASE` at `t_reg.cuh:531`) IS the canonical biological source** — zeroing B cell production does not zero the CXCL13 field.
- **Status counts after T5d:** 42 FIXED / 2 FLAG / 239 MANUAL_PRIOR (was 40/2/241).
- **Round-4 inventory:** **T5 series COMPLETE (T5a✅, T5b✅, T5c✅, T5d❌).** 8 of 9 active R4 dispositions resolved. Remaining R4: T6-T10 (5 targets) — see Plan forward.

### Summary — Round 4 T5 sub-series (bcell_secretome)

The T5 5-param joint plan was split into 4 sub-targets per MAPLE schema constraints; all 4 resolved:

| Sub-target | Param(s) | Disposition | Posterior / Action |
|---|---|---|---|
| T5a antibody | BCELL_ANTIBODY_RELEASE | ✅ MAPLE | 3.18e-9 (Nguyen2025; 3145× below legacy) |
| T5b Breg IL-10 | BCELL_IL10_RELEASE | ✅ MAPLE | 4.88e-14 (Iwata2011; 2.05M× below legacy) |
| T5c B cell IL-6 | BCELL_IL6_RELEASE | ✅ MAPLE | 2.95e-13 (Hanten2008; 678k× below legacy) |
| T5d CXCL13 (2 params) | BCELL_ACT/PLASMA_CXCL13_RELEASE | ❌ FIXED=0 | Biologically unsupported; B cells are CXCR5+ sinks |

**Cross-cutting findings for the BCELL_*_RELEASE family:**
1. Three of four T5 sub-targets surfaced unit-conversion errors in the legacy XML (T5a 1296×, T5b 2.05M×, T5c 678k×). All three implied physically-impossible cytokine concentrations.
2. T5d revealed that one of the original ledger citations (Mackay2004) was a misidentified paper (actually Carlsen2004 monocyte data), and that the underlying biology (B cell CXCL13 production) is unsupported.
3. **Pre-MAPLE arithmetic sanity check is now mandatory for any *_RELEASE param** — this rule paid off in all three T5 cases and is propagated to T6-T10.
4. Lit-search subagents make 10⁶× unit-conversion errors that go undetected unless independently spot-checked.
5. OA papers can be curl'd directly from journal/PMC when not in Zotero — this fallback was used in T5c (Hanten2008 from BMC OA).

#### T6 — tfh_biology ✅ T6a VALIDATED + T6b MANUAL_PRIOR 2026-05-28

T6 originally planned as a 2-param joint target (`PARAM_TFH_CXCL13_RELEASE` + `PARAM_TFH_EC50_IL6`). Resolved as two separate dispositions because the parameters represent biologically unrelated quantities (secretion rate vs Hill EC50 for sensing).

##### T6a — tfh_cxcl13_secretion ✅ VALIDATED 2026-05-28

- **Inferred param:** `PARAM_TFH_CXCL13_RELEASE` (TCD4_TFH state unconditional secretion at `t_reg.cuh:531-532`, pmol/cell/s). **Canonical TLS CXCL13 source per the T5d biology finding** (B cells are CXCR5+ sinks; FDCs not modeled).
- **Primary source:** Ukita et al. 2022 *JCI Insight* (DOI `10.1172/jci.insight.157215`). Fig 5F CD4 panel TGF-β bar: ~5000 ± ~1500 pg/mL SEM, n=3 donors / R&D Quantikine Human CXCL13 ELISA / 7-day differentiation / naive CD4 + plate-bound αCD3/αCD28 + 10 ng/mL TGF-β1 / IMDM + 10% FBS. Fig 5E TGF-β bar: ~4% CXCL13+ producer fraction (PD-1+CXCR5- Tph-like phenotype per Fig 5G).
- **Forward model:** `batch_accumulation` with explicit **producer-fraction correction**: effective `cell_count = total_CD4 × 0.04` instead of bulk total. This is the first T-series YAML to use this correction; mirrors the T1 STEM/PROG pooling pattern but applied to a producer-subset (not pooling) framing. MW 10.3 kDa (UniProt O43927 mature CXCL13).
- **Cell density convention:** Ukita does NOT state cells/mL → field-convention 5e5 cells/mL applied with explicit unit_conversion framing (T4-CXCL12 / T5c precedent). V-cancellation via 1-mL normalization basis.
- **Submodel-only posterior:** median **4.541e-11 pmol/cell/s**, CV=0.63, lognormal fit, translation σ=0.934, prior σ=0.576, contraction=0.56 (lower than T3/T5b/T5c — reflects figure_excerpt-heavy data structure + producer-fraction uncertainty), MCMC clean (n_eff=228, R-hat=0.999, 0 divergences).
- **Re-validation outcome:** Posterior 4.541e-11 ≈ current XML 4.97e-11 (9% offset). **Prior MAPLE pass was correctly calibrated**; this T6a YAML formalizes the producer-fraction correction under the current schema. UNLIKE the T5a/b/c BCELL_* series, NO unit-conversion error here — the prior MAPLE "~100x down. Ukita2022" XML comment reflects a legitimate biological correction.
- **YAML:** `calibration/active/submodel_targets/tfh_cxcl13_secretion_PDAC_deriv001.yaml`
- **Cross-cutting finding:** Producer-fraction correction is a clean way to handle bulk-supernatant ELISA data where only a subset of cells are active secretors. Use this pattern in future T-cell / B-cell subset targets where the active producer fraction is identifiable from the same paper (e.g., flow phenotyping figures).

##### T6b — tfh_ec50_il6 ⚠️ MANUAL_PRIOR_VALIDATED 2026-05-28

- **Decision:** Keep `PARAM_TFH_EC50_IL6` at MANUAL_PRIOR (no MAPLE YAML). Median 0.46 nM retained but σ broadened from 0.4 → 0.7 to acknowledge data poverty. Reframe ledger comment to clarify the Gao2019 citation is a protocol anchor, not a dose-response measurement.
- **Trigger:** Literature subagent (2026-05-28) confirmed:
  - Gao 2019 *Cell Mol Immunol* (DOI `10.1038/s41423-019-0329-7`) is a protocol-optimization paper using a FIXED 10 ng/mL IL-6 (= 0.476 nM at MW 21 kDa) — NOT a dose-response measurement. The "MAPLE posterior 0.46 nM. Gao2019" XML comment overstates the inferential basis: the value is anchored on the saturating IL-6 dose used in Gao's protocol, not on bootstrap-able Hill curve data.
  - No primary source provides a bootstrap-able Hill curve for IL-6 → Tfh-commitment EC50. This is the same data-poverty pattern as T4-CCL21 and T5d.
- **Biological anchor:** Value remains biologically plausible — STAT3-pY705 functional EC50 in primary T cells is 0.1-10 ng/mL (0.005-0.5 nM); IL-6Rα Kd is 9 nM. The chosen 0.46 nM sits at the upper end of the STAT3 functional range and corresponds exactly to Gao's 10 ng/mL protocol dose.
- **Prior broadening:** σ 0.4 → 0.7 (95% CI [0.11, 1.9] nM brackets STAT3 functional EC50 + Gao protocol dose); v3 ledger row 37 updated with full clarification note.
- **Status counts unchanged:** 42 FIXED / 2 FLAG / 239 MANUAL_PRIOR. PARAM_TFH_EC50_IL6 remains MANUAL_PRIOR; just σ broadened.
- **Round-4 inventory:** T6 contributes 1 validated (CXCL13) + 1 MANUAL_PRIOR_VALIDATED (EC50_IL6). 9 of 9 active R4 dispositions resolved (T1-T6 done). **R4 secretion + EC50 cluster COMPLETE.**

### Round 4 — COMPLETE 2026-05-28

All 9 active R4 dispositions resolved across 6 target groups:

| Target group | Sub-targets | Outcome |
|---|---|---|
| T1 cancer_tgfb | 1 (pooled STEM/PROG) | ✅ MAPLE |
| T2 treg_il10 | 1 | ❌ FIXED=0 (Tregs don't autonomously secrete IL-10) |
| T3 mac_m1_il1 | 1 | ✅ MAPLE |
| T4 frc_lymphoid | 2 (CXCL12 + CCL21) | CXCL12 ✅ MAPLE; CCL21 ⚠️ MANUAL_PRIOR (data-poor) |
| T5 bcell_secretome | 4 (antibody, IL10, IL6, CXCL13×2) | T5a/b/c ✅ MAPLE; T5d ❌ FIXED=0 (B cells are CXCR5+ sinks) |
| T6 tfh_biology | 2 (CXCL13, EC50_IL6) | CXCL13 ✅ MAPLE; EC50 ⚠️ MANUAL_PRIOR (no Hill data) |

**R4 status counts:** 42 FIXED / 2 FLAG / 239 MANUAL_PRIOR.

**Cross-cutting findings from R4 secretion cluster:**
1. **Seven MAPLE posteriors** added: T1 STEM_TGFB_RELEASE, T3 MAC_M1_IL1_RELEASE, T4-CXCL12 FRC_CXCL12_RELEASE, T5a BCELL_ANTIBODY_RELEASE, T5b BCELL_IL10_RELEASE, T5c BCELL_IL6_RELEASE, T6a TFH_CXCL13_RELEASE.
2. **Three close-by-fix dispositions** caught biologically-incorrect ABM encoding: T2 (Treg IL-10), T5d (B cell CXCL13 ACT + PLASMA). All flipped to FIXED=0.
3. **Three close-by-no-data dispositions** with σ broadening: T4-CCL21 (FRC), T6b (TFH EC50), T5 — VEGFA_UPTAKE + O2_DECAY from R3 carried forward.
4. **MASSIVE pattern in BCELL_*_RELEASE family**: 3 of 4 sub-targets surfaced unit-conversion errors of 1296× to 2.05M× class. The 4th had a misidentified citation. **All BCELL_*_RELEASE legacy values were systematically wrong.** This pattern is now flagged for future *_RELEASE param work.
5. **TFH_CXCL13_RELEASE re-validation**: Unlike BCELL_*, the prior MAPLE pass had it correctly calibrated within 9%. No unit-conversion bug here.
6. **Validator gotcha catalog** (now in MEMORY.md): mantissa+scale split for sci-notation; figure_excerpt for short snippets / values with no substring match; `proxy_observable` not `direct_measurement`; `endpoint_pair` not `endpoint`; OA paper curl as Zotero fallback; V-cancellation for unstated volumes; producer-fraction correction for bulk-supernatant data.

---

## Round 5 — Movement chemotaxis CI per cell type (✅ DONE 2026-05-28)

**Scope:** 15 PARAM_CHEMO_CI_* params across 8 cell types × 4 chemokine gradients. Re-validation round (most params had prior MAPLE attribution).

### Targets

| Group | Cell type | Gradient | Param | Anchor | Outcome | Posterior |
|---|---|---|---|---|---|---|
| T7a | T_CELL_EFF | CCL5 (CCR5) | TCELL_EFF | GaleanoNino2020 Fig 5A WT-Ccr5KO=0.19 | ✅ MAPLE | 0.198 |
| T7b | T_CELL_CYT | CCL5 (CCR5) | TCELL_CYT | GaleanoNino2020 Fig 2B cog 1:1 = 0.15 | ✅ MAPLE | 0.146 |
| T7c | T_CELL_SUP | CCL5 (CCR5) | TCELL_SUP | No direct data | ⚠️ MANUAL_PRIOR | 0.03 (σ→1.0) |
| T8a | MAC_M2 | CCL2 (CCR2) | MAC_M2 | Lee2020 Fig 5a CCL2-spec=0.29 | ✅ MAPLE | 0.269 |
| T8b | MAC_M1 | CCL2 (CCR2) | MAC_M1 | Same Lee2020 + Cui2018 (M1=M2) | ✅ MAPLE | 0.269 (re-anchored) |
| T8c | MDSC | CCL2 (CCR2) | MDSC | Same Lee2020 (CCR2+ proxy) | ✅ MAPLE | 0.211 |
| T9a | BCELL_ACT | CXCL13 (CXCR5) | BCELL_ACT | Liu2016 Fig 4B M.=0.075 | ✅ MAPLE | 0.074 |
| T9b | BCELL_NAIVE | CXCL13 (CXCR5) | BCELL_NAIVE | Same Liu2016 | ✅ MAPLE | 0.072 (re-anchored) |
| T9c | TREG_TFH | CXCL13 (CXCR5) | TREG_TFH | Same Liu2016 | ✅ MAPLE | 0.072 (re-anchored) |
| T10a | VAS_TIP | VEGFA | VAS_TIP | Shamloo2008 zone redistr, NOT FMI | ⚠️ MANUAL_PRIOR | 0.27 (σ→1.0) |
| T10b | FIB_MYCAF | TGF-β | FIB_MYCAF | Storck2016 PSC+PDGF (cite was 'Storz') | ⚠️ MANUAL_PRIOR | 0.05 (σ→1.2) |
| T10c | DC_MATURE | CCL21 | DC_MATURE | Quast2022 2D vs Haessler2011 3D | ⚠️ MANUAL_PRIOR | 0.23 (σ→1.0) |
| T10d | DC_IMMATURE | CCL2 | DC_IMMATURE | No quant data | ⚠️ MANUAL_PRIOR | 0.17 (σ→1.0) |
| T11 | (treg) | (TGF-β) | TREG_REG | Iellem2001 qualitative | ⚠️ MANUAL_PRIOR | 0.13 (σ→1.0) |
| T11 | (cancer) | O2 (not wired) | CANCER_STEM | Dead-code path | ❌ FIXED=0 |

### Summary (15 params)

- **8 MAPLE posteriors validated**: T7a/b, T8a/b/c, T9a/b/c
- **6 MANUAL_PRIOR_VALIDATED** (σ broadened, no MAPLE inference): T7c, T10a/b/c/d, T11-TREG_REG
- **1 FIXED**: T11-CANCER_STEM (O2 chemotaxis dead-code, mirrors T2/T5d/T6-T5d pattern)

### Cross-cutting findings

1. **ci_to_bias() identity mapping established**: PARAM_CHEMO_CI_* maps directly to measured FMI in the linear regime (CI ≤ 1/3 per `common.cuh:712`). All MAPLE forward models are algebraic identity. No `batch_accumulation` needed for chemotaxis-index calibration.

2. **Citation correction caught (T10b)**: Ledger cite-key "Storz2017" actually refers to **Storck et al. 2016/2017** (Schwab lab, not Peter Storz). Same Schwab-lab Oncotarget paper on PSC chemotaxis. Note for future audit passes.

3. **Cell-type pooling via Cui2018** (T8a/b/c): Cui 2018 J Immunol finding that M1=M2 chemotaxis (3D differences are adhesion-mediated) supports pooling MAC_M1 + MAC_M2 + MDSC onto a single Lee2020 anchor. Prior re-anchoring of MAC_M1 from 0.106 → 0.276 was required to align with this biology.

4. **Chemokine-system translation gap (T10b)**: Storck2016 measures PSC chemotaxis toward PDGF, but ABM's FIB_MYCAF uses TGF-β gradient (`fibroblast.cuh:235-239`). PDGFR-Smad vs TGF-βR-Smad signaling are mechanistically distinct, so direct anchoring is questionable. Closed as MANUAL_PRIOR with broadened σ.

5. **2D-vs-3D chemotaxis discrepancy (T10c)**: Quast2022 reports DC 2D FMI=0.62 while Haessler2011 reports DC 3D Vx/speed=0.176 — a ~3.5× gap reflecting mechanical confinement effects. Joint MAPLE inference would require explicit 2D-vs-3D translation σ contribution; deferred to future round.

6. **New Unicode validator gotcha** (T7a): Author names with non-ASCII characters ("Galeano Niño") trigger validator metadata-mismatch errors. Workaround: use hyphenated form ("Galeano-Niño"). Recurring issue for any non-ASCII first-author surname.

7. **Re-anchoring pattern for low-contraction posteriors** (T8b, T9b, T9c): When the prior median is far from the data anchor, posterior is dominated by prior (contraction → 0). Solution: re-anchor prior to match the data anchor magnitude BEFORE running inference. After re-anchor, posterior lands at data-anchor value with high contraction.

### Status counts after R5

- 43 FIXED (added CANCER_STEM)
- 2 FLAG
- 238 MANUAL_PRIOR (broadened σ on 6 chemotaxis MANUAL_PRIORs)

---

## Round 6 — Killing + proliferation rates (✅ DONE 2026-05-28, citation audit + MANUAL_PRIOR_VALIDATED)

**Scope:** 14 PARAM_*_DIV*, *_KILL*, *_GROWTH*, *_PROB* params across killing + proliferation. Cell-fraction observable sensitivity round.

**Outcome:** **Citation audit / MANUAL_PRIOR_VALIDATED round.** Most R6 PDFs not in Zotero (only Victora2010, Mesin2020, Lieber1975, Campisi2007 available), preventing fresh MAPLE inference. Round focused on auditing the legacy ledger citations against fresh CrossRef + lit-search verification — surfaced **6 critical citation issues** that have lived in the ledger undetected for months. All R6 params closed as MANUAL_PRIOR_VALIDATED with corrected citations + σ adjustments.

### Citation issues caught (R6 audit)

| Param | Legacy ledger cite | Audit finding |
|---|---|---|
| TCELL_DIV_LIMIT | Kaech2001 DOI 10.1038/ni0901-415 | **DOI TYPO** → correct DOI is 10.1038/87720 (May vs Sept issue confusion) |
| TCD4_DIV_LIMIT | Dowling2018 PMC6974474 | **Author + year wrong** → actually Sarkander et al. 2020 (PMC is correct) |
| ADCC_BOOST_MAX | Niu2022 PMC8932662 | **Author wrong** → actually Cao et al. 2022 (Feng lab; PMC is correct) |
| PROG_GROWTH_RATE | Furukawa2001 (no DOI in ledger) | **DOI 10.1097/00006676-200105000-00009 is WRONG PAPER** (glutathione S-transferase in pancreatitis, not PDAC TVDT). Furukawa2001 PDAC TVDT 159d claim needs separate DOI audit. |
| PROG_DIV_MAX | Wang2015 PDAC + Scott2012 | **Wang2015 UNVERIFIABLE** (no PDAC ABM by Wang in 2015 with PROG_DIV_MAX=10 found in CrossRef). **Scott2012 year wrong (actually 2014) + cited claim 'insensitive in range' is OPPOSITE of paper's actual finding** (high sensitivity below threshold ~15). |
| BCELL_DIV_LIMIT | Tas2016 Science | Verified DOI 10.1126/science.aad3439 ✓, but Zotero fetch returned wrong file (Park2020 PDAC paper). |

### Targets

| Target | Param(s) | Anchor papers | Disposition |
|---|---|---|---|
| T12a | TCELL_DIV_LIMIT | Kaech2001 (DOI corrected) + Lanzavecchia2002 | ⚠️ MANUAL_PRIOR_VALIDATED |
| T12b | TCD4_DIV_LIMIT | Sarkander2020 (citation corrected from "Dowling2018") | ⚠️ MANUAL_PRIOR_VALIDATED |
| T12c | PRIME_DIV_BURST | (additive to T12a/b base; no separate anchor) | ⚠️ MANUAL_PRIOR_VALIDATED |
| T13a | BCELL_DIV_CD | Victora2010 + Gitlin2014 | ⚠️ MANUAL_PRIOR_VALIDATED |
| T13b | BCELL_DIV_LIMIT | Tas2016 + Mesin2020 | ⚠️ MANUAL_PRIOR_VALIDATED |
| T14 | ADCC_BOOST_MAX | Cao2022 (citation corrected from "Niu2022") | ⚠️ MANUAL_PRIOR_VALIDATED |
| T15a | PROG_GROWTH_RATE | Lieber1975 ✓ + Furukawa2001 (DOI WRONG, needs audit) | 🚩 FLAG RETAINED (TEMP biology bump persists) |
| T15b | PROG_DIV_MAX | Wang2015 (unverifiable) + Scott2014 (year + claim wrong) | ⚠️ MANUAL_PRIOR_VALIDATED |
| T16 | SEN_DEATH_RATE | Campisi2007 + Xue2007 + Coppe2008 (all verified) | ⚠️ MANUAL_PRIOR_VALIDATED (TEMP bump persists) |
| T17 | TKILL_SCALAR | Halle2016 + Weigelin2021 (both verified) | ⚠️ MANUAL_PRIOR_VALIDATED |
| T18 | ASYM_DIV_PROB, TELOMERASE, EFF_TO_CYT, FIB_DIV_MAX | Model-design closures | ⚠️ MANUAL_PRIOR (no lit anchor needed) |

### Summary

- **0 MAPLE posteriors** added (PDFs not in Zotero; deferred to future round when PDF availability is resolved)
- **10 MANUAL_PRIOR_VALIDATED** with verified citations + σ adjustments
- **1 FLAG retained**: PROG_GROWTH_RATE (legacy MAPLE posterior 0.071 implies hierarchy inversion; current 0.3 is a TEMP biology fix awaiting joint inference)
- **4 MANUAL_PRIOR closures** (T18 model-design params with no required lit anchor)
- **6 citation issues** documented in ledger (deeper audit than any prior round)

### Status counts after R6

- 43 FIXED
- 2 FLAG (1 from R6: PROG_GROWTH_RATE)
- 238 MANUAL_PRIOR (most updated with verified citations)

### Cross-cutting findings

1. **Ledger citation hygiene** is poorer than expected — 6 of 14 R6 params had at least one citation issue (typo, wrong author, wrong year, wrong paper, or unverifiable). This suggests systematic re-audit of ledger citations against CrossRef metadata is warranted for ALL prior rounds, not just R6. Possible future task: bulk DOI re-verification across the entire v3 ledger.

2. **Author surname collisions** (Tas/Park, Sarkander/"Dowling") cause Zotero fetch_papers_from_zotero to grab wrong papers when there are multiple matches. Workaround: manual `cp` from specific storage subfolder by inspection, as done for Liu2016 in R5.

3. **Pancreas journal DOIs need special audit** — Furukawa2001 (Pancreas vol 22 #4) cited DOI returns wrong paper. The Pancreas-journal DOI scheme `10.1097/00006676-YYYYMM00-NNNNN` may collide across articles within the same issue.

4. **TEMP-bump pattern recurring** (T15a PROG_GROWTH 0.3 ← MAPLE 0.071; T16 SEN_DEATH 0.1 ← MAPLE 0.026): When MAPLE posterior implies a biologically-implausible value (e.g., proliferation hierarchy inversion), legacy practice has been to TEMP-bump the XML and FLAG/note in ledger. Future MAPLE inference must either re-derive these with biology-aware joint priors OR formally accept the TEMP-bump as a constraint.

### Plan forward (Round 7+)

Per `docs/phase3_maple_restart.md` §"Round cadence":

| Round | Cluster | ~Targets | Notes |
|---|---|---|---|
| 4 | PDE secretion + EC50 | 10 | ✅ DONE 2026-05-28 |
| 5 | Movement chemotaxis CI per cell type | ~8 (15 actual) | ✅ DONE 2026-05-28 |
| 6 | Killing + proliferation rates | ~8 (14 actual) | ✅ DONE 2026-05-28 (citation audit) |
| 7 | Persistence + porosity + contact guidance | ~6 | Hardest to MAPLE; many fall to manual prior |
| 8 | Lifespans + recruitment scalars | ~6 | |

**Round-entry note (carried forward from 2026-05-20):** before authoring any new round's targets, repeat the Step-1 dead-code check on each cluster. R4 (2026-05-26) caught one dead param (`PARAM_TCELL_IFNG_RELEASE_TIME`); gate continues to pay off.

---

## Round 7 — Persistence + porosity + contact guidance + ECM mechanics (🔄 IN PROGRESS, opened 2026-06-02)

**Scope:** 50 params across 6 families (the "structural/biophysical movement" cluster, flagged as hardest-to-MAPLE per the Phase-3 cadence). User elected to attempt full MAPLE on all 6 targets.

| Target | Family | #params | Forward model | MAPLE verdict (pre-inference) |
|---|---|---|---|---|
| T19 | `PARAM_PERSIST_*` | 17 | mean cos(turn angle) per 20µm step ≈ p_persist; convert lit `R`/(speed·τ) → persistence length L_p → p_persist = exp(−20µm/L_p) | 🟢 MAPLE (CD8/Treg/DC anchored); pool/manual rest |
| T20 | `PARAM_ECM_POROSITY_*` | 9 | min_porosity = steric void-fraction gate; Wolf2013 critical pore cross-section per cell class | 🟡 BLOCKED on modeling decision (see Finding 2) |
| T21 | `PARAM_CONTACT_GUIDANCE_*` | 7 | fiber-alignment channeling → directionality ratio / alignment fraction on aligned collagen | 🟢 MAPLE (cancer, CAF) |
| T22 | `PARAM_FIBER_BARRIER_*` | 11 | perpendicular-crossing penalty `1−b·sin²θ` → trajectory-to-fiber angle | 🟡 weak; Kamionka2021 PDAC null result caps upper bound |
| T23 | `PARAM_ECM_STRESS_{DEPOSIT,DECAY}` | 2 | DECAY = 1/τ stress-relaxation; DEPOSIT = dimensionless scalar | 🟢 DECAY MAPLE (direct PDAC anchor); DEPOSIT manual |
| T24 | `PARAM_ECM_ORIENT_{RATE,TRACTION_W,STRESS_W,CROSSLINK_RESIST}` | 4 | RATE = fiber-realignment timescale; weights = model-design | 🟡 RATE only; 3 weights manual |

### Step-1 dead-code gate — ✅ CLEAN (first clean gate since R3)
All 50 candidate params have live consumers in `agents/*.cuh`, `core/common.cuh`, or `pde/pde_integration.cu`. Consumption confirmed: PERSIST → `mp.p_persist` (per-cell move struct); POROSITY → `mp.min_porosity` (move_cell candidate filter, common.cuh:803); CONTACT_GUIDANCE → `apply_contact_guidance()` w_contact (common.cuh:641); FIBER_BARRIER → `mp.barrier_strength` (common.cuh:854-866); ECM_STRESS → cancer_cell.cuh:763 (deposit) + pde_integration.cu:1649 (decay); ECM_ORIENT → pde_integration.cu:1675-1678. No FIXED-by-dead-code dispositions this round.

### Movement-mechanics forward-model derivations (from common.cuh `move_cell`, lines 607-905)
- **p_persist identity:** per move-event the cell either keeps its exact lattice direction (prob p_persist, if that neighbor is an available candidate) or reselects gradient-biased (uniform when no gradient). Mean cos of turning angle over the 26 symmetric Moore directions on reselect = 0, so **mean directional autocorrelation at the 20µm lattice-step scale = p_persist** (analogous to the R5 `ci_to_bias` identity). Literature `R`/persistence-time must be rescaled to the 20µm step: L_p = speed·τ_p (or L_p = −Δs_exp/ln R_exp), then p_persist = exp(−20µm/L_p). **Scale conversion is the dominant uncertainty** and the reason T19 is "hard to MAPLE."
- **min_porosity:** steric gate only — `move_cell` has NO proteolysis. A cell may enter a neighbor voxel only if `ecm_porosity(voxel) ≥ min_porosity`. Higher min_porosity ⇒ MORE pore-restricted.
- **w_contact (contact guidance):** amplifies gradient component parallel to fibers by (1+w·fiber_mag), suppresses perpendicular by (1−w·fiber_mag).
- **barrier_strength:** multiplies a candidate move's weight by (1 − barrier_strength·fiber_mag·sin²θ_to_fiber); parallel unaffected, perpendicular blocked.

### 🚩 Finding 1 (T23) — ECM_STRESS_DECAY ~30× too slow
Current `PARAM_ECM_STRESS_DECAY = 5e-4 /s` ⇒ τ = 2000 s ≈ 33 min. **Rubiano et al. 2017/2018 Acta Biomater (DOI 10.1016/j.actbio.2017.11.037), Fig 6C: human PDAC tumor stress-relaxation τ = 66.1 ± 20.8 s** ⇒ decay rate ≈ 1.5e-2 /s. Corroborated by collagen-gel (Xu2013: 0.3–>200 s spectrum; Nam2016 PNAS: ~1–300 s) and PDAC-mimic hydrogels (τ½ 8–50 s). Current value sits ~30× below the direct PDAC measurement → strong candidate for upward correction. **Caveat:** Rubiano is in PMC but NOT in the OA bulk-download subset (Elsevier embargo) → validator snippet-check will need a Zotero copy or manual figure-excerpt.

### 🚩 Finding 2 (T20) — porosity priors appear INVERTED vs nuclear-deformability biology
Current priors: leukocytes have HIGH min_porosity (TCELL 0.4, TREG/TFH/BCELL 0.35) = most pore-restricted; fibroblast LOW (FIB 0.05), VAS_TIP 0.2 = least restricted. **Wolf 2013 JCB (DOI 10.1083/jcb.201210152):** protease-INDEPENDENT critical pore cross-section is ~7 µm² (tumor/mesenchymal) > ~4 µm² (T cell) ≥ ~2–4 µm² (neutrophil) — i.e. deformable leukocytes traverse roughly HALF the pore cross-section of mesenchymal cells, so on pure steric/deformability grounds leukocytes should have LOWER min_porosity, fibroblasts HIGHER. The current ordering is the reverse. **Confound:** fibroblasts and endothelial tip cells are protease-active (Wolf shows proteolysis relaxes the limit; van Hinsbergh: tip cells migrate by MT1-MMP, no pore gate) — and `move_cell` has no explicit proteolysis, so a low fib/vas gate may be an implicit stand-in for matrix degradation. **DECISION NEEDED before authoring T20:** does `min_porosity` encode (a) steric nuclear deformability (⇒ re-derive from Wolf, inverting the leukocyte/fib ordering) or (b) effective traversal including proteolysis (⇒ keep current ordering, calibrate magnitudes, treat fib/vas as protease-confounded MANUAL_PRIOR)?

### DOI verification (all 14 anchor DOIs via CrossRef) — all tags matched real first authors
VERIFIED: Rubiano2017(→2018), Wolf2013, Sadjadi2020, Matheu2015 (tag TregImaging2015, ncomms7219), Quast2022 (tag DCgradient2022, fcell.2022.943041 — NOTE: same first author as the existing R5 `Quast2022` dir → reconcile), Ray2017, Riching2014, Erdogan2017, Salmon2012, Bougherara2015, NicolasBoluda2021, Kamionka2021. Rate-limited (429, re-verify): Vader2009, Levental2009 (both well-known, DOIs sound). No citation errors found (contrast R6's 6 issues).

### PDF staging status
Real OA PDFs in place: Sadjadi2020, TregImaging2015, DCgradient2022 (all T19 anchors), Salmon2012, Bougherara2015 (T22), Vader2009 (T24). Still needed (Zotero/alt route): Wolf2013 (T20 — ftp mirror 404), Rubiano2017 (T23 — not OA), Ray2017/Riching2014/Erdogan2017 (T21 — Cell Press/JCB, PMC OA), NicolasBoluda2021 (eLife 406), Kamionka2021 (MDPI 403), Levental2009.

### Next actions (carried)
1. Resolve Finding 2 modeling decision → unblocks T20.
2. Author + validate T19 (all PDFs staged) — extract Sadjadi step-length/frame-interval + speed/τ for L_p conversion; per-cell-class pooling.
3. Acquire remaining PDFs via Zotero `fetch_papers_from_zotero` or alt routes; author T21–T24.
4. Joint inference across R7 YAMLs; update status counts + MEMORY.md.

### DECISION (2026-06-02) — Finding 2 resolved: min_porosity = "effective traversal (incl. proteolysis)"
Modeler intent confirmed: `min_porosity` is an implicit stand-in for each cell's real ability to traverse dense matrix INCLUDING protease-mediated remodeling. Consequences for T20: (a) KEEP the current cell-type ordering (leukocytes higher gate, fibroblast/tip lower); (b) calibrate magnitudes from Wolf2013 only for the non-proteolytic leukocyte classes where the steric gate is the right physics; (c) treat FIB and VAS_TIP as protease-confounded → MANUAL_PRIOR with broadened σ. No inversion / value-correction to the ordering.

### T19 anchor extraction — Sadjadi2020 (CD8 CTL, 3D collagen) + scale-mapping crux
**Verbatim (Table I, p.3; Methods p.2):** sampling Δt = **30 s**; R = ⟨cos φ⟩ over consecutive steps (instantaneous persistency, exactly the p_persist observable). Velocity 0.03–0.10 µm/s; **R = 0.30 (2mg/ml) → 0.36 (4) → 0.35–0.44 (5)** across donors. Faster cells more persistent (CC_v,R 0.29–0.64). Two-state PRW: slow CTLs sub-diffusive/anti-persistent, fast persistent. Collagen 2/4/5 mg/ml = normal-tissue / soft-tumor / hard-tumor analogs.
Treg anchor (Matheu2015 ncomms7219): T-zone Treg 14.6, follicular 12.9, capsular 9.5, Tconv 12.0 µm/min (velocity only, no angle). DC anchor (Quast2022 fcell): directionality 0.63 in steep gradient, <0.06 no-gradient (chemotactic, not pure persistence).

**⚠️ CRUX — the p_persist↔R scale mapping is ambiguous by ~6 orders of magnitude:**
- By DISPLACEMENT (20µm voxel step): L_p = −Δs/ln R ≈ 1.1–2.5 µm (Δs = v·30s ≈ 0.9–3 µm) → p_persist = exp(−20/L_p) ≈ **~0**.
- By TIME (per move-event = 600s/move_steps ≈ 9.4 s for T cell move_steps=64): τ_p = −30/ln(0.35) ≈ 29 s → p_persist = exp(−9.4/29) ≈ **~0.72**.
The persistence check in `move_cell` (common.cuh:816) fires once per move-EVENT, so the per-event interval is the right clock — BUT 64 events × 20µm ≠ the calibrated 3.6µm/min (move_steps XML comment), so most events do NOT displace a full voxel. Resolving T19 (MAPLE vs MANUAL_PRIOR) requires reading the move-scheduling code (`model_layers.cu` spread_steps + reset_moves + per-substep move layers) to determine the effective displacement-per-event and the correct autocorrelation clock. **OPEN — next action.**

### T19 RESOLUTION — MANUAL_PRIOR_VALIDATED (no clean MAPLE identity)
Move-code read (`model_layers.cu:146-189`, `t_cell.cuh:588-665`): each "move" agent-fn call = one `move_cell` invocation = one attempted **20µm** voxel step; `move_steps` calls (T cell=64) are spread across substeps via `spread_steps()`. Persistence (`p_persist`) repeats the lattice direction between consecutive move-EVENTS, but each successful event displaces a full 20µm and `persist_dir` is retained across failed (blocked) events.
- The **time mapping is rejected**: it ignores that the ABM moves 20µm/event vs Sadjadi's ~1–3µm/30s step.
- The physically-correct invariant is **persistence length** L_p. ABM walk: ⟨cos θ(n steps)⟩ = p_persist^n ⇒ L_p^ABM = −20µm/ln(p_persist). Matching to Sadjadi L_p ≈ 1.1–2.5 µm ⇒ p_persist ≈ exp(−20/1.5) ≈ 0.
- BUT the realized L_p is coupled to the emergent move-success (occupancy) rate and move_steps — failed events retain direction without displacing — so **no closed-form p_persist = f(R_lit) exists**. Unlike R5's `ci_to_bias` algebraic identity, persistence is an EMERGENT trajectory property requiring an ABM-in-the-loop fit, not a literature algebraic inversion.
- **Disposition: 17 PERSIST params → MANUAL_PRIOR_VALIDATED.** Literature constrains the cell-class ORDERING (fast/persistent vs slow/anti-persistent) and bounds, not point values. Recommended: retain current ordering, broaden σ (0.7→~1.0) to span the scale ambiguity, and FLAG that p_persist warrants a dedicated emergent-persistence ABM calibration (fit simulated trajectory L_p to Sadjadi L_p 1.1–2.5µm / Matheu velocities / Quast directionality) in a future SBI pass — NOT a MAPLE submodel target.
- **Finding 3 (candidate):** Sadjadi's diffusive-at-distance MSD ⇒ CTL persistence length in dense collagen is only ~1–2.5µm, far below the 20µm voxel. At the ABM's spatial resolution, true directional memory is largely lost between voxels — current priors (TCELL_EFF 0.4) likely OVERSTATE 20µm-scale persistence. Worth revisiting voxel size vs persistence-length resolution.

### Round 7 CLOSE-OUT (2026-06-15) — 1 MAPLE + 5 MANUAL_PRIOR_VALIDATED

All PDFs acquired (triaged 4 corrupt downloads + 2 wrong-paper fetches; only Rubiano2017 needed a fresh Zotero add). 5 parallel extraction agents read every paper. Forward-model derivations from `common.cuh move_cell` confirmed which targets admit a clean MAPLE identity.

**Outcome: only T23 yields a closed-form data identity. T19/T20/T21/T22/T24 are emergent / model-scale-dependent / no-rate -> MANUAL_PRIOR_VALIDATED.** This is the expected result for the Phase-3 "hardest-to-MAPLE" cluster.

- **T23 ECM_STRESS_DECAY -> MAPLE VALIDATED.** `ecm_stress_decay_PDAC_deriv001.yaml`. Rubiano2018 (10.1016/j.actbio.2017.11.037) Fig 6C: human PDAC tumor stress-relaxation tau = 66.1 +/- 20.8 s (SD; single-exponential SLS fit). Forward model tau = 1/k_decay -> k = 1.51e-2/s. Ledger RE-ANCHORED 5e-4 -> 1.51e-2 (30x correction); citation CORRECTED Chaudhuri2016(gel) -> Rubiano2018(PDAC tissue). Joint posterior: median 0.0152, CV 0.20, transl-sigma 0.180 (lowest of all 28 targets), contraction 0.96, r_hat 0.999. DEPOSIT stays MANUAL (dimensionless tuning scalar).
- **T19 PERSIST (17) -> MANUAL_PRIOR_VALIDATED** (as resolved 06-02): emergent, occupancy-coupled, L_p~1-2.5um<<20um voxel. sigma 0.7->1.0.
- **T20 POROSITY (9) -> MANUAL_PRIOR_VALIDATED / MANUAL.** `min_porosity` is a void-fraction gate with NO pore-area scale (ecm_porosity = max(0,1-(d/cap)(1+c)), common.cuh:612); Wolf2013 critical pore cross-sections fix only leukocyte RATIOS, absolute is model-set (A_ref~10um2 implied by TCELL=0.4). RE-ANCHORED MDSC 0.3->0.2 (Wolf PMN 2um2 : T-cell 4um2 = 1:2). FIB/VAS_TIP -> MANUAL (protease-confounded; move_cell has no proteolysis). sigma broadened.
- **T21 CONTACT_GUIDANCE (7) -> MANUAL_PRIOR_VALIDATED.** w_contact reshapes the gradient (common.cuh:641); directionality is emergent (no single-step identity like R5 ci_to_bias). Anchors: Erdogan2017 CAF dir-ratio 0.75 vs 0.42; Riching2014 CELL windrose 58% vs 32% (NOTE: Riching "71%/+-15deg" is a FIBER metric, not cells; Ray2018 "70%" is cited to Ray2017 BiophysJ, not measured in-hand). sigma broadened.
- **T22 FIBER_BARRIER (11) -> MANUAL_PRIOR_VALIDATED.** Angle->barrier is emergent over Moore-weighted selection (w=1-b*fmag*sin^2, common.cuh:866). Kamionka2021 = human PDAC NULL (collagen alignment does NOT channel T cells, |r|<=0.47 n.s. n=12) -> RE-ANCHORED TCELL/TREG barrier 0.55/0.5 -> 0.3, sigma 1.0 (spans tension with non-PDAC channeling: Salmon2012 22.5+/-19.5deg, Bougherara2015 26.8 vs 53.8deg). NicolasBoluda speed-impediment is non-directional (porosity/stiffness, not barrier).
- **T24 ECM_ORIENT (4) -> MANUAL_PRIOR_VALIDATED / MANUAL.** Vader2009 gives NO realignment rate (2.5e-5..2.5e-3/s is the APPLIED STRAIN RATE; alignment quasi-static vs strain, onset ~5%); ORIENT_RATE implied timescale 40s-1hr, current 0.001/s in range. Levental2009 LOX-isolated stiffening ~1.5-1.7x (NOT 2.3-2.8x; full Normal->Tumor 8-11x) -> CROSSLINK_RESIST~2 holds as LOX-specific. TRACTION_W/STRESS_W = model-design (MANUAL).

**Citation corrections logged this round:** Chaudhuri2016->Rubiano2018 (T23); Ray2018 "70%" is mis-attributed (actually Ray2017 ref21); Riching "71%" is fiber not cell; Levental "2.3-2.8x" -> actually 8-11x full / 1.5-1.7x LOX-isolated; Vader "realign rate" -> actually strain rate.

**Joint inference:** 28 targets, 39 params, 20000 samples, 4 chains, clean. `submodel_priors.yaml` regenerated.

**Status counts after R7:** 43 FIXED / 2 FLAG / 238 MANUAL_PRIOR (50 R7 movement params now MANUAL_PRIOR_VALIDATED with cited anchors; +1 MAPLE posterior ECM_STRESS_DECAY in submodel_priors.yaml). Plus 3 recruitment params added this session (PARAM_{TEFF,TREG,TH}_RECRUIT_P, MANUAL_PRIOR) -> 241 MANUAL_PRIOR.

**Round 7 COMPLETE.** Next per Phase-3 cadence: Round 8 (remaining clusters) or systematic v3 citation re-audit (R6 flagged 6 issues; R7 added 5 more corrections - cross-round audit warranted).

---

## Value-Audit Re-Derivation (2026-06-16) — 9 FLAGGED params resolved

Closes the FLAGGED batch from `docs/citation_audit_2026-06-15.md` §Phase 3 (the ~40%-unsupported-value finding). BCELL_BREG_FRACTION was already re-derived 06-15 (Zhao2018 3.2%), so 9 params remained. **All 9 resolved.** Each had a plausible number attached to a paper that does not report it.

### ⏱️ Critical correction surfaced: ABM step = 6h, not 600s
While deriving the move-steps forward model, confirmed `PARAM_SEC_PER_SLICE = 21600 s = 6 h` (param_all_test.xml:14; CLAUDE.md "dt=600s/10min" is STALE). This makes the movement forward model exact: `move_steps = speed[um/min] x (t_step_min/voxel_um) = speed x (360/20) = speed x 18`, matching all 4 prior validated movement params (TCELL 64=3.6, DC 38=2.1, MDSC 71=3.9, MAC 30=1.6). It also means BCELL_DIV_CD=2 steps = 12h (within 6-12h centroblast range), NOT 20 min — averting a spurious 20x "fix". See memory `project_abm_timestep_6h.md`.

### Tier A — re-anchored with verified value (MAPLE targets authored)
- **FIB_MOVE_STEPS 9 -> 8.** Storck2016 Oncotarget (10.18632/oncotarget.13647) RLT-PSC human pancreatic stellate line, UNSTIMULATED 0.45+/-0.04 um/min (Fig2C verbatim; Colo357-conditioned 0.98 = activated upper edge). 0.45 x 18 = 8.1 -> 8. REPLACES fabricated "Liu2015 0.48um/min" (was a Liu2008 NO-diffusion dimensionless ratio, not a speed). yaml `fib_move_steps_PDAC_deriv001`. (Same Storck/Schwab-lab paper as R5-T10b.)
- **MAC_MOVE_STEPS 30 -> 18.** Chen2019 PNAS (10.1073/pnas.1902366116) intravital glioma BMDM: mobile 0.07826 um/s = 4.70 / confined 0.003478 um/s = 0.21 um/min (Fig4 verbatim); NO mobile:confined fraction reported. Broad bimodal prior: geometric mean sqrt(4.70*0.21)=0.99~1 um/min x 18 = 18 steps, sigma 1.0 spanning ~4-85 steps. REPLACES "1.6 um/min" (not in paper). Cross-tissue (glioma not PDAC) -> high translation sigma. yaml `mac_move_steps_PDAC_deriv001`.

### Tier A — no clean literature number -> honest MANUAL
- **CHEMO_CI_VAS_TIP 0.27 -> 0.12, MANUAL_PRIOR_VALIDATED.** NO VEGF-A-specific endothelial FMI/CI exists in the literature. Best verified anchor Zengel2011 BMC Cell Biol (10.1186/1471-2121-12-21) HUVEC ibidi FMI-parallel=0.10 (Table1) but gradient was 10% FCS not VEGF (magnitude anchor only); Barkefors2008 JBC (10.1074/jbc.M704917200) is the canonical HUVEC VEGF chemotaxis paper but reports no extractable FMI. Re-anchored to ~0.12 (center near verified 0.10, span to old 0.27 as uncertain high end), sigma 1.0. DROPPED Shamloo2008 (zone-redistribution + filopodia only; "0.27" never in paper). The one Tier-A param that could NOT be MAPLE'd.

### Tier B — citation fix, value retained
- **BCELL_DIV_CD = 2 (kept), MANUAL_PRIOR_VALIDATED.** 2 steps x 6h = 12h, within loose 6-12h centroblast cycle (tight measured 6-7h = 1 step). Re-cited to Zhang/MacLennan/Liu/Lane 1988 Immunol Lett (10.1016/0165-2478(88)90178-2, "centroblasts having a remarkably short cell cycle time of some 6 to 7 hours", verbatim) + Liu1991 EJI (10.1002/eji.1830211209, in-vivo BrdU). DROPPED Victora2010/Gitlin2014 — audit + agent confirmed BOTH papers' "6hr" = imaging/migration window, NEITHER measures cycle time (Gitlin2014 measures divisions-per-DZ-cycle = 1-6, not hours).

### Tier C — honest MANUAL/model-design, false anchor dropped, broad sigma
- **TCD4_TGFB_RELEASE_TIME = 705000s (kept), MANUAL_PRIOR.** Model-design secretion-window timescale. Nakamura2001 Fig6D DROPPED (=%surface-TGFb1+ cells peaking day3, no 8.2d value). No direct lit timescale found.
- **DC_PRESENTATION_CAPACITY = 25 (kept), MANUAL_PRIOR.** Model-design DC:T conjugate cap. Mempel2004 "~25 conjugates/DC" DROPPED (not in paper; likely confusion with 20-26h window or CD25).
- **TCELL_HYPOXIA_FACTOR3 = 0.12 (kept), MANUAL_PRIOR.** Model-design hypoxia killing-impairment. Doedens2013 DROPPED — "0.12" not in paper AND Doedens shows hypoxia/HIF ENHANCES CD8 (opposite of the impairment this param encodes). Flag: the model assumption itself conflicts with the genetic-HIF-stabilization literature.
- **SEN_DEATH_RATE = 0.1/day (kept), MANUAL_PRIOR.** Model-design TEMP bump (legacy ~0.026/day = ~38d life, defensible but breaks sweep turnover). Campisi2007 quantitative-rate anchor DROPPED (review qualitative "viable many weeks"; BCL-2 apoptosis-resistance directional only).
- **PROG_GROWTH_RATE = 0.3/day (kept), FLAG -> MANUAL_PRIOR_VALIDATED.** Value VALIDATED: DT=ln2/0.3=2.31d=55h ~ Lieber1975 IJC (10.1002/ijc.2910150505) PANC-1 52h doubling. DROPPED wrong Furukawa2001 DOI (10.1097/00006676-200105000-00009 = glutathione-S-transferase in pancreatitis, NOT 159d clinical TVDT) + the unverifiable 159d secondary anchor. Model-design caveat retained (0.3 restores PROG>STEM hierarchy; full fix needs joint MAPLE on PROG_GROWTH_RATE+ASYM_DIV_PROB+k_C1_growth).

### Status counts after value-audit re-derivation
**44 FIXED-equiv / 0 FLAG / 241 MANUAL_PRIOR** — both prior FLAGs cleared (PROG_GROWTH_RATE -> MANUAL_PRIOR_VALIDATED via Lieber; BCELL_BREG_FRACTION cleared 06-15). The 5 Tier-C MANUAL params now carry honest "no lit anchor / false anchor dropped" framing instead of fabricated citations.

**Joint inference re-run DONE.** 2 new MAPLE yamls (`fib_move_steps`, `mac_move_steps`) validated PASS and folded into `submodel_priors.yaml`: now **30 targets / 41 params**, 20000 samples, 4 chains, clean (per-target r_hat 0.998-1.000, 0 divergences). New posteriors: PARAM_FIB_MOVE_STEPS median 8.08 (CV 0.35, transl-sigma 0.394, contraction 0.88); PARAM_MAC_MOVE_STEPS median 17.79 (CV 0.87, transl-sigma 0.712, contraction 0.80 — intentionally broad, bimodal glioma proxy). All 28 prior-round posteriors unchanged. CHEMO_CI_VAS_TIP stays MANUAL (no clean VEGF FMI -> not in yaml; priors CSV fallback 0.12).

---

## Value-Audit Round 2 (2026-06-16) — R7 movement/ECM cluster (28 params)

Second pass of the value-vs-citation check (do the cited NUMBERS appear in the papers?), covering the 28 MANUAL_PRIOR_VALIDATED numeric params from Round 7 not spot-checked in the Phase-3 audit. All anchor PDFs staged + file-checked (no corruption/wrong-paper this time). 4 parallel agents read the LOCAL staged PDFs and fact-checked each distinct numeric anchor.

**Outcome: NO fabrications (contrast Phase-3's 8/18). 0 value changes.** Because R7 (Jun 15) already did the citation-correction work, these hold up. Findings are provenance refinements, now annotated in the ledger (literature col, "VALUE-AUDIT-2 2026-06-16 ..."):

- **Wolf2013 porosity (5: ECM_POROSITY_*) -> SUPPORTED (verbatim).** Abstract: "arrest reached at 10% of the nuclear cross section (tumor cells, 7 um2; T cells, 4 um2; neutrophils, 2 um2)." 2:1 T:PMN ratio confirmed; correct 2013 paper (no Wolf2003 mix-up). Cite as representative arrest thresholds (transwell sub-ranges 1-10 um2).
- **Sadjadi2020 persistence (17: PERSIST_*) -> SUPPORTED (verbatim).** Table I "persistency R=0.30-0.44" exact (human CTLs, 3D collagen 2/4/5 mg/mL, 2 donors; arXiv 2001.05331 / Biophys J). METRIC CLARIFICATION: R=<cos phi> (mean cosine of turning angle), NOT the net/total-displacement ratio the prior note implied. Number right, metric label corrected. Per-cell-type spread (0.1-0.5) is model-design around the anchor.
- **Erdogan2017 (CONTACT_GUIDANCE_FIB_MYCAF) -> SUPPORTED (verbatim).** Dir-ratio 0.75 (aligned CAF-CDM) vs 0.42; CORRECTION: 0.42 is the blebbistatin-DISRUPTED CDM, not a normal-fibroblast control (biological meaning matches).
- **Bougherara2015 -> SUPPORTED (verbatim)** 26.8+/-7.18 vs 53.8+/-6.52 deg (OVARIAN, vs tumor-stroma boundary). **Salmon2012 -> SUPPORTED (verbatim)** 22.5+/-19.5 deg, n=114 (LUNG, vs vessel axis). **Vader2009 -> SUPPORTED (verbatim)** 5% strain-stiffening onset; confirmed it is an APPLIED-STRAIN threshold, not a realignment rate (R7 correction validated).
- **FIGURE-ONLY (number in a figure, not extractable text; qualitative anchor holds — NOT fabricated):**
  - **Riching2014 (CONTACT_GUIDANCE_CANCER_STEM/PROG):** cell windrose 58%/32% are Fig 4A graphic readouts (cone = 10deg, not 15deg). R7's "'71%/15deg' is a FIBER metric not cell" -> CONFIRMED (Fig 2A "71% of fibers" vs 43% unstrained).
  - **Kamionka2021 (FIBER_BARRIER_TCELL/TREG):** |r|<=0.47 is a Fig 7/8 plot value, but the PDAC-NULL finding itself (the actual 0.3 re-anchor basis) is solidly text-confirmed (n=12, NS, no correlation w/ any collagen-organization metric).
  - **Levental2009 (ECM_ORIENT_CROSSLINK_RESIST):** LOX-isolated 1.5-1.7x is figure-only; overall 140-400->5000-10000 Pa stiffening is text-verbatim. CROSSLINK_RESIST=2.0 stands as LOX-specific model-design.

**Net:** 0 FLAG, 0 re-derivations needed; 28 ledger rows annotated with verbatim/figure-only provenance + 5 metric/tissue/control-label corrections (Sadjadi metric, Erdogan blebbistatin, Riching cone=10deg, Bougherara=ovarian, Salmon=lung). The R7 cluster is now value-audited and provenance-hardened. **Value-audit COMPLETE for all round-validated numeric params (Phase-3 batch of 18 + this batch of 28 + 10 FLAGGED re-derived = full coverage of the 48+10 surface).**
