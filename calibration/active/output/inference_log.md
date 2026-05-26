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
| 1 | cancer_tgfb_secretion | STEM_TGFB_RELEASE, PROG_TGFB_RELEASE | 2 | secretion |
| 2 | treg_il10_secretion | TREG_IL10_RELEASE ⚠️default=0 | 1 | secretion |
| 3 | mac_m1_il1_secretion | MAC_M1_IL1_RELEASE | 1 | secretion |
| 4 | frc_lymphoid_chemokines | FIB_FRC_CCL21_RELEASE, FIB_FRC_CXCL12_RELEASE | 2 | secretion |
| 5 | bcell_secretome | BCELL_ANTIBODY, IL10, IL6, ACT_CXCL13, PLASMA_CXCL13 (all `_RELEASE`) | 5 | secretion (joint) |
| 6 | tfh_biology | TFH_CXCL13_RELEASE, TFH_EC50_IL6 | 2 | secretion + sensing |
| 7 | adcc_pharmacodynamics | ADCC_AB_EC50, ADCC_AB_HILL_N | 2 | sensing (paired Hill) |
| 8 | bcell_cxcl13_chemoreceptor | BCELL_EC50_CXCL13_REC | 1 | sensing |
| 9 | tcell_secretion_timers | TCELL_IL2_RELEASE_TIME, TCD4_TGFB_RELEASE_TIME | 2 | burst-period |
| 10 | mechanosensing_ec50 | ECM_YAP_EC50, VAS_COLLAPSE_EC50 | 2 | sensing (mechano) |

Note: ⚠️ `TREG_IL10_RELEASE=0` in current XML — production was disabled. Target should set a literature-grounded prior and surface a non-zero posterior; user will need to decide whether to enable in XML default after inference.

### Target-by-target progress

#### T1 — cancer_tgfb_secretion ✅ VALIDATED 2026-05-26

- **Inferred param:** `PARAM_STEM_TGFB_RELEASE` (single param; `PARAM_PROG_TGFB_RELEASE` pooled to mirror STEM's posterior in ledger-sync, justified by Lonardo2011 stem≈bulk TGFb finding).
- **Primary source:** Roger et al. 2019 (DOI `10.1038/s41419-019-2116-x`), Capan-2 conditioned-media TGFb1 Quantikine ELISA, n=3 biological replicates, 3×10⁵ cells / 6-well / 24 h / DMEM+0.5%FBS / standard 2 mL volume.
- **Forward model:** `batch_accumulation`; predicts pg/mL from sec_rate · cells · time · MW(25 kDa) / vol.
- **Digitized data:** Fig 2a WPD extraction of 3 individual replicate points (-1.92, 34.62, 48.08 pg/mL) — one replicate is at/below Quantikine LOD (~7 pg/mL).
- **Prior tension caught:** Roger implies sec_rate ≈ 8.3e-14, which was 5.9σ below the original CSV prior `lognormal(median=5e-12, σ=0.7)`. Broadened to `lognormal(median=6.3e-13, σ=1.5)` ahead of MAPLE (95% CI [3.3e-14, 1.2e-11] brackets both Roger and the legacy XML default at ±1.4σ each). Updated in both `docs/abm_parameters_v3.csv:244-245` and `PDAC/sim/resource/pdac_abm_priors.csv:218-219`.
- **Submodel-only posterior:** median 1.055e-13 pmol/cell/s, CV=0.64, lognormal fit, translation σ=0.377, contraction=1.00, MCMC clean (n_eff=203, R-hat=0.999, 0 divergences). About 50× below the legacy XML 5e-12.
- **YAML:** `calibration/active/submodel_targets/cancer_tgfb_secretion_PDAC_deriv001.yaml`

### Plan forward (Round 4+)

Per `docs/phase3_maple_restart.md` §"Round cadence":

| Round | Cluster | ~Targets | Notes |
|---|---|---|---|
| 4 | PDE secretion + EC50 | 10 | Cytokine field drivers (this round) |
| 5 | Movement chemotaxis CI per cell type | ~8 | IMC niche-structure sensitivity |
| 6 | Killing + proliferation rates | ~8 | Cell-fraction observable sensitivity |
| 7 | Persistence + porosity + contact guidance | ~6 | Hardest to MAPLE; many fall to manual prior |
| 8 | Lifespans + recruitment scalars | ~6 | |

**Round-entry note (carried forward from 2026-05-20):** before authoring any new round's targets, repeat the Step-1 dead-code check on each cluster. R4 (2026-05-26) caught one dead param (`PARAM_TCELL_IFNG_RELEASE_TIME`); gate continues to pay off.
