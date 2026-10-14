# Excluded submodel targets

YAMLs here pass Pydantic schema validation but target **parameters that are
not currently in the active PDAC model** — their upstream modules (e.g.,
`chemotaxis_module`, `NK_module`, `hypoxia_module`) exist in `core/modules/`
but are not wired into `scripts/immune_oncology_model_PDAC.m`.

Retained for provenance so the literature extractions aren't lost. If you
wire the module back in, move the YAML back to `../` and re-run
`make check-submodel-sync`.

Scanned and skipped by:
- `scripts/check_submodel_sync.py`
- `tests/scenarios/test_calibration_targets.py`
- the submodel-posterior aggregator that produces `submodel_priors.yaml`

Current contents — module not active (moved 2026-04-21):
- `CXCL9_50_Trec_PDAC_deriv001.yaml` — chemotaxis module
- `IFNg_50_CXCL9_PDAC_deriv001.yaml` — chemotaxis module
- `IL10_50_NK_PDAC_deriv001.yaml` — NK module
- `k_NK_kill_PDAC_deriv001.yaml` — NK module
- `TGFb_50_NK_PDAC_deriv001.yaml` — NK module
- `VEGF_norm_50_PDAC_deriv001.yaml` — hypoxia/VEGF module

Current contents — parameter-semantic mismatch (moved 2026-04-28):
- `k_C1_growth_kim2017_PDAC_deriv001.yaml` — Kim 2017 reports 2D-monolayer
  population doubling time on PDAC primary cell lines, which measures
  intrinsic cell-cycle proliferation. The QSP `k_C1_growth` parameter
  semantically represents the *net effective in-vivo growth rate*
  (the role TVDT-derived deriv001/deriv002 fill). Combining the 2D
  cell-cycle target with the TVDT-derived ones produced a composite
  posterior at 0.111/day that pushed evolve_to_diagnosis below the
  120-day natural-history floor (~38% sim failure rate), since the
  model parameter can't disentangle a fast cell-cycle from in-vivo
  death, hypoxia, and stromal accumulation. Retained for provenance —
  if a PDX/in-vivo cancer-cell-only target is later extracted (e.g.,
  human-mitochondrial or human-cytokeratin-quantified PDX growth),
  this upper-bound 2D anchor can be re-introduced alongside it.

Current contents — wrong-indication source (moved 2026-04-29):
- `q_CD8_T_in_PDAC_deriv001.yaml` — Boissonnas 2007 reports OT1 transgenic
  CD8 transfer kinetics in a mouse melanoma model. The cross-species
  translation to human PDAC TIL trafficking is off by ~3 OOM (the
  composed Boissonnas-pulled prior 95th-pct of 3.78e-4 caps `cd8_density`
  at ~2.3 cells/mm² vs obs 139 — joint reachability 1.27% in the
  feature/cd8-in-tumor-expansion 5k prior PC). Even after adding the
  in-tumor TIL proliferation mechanism (k_CD8_T_pro), the trafficking
  ceiling imposed by this submodel target is the binding constraint.
  Action: excluded the YAML and widened the CSV-only prior on
  q_CD8_T_in from σ_log=0.7 to σ_log=2.0 (95th-pct ratio over CSV
  median: 4× → 50×, accommodating the 60× density gap distributed
  across q_CD8_T_in + k_CD8_T_pro + upstream priming levers). Retained
  for provenance — if a primary human PDAC TIL trafficking measurement
  becomes available (intravital microscopy, adoptive transfer kinetics
  in PDX, or quantitative migration-rate inference from spatial mIHC
  gradients), this mouse-OT1 anchor can be re-introduced as one
  component of a multi-source extraction with appropriate
  translation_sigma.

Current contents — parameter-semantic mismatch (moved 2026-05-02):
- `APC0_cDC1_T_PDAC_deriv001.yaml` and `APC0_cDC2_T_PDAC_deriv001.yaml` —
  Plesca 2022 reports realized intratumoral cDC1/cDC2 density via mIHC. The
  forward models in these targets equate the QSP parameter `APC0_cDC{1,2}_T`
  directly to that realized density (cDC1 target uses a stale `H_ECM_APC_mig`
  with assumed occupancy ~0.6; cDC2 target asserts `H_pore_APC ~ 0` at
  baseline). At default ECM, `H_pore_APC ≈ 0.98` in the active model and
  the maturation drain (`k_APC_mat_cDC1 = 2.0/day` vs `k_APC_death ~ 0.4/day`)
  attenuates realized density ~50–200× below the parameter. Plus the
  literature consensus (Hegde 2020, Lin 2020, Salmon 2016, Böttcher 2018,
  Liu 2007/2009) is that intratumoral cDC1/cDC2 are recruitment-and-apoptosis
  dominated, not homeostatically defended — the V_T equation form is recruit
  + first-order death (NOT homeostatic relaxation), so the parameter is a
  *recruitment-target density* (`k_recruit / k_APC_death` numerator), not a
  realized density. The same Plesca data is anchored correctly by the
  full-model calibration targets at
  `calibration_targets/baseline_no_treatment/cdc{1,2}_density_baseline_*.yaml`,
  which evaluate the realized-density observable against the model's full
  attenuation chain. Renamed `APC0_cDC{1,2}_T → cDC{1,2}_recruit_target_T`
  in the same change; widened σ from 0.7 to 2.0 to allow the prior to reach
  Plesca-scale realized density via SBI. See
  `notes/model/cdc1_homeostasis_vs_recruitment.md` for the biology rationale.

Current contents — parameter-semantic mismatch (moved 2026-04-29):
- `f_stroma_max_torphy2018_PDAC_deriv001.yaml` — Torphy 2018 reports tumor
  stroma density (TSD = stromal area / total tumor area, median 0.60 in
  Denmark cohort n=123). The forward model is identity:
  `f_stroma_max = TSD/(1-TSD)` (stroma:epithelium *area* ratio). But the
  QSP `f_stroma_max` is a CAF *count* ratio (= aPSC_total / C_total at
  saturation). The YAML's own text correctly notes that "stroma includes
  acellular ECM (~40-60% of stromal area in PDAC) plus immune and
  vascular cells" and "the effective CAF:cancer ratio is roughly 0.3×
  the stroma:epithelium ratio," but the code applies no such correction
  and there is no implemented translation_sigma. The target therefore
  pushed the joint Ogawa+Torphy posterior to fs ≈ 2.43 — ~8× above the
  Ogawa-alone prior-median value (0.285) and the back-of-envelope cellular
  count ratio derived from Bengtsson + Ogawa area data (~0.3). With
  Ogawa2021 alone, joint NPE has the freedom to reach the single-observable
  optimum for stromal_fraction (~0.5). Retained for provenance — if a
  CAF-fraction-of-stroma submodel is added to apply the 0.3× correction
  with proper uncertainty propagation, this TSD anchor can be re-introduced
  in that revised form.
