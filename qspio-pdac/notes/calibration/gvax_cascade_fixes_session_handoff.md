# GVAX cascade fixes — session handoff (2026-04-21)

## TL;DR

Landed Fix 1 (cDC1 antigen transfer), Fix 2 (niche-scaled T-cell caps with
apCAF for Treg), and Fix 3 (lower `EC50_P1_mature`) for the GVAX cascade
bottleneck documented in
`notes/calibration/gvax_cascade_bottleneck_handoff.md`. Added the
post-model-change Makefile + sync scripts. Cleaned 40 orphan priors.
Regenerated model_structure.json, SBML, C++ codegen. Three commits on
`fix/gvax-cascade-bottleneck` (**unpushed**). All Python / submodel /
calibration-target tests pass; C++ equivalence test has 4 failures that
look like a pre-existing codegen issue with forward-referenced species.

## Branch + commits

Working directory: `/Users/joeleliason/Projects/pdac-build-gvax-cascade`
(git worktree on branch `fix/gvax-cascade-bottleneck`).

```
27b1737 Finish model refresh: fix Makefile worktree support, hpc-status parsing, drop v_T_motility orphan
899d070 Remove fabricated Bhatia 2014 JACI citation from Fix 3 rationale
9835cfe Fix GVAX cascade bottleneck: cDC1 antigen transfer, niche-scaled T-cell caps, scenario Cy correction
```

Also dirty in `/Users/joeleliason/Projects/SPQSP_PDAC` on `PDAC_functionality`:
freshly-regenerated SBML + C++ codegen + param_all.xml + qsp_sim binary
for the new parameters. Not committed yet.

## What changed (cross-reference to DEVLOG entry 2026-04-21)

### Fix 1 — cDC1 antigen transfer in LN

- New reaction in `core/modules/gvax_module.m`:
  `V_LN.cDC1 → V_LN.mcDC1` at rate
  `k_cDC1_Ag_transfer × V_LN.cDC1 × V_LN.mcDC2`.
- Captures Route B (migratory cDC2/GMDC → resident cDC1 antigen handoff)
  per Ferris 2022 *Cancer Immunol Res* PMC9357132. Route A (direct
  migratory cDC1) is weak for GVAX because GM-CSF poorly expands cDC1.
- New param `k_cDC1_Ag_transfer = 1e-7 /(cell·day)` (parameters_PDAC.m,
  pdac_priors.csv, notes/targets.csv).
- Impact at median priors under `gvax_nivo_neoadjuvant_zheng2022`:
  `V_LN.mcDC1` 1.03× → **18×**; `V_T.CD8` 1.10× → **17×**.

### Fix 2 — niche-scaled logistic T-cell caps

- Th cap: `rho_Th_max × V_T × (1 + alpha_Th_cap × H_CD8_Tcap)` with
  `H_CD8_Tcap = V_T.CD8/(V_T.CD8 + CD8_50_Tcap)`. CD8 is the
  inflammation proxy because V_T.IFNg is dominated by Th-secreted IFNg
  (circular coupling).
- Treg cap: `rho_Treg_max × V_T × (1 + alpha_Treg_cap × H_apCAF_Tcap)` with
  `H_apCAF_Tcap = V_T.apCAF/(V_T.apCAF + apCAF_50_Tcap)`. apCAFs are the
  dedicated Treg-inducing CAF subtype (Elyada 2019, Huang 2022), and
  critically apCAF activation is TGFb-dependent via CAF interconversion
  — so TGFb blockade correctly shrinks the Treg cap (an earlier
  CD8-scaled attempt produced a paradoxical "TGFb blockade → more CD8 →
  more Treg" loop).
- New params: `alpha_Th_cap=5`, `alpha_Treg_cap=2`, `CD8_50_Tcap=1e5`,
  `apCAF_50_Tcap=5e6`.
- Impact: V_T.Th 1.04× → **1.47×**, V_T.Treg 1.03× → **0.38×** (Cy
  depletion dominates after Fix 1+2+Cy scenario correction).

### Fix 3 — EC50_P1_mature 10 nM → 0.1 nM

- Single parameter change. With 10 nM, the antigen-gating factor in
  `J_mature_ID` was ~9e-4 because `V_ID.P1_GVAX` steady-state is ~0.009 nM.
  Antigen-dependent maturation was numerically dead.
- **0.1 nM is an estimated value, not anchored to a primary source.** An
  earlier draft cited a non-existent "Bhatia 2014 JACI" paper; that
  citation was scrubbed in commit 899d070. Candidate for submodel-target
  extraction before Stage-2 SBI.
- Impact on CD8 cascade is modest (steady-state V_ID.mAPC is rate-limited
  by death/migration, not antigen gating). Main value is mechanistic
  correctness.

### Zheng 2022 scenario correction

- Both `gvax_neoadjuvant_zheng2022.yaml` (Arm A) and
  `gvax_nivo_neoadjuvant_zheng2022.yaml` (Arm B) were missing low-dose
  cyclophosphamide. Cy is in every arm of NCT02451982 (200 mg/m² day
  before vaccination; standard Hopkins GVAX preparation since
  Laheru 2008). Added to both scenarios.
- Without Cy, Fix 2's expanded Treg cap produced Treg rising instead of
  falling; the three `test_treatment_reduces_tregs` tests failed. With
  Cy, Treg falls 62% — matches Zheng 2022 reported Treg reduction.

### Post-model-change Makefile + sync scripts

- `Makefile` at repo root: 12 targets across Layer 1 (Python), Layer 2
  (submodel), Layer 3 (C++ / HPC). See `make help` or
  `notes/model/post_change_checklist.md`.
- `scripts/check_priors_sync.py` — active-module-aware check that every
  row in `pdac_priors.csv` corresponds to either a `params.X.Value` entry
  in `parameters_PDAC.m` or an `addparameter(model,'X',...)` call in an
  actively-called module.
- `scripts/check_submodel_sync.py` — same rule applied to
  `submodel_targets/*.yaml` and `submodel_config.yaml` group members.
  Skips `excluded/`, `retired/`, `to-review/`.
- `scripts/hpc_status.py` — single SSH call reports HPC's SPQSP_PDAC
  branch/SHA + installed distribution versions; flags drift vs configured
  `credentials.yaml` `cpp.branch`.
- Workflow saved as user memory: `reference_refresh_workflow.md`.

### Orphan priors cleanup

40 rows removed from `pdac_priors.csv`:
- 39 stale params from never-wired modules (chemotaxis, NK, hypoxia,
  metabolism, naive-Treg compartment fragments).
- 1 true orphan (`v_T_motility` — declared in `parameters_PDAC.m` but
  never `addparameter`'d into an active module; `v_T_search_volume` is
  the actual model param).

6 of the 39 had corresponding submodel target YAMLs. Moved to
`calibration_targets/submodel_targets/excluded/` (not deleted; YAMLs are
provenance-tagged and could be reactivated if the modules are wired in).

### Regression guards

6 new tests in `TestGvaxCascadeFixes` in
`tests/scenarios/test_qualitative_behaviors.py`:
- `test_fix1_gvax_drives_mcDC1` (≥10×)
- `test_fix1_gvax_drives_cd8_infiltration` (≥5×)
- `test_fix2_gvax_expands_Th_pool` (≥1.2×)
- `test_cy_depletes_tregs_under_gvax` (≤0.8×)
- `test_th_treg_ratio_rises_under_treatment`
- `test_fix2_treg_cap_decoupled_from_cd8` (regression guard for the
  circular-coupling iteration we avoided)

Also flipped `test_worsening_exhaustion` from xfail to expected-pass — it
now consistently passes post-Fix-1/2 baseline shift.

## Test status

| Suite | Result |
|---|---|
| `make check-priors-sync` | ✅ 181 priors clean |
| `make check-submodel-sync` | ✅ 99 active YAMLs clean, 7 in skip dirs |
| `pytest -m "not slow"` | ✅ 283 pass, 0 fail, 15 xfail |
| `pytest tests/scenarios/test_calibration_targets.py -m slow` | ✅ 139/139 |
| SPQSP_PDAC `check_sync.py` | ✅ all 5 checks |
| `pytest PDAC/sim/tests/test_ode_vs_matlab.py --run-matlab` | ❌ 4/18 fail |

## Open issues to pick up

### 1. ~~C++ equivalence failures~~ — RESOLVED (2026-04-22)

**What looked like a forward-reference codegen bug was actually a
test-harness misconfiguration.** `test_ode_vs_matlab.py` hard-coded
`PDAC_BUILD = REPO_ROOT.parent / "pdac-build"` → the MATLAB side
always drove from the `main` pdac-build checkout regardless of which
worktree invoked `make cpp-equivalence`. The C++ side was built from
whichever worktree ran `make refresh-cpp`. So MATLAB was simulating
main's pre-Fix-1/2/3 model while C++ simulated the worktree's Fix-1/2/3
SBML — 5.8× cap divergence, 2.4× Treg divergence, etc. were the real
model response to Fix 2, not a bug.

Fix landed: `test_ode_vs_matlab.py` now reads `PDAC_BUILD_DIR` env var
with fallback to sibling `pdac-build/`. `make cpp-equivalence` in the
pdac-build Makefile already exports `PDAC_BUILD_DIR=$(CURDIR)`, so a
worktree at the root of `make` invocation is authoritative.

**Remaining equivalence failures (2 species, pre-existing):**
- `syn_M_C.PDL1_total`: MATLAB≈460, C++≈7.6e-10
- `syn_M_C.PDL2_total`: MATLAB≈32, C++≈5.3e-11

These are NOT Fix-1/2/3 regressions — they fail identically on main.
Likely a real codegen bug in how the macrophage-cancer synapse
repeatedAssignment rule for `PDL1_total` / `PDL2_total` is emitted to
C++. Not blocking SBI. Deferred.

**Investigation artifacts** (left in `scripts/debug_*.{m,py}` in the
gvax-cascade worktree, can be removed or kept as templates):
- `debug_cpp_vs_matlab_evolved_ic.py` — dumps all species at t=0 from
  the most-recent pytest run, sorts by rdiff, spotlights Fix-2 Hill
  inputs.
- `debug_dump_th_treg_reactions.m` — lists all reactions producing
  V_T.Th / V_T.Treg from the live MATLAB model.
- `debug_mathml_to_infix.py` — reconstructs a reaction's SBML MathML
  kineticLaw as infix, resolving species IDs to compartment.species.
- `debug_matlab_flux.m` — computes Reaction_192 / Reaction_31 fluxes
  at post-evolve state and cross-checks against SimBiology's
  evaluation.
- `debug_matlab_csv_mapping.m`, `debug_matlab_exactly_like_test.m` —
  replicate the CSV-export column mapping.

The handoff-harness diagnosis path is summarized in
`notes/codegen_decoupling_plan.md`; the harness fix is orthogonal to
the codegen split but discovered during it.

### 2. SPQSP_PDAC commit / push

Worktree status:
- SPQSP_PDAC dirty on `PDAC_functionality @ 7c72afa6` with freshly
  regenerated SBML + C++ codegen + param_all.xml + built binary.
- HPC tracks `cpp-sweep-binary-io @ d841dc63` per `credentials.yaml`
  `cpp.branch`.
- User mentioned (end of session) wanting to move HPC to
  `PDAC_functionality` and eventually split codegen into its own repo.

Options:
- A. Update `cpp.branch` in `credentials.yaml` to `PDAC_functionality`,
  commit + push on that branch, next HPC submit auto-pulls.
- B. Rebase/cherry-pick the codegen changes onto `cpp-sweep-binary-io`,
  push, HPC auto-pulls on next submit.
- C. Wait until the codegen-repo split. In the meantime HPC will
  continue using its current binary (built from `cpp-sweep-binary-io`)
  — safe but HPC simulations won't include Fix 1/2/3 behavior.

Option (C) is probably fine if no HPC PPC runs are imminent. Confirm
with user before pushing to either branch.

### 3. pdac-build push

3 unpushed commits on `fix/gvax-cascade-bottleneck`. Open PR via
`gh pr create` when ready. No pending review requests.

### 4. `submodel_priors.yaml` regen (deferred)

`calibration_targets/submodel_targets/submodel_priors.yaml` was last
regenerated 2026-03-18. New submodel targets (if any are extracted for
k_cDC1_Ag_transfer, alpha_Th_cap, alpha_Treg_cap, CD8_50_Tcap,
apCAF_50_Tcap, EC50_P1_mature from `notes/targets.csv`) would require
re-running the qsp-inference MCMC pipeline (hours). Not urgent — those
parameters go into Stage-2 with their uninformed CSV priors for now.

### 5. The other remaining xfails

14 xfails in the fast suite, all documented in DEVLOG. None are Fix-1/2/3
artifacts:
- 3 TGFb-saturation xfails: legitimate biology (TGFb ~17 nM >> EC50 0.07
  nM is correct for PDAC desmoplastic stroma).
- 7 EC50-miscalibration xfails for IL-6, CCL5, IL-10, NO, ArgI, pore_APC
  Hills — waiting on Stage-2 joint calibration.
- 1 H_P1_apCAF switch-high (0.08, needs >0.5) — ties to the antigen
  coupling work; partially Fix 3 territory but not fully addressed.
- 3 macrophage-polarization xfails (M2:M1 ratio at 11.6, M1/M2 flat
  under treatment) — requires joint calibration or a more sensitive
  repolarization Hill.

All tracked in DEVLOG as Stage-2 or Fix-4 concerns.

## Resume commands

```bash
# Verify current worktree state
cd /Users/joeleliason/Projects/pdac-build-gvax-cascade
git log --oneline -5
git status --short

# Re-run the refresh workflow (see reference_refresh_workflow.md memory)
make help
make refresh-model-artifacts   # Python-only, fast
make refresh-all               # Everything (including C++)

# Re-inspect Fix 1+2+3 cascade behavior at median priors
rm -f tests/scenarios/.sim_outputs.json
.venv/bin/pytest tests/scenarios/test_qualitative_behaviors.py::TestHillFunctionAudit::test_hill_audit_summary -x -s
.venv/bin/python -c "
import json
d = json.load(open('tests/scenarios/.sim_outputs.json'))
for k in ['Th','Treg','CD8','apCAF','VLN_mcDC1','VLN_mcDC2']:
    a=d['tx_d0'].get(k,0); b=d['tx_d21'].get(k,0)
    print(f'{k:12s} {a:10.3e} -> {b:10.3e}   {b/max(a,1e-12):.2f}x')
"

# Check what HPC is tracking
make hpc-status

# Regenerate artifacts after any module edit
make model-structure
```

## Related notes

- `DEVLOG.md` — entry 2026-04-21 has the full narrative
- `notes/calibration/gvax_cascade_bottleneck_handoff.md` — original
  diagnostic (pre-fix)
- `notes/model/post_change_checklist.md` — 12-artifact refresh table
- `calibration_targets/submodel_targets/excluded/README.md` — excluded
  submodel targets inventory
- User memory: `reference_refresh_workflow.md`,
  `feedback_no_fabricated_citations.md`
