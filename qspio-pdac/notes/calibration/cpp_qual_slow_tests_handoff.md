# C++ qual + slow tests port — handoff

Status as of 2026-04-24. Tracked in #40 (port) and #42 (driver-library extraction). PR #43 up for review.

Worktree: `.claude/worktrees/explore+cpp-qual-slow-tests` on branch `worktree-explore+cpp-qual-slow-tests`.

## TL;DR

- Slow MATLAB-driven tests (`test_qualitative_behaviors.py`, `test_evolve_to_diagnosis.py`) now run under the C++ simulator in ~5 s total instead of minutes via MATLAB.
- `QUAL_TESTS_BACKEND` env var flips backend: `cpp` (default), `matlab` (legacy). `QUAL_TESTS_INCLUDE_TX=1` enables the GVAX+nivo treatment tier.
- Driver + generated ODE + runtime YAMLs/XML moved in-repo: `cpp/`, `resources/cpp/`. `SPQSP_PDAC` no longer required as a sibling checkout for these tests.

## Repo layout after the move

```
cpp/
  sim/            qsp_sim.cpp, CMakeLists.txt, set_healthy_populations.{cpp,h},
                  evolve_to_diagnosis.{cpp,h}, profile_jacobian.cpp, build/
  qsp/
    ode/          ODE_system.{cpp,h}, QSPParam.{cpp,h}, QSP_enum.h,
                  qsp_params_xml_snippet.xml  (regenerated from SBML via qsp-codegen)
    cvode/        CVODEBase.{cpp,h}, MolecularModelCVode.h  (generic solver wrapper)
  core/           ParamBase.{cpp,h}
resources/cpp/
  param_all.xml           template passed to qsp_sim at runtime
  drug_metadata.yaml      MW + dose-basis per drug (adds cyclophosphamide)
  healthy_state.yaml      cell densities used by set_healthy_populations
tests/scenarios/
  _cpp_helper.py          builds sim_outputs dict via CppBatchRunner
  test_qualitative_behaviors.py  env-switched fixture
  test_evolve_to_diagnosis.py    env-switched fixture
```

## Building the binary locally

```
cd cpp/sim && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --target qsp_sim -j
```

`_cpp_helper.py` auto-picks up `cpp/sim/build/qsp_sim` when present; otherwise falls back to whatever `figures/sbi_stage2_current/run_config.json` specifies. The three runtime paths (param_all.xml, drug_metadata.yaml, healthy_state.yaml) are always pulled from `resources/cpp/`.

## Running the tests

```
# Everything except treatment tier
.venv/bin/pytest tests/scenarios/test_qualitative_behaviors.py -m slow

# Full suite including GVAX+nivo
QUAL_TESTS_INCLUDE_TX=1 .venv/bin/pytest tests/scenarios/test_qualitative_behaviors.py -m slow

# Legacy MATLAB path
QUAL_TESTS_BACKEND=matlab .venv/bin/pytest tests/scenarios/test_qualitative_behaviors.py -m slow

# Evolve sanity checks
.venv/bin/pytest tests/scenarios/test_evolve_to_diagnosis.py -v
```

## Current C++ results (PR #43 HEAD)

| Suite | Passed | Failed | xfail | xpass | Skipped | Wall |
| --- | --- | --- | --- | --- | --- | --- |
| `test_evolve_to_diagnosis.py` | 14 | 0 | 0 | 0 | 0 | 2.14 s |
| `test_qualitative_behaviors.py` (no tx) | 44 | 1 | 13 | 11 | 35 | 3.4 s |
| `test_qualitative_behaviors.py` (tx on) | 62 | 6 | 13 | 11 | 12 | 4.9 s |

Failures as of now:

- `icaf_fraction_range` — 18.6% vs 20% lower bound. Near-miss; probably a prior-median thing, worth retuning the bound or the iCAF/myCAF balance.
- `test_m1_m2_ratio_improves`, `test_fix1_gvax_drives_mcDC1`, `test_fix1_gvax_drives_cd8_infiltration`, `test_fix2_gvax_expands_Th_pool` (1.13x vs 1.20x), `test_mapc_transient_peak` (d14=96236 vs d21=96636). Cluster looks like GVAX cascade under-delivers compared to the thresholds these tests encode. Likely real biology findings — not port bugs. Worth an issue.

## MATLAB parity

Blocked: MATLAB `evolve_to_diagnosis` fails outright on this branch — every tier errors "Failed to establish initial conditions via evolve_to_diagnosis". Consistent with the `TestBaselineTME` xfail note referencing #32 (CD8-collapse refresh of 2026-04-23, p1_50 posterior ~135x prior pushes Th→Treg→CD8 balance).

C++ reaches 3.2 cm at t=856 d on the same prior-median theta. Either:

- MATLAB SBML has drifted past the SBML snapshot used to generate `cpp/qsp/ode/ODE_system.cpp`, so C++ is effectively running an older model. (Regenerate with qsp-codegen on current SBML to check.)
- MATLAB's hand-written `evolve_to_diagnosis.m` has a regression independent of the model itself.

Either is a separate problem from this port. Compare against `main` (pre-refresh) if you want a true parity diff.

## What didn't get done and why

- **Phase 4: GitHub Actions workflow** — build `qsp_sim` in CI (cache on template + source hash), run the qual + evolve tests, retire the test-receipt flow. Not started. Needs a fresh session with CI-surface context.
- **Makefile `refresh-cpp` target** — still writes into `SPQSP_PDAC/PDAC/sim/resource/`. Update to write `resources/cpp/` + `cpp/qsp/ode/` once the qsp-codegen workflow is exercised against the in-repo layout.
- **Driver library extraction** — generic bits of `qsp_sim.cpp` should ship with qsp-codegen (see #42). Today pdac-build carries a copy.
- **SPQSP_PDAC cleanup** — the files under `SPQSP_PDAC/PDAC/{qsp,sim,core}/` that we copied out still exist there. Leave them alone until non-PDAC consumers are migrated; then delete.

## File pointers

- Helper: `tests/scenarios/_cpp_helper.py` (`build_sim_outputs`, `build_evolve_outputs`)
- Env-gated fixtures: `tests/scenarios/test_qualitative_behaviors.py:105`, `tests/scenarios/test_evolve_to_diagnosis.py:65`
- Milligram passthrough: `cpp/sim/qsp_sim.cpp:361`
- Cyclophosphamide entry: `resources/cpp/drug_metadata.yaml:77`
- Tracking issues: #40 (port), #42 (codegen lib), #43 (PR for this work)
