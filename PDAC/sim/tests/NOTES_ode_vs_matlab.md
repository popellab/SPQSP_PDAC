# MATLAB vs C++ ODE trajectory validation — notes

## Harness

- `PDAC/qsp/sim/` — standalone CMake build of `qsp_sim` against the
  generated `PDAC/qsp/ode/*.cpp` + `PDAC/qsp/cvode/CVODEBase` +
  `PDAC/core/ParamBase`. SUNDIALS 7 found locally or fetched via `FetchContent`.
- `export_matlab_trajectories.m` — SimBiology reference exporter. Loads the
  live `immune_oncology_model_PDAC.m`, applies parameter/IC overrides from
  the same `param_all.xml` the C++ reads, tightens CVODE tolerances, emits
  a CSV at the shared time grid.
- `compare_trajectories.py` — name-normalized tolerance diff.
- `test_ode_vs_matlab.py` — pytest wrapper, gated by `--run-matlab`.

Build and run:
```
cmake -S PDAC/qsp/sim -B PDAC/qsp/sim/build
cmake --build PDAC/qsp/sim/build
.venv/bin/pytest PDAC/sim/tests/test_ode_vs_matlab.py -v --run-matlab
```

## Dumper fix (historical)

Use `getVarOriginalUnit(i)` via the friended `operator<<` on `CVODEBase`,
not `getSpeciesVar(i, raw=true)`. The latter only undoes the substance-unit
factor; species in non-unit compartments (`syn_*` = 37.8, `A_e` = 15,
`A_s` = 900, `V_LN` = 1112.6) need compartment-volume division as well to
match SimBiology concentrations.

## Current status: 100% pass

164/164 species, 598,764 comparisons, 0 failures at rtol=5%/atol=1e-6 over
365 days. Worst residual rdiff is 1.46% (`V_T.Treg` at t=316.8).

This was originally at 17.3% when ported. Real causes of the gap were:
1. **Dumper output bug** — `getSpeciesVar(raw=true)` misses the compartment
   factor. Fixed by switching to `getVarOriginalUnit` via `operator<<`.
2. **MATLAB's default solver tolerance (1e-3) is too loose** — set
   `RelativeTolerance=1e-5, AbsoluteTolerance=1e-9` in `export_matlab_trajectories.m`.
3. **Stale SBML + param rename drift** — when the SBML was re-exported, 19
   QSP params had been renamed in MATLAB. The old C++ expected the old
   names; `QSPParam.cpp` silently defaulted missing names to 0. Fixed by
   regen + swapping the `<QSP>` block of `param_all.xml` from the fresh
   `qsp_params_xml_snippet.xml`.

## Sync guards (important)

Seven tests in `test_ode_vs_matlab.py`, driven by the shared
`PDAC/codegen/sync_checks.py` module:

1. `test_sbml_newer_than_matlab_model_script` — SBML re-exported after
   edits to `immune_oncology_model_PDAC.m`.
2. `test_generated_ode_newer_than_sbml` — codegen ran since SBML changed.
3. `test_dump_binary_newer_than_ode_cpp` — binary rebuilt since codegen.
4. `test_param_xml_contains_snippet_names` — every name in the fresh
   `qsp_params_xml_snippet.xml` is present in `param_all.xml`. Catches
   the silent-default-to-0 failure mode.
5. `test_time_grids_match` — both sides sample on an identical t-grid.
6. `test_initial_conditions_match` — every species matches at t=0 to 1e-6.
7. `test_trajectories_match` — full 365-day trajectory diff, rtol=5%.

Run all four sync-only checks standalone (no MATLAB needed):
```
python PDAC/codegen/check_sync.py
```

## Build integration

`PDAC/sim/CMakeLists.txt` has a `qsp_codegen` custom target that:
- Runs `qsp_codegen.py` if `PDAC/qsp/PDAC_model.sbml` is newer than
  `ODE_system.cpp`.
- Runs `refresh_param_xml.py` to swap the `<QSP>` block of
  `param_all{,_test}.xml` from the fresh snippet.

The main `pdac` target depends on it, so a stale ODE can never link.

## Model bump workflow

```
./PDAC/codegen/reexport_sbml.sh      # runs MATLAB, copies SBML into this repo
cmake --build PDAC/sim/build         # auto-regenerates C++ and refreshes XML
python PDAC/codegen/check_sync.py    # all-green sanity check
```

## Gaps — not yet covered by the harness

1. **Drug dosing.** `nivoOn=0`, `ipiOn=0`, `caboOn=0`, `entOn=0` in both
   XMLs used today. We've only validated the baseline disease-progression
   trajectory. Need to exercise nivo-only, ipi-only, cabo-only, and
   combinations, with doses applied in SimBiology via `adddose()` and in
   C++ via the existing XML dose-schedule wiring.

2. **SBML `<listOfEvents>` are not codegen'd.** There are two events in
   the SBML (threshold resets on cDC populations). `qsp_codegen.py` does
   not process them, so `eventEvaluate` / `eventExecution` /
   `triggerComponentEvaluate` return `false` in the generated C++. The
   current pass rate is luck — neither event's trigger condition is
   crossed over the 365-day baseline. Any scenario that does will
   diverge silently in C++.

3. **Event-triggered scenarios.** Once events are codegen'd, need an
   explicit test case that pushes a species past the threshold so the
   event path is exercised on both sides.

## Candidate move (deferred)

Considered relocating codegen + validation harness to `pdac-build` so
the QSP model, its export, and its validation all live together; SBML
would live only there; C++ would consume pre-generated files + a
`version.txt` recording the pdac-build SHA. Downsides: `qsp_sim`
infrastructure (CVODEBase, ParamBase, SUNDIALS, CMake) would have to
move too or be vendored. Decision deferred.
