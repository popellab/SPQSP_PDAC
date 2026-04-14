# MATLAB vs C++ ODE trajectory validation — notes

## Harness

- `ode_compile/` — standalone CMake build of `dump_trajectories` against the
  generated `PDAC/qsp/ode/*.cpp` + `PDAC/qsp/cvode/CVODEBase` + `PDAC/core/ParamBase`.
  SUNDIALS 7 found locally or fetched via `FetchContent`.
- `export_matlab_trajectories.m` — SimBiology reference exporter.
  Assumes a sibling `../pdac-build/` with `startup.m` and `scripts/immune_oncology_model_PDAC.m`.
- `compare_trajectories.py` — name-normalized tolerance diff.
- `test_ode_vs_matlab.py` — pytest wrapper, gated by `--run-matlab`.

Build and run:
```
cmake -S PDAC/sim/tests/ode_compile -B PDAC/sim/tests/ode_compile/build
cmake --build PDAC/sim/tests/ode_compile/build
.venv/bin/pytest PDAC/sim/tests/test_ode_vs_matlab.py -v --run-matlab
```

## Dumper fix (important)

Use `getVarOriginalUnit(i)` via the friended `operator<<` on `CVODEBase`,
not `getSpeciesVar(i, raw=true)`. The latter only undoes the substance-unit
factor; it does not divide by compartment volume, so species in non-unit
compartments (`syn_*` = 37.8, `A_e` = 15, `A_s` = 900, `V_LN` = 1112.6) come
out as amount instead of concentration and disagree with SimBiology by
exactly the compartment size. The generated `getVarOriginalUnit` handles
fixed and dynamic compartments correctly.

## Current pass rate (model_all param XML, 365 days, rtol=5%)

- t=0 ICs: 164/164 match within 2.6e-9 relative (pure FP roundoff).
- 365-day trajectories: **62.5%** (target 95%).

## Remaining divergence — not yet root-caused

102/164 species drift above 5% over 365 days. Earliest divergences:

| Species       | First t>5% | Pattern                     |
|---------------|------------|-----------------------------|
| V_T.iCAF      | 0.1 d      | C++ 1.85× MATLAB            |
| V_LN.aCD8     | 0.3 d      | C++ ~200× smaller than MATLAB |
| V_T.IL6       | 0.3 d      | C++ 1.85× MATLAB (downstream of iCAF) |
| V_LN.IL2      | 0.7 d      | diverges to 100% by day 112 |

Not a tolerance issue — ratios are consistent within each species, pointing to
rate-constant or compartment-unit handling differences in a subset of reactions.

## Open investigation path

`ode_link` branch's modular `tools/sbml_converter/` package (with
`--model-units` mode) reportedly hit 87.7% — see memory `project_model_unit_progress.md`.
`PDAC/codegen/qsp_codegen.py` on this branch is the older monolithic generator.
Next step is likely to port the converter rather than patch individual
reactions. Scratch work left at `/tmp/ode_link_run/` (built but not run).
