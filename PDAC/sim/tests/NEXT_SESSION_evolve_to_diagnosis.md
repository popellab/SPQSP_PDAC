# Next session: port `evolve_to_diagnosis` to C++

## Goal

The C++ path must have **zero MATLAB dependencies**. Today, the dosing-parity
test (`test_scenario_trajectories_match`) is skipped because MATLAB's
`sbiosimulate` hits step-size-zero when a bolus fires against the
just-starting tumor in `param_all.xml`. pdac-build avoids this by calling
`evolve_to_diagnosis` (sets healthy populations → integrates until V_T
diameter hits target) before any treatment run. The C++ dumper and the main
`pdac` binary need the same capability natively.

## Current state on branch `PDAC_qsp_sync`

### What works
- Scenario YAML → C++ dose-plan expansion (`drug_metadata.yaml` + yaml-cpp)
- `<listOfEvents>` codegen with SimBiology-matching `initialValue` semantics
- Codegen-emitted `ODE_system::get_compartment_volume(name)` — returns any
  dynamic/static compartment's current volume in its SBML-native unit (so
  `get_compartment_volume("V_T")` returns mL, = cm³ → easy diameter math).
- `test_V_T_matches` asserts V_T agrees with MATLAB pointwise (rtol<1e-2)
  over the baseline 365-day run.
- All other tests green (11 passed, 1 skipped).

### What the skipped test needs
- `test_scenario_trajectories_match` uses `nivo_single_day7.yaml`. It runs
  both sides 30 days with a nivo bolus at day 7. Currently skipped.

## Port plan for `evolve_to_diagnosis`

### Source to port
- `pdac-build/scripts/evolve_to_diagnosis.m` (344 lines) — main logic
- `pdac-build/scripts/set_healthy_populations.m` (253 lines) — initial state
  setup (cell densities × compartment volumes → per-species amounts)

### Architecture
New files (not test-only — main `pdac` sim should call this too):
- `PDAC/sim/evolve_to_diagnosis.h` — header
- `PDAC/sim/evolve_to_diagnosis.cpp` — `evolve_to_diagnosis(ODE_system&,
  target_diameter_cm, opts)` → integrates until diameter crossed; returns
  time-of-diagnosis. State left in `_species_var`.
- `PDAC/sim/set_healthy_populations.cpp` — sets ICs from hardcoded
  densities + model parameters (D_cell, compartment volumes, etc).
- Dumper gains `--evolve-to-diagnosis` flag (or reads
  `sim_config.initialization.function: evolve_to_diagnosis` from scenario
  YAML). Silently integrates until target, then shifts dose times by
  `t_diag` and starts CSV output at `t - t_diag`.

### Stopping criterion
Use `get_compartment_volume("V_T")` which is already available. Diameter:
```cpp
double V_T_mL = ode.get_compartment_volume("V_T");
double diam_cm = 2.0 * std::cbrt(3.0 * V_T_mL / (4.0 * M_PI));
if (diam_cm >= target_diameter_cm) break;
```
`target_diameter_cm` comes from `initial_tumour_diameter` in
`param_all.xml` (already 3.2 cm). The MATLAB `TargetDiameter` fallback (via
variant) is not needed — model parameter is authoritative.

### Biological plausibility guard
Reject if target reached in < 120 days (matches MATLAB). Also cap at
7300 days (20 yr) evolution time as MATLAB does.

### Open design questions

1. **`set_healthy_populations` port scope.** MATLAB hardcodes densities
   (CD8=50 cells/mm³, Treg=20 cells/mm³, MDSC=5 cells/mm³, TAM=100 cells/mm³,
   m1_m2_ratio=1.5, iCAF=3.5, myCAF=6.0, apCAF=0.5, qPSC=200 cells/mm³,
   plus compartment-specific LN populations). Options:
   - (a) Hardcode in C++ as constants — simplest, matches MATLAB 1:1
   - (b) Move densities to a `healthy_state.yaml` both sides read (matches
         the `drug_metadata.yaml` pattern we already established)
   - (c) Commit a pre-generated `healthy_ic.xml` produced by running
         set_healthy_populations once (static; goes stale on param edits)
   Preference: (b) — keeps the "biology inputs in YAML" theme.

2. **Starting state before evolve.** MATLAB's `set_healthy_populations`
   operates on the live model (which has its own default values). C++
   starts from `param_all.xml`. These may differ. If we want the evolved
   endpoints to match for validation, we need either (i) `healthy_state.yaml`
   fed to both sides, or (ii) accept that the two sides' endpoints differ
   by initial-condition-dependent amounts.

3. **MATLAB validation reference.** Does the MATLAB side of the test still
   call its own `evolve_to_diagnosis`, or do we mirror the C++ algorithm
   (simulate from a shared healthy state until V_T target)? Cleanest for
   validation: both mirror the same algorithm. MATLAB's
   `export_matlab_trajectories.m` gets an `evolve_to_diagnosis` mode that
   runs sbiosimulate from the same starting state until V_T crosses,
   extracts state, runs scenario from there. No dependency on
   `run_median_simulation` or MATLAB's `evolve_to_diagnosis.m`.

4. **initial_tumour_diameter semantics.** MATLAB uses full-V_T diameter
   (cancer + immune + stroma + collagen) for the stopping criterion — so do
   we, now that `get_compartment_volume("V_T")` is exact.

### Rough estimate
~400-600 lines new C++ (including set_healthy_populations port)
+ ~100 lines MATLAB (mirror algorithm in export_matlab_trajectories.m)
+ ~50 lines CMake + pytest fixture wiring.
Half a day of focused work.

### File-level touch list
- NEW: `PDAC/sim/evolve_to_diagnosis.{h,cpp}`
- NEW: `PDAC/sim/set_healthy_populations.{h,cpp}`
- NEW (option 1b): `PDAC/sim/resource/healthy_state.yaml`
- MOD: `PDAC/sim/tests/ode_compile/dump_trajectories.cpp` — add `--evolve-to-diagnosis` flag, link new files
- MOD: `PDAC/sim/tests/ode_compile/CMakeLists.txt` — add new sources
- MOD: `PDAC/sim/CMakeLists.txt` — link into main `pdac` too
- MOD: `PDAC/sim/tests/export_matlab_trajectories.m` — mirror evolve
  algorithm (or toggle between that and MATLAB's evolve_to_diagnosis for
  comparison)
- MOD: `PDAC/sim/tests/test_ode_vs_matlab.py` — unskip
  `test_scenario_trajectories_match`; add an `evolved_state_match` test
  that compares species values at t_diagnosis between sides.

### Where to start

1. Read `set_healthy_populations.m` end-to-end. Identify every
   `sbioselect` call, every hardcoded density, every parameter dependency.
2. Decide on (1b) vs (1a) above — consult user.
3. Write `set_healthy_populations.cpp` that populates an `ODE_system`'s
   `_species_var`. Spot-check a few species against MATLAB's output.
4. Write the evolve loop — use `get_compartment_volume("V_T")`, simple
   simOdeStep until cross. Verify C++ reaches diagnosis at a reasonable
   time (~857 d per today's MATLAB log).
5. Wire to dumper via `--evolve-to-diagnosis`. Integrate continuous-time:
   offset dose times and CSV time axis by t_diagnosis.
6. Mirror algorithm in MATLAB. Unskip the parity test.

## Tangential TODOs uncovered

- `LymphCentral_wrapper.cpp` wires nivo + partial cabo only. `ipiOn` /
  `entOn` XML flags are read nowhere → silent no-op if enabled. Fix when
  touching the main sim's dosing.
- `evolve_to_diagnosis` in pdac-build has a `initialValue` event trigger
  bug? Not verified — MATLAB's events are honored, but the scenario
  events (`C_total < 0.5*cell → V_T.K = 0.01*cell`) didn't fire during
  the evolve log we saw at day 857. Sanity check once C++ evolve works.
