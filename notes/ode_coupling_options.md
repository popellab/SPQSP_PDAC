# ODE-ABM Coupling: Unit Conversion Analysis

## Problem

The custom SBML→C++ converter has a ~22% drift on ArgI and 1.5% failure rate vs MATLAB over 365 days. **The C++ solver is wrong** — three independent MATLAB solvers agree perfectly while all disagree with C++.

## Three-solver verification (2026-03-26)

| Comparison | Algorithm families | Failures (5% tol) | Worst diff |
|---|---|---|---|
| ode15s vs ode23t | BDF multistep vs trapezoidal rule | **0** | 0.011% |
| ode15s vs sundials | ode15s vs CVODE (same as C++) | **0** | 0.000% |
| ode15s vs C++ | MATLAB vs C++ SI-units | 8877 | 22.3% |

Drift starts at 0.017% at day 0.1 and grows monotonically, centered on V_T.ArgI.

## Root cause (confirmed)

**SimBiology solves in model units** (cells, nM, molecules/µm²) — NOT SI. Its `UnitConversion=ON` inserts conversion factors into rate law expressions for dimensional consistency while keeping species values in their declared units. The exported `InitialValues` vector confirms: nCD4 = 3706.9 cell, not 6.15e-21 mole.

**Our converter converts to SI** (mole, m³, second), which produces physically correct but numerically different results. The solver takes a different path because species magnitudes differ by up to 24 orders of magnitude from SimBiology's.

### Direct RHS comparison at t=0

Compared f(t=0, y0) between our C++ and SimBiology (55/162 species mismatched):

| Species | C++ dy/dt | SimBiology dy/dt | Issue |
|---|---|---|---|
| V_T.VEGF | 3.99e-9 | 3990 | ×1e-9 (nM→mol conversion) |
| V_T.qPSC | 1.66e-24 | 1.0 | ×1.66e-24 (cell→mol conversion) |
| V_LN.cDC2 | 2.66e-19 | 1.6e5 | ×1.66e-24 (cell→mol) |
| syn.CD28_CD86 | 0.128 | 7.7e10 | ×1.66e-12 (molecule→mol) |

The RHS values differ by exactly the unit conversion factors. Both are physically correct in their respective unit systems, but CVODE takes different internal steps.

### Why model-unit mode crashes

Without unit conversion, the model is **not dimensionally consistent**: parameters from different unit systems interact in rate laws (e.g., `vol_cell` in µm³ added to `V_Tmin` in mL). SimBiology also crashes with `UnitConversion=OFF`.

With our `--no-unit-conversion` flag, V_T evaluates to 2572 (µm³, wrong) instead of ~1e-6 (mL, correct). The V_T assignment rule mixes units that need conversion factors.

## What was tried

| Approach | Result |
|---|---|
| SI conversion (m³-mol-s) | Works, 98.5% — 1.5% drift |
| Tighter tolerances (1e-10) | No improvement — both converge to different solutions |
| Litre-mol-second target | Same 1.5% drift |
| Litre-mol-day target (skip time conversion) | Same 1.5% drift |
| Raw model units (--no-unit-conversion) | Crashes (dimensional inconsistency) |
| Species non-dimensionalization | Crashes (1/V_e still dominates Jacobian) |
| SPGMR iterative solver | mxstep exceeded |
| libRoadRunner v2.9.2 | Crashes on `<ci>max</ci>` + V_e stiffness |
| AMICI v1.0.1 | Crashes: `max` func, missing modifiers, non-constant V_T, SWIG mismatch |

## Solution: unit-aware code generator

The only path to matching SimBiology is generating C++ code that solves in model units with appropriate conversion factors — exactly what SimBiology's `UnitConversion=ON` does internally.

### What needs to happen

1. **Keep species values in model units** (scale = 1.0 for all species)
2. **For each parameter**, compute a conversion factor that makes it dimensionally consistent with the model-unit species it interacts with
   - Example: `vol_cell` in µm³/cell needs ×1e-12 to become mL/cell (matching V_T's mL)
   - NOT the SI conversion (×6e5 to become m³/mol)
3. **Convert compartment volumes** to a common unit within their dimension (litres for 3D, m² for 2D)
4. **Handle cross-compartment reactions** where species with different substance units interact
5. **Time stays in days** (matching SimBiology's TimeUnits=day)

### Implementation complexity

~300-500 lines of Python across the parser and generators. The core challenge: the correct conversion factor for a parameter depends on which species and compartments it appears with in each expression. This requires tracking units through rate law ASTs.

### Validation data available

- SimBiology's full RHS at t=0: `/tmp/simbio_rhs.csv` (243 species, y0 and dy/dt)
- Our C++ RHS at t=0: `/tmp/cpp_rhs.csv` (162 species, y0 and dy/dt)
- MATLAB trajectories: `/tmp/matlab_ode15s.csv` (365 days, 0.1 day resolution)
- SimBiology's exported state vector: `/tmp/simbio_internal_state.csv` (575 entries with units)

### SBML preprocessing (for any third-party tool)

SimBiology's SBML export requires these fixes:
- `<ci>max</ci>` → functionDefinition with piecewise (not standard MathML `<max/>` — invalid in L2V4)
- Missing modifierSpeciesReferences on ~71 reactions
- Non-constant compartment V_T with assignment rule depending on species (breaks AMICI's concentration transform)

Script: `tools/amici_test/fix_sbml.py`

## Progress: model-unit mode (`--model-units`, 2026-03-27)

### Approach

Per-parameter conversion factors applied at load time. The `unit_scale_annotator.py` module walks SBML ASTs, computes SI scales per node, and at additive mismatches traces the correction back to a parameter. The parameter is then converted at load time so expressions evaluate naturally in model units.

**Key conversions discovered:**
- `vol_cell`, `vol_Tcell`, `vol_Mcell`, etc.: ×1e-12 (µm³/cell → mL/cell)
- `rho_collagen`: ×1e3 (g/mL → mg/mL, matching collagen's mg units)
- `q_nCD8_P_in`, `q_nCD4_P_in`: handled via stoich factors (1/min → 1/day = ×1440)
- Various `koff_*` binding rates: ×0.6 or ×6e5 (molecule/µm² ↔ cell conversions)
- `IFNg_50`, `IL10_50`, etc.: ×1e-3 (concentration threshold conversions)

Additionally, per-stoichiometry-term factors handle cross-unit reactions:
`factor = kl_scale * time_scale / substance_scale`

### Results

| Metric | SI mode | Model-unit mode |
|---|---|---|
| RHS match at t=0 (non-zero species) | 107/162 exact | 68/94 within 1% |
| Trajectory pass rate (5% tol, 365 days) | 98.5% | **85.3%** |
| V_T.ArgI worst drift | 22.3% at day 265 | 28.7% at day 265 |
| V_T.C1 at t=100 | ratio ~1.01 | ratio **0.9999** |
| V_T.VEGF at t=100 | ratio ~0.98 | ratio **1.0009** |

Most remaining failures are marginal (5-6% at late timepoints, near the 5% threshold). Dominant failures: V_LN.cDC1/2 (52x at t=365), V_T.CD8_exh (1.37x), V_T.ArgI (29%).

### Remaining issues

1. **V_LN.cDC1/2**: 52x divergence at t=365 — likely a missing parameter conversion for lymph node dendritic cell reactions
2. **V_T.ArgI**: Still ~29% drift, same cascade as SI mode (ArgI → Treg → DC)
3. **24 species fail** the 5% threshold (vs 7 in SI mode), mostly at late timepoints (t > 260)
4. Some parameters may still need context-dependent conversions not yet discovered by the single-pass algorithm
