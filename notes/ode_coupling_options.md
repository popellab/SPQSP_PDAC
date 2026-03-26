# ODE-ABM Coupling Options

## Problem

The custom SBML→C++ converter works (98.5% species agreement over 365 days) but has a ~22% drift on ArgI due to structural differences between MATLAB's model-unit solver and our SI-unit CVODE setup.

**Root cause:** MATLAB solves ODEs in model units (cells, nM, mL, days). The C++ converter transforms to SI units (mol/m³, m³, seconds), creating a 24 order-of-magnitude spread in species values. Even with per-species absolute tolerances and matched relative tolerances, the BDF solver takes different internal steps at different scales. The Jacobian condition number differs between unit systems, causing path-dependent divergence in the stiff ArgI → Treg → DC cascade.

**What was tried (2026-03-26):**

| Approach | Pass rate | Notes |
|----------|-----------|-------|
| Baseline (per-species abstol × SI scale, reltol=1e-6) | 98.4% | 9500 failures, first at day 278 |
| Tighter reltol=1e-8, abstol=1e-11 | 98.5% | 8877 failures, first at day 281 |
| Tighter reltol=1e-10, abstol=1e-13 | 98.5% | 8862 failures — converged |
| Both MATLAB and C++ at 1e-10 | 98.5% | Identical — both converged to different solutions |
| Scalar tolerance (CVodeSStolerances) | 39.8% | Much worse — per-species scaling is essential |
| --no-unit-conversion (model units) | crash | Solver stuck at t=3e-8 (SBML math assumes SI) |

**Conclusion:** Tolerance tuning cannot close the remaining 1.5% gap. Both solvers have converged to their respective "true" numerical solutions. The divergence is intrinsic to solving in different unit systems.

## Options

### 1. libRoadRunner (recommended)

C/C++ library that loads SBML at runtime and JIT-compiles via LLVM.

- Uses CVODE internally, solves in model units like MATLAB
- C and C++ APIs for get/set species values between solver steps
- No code generation needed — link the library and load SBML
- Apache 2.0 license
- Designed for embedding in other applications
- https://github.com/sys-bio/roadrunner
- [Paper: libRoadRunner 2.0](https://academic.oup.com/bioinformatics/article/39/1/btac770/6883908)

**Integration sketch:**
```cpp
#include "rr/rrRoadRunner.h"

rr::RoadRunner rr("PDAC_model.sbml");
rr.simulate(t, t + dt);
double c1 = rr.getValue("V_T.C1");
rr.setValue("V_T.C1", new_value);
```

**Why this should work:** libRoadRunner solves in SBML's declared units (model units), same as MATLAB's SimBiology. This eliminates the unit-system divergence that causes the 1.5% gap.

### 2. AMICI

Reads SBML, symbolically derives ODE equations, generates native C++ code + SUNDIALS integration.

- Compiled C++ library, no runtime SBML parsing
- Maintained by a team solving these exact unit/solver issues
- Good fallback if generated-code approach is needed (e.g., GPU portability later)
- https://github.com/AMICI-dev/AMICI
- [Docs](https://amici.readthedocs.io/en/latest/about.html)

### 3. SimBiology Export (limited)

- Creates `SimBiology.export.Model` for MATLAB Compiler deployment
- Requires MATLAB runtime — too heavy for CUDA binary
- Not viable for standalone C++ integration
- https://www.mathworks.com/help/simbio/ref/export.html

### 4. Custom converter (current)

- Works: 98.5% pass rate, 155/162 species within 5% over 365 days (with reltol=1e-8)
- 7 immune species drift >5% after day 281+ (ArgI → Treg → DC cascade)
- Worst: V_T.ArgI at 22.3% relative diff at day 265
- Structural limit: cannot match MATLAB exactly due to SI vs model-unit solver paths

## Recommendation

**libRoadRunner** replaces the custom converter entirely. The ABM wrapper (`LymphCentral_wrapper.cpp`) would call roadrunner's C API instead of the generated ODE_system class. Crosstalk (ABM ↔ ODE) works the same way — read/write species values between solver steps. No unit conversion issues since roadrunner handles SBML semantics natively.

If exact MATLAB agreement is not required and 98.5% is acceptable, the custom converter works as-is with `reltol=1e-8, abstol=1e-11`.
