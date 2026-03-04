# GPU PDE Solver Options: Quick Reference & Decision Tree

## Decision Tree: Which Solver Should I Use?

```
START: Do you need to change PDE solvers?
│
├─ "No, current solver works fine"
│  └─→ STOP: Keep custom CG solver ✅
│      (lowest risk, tightest integration)
│
├─ "Single-GPU, want better performance or preconditioning"
│  │
│  ├─ "Worried about solver becoming bottleneck?"
│  │  └─→ GINKGO (3-4 weeks)
│  │      ✅ Matrix-free, modern C++, good preconditioning
│  │      ✅ Minimal FLAME GPU friction
│  │
│  └─ "Matrix assembly acceptable for better AMG?"
│     └─→ AMGX (4-6 weeks)
│         ✅ Native CUDA AMG, no assembly each step (static D/λ)
│         ❌ Community-maintained (maintenance risk)
│
└─ "Multi-GPU MPI research needed"
   │
   ├─ "Want unified SUNDIALS infrastructure with QSP?"
   │  └─→ SUNDIALS + hypre (6-8 weeks)
   │      ✅ LLNL-proven at exascale, MPI support
   │      ⚠️ Data transfer overhead (GPU↔CPU↔GPU)
   │
   └─ "Want most flexible solver library?"
      └─→ PETSc (8-12 weeks, long-term benefit)
          ✅ Best for unstructured meshes, distributed computing
          ❌ Steep learning curve, overkill for Cartesian grids
```

---

## Quick Comparison: For Your Problem (3D Structured Grid + Diffusion-Reaction)

| Factor | Best | Good | Avoid |
|--------|------|------|-------|
| **Ease of Integration** | Custom CG | Ginkgo | PETSc |
| **Single-GPU Speed** | cuSPARSE | Custom CG | Julia |
| **Multi-GPU Scaling** | SUNDIALS | Ginkko | AmgX |
| **Preconditioning** | PETSc | Ginkgo, AmgX | Custom CG |
| **Maintenance** | Ginkko | SUNDIALS | AmgX |
| **Documentation** | SUNDIALS, PETSc | cuSPARSE | Kokkos |

---

## Library Maturity & Roadmap (Feb 2026)

| Library | Status | Actively Developed? | Risk Level | Notes |
|---------|--------|---------------------|-----------|-------|
| **SUNDIALS 6.x** | ✅ Prod | Yes (LLNL) | Low | Major releases yearly; GPU support expanding |
| **Ginkgo 1.10+** | ✅ Prod | Yes (Jülich) | Very Low | Monthly releases; excellent maintenance |
| **AmgX 2.3** | ⚠️ Stable | No (NVIDIA paused) | Medium | Last major update 2020; community forks exist |
| **PETSc 3.20+** | ✅ Prod | Yes (ANL) | Low | Major releases yearly; HPC standard |
| **cuSPARSE 13.x** | ✅ Prod | Yes (NVIDIA) | Very Low | Updated with CUDA releases |
| **Julia SciML** | ✅ Prod | Yes (Rackauckas) | Low | Monthly releases; language-dependent risk |
| **Kokkos** | ✅ Prod | Yes (Sandia/DoE) | Very Low | Quarterly releases; exascale-funded |

---

## Integration Effort Breakdown (in Weeks)

### Ginkko (Recommended for Single-GPU Scaling)
```
Week 1:  Setup Ginkgo build, understand executor model      (5 hrs)
Week 2:  Implement DiffusionOperator custom operator        (10 hrs)
Week 3:  Integrate with FLAME GPU, numerical validation     (12 hrs)
Week 4:  Preconditioner tuning, performance testing         (8 hrs)
─────────────────────────────────────────────────────────────
Total:   ~35 hours (4.5 person-days)
         Can parallelize with biology development
```

### SUNDIALS + hypre (Multi-GPU/MPI)
```
Week 1-2:  Learn SUNDIALS residual-based interface          (15 hrs)
Week 3:    Wrap hypre StructMat for your grid               (12 hrs)
Week 4:    Implement residual function + source coupling    (15 hrs)
Week 5:    Integrate with QSP CVODE wrapper                 (10 hrs)
Week 6:    MPI data layout, halo exchange                   (15 hrs)
Week 7:    Numerical validation vs current solver           (12 hrs)
Week 8:    Performance tuning, solver option tuning         (8 hrs)
─────────────────────────────────────────────────────────────
Total:   ~87 hours (11 person-days)
         Can't easily parallelize (architecture changes)
```

---

## Solver Choice Checklist

### Use Current Custom CG If:
- [ ] Grids currently < 100³
- [ ] PDE solve < 30% of total ABM time
- [ ] No multi-GPU plans
- [ ] CG convergence acceptable (< 100 iters/step/substrate)
- [ ] Team familiar with current code
- [ ] Development velocity prioritized over robustness

### Switch to Ginkgo If:
- [ ] Grids approaching 100³+
- [ ] PDE solve becoming bottleneck (> 50% time)
- [ ] Want multigrid preconditioning
- [ ] Prefer modern C++ (C++17)
- [ ] May need AMD/Intel GPU support later
- [ ] Can invest 3-4 weeks once

### Switch to SUNDIALS + hypre If:
- [ ] Planning multi-GPU research (>1 GPU)
- [ ] Want unified SUNDIALS infrastructure (already using for QSP)
- [ ] Long-term production code (> 5 years)
- [ ] Can accept 6-8 week integration
- [ ] Have HPC allocation (MPI needed for benefit)

### Never Use:
- [ ] Julia SciML (incompatible with FLAME GPU 2 C++ base)
- [ ] PETSc (overkill for Cartesian grids)
- [ ] cuSPARSE alone (Ginkgo better for matrix-free; AmgX better for assembled)
- [ ] Kokkos (adds complexity for structured grids)

---

## Performance Expectations

### Current Custom CG (Baseline)
```
Grid:        50³          100³         150³
Chemicals:   10           10           10
Solver Time: 50 ms/step   400 ms/step  1500 ms/step
Iter/Subst:  20-50        50-150       150-300+
```

### With Ginkko (Estimate)
```
Grid:        50³          100³         150³
Chemicals:   10           10           10
Solver Time: 35 ms/step   250 ms/step  1000 ms/step  (~30-35% faster, better PC)
Iter/Subst:  15-30        30-80        80-150        (fewer iterations w/ AMG/ILU)
```

### With SUNDIALS (Multi-GPU, MPI)
```
Grid:        100³         200³         300³
Chemicals:   10           10           10
1 GPU:       400 ms/step  2500 ms/step OOM
2 GPU:       250 ms/step  1200 ms/step 3000 ms/step
4 GPU:       150 ms/step  700 ms/step  1500 ms/step  (strong scaling)
```

---

## Key Parameters to Monitor

When evaluating solver performance, track:

```cpp
struct SolverMetrics {
    float solver_time_ms;              // Wall clock time
    int   iterations;                  // CG/GMRES iterations
    float residual_final;              // Final ||Ax - b||
    float residual_reduction;          // Relative reduction (||r_n|| / ||r_0||)
    float convergence_rate;            // Iteration-to-iteration rate
    int   preconditioner_setup_time;   // If using PC
    int   matvec_calls;                // SpMV kernel calls
};
```

**Red Flags**:
- Iterations > 200 per substrate → change preconditioner
- solver_time > 50% of ABM step → consider matrix-free solver
- residual_final not < tolerance → reduce tolerance or increase max_iters

---

## Integration Points: FLAME GPU ↔ Solver

### Data Flow (Current CG)
```
FLAME GPU PDE Grid
        ↓ (device pointers)
   Custom CG Kernels (GPU)
        ↓ (solution)
FLAME GPU PDE Grid
```
**Overhead**: Minimal, all GPU-side

### Data Flow (Ginkko)
```
FLAME GPU PDE Grid
        ↓ (Ginkgo Vector from device ptr)
   Ginkgo CG + Preconditioner
        ↓ (Ginkgo Vector)
FLAME GPU PDE Grid
```
**Overhead**: Minimal, all GPU-side, one abstraction layer

### Data Flow (SUNDIALS)
```
FLAME GPU PDE Grid
        ↓ (copy to CPU)
   Host Data Packing
        ↓
   SUNDIALS N_Vector (CPU)
        ↓
   SUNDIALS Solver
        ↓
   N_Vector → CPU
        ↓
   Copy to GPU
        ↓
FLAME GPU PDE Grid
```
**Overhead**: GPU↔CPU transfer is expensive (50-200 ms per solve!)

---

## Build Integration Examples

### Current (Custom CG)
```cmake
# In CMakeLists.txt - no new dependencies
target_sources(pdac PRIVATE ../pde/pde_solver.cu)
```

### Ginkko
```cmake
# Add Ginkgo dependency (header-only friendly)
FetchContent_Declare(ginkgo
    GIT_REPOSITORY https://github.com/ginkgo-project/ginkgo.git
    GIT_TAG master)
FetchContent_MakeAvailable(ginkgo)
target_link_libraries(pdac PRIVATE ginkgo)
```

### SUNDIALS + hypre
```cmake
# Existing SUNDIALS setup + new hypre
find_package(HYPRE REQUIRED)  # Requires separate hypre install
target_link_libraries(pdac PRIVATE ${SUNDIALS_LIBRARIES} ${HYPRE_LIBRARIES})
target_include_directories(pdac PRIVATE ${HYPRE_INCLUDE_DIR})
```

### PETSc
```cmake
# Complex, PETSc recommends using PETSc's own CMake
# or invoking PETSc compiler wrappers
execute_process(COMMAND petsc-build-dir/lib/petsc/conf/petscvariables
                OUTPUT_VARIABLE PETSC_LIBS)
```

---

## Solver Option Strengths by Use Case

| Use Case | Best Choice | Why |
|----------|------------|-----|
| **Rapid Prototyping** | Custom CG | Working, zero learning curve |
| **Publication (Single-GPU)** | Ginkko | Clean code, reproducible |
| **Production Single-GPU** | Ginkko | Modern, maintained, portable |
| **HPC Exascale** | SUNDIALS + hypre | MPI, proven at scale |
| **Unstructured Mesh** | PETSc | Best for adaptive meshes |
| **Teaching/Learning** | Ginkko | Best documentation |

---

## Red Flags: When Current CG May Fail

- [ ] CG diverges on large grids (conditioning issues)
- [ ] Solver time > 60% of ABM step
- [ ] Memory usage > 80% GPU VRAM
- [ ] CG iterations > 300 per substrate
- [ ] Numerical instability (NaN/Inf concentrations)
- [ ] Grids > 150³ causing timeout

**Action**: Run scaling test on 100³ and 150³ grids; benchmark CG iterations. If any red flags appear, move Ginkko adoption to priority.

---

## Summary Decision Matrix

```
Your Question              → Answer           → Recommendation
────────────────────────────────────────────────────────────────
"Solver becoming bottleneck?"
  No                          → Keep current      STATUS: ✅ All OK
  Yes                         → Upgrade solver    WHEN: Now

"Need multi-GPU/MPI?"
  No                          → Ginkko            EFFORT: 3-4 weeks
  Yes                         → SUNDIALS+hypre    EFFORT: 6-8 weeks

"Prefer modern C++?"
  Yes                         → Ginkko            BENEFIT: ⭐⭐⭐⭐
  No                          → Custom/SUNDIALS   BENEFIT: ⭐⭐⭐

"Have HPC allocation?"
  Yes                         → SUNDIALS+hypre    PAYOFF: ⭐⭐⭐⭐⭐
  No                          → Ginkko            PAYOFF: ⭐⭐⭐⭐

"Can spend 6+ weeks on solver?"
  Yes (long-term)             → PETSc             FUTURE: ⭐⭐⭐⭐⭐
  No (focused on biology)     → Ginkko/Keep      FUTURE: ⭐⭐⭐
```

---

## Key Contacts & Resources

### Library Maintainers (Questions/Issues)
- **Ginkko**: GitHub Issues @ ginkgo-project/ginkgo
- **SUNDIALS**: Bitbucket issues + LLNL support mailing list
- **AmgX**: GitHub (NVIDIA/AmgX), but slower response
- **PETSc**: PETSc mailing list, excellent support
- **NVIDIA cuSPARSE**: NVIDIA Developer Forum

### Technical Documentation
- Ginkgo: [ginkgo-project.github.io](https://ginkgo-project.github.io/)
- SUNDIALS: [sundials.readthedocs.io](https://sundials.readthedocs.io/)
- PETSc: [petsc.org](https://petsc.org/)
- AmgX: [github.com/NVIDIA/AMGX](https://github.com/NVIDIA/AMGX) Wiki
- FLAME GPU: [flamegpu.com](https://flamegpu.com/)

---

**Last Updated**: February 23, 2026
**For**: SPQSP PDAC GPU-accelerated ABM
**Status**: Ready for decision & implementation planning
