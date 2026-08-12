# Issue 1: CAF Gap-Closing Mechanism

**Status:** Temporarily solved (Aug 12 2026). α=0.01 gives ~53% fill; can push higher if needed.

## Problem

The QSP model predicts fibroblast populations via ODE terms that include a volume-expansion
dilution compensator — CAF proliferation offsets the density drop as V_T grows. The ABM domain
is fixed (100×100×12 voxels, 20 µm), so this compensator has no analogue. Directly porting
the QSP proliferation rates into the ABM caused unbounded fibroblast accumulation.

The original approach — seeding quiescent PSCs (qPSC) and letting them proliferate to fill the
gap — failed because at high domain occupancy (~99.7% of voxels occupied), the effective
division rate (2.3%/step) could not keep pace with activation drain (qPSC→iCAF/myCAF/apCAF at
5.6%/step) plus death (0.8%/step). This produced a sawtooth oscillation: qPSC pool collapses,
bootstrap fires, repeat every ~150 steps.

## Root Cause Analysis

Two separate issues were conflated:

1. **Mechanism failure (sawtooth):** qPSC proliferation requires empty neighbor voxels.
   At 99.7% voxel-count occupancy, division succeeds only ~2.3% of attempts. But activation
   drains qPSC at 5.6%/step regardless of occupancy → net qPSC rate is always negative →
   extinction is inevitable.

2. **Undershoot (10% fill at α=0.001):** After switching to direct recruitment, the initial
   α=0.001 was chosen conservatively (an earlier α=0.10 attempt recruited ~4,280 cells/step
   and crowded out cancer 108k→87k). But the crowding was voxel-count limited, not
   volume-limited. Actual volume occupancy was only 12% — the domain had 88% free volume.
   The steady-state formula `N_ss = α·T/(α+d)` with α=0.001 and d_avg=0.007 gives
   α/(α+d) = 12.4% fill, matching the observed ~10%.

## Solution: Direct CAF Recruitment

Replaced the qPSC-proliferation pathway with direct per-subtype gap-closing recruitment,
identical in pattern to the immune cell gap-closing (recruit_systematic in pde_integration.cu).

### Mechanism

Each ABM step, the host function computes per-subtype deficits against the QSP target
(scaled by the geometric fraction f = V_box/V_T):

```
deficit_Z = max(0, f · X_QSP_tumor,Z · N_A − N_ABM,Z)    for Z ∈ {iCAF, myCAF, apCAF}
deficit_total = deficit_iCAF + deficit_myCAF + deficit_apCAF
N_recruit = stochastic_round(α · deficit_total)
```

Each recruit is assigned a subtype proportional to its share of the deficit, placed at a
random voxel via `place_one` (linear scan for volume capacity), and given an exponentially
distributed lifespan from PARAM_FIB_LIFE_MEAN.

### Key Design Decisions

1. **CAF proliferation DISABLED.** The QSP's CAF prolif term compensates for V_T expansion
   (density dilution). The ABM domain is fixed → that term has no role and causes unbounded
   accumulation. Comment block in `fibroblast.cuh` state_step (~line 479).

2. **No qPSC intermediate.** The qPSC-proliferation pathway is structurally unable to sustain
   against activation drain at high occupancy. Direct recruitment eliminates the intermediate
   pool and the sawtooth.

3. **Random voxel placement.** Despite recruiting to random locations, spatial organization
   (φ_iCAF×C1) converges to 1.16–1.25 (>1 = positive colocalization with cancer). Cancer
   cells occupy ~85% of inhabited voxels, so random placement IS mostly near cancer.
   Differential survival amplifies the signal.

4. **α is SBI-tunable.** PARAM_FIB_GAP_ALPHA is an XML parameter, not hardcoded.

### Steady-State Theory

At equilibrium, recruitment equals death:

```
α · (T − N) = d · N
N_ss = α · T / (α + d)
Fill = α / (α + d)
```

Where d_avg ≈ 0.007/step is the composition-weighted death rate:
- iCAF: k_death=0.05/day → d=0.0125/step
- myCAF: k_death=0.015/day → d=0.00375/step
- apCAF: k_death=0.03/day → d=0.0075/step

| α     | N_ss  | Fill  | Recruits/step (at SS) |
|-------|-------|-------|-----------------------|
| 0.001 | 1,786 | 12.4% | ~13                   |
| 0.007 | 7,180 | 50.0% | ~51                   |
| 0.010 | 8,443 | 58.8% | ~68                   |
| 0.020 | 10,632| 74.0% | ~75                   |
| 0.063 | 12,928| 90.0% | ~91                   |

Time constant τ = 1/(α+d). At α=0.01: τ ≈ 59 steps ≈ 15 days.

### Cancer Crowding Boundary

- α=0.01 (current): peak recruitment ~428/step initially → cancer stable at 112k. No effect.
- α=0.10 (rejected): ~4,280/step → cancer dropped 108k→87k. Too aggressive.
- The crowding threshold is somewhere in (0.01, 0.10). The crowding mechanism is voxel-count
  competition (agents per voxel), not volume — domain is 80%+ free by volume at α=0.01.

## Code Changes

### `PDAC/agents/fibroblast.cuh`

- **state_step:** CAF proliferation block disabled with explanatory comment (~line 479).
  The `divide_reserve` gate rejects FIB_QUIESCENT, FIB_FRC, FIB_APCAF — only iCAF and myCAF
  can divide (and currently never trigger because the cooldown is never set).
- **divide_reserve / divide_confirm:** Reverted to 2-branch volume lookup (myCAF vs iCAF).

### `PDAC/pde/pde_integration.cu`

- **recruit_systematic (~line 2300):** Replaced the qPSC proliferation + boundary bootstrap
  block with direct CAF recruitment. Per-subtype deficits, proportional allocation,
  random-voxel placement via `place_one`, exponential lifespan.
- **place_recruited_agents (~line 1921):** CELL_TYPE_FIB case sets x,y,z, cell_state, life,
  divide_cooldown=0, divide_count=0, divide_flag=0, frc_dwell_counter=0, persist_dir=0,
  det_tag (type_nibble=8u).
- **Class audit CSV:** Removed `qpsc_div_prob` column.

### `PDAC/sim/resource/param_all_test.xml`

- `PARAM_FIB_GAP_ALPHA`: 0.001 → 0.01

### `PDAC/sim/model_definition.cu`

- Removed: `env.newProperty<float>("qpsc_div_prob", 0.0f);`

### `PDAC/abm/gpu_param.h` / `gpu_param.cu`

- PARAM_FIB_GAP_ALPHA enum entry, description, and environment registration (added in prior
  session, retained).

## Validation

Run: 200 steps (50 days), seed=42, 100×100×12 grid, QSP-coupled, Potts-seeded placement.

| Metric | α=0.001 | α=0.01 |
|--------|---------|--------|
| CAF total | 4,116 | 7,570 |
| CAF fill | 28.7% | 52.7% |
| Cancer | ~107k | 112,278 |
| φ(iCAF×C1) | 1.25 ± 0.015 | 1.16 ± 0.02 |
| φ CV (late) | 1.2% | <2% |
| Subtype ratio | iCAF:myCAF:apCAF = 33:57:10 | 36:54:10 |

Sub-type proportions track the QSP target well (QSP: 35:49:9).

## Diagnostics

- `results/2026-08-12_caf_diagnostic/caf_diagnostic_full.png` — 6-panel overview
- `results/2026-08-12_caf_diagnostic/caf_substate_tracking.png` — per-subtype trajectories
- `results/2026-08-12_caf_diagnostic/caf_composition_stack.png` — stacked composition
- `results/2026-08-12_caf_diagnostic/outputs_alpha01/` — raw output from α=0.01 run

## Open Items

- **α can increase further** if higher fill is needed. Next threshold to test: α=0.02 (74%
  predicted fill). Cancer crowding boundary unknown between 0.01 and 0.10.
- **B cell overabundance** (>3000% of QSP target) is the next composition issue — uncontrolled
  proliferation, not gap-closing. See Issue 2.
- **M1 macrophage recruitment = 0** — all MAC recruits are M2 (IFN-γ/IL-10 ratio never
  exceeds M1 threshold). Parameterization issue from QSP fit. See Issue 3.
