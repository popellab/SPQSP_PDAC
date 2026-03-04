# Occupancy Grid Implementation Work Notes
## Session: Feb 18, 2026

---

## Goal
Replace two-phase select+execute divide (via messages) with single-phase
atomicCAS divide via a shared `occ_grid` MacroProperty. Benefits:
- No message-passing overhead for division
- Runtime grid size flexibility (compile-time max = 128)
- Immediate grid update when a thread wins CAS (other threads see it)
- Reroll to next available voxel if first attempt loses to contention

---

## What Was Implemented

### New constants in `PDAC/core/common.cuh`
```cpp
constexpr int OCC_GRID_MAX = 128;   // Compile-time max; [0..grid_size-1] used at runtime
constexpr int NUM_OCC_TYPES = 8;    // Matches AgentType enum max index + 1
```

### MacroProperty in `PDAC/sim/model_definition.cu` → `defineEnvironment()`
```cpp
env.newMacroProperty<unsigned int,
    OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX, NUM_OCC_TYPES>("occ_grid");
```
Only in main model environment. Submodel environments do NOT get occ_grid.

### `write_to_occ_grid` functions (one per agent type)
Each writes the agent's current position into the occ_grid each step.

**cancer_write_to_occ_grid** (`cancer_cell.cuh`):
```cpp
occ[x][y][z][CELL_TYPE_CANCER].exchange(static_cast<unsigned int>(cell_state) + 1u);
// Stores cell_state+1 so 0=empty, 1=stem, 2=progenitor, 3=senescent
```

**tcell_write_to_occ_grid** (`t_cell.cuh`):
```cpp
occ[x][y][z][CELL_TYPE_T] += 1u;  // additive (multiple T cells per voxel allowed)
```

**treg_write_to_occ_grid** (`t_reg.cuh`):
```cpp
occ[x][y][z][CELL_TYPE_TREG] += 1u;
```

**mdsc_write_to_occ_grid** (`mdsc.cuh`):
```cpp
occ[x][y][z][CELL_TYPE_MDSC].exchange(1u);  // exclusive (1 MDSC per voxel)
```

**vascular_write_to_occ_grid** (`vascular_cell.cuh`):
```cpp
occ[x][y][z][CELL_TYPE_VASCULAR] += 1u;
```

### `zero_occupancy_grid` host function (`pde_integration.cu`)
```cpp
FLAMEGPU_HOST_FUNCTION(zero_occupancy_grid) {
    auto occ = FLAMEGPU->environment.getMacroProperty<unsigned int,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX, NUM_OCC_TYPES>("occ_grid");
    occ.zero();  // cudaMemset to 0
}
```
Declaration in `pde_integration.cuh`: `extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER zero_occupancy_grid;`

### New `divide` functions (single-phase)

**cancer_divide** (`cancer_cell.cuh`):
- Checks divideFlag, divideCD, cell_state (early return if no division needed)
- Scans 6 Von Neumann neighbors for voxels where CELL_TYPE_CANCER==0 && CELL_TYPE_MDSC==0
- Fisher-Yates partial shuffle for random order
- Atomically claims with `.CAS(0u, 1u)`: if returns 0 → won; else try next
- On win: inlines full divide logic (same as old execute_divide, with divideCountRemaining fix)
- `break` after successful divide

**tcell_divide** (`t_cell.cuh`):
- Only T_CELL_CYT state can divide; needs divide_limit > 0 and divide_cd == 0
- Checks `occ[nx][ny][nz][CELL_TYPE_T] < MAX_T_PER_VOXEL` for candidates
- AtomicAdd+undo pattern:
  ```cpp
  const unsigned int old = occ[cx][cy][cz][CELL_TYPE_T] + 1u;
  if (old >= MAX_T_PER_VOXEL) {
      occ[cx][cy][cz][CELL_TYPE_T] -= 1u;  // undo
      continue;
  }
  // Won: create daughter
  ```

**treg_divide** (`t_reg.cuh`):
- Same atomicAdd+undo pattern as T cells
- Uses CELL_TYPE_TREG

### Conditional registration in `model_definition.cu`
CRITICAL FIX: `write_to_occ_grid` must only be registered for the MAIN model (not submodels),
because submodel environments do NOT define `occ_grid`.

```cpp
// Cancer, TCell, TReg: inside if (include_state_divide) block
if (include_state_divide) {
    cancer_cell.newFunction("write_to_occ_grid", cancer_write_to_occ_grid);
    cancer_cell.newFunction("state_step", ...);
    cancer_cell.newFunction("divide", cancer_divide).setAgentOutput(cancer_cell);
}

// MDSC: inside if (include_state) block
if (include_state) {
    mdsc.newFunction("write_to_occ_grid", mdsc_write_to_occ_grid);
    mdsc.newFunction("state_step", ...);
}

// Vascular: always registered (defineVascularCellAgent has no submodel variant)
agent.newFunction("write_to_occ_grid", vascular_write_to_occ_grid);
```

### Layer changes in `model_layers.cu`

**Removed (8 layers):**
- select_divide_cancer, execute_divide_cancer
- postdivision_broadcast_cancer, postdivision_check_packing
- select_divide_tcell, execute_divide_tcell
- select_divide_treg, execute_divide_treg

**Added (5 layers, after vascular movement, before solve_qsp):**
```cpp
// Layer: zero_occ_grid (host)
layer.addHostFunction(zero_occupancy_grid);

// Layer: write_to_occ_grid (all agent types in one device layer)
layer.addAgentFunction(AGENT_CANCER_CELL, "write_to_occ_grid");
layer.addAgentFunction(AGENT_TCELL,       "write_to_occ_grid");
layer.addAgentFunction(AGENT_TREG,        "write_to_occ_grid");
layer.addAgentFunction(AGENT_MDSC,        "write_to_occ_grid");
layer.addAgentFunction(AGENT_VASCULAR,    "write_to_occ_grid");

// Layer: divide_cancer
layer.addAgentFunction(AGENT_CANCER_CELL, "divide");

// Layer: divide_tcell
layer.addAgentFunction(AGENT_TCELL, "divide");

// Layer: divide_treg
layer.addAgentFunction(AGENT_TREG, "divide");

// Vascular still uses two-phase (select_divide_vascular + execute_divide_vascular) - KEPT
```

### CMakeLists.txt change
Added to disable FLAMEGPU2 seatbelt checks:
```cmake
set(FLAMEGPU_SEATBELTS OFF CACHE BOOL "Disable FLAMEGPU2 seatbelt checks" FORCE)
```

---

## FLAMEGPU2 MacroProperty API (v2.0.0-rc.4)

The DeviceMacroProperty uses operator overloads for atomics:
```cpp
auto occ = FLAMEGPU->environment.getMacroProperty<unsigned int,
    OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX, NUM_OCC_TYPES>("occ_grid");

occ[x][y][z][t].CAS(compare, val)   // atomic compare-and-swap, returns old value
occ[x][y][z][t] + 1u                // atomicAdd, returns OLD value
occ[x][y][z][t] += 1u               // atomicAdd, no return
occ[x][y][z][t] - 1u                // atomicSub, returns old value
occ[x][y][z][t] -= 1u               // atomicSub, no return
occ[x][y][z][t].exchange(val)       // atomicExch, returns old value

// Host only:
occ.zero()                          // cudaMemset whole property to 0
```
NOT: `.atomicCAS()`, `.atomicAdd()`, etc. — those do NOT exist in this version.

---

## Current Crash Status (Feb 18, 2026)

### Error 1: `cudaErrorIllegalAddress`
- Occurs after "PDE solved for step 0" in pre-simulation Phase 3
- Location: `CUDAFatAgent.cu(133)` = `gpuErrchk(cudaStreamSynchronize(stream))`
  inside `processDeath` → `scatterDeath`
- This sync catches an async error from a kernel that ran after the PDE solve

### Error 2: FLAMEGPU2 DeviceError (the REAL root cause)
```
Device function 'divide' reported 3650 errors.
DeviceMacroProperty read and atomic write operations cannot be mixed in the same layer.
DeviceMacroProperty.cuh(214)
```
- Even though `FLAMEGPU_SEATBELTS OFF` is set in CMakeLists.txt, the check still runs
- The device-side check at `DeviceMacroProperty.cuh(214)` is:
  `#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS`
- If `FLAMEGPU_SEATBELTS` is NOT defined (vs defined as 0), check still triggers
- Setting `OFF` in CMake might not emit `-DFLAMEGPU_SEATBELTS=0` to the compiler

### Why `cancer_divide` triggers the seatbelt:
In `cancer_divide`, within the SAME function:
1. **READ**: `occ[nx][ny][nz][CELL_TYPE_CANCER] == 0u` (checking candidates)
2. **ATOMIC WRITE**: `occ[cand_x[i]][...][CELL_TYPE_CANCER].CAS(0u, 1u)` (claiming)
FLAMEGPU2 seatbelt flags any mixed read + atomic-write in the same layer.

---

## Fix Options (To Try Next Session)

### Option A: Force `-DFLAMEGPU_SEATBELTS=0` explicitly
In CMakeLists.txt, after the flamegpu target is available:
```cmake
target_compile_definitions(pdac PRIVATE FLAMEGPU_SEATBELTS=0)
```
OR add to compile options:
```cmake
target_compile_options(pdac PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-DFLAMEGPU_SEATBELTS=0>
)
```
This ensures the preprocessor symbol is DEFINED as 0, satisfying
`#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS` → `false || 0 = false`.

### Option B: Split divide into two layers (seatbelt-compliant)
1. **Layer `scan_divide_candidates`**: READ occ_grid, save N candidates to agent variables
   (e.g., `cand_voxel_0`, `cand_voxel_1`, ..., `n_cands` as agent variables)
2. **Layer `divide`**: Only WRITE (CAS) to occ_grid using saved candidates — no reads

Agent variable changes needed for cancer:
```cpp
cancer_cell.newVariable<int>("cand_x_0"); cancer_cell.newVariable<int>("cand_y_0"); ...
cancer_cell.newVariable<int>("n_cands", 0);
```

### Option C: Encode candidate positions in available_neighbors bitmask
The `available_neighbors` bitmask (already exists as agent variable) encodes which
of the 6 Von Neumann neighbors are empty. Set this in `write_to_occ_grid` or a
separate `scan_candidates` layer, then `divide` uses the bitmask without reading occ_grid.
But still need CAS write — FLAMEGPU2 might allow write-only operations in a layer.

### Option D: Use a separate "intent" MacroProperty
Two-step with occupancy:
1. **Layer `claim_divide`**: Only WRITE to a `claim_grid` MacroProperty via CAS (no reads)
2. **Layer `execute_divide`**: Only READ `claim_grid` to see if this agent won (no writes)

---

## Also Fixed This Session

### divideCountRemaining bug (cancer progenitor daughters)
In `cancer_execute_divide` (old two-phase function) AND in new `cancer_divide`:
- **Before (BUG)**: `FLAMEGPU->agent_out.setVariable<int>("divideCountRemaining", divMax)`
  → ALL daughters got 9 fresh divisions regardless of parent's state
- **After (FIX)**: `FLAMEGPU->agent_out.setVariable<int>("divideCountRemaining", divideCountRemaining)`
  where `divideCountRemaining` is the parent's count AFTER decrementing
- Matches HCC CPU model: daughters inherit parent's already-decremented count
- Only asymmetric stem→progenitor birth still uses `divMax` (new progenitor, fresh divisions)

---

## Files Modified Summary
| File | Change |
|------|--------|
| `PDAC/core/common.cuh` | Added OCC_GRID_MAX=128, NUM_OCC_TYPES=8 |
| `PDAC/agents/cancer_cell.cuh` | Added cancer_write_to_occ_grid, cancer_divide; fixed divideCountRemaining |
| `PDAC/agents/t_cell.cuh` | Added tcell_write_to_occ_grid, tcell_divide |
| `PDAC/agents/t_reg.cuh` | Added treg_write_to_occ_grid, treg_divide |
| `PDAC/agents/mdsc.cuh` | Added mdsc_write_to_occ_grid |
| `PDAC/agents/vascular_cell.cuh` | Added vascular_write_to_occ_grid |
| `PDAC/pde/pde_integration.cu` | Added zero_occupancy_grid host function |
| `PDAC/pde/pde_integration.cuh` | Added extern declaration for zero_occupancy_grid |
| `PDAC/sim/model_definition.cu` | Registered new functions; added MacroProperty; write_to_occ_grid conditional |
| `PDAC/sim/model_layers.cu` | Replaced 8 old divide layers with 5 new layers |
| `PDAC/sim/CMakeLists.txt` | Added FLAMEGPU_SEATBELTS OFF (not fully working) |

---

## ✅ RESOLVED (Feb 18, 2026)

### Root cause of both errors

**Error 1 (DeviceError / seatbelt)**: `write_to_occ_grid` was registered unconditionally
in `defineCancerCellAgent()` etc., so it was also registered in movement submodels.
Movement submodel environments don't define `occ_grid`. FLAMEGPU2 tried to access an
undefined macro property, triggering the seatbelt flag check on a null pointer.

**Error 2 (cudaErrorIllegalAddress)**: The old two-phase `select_divide_vascular +
execute_divide_vascular` used MSG_INTENT for conflict resolution. MSG_INTENT messages
from the movement phase were still in the spatial hash when `execute_divide_vascular`
tried to read them, causing an illegal spatial message access.

### Fixes applied
1. Moved `write_to_occ_grid` registrations inside `if (include_state_divide)` and
   `if (include_state)` conditional blocks → only registered in main model.
2. Replaced two-phase vascular divide with single-phase `vascular_divide` (no messages).

### Verified working
```
./build/bin/pdac -s 20 -g 21 -oa 0 -op 0 -i 1
# Pre-simulation (56 steps) + 20 main steps: clean run, no errors
```

---

## ✅ RESOLVED (Feb 18, 2026 Session 2): Movement Submodels Replaced

### What was done
Replaced all 4 movement submodels (cancer, tcell, treg, mdsc) and the two-phase
vascular move (select_move_target + execute_move) with single-phase functions
that directly use the occ_grid MacroProperty.

### New `X_move` functions added

**cancer_move** (`cancer_cell.cuh`):
- ECM_sat=0.2 skip probability
- Check 6 VN neighbors: CELL_TYPE_CANCER==0 && CELL_TYPE_MDSC==0
- Fisher-Yates shuffle candidates
- CAS(0u, cell_state+1u) to claim; exchange(0u) to release old

**tcell_move** (`t_cell.cuh`):
- ECM_sat=0.2 skip; dead check
- Check capacity: MAX_T_PER_VOXEL (8) normally, MAX_T_PER_VOXEL_WITH_CANCER (1) if cancer present
- Fisher-Yates shuffle candidates
- atomicAdd+undo: `old = occ[CELL_TYPE_T]+1u; if old>=max_t undo, else won`
- -= 1u to release old voxel

**treg_move** (`t_reg.cuh`):
- Same as tcell_move but for CELL_TYPE_TREG
- Also fixes old direction function bug (old code called `get_moore_direction_t` instead of treg version)

**mdsc_move** (`mdsc.cuh`):
- ECM_sat=0.2 skip
- Check CELL_TYPE_MDSC==0 only (MDSCs can share with cancer)
- Fisher-Yates shuffle
- CAS(0u, 1u) to claim; exchange(0u) to release

**vascular_move** (`vascular_cell.cuh`):
- Only VAS_TIP cells move; STALK/PHALANX return early
- Run-tumble algorithm from old vascular_select_move_target (inline combined)
- No occ_grid access needed (tip cells not tracked in occ_grid)
- No conflict resolution (tip cells can share voxels)
- Directly sets x, y, z

### model_definition.cu changes
- Removed: buildMovementSubmodel, createMovementSubmodel, defineSubmodelEnvironment
- Removed: defineIntentMessage() call from buildModel() (still defined as dead code)
- Removed: old select_move_target / execute_move registrations from all define*Agent()
- Removed: reset_moves registration from defineCancerCellAgent()
- Removed: old vascular two-phase divide registrations (select_divide_target, execute_divide)
- Added: `move` function registration inside if(include_state_divide) for cancer/tcell/treg
- Added: `move` function registration inside if(include_state) for MDSC
- Added: `move` function registration (unconditional) for vascular
- buildModel() simplified: no submodel loop, no reset_cancer_moves layer

### model_layers.cu changes
- Removed: vascular_select_move + vascular_execute_move layers
- Added: zero_occ_grid + write_to_occ_grid BEFORE movement layers
- Added: repeated `move_X_N` layers (N=move_steps from XML) for cancer/tcell/treg/mdsc
- Added: single `move_vascular` layer
- Reads move step counts from model.Environment().getProperty<int>() at model build time

### Layer order (final)
1. update_agent_counts ... solve_pde (unchanged)
2. zero_occ_grid + write_to_occ_grid
3. move_cancer_0..N, move_tcell_0..N, move_treg_0..N, move_mdsc_0..N, move_vascular
4. divide_cancer, divide_tcell, divide_treg, divide_vascular
5. solve_qsp

### Verified working
```
./build/bin/pdac -s 10 -g 21 -oa 0 -op 0 -i 0  # Clean, 18s
./build/bin/pdac -s 20 -g 21 -oa 0 -op 0 -i 1  # Clean, 27s
```
