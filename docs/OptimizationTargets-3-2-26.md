# SPQSP PDAC — Optimization Targets (Mar 2, 2026)

## Methodology

Per-layer timing infrastructure added Mar 2, 2026:
- 8 FLAMEGPU host-function "checkpoint" layers inserted at phase boundaries in `model_layers.cu`
- Timing written to `outputs/layer_timing.csv` (long format: `step, layer, ms`) each step
- NVTX markers on all host functions (Nsight Systems compatible)
- Analysis script: `python/analyze_layers.py`
- Nsight profile command: `nsys profile --nvtx-include ".*" --output profile ./build/bin/pdac -s 2 -g 50 -oa 0 -op 0`

Test conditions: 50 steps, initial defaults (radius=5, 50 T cells, 50 fibroblasts, 50 vascular, etc.)

---

## Benchmark Results

### 50³ grid — no I/O (`-oa 0 -op 0`)

| Phase | Mean (ms) | Median (ms) | Std | % Total |
|---|---|---|---|---|
| ECM Update | 164 | 168 | 26 | **65.1%** |
| State + Sources | 42 | 18 | 44 | 16.7% |
| PDE Solve | 28 | 30 | 7 | 10.9% |
| QSP Solve | 21 | 19 | 18 | 8.4% |
| Movement | 14 | 13 | 3 | 5.7% |
| Other/Overhead | 4 | 4 | 0.5 | 1.7% |
| Broadcast + Scan | 2 | 2 | 0.6 | 0.7% |
| Division | 2 | 2 | 1 | 0.7% |
| Recruitment | 2 | 1 | 0.3 | 0.6% |
| **TOTAL** | **253** | | | |

### 150³ grid — no I/O (`-oa 0 -op 0`)

| Phase | Mean (ms) | Median (ms) | Std | % Total |
|---|---|---|---|---|
| ECM Update | 520 | 527 | 76 | **45.5%** |
| State + Sources | 352 | 345 | 127 | **30.8%** |
| PDE Solve | 142 | 142 | 1 | 12.4% |
| Other/Overhead | 80 | 83 | 18 | 7.0% |
| Broadcast + Scan | 28 | 28 | 4 | 2.4% |
| QSP Solve | 20 | 18 | 17 | 1.8% |
| Movement | 20 | 21 | 3 | 1.8% |
| Recruitment | 17 | 17 | 1 | 1.5% |
| Division | 10 | 8 | 8 | 0.9% |
| **TOTAL** | **1,143** | | | |

### 150³ grid — with full I/O (`-oa 1 -op 1`)

| Phase | Mean (ms) | % Total |
|---|---|---|
| **Other/Overhead (I/O)** | **5,400** | **82.7%** |
| ECM Update | 564 | 8.6% |
| State + Sources | 318 | 4.9% |
| PDE Solve | 144 | 2.2% |
| All other compute | ~100 | 1.6% |
| **TOTAL** | **6,526** | |

---

## Scaling Analysis

Grid volume grows 27× from 50³ → 150³. Actual scale factors:

| Phase | Actual scale | Expected (×27) | Interpretation |
|---|---|---|---|
| ECM Update | 3.1× | 27× | CPU-bound, memory bandwidth limited — will keep growing linearly with voxels |
| State + Sources | 19× | 27× | Tracks agent count (~25× more agents); healthy GPU scaling |
| PDE Solve | 4.7× | 27× | Excellent — GPU far better saturated at 150³ than 50³ |
| Broadcast + Scan | 17× | 27× | Near-linear with agents, expected |
| Movement | 1.6× | ~1× | Agent-count limited, mostly saturated |
| I/O Overhead | ~68× | — | Catastrophic — ASCII CSV of 3.375M voxels × 10 chemicals every step |

---

## Optimization Targets (Priority Order)

### 🔴 1. Output interval / binary format (immediate, no rebuild needed)

**Problem:** At 150³ with I/O enabled, CSV writes consume 82.7% of wall time (5.4 sec/step).
The PDE CSV alone writes 3.375M voxels × 10 chemicals as ASCII text every step.

**Options (in increasing effort):**
- **Increase output interval** — pass `--interval-out N` (or set `<interval_out>` in XML). Every 10 steps = 10× I/O reduction, zero code change.
- **Binary output (HDF5/raw float32)** — would cut PDE I/O by ~50–100× vs ASCII CSV. Requires rewriting `exportPDEData`.

**Expected gain:** 5.4 sec/step → ~0.1–0.5 sec/step (with interval=10 and/or binary).

---

### 🔴 2. ECM Update → CUDA kernel

**Problem:** `update_ecm_grid` in `PDAC/pde/pde_integration.cu:757` is a triple nested CPU loop over all voxels:
```cpp
for (int i = 0; i < grid_x; i++)
    for (int j = 0; j < grid_y; j++)
        for (int k = 0; k < grid_z; k++) { /* per-voxel ECM update */ }
```
Each voxel is independent (reads `ecm[i][j][k]` and `field[i][j][k]`, writes `ecm[i][j][k]`).
At 150³ this takes 527 ms/step. At 320³ it will take ~5+ sec/step.

**Fix:** Replace with a CUDA kernel — one thread per voxel. The MacroProperty arrays (`ecm_grid`, `fib_density_field`) can be accessed via raw device pointers.

**Expected gain:** 527 ms → ~2–5 ms (100–200× speedup from parallelism alone).

**Files to change:**
- `PDAC/pde/pde_integration.cu` — `update_ecm_grid` host function
- `PDAC/pde/pde_integration.cuh` — no interface changes needed

---

### 🟡 3. Recruitment source scan → GPU

**Problem:** `recruit_t_cells`, `recruit_mdscs`, `recruit_macrophages` all do:
```cpp
cudaMemcpy(h_sources.data(), d_sources, total_voxels * sizeof(int), cudaMemcpyDeviceToHost);
for (int idx = 0; idx < total_voxels; idx++) { /* scan and create agents */ }
```
At 150³ this copies 3.375M ints host-side every step and loops through them serially.
Currently 80 ms/step (7% at 150³ no-I/O); will grow linearly with voxels.

**Fix:** GPU kernel to count and collect candidate source voxels into a compact list (prefix-sum / stream compaction), then loop only over the ~few hundred actual sources. The agent creation loop remains on host but operates on a much smaller list.

**Expected gain:** 80 ms → ~5–10 ms.

**Files to change:**
- `PDAC/pde/pde_integration.cu` — `recruit_t_cells`, `recruit_mdscs`, `recruit_macrophages`

---

### 🟡 4. State + Sources — investigate scaling (TODO)

**Observation:** State + Sources scales 19× from 50³ → 150³, tracking the ~25× agent count growth.
The high variance (Std=127ms at 150³, IQR ~180–460ms) and outlier spikes (735ms) suggest something
beyond simple O(N) kernel launch overhead.

**To investigate:**
- Use Nsight Systems / Nsight Compute to profile the individual agent kernels inside this phase:
  `state_transitions` (7 agent types in one layer) and `compute_chemical_sources` (7 agent types)
- Check for: warp divergence in `cancer_cell state_step` (PDL1 Newton-Raphson), register pressure,
  shared memory usage, occupancy on each kernel
- The early-step spikes (steps 0–5) are likely cold-cache effects or FLAMEGPU2 internal memory reallocation
  as agent counts grow rapidly from cancer cell division
- Check if `aggregate_abm_events` (which iterates all cancer agents on host to check `dead` flag)
  is leaking into this phase's timing

**Files to examine:**
- `PDAC/agents/cancer_cell.cuh` — `state_step`, `compute_chemical_sources`
- `PDAC/agents/t_cell.cuh`, `macrophage.cuh`, etc.
- `PDAC/sim/model_layers.cu` — verify layer grouping in `state_transitions` layer

---

### 🟢 5. PDE Solve — already well-optimized, monitor at 320³

The CG solver scales well (4.7× for 27× more voxels) because it was GPU-bound but under-utilized at
small grid sizes. At 320³ (~27× more voxels than 150³), expect roughly another 4–5× increase, putting
it at ~600–700 ms/step. Worth monitoring but not a priority now.

---

## Summary Table

| Target | Effort | Expected gain (150³) | Priority |
|---|---|---|---|
| Output interval / binary I/O | Low | -5,400 ms/step | 🔴 Immediate |
| ECM → CUDA kernel | Medium | -520 ms/step | 🔴 High |
| Recruitment scan → GPU | Medium | -70 ms/step | 🟡 Medium |
| State+Sources investigation | Investigation | TBD | 🟡 Medium |
| PDE solver | Low (monitor) | negligible now | 🟢 Low |
