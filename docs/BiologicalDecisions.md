# Biological Decisions & Inconsistencies

This file tracks places where the PDAC GPU model intentionally or unintentionally diverges
from the HCC CPU reference, along with the biological justification (or lack thereof).

Each entry notes: the location, what HCC does, what PDAC does, which is more biologically
correct, and the current status.

---

## T Cell Exhaustion: TReg vs PDL1 Suppression (Independent or Mutually Exclusive?)

**File**: `PDAC/agents/t_cell.cuh` — `tcell_state_step`, inside `if (cell_state == T_CELL_CYT)` block

**HCC behavior** (`TCell.cpp:308`):
```cpp
if (_count_neighbor_Treg > 0) { /* TReg suppression roll */ }
else if (_count_neighbor_all > 0) { /* PDL1 suppression roll */ }
```
TReg suppression and PDL1 suppression are **mutually exclusive**: if any TReg neighbor is
present, the PDL1 check is skipped entirely.

**PDAC behavior** (after revert, currently matches HCC with `else if`):
```cpp
if (neighbor_Treg > 0) { /* TReg suppression roll */ }
else if (neighbor_all > 0) { /* PDL1 suppression roll */ }
```

**Biological argument for double `if` (independent checks)**:
TReg-mediated suppression (IL-10, TGF-β, direct contact) and PD1-PDL1 checkpoint signaling
are completely independent molecular pathways. A cytotoxic T cell surrounded by both TReg
cells and PDL1-expressing cancer cells would face both suppression signals simultaneously.
The `else if` formulation means TReg presence causes the model to ignore PDL1 suppression,
which has no mechanistic justification.

**Current status**: Using `else if` (matches HCC reference) to aid comparison.
**Recommended future change**: Switch to independent `if` / `if` blocks once HCC comparison
is no longer the primary validation target, or update both models simultaneously.

---

## T Cell Tumble Phase: Weighted Direction vs Uniform Random

**File**: `PDAC/agents/t_cell.cuh` — `tcell_move`, tumble phase (~line 540)

**HCC behavior** (`TCell.cpp:149-162`):
During tumble, picks any open Von Neumann neighbor with **uniform random probability**
(`getOneOpenVoxel` from `getMoveDestinationVoxels`). No directional weighting.

**PDAC behavior**:
Applies a sigma-weighted formula `exp(cos_theta/(σ²)) / exp(1/σ²)` (σ=0.524) that biases
the new direction toward the previous movement direction even during tumble.

**Biological argument**:
The uniform random tumble is the standard run-and-tumble model (e.g., Berg & Brown). The
weighted formula gives PDAC T cells stronger directional memory than the reference model,
causing different spatial clustering behavior.

**Current status**: Fixed — replaced sigma-weighted formula with uniform random pick from all 26 Moore neighbors (matches HCC).

---

