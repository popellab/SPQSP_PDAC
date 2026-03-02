# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SPQSP PDAC is a GPU-accelerated agent-based model (ABM) with CPU QSP coupling for simulating pancreatic ductal adenocarcinoma (PDAC) tumor microenvironment dynamics. It combines:

- **GPU (FLAME GPU 2)**:
  - Discrete agent-based modeling (cancer cells with functional behavior)
  - Continuous PDE-based chemical diffusion (O2, IFN, IL2, IL10, TGFB, CCL2, ArgI, NO, IL12, VEGFA)
  - Implicit conjugate gradient solver for unconditionally stable diffusion-decay

- **CPU (SUNDIALS CVODE)**:
  - Systemic QSP model for LymphCentral compartment (59+ species ODE system)
  - Drug pharmacokinetics (NIVO, CABO) and pharmacodynamics
  - Immune cell dynamics (recruitment, exhaustion, activation)

- **Integration**:
  - XML-based parameter system for runtime configuration
  - PDE-ABM coupling: chemicals flow from agents to PDE and back
  - QSP-ABM coupling infrastructure (partial implementation)
  - Voxel-based spatial discretization (configurable 10-100 µm resolution)

## Current Implementation Status (Mar 2026)

### ✅ **Fully Functional Components**

**Core Infrastructure**
- ✅ PDE Solver: Implicit CG with operator splitting, 36 substeps per ABM step, exact decay
- ✅ QSP Model: LymphCentral ODE (153 species), CVODE integration, synchronized timesteps
- ✅ Parameter System: XML-based, 124+ parameters, type-safe enums, CLI overrides
- ✅ Build & Output: CMake+CUDA, agent/PDE CSV output, scaling infrastructure
- ✅ Spatial Grid: 320³ voxel support, occupancy grid, ECM Gaussian density field

**Cancer Cell Agent (CancerCell)**
- ✅ States: Stem, Progenitor, Senescent
- ✅ Movement: Random walk with ECM-restricted speed (Gaussian density field)
- ✅ Division: Cooldown + max count → senescence
- ✅ PDL1: IFN-γ upregulation via Hill equation
- ✅ Chemicals: CCL2, TGFB (stem), VEGFA secretion; O2, IFN-γ consumption
- ✅ Neighbors: Cancer, T cell, TReg, MDSC counts via Moore neighborhood
- ✅ Killing: Killed by T cells and macrophages (M1 state)

**T Cell Agent (TCell)**
- ✅ States: Effector, Cytotoxic, Suppressed
- ✅ Movement: Random walk + IL-2 chemotaxis (run-tumble model)
- ✅ Killing: Stochastically kill cancer & fibroblasts, check PDL1 blocking
- ✅ Division: Variable cooldown (0-100% of base), max count limit
- ✅ Recruitment: From QSP-driven vascular mark sources (IFN-γ dependent)
- ✅ Chemicals: IL-2, IFN-γ secretion
- ✅ Death: Apoptosis mechanism

**Macrophage Agent (Macrophage)**
- ✅ States: M1 (pro-inflammatory), M2 (anti-inflammatory)
- ✅ Movement: CCL2-guided chemotaxis (run-tumble)
- ✅ Recruitment: From QSP-driven sources (M1 via IFN-γ, M2 via IL-10)
- ✅ Killing: M1 macrophages kill cancer cells
- ✅ Chemicals: IFN-γ (M1), IL-10 (M2), TNF-α secretion
- ✅ Neighbor scanning: Cancer, T cell, TReg counts
- ✅ Death: Apoptosis mechanism

**Fibroblast Agent (Fibroblast)**
- ✅ States: Normal (FIB_NORMAL), CAF (FIB_CAF)
- ✅ Chain Movement: HEAD → MIDDLE → TAIL leader-follower system
- ✅ Chemotaxis: TAIL cell follows TGF-β gradient (run-tumble)
- ✅ ECM Deposition: Gaussian density field (per-voxel Gaussian smoothing)
- ✅ Chemicals: TGF-β secretion (CAF only)
- ✅ Neighbor scanning: Cancer, T cell, MDSC counts
- ✅ Killing: Killed by T cells when in ECM
- ⚠️ Division: DISABLED (fib_execute_divide corrupts FLAMEGPU device state)

**Vascular Cell Agent (Vascular)**
- ✅ Types: PHALANX (stalk, O2 secretion), TIP (sprouting, migration)
- ✅ Phenotype: HEURISTIC, SPROUTING, QUIESCENT
- ✅ Movement: VEGFA chemotaxis (TIP cells only)
- ✅ Sprouting: TIP→PHALANX division + new TIP generation
- ✅ O2 Sourcing: Phalanx cells secrete O2 (implicit decay formula)
- ✅ Recruitment: IFN-γ dependent T cell mark sources
- ✅ Chemicals: O2 secretion; reads VEGFA gradient
- ✅ Neighbor scanning: T cell, cancer density checks

**TReg Agent (TReg)**
- ✅ States: Regulatory (TCD4_TREG), Helper T (TCD4_TH)
- ✅ Movement: Random walk + TGF-β chemotaxis (run-tumble)
- ✅ Recruitment: From QSP-driven sources
- ⚠️ Suppression: Infrastructure present but CTLA4-IPI block disabled (awaiting QSP data)
- ⚠️ Division: Partially implemented

**MDSC Agent (MDSC)**
- ✅ States: Single undifferentiated state
- ✅ Movement: Random walk + CCL2 chemotaxis (run-tumble)
- ✅ Recruitment: From CCL2 marking layer
- ⚠️ Suppression: Defined but not active (NO/ArgI production without effect)
- ⚠️ Division: Partially implemented

**PDE Solver (Updated Feb 2026)**
- ✅ 10 chemicals: O2, IFN-γ, IL-2, IL-10, TGF-β, CCL2, ArgI, NO, IL-12, VEGF-A
- ✅ Operator splitting: LOD diffusion (7-point stencil) + exact ODE decay
- ✅ 36 substeps per ABM step (dt=600s ABM → dt=16.67s per substep)
- ✅ Unconditionally stable (implicit Backward Euler)
- ✅ Agent atomic writes: Direct PDE atomicAdd from agent functions
- ✅ Boundary conditions: Neumann (no-flux)
- ✅ Performance: ~36 ms per ABM step on modern GPU

**PDE-ABM Coupling (Rewritten Feb 2026)**
- ✅ Agent read: Pointer-based direct read from `pde_concentration_ptr_N`
- ✅ Agent write: AtomicAdd to `pde_source_ptr_N` and `pde_uptake_ptr_N`
- ✅ Gradient computation: Compute all 10 chemical gradients after PDE solve
- ✅ No local agent vars: All PDE state in device arrays, no per-agent copies
- ✅ Source reset: Host-side buffer zero before agent layer
- ✅ Unit handling: Source divided by voxel_volume; uptake (1/s) direct

**Immune Recruitment (NEW Feb 2027)**
- ✅ Vascular mark_t_sources layer: Phalanx cells mark IFN-γ dependent recruitment spots
- ✅ Recruit_t_cells layer: Host function creates T cells at marked sources
- ✅ Mark_mac_sources layer: CCL2 threshold triggers macrophage recruitment
- ✅ Recruit_macrophages layer: M1/M2 polarization based on local cytokine ratios
- ✅ Recruit_mdsc layer: Alternative to macrophages, CCL2 threshold
- ✅ Scaling: Recruitment rate scales with n_vasculature_total (updated per step)

### 🔄 **Partially Implemented**

**QSP-ABM Bidirectional Coupling**
- ✅ Infrastructure: Data structures, wrappers, bookkeeping
- ⚠️ ABM→QSP: Event bookkeeping (cancer deaths, immune counts) but QSP effect TBD
- ⚠️ QSP→ABM: QSP state exported to CSV but not driving ABM recruitment
- ❌ Species mapping: Index extraction not implemented for drug concentrations
- ❌ Drug effects: NIVO/CABO not affecting PDL1 or killing rates

**Known Limitations**

**O2 Dynamics**
- ✅ Decay rate = 1e-5 [1/s] in XML (slow, ~115 min half-life)
- ✅ Vascular cells produce O2 at PHALANX locations via implicit formula
- ✅ Implicit split formula: Cell can only uptake if C_local < C_blood (fixed Feb 26)
- ⚠️ Vascular network still developing, O2 distribution may be patchy

**TReg Suppression**
- ⚠️ CTLA4-IPI block present in code but commented out
- ⚠️ Awaiting IPI concentration from QSP
- ✅ TReg movement and recruitment working

**MDSC Suppression**
- ⚠️ NO/ArgI secretion code present
- ❌ No effect on T cell killing or division

### ❌ **Not Implemented**

**Additional Cell Types**
- ❌ Dendritic cells (APCs)
- ❌ B cells / plasma cells
- ❌ Other T helper subtypes (TH1, TH2, TH17)

**Key Missing Mechanisms**
- ❌ Checkpoint blockade (PD1-PDL1 blocking via drugs)
- ❌ Antigen presentation & T cell priming
- ❌ Metabolic competition (glucose, lactate, amino acids)
- ❌ Full immunosuppression effects (IL-10 & TGF-β blocking T cells)
- ❌ Drug delivery kinetics / penetration into tissue

**Validation & Testing**
- ❌ Unit tests for PDE solver
- ❌ Sensitivity analysis
- ❌ Calibration to experimental data

## Architecture

### Directory Structure
```
PDAC/
├── sim/               # Main simulation entry points
│   ├── main.cu        # Entry point, builds model, initializes QSP/PDE, timing instrumentation
│   ├── model_definition.cu  # FLAME GPU agent definitions (7 types)
│   ├── model_layers.cu      # ~30 layers: recruitment, movement, PDE, division, QSP
│   ├── initialization.cu    # Population init + QSP presim option (-i flag)
│   ├── CMakeLists.txt       # Build configuration, OCC_GRID_MAX=320
│   └── resource/
│       └── param_all_test.xml  # Parameter configuration (164+ params)
│
├── abm/               # GPU parameter system
│   └── gpu_param.h/.cu # Type-safe parameter enums, XML loading, FLAMEGPU environment setup
│
├── core/              # Core simulation infrastructure
│   ├── common.cuh     # 7 agent types, state enums, constants (OCC_GRID_MAX=320)
│   ├── model_functions.cu/cuh  # QSP-ABM coupling, recruitment host functions
│   └── ParamBase.h/.cpp        # Base parameter class
│
├── agents/            # CUDA device functions for 7 agent types
│   ├── cancer_cell.cuh       # ✅ Stem/Progenitor/Senescent, PDL1, ECM interaction
│   ├── t_cell.cuh            # ✅ Effector/Cytotoxic/Suppressed, killing, IL-2 chemotaxis
│   ├── macrophage.cuh        # ✅ M1/M2 states, cancer killing, IFN-γ/IL-10 secretion
│   ├── fibroblast.cuh        # ✅ HEAD/TAIL chain, TGF-β chemotaxis, ECM deposition
│   ├── vascular_cell.cuh     # ✅ PHALANX/TIP, VEGFA chemotaxis, O2 secretion
│   ├── t_reg.cuh             # ⚠️ Movement + recruitment (suppression awaiting QSP IPI)
│   └── mdsc.cuh              # ⚠️ Movement + recruitment (suppression inactive)
│
├── pde/               # Chemical transport solver (operator splitting architecture)
│   ├── pde_solver.cu/cuh      # ✅ Implicit CG + exact ODE decay, matrix-free ops
│   └── pde_integration.cu/cuh # ✅ Host functions: buffer reset, gradient compute, solve
│
└── qsp/               # QSP model integration (CPU)
    ├── LymphCentral_wrapper.h/.cpp  # ✅ CVODE wrapper, time stepping, event bookkeeping
    ├── qsp_integration.cu             # ✅ Host layer for QSP stepping + CSV export
    ├── cvode/
    │   ├── CVODEBase.h/.cpp  # ✅ SUNDIALS CVODE interface
    │   └── MolecularModelCVode.h
    └── ode/
        ├── ODE_system.h/.cpp  # ✅ 153 species LymphCentral ODE
        ├── QSPParam.h/.cpp    # ✅ Parameter loader from XML
        └── QSP_enum.h         # ✅ Species/param index enums
```

## Simulation Execution Loop (model_layers.cu)

Each ABM timestep (default 600s = 10 min) executes ~30 layers in strict order:

### **Phase 0: Population & Recruitment Setup**
1. **update_agent_counts** (host) - Count all agent types
2. **reset_abm_event_counters_start** (host) - Clear cancer death counts, immune events
3. **reset_recruitment_sources** (host) - Clear CCL2 source map
4. **update_vasculature_count** (host) - Update vas_scaler = 100/n_vasculature_total
5. **mark_vascular_t_sources** (agent: VASCULAR) - IFN-γ dependent T source spots
6. **mark_mdsc_sources** (host) - CCL2 threshold MDSC sources
7. **recruit_t_cells** (host) - Create T cells at marked vascular spots
8. **recruit_mdscs** (host) - Create MDSCs at CCL2 high spots
9. **mark_mac_sources** (host) - CCL2 threshold macrophage sources
10. **recruit_macrophages** (host) - Create M1/M2 MACs at sources

### **Phase 1: Broadcast & Neighbor Scanning**
11-17. **final_broadcast_XXX** (agent: all types) - Broadcast (x,y,z) positions via messages
18. **final_scan_neighbors** (agent: CANCER/T/TREG/MDSC/MAC) - Count neighbors in 26-voxel Moore

### **Phase 2: PDE & Gradient Setup**
19. **reset_pde_buffers** (host) - Zero source & uptake arrays
20. **state_transitions** (agent: all types) - State machines, division intents, death checks
21. **compute_chemical_sources** (agent: all types) - AtomicAdd to pde_src/upt arrays
22. **solve_pde** (host) - Implicit CG with 36 substeps, exact ODE decay
23. **compute_pde_gradients** (host) - ∇C for all 10 chemicals (used by chemotaxis)

### **Phase 3: Fibroblast ECM**
24. **zero_fib_density_field** (host) - Clear Gaussian density buffer
25. **build_density_field** (agent: FIBROBLAST) - AtomicAdd Gaussian bumps
26. **update_ecm_grid** (host) - Smooth ECM with per-voxel Gaussian (matches HCC)

### **Phase 4: Occupancy Grid & Movement**
27. **zero_occ_grid** (host) - Clear occupancy array
28. **write_to_occ_grid** (agent: all types) - Mark voxel occupancy (CAS-based)
29. **reset_moves_XXX** (agent: all types) - Clear moves_remaining counter
30-35. **move_XXX_0..N** (agent: all types, 6 substeps) - Random walk + chemotaxis
   - Cancer: ECM-restricted (Gaussian density slows movement)
   - T cells/TRegs/MDSC: IL-2/TGF-β/CCL2 chemotaxis (run-tumble)
   - Macrophage: CCL2 chemotaxis
   - Fibroblast: HEAD→MIDDLE→TAIL chain (HEAD leads, TAIL follows TGF-β)
   - Vascular: TIP cells follow VEGFA gradient (PHALANX fixed at initial position)

### **Phase 5: Division (Two-Phase)**
36. **mark_divide_intent_XXX** (agent: types with division) - Set divideFlag
37. **execute_divide_XXX** (agent: types with division) - Create offspring (except FIB)

### **Phase 6: QSP ODE Integration**
38. **solve_qsp** (host) - CVODE integration forward by dt=600s
39. **export_qsp_data** (host) - Write QSP state to outputs/qsp.csv

**Key Architectural Points:**
- **No agent-local PDE state**: Agents read/write directly to GPU arrays via pointers
- **Operator splitting**: LOD diffusion (7-point stencil) + exact decay ODE in each substep
- **Unconditionally stable**: Implicit Backward Euler → single solve per ABM step
- **Chemotaxis**: Run-tumble model uses PDE gradients (∇C direction + random tumble)
- **ECM effect**: Gaussian density field (per-voxel smoothing) slows cancer movement
- **Occupancy**: CAS-based exclusive voxel access for cancer/macrophage; multi-occupancy for T cells
- **QSP sync**: ODE stepped once per ABM step, decoupled but synchronous

## Spatial Structures & Coupling

### **Voxel Grid & Coordinates**
- **Grid type**: Cartesian 3D (x, y, z in [0, grid_size))
- **Voxel size**: 20 µm default (configurable via XML)
- **Max grid size**: 320³ voxels (65 million; ~8GB GPU headroom on 12GB card)
- **Boundary**: Neumann (no-flux) for PDE; agents wrap at boundaries (periodic)
- **Coordinate system**: Integer voxel coords (x, y, z); agents occupy single voxel

### **Occupancy Grid (Exclusive Cells)**
- **Purpose**: Track exclusive cell types (cancer, macrophage, fibroblast HEAD, vascular)
- **Structure**: `d_cancer_occ` uint64_t array, one entry per voxel
- **CAS operations**: Compare-and-swap for exclusive access (0→cellID)
- **Reset**: Zeroed each ABM step before movement phase (zero_occ_grid layer)
- **Multi-occupancy**: T cells, TRegs, MDSCs use atomicAdd (not exclusive)

### **ECM Density Field (Fibroblasts)**
- **Purpose**: Gaussian fibroblast density, restricts cancer cell movement
- **Structure**: `d_ecm_field` float array, one per voxel
- **Update**:
  1. Zero field (zero_fib_density_field)
  2. Fibroblasts atomicAdd Gaussian bumps (build_density_field)
  3. Host Gaussian smoothing per voxel (update_ecm_grid) - matches HCC
- **Effect**: Cancer movement speed reduced to `1 - density_factor * ecm_density`
- **Range**: 0 (no ECM) → 1 (full ECM blockage)

### **PDE Concentration & Gradient Grids**
- **Substrates**: 10 chemicals (O2, IFN-γ, IL-2, IL-10, TGF-β, CCL2, ArgI, NO, IL-12, VEGF-A)
- **Storage**: `pde_conc_N` (float*) in FLAMEGPU environment for each chemical
- **Sources**: `pde_src_N`, `pde_upt_N` (float*) aggregated via atomicAdd per layer
- **Gradients**: `pde_gradx/y/z_N` computed after PDE solve for chemotaxis
- **Resolution**: One value per voxel

### **Immune Cell Recruitment Sources**
- **T cell sources**:
  - Host marking: Vascular PHALANX cells with IFN-γ > threshold
  - Formula: `p_entry = H_IFNg(local_IFNg) * tumor_scaler * vas_scaler`
  - Scaling: vas_scaler = 100 / n_vasculature_total (accounts for team size)
- **Macrophage sources**:
  - Host marking: CCL2 > threshold on host-side map
  - Polarization: M1 if `IFNg/IL10 > ratio_threshold`, else M2
- **MDSC sources**:
  - Similar to MAC: CCL2 threshold, no polarization
  - Alternative immune cell type

### **PDE-ABM Coupling (Rewritten Feb 2026)**

**Agent Read (compute_chemical_sources layer):**
```cpp
// Agent reads local concentration via pointer (uint64_t environment property)
float* pde_conc_ptr = reinterpret_cast<float*>(
  FLAMEGPU->environment.getProperty<uint64_t>("pde_concentration_ptr_N"));
float local_conc = pde_conc_ptr[voxel_idx];
```

**Agent Write (same layer):**
```cpp
// Agent atomically accumulates source/uptake
float* pde_src_ptr = reinterpret_cast<float*>(
  FLAMEGPU->environment.getProperty<uint64_t>("pde_source_ptr_N"));
atomicAdd(&pde_src_ptr[voxel_idx], release_rate / voxel_volume);  // Source

float* pde_upt_ptr = reinterpret_cast<float*>(
  FLAMEGPU->environment.getProperty<uint64_t>("pde_uptake_ptr_N"));
atomicAdd(&pde_upt_ptr[voxel_idx], uptake_rate);  // Uptake [1/s]
```

**Unit Convention:**
- **Source (secretion)**: `release_amount/(cell·s) → divide by voxel_volume → [concentration/s]`
- **Uptake (consumption)**: `uptake_rate [1/s] → direct (no volume factor)`
- **Decay**: Handled by exact ODE solver (exponential decay per substep)

### **PDE Solver Architecture (Operator Splitting)**

**Per ABM step (dt=600s):**
1. Reset buffers (source, uptake arrays)
2. All agents accumulate source/uptake rates
3. For substep i = 1 to 36 (dt_sub = 16.67s):
   - **LOD diffusion**: `dC/dt = D∇²C` via conjugate gradient
   - **Exact decay**: `C(t) = C(t-dt)·exp(-λ·dt_sub) + integral(S)`
4. Compute gradients for chemotaxis next step

**Key innovations:**
- **No substep threshold**: Implicit → unconditionally stable, dt can be arbitrarily large
- **Split architecture**: Decouples stiff decay from diffusion
- **Matrix-free**: Apply operator without storing full matrix (saves memory)
- **Atomic sources**: Direct GPU writes from agents during compute phase

### **QSP-ABM Coupling (Partial)**

**ABM→QSP (Event Bookkeeping):**
- Cancer deaths tracked by type (stem/progenitor/senescent)
- T cell, MAC, TREG, MDSC recruitment counts per step
- Data stored in host-side accumulators, written to event log

**QSP→ABM (Planned):**
- QSP exports drug concentrations (NIVO, CABO) via ODE state
- Recruitment scaling: immune cell input rates from QSP lymphocyte populations
- PDL1 blocking: PD1-PDL1 affects T cell killing probability
- ⚠️ **Currently**: QSP steps forward but doesn't drive ABM feedback

### **Movement Models**

**Random Walk (Cancer, T cell base):**
- Pick random direction: 26-voxel Moore neighborhood
- Check occupancy grid, reject if occupied
- Move if free, else stay

**Chemotaxis (Run-Tumble):**
- Compute chemical gradient ∇C from PDE
- Run phase: Move in gradient direction (prob ~70%)
- Tumble phase: Random direction (prob ~30%)
- Used by: T cells (IL-2), TRegs (TGF-β), MDSC (CCL2), MAC (CCL2), VASCULAR TIP (VEGFA)

**Fibroblast Chain (Leader-Follower):**
- HEAD: Leads movement (future division)
- MIDDLE: Follows leader via MacroProperty array
- TAIL: Senses gradient, chemotaxis guides HEAD
- No crossover: Each slot has unique position

## Agent Types Reference

### Cancer Cell (agents/cancer_cell.cuh)
**States:** STEM (0) | PROGENITOR (1) | SENESCENT (2) | PDL1_POS (3) | PDL1_NEG (4)

**Behavior:**
- **Movement**: Random walk, restricted by ECM density (moves_remaining counter)
- **Division**: Progenitor cells divide every ~200 steps; stem every ~400; max ~10 divisions → senescent
- **PDL1**: Upregulated by IFN-γ via Hill equation; blocks T cell killing
- **Chemicals**: Produces CCL2 (all), TGF-β (stem only), VEGFA (all); consumes O2, IFN-γ
- **Killing**: Killed by T cells (if PDL1⁻), macrophages (if M1)
- **Neighbors**: Counts cancer, T cells, TRegs, MDSCs in 26-voxel Moore neighborhood

**Key Parameters:**
- Division interval: stem ~400 steps, progenitor ~200 steps
- Max divisions: ~10 (then forced senescence)
- PDL1 EC50: 0.01 mM IFN-γ
- CCL2 release: 2.56e-13 per cell per step

### T Cell (agents/t_cell.cuh)
**States:** EFFECTOR (0) | CYTOTOXIC (1) | SUPPRESSED (2)

**Behavior:**
- **Recruitment**: From QSP-marked vascular T sources (IFN-γ dependent)
- **Movement**: Random walk + IL-2 chemotaxis (run-tumble, 70% run / 30% tumble)
- **Killing**: Stochastic kill cancer (~10% per neighbor); blocked by PDL1; killed by TGFB/IL-10
- **Division**: Effector cells divide every ~80-120 steps; max ~10 divisions then death
- **Chemotaxis**: IL-2 gradient following; uses PDE gradients
- **Chemicals**: Produces IL-2 (autocrine), IFN-γ (signals M1 MAC polarization)
- **State transitions**: Suppressed if high local TGFB or IL-10
- **Death**: Apoptosis after max divisions or suppression signal

**Key Parameters:**
- Killing probability: ~10% per cancer neighbor per step (if PDL1⁻)
- IL-2 chemotaxis strength: medium (slower than MDSC CCL2)
- Division cooldown: 80-120 steps base, randomized 0-100%
- Max divisions: ~10

### Macrophage (agents/macrophage.cuh)
**States:** M1 (0, pro-inflammatory) | M2 (1, anti-inflammatory) | INTERMEDIATE (2)

**Behavior:**
- **Recruitment**: From CCL2-marked sources; polarization based on IFN-γ/IL-10 ratio
- **Movement**: CCL2 chemotaxis (run-tumble, similar to T cells)
- **Killing**: M1 macrophages kill cancer cells (no PDL1 blocking)
- **Polarization**: M1 if IFN-γ/IL-10 > threshold, else M2; can flip over time
- **Chemicals**: M1 produces IFN-γ (pro-inflammatory); M2 produces IL-10 (anti-inflammatory)
- **Neighbors**: Counts cancer cells, T cells, TRegs
- **Death**: Apoptosis mechanism

**Key Parameters:**
- Recruitment threshold: CCL2 > 0.001 mM
- M1 activation: IFN-γ/IL-10 ratio > 10
- Killing probability: M1 ~5-10% per cancer neighbor
- Chemotaxis speed: High (sensitive to CCL2)

### Fibroblast (agents/fibroblast.cuh)
**States:** NORMAL (0) | CAF (1, cancer-associated)

**Behavior:**
- **Chain Movement**: HEAD (leader) → MIDDLE → TAIL (follower)
  - TAIL cell senses TGF-β gradient and directs HEAD movement
  - Head/Middle/Tail form exclusivity chain via MacroProperty array slots
- **Chemotaxis**: TAIL cell follows TGF-β gradient (run-tumble)
- **ECM Deposition**: Fibroblasts build Gaussian density field (restricts cancer movement)
- **State Transition**: Normal→CAF when exposed to high TGFB
- **Chemicals**: CAF produces TGF-β (immunosuppressive, ECM-building signal)
- **Neighbors**: Counts cancer, T cells, MDSC
- **Killing**: Can be killed by T cells (especially when in CAF state)
- **Division**: DISABLED (corrupts FLAMEGPU state)

**Key Parameters:**
- Density field range: 0 (no effect) → 1 (full blockage)
- TGF-β secretion: CAF-specific, activates fibroblasts & suppresses T cells
- Gaussian sigma: ~2 voxels (smooth local field)
- Chain length: 2-4 cells typically

### Vascular Cell (agents/vascular_cell.cuh)
**Types:** PHALANX (stalk) | TIP (sprouting, 1 per connected component)

**Phenotypes:** QUIESCENT | HEURISTIC | SPROUTING

**Behavior:**
- **PHALANX (Stalk) Cells:**
  - Fixed initial position (non-mobile in current implementation)
  - Produce O2 via implicit source formula: `S = K_vl * max(0, C_blood - C_local) / V`
  - Mark T cell recruitment sources (IFN-γ dependent)
  - Can sense hypoxia and anoxia for sprouting trigger
- **TIP (Sprouting) Cells:**
  - Actively migrate toward VEGFA gradient (run-tumble)
  - Divide into PHALANX + new TIP (proliferative sprouting)
  - One TIP per vessel component (tracked via unique tip_id)
- **Sprouting Mechanics:**
  - Phenotype transitions: HEURISTIC (sensing) → SPROUTING (migrating) → QUIESCENT (mature)
  - Trigger: High VEGFA or hypoxia markers
  - New vessel formation: TIP divides into PHALANX (stalk) + TIP (new tip)
- **Chemicals**: O2 source (phalanx only); reads VEGFA gradient (TIP)

**Key Parameters:**
- O2 source formula: Krogh cylinder model (implicit for stability)
- C_blood: ~0.51 mM (initial O2, from HCC)
- VEGFA chemotaxis: Strong (TIP cells actively seek)
- Sprouting rate: Low (network still developing)
- Vessel stability: Mature vessels less likely to regress

### TReg (T regulatory, agents/t_reg.cuh)
**States:** REGULATORY (0) | HELPER_T (1, Th)

**Behavior:**
- **Recruitment**: From QSP-marked vascular sources (IL-10/TGFB dependent)
- **Movement**: Random walk + TGF-β chemotaxis (run-tumble)
- **Suppression**: CTLA4-IPI block present but awaiting IPI from QSP
  - When active: Suppresses T cell division and killing via IL-10 paracrine
- **Division**: Partially implemented (needs QSP input)
- **Chemicals**: Produces IL-10 (suppressive)
- **Neighbors**: Counts cancer, T cells, MDSC
- **Death**: Apoptosis mechanism

**Key Parameters:**
- Recruitment: TGF-β dependent (higher in fibroblast-rich regions)
- Suppression strength: IL-10 paracrine (not yet wired)
- TGF-β chemotaxis: Moderate

### MDSC (Myeloid-derived suppressor cells, agents/mdsc.cuh)
**States:** Single undifferentiated state

**Behavior:**
- **Recruitment**: From CCL2-marked sources (alternative to macrophages)
- **Movement**: Random walk + CCL2 chemotaxis (run-tumble)
- **Suppression**: Code present (NO/ArgI secretion) but effects not implemented
  - When active: Should reduce T cell killing and division
- **Neighbors**: Counts cancer, T cells (for targeting context)
- **Death**: Apoptosis mechanism

**Key Parameters:**
- Recruitment: CCL2 > 0.001 mM
- Suppression: NO/ArgI (code present, effect not wired)
- CCL2 chemotaxis: Similar to macrophages

## Build & Run

### Requirements
- CUDA Toolkit 11.0+ (tested with CUDA 12.x)
- CMake 3.18+
- C++17 compiler (g++ 7+)
- SUNDIALS 4.0.1 (for QSP CVODE solver)
- Boost 1.70+ (for serialization)
- FLAME GPU 2 v2.0.0-rc.4 (auto-fetched by CMake)

**Installation Paths (customize in CMakeLists.txt):**
- SUNDIALS: `$HOME/lib/sundials-4.0.1`
- Boost: `$HOME/lib/boost_1_70_0`

### Build Commands
```bash
cd PDAC/sim
./build.sh                    # Release build (~8 min first time)
./build.sh --debug            # Debug build with symbols
./build.sh --clean            # Clean and rebuild
```

**Build Options in CMakeLists.txt:**
- `CMAKE_CUDA_ARCHITECTURES`: Default 75;80;86 (Turing/Ampere/Ada)
- `FLAMEGPU_VERSION`: v2.0.0-rc.4

### Run Commands
```bash
./build/bin/pdac                    # Defaults: 50³ grid, 500 steps
./build/bin/pdac -s 200 -g 50       # 200 steps, 50³ grid
./build/bin/pdac -s 10 -g 11        # Quick test: 10 steps, 11³ grid
./build/bin/pdac -oa 1 -op 1        # Enable agent and PDE output
```

**Command-line Options:**
```
-g, --grid-size N       Grid dimensions [8-320] (overrides XML, default: 50)
-s, --steps N           ABM steps to run (default: 500)
-r, --radius N          Initial tumor radius in voxels (default: 5)
-t, --tcells N          Initial T cell count (default: 50)
--tregs N               Initial TReg count (default: 10)
--mdscs N               Initial MDSC count (default: 5)
--macs N                Initial macrophage count (default: 5)
--fibs N                Initial fibroblast count (default: 50)
--vascular N            Initial vascular cell count (default: 50, mostly PHALANX)
-p, --param-file PATH   XML parameter file (default: resource/param_all_test.xml)
-oa, --output-agents    0=no agent output, 1=output agents (default: 1)
-op, --output-pde       0=no PDE output, 1=output PDE (default: 1)
-i, --qsp-init FLAG     0=no QSP presim, 1=run QSP to steady state (default: 0)
```

**Output Files:**
- `outputs/abm/agents_step_NNNNNN.csv`: Agent positions, states, properties, per-step
- `outputs/pde/pde_step_NNNNNN.csv`: Spatial chemical concentrations, per-step
- `outputs/qsp.csv`: QSP ODE state (153 species), per ABM step
- `outputs/timing.csv`: Per-step wall-time breakdown (total, pde, qsp, abm)
- `outputs/init_timing.csv`: Initialization phase timing (build_model, init_pde, init_qsp, etc.)

### Typical Run Times
- **Small test** (11³, 10 steps): ~10-15 sec
- **Medium** (50³, 200 steps): ~3-5 minutes
- **Large** (101³, 500 steps): ~20-30 minutes
- **Scaling study** (8 grids, 200 steps each): ~2-3 hours wall time

**First run after build**: Takes 5-10 minutes for CUDA JIT compilation (normal, not a hang!)

## Key Parameter Values (param_all_test.xml)

### PDE Chemicals
| Chemical | Diffusivity (cm²/s) | Decay Rate (1/s) | Chemotaxis | Notes |
|----------|---------------------|------------------|-----------|-------|
| O2 | 2.8e-5 | 1e-5 | No | Vascular PHALANX source only |
| IFN-γ | 1e-7 | 6.5e-5 | T cells | M1 MAC, M2 to lesser extent |
| IL-2 | 4e-8 | 2.78e-5 | T cells | T cell autocrine; signals proliferation |
| IL-10 | 1.4e-8 | 4.6e-5 | No | M2 MAC, suppressive cytokine |
| TGF-β | 2.6e-7 | 1.65e-4 | TReg, FIB | Fibroblast CAF activation |
| CCL2 | 1.31e-8 | 1.67e-5 | MAC, MDSC | Immune recruitment driver |
| ArgI | 1e-6 | 2e-6 | No | MDSC (effect not implemented) |
| NO | 3.8e-5 | 1.56e-3 | No | MDSC (effect not implemented); fastest decay |
| IL-12 | 2.4e-8 | 6.4e-5 | No | Pro-inflammatory signal |
| VEGF-A | 2.9e-7 | 1.921e-4 | Vascular TIP | Angiogenesis driver |

### Cancer Cell Parameters (per cell, per ABM step)
- **O2 uptake**: 1e-3 [mM·s⁻¹ or equiv]
- **IFN-γ uptake**: 1e-3
- **CCL2 release**: 2.56e-13
- **TGF-β release**: 1.06e-10 (stem only); 0 (progenitor)
- **VEGF-A release**: 1.27e-12 (stem + progenitor)
- **Division interval**: ~200 ABM steps (progenitor)
- **Max divisions**: ~10 (before senescence)
- **PDL1 EC50 (IFN-γ)**: 0.01 [Hill equation]

### Immune Cell Parameters
- **T cell**: IL-2 chemotaxis, division interval ~80-120 steps, killing prob ~10%/neighbor
- **TReg**: TGF-β chemotaxis, recruitment from vascular IFN-γ mark
- **MDSC**: CCL2 chemotaxis, suppression (code present, effect not active)
- **MAC M1**: IFN-γ activated, kills cancer cells
- **MAC M2**: IL-10 activated, anti-inflammatory
- **Vascular TIP**: VEGFA chemotaxis, sprouts new PHALANX
- **Vascular PHALANX**: Fixed position (initially), O2 source

### Recruitment Thresholds
- **T cell sources**: IFN-γ > 0.015 mM (Hill function with ec50=0.0725)
- **MAC sources**: CCL2 > 0.001 mM (humanized from HCC)
- **MDSC sources**: CCL2 > 0.001 mM (alternative to MAC)
- **M1 polarization**: IFN-γ/IL-10 ratio > 10 (tunable)

### Spatial Parameters
- **Voxel size**: 20 µm (configurable via XML)
- **Grid size**: 50³ voxels (default) = (50 voxels × 20 µm/voxel)³ = 1 mm³
- **Max grid size**: 320³ voxels (~12.8 mm³, uses ~8GB on 12GB GPU)
- **ABM timestep**: 600 s = 10 min (one QSP step)
- **PDE substep**: 16.67 s (36 substeps per ABM step)
- **Moore neighborhood**: 26 voxels (3×3×3 - center)

## Important Code Locations

### Critical Bug Fixes
**CRITICAL (Feb 11, 2026): Double-Offset Bug Fix**
- **File**: `PDAC/pde/pde_solver.cu` line ~190
- **Issue**: Kernel was adding substrate offset twice (worked only for substrate 0)
- **Fix**: Use `source_idx = voxel_idx` (pointer already offset by `get_device_source_ptr`)
- **Impact**: CCL2, TGFB, VEGFA, IFN-γ now work correctly!

### Key Functions to Understand

**Cancer Cell Behavior (agents/cancer_cell.cuh):**
- `cancer_update_chemicals()`: Read PDE, compute PDL1, detect hypoxia
- `cancer_compute_chemical_sources()`: Set CCL2, TGFB, VEGFA, O2 uptake rates
- `cancer_state_step()`: Division countdown, senescence
- `cancer_execute_divide()`: Create daughter cell, update state

**PDE Integration (pde/pde_integration.cu):**
- `update_agent_chemicals()`: Read PDE → agents (all chemicals, all agent types)
- `collect_agent_sources()`: Agents → PDE (reset, then aggregate all sources)
- `solve_pde_step()`: Call `PDESolver::solve_timestep()`
- `initialize_pde_solver()`: Setup grid, diffusion coeffs, decay rates

**PDE Solver (pde/pde_solver.cu):**
- `solve_timestep()`: Main solve loop (for each substrate, build RHS, run CG)
- `solve_implicit_cg()`: Conjugate gradient with matrix-free operator
- `apply_diffusion_operator()`: Compute A·x = (I + dt·λ - dt·D·∇²)·x
- `add_sources_from_agents()`: Kernel to write agent sources (CRITICAL: fixed Feb 11)

**QSP Integration (qsp/LymphCentral_wrapper.cpp):**
- `initialize()`: Setup ODE system, CVODE solver
- `time_step()`: Advance ODE by dt
- `update_from_abm()`: ⚠️ NOT IMPLEMENTED - should receive ABM data
- `get_state_for_abm()`: ⚠️ NOT IMPLEMENTED - should send QSP data

## Known Issues & Limitations (Mar 2026)

### Current Bugs / Limitations

**T Cell Recruitment Delay (IFN-γ Deadlock):**
- PDAC T cell recruitment lags HCC by ~6 ABM steps
- Root cause: `vascular_mark_t_sources` formula has low initial p_entry (IFN-γ starts at 0)
- Impact: Fewer T cells → more cancer growth → suppression feedback
- Status: Under investigation; likely parameter tuning (PARAM_TEFF_RECRUIT_K magnitude)
- Downstream: PDAC T cytotoxic 2.1x lower than HCC; T suppressed 2.3x higher

**Fibroblast Division Disabled:**
- `fib_execute_divide` in `agents/fibroblast.cuh` commented out
- Cause: `FLAMEGPU newAgent()` called from device layer corrupts device state
- Workaround: Host-side division queue + bulk create (future)
- Impact: Fibroblast count grows only by recruitment, not division

**TReg Suppression Not Active:**
- CTLA4-IPI block in `agents/t_reg.cuh` commented out (stack overflow fix Feb 25)
- Awaiting IPI concentration data from QSP (not yet connected)
- Impact: TRegs present but don't suppress T cell division/killing
- Status: Code ready, waiting on QSP-ABM feedback connection

**MDSC Suppression Not Active:**
- NO/ArgI secretion code present but not wired to T cell effects
- Macrophage M1/M2 polarization works, but MDSC alternative pathway incomplete
- Impact: MDSC recruitment counts correct, but effects not implemented

**Vascular Network Maturation:**
- TIP sprouting rate slower than HCC (network still developing)
- PHALANX cells have high O2 production but limited diffusion due to slow network spread
- Impact: Tumor center may still show some O2 gradients despite vascular presence
- Status: Sprouting mechanics working; tuning sprouting triggers needed

**QSP→ABM Feedback Not Connected:**
- QSP steps forward each ABM step (CVODE integration working)
- Drug concentrations available in ODE state but not extracted/passed to ABM
- Species index mapping for NIVO/CABO not implemented
- Impact: Drugs don't affect T cell PDL1-blocking or killing rates
- Status: Infrastructure ready; needs species mapping + host-side data transfer

### Build & Compile Issues

**CUDA 12.6 + FLAME GPU Incompatibility:**
- May fail with CUB template errors during compilation
- Workaround: Use CUDA 11.8, 12.0-12.5, or newer FLAME GPU version
- Current build: CUDA 12.x on WSL2 (tested up to 12.5)

**WSL2 Performance:**
- GPU performance reduced ~10-20% vs native Linux
- Network drive access (if /home on network) slows I/O
- Timing measurements include WSL overhead

### Known Limitations

**Hypoxia/O2 Dynamics:**
- O2 decay = 1e-5 (slow, ~115 min half-life)
- Vascular network still sparse; O2 distribution patchy
- Tissue center may still show some O2 gradients (expected given current vasculature density)

**Chemotaxis Run-Tumble Model:**
- Simplified model (70% run, 30% tumble) doesn't account for gradient magnitude
- Actual cells have gradient-adaptive response
- Current model sufficient for medium-range attraction

**Fibroblast ECM:**
- Gaussian density field slows cancer movement (working)
- Does not model ECM mechanics (stiffness, porosity, remodeling)
- Effect is kinetic (speed reduction) not mechanical (obstacle avoidance)

## Development Workflow

### Adding a New Agent Type
1. Define agent properties in `core/common.cuh`
2. Add agent definition in `model_definition.cu::defineXXXAgent()`
3. Create agent functions in `agents/xxx.cuh`
4. Add layers in `model_layers.cu::defineMainModelLayers()`
5. Initialize population in `initialization.cu::initializeAllAgents()`
6. Add parameters to XML and `gpu_param.h`

### Adding a New Chemical
1. Add enum to `pde/pde_solver.cuh::ChemicalType`
2. Add diffusivity/decay to `pde/pde_integration.cu::initialize_pde_solver()`
3. Add XML parameters to `param_all_test.xml::Molecular::biofvm`
4. Add collection in `pde_integration.cu::collect_agent_sources()`
5. Add reading in `pde_integration.cu::update_agent_chemicals()`

### Modifying Parameters
- Edit `resource/param_all_test.xml`
- Rebuild only if adding NEW parameters (need to update `gpu_param.h`)
- Otherwise, just re-run (XML loaded at runtime)

## Next Development Priorities

### High Priority (Core Functionality - Mar 2026)
1. **Debug T cell recruitment IFN-γ deadlock** - PDAC lags HCC by ~6 steps
   - Trace: vascular_mark_t_sources → recruit_t_cells formula sensitivity
   - Check: PARAM_TEFF_RECRUIT_K magnitude vs actual IFN-γ concentration
   - Compare: HCC BG recruitment vs PDAC to identify bugs
2. **Connect QSP→ABM feedback** - Pass drug concentrations, immune state to ABM
   - Implement species index mapping for NIVO, CABO extraction
   - Update T cell PDL1-blocking effects
   - Scale recruitment rates from QSP lymphocyte populations
3. **Activate TReg suppression** - Awaiting IPI concentration from QSP
   - Uncomment CTLA4-IPI block in t_reg.cuh
   - Implement IL-10 & TGF-β immune cell blocking
4. **Activate MDSC suppression** - NO/ArgI effects on T cell killing
   - Wire NO/ArgI secretion to T cell kill probability reduction

### Medium Priority (Biological Realism - Apr/May 2026)
5. **Vasculature maturation** - Current network still developing (sprouts slow)
   - Debug TIP sprouting rate vs HCC
   - Implement vessel stabilization (PHALANX→wall) after time
   - Add regression mechanic for isolated vessels
6. **Add Th1/Th2 polarization** - TCD4_TH cell subtypes
   - IFN-γ→Th1, IL-10→Th2
   - Different cytokine production rates per subtype
7. **Fibroblast division** - Re-enable fib_execute_divide
   - Investigate FLAMEGPU newAgent() device state corruption
   - Alternative: Host-side division queue + bulk create
8. **Checkpoint blockade drug effects** - PD1-PDL1 fully blocking
   - Model anti-PD-L1 (NIVO) concentration effect
   - Model CTLA-4 (CABO) concentration effect

### Low Priority (Validation & Optimization - Jun 2026+)
9. Parameter sensitivity analysis (vary recruitment k, decay rates, diffusion)
10. Unit tests for PDE solver accuracy vs analytical solutions
11. Performance profiling: identify GPU bottlenecks (solve_pde vs chemotaxis vs other)
12. Calibration to PDAC experimental data (IHC, flow cytometry, imaging)
13. Documentation & tutorials for extending simulator

## Debugging Tips

```bash
# Quick iteration with small grid
./build/bin/pdac -g 11 -s 5 -oa 1 -op 1

# Enable CUDA error checking (slower but catches issues)
export CUDA_LAUNCH_BLOCKING=1

# Check parameter loading
./build/bin/pdac 2>&1 | grep -i "parameters loaded"

# Monitor chemical evolution
tail -f outputs/pde/pde_step_*.csv

# Check agent counts over time
for f in outputs/abm/agents_step_*.csv; do
    echo -n "$f: "; tail -n +2 "$f" | wc -l
done

# Visualize PDE in Python
cd python
jupyter notebook SimpleABMPDEvis.ipynb
```

**Common Issues:**
- **All zeros in PDE**: Check source collection is enabled, verify agents exist
- **Simulation hangs**: First run takes 5-10 min (CUDA init), subsequent runs fast
- **Out of memory**: Reduce grid size or number of agents
- **Slow simulation**: Use smaller grid (`-g 21`) or fewer steps (`-s 10`)

## References

- **FLAME GPU 2**: https://github.com/FLAMEGPU/FLAMEGPU2
- **SUNDIALS**: https://computing.llnl.gov/projects/sundials
- **BioFVM**: http://biofvm.org (inspiration for PDE solver)
- **PhysiCell**: http://physicell.org (agent-based model reference)
