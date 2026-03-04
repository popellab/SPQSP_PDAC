# PORTING NOTES: CPU to GPU/CPU Hybrid Implementation

This document outlines the differences between the CPU-based SPQSP_HCC implementation and the GPU-accelerated SPQSP_PDAC implementation, and identifies areas where alignment is still needed.

## Executive Summary

The GPU port replaces:
- **ABM Engine**: Object-oriented C++ classes → FLAMEGPU2 agent framework
- **PDE Solver**: BioFVM CPU library → Custom CUDA kernels
- **Agent Functions**: Virtual C++ methods → CUDA device functions

**Gaps remaining**:
1. CVODE ODE integration not yet connected to GPU ABM
2. Model parameter values need verification/alignment
3. Some behavioral subtleties may differ between implementations
4. Chemical source/sink discretization scheme differs

---

## 1. AGENT REPRESENTATION & LIFECYCLE

### CPU Implementation (SPQSP_HCC)
```cpp
// Inheritance hierarchy
BaseAgent (virtual methods, type/state)
  └── CellAgent (coordinates, virtual agent_*_step methods)
      ├── CancerCellCPU
      ├── TCellCPU
      ├── TRegCPU
      └── MDSCCpu
```

**Agent Storage**:
- Flat vector: `std::vector<CellAgent*> _vecAgent`
- Spatial index: `Grid3D<AgentGridVoxel*>` mapping voxels to agent containers
- Movement: Direct object pointers passed between voxels

**Agent Lifecycle** (virtual methods per timestep):
```cpp
bool agent_movement_step(double t, double dt, Coord& c)   // Select + execute move
bool agent_state_step(double t, double dt, Coord& c)      // Transitions
void molecularStep(double t, double dt)                   // ODE integration
void odeStep(double t, double dt) {}                      // Override in subclasses
```

### GPU Implementation (SPQSP_PDAC)
```
FLAMEGPU2 AgentDescription
├── CancerCell agent type
│   ├── Variables: id, x, y, z, state, PDL1_syn, ...
│   └── Functions: cancer_broadcast_location, cancer_scan_neighbors, ...
├── TCell agent type
│   ├── Variables: id, x, y, z, state, IL2_exposure, ...
│   └── Functions: tcell_scan_neighbors, tcell_state_step, ...
├── TReg agent type
└── MDSC agent type
```

**Agent Storage**:
- FLAMEGPU2-managed agent list (GPU memory)
- Messaging system for spatial communication
- Two-phase movement (intent → execution)

**Agent Lifecycle** (FLAMEGPU2 agent functions):
```cuda
__device__ void cancer_broadcast_location(...)      // Output MSG_CELL_LOCATION
__device__ void cancer_scan_neighbors(...)          // Read MSG_CELL_LOCATION, cache
__device__ void cancer_state_step(...)              // State transitions
__device__ void cancer_compute_chemical_sources(...) // Set rates
__device__ void cancer_select_move_target(...)      // Phase 1 movement
__device__ void cancer_execute_move(...)            // Phase 2 movement
```

### Key Differences

| Aspect | CPU | GPU |
|--------|-----|-----|
| **Storage** | Object pointers in grid | FLAMEGPU2 agent lists (GPU memory) |
| **Movement** | Direct grid update | Two-phase message passing |
| **Neighbor access** | Direct grid query / callback | Spatial messages (radius-based) |
| **Conflict resolution** | Callback-based during agent step | Intent messages + priority ID |
| **State management** | Virtual methods (polymorphic) | Device functions (static dispatch) |
| **Type extension** | Derive from CellAgent | Add new agent description |

**Implication**: GPU forces explicit two-phase movement; CPU can resolve conflicts synchronously. Both should reach same outcome if priority rules match.

---

## 2. PDE SOLVER ARCHITECTURE

### CPU Implementation (BioFVM)

**BioFVMGrid wrapper**:
```cpp
class BioFVMGrid {
    BioFVM::Microenvironment _tme;        // Core solver
    std::vector<BioFVMSinkSource*> _sink_source;  // Point sources/sinks
    double operator()(const Coord3D&, size_t substrate);  // Read concentration
    BioFVMSinkSource* add_point_source(size_t chem, pos, vol, rate, saturation);
    void source_sink_step(double dt);     // Update concentrations
    void timestep(double dt);             // Solve PDE
};
```

**Diffusion-Reaction Equation**:
```
∂c/∂t = D∇²c - λc + S(x,y,z,t)
```

**Discretization**:
- **Method**: Finite volume (BioFVM library)
- **Space**: Voxel-based, center-difference for Laplacian
- **Time**: Explicit (forward Euler) for diffusion, implicit for decay
- **Boundary**: No-flux Neumann (reflection at edges)

**Source/Sink Handling**:
```cpp
// Three discretization schemes (selected at compile-time):
#define SOURCE_SINK_INTERNAL = SOURCE_SINK_NO_SATURATION  // Default

class BioFVMSinkSource {
    void update_source(double vol, double volV, double dt,
                      double secretion, double saturate);
    void update_sink(double vol, double volV, double dt, double uptake);
    double evaluate_source_sink(double c, double dt);
};
```

**For SOURCE_SINK_NO_SATURATION scheme**:
```
dC_voxel/dt = (secretion - uptake * C) dt
C_new = C_old + (secretion - uptake * C_old) * dt
```

### GPU Implementation (Custom CUDA Kernels)

**PDESolver class**:
```cpp
class PDESolver {
    float* d_concentrations_current_;  // Device memory [num_substrates][z][y][x]
    float* d_concentrations_next_;     // Double buffering
    float* d_sources_;                 // Agent source rates

    void read_concentrations_at_voxels(agents);
    void add_sources_from_agents(agents);
    void solve_step();
};
```

**CUDA Kernel Design**:
```cuda
__global__ void diffusion_reaction_kernel(
    const float* C_curr,
    float* C_next,
    const float* sources,
    int nx, int ny, nz,
    float D, float lambda, float dt, float dx) {

    // 3D index from thread blocks
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (in_bounds(x, y, z)) {
        // Compute 3D Laplacian using finite differences
        float laplacian = (C_curr[x±1,y,z] + ... - 6*C_curr[x,y,z]) / (dx*dx);

        // Update: C_next = C_curr + dt*(D*Laplacian - lambda*C + sources)
        C_next[x,y,z] = C_curr[x,y,z] +
                        dt * (D * laplacian - lambda * C_curr[x,y,z] + sources[x,y,z]);
    }
}
```

**Discretization**:
- **Method**: Finite difference, 7-point stencil (±1 neighbors in x,y,z)
- **Space**: Voxel-centered (same as BioFVM)
- **Time**: Explicit Euler (same as BioFVM for diffusion)
- **Boundary**: Neumann via padding (concentrations fixed in ghost cells)

**Source/Sink Handling**:
```cpp
// Sources accumulated in agent-to-PDE coupling:
for (int substrate = 0; substrate < num_substrates; ++substrate) {
    pde_solver.reset_sources(substrate);
    for (each agent) {
        pde_solver.add_source_at_voxel(agent.x, agent.y, agent.z,
                                       substrate, agent.source_rate[substrate]);
    }
}
// Source term entered directly into PDE kernel
```

### Key Differences

| Aspect | CPU (BioFVM) | GPU (CUDA) |
|--------|--------------|-----------|
| **Library** | External BioFVM library | Custom kernels |
| **Discretization** | Finite volume | Finite difference |
| **Stencil** | Laplacian (varies by voxel distance) | 7-point (±1 neighbors) |
| **Time stepping** | Explicit/Implicit (selectable) | Explicit Euler (fixed) |
| **Source saturation** | Three schemes (implicit, analytical, no-sat) | No saturation (direct rate) |
| **Memory** | CPU (vector) | GPU (device arrays) |
| **Granularity** | Flexible geometry | Regular grid only |
| **Boundary condition** | Neumann via mirror | Neumann via ghost cells |
| **Coupling loop** | Agent sources collected, updated per step | Agent sources reset/accumulated before PDE step |

**Numerical Impact**: Both should converge to same solution with appropriate parameters (D, λ, dt, dx). Finite difference is simpler but less flexible than finite volume.

---

## 3. AGENT-PDE COUPLING

### CPU Implementation

**Coupling sequence per ABM step**:

```
1. Agents: Movement phase
   - Each agent selects move target (Von Neumann, conflict resolution)

2. Agents: State update phase
   - Read from PDE grid: local_chem = pde_grid(x, y, z, substrate)
   - Update internal state based on chemical exposure
   - Compute source/sink rates

3. PDE: Diffusion solver
   - Agents' source/sink rates baked into PDE via BioFVMSinkSource
   - PDE solver updates concentrations

4. Agents: Next iteration reads updated concentrations
```

**Chemical access**:
```cpp
// In agent methods:
double O2 = pde_grid(agent_x, agent_y, agent_z, CHEM_O2);
double IFNg = pde_grid(agent_x, agent_y, agent_z, CHEM_IFN);
// Gradients available
auto grad = pde_grid.get_gradient(agent_pos);
```

**Source/sink specification**:
```cpp
// In agent molecularStep or state_step:
double O2_uptake_rate = hypoxic ? 0.5e-5 : 1.0e-5;
double IFNg_release_rate = (state == T_CELL_CYT) ? 1.0e-6 : 0.0;
pde_grid.update_sink(CHEM_O2, agent_pos, volume, O2_uptake_rate);
pde_grid.update_source(CHEM_IFNg, agent_pos, volume, IFNg_release_rate, saturation);
```

### GPU Implementation

**Coupling via host functions** (layer structure):

```
Layer: Read Chemicals from PDE (host function)
  for each agent:
    agent.local_O2 = pde_solver.get_concentration(agent.x, agent.y, agent.z, CHEM_O2)
    agent.local_IFN = pde_solver.get_concentration(agent.x, agent.y, agent.z, CHEM_IFN)
    ...

Layer: Update Chemical States (device function per agent)
  agent.update_chemical_state()  // Accumulate exposures, state transitions

Layer: Compute Chemical Sources (device function per agent)
  agent.compute_chemical_sources()  // Set source rates

Layer: Write Sources to PDE (host function)
  pde_solver.reset_sources()
  for each agent:
    pde_solver.add_source_at_voxel(agent.x, agent.y, agent.z, CHEM_*, agent.rate)

Layer: Solve PDE (host function)
  for i = 0 to molecular_steps-1:
    pde_solver.solve_step()  // Runs CUDA kernel
```

**Chemical access**:
```cuda
// In device function:
__device__ void tcell_update_chemical_state(...) {
    // Agent variables already contain local_O2, local_IFN, etc.
    // (read in "Read Chemicals from PDE" layer)
    IL2_exposure += local_IL2 * dt;
    activation_level = hill_equation(IL2_exposure, EC50, n);
}
```

**Source rate specification**:
```cuda
__device__ void tcell_compute_chemical_sources(...) {
    if (cell_state == T_CELL_CYT) {
        IFNg_release_rate = 1.0e-6;
        IL2_release_rate = 5.0e-7;
    } else {
        IFNg_release_rate = 0.0;
        IL2_release_rate = 0.0;
    }
}
```

### Key Differences

| Aspect | CPU | GPU |
|--------|-----|-----|
| **Coupling timing** | Per agent, during step | Synchronized, layer-based |
| **Chemical read** | On-demand in agent method | Layer pulls all at once |
| **Source updates** | Continuous via BioFVMSinkSource | Accumulated per layer |
| **Gradients** | Computed on-demand | Not currently used |
| **Saturation kinetics** | Available (configurable) | Not implemented |
| **Molecular coupling** | Direct ODE integration | Hill equations only |

**Implication**: GPU approach is more batch-efficient but less flexible. Chemical computations must be deterministic (no time-of-read dependencies).

---

## 4. ODE INTEGRATION (MOLECULAR MODELS)

### CPU Implementation

**CVODE Framework**:
```cpp
class CVODEBase {
    state_type _species_var;              // ODE LHS variables
    state_type _species_other;            // SBML variables not in ODE
    state_type _nonspecies_var;           // Parameters

    void simOdeStep(double tStart, double tStep);  // CVODE integration
    void resolveEvents(double t);                  // Discontinuous transitions

    // Per-cell ODE models inherit from CVODEBase
};

class MolecularModelCVode : public CVODEBase {
    // Template wrapper for specific ODE models
};
```

**ODE Integration Loop** (per agent, per ABM step):
```cpp
agent.molecularStep(t, dt) {
    agent.ode_model.simOdeStep(t, dt);  // CVODE (BDF solver)
}
```

**Event Handling**:
- Root-finding detects when `g(y,t) = 0` (state transition condition)
- Discontinuous jumps update state variables
- Example: PDL1 upregulation when IFNg > threshold

### GPU Implementation

**Current Status**: NOT INTEGRATED

**Placeholder in code**:
```cpp
// SP_QSP_shared/Numerical_Adaptor/CVODE/ classes exist but not used in GPU simulation
// MolecularModelCVode available for future integration
```

**Simplified molecular models** (using Hill equations):
```cuda
__device__ float hill_equation(float x, float k50, float n) {
    if (x <= 0.0f) return 0.0f;
    const float xn = powf(x, n);
    const float kn = powf(k50, n);
    return xn / (kn + xn);
}

// Example: PDL1 synthesis
__device__ void cancer_update_pdl1(float tcell_cyt_count, float IFNg, ...) {
    float syn_response = hill_equation(IFNg, EC50_IFN, n_IFN);
    PDL1_new = PDL1 * (1.0f - decay_rate) + basal + syn_max * syn_response;
}
```

### Key Differences

| Aspect | CPU | GPU |
|--------|-----|-----|
| **Molecular models** | CVODE (stiff ODE solvers) | Hill equations (algebraic) |
| **Integration method** | BDF (variable order/step) | Lookup tables / functions |
| **Event handling** | Root-finding discontinuities | Explicit if-then conditions |
| **Subcellular detail** | Full signaling cascades | Dose-response only |
| **Computational cost** | Higher (ODE solver overhead) | Lower (algebraic) |
| **Accuracy** | High (adaptive order) | Approximate (depends on parameters) |

**Gap**: GPU simplifies molecular dynamics to make GPU computation tractable. Requires parameter fitting to match ODE behavior.

---

## 5. PARAMETER ALIGNMENT

### Critical Parameters to Verify

#### Grid & Timing
- [ ] Grid size: 51 (CPU) vs 51 (GPU) ✓ Aligned
- [ ] Voxel size: 20 μm (CPU) vs 20 μm (GPU) ✓ Aligned
- [ ] ABM timestep: 1200 seconds (CPU) vs 1200 seconds (GPU) ✓ Aligned
- [ ] PDE substeps: 10 (CPU) vs 10 (GPU) ✓ Aligned

#### Cancer Cell Parameters
- [ ] Movement probability (stem vs progenitor)
- [ ] Division interval (stem ~24h, progenitor ~12h)
- [ ] PDL1 synthesis rate and kinetics
- [ ] T cell killing probability formula
- [ ] Senescence transition conditions

#### T Cell Parameters
- [ ] Movement probability (effector vs cytotoxic)
- [ ] Activation threshold (progenitor neighbor required?)
- [ ] Division interval and limit
- [ ] Lifespan distribution
- [ ] IFNg/IL2 release rates and kinetics

#### TReg Parameters
- [ ] Movement probability
- [ ] Division probability and limit
- [ ] IL10/TGFB release rates
- [ ] Suppression strength against T cells
- [ ] Lifespan

#### MDSC Parameters
- [ ] Movement probability
- [ ] Lifespan
- [ ] Suppression strength (IC50_MDSC for 50% T cell suppression)

#### Chemical Parameters
| Chemical | CPU D (cm²/s) | CPU λ (/s) | GPU D | GPU λ | Status |
|----------|---------------|-----------|-------|-------|--------|
| O₂ | ? | ? | ? | ? | **NEEDS VERIFICATION** |
| IFN | ? | ? | ? | ? | **NEEDS VERIFICATION** |
| IL2 | ? | ? | ? | ? | **NEEDS VERIFICATION** |
| IL10 | ? | ? | ? | ? | **NEEDS VERIFICATION** |
| TGFB | ? | ? | ? | ? | **NEEDS VERIFICATION** |
| CCL2 | ? | ? | ? | ? | **NEEDS VERIFICATION** |
| NIVO | ? | ? | ? | ? | **NEEDS VERIFICATION** |
| CABO | ? | ? | ? | ? | **NEEDS VERIFICATION** |

#### Hill Function Parameters
- [ ] PDL1 upregulation by IFNg (EC50, Hill coefficient n)
- [ ] T cell activation by IL2 (EC50, n)
- [ ] T cell exhaustion by IL10/TGFB (EC50s, n)
- [ ] PDL1-PD1 interaction (EC50, n)

**Action**: Extract parameter values from CPU config files and verify GPU code uses identical values.

---

## 6. MODEL BEHAVIORAL DIFFERENCES

### Movement Conflict Resolution

**CPU**: Callback-based conflict resolution during agent step
```cpp
// If multiple agents target same voxel:
// - Priority by ID (lower ID wins)
// - Deterministic outcome
```

**GPU**: Two-phase message passing with intent resolution
```cuda
// Phase 1: select_move_target broadcasts intent
// Phase 2: execute_move checks intent messages, applies priority
// - Same priority rule (lower ID) but via explicit messages
```

**Status**: Should be equivalent if priority rules match. ✓ Need to verify

### State Transition Logic

**CPU**: In `agent_state_step()` methods
```cpp
// Example: T cell exhaustion
if (Treg_neighbors > threshold && IL10_exposure > threshold) {
    state = T_CELL_SUPP;
}
```

**GPU**: In `tcell_state_step()` device function
```cuda
// Must match CPU logic exactly
if (neighbor_Treg_count > threshold && local_IL10 > threshold) {
    cell_state = T_CELL_SUPP;
}
```

**Status**: Logic should match, but timing of state reads may differ. Need code comparison.

### Division Mechanics

**CPU**: Division creates copy (deep copy of all variables)
```cpp
CellAgent* child = agent->createCellCopy();  // Inherits all state
```

**GPU**: Division creates new agent with FLAMEGPU2
```cuda
// In execute_divide function:
// New agent created with same variables as parent
```

**Status**: Should be equivalent. ✓ Need to verify asymmetric division logic.

### Chemical Exposure & Accumulation

**CPU**: Agents accumulate exposure over time
```cpp
agent.IL2_exposure += local_IL2 * dt;  // Cumulative
```

**GPU**: Same pattern in device functions
```cuda
IL2_exposure += local_IL2 * dt;  // Cumulative
```

**Status**: Equivalent. ✓

---

## 7. COUPLING CPU ODE SYSTEM TO GPU ABM/PDE

### Current Gap

GPU code does NOT currently integrate with CVODE ODE systems. The framework exists in SP_QSP_shared but is not used.

### Integration Strategy (Proposed)

#### Option A: CPU-side ODE Thread

Maintain a separate CPU thread running per-agent ODE models:

```cpp
// Main GPU simulation loop
while (simulation_running) {
    // 1. Run GPU ABM step
    simulation.simulate(1 step);

    // 2. Copy agent state to CPU
    get_population_data(agents_cpu);

    // 3. Run CPU ODE models (parallel per agent)
    for (each agent) {
        agent_ode_model.simOdeStep(t, dt);
        // Updates: PDL1, signaling state, internal concentration
    }

    // 4. Copy ODE results back to GPU
    set_population_data(agents_gpu, agents_cpu);

    // 5. Repeat
}
```

**Pros**:
- Minimal GPU code changes
- Leverages existing CVODE infrastructure
- Can use multi-threaded CVODE

**Cons**:
- GPU-CPU data transfer overhead
- Synchronization point every step
- Breaks GPU acceleration benefits

#### Option B: CUDA ODE Kernels

Implement CUDA-compatible ODE solvers (e.g., simple Euler, RK4):

```cuda
__global__ void solve_molecular_odes(
    float* ode_state,      // Per-agent ODE variables
    const float* chemical_exposure,  // From PDE
    int num_agents,
    float dt) {

    // Thread per agent
    int agent_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (agent_id < num_agents) {
        // Simplified RK4 or implicit Euler for this agent's ODE
        solve_agent_ode(agent_id, dt);
    }
}
```

**Pros**:
- Fully GPU-accelerated
- No CPU-GPU data transfer
- Scales with agent count

**Cons**:
- Requires ODE solver rewrite for GPU
- Loss of advanced CVODE features (adaptive order, event detection)
- More complex code

#### Option C: Hybrid - Keep GPU Simple, CPU Handles Molecular Detail

Current GPU approach is simplified:
- Use Hill equations on GPU (fast, approximate)
- When detailed molecular models needed: CPU handles that agent's ODE

```cpp
// GPU: Fast approximation
PDL1_synthesis_rate = syn_max * hill_equation(IFNg, K50, n);

// CPU (on-demand, per agent):
// For agents requiring detailed modeling:
// - Copy agent state to CPU
// - Solve full CVODE ODE system
// - Copy molecular state back
```

**Pros**:
- Leverages GPU for bulk simulation (fast)
- Allows detailed models when needed
- Gradually increases complexity

**Cons**:
- Requires infrastructure to track which agents need CPU
- Adds complexity to CPU-GPU interface
- Potential load imbalance

### Recommended Approach: Option C (Hybrid)

1. **Immediate** (keep GPU simpler):
   - GPU runs ABM with Hill equation molecular models
   - CVODE infrastructure ready but unused

2. **Phase 2** (add CPU-GPU coupling):
   - Agents can optionally use CPU ODE solver
   - Flag in agent state: `use_detailed_ode`
   - Per-step: copy flagged agents CPU → solve ODE → copy back

3. **Future** (full GPU):
   - Implement adaptive ODE solver in CUDA
   - Migrate CPU-side agents to GPU kernels

---

## 8. IMPLEMENTATION CHECKLIST

### Phase 1: Verification & Alignment (Current State)

- [ ] Extract diffusion coefficients (D) for all 8 chemicals from CPU code
- [ ] Extract decay rates (λ) for all 8 chemicals from CPU code
- [ ] Verify all cell-type parameters match between CPU and GPU
- [ ] Verify Hill equation parameters match
- [ ] Compare agent behavior code (CPU vs GPU) for discrepancies
- [ ] Test movement conflict resolution logic equivalence
- [ ] Test state transition logic equivalence

### Phase 2: Model Refinements

- [ ] Implement gradient-based chemotaxis (if used in CPU)
- [ ] Add PDL1 dynamics if missing
- [ ] Verify T cell activation logic matches CPU
- [ ] Verify cancer cell killing probability formula
- [ ] Test division mechanics (especially asymmetric division for cancer)
- [ ] Compare chemical source/sink rates with CPU

### Phase 3: ODE Integration

- [ ] Design CPU-GPU ODE coupling interface
- [ ] Implement Option C (hybrid) infrastructure
- [ ] Create ODE agent wrapper class
- [ ] Test data transfer efficiency
- [ ] Benchmark GPU vs hybrid vs CPU-only

### Phase 4: Validation

- [ ] Run side-by-side CPU vs GPU simulations
- [ ] Compare trajectory of cell populations
- [ ] Compare chemical concentration profiles
- [ ] Compare final state distributions
- [ ] Sensitivity analysis on key parameters

---

## 9. KEY FILES FOR REFERENCE

### CPU Implementation (SPQSP_HCC-main)
- `SP_QSP_shared/ABM_Base/` - Agent base classes
- `SP_QSP_shared/Numerical_Adaptor/BioFVM/` - PDE wrapper
- `SP_QSP_shared/Numerical_Adaptor/CVODE/` - ODE integration
- `PDAC/` - Problem-specific implementation (agent behavior)

### GPU Implementation (SPQSP_PDAC-main)
- `PDAC/agents/` - CUDA agent functions
- `PDAC/pde/pde_solver.cuh` - PDE solver class
- `PDAC/sim/main.cu` - Initialization and parameters
- `PDAC/sim/model_definition.cu` - FLAMEGPU2 model construction

### Documentation
- `CLAUDE.md` - High-level GPU architecture
- This file (`PORTING_NOTES.md`) - Detailed differences

---

## 10. OPEN QUESTIONS

1. **Parameter sources**: Are exact CPU parameter values documented? (diffusion coefficients, decay rates, Hill parameters)

2. **Gradient chemotaxis**: Is gradient-based movement used in CPU? Not implemented in GPU yet.

3. **Asymmetric division**: What is exact probability model for cancer progenitor asymmetric division?

4. **PDL1 kinetics**: Is PDL1 continuous function or discrete state?

5. **Source saturation**: CPU supports three schemes; GPU uses simple linear rate. Is this sufficient?

6. **Boundary conditions**: Are boundaries truly no-flux, or are there other boundary types?

7. **Grid spacing**: Voxel size 20μm - is this biologically motivated or computational choice?

8. **Molecular network**: What is full reaction network for cytokines? Hill equations sufficient approximation?

---

## Summary Table: Implementation Gaps

| Component | Status | Priority | Notes |
|-----------|--------|----------|-------|
| **Agent types** | ✓ Aligned | - | 4 types, 3-5 states each |
| **Movement** | ⚠️ Different mechanism | Medium | Two-phase GPU vs direct CPU, same logic |
| **Division** | ⚠️ Check asymmetric | Medium | Verify cancer progenitor behavior |
| **PDE solver** | ⚠️ Different method | Medium | Finite difference vs finite volume |
| **Chemical coupling** | ⚠️ Different timing | Medium | Batched (GPU) vs continuous (CPU) |
| **ODE integration** | ❌ Not integrated | High | CVODE available but unused in GPU |
| **Parameters** | ❌ Unverified | Critical | Need to compare D, λ, and agent parameters |
| **Gradients** | ❌ Missing | Low | Not used in current CPU model? |
| **Validation** | ❌ Not done | Critical | Side-by-side CPU vs GPU comparison |

