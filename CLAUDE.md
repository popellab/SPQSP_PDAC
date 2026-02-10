# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SPQSP PDAC is a GPU-accelerated agent-based model (ABM) with CPU QSP coupling for simulating pancreatic ductal adenocarcinoma (PDAC) tumor microenvironment dynamics. It combines:

- **GPU (FLAME GPU 2)**:
  - Discrete agent-based modeling (cancer cells, T cells, regulatory T cells, MDSCs)
  - Continuous PDE-based chemical diffusion (O2, IFN, IL2, IL10, TGFB, CCL2, ArgI, NO, IL12, VEGFA)

- **CPU (SUNDIALS CVODE)**:
  - Systemic QSP model for LymphCentral compartment (59+ species)
  - Drug pharmacokinetics (NIVO, CABO) and pharmacodynamics
  - Immune cell recruitment/exhaustion feedback loops

- **Integration**:
  - XML-based parameter system for dynamic configuration
  - GPU-CPU data exchange for bidirectional coupling
  - Voxel-based spatial discretization (configurable 10-100 µm resolution)

## Architecture

### Directory Structure
```
PDAC/
├── sim/               # Main simulation entry points
│   ├── main.cu        # Entry point, builds model, initializes QSP/PDE
│   ├── model_definition.cu  # FLAME GPU agent/message definitions
│   ├── initialization.cu    # Agent population initialization
│   ├── CMakeLists.txt       # Build configuration
│   └── resource/
│       └── param_all_test.xml  # Parameter configuration file
│
├── abm/               # GPU parameter system
│   ├── gpu_param.h/.cu # Type-safe parameter class, XML loading
│
├── core/              # Core simulation infrastructure
│   ├── common.cuh     # Agent types, enums, constants
│   ├── model_functions.cu/cuh  # QSP-ABM coupling functions
│   ├── ParamBase.h/.cpp        # Base parameter class
│
├── agents/            # CUDA device functions for agent behavior
│   ├── cancer_cell.cuh    # Cancer stem/progenitor cell dynamics
│   ├── t_cell.cuh         # Effector/Cytotoxic/Suppressed T cell states
│   ├── t_reg.cuh          # Regulatory T cell suppression
│   ├── mdsc.cuh           # Myeloid-derived suppressor cells
│
├── pde/               # Chemical transport solver
│   ├── pde_solver.cu/cuh      # BioFVM-based diffusion/decay
│   ├── pde_integration.cu/cuh # GPU-PDE coupling functions
│
└── qsp/               # QSP model integration (CPU)
    ├── LymphCentral_wrapper.h/.cpp  # CVODE wrapper for ODE system
    ├── qsp_integration.cu            # QSP stepping and ABM feedback
    ├── cvode/
    │   ├── CVODEBase.h/.cpp  # SUNDIALS CVODE interface
    │   └── MolecularModelCVode.h
    └── ode/
        ├── ODE_system.h/.cpp  # 59+ species HCC biological model
        ├── QSPParam.h/.cpp    # QSP parameter loader
        └── QSP_enum.h
```

### Agent Types (PDAC namespace in core/common.cuh)

**CancerCell**
- States: Stem, Progenitor, PDL1+, PDL1-, Senescent
- Variables: position, cell_state, division countdown, PDL1 expression, neighbor counts
- Functions: broadcast_location, movement, division, PDL1 dynamics, neighbor scanning

**TCell** (Effector CD8+ T cells)
- States: Effector, Cytotoxic, Suppressed
- Variables: position, state, migration, cytotoxic activity, PDL1 interaction, lifespan
- Functions: migration, state transitions, cytokine sensing, cancer killing

**TReg** (Regulatory T cells)
- Variables: position, suppression strength, lifespan
- Functions: migration, suppression of effector T cells, recruitment

**MDSC** (Myeloid-Derived Suppressor Cells)
- Variables: position, suppression capacity, lifespan
- Functions: migration, T cell suppression via NO/ArgI

### Key Simulation Flow

1. **Initialization** (main.cu:55-125)
   - Load XML parameters via `GPUParam::initializeParams()`
   - Initialize PDE solver via `initialize_pde_solver()`
   - Initialize QSP model via `LymphCentralWrapper::initialize()`
   - Setup internal derived parameters via `set_internal_params()`
   - Create initial agent population via `initializeAllAgents()`

2. **Per ABM Step** (main.cu:129, model_definition.cu)
   - **Phase 1: Chemical Updates**
     - `update_agent_chemicals`: Read chemical concentrations from PDE into agents
     - Agents compute source/sink rates based on state and local chemicals

   - **Phase 2: Agent Behavior**
     - `cancer_broadcast_location`, `tcell_scan_neighbors`, etc.
     - Agent movement, division, state transitions, killing interactions

   - **Phase 3: PDE Update**
     - `collect_agent_sources`: Aggregate sources/sinks from all agents
     - `solve_pde_step`: Run multiple PDE substeps (usually ~111 per ABM step)

   - **Phase 4: QSP Coupling**
     - `solve_qsp_step`: Advance CPU ODE system
     - Extract drug concentrations, immune mediators
     - Update environment properties for next ABM step

3. **Termination**
   - Exit condition: all cancer cells eliminated
   - Output final population counts, agent state distributions

### Parameter System

**XML Structure** (param_all_test.xml)
```xml
<Param>
  <Molecular>
    <biofvm>           <!-- PDE chemical properties -->
      <DiffusionCoeff>
        <O2>...</O2>
        ...
      </DiffusionCoeff>
      <DecayRate>...</DecayRate>
    </biofvm>
  </Molecular>
  <ABM>               <!-- Agent-based model parameters -->
    <CancerCell>      <!-- Movement, division, senescence -->
    <TCell>           <!-- Recruitment, state transitions -->
    <TCD4>            <!-- TReg parameters -->
    <MDSC>            <!-- Suppression, recruitment -->
    <PD1_PDL1>        <!-- Checkpoint blocking mechanics -->
    <PDL1>            <!-- PDL1 upregulation dynamics -->
  </ABM>
</Param>
```

**Loading Process**
- `main.cu`: `GPUParam::initializeParams(param_file)` loads XML
- `model_definition.cu`: `params.populateFlameGPUEnvironment(env)` populates FLAME GPU environment
- `model_functions.cu`: `set_internal_params()` derives secondary parameters from QSP model

### Two-Phase Conflict Resolution

Agents use intent-based movement/division to avoid conflicts:

1. **Phase 1: Intent Broadcasting**
   - Agents select targets and broadcast intent via `MSG_INTENT` messages
   - Check for spatial conflicts with neighboring voxels

2. **Phase 2: Action Execution**
   - Execute confirmed actions (move, divide) if space available
   - Resolve ties via agent priority ordering

### Spatial Messaging

- All agents broadcast location via `MSG_CELL_LOCATION`
- Receivers query 26-neighbor Moore neighborhood
- Neighbor counts cached: `neighbor_Teff_count`, `neighbor_Treg_count`, etc.

### GPU-CPU Coupling Architecture

**Data Flow: GPU → CPU**
- ABM → LymphCentralWrapper via `update_from_abm()`
  - Cancer deaths (T cell kills)
  - T cell deaths and recruitment
  - Current tumor volume and cell count

**Data Flow: CPU → GPU**
- QSPState → Environment via `solve_qsp_step()`
  - Drug concentrations (nivo_tumor, cabo_tumor)
  - Immune mediator levels (teff_central, ifn_central, etc.)
  - Tumor status (remaining capacity, necrotic fraction)

## Build & Run

### Requirements
- CUDA Toolkit 11.0+ (tested with CUDA 12.x)
- CMake 3.18+
- C++17 compiler
- SUNDIALS 4.0.1 (for QSP CVODE solver)
- Boost 1.70+ (for serialization)
- FLAME GPU 2 v2.0.0-rc.4 (auto-fetched)

### Build Commands
```bash
cd PDAC/sim
./build.sh                    # Release build (default CUDA arch 75,80,86)
./build.sh --debug            # Debug build with symbols
./build.sh --cuda-arch 86     # Override CUDA architecture
./build.sh --clean            # Clean and rebuild
./build.sh --flamegpu ~/path  # Use local FLAME GPU source
```

### Run Commands
```bash
./build/bin/pdac                    # Run with defaults (51³ grid, 500 steps)
./build/bin/pdac -s 100             # Run 100 steps
./build/bin/pdac -g 21 -s 10        # Run on 21³ grid, 10 steps
./build/bin/pdac -p custom.xml      # Use custom parameter file
./build/bin/pdac -t 30 --tregs 5    # 30 T cells, 5 TRegs initially
```

**Command-line Options**
```
-g, --grid-size N       Grid dimensions (default: 51)
-s, --steps N           Simulation steps (default: 500)
-r, --radius N          Initial tumor radius (default: 5)
-t, --tcells N          Initial T cell count (default: 50)
--tregs N               Initial TReg count (default: 10)
--mdscs N               Initial MDSC count (default: 5)
-m, --move-steps N      PDE steps per ABM step (default: 111)
-p, --param-file PATH   XML parameter file path
```

## Key Files to Understand

### GPU Agent Behavior
- **cancer_cell.cuh**: PDL1 upregulation via Hill equation, hypoxic response, division, senescence
- **t_cell.cuh**: State machine (EFF→CYT→SUPP), neighbor scanning, killing probability, cytokine sensing
- **t_reg.cuh**: Suppression radius and strength, recruitment dynamics
- **mdsc.cuh**: Suppression via NO/ArgI, migration patterns

### QSP Integration
- **LymphCentral_wrapper.h/cpp**: CVODE wrapper, species access, ABM feedback
- **qsp_integration.cu**: `solve_qsp_step` host function to call wrapper from GPU
- **model_functions.cu**: `set_internal_params()` to derive all derived parameters from ODE system
- **ODE_system.h/cpp**: The actual 59+ species biological model from HCC codebase

### Parameter System
- **gpu_param.h/cpp**: Type-safe enums `GPUParamFloat`, `GPUParamInt`, `GPUParamBool` with XML mappings
- **param_all_test.xml**: Master configuration file with all PDE/ABM/QSP parameters

### PDE Solver
- **pde_solver.cuh**: Chemical enum (10 substances), diffusion coefficients, decay rates
- **pde_integration.cuh**: Host functions for chemical updates and PDE stepping

## Implementation Status

### ✅ Complete
- GPU agent types and behavior functions
- PDE chemical diffusion (10 substances)
- Parameter system (XML loading + GPU environment population)
- QSP model integration (CVODE wrapper)
- Two-phase conflict resolution
- Spatial messaging system
- Build system with SUNDIALS/Boost/FLAMEGPU
- Voxel capacity constraints
- Population tracking and reporting

### 🔄 In Progress
- QSP-ABM feedback loop (infrastructure complete, need species index mapping)
- Drug concentration transfer to GPU environment
- Recruitment rate modulation from QSP outputs
- Comprehensive validation against CPU HCC outputs

### 📋 Todo
- Extract species indices from ODE_system for drug/immune mediators
- Implement `LymphCentral_wrapper::_extract_species_indices()`
- Populate `_apply_abm_feedback()` for ABM→QSP coupling
- Bidirectional parameter exchange (QSP→ABM drug effects)
- Unit tests for PDE solver accuracy
- Comparative validation (GPU vs CPU on same initial conditions)
- Performance profiling and optimization

## Important Constants

**Voxel Capacity** (common.cuh)
- `MAX_T_PER_VOXEL = 8` (no cancer)
- `MAX_T_PER_VOXEL_WITH_CANCER = 1`
- `MAX_CANCER_PER_VOXEL = 1`
- `MAX_MDSC_PER_VOXEL = 1`

**Chemical Substances** (pde_solver.cuh)
- 0: O2 (oxygen)
- 1: IFN (interferon-gamma)
- 2: IL2 (interleukin-2)
- 3: IL10 (interleukin-10)
- 4: TGFB (TGF-beta)
- 5: CCL2 (chemokine)
- 6: ARGI (arginase I)
- 7: NO (nitric oxide)
- 8: IL12 (interleukin-12)
- 9: VEGFA (VEGF-alpha)

**Paths**
- SUNDIALS: `$HOME/lib/sundials-4.0.1` (manually configured)
- Boost: `$HOME/lib/boost_1_70_0` (manually configured)
- FLAME GPU: Auto-fetched from GitHub
- Parameters: `/PDAC/sim/resource/param_all_test.xml`

## Known Issues & Workarounds

**Issue**: CUDA 12.6 compatibility with FLAME GPU CUB templates
- **Status**: Build may fail with CUB template errors in some CUDA versions
- **Workaround**: Fall back to working binary or use compatible CUDA version
- **Location**: `_deps/cccl-src/dispatch_transform.cuh`

**Issue**: QSP species indices not yet mapped
- **Status**: Need to extract indices from ODE_system for drug/immune mediators
- **Next Step**: Implement `_extract_species_indices()` in LymphCentralWrapper

## Development Workflow

1. **Making agent changes**: Edit `.cuh` files in `/agents/`, rebuild
2. **Adding parameters**: Update XML, add entry to `GPUParamFloat`/`GPUParamInt` in gpu_param.h
3. **Changing PDE properties**: Update `pde_solver.cuh` enum and parameters
4. **QSP integration**: Modify `model_functions.cu::set_internal_params()` and `qsp_integration.cu::solve_qsp_step()`

## Debugging Tips

```bash
# Run small grid for quick iteration
./build/bin/pdac -g 11 -s 2

# Enable GPU error checking
export CUDA_LAUNCH_BLOCKING=1

# Check parameter loading
./build/bin/pdac -p param_all_test.xml 2>&1 | grep -i param

# Profile FLAME GPU
./build/bin/pdac -s 10 --profile
```
