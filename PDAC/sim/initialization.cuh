#ifndef INITIALIZATION_CUH
#define INITIALIZATION_CUH

#include "flamegpu/flamegpu.h"
#include "gpu_param.h"
#include "../qsp/LymphCentral_wrapper.h"
#include <string>
#include <vector>

namespace PDAC {

// ============================================================================
// Simulation Configuration Structure
// ============================================================================

struct SimulationConfig {
    // ── Grid ──
    int grid_x;
    int grid_y;
    int grid_z;
    float voxel_size;  // micrometers

    // ── Simulation ──
    unsigned int steps;          // main-sim steps (after presim)
    unsigned int random_seed;

    // ── Initialization ──
    //   0 = single stem cell at grid center (default)
    //   1 = simulated-annealing placement (requires --phi-targets; TBD)
    int init_method;

    // Placement file (.plc). When non-empty, SKIPS QSP warmup and procedural
    // init: agents and V_vas(t=0) come from the file. Requires --qsp-state.
    std::string placement_file;

    // QSP state binary (.qsp). Required with --placement; written automatically
    // during warmup when no placement is provided.
    std::string qsp_state_file;

    // ── Presim ──
    //  -1 = volume stopper: run coupled presim until QSP tumor volume reaches
    //       the XML target (initial_tumour_diameter → π/6 × D³). DEFAULT.
    //  ≥0 = fixed step count (only valid with --placement + --qsp-state)
    int presim_steps;

    // ── Output ──
    // grid_out: 0=none, 1=ABM only, 2=PDE+ECM only, 3=both
    int grid_out;
    int interval_out;

    // ── ABM→QSP spatial feedback ──
    // When true, sets PARAM_COUPLING_MODE=1 (contact corrections).
    // When false (default), PARAM_COUPLING_MODE stays at XML value (typically 0).
    bool spatial_feedback;

    // ── Derived from XML (populated after param load, not CLI) ──
    float dt_abm;           // ABM timestep (seconds)
    int molecular_steps;    // PDE substeps per ABM step
    int cancer_move_steps;
    int tcell_move_steps;
    int treg_move_steps;
    int mdsc_move_steps;

    // ── Hidden / legacy (not in --help) ──
    // Scenario bundles (SingleStemEdge forces ABM-only + edge seed).
    enum class Scenario { Default, SingleStemEdge };
    Scenario scenario;
    std::string seed_position;
    bool ripley_enabled;

    // Constructor with defaults
    SimulationConfig();

    // Parse from command line arguments
    void parseCommandLine(int argc, const char** argv, const PDAC::GPUParam gpu_params);

    // Print configuration
    void print() const;
};

// ============================================================================
// Agent Population Initialization Functions
// ============================================================================

// Resident quiescent-PSC (stroma) seeding. Places quiescent fibroblasts at a per-voxel
// probability (from PARAM_DOMAIN_FIB_DENSITY); they activate + proliferate into CAF at run time.
// (Retired Jul 2026: initializeTH/TReg/MDSC/MacsFromQSP — the QSP-population TME seeding — deleted;
//  immune now emerge via QSP-coupled gap-closing recruitment, stroma from this resident field.)
void seedQuiescentPSCs(
    flamegpu::AgentVector& fib_agents,
    int grid_x, int grid_y, int grid_z,
    double p_fib, std::vector<std::vector<int>>& occupied,
    float life_mean);

// Vascular agent init functions removed (ABM rebuild Step 3b).
// O2 delivery + immune entry via Vvas continuous field.

// ============================================================================
// Domain Structure Generation (Structured Init, -i 1)
// ============================================================================

struct Point3 { float x, y, z; };

// Domain structure: temporary init-time data (no persistent voxel types)
struct DomainStructure {
    std::vector<bool> is_septum;          // true for Voronoi boundary voxels
    std::vector<bool> is_tumor;           // true for tumor hemisphere voxels
    std::vector<Point3> lobule_centers;   // for orientation computation
};

// Generate lobular tissue structure via Poisson disk + Voronoi tessellation.
// Identifies septum and tumor voxels (temporary, not stored on GPU).
DomainStructure generate_domain_structure(
    int grid_x, int grid_y, int grid_z,
    float lobule_spacing, float septum_thickness,
    float tumor_radius_frac,
    unsigned int seed);

// ECM pre-seeding: density, crosslink, floor, and fiber orientation.
// Septa get high density+crosslink+aligned orientation; lobule gets low baseline.
void preseed_ecm(
    const DomainStructure& domain,
    int grid_x, int grid_y, int grid_z,
    float septum_density, float septum_crosslink,
    float lobule_density, float tumor_density,
    float lobule_floor,
    unsigned int seed);

// ============================================================================
// Master Initialization Functions
// ============================================================================

// Initialize agent populations seeded from QSP steady-state (after QSP warmup)
// Computes cluster_radius and immune cell counts from QSP tumor volume and species
// Used by -i 0 (simple central cluster for quick testing)
void initializeSingleStem(
    flamegpu::CUDASimulation& simulation,
    flamegpu::ModelDescription& model,
    const SimulationConfig& config,
    const LymphCentralWrapper& lymph);

// Structured domain initialization with lobular architecture (-i 1)
// Generates tissue structure, pre-seeds ECM, places cells by region
void initializeStructuredDomain(
    flamegpu::CUDASimulation& simulation,
    flamegpu::ModelDescription& model,
    const SimulationConfig& config,
    const LymphCentralWrapper& lymph);

// Seed exactly one cancer stem cell at the grid center.
// Builds the agent by hand (not via initializeCancerCellsRandom + radius=0,
// which samples a CDF and may produce a non-stem state).
void seedSingleStemAtCenter(
    flamegpu::AgentVector& cancer_pop,
    int grid_x, int grid_y, int grid_z,
    float stem_div_interval);

// Single-stem-cell-at-edge diagnostic init.
// Domain pre-populated with resident healthy cells only:
//   - vasculature network (random sprouting, no tumor exclusion zone)
//   - quiescent pancreatic stellate cells (PSCs) at fixed density
// Then places exactly one cancer stem cell on the chosen domain face
// (config.seed_position). No immune agents are placed; FLAMEGPU requires
// setPopulationData on every agent type so empty populations are submitted.
// Designed for diagnostic edge-seeding runs (legacy).
void initializeSingleStemEdge(
    flamegpu::CUDASimulation& simulation,
    flamegpu::ModelDescription& model,
    const SimulationConfig& config,
    const LymphCentralWrapper& lymph);

// ============================================================================
// Initialization by data-driven PLACEMENT (workplan §C6) — `--placement <file.plc>`
// ============================================================================
// Reads a .plc written by abm-inference `placement_io.write_placement`: an agent table
// (type_id, x, y, z, cell_state, subtype) plus an optional V_vas(t=0) field. Replaces the
// presim entirely — the QSP owns population totals, the IMC data owns their arrangement.
//
// Two things this does NOT take from the file, on purpose:
//   * Per-agent internal state (lifespans, division counters, persistence, exhaustion) is
//     drawn here from the SAME distributions the recruitment path uses, so a placed cell and
//     a recruited cell differ only in how their position was chosen (workplan D-11).
//   * `vas_phalanx` places NO agent (D-4 dropped vascular agents); it arrives already
//     rasterized and smoothed into the V_vas field, which is uploaded via set_vvas_from_host.
//
// Returns false if the file is missing/malformed/grid-mismatched, so main can fail loudly
// rather than silently running an unseeded domain.
bool initializeFromPlacement(
    flamegpu::CUDASimulation& simulation,
    flamegpu::ModelDescription& model,
    const SimulationConfig& config,
    const LymphCentralWrapper& lymph);

} // namespace PDAC

#endif // INITIALIZATION_CUH
