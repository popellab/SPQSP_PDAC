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
    // Grid parameters
    int grid_x;
    int grid_y;
    int grid_z;
    float voxel_size;  // micrometers

    // Simulation parameters
    unsigned int steps;
    unsigned int random_seed;

    // Initialization method (0 = QSP-seeded, only supported mode)
    int init_method;

    // Vasculature initialization mode
    std::string vascular_mode;  // "random", "xml", "test"
    std::string vascular_xml_file;

    // Movement iterations per timestep
    int cancer_move_steps;
    int tcell_move_steps;
    int treg_move_steps;
    int mdsc_move_steps;

    // PDE parameters
    float dt_abm;           // ABM timestep (seconds)
    int molecular_steps;    // PDE substeps per ABM step

    // Output parameters
    // grid_out: 0=none, 1=ABM only, 2=PDE+ECM only, 3=both
    int grid_out;
    int interval_out;

    // Constructor with defaults
    SimulationConfig();

    // Parse from command line arguments
    void parseCommandLine(int argc, const char** argv, const PDAC::GPUParam gpu_params);

    // Print configuration
    void print() const;
};

// ============================================================================
// Agent Population Initialization Functions (QSP-seeded)
// ============================================================================

void initializeFibroblastsFromQSP(
    flamegpu::AgentVector& fib_agents,
    int grid_x, int grid_y, int grid_z,
    double p_fib, std::vector<std::vector<int>>& occupied,
    float life_mean);

void initializeTHCellsFromQSP(
    flamegpu::AgentVector& treg_agents,
    int grid_x, int grid_y, int grid_z,
    double p_th, std::vector<std::vector<int>>& occupied,
    float life_mean, int div_limit, int div_interval);

void initializeTRegCellsFromQSP(
    flamegpu::AgentVector& treg_agents,
    int grid_x, int grid_y, int grid_z,
    double p_treg, std::vector<std::vector<int>>& occupied,
    float life_mean, int div_limit, int div_interval);

void initializeMDSCsFromQSP(
    flamegpu::AgentVector& mdsc_agents,
    int grid_x, int grid_y, int grid_z,
    double p_mdsc, std::vector<std::vector<int>>& occupied,
    float life_mean);

// Initialize Vascular Cells with random walk (HCC-style)
void initializeVascularCellsRandom(
    flamegpu::AgentVector& vascular_agents,
    int grid_x, int grid_y, int grid_z,
    int tumor_radius,
    int num_segments,
    float branch_prob,
    unsigned int seed);

// Sequentially assign INTENT_DIVIDE to spatially distributed PHALANX cells pre-simulation.
// Must be called after initializeVascularCellsRandom; step 0 of vascular_state_step is skipped
// so these intents are preserved until vascular_divide executes.
void assignInitialVascularTips(
    flamegpu::AgentVector& vascular_agents,
    int grid_x, int grid_y, int grid_z,
    int min_neighbor_range,
    unsigned int seed);

// Initialize Vascular Cells with manual test pattern
void initializeVascularCellsTest(
    flamegpu::AgentVector& vascular_agents,
    int grid_x, int grid_y, int grid_z);

// ============================================================================
// Domain Structure Generation (Structured Init, -i 1)
// ============================================================================

// Generate lobular tissue structure: Poisson disk lobule centers → Voronoi
// tessellation → septum/lobule labels → tumor hemisphere + margin overlay.
// Returns host-side voxel type array (uint8_t, total_voxels elements).
std::vector<uint8_t> generate_domain_structure(
    int grid_x, int grid_y, int grid_z,
    float lobule_spacing, float septum_thickness,
    float tumor_radius_frac, float margin_thickness,
    unsigned int seed);

// ECM density/crosslink pre-seeding by voxel type.
// Reads voxel_type array and DomainInit params, writes to d_ecm_density and
// d_ecm_crosslink device arrays via pde_integration helpers.
struct ECMInitParams {
    float septum_density, septum_crosslink;
    float stroma_density;
    float lobule_density;
    float margin_density, margin_crosslink;
    float tumor_density;
};

void preseed_ecm_by_voxel_type(
    const std::vector<uint8_t>& voxel_type,
    const ECMInitParams& ecm_params,
    int total_voxels, unsigned int seed);

// ============================================================================
// Master Initialization Functions
// ============================================================================

// Initialize agent populations seeded from QSP steady-state (after QSP warmup)
// Computes cluster_radius and immune cell counts from QSP tumor volume and species
// Used by -i 0 (simple central cluster for quick testing)
void initializeToQSP(
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

} // namespace PDAC

#endif // INITIALIZATION_CUH
