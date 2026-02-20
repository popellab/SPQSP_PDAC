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

    // Initialization parameters
    int init_method;
    
    // Initial cell populations
    int cluster_radius;
    int num_tcells;
    int num_tregs;
    int num_mdscs;
    int num_macrophages;
    int num_fibroblasts;

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
    bool abm_out;
    bool pde_out;
    int interval_out;
    
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

// Initialize cancer cell cluster at grid center
void initializeCancerCellCluster(
    flamegpu::AgentVector& cancer_agents,
    int grid_x, int grid_y, int grid_z,
    int cluster_radius,
    float stem_div_interval,
    float progenitor_div_interval,
    int progenitor_div_max);

// Initialize T cells around tumor margin
void initializeTCells(
    flamegpu::AgentVector& tcell_agents,
    int grid_x, int grid_y, int grid_z,
    int tumor_radius, int num_tcells,
    float tcell_life_mean, int div_limit,
    float IL2_release_time, float IFN_release_time);

// Initialize TReg cells around tumor margin
void initializeTRegs(
    flamegpu::AgentVector& treg_agents,
    int grid_x, int grid_y, int grid_z,
    int tumor_radius, int num_tregs,
    float treg_life_mean, int div_limit);

// Initialize MDSCs around tumor margin
void initializeMDSCs(
    flamegpu::AgentVector& mdsc_agents,
    int grid_x, int grid_y, int grid_z,
    int tumor_radius, int num_mdscs,
    float mdsc_life_mean);

// Initialize Macrophages around tumor margin
void initializeMacrophages(
    flamegpu::AgentVector& mac_agents,
    int grid_x, int grid_y, int grid_z,
    int tumor_radius, int num_macrophages,
    float mac_life_mean);

// Initialize Fibroblasts around tumor margin
void initializeFibroblasts(
    flamegpu::AgentVector& fib_agents,
    int grid_x, int grid_y, int grid_z,
    int tumor_radius, int num_fibroblasts,
    float fib_life_mean);

void initializeFibroblastsFromQSP(
    flamegpu::AgentVector& fib_agents,
    int grid_x, int grid_y, int grid_z,
    double p_fib, std::vector<std::vector<int>>& occupied,
    float life_mean);

// QSP probability-based initialization (iterate all voxels, place if rand < p and voxel empty)
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
    int num_segments);

// Initialize Vascular Cells with manual test pattern
void initializeVascularCellsTest(
    flamegpu::AgentVector& vascular_agents,
    int grid_x, int grid_y, int grid_z);

// ============================================================================
// Master Initialization Functions
// ============================================================================

// Initialize all agent populations with fixed config values (manual/default init)
void initializeAllAgents(
    flamegpu::CUDASimulation& simulation,
    flamegpu::ModelDescription& model,
    const SimulationConfig& config);

// Initialize agent populations seeded from QSP steady-state (after QSP warmup)
// Computes cluster_radius and immune cell counts from QSP tumor volume and species
void initializeToQSP(
    flamegpu::CUDASimulation& simulation,
    flamegpu::ModelDescription& model,
    const SimulationConfig& config,
    const LymphCentralWrapper& lymph);

} // namespace PDAC

#endif // INITIALIZATION_H