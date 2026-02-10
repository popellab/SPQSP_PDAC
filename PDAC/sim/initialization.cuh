#ifndef INITIALIZATION_CUH
#define INITIALIZATION_CUH

#include "flamegpu/flamegpu.h"
#include "gpu_param.h"
#include <string>

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

// ============================================================================
// Master Initialization Function
// ============================================================================

// Initialize all agent populations in the simulation
void initializeAllAgents(
    flamegpu::CUDASimulation& simulation,
    flamegpu::ModelDescription& model,
    const SimulationConfig& config);

} // namespace PDAC

#endif // INITIALIZATION_H