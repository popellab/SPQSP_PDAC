#ifndef PDE_INTEGRATION_CUH
#define PDE_INTEGRATION_CUH

#include "flamegpu/flamegpu.h"
#include "pde_solver.cuh"
#include "gpu_param.h"

namespace PDAC {

// Global PDE solver instance (managed by host)
// Note: This is a workaround since FLAME GPU 2 host functions
// can't easily pass custom objects. In production, you'd use
// environment properties or a singleton pattern.
extern PDESolver* g_pde_solver;

// Read a specific chemical substrate from PDE into agent variable
void read_chemical_to_agents(
    flamegpu::HostAPI& host_api,
    const std::string& agent_name,
    int substrate_idx,
    const std::string& output_var_name);

// Collect a specific chemical source from agents into PDE
void collect_chemical_from_agents(
    flamegpu::HostAPI& host_api,
    const std::string& agent_name,
    int substrate_idx,
    const std::string& source_var_name);

// FLAME GPU 2 host functions for PDE integration
// (Declared and implemented in pde_integration.cu)

// Initialize PDE solver (call once at start)
void initialize_pde_solver(int grid_x, int grid_y, int grid_z,
                           float voxel_size, float dt_abm, int molecular_steps,
                            const PDAC::GPUParam& gpu_params);

// Set PDE solver pointers in FLAME GPU environment properties
void set_pde_pointers_in_environment(flamegpu::ModelDescription& model);

// Cleanup PDE solver (call at end)
void cleanup_pde_solver();

// Recruitment host functions (declared and implemented in pde_integration.cu)

// Mark recruitment sources based on chemical concentrations
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER mark_mdsc_sources;

// Recruit new immune cells at marked sources
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER recruit_t_cells;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER recruit_mdscs;

// Reset recruitment sources (call at start of each step)
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER reset_recruitment_sources;

} // namespace PDAC

#endif // PDE_INTEGRATION_CUH