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
extern double g_last_pde_ms;

// Accessor for last PDE solve time (milliseconds)
double get_last_pde_ms();

// FLAME GPU 2 host functions for PDE integration
// (Declared and implemented in pde_integration.cu)

// Reset source/uptake buffers before agent compute_chemical_sources functions run
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER reset_pde_buffers;

// Compute gradients for chemotaxis substrates (call after solve_pde_step)
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER compute_pde_gradients;

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
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER mark_mac_sources;

// Recruit new immune cells at marked sources
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER recruit_t_cells;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER recruit_mdscs;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER recruit_macrophages;

// Reset recruitment sources (call at start of each step)
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER reset_recruitment_sources;

// Update vasculature count env property (call before vascular source marking)
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER update_vasculature_count;

// Occupancy grid: zero before agents write their positions each step
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER zero_occupancy_grid;

// ECM grid: decay ECM each step after fibroblasts have deposited
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER update_ecm_grid;

// ============================================================================
// Timing Checkpoint Host Functions (inserted at phase boundaries)
// ============================================================================
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER timing_step_start;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER timing_after_recruit;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER timing_after_broadcast;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER timing_after_sources;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER timing_after_pde;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER timing_after_gradients;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER timing_after_ecm;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER timing_after_movement;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER timing_after_division;

} // namespace PDAC

#endif // PDE_INTEGRATION_CUH