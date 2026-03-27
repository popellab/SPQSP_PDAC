#ifndef PDE_INTEGRATION_CUH
#define PDE_INTEGRATION_CUH

#include <cstdint>
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

// Per-step recruitment stats — populated by recruit_gpu / place_recruited_agents
struct RecruitStats {
    int teff_rec = 0, treg_rec = 0, th_rec = 0;
    int mdsc_rec = 0, mac_rec = 0, mac_m1_rec = 0, mac_m2_rec = 0;
    float p_teff = 0.f, p_treg = 0.f, p_th = 0.f;
    int t_sources = 0, mdsc_sources = 0, mac_sources = 0;
    float qsp_teff = 0.f, qsp_treg = 0.f, qsp_th = 0.f;
};
RecruitStats get_last_recruit_stats();

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

// Initialize ECM grid to saturation value (call after QSP is initialized, before simulation)
// Matches HCC behavior: all voxels start at ECM_saturation, not 0/baseline.
void initialize_ecm_to_saturation(float ecm_saturation);

// Return device pointer for voxel type grid (domain initialization labels)
uint8_t* get_voxel_type_device_ptr();

// Copy host-side voxel type array to device (call after generate_domain_structure)
void set_voxel_type_from_host(const uint8_t* host_data, int total_voxels);

// Copy host-side ECM arrays to device (call after preseed_ecm_by_voxel_type)
void set_ecm_density_from_host(const float* host_data, int total_voxels);
void set_ecm_crosslink_from_host(const float* host_data, int total_voxels);

// Return device pointers for ECM arrays (for output)
float* get_ecm_density_device_ptr();
float* get_ecm_crosslink_device_ptr();
float* get_fib_density_field_device_ptr();

// Return device pointer for vascular tip_id grid (for output/debug)
unsigned int* get_vas_tip_id_grid_device_ptr();

// Run PDE-only warmup (N substeps, no agent sources — just diffusion+decay)
// Call after agent init to establish baseline chemical field before first ABM step.
void run_pde_warmup(int substeps);

// Cleanup PDE solver (call at end)
void cleanup_pde_solver();

// Recruitment host functions (declared and implemented in pde_integration.cu)

// Mark recruitment sources based on chemical concentrations
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER mark_mdsc_sources;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER mark_mac_sources;

// GPU recruitment: kernel decides placement, host fn creates agents
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER recruit_gpu;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER place_recruited_agents;

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
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER reset_divide_wave;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER increment_divide_wave;

} // namespace PDAC

#endif // PDE_INTEGRATION_CUH