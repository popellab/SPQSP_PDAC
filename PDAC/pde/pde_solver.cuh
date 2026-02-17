#ifndef PDE_SOLVER_CUH
#define PDE_SOLVER_CUH

#include <cuda_runtime.h>
#include <vector>
#include <string>

namespace PDAC {

// Chemical substrate indices (matching CPU HCC implementation)
// Note: NIVO and CABO moved to QSP compartments, 4 new chemicals added
enum ChemicalSubstrate {
    CHEM_O2 = 0,
    CHEM_IFN,       // IFN-gamma
    CHEM_IL2,       // IL-2
    CHEM_IL10,      // IL-10
    CHEM_TGFB,      // TGF-beta
    CHEM_CCL2,      // CCL2
    CHEM_ARGI,      // Arginase I (MDSC produces)
    CHEM_NO,        // Nitric Oxide (MDSC produces)
    CHEM_IL12,      // IL-12 (M1 macrophages produce)
    CHEM_VEGFA,     // VEGF-A (cancer cells produce)
    NUM_SUBSTRATES
};

// Configuration for PDE solver
struct PDEConfig {
    int nx, ny, nz;                    // Grid dimensions
    int num_substrates;                 // Number of chemical species
    float voxel_size;                   // Spatial resolution (cm)
    float dt_abm;                       // ABM timestep (seconds)
    float dt_pde;                       // PDE timestep (seconds)
    int substeps_per_abm;               // PDE substeps per ABM step
    
    // Diffusion coefficients (cm²/s) for each substrate
    float diffusion_coeffs[NUM_SUBSTRATES];
    
    // Decay rates (1/s) for each substrate
    float decay_rates[NUM_SUBSTRATES];
    
    // Boundary conditions (0 = Neumann/no-flux, 1 = Dirichlet)
    int boundary_type;
};

class PDESolver {
public:
    PDESolver(const PDEConfig& config);
    ~PDESolver();
    
    // Initialize solver and allocate memory
    void initialize();
    
    // Run PDE for one ABM timestep (runs substeps internally)
    void solve_timestep();
    
    // Agent-PDE coupling: set source/sink values
    // sources: array of size [num_substrates][nz][ny][nx]
    void set_sources(const float* h_sources, int substrate_idx);
    void add_source_at_voxel(int x, int y, int z, int substrate_idx, float value);
    
    // Agent-PDE coupling: get concentration values
    void get_concentrations(float* h_concentrations, int substrate_idx) const;
    float get_concentration_at_voxel(int x, int y, int z, int substrate_idx) const;
    
    // Direct device pointer access (for FLAME GPU integration)
    float* get_device_concentration_ptr(int substrate_idx);
    float* get_device_source_ptr(int substrate_idx);
    
    // Reset all concentrations to zero
    void reset_concentrations();
    void reset_sources();
    
    // Set uniform initial concentration for a substrate
    void set_initial_concentration(int substrate_idx, float value);

    // Get total source for a substrate (for debugging)
    float get_total_source(int substrate_idx);

    // Recruitment source management
    int* get_device_recruitment_sources_ptr() { return d_recruitment_sources_; }
    void reset_recruitment_sources();

    // Utility
    int get_total_voxels() const { return config_.nx * config_.ny * config_.nz; }
    
private:
    PDEConfig config_;

    // Device memory
    float* d_concentrations_current_;   // [num_substrates][nz][ny][nx]
    float* d_concentrations_next_;      // Double buffering (for output)
    float* d_sources_;                   // [num_substrates][nz][ny][nx]

    // Recruitment sources (bit flags: 1=T_source, 2=MDSC_source, 3=both)
    int* d_recruitment_sources_;        // [nz][ny][nx]

    // CG solver workspace (per voxel, not per substrate)
    float* d_cg_r_;      // Residual vector
    float* d_cg_p_;      // Search direction
    float* d_cg_Ap_;     // A*p product
    float* d_cg_z_;      // Preconditioned residual (for diagonal preconditioner)
    float* d_cg_temp_;   // Temporary storage
    float* d_dot_buffer_; // Reduction buffer for dot products
    float* d_precond_diag_inv_; // Diagonal preconditioner M^{-1} (inverse stored)
    int cg_reduction_blocks_;
    float last_residual_norm_;  // Final residual from last CG solve (for diagnostics)

    // Multigrid workspace
    float* d_mg_residual_;     // Residual on fine grid
    float* d_mg_correction_;   // Correction from coarse grid
    float* d_mg_coarse_;       // Solution on coarse grid (nx/2 × ny/2 × nz/2)
    float* d_mg_coarse_rhs_;   // RHS on coarse grid
    int mg_coarse_nx_, mg_coarse_ny_, mg_coarse_nz_;  // Coarse grid dimensions

    // Host memory (for transfers)
    float* h_temp_buffer_;

    // Internal indexing
    inline int idx(int x, int y, int z) const {
        return z * (config_.nx * config_.ny) + y * config_.nx + x;
    }

    inline int idx_substrate(int x, int y, int z, int substrate) const {
        return substrate * (config_.nx * config_.ny * config_.nz) + idx(x, y, z);
    }

    // CG solver internals
    int solve_implicit_cg(float* d_C, const float* d_rhs, float D, float lambda, float dt, float dx);

    // Multigrid solver internals
    int solve_multigrid(float* d_C, const float* d_rhs, float D, float lambda, float dt, float dx);
    void mg_smooth(float* d_x, const float* d_rhs, float D, float lambda, float dt, float dx,
                   int nx, int ny, int nz, int num_iters, float omega);
    void mg_compute_residual(const float* d_x, const float* d_rhs, float* d_residual,
                             float D, float lambda, float dt, float dx, int nx, int ny, int nz);

    // Swap current and next buffers
    void swap_buffers();
};

// CUDA kernel declarations
__global__ void diffusion_reaction_kernel(
    const float* __restrict__ C_curr,
    float* __restrict__ C_next,
    const float* __restrict__ sources,
    int nx, int ny, int nz,
    float D, float lambda, float dt, float dx
);

__global__ void copy_substrate_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int n
);

__global__ void read_concentrations_at_voxels(
    const float* __restrict__ d_concentrations,
    const int* __restrict__ d_agent_x,
    const int* __restrict__ d_agent_y,
    const int* __restrict__ d_agent_z,
    float* __restrict__ d_agent_concentrations,
    int num_agents,
    int substrate_idx,
    int nx, int ny, int nz);

__global__ void add_sources_from_agents(
    float* __restrict__ d_sources,
    const int* __restrict__ d_agent_x,
    const int* __restrict__ d_agent_y,
    const int* __restrict__ d_agent_z,
    const float* __restrict__ d_agent_source_rates,
    int num_agents,
    int substrate_idx,
    int nx, int ny, int nz,
    float voxel_volume);

// CG solver kernels
__global__ void apply_diffusion_operator(
    const float* __restrict__ x,
    float* __restrict__ Ax,
    int nx, int ny, int nz,
    float D, float lambda, float dt, float dx);

__global__ void vector_axpy(
    float* __restrict__ y,
    const float* __restrict__ x,
    float alpha,
    int n);

__global__ void vector_scale(
    float* __restrict__ y,
    const float* __restrict__ x,
    float alpha,
    int n);

__global__ void vector_copy(
    float* __restrict__ dst,
    const float* __restrict__ src,
    int n);

__global__ void dot_product_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ partial_sums,
    int n);

__global__ void clamp_nonnegative(
    float* __restrict__ x,
    int n);

// Multigrid kernels
__global__ void restrict_residual(
    const float* __restrict__ fine,
    float* __restrict__ coarse,
    int nx_fine, int ny_fine, int nz_fine);

__global__ void prolong_correction(
    const float* __restrict__ coarse,
    float* __restrict__ fine,
    int nx_fine, int ny_fine, int nz_fine);

__global__ void weighted_jacobi_kernel(
    const float* __restrict__ x_old,
    const float* __restrict__ rhs,
    float* __restrict__ x_new,
    int nx, int ny, int nz,
    float D, float lambda, float dt, float dx, float omega);

} // namespace PDAC

#endif // PDE_SOLVER_CUH