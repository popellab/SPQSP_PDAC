#ifndef PDE_SOLVER_CUH
#define PDE_SOLVER_CUH

#include <cuda_runtime.h>
#include <vector>
#include <string>

namespace PDAC {

// Chemical substrate indices
enum ChemicalSubstrate {
    CHEM_O2 = 0,
    CHEM_IFN,       // IFN-gamma
    CHEM_IL2,       // IL-2
    CHEM_IL10,      // IL-10
    CHEM_TGFB,      // TGF-beta
    CHEM_CCL2,      // CCL2
    CHEM_ARGI,      // Arginase I
    CHEM_NO,        // Nitric Oxide
    CHEM_IL12,      // IL-12
    CHEM_VEGFA,     // VEGF-A
    NUM_SUBSTRATES
};

// Gradient substrates (subset used for chemotaxis)
// GRAD_IFN=0 → CHEM_IFN, GRAD_TGFB=1 → CHEM_TGFB, GRAD_CCL2=2 → CHEM_CCL2, GRAD_VEGFA=3 → CHEM_VEGFA
enum GradientSubstrate {
    GRAD_IFN = 0,
    GRAD_TGFB,
    GRAD_CCL2,
    GRAD_VEGFA,
    NUM_GRAD_SUBSTRATES
};

struct PDEConfig {
    int nx, ny, nz;                    // Grid dimensions
    int num_substrates;                 // Number of chemical species
    float voxel_size;                   // Spatial resolution (cm)
    float dt_abm;                       // ABM timestep (seconds)
    float dt_pde;                       // PDE substep (seconds, = dt_abm / substeps_per_abm)
    int substeps_per_abm;               // Molecular substeps per ABM step (36, matches BioFVM)
    int boundary_type;                  // Unused (always Neumann no-flux)

    float diffusion_coeffs[NUM_SUBSTRATES];  // cm²/s
    float decay_rates[NUM_SUBSTRATES];       // 1/s (background decay λ)
};

/**
 * PDESolver: LOD (Locally One-Dimensional) implicit diffusion + exact ODE source/uptake
 *
 * Mathematics (matches BioFVM LOD_3D exactly):
 *
 * Step 1 — Source/uptake (exact ODE, cell terms only):
 *   dp/dt = S - U*p
 *   if U > 1e-10: p_new = (p - S/U)*exp(-U*dt) + S/U
 *   else:         p_new = p + S*dt
 *   S [conc/s] = secretion/voxel_volume, U [1/s] = cell uptake only (no λ here)
 *
 * Step 2 — LOD diffusion+decay (3 implicit 1D Thomas sweeps):
 *   c1 = dt*D/dx²,  c2 = dt*λ/3 (decay split over 3 sweeps, matching BioFVM)
 *   Interior diagonal: 1 + 2*c1 + c2
 *   Boundary diagonal: 1 + c1 + c2
 *   Off-diagonal: -c1
 *
 * Agent-PDE coupling (direct device pointer access):
 *   Agent FLAMEGPU functions atomicAdd to d_src_[] / d_upt_[] via uint64_t env pointers
 *   Agents read d_conc_[] directly via env pointers (no host loops needed)
 */
class PDESolver {
public:
    PDESolver(const PDEConfig& config);
    ~PDESolver();

    // Initialize solver: allocate memory, precompute Thomas coefficients
    void initialize();

    // Run one timestep: apply sources/uptakes (exact ODE) then LOD diffusion+decay
    void solve_timestep();

    // Compute gradients for chemotaxis substrates (call after solve_timestep)
    void compute_gradients();

    // Reset source/uptake arrays to zero (call before agent compute functions)
    void reset_sources();
    void reset_uptakes();
    void reset_recruitment_sources();
    void reset_concentrations();

    // D2H copy for CSV output
    void get_concentrations(float* h_buf, int substrate_idx) const;

    // D2H copy of ALL substrates in one transfer: [NUM_SUBSTRATES * V] floats
    void get_all_concentrations(float* h_buf) const;

    // Async D2H copy of all substrates on the given stream (h_buf must be pinned memory)
    void get_all_concentrations_async(float* h_buf, cudaStream_t stream) const;

    // Set uniform initial concentration
    void set_initial_concentration(int substrate_idx, float value);

    // Device pointer accessors (stored as uint64_t env properties for agent access)
    float* get_device_concentration_ptr(int substrate_idx);
    float* get_device_source_ptr(int substrate_idx);
    float* get_device_uptake_ptr(int substrate_idx);
    // Gradient pointers: grad_substrate_idx in [0, NUM_GRAD_SUBSTRATES)
    float* get_device_gradx_ptr(int grad_substrate_idx);
    float* get_device_grady_ptr(int grad_substrate_idx);
    float* get_device_gradz_ptr(int grad_substrate_idx);
    // Recruitment sources (integer bit-flags per voxel)
    int*   get_device_recruitment_sources_ptr();

    // Diagnostics
    float get_total_source(int substrate_idx);
    int   get_total_voxels() const { return config_.nx * config_.ny * config_.nz; }

    // Point query (D2H, for diagnostics only - slow)
    float get_concentration_at_voxel(int x, int y, int z, int substrate_idx) const;

    // Compatibility stubs (unused)
    void set_sources(const float* h_sources, int substrate_idx) {}
    void add_source_at_voxel(int x, int y, int z, int substrate_idx, float value) {}

private:
    PDEConfig config_;

    // --- Device arrays ---
    float* d_conc_;        // Concentrations:   [NUM_SUBSTRATES * V]
    float* d_src_;         // Sources [conc/s]:  [NUM_SUBSTRATES * V]
    float* d_upt_;         // Uptakes [1/s]:     [NUM_SUBSTRATES * V]
    // Gradients: layout [grad_s * 3 * V + dim * V + voxel_idx]
    //   dim 0=x, 1=y, 2=z; grad_s ∈ {GRAD_IFN, GRAD_TGFB, GRAD_CCL2, GRAD_VEGFA}
    float* d_grad_;        // [NUM_GRAD_SUBSTRATES * 3 * V]
    int*   d_recruitment_; // [V]

    // --- Precomputed Thomas coefficients ---
    // Layout per array: [substrate_idx * N + element_idx]
    float* d_thomas_denom_x_; // Modified pivots, x-direction: [NUM_SUBSTRATES * nx]
    float* d_thomas_c_x_;     // Back-sub coefficients, x:     [NUM_SUBSTRATES * nx]
    float* d_thomas_denom_y_; // [NUM_SUBSTRATES * ny]
    float* d_thomas_c_y_;     // [NUM_SUBSTRATES * ny]
    float* d_thomas_denom_z_; // [NUM_SUBSTRATES * nz]
    float* d_thomas_c_z_;     // [NUM_SUBSTRATES * nz]

    // Per-substrate c1 = dt*D/dx² values (for LOD kernel arguments)
    float h_c1_[NUM_SUBSTRATES];

    // Precompute Thomas coefficients on host and upload
    void precompute_thomas_coefficients();
};

} // namespace PDAC

#endif // PDE_SOLVER_CUH
