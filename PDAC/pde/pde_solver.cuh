#ifndef PDE_SOLVER_CUH
#define PDE_SOLVER_CUH

#include <cuda_runtime.h>
#include <vector>
#include <string>

namespace PDAC {

// Chemical substrate indices
// Units: all cytokines/chemokines/MMP/Antibody in nM; O2 in mM.
// Sources: pmol/(cell*s) / voxel_volume [cm³] → nM/s (all substrates except O2)
//          O2 uses implicit Krogh cylinder (source in mM/s via KvLv*C_blood/vol)
// EC50s: nM (matching grid), except O2 thresholds in mM
// Decay/uptake rates: 1/s (unit-independent)
// Diffusivities: cm²/s (unit-independent)
enum ChemicalSubstrate {
    CHEM_O2 = 0,    // [mM] — oxygen, vascular PHALANX source (0.065 mM ≈ 65 µM ≈ 50 mmHg)
    CHEM_IFN,       // [nM] — IFN-gamma
    CHEM_IL2,       // [nM] — IL-2
    CHEM_IL10,      // [nM] — IL-10
    CHEM_TGFB,      // [nM] — TGF-beta
    CHEM_CCL2,      // [nM] — CCL2/MCP-1
    CHEM_ARGI,      // [nM] — Arginase I
    CHEM_NO,        // [nM] — Nitric Oxide
    CHEM_IL12,      // [nM] — IL-12
    CHEM_VEGFA,     // [nM] — VEGF-A
    CHEM_IL1,       // [nM] — IL-1beta
    CHEM_IL6,       // [nM] — IL-6
    CHEM_CXCL13,    // [nM] — CXCL13 (B cell/TLS chemokine)
    CHEM_MMP,       // [nM] — MMP-2/9 (matrix metalloproteinase)
    CHEM_ANTIBODY,  // [nM] — IgG antibody (B cell plasma secretion, ADCC)
    CHEM_CCL21,     // [nM] — CCL21 (mature DC secretion, TLS T-zone homing)
    CHEM_CXCL12,    // [nM] — CXCL12/SDF-1 (iCAF + cancer, T cell exclusion)
    CHEM_CCL5,      // [nM] — CCL5/RANTES (cancer + iCAF, Treg CCR5 recruitment)
    CHEM_CXCL9_10,  // [nM] — CXCL9/10/11 (CXCR3 ligand, CAF-derived; effector CD8/Th recruitment gate)
    NUM_SUBSTRATES
};

// Gradient substrates (subset used for chemotaxis)
// Maps gradient indices to chemical indices for chemotaxis
enum GradientSubstrate {
    GRAD_IFN = 0,
    GRAD_TGFB,
    GRAD_CCL2,
    GRAD_VEGFA,
    GRAD_CXCL13,
    GRAD_CCL21,
    GRAD_CXCL12,
    GRAD_CCL5,
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

    // Solve mode: 0 = transient LOD (solve_timestep ×substeps), 1 = quasi-steady-state (solve_steadystate)
    int solve_mode = 0;
    float cg_tol = 1e-5f;               // Steady-state CG relative-residual tolerance
    int cg_maxiter = 500;              // Steady-state CG iteration cap

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

    // Solve the quasi-steady-state field directly: for each substrate solve
    //   M c = b,   M = (λ + U)·I − D·∇²,   b = S   (screened-Poisson, SPD)
    // via matrix-free Jacobi-preconditioned CG. Replaces the 36-substep transient
    // relaxation for the equilibrating (fast-diffusion) regime. Warm-starts from the
    // current d_conc_ field. D=0 substrates use the closed form c = S/(λ+U).
    void solve_steadystate();

    // Solve the quasi-steady-state field spectrally (FFT/DCT). For the constant-coefficient
    // operator (λ − D∇²) with Neumann BCs, a Discrete Cosine Transform diagonalizes the
    // system, so decay-only substrates are solved EXACTLY in one transform pair (no
    // iteration, correct gradients). Substrates with per-voxel uptake (O2/CCL2/VEGFA) use
    // the DCT solve as a preconditioner in CG. Realized via cuFFT with a mirror (even)
    // extension to 2N per dimension. Same fixed point as solve_steadystate() but far faster.
    void solve_spectral();

    // Diagnostics for the last solve_steadystate() call (per substrate).
    int   get_last_cg_iters(int substrate_idx) const;
    float get_last_cg_residual(int substrate_idx) const;

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
    int   get_solve_mode()   const { return config_.solve_mode; }

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
    //   dim 0=x, 1=y, 2=z; grad_s ∈ GradientSubstrate enum
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

    // --- Steady-state CG workspace (per-voxel scratch, shared across substrates) ---
    float* d_cg_r_   = nullptr;   // residual
    float* d_cg_p_   = nullptr;   // search direction
    float* d_cg_Ap_  = nullptr;   // A·p
    float* d_cg_z_   = nullptr;   // preconditioned residual
    float* d_cg_tmp_ = nullptr;   // temp for p update
    float* d_cg_minv_ = nullptr;  // per-voxel Jacobi preconditioner M^{-1}
    float* d_dot_buf_ = nullptr;  // partial-sum reduction buffer
    int    cg_blocks_ = 0;        // number of reduction blocks
    void allocate_cg_workspace();

    // Solve one substrate's steady-state system in-place on d_C. Returns iterations.
    int solve_steadystate_substrate(float* d_C, const float* d_S, const float* d_U,
                                    float D, float lambda);
    float host_dot(const float* a, const float* b, int n);

    int   last_cg_iters_[NUM_SUBSTRATES]    = {0};
    float last_cg_residual_[NUM_SUBSTRATES] = {0.0f};
    float cg_scratch_residual_ = 0.0f;  // per-substrate residual handed back by the helper

    // --- Spectral steady-state workspace: true 3D DCT-II/III via Makhoul (no inflation) ---
    // Each 1D DCT runs on the CONTIGUOUS axis; a cyclic axis-rotation between passes keeps
    // every transform contiguous-batched (one cuFFT call each — no strides, no per-slab loop).
    int     dct_plan_x_ = -1;         // single-substrate contiguous C2C, len nx, batch ny*nz
    int     dct_plan_y_ = -1;         // single, len ny, batch nx*nz
    int     dct_plan_z_ = -1;         // single, len nz, batch nx*ny
    int     dct_plan_xb_ = -1;        // batched (×n_decay) contiguous C2C, len nx
    int     dct_plan_yb_ = -1;        // batched, len ny
    int     dct_plan_zb_ = -1;        // batched, len nz
    int     dct_plan_x3_ = -1;        // batched (×3 uptake substrates) contiguous C2C, len nx
    int     dct_plan_y3_ = -1;        // batched ×3, len ny
    int     dct_plan_z3_ = -1;        // batched ×3, len nz
    float2* d_dct_a_ = nullptr;       // complex work buffer A [n_decay*V]
    float2* d_dct_b_ = nullptr;       // complex work buffer B [n_decay*V]
    float*  d_dct_mu_ = nullptr;      // Laplacian eigenvalues, natural (z,y,x) layout [V]
    // Decay-only substrates (no per-voxel uptake) solved together in one batched DCT.
    int     n_decay_ = 0;
    int*    d_decay_subs_ = nullptr;  // [n_decay_]
    float*  d_sub_D_ = nullptr;       // [n_decay_]
    float*  d_sub_lambda_ = nullptr;  // [n_decay_]
    // Uptake substrates (O2/CCL2/VEGFA): mean-shifted batched defect-correction workspace.
    int*    d_upt_subs_ = nullptr;    // [3] substrate indices
    float*  d_upt_D_ = nullptr;       // [3] diffusivities
    float*  d_upt_lambda_ = nullptr;  // [3] decay rates λ
    float*  d_upt_ubar_ = nullptr;    // [3] mean uptake Ū, recomputed each step
    float*  d_upt_lamshift_ = nullptr;// [3] (λ + Ū) per substrate, recomputed each step
    bool    fft_ready_ = false;
    void allocate_fft_workspace();
    // One constant-coefficient elliptic solve c = (λ − D∇²)^{-1} b via 3D DCT (single substrate).
    void fft_solve_const(float* d_C, const float* d_b, float D, float lambda);
    // Batched 3D DCT solve for all decay-only substrates (reads d_src_, writes d_conc_, clamped).
    void solve_spectral_decay_batched();
    // Mean-shifted batched defect-correction for the 3 per-voxel-uptake substrates.
    void solve_spectral_uptake_split();
    // 3D forward/inverse DCT-II over `nb` substrate blocks held in d_dct_a_ using plans (px,py,pz).
    void dct3d_forward(int px, int py, int pz, int nb);
    void dct3d_inverse(int px, int py, int pz, int nb);
    void dct_contig(int plan, int n, bool inverse, int V_total);
    void dct_rotate(int na, int nb, int nc, int V_total);
    float host_sum(const float* d_x, int n);  // device-array sum (for mean uptake)
    // FFT-preconditioned CG for substrates with per-voxel uptake. Returns iterations.
    int  solve_spectral_uptake(float* d_C, const float* d_S, const float* d_U,
                               float D, float lambda);
};

} // namespace PDAC

#endif // PDE_SOLVER_CUH
