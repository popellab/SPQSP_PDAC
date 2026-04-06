#include "pde_integration.cuh"
#include "../core/common.cuh"
#include "../core/layer_timing.h"
#include <iostream>
#include <vector>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <nvtx3/nvToolsExt.h>
#include <cmath>

// ============================================================================
// Layer Timing Globals (defined once here, declared extern in layer_timing.h)
// ============================================================================
namespace PDAC {
    std::vector<LayerTime> g_layer_timings;
    ClockPoint g_checkpoint_t = std::chrono::high_resolution_clock::now();
} // namespace PDAC

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
            throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err))); \
        } \
    } while(0)

namespace PDAC {

// ============================================================================
// Global PDE solver instance
// ============================================================================
PDESolver* g_pde_solver = nullptr;
double g_last_pde_ms = 0.0;  // exposed for timing CSV

static RecruitStats g_recruit_stats;

RecruitStats get_last_recruit_stats() { return g_recruit_stats; }

// Flat device array: cancer occupancy per voxel (0 = empty, >0 = cancer present).
// Populated by cancer_write_to_occ_grid, zeroed by zero_occupancy_grid.
// Used by recruitment source-marking kernels to skip tumor-dense voxels.
static unsigned int* d_cancer_occ = nullptr;

// Per-voxel vascular tip_id map.
// Written by vascular_write_to_occ_grid (PHALANX/STALK cells write their tip_id).
// Zeroed by zero_occupancy_grid. Read by vascular_state_step for neighbor check.
// Layout: idx = z*(nx*ny) + y*nx + x. Value 0 = empty; nonzero = tip_id of vessel.
static unsigned int* d_vas_tip_id_grid = nullptr;

// Flat device arrays for ECM (extracellular matrix) and fibroblast density field.
// Replace MacroProperty-based approach to eliminate D2H/H2D copies every step.
// Layout: idx = z * (nx * ny) + y * nx + x  (z-major, x-minor; matches PDE convention)
static float* d_ecm_density = nullptr;
static float* d_ecm_crosslink = nullptr;   // Per-voxel crosslink accumulator (LOX-driven)
static float* d_fib_density_field = nullptr;

// Voxel tissue type labels (static, set once during initialization).
// uint8_t per voxel: VOXEL_STROMA(0), VOXEL_SEPTUM(1), VOXEL_LOBULE(2), VOXEL_TUMOR(3), VOXEL_MARGIN(4)
static uint8_t* d_voxel_type = nullptr;

// Volume-based occupancy: single float per voxel tracking total cell volume (µm³).
// Replaces the old per-type occ_grid MacroProperty + flat arrays (d_t_occ, d_mac_occ, d_mdsc_occ).
static float* d_volume_used = nullptr;

// Antigen grid: persistent per-voxel antigen signal deposited by dying cancer cells.
// DCs and B cells read this to capture antigen (replaces dead-neighbor scanning).
// Decays exponentially each ABM step via decay_antigen_grid host function.
static float* d_antigen_grid = nullptr;

// ECM fiber orientation: per-voxel axis vector (magnitude = alignment strength).
// 0 = isotropic, 1 = fully aligned. Updated by orient update kernel each step.
static float* d_ecm_orient_x = nullptr;
static float* d_ecm_orient_y = nullptr;
static float* d_ecm_orient_z = nullptr;

// Mechanical stress field: transient per-voxel stress from cancer cell movement.
// Accumulated via atomicAdd during cancer_move, decayed each step.
static float* d_mech_stress_x = nullptr;
static float* d_mech_stress_y = nullptr;
static float* d_mech_stress_z = nullptr;

static RecruitRequest* d_recruit_requests = nullptr;
static int* d_recruit_count = nullptr;

// ── Recruitment diagnostics (device-side atomic counters) ──
struct RecruitDiag {
    int t_sources;         // voxels with T source flag
    int mdsc_sources;      // voxels with MDSC source flag
    int mac_sources;       // voxels with MAC source flag
    int teff_roll_pass;    // p_teff rolls that passed
    int teff_place_ok;     // Teff placements that succeeded
    int teff_place_fail;   // Teff placements that failed (cap blocked)
    int treg_roll_pass;
    int treg_place_ok;
    int treg_place_fail;
    int th_roll_pass;
    int th_place_ok;
    int th_place_fail;
    int mdsc_roll_pass;
    int mdsc_place_ok;
    int mdsc_place_fail;
    int mac_roll_pass;
    int mac_place_ok;
    int mac_place_fail;
    int bcell_sources;
    int bcell_roll_pass;
    int bcell_place_ok;
    int bcell_place_fail;
    int dc_sources;
    int dc_roll_pass;
    int dc_place_ok;
    int dc_place_fail;
};
static RecruitDiag* d_recruit_diag = nullptr;

// ============================================================================
// CUDA Kernel: ECM Grid Update
// Applies decay + myCAF deposition + MMP degradation + crosslink accumulation per voxel.
// Called from update_ecm_grid host function after fib_build_density_field runs.
// ============================================================================
__global__ void update_ecm_grid_kernel(
    float* ecm_density, float* ecm_crosslink,
    const float* fib_field, const float* tgfb_conc, const float* mmp_conc,
    int nx, int ny, int nz,
    float voxel_vol_cm3, float dt,
    float k_decay, float k_depo, float density_cap,
    float tgfb_ec50, float ecm_baseline,
    float k_mmp, float alpha_crosslink,
    float k_lox, float yap_ec50)
{
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ty = blockIdx.y * blockDim.y + threadIdx.y;
    const int tz = blockIdx.z * blockDim.z + threadIdx.z;
    if (tx >= nx || ty >= ny || tz >= nz) return;

    const int idx = tz * (nx * ny) + ty * nx + tx;

    float density   = ecm_density[idx];
    float crosslink = ecm_crosslink[idx];
    float fib       = fib_field[idx];
    float tgfb      = tgfb_conc[idx];
    float mmp       = mmp_conc[idx];

    // 1. Exponential baseline decay
    float density_amt = density * voxel_vol_cm3;
    density_amt *= expf(-k_decay * dt);
    density = density_amt / voxel_vol_cm3;

    // 2. myCAF deposition (TGF-β gated, saturation-limited, YAP/TAZ Hill ceiling)
    float H_TGFB = tgfb / (tgfb + tgfb_ec50 + 1e-30f);
    float sat_frac = fminf(density / density_cap, 1.0f);
    // YAP/TAZ ceiling: stiff ECM (high density × crosslink) → feedback limits further deposition
    // Hill function: yap_factor = 1 / (1 + (stiffness/yap_ec50)^2)
    float stiffness = density * (1.0f + crosslink);
    float yap_factor = 1.0f / (1.0f + (stiffness * stiffness) / (yap_ec50 * yap_ec50 + 1e-30f));
    float deposition = fib * (1.0f + H_TGFB) * k_depo / 3.0f * (1.0f - sat_frac) * yap_factor * dt;
    density += deposition / voxel_vol_cm3;

    // 3. MMP degradation (crosslink-resistant)
    float mmp_degrade = k_mmp * mmp * density / (1.0f + alpha_crosslink * crosslink) * dt;
    density -= mmp_degrade;

    // 4. Floor to baseline
    density = fmaxf(density, ecm_baseline);
    density = fminf(density, density_cap);
    ecm_density[idx] = density;

    // 5. Crosslink accumulation (LOX from myCAFs, saturates at 1.0)
    float mycaf_present = (fib > 0.0f) ? 1.0f : 0.0f;
    crosslink += k_lox * (1.0f - crosslink) * mycaf_present * dt;
    crosslink = fminf(crosslink, 1.0f);
    crosslink = fmaxf(crosslink, 0.0f);
    ecm_crosslink[idx] = crosslink;
}

// ============================================================================
// CUDA Kernel: Fill ECM grid with a constant value (initialization only)
// ============================================================================
__global__ void fill_ecm_kernel(float* ecm, int total_voxels, float value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_voxels) ecm[idx] = value;
}

// ============================================================================
// CUDA Kernel: Decay antigen grid (exponential decay each ABM step)
// ============================================================================
__global__ void decay_antigen_kernel(float* antigen, int total_voxels, float decay_factor)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_voxels) {
        antigen[idx] *= decay_factor;
    }
}

// ============================================================================
// CUDA Kernel: Decay mechanical stress field (exponential decay each ABM step)
// ============================================================================
__global__ void decay_stress_kernel(float* stress_x, float* stress_y, float* stress_z,
                                     int total_voxels, float decay_factor)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_voxels) {
        stress_x[idx] *= decay_factor;
        stress_y[idx] *= decay_factor;
        stress_z[idx] *= decay_factor;
    }
}

// ============================================================================
// CUDA Kernel: Update ECM fiber orientation
// Two competing forces: myCAF traction (TACS-2) and mechanical stress (TACS-3).
// Crosslinked ECM resists reorientation.
// ============================================================================
__global__ void update_ecm_orient_kernel(
    float* orient_x, float* orient_y, float* orient_z,
    const float* stress_x, const float* stress_y, const float* stress_z,
    const float* fib_field, const float* ecm_crosslink,
    const float* tgfb_grad_x, const float* tgfb_grad_y, const float* tgfb_grad_z,
    int nx, int ny, int nz,
    float dt, float base_rate,
    float w_traction, float w_stress, float alpha_crosslink)
{
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ty = blockIdx.y * blockDim.y + threadIdx.y;
    const int tz = blockIdx.z * blockDim.z + threadIdx.z;
    if (tx >= nx || ty >= ny || tz >= nz) return;

    const int idx = tz * (nx * ny) + ty * nx + tx;

    float ox = orient_x[idx];
    float oy = orient_y[idx];
    float oz = orient_z[idx];

    // Effective rate: crosslinked ECM resists reorientation
    float crosslink = ecm_crosslink[idx];
    float effective_rate = base_rate / (1.0f + alpha_crosslink * crosslink);

    // Force 1: myCAF traction (TACS-2) — align fibers perpendicular to TGF-β gradient
    float fib = fib_field[idx];
    float fx_trac = 0.0f, fy_trac = 0.0f, fz_trac = 0.0f;
    if (fib > 0.0f) {
        float tg_x = tgfb_grad_x[idx];
        float tg_y = tgfb_grad_y[idx];
        float tg_z = tgfb_grad_z[idx];
        float tg_mag = sqrtf(tg_x * tg_x + tg_y * tg_y + tg_z * tg_z);
        if (tg_mag > 1e-12f) {
            float inv_tg = 1.0f / tg_mag;
            float th_x = tg_x * inv_tg;
            float th_y = tg_y * inv_tg;
            float th_z = tg_z * inv_tg;
            // Remove component of orientation parallel to TGF-β gradient
            float dot_ot = ox * th_x + oy * th_y + oz * th_z;
            float perp_x = ox - dot_ot * th_x;
            float perp_y = oy - dot_ot * th_y;
            float perp_z = oz - dot_ot * th_z;
            // Force toward perpendicular plane
            fx_trac = w_traction * fib * (perp_x - ox);
            fy_trac = w_traction * fib * (perp_y - oy);
            fz_trac = w_traction * fib * (perp_z - oz);
        }
    }

    // Force 2: Mechanical stress (TACS-3) — align fibers with stress direction
    float sx = stress_x[idx];
    float sy = stress_y[idx];
    float sz = stress_z[idx];
    float fx_stress = w_stress * sx;
    float fy_stress = w_stress * sy;
    float fz_stress = w_stress * sz;

    // Combined update
    ox += effective_rate * dt * (fx_trac + fx_stress);
    oy += effective_rate * dt * (fy_trac + fy_stress);
    oz += effective_rate * dt * (fz_trac + fz_stress);

    // Clamp magnitude to [0, 1]
    float mag = sqrtf(ox * ox + oy * oy + oz * oz);
    if (mag > 1.0f) {
        float inv_mag = 1.0f / mag;
        ox *= inv_mag;
        oy *= inv_mag;
        oz *= inv_mag;
    }

    orient_x[idx] = ox;
    orient_y[idx] = oy;
    orient_z[idx] = oz;
}

// ============================================================================
// Host Function: Reset PDE Buffers (call before compute_chemical_sources)
// ============================================================================

FLAMEGPU_HOST_FUNCTION(reset_pde_buffers) {
    nvtxRangePush("Reset PDE Buffers");
    if (!g_pde_solver) { nvtxRangePop(); return; }
    g_pde_solver->reset_sources();
    g_pde_solver->reset_uptakes();
    nvtxRangePop();
}

// ============================================================================
// Host Function: Solve PDE
// ============================================================================

FLAMEGPU_HOST_FUNCTION(solve_pde_step) {
    nvtxRangePush("PDE Solve");
    if (!g_pde_solver) {
        nvtxRangePop();
        return;
    }

    int substeps = FLAMEGPU->environment.getProperty<int>("PARAM_MOLECULAR_STEPS");
    auto pde_t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < substeps; i++) {
        g_pde_solver->solve_timestep();
    }
    auto pde_t1 = std::chrono::high_resolution_clock::now();
    g_last_pde_ms = std::chrono::duration<double, std::milli>(pde_t1 - pde_t0).count();

    unsigned int step = FLAMEGPU->environment.getProperty<unsigned int>("current_step");
    if (step % 50 == 0) {
        std::cout << "PDE solved for step " << step << std::endl;
    }
    nvtxRangePop();
}

// ============================================================================
// Host Function: Compute PDE Gradients (call after solve_pde_step)
// ============================================================================

FLAMEGPU_HOST_FUNCTION(compute_pde_gradients) {
    nvtxRangePush("Compute PDE Gradients");
    if (!g_pde_solver) { nvtxRangePop(); return; }
    g_pde_solver->compute_gradients();
    nvtxRangePop();
}

// ============================================================================
// Initialize/Cleanup
// ============================================================================

void initialize_pde_solver(int grid_x, int grid_y, int grid_z,
                           float voxel_size, float dt_abm, int molecular_steps,
                            const PDAC::GPUParam& gpu_params,
                            flamegpu::ModelDescription& model) {
    PDEConfig config;
    config.nx = grid_x;
    config.ny = grid_y;
    config.nz = grid_z;
    config.num_substrates = NUM_SUBSTRATES;
    config.voxel_size = voxel_size * 1.0e-4f;  // Convert µm to cm
    config.dt_abm = dt_abm;
    config.dt_pde = dt_abm / molecular_steps;
    config.substeps_per_abm = molecular_steps;
    config.boundary_type = 0;  // Neumann (no-flux)
    
    // Set diffusion coefficients (cm²/s) from params file
    config.diffusion_coeffs[CHEM_O2]    = gpu_params.getFloat(PARAM_O2_DIFFUSIVITY);
    config.diffusion_coeffs[CHEM_IFN]   = gpu_params.getFloat(PARAM_IFNG_DIFFUSIVITY);
    config.diffusion_coeffs[CHEM_IL2]   = gpu_params.getFloat(PARAM_IL2_DIFFUSIVITY);
    config.diffusion_coeffs[CHEM_IL10]  = gpu_params.getFloat(PARAM_IL10_DIFFUSIVITY);
    config.diffusion_coeffs[CHEM_TGFB]  = gpu_params.getFloat(PARAM_TGFB_DIFFUSIVITY);
    config.diffusion_coeffs[CHEM_CCL2]  = gpu_params.getFloat(PARAM_CCL2_DIFFUSIVITY);
    config.diffusion_coeffs[CHEM_ARGI]  = gpu_params.getFloat(PARAM_ARGI_DIFFUSIVITY);
    config.diffusion_coeffs[CHEM_NO]    = gpu_params.getFloat(PARAM_NO_DIFFUSIVITY);
    config.diffusion_coeffs[CHEM_IL12]  = gpu_params.getFloat(PARAM_IL12_DIFFUSIVITY);
    config.diffusion_coeffs[CHEM_VEGFA] = gpu_params.getFloat(PARAM_VEGFA_DIFFUSIVITY);
    config.diffusion_coeffs[CHEM_IL1]   = gpu_params.getFloat(PARAM_IL1_DIFFUSIVITY);
    config.diffusion_coeffs[CHEM_IL6]   = gpu_params.getFloat(PARAM_IL6_DIFFUSIVITY);
    config.diffusion_coeffs[CHEM_CXCL13]= gpu_params.getFloat(PARAM_CXCL13_DIFFUSIVITY);
    config.diffusion_coeffs[CHEM_MMP]   = gpu_params.getFloat(PARAM_MMP_DIFFUSIVITY);
    config.diffusion_coeffs[CHEM_ANTIBODY] = gpu_params.getFloat(PARAM_ANTIBODY_DIFFUSIVITY);
    config.diffusion_coeffs[CHEM_CCL21]    = gpu_params.getFloat(PARAM_CCL21_DIFFUSIVITY);
    config.diffusion_coeffs[CHEM_CXCL12]   = gpu_params.getFloat(PARAM_CXCL12_DIFFUSIVITY);
    config.diffusion_coeffs[CHEM_CCL5]     = gpu_params.getFloat(PARAM_CCL5_DIFFUSIVITY);

    // Set decay rates (1/s) — QSP-derived read from model env, ABM-only from gpu_params
    const auto env = model.Environment();
    config.decay_rates[CHEM_O2]       = gpu_params.getFloat(PARAM_O2_DECAY_RATE);     // ABM-only (no QSP equivalent)
    config.decay_rates[CHEM_IFN]      = env.getProperty<float>("PARAM_IFNG_DECAY_RATE");
    config.decay_rates[CHEM_IL2]      = env.getProperty<float>("PARAM_IL2_DECAY_RATE");
    config.decay_rates[CHEM_IL10]     = env.getProperty<float>("PARAM_IL10_DECAY_RATE");
    config.decay_rates[CHEM_TGFB]     = env.getProperty<float>("PARAM_TGFB_DECAY_RATE");
    config.decay_rates[CHEM_CCL2]     = env.getProperty<float>("PARAM_CCL2_DECAY_RATE");
    config.decay_rates[CHEM_ARGI]     = env.getProperty<float>("PARAM_ARGI_DECAY_RATE");
    config.decay_rates[CHEM_NO]       = env.getProperty<float>("PARAM_NO_DECAY_RATE");
    config.decay_rates[CHEM_IL12]     = env.getProperty<float>("PARAM_IL12_DECAY_RATE");
    config.decay_rates[CHEM_VEGFA]    = gpu_params.getFloat(PARAM_VEGFA_DECAY_RATE);  // ABM-only
    config.decay_rates[CHEM_IL1]      = env.getProperty<float>("PARAM_IL1_DECAY_RATE");
    config.decay_rates[CHEM_IL6]      = env.getProperty<float>("PARAM_IL6_DECAY_RATE");
    config.decay_rates[CHEM_CXCL13]   = gpu_params.getFloat(PARAM_CXCL13_DECAY_RATE); // ABM-only
    config.decay_rates[CHEM_MMP]      = gpu_params.getFloat(PARAM_MMP_DECAY_RATE);     // ABM-only
    config.decay_rates[CHEM_ANTIBODY] = gpu_params.getFloat(PARAM_ANTIBODY_DECAY_RATE);// ABM-only
    config.decay_rates[CHEM_CCL21]    = gpu_params.getFloat(PARAM_CCL21_DECAY_RATE);   // ABM-only
    config.decay_rates[CHEM_CXCL12]   = env.getProperty<float>("PARAM_CXCL12_DECAY_RATE");
    config.decay_rates[CHEM_CCL5]     = env.getProperty<float>("PARAM_CCL5_DECAY_RATE");

    g_pde_solver = new PDESolver(config);
    g_pde_solver->initialize();

    // Set initial O2 = 0 so vessels actively source from step 1.
    // Old value 0.673 > C_blood (0.51) caused vessels to be silent initially,
    // crashing O2 to near-zero under cancer uptake before vessels could respond.
    g_pde_solver->set_initial_concentration(CHEM_O2, 0.0f);

    // Allocate flat cancer occupancy array for recruitment density checks
    int total_voxels = g_pde_solver->get_total_voxels();
    CUDA_CHECK(cudaMalloc(&d_cancer_occ, total_voxels * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_cancer_occ, 0, total_voxels * sizeof(unsigned int)));

    // Allocate vascular tip_id grid for efficient neighbor check in vascular_state_step
    CUDA_CHECK(cudaMalloc(&d_vas_tip_id_grid, total_voxels * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_vas_tip_id_grid, 0, total_voxels * sizeof(unsigned int)));

    // Allocate ECM and fibroblast density field device arrays.
    // ECM starts at 0 here; initialize_ecm_to_saturation() is called after QSP init
    // (in set_internal_params) to fill with PARAM_FIB_ECM_SATURATION — matching HCC.
    // Allocate voxel type grid (domain initialization labels)
    CUDA_CHECK(cudaMalloc(&d_voxel_type, total_voxels * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemset(d_voxel_type, 0, total_voxels * sizeof(uint8_t)));  // Default VOXEL_STROMA=0

    CUDA_CHECK(cudaMalloc(&d_ecm_density, total_voxels * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_ecm_density, 0, total_voxels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ecm_crosslink, total_voxels * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_ecm_crosslink, 0, total_voxels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fib_density_field, total_voxels * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_fib_density_field, 0, total_voxels * sizeof(float)));

    // Allocate volume-based occupancy grid
    CUDA_CHECK(cudaMalloc(&d_volume_used, total_voxels * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_volume_used, 0, total_voxels * sizeof(float)));

    // Allocate antigen grid (persistent per-voxel antigen from dying cancer)
    CUDA_CHECK(cudaMalloc(&d_antigen_grid, total_voxels * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_antigen_grid, 0, total_voxels * sizeof(float)));

    // Allocate ECM fiber orientation arrays (per-voxel axis vector, init isotropic = 0)
    CUDA_CHECK(cudaMalloc(&d_ecm_orient_x, total_voxels * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_ecm_orient_x, 0, total_voxels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ecm_orient_y, total_voxels * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_ecm_orient_y, 0, total_voxels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ecm_orient_z, total_voxels * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_ecm_orient_z, 0, total_voxels * sizeof(float)));

    // Allocate mechanical stress field arrays (transient, init = 0)
    CUDA_CHECK(cudaMalloc(&d_mech_stress_x, total_voxels * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_mech_stress_x, 0, total_voxels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mech_stress_y, total_voxels * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_mech_stress_y, 0, total_voxels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mech_stress_z, total_voxels * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_mech_stress_z, 0, total_voxels * sizeof(float)));

    // Allocate GPU recruitment output buffer + atomic counter
    CUDA_CHECK(cudaMalloc(&d_recruit_requests, MAX_RECRUITS_PER_STEP * sizeof(RecruitRequest)));
    CUDA_CHECK(cudaMalloc(&d_recruit_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_recruit_count, 0, sizeof(int)));

    // Recruitment diagnostics buffer
    CUDA_CHECK(cudaMalloc(&d_recruit_diag, sizeof(RecruitDiag)));
    CUDA_CHECK(cudaMemset(d_recruit_diag, 0, sizeof(RecruitDiag)));

    std::cout << "PDE Solver initialized and coupled to FLAME GPU 2" << std::endl;
}

// Call this after model initialization but before simulation starts
void set_pde_pointers_in_environment(flamegpu::ModelDescription& model) {
    if (!g_pde_solver) {
        std::cerr << "[ERROR] g_pde_solver is NULL!" << std::endl;
        return;
    }



    // Store device pointers as unsigned long long (can be cast back to float*)
    for (int sub = 0; sub < NUM_SUBSTRATES; sub++) {
        std::string concentration_key = "pde_concentration_ptr_" + std::to_string(sub);
        std::string source_key = "pde_source_ptr_" + std::to_string(sub);
        std::string uptake_key = "pde_uptake_ptr_" + std::to_string(sub);

        uintptr_t conc_ptr = reinterpret_cast<uintptr_t>(g_pde_solver->get_device_concentration_ptr(sub));
        uintptr_t src_ptr  = reinterpret_cast<uintptr_t>(g_pde_solver->get_device_source_ptr(sub));
        uintptr_t upt_ptr  = reinterpret_cast<uintptr_t>(g_pde_solver->get_device_uptake_ptr(sub));

        model.Environment().newProperty<unsigned long long>(concentration_key, static_cast<unsigned long long>(conc_ptr));
        model.Environment().newProperty<unsigned long long>(source_key,        static_cast<unsigned long long>(src_ptr));
        model.Environment().newProperty<unsigned long long>(uptake_key,        static_cast<unsigned long long>(upt_ptr));
    }

    // Store gradient pointers for chemotaxis substrates (IFN=0, TGFB=1, CCL2=2, VEGFA=3)
    // Naming: pde_grad_IFN_x, pde_grad_IFN_y, pde_grad_IFN_z, pde_grad_TGFB_x, ...
    static const char* grad_names[NUM_GRAD_SUBSTRATES] = {"IFN", "TGFB", "CCL2", "VEGFA", "CXCL13", "CCL21"};
    for (int g = 0; g < NUM_GRAD_SUBSTRATES; g++) {
        model.Environment().newProperty<unsigned long long>(
            std::string("pde_grad_") + grad_names[g] + "_x",
            static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(g_pde_solver->get_device_gradx_ptr(g))));
        model.Environment().newProperty<unsigned long long>(
            std::string("pde_grad_") + grad_names[g] + "_y",
            static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(g_pde_solver->get_device_grady_ptr(g))));
        model.Environment().newProperty<unsigned long long>(
            std::string("pde_grad_") + grad_names[g] + "_z",
            static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(g_pde_solver->get_device_gradz_ptr(g))));
    }

    // Store recruitment sources pointer
    uintptr_t recruit_ptr = reinterpret_cast<uintptr_t>(g_pde_solver->get_device_recruitment_sources_ptr());
    model.Environment().newProperty<unsigned long long>("pde_recruitment_sources_ptr", static_cast<unsigned long long>(recruit_ptr));

    // Store flat cancer occupancy pointer (for recruitment density checks)
    model.Environment().newProperty<unsigned long long>("cancer_occ_ptr",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(d_cancer_occ)));

    // Store vascular tip_id grid pointer (for efficient neighbor check in vascular_state_step)
    model.Environment().newProperty<unsigned long long>("vas_tip_id_grid_ptr",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(d_vas_tip_id_grid)));

    // Store voxel type grid pointer (domain initialization labels)
    model.Environment().newProperty<unsigned long long>("voxel_type_ptr",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(d_voxel_type)));

    // Store ECM and fibroblast density field pointers (replace MacroProperty approach)
    model.Environment().newProperty<unsigned long long>("ecm_density_ptr",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(d_ecm_density)));
    model.Environment().newProperty<unsigned long long>("ecm_crosslink_ptr",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(d_ecm_crosslink)));
    model.Environment().newProperty<unsigned long long>("fib_density_field_ptr",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(d_fib_density_field)));

    // Volume-based occupancy grid pointer
    model.Environment().newProperty<unsigned long long>("volume_used_ptr",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(d_volume_used)));

    // Antigen grid pointer (persistent antigen from dying cancer cells)
    model.Environment().newProperty<unsigned long long>("antigen_grid_ptr",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(d_antigen_grid)));

    // ECM fiber orientation pointers
    model.Environment().newProperty<unsigned long long>("ecm_orient_x_ptr",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(d_ecm_orient_x)));
    model.Environment().newProperty<unsigned long long>("ecm_orient_y_ptr",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(d_ecm_orient_y)));
    model.Environment().newProperty<unsigned long long>("ecm_orient_z_ptr",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(d_ecm_orient_z)));

    // Mechanical stress field pointers
    model.Environment().newProperty<unsigned long long>("stress_x_ptr",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(d_mech_stress_x)));
    model.Environment().newProperty<unsigned long long>("stress_y_ptr",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(d_mech_stress_y)));
    model.Environment().newProperty<unsigned long long>("stress_z_ptr",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(d_mech_stress_z)));

    std::cout << "PDE device pointers stored in FLAME GPU environment" << std::endl;
}

void run_pde_warmup(int substeps) {
    if (!g_pde_solver || substeps <= 0) return;
    std::cout << "  Running PDE warmup (" << substeps << " substeps)..." << std::endl;
    g_pde_solver->reset_sources();
    g_pde_solver->reset_uptakes();
    for (int i = 0; i < substeps; i++) {
        g_pde_solver->solve_timestep();
    }
    g_pde_solver->compute_gradients();
    std::cout << "  PDE warmup complete" << std::endl;
}

void cleanup_pde_solver() {
    if (g_pde_solver) {
        delete g_pde_solver;
        g_pde_solver = nullptr;
    }
    if (d_cancer_occ) {
        cudaFree(d_cancer_occ);
        d_cancer_occ = nullptr;
    }
    if (d_voxel_type) {
        cudaFree(d_voxel_type);
        d_voxel_type = nullptr;
    }
    if (d_ecm_density) {
        cudaFree(d_ecm_density);
        d_ecm_density = nullptr;
    }
    if (d_ecm_crosslink) {
        cudaFree(d_ecm_crosslink);
        d_ecm_crosslink = nullptr;
    }
    if (d_fib_density_field) {
        cudaFree(d_fib_density_field);
        d_fib_density_field = nullptr;
    }
    if (d_vas_tip_id_grid) {
        cudaFree(d_vas_tip_id_grid);
        d_vas_tip_id_grid = nullptr;
    }
    if (d_volume_used) { cudaFree(d_volume_used); d_volume_used = nullptr; }
    if (d_antigen_grid) { cudaFree(d_antigen_grid); d_antigen_grid = nullptr; }
    if (d_ecm_orient_x) { cudaFree(d_ecm_orient_x); d_ecm_orient_x = nullptr; }
    if (d_ecm_orient_y) { cudaFree(d_ecm_orient_y); d_ecm_orient_y = nullptr; }
    if (d_ecm_orient_z) { cudaFree(d_ecm_orient_z); d_ecm_orient_z = nullptr; }
    if (d_mech_stress_x) { cudaFree(d_mech_stress_x); d_mech_stress_x = nullptr; }
    if (d_mech_stress_y) { cudaFree(d_mech_stress_y); d_mech_stress_y = nullptr; }
    if (d_mech_stress_z) { cudaFree(d_mech_stress_z); d_mech_stress_z = nullptr; }
    if (d_recruit_requests) { cudaFree(d_recruit_requests); d_recruit_requests = nullptr; }
    if (d_recruit_count)    { cudaFree(d_recruit_count);    d_recruit_count = nullptr; }
    if (d_recruit_diag)     { cudaFree(d_recruit_diag);     d_recruit_diag = nullptr; }
}

void initialize_ecm_to_saturation(float ecm_saturation) {
    if (!d_ecm_density || !g_pde_solver) return;
    int total_voxels = g_pde_solver->get_total_voxels();
    int block_size = 256;
    int grid_size = (total_voxels + block_size - 1) / block_size;
    fill_ecm_kernel<<<grid_size, block_size>>>(d_ecm_density, total_voxels, ecm_saturation);
    CUDA_CHECK(cudaDeviceSynchronize());

}

uint8_t* get_voxel_type_device_ptr() { return d_voxel_type; }

void set_voxel_type_from_host(const uint8_t* host_data, int total_voxels) {
    CUDA_CHECK(cudaMemcpy(d_voxel_type, host_data, total_voxels * sizeof(uint8_t), cudaMemcpyHostToDevice));
}

void set_ecm_density_from_host(const float* host_data, int total_voxels) {
    CUDA_CHECK(cudaMemcpy(d_ecm_density, host_data, total_voxels * sizeof(float), cudaMemcpyHostToDevice));
}

void set_ecm_crosslink_from_host(const float* host_data, int total_voxels) {
    CUDA_CHECK(cudaMemcpy(d_ecm_crosslink, host_data, total_voxels * sizeof(float), cudaMemcpyHostToDevice));
}
float* get_ecm_density_device_ptr() { return d_ecm_density; }
float* get_ecm_crosslink_device_ptr() { return d_ecm_crosslink; }
float* get_fib_density_field_device_ptr() { return d_fib_density_field; }
unsigned int* get_vas_tip_id_grid_device_ptr() { return d_vas_tip_id_grid; }
float* get_ecm_orient_x_device_ptr() { return d_ecm_orient_x; }
float* get_ecm_orient_y_device_ptr() { return d_ecm_orient_y; }
float* get_ecm_orient_z_device_ptr() { return d_ecm_orient_z; }

// ============================================================================
// Recruitment System Implementation
// ============================================================================

// Helper: check if radius-3 BOX around (x,y,z) is completely filled with cancer.
// Returns true if every in-bounds voxel within 7x7x7 cube has cancer present.
// Matches HCC: 7x7x7 window_counts_inplace with local_cancer_ratio < 1 gate.
__device__ bool is_tumor_dense_r3(
    const unsigned int* d_cancer_occ,
    int x, int y, int z,
    int nx, int ny, int nz)
{
    int box_total = 0, box_cancer = 0;
    for (int dz = -3; dz <= 3; dz++) {
        for (int dy = -3; dy <= 3; dy++) {
            for (int dx = -3; dx <= 3; dx++) {
                int cx = x + dx, cy = y + dy, cz = z + dz;
                if (cx < 0 || cx >= nx || cy < 0 || cy >= ny || cz < 0 || cz >= nz) continue;
                box_total++;
                box_cancer += (d_cancer_occ[cz*(nx*ny) + cy*nx + cx] > 0u) ? 1 : 0;
            }
        }
    }
    return (box_total > 0 && box_cancer >= box_total);
}

// Source marking for MDSC, MAC, B cell, DC now handled by vascular_mark_sources
// agent function (vascular_cell.cuh) — all immune cells extravasate from vasculature.

// Update vasculature count env property (used by vascular_mark_sources device function)
FLAMEGPU_HOST_FUNCTION(update_vasculature_count) {
    nvtxRangePush("Update Vas Count");
    int n_vas = static_cast<int>(FLAMEGPU->agent(AGENT_VASCULAR).count());
    FLAMEGPU->environment.setProperty<int>("n_vasculature_total", std::max(1, n_vas));
    nvtxRangePop();
}

// Reset recruitment sources at start of each step
FLAMEGPU_HOST_FUNCTION(reset_recruitment_sources) {
    nvtxRangePush("Reset Recruit Sources");
    if (!g_pde_solver) { nvtxRangePop(); return; }
    g_pde_solver->reset_recruitment_sources();
    nvtxRangePop();
}


// ============================================================================
// GPU Recruitment Kernel: Packed parameters struct
// ============================================================================
struct RecruitKernelParams {
    int nx, ny, nz;
    // T cell (CD8) recruitment
    float p_teff, p_treg, p_th;
    // nr_t_voxel/nr_t_voxel_c removed — volume-based occupancy replaces per-type caps
    float t_life_mean, t_life_sd;
    int t_divide_cd, t_divide_limit;
    float t_IL2_release;
    // TCD4 (TReg/TH) recruitment
    float tcd4_life_mean, tcd4_life_sd;
    int tcd4_divide_cd, tcd4_divide_limit;
    float tcd4_TGFB_release;
    float ctla4_treg;
    // MDSC recruitment
    float p_mdsc;
    float mdsc_life_mean;
    const float* il6_conc;    // IL-6 PDE concentration (for MDSC recruitment boost)
    float il6_ec50_mdsc;      // IL-6 EC50 for MDSC recruitment [nM]
    // MAC recruitment
    float p_mac;
    float mac_life_mean;
    // B cell recruitment
    float p_bcell;
    float bcell_life_mean, bcell_life_sd;
    // DC recruitment (per-subtype, homeostatic)
    float p_dc_cdc1;
    float p_dc_cdc2;
    float dc_life_immature_mean, dc_life_immature_sd;
    float dc_life_mature_mean, dc_life_mature_sd;
    float vol_dc;
    // RNG seed
    unsigned int seed;
    // CXCL12 T cell exclusion + CCL5 Treg recruitment
    const float* cxcl12_conc;
    float cxcl12_ec50;
    const float* ccl5_conc;
    float ccl5_ec50;
    float ccl5_ratio;  // k_CCR5_Treg_rec / q_Treg_T_in — scales CCL5 boost relative to base
    // Volume-based occupancy
    float voxel_capacity;
    float vol_teff, vol_treg, vol_th;
    float vol_mdsc, vol_mac_m1, vol_mac_m2;
    float vol_bcell;
};

// Device helper: xorshift32 RNG (fast, good enough for recruitment)
__device__ __forceinline__ unsigned int xorshift32(unsigned int& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

// Device helper: uniform float in [0, 1) from RNG state
__device__ __forceinline__ float rng_uniform(unsigned int& state) {
    return (xorshift32(state) & 0x7FFFFFFFu) / 2147483648.0f;
}

// Device helper: Box-Muller normal sample (mean + sd * N(0,1)), clamped to >= 1
__device__ __forceinline__ int sample_normal_life_gpu(float mean, float sd, unsigned int& rng) {
    float u1 = rng_uniform(rng) + 1e-10f;  // Avoid log(0)
    float u2 = rng_uniform(rng);
    float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
    int life = __float2int_rn(mean + z * sd);
    return life > 0 ? life : 1;
}

// Device helper: exponential life sample (mean * log(1/u)), clamped to >= 1
__device__ __forceinline__ int sample_exp_life_gpu(float mean, unsigned int& rng) {
    float u = rng_uniform(rng) + 1e-4f;  // Avoid log(0)
    int life = __float2int_rn(mean * logf(1.0f / u));
    return life > 0 ? life : 1;
}

// Device helper: try to place a cell at one of the 26 Moore neighbors of (sx, sy, sz).
// Uses Fisher-Yates shuffle with thread-local RNG. Claims voxel via volume-based
// occupancy (atomicAdd + capacity check + undo on overflow).
// Returns true + sets (out_x, out_y, out_z) on success.
__device__ bool try_find_open_neighbor(
    int sx, int sy, int sz,
    int nx, int ny, int nz,
    float* d_vol_used,
    float cell_volume,
    float voxel_capacity,
    unsigned int& rng,
    int& out_x, int& out_y, int& out_z)
{
    // Build 26 Moore neighbor offsets
    int offsets[26][3];
    int n = 0;
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                offsets[n][0] = dx; offsets[n][1] = dy; offsets[n][2] = dz;
                n++;
            }
        }
    }

    // Fisher-Yates shuffle
    for (int i = n - 1; i > 0; i--) {
        int j = xorshift32(rng) % (i + 1);
        int tmp0 = offsets[i][0]; offsets[i][0] = offsets[j][0]; offsets[j][0] = tmp0;
        int tmp1 = offsets[i][1]; offsets[i][1] = offsets[j][1]; offsets[j][1] = tmp1;
        int tmp2 = offsets[i][2]; offsets[i][2] = offsets[j][2]; offsets[j][2] = tmp2;
    }

    for (int i = 0; i < n; i++) {
        int cx = sx + offsets[i][0];
        int cy = sy + offsets[i][1];
        int cz = sz + offsets[i][2];
        if (cx < 0 || cx >= nx || cy < 0 || cy >= ny || cz < 0 || cz >= nz) continue;

        int vidx = cz * (nx * ny) + cy * nx + cx;

        // Volume-based occupancy check (primary gate)
        if (d_vol_used[vidx] + cell_volume > voxel_capacity) continue;

        // Atomic volume claim
        float old_vol = atomicAdd(&d_vol_used[vidx], cell_volume);
        if (old_vol + cell_volume > voxel_capacity) {
            atomicAdd(&d_vol_used[vidx], -cell_volume);  // undo
            continue;
        }

        out_x = cx; out_y = cy; out_z = cz;
        return true;
    }
    return false;
}

// ============================================================================
// GPU Recruitment Kernel: One thread per voxel. Checks recruitment source flags,
// rolls probabilities, finds open neighbors, writes compact RecruitRequest buffer.
// ============================================================================
__global__ void recruit_all_kernel(
    const int* __restrict__ d_recruitment_sources,
    const unsigned int* __restrict__ d_cancer_occ,
    float* d_vol_used,
    RecruitRequest* d_requests,
    int* d_request_count,
    RecruitDiag* diag,
    RecruitKernelParams p)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= p.nx || y >= p.ny || z >= p.nz) return;

    int idx = z * (p.nx * p.ny) + y * p.nx + x;
    int flags = d_recruitment_sources[idx];
    if (flags == 0) return;  // No sources marked here

    // Thread-local RNG seeded from position + step seed
    unsigned int rng = p.seed ^ (idx * 2654435761u + 1);
    xorshift32(rng);  // Warm up

    int px, py, pz;

    // ── T source (bit 0): recruit Teff, TReg, TH ──
    if (flags & 1) {
        atomicAdd(&diag->t_sources, 1);

        // Local CXCL12 exclusion: reduces Teff and TH recruitment
        float cxcl12_local = (p.cxcl12_conc && p.cxcl12_ec50 > 0.0f) ? p.cxcl12_conc[idx] : 0.0f;
        float cxcl12_inhib = (p.cxcl12_ec50 > 0.0f) ? cxcl12_local / (cxcl12_local + p.cxcl12_ec50) : 0.0f;
        // Local CCL5 boost: enhances Treg CCR5 recruitment
        float ccl5_local = (p.ccl5_conc && p.ccl5_ec50 > 0.0f) ? p.ccl5_conc[idx] : 0.0f;
        float ccl5_boost = (p.ccl5_ec50 > 0.0f) ? ccl5_local / (ccl5_local + p.ccl5_ec50) : 0.0f;

        // Try Teff (reduced by CXCL12)
        if (rng_uniform(rng) < p.p_teff * (1.0f - cxcl12_inhib)) {
            atomicAdd(&diag->teff_roll_pass, 1);
            if (try_find_open_neighbor(x, y, z, p.nx, p.ny, p.nz,
                    d_vol_used, p.vol_teff, p.voxel_capacity,
                    rng, px, py, pz)) {
                atomicAdd(&diag->teff_place_ok, 1);
                int slot = atomicAdd(d_request_count, 1);
                if (slot < MAX_RECRUITS_PER_STEP) {
                    RecruitRequest req;
                    req.x = px; req.y = py; req.z = pz;
                    req.cell_type = CELL_TYPE_T;
                    req.cell_state = T_CELL_EFF;
                    req.life = sample_normal_life_gpu(p.t_life_mean, p.t_life_sd, rng);
                    req.divide_cd = p.t_divide_cd;
                    req.divide_limit = p.t_divide_limit;
                    req.IL2_release_remain = p.t_IL2_release;
                    req.TGFB_release_remain = 0.0f;
                    req.CTLA4 = 0.0f;
                    d_requests[slot] = req;
                }
            } else {
                atomicAdd(&diag->teff_place_fail, 1);
            }
        }

        // Try TReg (boosted by CCL5 via CCR5, ratio = k_CCR5/q_Treg)
        if (rng_uniform(rng) < fminf(p.p_treg * (1.0f + p.ccl5_ratio * ccl5_boost), 1.0f)) {
            atomicAdd(&diag->treg_roll_pass, 1);
            if (try_find_open_neighbor(x, y, z, p.nx, p.ny, p.nz,
                    d_vol_used, p.vol_treg, p.voxel_capacity,
                    rng, px, py, pz)) {
                atomicAdd(&diag->treg_place_ok, 1);
                int slot = atomicAdd(d_request_count, 1);
                if (slot < MAX_RECRUITS_PER_STEP) {
                    RecruitRequest req;
                    req.x = px; req.y = py; req.z = pz;
                    req.cell_type = CELL_TYPE_TREG;
                    req.cell_state = TCD4_TREG;
                    req.life = sample_normal_life_gpu(p.tcd4_life_mean, p.tcd4_life_sd, rng);
                    req.divide_cd = p.tcd4_divide_cd;
                    req.divide_limit = p.tcd4_divide_limit;
                    req.IL2_release_remain = 0.0f;
                    req.TGFB_release_remain = p.tcd4_TGFB_release;
                    req.CTLA4 = p.ctla4_treg;
                    d_requests[slot] = req;
                }
            } else {
                atomicAdd(&diag->treg_place_fail, 1);
            }
        }

        // Try TH (reduced by CXCL12, same as Teff)
        if (rng_uniform(rng) < p.p_th * (1.0f - cxcl12_inhib)) {
            atomicAdd(&diag->th_roll_pass, 1);
            if (try_find_open_neighbor(x, y, z, p.nx, p.ny, p.nz,
                    d_vol_used, p.vol_th, p.voxel_capacity,
                    rng, px, py, pz)) {
                atomicAdd(&diag->th_place_ok, 1);
                int slot = atomicAdd(d_request_count, 1);
                if (slot < MAX_RECRUITS_PER_STEP) {
                    RecruitRequest req;
                    req.x = px; req.y = py; req.z = pz;
                    req.cell_type = CELL_TYPE_TREG;  // TH uses TReg agent type
                    req.cell_state = TCD4_TH;
                    req.life = sample_normal_life_gpu(p.tcd4_life_mean, p.tcd4_life_sd, rng);
                    req.divide_cd = p.tcd4_divide_cd;
                    req.divide_limit = p.tcd4_divide_limit;
                    req.IL2_release_remain = 0.0f;
                    req.TGFB_release_remain = p.tcd4_TGFB_release;
                    req.CTLA4 = 0.0f;
                    d_requests[slot] = req;
                }
            } else {
                atomicAdd(&diag->th_place_fail, 1);
            }
        }
    }

    // ── MDSC source (bit 1) ──
    if (flags & 2) {
        atomicAdd(&diag->mdsc_sources, 1);
        // ODE: k_MDSC_rec * V_T * H_CCL2 * (1 + H_IL6_MDSC)
        // CCL2 gating already done in vascular_mark_sources; apply IL-6 boost here
        float p_mdsc_eff = p.p_mdsc;
        if (p.il6_conc) {
            float il6 = p.il6_conc[idx];
            float H_IL6 = il6 / (il6 + p.il6_ec50_mdsc + 1e-30f);
            p_mdsc_eff *= (1.0f + H_IL6);
        }
        if (rng_uniform(rng) < p_mdsc_eff) {
            atomicAdd(&diag->mdsc_roll_pass, 1);
            if (try_find_open_neighbor(x, y, z, p.nx, p.ny, p.nz,
                    d_vol_used, p.vol_mdsc, p.voxel_capacity,
                    rng, px, py, pz)) {
                atomicAdd(&diag->mdsc_place_ok, 1);
                int slot = atomicAdd(d_request_count, 1);
                if (slot < MAX_RECRUITS_PER_STEP) {
                    RecruitRequest req;
                    req.x = px; req.y = py; req.z = pz;
                    req.cell_type = CELL_TYPE_MDSC;
                    req.cell_state = 0;
                    req.life = sample_exp_life_gpu(p.mdsc_life_mean, rng);
                    req.divide_cd = 0;
                    req.divide_limit = 0;
                    req.IL2_release_remain = 0.0f;
                    req.TGFB_release_remain = 0.0f;
                    req.CTLA4 = 0.0f;
                    d_requests[slot] = req;
                }
            } else {
                atomicAdd(&diag->mdsc_place_fail, 1);
            }
        }
    }

    // ── MAC source (bit 2) ──
    if (flags & 4) {
        atomicAdd(&diag->mac_sources, 1);
        if (rng_uniform(rng) < p.p_mac) {
            atomicAdd(&diag->mac_roll_pass, 1);
            if (try_find_open_neighbor(x, y, z, p.nx, p.ny, p.nz,
                    d_vol_used, p.vol_mac_m1, p.voxel_capacity,
                    rng, px, py, pz)) {
                atomicAdd(&diag->mac_place_ok, 1);
                int slot = atomicAdd(d_request_count, 1);
                if (slot < MAX_RECRUITS_PER_STEP) {
                    RecruitRequest req;
                    req.x = px; req.y = py; req.z = pz;
                    req.cell_type = CELL_TYPE_MAC;
                    // ODE recruits all macrophages as M1; polarization handled by M1↔M2 transitions
                    req.cell_state = MAC_M1;
                    req.life = sample_exp_life_gpu(p.mac_life_mean, rng);
                    req.divide_cd = 0;
                    req.divide_limit = 0;
                    req.IL2_release_remain = 0.0f;
                    req.TGFB_release_remain = 0.0f;
                    req.CTLA4 = 0.0f;
                    d_requests[slot] = req;
                }
            } else {
                atomicAdd(&diag->mac_place_fail, 1);
            }
        }
    }

    // ── BCell source (bit 3) ──
    if (flags & 8) {
        atomicAdd(&diag->bcell_sources, 1);
        if (rng_uniform(rng) < p.p_bcell) {
            atomicAdd(&diag->bcell_roll_pass, 1);
            if (try_find_open_neighbor(x, y, z, p.nx, p.ny, p.nz,
                    d_vol_used, p.vol_bcell, p.voxel_capacity,
                    rng, px, py, pz)) {
                atomicAdd(&diag->bcell_place_ok, 1);
                int slot = atomicAdd(d_request_count, 1);
                if (slot < MAX_RECRUITS_PER_STEP) {
                    RecruitRequest req;
                    req.x = px; req.y = py; req.z = pz;
                    req.cell_type = CELL_TYPE_BCELL;
                    req.cell_state = BCELL_NAIVE;
                    req.life = sample_normal_life_gpu(p.bcell_life_mean, p.bcell_life_sd, rng);
                    req.divide_cd = 0;
                    req.divide_limit = 0;
                    req.IL2_release_remain = 0.0f;
                    req.TGFB_release_remain = 0.0f;
                    req.CTLA4 = 0.0f;
                    d_requests[slot] = req;
                }
            } else {
                atomicAdd(&diag->bcell_place_fail, 1);
            }
        }
    }

    // ── DC source (bit 4) — homeostatic, per-subtype ──
    if (flags & 16) {
        atomicAdd(&diag->dc_sources, 1);
        // Try cDC1 recruitment
        if (rng_uniform(rng) < p.p_dc_cdc1) {
            if (try_find_open_neighbor(x, y, z, p.nx, p.ny, p.nz,
                    d_vol_used, p.vol_dc, p.voxel_capacity,
                    rng, px, py, pz)) {
                atomicAdd(&diag->dc_roll_pass, 1);
                atomicAdd(&diag->dc_place_ok, 1);
                int slot = atomicAdd(d_request_count, 1);
                if (slot < MAX_RECRUITS_PER_STEP) {
                    RecruitRequest req;
                    req.x = px; req.y = py; req.z = pz;
                    req.cell_type = CELL_TYPE_DC;
                    req.cell_state = DC_IMMATURE;
                    req.subtype = DC_CDC1;
                    req.life = sample_normal_life_gpu(p.dc_life_immature_mean, p.dc_life_immature_sd, rng);
                    req.divide_cd = 0;
                    req.divide_limit = 0;
                    req.IL2_release_remain = 0.0f;
                    req.TGFB_release_remain = 0.0f;
                    req.CTLA4 = 0.0f;
                    d_requests[slot] = req;
                }
            }
        }
        // Try cDC2 recruitment
        if (rng_uniform(rng) < p.p_dc_cdc2) {
            if (try_find_open_neighbor(x, y, z, p.nx, p.ny, p.nz,
                    d_vol_used, p.vol_dc, p.voxel_capacity,
                    rng, px, py, pz)) {
                atomicAdd(&diag->dc_roll_pass, 1);
                atomicAdd(&diag->dc_place_ok, 1);
                int slot = atomicAdd(d_request_count, 1);
                if (slot < MAX_RECRUITS_PER_STEP) {
                    RecruitRequest req;
                    req.x = px; req.y = py; req.z = pz;
                    req.cell_type = CELL_TYPE_DC;
                    req.cell_state = DC_IMMATURE;
                    req.subtype = DC_CDC2;
                    req.life = sample_normal_life_gpu(p.dc_life_immature_mean, p.dc_life_immature_sd, rng);
                    req.divide_cd = 0;
                    req.divide_limit = 0;
                    req.IL2_release_remain = 0.0f;
                    req.TGFB_release_remain = 0.0f;
                    req.CTLA4 = 0.0f;
                    d_requests[slot] = req;
                }
            }
        }
    }
}

// ============================================================================
// Host Function: Launch GPU recruitment kernel
// ============================================================================
FLAMEGPU_HOST_FUNCTION(recruit_gpu) {
    nvtxRangePush("Recruit GPU");
    if (!g_pde_solver) { nvtxRangePop(); return; }

    const int nx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int nz = FLAMEGPU->environment.getProperty<int>("grid_size_z");

    // Reset atomic counter and diagnostics
    CUDA_CHECK(cudaMemset(d_recruit_count, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_recruit_diag, 0, sizeof(RecruitDiag)));

    // Build parameter struct
    RecruitKernelParams p;
    p.nx = nx; p.ny = ny; p.nz = nz;

    // T cell recruitment probabilities
    float qsp_teff = FLAMEGPU->environment.getProperty<float>("qsp_teff_central");
    float qsp_treg = FLAMEGPU->environment.getProperty<float>("qsp_treg_central");
    float qsp_th   = FLAMEGPU->environment.getProperty<float>("qsp_th_central");
    p.p_teff = std::min(qsp_teff * FLAMEGPU->environment.getProperty<float>("PARAM_TEFF_RECRUIT_K"), 1.0f);
    p.p_treg = std::min(qsp_treg * FLAMEGPU->environment.getProperty<float>("PARAM_TREG_RECRUIT_K"), 1.0f);
    p.p_th   = std::min(qsp_th   * FLAMEGPU->environment.getProperty<float>("PARAM_TH_RECRUIT_K"),   1.0f);

    // Voxel caps
    // nr_t_voxel/nr_t_voxel_c removed — replaced by volume capacity

    // T cell life/division params
    p.t_life_mean   = FLAMEGPU->environment.getProperty<float>("PARAM_T_CELL_LIFE_MEAN_SLICE");
    p.t_life_sd     = FLAMEGPU->environment.getProperty<float>("PARAM_TCELL_LIFESPAN_SD");
    p.t_divide_cd   = FLAMEGPU->environment.getProperty<int>("PARAM_TCELL_DIV_INTERNAL");
    p.t_divide_limit = FLAMEGPU->environment.getProperty<int>("PARAM_TCELL_DIV_LIMIT");
    p.t_IL2_release = FLAMEGPU->environment.getProperty<float>("PARAM_TCELL_IL2_RELEASE_TIME");

    // TCD4 life/division params
    p.tcd4_life_mean   = FLAMEGPU->environment.getProperty<float>("PARAM_TCD4_LIFE_MEAN_SLICE");
    p.tcd4_life_sd     = FLAMEGPU->environment.getProperty<float>("PARAM_TCD4_LIFESPAN_SD");
    p.tcd4_divide_cd   = FLAMEGPU->environment.getProperty<int>("PARAM_TREG_DIV_INTERVAL");
    p.tcd4_divide_limit = FLAMEGPU->environment.getProperty<int>("PARAM_TCD4_DIV_LIMIT");
    p.tcd4_TGFB_release = FLAMEGPU->environment.getProperty<float>("PARAM_TCD4_TGFB_RELEASE_TIME");
    p.ctla4_treg = FLAMEGPU->environment.getProperty<float>("PARAM_CTLA4_TREG");

    // MDSC params
    p.p_mdsc = FLAMEGPU->environment.getProperty<float>("PARAM_MDSC_RECRUIT_K");
    p.mdsc_life_mean = FLAMEGPU->environment.getProperty<float>("PARAM_MDSC_LIFE_MEAN_SLICE");
    p.il6_conc = g_pde_solver ? reinterpret_cast<const float*>(
        g_pde_solver->get_device_concentration_ptr(CHEM_IL6)) : nullptr;
    p.il6_ec50_mdsc = FLAMEGPU->environment.getProperty<float>("PARAM_MDSC_EC50_IL6_REC");

    // MAC params
    p.p_mac = FLAMEGPU->environment.getProperty<float>("PARAM_MAC_RECRUIT_K");
    p.mac_life_mean = FLAMEGPU->environment.getProperty<float>("PARAM_MAC_LIFE_MEAN");

    // B cell params
    p.p_bcell = FLAMEGPU->environment.getProperty<float>("PARAM_BCELL_RECRUIT_K");
    p.bcell_life_mean = FLAMEGPU->environment.getProperty<float>("PARAM_BCELL_LIFE_MEAN");
    p.bcell_life_sd   = FLAMEGPU->environment.getProperty<float>("PARAM_BCELL_LIFE_SD");

    // DC params
    p.p_dc_cdc1 = FLAMEGPU->environment.getProperty<float>("PARAM_DC_RECRUIT_K_CDC1");
    p.p_dc_cdc2 = FLAMEGPU->environment.getProperty<float>("PARAM_DC_RECRUIT_K_CDC2");
    p.dc_life_immature_mean = FLAMEGPU->environment.getProperty<float>("PARAM_DC_LIFE_IMMATURE_MEAN");
    p.dc_life_immature_sd   = FLAMEGPU->environment.getProperty<float>("PARAM_DC_LIFE_IMMATURE_SD");
    p.dc_life_mature_mean   = FLAMEGPU->environment.getProperty<float>("PARAM_DC_LIFE_MATURE_MEAN");
    p.dc_life_mature_sd     = FLAMEGPU->environment.getProperty<float>("PARAM_DC_LIFE_MATURE_SD");
    p.vol_dc = FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_DC_IMMATURE");

    // RNG seed: combine environment seed with step counter (salt=0x1 for recruitment)
    unsigned int base_seed = FLAMEGPU->environment.getProperty<unsigned int>("sim_seed");
    p.seed = base_seed ^ (static_cast<unsigned int>(FLAMEGPU->getStepCounter()) * 2654435761u + 1u);

    // CXCL12 T cell exclusion + CCL5 Treg recruitment pointers
    p.cxcl12_conc = g_pde_solver ? reinterpret_cast<const float*>(
        g_pde_solver->get_device_concentration_ptr(CHEM_CXCL12)) : nullptr;
    p.cxcl12_ec50 = FLAMEGPU->environment.getProperty<float>("PARAM_CXCL12_EC50_TEXCL");
    p.ccl5_conc = g_pde_solver ? reinterpret_cast<const float*>(
        g_pde_solver->get_device_concentration_ptr(CHEM_CCL5)) : nullptr;
    p.ccl5_ec50 = FLAMEGPU->environment.getProperty<float>("PARAM_CCL5_EC50_TREG");
    p.ccl5_ratio = FLAMEGPU->environment.getProperty<float>("PARAM_CCL5_TREG_RATIO");

    // Volume-based occupancy params
    p.voxel_capacity = FLAMEGPU->environment.getProperty<float>("PARAM_VOXEL_CAPACITY");
    p.vol_teff   = FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_TCELL_EFF");
    p.vol_treg   = FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_TREG_REG");
    p.vol_th     = FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_TREG_TH");
    p.vol_mdsc   = FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_MDSC");
    p.vol_mac_m1 = FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_MAC_M1");
    p.vol_mac_m2 = FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_MAC_M2");
    p.vol_bcell  = FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_BCELL_NAIVE");

    // Launch kernel
    dim3 block(8, 8, 8);
    dim3 grid((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);

    recruit_all_kernel<<<grid, block>>>(
        g_pde_solver->get_device_recruitment_sources_ptr(),
        d_cancer_occ,
        d_volume_used,
        d_recruit_requests, d_recruit_count, d_recruit_diag, p);

    CUDA_CHECK(cudaDeviceSynchronize());

    // Stash stats for logging
    g_recruit_stats.p_teff = p.p_teff;
    g_recruit_stats.p_treg = p.p_treg;
    g_recruit_stats.p_th   = p.p_th;
    g_recruit_stats.qsp_teff = qsp_teff;
    g_recruit_stats.qsp_treg = qsp_treg;
    g_recruit_stats.qsp_th   = qsp_th;

    // Copy diagnostics from device
    RecruitDiag diag_host;
    CUDA_CHECK(cudaMemcpy(&diag_host, d_recruit_diag, sizeof(RecruitDiag), cudaMemcpyDeviceToHost));
    g_recruit_stats.t_sources = diag_host.t_sources;
    g_recruit_stats.mdsc_sources = diag_host.mdsc_sources;
    g_recruit_stats.mac_sources = diag_host.mac_sources;
    g_recruit_stats.bcell_sources = diag_host.bcell_sources;
    g_recruit_stats.dc_sources = diag_host.dc_sources;

    nvtxRangePop();
}

// ============================================================================
// Host Function: Read GPU recruitment results and create FLAMEGPU agents.
// Thin loop — all heavy work (occupancy check, RNG) was done on GPU.
// ============================================================================
FLAMEGPU_HOST_FUNCTION(place_recruited_agents) {
    nvtxRangePush("Place Recruited Agents");

    // Read count from device
    int count = 0;
    CUDA_CHECK(cudaMemcpy(&count, d_recruit_count, sizeof(int), cudaMemcpyDeviceToHost));
    if (count > MAX_RECRUITS_PER_STEP) count = MAX_RECRUITS_PER_STEP;

    if (count == 0) {
        // Zero stats
        g_recruit_stats.teff_rec   = 0;
        g_recruit_stats.treg_rec   = 0;
        g_recruit_stats.th_rec     = 0;
        g_recruit_stats.mdsc_rec   = 0;
        g_recruit_stats.mac_rec    = 0;
        g_recruit_stats.mac_m1_rec = 0;
        g_recruit_stats.mac_m2_rec = 0;
        g_recruit_stats.bcell_rec  = 0;
        g_recruit_stats.dc_rec     = 0;
        nvtxRangePop();
        return;
    }

    // D2H copy of compact request buffer
    std::vector<RecruitRequest> requests(count);
    CUDA_CHECK(cudaMemcpy(requests.data(), d_recruit_requests,
        count * sizeof(RecruitRequest), cudaMemcpyDeviceToHost));

    // Get agent APIs
    auto tcell_api = FLAMEGPU->agent(AGENT_TCELL);
    auto treg_api  = FLAMEGPU->agent(AGENT_TREG);
    auto mdsc_api  = FLAMEGPU->agent(AGENT_MDSC);
    auto mac_api   = FLAMEGPU->agent(AGENT_MACROPHAGE);
    auto bcell_api = FLAMEGPU->agent(AGENT_BCELL);
    auto dc_api    = FLAMEGPU->agent(AGENT_DC);

    int teff_recruited = 0, treg_recruited = 0, th_recruited = 0;
    int mdsc_recruited = 0, mac_recruited = 0, mac_m1_recruited = 0, mac_m2_recruited = 0;
    int bcell_recruited = 0, dc_recruited = 0;

    for (int i = 0; i < count; i++) {
        const auto& r = requests[i];

        if (r.cell_type == CELL_TYPE_T) {
            auto a = tcell_api.newAgent();
            a.setVariable<int>("x", r.x);
            a.setVariable<int>("y", r.y);
            a.setVariable<int>("z", r.z);
            a.setVariable<int>("cell_state", r.cell_state);
            a.setVariable<int>("life", r.life);
            a.setVariable<int>("divide_cd", r.divide_cd);
            a.setVariable<int>("divide_limit", r.divide_limit);
            a.setVariable<float>("IL2_release_remain", r.IL2_release_remain);
            a.setVariable<int>("persist_dir_x", 0);
            a.setVariable<int>("persist_dir_y", 0);
            a.setVariable<int>("persist_dir_z", 0);
            a.setVariable<int>("hypoxia_exposure", 0);
            a.setVariable<float>("hypoxia_kill_factor", 1.0f);
            teff_recruited++;
        }
        else if (r.cell_type == CELL_TYPE_TREG) {
            auto a = treg_api.newAgent();
            a.setVariable<int>("x", r.x);
            a.setVariable<int>("y", r.y);
            a.setVariable<int>("z", r.z);
            a.setVariable<int>("cell_state", r.cell_state);
            a.setVariable<int>("life", r.life);
            a.setVariable<int>("divide_cd", r.divide_cd);
            a.setVariable<int>("divide_limit", r.divide_limit);
            a.setVariable<float>("TGFB_release_remain", r.TGFB_release_remain);
            a.setVariable<float>("CTLA4", r.CTLA4);
            a.setVariable<int>("persist_dir_x", 0);
            a.setVariable<int>("persist_dir_y", 0);
            a.setVariable<int>("persist_dir_z", 0);
            if (r.cell_state == TCD4_TREG) treg_recruited++;
            else th_recruited++;
        }
        else if (r.cell_type == CELL_TYPE_MDSC) {
            auto a = mdsc_api.newAgent();
            a.setVariable<int>("x", r.x);
            a.setVariable<int>("y", r.y);
            a.setVariable<int>("z", r.z);
            a.setVariable<int>("life", r.life);
            a.setVariable<int>("persist_dir_x", 0);
            a.setVariable<int>("persist_dir_y", 0);
            a.setVariable<int>("persist_dir_z", 0);
            mdsc_recruited++;
        }
        else if (r.cell_type == CELL_TYPE_MAC) {
            auto a = mac_api.newAgent();
            a.setVariable<int>("x", r.x);
            a.setVariable<int>("y", r.y);
            a.setVariable<int>("z", r.z);
            a.setVariable<int>("cell_state", r.cell_state);
            a.setVariable<int>("life", r.life);
            a.setVariable<int>("persist_dir_x", 0);
            a.setVariable<int>("persist_dir_y", 0);
            a.setVariable<int>("persist_dir_z", 0);
            mac_recruited++;
            if (r.cell_state == MAC_M1) mac_m1_recruited++;
            else                        mac_m2_recruited++;
        }
        else if (r.cell_type == CELL_TYPE_BCELL) {
            auto a = bcell_api.newAgent();
            a.setVariable<int>("x", r.x);
            a.setVariable<int>("y", r.y);
            a.setVariable<int>("z", r.z);
            a.setVariable<int>("cell_state", BCELL_NAIVE);
            a.setVariable<int>("life", r.life);
            a.setVariable<int>("dead", 0);
            a.setVariable<int>("has_antigen", 0);
            a.setVariable<int>("is_breg", 0);
            a.setVariable<int>("activation_timer", 0);
            a.setVariable<int>("divide_flag", 0);
            a.setVariable<int>("divide_cd", 0);
            a.setVariable<int>("divide_limit", 0);
            a.setVariable<int>("divide_wave", 0);
            a.setVariable<int>("persist_dir_x", 0);
            a.setVariable<int>("persist_dir_y", 0);
            a.setVariable<int>("persist_dir_z", 0);
            a.setVariable<int>("neighbor_cancer_count", 0);
            a.setVariable<int>("neighbor_th_count", 0);
            a.setVariable<int>("neighbor_bcell_count", 0);
            a.setVariable<int>("neighbor_fib_count", 0);
            bcell_recruited++;
        }
        else if (r.cell_type == CELL_TYPE_DC) {
            auto a = dc_api.newAgent();
            a.setVariable<int>("x", r.x);
            a.setVariable<int>("y", r.y);
            a.setVariable<int>("z", r.z);
            a.setVariable<int>("cell_state", DC_IMMATURE);
            a.setVariable<int>("dc_subtype", r.subtype);
            a.setVariable<int>("life", r.life);
            a.setVariable<int>("dead", 0);
            a.setVariable<int>("presentation_capacity", 0);
            a.setVariable<int>("persist_dir_x", 0);
            a.setVariable<int>("persist_dir_y", 0);
            a.setVariable<int>("persist_dir_z", 0);
            a.setVariable<int>("neighbor_tcell_count", 0);
            a.setVariable<int>("neighbor_bcell_count", 0);
            dc_recruited++;
        }
    }

    // Update stats
    g_recruit_stats.teff_rec   = teff_recruited;
    g_recruit_stats.treg_rec   = treg_recruited;
    g_recruit_stats.th_rec     = th_recruited;
    g_recruit_stats.mdsc_rec   = mdsc_recruited;
    g_recruit_stats.mac_rec    = mac_recruited;
    g_recruit_stats.mac_m1_rec = mac_m1_recruited;
    g_recruit_stats.mac_m2_rec = mac_m2_recruited;
    g_recruit_stats.bcell_rec  = bcell_recruited;
    g_recruit_stats.dc_rec     = dc_recruited;

    // Update QSP MacroProperty counters (used by QSP coupling, separate from stats)
    auto counters = FLAMEGPU->environment.getMacroProperty<int, ABM_EVENT_COUNTER_SIZE>("abm_event_counters");
    counters[ABM_COUNT_TEFF_REC] += teff_recruited;
    counters[ABM_COUNT_TH_REC]   += th_recruited;
    counters[ABM_COUNT_TREG_REC] += treg_recruited;
    counters[ABM_COUNT_MDSC_REC] += mdsc_recruited;
    counters[ABM_COUNT_MAC_REC]  += mac_recruited;
    counters[ABM_COUNT_BCELL_REC] += bcell_recruited;
    counters[ABM_COUNT_DC_REC] += dc_recruited;

    nvtxRangePop();
}

// ============================================================================
// Zero occupancy grids at the start of each step
// ============================================================================
FLAMEGPU_HOST_FUNCTION(zero_occupancy_grid) {
    nvtxRangePush("Zero Occ Grid");
    if (g_pde_solver) {
        int total_voxels = g_pde_solver->get_total_voxels();
        if (d_cancer_occ)      cudaMemset(d_cancer_occ,      0, total_voxels * sizeof(unsigned int));
        if (d_vas_tip_id_grid) cudaMemset(d_vas_tip_id_grid, 0, total_voxels * sizeof(unsigned int));
        if (d_volume_used)     cudaMemset(d_volume_used,     0, total_voxels * sizeof(float));
    }
    nvtxRangePop();
}

// ============================================================================
// Zero Fibroblast Density Field (reset before scatter)
// Uses cudaMemset on flat device array — no D2H/H2D copy.
// ============================================================================
FLAMEGPU_HOST_FUNCTION(zero_fib_density_field) {
    nvtxRangePush("Zero Fib Density");
    if (d_fib_density_field && g_pde_solver) {
        int total_voxels = g_pde_solver->get_total_voxels();
        cudaMemset(d_fib_density_field, 0, total_voxels * sizeof(float));
    }
    nvtxRangePop();
}

// ============================================================================
// ECM Grid: decay + myCAF deposition + MMP degradation + crosslink accumulation.
// Operates entirely on device arrays — no MacroProperty D2H/H2D.
// ============================================================================
FLAMEGPU_HOST_FUNCTION(update_ecm_grid) {
    nvtxRangePush("Update ECM Grid");

    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int grid_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");

    float voxel_size_cm  = FLAMEGPU->environment.getProperty<float>("PARAM_VOXEL_SIZE_CM");
    float dt             = FLAMEGPU->environment.getProperty<float>("PARAM_SEC_PER_SLICE");
    float voxel_vol_cm3  = voxel_size_cm * voxel_size_cm * voxel_size_cm;

    // ECM parameters (new unified system)
    float k_decay           = FLAMEGPU->environment.getProperty<float>("PARAM_ECM_DECAY_RATE");
    float k_depo            = FLAMEGPU->environment.getProperty<float>("PARAM_ECM_DEPOSITION_RATE");
    float density_cap       = FLAMEGPU->environment.getProperty<float>("PARAM_ECM_DENSITY_CAP");
    float tgfb_ec50         = FLAMEGPU->environment.getProperty<float>("PARAM_ECM_TGFB_EC50");
    float ecm_baseline      = FLAMEGPU->environment.getProperty<float>("PARAM_ECM_BASELINE");
    float k_mmp             = FLAMEGPU->environment.getProperty<float>("PARAM_ECM_MMP_DEGRADE_RATE");
    float alpha_crosslink   = FLAMEGPU->environment.getProperty<float>("PARAM_ECM_CROSSLINK_RESISTANCE");
    float k_lox             = FLAMEGPU->environment.getProperty<float>("PARAM_ECM_CROSSLINK_RATE");
    float yap_ec50          = FLAMEGPU->environment.getProperty<float>("PARAM_ECM_YAP_EC50");

    // PDE concentration pointers
    const float* tgfb_ptr = g_pde_solver->get_device_concentration_ptr(CHEM_TGFB);
    const float* mmp_ptr  = g_pde_solver->get_device_concentration_ptr(CHEM_MMP);

    dim3 block(8, 8, 8);
    dim3 grid((grid_x + 7) / 8, (grid_y + 7) / 8, (grid_z + 7) / 8);
    update_ecm_grid_kernel<<<grid, block>>>(
        d_ecm_density, d_ecm_crosslink,
        d_fib_density_field, tgfb_ptr, mmp_ptr,
        grid_x, grid_y, grid_z,
        voxel_vol_cm3, dt,
        k_decay, k_depo, density_cap,
        tgfb_ec50, ecm_baseline,
        k_mmp, alpha_crosslink,
        k_lox, yap_ec50);
    cudaDeviceSynchronize();

    nvtxRangePop();
}

// ============================================================================
// Antigen Grid Decay: exponential decay of persistent antigen signal.
// Called once per ABM step, before agents read the grid.
// decay_factor = exp(-decay_rate * dt)
// ============================================================================
FLAMEGPU_HOST_FUNCTION(decay_antigen_grid) {
    nvtxRangePush("Decay Antigen Grid");
    if (d_antigen_grid && g_pde_solver) {
        int total_voxels = g_pde_solver->get_total_voxels();
        float decay_rate = FLAMEGPU->environment.getProperty<float>("PARAM_ANTIGEN_DECAY_RATE");
        float dt = FLAMEGPU->environment.getProperty<float>("PARAM_SEC_PER_SLICE");
        float decay_factor = expf(-decay_rate * dt);

        int block_size = 256;
        int grid_size = (total_voxels + block_size - 1) / block_size;
        decay_antigen_kernel<<<grid_size, block_size>>>(d_antigen_grid, total_voxels, decay_factor);
        cudaDeviceSynchronize();
    }
    nvtxRangePop();
}

// ============================================================================
// Stress Field Decay: exponential decay of mechanical stress from cancer movement.
// Called once per ABM step, early in the step (before agents move and deposit new stress).
// ============================================================================
FLAMEGPU_HOST_FUNCTION(decay_stress_field) {
    nvtxRangePush("Decay Stress Field");
    if (d_mech_stress_x && g_pde_solver) {
        int total_voxels = g_pde_solver->get_total_voxels();
        float decay_rate = FLAMEGPU->environment.getProperty<float>("PARAM_ECM_STRESS_DECAY");
        float dt = FLAMEGPU->environment.getProperty<float>("PARAM_SEC_PER_SLICE");
        float decay_factor = expf(-decay_rate * dt);

        int block_size = 256;
        int grid_size = (total_voxels + block_size - 1) / block_size;
        decay_stress_kernel<<<grid_size, block_size>>>(
            d_mech_stress_x, d_mech_stress_y, d_mech_stress_z,
            total_voxels, decay_factor);
        cudaDeviceSynchronize();
    }
    nvtxRangePop();
}

// ============================================================================
// ECM Fiber Orientation Update: reorient fibers based on myCAF traction + stress.
// Called once per ABM step, after update_ecm_grid (needs current fib_field + crosslink).
// ============================================================================
FLAMEGPU_HOST_FUNCTION(update_ecm_orientation) {
    nvtxRangePush("Update ECM Orientation");
    if (d_ecm_orient_x && g_pde_solver) {
        const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
        const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
        const int grid_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");
        float dt = FLAMEGPU->environment.getProperty<float>("PARAM_SEC_PER_SLICE");

        float base_rate = FLAMEGPU->environment.getProperty<float>("PARAM_ECM_ORIENT_RATE");
        float w_traction = FLAMEGPU->environment.getProperty<float>("PARAM_ECM_ORIENT_TRACTION_W");
        float w_stress = FLAMEGPU->environment.getProperty<float>("PARAM_ECM_ORIENT_STRESS_W");
        float alpha_crosslink = FLAMEGPU->environment.getProperty<float>("PARAM_ECM_ORIENT_CROSSLINK_RESIST");

        // TGF-β gradient pointers (already computed by compute_pde_gradients last step)
        const float* tgfb_gx = g_pde_solver->get_device_gradx_ptr(1);  // TGFB = grad substrate index 1
        const float* tgfb_gy = g_pde_solver->get_device_grady_ptr(1);
        const float* tgfb_gz = g_pde_solver->get_device_gradz_ptr(1);

        dim3 block(8, 8, 8);
        dim3 grid((grid_x + 7) / 8, (grid_y + 7) / 8, (grid_z + 7) / 8);
        update_ecm_orient_kernel<<<grid, block>>>(
            d_ecm_orient_x, d_ecm_orient_y, d_ecm_orient_z,
            d_mech_stress_x, d_mech_stress_y, d_mech_stress_z,
            d_fib_density_field, d_ecm_crosslink,
            tgfb_gx, tgfb_gy, tgfb_gz,
            grid_x, grid_y, grid_z,
            dt, base_rate, w_traction, w_stress, alpha_crosslink);
        cudaDeviceSynchronize();
    }
    nvtxRangePop();
}

// ============================================================================
// Aggregate ABM Event Counters from Agent States
// Counts cancer cell deaths by cause from agents marked as dead
// ============================================================================
FLAMEGPU_HOST_FUNCTION(aggregate_abm_events) {
    nvtxRangePush("Aggregate ABM Events");
    auto counters = FLAMEGPU->environment.getMacroProperty<int, ABM_EVENT_COUNTER_SIZE>("abm_event_counters");
    auto cc_api = FLAMEGPU->agent("CancerCell");

    // Get population data for iteration
    flamegpu::DeviceAgentVector cc_agents = cc_api.getPopulationData();
    const unsigned int cc_count = cc_agents.size();

    // Count cancer cell deaths by cause from dead agents
    int cc_death_total = 0;
    int cc_death_natural = 0;
    int cc_death_t_kill = 0;
    int cc_death_mac_kill = 0;

    for (unsigned int i = 0; i < cc_count; i++) {
        if (cc_agents[i].getVariable<int>("dead") != 0) {
            cc_death_total++;
            int reason = cc_agents[i].getVariable<int>("death_reason");
            if (reason == 0) {
                cc_death_natural++;
            } else if (reason == 1) {
                cc_death_t_kill++;
            } else if (reason == 2) {
                cc_death_mac_kill++;
            }
        }
    }

    // Update counter MacroProperty with aggregated values
    counters[ABM_COUNT_CC_DEATH] = cc_death_total;
    counters[ABM_COUNT_CC_DEATH_NATURAL] = cc_death_natural;
    counters[ABM_COUNT_CC_DEATH_T_KILL] = cc_death_t_kill;
    counters[ABM_COUNT_CC_DEATH_MAC_KILL] = cc_death_mac_kill;
    nvtxRangePop();
}

// ============================================================================
// Copy ABM Event Counters from MacroProperty to Environment Properties
// Called BEFORE QSP so the ODE model can read accumulated counts this step
// ============================================================================
FLAMEGPU_HOST_FUNCTION(copy_abm_counters_to_environment) {
    nvtxRangePush("Copy ABM Counters");
    auto counters = FLAMEGPU->environment.getMacroProperty<int, ABM_EVENT_COUNTER_SIZE>("abm_event_counters");

    // Copy from MacroProperty array to environment properties for QSP access
    FLAMEGPU->environment.setProperty<int>("ABM_cc_death", static_cast<int>(counters[ABM_COUNT_CC_DEATH]));
    FLAMEGPU->environment.setProperty<int>("ABM_cc_death_t_kill", static_cast<int>(counters[ABM_COUNT_CC_DEATH_T_KILL]));
    FLAMEGPU->environment.setProperty<int>("ABM_cc_death_mac_kill", static_cast<int>(counters[ABM_COUNT_CC_DEATH_MAC_KILL]));
    FLAMEGPU->environment.setProperty<int>("ABM_cc_death_natural", static_cast<int>(counters[ABM_COUNT_CC_DEATH_NATURAL]));
    FLAMEGPU->environment.setProperty<int>("ABM_TEFF_REC", static_cast<int>(counters[ABM_COUNT_TEFF_REC]));
    FLAMEGPU->environment.setProperty<int>("ABM_TH_REC", static_cast<int>(counters[ABM_COUNT_TH_REC]));
    FLAMEGPU->environment.setProperty<int>("ABM_TREG_REC", static_cast<int>(counters[ABM_COUNT_TREG_REC]));
    FLAMEGPU->environment.setProperty<int>("ABM_MDSC_REC", static_cast<int>(counters[ABM_COUNT_MDSC_REC]));
    FLAMEGPU->environment.setProperty<int>("ABM_MAC_REC", static_cast<int>(counters[ABM_COUNT_MAC_REC]));
    nvtxRangePop();
}

// ============================================================================
// Reset ABM → QSP Event Counters (called at END of each step)
// Clears MacroProperty array for next step's accumulation
// ============================================================================
FLAMEGPU_HOST_FUNCTION(reset_abm_event_counters) {
    nvtxRangePush("Reset ABM Counters");
    auto counters = FLAMEGPU->environment.getMacroProperty<int, ABM_EVENT_COUNTER_SIZE>("abm_event_counters");

    // Reset all counter elements to zero
    for (int i = 0; i < ABM_EVENT_COUNTER_SIZE; i++) {
        counters[i] = 0;
    }
    nvtxRangePop();
}

// ============================================================================
// fib_execute_divide removed: activation is now device-side (fib_activate in fibroblast.cuh).
// Each fibroblast agent is a multi-voxel chain that extends itself on activation.

// ============================================================================
// Timing Accessor: Last PDE Solve Time (milliseconds)
// ============================================================================
double get_last_pde_ms() {
    return g_last_pde_ms;
}

// ============================================================================
// Timing Checkpoint Host Functions
//
// These are thin FLAMEGPU host function layers inserted at phase boundaries.
// Each records elapsed wall-clock time since the previous checkpoint.
// Because FLAMEGPU2 fully completes all GPU kernels in a layer before calling
// the next host function, wall-clock accurately captures GPU time per phase.
//
// Phase map (in layer execution order):
//   timing_step_start        → resets the clock (very first layer)
//   [Phase 0: recruitment]
//   timing_after_recruit     → captures recruit time
//   [Phase 1: broadcast + neighbor scan]
//   timing_after_broadcast   → captures broadcast+scan time
//   [reset_pde_buffers + state_transitions + compute_chemical_sources]
//   timing_after_sources     → captures state+sources time
//   [solve_pde  -- internally timed via g_last_pde_ms]
//   timing_after_pde         → captures solve_pde wall time (for cross-check)
//   [compute_pde_gradients]
//   timing_after_gradients   → captures gradients time
//   [Phase 3: ECM]
//   timing_after_ecm         → captures ECM time
//   [Phase 4: occ + movement]
//   timing_after_movement    → captures movement time
//   [Phase 5: division]
//   timing_after_division    → captures division time
//   [Phase 6: QSP -- internally timed via g_last_qsp_ms]
// ============================================================================

FLAMEGPU_HOST_FUNCTION(timing_step_start) {
    nvtxRangePush("Step Start");
    reset_step_timer();
    nvtxRangePop();
}

FLAMEGPU_HOST_FUNCTION(timing_after_recruit) {
    nvtxRangePush("Timing Checkpoint: recruit");
    record_checkpoint("recruit");
    nvtxRangePop();
}

FLAMEGPU_HOST_FUNCTION(timing_after_broadcast) {
    nvtxRangePush("Timing Checkpoint: broadcast_scan");
    record_checkpoint("broadcast_scan");
    nvtxRangePop();
}

FLAMEGPU_HOST_FUNCTION(timing_after_sources) {
    nvtxRangePush("Timing Checkpoint: state_sources");
    record_checkpoint("state_sources");
    nvtxRangePop();
}

FLAMEGPU_HOST_FUNCTION(timing_after_pde) {
    nvtxRangePush("Timing Checkpoint: pde_wall");
    record_checkpoint("pde_wall");
    nvtxRangePop();
}

FLAMEGPU_HOST_FUNCTION(timing_after_gradients) {
    nvtxRangePush("Timing Checkpoint: gradients");
    record_checkpoint("gradients");
    nvtxRangePop();
}

FLAMEGPU_HOST_FUNCTION(timing_after_ecm) {
    nvtxRangePush("Timing Checkpoint: ecm");
    record_checkpoint("ecm");
    nvtxRangePop();
}

FLAMEGPU_HOST_FUNCTION(timing_after_movement) {
    nvtxRangePush("Timing Checkpoint: movement");
    record_checkpoint("movement");
    nvtxRangePop();
}

FLAMEGPU_HOST_FUNCTION(timing_after_division) {
    nvtxRangePush("Timing Checkpoint: division");
    record_checkpoint("division");
    nvtxRangePop();
}

// ── Wave-interleaved division control ──────────────────────────────────────
// reset_divide_wave: called before the first wave each step (sets wave index to 0)
FLAMEGPU_HOST_FUNCTION(reset_divide_wave) {
    FLAMEGPU->environment.setProperty<int>("divide_current_wave", 0);
}

// increment_divide_wave: called after each wave's agent layers to advance to next wave
FLAMEGPU_HOST_FUNCTION(increment_divide_wave) {
    int w = FLAMEGPU->environment.getProperty<int>("divide_current_wave");
    FLAMEGPU->environment.setProperty<int>("divide_current_wave", w + 1);
}

} // namespace PDAC