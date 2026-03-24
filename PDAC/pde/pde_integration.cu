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
static float* d_ecm_grid = nullptr;
static float* d_fib_density_field = nullptr;

// Flat occupancy arrays for GPU recruitment kernel (populated by agent write_to_occ_grid).
// d_t_occ: T cell + TReg count per voxel (only CELL_TYPE_T and CELL_TYPE_TREG increment).
//          Used with d_cancer_occ for combined cap check matching HCC isOpenToType logic.
// d_mac_occ / d_mdsc_occ: per-type counts for exclusive placement checks (atomicCAS).
// d_recruit_requests: compact output buffer for placement decisions (GPU→host).
// d_recruit_count: atomic counter for number of valid requests in buffer.
static unsigned int* d_t_occ = nullptr;
static unsigned int* d_mac_occ = nullptr;
static unsigned int* d_mdsc_occ = nullptr;
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
};
static RecruitDiag* d_recruit_diag = nullptr;

// ============================================================================
// CUDA Kernel: ECM Grid Update
// Applies decay + fibroblast deposition + saturation clamping per voxel in parallel.
// Called from update_ecm_grid host function after fib_build_density_field runs.
// ============================================================================
__global__ void update_ecm_grid_kernel(
    float* ecm, const float* fib_field, const float* tgfb_conc,
    int nx, int ny, int nz,
    float voxel_vol_cm3, float decay_rate, float dt,
    float ecm_baseline, float ecm_saturation, float release_rate,
    float tgfb_ec50)
{
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ty = blockIdx.y * blockDim.y + threadIdx.y;
    const int tz = blockIdx.z * blockDim.z + threadIdx.z;
    if (tx >= nx || ty >= ny || tz >= nz) return;

    const int idx = tz * (nx * ny) + ty * nx + tx;

    float curr_ecm     = ecm[idx];
    float curr_ecm_amt = curr_ecm * voxel_vol_cm3;

    // Exponential decay — matches HCC: exp(-SEC_PER_SLICE * decay_rate)
    float decayed = curr_ecm_amt * expf(-decay_rate * dt);

    // Per-voxel TGFB amplification — matches HCC: (1 + H_CAF_TGFB) at target voxel
    float tgfb    = tgfb_conc[idx];
    float H_TGFB  = tgfb / (tgfb + tgfb_ec50);

    // Deposition: fib_field * (1 + H_TGFB) * release_rate / 3 * (1 - saturation)
    float saturation = fminf(curr_ecm / ecm_saturation, 1.0f);
    float deposition = fib_field[idx] * (1.0f + H_TGFB) * release_rate / 3.0f * (1.0f - saturation);

    float new_ecm = (decayed + deposition) / voxel_vol_cm3;

    // Floor to baseline only — matches HCC (no upper saturation clamp)
    new_ecm = fmaxf(new_ecm, ecm_baseline);

    ecm[idx] = new_ecm;
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
                            const PDAC::GPUParam& gpu_params) {
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

    // Set decay rates (1/s) from params file
    config.decay_rates[CHEM_O2]    = gpu_params.getFloat(PARAM_O2_DECAY_RATE);
    config.decay_rates[CHEM_IFN]   = gpu_params.getFloat(PARAM_IFNG_DECAY_RATE);
    config.decay_rates[CHEM_IL2]   = gpu_params.getFloat(PARAM_IL2_DECAY_RATE);
    config.decay_rates[CHEM_IL10]  = gpu_params.getFloat(PARAM_IL10_DECAY_RATE);
    config.decay_rates[CHEM_TGFB]  = gpu_params.getFloat(PARAM_TGFB_DECAY_RATE);
    config.decay_rates[CHEM_CCL2]  = gpu_params.getFloat(PARAM_CCL2_DECAY_RATE);
    config.decay_rates[CHEM_ARGI]  = gpu_params.getFloat(PARAM_ARGI_DECAY_RATE);
    config.decay_rates[CHEM_NO]    = gpu_params.getFloat(PARAM_NO_DECAY_RATE);
    config.decay_rates[CHEM_IL12]  = gpu_params.getFloat(PARAM_IL12_DECAY_RATE);
    config.decay_rates[CHEM_VEGFA] = gpu_params.getFloat(PARAM_VEGFA_DECAY_RATE);
    
    g_pde_solver = new PDESolver(config);
    g_pde_solver->initialize();

    // Set initial O2 concentration, all others start at 0.0
    g_pde_solver->set_initial_concentration(CHEM_O2, 0.673);  // Oxygen starts at 0.673 (amount/mL)

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
    CUDA_CHECK(cudaMalloc(&d_ecm_grid, total_voxels * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_ecm_grid, 0, total_voxels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fib_density_field, total_voxels * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_fib_density_field, 0, total_voxels * sizeof(float)));

    // Allocate flat occupancy arrays for GPU recruitment kernel
    CUDA_CHECK(cudaMalloc(&d_t_occ, total_voxels * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_t_occ, 0, total_voxels * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_mac_occ, total_voxels * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_mac_occ, 0, total_voxels * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_mdsc_occ, total_voxels * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_mdsc_occ, 0, total_voxels * sizeof(unsigned int)));

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

    std::cout << "[DEBUG] Storing PDE device pointers in environment..." << std::endl;

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
    static const char* grad_names[NUM_GRAD_SUBSTRATES] = {"IFN", "TGFB", "CCL2", "VEGFA"};
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

    // Store ECM and fibroblast density field pointers (replace MacroProperty approach)
    model.Environment().newProperty<unsigned long long>("ecm_grid_ptr",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(d_ecm_grid)));
    model.Environment().newProperty<unsigned long long>("fib_density_field_ptr",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(d_fib_density_field)));

    // Store flat occupancy array pointers for GPU recruitment kernel
    model.Environment().newProperty<unsigned long long>("t_occ_ptr",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(d_t_occ)));
    model.Environment().newProperty<unsigned long long>("mac_occ_ptr",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(d_mac_occ)));
    model.Environment().newProperty<unsigned long long>("mdsc_occ_ptr",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(d_mdsc_occ)));

    std::cout << "PDE device pointers stored in FLAME GPU environment" << std::endl;
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
    if (d_ecm_grid) {
        cudaFree(d_ecm_grid);
        d_ecm_grid = nullptr;
    }
    if (d_fib_density_field) {
        cudaFree(d_fib_density_field);
        d_fib_density_field = nullptr;
    }
    if (d_vas_tip_id_grid) {
        cudaFree(d_vas_tip_id_grid);
        d_vas_tip_id_grid = nullptr;
    }
    if (d_t_occ) { cudaFree(d_t_occ); d_t_occ = nullptr; }
    if (d_mac_occ)   { cudaFree(d_mac_occ);   d_mac_occ = nullptr; }
    if (d_mdsc_occ)  { cudaFree(d_mdsc_occ);  d_mdsc_occ = nullptr; }
    if (d_recruit_requests) { cudaFree(d_recruit_requests); d_recruit_requests = nullptr; }
    if (d_recruit_count)    { cudaFree(d_recruit_count);    d_recruit_count = nullptr; }
    if (d_recruit_diag)     { cudaFree(d_recruit_diag);     d_recruit_diag = nullptr; }
}

void initialize_ecm_to_saturation(float ecm_saturation) {
    if (!d_ecm_grid || !g_pde_solver) return;
    int total_voxels = g_pde_solver->get_total_voxels();
    int block_size = 256;
    int grid_size = (total_voxels + block_size - 1) / block_size;
    fill_ecm_kernel<<<grid_size, block_size>>>(d_ecm_grid, total_voxels, ecm_saturation);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "[ECM] Initialized ECM grid to saturation value: " << ecm_saturation << std::endl;
}

float* get_ecm_grid_device_ptr() { return d_ecm_grid; }
float* get_fib_density_field_device_ptr() { return d_fib_density_field; }
unsigned int* get_vas_tip_id_grid_device_ptr() { return d_vas_tip_id_grid; }

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

// CUDA kernel to mark MDSC recruitment sources based on CCL2
__global__ void mark_mdsc_sources_kernel(
    int* d_recruitment_sources,
    const float* d_ccl2,
    const unsigned int* d_cancer_occ,
    int nx, int ny, int nz,
    float ec50_ccl2,
    unsigned int seed)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= nz) return;

    // Skip voxels where radius-3 sphere is completely filled with cancer
    if (is_tumor_dense_r3(d_cancer_occ, x, y, z, nx, ny, nz)) return;

    int idx = z * (nx * ny) + y * nx + x;

    float ccl2 = d_ccl2[idx];
    float H_CCL2 = ccl2 / (ccl2 + ec50_ccl2);

    // Simple random number generation (thread-local)
    unsigned int rng_state = seed + idx;
    rng_state = rng_state * 1103515245u + 12345u;
    float rand_val = (rng_state & 0x7FFFFFFF) / float(0x7FFFFFFF);

    if (rand_val < H_CCL2) {
        atomicOr(&d_recruitment_sources[idx], 2);  // Set MDSC bit (bit 1)
    }
}

// CUDA kernel to mark macrophage recruitment sources based on CCL2
__global__ void mark_mac_sources_kernel(
    int* d_recruitment_sources,
    const float* d_ccl2,
    const unsigned int* d_cancer_occ,
    int nx, int ny, int nz,
    float ec50_ccl2,
    unsigned int seed)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= nz) return;

    // Skip voxels where radius-3 sphere is completely filled with cancer
    if (is_tumor_dense_r3(d_cancer_occ, x, y, z, nx, ny, nz)) return;

    int idx = z * (nx * ny) + y * nx + x;

    float ccl2 = d_ccl2[idx];
    float H_CCL2 = ccl2 / (ccl2 + ec50_ccl2);

    // Simple random number generation (thread-local)
    unsigned int rng_state = seed + idx;
    rng_state = rng_state * 1103515245u + 12345u;
    float rand_val = (rng_state & 0x7FFFFFFF) / float(0x7FFFFFFF);

    if (rand_val < H_CCL2) {
        atomicOr(&d_recruitment_sources[idx], 4);  // Set macrophage bit (bit 2)
    }
}

// Update vasculature count env property (used by vascular_mark_t_sources device function)
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

// Mark MDSC sources based on CCL2 concentration
FLAMEGPU_HOST_FUNCTION(mark_mdsc_sources) {
    nvtxRangePush("Mark MDSC Sources");
    if (!g_pde_solver) { nvtxRangePop(); return; }

    const int nx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int nz = FLAMEGPU->environment.getProperty<int>("grid_size_z");

    int* d_recruitment_sources = g_pde_solver->get_device_recruitment_sources_ptr();
    const float* d_ccl2 = g_pde_solver->get_device_concentration_ptr(CHEM_CCL2);

    // Get parameter
    float ec50_ccl2 = FLAMEGPU->environment.getProperty<float>("PARAM_MDSC_EC50_CCL2_REC");

    // Seed: combine environment seed with step counter (salt=0x9E3779B9 for MDSC)
    unsigned int base_seed = FLAMEGPU->environment.getProperty<unsigned int>("sim_seed");
    unsigned int seed = base_seed ^ (static_cast<unsigned int>(FLAMEGPU->getStepCounter()) * 2654435761u + 0x9E3779B9u);

    dim3 block(8, 8, 8);
    dim3 grid((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);

    mark_mdsc_sources_kernel<<<grid, block>>>(
        d_recruitment_sources, d_ccl2, d_cancer_occ, nx, ny, nz, ec50_ccl2, seed);

    cudaDeviceSynchronize();
    nvtxRangePop();
}

// ============================================================================
// GPU Recruitment Kernel: Packed parameters struct
// ============================================================================
struct RecruitKernelParams {
    int nx, ny, nz;
    // T cell (CD8) recruitment
    float p_teff, p_treg, p_th;
    int nr_t_voxel, nr_t_voxel_c;
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
    // MAC recruitment
    float p_mac;
    float mac_life_mean;
    // RNG seed
    unsigned int seed;
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
// Uses Fisher-Yates shuffle with thread-local RNG. Claims voxel via atomic ops on
// d_t_occ + d_cancer_occ (for T/TReg cap, matching HCC isOpenToType) and
// d_mac_occ/d_mdsc_occ (for exclusive placement).
// Returns true + sets (out_x, out_y, out_z) on success.
__device__ bool try_find_open_neighbor(
    int sx, int sy, int sz,
    int cell_type,
    int nx, int ny, int nz,
    int nr_t_voxel, int nr_t_voxel_c,
    unsigned int* d_t_occ,
    const unsigned int* d_cancer_occ,
    unsigned int* d_mac_occ,
    unsigned int* d_mdsc_occ,
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

        if (cell_type == CELL_TYPE_T) {
            // HCC isOpenToType: cap check uses cancer + T count combined.
            // With nr_t_voxel_c=1 and cancer present, count starts at >=1 → no T cells allowed.
            // With no cancer, up to nr_t_voxel (8) T cells per voxel.
            unsigned int cancer_count = d_cancer_occ[vidx];
            int cap = (cancer_count > 0u) ? nr_t_voxel_c : nr_t_voxel;

            // Atomic claim: increment T count, check combined (cancer + T) against cap
            unsigned int old_t = atomicAdd(&d_t_occ[vidx], 1u);
            if ((int)(cancer_count + old_t) >= cap) {
                atomicSub(&d_t_occ[vidx], 1u);  // Undo
                continue;
            }
            out_x = cx; out_y = cy; out_z = cz;
            return true;
        }
        else if (cell_type == CELL_TYPE_TREG) {
            // TRegs check T cell conditions for placement but don't consume T capacity.
            // HCC: TRegs are invisible to T cell cap — T cells don't count TRegs.
            unsigned int cancer_count = d_cancer_occ[vidx];
            int cap = (cancer_count > 0u) ? nr_t_voxel_c : nr_t_voxel;
            unsigned int t_count = d_t_occ[vidx];  // read-only, don't increment
            if ((int)(cancer_count + t_count) >= cap) continue;
            out_x = cx; out_y = cy; out_z = cz;
            return true;
        }
        else if (cell_type == CELL_TYPE_MAC) {
            // Exclusive: atomicCAS, only place if slot == 0
            unsigned int old = atomicCAS(&d_mac_occ[vidx], 0u, 1u);
            if (old != 0u) continue;
            out_x = cx; out_y = cy; out_z = cz;
            return true;
        }
        else if (cell_type == CELL_TYPE_MDSC) {
            // Exclusive: atomicCAS, only place if slot == 0
            unsigned int old = atomicCAS(&d_mdsc_occ[vidx], 0u, 1u);
            if (old != 0u) continue;
            out_x = cx; out_y = cy; out_z = cz;
            return true;
        }
    }
    return false;
}

// ============================================================================
// GPU Recruitment Kernel: One thread per voxel. Checks recruitment source flags,
// rolls probabilities, finds open neighbors, writes compact RecruitRequest buffer.
// ============================================================================
__global__ void recruit_all_kernel(
    const int* __restrict__ d_recruitment_sources,
    unsigned int* d_t_occ,
    const unsigned int* __restrict__ d_cancer_occ,
    unsigned int* d_mac_occ,
    unsigned int* d_mdsc_occ,
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

        // Try Teff
        if (rng_uniform(rng) < p.p_teff) {
            atomicAdd(&diag->teff_roll_pass, 1);
            if (try_find_open_neighbor(x, y, z, CELL_TYPE_T, p.nx, p.ny, p.nz,
                    p.nr_t_voxel, p.nr_t_voxel_c, d_t_occ, d_cancer_occ,
                    d_mac_occ, d_mdsc_occ, rng, px, py, pz)) {
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

        // Try TReg
        if (rng_uniform(rng) < p.p_treg) {
            atomicAdd(&diag->treg_roll_pass, 1);
            if (try_find_open_neighbor(x, y, z, CELL_TYPE_TREG, p.nx, p.ny, p.nz,
                    p.nr_t_voxel, p.nr_t_voxel_c, d_t_occ, d_cancer_occ,
                    d_mac_occ, d_mdsc_occ, rng, px, py, pz)) {
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

        // Try TH
        if (rng_uniform(rng) < p.p_th) {
            atomicAdd(&diag->th_roll_pass, 1);
            if (try_find_open_neighbor(x, y, z, CELL_TYPE_TREG, p.nx, p.ny, p.nz,
                    p.nr_t_voxel, p.nr_t_voxel_c, d_t_occ, d_cancer_occ,
                    d_mac_occ, d_mdsc_occ, rng, px, py, pz)) {
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
        if (rng_uniform(rng) < p.p_mdsc) {
            atomicAdd(&diag->mdsc_roll_pass, 1);
            if (try_find_open_neighbor(x, y, z, CELL_TYPE_MDSC, p.nx, p.ny, p.nz,
                    p.nr_t_voxel, p.nr_t_voxel_c, d_t_occ, d_cancer_occ,
                    d_mac_occ, d_mdsc_occ, rng, px, py, pz)) {
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
            if (try_find_open_neighbor(x, y, z, CELL_TYPE_MAC, p.nx, p.ny, p.nz,
                    p.nr_t_voxel, p.nr_t_voxel_c, d_t_occ, d_cancer_occ,
                    d_mac_occ, d_mdsc_occ, rng, px, py, pz)) {
                atomicAdd(&diag->mac_place_ok, 1);
                int slot = atomicAdd(d_request_count, 1);
                if (slot < MAX_RECRUITS_PER_STEP) {
                    RecruitRequest req;
                    req.x = px; req.y = py; req.z = pz;
                    req.cell_type = CELL_TYPE_MAC;
                    // 30% chance M2, else M1
                    req.cell_state = (rng_uniform(rng) < 0.3f) ? MAC_M2 : MAC_M1;
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
    p.nr_t_voxel   = FLAMEGPU->environment.getProperty<int>("PARAM_NR_T_VOXELS");
    p.nr_t_voxel_c = FLAMEGPU->environment.getProperty<int>("PARAM_NR_T_VOXELS_C");

    // T cell life/division params
    p.t_life_mean   = FLAMEGPU->environment.getProperty<float>("PARAM_T_CELL_LIFE_MEAN_SLICE");
    p.t_life_sd     = FLAMEGPU->environment.getProperty<float>("PARAM_TCELL_LIFESPAN_SD");
    p.t_divide_cd   = FLAMEGPU->environment.getProperty<int>("PARAM_TCELL_DIV_INTERNAL");
    p.t_divide_limit = FLAMEGPU->environment.getProperty<int>("PARAM_TCELL_DIV_LIMIT");
    p.t_IL2_release = FLAMEGPU->environment.getProperty<float>("PARAM_TCELL_IL2_RELEASE_TIME");

    // TCD4 life/division params
    p.tcd4_life_mean   = FLAMEGPU->environment.getProperty<float>("PARAM_TCD4_LIFE_MEAN_SLICE");
    p.tcd4_life_sd     = FLAMEGPU->environment.getProperty<float>("PARAM_TCD4_LIFESPAN_SD");
    p.tcd4_divide_cd   = FLAMEGPU->environment.getProperty<int>("PARAM_TCD4_DIV_INTERNAL");
    p.tcd4_divide_limit = FLAMEGPU->environment.getProperty<int>("PARAM_TCD4_DIV_LIMIT");
    p.tcd4_TGFB_release = FLAMEGPU->environment.getProperty<float>("PARAM_TCD4_TGFB_RELEASE_TIME");
    p.ctla4_treg = FLAMEGPU->environment.getProperty<float>("PARAM_CTLA4_TREG");

    // MDSC params
    p.p_mdsc = FLAMEGPU->environment.getProperty<float>("PARAM_MDSC_RECRUIT_K");
    p.mdsc_life_mean = FLAMEGPU->environment.getProperty<float>("PARAM_MDSC_LIFE_MEAN_SLICE");

    // MAC params
    p.p_mac = FLAMEGPU->environment.getProperty<float>("PARAM_MAC_RECRUIT_K");
    p.mac_life_mean = FLAMEGPU->environment.getProperty<float>("PARAM_MAC_LIFE_MEAN");

    // RNG seed: combine environment seed with step counter (salt=0x1 for recruitment)
    unsigned int base_seed = FLAMEGPU->environment.getProperty<unsigned int>("sim_seed");
    p.seed = base_seed ^ (static_cast<unsigned int>(FLAMEGPU->getStepCounter()) * 2654435761u + 1u);

    // Launch kernel
    dim3 block(8, 8, 8);
    dim3 grid((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);

    recruit_all_kernel<<<grid, block>>>(
        g_pde_solver->get_device_recruitment_sources_ptr(),
        d_t_occ, d_cancer_occ, d_mac_occ, d_mdsc_occ,
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

    // Print recruitment diagnostics
    int step = FLAMEGPU->getStepCounter();
    std::cout << "[Recruit Diag step=" << step << "] "
              << "T_src=" << diag_host.t_sources
              << " | Teff: roll=" << diag_host.teff_roll_pass
              << " ok=" << diag_host.teff_place_ok
              << " fail=" << diag_host.teff_place_fail
              << " | TReg: roll=" << diag_host.treg_roll_pass
              << " ok=" << diag_host.treg_place_ok
              << " fail=" << diag_host.treg_place_fail
              << " | TH: roll=" << diag_host.th_roll_pass
              << " ok=" << diag_host.th_place_ok
              << " fail=" << diag_host.th_place_fail
              << " | MDSC: roll=" << diag_host.mdsc_roll_pass
              << " ok=" << diag_host.mdsc_place_ok
              << " fail=" << diag_host.mdsc_place_fail
              << " | MAC: roll=" << diag_host.mac_roll_pass
              << " ok=" << diag_host.mac_place_ok
              << " fail=" << diag_host.mac_place_fail
              << " | p_teff=" << p.p_teff
              << " p_treg=" << p.p_treg
              << " p_th=" << p.p_th
              << std::endl;

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

    int teff_recruited = 0, treg_recruited = 0, th_recruited = 0;
    int mdsc_recruited = 0, mac_recruited = 0, mac_m1_recruited = 0, mac_m2_recruited = 0;

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
            a.setVariable<int>("tumble", 1);
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
            a.setVariable<int>("tumble", 1);
            if (r.cell_state == TCD4_TREG) treg_recruited++;
            else th_recruited++;
        }
        else if (r.cell_type == CELL_TYPE_MDSC) {
            auto a = mdsc_api.newAgent();
            a.setVariable<int>("x", r.x);
            a.setVariable<int>("y", r.y);
            a.setVariable<int>("z", r.z);
            a.setVariable<int>("life", r.life);
            a.setVariable<int>("tumble", 1);
            mdsc_recruited++;
        }
        else if (r.cell_type == CELL_TYPE_MAC) {
            auto a = mac_api.newAgent();
            a.setVariable<int>("x", r.x);
            a.setVariable<int>("y", r.y);
            a.setVariable<int>("z", r.z);
            a.setVariable<int>("cell_state", r.cell_state);
            a.setVariable<int>("life", r.life);
            a.setVariable<int>("tumble", 1);
            mac_recruited++;
            if (r.cell_state == MAC_M1) mac_m1_recruited++;
            else                        mac_m2_recruited++;
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

    // Update QSP MacroProperty counters (used by QSP coupling, separate from stats)
    auto counters = FLAMEGPU->environment.getMacroProperty<int, ABM_EVENT_COUNTER_SIZE>("abm_event_counters");
    counters[ABM_COUNT_TEFF_REC] += teff_recruited;
    counters[ABM_COUNT_TH_REC]   += th_recruited;
    counters[ABM_COUNT_TREG_REC] += treg_recruited;
    counters[ABM_COUNT_MDSC_REC] += mdsc_recruited;
    counters[ABM_COUNT_MAC_REC]  += mac_recruited;

    nvtxRangePop();
}

// Mark macrophage sources based on CCL2 concentration
FLAMEGPU_HOST_FUNCTION(mark_mac_sources) {
    nvtxRangePush("Mark MAC Sources");
    if (!g_pde_solver) { nvtxRangePop(); return; }

    const int nx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int nz = FLAMEGPU->environment.getProperty<int>("grid_size_z");

    int* d_recruitment_sources = g_pde_solver->get_device_recruitment_sources_ptr();
    const float* d_ccl2 = g_pde_solver->get_device_concentration_ptr(CHEM_CCL2);

    // Get parameter
    float ec50_ccl2 = FLAMEGPU->environment.getProperty<float>("PARAM_MAC_EC50_CCL2_REC");

    // Seed: combine environment seed with step counter (salt=0x517CC1B7 for MAC)
    unsigned int base_seed = FLAMEGPU->environment.getProperty<unsigned int>("sim_seed");
    unsigned int seed = base_seed ^ (static_cast<unsigned int>(FLAMEGPU->getStepCounter()) * 2654435761u + 0x517CC1B7u);

    dim3 block(8, 8, 8);
    dim3 grid((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);

    mark_mac_sources_kernel<<<grid, block>>>(
        d_recruitment_sources, d_ccl2, d_cancer_occ, nx, ny, nz, ec50_ccl2, seed);

    cudaDeviceSynchronize();
    nvtxRangePop();
}

// ============================================================================
// Occupancy Grid: Zero the grid at the start of each step's division phase
// ============================================================================
FLAMEGPU_HOST_FUNCTION(zero_occupancy_grid) {
    nvtxRangePush("Zero Occ Grid");
    auto occ = FLAMEGPU->environment.getMacroProperty<unsigned int,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX, NUM_OCC_TYPES>("occ_grid");
    occ.zero();

    // Also reset the flat cancer occupancy, vascular tip_id, and recruitment occ arrays
    if (g_pde_solver) {
        int total_voxels = g_pde_solver->get_total_voxels();
        if (d_cancer_occ)      cudaMemset(d_cancer_occ,      0, total_voxels * sizeof(unsigned int));
        if (d_vas_tip_id_grid) cudaMemset(d_vas_tip_id_grid, 0, total_voxels * sizeof(unsigned int));
        if (d_t_occ)           cudaMemset(d_t_occ,           0, total_voxels * sizeof(unsigned int));
        if (d_mac_occ)         cudaMemset(d_mac_occ,         0, total_voxels * sizeof(unsigned int));
        if (d_mdsc_occ)        cudaMemset(d_mdsc_occ,        0, total_voxels * sizeof(unsigned int));
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
// ECM Grid: Apply decay, deposition from fibroblast density field, and clamp.
// Replaces the CPU triple-nested loop with a GPU kernel launch.
// No MacroProperty D2H/H2D — operates entirely on device arrays.
// ============================================================================
FLAMEGPU_HOST_FUNCTION(update_ecm_grid) {
    nvtxRangePush("Update ECM Grid");

    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int grid_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");

    float voxel_size_cm  = FLAMEGPU->environment.getProperty<float>("PARAM_VOXEL_SIZE_CM");
    float decay_rate     = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ECM_DECAY_RATE");
    float ecm_baseline   = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ECM_BASELINE");
    float ecm_saturation = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ECM_SATURATION");
    float release_rate   = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ECM_RELEASE_CAF");
    float tgfb_ec50      = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_CAF_EC50");
    float dt             = FLAMEGPU->environment.getProperty<float>("PARAM_SEC_PER_SLICE");
    // dt is in seconds; PARAM_FIB_ECM_DECAY_RATE is in [1/s] (QSP param converted by 1/86400)
    // HCC formula: exp(-SEC_PER_TIME_SLICE [s] * k_ECM_deg [1/s])  →  exp(-dt * decay_rate)
    float voxel_vol_cm3  = voxel_size_cm * voxel_size_cm * voxel_size_cm;

    // TGFB concentration pointer — read directly from PDE solver to avoid FLAMEGPU type issues
    // (pde_concentration_ptr_* properties are stored as unsigned long long, not uint64_t)
    const float* tgfb_ptr = g_pde_solver->get_device_concentration_ptr(CHEM_TGFB);

    dim3 block(8, 8, 8);
    dim3 grid((grid_x + 7) / 8, (grid_y + 7) / 8, (grid_z + 7) / 8);
    update_ecm_grid_kernel<<<grid, block>>>(
        d_ecm_grid, d_fib_density_field, tgfb_ptr,
        grid_x, grid_y, grid_z,
        voxel_vol_cm3, decay_rate, dt,
        ecm_baseline, ecm_saturation, release_rate,
        tgfb_ec50);
    cudaDeviceSynchronize();

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