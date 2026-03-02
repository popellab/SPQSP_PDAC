#include "pde_integration.cuh"
#include "../core/common.cuh"
#include "../core/layer_timing.h"
#include <iostream>
#include <vector>
#include <nvtx3/nvToolsExt.h>

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

// Flat device array: cancer occupancy per voxel (0 = empty, >0 = cancer present).
// Populated by cancer_write_to_occ_grid, zeroed by zero_occupancy_grid.
// Used by recruitment source-marking kernels to skip tumor-dense voxels.
static unsigned int* d_cancer_occ = nullptr;

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
}

// ============================================================================
// Recruitment System Implementation
// ============================================================================

// Helper: check if radius-3 sphere around (x,y,z) is completely filled with cancer.
// Returns true if every in-bounds voxel within radius 3 has cancer present.
__device__ bool is_tumor_dense_r3(
    const unsigned int* d_cancer_occ,
    int x, int y, int z,
    int nx, int ny, int nz)
{
    int sphere_total = 0, sphere_cancer = 0;
    for (int dz = -3; dz <= 3; dz++) {
        for (int dy = -3; dy <= 3; dy++) {
            for (int dx = -3; dx <= 3; dx++) {
                if (dx*dx + dy*dy + dz*dz > 9) continue;
                int cx = x + dx, cy = y + dy, cz = z + dz;
                if (cx < 0 || cx >= nx || cy < 0 || cy >= ny || cz < 0 || cz >= nz) continue;
                sphere_total++;
                sphere_cancer += (d_cancer_occ[cz*(nx*ny) + cy*nx + cx] > 0u) ? 1 : 0;
            }
        }
    }
    return (sphere_total > 0 && sphere_cancer >= sphere_total);
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

    // Generate random seed from step number
    unsigned int seed = static_cast<unsigned int>(FLAMEGPU->getStepCounter()) * 12345u;

    dim3 block(8, 8, 8);
    dim3 grid((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);

    mark_mdsc_sources_kernel<<<grid, block>>>(
        d_recruitment_sources, d_ccl2, d_cancer_occ, nx, ny, nz, ec50_ccl2, seed);

    cudaDeviceSynchronize();
    nvtxRangePop();
}

// Recruit T cells at marked T source voxels
FLAMEGPU_HOST_FUNCTION(recruit_t_cells) {
    nvtxRangePush("Recruit T Cells");
    if (!g_pde_solver) { nvtxRangePop(); return; }

    const int nx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int nz = FLAMEGPU->environment.getProperty<int>("grid_size_z");
    const float voxel_size = FLAMEGPU->environment.getProperty<float>("voxel_size");

    // Get QSP concentrations and recruitment rates from environment
    float qsp_teff_conc = FLAMEGPU->environment.getProperty<float>("qsp_teff_central");
    float k_teff = FLAMEGPU->environment.getProperty<float>("PARAM_TEFF_RECRUIT_K");
    float p_recruit_teff = std::min(qsp_teff_conc * k_teff, 1.0f);

    float qsp_treg_conc = FLAMEGPU->environment.getProperty<float>("qsp_treg_central");
    float k_treg = FLAMEGPU->environment.getProperty<float>("PARAM_TREG_RECRUIT_K");
    float p_recruit_treg = std::min(qsp_treg_conc * k_treg, 1.0f);

    float qsp_th_conc = FLAMEGPU->environment.getProperty<float>("qsp_th_central");
    float k_th = FLAMEGPU->environment.getProperty<float>("PARAM_TH_RECRUIT_K");
    float p_recruit_th = std::min(qsp_th_conc * k_th, 1.0f);

    // Copy recruitment sources to host
    int total_voxels = nx * ny * nz;
    std::vector<int> h_sources(total_voxels);
    int* d_sources = g_pde_solver->get_device_recruitment_sources_ptr();
    cudaMemcpy(h_sources.data(), d_sources, total_voxels * sizeof(int), cudaMemcpyDeviceToHost);

    // Get agent APIs for creating new agents
    auto tcell_api = FLAMEGPU->agent(AGENT_TCELL);
    auto treg_api = FLAMEGPU->agent(AGENT_TREG);

    int teff_recruited = 0;
    int treg_recruited = 0;
    int th_recruited = 0;

    // Count total sources available
    int total_t_sources = 0;
    for (int idx = 0; idx < total_voxels; idx++) {
        if ((h_sources[idx] & 1) != 0) total_t_sources++;
    }

    // Scan for T source voxels (bit 0 set)
    for (int idx = 0; idx < total_voxels; idx++) {
        if ((h_sources[idx] & 1) == 0) continue;  // Not a T source

        int z = idx / (nx * ny);
        int y = (idx % (nx * ny)) / nx;
        int x = idx % nx;

        // Try to recruit Teff
        if (FLAMEGPU->random.uniform<float>() < p_recruit_teff) {
            // Find empty neighbor voxel for placement (Moore neighborhood)
            bool placed = false;
            for (int dz = -1; dz <= 1 && !placed; dz++) {
                for (int dy = -1; dy <= 1 && !placed; dy++) {
                    for (int dx = -1; dx <= 1 && !placed; dx++) {
                        if (dx == 0 && dy == 0 && dz == 0) continue;

                        int nx_new = x + dx;
                        int ny_new = y + dy;
                        int nz_new = z + dz;

                        if (nx_new >= 0 && nx_new < nx &&
                            ny_new >= 0 && ny_new < ny &&
                            nz_new >= 0 && nz_new < nz) {

                            // Create new T cell (simplified initialization)
                            auto new_agent = tcell_api.newAgent();
                            new_agent.setVariable<int>("x", nx_new);
                            new_agent.setVariable<int>("y", ny_new);
                            new_agent.setVariable<int>("z", nz_new);
                            new_agent.setVariable<int>("cell_state", 0);  // Effector state

                            float lifeMean = FLAMEGPU->environment.getProperty<float>("PARAM_T_CELL_LIFE_MEAN_SLICE");

                            float rnd = static_cast<float>(rand()) / RAND_MAX;
                            int life = static_cast<int>(lifeMean * std::log(1.0f / (rnd + 0.0001f)) + 0.5f);
                            if (life < 1) life = 1;

                            new_agent.setVariable<int>("life", life);

                            new_agent.setVariable<int>("divide_cd", FLAMEGPU->environment.getProperty<int>("PARAM_TCELL_DIV_INTERNAL"));
                            new_agent.setVariable<int>("divide_limit", FLAMEGPU->environment.getProperty<int>("PARAM_TCELL_DIV_LIMIT"));

                            new_agent.setVariable<float>("IL2_release_remain", FLAMEGPU->environment.getProperty<float>("PARAM_TCELL_IL2_RELEASE_TIME"));

                            teff_recruited++;
                            placed = true;
                        }
                    }
                }
            }
        }

        // Try to recruit Treg
        if (FLAMEGPU->random.uniform<float>() < p_recruit_treg) {
            bool placed = false;
            for (int dz = -1; dz <= 1 && !placed; dz++) {
                for (int dy = -1; dy <= 1 && !placed; dy++) {
                    for (int dx = -1; dx <= 1 && !placed; dx++) {
                        if (dx == 0 && dy == 0 && dz == 0) continue;

                        int nx_new = x + dx;
                        int ny_new = y + dy;
                        int nz_new = z + dz;

                        if (nx_new >= 0 && nx_new < nx &&
                            ny_new >= 0 && ny_new < ny &&
                            nz_new >= 0 && nz_new < nz) {

                            auto new_agent = treg_api.newAgent();
                            new_agent.setVariable<int>("x", nx_new);
                            new_agent.setVariable<int>("y", ny_new);
                            new_agent.setVariable<int>("z", nz_new);
                            new_agent.setVariable<int>("cell_state", 0);

                            float lifeMean_treg = FLAMEGPU->environment.getProperty<float>("PARAM_TCD4_LIFE_MEAN_SLICE");

                            float rnd_treg = static_cast<float>(rand()) / RAND_MAX;
                            int life_treg = static_cast<int>(lifeMean_treg * std::log(1.0f / (rnd_treg + 0.0001f)) + 0.5f);
                            if (life_treg < 1) life_treg = 1;

                            new_agent.setVariable<int>("life", life_treg);

                            new_agent.setVariable<int>("divide_cd", FLAMEGPU->environment.getProperty<int>("PARAM_TCD4_DIV_INTERNAL"));
                            new_agent.setVariable<int>("divide_limit", FLAMEGPU->environment.getProperty<int>("PARAM_TCD4_DIV_LIMIT"));

                            new_agent.setVariable<float>("TGFB_release_remain", FLAMEGPU->environment.getProperty<float>("PARAM_TCD4_TGFB_RELEASE_TIME"));

                            new_agent.setVariable<float>("CTLA4", FLAMEGPU->environment.getProperty<float>("PARAM_CTLA4_TREG"));

                            treg_recruited++;
                            placed = true;
                        }
                    }
                }
            }
        }
        // Try to recruit TH
        if (FLAMEGPU->random.uniform<float>() < p_recruit_th) {
            bool placed = false;
            for (int dz = -1; dz <= 1 && !placed; dz++) {
                for (int dy = -1; dy <= 1 && !placed; dy++) {
                    for (int dx = -1; dx <= 1 && !placed; dx++) {
                        if (dx == 0 && dy == 0 && dz == 0) continue;

                        int nx_new = x + dx;
                        int ny_new = y + dy;
                        int nz_new = z + dz;

                        if (nx_new >= 0 && nx_new < nx &&
                            ny_new >= 0 && ny_new < ny &&
                            nz_new >= 0 && nz_new < nz) {

                            auto new_agent_th = treg_api.newAgent();
                            new_agent_th.setVariable<int>("x", nx_new);
                            new_agent_th.setVariable<int>("y", ny_new);
                            new_agent_th.setVariable<int>("z", nz_new);
                            new_agent_th.setVariable<int>("cell_state", 1);

                            float lifeMean_th = FLAMEGPU->environment.getProperty<float>("PARAM_TCD4_LIFE_MEAN_SLICE");

                            float rnd_th = static_cast<float>(rand()) / RAND_MAX;
                            int life_th = static_cast<int>(lifeMean_th * std::log(1.0f / (rnd_th + 0.0001f)) + 0.5f);
                            if (life_th < 1) life_th = 1;

                            new_agent_th.setVariable<int>("life", life_th);

                            new_agent_th.setVariable<int>("divide_cd", FLAMEGPU->environment.getProperty<int>("PARAM_TCD4_DIV_INTERNAL"));
                            new_agent_th.setVariable<int>("divide_limit", FLAMEGPU->environment.getProperty<int>("PARAM_TCD4_DIV_LIMIT"));

                            new_agent_th.setVariable<float>("TGFB_release_remain", FLAMEGPU->environment.getProperty<float>("PARAM_TCD4_TGFB_RELEASE_TIME"));

                            new_agent_th.setVariable<float>("CTLA4", 0.0f);

                            th_recruited++;
                            placed = true;
                        }
                    }
                }
            }
        }
    }

    // Update MacroProperty counters (will be copied to environment by copy_abm_counters_to_environment)
    auto counters = FLAMEGPU->environment.getMacroProperty<int, ABM_EVENT_COUNTER_SIZE>("abm_event_counters");
    counters[ABM_COUNT_TEFF_REC] += teff_recruited;
    counters[ABM_COUNT_TH_REC] += th_recruited;
    counters[ABM_COUNT_TREG_REC] += treg_recruited;

    // Track events: T cell recruitment (CD8 effector) and TREG recruitment
    uint64_t tcell_recruit_ptr = FLAMEGPU->environment.getProperty<uint64_t>("event_tcell_recruit_ptr");
    if (tcell_recruit_ptr != 0 && teff_recruited > 0) {
        unsigned int host_val;
        cudaMemcpy(&host_val, reinterpret_cast<unsigned int*>(tcell_recruit_ptr), sizeof(unsigned int), cudaMemcpyDeviceToHost);
        host_val += teff_recruited;
        cudaMemcpy(reinterpret_cast<unsigned int*>(tcell_recruit_ptr), &host_val, sizeof(unsigned int), cudaMemcpyHostToDevice);
    }

    // Track TREG recruitment (TREGs are recruited as TH cells, so count as TH recruitment)
    uint64_t th_recruit_ptr = FLAMEGPU->environment.getProperty<uint64_t>("event_th_recruit_ptr");
    if (th_recruit_ptr != 0 && treg_recruited > 0) {
        unsigned int host_val;
        cudaMemcpy(&host_val, reinterpret_cast<unsigned int*>(th_recruit_ptr), sizeof(unsigned int), cudaMemcpyDeviceToHost);
        host_val += treg_recruited;
        cudaMemcpy(reinterpret_cast<unsigned int*>(th_recruit_ptr), &host_val, sizeof(unsigned int), cudaMemcpyHostToDevice);
    }
    nvtxRangePop();
}

// Recruit MDSCs at marked MDSC source voxels
FLAMEGPU_HOST_FUNCTION(recruit_mdscs) {
    nvtxRangePush("Recruit MDSCs");
    if (!g_pde_solver) { nvtxRangePop(); return; }

    const int nx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int nz = FLAMEGPU->environment.getProperty<int>("grid_size_z");

    // Calculate recruitment probability: p = min(concentration * k_recruit, 1.0)
    float p_recruit_mdsc = FLAMEGPU->environment.getProperty<float>("PARAM_MDSC_RECRUIT_K");

    int total_voxels = nx * ny * nz;
    std::vector<int> h_sources(total_voxels);
    int* d_sources = g_pde_solver->get_device_recruitment_sources_ptr();
    cudaMemcpy(h_sources.data(), d_sources, total_voxels * sizeof(int), cudaMemcpyDeviceToHost);

    // Get agent API for creating new agents
    auto mdsc_api = FLAMEGPU->agent(AGENT_MDSC);
    int mdsc_recruited = 0;

    // Scan for MDSC source voxels (bit 1 set)
    for (int idx = 0; idx < total_voxels; idx++) {
        if ((h_sources[idx] & 2) == 0) continue;  // Not an MDSC source

        if (FLAMEGPU->random.uniform<float>() < p_recruit_mdsc) {
            int z = idx / (nx * ny);
            int y = (idx % (nx * ny)) / nx;
            int x = idx % nx;

            // Find empty neighbor voxel
            bool placed = false;
            for (int dz = -1; dz <= 1 && !placed; dz++) {
                for (int dy = -1; dy <= 1 && !placed; dy++) {
                    for (int dx = -1; dx <= 1 && !placed; dx++) {
                        if (dx == 0 && dy == 0 && dz == 0) continue;

                        int nx_new = x + dx;
                        int ny_new = y + dy;
                        int nz_new = z + dz;

                        if (nx_new >= 0 && nx_new < nx &&
                            ny_new >= 0 && ny_new < ny &&
                            nz_new >= 0 && nz_new < nz) {

                            auto new_agent = mdsc_api.newAgent();
                            new_agent.setVariable<int>("x", nx_new);
                            new_agent.setVariable<int>("y", ny_new);
                            new_agent.setVariable<int>("z", nz_new);

                            double lifeMean = FLAMEGPU->environment.getProperty<float>("PARAM_MDSC_LIFE_MEAN_SLICE");
                            float rnd = static_cast<float>(rand()) / RAND_MAX;
                            int life = static_cast<int>(lifeMean * std::log(1.0f / (rnd + 0.0001f)) + 0.5f);
                            if (life < 1) life = 1;

                            new_agent.setVariable<int>("life", life);

                            mdsc_recruited++;
                            placed = true;
                        }
                    }
                }
            }
        }
    }

    // Update MacroProperty counters (will be copied to environment by copy_abm_counters_to_environment)
    auto counters = FLAMEGPU->environment.getMacroProperty<int, ABM_EVENT_COUNTER_SIZE>("abm_event_counters");
    counters[ABM_COUNT_MDSC_REC] += mdsc_recruited;
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

    // Generate random seed from step number
    unsigned int seed = static_cast<unsigned int>(FLAMEGPU->getStepCounter()) * 12345u + 54321u;

    dim3 block(8, 8, 8);
    dim3 grid((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);

    mark_mac_sources_kernel<<<grid, block>>>(
        d_recruitment_sources, d_ccl2, d_cancer_occ, nx, ny, nz, ec50_ccl2, seed);

    cudaDeviceSynchronize();
    nvtxRangePop();
}

// Recruit macrophages at marked macrophage source voxels
FLAMEGPU_HOST_FUNCTION(recruit_macrophages) {
    nvtxRangePush("Recruit MACs");
    if (!g_pde_solver) { nvtxRangePop(); return; }

    const int nx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int nz = FLAMEGPU->environment.getProperty<int>("grid_size_z");

    // Calculate recruitment probability
    float p_recruit_mac = FLAMEGPU->environment.getProperty<float>("PARAM_MAC_RECRUIT_K");

    int total_voxels = nx * ny * nz;
    std::vector<int> h_sources(total_voxels);
    int* d_sources = g_pde_solver->get_device_recruitment_sources_ptr();
    cudaMemcpy(h_sources.data(), d_sources, total_voxels * sizeof(int), cudaMemcpyDeviceToHost);

    // Get agent API for creating new agents
    auto mac_api = FLAMEGPU->agent(AGENT_MACROPHAGE);
    int mac_recruited = 0;

    // Scan for macrophage source voxels (bit 2 set)
    for (int idx = 0; idx < total_voxels; idx++) {
        if ((h_sources[idx] & 4) == 0) continue;  // Not a macrophage source

        if (FLAMEGPU->random.uniform<float>() < p_recruit_mac) {
            int z = idx / (nx * ny);
            int y = (idx % (nx * ny)) / nx;
            int x = idx % nx;

            // Find empty neighbor voxel
            bool placed = false;
            for (int dz = -1; dz <= 1 && !placed; dz++) {
                for (int dy = -1; dy <= 1 && !placed; dy++) {
                    for (int dx = -1; dx <= 1 && !placed; dx++) {
                        if (dx == 0 && dy == 0 && dz == 0) continue;

                        int nx_new = x + dx;
                        int ny_new = y + dy;
                        int nz_new = z + dz;

                        if (nx_new >= 0 && nx_new < nx &&
                            ny_new >= 0 && ny_new < ny &&
                            nz_new >= 0 && nz_new < nz) {

                            auto new_agent = mac_api.newAgent();
                            new_agent.setVariable<int>("x", nx_new);
                            new_agent.setVariable<int>("y", ny_new);
                            new_agent.setVariable<int>("z", nz_new);

                            // Recruit as M1 state
                            int cell_state = MAC_M1;

                            // 30% chance to become M2
                            if (FLAMEGPU->random.uniform<float>() < 0.3f) {
                                cell_state = MAC_M2;
                            }
                            new_agent.setVariable<int>("cell_state", cell_state);

                            // Set lifespan
                            double lifeMean = FLAMEGPU->environment.getProperty<float>("PARAM_MAC_LIFE_MEAN");
                            float rnd = static_cast<float>(rand()) / RAND_MAX;
                            int life = static_cast<int>(lifeMean * std::log(1.0f / (rnd + 0.0001f)) + 0.5f);
                            if (life < 1) life = 1;

                            new_agent.setVariable<int>("life", life);

                            mac_recruited++;
                            placed = true;
                        }
                    }
                }
            }
        }
    }

    // Update MacroProperty counters (will be copied to environment by copy_abm_counters_to_environment)
    auto counters = FLAMEGPU->environment.getMacroProperty<int, ABM_EVENT_COUNTER_SIZE>("abm_event_counters");
    counters[ABM_COUNT_MAC_REC] += mac_recruited;
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

    // Also reset the flat cancer occupancy array used by recruitment kernels
    if (d_cancer_occ && g_pde_solver) {
        int total_voxels = g_pde_solver->get_total_voxels();
        cudaMemset(d_cancer_occ, 0, total_voxels * sizeof(unsigned int));
    }
    nvtxRangePop();
}

// ============================================================================
// Zero Fibroblast Density Field (reset before scatter)
// ============================================================================
FLAMEGPU_HOST_FUNCTION(zero_fib_density_field) {
    nvtxRangePush("Zero Fib Density");
    auto field = FLAMEGPU->environment.getMacroProperty<float,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX>("fib_density_field");
    field.zero();
    nvtxRangePop();
}

// ============================================================================
// ECM Grid: Apply decay, deposition from fibroblast density field, and clamp
// ============================================================================
FLAMEGPU_HOST_FUNCTION(update_ecm_grid) {
    nvtxRangePush("Update ECM Grid");
    auto ecm = FLAMEGPU->environment.getMacroProperty<float,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX>("ecm_grid");
    auto field = FLAMEGPU->environment.getMacroProperty<float,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX>("fib_density_field");

    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int grid_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");

    float voxel_size_cm = FLAMEGPU->environment.getProperty<float>("PARAM_VOXEL_SIZE_CM");
    // ECM parameters
    float decay_rate = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ECM_DECAY_RATE");
    float ecm_baseline = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ECM_BASELINE");
    float ecm_saturation = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ECM_SATURATION");
    float release_rate = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ECM_RELEASE_CAF");
    float dt_sec = FLAMEGPU->environment.getProperty<float>("PARAM_SEC_PER_SLICE");
    float dt = dt_sec / 86400.0f;  // seconds → days

    // ============================================================================
    // Apply ECM dynamics: decay + deposition from Gaussian density field + saturation
    // ============================================================================
    for (int i = 0; i < grid_x; i++) {
        for (int j = 0; j < grid_y; j++) {
            for (int k = 0; k < grid_z; k++) {
                float curr_ecm = ecm[i][j][k];
                float curr_ecm_amt = curr_ecm * std::pow(voxel_size_cm, 3);

                // Exponential decay: ECM_n = ECM_{n-1} * exp(-decay_rate * dt)
                float decayed = curr_ecm_amt * expf(-decay_rate * dt);

                // Deposition from Gaussian density field
                float fib_field_val = field[i][j][k];
                float saturation = fminf(curr_ecm / ecm_saturation, 1.0f);
                float deposition = fib_field_val * release_rate / 3.0f * (1.0f - saturation);

                float new_ecm = (decayed + deposition) / std::pow(voxel_size_cm, 3);

                // Enforce bounds [baseline, saturation]
                if (new_ecm < ecm_baseline) new_ecm = ecm_baseline;
                if (new_ecm > ecm_saturation) new_ecm = ecm_saturation;

                ecm[i][j][k] = new_ecm;
            }
        }
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
// Fibroblast HEAD Division: Create 2 new HEAD cells and convert chain to CAF
// ============================================================================
// DISABLED 2026-02-24: Fibroblast division function causes FLAME GPU device memory corruption
// Issue: Calling fib_api.newAgent() in a host function corrupts DeviceAgentVector state
// This causes cudaErrorInvalidValue on subsequent steps
// TODO: Either rewrite to avoid newAgent() calls, or use different mechanism for fibroblast growth
FLAMEGPU_HOST_FUNCTION(fib_execute_divide) {
    // DISABLED - fibroblast division was not functioning correctly and caused CUDA crashes
    // Fibroblasts will persist but not divide
    return;
}

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

} // namespace PDAC