#include "pde_integration.cuh"
#include "../core/common.cuh"
#include "../core/layer_timing.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <nvtx3/nvToolsExt.h>
#include <cmath>

// ── Class-mapping audit (contract §4.4 / gates G3-G4) ───────────────────────────
// Per-step record of the ABM population in each class against its QSP target f·X_QSP.
// Written from recruit_systematic, where f and the qsp_* env properties are already in
// hand. Separate from stats_*.csv on purpose: that file is pure ABM state, this one is
// the ABM↔QSP comparison, and the two have different consumers.
static std::ofstream g_class_audit_csv;
static std::string   g_class_audit_path = "outputs/class_audit.csv";
static std::ofstream g_phi_csv;
static std::string   g_phi_csv_path = "outputs/phi.csv";

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

// Redirect the class-mapping audit CSV (mirrors set_qsp_output_path; called from main
// once the output root is known). Must be called before the first step.
void set_class_audit_path(const std::string& path) { g_class_audit_path = path; }
void set_phi_csv_path(const std::string& path) { g_phi_csv_path = path; }

// Defined in qsp_integration.cu — the audit tags presim rows so the two phases, whose
// step counters both start at 0, stay separable in one file.
extern bool is_presim_mode_active();

// Flat device array: cancer occupancy per voxel (0 = empty, >0 = cancer present).
// Populated by cancer_write_to_occ_grid, zeroed by zero_occupancy_grid.
// Used by recruitment source-marking kernels to skip tumor-dense voxels.
static unsigned int* d_cancer_occ = nullptr;

// Per-voxel vascular volume fraction field Vvas = Hill(VEGF)*(1-f_cancer_moore)*cabo_term
// (media-2 Eq.1). Drives O2 delivery (Krogh) + immune recruitment entry points,
// replacing explicit vascular agents. Computed each step by compute_vvas_and_o2.
static float* d_vvas_field = nullptr;

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
static float* d_ecm_floor = nullptr;       // Per-voxel density floor (passive decay clamped here; MMP bypasses)
static unsigned long long* d_fib_density_field = nullptr;  // fixed-point ECM density (ecm_fp_encode/decode)

// ── phi estimator state (contract D-1/D-2). See common.cuh for the statistic. ──
static unsigned long long* d_phi_kde   = nullptr;  // [N_PHI_KDE][nvox] fixed-point KDE fields
static unsigned long long* d_phi_accum = nullptr;  // [N_PHI_PAIR] fixed-point SUM of edge-corrected ratios
static unsigned int*       d_phi_count = nullptr;  // [N_PHI_PAIR targets | N_PHI_KDE sources]
static float*              d_phi_sumk  = nullptr;  // [nvox] SumK_local — geometry only, computed once
static double              g_phi[N_PHI_PAIR] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
static bool                g_phi_valid[N_PHI_PAIR] = {};

// SumK_local(v) = SUM over in-domain offsets of K(v-u). Pure geometry (see common.cuh edge note),
// so this runs once at init rather than per step.
__global__ void phi_sumk_kernel(float* sumk, int gx, int gy, int gz, float sigma, int radius) {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    const int nvox = gx * gy * gz;
    if (v >= nvox) return;
    const int x = v % gx;
    const int y = (v / gx) % gy;
    const int z = v / (gx * gy);
    const float two_s2 = 2.0f * sigma * sigma;
    float s = 0.0f;
    for (int dz = -radius; dz <= radius; ++dz) {
        if (z + dz < 0 || z + dz >= gz) continue;
        for (int dy = -radius; dy <= radius; ++dy) {
            if (y + dy < 0 || y + dy >= gy) continue;
            for (int dx = -radius; dx <= radius; ++dx) {
                if (x + dx < 0 || x + dx >= gx) continue;
                s += phi_kernel(dx, dy, dz, two_s2);
            }
        }
    }
    sumk[v] = s;
}

void init_phi_sumk(int gx, int gy, int gz, float sigma, int radius) {
    const int nvox = gx * gy * gz;
    const int threads = 256;
    phi_sumk_kernel<<<(nvox + threads - 1) / threads, threads>>>(
        d_phi_sumk, gx, gy, gz, sigma, radius);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "  phi estimator: sigma=" << sigma << " vox, radius=" << radius
              << " vox, SumK_local precomputed over " << nvox << " voxels" << std::endl;
}

double get_phi(int pair) { return (pair >= 0 && pair < N_PHI_PAIR) ? g_phi[pair] : 1.0; }
bool   phi_is_valid(int pair) { return (pair >= 0 && pair < N_PHI_PAIR) && g_phi_valid[pair]; }

// Volume-based occupancy: single float per voxel tracking total cell volume (µm³).
// Replaces the old per-type occ_grid MacroProperty + flat arrays (d_t_occ, d_mac_occ, d_mdsc_occ).
static float* d_volume_used = nullptr;

// Systematic-resampling recruitment: per-voxel expected-recruitment field, N_QPLANE planes
// (8 stream q-fields + 1 mac M1/M2 polarization plane), plane-major. Filled each step by
// compute_qfields_kernel, read host-side by recruit_systematic. Allocated in
// initialize_pde_solver alongside d_vvas_field.
static float* d_qfields = nullptr;

// ── Systematic-resampling recruitment: q-field layout ──
// One expected-count plane per recruitment stream + one M1/M2 polarization plane, packed
// plane-major into a single [N_QPLANE*nvox] buffer for a single D2H.
//   q_type[v] = p_entry[v] · gate_type[v]   (expected recruits of `type` at voxel v)
// Replaces the two independent Bernoulli rolls (mark_entry_points ⊗ recruit_all_kernel): their
// product is exactly this expected field, so the mean rate is unchanged and the per-voxel
// Bernoulli realization noise (the ICC-killing term) is removed. See
// project-recruitment-systematic-resampling.
enum QStream : int {
    QS_TEFF = 0, QS_TREG, QS_TH, QS_MDSC, QS_MAC, QS_BCELL, QS_DC1, QS_DC2,
    N_QSTREAM        // = 8
};
constexpr int QPLANE_PM2 = N_QSTREAM;          // plane 8: mac M1/M2 spawn polarization P(M2)
constexpr int N_QPLANE   = N_QSTREAM + 1;      // = 9

// Deterministic voxel-claim ownership buffer (Step-5 determinism). One uint64 per voxel,
// reset to ULLONG_MAX before each reserve pass; contenders atomicMin their packed
// (source-voxel, agent-id) priority; the confirm pass commits the unique winner.
static unsigned long long* d_voxel_owner = nullptr;

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
    const float* ecm_floor,
    const unsigned long long* fib_field, const float* tgfb_conc, const float* mmp_conc,
    int nx, int ny, int nz,
    float voxel_vol_cm3, float dt,
    float k_decay, float k_depo, float density_cap,
    float tgfb_ec50,
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
    float fib       = ecm_fp_decode(fib_field[idx]);  // fixed-point → float (deterministic)
    float tgfb      = tgfb_conc[idx];
    float mmp       = mmp_conc[idx];

    // 1. Exponential passive decay
    float density_amt = density * voxel_vol_cm3;
    density_amt *= expf(-k_decay * dt);
    density = density_amt / voxel_vol_cm3;

    // 2. Per-voxel floor (catches passive decay only — septa stay stable without MMP)
    density = fmaxf(density, ecm_floor[idx]);

    // 3. myCAF deposition (TGF-β gated, saturation-limited, YAP/TAZ Hill ceiling)
    float H_TGFB = tgfb / (tgfb + tgfb_ec50 + 1e-30f);
    float sat_frac = fminf(density / density_cap, 1.0f);
    float stiffness = density * (1.0f + crosslink);
    float yap_factor = 1.0f / (1.0f + (stiffness * stiffness) / (yap_ec50 * yap_ec50 + 1e-30f));
    float deposition = fib * (1.0f + H_TGFB) * k_depo / 3.0f * (1.0f - sat_frac) * yap_factor * dt;
    density += deposition / voxel_vol_cm3;

    // 4. MMP degradation (crosslink-resistant, can push below per-voxel floor)
    float mmp_degrade = k_mmp * mmp * density / (1.0f + alpha_crosslink * crosslink) * dt;
    density -= mmp_degrade;

    // 5. Hard non-negative floor + density cap
    density = fmaxf(density, 0.0f);
    density = fminf(density, density_cap);
    ecm_density[idx] = density;

    // 6. Crosslink accumulation (LOX from myCAFs, saturates at 1.0)
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
// CUDA Kernel: Background O2 floor. Applied post-solve so every voxel holds at
// least a baseline dissolved-O2 level (diffuse background perfusion that we do
// NOT model as vascular agents). Prevents the whole avascular presim tumour from
// sitting below the ~0.01 mM hypoxia thresholds and switching the HIF cascade on
// globally (M1->M2 forcing, DC-maturation suppression, PDL1 up / MHC-I down).
// Krogh Vvas delivery still adds O2 on top where vasculature is present, since
// the floor (~0.02 mM) is below C_blood (PARAM_VAS_O2_CONC ~0.065 mM).
// ============================================================================
__global__ void floor_o2_kernel(float* o2_conc, int total_voxels, float background)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_voxels && o2_conc[idx] < background) o2_conc[idx] = background;
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
    const unsigned long long* fib_field, const float* ecm_crosslink,
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
    float fib = ecm_fp_decode(fib_field[idx]);  // fixed-point → float (deterministic)
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
    int mode = g_pde_solver->get_solve_mode();
    if (mode == 2) {
        // Quasi-steady-state via spectral (FFT/DCT) solve — exact elliptic field
        g_pde_solver->solve_spectral();
    } else if (mode == 1) {
        // Quasi-steady-state via coupled CG (exact elliptic; slow reference/oracle)
        g_pde_solver->solve_steadystate();
    } else {
        // Transient LOD: relax toward steady state over `substeps` substeps
        for (int i = 0; i < substeps; i++) {
            g_pde_solver->solve_timestep();
        }
    }
    // Background O2 floor: hold every voxel at >= PARAM_O2_BACKGROUND so the avascular
    // presim tumour never falls below the hypoxia thresholds and drives a global HIF
    // cascade. Disabled when the param is 0 (its fallback if absent from the XML), so
    // this is a no-op for unmodified parameter files.
    const float o2_background = FLAMEGPU->environment.getProperty<float>("PARAM_O2_BACKGROUND");
    if (o2_background > 0.0f) {
        const int nx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
        const int ny = FLAMEGPU->environment.getProperty<int>("grid_size_y");
        const int nz = FLAMEGPU->environment.getProperty<int>("grid_size_z");
        const int total_voxels = nx * ny * nz;
        float* o2_conc = g_pde_solver->get_device_concentration_ptr(CHEM_O2);
        const int threads = 256;
        const int blocks = (total_voxels + threads - 1) / threads;
        floor_o2_kernel<<<blocks, threads>>>(o2_conc, total_voxels, o2_background);
        cudaDeviceSynchronize();
    }

    auto pde_t1 = std::chrono::high_resolution_clock::now();
    g_last_pde_ms = std::chrono::duration<double, std::milli>(pde_t1 - pde_t0).count();

    unsigned int step = FLAMEGPU->environment.getProperty<unsigned int>("current_step");
    if (step % 50 == 0) {
        std::cout << "PDE solved for step " << step
                  << " (mode=" << g_pde_solver->get_solve_mode() << ")" << std::endl;
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

    // PDE solve mode + steady-state CG controls (quasi-steady-state path)
    config.solve_mode = gpu_params.getInt(PARAM_PDE_SOLVE_MODE);
    config.cg_tol     = gpu_params.getFloat(PARAM_PDE_CG_TOL);
    config.cg_maxiter = gpu_params.getInt(PARAM_PDE_CG_MAXITER);
    if (config.cg_tol     <= 0.0f) config.cg_tol = 1e-5f;   // guard against unset XML
    if (config.cg_maxiter <= 0)    config.cg_maxiter = 500;

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
    config.diffusion_coeffs[CHEM_CXCL9_10] = 1.47e-7f;  // small chemokine ~10-12 kDa (≈CXCL13); promote to PARAM if SBI needs it

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
    config.decay_rates[CHEM_CXCL9_10] = 1e-4f;  // ABM-only chemokine decay (≈CXCL13); promote to PARAM if SBI needs it

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

    // Vvas field (vascular volume fraction; media-2). Persists across steps; recomputed each step.
    CUDA_CHECK(cudaMalloc(&d_vvas_field, total_voxels * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_vvas_field, 0, total_voxels * sizeof(float)));

    // Systematic-resampling q-fields: N_QPLANE planes (8 stream expected-recruitment fields +
    // 1 mac M1/M2 polarization plane), plane-major. Recomputed each step by compute_qfields_kernel.
    CUDA_CHECK(cudaMalloc(&d_qfields, static_cast<size_t>(N_QPLANE) * total_voxels * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_qfields, 0, static_cast<size_t>(N_QPLANE) * total_voxels * sizeof(float)));

    // Allocate vascular tip_id grid for efficient neighbor check in vascular_state_step
    CUDA_CHECK(cudaMalloc(&d_vas_tip_id_grid, total_voxels * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_vas_tip_id_grid, 0, total_voxels * sizeof(unsigned int)));

    // Allocate ECM and fibroblast density field device arrays.
    // ECM starts at 0 here; preseed_ecm() sets initial values per region.
    CUDA_CHECK(cudaMalloc(&d_ecm_density, total_voxels * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_ecm_density, 0, total_voxels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ecm_crosslink, total_voxels * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_ecm_crosslink, 0, total_voxels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ecm_floor, total_voxels * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_ecm_floor, 0, total_voxels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fib_density_field, total_voxels * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_fib_density_field, 0, total_voxels * sizeof(unsigned long long)));

    // ── phi estimator buffers (contract D-1/D-2) ─────────────────────────────────────
    CUDA_CHECK(cudaMalloc(&d_phi_kde, static_cast<size_t>(N_PHI_KDE) * total_voxels
                                      * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_phi_kde, 0, static_cast<size_t>(N_PHI_KDE) * total_voxels
                                        * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_phi_accum, N_PHI_PAIR * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_phi_accum, 0, N_PHI_PAIR * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_phi_count, (N_PHI_PAIR + N_PHI_KDE) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_phi_count, 0, (N_PHI_PAIR + N_PHI_KDE) * sizeof(unsigned int)));
    // SumK_local is pure geometry — computed ONCE here, never per step.
    CUDA_CHECK(cudaMalloc(&d_phi_sumk, total_voxels * sizeof(float)));
    init_phi_sumk(grid_x, grid_y, grid_z,
                  gpu_params.getFloat(PARAM_PHI_SIGMA_VOX),
                  gpu_params.getInt(PARAM_PHI_RADIUS_VOX));

    // Allocate volume-based occupancy grid
    CUDA_CHECK(cudaMalloc(&d_volume_used, total_voxels * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_volume_used, 0, total_voxels * sizeof(float)));

    // Allocate deterministic voxel-claim ownership buffer (init to ULLONG_MAX = 0xFF bytes)
    CUDA_CHECK(cudaMalloc(&d_voxel_owner, total_voxels * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_voxel_owner, 0xFF, total_voxels * sizeof(unsigned long long)));

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

        // Agents write to the fixed-point STAGING buffers (deterministic atomicAdd of scaled
        // ints); decode_pde_sources adds the decoded sum onto d_src_/d_upt_ before the solve.
        uintptr_t conc_ptr = reinterpret_cast<uintptr_t>(g_pde_solver->get_device_concentration_ptr(sub));
        uintptr_t src_ptr  = reinterpret_cast<uintptr_t>(g_pde_solver->get_device_source_fp_ptr(sub));
        uintptr_t upt_ptr  = reinterpret_cast<uintptr_t>(g_pde_solver->get_device_uptake_fp_ptr(sub));

        model.Environment().newProperty<unsigned long long>(concentration_key, static_cast<unsigned long long>(conc_ptr));
        model.Environment().newProperty<unsigned long long>(source_key,        static_cast<unsigned long long>(src_ptr));
        model.Environment().newProperty<unsigned long long>(uptake_key,        static_cast<unsigned long long>(upt_ptr));
    }

    // Store gradient pointers for chemotaxis substrates
    // Naming: pde_grad_IFN_x, pde_grad_IFN_y, pde_grad_IFN_z, pde_grad_TGFB_x, ...
    static const char* grad_names[NUM_GRAD_SUBSTRATES] = {"IFN", "TGFB", "CCL2", "VEGFA", "CXCL13", "CCL21", "CXCL12", "CCL5"};
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

    // Vvas field pointer (read by recruitment marking; media-2 entry points).
    model.Environment().newProperty<unsigned long long>("vvas_ptr",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(d_vvas_field)));

    // Store vascular tip_id grid pointer (for efficient neighbor check in vascular_state_step)
    model.Environment().newProperty<unsigned long long>("vas_tip_id_grid_ptr",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(d_vas_tip_id_grid)));

    // Store ECM and fibroblast density field pointers (replace MacroProperty approach)
    model.Environment().newProperty<unsigned long long>("ecm_density_ptr",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(d_ecm_density)));
    model.Environment().newProperty<unsigned long long>("ecm_crosslink_ptr",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(d_ecm_crosslink)));
    model.Environment().newProperty<unsigned long long>("ecm_floor_ptr",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(d_ecm_floor)));
    model.Environment().newProperty<unsigned long long>("fib_density_field_ptr",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(d_fib_density_field)));

    // phi estimator buffers
    model.Environment().newProperty<unsigned long long>("phi_kde_ptr",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(d_phi_kde)));
    model.Environment().newProperty<unsigned long long>("phi_accum_ptr",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(d_phi_accum)));
    model.Environment().newProperty<unsigned long long>("phi_count_ptr",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(d_phi_count)));
    model.Environment().newProperty<unsigned long long>("phi_sumk_ptr",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(d_phi_sumk)));

    // Volume-based occupancy grid pointer
    model.Environment().newProperty<unsigned long long>("volume_used_ptr",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(d_volume_used)));

    // Deterministic voxel-claim ownership buffer pointer
    model.Environment().newProperty<unsigned long long>("voxel_owner_ptr",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(d_voxel_owner)));

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
    if (d_vvas_field) {
        cudaFree(d_vvas_field);
        d_vvas_field = nullptr;
    }
    if (d_qfields) {
        cudaFree(d_qfields);
        d_qfields = nullptr;
    }
    if (d_ecm_density) {
        cudaFree(d_ecm_density);
        d_ecm_density = nullptr;
    }
    if (d_ecm_crosslink) {
        cudaFree(d_ecm_crosslink);
        d_ecm_crosslink = nullptr;
    }
    if (d_ecm_floor) {
        cudaFree(d_ecm_floor);
        d_ecm_floor = nullptr;
    }
    if (d_fib_density_field) {
        cudaFree(d_fib_density_field);
        d_fib_density_field = nullptr;
    }
    if (d_phi_kde)   { cudaFree(d_phi_kde);   d_phi_kde = nullptr; }
    if (d_phi_accum) { cudaFree(d_phi_accum); d_phi_accum = nullptr; }
    if (d_phi_count) { cudaFree(d_phi_count); d_phi_count = nullptr; }
    if (d_phi_sumk)  { cudaFree(d_phi_sumk);  d_phi_sumk = nullptr; }
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

void initialize_ecm_floor_uniform(float baseline) {
    if (!d_ecm_floor || !g_pde_solver) return;
    int total_voxels = g_pde_solver->get_total_voxels();
    int block_size = 256;
    int grid_size = (total_voxels + block_size - 1) / block_size;
    fill_ecm_kernel<<<grid_size, block_size>>>(d_ecm_floor, total_voxels, baseline);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void set_ecm_density_from_host(const float* host_data, int total_voxels) {
    CUDA_CHECK(cudaMemcpy(d_ecm_density, host_data, total_voxels * sizeof(float), cudaMemcpyHostToDevice));
}

void set_ecm_crosslink_from_host(const float* host_data, int total_voxels) {
    CUDA_CHECK(cudaMemcpy(d_ecm_crosslink, host_data, total_voxels * sizeof(float), cudaMemcpyHostToDevice));
}

void set_ecm_floor_from_host(const float* host_data, int total_voxels) {
    CUDA_CHECK(cudaMemcpy(d_ecm_floor, host_data, total_voxels * sizeof(float), cudaMemcpyHostToDevice));
}

void set_ecm_orient_from_host(const float* ox, const float* oy, const float* oz, int total_voxels) {
    CUDA_CHECK(cudaMemcpy(d_ecm_orient_x, ox, total_voxels * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ecm_orient_y, oy, total_voxels * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ecm_orient_z, oz, total_voxels * sizeof(float), cudaMemcpyHostToDevice));
}

float* get_ecm_density_device_ptr() { return d_ecm_density; }
float* get_vvas_device_ptr() { return d_vvas_field; }

// Set true by set_vvas_from_host so compute_vvas_and_o2 does not overwrite a placement seed
// with the uniform PARAM_VAS_INIT fallback. File-scope rather than a static local because the
// seeding call and the consuming call are in different translation-unit positions.
static bool g_vvas_seeded = false;

bool set_vvas_from_host(const float* host_data, int total_voxels) {
    if (!d_vvas_field) {
        std::cerr << "ERROR: set_vvas_from_host called before initialize_pde_solver — "
                     "V_vas seed DROPPED. Placement V_vas will fall back to PARAM_VAS_INIT."
                  << std::endl;
        return false;
    }
    CUDA_CHECK(cudaMemcpy(d_vvas_field, host_data, total_voxels * sizeof(float),
                          cudaMemcpyHostToDevice));
    g_vvas_seeded = true;
    return true;
}
float* get_ecm_crosslink_device_ptr() { return d_ecm_crosslink; }
unsigned long long* get_fib_density_field_device_ptr() { return d_fib_density_field; }
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
    // MAC recruitment + probabilistic M1/M2 polarization at spawn
    float p_mac;
    float mac_life_mean;
    const float* mac_ifng_conc;   // IFN-γ PDE concentration (M1 driver)
    const float* mac_il12_conc;   // IL-12 PDE concentration (M1 driver)
    const float* mac_il10_conc;   // IL-10 PDE concentration (M2 driver)
    const float* mac_tgfb_conc;   // TGF-β PDE concentration (M2 driver)
    // mac_il6_conc reuses il6_conc pointer above (same PDE field)
    float mac_ifng_ec50, mac_il12_ec50;
    float mac_il10_ec50, mac_tgfb_ec50, mac_il6_ec50;
    // B cell recruitment
    float p_bcell;
    float bcell_life_mean, bcell_life_sd;
    const float* cxcl13_conc;  // CXCL13 PDE concentration (B cell recruitment boost)
    float cxcl13_ec50_bcell;   // CXCL13 EC50 for B cell recruitment boost
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
    // CXCL9/10 (CXCR3 ligand) positive effector (Teff/Th) recruitment gate — CAF-sourced
    const float* cxcl9_conc;
    float cxcl9_ec50;
    // CCL2 (CCR2) primary myeloid recruitment gate — MDSC + MAC (moved from marking to roll)
    const float* ccl2_conc;
    float ccl2_ec50_mdsc;
    float ccl2_ec50_mac;
    // Volume-based occupancy
    float voxel_capacity;
    float vol_teff, vol_treg, vol_th;
    float vol_mdsc, vol_mac_m1, vol_mac_m2;
    float vol_bcell;
    // Naive T cell recruitment (DC priming pathway)
    float naive_cd8_frac;     // Fraction of Teff events that also spawn a naive CD8
    float naive_cd4_frac;     // Fraction of TH events that also spawn a naive CD4
    int naive_lifespan;       // Lifespan in ABM steps for naive cells
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

// Device helper: claim the cell's OWN (marked/source) voxel (sx, sy, sz) — no Moore
// neighbor search. The recruit kernel runs one thread per voxel, so each voxel's
// volume-claim has a SINGLE writer per kernel launch → race-free and deterministic at
// fixed seed (Step-5 determinism fix: the old neighbor-search + cross-thread atomicAdd
// claim into shared neighbors was the immune-cell placement race). Recruits now land at
// the entry voxel where they extravasate; movement spreads them afterward.
// rng is retained for call-site signature compatibility but no longer consumed (placement
// is deterministic), so the RNG stream is unchanged by placement.
// Returns true + sets (out_x, out_y, out_z) on success; false if the voxel is full.
__device__ bool try_claim_voxel(
    int sx, int sy, int sz,
    int nx, int ny, int nz,
    float* d_vol_used,
    float cell_volume,
    float voxel_capacity,
    unsigned int& rng,
    int& out_x, int& out_y, int& out_z)
{
    (void)rng;  // placement no longer consumes RNG
    if (sx < 0 || sx >= nx || sy < 0 || sy >= ny || sz < 0 || sz >= nz) return false;

    int vidx = sz * (nx * ny) + sy * nx + sx;

    // Volume-based occupancy check (primary gate)
    if (d_vol_used[vidx] + cell_volume > voxel_capacity) return false;

    // Volume claim (single writer per voxel in this kernel → deterministic; atomicAdd
    // kept for safety/consistency with the multi-cell-type sequential claims in one thread).
    float old_vol = atomicAdd(&d_vol_used[vidx], cell_volume);
    if (old_vol + cell_volume > voxel_capacity) {
        atomicAdd(&d_vol_used[vidx], -cell_volume);  // undo overcommit
        return false;
    }

    out_x = sx; out_y = sy; out_z = sz;
    return true;
}

// CXCL13-independent B-cell seeding floor for the soft TLS gate (see BCell block).
// Bootstrap constant: CXCL13 is Tfh/activated-B-derived only, so a pure CXCL13 gate
// deadlocks cold-start. floor ∈ (0,1]; B recruit rate = p_bcell·(floor + (1-floor)·H_cxcl13).
// Candidate SBI param (TLS-seeding rate) — promote to PARAM_* if calibration needs it.
__device__ constexpr float BCELL_CXCL13_GATE_FLOOR = 0.1f;

// CXCL9/10-independent effector (Teff/Th) seeding floor for the soft CXCR3 gate (Step 4b).
// CAFs source CXCL9/10 from ~step 0 (TGFβ-driven myCAF), so this floor mostly covers the
// brief pre-CAF window. floor ∈ (0,1]; effector rate ∝ (floor + (1-floor)·H_cxcl9).
// gate-vs-probability-roll semantics for each cytokine to be reviewed after the rebuild.
__device__ constexpr float TEFF_CXCL9_GATE_FLOOR = 0.1f;

// CCL2 (CCR2) myeloid recruitment soft-gate floor (MDSC + MAC). Gate-vs-roll review outcome:
// CCL2 promoted from a Stage-A binary marking threshold to a Stage-B soft gate, unifying the
// myeloid axis with the lymphoid pattern. CCL2 is cancer/CAF-sourced (present early → no
// bootstrap deadlock); floor=0.1 = universal floor policy. floor ∈ (0,1]; rate ∝ floor+(1-floor)·H_ccl2.
__device__ constexpr float MYELOID_CCL2_GATE_FLOOR = 0.1f;

// One thread per voxel. Writes the expected-recruitment field for every stream (no roll).
// Mirrors the per-type gate math in recruit_all_kernel exactly, minus the rng_uniform() rolls.
__global__ void compute_qfields_kernel(
    float* __restrict__ qf,                    // [N_QPLANE * nvox], plane-major
    const float* __restrict__ vvas,
    float entry_scale,
    RecruitKernelParams p)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= p.nx || y >= p.ny || z >= p.nz) return;

    const int nvox = p.nx * p.ny * p.nz;
    const int idx  = z * (p.nx * p.ny) + y * p.nx + x;

    // Single-ledger normalized scheme: this kernel writes only the SPATIAL SHAPE weight
    //   w_type(v) = p_entry(Vvas) · type-gate(v)
    // with NO magnitude (no central pool, no rate constant, no f, no clamp). The per-stream
    // magnitude is applied on the host (recruit_systematic), which rescales each plane so
    // Σ_v q = f·Φ_type (QSP mode) or = P_type·Σw (parametric mode). Because that host step
    // normalizes by Σ_v w, any uniform factor here (entry_scale) cancels — REC_PORT/voxel-vol
    // are retired. Vvas is the H_C2 spatial analog; gates mirror the QSP trafficking fluxes
    // (CXCL12 exclusion for Teff/TH; CCL2 pure Hill ± IL-6 for myeloid; none for Treg/B/DC).
    const float vv = vvas[idx];
    const float p_entry = fminf(vv * entry_scale, 1.0f);
    if (p_entry <= 0.0f) {
        for (int s = 0; s < N_QPLANE; ++s) qf[s * nvox + idx] = 0.0f;
        return;
    }

    // ── Effector (Teff/TH): CXCL12 (CXCR4) exclusion — matches QSP RF68/RF193 ──
    float cxcl12_local = (p.cxcl12_conc && p.cxcl12_ec50 > 0.0f) ? p.cxcl12_conc[idx] : 0.0f;
    float cxcl12_inhib = (p.cxcl12_ec50 > 0.0f) ? cxcl12_local / (cxcl12_local + p.cxcl12_ec50) : 0.0f;
    const float eff_gate = (1.0f - cxcl12_inhib);
    qf[QS_TEFF * nvox + idx] = p_entry * eff_gate;
    qf[QS_TH   * nvox + idx] = p_entry * eff_gate;

    // ── Treg: no cytokine gate — matches QSP RF36 (H_C2 only, carried by Vvas) ──
    qf[QS_TREG * nvox + idx] = p_entry;

    // ── MDSC: CCL2/CCR2 pure Hill × IL-6 boost — matches QSP RF241 ──
    float mdsc_gate = 1.0f;
    if (p.ccl2_conc) {
        float c = p.ccl2_conc[idx];
        mdsc_gate *= c / (c + p.ccl2_ec50_mdsc + 1e-30f);
    }
    if (p.il6_conc) {
        float il6 = p.il6_conc[idx];
        mdsc_gate *= (1.0f + il6 / (il6 + p.il6_ec50_mdsc + 1e-30f));
    }
    qf[QS_MDSC * nvox + idx] = p_entry * mdsc_gate;

    // ── MAC: CCL2/CCR2 pure Hill — matches QSP RF276 ──
    float mac_gate = 1.0f;
    if (p.ccl2_conc) {
        float c = p.ccl2_conc[idx];
        mac_gate *= c / (c + p.ccl2_ec50_mac + 1e-30f);
    }
    qf[QS_MAC * nvox + idx] = p_entry * mac_gate;

    // ── MAC spawn polarization P(M2) — mirrors recruit_all_kernel / state-step formula ──
    //   H_M1 = H(IFN-γ)+H(IL-12);  H_M2 = H(TGF-β)+H(IL-10)+H(IL-6);  default M2 if no signal.
    float h_m1 = 0.0f, h_m2 = 0.0f;
    if (p.mac_ifng_conc) { float c = p.mac_ifng_conc[idx]; h_m1 += c / (c + p.mac_ifng_ec50 + 1e-30f); }
    if (p.mac_il12_conc) { float c = p.mac_il12_conc[idx]; h_m1 += c / (c + p.mac_il12_ec50 + 1e-30f); }
    if (p.mac_tgfb_conc) { float c = p.mac_tgfb_conc[idx]; h_m2 += c / (c + p.mac_tgfb_ec50 + 1e-30f); }
    if (p.mac_il10_conc) { float c = p.mac_il10_conc[idx]; h_m2 += c / (c + p.mac_il10_ec50 + 1e-30f); }
    if (p.il6_conc)      { float c = p.il6_conc[idx];      h_m2 += c / (c + p.mac_il6_ec50  + 1e-30f); }
    qf[QPLANE_PM2 * nvox + idx] = (h_m1 + h_m2 < 1e-9f) ? 1.0f : h_m2 / (h_m1 + h_m2);

    // ── B cell: no cytokine gate — matches QSP RF237 (q_Treg_T_in · H_C2, carried by Vvas).
    //    The former CXCL13/CXCR5 soft gate (ABM-only TLS heuristic) is removed. ──
    qf[QS_BCELL * nvox + idx] = p_entry;

    // ── DC: homeostatic per-subtype (no cytokine gate) ──
    qf[QS_DC1 * nvox + idx] = p_entry;
    qf[QS_DC2 * nvox + idx] = p_entry;
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
        // Local CXCL9/10 positive gate (CXCR3): enhances effector (Teff/TH) recruitment toward
        // CAF-rich stroma. Soft gate: floor + (1-floor)·H_cxcl9 (effector-specific; not Treg).
        float cxcl9_local = (p.cxcl9_conc && p.cxcl9_ec50 > 0.0f) ? p.cxcl9_conc[idx] : 0.0f;
        float cxcl9_gate = (p.cxcl9_ec50 > 0.0f)
            ? TEFF_CXCL9_GATE_FLOOR + (1.0f - TEFF_CXCL9_GATE_FLOOR) * (cxcl9_local / (cxcl9_local + p.cxcl9_ec50))
            : 1.0f;

        // Try Teff (CXCR3 CXCL9/10 gate × CXCL12 exclusion)
        if (rng_uniform(rng) < p.p_teff * cxcl9_gate * (1.0f - cxcl12_inhib)) {
            atomicAdd(&diag->teff_roll_pass, 1);
            if (try_claim_voxel(x, y, z, p.nx, p.ny, p.nz,
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

                // Also try to recruit a naive CD8 alongside (TLS priming pathway)
                if (p.naive_cd8_frac > 0.0f && rng_uniform(rng) < p.naive_cd8_frac) {
                    int npx, npy, npz;
                    if (try_claim_voxel(x, y, z, p.nx, p.ny, p.nz,
                            d_vol_used, p.vol_teff, p.voxel_capacity,
                            rng, npx, npy, npz)) {
                        int nslot = atomicAdd(d_request_count, 1);
                        if (nslot < MAX_RECRUITS_PER_STEP) {
                            RecruitRequest nreq;
                            nreq.x = npx; nreq.y = npy; nreq.z = npz;
                            nreq.cell_type = CELL_TYPE_T;
                            nreq.cell_state = T_CELL_NAIVE;
                            nreq.life = p.naive_lifespan;
                            nreq.divide_cd = 0;
                            nreq.divide_limit = 0;  // No division until primed
                            nreq.IL2_release_remain = 0.0f;
                            nreq.TGFB_release_remain = 0.0f;
                            nreq.CTLA4 = 0.0f;
                            d_requests[nslot] = nreq;
                        }
                    }
                }
            } else {
                atomicAdd(&diag->teff_place_fail, 1);
            }
        }

        // Try TReg (boosted by CCL5 via CCR5, ratio = k_CCR5/q_Treg)
        if (rng_uniform(rng) < fminf(p.p_treg * (1.0f + p.ccl5_ratio * ccl5_boost), 1.0f)) {
            atomicAdd(&diag->treg_roll_pass, 1);
            if (try_claim_voxel(x, y, z, p.nx, p.ny, p.nz,
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

        // Try TH (CXCR3 CXCL9/10 gate × CXCL12 exclusion, same as Teff)
        if (rng_uniform(rng) < p.p_th * cxcl9_gate * (1.0f - cxcl12_inhib)) {
            atomicAdd(&diag->th_roll_pass, 1);
            if (try_claim_voxel(x, y, z, p.nx, p.ny, p.nz,
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

                // Also try to recruit a naive CD4 alongside (TLS priming pathway)
                if (p.naive_cd4_frac > 0.0f && rng_uniform(rng) < p.naive_cd4_frac) {
                    int npx, npy, npz;
                    if (try_claim_voxel(x, y, z, p.nx, p.ny, p.nz,
                            d_vol_used, p.vol_th, p.voxel_capacity,
                            rng, npx, npy, npz)) {
                        int nslot = atomicAdd(d_request_count, 1);
                        if (nslot < MAX_RECRUITS_PER_STEP) {
                            RecruitRequest nreq;
                            nreq.x = npx; nreq.y = npy; nreq.z = npz;
                            nreq.cell_type = CELL_TYPE_TREG;  // Naive CD4 uses TReg agent type
                            nreq.cell_state = TCD4_NAIVE;
                            nreq.life = p.naive_lifespan;
                            nreq.divide_cd = 0;
                            nreq.divide_limit = 0;  // No division until primed
                            nreq.IL2_release_remain = 0.0f;
                            nreq.TGFB_release_remain = 0.0f;
                            nreq.CTLA4 = 0.0f;
                            d_requests[nslot] = nreq;
                        }
                    }
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
        // CCL2/CCR2 SOFT GATE (primary myeloid axis) — promoted from Stage-A marking to the roll
        // here, then IL-6 BOOST on top (gate-vs-roll review: CCL2=gate, IL-6=boost).
        float p_mdsc_eff = p.p_mdsc;
        if (p.ccl2_conc) {
            float c = p.ccl2_conc[idx];
            float H_ccl2 = c / (c + p.ccl2_ec50_mdsc + 1e-30f);
            p_mdsc_eff *= MYELOID_CCL2_GATE_FLOOR + (1.0f - MYELOID_CCL2_GATE_FLOOR) * H_ccl2;
        }
        if (p.il6_conc) {
            float il6 = p.il6_conc[idx];
            float H_IL6 = il6 / (il6 + p.il6_ec50_mdsc + 1e-30f);
            p_mdsc_eff *= (1.0f + H_IL6);
        }
        if (rng_uniform(rng) < p_mdsc_eff) {
            atomicAdd(&diag->mdsc_roll_pass, 1);
            if (try_claim_voxel(x, y, z, p.nx, p.ny, p.nz,
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
        // CCL2/CCR2 SOFT GATE (primary myeloid axis) — promoted from Stage-A marking to the roll.
        float p_mac_eff = p.p_mac;
        if (p.ccl2_conc) {
            float c = p.ccl2_conc[idx];
            float H_ccl2 = c / (c + p.ccl2_ec50_mac + 1e-30f);
            p_mac_eff *= MYELOID_CCL2_GATE_FLOOR + (1.0f - MYELOID_CCL2_GATE_FLOOR) * H_ccl2;
        }
        if (rng_uniform(rng) < p_mac_eff) {
            atomicAdd(&diag->mac_roll_pass, 1);
            if (try_claim_voxel(x, y, z, p.nx, p.ny, p.nz,
                    d_vol_used, p.vol_mac_m1, p.voxel_capacity,
                    rng, px, py, pz)) {
                atomicAdd(&diag->mac_place_ok, 1);
                int slot = atomicAdd(d_request_count, 1);
                if (slot < MAX_RECRUITS_PER_STEP) {
                    RecruitRequest req;
                    req.x = px; req.y = py; req.z = pz;
                    req.cell_type = CELL_TYPE_MAC;

                    // Probabilistic M1/M2 polarization at spawn — mirrors the
                    // state-step polarization formula so a freshly-recruited
                    // macrophage's initial phenotype tracks the local cytokine
                    // milieu rather than always defaulting to M1.
                    //   H_M1 = H(IFN-γ) + H(IL-12)              [M1 drivers]
                    //   H_M2 = H(TGF-β) + H(IL-10) + H(IL-6)    [M2 drivers]
                    //   P(M2) = H_M2 / (H_M1 + H_M2); default M2 if no signal
                    float h_m1 = 0.0f, h_m2 = 0.0f;
                    if (p.mac_ifng_conc) {
                        float c = p.mac_ifng_conc[idx];
                        h_m1 += c / (c + p.mac_ifng_ec50 + 1e-30f);
                    }
                    if (p.mac_il12_conc) {
                        float c = p.mac_il12_conc[idx];
                        h_m1 += c / (c + p.mac_il12_ec50 + 1e-30f);
                    }
                    if (p.mac_tgfb_conc) {
                        float c = p.mac_tgfb_conc[idx];
                        h_m2 += c / (c + p.mac_tgfb_ec50 + 1e-30f);
                    }
                    if (p.mac_il10_conc) {
                        float c = p.mac_il10_conc[idx];
                        h_m2 += c / (c + p.mac_il10_ec50 + 1e-30f);
                    }
                    if (p.il6_conc) {  // shared PDE field with MDSC
                        float c = p.il6_conc[idx];
                        h_m2 += c / (c + p.mac_il6_ec50 + 1e-30f);
                    }
                    float p_m2 = (h_m1 + h_m2 < 1e-9f)
                        ? 1.0f                          // no cytokine signal → PDAC TAM bias (M2)
                        : h_m2 / (h_m1 + h_m2);
                    req.cell_state = (rng_uniform(rng) < p_m2) ? MAC_M2 : MAC_M1;

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

    // ── BCell source (bit 3) — Vvas entry × CXCL13 Hill SOFT-GATE (TLS localization) ──
    if (flags & 8) {
        atomicAdd(&diag->bcell_sources, 1);
        // B cells home via CXCR5 to CXCL13. Recruitment is GATED by local CXCL13 so
        // B cells localize to TLS/follicle sites. p_entry (Vvas) already gated WHERE B
        // may enter (bit 8); the CXCL13 Hill gates the TLS-forming subset on top.
        // A small floor (BCELL_CXCL13_GATE_FLOOR) permits CXCL13-independent seeding so
        // nascent TLS can self-organize (CXCL13 is Tfh/activated-B-derived only — see
        // XML CXCL13 block, O'Connor2023 — so a pure gate would deadlock cold-start:
        // no B → no plasma B → no CXCL13 → no B). gate ∈ [floor, 1].
        const float floor_cx = BCELL_CXCL13_GATE_FLOOR;
        float gate = floor_cx;
        if (p.cxcl13_conc) {
            float cxcl13 = p.cxcl13_conc[idx];
            float H_cxcl13 = cxcl13 / (cxcl13 + p.cxcl13_ec50_bcell + 1e-30f);
            gate = floor_cx + (1.0f - floor_cx) * H_cxcl13;
        }
        float p_bcell_eff = p.p_bcell * gate;
        if (rng_uniform(rng) < p_bcell_eff) {
            atomicAdd(&diag->bcell_roll_pass, 1);
            if (try_claim_voxel(x, y, z, p.nx, p.ny, p.nz,
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
            if (try_claim_voxel(x, y, z, p.nx, p.ny, p.nz,
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
            if (try_claim_voxel(x, y, z, p.nx, p.ny, p.nz,
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
// Build the recruitment parameter struct from the FLAMEGPU environment.
// Shared by the legacy Bernoulli recruit_gpu path and the systematic-resampling
// path (compute_qfields_kernel + recruit_systematic). Returns the QSP central
// counts (or 0 in parametric mode) via the out-params for stats/logging.
// ============================================================================
static RecruitKernelParams build_recruit_kernel_params(
    flamegpu::HostAPI* FLAMEGPU, int nx, int ny, int nz,
    float& qsp_teff, float& qsp_treg, float& qsp_th)
{
    RecruitKernelParams p;
    p.nx = nx; p.ny = ny; p.nz = nz;

    // Recruitment probabilities. The T-cell and B-cell streams are coupled to the QSP
    // central compartment (their QSP trafficking flux is q_·_T_in · V_T · [central] · gates);
    // MDSC/MAC/DC have no systemic reservoir in the QSP (flux is k_·_rec · V_T · gates), so
    // their p is the derived rate constant alone. recruitment_mode selects the source:
    //   0 = QSP-coupled : p = qsp_*_central × PARAM_*_RECRUIT_K (rate constant)
    //   1 = PARAMETRIC  : p = PARAM_*_RECRUIT_P (direct XML probability, no QSP read)
    const int recruitment_mode = FLAMEGPU->environment.getProperty<int>("recruitment_mode");
    qsp_teff = 0.0f; qsp_treg = 0.0f; qsp_th = 0.0f;
    if (recruitment_mode == 0) {
        qsp_teff = FLAMEGPU->environment.getProperty<float>("qsp_teff_central");
        qsp_treg = FLAMEGPU->environment.getProperty<float>("qsp_treg_central");
        qsp_th   = FLAMEGPU->environment.getProperty<float>("qsp_th_central");
        p.p_teff = std::min(qsp_teff * FLAMEGPU->environment.getProperty<float>("PARAM_TEFF_RECRUIT_K"), 1.0f);
        p.p_treg = std::min(qsp_treg * FLAMEGPU->environment.getProperty<float>("PARAM_TREG_RECRUIT_K"), 1.0f);
        p.p_th   = std::min(qsp_th   * FLAMEGPU->environment.getProperty<float>("PARAM_TH_RECRUIT_K"),   1.0f);
    } else {
        p.p_teff = std::min(FLAMEGPU->environment.getProperty<float>("PARAM_TEFF_RECRUIT_P"), 1.0f);
        p.p_treg = std::min(FLAMEGPU->environment.getProperty<float>("PARAM_TREG_RECRUIT_P"), 1.0f);
        p.p_th   = std::min(FLAMEGPU->environment.getProperty<float>("PARAM_TH_RECRUIT_P"),   1.0f);
    }
    // NOTE: the sampling fraction f and the per-stream magnitude Φ are now applied on the HOST
    // in recruit_systematic (normalized f·Φ scheme), not here — compute_qfields_kernel emits
    // only spatial shape weights. p.p_teff etc. below are retained for logging/parametric only.

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
    // Cytokine fields for M1/M2 spawn polarization (same EC50s as state-step)
    p.mac_ifng_conc = g_pde_solver ? reinterpret_cast<const float*>(
        g_pde_solver->get_device_concentration_ptr(CHEM_IFN)) : nullptr;
    p.mac_il12_conc = g_pde_solver ? reinterpret_cast<const float*>(
        g_pde_solver->get_device_concentration_ptr(CHEM_IL12)) : nullptr;
    p.mac_il10_conc = g_pde_solver ? reinterpret_cast<const float*>(
        g_pde_solver->get_device_concentration_ptr(CHEM_IL10)) : nullptr;
    p.mac_tgfb_conc = g_pde_solver ? reinterpret_cast<const float*>(
        g_pde_solver->get_device_concentration_ptr(CHEM_TGFB)) : nullptr;
    p.mac_ifng_ec50 = FLAMEGPU->environment.getProperty<float>("PARAM_MAC_IFN_G_EC50");
    p.mac_il12_ec50 = FLAMEGPU->environment.getProperty<float>("PARAM_MAC_IL_12_EC50");
    p.mac_il10_ec50 = FLAMEGPU->environment.getProperty<float>("PARAM_MAC_IL_10_EC50");
    p.mac_tgfb_ec50 = FLAMEGPU->environment.getProperty<float>("PARAM_MAC_TGFB_EC50");
    p.mac_il6_ec50  = FLAMEGPU->environment.getProperty<float>("PARAM_MAC_IL6_M2_EC50");

    // B cell params. QSP couples B trafficking through the same functional form as the
    // T cells (RF237 = q_Treg_T_in · V_T · [V_C_B] · H_C2), so B is central-pool-driven in
    // mode 0 (PARAM_BCELL_RECRUIT_K is now QSP-derived from q_Treg_T_in, mirroring Th) and
    // uses the parametric probability PARAM_BCELL_RECRUIT_P in mode 1.
    if (recruitment_mode == 0) {
        const float qsp_bcell = FLAMEGPU->environment.getProperty<float>("qsp_bcell_central");
        p.p_bcell = std::min(qsp_bcell * FLAMEGPU->environment.getProperty<float>("PARAM_BCELL_RECRUIT_K"), 1.0f);
    } else {
        p.p_bcell = std::min(FLAMEGPU->environment.getProperty<float>("PARAM_BCELL_RECRUIT_P"), 1.0f);
    }
    p.bcell_life_mean = FLAMEGPU->environment.getProperty<float>("PARAM_BCELL_LIFE_MEAN");
    p.bcell_life_sd   = FLAMEGPU->environment.getProperty<float>("PARAM_BCELL_LIFE_SD");
    p.cxcl13_conc = g_pde_solver ? reinterpret_cast<const float*>(
        g_pde_solver->get_device_concentration_ptr(CHEM_CXCL13)) : nullptr;
    p.cxcl13_ec50_bcell = FLAMEGPU->environment.getProperty<float>("PARAM_BCELL_EC50_CXCL13_REC");

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
    // CXCL9/10 effector recruitment gate (CAF-sourced CXCR3 ligand)
    p.cxcl9_conc = g_pde_solver ? reinterpret_cast<const float*>(
        g_pde_solver->get_device_concentration_ptr(CHEM_CXCL9_10)) : nullptr;
    p.cxcl9_ec50 = FLAMEGPU->environment.getProperty<float>("PARAM_TEFF_EC50_CXCL9_REC");
    // CCL2 myeloid recruitment gate (CCR2) — promoted from Stage-A marking to Stage-B roll
    p.ccl2_conc = g_pde_solver ? reinterpret_cast<const float*>(
        g_pde_solver->get_device_concentration_ptr(CHEM_CCL2)) : nullptr;
    p.ccl2_ec50_mdsc = FLAMEGPU->environment.getProperty<float>("PARAM_MDSC_EC50_CCL2_REC");
    p.ccl2_ec50_mac  = FLAMEGPU->environment.getProperty<float>("PARAM_MAC_EC50_CCL2_REC");

    // Volume-based occupancy params
    p.voxel_capacity = FLAMEGPU->environment.getProperty<float>("PARAM_VOXEL_CAPACITY");
    p.vol_teff   = FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_TCELL_EFF");
    p.vol_treg   = FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_TREG_REG");
    p.vol_th     = FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_TREG_TH");
    p.vol_mdsc   = FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_MDSC");
    p.vol_mac_m1 = FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_MAC_M1");
    p.vol_mac_m2 = FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_MAC_M2");
    p.vol_bcell  = FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_BCELL_NAIVE");

    // Naive T cell recruitment params
    p.naive_cd8_frac = FLAMEGPU->environment.getProperty<float>("PARAM_NAIVE_CD8_RECRUIT_FRAC");
    p.naive_cd4_frac = FLAMEGPU->environment.getProperty<float>("PARAM_NAIVE_CD4_RECRUIT_FRAC");
    p.naive_lifespan = FLAMEGPU->environment.getProperty<int>("PARAM_NAIVE_LIFESPAN_STEPS");

    return p;
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

    // Build parameter struct (shared with the systematic-resampling path)
    float qsp_teff = 0.0f, qsp_treg = 0.0f, qsp_th = 0.0f;
    RecruitKernelParams p = build_recruit_kernel_params(FLAMEGPU, nx, ny, nz,
                                                        qsp_teff, qsp_treg, qsp_th);

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

    // Read count from device. d_recruit_count is atomically incremented even when
    // the slot would exceed the buffer, so (count > MAX) means requests were dropped.
    int count = 0;
    CUDA_CHECK(cudaMemcpy(&count, d_recruit_count, sizeof(int), cudaMemcpyDeviceToHost));
    if (count > MAX_RECRUITS_PER_STEP) {
        std::cerr << "[WARN] Recruitment cap hit at step "
                  << FLAMEGPU->getStepCounter()
                  << ": " << count << " requests generated, buffer capped at "
                  << MAX_RECRUITS_PER_STEP << " ("
                  << (count - MAX_RECRUITS_PER_STEP) << " dropped). "
                  << "Raise MAX_RECRUITS_PER_STEP in core/common.cuh." << std::endl;
        count = MAX_RECRUITS_PER_STEP;
    }

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
    auto fib_api   = FLAMEGPU->agent(AGENT_FIBROBLAST);

    int teff_recruited = 0, treg_recruited = 0, th_recruited = 0;
    int mdsc_recruited = 0, mac_recruited = 0, mac_m1_recruited = 0, mac_m2_recruited = 0;
    int bcell_recruited = 0, dc_recruited = 0, qpsc_recruited = 0;

    // Step-5 determinism: mint each recruit's lineage tag from its IDENTITY (voxel, step, type),
    // NOT its buffer index i — the compact request buffer order is nondeterministic (atomic
    // compaction), but (voxel,step,type) is deterministic and unique (≤1 recruit/voxel/type/step).
    const int rec_gx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int rec_gy = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const unsigned int rec_seed = FLAMEGPU->environment.getProperty<unsigned int>("sim_seed");
    const unsigned int rec_step = FLAMEGPU->getStepCounter();

    for (int i = 0; i < count; i++) {
        const auto& r = requests[i];
        // High nibble = type (matches the init type-offsets: mac=1,T=2,treg=3,dc=4,bcell=5,mdsc=6)
        // so det_tags are unique ACROSS types — required since immune share the deterministic wave
        // movement rounds (one voxel_owner buffer); low 28 bits = identity hash (voxel,step,type).
        unsigned int type_nibble;
        switch (r.cell_type) {
            case CELL_TYPE_MAC:   type_nibble = 1u; break;
            case CELL_TYPE_T:     type_nibble = 2u; break;
            case CELL_TYPE_TREG:  type_nibble = 3u; break;
            case CELL_TYPE_DC:    type_nibble = 4u; break;
            case CELL_TYPE_BCELL: type_nibble = 5u; break;
            case CELL_TYPE_MDSC:  type_nibble = 6u; break;
            case CELL_TYPE_FIB:   type_nibble = 8u; break;
            default:              type_nibble = 7u; break;
        }
        // Seed from the globally-unique deterministic request index r.seq (set by
        // recruit_systematic) rather than (voxel,step,type): systematic resampling can place
        // multiple same-type cells in one voxel, so the voxel key is no longer unique.
        unsigned int rec_hash = det_rng_seed(r.seq, rec_step,
                                             0x5EC0u + static_cast<unsigned int>(r.cell_type), rec_seed);
        unsigned int rec_dt = (type_nibble << 28) | (rec_hash & 0x0FFFFFFFu);

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
            a.setVariable<unsigned int>("det_tag", rec_dt);
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
            a.setVariable<unsigned int>("det_tag", rec_dt);
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
            a.setVariable<unsigned int>("det_tag", rec_dt);
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
            a.setVariable<unsigned int>("det_tag", rec_dt);
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
            a.setVariable<unsigned int>("det_tag", rec_dt);
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
            a.setVariable<unsigned int>("det_tag", rec_dt);
            dc_recruited++;
        }
        else if (r.cell_type == CELL_TYPE_FIB) {
            auto a = fib_api.newAgent();
            a.setVariable<int>("x", r.x);
            a.setVariable<int>("y", r.y);
            a.setVariable<int>("z", r.z);
            a.setVariable<int>("cell_state", r.cell_state);
            a.setVariable<int>("life", r.life);
            a.setVariable<int>("divide_cooldown", 0);
            a.setVariable<int>("divide_count", 0);
            a.setVariable<int>("divide_flag", 0);
            a.setVariable<int>("frc_dwell_counter", 0);
            a.setVariable<int>("persist_dir_x", 0);
            a.setVariable<int>("persist_dir_y", 0);
            a.setVariable<int>("persist_dir_z", 0);
            a.setVariable<unsigned int>("det_tag", rec_dt);
            qpsc_recruited++;
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
// Host Function: systematic-resampling recruitment.
//
// Replaces the two-stage per-voxel Bernoulli path (mark_entry_points ⊗ recruit_all_kernel).
// Rationale (project-recruitment-systematic-resampling): the Bernoulli realization of a
// low-probability entry field, re-drawn every step, is the dominant immune spatial noise that
// keeps ICC seed-bound. Here we instead draw a deterministic STRATIFIED sample from the
// reproducible expected-recruitment field q_type[v] = p_entry·gate, so the mean rate is
// unchanged but the per-voxel coin-flip noise is removed.
//
// Per stream: Σq = Σ_v q[v];  N = round(Σq·(1 + CV·𝒩(0,1)));  systematic resample with a single
// offset u∈[0,1) → each of N cells targets the voxel whose cumulative q crosses (k+u)·Σq/N.
// Capacity-skip: if the target voxel is full, advance forward (cumulative/index order) to the
// next voxel with room; if none, skip the cell. All RNG is det_rng seeded from (stream,step,seed)
// → bit-reproducible at fixed seed, decorrelated across seeds. Emits RecruitRequests consumed
// unchanged by place_recruited_agents.
//
// d_volume_used is read (stale post-movement occupancy) only to gate intra-recruitment packing;
// it is NOT written back — the later zero_occ_grid layer wipes it and write_to_occ_grid rebuilds
// it from the actual agent population (including these recruits). Mirrors the old kernel's
// transient try_claim_voxel writes.
// ============================================================================
FLAMEGPU_HOST_FUNCTION(recruit_systematic) {
    nvtxRangePush("Recruit Systematic");
    if (!g_pde_solver || !d_qfields || !d_vvas_field) { nvtxRangePop(); return; }

    const int nx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int nz = FLAMEGPU->environment.getProperty<int>("grid_size_z");
    const int nxy = nx * ny;
    const int nvox = nxy * nz;

    // place_recruited_agents reads d_recruit_count; reset before we (re)populate it.
    CUDA_CHECK(cudaMemset(d_recruit_count, 0, sizeof(int)));

    // Shared param struct (same env reads as the legacy path).
    float qsp_teff = 0.0f, qsp_treg = 0.0f, qsp_th = 0.0f;
    RecruitKernelParams p = build_recruit_kernel_params(FLAMEGPU, nx, ny, nz,
                                                        qsp_teff, qsp_treg, qsp_th);
    const float entry_scale = FLAMEGPU->environment.getProperty<float>("PARAM_ENTRY_ADHESION_SCALE");

    // 1) Compute the expected-recruitment q-fields on device (no roll).
    dim3 block(8, 8, 8);
    dim3 grid((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);
    compute_qfields_kernel<<<grid, block>>>(d_qfields, d_vvas_field, entry_scale, p);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 2) D2H the q-fields + current occupancy (for capacity gating; not written back).
    std::vector<float> qf(static_cast<size_t>(N_QPLANE) * nvox);
    CUDA_CHECK(cudaMemcpy(qf.data(), d_qfields, qf.size() * sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<float> vol_used(nvox);
    CUDA_CHECK(cudaMemcpy(vol_used.data(), d_volume_used, nvox * sizeof(float), cudaMemcpyDeviceToHost));

    const float cap = p.voxel_capacity;
    const unsigned int sim_seed = FLAMEGPU->environment.getProperty<unsigned int>("sim_seed");
    const unsigned int step = static_cast<unsigned int>(FLAMEGPU->getStepCounter());
    const float count_cv = FLAMEGPU->environment.getProperty<float>("PARAM_RECRUIT_COUNT_CV");  // per-stream demographic count CV (SBI-tunable)

    // ── Deterministic host RNG helpers (det_rng is __host__ __device__) ──
    auto det_normal = [](unsigned int& s) -> float {
        float u1 = det_rng_uniform(s) + 1e-10f;
        float u2 = det_rng_uniform(s);
        return std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * 3.14159265f * u2);
    };
    auto det_normal_life = [&](unsigned int& s, float mean, float sd) -> int {
        int life = static_cast<int>(std::lround(mean + det_normal(s) * sd));
        return life > 0 ? life : 1;
    };
    auto det_exp_life = [](unsigned int& s, float mean) -> int {
        float u = det_rng_uniform(s) + 1e-4f;
        int life = static_cast<int>(std::lround(mean * std::log(1.0f / u)));
        return life > 0 ? life : 1;
    };
    auto decode = [&](int vidx, int& x, int& y, int& z) {
        x = vidx % nx; y = (vidx / nx) % ny; z = vidx / nxy;
    };

    std::vector<RecruitRequest> requests;
    int req_seq = 0;

    // Capacity-skip placement: target voxel, else advance forward to the next voxel with room.
    auto place_one = [&](int target_vox, float cell_vol) -> int {
        for (int v = target_vox; v < nvox; ++v) {
            if (vol_used[v] + cell_vol <= cap) { vol_used[v] += cell_vol; return v; }
        }
        return -1;  // no capacity downstream → skip
    };

    // Emit one cell of stream `s` targeting voxel `tvox` (+ any coupled naive cell).
    auto emit_cell = [&](int s, int tvox, unsigned int& crng) {
        RecruitRequest req{};
        switch (s) {
        case QS_TEFF: {
            int pv = place_one(tvox, p.vol_teff);
            if (pv < 0) return;
            decode(pv, req.x, req.y, req.z);
            req.cell_type = CELL_TYPE_T; req.cell_state = T_CELL_EFF;
            req.life = det_normal_life(crng, p.t_life_mean, p.t_life_sd);
            req.divide_cd = p.t_divide_cd; req.divide_limit = p.t_divide_limit;
            req.IL2_release_remain = p.t_IL2_release;
            req.seq = req_seq++;
            requests.push_back(req);
            if (p.naive_cd8_frac > 0.0f && det_rng_uniform(crng) < p.naive_cd8_frac) {
                int npv = place_one(tvox, p.vol_teff);
                if (npv >= 0) {
                    RecruitRequest nr{};
                    decode(npv, nr.x, nr.y, nr.z);
                    nr.cell_type = CELL_TYPE_T; nr.cell_state = T_CELL_NAIVE;
                    nr.life = p.naive_lifespan;
                    nr.seq = req_seq++;
                    requests.push_back(nr);
                }
            }
            break;
        }
        case QS_TREG: {
            int pv = place_one(tvox, p.vol_treg);
            if (pv < 0) return;
            decode(pv, req.x, req.y, req.z);
            req.cell_type = CELL_TYPE_TREG; req.cell_state = TCD4_TREG;
            req.life = det_normal_life(crng, p.tcd4_life_mean, p.tcd4_life_sd);
            req.divide_cd = p.tcd4_divide_cd; req.divide_limit = p.tcd4_divide_limit;
            req.TGFB_release_remain = p.tcd4_TGFB_release;
            req.CTLA4 = p.ctla4_treg;
            req.seq = req_seq++;
            requests.push_back(req);
            break;
        }
        case QS_TH: {
            int pv = place_one(tvox, p.vol_th);
            if (pv < 0) return;
            decode(pv, req.x, req.y, req.z);
            req.cell_type = CELL_TYPE_TREG; req.cell_state = TCD4_TH;
            req.life = det_normal_life(crng, p.tcd4_life_mean, p.tcd4_life_sd);
            req.divide_cd = p.tcd4_divide_cd; req.divide_limit = p.tcd4_divide_limit;
            req.TGFB_release_remain = p.tcd4_TGFB_release;
            req.CTLA4 = 0.0f;
            req.seq = req_seq++;
            requests.push_back(req);
            if (p.naive_cd4_frac > 0.0f && det_rng_uniform(crng) < p.naive_cd4_frac) {
                int npv = place_one(tvox, p.vol_th);
                if (npv >= 0) {
                    RecruitRequest nr{};
                    decode(npv, nr.x, nr.y, nr.z);
                    nr.cell_type = CELL_TYPE_TREG; nr.cell_state = TCD4_NAIVE;
                    nr.life = p.naive_lifespan;
                    nr.seq = req_seq++;
                    requests.push_back(nr);
                }
            }
            break;
        }
        case QS_MDSC: {
            int pv = place_one(tvox, p.vol_mdsc);
            if (pv < 0) return;
            decode(pv, req.x, req.y, req.z);
            req.cell_type = CELL_TYPE_MDSC; req.cell_state = 0;
            req.life = det_exp_life(crng, p.mdsc_life_mean);
            req.seq = req_seq++;
            requests.push_back(req);
            break;
        }
        case QS_MAC: {
            // M1/M2 polarization from the precomputed P(M2) plane at the TARGET voxel
            // (matches the local-cytokine spawn polarization of the old kernel).
            float p_m2 = qf[static_cast<size_t>(QPLANE_PM2) * nvox + tvox];
            bool is_m2 = det_rng_uniform(crng) < p_m2;
            int pv = place_one(tvox, is_m2 ? p.vol_mac_m2 : p.vol_mac_m1);
            if (pv < 0) return;
            decode(pv, req.x, req.y, req.z);
            req.cell_type = CELL_TYPE_MAC;
            req.cell_state = is_m2 ? MAC_M2 : MAC_M1;
            req.life = det_exp_life(crng, p.mac_life_mean);
            req.seq = req_seq++;
            requests.push_back(req);
            break;
        }
        case QS_BCELL: {
            int pv = place_one(tvox, p.vol_bcell);
            if (pv < 0) return;
            decode(pv, req.x, req.y, req.z);
            req.cell_type = CELL_TYPE_BCELL; req.cell_state = BCELL_NAIVE;
            req.life = det_normal_life(crng, p.bcell_life_mean, p.bcell_life_sd);
            req.seq = req_seq++;
            requests.push_back(req);
            break;
        }
        case QS_DC1:
        case QS_DC2: {
            int pv = place_one(tvox, p.vol_dc);
            if (pv < 0) return;
            decode(pv, req.x, req.y, req.z);
            req.cell_type = CELL_TYPE_DC; req.cell_state = DC_IMMATURE;
            req.subtype = (s == QS_DC1) ? DC_CDC1 : DC_CDC2;
            req.life = det_normal_life(crng, p.dc_life_immature_mean, p.dc_life_immature_sd);
            req.seq = req_seq++;
            requests.push_back(req);
            break;
        }
        default: break;
        }
    };

    // ── Single-ledger per-stream magnitude. The kernel wrote pure spatial shape weights
    //    w_type(v) = Vvas·gate; here we rescale each plane so its TOTAL is the number of type-Z
    //    cells to recruit this step, then the sampler distributes that count by w:
    //      QSP mode (0): SOFT GAP-CLOSING to the QSP tumor composition. The ABM window should
    //        hold N_target,Z = f · X_QSP_tumor,Z cells (geometric f = min(1, V_box/V_T); the
    //        window's volume-share of the whole tumor). Per step we recruit only the deficit,
    //        relaxed by α:  recruit_Z = α · max(0, N_target,Z − N_abm,Z). ABM death/division run
    //        free underneath — the controller pins the trajectory to f·X_QSP_tumor, it does not
    //        force ABM removal to match the QSP. This self-caps abundant streams (e.g. B) and
    //        replaces the old gross-influx flux (which over-populated efflux-limited types).
    //      Parametric (1): Σ_v q = P_type·Σw  (per-voxel probability; no QSP, no normalization).
    //    Only the N_QSTREAM recruitment planes are rescaled; the PM2 polarization plane is left. ──
    {
        auto envf = [&](const char* k){ return static_cast<double>(FLAMEGPU->environment.getProperty<float>(k)); };
        const int rec_mode = FLAMEGPU->environment.getProperty<int>("recruitment_mode");
        double scale[N_QSTREAM];
        if (rec_mode == 0) {
            const double NA    = envf("AVOGADROS");
            const double alpha = envf("PARAM_RECRUIT_ALPHA");
            // Geometric sampling fraction: fraction of the whole tumor VOLUME this window resolves.
            // Uses total tumor content (V_T, all cell types + stroma), not cancer count — robust to
            // intratumoral heterogeneity (a stroma-rich, cancer-poor window still gets its share).
            const double voxel_m = FLAMEGPU->environment.getProperty<int>("PARAM_VOXEL_SIZE") / 1e6;
            const double V_box = static_cast<double>(nx) * ny * nz * voxel_m * voxel_m * voxel_m;
            const double V_T   = envf("qsp_tum_vol");
            const double f     = (V_T > 0.0) ? std::min(1.0, V_box / V_T) : 1.0;

            // Target ABM population per stream = f · X_QSP_tumor,Z (cells). qsp_*_tumor are in mol → ×NA.
            double Ntarget[N_QSTREAM];
            Ntarget[QS_TEFF]  = f * envf("qsp_teff_tumor")  * NA;
            Ntarget[QS_MDSC]  = f * envf("qsp_mdsc_tumor")  * NA;
            Ntarget[QS_MAC]   = f * (envf("qsp_m1_tumor") + envf("qsp_m2_tumor")) * NA;
            Ntarget[QS_BCELL] = f * envf("qsp_bcell_tumor") * NA;
            // Th + Treg share one ABM agent (AGENT_TREG); DC1 + DC2 share AGENT_DC. Compute the
            // combined deficit against the combined ABM count, then split by the QSP subtype ratio.
            const double th_q = envf("qsp_th_tumor"), treg_q = envf("qsp_treg_tumor");
            const double dc_q = envf("qsp_dc_tumor");
            const double tcd4_target = f * (th_q + treg_q) * NA;
            const double dc_target   = f * dc_q * NA;

            // ── Current ABM counts, COUNTED BY STATE so each count means the same class as its
            //    QSP target (contract §4.4 leaky class mapping).
            //    RULE: an ABM state is counted into a stream's gap iff it has NO QSP species of
            //    its own. States the QSP tracks separately are excluded from BOTH sides and
            //    monitored instead — otherwise the deficit is closed against a pool the target
            //    never described, and the controller under-recruits by that pool's size.
            //      CD8:  EFF+CYT ↔ SP_V_T_CD8. SUPP ↔ SP_V_T_CD8_exh (own species, excluded).
            //            NAIVE is a side-channel recruit (naive_cd8_frac bonus, never counted
            //            against target[QS_TEFF]) and the QSP holds naive CD8 in the LN, not
            //            in V_T_CD8 — excluded.
            //      TCD4: TH+TREG ↔ SP_V_T_Th + SP_V_T_Treg. TFH ↔ SP_V_T_Tfh (own species,
            //            arises by TH→TFH transition, t_reg.cuh:328) and NAIVE is likewise a
            //            bonus recruit — both excluded.
            //      B:    NAIVE+ACT ↔ SP_V_T_B. PLASMA ↔ SP_V_T_Plasma (own species, arises by
            //            ACT→PLASMA transition, b_cell.cuh:225) — excluded.
            //      MAC (M1+M2 ↔ Mac_M1+Mac_M2), DC (immature+mature ↔ cDC+mcDC) and MDSC
            //      (single state) are already class-consistent; counted whole, deliberately.
            auto nstate = [&](const char* agent, int st) {
                return static_cast<double>(FLAMEGPU->agent(agent).count<int>("cell_state", st));
            };
            const double n_teff  = nstate(AGENT_TCELL, T_CELL_EFF) + nstate(AGENT_TCELL, T_CELL_CYT);
            const double n_tcd4  = nstate(AGENT_TREG, TCD4_TH) + nstate(AGENT_TREG, TCD4_TREG);
            const double n_bcell = nstate(AGENT_BCELL, BCELL_NAIVE) + nstate(AGENT_BCELL, BCELL_ACTIVATED);
            const double n_mdsc  = FLAMEGPU->agent(AGENT_MDSC).count();
            const double n_mac   = FLAMEGPU->agent(AGENT_MACROPHAGE).count();
            const double n_dc    = FLAMEGPU->agent(AGENT_DC).count();

            // Per-stream recruit count = α·max(0, target − current). Split the shared TCD4/DC gaps.
            double target[N_QSTREAM];
            target[QS_TEFF]  = alpha * std::max(0.0, Ntarget[QS_TEFF]  - n_teff);
            target[QS_MDSC]  = alpha * std::max(0.0, Ntarget[QS_MDSC]  - n_mdsc);
            target[QS_MAC]   = alpha * std::max(0.0, Ntarget[QS_MAC]   - n_mac);
            target[QS_BCELL] = alpha * std::max(0.0, Ntarget[QS_BCELL] - n_bcell);
            const double tcd4_gap = alpha * std::max(0.0, tcd4_target - n_tcd4);
            const double tcd4_den = th_q + treg_q + 1e-300;
            target[QS_TH]   = tcd4_gap * (th_q   / tcd4_den);
            target[QS_TREG] = tcd4_gap * (treg_q / tcd4_den);
            const double dc_gap = alpha * std::max(0.0, dc_target - n_dc);

            // ── CLASS AUDIT (G3/G4 monitor) ────────────────────────────────────────
            // Report the residual per class; never act on it. The three classes the QSP
            // tracks separately are reported against THEIR own species, so a discrepancy
            // is attributable to a class rather than smeared across the CD8/TCD4/B pools.
            // ⚠️ CD8_exh, Tfh and Plasma have no recruitment stream: the ABM reaches them
            // only by transition, so their residual tests the TRANSITION rates, not entry.
            {
                if (!g_class_audit_csv.is_open()) {
                    g_class_audit_csv.open(g_class_audit_path);
                    g_class_audit_csv << "step,presim,f,V_T_m3,"
                        << "tgt.CD8,abm.CD8,tgt.CD8_exh,abm.CD8_exh,abm.CD8_naive,"
                        << "tgt.TCD4,abm.TCD4,tgt.Tfh,abm.Tfh,abm.TCD4_naive,"
                        << "tgt.B,abm.B,tgt.Plasma,abm.Plasma,"
                        << "tgt.MDSC,abm.MDSC,tgt.MAC,abm.MAC,tgt.DC,abm.DC,"
                        << "tgt.fib_total,abm.fib_total,tgt.qPSC,abm.qPSC,"
                        << "tgt.iCAF,abm.iCAF,tgt.myCAF,abm.myCAF,tgt.apCAF,abm.apCAF\n";
                }
                if (g_class_audit_csv.is_open()) {
                    g_class_audit_csv << FLAMEGPU->getStepCounter()
                        << ',' << (is_presim_mode_active() ? 1 : 0)
                        << ',' << f << ',' << V_T
                        << ',' << Ntarget[QS_TEFF]              << ',' << n_teff
                        << ',' << f * envf("qsp_t_exh_tumor") * NA
                        << ',' << nstate(AGENT_TCELL, T_CELL_SUPP)
                        << ',' << nstate(AGENT_TCELL, T_CELL_NAIVE)
                        << ',' << tcd4_target                   << ',' << n_tcd4
                        << ',' << f * envf("qsp_tfh_tumor") * NA
                        << ',' << nstate(AGENT_TREG, TCD4_TFH)
                        << ',' << nstate(AGENT_TREG, TCD4_NAIVE)
                        << ',' << Ntarget[QS_BCELL]             << ',' << n_bcell
                        << ',' << f * envf("qsp_plasma_tumor") * NA
                        << ',' << nstate(AGENT_BCELL, BCELL_PLASMA)
                        << ',' << Ntarget[QS_MDSC]              << ',' << n_mdsc
                        << ',' << Ntarget[QS_MAC]               << ',' << n_mac
                        << ',' << dc_target                     << ',' << n_dc
                        << ',' << f * (envf("qsp_qpsc_tumor") + envf("qsp_icaf_tumor") + envf("qsp_mycaf_tumor") + envf("qsp_apcaf_tumor")) * NA
                        << ',' << (nstate(AGENT_FIBROBLAST, FIB_QUIESCENT) + nstate(AGENT_FIBROBLAST, FIB_ICAF) + nstate(AGENT_FIBROBLAST, FIB_MYCAF) + nstate(AGENT_FIBROBLAST, FIB_APCAF))
                        << ',' << f * envf("qsp_qpsc_tumor") * NA
                        << ',' << nstate(AGENT_FIBROBLAST, FIB_QUIESCENT)
                        << ',' << f * envf("qsp_icaf_tumor") * NA
                        << ',' << nstate(AGENT_FIBROBLAST, FIB_ICAF)
                        << ',' << f * envf("qsp_mycaf_tumor") * NA
                        << ',' << nstate(AGENT_FIBROBLAST, FIB_MYCAF)
                        << ',' << f * envf("qsp_apcaf_tumor") * NA
                        << ',' << nstate(AGENT_FIBROBLAST, FIB_APCAF) << '\n';
                    g_class_audit_csv.flush();
                }
            }
            target[QS_DC1] = 0.5 * dc_gap;   // cDC1/cDC2 split 50/50 (subtype-resolved QSP DC not exported)
            target[QS_DC2] = 0.5 * dc_gap;

            // ── Direct CAF gap-closing recruitment ─────────────────────────
            // The ABM domain is fixed (no V_T expansion), so the QSP's CAF
            // proliferation term (which compensates density dilution) has no
            // analogue. Instead we directly recruit CAFs at sub-type-specific
            // deficit rates, mirroring the immune gap-closing pattern.
            // Placement: random voxel via place_one (linear scan for capacity).
            // CAF proliferation is disabled in fibroblast.cuh; sub-state
            // composition tracks the QSP composition ratios.
            {
                const double alpha_fib = static_cast<double>(envf("PARAM_FIB_GAP_ALPHA"));
                const double tgt_icaf  = f * envf("qsp_icaf_tumor")  * NA;
                const double tgt_mycaf = f * envf("qsp_mycaf_tumor") * NA;
                const double tgt_apcaf = f * envf("qsp_apcaf_tumor") * NA;
                const double n_icaf  = nstate(AGENT_FIBROBLAST, FIB_ICAF);
                const double n_mycaf = nstate(AGENT_FIBROBLAST, FIB_MYCAF);
                const double n_apcaf = nstate(AGENT_FIBROBLAST, FIB_APCAF);
                const double def_icaf  = std::max(0.0, tgt_icaf  - n_icaf);
                const double def_mycaf = std::max(0.0, tgt_mycaf - n_mycaf);
                const double def_apcaf = std::max(0.0, tgt_apcaf - n_apcaf);
                const double def_total = def_icaf + def_mycaf + def_apcaf;

                if (def_total > 0.0) {
                    double raw_N = alpha_fib * def_total;
                    double Nfloor = std::floor(raw_N);
                    unsigned int srng = det_rng_seed(0xBF1B, step, 0x9A5Cu, sim_seed);
                    int N_recruit = static_cast<int>(Nfloor)
                        + (det_rng_uniform(srng) < (raw_N - Nfloor) ? 1 : 0);

                    const float fib_life_mean = envf("PARAM_FIB_LIFE_MEAN");
                    const float vol_icaf  = envf("PARAM_VOLUME_FIB_ICAF");
                    const float vol_mycaf = envf("PARAM_VOLUME_FIB_MYCAF");
                    const float frac_icaf  = def_icaf  / def_total;
                    const float frac_mycaf = def_mycaf / def_total;

                    for (int k = 0; k < N_recruit; ++k) {
                        float r = det_rng_uniform(srng);
                        int cstate;
                        float cvol;
                        if (r < frac_icaf) {
                            cstate = FIB_ICAF;  cvol = vol_icaf;
                        } else if (r < frac_icaf + frac_mycaf) {
                            cstate = FIB_MYCAF; cvol = vol_mycaf;
                        } else {
                            cstate = FIB_APCAF; cvol = vol_icaf;
                        }

                        int vx = static_cast<int>(det_rng_uniform(srng) * nx);
                        int vy = static_cast<int>(det_rng_uniform(srng) * ny);
                        int vz = static_cast<int>(det_rng_uniform(srng) * nz);
                        if (vx >= nx) vx = nx - 1;
                        if (vy >= ny) vy = ny - 1;
                        if (vz >= nz) vz = nz - 1;
                        int vox = vz * nxy + vy * nx + vx;
                        int pv = place_one(vox, cvol);
                        if (pv < 0) continue;

                        RecruitRequest req{};
                        decode(pv, req.x, req.y, req.z);
                        req.cell_type = CELL_TYPE_FIB;
                        req.cell_state = cstate;
                        req.life = static_cast<int>(fib_life_mean
                            * logf(1.0f / (det_rng_uniform(srng) + 1e-4f)) + 0.5f);
                        if (req.life <= 0) req.life = 1;
                        req.seq = req_seq++;
                        requests.push_back(req);
                    }
                }
            }

            for (int s = 0; s < N_QSTREAM; ++s) {
                const float* q = &qf[static_cast<size_t>(s) * nvox];
                double W = 0.0; for (int v = 0; v < nvox; ++v) W += q[v];
                scale[s] = (W > 0.0) ? (target[s] / W) : 0.0;
            }
        } else {
            scale[QS_TEFF]  = envf("PARAM_TEFF_RECRUIT_P");
            scale[QS_TREG]  = envf("PARAM_TREG_RECRUIT_P");
            scale[QS_TH]    = envf("PARAM_TH_RECRUIT_P");
            scale[QS_MDSC]  = envf("PARAM_MDSC_RECRUIT_P");
            scale[QS_MAC]   = envf("PARAM_MAC_RECRUIT_P");
            scale[QS_BCELL] = envf("PARAM_BCELL_RECRUIT_P");
            scale[QS_DC1]   = envf("PARAM_DC_RECRUIT_P");
            scale[QS_DC2]   = envf("PARAM_DC_RECRUIT_P");
        }
        for (int s = 0; s < N_QSTREAM; ++s) {
            const float sc = static_cast<float>(scale[s]);
            float* q = &qf[static_cast<size_t>(s) * nvox];
            for (int v = 0; v < nvox; ++v) q[v] *= sc;
        }
    }

    int drawnN[N_QSTREAM] = {0};

    // ── Per-stream systematic (stratified) resampling ──
    for (int s = 0; s < N_QSTREAM; ++s) {
        const float* q = &qf[static_cast<size_t>(s) * nvox];
        double Sigma = 0.0;
        for (int v = 0; v < nvox; ++v) Sigma += q[v];
        if (Sigma <= 0.0) continue;

        // N ~ Σq·(1 + CV·𝒩), deterministic per (stream, step, seed).
        // STOCHASTIC rounding of the fractional part (NOT round-to-nearest): a sparse stream with
        // per-step Σq < 0.5 would otherwise round to 0 EVERY step and never recruit — that zeroed
        // MDSC and B cells (per-step expectation <1) in validation. Stochastic rounding preserves
        // E[N] = Σq for sparse streams. It's a single per-STREAM coin on the fraction, NOT the
        // per-voxel Bernoulli that caused the spatial ICC noise → spatial reproducibility intact.
        unsigned int nrng = det_rng_seed(s, step, 0x7100u, sim_seed);
        double x = Sigma * (1.0 + count_cv * det_normal(nrng));
        if (x <= 0.0) continue;
        double xfloor = std::floor(x);
        int N = static_cast<int>(xfloor) + (det_rng_uniform(nrng) < (x - xfloor) ? 1 : 0);
        if (N <= 0) continue;
        drawnN[s] = N;
        const float u = det_rng_uniform(nrng);           // single systematic offset in [0,1)

        // Distinct per-cell RNG stream for life/state/naive rolls.
        unsigned int crng = det_rng_seed(s, step, 0x7200u, sim_seed);

        const double step_size = Sigma / static_cast<double>(N);
        double thr = u * step_size;                       // first stratum crossing
        double cum = 0.0;
        int k = 0;
        // Σ accumulated in the SAME order as above → cum ends bit-identical to Sigma, and
        // thr_max = (N-1+u)/N·Σ < Σ, so all N strata fire.
        for (int v = 0; v < nvox && k < N; ++v) {
            cum += q[v];
            while (k < N && thr < cum) {
                emit_cell(s, v, crng);
                ++k;
                thr += step_size;
            }
        }
    }

    // ── Stats (diagnostic; *_rec counts filled later by place_recruited_agents) ──
    g_recruit_stats.p_teff = p.p_teff;
    g_recruit_stats.p_treg = p.p_treg;
    g_recruit_stats.p_th   = p.p_th;
    g_recruit_stats.qsp_teff = qsp_teff;
    g_recruit_stats.qsp_treg = qsp_treg;
    g_recruit_stats.qsp_th   = qsp_th;
    g_recruit_stats.t_sources     = drawnN[QS_TEFF] + drawnN[QS_TREG] + drawnN[QS_TH];
    g_recruit_stats.mdsc_sources  = drawnN[QS_MDSC];
    g_recruit_stats.mac_sources   = drawnN[QS_MAC];
    g_recruit_stats.bcell_sources = drawnN[QS_BCELL];
    g_recruit_stats.dc_sources    = drawnN[QS_DC1] + drawnN[QS_DC2];

    // ── Upload requests for place_recruited_agents ──
    int count = static_cast<int>(requests.size());
    if (count > MAX_RECRUITS_PER_STEP) {
        std::cerr << "[WARN] recruit_systematic: " << count << " requests > cap "
                  << MAX_RECRUITS_PER_STEP << " (truncating). Raise MAX_RECRUITS_PER_STEP."
                  << std::endl;
        count = MAX_RECRUITS_PER_STEP;
    }
    if (count > 0) {
        CUDA_CHECK(cudaMemcpy(d_recruit_requests, requests.data(),
            count * sizeof(RecruitRequest), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_recruit_count, &count, sizeof(int), cudaMemcpyHostToDevice));

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

// Reset the deterministic voxel-claim ownership buffer to ULLONG_MAX (0xFF bytes).
// Placed as a layer before each reserve pass (per-step for division, per-round for movement).
FLAMEGPU_HOST_FUNCTION(reset_voxel_owner) {
    nvtxRangePush("Reset Voxel Owner");
    if (g_pde_solver && d_voxel_owner) {
        int total_voxels = g_pde_solver->get_total_voxels();
        cudaMemset(d_voxel_owner, 0xFF, total_voxels * sizeof(unsigned long long));
    }
    nvtxRangePop();
}

// ============================================================================
// Zero Fibroblast Density Field (reset before scatter)
// Uses cudaMemset on flat device array — no D2H/H2D copy.
// ============================================================================
// ── phi: zero the per-step accumulators. The KDE fields are rebuilt from scratch each step
//    (agents move), so nothing carries over — unlike V_vas/ECM which are state. ──
FLAMEGPU_HOST_FUNCTION(zero_phi_buffers) {
    if (!g_pde_solver) return;
    const int nvox = g_pde_solver->get_total_voxels();
    if (d_phi_kde)   cudaMemset(d_phi_kde, 0,
                                static_cast<size_t>(N_PHI_KDE) * nvox * sizeof(unsigned long long));
    if (d_phi_accum) cudaMemset(d_phi_accum, 0, N_PHI_PAIR * sizeof(unsigned long long));
    if (d_phi_count) cudaMemset(d_phi_count, 0, (N_PHI_PAIR + N_PHI_KDE) * sizeof(unsigned int));
}

// ── phi: reduce the accumulators to the three multipliers and hand them to the QSP. ──
//
//   phi_p = mean_a[ S_B(v_a) / SumK_local(v_a) ]  /  ( N_B / N_vox )
//
// The numerator is the measured local B-density around A; the denominator is the well-mixed
// density the ODE's rate law already assumes. Their ratio is the correction, and phi=1
// reproduces the rate law exactly.
//
// ⚠️ A pair with NO cells on either side has no measurement — phi falls back to 1 (mean-field)
// and is flagged invalid, rather than being computed as 0/0 or silently as 0. Seeding phi=0
// into the ODE would switch the corresponding interaction OFF, which is a very different
// statement from "the ABM could not measure this".
FLAMEGPU_HOST_FUNCTION(finalize_phi) {
    if (!g_pde_solver || !d_phi_accum) return;
    const int nvox = g_pde_solver->get_total_voxels();

    unsigned long long acc[N_PHI_PAIR];
    unsigned int cnt[N_PHI_PAIR + N_PHI_KDE];
    cudaMemcpy(acc, d_phi_accum, sizeof(acc), cudaMemcpyDeviceToHost);
    cudaMemcpy(cnt, d_phi_count, sizeof(cnt), cudaMemcpyDeviceToHost);

    for (int p = 0; p < N_PHI_PAIR; ++p) {
        const unsigned int n_tgt = cnt[p];
        const unsigned int n_src = cnt[N_PHI_PAIR + phi_pair_kde(p)];
        if (n_tgt == 0 || n_src == 0) {
            g_phi[p] = 1.0;
            g_phi_valid[p] = false;
            continue;
        }
        const double mean_local = static_cast<double>(acc[p])
                                / static_cast<double>(ecm_fp_scale())
                                / static_cast<double>(n_tgt);
        const double rho_B = static_cast<double>(n_src) / static_cast<double>(nvox);
        g_phi[p] = mean_local / rho_B;
        g_phi_valid[p] = true;
    }

    FLAMEGPU->environment.setProperty<float>("phi_cd8_c1",   static_cast<float>(g_phi[PHI_CD8_C1]));
    FLAMEGPU->environment.setProperty<float>("phi_m1_c1",    static_cast<float>(g_phi[PHI_M1_C1]));
    FLAMEGPU->environment.setProperty<float>("phi_cd8_treg", static_cast<float>(g_phi[PHI_CD8_TREG]));
    FLAMEGPU->environment.setProperty<float>("phi_m2_c1",    static_cast<float>(g_phi[PHI_M2_C1]));
    FLAMEGPU->environment.setProperty<float>("phi_icaf_c1",  static_cast<float>(g_phi[PHI_ICAF_C1]));
    FLAMEGPU->environment.setProperty<float>("phi_mdsc_cd8", static_cast<float>(g_phi[PHI_MDSC_CD8]));
    FLAMEGPU->environment.setProperty<float>("phi_bcell_cd8",static_cast<float>(g_phi[PHI_BCELL_CD8]));
    FLAMEGPU->environment.setProperty<float>("phi_vas_c1",   static_cast<float>(g_phi[PHI_VAS_C1]));

    if (!g_phi_csv.is_open()) {
        g_phi_csv.open(g_phi_csv_path);
        g_phi_csv << "step"
                  << ",phi_cd8_c1,valid_cd8_c1,n_c1,n_cd8"
                  << ",phi_m1_c1,valid_m1_c1,n_m1"
                  << ",phi_cd8_treg,valid_cd8_treg,n_treg"
                  << ",phi_m2_c1,valid_m2_c1,n_m2"
                  << ",phi_icaf_c1,valid_icaf_c1,n_icaf"
                  << ",phi_mdsc_cd8,valid_mdsc_cd8,n_mdsc"
                  << ",phi_bcell_cd8,valid_bcell_cd8,n_bcell"
                  << ",phi_vas_c1,valid_vas_c1,n_vas\n";
    }
    if (g_phi_csv.is_open()) {
        g_phi_csv << FLAMEGPU->getStepCounter()
            << ',' << g_phi[PHI_CD8_C1]   << ',' << (g_phi_valid[PHI_CD8_C1] ? 1 : 0)
            << ',' << cnt[PHI_CD8_C1]     << ',' << cnt[N_PHI_PAIR + PHI_KDE_CD8]
            << ',' << g_phi[PHI_M1_C1]    << ',' << (g_phi_valid[PHI_M1_C1] ? 1 : 0)
            << ',' << cnt[N_PHI_PAIR + PHI_KDE_MAC_M1]
            << ',' << g_phi[PHI_CD8_TREG] << ',' << (g_phi_valid[PHI_CD8_TREG] ? 1 : 0)
            << ',' << cnt[PHI_CD8_TREG]
            << ',' << g_phi[PHI_M2_C1]    << ',' << (g_phi_valid[PHI_M2_C1] ? 1 : 0)
            << ',' << cnt[N_PHI_PAIR + PHI_KDE_MAC_M2]
            << ',' << g_phi[PHI_ICAF_C1]  << ',' << (g_phi_valid[PHI_ICAF_C1] ? 1 : 0)
            << ',' << cnt[N_PHI_PAIR + PHI_KDE_FIB_ICAF]
            << ',' << g_phi[PHI_MDSC_CD8] << ',' << (g_phi_valid[PHI_MDSC_CD8] ? 1 : 0)
            << ',' << cnt[N_PHI_PAIR + PHI_KDE_MDSC]
            << ',' << g_phi[PHI_BCELL_CD8]<< ',' << (g_phi_valid[PHI_BCELL_CD8] ? 1 : 0)
            << ',' << cnt[N_PHI_PAIR + PHI_KDE_BCELL]
            << ',' << g_phi[PHI_VAS_C1]   << ',' << (g_phi_valid[PHI_VAS_C1] ? 1 : 0)
            << ',' << cnt[N_PHI_PAIR + PHI_KDE_VAS] << '\n';
        g_phi_csv.flush();
    }
}

FLAMEGPU_HOST_FUNCTION(zero_fib_density_field) {
    nvtxRangePush("Zero Fib Density");
    if (d_fib_density_field && g_pde_solver) {
        int total_voxels = g_pde_solver->get_total_voxels();
        cudaMemset(d_fib_density_field, 0, total_voxels * sizeof(unsigned long long));
    }
    nvtxRangePop();
}

// ============================================================================
// Vvas as a STATE WITH MEMORY (coupling contract §6.3), + Krogh O2 (media-2 Eq.1-3).
//
//   ∂V_vas/∂t = k_growth·Hill(VEGFA)·V_vas·(1 − V_vas/V_max) [angiogenesis]
//             − k_decay·V_vas                                [regression]
//             − occlusion(ECM, cancer)·V_vas                 [desmoplastic compression]
//             + recovery·V_vas·(1 − V_vas/V_max)             [ECM relief, e.g. losartan]
//
// REPLACES the previous `V_vas = Hill(VEGFA)·(1 − f_cancer)·cabo`, which was a category
// error rather than a calibration one: VEGFA is a *demand* signal and was being read as a
// *supply* state. Hypoxic cells secrete VEGFA precisely because they are under-perfused, so
// the old form inverted the perfusion↔oxygenation correlation, built an unintended
// homeostatic controller (VEGFA↑→V_vas↑→O2↑→HIF↓→VEGFA↓) that flattens the O2 field, and
// would have delivered drug preferentially to avascular hypoxic regions — inverting the
// headline chemo result. Under placement it would also have set the INITIAL CONDITION, so
// the inversion would be baked into the seed rather than merely accumulated over a run.
//
// All rate parameters are ALREADY per-step (PARAM_VAS_REGRESS_RATE is
// `QP(P_k_vas_decay) * t_step_sec`), so nothing here is multiplied by dt again.
//
// GROWTH IS LOGISTIC IN V_vas (2026-08-08). The first version of this law had growth
// independent of V — `k_growth·Hill(VEGFA)·(1 − V/Vmax)` — i.e. a constant source wherever
// VEGFA was present, so a voxel at V_vas = 0 still grew at k_growth per step. Vessels
// appeared from nothing in avascular tissue, which made the fixed point globally attracting:
// measured (analysis/vvas_o2_check.py) the equilibrium field was identical to six digits
// whether started from the placement seed or from the uniform fallback — 0.1877 either way,
// with the seeded contrast of 4.65 collapsing to 1.31 in ~7 steps. Under §C6 placement that
// is fatal: it erases V_vas(t=0) before treatment onset, so seeding could not matter.
// Multiplying by V restores V=0 as a fixed point, so placed avascular tissue STAYS avascular
// and occlusion can drive vessels to genuine extinction rather than to a lower uniform level.
// The same argument applies to the recovery branch, which was also V-independent: a voxel
// with no vessel left has nothing to decompress.
//
// Two departures from the old kernel worth noting:
//   * `cabo_term` now multiplies GROWTH only. Anti-VEGFR2 blocks new vessel formation; it
//     does not instantaneously delete existing vessels, which is what scaling the whole
//     field implied.
//   * `f_cancer` is no longer a multiplicative mask but a COMPRESSION DRIVER. Compression is
//     otherwise ECM-driven, and dense cancer nests contain no fibroblasts depositing ECM —
//     so an ECM-only law gives nests zero compression, which is backwards. Cancer occupancy
//     and ECM are treated as independent occlusion pathways.
// ============================================================================
__global__ void update_vvas_kernel(
    float* vvas,                                        // in/out: the state
    const float* __restrict__ vegf,
    const unsigned int* __restrict__ cancer_occ,
    const float* __restrict__ ecm_density,
    int nx, int ny, int nz,
    float vegf_ec50, float cabo_term,
    float k_growth, float k_decay, float v_max,
    float collapse_th, float collapse_ec50,
    float recovery_th, float recovery_rate,
    float maturity_factor, float cancer_w)
{
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ty = blockIdx.y * blockDim.y + threadIdx.y;
    const int tz = blockIdx.z * blockDim.z + threadIdx.z;
    if (tx >= nx || ty >= ny || tz >= nz) return;
    const int idx = tz * (nx * ny) + ty * nx + tx;

    float V = vvas[idx];

    // Hill(VEGF) — angiogenic drive (EC50 reuses PARAM_VAS_50, the sprouting VEGF EC50)
    const float v = vegf[idx];
    const float hill = v / (v + vegf_ec50 + 1e-30f);

    // cancer Moore fraction, n=2 (5x5x5, up to 124 closest), edge-corrected
    int box_total = 0, box_cancer = 0;
    for (int dz = -2; dz <= 2; dz++)
    for (int dy = -2; dy <= 2; dy++)
    for (int dx = -2; dx <= 2; dx++) {
        int cx = tx + dx, cy = ty + dy, cz = tz + dz;
        if (cx < 0 || cx >= nx || cy < 0 || cy >= ny || cz < 0 || cz >= nz) continue;
        box_total++;
        box_cancer += (cancer_occ[cz * (nx * ny) + cy * nx + cx] > 0u) ? 1 : 0;
    }
    const float f_cancer = (box_total > 0) ? (float)box_cancer / (float)box_total : 0.0f;

    // --- growth: logistic IN V, VEGFA-gated, blocked by anti-angiogenic therapy ---
    // The V factor is what keeps V=0 a fixed point; see the header note.
    const float logistic = V * fmaxf(0.0f, 1.0f - V / (v_max + 1e-30f));
    const float growth = k_growth * hill * cabo_term * logistic;

    // --- regression ---
    const float decay = k_decay * V;

    // --- occlusion: ECM and cancer packing as INDEPENDENT pathways ---
    // The ECM branch is the agent-era collapse law ported to a continuous rate
    // (vascular_cell.cuh: p_collapse = Hill(ECM)/(1 + mat_resist·maturity), gated above
    // PARAM_VAS_COLLAPSE_THRESHOLD). `maturity_factor` folds in that resistance at the
    // pre-existing-vessel maturity, which is the right regime: PDAC vasculature is largely
    // co-opted mature stromal vessels, not de novo sprouts.
    const float ecm = ecm_density[idx];
    const float c_ecm = (ecm > collapse_th)
                      ? (ecm / (ecm + collapse_ec50 + 1e-30f)) * maturity_factor
                      : 0.0f;
    const float c_can = fminf(1.0f, cancer_w * f_cancer) * maturity_factor;
    const float occl = 1.0f - (1.0f - c_ecm) * (1.0f - c_can);   // P(occluded by either)
    const float compress = occl * V;

    // --- recovery once ECM relaxes below the recovery threshold (losartan-anchored) ---
    // Also logistic in V: decompression re-opens vessels that are still there, so a voxel
    // scoured to V=0 does not spontaneously re-vascularize.
    const float recover = (ecm < recovery_th) ? recovery_rate * logistic : 0.0f;

    V += growth - decay - compress + recover;
    vvas[idx] = fminf(fmaxf(V, 0.0f), v_max);
}

// Uniform initial perfusion, used when no placement seed has been supplied. Once
// `seeding_vvas.seed_vvas` is wired into the handoff this is overwritten by the
// rasterized-and-smoothed vessel field; until then it keeps O2 delivery sane, since a
// field starting at exactly zero would starve the domain while growth ramps.
__global__ void fill_vvas_kernel(float* vvas, int total, float value)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) vvas[i] = value;
}

__global__ void source_o2_from_vvas_kernel(
    const float* __restrict__ vvas,
    const float* __restrict__ o2_conc,
    float* o2_src, float* o2_upt,
    int nx, int ny, int nz,
    float KvLv_base, float C_blood, float voxel_volume)
{
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ty = blockIdx.y * blockDim.y + threadIdx.y;
    const int tz = blockIdx.z * blockDim.z + threadIdx.z;
    if (tx >= nx || ty >= ny || tz >= nz) return;
    const int idx = tz * (nx * ny) + ty * nx + tx;

    const float vv = vvas[idx];
    if (vv <= 0.0f) return;
    // implicit-split gate: only deliver where local O2 below blood (drives up to C_blood)
    if (o2_conc[idx] >= C_blood) return;

    const float G = KvLv_base * vv;  // effective Krogh transport coeff [cm^3/s], scaled by vascular fraction
    // Additive into reset buffers (reset_pde_buffers + cancer O2 uptake already applied this step)
    atomicAdd(&o2_src[idx], G * C_blood / voxel_volume);  // [conc/s]
    atomicAdd(&o2_upt[idx], G / voxel_volume);            // [1/s]
}

// Host layer: decode the fixed-point agent source/uptake staging buffers onto the float
// d_src_/d_upt_ the solver reads. Runs AFTER compute_chemical_sources (agents accumulate FP)
// and BEFORE compute_vvas_and_o2 (Krogh adds float O2 on top) and solve_pde. Makes the source
// accumulation deterministic (exact integer sum) — see PDE_SECRETE in common.cuh.
FLAMEGPU_HOST_FUNCTION(decode_pde_sources) {
    if (g_pde_solver) g_pde_solver->decode_sources_from_fp();
}

// Host layer: compute Vvas, then source O2 from it. Runs after compute_chemical_sources,
// before solve_pde. d_vvas_field persists for recruitment entry-point marking (Step 3).
FLAMEGPU_HOST_FUNCTION(compute_vvas_and_o2) {
    nvtxRangePush("Vvas + O2");
    if (!g_pde_solver || !d_vvas_field || !d_cancer_occ) { nvtxRangePop(); return; }

    const int nx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int nz = FLAMEGPU->environment.getProperty<int>("grid_size_z");

    const float vegf_ec50 = FLAMEGPU->environment.getProperty<float>("PARAM_VAS_50");
    // cabo term (media-2 Eq.1): (1 - k_cabo*cabo/(cabo+IC50_VEGFR2)). cabo=0 pre-treatment -> 1.0.
    // TODO Step 4 (therapy): wire PARAM_K_CABO + PARAM_IC50_VEGFR2 for the anti-angiogenic arm.
    const float cabo_term = 1.0f;

    // --- V_vas state-law parameters (contract §6.3) ---
    const float k_growth = FLAMEGPU->environment.getProperty<float>("PARAM_VAS_GROWTH_RATE");
    const float k_decay  = FLAMEGPU->environment.getProperty<float>("PARAM_VAS_REGRESS_RATE");
    const float v_max    = FLAMEGPU->environment.getProperty<float>("PARAM_VAS_MAX");
    const float collapse_th    = FLAMEGPU->environment.getProperty<float>("PARAM_VAS_COLLAPSE_THRESHOLD");
    const float collapse_ec50  = FLAMEGPU->environment.getProperty<float>("PARAM_VAS_COLLAPSE_EC50");
    const float recovery_th    = FLAMEGPU->environment.getProperty<float>("PARAM_VAS_RECOVERY_THRESHOLD");
    const float recovery_rate  = FLAMEGPU->environment.getProperty<float>("PARAM_VAS_RECOVERY_RATE");
    const float cancer_w       = FLAMEGPU->environment.getProperty<float>("PARAM_VAS_COMPRESS_CANCER_W");
    // Mature-vessel resistance from the agent law, evaluated at the pre-existing-vessel
    // maturity: 1/(1 + 0.04*100) = 0.2, i.e. mature vessels collapse 5x slower. Dropping it
    // would make compression 5x too fast for exactly the co-opted vessels PDAC runs on.
    const float mat_resist = FLAMEGPU->environment.getProperty<float>("PARAM_VAS_MATURITY_RESISTANCE");
    const float init_mat   = FLAMEGPU->environment.getProperty<float>("PARAM_VAS_INITIAL_MATURITY");
    const float maturity_factor = 1.0f / (1.0f + mat_resist * init_mat);

    // One-shot uniform seed — the FALLBACK only. When a placement supplied V_vas(t=0)
    // (set_vvas_from_host), g_vvas_seeded is already true and this is skipped, which is what
    // preserves the placed avascular zero set: the logistic growth law keeps V=0 fixed, but a
    // uniform overwrite here would destroy the zero set before step 1.
    if (!g_vvas_seeded) {
        const float v0 = FLAMEGPU->environment.getProperty<float>("PARAM_VAS_INIT");
        const int total = nx * ny * nz;
        fill_vvas_kernel<<<(total + 255) / 256, 256>>>(d_vvas_field, total, v0);
        g_vvas_seeded = true;
    }

    // Krogh transport coefficient — constants only; mirrors vascular_cell.cuh O2 block exactly.
    const float pi      = 3.1415926f;
    const float sigma   = FLAMEGPU->environment.getProperty<float>("PARAM_VAS_SIGMA");
    const float RC      = FLAMEGPU->environment.getProperty<float>("PARAM_VAS_RC");
    const float C_blood = FLAMEGPU->environment.getProperty<float>("PARAM_VAS_O2_CONC");
    const float D_O2    = FLAMEGPU->environment.getProperty<float>("PARAM_O2_DIFFUSIVITY");
    const float vs_cm   = FLAMEGPU->environment.getProperty<float>("voxel_size") * 1.0e-4f;
    const float voxel_volume = vs_cm * vs_cm * vs_cm;
    const float Lv     = voxel_volume / (RC * RC * pi);
    const float Rt     = 1.0f / std::sqrt(Lv * pi);
    const float w      = RC / Rt;
    const float lambda = 1.0f - w * w;
    const float Kv = 2.0f * pi * D_O2
                     * (lambda / (sigma * lambda - (2.0f + lambda) / 4.0f + (1.0f / lambda) * std::log(1.0f / w)));
    const float KvLv_base = Kv * Lv;

    const float* vegf   = g_pde_solver->get_device_concentration_ptr(CHEM_VEGFA);
    const float* o2conc = g_pde_solver->get_device_concentration_ptr(CHEM_O2);
    float* o2_src = g_pde_solver->get_device_source_ptr(CHEM_O2);
    float* o2_upt = g_pde_solver->get_device_uptake_ptr(CHEM_O2);

    dim3 block(8, 8, 8);
    dim3 grid((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);
    update_vvas_kernel<<<grid, block>>>(d_vvas_field, vegf, d_cancer_occ, d_ecm_density,
                                        nx, ny, nz, vegf_ec50, cabo_term,
                                        k_growth, k_decay, v_max,
                                        collapse_th, collapse_ec50,
                                        recovery_th, recovery_rate,
                                        maturity_factor, cancer_w);
    source_o2_from_vvas_kernel<<<grid, block>>>(d_vvas_field, o2conc, o2_src, o2_upt, nx, ny, nz,
                                                KvLv_base, C_blood, voxel_volume);
    cudaDeviceSynchronize();
    nvtxRangePop();
}

// ============================================================================
// Recruitment entry points from Vvas (media-2 Eq.4): replaces vascular-agent marking.
//   p_entry(voxel) = clamp(Vvas * PARAM_ENTRY_ADHESION_SCALE, 1)   [ = Vvas*rho_adh*Vvox/n_adh ]
// At an entry point, set ALL immune-type bits in d_recruitment_sources (T=1,MDSC=2,MAC=4,B=8,DC=16).
// Per-type cytokine gating (CXCL9/CXCL12 for effectors, CXCL13 for B, CCL2/IL-6 for MDSC, CCL2 for
// MAC, CCL5 for Treg) is applied as continuous soft gates in the Stage-B recruit_all_kernel roll.
// ============================================================================
__global__ void mark_entry_points_kernel(
    int* recruit_sources,
    const float* __restrict__ vvas,
    int nx, int ny, int nz,
    float entry_scale,
    unsigned int seed)
{
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ty = blockIdx.y * blockDim.y + threadIdx.y;
    const int tz = blockIdx.z * blockDim.z + threadIdx.z;
    if (tx >= nx || ty >= ny || tz >= nz) return;
    const int idx = tz * (nx * ny) + ty * nx + tx;

    const float vv = vvas[idx];
    if (vv <= 0.0f) return;
    const float p_entry = fminf(vv * entry_scale, 1.0f);

    unsigned int rng = seed ^ (idx * 2654435761u + 1u);
    xorshift32(rng);  // warm up
    if (rng_uniform(rng) >= p_entry) return;  // not an entry point this step

    // All immune types are Vvas-entry-gated here; cytokine gating (incl. CCL2/CCR2 for MDSC/MAC)
    // now lives in the Stage-B recruit roll as continuous soft gates (gate-vs-roll review).
    const int bits = 1 | 2 | 4 | 8 | 16;  // T, MDSC, MAC, B, DC
    atomicOr(&recruit_sources[idx], bits);
}

// Host layer: mark Vvas-driven recruitment entry points into the bitmask the recruit
// kernel consumes. Replaces vascular_mark_sources; runs after reset_recruitment_sources,
// before recruit_gpu. Reads d_vvas_field (computed previous step's PDE phase; 1-step lag).
FLAMEGPU_HOST_FUNCTION(mark_entry_points) {
    nvtxRangePush("Mark Entry Points");
    if (!g_pde_solver || !d_vvas_field) { nvtxRangePop(); return; }
    const int nx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int nz = FLAMEGPU->environment.getProperty<int>("grid_size_z");

    const float entry_scale = FLAMEGPU->environment.getProperty<float>("PARAM_ENTRY_ADHESION_SCALE");
    const unsigned int base_seed = FLAMEGPU->environment.getProperty<unsigned int>("sim_seed");
    // distinct salt from recruit_all_kernel to decorrelate the two stochastic passes
    const unsigned int seed = base_seed ^ (static_cast<unsigned int>(FLAMEGPU->getStepCounter()) * 40503u + 0x9E3779B9u);

    // CCL2 myeloid gating moved to the Stage-B recruit roll (gate-vs-roll review) — not needed here.
    int* recruit_sources = g_pde_solver->get_device_recruitment_sources_ptr();

    dim3 block(8, 8, 8);
    dim3 grid((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);
    mark_entry_points_kernel<<<grid, block>>>(recruit_sources, d_vvas_field, nx, ny, nz,
                                              entry_scale, seed);
    cudaDeviceSynchronize();
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
        d_ecm_floor,
        d_fib_density_field, tgfb_ptr, mmp_ptr,
        grid_x, grid_y, grid_z,
        voxel_vol_cm3, dt,
        k_decay, k_depo, density_cap,
        tgfb_ec50,
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
// Seed the antigen grid at specified voxel indices with `value`.
// Builds a sparse host-side delta and copies into device memory voxel-by-voxel via
// a transient buffer (one cudaMemcpy). Called from initialization before main sim.
void seed_antigen_grid_voxels(const int* voxel_indices, int n, float value) {
    if (!d_antigen_grid || !g_pde_solver || n <= 0 || value == 0.0f) return;
    int total_voxels = g_pde_solver->get_total_voxels();
    std::vector<float> host_grid(total_voxels, 0.0f);
    // Pull current device contents so we add to (not overwrite) any existing antigen.
    cudaMemcpy(host_grid.data(), d_antigen_grid,
               total_voxels * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) {
        int v = voxel_indices[i];
        if (v >= 0 && v < total_voxels) host_grid[v] += value;
    }
    cudaMemcpy(d_antigen_grid, host_grid.data(),
               total_voxels * sizeof(float), cudaMemcpyHostToDevice);
}

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
    // ABM-only mode: QSP is frozen, no point aggregating ABM→QSP feedback events.
    if (FLAMEGPU->environment.getProperty<int>("step_qsp") == 0) {
        nvtxRangePop();
        return;
    }
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
    // ABM-only mode: QSP is frozen, skip the copy (downstream QSP reads are also gated).
    if (FLAMEGPU->environment.getProperty<int>("step_qsp") == 0) {
        nvtxRangePop();
        return;
    }
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
    FLAMEGPU->environment.setProperty<int>("ABM_BCELL_REC", static_cast<int>(counters[ABM_COUNT_BCELL_REC]));
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

FLAMEGPU_HOST_FUNCTION(timing_after_occ) {
    nvtxRangePush("Timing Checkpoint: occupancy");
    record_checkpoint("occupancy");
    nvtxRangePop();
}

FLAMEGPU_HOST_FUNCTION(timing_after_bcast_out) {
    nvtxRangePush("Timing Checkpoint: bcast_out");
    record_checkpoint("bcast_out");
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