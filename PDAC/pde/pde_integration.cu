#include "pde_integration.cuh"
#include "../core/common.cuh"
#include "../core/layer_timing.h"
#include "../diagnostics/ripley.cuh"
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

// Snapshot of the SA-placed Vvas at t=0. Serves as the recovery target: compression
// can push Vvas below this, but recovery drives it back toward the initial value
// rather than toward zero. Healthy tissue vasculature is permanent infrastructure.
static float* d_vvas_initial = nullptr;

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

// ── phi estimator state (permutation-null FunCN, contract D-1/D-2). ──
static unsigned long long* d_phi_kde   = nullptr;  // [N_PHI_KDE][nvox] fixed-point KDE fields
static unsigned long long* d_phi_accum = nullptr;  // [N_PHI_PAIR] fixed-point SUM of raw KDE at targets
static unsigned int*       d_phi_count = nullptr;  // [N_PHI_PAIR targets | N_PHI_KDE sources | N_PHI_KDE in-window sources]
static double              g_phi[N_PHI_PAIR] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
static bool                g_phi_valid[N_PHI_PAIR] = {};

// Cancer window for phi: bounding box of cancer cells + margin, updated per step.
static int g_phi_win[6] = {};  // {x0, x1, y0, y1, z0, z1} inclusive
static int g_phi_win_nvox = 0;

// Permutation-null W = S/N - k0, computed per step by phi_compute_W.
static double g_phi_W = 0.0;
static unsigned int g_phi_N_total = 0;
// Persistent device accumulators for W computation (allocated once).
static double*       d_phi_S = nullptr;
static unsigned int* d_phi_N = nullptr;

// Compute W from type_counts: for each window voxel, convolve total cell counts
// with the Gaussian stencil to get kde_all, then accumulate S = Σ n_cells · kde_all
// and N = Σ n_cells. W = S/N - 1.0 (k0=1 for unnormalized kernel).
__global__ void phi_W_kernel(
    const unsigned int* type_counts, int n_classes, int nvox,
    int gx, int gy, int gz,
    float two_sigma_sq, int radius,
    int x0, int x1, int y0, int y1, int z0, int z1,
    double* out_S, unsigned int* out_N)
{
    const int win_x = x1 - x0 + 1;
    const int win_y = y1 - y0 + 1;
    const int total_win = win_x * win_y * (z1 - z0 + 1);
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_win) return;

    const int lz = tid / (win_x * win_y);
    const int ly = (tid / win_x) % win_y;
    const int lx = tid % win_x;
    const int x = x0 + lx, y = y0 + ly, z = z0 + lz;

    const int vidx = z * (gx * gy) + y * gx + x;
    unsigned int nc = 0;
    for (int c = 0; c < n_classes; ++c)
        nc += type_counts[c * nvox + vidx];
    if (nc == 0) return;

    // Gaussian stencil: kde_all at (x,y,z) = Σ_neighbors total_cells(nbr) · K(d)
    float kde = 0.0f;
    for (int dz = -radius; dz <= radius; ++dz) {
        const int nz = z + dz; if (nz < 0 || nz >= gz) continue;
        for (int dy = -radius; dy <= radius; ++dy) {
            const int ny = y + dy; if (ny < 0 || ny >= gy) continue;
            for (int dx = -radius; dx <= radius; ++dx) {
                const int nx = x + dx; if (nx < 0 || nx >= gx) continue;
                const int nidx = nz * (gx * gy) + ny * gx + nx;
                unsigned int nc_nbr = 0;
                for (int c = 0; c < n_classes; ++c)
                    nc_nbr += type_counts[c * nvox + nidx];
                kde += static_cast<float>(nc_nbr) * phi_kernel(dx, dy, dz, two_sigma_sq);
            }
        }
    }

    atomicAdd(out_S, static_cast<double>(nc) * static_cast<double>(kde));
    atomicAdd(out_N, nc);
}

double get_phi(int pair) { return (pair >= 0 && pair < N_PHI_PAIR) ? g_phi[pair] : 1.0; }
bool   phi_is_valid(int pair) { return (pair >= 0 && pair < N_PHI_PAIR) && g_phi_valid[pair]; }

// Epsilon-kernel occupancy: per-class cell counts and interaction matrix.
// d_type_counts: [N_OCC_CLASSES * total_voxels] unsigned int, row-major by class.
// d_epsilon: [N_OCC_CLASSES * N_OCC_CLASSES] float, the interaction matrix.
static unsigned int* d_type_counts = nullptr;
static float* d_epsilon = nullptr;

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

// Proliferative pressure field: scalar per-voxel pressure from cancer division events.
// Cancer division deposits a Gaussian kernel centered at the daughter voxel; field decays
// each step. Gradient biases movement of mechanosensitive cell types away from high pressure.
static float* d_pressure_field = nullptr;
static float* d_pressure_grad_x = nullptr;
static float* d_pressure_grad_y = nullptr;
static float* d_pressure_grad_z = nullptr;

// Necrotic debris field: dead cancer cell remnants. Blocks movement, suppresses
// Vvas recovery. Deposited by cancer death (atomicAdd), decayed each step.
static float* d_debris_field = nullptr;

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

    // 4. MMP degradation (crosslink-resistant, can push below per-voxel floor) - turned off for now, 10-13-26
    // float mmp_degrade = k_mmp * mmp * density / (1.0f + alpha_crosslink * crosslink) * dt;
    // density -= mmp_degrade;

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
// CUDA Kernel: Decay proliferative pressure field (exponential decay each step)
// ============================================================================
__global__ void decay_pressure_kernel(float* pressure, int total_voxels, float decay_factor)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_voxels) {
        pressure[idx] *= decay_factor;
    }
}

__global__ void decay_debris_kernel(float* debris, int total_voxels, float decay_factor)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_voxels) {
        debris[idx] *= decay_factor;
    }
}

// ============================================================================
// CUDA Kernel: Compute pressure field gradient (central differences)
// ============================================================================
__global__ void compute_pressure_gradient_kernel(
    const float* pressure, float* grad_x, float* grad_y, float* grad_z,
    int nx, int ny, int nz, float inv_2dx)
{
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ty = blockIdx.y * blockDim.y + threadIdx.y;
    const int tz = blockIdx.z * blockDim.z + threadIdx.z;
    if (tx >= nx || ty >= ny || tz >= nz) return;

    const int idx = tz * (nx * ny) + ty * nx + tx;

    // Central differences with Neumann (no-flux) boundary.
    // For dimensions of size 1: gradient is identically zero (no neighbor to difference).
    // For dimensions of size >1: reflect at edges (forward/backward difference via reflected index).
    int xm = (nx > 1) ? ((tx > 0)      ? idx - 1       : idx + 1)       : idx;
    int xp = (nx > 1) ? ((tx < nx - 1) ? idx + 1       : idx - 1)       : idx;
    int ym = (ny > 1) ? ((ty > 0)      ? idx - nx      : idx + nx)      : idx;
    int yp = (ny > 1) ? ((ty < ny - 1) ? idx + nx      : idx - nx)      : idx;
    int zm = (nz > 1) ? ((tz > 0)      ? idx - nx * ny : idx + nx * ny) : idx;
    int zp = (nz > 1) ? ((tz < nz - 1) ? idx + nx * ny : idx - nx * ny) : idx;

    grad_x[idx] = (pressure[xp] - pressure[xm]) * inv_2dx;
    grad_y[idx] = (pressure[yp] - pressure[ym]) * inv_2dx;
    grad_z[idx] = (pressure[zp] - pressure[zm]) * inv_2dx;
}

// ============================================================================
// Host Function: Reset PDE Buffers (call before compute_chemical_sources)
// ============================================================================

FLAMEGPU_HOST_FUNCTION(reset_pde_buffers) {
    nvtxRangePush("Reset PDE Buffers");
    if (!g_pde_solver) { nvtxRangePop(); return; }
    g_pde_solver->reset_sources();
    g_pde_solver->reset_uptakes();
    cuda_layer_check("reset_pde_buffers");
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
    cuda_layer_check("solve_pde_step");
    nvtxRangePop();
}

// ============================================================================
// Host Function: Compute PDE Gradients (call after solve_pde_step)
// ============================================================================

FLAMEGPU_HOST_FUNCTION(compute_pde_gradients) {
    nvtxRangePush("Compute PDE Gradients");
    if (!g_pde_solver) { nvtxRangePop(); return; }
    g_pde_solver->compute_gradients();
    cuda_layer_check("compute_pde_gradients");
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
    config.decay_rates[CHEM_ARGI]     = gpu_params.getFloat(PARAM_ARGI_DECAY_RATE);  // ABM override (QSP value too slow for spatial)
    config.decay_rates[CHEM_NO]       = gpu_params.getFloat(PARAM_NO_DECAY_RATE);    // ABM override (QSP value too slow for spatial)
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
    // Initial Vvas snapshot — recovery target for compression rebound.
    CUDA_CHECK(cudaMalloc(&d_vvas_initial, total_voxels * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_vvas_initial, 0, total_voxels * sizeof(float)));

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
    CUDA_CHECK(cudaMalloc(&d_phi_count, (N_PHI_PAIR + 2 * N_PHI_KDE) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_phi_count, 0, (N_PHI_PAIR + 2 * N_PHI_KDE) * sizeof(unsigned int)));
    // Persistent device accumulators for permutation-null W computation.
    CUDA_CHECK(cudaMalloc(&d_phi_S, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_phi_N, sizeof(unsigned int)));
    std::cout << "  phi estimator: permutation-null, sigma="
              << gpu_params.getFloat(PARAM_PHI_SIGMA_VOX)
              << " vox, radius=" << gpu_params.getInt(PARAM_PHI_RADIUS_VOX) << " vox" << std::endl;

    // Allocate epsilon-kernel type count grid: N_OCC_CLASSES planes of total_voxels each
    {
        size_t tc_bytes = static_cast<size_t>(N_OCC_CLASSES) * total_voxels * sizeof(unsigned int);
        CUDA_CHECK(cudaMalloc(&d_type_counts, tc_bytes));
        CUDA_CHECK(cudaMemset(d_type_counts, 0, tc_bytes));
        std::cout << "  Epsilon type_counts: " << N_OCC_CLASSES << " classes × "
                  << total_voxels << " voxels = " << (tc_bytes / (1024*1024)) << " MB" << std::endl;
    }
    // Allocate epsilon interaction matrix on device
    {
        size_t eps_bytes = N_OCC_CLASSES * N_OCC_CLASSES * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_epsilon, eps_bytes));
        float h_epsilon[N_OCC_CLASSES * N_OCC_CLASSES];
        for (int i = 0; i < N_OCC_CLASSES; i++)
            for (int j = 0; j < N_OCC_CLASSES; j++)
                h_epsilon[i * N_OCC_CLASSES + j] = gpu_params.getFloat(
                    static_cast<GPUParamFloat>(PARAM_EPS_CANCER_CANCER + i * N_OCC_CLASSES + j));
        CUDA_CHECK(cudaMemcpy(d_epsilon, h_epsilon, eps_bytes, cudaMemcpyHostToDevice));
        std::cout << "  Epsilon matrix loaded (" << N_OCC_CLASSES << "×" << N_OCC_CLASSES << ")" << std::endl;
    }
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

    // Allocate proliferative pressure field + gradient arrays (init = 0)
    CUDA_CHECK(cudaMalloc(&d_pressure_field, total_voxels * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_pressure_field, 0, total_voxels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pressure_grad_x, total_voxels * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_pressure_grad_x, 0, total_voxels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pressure_grad_y, total_voxels * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_pressure_grad_y, 0, total_voxels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pressure_grad_z, total_voxels * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_pressure_grad_z, 0, total_voxels * sizeof(float)));

    // Allocate necrotic debris field (init = 0)
    CUDA_CHECK(cudaMalloc(&d_debris_field, total_voxels * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_debris_field, 0, total_voxels * sizeof(float)));

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
    static const char* grad_names[NUM_GRAD_SUBSTRATES] = {"IFN", "TGFB", "CCL2", "VEGFA", "CXCL13", "CCL21", "CXCL12", "CCL5", "IL12"};
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
    // Cancer window bounds for phi (updated per step by zero_phi_buffers)
    model.Environment().newProperty<int>("phi_win_x0", 0);
    model.Environment().newProperty<int>("phi_win_x1", g_pde_solver->get_nx() - 1);
    model.Environment().newProperty<int>("phi_win_y0", 0);
    model.Environment().newProperty<int>("phi_win_y1", g_pde_solver->get_ny() - 1);
    model.Environment().newProperty<int>("phi_win_z0", 0);
    model.Environment().newProperty<int>("phi_win_z1", g_pde_solver->get_nz() - 1);

    // Epsilon-kernel occupancy pointers
    model.Environment().newProperty<unsigned long long>("type_counts_ptr",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(d_type_counts)));
    model.Environment().newProperty<unsigned long long>("epsilon_ptr",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(d_epsilon)));
    // Legacy volume grid pointer
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

    // Proliferative pressure field + gradient pointers
    model.Environment().newProperty<unsigned long long>("pressure_field_ptr",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(d_pressure_field)));
    model.Environment().newProperty<unsigned long long>("pressure_grad_x_ptr",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(d_pressure_grad_x)));
    model.Environment().newProperty<unsigned long long>("pressure_grad_y_ptr",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(d_pressure_grad_y)));
    model.Environment().newProperty<unsigned long long>("pressure_grad_z_ptr",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(d_pressure_grad_z)));

    // Necrotic debris field pointer
    model.Environment().newProperty<unsigned long long>("debris_field_ptr",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(d_debris_field)));

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
    if (d_vvas_initial) {
        cudaFree(d_vvas_initial);
        d_vvas_initial = nullptr;
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
    if (d_phi_S)     { cudaFree(d_phi_S);     d_phi_S = nullptr; }
    if (d_phi_N)     { cudaFree(d_phi_N);     d_phi_N = nullptr; }
    if (d_vas_tip_id_grid) {
        cudaFree(d_vas_tip_id_grid);
        d_vas_tip_id_grid = nullptr;
    }
    if (d_type_counts) { cudaFree(d_type_counts); d_type_counts = nullptr; }
    if (d_epsilon)     { cudaFree(d_epsilon);     d_epsilon = nullptr; }
    if (d_antigen_grid) { cudaFree(d_antigen_grid); d_antigen_grid = nullptr; }
    if (d_ecm_orient_x) { cudaFree(d_ecm_orient_x); d_ecm_orient_x = nullptr; }
    if (d_ecm_orient_y) { cudaFree(d_ecm_orient_y); d_ecm_orient_y = nullptr; }
    if (d_ecm_orient_z) { cudaFree(d_ecm_orient_z); d_ecm_orient_z = nullptr; }
    if (d_mech_stress_x) { cudaFree(d_mech_stress_x); d_mech_stress_x = nullptr; }
    if (d_mech_stress_y) { cudaFree(d_mech_stress_y); d_mech_stress_y = nullptr; }
    if (d_mech_stress_z) { cudaFree(d_mech_stress_z); d_mech_stress_z = nullptr; }
    if (d_pressure_field)  { cudaFree(d_pressure_field);  d_pressure_field = nullptr; }
    if (d_pressure_grad_x) { cudaFree(d_pressure_grad_x); d_pressure_grad_x = nullptr; }
    if (d_pressure_grad_y) { cudaFree(d_pressure_grad_y); d_pressure_grad_y = nullptr; }
    if (d_pressure_grad_z) { cudaFree(d_pressure_grad_z); d_pressure_grad_z = nullptr; }
    if (d_debris_field) { cudaFree(d_debris_field); d_debris_field = nullptr; }
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
    // Snapshot the placement seed as the recovery floor.
    if (d_vvas_initial) {
        CUDA_CHECK(cudaMemcpy(d_vvas_initial, host_data, total_voxels * sizeof(float),
                              cudaMemcpyHostToDevice));
    }
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

// Vascular agents removed (ABM rebuild Step 3b). O2 + entry via Vvas continuous field.
// n_vasculature_total stays at its default value (1) since there are no discrete VAS agents.

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
    float tcd4_life_mean, tcd4_treg_life_mean, tcd4_life_sd;
    int tcd4_divide_cd, tcd4_th_divide_cd, tcd4_divide_limit;
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
    float mac_ifng_response;  // linear gain [1/nM] for IFN-γ M1 drive (NOT an EC50)
    float mac_il12_ec50;
    float mac_il10_ec50, mac_tgfb_ec50, mac_il6_ec50;
    float mac_m1_pol, mac_m2_pol;  // polarization rate constants (per step)
    // B cell recruitment
    float p_bcell;
    float bcell_life_mean;  // QSP-derived, exponential draw at creation
    const float* cxcl13_conc;  // CXCL13 PDE concentration (B cell recruitment boost)
    float cxcl13_ec50_bcell;   // CXCL13 EC50 for B cell recruitment boost
    // DC recruitment (per-subtype, homeostatic)
    float p_dc_cdc1;
    float p_dc_cdc2;
    float dc_life_immature_mean, dc_life_immature_sd;
    float dc_life_mature_mean, dc_life_mature_sd;
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
    // Epsilon-kernel occupancy (passed through to try_claim_voxel)
    unsigned int* d_tc;   // type_counts device pointer
    const float* d_eps;   // epsilon matrix device pointer
    int n_voxels;         // nx * ny * nz
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
// Returns true + sets (out_x, out_y, out_z) on success; false if the voxel is too crowded.
__device__ bool try_claim_voxel(
    int sx, int sy, int sz,
    int nx, int ny, int nz,
    unsigned int* d_tc,
    int occ_class,
    int n_voxels,
    const float* d_eps,
    unsigned int& rng,
    int& out_x, int& out_y, int& out_z)
{
    (void)rng;  // placement no longer consumes RNG
    if (sx < 0 || sx >= nx || sy < 0 || sy >= ny || sz < 0 || sz >= nz) return false;

    int vidx = sz * (nx * ny) + sy * nx + sx;

    // Epsilon kernel occupancy check
    float w = epsilon_weight(d_tc, d_eps, occ_class, vidx, n_voxels);
    if (w < EPS_DIVIDE_THRESHOLD) return false;

    // Claim: increment type count
    type_count_claim(d_tc, occ_class, vidx, n_voxels);

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

    // QSP RF276: all recruits enter as M1; polarization happens post-recruitment.
    // QPLANE_PM2 = 0 → all spawns are M1 (P(M2) = 0).
    qf[QPLANE_PM2 * nvox + idx] = 0.0f;

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
                    p.d_tc, OCC_TCELL, p.n_voxels, p.d_eps,
                    rng, px, py, pz)) {
                atomicAdd(&diag->teff_place_ok, 1);
                int slot = atomicAdd(d_request_count, 1);
                if (slot < MAX_RECRUITS_PER_STEP) {
                    RecruitRequest req;
                    req.x = px; req.y = py; req.z = pz;
                    req.cell_type = CELL_TYPE_T;
                    req.cell_state = T_CELL_EFF;
                    req.life = sample_normal_life_gpu(p.t_life_mean, p.t_life_sd, rng);
                    req.divide_cd = 1 + static_cast<int>(rng_uniform(rng) * (p.t_divide_cd - 1));
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
                            p.d_tc, OCC_TCELL, p.n_voxels, p.d_eps,
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
                    p.d_tc, OCC_TREG, p.n_voxels, p.d_eps,
                    rng, px, py, pz)) {
                atomicAdd(&diag->treg_place_ok, 1);
                int slot = atomicAdd(d_request_count, 1);
                if (slot < MAX_RECRUITS_PER_STEP) {
                    RecruitRequest req;
                    req.x = px; req.y = py; req.z = pz;
                    req.cell_type = CELL_TYPE_TREG;
                    req.cell_state = TCD4_TREG;
                    req.life = sample_normal_life_gpu(p.tcd4_treg_life_mean, p.tcd4_life_sd, rng);
                    req.divide_cd = 1 + static_cast<int>(rng_uniform(rng) * (p.tcd4_divide_cd - 1));
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
                    p.d_tc, OCC_TREG, p.n_voxels, p.d_eps,
                    rng, px, py, pz)) {
                atomicAdd(&diag->th_place_ok, 1);
                int slot = atomicAdd(d_request_count, 1);
                if (slot < MAX_RECRUITS_PER_STEP) {
                    RecruitRequest req;
                    req.x = px; req.y = py; req.z = pz;
                    req.cell_type = CELL_TYPE_TREG;  // TH uses TReg agent type
                    req.cell_state = TCD4_TH;
                    req.life = sample_normal_life_gpu(p.tcd4_life_mean, p.tcd4_life_sd, rng);
                    req.divide_cd = 1 + static_cast<int>(rng_uniform(rng) * (p.tcd4_th_divide_cd - 1));
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
                            p.d_tc, OCC_TREG, p.n_voxels, p.d_eps,
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
                    p.d_tc, OCC_MDSC, p.n_voxels, p.d_eps,
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
                    p.d_tc, OCC_MAC, p.n_voxels, p.d_eps,
                    rng, px, py, pz)) {
                atomicAdd(&diag->mac_place_ok, 1);
                int slot = atomicAdd(d_request_count, 1);
                if (slot < MAX_RECRUITS_PER_STEP) {
                    RecruitRequest req;
                    req.x = px; req.y = py; req.z = pz;
                    req.cell_type = CELL_TYPE_MAC;

                    // QSP RF276: all monocyte recruits enter as M1.
                    // Polarization to M2 happens post-recruitment via mac_state_step
                    // (RF287/RF288). The old probabilistic split produced ~97% M2
                    // recruits with high k_M2_pol, collapsing M1 to <2%.
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
                    p.d_tc, OCC_BCELL, p.n_voxels, p.d_eps,
                    rng, px, py, pz)) {
                atomicAdd(&diag->bcell_place_ok, 1);
                int slot = atomicAdd(d_request_count, 1);
                if (slot < MAX_RECRUITS_PER_STEP) {
                    RecruitRequest req;
                    req.x = px; req.y = py; req.z = pz;
                    req.cell_type = CELL_TYPE_BCELL;
                    req.cell_state = BCELL_NAIVE;
                    req.life = sample_exp_life_gpu(p.bcell_life_mean, rng);
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
                    p.d_tc, OCC_DC, p.n_voxels, p.d_eps,
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
                    p.d_tc, OCC_DC, p.n_voxels, p.d_eps,
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
    p.tcd4_life_mean      = FLAMEGPU->environment.getProperty<float>("PARAM_TCD4_LIFE_MEAN_SLICE");
    p.tcd4_treg_life_mean = FLAMEGPU->environment.getProperty<float>("PARAM_TREG_LIFE_MEAN_SLICE");
    p.tcd4_life_sd        = FLAMEGPU->environment.getProperty<float>("PARAM_TCD4_LIFESPAN_SD");
    p.tcd4_divide_cd    = FLAMEGPU->environment.getProperty<int>("PARAM_TREG_DIV_INTERVAL");
    p.tcd4_th_divide_cd = FLAMEGPU->environment.getProperty<int>("PARAM_TH_DIV_INTERVAL");
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
    p.mac_ifng_response = FLAMEGPU->environment.getProperty<float>("PARAM_MAC_IFNG_RESPONSE");
    p.mac_il12_ec50 = FLAMEGPU->environment.getProperty<float>("PARAM_MAC_IL_12_EC50");
    p.mac_il10_ec50 = FLAMEGPU->environment.getProperty<float>("PARAM_MAC_IL_10_EC50");
    p.mac_tgfb_ec50 = FLAMEGPU->environment.getProperty<float>("PARAM_MAC_TGFB_EC50");
    p.mac_il6_ec50  = FLAMEGPU->environment.getProperty<float>("PARAM_MAC_IL6_M2_EC50");
    p.mac_m1_pol = FLAMEGPU->environment.getProperty<float>("PARAM_MAC_M1_POL");
    p.mac_m2_pol = FLAMEGPU->environment.getProperty<float>("PARAM_MAC_M2_POL");

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

    // Epsilon-kernel occupancy pointers
    p.d_tc = d_type_counts;
    p.d_eps = d_epsilon;
    p.n_voxels = p.nx * p.ny * p.nz;

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
            a.setVariable<int>("divide_limit", static_cast<int>(FLAMEGPU->environment.getProperty<float>("PARAM_FIB_DIV_MAX")));
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
// d_type_counts is read (stale post-movement occupancy) only to gate intra-recruitment packing;
// it is NOT written back — the later zero_occ_grid layer wipes it and write_to_occ_grid rebuilds
// it from the actual agent population (including these recruits).
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

    // 2) D2H the q-fields + current type counts (for epsilon-kernel gating; not written back).
    std::vector<float> qf(static_cast<size_t>(N_QPLANE) * nvox);
    CUDA_CHECK(cudaMemcpy(qf.data(), d_qfields, qf.size() * sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<unsigned int> h_tc(static_cast<size_t>(N_OCC_CLASSES) * nvox);
    CUDA_CHECK(cudaMemcpy(h_tc.data(), d_type_counts,
        h_tc.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    // Host-side epsilon matrix (small — 81 floats)
    std::vector<float> h_eps(N_OCC_CLASSES * N_OCC_CLASSES);
    CUDA_CHECK(cudaMemcpy(h_eps.data(), d_epsilon,
        h_eps.size() * sizeof(float), cudaMemcpyDeviceToHost));

    const unsigned int sim_seed = FLAMEGPU->environment.getProperty<unsigned int>("sim_seed");
    const unsigned int step = static_cast<unsigned int>(FLAMEGPU->getStepCounter());
    const float count_cv = FLAMEGPU->environment.getProperty<float>("PARAM_RECRUIT_COUNT_CV");

    // ── Host-side epsilon weight (mirrors device epsilon_weight in common.cuh) ──
    auto host_eps_weight = [&](int occ_class, int vidx) -> float {
        float energy = 0.0f;
        for (int j = 0; j < N_OCC_CLASSES; j++)
            energy += h_eps[occ_class * N_OCC_CLASSES + j]
                    * static_cast<float>(h_tc[j * nvox + vidx]);
        return expf(-energy);
    };

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

    // Epsilon-kernel placement: target voxel, else scan forward for an acceptable voxel.
    auto place_one = [&](int target_vox, int occ_class) -> int {
        for (int v = target_vox; v < nvox; ++v) {
            if (host_eps_weight(occ_class, v) > EPS_DIVIDE_THRESHOLD) {
                h_tc[occ_class * nvox + v] += 1u;
                return v;
            }
        }
        return -1;
    };

    // Emit one cell of stream `s` targeting voxel `tvox` (+ any coupled naive cell).
    auto emit_cell = [&](int s, int tvox, unsigned int& crng) {
        RecruitRequest req{};
        switch (s) {
        case QS_TEFF: {
            int pv = place_one(tvox, OCC_TCELL);
            if (pv < 0) return;
            decode(pv, req.x, req.y, req.z);
            req.cell_type = CELL_TYPE_T; req.cell_state = T_CELL_EFF;
            req.life = det_normal_life(crng, p.t_life_mean, p.t_life_sd);
            req.divide_cd = 1 + static_cast<int>(det_rng_uniform(crng) * (p.t_divide_cd - 1));
            req.divide_limit = p.t_divide_limit;
            req.IL2_release_remain = p.t_IL2_release;
            req.seq = req_seq++;
            requests.push_back(req);
            if (p.naive_cd8_frac > 0.0f && det_rng_uniform(crng) < p.naive_cd8_frac) {
                int npv = place_one(tvox, OCC_TCELL);
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
            int pv = place_one(tvox, OCC_TREG);
            if (pv < 0) return;
            decode(pv, req.x, req.y, req.z);
            req.cell_type = CELL_TYPE_TREG; req.cell_state = TCD4_TREG;
            req.life = det_normal_life(crng, p.tcd4_treg_life_mean, p.tcd4_life_sd);
            req.divide_cd = 1 + static_cast<int>(det_rng_uniform(crng) * (p.tcd4_divide_cd - 1));
            req.divide_limit = p.tcd4_divide_limit;
            req.TGFB_release_remain = p.tcd4_TGFB_release;
            req.CTLA4 = p.ctla4_treg;
            req.seq = req_seq++;
            requests.push_back(req);
            break;
        }
        case QS_TH: {
            int pv = place_one(tvox, OCC_TREG);
            if (pv < 0) return;
            decode(pv, req.x, req.y, req.z);
            req.cell_type = CELL_TYPE_TREG; req.cell_state = TCD4_TH;
            req.life = det_normal_life(crng, p.tcd4_life_mean, p.tcd4_life_sd);
            req.divide_cd = 1 + static_cast<int>(det_rng_uniform(crng) * (p.tcd4_th_divide_cd - 1));
            req.divide_limit = p.tcd4_divide_limit;
            req.TGFB_release_remain = p.tcd4_TGFB_release;
            req.CTLA4 = 0.0f;
            req.seq = req_seq++;
            requests.push_back(req);
            if (p.naive_cd4_frac > 0.0f && det_rng_uniform(crng) < p.naive_cd4_frac) {
                int npv = place_one(tvox, OCC_TREG);
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
            int pv = place_one(tvox, OCC_MDSC);
            if (pv < 0) return;
            decode(pv, req.x, req.y, req.z);
            req.cell_type = CELL_TYPE_MDSC; req.cell_state = 0;
            req.life = det_exp_life(crng, p.mdsc_life_mean);
            req.seq = req_seq++;
            requests.push_back(req);
            break;
        }
        case QS_MAC: {
            int pv = place_one(tvox, OCC_MAC);
            if (pv < 0) return;
            decode(pv, req.x, req.y, req.z);
            req.cell_type = CELL_TYPE_MAC;
            req.cell_state = MAC_M1;
            req.life = det_exp_life(crng, p.mac_life_mean);
            req.seq = req_seq++;
            requests.push_back(req);
            break;
        }
        case QS_BCELL: {
            int pv = place_one(tvox, OCC_BCELL);
            if (pv < 0) return;
            decode(pv, req.x, req.y, req.z);
            req.cell_type = CELL_TYPE_BCELL; req.cell_state = BCELL_NAIVE;
            req.life = det_exp_life(crng, p.bcell_life_mean);
            req.seq = req_seq++;
            requests.push_back(req);
            break;
        }
        case QS_DC1:
        case QS_DC2: {
            int pv = place_one(tvox, OCC_DC);
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

    // ── QSP-flux magnitude + per-voxel probability roll ──
    // The kernel wrote spatial shape weights w_type(v) = Vvas·gate. Here we:
    //   1. Compute V_T_ABM (binary voxel occupancy), H_C2_ABM (cancer Hill), mean cytokine Hills
    //   2. Apply cancer proximity gate to q-field (restrict placement near tumor)
    //   3. Compute per-stream QSP flux Φ [cells/step]
    //   4. Rescale q(v) = w(v) · Φ / Σw (expected cells at each voxel)
    //   5. Per-voxel Bernoulli roll: recruit cell with prob q(v)
    {
        auto envf = [&](const char* k){ return static_cast<double>(FLAMEGPU->environment.getProperty<float>(k)); };
        const int rec_mode = FLAMEGPU->environment.getProperty<int>("recruitment_mode");
        double scale[N_QSTREAM];
        if (rec_mode == 0) {
            const double NA = envf("AVOGADROS");
            const double voxel_m = FLAMEGPU->environment.getProperty<int>("PARAM_VOXEL_SIZE") / 1e6;
            const double voxel_vol = voxel_m * voxel_m * voxel_m;

            // ── V_T_ABM: binary voxel occupancy ──
            // A voxel is "tumor" if it contains cancer or any immune cell.
            // Quiescent fibroblasts and vasculature are resident tissue, excluded.
            int n_tumor_voxels = 0;
            double N_cancer = 0.0;
            for (int v = 0; v < nvox; ++v) {
                bool has_tumor = h_tc[OCC_CANCER * nvox + v] > 0
                    || h_tc[OCC_TCELL * nvox + v] > 0
                    || h_tc[OCC_TREG  * nvox + v] > 0
                    || h_tc[OCC_MAC   * nvox + v] > 0
                    || h_tc[OCC_MDSC  * nvox + v] > 0
                    || h_tc[OCC_BCELL * nvox + v] > 0
                    || h_tc[OCC_DC    * nvox + v] > 0;
                if (has_tumor) ++n_tumor_voxels;
                N_cancer += h_tc[OCC_CANCER * nvox + v];
            }
            const double V_T_ABM = n_tumor_voxels * voxel_vol;  // [m³]

            // ── H_C2_ABM: cancer concentration Hill (QSP RF68) ──
            // QSP C_total = V_T_C1 + V_T_C2 [mol] (total cancer, NOT per volume).
            // PARAM_K_C2_RECRUIT = K² [mol²] (already squared from QSP unit conversion).
            // Hill: C_total² / (C_total² + K²).  Half-max at sqrt(K²) ≈ 4494 cells.
            const double K_C2_sq = envf("PARAM_K_C2_RECRUIT");  // [mol²] — already K²
            const double C_total_mol = N_cancer / NA;            // [mol]
            const double H_C2_ABM = (C_total_mol > 0.0)
                ? (C_total_mol * C_total_mol) / (C_total_mol * C_total_mol + K_C2_sq)
                : 0.0;

            // ── Cancer proximity gate: DISABLED ──
            // Vvas spatial shape now drives recruitment location directly.
            // Cancer compression collapses intratumoral Vvas → near-zero q-field
            // inside tumor, so the gate is redundant. Kept as comment for reference.

            // ── Mean cytokine Hills over tumor voxels (myeloid magnitude) ──
            double sum_H_CCL2 = 0.0, sum_H_IL6 = 0.0;
            int n_gate_voxels = 0;
            if (n_tumor_voxels > 0) {
                std::vector<float> h_ccl2(nvox), h_il6(nvox);
                g_pde_solver->get_concentrations(h_ccl2.data(), CHEM_CCL2);
                g_pde_solver->get_concentrations(h_il6.data(), CHEM_IL6);
                const double ccl2_ec50 = static_cast<double>(p.ccl2_ec50_mdsc);
                const double il6_ec50  = static_cast<double>(p.il6_ec50_mdsc);
                for (int v = 0; v < nvox; ++v) {
                    if (h_tc[OCC_CANCER * nvox + v] == 0) continue;
                    ++n_gate_voxels;
                    double c = h_ccl2[v];
                    sum_H_CCL2 += c / (c + ccl2_ec50 + 1e-30);
                    double il6 = h_il6[v];
                    sum_H_IL6  += il6 / (il6 + il6_ec50 + 1e-30);
                }
            }
            const double H_CCL2_mean = (n_gate_voxels > 0) ? sum_H_CCL2 / n_gate_voxels : 0.0;
            const double H_IL6_mean  = (n_gate_voxels > 0) ? sum_H_IL6  / n_gate_voxels : 0.0;

            // ── Per-stream QSP flux Φ [cells/step] ──
            // Lymphocytes: Φ = RECRUIT_K · V_T_ABM · C_central · H_C2_ABM
            // Myeloid:     Φ = RECRUIT_K · V_T_ABM · H_CCL2_mean (× IL-6 for MDSC)
            // DC:          Φ = RECRUIT_K · V_T_ABM (homeostatic)
            double flux[N_QSTREAM];
            flux[QS_TEFF]  = envf("PARAM_TEFF_RECRUIT_K")  * V_T_ABM * envf("qsp_teff_central") * H_C2_ABM;
            flux[QS_TREG]  = envf("PARAM_TREG_RECRUIT_K")  * V_T_ABM * envf("qsp_treg_central") * H_C2_ABM;
            flux[QS_TH]    = envf("PARAM_TH_RECRUIT_K")    * V_T_ABM * envf("qsp_th_central")   * H_C2_ABM;
            flux[QS_BCELL] = envf("PARAM_BCELL_RECRUIT_K") * V_T_ABM * envf("qsp_bcell_central") * H_C2_ABM;
            flux[QS_MDSC]  = envf("PARAM_MDSC_RECRUIT_K")  * V_T_ABM * H_CCL2_mean * (1.0 + H_IL6_mean);
            flux[QS_MAC]   = envf("PARAM_MAC_RECRUIT_K")   * V_T_ABM * H_CCL2_mean;
            flux[QS_DC1]   = envf("PARAM_DC_RECRUIT_K_CDC1") * V_T_ABM;
            flux[QS_DC2]   = envf("PARAM_DC_RECRUIT_K_CDC2") * V_T_ABM;

            // ── CLASS AUDIT — report flux-based recruitment vs ABM populations ──
            {
                auto nstate = [&](const char* agent, int st) {
                    return static_cast<double>(FLAMEGPU->agent(agent).count<int>("cell_state", st));
                };
                const double n_teff  = nstate(AGENT_TCELL, T_CELL_EFF) + nstate(AGENT_TCELL, T_CELL_CYT);
                const double n_tcd4  = nstate(AGENT_TREG, TCD4_TH) + nstate(AGENT_TREG, TCD4_TREG);
                const double n_bcell = nstate(AGENT_BCELL, BCELL_NAIVE) + nstate(AGENT_BCELL, BCELL_ACTIVATED);
                const double n_mdsc  = FLAMEGPU->agent(AGENT_MDSC).count();
                const double n_mac   = FLAMEGPU->agent(AGENT_MACROPHAGE).count();
                const double n_dc    = FLAMEGPU->agent(AGENT_DC).count();
                if (!g_class_audit_csv.is_open()) {
                    g_class_audit_csv.open(g_class_audit_path);
                    g_class_audit_csv << "step,presim,V_T_ABM,H_C2_ABM,H_CCL2_mean,H_IL6_mean,"
                        << "flux.CD8,flux.Treg,flux.Th,flux.B,flux.MDSC,flux.MAC,flux.DC,"
                        << "abm.CD8,abm.CD8_exh,abm.CD8_naive,"
                        << "abm.TCD4,abm.Tfh,abm.TCD4_naive,"
                        << "abm.B,abm.Plasma,"
                        << "abm.MDSC,abm.MAC,abm.DC,"
                        << "abm.fib_total,abm.qPSC,abm.iCAF,abm.myCAF,abm.apCAF\n";
                }
                if (g_class_audit_csv.is_open()) {
                    g_class_audit_csv << FLAMEGPU->getStepCounter()
                        << ',' << (is_presim_mode_active() ? 1 : 0)
                        << ',' << V_T_ABM << ',' << H_C2_ABM
                        << ',' << H_CCL2_mean << ',' << H_IL6_mean
                        << ',' << flux[QS_TEFF] << ',' << flux[QS_TREG]
                        << ',' << flux[QS_TH] << ',' << flux[QS_BCELL]
                        << ',' << flux[QS_MDSC] << ',' << flux[QS_MAC]
                        << ',' << (flux[QS_DC1] + flux[QS_DC2])
                        << ',' << n_teff
                        << ',' << nstate(AGENT_TCELL, T_CELL_SUPP)
                        << ',' << nstate(AGENT_TCELL, T_CELL_NAIVE)
                        << ',' << n_tcd4
                        << ',' << nstate(AGENT_TREG, TCD4_TFH)
                        << ',' << nstate(AGENT_TREG, TCD4_NAIVE)
                        << ',' << n_bcell
                        << ',' << nstate(AGENT_BCELL, BCELL_PLASMA)
                        << ',' << n_mdsc << ',' << n_mac << ',' << n_dc
                        << ',' << (nstate(AGENT_FIBROBLAST, FIB_QUIESCENT) + nstate(AGENT_FIBROBLAST, FIB_ICAF) + nstate(AGENT_FIBROBLAST, FIB_MYCAF) + nstate(AGENT_FIBROBLAST, FIB_APCAF))
                        << ',' << nstate(AGENT_FIBROBLAST, FIB_QUIESCENT)
                        << ',' << nstate(AGENT_FIBROBLAST, FIB_ICAF)
                        << ',' << nstate(AGENT_FIBROBLAST, FIB_MYCAF)
                        << ',' << nstate(AGENT_FIBROBLAST, FIB_APCAF) << '\n';
                    g_class_audit_csv.flush();
                }
            }

            // Rescale q-fields: q(v) = w(v) · Φ / Σw
            for (int s = 0; s < N_QSTREAM; ++s) {
                const float* q = &qf[static_cast<size_t>(s) * nvox];
                double W = 0.0; for (int v = 0; v < nvox; ++v) W += q[v];
                scale[s] = (W > 0.0) ? (flux[s] / W) : 0.0;
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

    // ── Per-voxel probability roll (Bernoulli) ──
    // q(v) = expected cells at voxel v. If q(v) < 1: recruit one cell with prob q(v).
    // If q(v) >= 1: recruit floor(q(v)) guaranteed + one more with prob of fractional part.
    for (int s = 0; s < N_QSTREAM; ++s) {
        const float* q = &qf[static_cast<size_t>(s) * nvox];
        unsigned int crng = det_rng_seed(s, step, 0x7200u, sim_seed);
        for (int v = 0; v < nvox; ++v) {
            float qv = q[v];
            if (qv <= 0.0f) continue;
            int n_cells = static_cast<int>(qv);
            float frac = qv - static_cast<float>(n_cells);
            if (frac > 0.0f && det_rng_uniform(crng) < frac) ++n_cells;
            for (int k = 0; k < n_cells; ++k) {
                emit_cell(s, v, crng);
                ++drawnN[s];
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
        if (d_type_counts)     cudaMemset(d_type_counts,     0,
                                   static_cast<size_t>(N_OCC_CLASSES) * total_voxels * sizeof(unsigned int));
    }
    cuda_layer_check("zero_occupancy_grid");
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
    cuda_layer_check("reset_voxel_owner");
    nvtxRangePop();
}

// ============================================================================
// Zero Fibroblast Density Field (reset before scatter)
// Uses cudaMemset on flat device array — no D2H/H2D copy.
// ============================================================================
// ── phi: zero the per-step accumulators and compute the cancer window. ──
FLAMEGPU_HOST_FUNCTION(zero_phi_buffers) {
    if (!g_pde_solver) return;
    const int nvox = g_pde_solver->get_total_voxels();
    if (d_phi_kde)   cudaMemset(d_phi_kde, 0,
                                static_cast<size_t>(N_PHI_KDE) * nvox * sizeof(unsigned long long));
    if (d_phi_accum) cudaMemset(d_phi_accum, 0, N_PHI_PAIR * sizeof(unsigned long long));
    if (d_phi_count) cudaMemset(d_phi_count, 0, (N_PHI_PAIR + 2 * N_PHI_KDE) * sizeof(unsigned int));

    // Compute cancer bounding box + margin for windowed phi
    const int gx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int gy = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int gz = FLAMEGPU->environment.getProperty<int>("grid_size_z");
    const int margin = FLAMEGPU->environment.getProperty<int>("PARAM_PHI_WIN_MARGIN");

    int xmin = gx, xmax = -1, ymin = gy, ymax = -1, zmin = gz, zmax = -1;
    auto cc_api = FLAMEGPU->agent("CancerCell");
    if (cc_api.count() > 0) {
        flamegpu::DeviceAgentVector cc = cc_api.getPopulationData();
        for (unsigned i = 0; i < cc.size(); ++i) {
            if (cc[i].getVariable<int>("dead") != 0) continue;
            const int x = cc[i].getVariable<int>("x");
            const int y = cc[i].getVariable<int>("y");
            const int z = cc[i].getVariable<int>("z");
            if (x < xmin) xmin = x; if (x > xmax) xmax = x;
            if (y < ymin) ymin = y; if (y > ymax) ymax = y;
            if (z < zmin) zmin = z; if (z > zmax) zmax = z;
        }
    }
    if (xmax < 0) {
        // No live cancer cells — fall back to full domain
        xmin = 0; xmax = gx - 1;
        ymin = 0; ymax = gy - 1;
        zmin = 0; zmax = gz - 1;
    } else {
        xmin = std::max(0, xmin - margin); xmax = std::min(gx - 1, xmax + margin);
        ymin = std::max(0, ymin - margin); ymax = std::min(gy - 1, ymax + margin);
        zmin = std::max(0, zmin - margin); zmax = std::min(gz - 1, zmax + margin);
    }
    g_phi_win[0] = xmin; g_phi_win[1] = xmax;
    g_phi_win[2] = ymin; g_phi_win[3] = ymax;
    g_phi_win[4] = zmin; g_phi_win[5] = zmax;
    g_phi_win_nvox = (xmax - xmin + 1) * (ymax - ymin + 1) * (zmax - zmin + 1);

    FLAMEGPU->environment.setProperty<int>("phi_win_x0", xmin);
    FLAMEGPU->environment.setProperty<int>("phi_win_x1", xmax);
    FLAMEGPU->environment.setProperty<int>("phi_win_y0", ymin);
    FLAMEGPU->environment.setProperty<int>("phi_win_y1", ymax);
    FLAMEGPU->environment.setProperty<int>("phi_win_z0", zmin);
    FLAMEGPU->environment.setProperty<int>("phi_win_z1", zmax);
}

// ── phi: compute permutation-null W from type_counts (between deposit and sample). ──
FLAMEGPU_HOST_FUNCTION(phi_compute_W) {
    if (!g_pde_solver || !d_type_counts || !d_phi_S) return;

    const int gx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int gy = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int gz = FLAMEGPU->environment.getProperty<int>("grid_size_z");
    const int nvox = gx * gy * gz;
    const float sigma = FLAMEGPU->environment.getProperty<float>("PARAM_PHI_SIGMA_VOX");
    const int radius = FLAMEGPU->environment.getProperty<int>("PARAM_PHI_RADIUS_VOX");
    const float two_s2 = 2.0f * sigma * sigma;

    CUDA_CHECK(cudaMemset(d_phi_S, 0, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_phi_N, 0, sizeof(unsigned int)));

    const int win_total = g_phi_win_nvox;
    if (win_total > 0) {
        const int threads = 256;
        phi_W_kernel<<<(win_total + threads - 1) / threads, threads>>>(
            d_type_counts, N_OCC_CLASSES, nvox, gx, gy, gz,
            two_s2, radius,
            g_phi_win[0], g_phi_win[1], g_phi_win[2], g_phi_win[3],
            g_phi_win[4], g_phi_win[5],
            d_phi_S, d_phi_N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    double h_S;
    unsigned int h_N;
    CUDA_CHECK(cudaMemcpy(&h_S, d_phi_S, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_N, d_phi_N, sizeof(unsigned int), cudaMemcpyDeviceToHost));

    g_phi_N_total = h_N;
    g_phi_W = (h_N >= 2) ? (h_S / static_cast<double>(h_N) - 1.0) : 0.0;
}

// ── phi: reduce accumulators to permutation-null phi and hand to QSP. ──
//
//   phi(A,B) = T_obs / E_perm
//   T_obs    = (1/n_A) · Σ_a KDE_B(x_a)
//   E_perm   = n_B · W / (N - 1)
//   W        = S/N - 1.0          (k0 = 1 for unnormalized kernel)
//
// ⚠️ A pair with NO cells on either side has no measurement — phi falls back to 1 (mean-field)
// and is flagged invalid, rather than being computed as 0/0 or silently as 0.
FLAMEGPU_HOST_FUNCTION(finalize_phi) {
    if (!g_pde_solver || !d_phi_accum) return;

    unsigned long long acc[N_PHI_PAIR];
    unsigned int cnt[N_PHI_PAIR + 2 * N_PHI_KDE];
    cudaMemcpy(acc, d_phi_accum, sizeof(acc), cudaMemcpyDeviceToHost);
    cudaMemcpy(cnt, d_phi_count, sizeof(cnt), cudaMemcpyDeviceToHost);

    for (int p = 0; p < N_PHI_PAIR; ++p) {
        const unsigned int n_tgt = cnt[p];
        const int kde = phi_pair_kde(p);
        const unsigned int n_src_win = cnt[N_PHI_PAIR + N_PHI_KDE + kde];
        if (n_tgt == 0 || n_src_win == 0 || g_phi_W < 1e-12 || g_phi_N_total < 2) {
            g_phi[p] = 1.0;
            g_phi_valid[p] = false;
            continue;
        }
        const double T_obs = static_cast<double>(acc[p])
                           / static_cast<double>(ecm_fp_scale())
                           / static_cast<double>(n_tgt);
        const double E_perm = static_cast<double>(n_src_win) * g_phi_W
                            / static_cast<double>(g_phi_N_total - 1);
        g_phi[p] = T_obs / E_perm;
        g_phi_valid[p] = true;
    }

    FLAMEGPU->environment.setProperty<float>("phi_cd8_c1",   static_cast<float>(g_phi[PHI_CD8_C1]));
    FLAMEGPU->environment.setProperty<float>("phi_m1_c1",    static_cast<float>(g_phi[PHI_M1_C1]));
    FLAMEGPU->environment.setProperty<float>("phi_cd8_treg", static_cast<float>(g_phi[PHI_CD8_TREG]));
    FLAMEGPU->environment.setProperty<float>("phi_m2_c1",    static_cast<float>(g_phi[PHI_M2_C1]));
    FLAMEGPU->environment.setProperty<float>("phi_icaf_c1",  static_cast<float>(g_phi[PHI_ICAF_C1]));
    FLAMEGPU->environment.setProperty<float>("phi_mdsc_cd8", static_cast<float>(g_phi[PHI_MDSC_CD8]));
    FLAMEGPU->environment.setProperty<float>("phi_bcell_cd8",static_cast<float>(g_phi[PHI_BCELL_CD8]));
    FLAMEGPU->environment.setProperty<float>("phi_mycaf_c1", static_cast<float>(g_phi[PHI_MYCAF_C1]));

    if (!g_phi_csv.is_open()) {
        g_phi_csv.open(g_phi_csv_path);
        g_phi_csv << "step,win_nvox,W,N_total"
                  << ",phi_cd8_c1,valid_cd8_c1,n_c1,n_cd8,n_cd8_win"
                  << ",phi_m1_c1,valid_m1_c1,n_m1,n_m1_win"
                  << ",phi_cd8_treg,valid_cd8_treg,n_treg"
                  << ",phi_m2_c1,valid_m2_c1,n_m2,n_m2_win"
                  << ",phi_icaf_c1,valid_icaf_c1,n_icaf,n_icaf_win"
                  << ",phi_mdsc_cd8,valid_mdsc_cd8,n_mdsc,n_mdsc_win"
                  << ",phi_bcell_cd8,valid_bcell_cd8,n_bcell,n_bcell_win"
                  << ",phi_mycaf_c1,valid_mycaf_c1,n_mycaf,n_mycaf_win\n";
    }
    if (g_phi_csv.is_open()) {
        g_phi_csv << FLAMEGPU->getStepCounter() << ',' << g_phi_win_nvox
            << ',' << g_phi_W << ',' << g_phi_N_total
            << ',' << g_phi[PHI_CD8_C1]   << ',' << (g_phi_valid[PHI_CD8_C1] ? 1 : 0)
            << ',' << cnt[PHI_CD8_C1]     << ',' << cnt[N_PHI_PAIR + PHI_KDE_CD8]
            << ',' << cnt[N_PHI_PAIR + N_PHI_KDE + PHI_KDE_CD8]
            << ',' << g_phi[PHI_M1_C1]    << ',' << (g_phi_valid[PHI_M1_C1] ? 1 : 0)
            << ',' << cnt[N_PHI_PAIR + PHI_KDE_MAC_M1]
            << ',' << cnt[N_PHI_PAIR + N_PHI_KDE + PHI_KDE_MAC_M1]
            << ',' << g_phi[PHI_CD8_TREG] << ',' << (g_phi_valid[PHI_CD8_TREG] ? 1 : 0)
            << ',' << cnt[PHI_CD8_TREG]
            << ',' << g_phi[PHI_M2_C1]    << ',' << (g_phi_valid[PHI_M2_C1] ? 1 : 0)
            << ',' << cnt[N_PHI_PAIR + PHI_KDE_MAC_M2]
            << ',' << cnt[N_PHI_PAIR + N_PHI_KDE + PHI_KDE_MAC_M2]
            << ',' << g_phi[PHI_ICAF_C1]  << ',' << (g_phi_valid[PHI_ICAF_C1] ? 1 : 0)
            << ',' << cnt[N_PHI_PAIR + PHI_KDE_FIB_ICAF]
            << ',' << cnt[N_PHI_PAIR + N_PHI_KDE + PHI_KDE_FIB_ICAF]
            << ',' << g_phi[PHI_MDSC_CD8] << ',' << (g_phi_valid[PHI_MDSC_CD8] ? 1 : 0)
            << ',' << cnt[N_PHI_PAIR + PHI_KDE_MDSC]
            << ',' << cnt[N_PHI_PAIR + N_PHI_KDE + PHI_KDE_MDSC]
            << ',' << g_phi[PHI_BCELL_CD8]<< ',' << (g_phi_valid[PHI_BCELL_CD8] ? 1 : 0)
            << ',' << cnt[N_PHI_PAIR + PHI_KDE_BCELL]
            << ',' << cnt[N_PHI_PAIR + N_PHI_KDE + PHI_KDE_BCELL]
            << ',' << g_phi[PHI_MYCAF_C1] << ',' << (g_phi_valid[PHI_MYCAF_C1] ? 1 : 0)
            << ',' << cnt[N_PHI_PAIR + PHI_KDE_FIB_MYCAF]
            << ',' << cnt[N_PHI_PAIR + N_PHI_KDE + PHI_KDE_FIB_MYCAF] << '\n';
        g_phi_csv.flush();
    }
}

FLAMEGPU_HOST_FUNCTION(zero_fib_density_field) {
    nvtxRangePush("Zero Fib Density");
    if (d_fib_density_field && g_pde_solver) {
        int total_voxels = g_pde_solver->get_total_voxels();
        cudaMemset(d_fib_density_field, 0, total_voxels * sizeof(unsigned long long));
    }
    cuda_layer_check("zero_fib_density_field");
    nvtxRangePop();
}

// ============================================================================
// Vvas as a STATE WITH MEMORY (coupling contract §6.3), + Krogh O2 (media-2 Eq.1-3).
//
//   ∂V_vas/∂t = k_growth·Hill(VEGFA)·V_vas·(1 − V_vas/V_max) [angiogenesis]
//             − k_decay·max(0, V_vas − V0)                   [immature neovessel regression]
//             − occlusion(ECM, cancer)·V_vas                 [desmoplastic compression]
//             + recovery_rate·max(0, V0 − V_vas)             [rebound toward SA floor]
//
//   V0 = SA-placed initial field (d_vvas_initial). The initial floor is the stable
//   equilibrium: decay only acts on VEGF-driven excess above V0 (immature neovessels),
//   recovery drives compressed voxels back toward V0. Remove all signals → V → V0.
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
// All rate parameters are ALREADY per-step, so nothing here is multiplied by dt again.
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
    const float* __restrict__ vvas_init,                // SA-placed floor / recovery target
    const float* __restrict__ vegf,
    const unsigned int* __restrict__ cancer_occ,
    const float* __restrict__ ecm_density,
    const float* __restrict__ debris_field,
    int nx, int ny, int nz,
    float vegf_ec50, float cabo_term,
    float k_growth, float k_decay, float v_max,
    float collapse_th, float collapse_ec50,
    float recovery_th, float recovery_rate,
    float maturity_factor, float cancer_w,
    float vegf_survival_ec50, float decay_unsupported,
    float debris_block_th)
{
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ty = blockIdx.y * blockDim.y + threadIdx.y;
    const int tz = blockIdx.z * blockDim.z + threadIdx.z;
    if (tx >= nx || ty >= ny || tz >= nz) return;
    const int idx = tz * (nx * ny) + ty * nx + tx;

    float V = vvas[idx];
    const float V0 = vvas_init[idx];  // placement floor for this voxel

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
    const float logistic = V * fmaxf(0.0f, 1.0f - V / (v_max + 1e-30f));
    const float growth = k_growth * hill * cabo_term * logistic;

    // --- regression of immature neovessels above the initial floor ---
    // Only VEGF-driven excess regresses; the pre-existing mature floor is stable.
    // VEGF-A trophic support: immature neovessels require ongoing VEGF signaling.
    // When VEGF-A drops below survival EC50, regression accelerates — vessels that
    // grew due to VEGF-A stimulation regress when the stimulus is removed.
    const float vegf_survival = v / (v + vegf_survival_ec50 + 1e-30f);
    const float effective_decay = k_decay * (1.0f + decay_unsupported * (1.0f - vegf_survival));
    const float decay = effective_decay * fmaxf(0.0f, V - V0);

    // --- occlusion: ECM and cancer packing as INDEPENDENT pathways ---
    // ECM compression: slow remodeling resisted by mature pericyte coverage.
    const float ecm = ecm_density[idx];
    const float c_ecm = (ecm > collapse_th)
                      ? (ecm / (ecm + collapse_ec50 + 1e-30f)) * maturity_factor
                      : 0.0f;
    // Cancer compression: mechanical stress — no maturity resistance.
    // Dense tumor mass physically crushes vessels regardless of pericyte coverage.
    const float c_can = fminf(1.0f, cancer_w * f_cancer);
    const float occl = 1.0f - (1.0f - c_ecm) * (1.0f - c_can);
    const float compress = occl * V;

    // --- recovery toward initial floor once compression source relaxes ---
    // Debris blocks recovery: dead cell remnants prevent vessel regrowth
    const float debris_val = debris_field ? debris_field[idx] : 0.0f;
    const float deficit = fmaxf(0.0f, V0 - V);
    const float recover = (ecm < recovery_th && debris_val < debris_block_th)
                        ? recovery_rate * deficit : 0.0f;

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
    cuda_layer_check("decode_pde_sources");
}

// Kernel: add ECM-density-proportional uptake to TGF-β (and VEGF-A) float uptake buffers.
// ECM sequesters TGF-β via LAP/LTBP binding and VEGF-A via HSPG binding. The effective
// additional uptake rate at each voxel is ecm_density[v] * sequester_rate [1/s].
// Runs AFTER decode_pde_sources (FP→float), BEFORE solve_pde.
__global__ void ecm_sequester_kernel(
    const float* __restrict__ ecm_density,
    float* __restrict__ tgfb_upt,
    float tgfb_ecm_rate,
    int total_voxels)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_voxels) return;
    const float rho = ecm_density[idx];
    if (rho <= 0.0f) return;
    atomicAdd(&tgfb_upt[idx], rho * tgfb_ecm_rate);
}

FLAMEGPU_HOST_FUNCTION(apply_ecm_sequestration) {
    if (!g_pde_solver || !d_ecm_density) return;
    const int nx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int nz = FLAMEGPU->environment.getProperty<int>("grid_size_z");
    const int V = nx * ny * nz;
    const float tgfb_ecm_rate = FLAMEGPU->environment.getProperty<float>("PARAM_TGFB_ECM_SEQUESTER_RATE");
    if (tgfb_ecm_rate <= 0.0f) return;
    float* tgfb_upt = g_pde_solver->get_device_uptake_ptr(CHEM_TGFB);
    const int threads = 256;
    const int blocks = (V + threads - 1) / threads;
    ecm_sequester_kernel<<<blocks, threads>>>(d_ecm_density, tgfb_upt, tgfb_ecm_rate, V);
    cuda_layer_check("apply_ecm_sequestration");
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
    // Mature-vessel resistance: applies to ECM-driven compression only.
    // Cancer mechanical compression bypasses this (no pericyte resistance to
    // solid stress from tumor mass).
    const float mat_resist = FLAMEGPU->environment.getProperty<float>("PARAM_VAS_MATURITY_RESISTANCE");
    const float init_mat   = FLAMEGPU->environment.getProperty<float>("PARAM_VAS_INITIAL_MATURITY");
    const float maturity_factor = 1.0f / (1.0f + mat_resist * init_mat);
    const float vegf_survival_ec50 = FLAMEGPU->environment.getProperty<float>("PARAM_VAS_VEGFA_SURVIVAL_EC50");
    const float decay_unsupported  = FLAMEGPU->environment.getProperty<float>("PARAM_VAS_DECAY_UNSUPPORTED");

    // One-shot uniform seed — the FALLBACK only. When a placement supplied V_vas(t=0)
    // (set_vvas_from_host), g_vvas_seeded is already true and this is skipped, which is what
    // preserves the placed avascular zero set: the logistic growth law keeps V=0 fixed, but a
    // uniform overwrite here would destroy the zero set before step 1.
    if (!g_vvas_seeded) {
        const float v0 = FLAMEGPU->environment.getProperty<float>("PARAM_VAS_INIT");
        const int total = nx * ny * nz;
        fill_vvas_kernel<<<(total + 255) / 256, 256>>>(d_vvas_field, total, v0);
        fill_vvas_kernel<<<(total + 255) / 256, 256>>>(d_vvas_initial, total, v0);
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
    const float debris_block_th = FLAMEGPU->environment.getProperty<float>("PARAM_DEBRIS_BLOCK_THRESHOLD");
    update_vvas_kernel<<<grid, block>>>(d_vvas_field, d_vvas_initial, vegf, d_cancer_occ, d_ecm_density,
                                        d_debris_field,
                                        nx, ny, nz, vegf_ec50, cabo_term,
                                        k_growth, k_decay, v_max,
                                        collapse_th, collapse_ec50,
                                        recovery_th, recovery_rate,
                                        maturity_factor, cancer_w,
                                        vegf_survival_ec50, decay_unsupported,
                                        debris_block_th);
    source_o2_from_vvas_kernel<<<grid, block>>>(d_vvas_field, o2conc, o2_src, o2_upt, nx, ny, nz,
                                                KvLv_base, C_blood, voxel_volume);
    cudaDeviceSynchronize();
    cuda_layer_check("compute_vvas_and_o2");
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
    cuda_layer_check("update_ecm_grid");

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
    cuda_layer_check("decay_antigen_grid");
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
    cuda_layer_check("decay_stress_field");
    nvtxRangePop();
}

// ============================================================================
// Proliferative Pressure: decay + gradient computation.
// Pressure deposited by cancer_divide_confirm (Gaussian kernel at daughter voxel).
// Decay runs at top of step (with other field decays); gradient computed after division
// so this step's divisions are included for next step's movement.
// ============================================================================
FLAMEGPU_HOST_FUNCTION(decay_pressure_field) {
    nvtxRangePush("Decay Pressure Field");
    if (d_pressure_field && g_pde_solver) {
        int total_voxels = g_pde_solver->get_total_voxels();
        float decay_rate = FLAMEGPU->environment.getProperty<float>("PARAM_PRESSURE_DECAY");
        float dt = FLAMEGPU->environment.getProperty<float>("PARAM_SEC_PER_SLICE");
        float decay_factor = expf(-decay_rate * dt);

        int block_size = 256;
        int grid_size = (total_voxels + block_size - 1) / block_size;
        decay_pressure_kernel<<<grid_size, block_size>>>(
            d_pressure_field, total_voxels, decay_factor);
        cudaDeviceSynchronize();
    }
    cuda_layer_check("decay_pressure_field");
    nvtxRangePop();
}

FLAMEGPU_HOST_FUNCTION(decay_debris_field) {
    nvtxRangePush("Decay Debris Field");
    if (d_debris_field && g_pde_solver) {
        int total_voxels = g_pde_solver->get_total_voxels();
        float decay_rate = FLAMEGPU->environment.getProperty<float>("PARAM_DEBRIS_CLEARANCE_RATE");
        float dt = FLAMEGPU->environment.getProperty<float>("PARAM_SEC_PER_SLICE");
        float decay_factor = expf(-decay_rate * dt);

        int block_size = 256;
        int grid_size = (total_voxels + block_size - 1) / block_size;
        decay_debris_kernel<<<grid_size, block_size>>>(
            d_debris_field, total_voxels, decay_factor);
        cudaDeviceSynchronize();
    }
    cuda_layer_check("decay_debris_field");
    nvtxRangePop();
}

float* get_debris_device_ptr() { return d_debris_field; }

FLAMEGPU_HOST_FUNCTION(compute_pressure_gradients) {
    nvtxRangePush("Compute Pressure Gradients");
    if (d_pressure_field && d_pressure_grad_x && g_pde_solver) {
        const int nx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
        const int ny = FLAMEGPU->environment.getProperty<int>("grid_size_y");
        const int nz = FLAMEGPU->environment.getProperty<int>("grid_size_z");
        float voxel_size_cm = FLAMEGPU->environment.getProperty<int>("PARAM_VOXEL_SIZE") * 1e-4f;
        float inv_2dx = 0.5f / voxel_size_cm;

        dim3 block(8, 8, 8);
        dim3 grid((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);
        compute_pressure_gradient_kernel<<<grid, block>>>(
            d_pressure_field, d_pressure_grad_x, d_pressure_grad_y, d_pressure_grad_z,
            nx, ny, nz, inv_2dx);
        cudaDeviceSynchronize();
    }
    cuda_layer_check("compute_pressure_gradients");
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
    cuda_layer_check("update_ecm_orientation");
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

// ── Ripley diagnostic: fill per-subtype voxel count grids ────────────────────
void ripley_fill_counts(flamegpu::HostAPI* api, int gx, int gy, int gz) {
    const int nvox = gx * gy * gz;
    const int nxy  = gx * gy;
    for (int t = 0; t < N_RIPLEY_TYPE; ++t)
        g_ripley_counts[t].assign(nvox, 0);

    auto scan = [&](const char* agent_name, auto&& classify) {
        auto a = api->agent(agent_name);
        flamegpu::DeviceAgentVector pop = a.getPopulationData();
        for (unsigned int i = 0; i < pop.size(); ++i) {
            int cs = pop[i].getVariable<int>("cell_state");
            int rip = classify(cs);
            if (rip < 0) continue;
            int x = pop[i].getVariable<int>("x");
            int y = pop[i].getVariable<int>("y");
            int z = pop[i].getVariable<int>("z");
            int v = z * nxy + y * gx + x;
            if (v >= 0 && v < nvox)
                g_ripley_counts[rip][v]++;
        }
    };

    auto scan_flat = [&](const char* agent_name, int rip) {
        auto a = api->agent(agent_name);
        flamegpu::DeviceAgentVector pop = a.getPopulationData();
        for (unsigned int i = 0; i < pop.size(); ++i) {
            int x = pop[i].getVariable<int>("x");
            int y = pop[i].getVariable<int>("y");
            int z = pop[i].getVariable<int>("z");
            int v = z * nxy + y * gx + x;
            if (v >= 0 && v < nvox)
                g_ripley_counts[rip][v]++;
        }
    };

    scan_flat(AGENT_CANCER_CELL, (int)RIP_CANCER);
    scan_flat(AGENT_TCELL,       (int)RIP_CD8);
    scan(AGENT_TREG, [](int cs) -> int {
        if (cs == TCD4_TREG) return (int)RIP_TREG;
        if (cs == TCD4_TH)   return (int)RIP_TH;
        return -1;
    });
    scan(AGENT_MACROPHAGE, [](int cs) -> int {
        if (cs == MAC_M1) return (int)RIP_M1;
        if (cs == MAC_M2) return (int)RIP_M2;
        return -1;
    });
    scan(AGENT_FIBROBLAST, [](int cs) -> int {
        if (cs == FIB_ICAF)  return (int)RIP_ICAF;
        if (cs == FIB_MYCAF) return (int)RIP_MYCAF;
        if (cs == FIB_APCAF) return (int)RIP_APCAF;
        return -1;
    });
    scan_flat(AGENT_MDSC,  (int)RIP_MDSC);
    scan_flat(AGENT_BCELL, (int)RIP_BCELL);
    scan_flat(AGENT_DC,    (int)RIP_DC);
}

} // namespace PDAC