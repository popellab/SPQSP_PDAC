/**
 * PDE Solver: LOD diffusion+decay + Exact ODE Source/Uptake (cell terms only)
 *
 * Matches BioFVM LOD_3D exactly (36 substeps per ABM step, dt_pde = 600s):
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
 * Agent coupling: direct device pointer atomicAdds (no host loops).
 */

#include "pde_solver.cuh"
#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

namespace PDAC {

// ============================================================================
// CUDA Error Checking
// ============================================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t _err = (call); \
        if (_err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " — " << cudaGetErrorString(_err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// KERNEL: Apply sources and uptakes (exact ODE, cell terms only)
//
// dp/dt = S - U*p
//   if U > 1e-10: p_new = (p - S/U)*exp(-U*dt) + S/U
//   else:         p_new = p + S*dt
//
// S [conc/s] = secretion [mol/s] / voxel_volume  (agent fns divide before atomicAdd)
// U [1/s]   = cell uptake rate constant only (background decay λ handled in LOD Thomas)
// ============================================================================

__global__ void apply_sources_uptakes_kernel(
    float* __restrict__ C,         // [V] concentration for one substrate (in-place update)
    const float* __restrict__ S,   // [V] source [conc/s]
    const float* __restrict__ U,   // [V] cell uptake rate constant [1/s]
    float dt,
    int V)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= V) return;

    float p = C[idx];
    float s = S[idx];
    float u = U[idx];              // cell uptake only; background decay handled in LOD sweeps

    float p_new;
    if (u > 1e-10f) {
        float su = s / u;
        p_new = (p - su) * expf(-u * dt) + su;
    } else {
        p_new = p + s * dt;
    }

    C[idx] = fmaxf(0.0f, p_new);
}

// ============================================================================
// KERNELS: LOD Thomas solver per line
//
// Each thread handles one independent 1D tridiagonal system.
// Thomas algorithm (precomputed pivots denom[], back-sub factors c[]):
//
//   Forward pass:
//     d[0] /= denom[0]
//     for i = 1..N-1:
//       d[i] += c1 * d[i-1]   // where c1 = dt*D/dx^2 (positive)
//       d[i] /= denom[i]
//
//   Back-substitution:
//     for i = N-2..0:
//       d[i] -= c[i] * d[i+1] // c[i] = -c1/denom[i] (precomputed, negative)
//
// This in-place approach modifies the concentration array directly.
// ============================================================================

// x-pass: ny*nz independent systems of length nx (stride-1 access, optimal)
__global__ void lod_x_kernel(
    float* __restrict__ C,
    const float* __restrict__ denom,  // [nx], precomputed modified pivots
    const float* __restrict__ cx,     // [nx], precomputed back-sub factors
    float c1,                          // dt*D/dx^2
    int nx, int ny, int nz)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // y-index
    int k = blockIdx.y;                               // z-index
    if (j >= ny) return;

    float* line = C + k * ny*nx + j * nx;  // pointer to this x-line (stride-1)

    // Forward elimination
    line[0] /= denom[0];
    for (int i = 1; i < nx; i++) {
        line[i] += c1 * line[i-1];
        line[i] /= denom[i];
    }

    // Back-substitution
    for (int i = nx-2; i >= 0; i--) {
        line[i] -= cx[i] * line[i+1];
    }
}

// y-pass: nx*nz independent systems of length ny (stride-nx access)
__global__ void lod_y_kernel(
    float* __restrict__ C,
    const float* __restrict__ denom,  // [ny]
    const float* __restrict__ cy,     // [ny]
    float c1,
    int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // x-index
    int k = blockIdx.y;                               // z-index
    if (i >= nx) return;

    // Base index for this y-line; consecutive elements are stride-nx apart
    int base = k * ny*nx + i;

    // Forward elimination
    C[base] /= denom[0];
    for (int j = 1; j < ny; j++) {
        C[base + j*nx] += c1 * C[base + (j-1)*nx];
        C[base + j*nx] /= denom[j];
    }

    // Back-substitution
    for (int j = ny-2; j >= 0; j--) {
        C[base + j*nx] -= cy[j] * C[base + (j+1)*nx];
    }
}

// z-pass: nx*ny independent systems of length nz (stride-nx*ny access)
__global__ void lod_z_kernel(
    float* __restrict__ C,
    const float* __restrict__ denom,  // [nz]
    const float* __restrict__ cz,     // [nz]
    float c1,
    int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // x-index
    int j = blockIdx.y;                               // y-index
    if (i >= nx) return;

    int stride = ny * nx;
    int base   = j * nx + i;  // base for k=0; consecutive elements are stride-ny*nx apart

    // Forward elimination
    C[base] /= denom[0];
    for (int k = 1; k < nz; k++) {
        C[base + k*stride] += c1 * C[base + (k-1)*stride];
        C[base + k*stride] /= denom[k];
    }

    // Back-substitution
    for (int k = nz-2; k >= 0; k--) {
        C[base + k*stride] -= cz[k] * C[base + (k+1)*stride];
    }
}

// ============================================================================
// KERNEL: Compute gradients via central differences
//
// grad_x[i,j,k] = (C[i+1,j,k] - C[i-1,j,k]) / (2*dx)  (forward/backward at boundaries)
//
// Gradient array layout: d_grad_[(g * 3 + dim) * V + voxel_idx]
//   g   ∈ {GRAD_IFN=0, GRAD_TGFB=1, GRAD_CCL2=2, GRAD_VEGFA=3}
//   dim ∈ {0=x, 1=y, 2=z}
// ============================================================================

__global__ void compute_gradients_kernel(
    float* __restrict__ d_grad,    // [NUM_GRAD_SUBSTRATES * 3 * V]
    const float* __restrict__ d_conc, // [NUM_SUBSTRATES * V]
    int nx, int ny, int nz,
    float inv2dx,                  // 1.0 / (2*dx)
    int V)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz) return;

    int voxel = iz * ny*nx + iy * nx + ix;

    // Gradient substrates: {CHEM_IFN=1, CHEM_TGFB=4, CHEM_CCL2=5, CHEM_VEGFA=9}
    const int grad_chems[4] = {1, 4, 5, 9};

    for (int g = 0; g < 4; g++) {
        const float* C = d_conc + grad_chems[g] * V;
        float* Gx = d_grad + (g*3 + 0) * V;
        float* Gy = d_grad + (g*3 + 1) * V;
        float* Gz = d_grad + (g*3 + 2) * V;

        float cp, cm;

        // x-gradient
        cp = (ix < nx-1) ? C[voxel + 1]    : C[voxel];
        cm = (ix > 0)    ? C[voxel - 1]    : C[voxel];
        Gx[voxel] = (cp - cm) * inv2dx;

        // y-gradient
        cp = (iy < ny-1) ? C[voxel + nx]   : C[voxel];
        cm = (iy > 0)    ? C[voxel - nx]   : C[voxel];
        Gy[voxel] = (cp - cm) * inv2dx;

        // z-gradient
        cp = (iz < nz-1) ? C[voxel + ny*nx] : C[voxel];
        cm = (iz > 0)    ? C[voxel - ny*nx] : C[voxel];
        Gz[voxel] = (cp - cm) * inv2dx;
    }
}

// ============================================================================
// PDESolver: Constructor / Destructor
// ============================================================================

PDESolver::PDESolver(const PDEConfig& config)
    : config_(config),
      d_conc_(nullptr), d_src_(nullptr), d_upt_(nullptr),
      d_grad_(nullptr), d_recruitment_(nullptr),
      d_thomas_denom_x_(nullptr), d_thomas_c_x_(nullptr),
      d_thomas_denom_y_(nullptr), d_thomas_c_y_(nullptr),
      d_thomas_denom_z_(nullptr), d_thomas_c_z_(nullptr)
{
    for (int s = 0; s < NUM_SUBSTRATES; s++) {
        h_c1_[s] = 0.0f;
    }
}

PDESolver::~PDESolver() {
    if (d_conc_)          CUDA_CHECK(cudaFree(d_conc_));
    if (d_src_)           CUDA_CHECK(cudaFree(d_src_));
    if (d_upt_)           CUDA_CHECK(cudaFree(d_upt_));
    if (d_grad_)          CUDA_CHECK(cudaFree(d_grad_));
    if (d_recruitment_)   CUDA_CHECK(cudaFree(d_recruitment_));
    if (d_thomas_denom_x_) CUDA_CHECK(cudaFree(d_thomas_denom_x_));
    if (d_thomas_c_x_)    CUDA_CHECK(cudaFree(d_thomas_c_x_));
    if (d_thomas_denom_y_) CUDA_CHECK(cudaFree(d_thomas_denom_y_));
    if (d_thomas_c_y_)    CUDA_CHECK(cudaFree(d_thomas_c_y_));
    if (d_thomas_denom_z_) CUDA_CHECK(cudaFree(d_thomas_denom_z_));
    if (d_thomas_c_z_)    CUDA_CHECK(cudaFree(d_thomas_c_z_));
}

// ============================================================================
// initialize(): allocate arrays and precompute Thomas coefficients
// ============================================================================

void PDESolver::initialize() {
    int V = get_total_voxels();
    size_t sz_sv = (size_t)NUM_SUBSTRATES * V * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_conc_,        sz_sv));  CUDA_CHECK(cudaMemset(d_conc_, 0, sz_sv));
    CUDA_CHECK(cudaMalloc(&d_src_,         sz_sv));  CUDA_CHECK(cudaMemset(d_src_,  0, sz_sv));
    CUDA_CHECK(cudaMalloc(&d_upt_,         sz_sv));  CUDA_CHECK(cudaMemset(d_upt_,  0, sz_sv));

    size_t sz_grad = (size_t)NUM_GRAD_SUBSTRATES * 3 * V * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_grad_,        sz_grad)); CUDA_CHECK(cudaMemset(d_grad_, 0, sz_grad));

    CUDA_CHECK(cudaMalloc(&d_recruitment_, (size_t)V * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_recruitment_, 0, (size_t)V * sizeof(int)));

    // Thomas coefficient arrays
    CUDA_CHECK(cudaMalloc(&d_thomas_denom_x_, NUM_SUBSTRATES * config_.nx * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_thomas_c_x_,     NUM_SUBSTRATES * config_.nx * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_thomas_denom_y_, NUM_SUBSTRATES * config_.ny * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_thomas_c_y_,     NUM_SUBSTRATES * config_.ny * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_thomas_denom_z_, NUM_SUBSTRATES * config_.nz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_thomas_c_z_,     NUM_SUBSTRATES * config_.nz * sizeof(float)));

    precompute_thomas_coefficients();

    std::cout << "[PDESolver v3] Initialized: " << config_.nx << "×" << config_.ny
              << "×" << config_.nz << " grid, " << NUM_SUBSTRATES
              << " substrates, LOD Thomas solver" << std::endl;
}

// ============================================================================
// precompute_thomas_coefficients()
//
// For each substrate s and each direction (with N grid points):
//   c1 = dt * D[s] / dx^2
//   c2 = dt * λ[s] / 3            (decay split equally over 3 LOD sweeps)
//   b_interior = 1 + 2*c1 + c2
//   b_boundary = 1 + c1 + c2      (Neumann: one fewer neighbor at each end)
//
// Modified pivot (Thomas forward elimination precomputed):
//   w[0]   = b_boundary
//   w[i]   = b_interior  - c1^2 / w[i-1]  (i = 1..N-2)
//   w[N-1] = b_boundary  - c1^2 / w[N-2]
//
// Back-sub factor (also precomputed):
//   c[i] = -c1 / w[i]  (i = 0..N-2; c[N-1] unused)
// ============================================================================

void PDESolver::precompute_thomas_coefficients() {
    const int Ns[3] = {config_.nx, config_.ny, config_.nz};
    float* h_denom[3];
    float* h_c[3];
    float* d_denom[3] = {d_thomas_denom_x_, d_thomas_denom_y_, d_thomas_denom_z_};
    float* d_c[3]     = {d_thomas_c_x_,     d_thomas_c_y_,     d_thomas_c_z_};

    for (int dim = 0; dim < 3; dim++) {
        h_denom[dim] = new float[NUM_SUBSTRATES * Ns[dim]];
        h_c[dim]     = new float[NUM_SUBSTRATES * Ns[dim]];
    }

    for (int s = 0; s < NUM_SUBSTRATES; s++) {
        float D      = config_.diffusion_coeffs[s];
        float dt     = config_.dt_pde;
        float dx     = config_.voxel_size;

        float c1 = dt * D / (dx * dx);
        float c2 = dt * config_.decay_rates[s] / 3.0f;  // decay split over 3 LOD sweeps
        h_c1_[s] = c1;

        float b_interior = 1.0f + 2.0f*c1 + c2;
        float b_boundary = 1.0f + c1       + c2;

        for (int dim = 0; dim < 3; dim++) {
            int N      = Ns[dim];
            float* wd  = h_denom[dim] + s * N;
            float* wc  = h_c[dim]     + s * N;

            if (N == 1) {
                // Single point: no off-diagonal, just decay
                wd[0] = 1.0f + c2;
                wc[0] = 0.0f;
                continue;
            }

            // Boundary at i=0
            wd[0] = b_boundary;
            wc[0] = (c1 > 0.0f) ? (-c1 / wd[0]) : 0.0f;

            // Interior points i = 1..N-2
            for (int i = 1; i < N-1; i++) {
                wd[i] = b_interior - (c1 * c1) / wd[i-1];
                wc[i] = (c1 > 0.0f) ? (-c1 / wd[i]) : 0.0f;
            }

            // Boundary at i=N-1
            wd[N-1] = b_boundary - (c1 * c1) / wd[N-2];
            wc[N-1] = 0.0f;  // not used in back-sub (last element)
        }
    }

    // Upload to device
    for (int dim = 0; dim < 3; dim++) {
        size_t sz = NUM_SUBSTRATES * Ns[dim] * sizeof(float);
        CUDA_CHECK(cudaMemcpy(d_denom[dim], h_denom[dim], sz, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_c[dim],     h_c[dim],     sz, cudaMemcpyHostToDevice));
        delete[] h_denom[dim];
        delete[] h_c[dim];
    }
}

// ============================================================================
// solve_timestep()
//
// For each substrate:
//   1. apply_sources_uptakes_kernel (exact ODE, cell source/uptake only)
//   2. lod_x_kernel (Thomas sweep in x, decay λ/3 included in coefficients)
//   3. lod_y_kernel (Thomas sweep in y, decay λ/3)
//   4. lod_z_kernel (Thomas sweep in z, decay λ/3)
// ============================================================================

void PDESolver::solve_timestep() {
    int V  = get_total_voxels();
    int nx = config_.nx;
    int ny = config_.ny;
    int nz = config_.nz;
    float dt = config_.dt_pde;

    const int threads = 256;

    for (int s = 0; s < NUM_SUBSTRATES; s++) {
        float* C  = d_conc_ + (size_t)s * V;
        float* S  = d_src_  + (size_t)s * V;
        float* U  = d_upt_  + (size_t)s * V;
        float  c1 = h_c1_[s];

        // --- Step 1: exact ODE for cell source/uptake (background decay in LOD) ---
        {
            int blocks = (V + threads - 1) / threads;
            apply_sources_uptakes_kernel<<<blocks, threads>>>(C, S, U, dt, V);
        }

        // Skip LOD entirely if D = 0 — diffusion-free substrate
        if (config_.diffusion_coeffs[s] == 0.0f) {
            continue;
        }

        // --- Step 2: LOD x-pass ---
        // Each thread handles one (j, k) pair → one x-line of length nx
        {
            const int bx = 32;
            dim3 grid((ny + bx-1)/bx, nz);
            lod_x_kernel<<<grid, bx>>>(C,
                d_thomas_denom_x_ + s * nx,
                d_thomas_c_x_     + s * nx,
                c1, nx, ny, nz);
        }

        // --- Step 3: LOD y-pass ---
        // Each thread handles one (i, k) pair → one y-line of length ny
        {
            const int bx = 32;
            dim3 grid((nx + bx-1)/bx, nz);
            lod_y_kernel<<<grid, bx>>>(C,
                d_thomas_denom_y_ + s * ny,
                d_thomas_c_y_     + s * ny,
                c1, nx, ny, nz);
        }

        // --- Step 4: LOD z-pass ---
        // Each thread handles one (i, j) pair → one z-line of length nz
        {
            const int bx = 32;
            dim3 grid((nx + bx-1)/bx, ny);
            lod_z_kernel<<<grid, bx>>>(C,
                d_thomas_denom_z_ + s * nz,
                d_thomas_c_z_     + s * nz,
                c1, nx, ny, nz);
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// compute_gradients(): central differences for 4 chemotaxis substrates
// ============================================================================

void PDESolver::compute_gradients() {
    int nx = config_.nx;
    int ny = config_.ny;
    int nz = config_.nz;
    int V  = get_total_voxels();
    float inv2dx = 1.0f / (2.0f * config_.voxel_size);

    dim3 block(16, 16, 1);
    dim3 grid((nx + 15)/16, (ny + 15)/16, nz);
    compute_gradients_kernel<<<grid, block>>>(d_grad_, d_conc_, nx, ny, nz, inv2dx, V);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Reset helpers
// ============================================================================

void PDESolver::reset_sources() {
    size_t sz = (size_t)NUM_SUBSTRATES * get_total_voxels() * sizeof(float);
    CUDA_CHECK(cudaMemset(d_src_, 0, sz));
}

void PDESolver::reset_uptakes() {
    size_t sz = (size_t)NUM_SUBSTRATES * get_total_voxels() * sizeof(float);
    CUDA_CHECK(cudaMemset(d_upt_, 0, sz));
}

void PDESolver::reset_recruitment_sources() {
    size_t sz = (size_t)get_total_voxels() * sizeof(int);
    CUDA_CHECK(cudaMemset(d_recruitment_, 0, sz));
}

void PDESolver::reset_concentrations() {
    size_t sz = (size_t)NUM_SUBSTRATES * get_total_voxels() * sizeof(float);
    CUDA_CHECK(cudaMemset(d_conc_, 0, sz));
}

// ============================================================================
// Device pointer accessors
// ============================================================================

float* PDESolver::get_device_concentration_ptr(int s) {
    return d_conc_ + (size_t)s * get_total_voxels();
}

float* PDESolver::get_device_source_ptr(int s) {
    return d_src_ + (size_t)s * get_total_voxels();
}

float* PDESolver::get_device_uptake_ptr(int s) {
    return d_upt_ + (size_t)s * get_total_voxels();
}

float* PDESolver::get_device_gradx_ptr(int g) {
    return d_grad_ + (size_t)(g*3 + 0) * get_total_voxels();
}

float* PDESolver::get_device_grady_ptr(int g) {
    return d_grad_ + (size_t)(g*3 + 1) * get_total_voxels();
}

float* PDESolver::get_device_gradz_ptr(int g) {
    return d_grad_ + (size_t)(g*3 + 2) * get_total_voxels();
}

int* PDESolver::get_device_recruitment_sources_ptr() {
    return d_recruitment_;
}

// ============================================================================
// Host-accessible helpers (D2H copies — use only for output/debugging)
// ============================================================================

void PDESolver::get_concentrations(float* h_buf, int substrate_idx) const {
    int V = get_total_voxels();
    CUDA_CHECK(cudaMemcpy(h_buf,
        d_conc_ + (size_t)substrate_idx * V,
        (size_t)V * sizeof(float),
        cudaMemcpyDeviceToHost));
}

void PDESolver::get_all_concentrations(float* h_buf) const {
    size_t total_bytes = (size_t)NUM_SUBSTRATES * get_total_voxels() * sizeof(float);
    CUDA_CHECK(cudaMemcpy(h_buf, d_conc_, total_bytes, cudaMemcpyDeviceToHost));
}

void PDESolver::get_all_concentrations_async(float* h_buf, cudaStream_t stream) const {
    size_t total_bytes = (size_t)NUM_SUBSTRATES * get_total_voxels() * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(h_buf, d_conc_, total_bytes, cudaMemcpyDeviceToHost, stream));
}

float PDESolver::get_concentration_at_voxel(int x, int y, int z, int substrate_idx) const {
    if (x < 0 || x >= config_.nx || y < 0 || y >= config_.ny || z < 0 || z >= config_.nz) return 0.0f;
    int V = get_total_voxels();
    int voxel = z * config_.ny*config_.nx + y * config_.nx + x;
    float val;
    CUDA_CHECK(cudaMemcpy(&val,
        d_conc_ + (size_t)substrate_idx * V + voxel,
        sizeof(float), cudaMemcpyDeviceToHost));
    return val;
}

void PDESolver::set_initial_concentration(int substrate_idx, float value) {
    int V = get_total_voxels();
    std::vector<float> h_data(V, value);
    CUDA_CHECK(cudaMemcpy(d_conc_ + (size_t)substrate_idx * V,
        h_data.data(), (size_t)V * sizeof(float), cudaMemcpyHostToDevice));
}

float PDESolver::get_total_source(int substrate_idx) {
    int V = get_total_voxels();
    std::vector<float> h_src(V);
    CUDA_CHECK(cudaMemcpy(h_src.data(),
        d_src_ + (size_t)substrate_idx * V,
        (size_t)V * sizeof(float), cudaMemcpyDeviceToHost));
    float total = 0.0f;
    for (int i = 0; i < V; i++) total += h_src[i];
    return total;
}

} // namespace PDAC
