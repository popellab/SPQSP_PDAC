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
#include <cufft.h>

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

#define CUFFT_CHECK(call) \
    do { \
        cufftResult _err = (call); \
        if (_err != CUFFT_SUCCESS) { \
            std::cerr << "cuFFT error at " << __FILE__ << ":" << __LINE__ \
                      << " — code " << static_cast<int>(_err) << std::endl; \
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

    // Gradient substrates: {IFN=1, TGFB=4, CCL2=5, VEGFA=9, CXCL13=12, CCL21=15, CXCL12=16, CCL5=17}
    const int grad_chems[NUM_GRAD_SUBSTRATES] = {1, 4, 5, 9, 12, 15, 16, 17};

    for (int g = 0; g < NUM_GRAD_SUBSTRATES; g++) {
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
// KERNELS: Steady-state screened-Poisson solve (matrix-free Jacobi-PCG)
//
// Per substrate, solve  M c = b  with
//   M = (λ + U)·I − D·∇²      (7-point Neumann Laplacian)
//   b = S                      (agent source field [conc/s])
// M is SPD for λ>0, U,D≥0 (and SPSD/diagonally-dominant elsewhere), so PCG
// with a per-voxel Jacobi (diagonal) preconditioner converges robustly.
// This is the dt→∞ fixed point of the transient operator, solved as a single
// coupled system (NOT achievable via big-dt LOD, which retains splitting error).
// ============================================================================

// A·x for the steady operator: Ax = (λ + U[i])·x[i] − D·∇²x  (Neumann no-flux)
__global__ void steady_operator_kernel(
    const float* __restrict__ x,
    float* __restrict__ Ax,
    const float* __restrict__ U,   // per-voxel uptake [1/s]
    int nx, int ny, int nz,
    float D, float lambda, float inv_dx2)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz) return;

    int idx = iz * (nx * ny) + iy * nx + ix;
    float xc = x[idx];

    // −∇²x via 7-point stencil (no-flux: missing neighbor contributes 0)
    float lap = 0.0f;
    if (ix > 0)      lap += x[idx - 1]        - xc;
    if (ix < nx - 1) lap += x[idx + 1]        - xc;
    if (iy > 0)      lap += x[idx - nx]       - xc;
    if (iy < ny - 1) lap += x[idx + nx]       - xc;
    if (iz > 0)      lap += x[idx - nx * ny]  - xc;
    if (iz < nz - 1) lap += x[idx + nx * ny]  - xc;
    lap *= inv_dx2;

    Ax[idx] = (lambda + U[idx]) * xc - D * lap;
}

// Per-voxel Jacobi preconditioner M^{-1}: diagonal of M = (λ+U) + D·(neighbor count)/dx²
__global__ void steady_precond_kernel(
    float* __restrict__ Minv,
    const float* __restrict__ U,
    int nx, int ny, int nz,
    float D, float lambda, float inv_dx2)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz) return;

    int idx = iz * (nx * ny) + iy * nx + ix;
    int neighbors = (ix > 0) + (ix < nx - 1) + (iy > 0) + (iy < ny - 1)
                  + (iz > 0) + (iz < nz - 1);
    float diag = (lambda + U[idx]) + D * neighbors * inv_dx2;
    if (!isfinite(diag) || diag <= 0.0f) diag = 1.0f;
    Minv[idx] = 1.0f / (diag + 1e-30f);
}

// D=0 closed form: c = S / (λ + U)   (no diffusion → algebraic per voxel)
__global__ void steady_diffusionless_kernel(
    float* __restrict__ C,
    const float* __restrict__ S,
    const float* __restrict__ U,
    float lambda, int V)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= V) return;
    float denom = lambda + U[i];
    C[i] = (denom > 1e-30f) ? fmaxf(0.0f, S[i] / denom) : 0.0f;
}

// z = Minv ⊙ r
__global__ void cg_apply_precond(const float* __restrict__ Minv,
                                 const float* __restrict__ r,
                                 float* __restrict__ z, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) z[i] = Minv[i] * r[i];
}
// y += a·x
__global__ void cg_axpy(float* __restrict__ y, const float* __restrict__ x, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}
// p = z + b·p
__global__ void cg_xpby(float* __restrict__ p, const float* __restrict__ z, float b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = z[i] + b * p[i];
}
__global__ void cg_copy(float* __restrict__ dst, const float* __restrict__ src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}
__global__ void cg_clamp_nonneg(float* __restrict__ x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && x[i] < 0.0f) x[i] = 0.0f;
}
// Block-partial dot product (256 threads/block)
__global__ void cg_dot_kernel(const float* __restrict__ a, const float* __restrict__ b,
                              float* __restrict__ partial, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (idx < n) ? a[idx] * b[idx] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = sdata[0];
}

// Block-partial sum of a single array (for mean uptake).
__global__ void cg_sum_kernel(const float* __restrict__ x, float* __restrict__ partial, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (idx < n) ? x[idx] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = sdata[0];
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
    if (d_cg_r_)    CUDA_CHECK(cudaFree(d_cg_r_));
    if (d_cg_p_)    CUDA_CHECK(cudaFree(d_cg_p_));
    if (d_cg_Ap_)   CUDA_CHECK(cudaFree(d_cg_Ap_));
    if (d_cg_z_)    CUDA_CHECK(cudaFree(d_cg_z_));
    if (d_cg_tmp_)  CUDA_CHECK(cudaFree(d_cg_tmp_));
    if (d_cg_minv_) CUDA_CHECK(cudaFree(d_cg_minv_));
    if (d_dot_buf_) CUDA_CHECK(cudaFree(d_dot_buf_));
    if (fft_ready_) {
        cufftDestroy((cufftHandle)dct_plan_x_);
        cufftDestroy((cufftHandle)dct_plan_y_);
        cufftDestroy((cufftHandle)dct_plan_z_);
        cufftDestroy((cufftHandle)dct_plan_xb_);
        cufftDestroy((cufftHandle)dct_plan_yb_);
        cufftDestroy((cufftHandle)dct_plan_zb_);
        cufftDestroy((cufftHandle)dct_plan_x3_);
        cufftDestroy((cufftHandle)dct_plan_y3_);
        cufftDestroy((cufftHandle)dct_plan_z3_);
        if (d_dct_a_)  CUDA_CHECK(cudaFree(d_dct_a_));
        if (d_dct_b_)  CUDA_CHECK(cudaFree(d_dct_b_));
        if (d_dct_mu_) CUDA_CHECK(cudaFree(d_dct_mu_));
        if (d_decay_subs_) CUDA_CHECK(cudaFree(d_decay_subs_));
        if (d_sub_D_)      CUDA_CHECK(cudaFree(d_sub_D_));
        if (d_sub_lambda_) CUDA_CHECK(cudaFree(d_sub_lambda_));
        if (d_upt_subs_)     CUDA_CHECK(cudaFree(d_upt_subs_));
        if (d_upt_D_)        CUDA_CHECK(cudaFree(d_upt_D_));
        if (d_upt_lambda_)   CUDA_CHECK(cudaFree(d_upt_lambda_));
        if (d_upt_ubar_)     CUDA_CHECK(cudaFree(d_upt_ubar_));
        if (d_upt_lamshift_) CUDA_CHECK(cudaFree(d_upt_lamshift_));
    }
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

    // Steady-state CG workspace (only allocated when steady mode may be used;
    // cheap enough — 6 per-voxel float arrays — to always allocate)
    allocate_cg_workspace();

    // Spectral (FFT/DCT) workspace — large (8V complex), so only when mode 2 is selected.
    // mode 2's uptake path also reuses the CG buffers above.
    if (config_.solve_mode == 2) {
        allocate_fft_workspace();
    }

    std::cout << "[PDESolver v3] Initialized: " << config_.nx << "×" << config_.ny
              << "×" << config_.nz << " grid, " << NUM_SUBSTRATES
              << " substrates, "
              << (config_.solve_mode == 1 ? "STEADY-STATE (Jacobi-PCG)" : "transient LOD Thomas")
              << " solve mode" << std::endl;
}

// ============================================================================
// allocate_cg_workspace(): per-voxel scratch for the steady-state PCG
// ============================================================================
void PDESolver::allocate_cg_workspace() {
    int V = get_total_voxels();
    size_t vsz = (size_t)V * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_cg_r_,    vsz));
    CUDA_CHECK(cudaMalloc(&d_cg_p_,    vsz));
    CUDA_CHECK(cudaMalloc(&d_cg_Ap_,   vsz));
    CUDA_CHECK(cudaMalloc(&d_cg_z_,    vsz));
    CUDA_CHECK(cudaMalloc(&d_cg_tmp_,  vsz));
    CUDA_CHECK(cudaMalloc(&d_cg_minv_, vsz));
    cg_blocks_ = (V + 255) / 256;
    CUDA_CHECK(cudaMalloc(&d_dot_buf_, (size_t)cg_blocks_ * sizeof(float)));
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
// solve_steadystate(): direct quasi-steady-state field per substrate
// ============================================================================

float PDESolver::host_dot(const float* a, const float* b, int n) {
    cg_dot_kernel<<<cg_blocks_, 256>>>(a, b, d_dot_buf_, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> h(cg_blocks_);
    CUDA_CHECK(cudaMemcpy(h.data(), d_dot_buf_, (size_t)cg_blocks_ * sizeof(float),
                          cudaMemcpyDeviceToHost));
    double sum = 0.0;  // accumulate in double to limit reduction error
    for (int i = 0; i < cg_blocks_; i++) sum += h[i];
    return (float)sum;
}

// Solve M c = b in-place on d_C, warm-started from its current contents.
//   M = (λ + U)·I − D·∇²,  b = d_S
// Returns iteration count; stores final relative residual in last_cg_residual_ via caller.
int PDESolver::solve_steadystate_substrate(float* d_C, const float* d_S, const float* d_U,
                                           float D, float lambda) {
    const int n   = get_total_voxels();
    const int nx = config_.nx, ny = config_.ny, nz = config_.nz;
    const float inv_dx2 = 1.0f / (config_.voxel_size * config_.voxel_size);
    const int t1 = 256;
    const int b1 = (n + t1 - 1) / t1;
    dim3 blk(8, 8, 8);
    dim3 grd((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);

    // Jacobi preconditioner M^{-1} (per-voxel diagonal; rebuilt per substrate)
    steady_precond_kernel<<<grd, blk>>>(d_cg_minv_, d_U, nx, ny, nz, D, lambda, inv_dx2);

    // r = b − A·x0   (x0 = warm start = current d_C)
    steady_operator_kernel<<<grd, blk>>>(d_C, d_cg_Ap_, d_U, nx, ny, nz, D, lambda, inv_dx2);
    cg_copy<<<b1, t1>>>(d_cg_r_, d_S, n);
    cg_axpy<<<b1, t1>>>(d_cg_r_, d_cg_Ap_, -1.0f, n);
    // z = M^{-1} r ;  p = z
    cg_apply_precond<<<b1, t1>>>(d_cg_minv_, d_cg_r_, d_cg_z_, n);
    cg_copy<<<b1, t1>>>(d_cg_p_, d_cg_z_, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    float rz_old   = host_dot(d_cg_r_, d_cg_z_, n);
    float rr_init  = host_dot(d_cg_r_, d_cg_r_, n);
    if (rr_init <= 0.0f) { cg_scratch_residual_ = 0.0f; return 0; }  // already converged
    const float rr_target = config_.cg_tol * config_.cg_tol * rr_init;

    int iter = 0;
    float rr_new = rr_init;
    for (; iter < config_.cg_maxiter; iter++) {
        steady_operator_kernel<<<grd, blk>>>(d_cg_p_, d_cg_Ap_, d_U, nx, ny, nz, D, lambda, inv_dx2);
        CUDA_CHECK(cudaDeviceSynchronize());
        float pAp = host_dot(d_cg_p_, d_cg_Ap_, n);
        float alpha = rz_old / (pAp + 1e-30f);

        cg_axpy<<<b1, t1>>>(d_C,    d_cg_p_,  alpha, n);   // x += α p
        cg_axpy<<<b1, t1>>>(d_cg_r_, d_cg_Ap_, -alpha, n); // r −= α Ap
        CUDA_CHECK(cudaDeviceSynchronize());

        rr_new = host_dot(d_cg_r_, d_cg_r_, n);
        if (rr_new < rr_target) { iter++; break; }

        cg_apply_precond<<<b1, t1>>>(d_cg_minv_, d_cg_r_, d_cg_z_, n);  // z = M^{-1} r
        CUDA_CHECK(cudaDeviceSynchronize());
        float rz_new = host_dot(d_cg_r_, d_cg_z_, n);
        float beta = rz_new / (rz_old + 1e-30f);
        cg_xpby<<<b1, t1>>>(d_cg_p_, d_cg_z_, beta, n);     // p = z + β p
        rz_old = rz_new;
    }

    cg_clamp_nonneg<<<b1, t1>>>(d_C, n);   // physical concentrations ≥ 0
    CUDA_CHECK(cudaDeviceSynchronize());

    cg_scratch_residual_ = sqrtf(rr_new / (rr_init + 1e-30f));
    return iter;
}

void PDESolver::solve_steadystate() {
    int V = get_total_voxels();
    const int t1 = 256, b1 = (V + t1 - 1) / t1;

    for (int s = 0; s < NUM_SUBSTRATES; s++) {
        float* C = d_conc_ + (size_t)s * V;
        float* S = d_src_  + (size_t)s * V;
        float* U = d_upt_  + (size_t)s * V;
        float  D = config_.diffusion_coeffs[s];
        float  lambda = config_.decay_rates[s];

        if (D == 0.0f) {
            // No diffusion → algebraic steady state per voxel: c = S/(λ+U)
            steady_diffusionless_kernel<<<b1, t1>>>(C, S, U, lambda, V);
            CUDA_CHECK(cudaDeviceSynchronize());
            last_cg_iters_[s] = 0;
            last_cg_residual_[s] = 0.0f;
            continue;
        }

        int iters = solve_steadystate_substrate(C, S, U, D, lambda);
        last_cg_iters_[s] = iters;
        last_cg_residual_[s] = cg_scratch_residual_;
    }
}

int   PDESolver::get_last_cg_iters(int s)    const { return (s >= 0 && s < NUM_SUBSTRATES) ? last_cg_iters_[s] : 0; }
float PDESolver::get_last_cg_residual(int s) const { return (s >= 0 && s < NUM_SUBSTRATES) ? last_cg_residual_[s] : 0.0f; }

// ============================================================================
// Spectral steady-state solve via the TRUE 3D DCT-II/III (Makhoul), no mirror inflation.
//
// For (λ − D∇²) with Neumann BCs the DCT-II diagonalizes the operator:
//     ĉ_k = b̂_k / (λ + D·μ_k),   μ_k = (2/dx²) Σ_d (1 − cos(π k_d/n_d)),  k_d = 0..n_d−1.
// The DCT is applied separably (one 1D DCT per axis) on the native N³ grid. Each 1D
// DCT-II uses Makhoul's method: reorder → length-N complex FFT → twiddle. The formulas
// were validated to 1e-14 against a direct Thomas solve (/tmp/dct_validate.cpp), so this
// matches the CG operator (steady_operator_kernel) — CG is the oracle.
//
// Makhoul, 0-indexed length N:
//   reorder:  v[i] = x[(2i<N) ? 2i : 2N−1−2i]
//   fwd tw:   X[k] = 2·Re(e^{−iπk/2N}·V[k]) = 2(cosθ·Vr + sinθ·Vi),  θ=πk/2N
//   inv tw:   W[0]=½X[0];  W[k]=½·e^{+iπk/2N}·(X[k] − i·X[N−k])
//   inv IFFT then ×(1/N) (cuFFT inverse is unnormalized), then un-reorder.
// ============================================================================

__device__ __forceinline__ int dct_perm(int i, int n) { return (2*i < n) ? (2*i) : (2*n - 1 - 2*i); }

// μ_k in natural (z,y,x) layout; k_x=x, k_y=y, k_z=z, each in [0,n_d).
__global__ void dct_build_mu_kernel(float* __restrict__ mu, int nx, int ny, int nz, float inv_dx2) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int V = nx * ny * nz;
    if (t >= V) return;
    int x = t % nx, y = (t / nx) % ny, z = t / (nx * ny);
    const float PI = 3.14159265358979323846f;
    float m = (1.0f - cosf(PI * x / nx)) + (1.0f - cosf(PI * y / ny)) + (1.0f - cosf(PI * z / nz));
    mu[t] = 2.0f * inv_dx2 * m;
}

__global__ void dct_real2cplx_kernel(const float* __restrict__ b, float2* __restrict__ a, int V) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < V) a[t] = make_float2(b[t], 0.0f);
}

// Reorder along `axis` (Makhoul): out[t] = (in[perm-source].x, 0).  in/out complex [V].
// All DCT passes act on the CONTIGUOUS axis (length n; the array is n_lines lines of n).
// Cyclic rotation between passes keeps every axis contiguous in turn.

// Makhoul reorder on contiguous lines: out[line·n + i] = (in[line·n + perm(i,n)].x, 0).
__global__ void dct_reorder_kernel(const float2* __restrict__ in, float2* __restrict__ out, int n, int V) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= V) return;
    int i = t % n; size_t base = (size_t)(t / n) * n;
    out[t] = make_float2(in[base + dct_perm(i, n)].x, 0.0f);
}

// Forward twiddle: X[k] = 2(cosθ·Vr + sinθ·Vi), θ=πk/2n.  In place.
__global__ void dct_twiddle_fwd_kernel(float2* __restrict__ buf, int n, int V) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= V) return;
    int k = t % n;
    const float PI = 3.14159265358979323846f;
    float th = PI * k / (2.0f * n);
    float2 V2 = buf[t];
    buf[t] = make_float2(2.0f * (cosf(th) * V2.x + sinf(th) * V2.y), 0.0f);
}

// Inverse pretwiddle: build W from X[k], X[n-k] (reflected within the line).  out separate.
__global__ void dct_pretwiddle_inv_kernel(const float2* __restrict__ in, float2* __restrict__ out, int n, int V) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= V) return;
    int k = t % n; size_t base = (size_t)(t / n) * n;
    if (k == 0) { out[t] = make_float2(0.5f * in[t].x, 0.0f); return; }
    float Xk = in[t].x, Xnk = in[base + (n - k)].x;
    const float PI = 3.14159265358979323846f;
    float th = PI * k / (2.0f * n);
    float c = cosf(th), s = sinf(th);
    out[t] = make_float2(0.5f * (c * Xk + s * Xnk), 0.5f * (s * Xk - c * Xnk));
}

// Un-reorder + 1/n normalization: out[line·n + perm(i,n)] = (in[t].x / n, 0).
__global__ void dct_unreorder_kernel(const float2* __restrict__ in, float2* __restrict__ out, int n, int V) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= V) return;
    int i = t % n; size_t base = (size_t)(t / n) * n;
    out[base + dct_perm(i, n)] = make_float2(in[t].x / (float)n, 0.0f);
}

// Cyclic axis rotation per V-block: dims (na,nb,nc) [slow,mid,fast] -> (nc,na,nb).
// out[c,a,b] = in[a,b,c]. Batch-aware: each substrate's V=na*nb*nc block rotated independently.
__global__ void dct_rotate_kernel(const float2* __restrict__ in, float2* __restrict__ out,
    int na, int nb, int nc, int V_total) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= V_total) return;
    int V = na * nb * nc;
    size_t sub = t / V; int loc = (int)(t - sub * V);
    int c = loc % nc, b = (loc / nc) % nb, a = loc / (nb * nc);
    out[sub * V + (size_t)c * na * nb + (size_t)a * nb + b] = in[t];
}

// ---- Batched gather/divide/scatter for the decay-only substrates ----
// real2cplx gather: a[sub·V + loc] = (src[subs[sub]·V + loc], 0).
__global__ void dct_gather_kernel(const float* __restrict__ d_src, const int* __restrict__ subs,
    float2* __restrict__ a, int n_decay, int V) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_decay * V) return;
    int sub = t / V, loc = t - sub * V;
    a[t] = make_float2(d_src[(size_t)subs[sub] * V + loc], 0.0f);
}
// Divide each substrate block by (λ_sub + D_sub·μ_loc).  In place.
__global__ void dct_divide_mu_batched(float2* __restrict__ buf, const float* __restrict__ mu,
    const float* __restrict__ Dv, const float* __restrict__ lam, int n_decay, int V) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_decay * V) return;
    int sub = t / V, loc = t - sub * V;
    float denom = lam[sub] + Dv[sub] * mu[loc];
    buf[t].x = (denom > 1e-30f) ? (buf[t].x / denom) : 0.0f;
    buf[t].y = 0.0f;
}
// cplx2real scatter (clamped ≥0): conc[subs[sub]·V + loc] = max(0, a[t].x).
__global__ void dct_scatter_kernel(float* __restrict__ d_conc, const int* __restrict__ subs,
    const float2* __restrict__ a, int n_decay, int V) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_decay * V) return;
    int sub = t / V, loc = t - sub * V;
    d_conc[(size_t)subs[sub] * V + loc] = fmaxf(0.0f, a[t].x);
}

// Defect-correction RHS for the mean-shifted uptake split:
//   a[sub·V+loc] = S[s,loc] − (U[s,loc] − Ū_sub)·c[s,loc]   (complex; s = subs[sub])
// At the fixed point of c ← M0⁻¹·a (M0 = (λ+Ū) − D∇²), c solves (λ+U−D∇²)c = S exactly.
__global__ void dct_uptake_rhs_kernel(float2* __restrict__ a,
    const float* __restrict__ d_src, const float* __restrict__ d_upt, const float* __restrict__ d_conc,
    const int* __restrict__ subs, const float* __restrict__ ubar, int n3, int V) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n3 * V) return;
    int sub = t / V, loc = t - sub * V;
    size_t g = (size_t)subs[sub] * V + loc;
    float rhs = d_src[g] - (d_upt[g] - ubar[sub]) * d_conc[g];
    a[t] = make_float2(rhs, 0.0f);
}

// Divide DCT coefficients (real, in .x) by (λ + D·μ_k).  In place.
__global__ void dct_divide_mu_kernel(float2* __restrict__ buf, const float* __restrict__ mu,
    float D, float lambda, int V) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= V) return;
    float denom = lambda + D * mu[t];
    buf[t].x = (denom > 1e-30f) ? (buf[t].x / denom) : 0.0f;
    buf[t].y = 0.0f;
}

// Extract real part of the solution into C (no clamp — caller decides).
__global__ void dct_cplx2real_kernel(float* __restrict__ C, const float2* __restrict__ a, int V) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < V) C[t] = a[t].x;
}

void PDESolver::allocate_fft_workspace() {
    int V = get_total_voxels();
    int nx = config_.nx, ny = config_.ny, nz = config_.nz;

    // Decay-only substrates (no per-voxel uptake) — batched together through one DCT.
    std::vector<int> decay; std::vector<float> Dv, lam;
    for (int s = 0; s < NUM_SUBSTRATES; s++) {
        if (s == CHEM_O2 || s == CHEM_CCL2 || s == CHEM_VEGFA) continue;
        decay.push_back(s); Dv.push_back(config_.diffusion_coeffs[s]); lam.push_back(config_.decay_rates[s]);
    }
    n_decay_ = (int)decay.size();
    CUDA_CHECK(cudaMalloc(&d_decay_subs_, n_decay_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sub_D_,      n_decay_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sub_lambda_, n_decay_ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_decay_subs_, decay.data(), n_decay_*sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sub_D_,      Dv.data(),    n_decay_*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sub_lambda_, lam.data(),   n_decay_*sizeof(float), cudaMemcpyHostToDevice));

    // Work buffers sized for the full decay batch; single-substrate solves use slot 0.
    CUDA_CHECK(cudaMalloc(&d_dct_a_,  (size_t)n_decay_ * V * sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&d_dct_b_,  (size_t)n_decay_ * V * sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&d_dct_mu_, (size_t)V * sizeof(float)));
    float inv_dx2 = 1.0f / (config_.voxel_size * config_.voxel_size);
    int t = 256, b = (V + t - 1) / t;
    dct_build_mu_kernel<<<b, t>>>(d_dct_mu_, nx, ny, nz, inv_dx2);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Uptake substrates (O2/CCL2/VEGFA): per-substrate constants for the defect-correction.
    int upt_subs[3] = {CHEM_O2, CHEM_CCL2, CHEM_VEGFA};
    float upt_D[3], upt_lam[3];
    for (int i = 0; i < 3; i++) { upt_D[i] = config_.diffusion_coeffs[upt_subs[i]];
                                  upt_lam[i] = config_.decay_rates[upt_subs[i]]; }
    CUDA_CHECK(cudaMalloc(&d_upt_subs_,     3 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_upt_D_,        3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_upt_lambda_,   3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_upt_ubar_,     3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_upt_lamshift_, 3 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_upt_subs_,   upt_subs, 3*sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_upt_D_,      upt_D,    3*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_upt_lambda_, upt_lam,  3*sizeof(float), cudaMemcpyHostToDevice));

    // Contiguous C2C plans per pass — single, ×n_decay, and ×3 (uptake) batched variants.
    cufftHandle px, py, pz, pxb, pyb, pzb, px3, py3, pz3;
    CUFFT_CHECK(cufftPlan1d(&px,  nx, CUFFT_C2C, ny*nz));
    CUFFT_CHECK(cufftPlan1d(&py,  ny, CUFFT_C2C, nx*nz));
    CUFFT_CHECK(cufftPlan1d(&pz,  nz, CUFFT_C2C, nx*ny));
    CUFFT_CHECK(cufftPlan1d(&pxb, nx, CUFFT_C2C, n_decay_*ny*nz));
    CUFFT_CHECK(cufftPlan1d(&pyb, ny, CUFFT_C2C, n_decay_*nx*nz));
    CUFFT_CHECK(cufftPlan1d(&pzb, nz, CUFFT_C2C, n_decay_*nx*ny));
    CUFFT_CHECK(cufftPlan1d(&px3, nx, CUFFT_C2C, 3*ny*nz));
    CUFFT_CHECK(cufftPlan1d(&py3, ny, CUFFT_C2C, 3*nx*nz));
    CUFFT_CHECK(cufftPlan1d(&pz3, nz, CUFFT_C2C, 3*nx*ny));
    dct_plan_x_ = (int)px;  dct_plan_y_ = (int)py;  dct_plan_z_ = (int)pz;
    dct_plan_xb_ = (int)pxb; dct_plan_yb_ = (int)pyb; dct_plan_zb_ = (int)pzb;
    dct_plan_x3_ = (int)px3; dct_plan_y3_ = (int)py3; dct_plan_z3_ = (int)pz3;
    fft_ready_ = true;

    double mb = ((size_t)n_decay_ * 2.0 * V * sizeof(float2) + V * sizeof(float)) / (1024.0*1024.0);
    std::cout << "  [PDESolver] Spectral workspace (batched 3D Makhoul DCT): " << nx << "×" << ny
              << "×" << nz << ", batch=" << n_decay_ << " (" << mb << " MB, no inflation)" << std::endl;
}

// One 1D DCT pass on the contiguous axis (length n) over V_total elements; scratch d_dct_b_.
void PDESolver::dct_contig(int plan_i, int n, bool inverse, int V_total) {
    const int t = 256, bV = (V_total + t - 1) / t;
    cufftHandle plan = (cufftHandle)plan_i;
    int dir = inverse ? CUFFT_INVERSE : CUFFT_FORWARD;
    if (!inverse) {
        dct_reorder_kernel<<<bV, t>>>(d_dct_a_, d_dct_b_, n, V_total);
        CUFFT_CHECK(cufftExecC2C(plan, (cufftComplex*)d_dct_b_, (cufftComplex*)d_dct_b_, dir));
        dct_twiddle_fwd_kernel<<<bV, t>>>(d_dct_b_, n, V_total);
        std::swap(d_dct_a_, d_dct_b_);
    } else {
        dct_pretwiddle_inv_kernel<<<bV, t>>>(d_dct_a_, d_dct_b_, n, V_total);
        CUFFT_CHECK(cufftExecC2C(plan, (cufftComplex*)d_dct_b_, (cufftComplex*)d_dct_b_, dir));
        dct_unreorder_kernel<<<bV, t>>>(d_dct_b_, d_dct_a_, n, V_total);
    }
}

void PDESolver::dct_rotate(int na, int nb, int nc, int V_total) {
    const int t = 256, bV = (V_total + t - 1) / t;
    dct_rotate_kernel<<<bV, t>>>(d_dct_a_, d_dct_b_, na, nb, nc, V_total);
    std::swap(d_dct_a_, d_dct_b_);
}

// Forward 3D DCT over `nb` substrate blocks using contiguous plans (px,py,pz) for lengths (nx,ny,nz).
void PDESolver::dct3d_forward(int px, int py, int pz, int nb) {
    int nx = config_.nx, ny = config_.ny, nz = config_.nz, V = nx*ny*nz, Vt = nb * V;
    dct_contig(px, nx, false, Vt); dct_rotate(nz, ny, nx, Vt);
    dct_contig(py, ny, false, Vt); dct_rotate(nx, nz, ny, Vt);
    dct_contig(pz, nz, false, Vt); dct_rotate(ny, nx, nz, Vt);
}
void PDESolver::dct3d_inverse(int px, int py, int pz, int nb) {
    int nx = config_.nx, ny = config_.ny, nz = config_.nz, V = nx*ny*nz, Vt = nb * V;
    dct_contig(px, nx, true, Vt); dct_rotate(nz, ny, nx, Vt);
    dct_contig(py, ny, true, Vt); dct_rotate(nx, nz, ny, Vt);
    dct_contig(pz, nz, true, Vt); dct_rotate(ny, nx, nz, Vt);
}

// Single-substrate constant-coefficient elliptic solve via 3D DCT (raw, no clamp).
void PDESolver::fft_solve_const(float* d_C, const float* d_b, float D, float lambda) {
    int V = get_total_voxels();
    const int t = 256, bV = (V + t - 1) / t;
    dct_real2cplx_kernel<<<bV, t>>>(d_b, d_dct_a_, V);
    dct3d_forward(dct_plan_x_, dct_plan_y_, dct_plan_z_, 1);
    dct_divide_mu_kernel<<<bV, t>>>(d_dct_a_, d_dct_mu_, D, lambda, V);
    dct3d_inverse(dct_plan_x_, dct_plan_y_, dct_plan_z_, 1);
    dct_cplx2real_kernel<<<bV, t>>>(d_C, d_dct_a_, V);
}

// All decay-only substrates solved together in one batched 3D DCT.
void PDESolver::solve_spectral_decay_batched() {
    if (n_decay_ <= 0) return;
    int V = get_total_voxels();
    const int t = 256, bT = (n_decay_ * V + t - 1) / t;
    dct_gather_kernel<<<bT, t>>>(d_src_, d_decay_subs_, d_dct_a_, n_decay_, V);
    dct3d_forward(dct_plan_xb_, dct_plan_yb_, dct_plan_zb_, n_decay_);
    dct_divide_mu_batched<<<bT, t>>>(d_dct_a_, d_dct_mu_, d_sub_D_, d_sub_lambda_, n_decay_, V);
    dct3d_inverse(dct_plan_xb_, dct_plan_yb_, dct_plan_zb_, n_decay_);
    dct_scatter_kernel<<<bT, t>>>(d_conc_, d_decay_subs_, d_dct_a_, n_decay_, V);
}

float PDESolver::host_sum(const float* d_x, int n) {
    cg_sum_kernel<<<cg_blocks_, 256>>>(d_x, d_dot_buf_, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> h(cg_blocks_);
    CUDA_CHECK(cudaMemcpy(h.data(), d_dot_buf_, (size_t)cg_blocks_ * sizeof(float), cudaMemcpyDeviceToHost));
    double s = 0.0;
    for (int i = 0; i < cg_blocks_; i++) s += h[i];
    return (float)s;
}

// Mean-shifted batched defect-correction for the 3 per-voxel-uptake substrates (O2/CCL2/VEGFA).
// Split M = (λ+Ū − D∇²) + (U−Ū); iterate c ← M0⁻¹(S − (U−Ū)c). The mean-zero (U−Ū) can't
// excite the low mode, so it converges in a few iterations even for O2 (tiny λ, strong U).
// Exact at the fixed point; sync-free per iteration; batched over the 3 substrates.
void PDESolver::solve_spectral_uptake_split() {
    int V = get_total_voxels();
    const int t = 256, b3 = (3 * V + t - 1) / t;
    const int subs[3] = {CHEM_O2, CHEM_CCL2, CHEM_VEGFA};

    // Per-substrate mean uptake Ū and the shifted decay (λ+Ū) for M0's diagonal.
    float ubar[3], lamshift[3];
    for (int i = 0; i < 3; i++) {
        ubar[i] = host_sum(d_upt_ + (size_t)subs[i] * V, V) / (float)V;
        lamshift[i] = config_.decay_rates[subs[i]] + ubar[i];
    }
    CUDA_CHECK(cudaMemcpy(d_upt_ubar_,     ubar,     3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_upt_lamshift_, lamshift, 3 * sizeof(float), cudaMemcpyHostToDevice));

    const int K = 8;  // fixed iterations (ρ≈0.005 for O2 ⇒ converges in 2–3; 8 is safe margin)
    for (int k = 0; k < K; k++) {
        dct_uptake_rhs_kernel<<<b3, t>>>(d_dct_a_, d_src_, d_upt_, d_conc_,
                                         d_upt_subs_, d_upt_ubar_, 3, V);     // a = S − (U−Ū)c
        dct3d_forward(dct_plan_x3_, dct_plan_y3_, dct_plan_z3_, 3);
        dct_divide_mu_batched<<<b3, t>>>(d_dct_a_, d_dct_mu_, d_upt_D_, d_upt_lamshift_, 3, V);  // M0⁻¹
        dct3d_inverse(dct_plan_x3_, dct_plan_y3_, dct_plan_z3_, 3);
        dct_scatter_kernel<<<b3, t>>>(d_conc_, d_upt_subs_, d_dct_a_, 3, V);  // c_{k+1} (clamped ≥0)
    }
    for (int i = 0; i < 3; i++) { last_cg_iters_[subs[i]] = K; last_cg_residual_[subs[i]] = 0.0f; }
}

// FFT-preconditioned CG for substrates with per-voxel uptake (O2/CCL2/VEGFA).
// Operator = full (λ + U − D∇²); preconditioner = constant-coeff DCT solve (λ − D∇²)^{-1}.
int PDESolver::solve_spectral_uptake(float* d_C, const float* d_S, const float* d_U,
                                     float D, float lambda) {
    const int n = get_total_voxels();
    const int nx = config_.nx, ny = config_.ny, nz = config_.nz;
    const float inv_dx2 = 1.0f / (config_.voxel_size * config_.voxel_size);
    const int t1 = 256, b1 = (n + t1 - 1) / t1;
    dim3 blk(8, 8, 8), grd((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);

    // r = b − A x0   (warm start = current d_C)
    steady_operator_kernel<<<grd, blk>>>(d_C, d_cg_Ap_, d_U, nx, ny, nz, D, lambda, inv_dx2);
    cg_copy<<<b1, t1>>>(d_cg_r_, d_S, n);
    cg_axpy<<<b1, t1>>>(d_cg_r_, d_cg_Ap_, -1.0f, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    fft_solve_const(d_cg_z_, d_cg_r_, D, lambda);  // z = A0^{-1} r (no clamp — preconditioner)
    cg_copy<<<b1, t1>>>(d_cg_p_, d_cg_z_, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    float rz_old  = host_dot(d_cg_r_, d_cg_z_, n);
    float rr_init = host_dot(d_cg_r_, d_cg_r_, n);
    if (rr_init <= 0.0f) { cg_scratch_residual_ = 0.0f; return 0; }
    const float rr_target = config_.cg_tol * config_.cg_tol * rr_init;

    int iter = 0; float rr_new = rr_init;
    for (; iter < config_.cg_maxiter; iter++) {
        steady_operator_kernel<<<grd, blk>>>(d_cg_p_, d_cg_Ap_, d_U, nx, ny, nz, D, lambda, inv_dx2);
        CUDA_CHECK(cudaDeviceSynchronize());
        float pAp = host_dot(d_cg_p_, d_cg_Ap_, n);
        float alpha = rz_old / (pAp + 1e-30f);
        cg_axpy<<<b1, t1>>>(d_C,     d_cg_p_,  alpha,  n);
        cg_axpy<<<b1, t1>>>(d_cg_r_, d_cg_Ap_, -alpha, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        rr_new = host_dot(d_cg_r_, d_cg_r_, n);
        if (rr_new < rr_target) { iter++; break; }
        fft_solve_const(d_cg_z_, d_cg_r_, D, lambda);  // z = A0^{-1} r
        float rz_new = host_dot(d_cg_r_, d_cg_z_, n);
        float beta = rz_new / (rz_old + 1e-30f);
        cg_xpby<<<b1, t1>>>(d_cg_p_, d_cg_z_, beta, n);
        rz_old = rz_new;
    }
    cg_clamp_nonneg<<<b1, t1>>>(d_C, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    cg_scratch_residual_ = sqrtf(rr_new / (rr_init + 1e-30f));
    return iter;
}

void PDESolver::solve_spectral() {
    if (!fft_ready_) { std::cerr << "solve_spectral: FFT workspace not allocated\n"; return; }

    // (1) All decay-only substrates (U=0) in one batched 3D DCT — exact one-shot.
    //     D=0 substrates fall out correctly (denom=λ ⇒ c=S/λ).
    solve_spectral_decay_batched();

    // (2) Per-voxel-uptake substrates (O2/CCL2/VEGFA): mean-shifted batched defect-correction.
    solve_spectral_uptake_split();

    CUDA_CHECK(cudaDeviceSynchronize());  // single barrier before downstream (gradients) read d_conc_
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
