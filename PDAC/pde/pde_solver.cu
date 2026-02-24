/**
 * PDE SOLVER v2: Clean Operator-Splitting Implementation
 *
 * Architecture:
 * 1. Collect agent sources → S[voxel, substrate] via atomic adds
 * 2. Apply sources and decay → C = C + dt*(S - λ*C)
 * 3. Implicit diffusion → (I - dt*D*∇²)C_new = C_old
 * 4. Read concentrations back to agents
 *
 * Solver: Preconditioned Conjugate Gradient (diagonal preconditioner)
 * Validation: Mass conservation, decay correctness, diffusion spread
 */

#include "pde_solver.cuh"
#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>

namespace PDAC {

// ============================================================================
// CUDA Error Checking
// ============================================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// INTEGRATION HELPER KERNELS (for coupling with FLAME GPU agents)
// ============================================================================

/**
 * Read concentration values at agent voxels
 */
__global__ void read_concentrations_at_voxels(
    const float* __restrict__ concentrations,
    const int* __restrict__ agent_x,
    const int* __restrict__ agent_y,
    const int* __restrict__ agent_z,
    float* __restrict__ output,
    int num_agents,
    int substrate_idx,
    int nx, int ny, int nz)
{
    int agent_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (agent_idx >= num_agents) return;

    int x = agent_x[agent_idx];
    int y = agent_y[agent_idx];
    int z = agent_z[agent_idx];

    if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz) {
        output[agent_idx] = 0.0f;
        return;
    }

    int idx = z * (nx * ny) + y * nx + x;
    output[agent_idx] = concentrations[idx];
}

/**
 * Add sources from agents via atomic operations
 */
__global__ void add_sources_from_agents(
    float* __restrict__ sources,
    float* __restrict__ uptakes,
    const int* __restrict__ agent_x,
    const int* __restrict__ agent_y,
    const int* __restrict__ agent_z,
    const float* __restrict__ rates,
    int num_agents,
    int substrate_idx,
    int nx, int ny, int nz,
    float voxel_volume,
    float dt)
{
    int agent_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (agent_idx >= num_agents) return;

    int x = agent_x[agent_idx];
    int y = agent_y[agent_idx];
    int z = agent_z[agent_idx];

    if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz) return;

    float rate = rates[agent_idx];
    if (rate == 0.0f) return;

    int idx = z * (nx * ny) + y * nx + x;

    if (rate > 0.0f) {
        // Source (release)
        float source_contrib = rate / voxel_volume;
        atomicAdd(&sources[idx], source_contrib);
    } else {
        // Uptake (sink)
        float uptake_mag = fabsf(rate);
        atomicAdd(&uptakes[idx], uptake_mag);
    }
}

// ============================================================================
// HELPER KERNELS
// ============================================================================

/**
 * Collect agent sources via atomic add
 * Each thread processes one agent, reads its source for a substrate,
 * and atomically adds to the voxel's source array
 */
__global__ void collect_agent_sources_kernel(
    float* __restrict__ d_sources,           // [num_voxels] source array
    const int* __restrict__ d_agent_x,       // Agent X coordinates
    const int* __restrict__ d_agent_y,       // Agent Y coordinates
    const int* __restrict__ d_agent_z,       // Agent Z coordinates
    const float* __restrict__ d_agent_rates, // Agent source rate for this substrate
    int num_agents,
    int nx, int ny, int nz,
    float voxel_volume)
{
    int agent_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (agent_idx >= num_agents) return;

    int x = d_agent_x[agent_idx];
    int y = d_agent_y[agent_idx];
    int z = d_agent_z[agent_idx];

    // Validate bounds
    if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz) return;

    float rate = d_agent_rates[agent_idx];
    if (rate == 0.0f) return;

    // Convert: amount/(cell·time) → concentration/time by dividing by voxel volume
    float source_contribution = rate / voxel_volume;

    int voxel_idx = z * (nx * ny) + y * nx + x;
    atomicAdd(&d_sources[voxel_idx], source_contribution);
}

/**
 * Apply sources and decay: C = C + dt*(S - λ*C)
 * This is a simple explicit step that applies accumulated sources and decay
 */
__global__ void apply_sources_and_decay_kernel(
    float* __restrict__ C,           // Concentrations to update
    const float* __restrict__ S,     // Source array
    float decay_rate,                // λ (decay rate, 1/s)
    float dt,                        // Timestep
    int n)                           // Total number of voxels
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Decay term: exp(-λ*dt)
    float decay_factor = expf(-decay_rate * dt);

    // Apply: C_new = C_old * exp(-λ*dt) + S * (1 - exp(-λ*dt)) / λ
    // Simplified: C_new = C_old * decay_factor + S * dt (for small λ*dt, approximately)
    // More accurate: C_new = C_old * exp(-λ*dt) + S / λ * (1 - exp(-λ*dt))

    float source_term = S[idx];
    if (decay_rate > 1e-10f) {
        // Exact solution for source + decay ODE
        float decay_decay = (1.0f - decay_factor) / decay_rate;
        C[idx] = C[idx] * decay_factor + source_term * decay_decay;
    } else {
        // No decay: just add source
        C[idx] = C[idx] + source_term * dt;
    }
}

/**
 * Apply diffusion operator: Ax = (I - dt*D*∇²)x
 * Uses 7-point stencil with Neumann (no-flux) boundary conditions
 */
__global__ void apply_diffusion_operator_kernel(
    const float* __restrict__ x,     // Input concentration
    float* __restrict__ Ax,          // Output: A*x
    float D,                         // Diffusion coefficient
    float dt,                        // Timestep
    float dx,                        // Voxel size
    int nx, int ny, int nz)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= nx || iy >= ny || iz >= nz) return;

    int idx = iz * (nx * ny) + iy * nx + ix;
    float x_center = x[idx];

    // Compute Laplacian using 7-point stencil with Neumann BC
    float laplacian = 0.0f;
    float dx2 = dx * dx;

    // X-direction (periodic or Neumann)
    if (ix > 0) {
        laplacian += x[idx - 1] - x_center;
    } else {
        laplacian += 0.0f;  // Neumann: no flux at boundary
    }
    if (ix < nx - 1) {
        laplacian += x[idx + 1] - x_center;
    } else {
        laplacian += 0.0f;
    }

    // Y-direction
    if (iy > 0) {
        laplacian += x[idx - nx] - x_center;
    } else {
        laplacian += 0.0f;
    }
    if (iy < ny - 1) {
        laplacian += x[idx + nx] - x_center;
    } else {
        laplacian += 0.0f;
    }

    // Z-direction
    if (iz > 0) {
        laplacian += x[idx - nx * ny] - x_center;
    } else {
        laplacian += 0.0f;
    }
    if (iz < nz - 1) {
        laplacian += x[idx + nx * ny] - x_center;
    } else {
        laplacian += 0.0f;
    }

    laplacian /= dx2;

    // Operator: Ax = x - dt*D*∇²x
    // (This is: (I - dt*D*∇²)x = x + dt*D*laplacian)
    // Wait, let me reconsider: for implicit backward Euler:
    // (I + dt*D*∇²)C_new = C_old
    // So A = I + dt*D*∇² means Ax = x + dt*D*∇²x
    // No wait, we're solving (I - dt*D*∇²)C_new = C_old for diffusion with Neumann
    // That's: C_new - dt*D*∇²C_new = C_old
    // Rearranged: A*C_new = C_old where A = I - dt*D*∇²
    // So: A*x = x - dt*D*∇²x

    Ax[idx] = x_center - dt * D * laplacian;
}

/**
 * Compute diagonal preconditioner: M = I + dt*D*6/dx²
 */
__global__ void compute_diagonal_preconditioner_kernel(
    float* __restrict__ M_inv,  // Output: M^{-1}
    float D,                    // Diffusion coefficient
    float dt,                   // Timestep
    float dx,                   // Voxel size
    int n)                      // Number of voxels
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float dx2 = dx * dx;
    // Diagonal of A = I - dt*D*∇² is approximately 1 + dt*D*6/dx²
    // (6 neighbors in 3D, each contributing -dt*D/dx²)
    float diag = 1.0f + dt * D * 6.0f / dx2;

    // Clamp to ensure stability
    if (!isfinite(diag) || diag <= 0.0f) diag = 1.0f;

    M_inv[idx] = 1.0f / diag;
}

/**
 * Apply diagonal preconditioner: z = M^{-1} * r
 */
__global__ void apply_preconditioner_kernel(
    const float* __restrict__ M_inv,
    const float* __restrict__ r,
    float* __restrict__ z,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    z[idx] = M_inv[idx] * r[idx];
}

/**
 * Vector operations for CG: y = alpha * x
 */
__global__ void vector_scale_kernel(
    float* __restrict__ y,
    const float* __restrict__ x,
    float alpha,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = alpha * x[idx];
    }
}

/**
 * Vector operations for CG: y = y + alpha * x
 */
__global__ void vector_axpy_kernel(
    float* __restrict__ y,
    const float* __restrict__ x,
    float alpha,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] += alpha * x[idx];
    }
}

/**
 * Dot product partial reduction
 */
__global__ void dot_product_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ partial_sums,
    int n)
{
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    if (idx < n) {
        sum = x[idx] * y[idx];
    }
    sdata[tid] = sum;
    __syncthreads();

    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// ============================================================================
// PDESolver Class Implementation
// ============================================================================

PDESolver::PDESolver(const PDEConfig& config)
    : config_(config),
      d_concentrations_current_(nullptr),
      d_concentrations_next_(nullptr),
      d_sources_(nullptr),
      d_cg_r_(nullptr),
      d_cg_z_(nullptr),
      d_cg_p_(nullptr),
      d_cg_Ap_(nullptr),
      d_precond_diag_inv_(nullptr),
      d_dot_buffer_(nullptr),
      cg_reduction_blocks_(0)
{
}

PDESolver::~PDESolver() {
    if (d_concentrations_current_) CUDA_CHECK(cudaFree(d_concentrations_current_));
    if (d_concentrations_next_) CUDA_CHECK(cudaFree(d_concentrations_next_));
    if (d_sources_) CUDA_CHECK(cudaFree(d_sources_));
    if (d_uptakes_) CUDA_CHECK(cudaFree(d_uptakes_));
    if (d_recruitment_sources_) CUDA_CHECK(cudaFree(d_recruitment_sources_));
    if (d_cg_r_) CUDA_CHECK(cudaFree(d_cg_r_));
    if (d_cg_z_) CUDA_CHECK(cudaFree(d_cg_z_));
    if (d_cg_p_) CUDA_CHECK(cudaFree(d_cg_p_));
    if (d_cg_Ap_) CUDA_CHECK(cudaFree(d_cg_Ap_));
    if (d_precond_diag_inv_) CUDA_CHECK(cudaFree(d_precond_diag_inv_));
    if (d_dot_buffer_) CUDA_CHECK(cudaFree(d_dot_buffer_));
}

void PDESolver::initialize() {
    int total_voxels = config_.nx * config_.ny * config_.nz;
    size_t voxel_size = total_voxels * sizeof(float);
    size_t total_size = total_voxels * config_.num_substrates * sizeof(float);

    // Allocate concentration arrays for all substrates
    CUDA_CHECK(cudaMalloc(&d_concentrations_current_, total_size));
    CUDA_CHECK(cudaMalloc(&d_concentrations_next_, total_size));

    // Initialize to zero
    CUDA_CHECK(cudaMemset(d_concentrations_current_, 0, total_size));
    CUDA_CHECK(cudaMemset(d_concentrations_next_, 0, total_size));

    // Allocate source array for ALL substrates (not just one!)
    CUDA_CHECK(cudaMalloc(&d_sources_, total_size));  // FIX: was voxel_size, now total_size
    CUDA_CHECK(cudaMemset(d_sources_, 0, total_size));

    // Allocate uptakes array for ALL substrates
    CUDA_CHECK(cudaMalloc(&d_uptakes_, total_size));
    CUDA_CHECK(cudaMemset(d_uptakes_, 0, total_size));

    // Allocate recruitment sources array (int per voxel)
    size_t recruitment_size = total_voxels * sizeof(int);
    CUDA_CHECK(cudaMalloc(&d_recruitment_sources_, recruitment_size));  // FIX: was missing
    CUDA_CHECK(cudaMemset(d_recruitment_sources_, 0, recruitment_size));

    // Allocate CG working arrays
    CUDA_CHECK(cudaMalloc(&d_cg_r_, voxel_size));
    CUDA_CHECK(cudaMalloc(&d_cg_z_, voxel_size));
    CUDA_CHECK(cudaMalloc(&d_cg_p_, voxel_size));
    CUDA_CHECK(cudaMalloc(&d_cg_Ap_, voxel_size));
    CUDA_CHECK(cudaMalloc(&d_precond_diag_inv_, voxel_size));

    // Allocate reduction buffer for dot products
    cg_reduction_blocks_ = (total_voxels + 256 - 1) / 256;
    CUDA_CHECK(cudaMalloc(&d_dot_buffer_, cg_reduction_blocks_ * sizeof(float)));

    std::cout << "[PDESolver] Initialized: " << config_.nx << "x" << config_.ny << "x" << config_.nz
              << " grid, " << config_.num_substrates << " substrates" << std::endl;
}

void PDESolver::reset_sources() {
    int total_voxels = config_.nx * config_.ny * config_.nz;
    size_t total_size = total_voxels * config_.num_substrates * sizeof(float);  // FIX: reset all substrates
    CUDA_CHECK(cudaMemset(d_sources_, 0, total_size));
}

void PDESolver::reset_recruitment_sources() {
    if (!d_recruitment_sources_) return;  // FIX: actually reset the memory
    int total_voxels = config_.nx * config_.ny * config_.nz;
    size_t recruitment_size = total_voxels * sizeof(int);
    CUDA_CHECK(cudaMemset(d_recruitment_sources_, 0, recruitment_size));
}

void PDESolver::reset_uptakes() {
    int total_voxels = config_.nx * config_.ny * config_.nz;
    size_t total_size = total_voxels * config_.num_substrates * sizeof(float);
    CUDA_CHECK(cudaMemset(d_uptakes_, 0, total_size));
}

void PDESolver::reset_concentrations() {
    int total_voxels = config_.nx * config_.ny * config_.nz;
    size_t total_size = total_voxels * config_.num_substrates * sizeof(float);
    CUDA_CHECK(cudaMemset(d_concentrations_current_, 0, total_size));
}

float* PDESolver::get_device_concentration_ptr(int substrate_idx) {
    int total_voxels = config_.nx * config_.ny * config_.nz;
    return d_concentrations_current_ + substrate_idx * total_voxels;
}

float* PDESolver::get_device_source_ptr(int substrate_idx) {
    // Return offset pointer for this substrate (like get_device_concentration_ptr)
    int total_voxels = config_.nx * config_.ny * config_.nz;
    return d_sources_ + substrate_idx * total_voxels;
}

float* PDESolver::get_device_uptake_ptr(int substrate_idx) {
    int total_voxels = config_.nx * config_.ny * config_.nz;
    return d_uptakes_ + substrate_idx * total_voxels;
}

/**
 * Solve diffusion for one substrate using CG
 * Solves: (I - dt*D*∇²)C_new = C_old
 */
int PDESolver::solve_cg_diffusion(
    float* d_C_current,
    const float* d_RHS,
    float D,
    float dt,
    float dx,
    int n)
{
    const int max_iters = 500;
    const float tolerance = 1e-4f;
    const int threads_1d = 256;
    const int blocks_1d = (n + threads_1d - 1) / threads_1d;

    dim3 block_3d(8, 8, 8);
    int grid_x = (config_.nx + 7) / 8;
    int grid_y = (config_.ny + 7) / 8;
    int grid_z = (config_.nz + 7) / 8;
    dim3 grid_3d(grid_x, grid_y, grid_z);

    // Compute preconditioner diagonal
    compute_diagonal_preconditioner_kernel<<<blocks_1d, threads_1d>>>(
        d_precond_diag_inv_, D, dt, dx, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Helper lambda for dot product
    auto dot_product = [&](const float* x, const float* y) -> float {
        dot_product_kernel<<<cg_reduction_blocks_, threads_1d>>>(x, y, d_dot_buffer_, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> h_partial(cg_reduction_blocks_);
        CUDA_CHECK(cudaMemcpy(h_partial.data(), d_dot_buffer_,
                              cg_reduction_blocks_ * sizeof(float), cudaMemcpyDeviceToHost));
        float sum = 0.0f;
        for (int i = 0; i < cg_reduction_blocks_; i++) {
            sum += h_partial[i];
        }
        return sum;
    };

    // Initial residual: r = b - A*x0
    apply_diffusion_operator_kernel<<<grid_3d, block_3d>>>(
        d_C_current, d_cg_Ap_, D, dt, dx, config_.nx, config_.ny, config_.nz);
    CUDA_CHECK(cudaDeviceSynchronize());

    // r = RHS - Ax
    vector_scale_kernel<<<blocks_1d, threads_1d>>>(d_cg_r_, d_RHS, 1.0f, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    vector_axpy_kernel<<<blocks_1d, threads_1d>>>(d_cg_r_, d_cg_Ap_, -1.0f, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Initial preconditioner: z = M^{-1} * r
    apply_preconditioner_kernel<<<blocks_1d, threads_1d>>>(
        d_precond_diag_inv_, d_cg_r_, d_cg_z_, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // p = z
    vector_scale_kernel<<<blocks_1d, threads_1d>>>(d_cg_p_, d_cg_z_, 1.0f, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    float rzold = dot_product(d_cg_r_, d_cg_z_);
    float r_norm_sq = dot_product(d_cg_r_, d_cg_r_);

    int iter;
    for (iter = 0; iter < max_iters; iter++) {
        // Ap = A*p
        apply_diffusion_operator_kernel<<<grid_3d, block_3d>>>(
            d_cg_p_, d_cg_Ap_, D, dt, dx, config_.nx, config_.ny, config_.nz);
        CUDA_CHECK(cudaDeviceSynchronize());

        // alpha = (r·z) / (p·Ap)
        float pAp = dot_product(d_cg_p_, d_cg_Ap_);
        float alpha = rzold / (pAp + 1e-30f);

        // x = x + alpha*p
        vector_axpy_kernel<<<blocks_1d, threads_1d>>>(d_C_current, d_cg_p_, alpha, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        // r = r - alpha*Ap
        vector_axpy_kernel<<<blocks_1d, threads_1d>>>(d_cg_r_, d_cg_Ap_, -alpha, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Check convergence
        float r_norm = dot_product(d_cg_r_, d_cg_r_);
        if (r_norm / r_norm_sq < tolerance * tolerance) {
            std::cout << "  [CG] Converged at iter " << iter << ", residual norm: " << sqrtf(r_norm / r_norm_sq) << std::endl;
            return iter;
        }

        // z = M^{-1} * r
        apply_preconditioner_kernel<<<blocks_1d, threads_1d>>>(
            d_precond_diag_inv_, d_cg_r_, d_cg_z_, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        float rznew = dot_product(d_cg_r_, d_cg_z_);
        float beta = rznew / (rzold + 1e-30f);

        // p = z + beta*p
        vector_scale_kernel<<<blocks_1d, threads_1d>>>(d_cg_p_, d_cg_z_, 1.0f, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        vector_axpy_kernel<<<blocks_1d, threads_1d>>>(d_cg_p_, d_cg_p_, beta, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        rzold = rznew;
    }

    std::cout << "  [CG] Max iterations reached (" << iter << ")" << std::endl;
    return iter;
}

void PDESolver::solve_timestep() {
    int total_voxels = config_.nx * config_.ny * config_.nz;
    const int threads_1d = 256;
    const int blocks_1d = (total_voxels + threads_1d - 1) / threads_1d;

    // For each substrate
    for (int s = 0; s < config_.num_substrates; s++) {
        float* C_current = d_concentrations_current_ + s * total_voxels;

        // Step 1: Apply sources and decay
        apply_sources_and_decay_kernel<<<blocks_1d, threads_1d>>>(
            C_current, d_sources_, config_.decay_rates[s], config_.dt_pde, total_voxels);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Step 2: Solve implicit diffusion
        solve_cg_diffusion(C_current, C_current, config_.diffusion_coeffs[s],
                          config_.dt_pde, config_.voxel_size, total_voxels);
    }
}

// Stubs for compatibility
void PDESolver::set_sources(const float* h_sources, int substrate_idx) {}
void PDESolver::add_source_at_voxel(int x, int y, int z, int substrate_idx, float value) {}
void PDESolver::get_concentrations(float* h_concentrations, int substrate_idx) const {
    if (substrate_idx < 0 || substrate_idx >= config_.num_substrates) {
        throw std::runtime_error("Invalid substrate index");
    }

    int total_voxels = config_.nx * config_.ny * config_.nz;

    // Copy concentrations from device to host
    CUDA_CHECK(cudaMemcpy(
        h_concentrations,
        d_concentrations_current_ + substrate_idx * total_voxels,
        total_voxels * sizeof(float),
        cudaMemcpyDeviceToHost
    ));
}

float PDESolver::get_concentration_at_voxel(int x, int y, int z, int substrate_idx) const {
    if (x < 0 || x >= config_.nx || y < 0 || y >= config_.ny || z < 0 || z >= config_.nz) {
        return 0.0f;
    }
    if (substrate_idx < 0 || substrate_idx >= config_.num_substrates) {
        return 0.0f;
    }

    int voxel_idx = z * (config_.nx * config_.ny) + y * config_.nx + x;
    int total_voxels = config_.nx * config_.ny * config_.nz;
    int offset_voxel = substrate_idx * total_voxels + voxel_idx;

    float value = 0.0f;
    CUDA_CHECK(cudaMemcpy(&value, d_concentrations_current_ + offset_voxel, sizeof(float), cudaMemcpyDeviceToHost));
    return value;
}

void PDESolver::set_initial_concentration(int substrate_idx, float value) {
    if (substrate_idx < 0 || substrate_idx >= config_.num_substrates) {
        throw std::runtime_error("Invalid substrate index");
    }

    int total_voxels = config_.nx * config_.ny * config_.nz;
    std::vector<float> h_data(total_voxels, value);

    CUDA_CHECK(cudaMemcpy(
        d_concentrations_current_ + substrate_idx * total_voxels,
        h_data.data(),
        total_voxels * sizeof(float),
        cudaMemcpyHostToDevice
    ));
}

float PDESolver::get_total_source(int substrate_idx) {
    if (substrate_idx < 0 || substrate_idx >= config_.num_substrates) {
        return 0.0f;
    }

    int total_voxels = config_.nx * config_.ny * config_.nz;
    std::vector<float> h_sources(total_voxels);

    CUDA_CHECK(cudaMemcpy(
        h_sources.data(),
        d_sources_ + substrate_idx * total_voxels,
        total_voxels * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    float total = 0.0f;
    for (int i = 0; i < total_voxels; i++) {
        total += h_sources[i];
    }
    return total;
}

}  // namespace PDAC
