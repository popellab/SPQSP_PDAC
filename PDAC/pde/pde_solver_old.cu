#include "pde_solver.cuh"
#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <nvtx3/nvToolsExt.h>

namespace PDAC {

// CUDA error checking macro
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
// CUDA Kernels
// ============================================================================

__global__ void diffusion_reaction_kernel(
    const float* __restrict__ C_curr,
    float* __restrict__ C_next,
    const float* __restrict__ sources,
    int nx, int ny, int nz,
    float D, float lambda, float dt, float dx)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= nx || y >= ny || z >= nz) return;
    
    int idx = z * (nx * ny) + y * nx + x;
    
    float C = C_curr[idx];
    
    // Compute Laplacian using 7-point stencil
    float laplacian = 0.0f;
    float dx2 = dx * dx;
    int neighbor_count = 0;
    
    // X-direction
    if (x > 0) {
        laplacian += (C_curr[idx - 1] - C) / dx2;
        neighbor_count++;
    } else {
        // Neumann BC: zero flux (reflective)
        laplacian += 0.0f;
    }
    
    if (x < nx - 1) {
        laplacian += (C_curr[idx + 1] - C) / dx2;
        neighbor_count++;
    } else {
        laplacian += 0.0f;
    }
    
    // Y-direction
    if (y > 0) {
        laplacian += (C_curr[idx - nx] - C) / dx2;
        neighbor_count++;
    } else {
        laplacian += 0.0f;
    }
    
    if (y < ny - 1) {
        laplacian += (C_curr[idx + nx] - C) / dx2;
        neighbor_count++;
    } else {
        laplacian += 0.0f;
    }
    
    // Z-direction
    if (z > 0) {
        laplacian += (C_curr[idx - nx * ny] - C) / dx2;
        neighbor_count++;
    } else {
        laplacian += 0.0f;
    }
    
    if (z < nz - 1) {
        laplacian += (C_curr[idx + nx * ny] - C) / dx2;
        neighbor_count++;
    } else {
        laplacian += 0.0f;
    }
    
    // Reaction-diffusion-decay equation: dC/dt = D*∇²C - λ*C + S
    float diffusion = D * laplacian;
    float decay = -lambda * C;
    float source = sources[idx];
    
    // Forward Euler time integration
    C_next[idx] = C + dt * (diffusion + decay + source);
    
    // Ensure non-negative concentrations
    if (C_next[idx] < 0.0f) {
        C_next[idx] = 0.0f;
    }
}

__global__ void copy_substrate_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// Kernel: Read concentration at specific voxel for all agents
__global__ void read_concentrations_at_voxels(
    const float* __restrict__ d_concentrations,
    const int* __restrict__ d_agent_x,
    const int* __restrict__ d_agent_y,
    const int* __restrict__ d_agent_z,
    float* __restrict__ d_agent_concentrations,
    int num_agents,
    int substrate_idx,
    int nx, int ny, int nz)
{
    int agent_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (agent_idx >= num_agents) return;
    
    int x = d_agent_x[agent_idx];
    int y = d_agent_y[agent_idx];
    int z = d_agent_z[agent_idx];
    
    // Bounds check
    if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz) {
        d_agent_concentrations[agent_idx] = 0.0f;
        return;
    }
    
    // Compute flat index: substrate_offset + z*(nx*ny) + y*nx + x
    int voxel_idx = z * (nx * ny) + y * nx + x;
    int total_voxels = nx * ny * nz;
    int concentration_idx = substrate_idx * total_voxels + voxel_idx;
    
    d_agent_concentrations[agent_idx] = d_concentrations[concentration_idx];
}

// Kernel: Compute spatial gradients using central differences
// grad_x = (C[x+1,y,z] - C[x-1,y,z]) / (2*voxel_size)
// Boundary cells use forward/backward differences
__global__ void compute_gradients_at_voxels(
    const float* __restrict__ d_concentrations,
    const int* __restrict__ d_agent_x,
    const int* __restrict__ d_agent_y,
    const int* __restrict__ d_agent_z,
    float* __restrict__ d_grad_x,
    float* __restrict__ d_grad_y,
    float* __restrict__ d_grad_z,
    int num_agents,
    int substrate_idx,
    int nx, int ny, int nz,
    float voxel_size)
{
    int agent_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (agent_idx >= num_agents) return;

    int x = d_agent_x[agent_idx];
    int y = d_agent_y[agent_idx];
    int z = d_agent_z[agent_idx];

    // Bounds check
    if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz) {
        d_grad_x[agent_idx] = 0.0f;
        d_grad_y[agent_idx] = 0.0f;
        d_grad_z[agent_idx] = 0.0f;
        return;
    }

    int total_voxels = nx * ny * nz;
    int substrate_offset = substrate_idx * total_voxels;

    // Helper lambda to get concentration at (ix, iy, iz)
    auto get_concentration = [&](int ix, int iy, int iz) -> float {
        if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz) {
            return 0.0f;  // Neumann BC: no flux at boundary
        }
        int voxel_idx = iz * (nx * ny) + iy * nx + ix;
        return d_concentrations[substrate_offset + voxel_idx];
    };

    // Compute gradients using central differences
    float two_dx = 2.0f * voxel_size;

    // Gradient in X direction
    float c_right = get_concentration(x + 1, y, z);
    float c_left = get_concentration(x - 1, y, z);
    d_grad_x[agent_idx] = (c_right - c_left) / two_dx;

    // Gradient in Y direction
    float c_up = get_concentration(x, y + 1, z);
    float c_down = get_concentration(x, y - 1, z);
    d_grad_y[agent_idx] = (c_up - c_down) / two_dx;

    // Gradient in Z direction
    float c_front = get_concentration(x, y, z + 1);
    float c_back = get_concentration(x, y, z - 1);
    d_grad_z[agent_idx] = (c_front - c_back) / two_dx;
}

// Kernel: Write (add) sources from agents to voxels
__global__ void add_sources_from_agents(
    float* __restrict__ d_sources,
    float* __restrict__ d_uptakes,  // NEW: separate uptakes array
    const int* __restrict__ d_agent_x,
    const int* __restrict__ d_agent_y,
    const int* __restrict__ d_agent_z,
    const float* __restrict__ d_agent_source_rates,
    int num_agents,
    int substrate_idx,
    int nx, int ny, int nz,
    float voxel_volume,
    float dt)
{
    int agent_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (agent_idx >= num_agents) return;

    int x = d_agent_x[agent_idx];
    int y = d_agent_y[agent_idx];
    int z = d_agent_z[agent_idx];

    // Bounds check
    if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz) {
        return;
    }

    float source_rate = d_agent_source_rates[agent_idx];

    // Skip if no source
    if (source_rate == 0.0f) return;

    // Unit conversion for implicit BioFVM solver:
    // Store RATES (concentration/time), NOT integrated over dt
    // For SOURCES (positive rates): amount/cell/time → divide by voxel_volume to get concentration/time
    // For UPTAKES (negative rates): magnitude as concentration/time
    // dt is applied in RHS construction, not here
    // Implicit solve: (I + dt*λ + dt*U - dt*D*∇²)C_new = C + dt*S

    // NEW BIOFVM APPROACH: Separate sources from uptakes
    // Sources (positive) go to d_sources array (for RHS of PDE)
    // Uptakes (negative) go to d_uptakes array (for diagonal coefficient)

    // Compute flat index
    int voxel_idx = z * (nx * ny) + y * nx + x;

    // NOTE: d_sources and d_uptakes are already offset to this substrate's data
    // by get_device_source_ptr() and get_device_uptake_ptr(), so just use voxel_idx

    if (source_rate > 0.0f) {
        // Release: amount/cell/time → divide by voxel_volume to get concentration/time (rate, not integrated)
        float source_contribution = source_rate / voxel_volume;
        atomicAdd(&d_sources[voxel_idx], source_contribution);
    } else {
        // Uptake: positive rate (magnitude), stored as concentration/time rate
        float uptake_magnitude = fabsf(source_rate);
        atomicAdd(&d_uptakes[voxel_idx], uptake_magnitude);
    }
}

__global__ void add_source_kernel(
    float* __restrict__ sources,
    int idx,
    float value)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        sources[idx] += value;
    }
}

// ============================================================================
// Diagonal Preconditioner Kernels
// ============================================================================

// Compute diagonal preconditioner M^{-1} for operator A = I + dt*λ - dt*D*∇²
// Diagonal: M[i] = 1 + dt*λ + 6*dt*D/dx² (for interior points, approx for boundary)
// Store inverse: M^{-1}[i] = 1 / M[i]
__global__ void compute_diagonal_preconditioner_inv(
    float* __restrict__ M_inv,
    float avg_uptake,  // Average uptake coefficient for this substrate
    float D, float lambda, float dt, float dx,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float dx2 = dx * dx;
    float diag = 1.0f + dt * (lambda + avg_uptake) + 6.0f * dt * D / dx2;
    // Clamp to ensure positive and finite
    if (!isfinite(diag) || diag <= 0.0f) diag = 1.0f;
    M_inv[i] = 1.0f / (diag + 1e-30f);
}

// Apply diagonal preconditioner: z = M^{-1} * r (element-wise multiplication)
__global__ void apply_diagonal_preconditioner(
    const float* __restrict__ M_inv,
    const float* __restrict__ r,
    float* __restrict__ z,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    z[i] = M_inv[i] * r[i];
}

// Compute average uptake from uptake array for preconditioner
__global__ void compute_average_uptake_kernel(
    const float* __restrict__ uptakes,
    float* __restrict__ partial_sums,
    int n)
{
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    if (idx < n) {
        sum = uptakes[idx];
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
// Multigrid Kernels
// ============================================================================

// Restrict residual from fine grid to coarse grid using full-weighting
// Coarse grid has dimensions (nx_fine+1)/2 × (ny_fine+1)/2 × (nz_fine+1)/2
__global__ void restrict_residual(
    const float* __restrict__ fine,
    float* __restrict__ coarse,
    int nx_fine, int ny_fine, int nz_fine)
{
    int ix_c = blockIdx.x * blockDim.x + threadIdx.x;
    int iy_c = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_c = blockIdx.z * blockDim.z + threadIdx.z;

    int nx_coarse = (nx_fine + 1) / 2;
    int ny_coarse = (ny_fine + 1) / 2;
    int nz_coarse = (nz_fine + 1) / 2;

    if (ix_c >= nx_coarse || iy_c >= ny_coarse || iz_c >= nz_coarse) return;

    // Map coarse grid point to fine grid
    int ix_f = 2 * ix_c;
    int iy_f = 2 * iy_c;
    int iz_f = 2 * iz_c;

    // Full-weighting: average 8 fine grid points (or fewer at boundaries)
    float sum = 0.0f;
    int count = 0;

    for (int dz = 0; dz <= 1 && iz_f + dz < nz_fine; dz++) {
        for (int dy = 0; dy <= 1 && iy_f + dy < ny_fine; dy++) {
            for (int dx = 0; dx <= 1 && ix_f + dx < nx_fine; dx++) {
                int idx_f = (iz_f + dz) * (nx_fine * ny_fine) + (iy_f + dy) * nx_fine + (ix_f + dx);
                sum += fine[idx_f];
                count++;
            }
        }
    }

    int idx_c = iz_c * (nx_coarse * ny_coarse) + iy_c * nx_coarse + ix_c;
    coarse[idx_c] = sum / count;  // Average
}

// Prolong correction from coarse grid to fine grid using trilinear interpolation
__global__ void prolong_correction(
    const float* __restrict__ coarse,
    float* __restrict__ fine,
    int nx_fine, int ny_fine, int nz_fine)
{
    int ix_f = blockIdx.x * blockDim.x + threadIdx.x;
    int iy_f = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_f = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix_f >= nx_fine || iy_f >= ny_fine || iz_f >= nz_fine) return;

    int nx_coarse = (nx_fine + 1) / 2;
    int ny_coarse = (ny_fine + 1) / 2;
    int nz_coarse = (nz_fine + 1) / 2;

    // Map fine grid point to coarse grid
    int ix_c = ix_f / 2;
    int iy_c = iy_f / 2;
    int iz_c = iz_f / 2;

    // For simplicity, use piecewise constant interpolation (inject)
    // This is the simplest prolongation and works well for diffusion
    int idx_c = iz_c * (nx_coarse * ny_coarse) + iy_c * nx_coarse + ix_c;
    int idx_f = iz_f * (nx_fine * ny_fine) + iy_f * nx_fine + ix_f;

    fine[idx_f] += coarse[idx_c];  // Add correction
}

// Weighted Jacobi smoother for multigrid
__global__ void weighted_jacobi_kernel(
    const float* __restrict__ x_old,
    const float* __restrict__ rhs,
    float* __restrict__ x_new,
    const float* __restrict__ uptakes,
    int nx, int ny, int nz,
    float D, float lambda, float dt, float dx, float omega)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= nx || iy >= ny || iz >= nz) return;

    int idx = iz * (nx * ny) + iy * nx + ix;
    float x_center = x_old[idx];
    float uptake = uptakes[idx];

    // Compute Laplacian using 7-point stencil
    float laplacian = 0.0f;
    float dx2 = dx * dx;
    int num_neighbors = 0;

    if (ix > 0) { laplacian += x_old[idx - 1]; num_neighbors++; }
    if (ix < nx - 1) { laplacian += x_old[idx + 1]; num_neighbors++; }
    if (iy > 0) { laplacian += x_old[idx - nx]; num_neighbors++; }
    if (iy < ny - 1) { laplacian += x_old[idx + nx]; num_neighbors++; }
    if (iz > 0) { laplacian += x_old[idx - nx * ny]; num_neighbors++; }
    if (iz < nz - 1) { laplacian += x_old[idx + nx * ny]; num_neighbors++; }

    laplacian = (laplacian - num_neighbors * x_center) / dx2;

    // Apply operator: A*x = (I + dt*λ - dt*D*∇²)x
    // Note: Uptakes handled via average in preconditioner, not per-voxel
    float Ax = x_center + dt * lambda * x_center - dt * D * laplacian;

    // Jacobi update: x_new = x_old + omega * (rhs - Ax) / diag(A)
    // Diagonal includes avg uptake from preconditioner
    float diagonal = 1.0f + dt * lambda + 6.0f * dt * D / dx2;
    float residual = rhs[idx] - Ax;
    x_new[idx] = x_old[idx] + omega * residual / diagonal;
}

// ============================================================================
// Conjugate Gradient Solver Kernels (Implicit Method)
// ============================================================================

// Apply the implicit diffusion-decay operator: A*x = (I + dt*λ - dt*D*∇²)x
__global__ void apply_diffusion_operator(
    const float* __restrict__ x,
    float* __restrict__ Ax,
    const float* __restrict__ uptakes,  // NEW: uptake rates per voxel
    int nx, int ny, int nz,
    float D, float lambda, float dt, float dx)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= nx || iy >= ny || iz >= nz) return;

    int idx = iz * (nx * ny) + iy * nx + ix;
    float x_center = x[idx];
    float uptake = uptakes[idx];  // NEW: get local uptake rate

    // Compute Laplacian using 7-point stencil
    float laplacian = 0.0f;
    float dx2 = dx * dx;

    // X-direction
    if (ix > 0) {
        laplacian += (x[idx - 1] - x_center) / dx2;
    }
    if (ix < nx - 1) {
        laplacian += (x[idx + 1] - x_center) / dx2;
    }

    // Y-direction
    if (iy > 0) {
        laplacian += (x[idx - nx] - x_center) / dx2;
    }
    if (iy < ny - 1) {
        laplacian += (x[idx + nx] - x_center) / dx2;
    }

    // Z-direction
    if (iz > 0) {
        laplacian += (x[idx - nx * ny] - x_center) / dx2;
    }
    if (iz < nz - 1) {
        laplacian += (x[idx + nx * ny] - x_center) / dx2;
    }

    // Apply operator: A*x = (1 + dt*λ + dt*U)*x - dt*D*∇²x
    // Uptakes are per-voxel and must be included in the main operator
    Ax[idx] = x_center + dt * lambda * x_center + dt * uptake * x_center - dt * D * laplacian;
}

// Vector addition: y = y + alpha*x
__global__ void vector_axpy(
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

// Vector scaling: y = alpha*x
__global__ void vector_scale(
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

// Vector copy: dst = src
__global__ void vector_copy(
    float* __restrict__ dst,
    const float* __restrict__ src,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// Dot product kernel (partial reduction)
__global__ void dot_product_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ partial_sums,
    int n)
{
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load and compute partial dot product
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

    // Write block result
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// Clamp negative values to zero
__global__ void clamp_nonnegative(float* __restrict__ x, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && x[idx] < 0.0f) {
        x[idx] = 0.0f;
    }
}

// ============================================================================
// PDESolver Implementation
// ============================================================================

PDESolver::PDESolver(const PDEConfig& config)
    : config_(config),
      d_concentrations_current_(nullptr),
      d_concentrations_next_(nullptr),
      d_sources_(nullptr),
      d_recruitment_sources_(nullptr),
      d_cg_r_(nullptr),
      d_cg_p_(nullptr),
      d_cg_Ap_(nullptr),
      d_cg_temp_(nullptr),
      d_dot_buffer_(nullptr),
      cg_reduction_blocks_(0),
      h_temp_buffer_(nullptr)
{
    // Validate config
    if (config_.nx <= 0 || config_.ny <= 0 || config_.nz <= 0) {
        throw std::runtime_error("Invalid grid dimensions");
    }
    if (config_.num_substrates <= 0 || config_.num_substrates > NUM_SUBSTRATES) {
        throw std::runtime_error("Invalid number of substrates");
    }
}

PDESolver::~PDESolver() {
    if (d_concentrations_current_) CUDA_CHECK(cudaFree(d_concentrations_current_));
    if (d_concentrations_next_) CUDA_CHECK(cudaFree(d_concentrations_next_));
    if (d_sources_) CUDA_CHECK(cudaFree(d_sources_));
    if (d_uptakes_) CUDA_CHECK(cudaFree(d_uptakes_));  // NEW: deallocate uptakes
    if (d_recruitment_sources_) CUDA_CHECK(cudaFree(d_recruitment_sources_));
    if (d_cg_r_) CUDA_CHECK(cudaFree(d_cg_r_));
    if (d_cg_p_) CUDA_CHECK(cudaFree(d_cg_p_));
    if (d_cg_Ap_) CUDA_CHECK(cudaFree(d_cg_Ap_));
    if (d_cg_z_) CUDA_CHECK(cudaFree(d_cg_z_));
    if (d_cg_temp_) CUDA_CHECK(cudaFree(d_cg_temp_));
    if (d_precond_diag_inv_) CUDA_CHECK(cudaFree(d_precond_diag_inv_));
    if (d_dot_buffer_) CUDA_CHECK(cudaFree(d_dot_buffer_));
    if (d_mg_residual_) CUDA_CHECK(cudaFree(d_mg_residual_));
    if (d_mg_correction_) CUDA_CHECK(cudaFree(d_mg_correction_));
    if (d_mg_coarse_) CUDA_CHECK(cudaFree(d_mg_coarse_));
    if (d_mg_coarse_rhs_) CUDA_CHECK(cudaFree(d_mg_coarse_rhs_));

    if (h_temp_buffer_) delete[] h_temp_buffer_;
}

void PDESolver::initialize() {
    int total_voxels = config_.nx * config_.ny * config_.nz;
    size_t total_size = total_voxels * config_.num_substrates * sizeof(float);
    size_t voxel_size = total_voxels * sizeof(float);

    // Allocate device memory for concentration fields
    CUDA_CHECK(cudaMalloc(&d_concentrations_current_, total_size));
    CUDA_CHECK(cudaMalloc(&d_concentrations_next_, total_size));
    CUDA_CHECK(cudaMalloc(&d_sources_, total_size));
    CUDA_CHECK(cudaMalloc(&d_uptakes_, total_size));  // NEW: uptake rates

    // Allocate recruitment sources array (int per voxel)
    size_t recruitment_size = total_voxels * sizeof(int);
    CUDA_CHECK(cudaMalloc(&d_recruitment_sources_, recruitment_size));

    // Initialize to zero
    CUDA_CHECK(cudaMemset(d_concentrations_current_, 0, total_size));
    CUDA_CHECK(cudaMemset(d_concentrations_next_, 0, total_size));
    CUDA_CHECK(cudaMemset(d_sources_, 0, total_size));
    CUDA_CHECK(cudaMemset(d_uptakes_, 0, total_size));  // NEW: initialize uptakes
    CUDA_CHECK(cudaMemset(d_recruitment_sources_, 0, recruitment_size));

    // Allocate CG workspace (per voxel, not per substrate)
    CUDA_CHECK(cudaMalloc(&d_cg_r_, voxel_size));
    CUDA_CHECK(cudaMalloc(&d_cg_p_, voxel_size));
    CUDA_CHECK(cudaMalloc(&d_cg_Ap_, voxel_size));
    CUDA_CHECK(cudaMalloc(&d_cg_z_, voxel_size));  // Preconditioned residual
    CUDA_CHECK(cudaMalloc(&d_cg_temp_, voxel_size));
    CUDA_CHECK(cudaMalloc(&d_precond_diag_inv_, voxel_size));  // Diagonal preconditioner (inverse)

    // Allocate reduction buffer for dot products
    int threads_per_block = 256;
    cg_reduction_blocks_ = (total_voxels + threads_per_block - 1) / threads_per_block;
    CUDA_CHECK(cudaMalloc(&d_dot_buffer_, cg_reduction_blocks_ * sizeof(float)));

    // Allocate multigrid workspace
    mg_coarse_nx_ = (config_.nx + 1) / 2;  // Coarsen by factor of 2
    mg_coarse_ny_ = (config_.ny + 1) / 2;
    mg_coarse_nz_ = (config_.nz + 1) / 2;
    int coarse_voxels = mg_coarse_nx_ * mg_coarse_ny_ * mg_coarse_nz_;
    size_t coarse_size = coarse_voxels * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_mg_residual_, voxel_size));
    CUDA_CHECK(cudaMalloc(&d_mg_correction_, voxel_size));
    CUDA_CHECK(cudaMalloc(&d_mg_coarse_, coarse_size));
    CUDA_CHECK(cudaMalloc(&d_mg_coarse_rhs_, coarse_size));

    // Allocate host buffer for transfers
    h_temp_buffer_ = new float[total_voxels];

    float cg_workspace_mb = 7.0f * voxel_size / (1024.0f * 1024.0f);
    float mg_workspace_mb = (2.0f * voxel_size + 2.0f * coarse_size) / (1024.0f * 1024.0f);
    std::cout << "PDE Solver initialized (Multigrid + PCG):" << std::endl;
    std::cout << "  Fine grid: " << config_.nx << "x" << config_.ny << "x" << config_.nz << std::endl;
    std::cout << "  Coarse grid: " << mg_coarse_nx_ << "x" << mg_coarse_ny_ << "x" << mg_coarse_nz_ << std::endl;
    std::cout << "  Substrates: " << config_.num_substrates << std::endl;
    std::cout << "  Concentration memory: " << (3 * total_size) / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "  CG workspace: " << cg_workspace_mb << " MB" << std::endl;
    std::cout << "  Multigrid workspace: " << mg_workspace_mb << " MB" << std::endl;
    std::cout << "  PDE timestep: " << config_.dt_pde << " s" << std::endl;
    std::cout << "  Substeps per ABM step: " << config_.substeps_per_abm << std::endl;
}

// Solve implicit system using Preconditioned Conjugate Gradient (diagonal preconditioner)
// A*x = b where A = (I + dt*(λ+U) - dt*D*∇²) with U = uptake rate
// Returns: number of iterations taken (for diagnostics)
int PDESolver::solve_implicit_cg(float* d_C, const float* d_rhs, const float* d_uptakes_per_voxel, float D, float lambda, float dt, float dx) {
    const int n = config_.nx * config_.ny * config_.nz;
    const int max_iters = 500;  // Increased to test convergence on 50^3 grid
    const float tolerance = 1e-4f;

    // Compute actual average uptake from uptake array for preconditioner
    // This ensures preconditioner diagonal matches operator diagonal on average
    float avg_uptake = 1e-4f;  // Default fallback

    int threads_1d = 256;
    int reduction_blocks = (n + threads_1d - 1) / threads_1d;

    // Compute partial sums (partial_sums_uptake_ buffer)
    compute_average_uptake_kernel<<<reduction_blocks, threads_1d>>>(
        d_uptakes_per_voxel, d_dot_buffer_, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Reduce partial sums on CPU
    std::vector<float> h_partial(reduction_blocks);
    CUDA_CHECK(cudaMemcpy(h_partial.data(), d_dot_buffer_,
                          reduction_blocks * sizeof(float), cudaMemcpyDeviceToHost));
    float total_uptake = 0.0f;
    for (int i = 0; i < reduction_blocks; i++) {
        total_uptake += h_partial[i];
    }
    avg_uptake = total_uptake / static_cast<float>(n);

    // Clamp to reasonable range to avoid numerical issues
    avg_uptake = std::max(1e-8f, std::min(avg_uptake, 1e-2f));

    // CUDA grid configuration
    dim3 block_3d(8, 8, 8);
    dim3 grid_3d(
        (config_.nx + block_3d.x - 1) / block_3d.x,
        (config_.ny + block_3d.y - 1) / block_3d.y,
        (config_.nz + block_3d.z - 1) / block_3d.z
    );

    int blocks_1d = reduction_blocks;

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

    // Step 1: Compute diagonal preconditioner M^{-1} (once per substrate)
    compute_diagonal_preconditioner_inv<<<blocks_1d, threads_1d>>>(
        d_precond_diag_inv_, avg_uptake, D, lambda, dt, dx, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 2: Initialize residual r = b - A*x0
    apply_diffusion_operator<<<grid_3d, block_3d>>>(d_C, d_cg_Ap_,
                                                     d_uptakes_per_voxel,
                                                     config_.nx, config_.ny, config_.nz,
                                                     D, lambda, dt, dx);
    CUDA_CHECK(cudaDeviceSynchronize());

    vector_copy<<<blocks_1d, threads_1d>>>(d_cg_r_, d_rhs, n);
    vector_axpy<<<blocks_1d, threads_1d>>>(d_cg_r_, d_cg_Ap_, -1.0f, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 3: Apply preconditioner z = M^{-1} * r
    apply_diagonal_preconditioner<<<blocks_1d, threads_1d>>>(
        d_precond_diag_inv_, d_cg_r_, d_cg_z_, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 4: Initialize search direction p = z
    vector_copy<<<blocks_1d, threads_1d>>>(d_cg_p_, d_cg_z_, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 5: Compute initial r·z (for alpha calculation)
    float rzold = dot_product(d_cg_r_, d_cg_z_);
    float rs_initial = dot_product(d_cg_r_, d_cg_r_);  // For residual norm

    // Preconditioned CG iteration
    int iter;
    float final_residual_norm = 0.0f;

    for (iter = 0; iter < max_iters; iter++) {
        // Ap = A*p
        apply_diffusion_operator<<<grid_3d, block_3d>>>(d_cg_p_, d_cg_Ap_,
                                                         d_uptakes_per_voxel,
                                                         config_.nx, config_.ny, config_.nz,
                                                         D, lambda, dt, dx);
        CUDA_CHECK(cudaDeviceSynchronize());

        // alpha = (r·z) / (p·Ap)  ← KEY: use r·z not r·r!
        float pAp = dot_product(d_cg_p_, d_cg_Ap_);
        float alpha = rzold / (pAp + 1e-30f);

        // x = x + alpha*p
        vector_axpy<<<blocks_1d, threads_1d>>>(d_C, d_cg_p_, alpha, n);

        // r = r - alpha*Ap
        vector_axpy<<<blocks_1d, threads_1d>>>(d_cg_r_, d_cg_Ap_, -alpha, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Check convergence (use actual residual norm)
        float rsnew = dot_product(d_cg_r_, d_cg_r_);
        final_residual_norm = sqrtf(rsnew / (rs_initial + 1e-30f));

        if (final_residual_norm < tolerance) {
            iter++;  // Count this iteration
            break;
        }

        // z = M^{-1} * r (apply preconditioner)
        apply_diagonal_preconditioner<<<blocks_1d, threads_1d>>>(
            d_precond_diag_inv_, d_cg_r_, d_cg_z_, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Compute new r·z
        float rznew = dot_product(d_cg_r_, d_cg_z_);

        // beta = rznew / rzold  ← KEY: use r·z not r·r!
        float beta = rznew / (rzold + 1e-30f);

        // p = z + beta*p  ← KEY: use z not r!
        vector_scale<<<blocks_1d, threads_1d>>>(d_cg_temp_, d_cg_p_, beta, n);
        vector_copy<<<blocks_1d, threads_1d>>>(d_cg_p_, d_cg_z_, n);
        vector_axpy<<<blocks_1d, threads_1d>>>(d_cg_p_, d_cg_temp_, 1.0f, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        rzold = rznew;
    }

    // Store final residual in a member variable for diagnostics
    last_residual_norm_ = final_residual_norm;

    // Ensure non-negative concentrations
    clamp_nonnegative<<<blocks_1d, threads_1d>>>(d_C, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    return iter;  // Return iteration count for diagnostics
}

// ============================================================================
// Multigrid Solver (2-Level V-Cycle)
// ============================================================================

int PDESolver::solve_multigrid(float* d_C, const float* d_rhs, const float* d_uptakes_per_voxel, float D, float lambda, float dt, float dx) {
    const int nx = config_.nx;
    const int ny = config_.ny;
    const int nz = config_.nz;
    const int n_fine = nx * ny * nz;
    const int n_coarse = mg_coarse_nx_ * mg_coarse_ny_ * mg_coarse_nz_;

    // Special handling for zero-decay substrates (like O2)
    const bool is_zero_decay = (lambda < 1e-10f);
    const int max_cycles = is_zero_decay ? 50 : 20;  // More cycles for stiff problems
    const float tolerance = is_zero_decay ? 5e-4f : 1e-4f;  // Relaxed tolerance for O2
    const int pre_smooth = is_zero_decay ? 5 : 3;   // More smoothing for stiff problems
    const int post_smooth = is_zero_decay ? 5 : 3;
    const float omega = 0.67f;  // Jacobi relaxation parameter

    // Grid configuration
    dim3 block_fine(8, 8, 8);
    dim3 grid_fine(
        (nx + block_fine.x - 1) / block_fine.x,
        (ny + block_fine.y - 1) / block_fine.y,
        (nz + block_fine.z - 1) / block_fine.z
    );

    dim3 block_coarse(8, 8, 8);
    dim3 grid_coarse(
        (mg_coarse_nx_ + block_coarse.x - 1) / block_coarse.x,
        (mg_coarse_ny_ + block_coarse.y - 1) / block_coarse.y,
        (mg_coarse_nz_ + block_coarse.z - 1) / block_coarse.z
    );

    int threads_1d = 256;
    int blocks_fine_1d = (n_fine + threads_1d - 1) / threads_1d;
    int blocks_coarse_1d = (n_coarse + threads_1d - 1) / threads_1d;

    // Helper for computing residual norm
    auto compute_residual_norm = [&](const float* d_x, const float* d_b) -> float {
        // Compute residual: r = b - A*x
        apply_diffusion_operator<<<grid_fine, block_fine>>>(d_x, d_mg_residual_,
                                                            d_cg_temp_,  // Use temp for zero uptakes in multigrid
                                                            nx, ny, nz, D, lambda, dt, dx);
        CUDA_CHECK(cudaDeviceSynchronize());

        vector_copy<<<blocks_fine_1d, threads_1d>>>(d_cg_temp_, d_b, n_fine);
        vector_axpy<<<blocks_fine_1d, threads_1d>>>(d_cg_temp_, d_mg_residual_, -1.0f, n_fine);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Compute norm
        dot_product_kernel<<<cg_reduction_blocks_, threads_1d>>>(d_cg_temp_, d_cg_temp_, d_dot_buffer_, n_fine);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> h_partial(cg_reduction_blocks_);
        CUDA_CHECK(cudaMemcpy(h_partial.data(), d_dot_buffer_,
                              cg_reduction_blocks_ * sizeof(float), cudaMemcpyDeviceToHost));
        float sum = 0.0f;
        for (int i = 0; i < cg_reduction_blocks_; i++) {
            sum += h_partial[i];
        }
        return sqrtf(sum) / sqrtf(n_fine);  // Normalized
    };

    // Initial residual
    float initial_residual = compute_residual_norm(d_C, d_rhs);

    // V-cycle iterations
    int cycle;
    for (cycle = 0; cycle < max_cycles; cycle++) {
        // 1. Pre-smoothing on fine grid
        mg_smooth(d_C, d_rhs, d_uptakes_per_voxel, D, lambda, dt, dx, nx, ny, nz, pre_smooth, omega);

        // 2. Compute residual: r = b - A*x
        apply_diffusion_operator<<<grid_fine, block_fine>>>(d_C, d_mg_residual_,
                                                            d_uptakes_per_voxel,
                                                            nx, ny, nz, D, lambda, dt, dx);
        CUDA_CHECK(cudaDeviceSynchronize());

        vector_copy<<<blocks_fine_1d, threads_1d>>>(d_cg_temp_, d_rhs, n_fine);
        vector_axpy<<<blocks_fine_1d, threads_1d>>>(d_cg_temp_, d_mg_residual_, -1.0f, n_fine);
        vector_copy<<<blocks_fine_1d, threads_1d>>>(d_mg_residual_, d_cg_temp_, n_fine);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 3. Restrict residual to coarse grid
        restrict_residual<<<grid_coarse, block_coarse>>>(d_mg_residual_, d_mg_coarse_rhs_,
                                                         nx, ny, nz);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 4. Solve on coarse grid (using smoothing iterations)
        CUDA_CHECK(cudaMemset(d_mg_coarse_, 0, n_coarse * sizeof(float)));  // Initial guess = 0
        float dx_coarse = 2.0f * dx;  // Coarse grid spacing

        // Smooth on coarse grid instead of exact solve
        // Use d_cg_temp_ as zero uptakes for coarse grid (uptakes not properly restricted)
        CUDA_CHECK(cudaMemset(d_cg_temp_, 0, n_coarse * sizeof(float)));
        mg_smooth(d_mg_coarse_, d_mg_coarse_rhs_, d_cg_temp_, D, lambda, dt, dx_coarse,
                  mg_coarse_nx_, mg_coarse_ny_, mg_coarse_nz_, 10, omega);

        // 5. Prolong correction to fine grid
        CUDA_CHECK(cudaMemset(d_mg_correction_, 0, n_fine * sizeof(float)));
        prolong_correction<<<grid_fine, block_fine>>>(d_mg_coarse_, d_mg_correction_,
                                                      nx, ny, nz);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 6. Update solution: x = x + correction
        vector_axpy<<<blocks_fine_1d, threads_1d>>>(d_C, d_mg_correction_, 1.0f, n_fine);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 7. Post-smoothing on fine grid
        mg_smooth(d_C, d_rhs, d_uptakes_per_voxel, D, lambda, dt, dx, nx, ny, nz, post_smooth, omega);

        // Check convergence
        float residual_norm = compute_residual_norm(d_C, d_rhs);
        last_residual_norm_ = residual_norm;

        if (residual_norm < tolerance) {
            cycle++;  // Count this cycle
            break;
        }
    }

    // Ensure non-negative concentrations
    clamp_nonnegative<<<blocks_fine_1d, threads_1d>>>(d_C, n_fine);
    CUDA_CHECK(cudaDeviceSynchronize());

    return cycle;  // Return number of V-cycles
}

// Weighted Jacobi smoother
void PDESolver::mg_smooth(float* d_x, const float* d_rhs, const float* d_uptakes, float D, float lambda, float dt, float dx,
                          int nx, int ny, int nz, int num_iters, float omega) {
    dim3 block(8, 8, 8);
    dim3 grid(
        (nx + block.x - 1) / block.x,
        (ny + block.y - 1) / block.y,
        (nz + block.z - 1) / block.z
    );

    // Use d_cg_temp_ as temporary buffer for ping-pong
    for (int iter = 0; iter < num_iters; iter++) {
        weighted_jacobi_kernel<<<grid, block>>>(d_x, d_rhs, d_cg_temp_, d_uptakes, nx, ny, nz, D, lambda, dt, dx, omega);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy result back
        int n = nx * ny * nz;
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        vector_copy<<<blocks, threads>>>(d_x, d_cg_temp_, n);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

void PDESolver::solve_timestep() {
    int n = config_.nx * config_.ny * config_.nz;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    static int step_count = 0;
    bool print_debug = false;
    step_count++;

    // Diagnostics: track convergence and timing
    std::vector<int> substrate_iters(config_.num_substrates);
    std::vector<cudaEvent_t> starts(config_.num_substrates);
    std::vector<cudaEvent_t> stops(config_.num_substrates);

    const char* substrate_names[NUM_SUBSTRATES] = {
        "O2", "IFNg", "IL2", "IL10", "TGFB", "CCL2", "ARGI", "NO", "IL12", "VEGFA"
    };

    // Sequential solve for each substrate
    for (int sub = 0; sub < config_.num_substrates; sub++) {
        // Create named NVTX range for this substrate
        std::string range_name = std::string("PDE ") + substrate_names[sub];
        nvtxRangePush(range_name.c_str());

        float D = config_.diffusion_coeffs[sub];
        float lambda = config_.decay_rates[sub];
        float* C_curr = d_concentrations_current_ + sub * n;
        float* sources = d_sources_ + sub * n;
        float* uptakes = d_uptakes_ + sub * n;  // NEW: get uptakes for this substrate

        // Build RHS: b = C^n + dt*S (sources are rates, apply dt here)
        vector_copy<<<blocks, threads>>>(d_cg_temp_, C_curr, n);
        vector_axpy<<<blocks, threads>>>(d_cg_temp_, sources, config_.dt_abm, n);  // Apply dt to sources
        CUDA_CHECK(cudaDeviceSynchronize());

        // Create timing events
        cudaEventCreate(&starts[sub]);
        cudaEventCreate(&stops[sub]);
        cudaEventRecord(starts[sub]);

        // Use hybrid approach: multigrid for most, PCG for zero-decay (O2)
        bool use_pcg = (lambda < 1e-10f);
        if (use_pcg) {
            substrate_iters[sub] = solve_implicit_cg(C_curr, d_cg_temp_, uptakes, D, lambda,
                                                     config_.dt_abm, config_.voxel_size);
        } else {
            substrate_iters[sub] = solve_multigrid(C_curr, d_cg_temp_, uptakes, D, lambda,
                                                   config_.dt_abm, config_.voxel_size);
        }

        cudaEventRecord(stops[sub]);
        nvtxRangePop();
    }

    // Collect diagnostics
    int total_iters = 0;
    int max_iters_seen = 0;
    double total_time = 0.0;

    for (int sub = 0; sub < config_.num_substrates; sub++) {
        float D = config_.diffusion_coeffs[sub];
        float lambda = config_.decay_rates[sub];
        int iters = substrate_iters[sub];

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, starts[sub], stops[sub]);
        cudaEventDestroy(starts[sub]);
        cudaEventDestroy(stops[sub]);

        total_iters += iters;
        max_iters_seen = (iters > max_iters_seen) ? iters : max_iters_seen;
        total_time += milliseconds;

        if (print_debug) {
            bool use_pcg = (lambda < 1e-10f);
            const char* solver_name = use_pcg ? "PCG" : "MG";
            const char* iter_unit = use_pcg ? "iters" : "V-cycles";
            printf("[%s] Step %d, %s (sub %d): %d %s, %.2f ms (D=%.2e, λ=%.2e)\n",
                   solver_name, step_count - 1, substrate_names[sub], sub, iters, iter_unit, milliseconds, D, lambda);
        }
    }

    // Print summary diagnostics
    if (print_debug) {
        printf("[Sequential Summary] Step %d: total_iters=%d, avg_iters=%.1f, max_iters=%d, total_time=%.2f ms\n\n",
               step_count - 1, total_iters, total_iters / (float)config_.num_substrates,
               max_iters_seen, total_time);
    }
}

void PDESolver::set_sources(const float* h_sources, int substrate_idx) {
    if (substrate_idx < 0 || substrate_idx >= config_.num_substrates) {
        throw std::runtime_error("Invalid substrate index");
    }
    
    int voxels = config_.nx * config_.ny * config_.nz;
    size_t offset = substrate_idx * voxels * sizeof(float);
    
    CUDA_CHECK(cudaMemcpy(
        d_sources_ + substrate_idx * voxels,
        h_sources,
        voxels * sizeof(float),
        cudaMemcpyHostToDevice
    ));
}

void PDESolver::add_source_at_voxel(int x, int y, int z, int substrate_idx, float value) {
    if (x < 0 || x >= config_.nx || y < 0 || y >= config_.ny || z < 0 || z >= config_.nz) {
        return; // Out of bounds
    }
    
    int voxel_idx = idx(x, y, z);
    int offset = substrate_idx * get_total_voxels() + voxel_idx;
    
    // Atomic add on device (launch simple kernel)
    add_source_kernel<<<1, 1>>>(d_sources_, offset, value);
    CUDA_CHECK(cudaGetLastError());
}

void PDESolver::get_concentrations(float* h_concentrations, int substrate_idx) const {
    if (substrate_idx < 0 || substrate_idx >= config_.num_substrates) {
        throw std::runtime_error("Invalid substrate index");
    }
    
    int voxels = config_.nx * config_.ny * config_.nz;
    
    CUDA_CHECK(cudaMemcpy(
        h_concentrations,
        d_concentrations_current_ + substrate_idx * voxels,
        voxels * sizeof(float),
        cudaMemcpyDeviceToHost
    ));
}

float PDESolver::get_concentration_at_voxel(int x, int y, int z, int substrate_idx) const {
    if (x < 0 || x >= config_.nx || y < 0 || y >= config_.ny || z < 0 || z >= config_.nz) {
        return 0.0f;
    }
    
    int voxel_idx = idx(x, y, z);
    int offset = substrate_idx * get_total_voxels() + voxel_idx;
    
    float value;
    CUDA_CHECK(cudaMemcpy(
        &value,
        d_concentrations_current_ + offset,
        sizeof(float),
        cudaMemcpyDeviceToHost
    ));
    
    return value;
}

float* PDESolver::get_device_concentration_ptr(int substrate_idx) {
    if (substrate_idx < 0 || substrate_idx >= config_.num_substrates) {
        return nullptr;
    }
    return d_concentrations_current_ + substrate_idx * get_total_voxels();
}

float* PDESolver::get_device_source_ptr(int substrate_idx) {
    if (substrate_idx < 0 || substrate_idx >= config_.num_substrates) {
        return nullptr;
    }
    return d_sources_ + substrate_idx * get_total_voxels();
}

float* PDESolver::get_device_uptake_ptr(int substrate_idx) {
    if (substrate_idx < 0 || substrate_idx >= config_.num_substrates) {
        return nullptr;
    }
    return d_uptakes_ + substrate_idx * get_total_voxels();
}

void PDESolver::reset_concentrations() {
    int total_voxels = get_total_voxels();
    size_t total_size = total_voxels * config_.num_substrates * sizeof(float);
    CUDA_CHECK(cudaMemset(d_concentrations_current_, 0, total_size));
    CUDA_CHECK(cudaMemset(d_concentrations_next_, 0, total_size));
}

void PDESolver::reset_sources() {
    int total_voxels = get_total_voxels();
    size_t total_size = total_voxels * config_.num_substrates * sizeof(float);
    CUDA_CHECK(cudaMemset(d_sources_, 0, total_size));
}

void PDESolver::reset_uptakes() {
    int total_voxels = get_total_voxels();
    size_t total_size = total_voxels * config_.num_substrates * sizeof(float);
    CUDA_CHECK(cudaMemset(d_uptakes_, 0, total_size));
}

void PDESolver::reset_recruitment_sources() {
    int total_voxels = get_total_voxels();
    size_t size = total_voxels * sizeof(int);
    CUDA_CHECK(cudaMemset(d_recruitment_sources_, 0, size));
}

void PDESolver::set_initial_concentration(int substrate_idx, float value) {
    if (substrate_idx < 0 || substrate_idx >= config_.num_substrates) {
        throw std::runtime_error("Invalid substrate index");
    }

    int voxels = get_total_voxels();
    std::vector<float> init_values(voxels, value);

    CUDA_CHECK(cudaMemcpy(
        d_concentrations_current_ + substrate_idx * voxels,
        init_values.data(),
        voxels * sizeof(float),
        cudaMemcpyHostToDevice
    ));
}

float PDESolver::get_total_source(int substrate_idx) {
    if (substrate_idx < 0 || substrate_idx >= config_.num_substrates) {
        return 0.0f;
    }

    int voxels = get_total_voxels();
    std::vector<float> h_sources(voxels);

    CUDA_CHECK(cudaMemcpy(
        h_sources.data(),
        d_sources_ + substrate_idx * voxels,
        voxels * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    return std::accumulate(h_sources.begin(), h_sources.end(), 0.0f);
}

void PDESolver::swap_buffers() {
    float* temp = d_concentrations_current_;
    d_concentrations_current_ = d_concentrations_next_;
    d_concentrations_next_ = temp;
}

} // namespace PDAC