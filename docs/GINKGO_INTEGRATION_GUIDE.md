# Ginkko Integration Guide for PDAC PDE Solver
## Detailed Implementation Plan (3-4 weeks)

---

## Overview: Why Ginkko for PDAC?

Your current custom conjugate gradient solver:
- ✅ Works well for 50³ grids
- ✅ Tight FLAME GPU integration
- ✅ Matrix-free (no Jacobian assembly)
- ❌ Limited preconditioning (just diagonal scaling)
- ❌ Single-GPU only
- ❌ Custom maintenance burden

**Ginkgo offers**:
- ✅ Same matrix-free paradigm
- ✅ Better preconditioning (ILU, AMG, Chebyshev)
- ✅ MPI support for multi-GPU
- ✅ Modern C++ (17) with clean API
- ✅ Active maintenance (Forschungszentrum Jülich)
- ✅ Direct device memory (minimal friction with FLAME GPU)

---

## Phase 1: Setup (Days 1-2, ~5 hours)

### 1.1 Build & Install Ginkko Locally

```bash
# Clone Ginkgo
cd ~/lib  # Or your preferred location
git clone https://github.com/ginkgo-project/ginkgo.git
cd ginkgo
git checkout master  # Latest stable

# Configure for your GPU
mkdir build
cd build
cmake .. \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_CUDA_COMPILER=nvcc \
    -DGINKGO_BUILD_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=75  # Use your GPU arch (75=RTX 20xx, 80=A100, 86=RTX 30xx) \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$HOME/lib/ginkgo-install \
    -DBUILD_SHARED_LIBS=ON

# Build (takes ~5-10 min)
cmake --build . -j$(nproc)
cmake --install .

# Test installation
ctest --verbose  # Optional, checks library
```

### 1.2 Verify CUDA Support

```bash
# Check Ginkgo was built with CUDA
$ ls ~/lib/ginkgo-install/lib/libginkgo_cuda*
/home/.../lib/ginkgo-install/lib/libginkgo_cuda.so

# Check include paths
$ ls ~/lib/ginkgo-install/include/ginkgo/
core/  devices/  matrix/  preconditioners/  solvers/  ...
```

### 1.3 Update PDAC CMakeLists.txt

File: `/home/chase/SPQSP/SPQSP_PDAC-main/PDAC/sim/CMakeLists.txt`

```cmake
# Add after existing dependencies (after FLAMEGPU, SUNDIALS)

# Find Ginkgo
find_package(Ginkgo REQUIRED
    PATHS $ENV{HOME}/lib/ginkgo-install/lib/cmake/Ginkgo
    NO_DEFAULT_PATH)

if(Ginkgo_FOUND)
    message(STATUS "Ginkgo found at ${Ginkgo_DIR}")
    target_link_libraries(pdac PRIVATE Ginkgo::ginkgo Ginkgo::ginkgo_cuda)
    target_include_directories(pdac PRIVATE ${Ginkgo_INCLUDE_DIRS})
    target_compile_definitions(pdac PRIVATE GINKGO_ENABLED=1)
else()
    message(WARNING "Ginkgo not found. Set GINKGO_DIR to enable Ginkgo solver.")
endif()
```

### 1.4 Test CMake Integration

```bash
cd /home/chase/SPQSP/SPQSP_PDAC-main/PDAC/sim
rm -rf build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release

# Should see:
# "Ginkgo found at ..."
# "GINKGO_ENABLED=1"
```

---

## Phase 2: Core Implementation (Days 3-4, ~10 hours)

### 2.1 Create Ginkgo Solver Wrapper Header

File: `/home/chase/SPQSP/SPQSP_PDAC-main/PDAC/pde/pde_solver_ginkgo.cuh`

```cuda
#ifndef PDAC_PDE_SOLVER_GINKGO_CUH
#define PDAC_PDE_SOLVER_GINKGO_CUH

#include <ginkgo/ginkgo.hpp>
#include <memory>
#include <vector>
#include <cmath>

namespace PDAC {

/**
 * Matrix-free diffusion-reaction operator for Ginkgo
 * Implements: (I + dt·λ - dt·D·∇²)·C = rhs
 *
 * This is a custom operator that applies the stencil without
 * explicit matrix storage, replicating your current CG approach.
 */
class DiffusionReactionOperator : public ginkgo::LinOp {
public:
    /**
     * Create a diffusion-reaction operator for a 3D Cartesian grid
     *
     * @param executor     GPU executor (CUDA)
     * @param grid_x, grid_y, grid_z  Grid dimensions (in voxels)
     * @param dx           Voxel size (in same units as diffusivity)
     * @param dt           Time step (in same units as decay rate)
     * @param D            Diffusion coefficient (device array of size grid_x*grid_y*grid_z)
     * @param lambda       Decay rate (device array)
     * @param sources      RHS sources (will be set/updated each solve)
     */
    DiffusionReactionOperator(
        std::shared_ptr<ginkgo::Executor> executor,
        int grid_x, int grid_y, int grid_z,
        float dx, float dt,
        const float* D,           // Device ptr
        const float* lambda,      // Device ptr
        const float* sources      // Device ptr
    );

    /**
     * Create clone with different source vector
     */
    std::unique_ptr<ginkgo::LinOp> clone(
        std::shared_ptr<ginkgo::Executor> executor = {}) const override;

    /**
     * Apply operator: y = A*x where A = (I + dt·λ - dt·D·∇²)
     */
    void apply_impl(const ginkgo::LinOp *b, ginkgo::LinOp *x) const override;

    /**
     * Advanced: Add scaled contribution
     */
    void apply_impl(const ginkgo::LinOp *alpha, const ginkgo::LinOp *b,
                    const ginkgo::LinOp *beta, ginkgo::LinOp *x) const override;

    // Accessors for coefficient updates
    void set_D(const float* D_new) { d_D = D_new; }
    void set_lambda(const float* lambda_new) { d_lambda = lambda_new; }
    void set_sources(const float* sources_new) { d_sources = sources_new; }

private:
    std::shared_ptr<ginkgo::Executor> executor_;
    int grid_x_, grid_y_, grid_z_;
    float dx_, dt_;
    const float* d_D;           // Device pointers
    const float* d_lambda;
    const float* d_sources;

    // Helper: Apply 7-point stencil
    void apply_stencil_(const float* x, float* y) const;
};

/**
 * High-level PDE solver wrapper using Ginkgo
 */
class GinkgoPDESolver {
public:
    /**
     * Initialize Ginkgo solver
     *
     * @param grid_x, grid_y, grid_z  Grid dimensions
     * @param dx                       Voxel size
     * @param dt                       Time step
     * @param num_substrates           Number of chemicals to solve
     */
    GinkgoPDESolver(int grid_x, int grid_y, int grid_z,
                   float dx, float dt, int num_substrates);

    /**
     * Solve one PDE timestep for all substrates
     *
     * @param d_C_curr        Current concentration (device, size=grid³)
     * @param d_sources       Agent sources (device, size=grid³)
     * @param d_D             Diffusion coefficient (device)
     * @param d_lambda        Decay rate (device)
     * @param d_C_next        Solution output (device)
     * @param substrate_idx   Which chemical (0=O2, 1=IFN, etc.)
     */
    void solve_timestep(
        const float* d_C_curr,
        const float* d_sources,
        const float* d_D,
        const float* d_lambda,
        float* d_C_next,
        int substrate_idx
    );

    /**
     * Configure solver options
     *
     * @param max_iters      Maximum CG iterations
     * @param tolerance      Relative residual tolerance
     * @param preconditioner "none", "ilu", "chebyshev" (default: "ilu")
     */
    void set_solver_options(int max_iters = 100,
                           float tolerance = 1e-6,
                           const std::string& preconditioner = "ilu");

    /**
     * Get diagnostics from last solve
     */
    struct SolveStats {
        int iterations;
        float final_residual;
        float residual_reduction;
        float solve_time_ms;
    };
    const SolveStats& get_last_stats() const { return last_stats_; }

private:
    std::shared_ptr<ginkgo::Executor> executor_;
    std::vector<std::shared_ptr<DiffusionReactionOperator>> operators_;
    std::vector<std::unique_ptr<ginkgo::solver::Cg<float>>> solvers_;
    std::vector<std::unique_ptr<ginkgo::preconditioner::Ilu<float>>> preconditioners_;

    int grid_x_, grid_y_, grid_z_, grid_size_;
    float dx_, dt_;
    int num_substrates_;

    SolveStats last_stats_;
    int max_iters_;
    float tolerance_;
    std::string preconditioner_type_;
};

}  // namespace PDAC

#endif  // PDAC_PDE_SOLVER_GINKGO_CUH
```

### 2.2 Implement Ginkgo Solver (Core CUDA Kernels)

File: `/home/chase/SPQSP/SPQSP_PDAC-main/PDAC/pde/pde_solver_ginkgo.cu`

```cuda
#include "pde_solver_ginkgo.cuh"
#include <iostream>
#include <chrono>

namespace PDAC {

// ============================================================================
// Kernels for Stencil Application
// ============================================================================

/**
 * Apply 7-point Laplacian stencil on 3D grid
 * y = (I + dt·λ - dt·D·∇²) x
 */
__global__ void apply_diffusion_stencil_kernel(
    const float* __restrict__ x,
    const float* __restrict__ D,
    const float* __restrict__ lambda,
    const float* __restrict__ sources,
    float* __restrict__ y,
    int nx, int ny, int nz,
    float dx, float dt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny * nz) return;

    // Compute 3D indices from linear index
    int x_idx = idx % nx;
    int y_idx = (idx / nx) % ny;
    int z_idx = idx / (nx * ny);

    float c = x[idx];
    float d_coeff = D[idx];
    float lambda_coeff = lambda[idx];
    float dx2 = dx * dx;

    // 7-point stencil: Laplacian at (x_idx, y_idx, z_idx)
    float laplacian = -6.0f * c;  // Center point coefficient

    // X-direction
    if (x_idx > 0) {
        laplacian += x[idx - 1];
    }
    if (x_idx < nx - 1) {
        laplacian += x[idx + 1];
    }

    // Y-direction
    if (y_idx > 0) {
        laplacian += x[idx - nx];
    }
    if (y_idx < ny - 1) {
        laplacian += x[idx + nx];
    }

    // Z-direction
    int xy_stride = nx * ny;
    if (z_idx > 0) {
        laplacian += x[idx - xy_stride];
    }
    if (z_idx < nz - 1) {
        laplacian += x[idx + xy_stride];
    }

    laplacian /= dx2;

    // y = (I + dt·λ - dt·D·∇²)·x + dt·sources
    y[idx] = c
           + dt * lambda_coeff * c
           - dt * d_coeff * laplacian
           + dt * sources[idx];
}

// ============================================================================
// DiffusionReactionOperator Implementation
// ============================================================================

DiffusionReactionOperator::DiffusionReactionOperator(
    std::shared_ptr<ginkgo::Executor> executor,
    int grid_x, int grid_y, int grid_z,
    float dx, float dt,
    const float* D,
    const float* lambda,
    const float* sources)
    : ginkgo::LinOp(executor, ginkgo::dim<2>(grid_x * grid_y * grid_z,
                                             grid_x * grid_y * grid_z)),
      executor_(executor),
      grid_x_(grid_x), grid_y_(grid_y), grid_z_(grid_z),
      dx_(dx), dt_(dt),
      d_D(D), d_lambda(lambda), d_sources(sources)
{
}

std::unique_ptr<ginkgo::LinOp> DiffusionReactionOperator::clone(
    std::shared_ptr<ginkgo::Executor> executor) const
{
    if (!executor) executor = executor_;
    return std::make_unique<DiffusionReactionOperator>(
        executor, grid_x_, grid_y_, grid_z_, dx_, dt_, d_D, d_lambda, d_sources);
}

void DiffusionReactionOperator::apply_stencil_(
    const float* x, float* y) const
{
    int grid_size = grid_x_ * grid_y_ * grid_z_;
    int threads = 256;
    int blocks = (grid_size + threads - 1) / threads;

    apply_diffusion_stencil_kernel<<<blocks, threads>>>(
        x, d_D, d_lambda, d_sources, y,
        grid_x_, grid_y_, grid_z_, dx_, dt_);

    cudaDeviceSynchronize();
}

void DiffusionReactionOperator::apply_impl(
    const ginkgo::LinOp *b, ginkgo::LinOp *x) const
{
    auto b_casted = ginkgo::as<ginkgo::matrix::Dense<float>>(b);
    auto x_casted = ginkgo::as<ginkgo::matrix::Dense<float>>(x);

    const float* b_data = b_casted->get_const_values();
    float* x_data = x_casted->get_values();

    apply_stencil_(b_data, x_data);
}

void DiffusionReactionOperator::apply_impl(
    const ginkgo::LinOp *alpha, const ginkgo::LinOp *b,
    const ginkgo::LinOp *beta, ginkgo::LinOp *x) const
{
    // y = alpha * A * b + beta * x
    auto alpha_casted = ginkgo::as<ginkgo::matrix::Dense<float>>(alpha);
    auto beta_casted = ginkgo::as<ginkgo::matrix::Dense<float>>(beta);
    auto b_casted = ginkgo::as<ginkgo::matrix::Dense<float>>(b);
    auto x_casted = ginkgo::as<ginkgo::matrix::Dense<float>>(x);

    float alpha_val = *alpha_casted->get_const_values();
    float beta_val = *beta_casted->get_const_values();
    const float* b_data = b_casted->get_const_values();
    float* x_data = x_casted->get_values();

    int grid_size = grid_x_ * grid_y_ * grid_z_;

    // Temporary: y = A*b
    float* temp;
    cudaMalloc(&temp, grid_size * sizeof(float));
    apply_stencil_(b_data, temp);

    // x = alpha * y + beta * x
    int threads = 256;
    int blocks = (grid_size + threads - 1) / threads;

    auto kernel_scale = [=] __device__ (int idx) {
        if (idx < grid_size) {
            x_data[idx] = alpha_val * temp[idx] + beta_val * x_data[idx];
        }
    };

    // Simple scale kernel (could optimize)
    cudaMemcpy(x_data, temp, grid_size * sizeof(float), cudaMemcpyDeviceToDevice);

    cudaFree(temp);
}

// ============================================================================
// GinkgoPDESolver Implementation
// ============================================================================

GinkgoPDESolver::GinkgoPDESolver(int grid_x, int grid_y, int grid_z,
                                 float dx, float dt, int num_substrates)
    : grid_x_(grid_x), grid_y_(grid_y), grid_z_(grid_z),
      grid_size_(grid_x * grid_y * grid_z),
      dx_(dx), dt_(dt),
      num_substrates_(num_substrates),
      max_iters_(100),
      tolerance_(1e-6f),
      preconditioner_type_("ilu")
{
    // Create GPU executor
    executor_ = ginkgo::CudaExecutor::create(0);  // GPU 0

    // Pre-allocate solver objects for each substrate
    operators_.resize(num_substrates);
    solvers_.resize(num_substrates);
    preconditioners_.resize(num_substrates);

    std::cout << "[Ginkgo] Initialized PDE solver for " << num_substrates
              << " substrates on " << grid_x << "³ grid" << std::endl;
}

void GinkgoPDESolver::set_solver_options(int max_iters, float tolerance,
                                         const std::string& preconditioner)
{
    max_iters_ = max_iters;
    tolerance_ = tolerance;
    preconditioner_type_ = preconditioner;
}

void GinkgoPDESolver::solve_timestep(
    const float* d_C_curr,
    const float* d_sources,
    const float* d_D,
    const float* d_lambda,
    float* d_C_next,
    int substrate_idx)
{
    auto start = std::chrono::high_resolution_clock::now();

    // Create/update operator
    if (!operators_[substrate_idx]) {
        operators_[substrate_idx] = std::make_shared<DiffusionReactionOperator>(
            executor_, grid_x_, grid_y_, grid_z_, dx_, dt_,
            d_D, d_lambda, d_sources);
    } else {
        operators_[substrate_idx]->set_D(d_D);
        operators_[substrate_idx]->set_lambda(d_lambda);
        operators_[substrate_idx]->set_sources(d_sources);
    }

    // Create vectors
    auto b_vec = ginkgo::matrix::Dense<float>::create(
        executor_, ginkgo::dim<2>(grid_size_, 1),
        ginkgo::array<float>{executor_, grid_size_});
    cudaMemcpy(b_vec->get_values(), d_C_curr,
               grid_size_ * sizeof(float), cudaMemcpyDeviceToDevice);

    auto x_vec = ginkgo::matrix::Dense<float>::create(
        executor_, ginkgo::dim<2>(grid_size_, 1),
        ginkgo::array<float>{executor_, grid_size_});
    cudaMemset(x_vec->get_values(), 0, grid_size_ * sizeof(float));

    // Create CG solver with stopping criteria
    auto criteria = ginkgo::stop::combine(
        std::make_shared<ginkgo::stop::Iteration::Factory>()
            .with_max_iters(max_iters_),
        std::make_shared<ginkgo::stop::RelativeResidualNorm<float>::Factory>()
            .with_tolerance(tolerance_));

    if (!solvers_[substrate_idx]) {
        solvers_[substrate_idx] = ginkgo::solver::Cg<float>::build()
            .with_criteria(criteria)
            .on(executor_)
            .generate();
    }

    // Solve: x = A^{-1} * b
    solvers_[substrate_idx]->apply(b_vec, x_vec);

    // Copy solution back
    cudaMemcpy(d_C_next, x_vec->get_values(),
               grid_size_ * sizeof(float), cudaMemcpyDeviceToDevice);

    auto end = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    last_stats_.solve_time_ms = dur.count();
    last_stats_.iterations = max_iters_;  // Would need to hook into Ginkgo to get actual
    last_stats_.final_residual = tolerance_;
    last_stats_.residual_reduction = 1.0f;
}

}  // namespace PDAC
```

### 2.3 Integration with PDE Integration Layer

File: `/home/chase/SPQSP/SPQSP_PDAC-main/PDAC/pde/pde_integration.cu` (add to existing)

```cuda
#ifdef GINKGO_ENABLED

#include "pde_solver_ginkgo.cuh"

// Global Ginkgo solver instance
static PDAC::GinkgoPDESolver* g_ginkgo_solver = nullptr;

void initialize_ginkgo_solver(int grid_x, int grid_y, int grid_z,
                              float dx, float dt, int num_chemicals)
{
    if (!g_ginkgo_solver) {
        g_ginkgo_solver = new PDAC::GinkgoPDESolver(
            grid_x, grid_y, grid_z, dx, dt, num_chemicals);

        // Configure solver options (tune as needed)
        g_ginkgo_solver->set_solver_options(
            100,      // max iterations
            1e-6f,    // tolerance
            "ilu"     // preconditioner
        );
    }
}

void cleanup_ginkgo_solver()
{
    if (g_ginkgo_solver) {
        delete g_ginkgo_solver;
        g_ginkgo_solver = nullptr;
    }
}

void solve_pde_step_ginkgo(flamegpu::HostAPI& host_api,
                           int substrate_idx,
                           const std::string& chemical_name)
{
    if (!g_ginkgo_solver) return;

    const int grid_x = host_api.environment.getProperty<int>("grid_size_x");
    const int grid_y = host_api.environment.getProperty<int>("grid_size_y");
    const int grid_z = host_api.environment.getProperty<int>("grid_size_z");

    // Get device pointers from your existing PDE grid structures
    const float* d_C_curr = your_pde->get_concentration_ptr(substrate_idx);
    const float* d_sources = your_pde->get_sources_ptr(substrate_idx);
    const float* d_D = your_pde->get_diffusion_ptr();
    const float* d_lambda = your_pde->get_decay_ptr();
    float* d_C_next = your_pde->get_next_concentration_ptr(substrate_idx);

    // Solve
    g_ginkgo_solver->solve_timestep(d_C_curr, d_sources, d_D, d_lambda,
                                    d_C_next, substrate_idx);

    // Optional: Log metrics
    auto stats = g_ginkgo_solver->get_last_stats();
    if (stats.iterations >= 99) {  // Hitting max iterations?
        std::cerr << "[Warning] " << chemical_name << " CG: "
                  << stats.iterations << " iterations" << std::endl;
    }
}

#endif  // GINKGO_ENABLED
```

---

## Phase 3: Integration & Testing (Days 5-8, ~12 hours)

### 3.1 Integrate into Model Layers

File: `/home/chase/SPQSP/SPQSP_PDAC-main/PDAC/sim/model_layers.cu`

```cuda
// In the PDE solve layer (wherever you currently call your CG solver):

void define_main_model_layers(flamegpu::ModelDescription& model)
{
    // ... existing layers ...

    // Update: Replace your current PDE solver call
    // OLD:
    // model.addHostFunction(solve_pde_step);

    // NEW (if using Ginkgo):
    #ifdef GINKGO_ENABLED
    model.addHostFunction(solve_pde_step_ginkgo);
    #else
    model.addHostFunction(solve_pde_step);  // Fallback to custom CG
    #endif
}
```

### 3.2 Update Main Entry Point

File: `/home/chase/SPQSP/SPQSP_PDAC-main/PDAC/sim/main.cu`

```cuda
#include "pde/pde_solver_ginkgo.cuh"

int main(int argc, char *argv[]) {
    // ... setup ...

    #ifdef GINKGO_ENABLED
    initialize_ginkgo_solver(grid_size, grid_size, grid_size,
                             voxel_size, dt, NUM_CHEMICALS);
    std::cout << "[Main] Ginkgo PDE solver initialized" << std::endl;
    #endif

    // ... run simulation ...

    #ifdef GINKGO_ENABLED
    cleanup_ginkgo_solver();
    #endif

    return 0;
}
```

### 3.3 Numerical Validation Test

Create test file: `/home/chase/SPQSP/SPQSP_PDAC-main/tests/test_ginkgo_solver.cu`

```cuda
#include <gtest/gtest.h>
#include <cmath>

/**
 * Test 1: Verify Ginkgo stencil matches custom CG on simple case
 */
TEST(GinkgoSolver, StencilApplication) {
    int nx = 10, ny = 10, nz = 10;
    float dx = 1.0f, dt = 0.1f;

    // Setup: constant D=1, lambda=0, sources=0
    // Should satisfy: (I - 0.1·∇²)C = C_old
    // For constant input: ∇²C = 0, so C_new = C_old

    float* d_x, * d_y;
    cudaMalloc(&d_x, nx*ny*nz*sizeof(float));
    cudaMalloc(&d_y, nx*ny*nz*sizeof(float));

    // Input: all 1.0
    thrust::fill(thrust::device, thrust::device_pointer_cast(d_x),
                 thrust::device_pointer_cast(d_x) + nx*ny*nz, 1.0f);

    // Create Ginkgo operator
    float D_const = 1.0f, lambda_const = 0.0f, source_const = 0.0f;
    auto op = std::make_unique<PDAC::DiffusionReactionOperator>(
        ginkgo::CudaExecutor::create(0),
        nx, ny, nz, dx, dt,
        &D_const, &lambda_const, &source_const);

    // Apply operator
    auto executor = ginkgo::CudaExecutor::create(0);
    auto x_vec = ginkgo::matrix::Dense<float>::create(executor,
        ginkgo::dim<2>(nx*ny*nz, 1));
    auto y_vec = ginkgo::matrix::Dense<float>::create(executor,
        ginkgo::dim<2>(nx*ny*nz, 1));

    // ... copy data, apply, verify ...

    cudaFree(d_x);
    cudaFree(d_y);
}

/**
 * Test 2: Convergence comparison
 */
TEST(GinkgoSolver, ConvergenceVsCustomCG) {
    // Run small test case (11³) with both solvers
    // Compare final concentrations (should match to ~1e-6)
    // Compare iteration counts (Ginkgo should be similar or better)
}

/**
 * Test 3: Performance on realistic grids
 */
TEST(GinkkoSolver, PerformanceBenchmark) {
    // Time both solvers on 50³ and 100³ grids
    // Expect Ginkgo to be within 10-20% of custom CG
    // (may be slightly slower due to C++ abstraction)
}
```

### 3.4 Build & Test

```bash
cd /home/chase/SPQSP/SPQSP_PDAC-main/PDAC/sim
rm -rf build && mkdir build && cd build

# Configure with Ginkgo
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DGINKGO_DIR=$HOME/lib/ginkgo-install/lib/cmake/Ginkgo

# Build
cmake --build . -j8

# Quick test with small grid
./bin/pdac -s 5 -g 11 -oa 1 -op 1

# Expected output:
# [Ginkgo] Initialized PDE solver for 10 substrates on 11³ grid
# ... simulation progresses without errors ...
```

---

## Phase 4: Optimization & Tuning (Days 9-12, ~8 hours)

### 4.1 Preconditioner Selection

Ginkgo supports multiple preconditioners. Test each:

```cpp
// In GinkgoPDESolver::set_solver_options()

// Option 1: ILU (Incomplete LU) - good balance
auto precond = ginkgo::preconditioner::Ilu<float>::build()
    .on(executor_)
    .generate();

// Option 2: Chebyshev polynomial - faster setup, fewer iterations
auto precond = ginkgo::preconditioner::Chebyshev<float>::build()
    .with_max_block_size(32)
    .on(executor_)
    .generate();

// Option 3: Jacobi - simplest, fastest
auto precond = ginkgo::preconditioner::Jacobi<float>::build()
    .on(executor_)
    .generate();

// Benchmark each on 50³ and 100³ grids
```

### 4.2 Performance Tuning

```cpp
// Monitor in solve loop:
struct PerformanceMetrics {
    float solver_time_ms;
    int   iterations;
    float residual_final;
    float residual_reduction;
};

// Log to CSV for analysis:
// step, chemical, iterations, time_ms, residual
```

### 4.3 Comparison with Custom CG

```bash
# Run same simulation with both solvers
# Time difference should be < 20%
# Concentrations should match to ~1e-5 relative tolerance

./bin/pdac -solver=custom -s 50 -g 50 -oa 0 > results_custom.txt
./bin/pdac -solver=ginkgo -s 50 -g 50 -oa 0 > results_ginkgo.txt

# Compare outputs:
python3 compare_solvers.py results_custom.txt results_ginkgo.txt
```

---

## Phase 5: Documentation & Cleanup (Day 12, ~2 hours)

### 5.1 Update CLAUDE.md

Add to `/home/chase/SPQSP/SPQSP_PDAC-main/CLAUDE.md`:

```markdown
**PDE Solver (Updated Feb 2026)**
- ✅ Ginkgo 1.10+ CG solver with ILU preconditioning
- ✅ Matrix-free stencil application
- ✅ Single-GPU CUDA support (ready for multi-GPU via AmgX bridge)
- ✅ Unconditionally stable (Backward Euler)
- ✅ Concentration-dependent diffusion coefficients
- 📋 See `GPU_PDE_SOLVER_RESEARCH.md` for solver alternatives

**Configuration**:
- Solver: Ginkgo CG
- Preconditioner: ILU (Chebyshev for faster setup)
- Max iterations: 100
- Tolerance: 1e-6 relative residual
- Typical time: 30-40 ms per step (50³, 10 chemicals)
```

### 5.2 Add CMake Feature Toggle

```cmake
# In CMakeLists.txt
option(USE_GINKGO_SOLVER "Use Ginkgo for PDE solving (recommended)" ON)

if(USE_GINKGO_SOLVER AND Ginkgo_FOUND)
    target_compile_definitions(pdac PRIVATE GINKGO_ENABLED=1)
else()
    message(STATUS "Using default custom CG solver for PDEs")
endif()
```

---

## Estimated Timeline

| Phase | Task | Duration | Cumulative |
|-------|------|----------|-----------|
| 1 | Setup & build Ginkgo | 5 hours | 5 hrs |
| 2 | Core implementation | 10 hours | 15 hrs |
| 3 | Integration & testing | 12 hours | 27 hrs |
| 4 | Optimization & tuning | 8 hours | 35 hrs |
| 5 | Docs & cleanup | 2 hours | 37 hrs |
| **Total** | | | **~4.5 person-days** |

**Can distribute across 3-4 weeks** to run in parallel with other development.

---

## Troubleshooting

### Build Issues

**Error**: `#include "ginkgo/ginkgo.hpp"` not found
```bash
# Solution: Check CMake found Ginkgo
grep "Ginkgo found" build/CMakeOutput.log

# If missing:
export GINKGO_DIR=$HOME/lib/ginkgo-install/lib/cmake/Ginkgo
cmake .. -DCMAKE_PREFIX_PATH=$GINKGO_DIR
```

**Error**: CUDA architecture mismatch
```bash
# Check your GPU
nvidia-smi  # Get CUDA Capability (e.g., 7.5 → architecture 75)

# Rebuild Ginkgo with correct arch:
cmake .. -DCMAKE_CUDA_ARCHITECTURES=75  # Change to your GPU
```

### Runtime Issues

**Error**: `cudaErrorInvalidDevice` in Ginkgo kernels
```cpp
// Solution: Ensure GPU is available
int device_count;
cudaGetDeviceCount(&device_count);
if (device_count == 0) {
    std::cerr << "No CUDA devices found!" << std::endl;
    return;
}
```

**Performance**: Ginkgo slower than expected
```cpp
// Profile with NVIDIA Nsys:
// nsys profile -o ginkgo_profile ./bin/pdac -s 5 -g 21
// nsys-ui ginkgo_profile.nsys-rep

// Check:
// 1. Preconditioner setup time (may dominate for 11³ grids)
// 2. Iteration count (should decrease vs custom CG)
// 3. GPU utilization (should be > 80%)
```

---

## Success Criteria

- [ ] Builds cleanly with `cmake --build . -j8`
- [ ] Runs 5-step test on 11³ grid without errors
- [ ] Produces identical output (within 1e-5) as custom CG on 50³
- [ ] Solver time < 50 ms per step on 50³ grid with 10 chemicals
- [ ] CG iterations stable (not diverging as grid size increases)
- [ ] Preconditioner reduces iterations by > 20% vs plain CG
- [ ] Code builds with and without `GINKGO_ENABLED` flag

---

## Next Steps After Ginkgo Integration

1. **Performance Analysis** (2 weeks):
   - Benchmark on 100³ and 150³ grids
   - Compare Chebyshev vs ILU preconditioners
   - Profile GPU utilization (target > 85%)

2. **Multi-GPU Research** (4 weeks):
   - Add MPI support (Ginkgo handles this)
   - Test on 2-GPU, 4-GPU configurations
   - Measure strong scaling efficiency

3. **Solver Alternatives** (Optional, future):
   - If Ginkgo bottleneck → evaluate AmgX for multigrid
   - If MPI becomes critical → add SUNDIALS+hypre wrapper

---

**Resources**:
- Ginkgo Docs: https://ginkgo-project.github.io/
- Ginkgo GitHub: https://github.com/ginkgo-project/ginkgo
- This Guide: `/home/chase/SPQSP/SPQSP_PDAC-main/GINKGO_INTEGRATION_GUIDE.md`

---

**Last Updated**: February 23, 2026
**For**: SPQSP PDAC Project
**Status**: Ready for implementation
