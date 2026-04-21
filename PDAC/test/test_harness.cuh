#ifndef TEST_HARNESS_CUH
#define TEST_HARNESS_CUH

#include "flamegpu/flamegpu.h"
#include "../pde/pde_solver.cuh"
#include "../pde/pde_integration.cuh"
#include "../core/common.cuh"
#include "../abm/gpu_param.h"
#include "../qsp/LymphCentral_wrapper.h"

#include <string>
#include <vector>
#include <array>
#include <functional>
#include <memory>

namespace PDAC {
namespace test {

// ============================================================================
// Chemical field pinning
// ============================================================================

// A pinned chemical has its concentration overwritten after every PDE solve.
// profile_fn(x, y, z) -> concentration value at that voxel.
using FieldProfileFn = std::function<float(int x, int y, int z)>;

struct PinnedField {
    int substrate_idx;         // CHEM_O2, CHEM_IFN, etc.
    FieldProfileFn profile;    // spatial profile function
};

// ============================================================================
// Agent seeding specification
// ============================================================================

struct AgentSeed {
    std::string agent_type;    // AGENT_CANCER_CELL, AGENT_TCELL, etc.
    int x, y, z;               // voxel coordinates
    int cell_state;            // agent-specific state enum
    // Optional per-agent overrides (name -> value)
    std::vector<std::pair<std::string, int>> int_vars;
    std::vector<std::pair<std::string, float>> float_vars;
};

// ============================================================================
// ECM preset specification
// ============================================================================

struct ECMPreset {
    // Per-voxel functions. If null, left at default (baseline).
    FieldProfileFn density;       // nullptr = use PARAM_ECM_BASELINE
    FieldProfileFn crosslink;     // nullptr = 0
    // Orientation: if orient_x is non-null, all three must be set.
    FieldProfileFn orient_x;
    FieldProfileFn orient_y;
    FieldProfileFn orient_z;
    FieldProfileFn floor;         // nullptr = use PARAM_ECM_BASELINE
};

// ============================================================================
// Per-step logging callback
// ============================================================================

// Called after each simulation step with the CUDASimulation reference.
// Tests use this to dump per-agent state, chemical concentrations, etc.
using StepCallback = std::function<void(
    flamegpu::CUDASimulation& sim,
    flamegpu::ModelDescription& model,
    unsigned int step)>;

// ============================================================================
// Layer selection for test scenarios
// ============================================================================

// Which simulation phases to include. Tests enable only what they need.
struct LayerConfig {
    bool ecm_update       = false;  // ECM density field scatter + update
    bool recruitment      = false;  // vascular marking + GPU recruitment
    bool occupancy        = true;   // zero + write occ grid (needed for movement)
    bool movement         = true;   // interleaved movement substeps
    bool neighbor_scan    = true;   // broadcast + scan neighbors
    bool state_transition = true;   // state_step for all agents
    bool chemical_sources = true;   // compute_chemical_sources
    bool pde_solve        = true;   // PDE diffusion + decay
    bool pde_gradients    = true;   // gradient computation
    bool division         = false;  // wave-interleaved division
    bool qsp              = false;  // QSP ODE step
    bool abm_export       = false;  // pack_for_export (usually off in tests)
};

// ============================================================================
// Test scenario configuration
// ============================================================================

struct TestConfig {
    std::string name;               // scenario identifier
    int grid_size       = 21;       // cubic grid (grid_size^3 voxels)
    float voxel_size    = 20.0f;    // micrometers
    unsigned int steps  = 100;      // ABM steps to run
    unsigned int seed   = 42;       // RNG seed
    std::string param_file;         // XML param file path (required)
    std::string output_dir;         // output dir (auto: ../test/scenarios/<name>/outputs/)

    // What to run
    LayerConfig layers;

    // Chemical pinning (overwrite substrate concentration each step)
    std::vector<PinnedField> pinned_fields;

    // Agent seeding
    std::vector<AgentSeed> agents;

    // ECM preset (applied after PDE init, before simulation)
    ECMPreset ecm;

    // Per-scenario overrides for FLAMEGPU environment properties.
    // Applied after defineEnvironment() populates XML params, but BEFORE
    // defineTestLayers() reads them to build layer structure. Use this to
    // force a single move substep, disable a subprocess rate, etc.
    std::vector<std::pair<std::string, int>>   int_env_overrides;
    std::vector<std::pair<std::string, float>> float_env_overrides;
    std::vector<std::pair<std::string, bool>>  bool_env_overrides;

    // Per-step logging callback
    StepCallback step_callback;
};

// ============================================================================
// Test harness API
// ============================================================================

// Build a FLAMEGPU model with agent definitions and selected layers.
// Returns the model and populates gpu_params from XML.
std::unique_ptr<flamegpu::ModelDescription> buildTestModel(
    const TestConfig& config,
    GPUParam& gpu_params);

// Define only the layers specified by LayerConfig.
// Called by buildTestModel; exposed for tests that need custom layer ordering.
void defineTestLayers(flamegpu::ModelDescription& model, const LayerConfig& layers);

// Seed agents into the simulation from the TestConfig spec.
void seedAgents(
    flamegpu::CUDASimulation& sim,
    flamegpu::ModelDescription& model,
    const std::vector<AgentSeed>& agents);

// Apply ECM preset: copy host-side density/crosslink/orientation arrays to device.
void applyECMPreset(
    const ECMPreset& ecm,
    int grid_x, int grid_y, int grid_z);

// Apply pinned chemical fields: overwrite device concentration arrays.
// Called after each PDE solve step (via a host function inserted into the layer list).
void applyPinnedFields(
    const std::vector<PinnedField>& pinned,
    int grid_x, int grid_y, int grid_z);

// Run a complete test scenario: build model, init PDE, seed agents, run, log.
// Returns 0 on success, nonzero on failure.
int runTest(const TestConfig& config);

// ============================================================================
// Utility: common field profile functions
// ============================================================================

// Uniform concentration everywhere
inline FieldProfileFn uniform_field(float value) {
    return [value](int, int, int) { return value; };
}

// Linear gradient along an axis: value = base + slope * coord[axis]
// axis: 0=x, 1=y, 2=z
inline FieldProfileFn linear_gradient(float base, float slope, int axis) {
    return [base, slope, axis](int x, int y, int z) {
        int coord = (axis == 0) ? x : (axis == 1) ? y : z;
        return base + slope * static_cast<float>(coord);
    };
}

// Radial from center: value = amplitude * exp(-r^2 / (2*sigma^2))
inline FieldProfileFn gaussian_source(int cx, int cy, int cz, float amplitude, float sigma) {
    float inv_2s2 = 1.0f / (2.0f * sigma * sigma);
    return [cx, cy, cz, amplitude, inv_2s2](int x, int y, int z) {
        float dx = x - cx, dy = y - cy, dz = z - cz;
        return amplitude * expf(-(dx*dx + dy*dy + dz*dz) * inv_2s2);
    };
}

// Krogh point-source steady state: C(r) = (S / (4*pi*D*r)) * exp(-r / lambda)
// where lambda = sqrt(D / k_decay), r = distance from (cx,cy,cz) in voxels * voxel_size
inline FieldProfileFn krogh_steady_state(
    int cx, int cy, int cz,
    float source_rate, float diffusivity, float decay_rate, float voxel_size)
{
    float lambda = std::sqrt(diffusivity / decay_rate);
    float prefactor = source_rate / (4.0f * 3.14159265f * diffusivity);
    return [cx, cy, cz, prefactor, lambda, voxel_size](int x, int y, int z) {
        float dx = (x - cx) * voxel_size;
        float dy = (y - cy) * voxel_size;
        float dz = (z - cz) * voxel_size;
        float r = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (r < voxel_size * 0.5f) r = voxel_size * 0.5f;  // clamp at half-voxel
        return prefactor / r * expf(-r / lambda);
    };
}

} // namespace test
} // namespace PDAC

#endif // TEST_HARNESS_CUH
