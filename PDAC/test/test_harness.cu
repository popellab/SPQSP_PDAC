#include "test_harness.cuh"
#include "../core/common.cuh"
#include "../core/model_functions.cuh"
#include "../pde/pde_integration.cuh"
#include "../pde/pde_solver.cuh"
#include "../abm/gpu_param.h"
#include "../qsp/LymphCentral_wrapper.h"

// Agent headers are NOT included here — they define __device__ functions that
// would cause duplicate symbol errors with model_definition.cu. We use forward
// declarations for the define*Agent() functions instead.

#include <iostream>
#include <fstream>
#include <filesystem>
#include <cmath>
#include <algorithm>
#include <set>
#include <chrono>

namespace PDAC {

// ---- Externs for host functions (from pde_integration.cu, model_functions.cu, qsp_integration.cu) ----
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER solve_pde_step;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER update_agent_counts;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER solve_qsp_step;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER zero_fib_density_field;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER aggregate_abm_events;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER copy_abm_counters_to_environment;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER reset_abm_event_counters;
extern PDESolver* g_pde_solver;

namespace test {

} // close namespace test temporarily

// Forward declarations for functions in model_definition.cu (PDAC namespace)
void defineCancerCellAgent(flamegpu::ModelDescription& model, bool include_state_divide);
void defineTCellAgent(flamegpu::ModelDescription& model, bool include_state_divide);
void defineTRegAgent(flamegpu::ModelDescription& model, bool include_state_divide);
void defineMDSCAgent(flamegpu::ModelDescription& model, bool include_state);
void defineMacrophageAgent(flamegpu::ModelDescription& model, bool include_state);
void defineFibroblastAgent(flamegpu::ModelDescription& model, bool include_state);
void defineVascularCellAgent(flamegpu::ModelDescription& model);
void defineBCellAgent(flamegpu::ModelDescription& model, bool include_state_divide);
void defineDCAgent(flamegpu::ModelDescription& model, bool include_state);
void defineCellLocationMessage(flamegpu::ModelDescription& model, float voxel_size, int grid_max);
void defineEnvironment(flamegpu::ModelDescription& model,
                       int grid_x, int grid_y, int grid_z, float voxel_size,
                       const GPUParam& gpu_params);

namespace test {

// ============================================================================
// Layer subset builder
// ============================================================================

// Spread N steps evenly across max_steps (same as model_layers.cu)
static std::set<int> spread_steps(int n_moves, int max_steps) {
    std::set<int> steps;
    if (n_moves <= 0 || max_steps <= 0) return steps;
    if (n_moves >= max_steps) {
        for (int i = 0; i < max_steps; i++) steps.insert(i);
        return steps;
    }
    for (int i = 0; i < n_moves; i++) {
        steps.insert(i * max_steps / n_moves);
    }
    return steps;
}

void defineTestLayers(flamegpu::ModelDescription& model, const LayerConfig& layers) {

    // ---- ECM update ----
    if (layers.ecm_update) {
        model.newLayer("zero_fib_density_field").addHostFunction(zero_fib_density_field);
        model.newLayer("build_density_field").addAgentFunction(AGENT_FIBROBLAST, "build_density_field");
        model.newLayer("update_ecm").addHostFunction(update_ecm_grid);
        model.newLayer("decay_antigen").addHostFunction(decay_antigen_grid);
        model.newLayer("update_ecm_orientation").addHostFunction(update_ecm_orientation);
        model.newLayer("decay_stress_field").addHostFunction(decay_stress_field);
    }

    // ---- Bookkeeping ----
    model.newLayer("update_agent_counts").addHostFunction(update_agent_counts);
    model.newLayer("reset_abm_event_counters_start").addHostFunction(reset_abm_event_counters);

    // ---- Recruitment ----
    if (layers.recruitment) {
        model.newLayer("reset_recruitment_sources").addHostFunction(reset_recruitment_sources);
        model.newLayer("update_vasculature_count").addHostFunction(update_vasculature_count);
        model.newLayer("mark_vascular_sources").addAgentFunction(AGENT_VASCULAR, "mark_sources");
        model.newLayer("recruit_gpu").addHostFunction(recruit_gpu);
        model.newLayer("place_recruited_agents").addHostFunction(place_recruited_agents);
    }

    // ---- Occupancy + Movement ----
    if (layers.occupancy) {
        model.newLayer("zero_occ_grid").addHostFunction(zero_occupancy_grid);
        {
            flamegpu::LayerDescription layer = model.newLayer("write_to_occ_grid");
            layer.addAgentFunction(AGENT_CANCER_CELL, "write_to_occ_grid");
            layer.addAgentFunction(AGENT_TCELL,       "write_to_occ_grid");
            layer.addAgentFunction(AGENT_TREG,        "write_to_occ_grid");
            layer.addAgentFunction(AGENT_MDSC,        "write_to_occ_grid");
            layer.addAgentFunction(AGENT_MACROPHAGE,  "write_to_occ_grid");
            layer.addAgentFunction(AGENT_FIBROBLAST,  "write_to_occ_grid");
            layer.addAgentFunction(AGENT_VASCULAR,    "write_to_occ_grid");
            layer.addAgentFunction(AGENT_BCELL,       "write_to_occ_grid");
            layer.addAgentFunction(AGENT_DC,          "write_to_occ_grid");
        }
    }

    if (layers.movement) {
        model.newLayer("reset_moves_cancer").addAgentFunction(AGENT_CANCER_CELL, "reset_moves");

        const int cancer_steps = model.Environment().getProperty<int>("PARAM_CANCER_MOVE_STEPS_STEM");
        const int tcell_steps  = model.Environment().getProperty<int>("PARAM_TCELL_MOVE_STEPS");
        const int treg_steps   = model.Environment().getProperty<int>("PARAM_TCELL_MOVE_STEPS");
        const int mdsc_steps   = model.Environment().getProperty<int>("PARAM_MDSC_MOVE_STEPS");
        const int mac_steps    = model.Environment().getProperty<int>("PARAM_MAC_MOVE_STEPS");
        const int fib_steps    = model.Environment().getProperty<int>("PARAM_FIB_MOVE_STEPS");
        const int bcell_steps  = model.Environment().getProperty<int>("PARAM_BCELL_MOVE_STEPS");
        const int dc_steps     = model.Environment().getProperty<int>("PARAM_DC_MOVE_STEPS");

        const int max_steps = std::max({cancer_steps, tcell_steps, treg_steps,
                                        mdsc_steps, mac_steps, fib_steps, bcell_steps, dc_steps});
        const auto cancer_on = spread_steps(cancer_steps, max_steps);
        const auto tcell_on  = spread_steps(tcell_steps,  max_steps);
        const auto treg_on   = spread_steps(treg_steps,   max_steps);
        const auto mdsc_on   = spread_steps(mdsc_steps,   max_steps);
        const auto mac_on    = spread_steps(mac_steps,     max_steps);
        const auto fib_on    = spread_steps(fib_steps,     max_steps);
        const auto bcell_on  = spread_steps(bcell_steps,   max_steps);
        const auto dc_on     = spread_steps(dc_steps,      max_steps);

        for (int i = 0; i < max_steps; i++) {
            flamegpu::LayerDescription layer = model.newLayer("move_interleaved_" + std::to_string(i));
            if (cancer_on.count(i)) layer.addAgentFunction(AGENT_CANCER_CELL, "move");
            if (tcell_on.count(i))  layer.addAgentFunction(AGENT_TCELL, "move");
            if (treg_on.count(i))   layer.addAgentFunction(AGENT_TREG, "move");
            if (mdsc_on.count(i))   layer.addAgentFunction(AGENT_MDSC, "move");
            if (mac_on.count(i))    layer.addAgentFunction(AGENT_MACROPHAGE, "move");
            if (fib_on.count(i))    layer.addAgentFunction(AGENT_FIBROBLAST, "move");
            if (bcell_on.count(i))  layer.addAgentFunction(AGENT_BCELL, "move");
            if (dc_on.count(i))     layer.addAgentFunction(AGENT_DC, "move");
        }
        model.newLayer("move_vascular").addAgentFunction(AGENT_VASCULAR, "move");
    }

    // ---- Neighbor scan ----
    if (layers.neighbor_scan) {
        model.newLayer("broadcast_cancer").addAgentFunction(AGENT_CANCER_CELL, "broadcast_location");
        model.newLayer("broadcast_tcell").addAgentFunction(AGENT_TCELL, "broadcast_location");
        model.newLayer("broadcast_treg").addAgentFunction(AGENT_TREG, "broadcast_location");
        model.newLayer("broadcast_mdsc").addAgentFunction(AGENT_MDSC, "broadcast_location");
        model.newLayer("broadcast_vascular").addAgentFunction(AGENT_VASCULAR, "broadcast_location");
        model.newLayer("broadcast_macrophage").addAgentFunction(AGENT_MACROPHAGE, "broadcast_location");
        model.newLayer("broadcast_fibroblast").addAgentFunction(AGENT_FIBROBLAST, "broadcast_location");
        model.newLayer("broadcast_bcell").addAgentFunction(AGENT_BCELL, "broadcast_location");
        model.newLayer("broadcast_dc").addAgentFunction(AGENT_DC, "broadcast_location");
        {
            flamegpu::LayerDescription layer = model.newLayer("scan_neighbors");
            layer.addAgentFunction(AGENT_CANCER_CELL, "count_neighbors");
            layer.addAgentFunction(AGENT_TCELL,       "scan_neighbors");
            layer.addAgentFunction(AGENT_TREG,        "scan_neighbors");
            layer.addAgentFunction(AGENT_MDSC,        "scan_neighbors");
            layer.addAgentFunction(AGENT_MACROPHAGE,  "scan_neighbors");
            layer.addAgentFunction(AGENT_FIBROBLAST,  "scan_neighbors");
            layer.addAgentFunction(AGENT_BCELL,       "scan_neighbors");
            layer.addAgentFunction(AGENT_DC,          "scan_neighbors");
        }
    }

    // ---- State transitions + chemical sources ----
    if (layers.state_transition || layers.chemical_sources) {
        model.newLayer("reset_pde_buffers").addHostFunction(reset_pde_buffers);
    }
    if (layers.state_transition) {
        flamegpu::LayerDescription layer = model.newLayer("state_transitions");
        layer.addAgentFunction(AGENT_CANCER_CELL, "state_step");
        layer.addAgentFunction(AGENT_TCELL,       "state_step");
        layer.addAgentFunction(AGENT_TREG,        "state_step");
        layer.addAgentFunction(AGENT_MDSC,        "state_step");
        layer.addAgentFunction(AGENT_MACROPHAGE,  "state_step");
        layer.addAgentFunction(AGENT_FIBROBLAST,  "state_step");
        layer.addAgentFunction(AGENT_VASCULAR,    "state_step");
        layer.addAgentFunction(AGENT_BCELL,       "state_step");
        layer.addAgentFunction(AGENT_DC,          "state_step");
    }
    if (layers.chemical_sources) {
        flamegpu::LayerDescription layer = model.newLayer("compute_chemical_sources");
        layer.addAgentFunction(AGENT_CANCER_CELL, "compute_chemical_sources");
        layer.addAgentFunction(AGENT_TCELL,       "compute_chemical_sources");
        layer.addAgentFunction(AGENT_TREG,        "compute_chemical_sources");
        layer.addAgentFunction(AGENT_MDSC,        "compute_chemical_sources");
        layer.addAgentFunction(AGENT_MACROPHAGE,  "compute_chemical_sources");
        layer.addAgentFunction(AGENT_FIBROBLAST,  "compute_chemical_sources");
        layer.addAgentFunction(AGENT_VASCULAR,    "compute_chemical_sources");
        layer.addAgentFunction(AGENT_BCELL,       "compute_chemical_sources");
        layer.addAgentFunction(AGENT_DC,          "compute_chemical_sources");
    }

    // ---- PDE solve + gradients ----
    if (layers.pde_solve) {
        model.newLayer("solve_pde").addHostFunction(solve_pde_step);
    }
    if (layers.pde_gradients) {
        model.newLayer("compute_pde_gradients").addHostFunction(compute_pde_gradients);
    }

    // ---- Division ----
    if (layers.division) {
        model.newLayer("reset_divide_wave").addHostFunction(reset_divide_wave);
        for (int w = 0; w < N_DIVIDE_WAVES; w++) {
            const std::string ws = std::to_string(w);
            model.newLayer("divide_cancer_w" + ws).addAgentFunction(AGENT_CANCER_CELL, "divide");
            model.newLayer("divide_tcell_w" + ws).addAgentFunction(AGENT_TCELL, "divide");
            model.newLayer("divide_treg_w" + ws).addAgentFunction(AGENT_TREG, "divide");
            model.newLayer("divide_bcell_w" + ws).addAgentFunction(AGENT_BCELL, "divide");
            model.newLayer("increment_divide_wave_" + ws).addHostFunction(increment_divide_wave);
        }
        model.newLayer("divide_vascular").addAgentFunction(AGENT_VASCULAR, "vascular_divide");
        model.newLayer("fib_divide").addAgentFunction(AGENT_FIBROBLAST, "divide");
    }

    // ---- QSP coupling ----
    if (layers.qsp) {
        model.newLayer("aggregate_abm_events").addHostFunction(aggregate_abm_events);
        model.newLayer("copy_abm_counters_to_environment").addHostFunction(copy_abm_counters_to_environment);
        model.newLayer("solve_qsp").addHostFunction(solve_qsp_step);
        model.newLayer("reset_abm_event_counters").addHostFunction(reset_abm_event_counters);
    }
}

// ============================================================================
// Build test model
// ============================================================================

std::unique_ptr<flamegpu::ModelDescription> buildTestModel(
    const TestConfig& config,
    GPUParam& gpu_params)
{
    // Load parameters from XML
    gpu_params.initializeParams(config.param_file);
    std::cout << "[test] Loaded params from " << config.param_file << std::endl;

    auto model = std::make_unique<flamegpu::ModelDescription>("PDAC_TEST_" + config.name);

    const int gs = config.grid_size;

    // Define messages + agents (all 9 types, with full state/divide functions)
    defineCellLocationMessage(*model, config.voxel_size, gs);
    defineCancerCellAgent(*model, true);
    defineTCellAgent(*model, true);
    defineTRegAgent(*model, true);
    defineMDSCAgent(*model, true);
    defineMacrophageAgent(*model, true);
    defineFibroblastAgent(*model, true);
    defineVascularCellAgent(*model);
    defineBCellAgent(*model, true);
    defineDCAgent(*model, true);

    // Define environment with GPU parameters
    defineEnvironment(*model, gs, gs, gs, config.voxel_size, gpu_params);

    // Define only the layers the test needs
    defineTestLayers(*model, config.layers);

    return model;
}

// ============================================================================
// Agent seeding
// ============================================================================

void seedAgents(
    flamegpu::CUDASimulation& sim,
    flamegpu::ModelDescription& model,
    const std::vector<AgentSeed>& agents)
{
    // Group agents by type
    std::map<std::string, std::vector<const AgentSeed*>> by_type;
    for (const auto& a : agents) {
        by_type[a.agent_type].push_back(&a);
    }

    for (auto& [type_name, seeds] : by_type) {
        flamegpu::AgentVector pop(model.Agent(type_name));
        for (const AgentSeed* s : seeds) {
            pop.push_back();
            flamegpu::AgentVector::Agent agent = pop.back();
            agent.setVariable<int>("x", s->x);
            agent.setVariable<int>("y", s->y);
            agent.setVariable<int>("z", s->z);
            agent.setVariable<int>("cell_state", s->cell_state);

            // Apply overrides
            for (const auto& [name, val] : s->int_vars) {
                agent.setVariable<int>(name, val);
            }
            for (const auto& [name, val] : s->float_vars) {
                agent.setVariable<float>(name, val);
            }
        }
        sim.setPopulationData(pop);
        std::cout << "[test] Seeded " << seeds.size() << " " << type_name << " agents" << std::endl;
    }

    // Set empty populations for agent types that have no seeds
    // (FLAMEGPU requires all agent types to have population data set)
    const std::vector<std::string> all_types = {
        AGENT_CANCER_CELL, AGENT_TCELL, AGENT_TREG, AGENT_MDSC,
        AGENT_MACROPHAGE, AGENT_FIBROBLAST, AGENT_VASCULAR,
        AGENT_BCELL, AGENT_DC
    };
    for (const auto& type_name : all_types) {
        if (by_type.find(type_name) == by_type.end()) {
            flamegpu::AgentVector empty_pop(model.Agent(type_name));
            sim.setPopulationData(empty_pop);
        }
    }
}

// ============================================================================
// ECM preset
// ============================================================================

void applyECMPreset(
    const ECMPreset& ecm,
    int grid_x, int grid_y, int grid_z)
{
    const int total = grid_x * grid_y * grid_z;

    if (ecm.density) {
        std::vector<float> data(total);
        for (int z = 0; z < grid_z; z++)
            for (int y = 0; y < grid_y; y++)
                for (int x = 0; x < grid_x; x++)
                    data[x + y * grid_x + z * grid_x * grid_y] = ecm.density(x, y, z);
        set_ecm_density_from_host(data.data(), total);
        std::cout << "[test] ECM density preset applied" << std::endl;
    }

    if (ecm.crosslink) {
        std::vector<float> data(total);
        for (int z = 0; z < grid_z; z++)
            for (int y = 0; y < grid_y; y++)
                for (int x = 0; x < grid_x; x++)
                    data[x + y * grid_x + z * grid_x * grid_y] = ecm.crosslink(x, y, z);
        set_ecm_crosslink_from_host(data.data(), total);
        std::cout << "[test] ECM crosslink preset applied" << std::endl;
    }

    if (ecm.orient_x) {
        std::vector<float> ox(total), oy(total), oz(total);
        for (int z = 0; z < grid_z; z++)
            for (int y = 0; y < grid_y; y++)
                for (int x = 0; x < grid_x; x++) {
                    int idx = x + y * grid_x + z * grid_x * grid_y;
                    ox[idx] = ecm.orient_x(x, y, z);
                    oy[idx] = ecm.orient_y(x, y, z);
                    oz[idx] = ecm.orient_z(x, y, z);
                }
        set_ecm_orient_from_host(ox.data(), oy.data(), oz.data(), total);
        std::cout << "[test] ECM orientation preset applied" << std::endl;
    }

    if (ecm.floor) {
        std::vector<float> data(total);
        for (int z = 0; z < grid_z; z++)
            for (int y = 0; y < grid_y; y++)
                for (int x = 0; x < grid_x; x++)
                    data[x + y * grid_x + z * grid_x * grid_y] = ecm.floor(x, y, z);
        set_ecm_floor_from_host(data.data(), total);
        std::cout << "[test] ECM floor preset applied" << std::endl;
    }
}

// ============================================================================
// Chemical field pinning
// ============================================================================

void applyPinnedFields(
    const std::vector<PinnedField>& pinned,
    int grid_x, int grid_y, int grid_z)
{
    if (pinned.empty()) return;
    const int total = grid_x * grid_y * grid_z;

    for (const auto& pf : pinned) {
        std::vector<float> data(total);
        for (int z = 0; z < grid_z; z++)
            for (int y = 0; y < grid_y; y++)
                for (int x = 0; x < grid_x; x++)
                    data[x + y * grid_x + z * grid_x * grid_y] = pf.profile(x, y, z);

        // Overwrite device concentration array
        float* d_conc = g_pde_solver->get_device_concentration_ptr(pf.substrate_idx);
        cudaMemcpy(d_conc, data.data(), total * sizeof(float), cudaMemcpyHostToDevice);
    }
}

// ============================================================================
// Run a complete test scenario
// ============================================================================

int runTest(const TestConfig& config) {
    auto t_start = std::chrono::high_resolution_clock::now();

    std::cout << "\n========================================" << std::endl;
    std::cout << "  Test: " << config.name << std::endl;
    std::cout << "  Grid: " << config.grid_size << "³, Steps: " << config.steps
              << ", Seed: " << config.seed << std::endl;
    std::cout << "========================================\n" << std::endl;

    const int gs = config.grid_size;

    // 1. Create output directory
    std::string out_dir = config.output_dir;
    if (out_dir.empty()) {
        out_dir = "outputs/tests/" + config.name;
    }
    std::filesystem::create_directories(out_dir);

    // 2. Build model
    GPUParam gpu_params;
    auto model = buildTestModel(config, gpu_params);

    // 3. Initialize QSP (needed for derived params even if QSP layer is off)
    LymphCentralWrapper lymph;
    lymph.initialize(config.param_file);
    set_internal_params(*model, lymph);

    // 4. Initialize PDE solver
    float dt_abm = model->Environment().getProperty<float>("PARAM_SEC_PER_SLICE");
    int mol_steps = model->Environment().getProperty<int>("PARAM_MOLECULAR_STEPS");
    initialize_pde_solver(gs, gs, gs, config.voxel_size, dt_abm, mol_steps, gpu_params, *model);
    set_pde_pointers_in_environment(*model);

    // Initialize ECM to baseline
    float ecm_baseline = model->Environment().getProperty<float>("PARAM_ECM_BASELINE");
    initialize_ecm_to_saturation(
        model->Environment().getProperty<float>("PARAM_ECM_DENSITY_CAP"));
    initialize_ecm_floor_uniform(ecm_baseline);

    // 5. Allocate GPU buffers for event/state counters (agents atomicAdd to these)
    unsigned int* device_event_counters = nullptr;
    unsigned int* device_state_counters = nullptr;
    cudaMalloc(&device_event_counters, ABM_EVENT_COUNTER_SIZE * sizeof(unsigned int));
    cudaMalloc(&device_state_counters, ABM_STATE_COUNTER_SIZE * sizeof(unsigned int));
    cudaMemset(device_event_counters, 0, ABM_EVENT_COUNTER_SIZE * sizeof(unsigned int));
    cudaMemset(device_state_counters, 0, ABM_STATE_COUNTER_SIZE * sizeof(unsigned int));
    model->Environment().setProperty<uint64_t>("event_counters_ptr",
        reinterpret_cast<uint64_t>(device_event_counters));
    model->Environment().setProperty<uint64_t>("state_counters_ptr",
        reinterpret_cast<uint64_t>(device_state_counters));

    // Increase CUDA stack size for complex kernels (Newton-Raphson in state_step)
    cudaDeviceSetLimit(cudaLimitStackSize, 16384);

    // 6. Create CUDA simulation
    flamegpu::CUDASimulation sim(*model);
    sim.SimulationConfig().steps = config.steps;
    sim.SimulationConfig().random_seed = config.seed;
    sim.setEnvironmentProperty<unsigned int>("sim_seed", config.seed);

    // 7. Seed agents
    seedAgents(sim, *model, config.agents);

    // 8. Apply ECM preset (after PDE init so device arrays exist)
    applyECMPreset(config.ecm, gs, gs, gs);

    // 9. Apply initial pinned fields
    applyPinnedFields(config.pinned_fields, gs, gs, gs);

    // 10. Run simulation with per-step callback
    std::cout << "[test] Running " << config.steps << " steps..." << std::endl;

    for (unsigned int step = 0; step < config.steps; step++) {
        sim.step();

        // Re-apply pinned fields after PDE solve
        applyPinnedFields(config.pinned_fields, gs, gs, gs);

        // Call user's per-step callback
        if (config.step_callback) {
            config.step_callback(sim, *model, step);
        }
    }

    // 11. Cleanup
    cleanup_pde_solver();
    cudaFree(device_event_counters);
    cudaFree(device_state_counters);

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();
    std::cout << "\n[test] " << config.name << " complete in " << elapsed << "s" << std::endl;

    return 0;
}

} // namespace test
} // namespace PDAC
