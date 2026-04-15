#include "test_harness.cuh"
#include "../core/common.cuh"
#include "../pde/pde_solver.cuh"
#include "../pde/pde_integration.cuh"

#include <iostream>
#include <string>
#include <map>
#include <functional>
#include <fstream>
#include <cmath>
#include <filesystem>

// Stub for prepare_abm_export (defined in main.cu, referenced by model_layers.cu)
// Tests don't use ABM export, but model_layers.cu references this symbol.
FLAMEGPU_HOST_FUNCTION(prepare_abm_export) {
    // no-op in test harness
}

using namespace PDAC;
using namespace PDAC::test;

// ============================================================================
// Default XML path (relative to working directory = PDAC/sim/)
// ============================================================================
static const char* DEFAULT_PARAM_FILE = "resource/param_all_test.xml";

// ============================================================================
// Test registry: map scenario name -> configuration function
// ============================================================================

// Each scenario function builds a TestConfig and returns it.
// Scenario files in scenarios/ register themselves via register_scenario().
using ScenarioBuilder = std::function<TestConfig(const std::string& param_file)>;
static std::map<std::string, ScenarioBuilder>& scenario_registry() {
    static std::map<std::string, ScenarioBuilder> reg;
    return reg;
}

// Called by scenario files at static init time
void register_scenario(const std::string& name, ScenarioBuilder builder) {
    scenario_registry()[name] = builder;
}

// ============================================================================
// Built-in smoke test: verifies the harness itself works
// ============================================================================

static TestConfig build_smoke_test(const std::string& param_file) {
    TestConfig cfg;
    cfg.name = "smoke";
    cfg.grid_size = 11;
    cfg.steps = 3;
    cfg.seed = 42;
    cfg.param_file = param_file;

    // Minimal layers: just movement + PDE
    cfg.layers.ecm_update = false;
    cfg.layers.recruitment = false;
    cfg.layers.division = false;
    cfg.layers.qsp = false;
    cfg.layers.abm_export = false;

    // Seed a few agents at center
    const int c = 5;
    cfg.agents.push_back({AGENT_CANCER_CELL, c, c, c, CANCER_PROGENITOR,
                          {{"divideFlag", 0}, {"divideCountRemaining", 5}}, {}});
    cfg.agents.push_back({AGENT_TCELL, c+1, c, c, T_CELL_EFF,
                          {{"life", 100}, {"divide_limit", 4}}, {}});
    cfg.agents.push_back({AGENT_MACROPHAGE, c-1, c, c, MAC_M2,
                          {{"life", 100}}, {}});

    // Pin O2 to uniform 0.065 mM (normal tissue)
    cfg.pinned_fields.push_back({CHEM_O2, uniform_field(0.065f)});

    // Per-step callback: print agent counts
    cfg.step_callback = [](flamegpu::CUDASimulation& sim,
                           flamegpu::ModelDescription& model,
                           unsigned int step)
    {
        flamegpu::AgentVector cc(model.Agent(AGENT_CANCER_CELL));
        flamegpu::AgentVector tc(model.Agent(AGENT_TCELL));
        flamegpu::AgentVector mc(model.Agent(AGENT_MACROPHAGE));
        sim.getPopulationData(cc);
        sim.getPopulationData(tc);
        sim.getPopulationData(mc);
        std::cout << "  [step " << step << "] CC=" << cc.size()
                  << " TC=" << tc.size() << " MAC=" << mc.size() << std::endl;
    };

    return cfg;
}

// ============================================================================
// Logging utilities for test scenarios
// ============================================================================

// Dump all agents of a given type to CSV (position + state + life)
static void dump_agent_csv(flamegpu::CUDASimulation& sim,
                           flamegpu::ModelDescription& model,
                           const std::string& agent_type,
                           const std::string& filepath,
                           unsigned int step)
{
    flamegpu::AgentVector pop(model.Agent(agent_type));
    sim.getPopulationData(pop);
    bool exists = std::filesystem::exists(filepath);
    std::ofstream f(filepath, std::ios::app);
    if (!exists) {
        f << "step,id,x,y,z,cell_state,life\n";
    }
    for (unsigned int i = 0; i < pop.size(); i++) {
        f << step << ","
          << pop[i].getID() << ","
          << pop[i].getVariable<int>("x") << ","
          << pop[i].getVariable<int>("y") << ","
          << pop[i].getVariable<int>("z") << ","
          << pop[i].getVariable<int>("cell_state") << ","
          << pop[i].getVariable<int>("life") << "\n";
    }
}

// Dump radial concentration profile from a point source
static void dump_radial_profile(const std::string& filepath,
                                int substrate_idx,
                                int cx, int cy, int cz,
                                int grid_size, float voxel_size)
{
    const int total = grid_size * grid_size * grid_size;
    std::vector<float> conc(total);

    float* d_conc = PDAC::g_pde_solver->get_device_concentration_ptr(substrate_idx);
    cudaMemcpy(conc.data(), d_conc, total * sizeof(float), cudaMemcpyDeviceToHost);

    std::ofstream f(filepath);
    f << "r_voxels,r_um,concentration\n";

    // Bin by integer distance
    std::map<int, std::pair<float, int>> bins;  // dist -> (sum, count)
    for (int z = 0; z < grid_size; z++)
        for (int y = 0; y < grid_size; y++)
            for (int x = 0; x < grid_size; x++) {
                float dx = x - cx, dy = y - cy, dz = z - cz;
                float r = std::sqrt(dx*dx + dy*dy + dz*dz);
                int ri = static_cast<int>(r + 0.5f);
                int idx = x + y * grid_size + z * grid_size * grid_size;
                bins[ri].first += conc[idx];
                bins[ri].second++;
            }

    for (auto& [ri, pair] : bins) {
        float avg = pair.first / pair.second;
        f << ri << "," << (ri * voxel_size) << "," << avg << "\n";
    }
}

// Dump full 3D concentration field to CSV
static void dump_concentration_field(const std::string& filepath,
                                     int substrate_idx,
                                     int grid_size)
{
    const int total = grid_size * grid_size * grid_size;
    std::vector<float> conc(total);

    float* d_conc = PDAC::g_pde_solver->get_device_concentration_ptr(substrate_idx);
    cudaMemcpy(conc.data(), d_conc, total * sizeof(float), cudaMemcpyDeviceToHost);

    std::ofstream f(filepath);
    f << "x,y,z,concentration\n";
    for (int z = 0; z < grid_size; z++)
        for (int y = 0; y < grid_size; y++)
            for (int x = 0; x < grid_size; x++) {
                int idx = x + y * grid_size + z * grid_size * grid_size;
                if (conc[idx] > 0.0f) {
                    f << x << "," << y << "," << z << "," << conc[idx] << "\n";
                }
            }
}

// ============================================================================
// Command-line parsing
// ============================================================================

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options] <scenario>\n"
              << "\nOptions:\n"
              << "  -p, --param-file PATH   XML parameter file (default: " << DEFAULT_PARAM_FILE << ")\n"
              << "  -g, --grid-size N       Override grid size\n"
              << "  -s, --steps N           Override step count\n"
              << "  -r, --seed N            Override RNG seed\n"
              << "  --list                  List available scenarios\n"
              << "  -h, --help              Show this help\n"
              << "\nAvailable scenarios:\n";
    // Always show smoke
    std::cout << "  smoke        Built-in harness verification\n";
    for (auto& [name, _] : scenario_registry()) {
        std::cout << "  " << name << "\n";
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, const char** argv) {
    std::string param_file = DEFAULT_PARAM_FILE;
    std::string scenario_name;
    int override_grid = -1;
    int override_steps = -1;
    int override_seed = -1;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
        if (arg == "--list") {
            std::cout << "smoke\n";
            for (auto& [name, _] : scenario_registry()) {
                std::cout << name << "\n";
            }
            return 0;
        }
        if ((arg == "-p" || arg == "--param-file") && i + 1 < argc) {
            param_file = argv[++i];
        } else if ((arg == "-g" || arg == "--grid-size") && i + 1 < argc) {
            override_grid = std::atoi(argv[++i]);
        } else if ((arg == "-s" || arg == "--steps") && i + 1 < argc) {
            override_steps = std::atoi(argv[++i]);
        } else if ((arg == "-r" || arg == "--seed") && i + 1 < argc) {
            override_seed = std::atoi(argv[++i]);
        } else if (arg[0] != '-') {
            scenario_name = arg;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    if (scenario_name.empty()) {
        std::cerr << "Error: no scenario specified\n" << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    // Build test config from scenario name
    TestConfig config;
    if (scenario_name == "smoke") {
        config = build_smoke_test(param_file);
    } else {
        auto it = scenario_registry().find(scenario_name);
        if (it == scenario_registry().end()) {
            std::cerr << "Unknown scenario: " << scenario_name << std::endl;
            print_usage(argv[0]);
            return 1;
        }
        config = it->second(param_file);
    }

    // Apply command-line overrides
    if (override_grid > 0)  config.grid_size = override_grid;
    if (override_steps > 0) config.steps = static_cast<unsigned int>(override_steps);
    if (override_seed >= 0) config.seed = static_cast<unsigned int>(override_seed);

    // Run the test
    return runTest(config);
}
