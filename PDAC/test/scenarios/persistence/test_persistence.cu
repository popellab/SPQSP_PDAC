// Test 2 — Persistence
//
// Verifies the direction-memory branch of move_cell() produces the expected
// lag-1 autocorrelation of movement directions. With zero chemical gradients,
// the non-persistence branch reduces to uniform-random selection over valid
// 26-Moore neighbors; persistence overrides that with probability
// PARAM_PERSIST_TCELL_EFF, repeating the prior direction exactly.
//
// Since E[cos(d_prev, D_uniform)] = 0 by lattice symmetry (every direction
// has its negation in the 26-Moore set), the expected lag-1 autocorrelation
// E[ĥ_t · ĥ_{t+1}] equals exactly PARAM_PERSIST_TCELL_EFF.
//
// Layers enabled: occupancy, movement.
// Layers disabled: everything else — pure random walk with persistence memory.

#include "../../test_harness.cuh"
#include "../../../core/common.cuh"
#include "../../../pde/pde_solver.cuh"
#include "../../../pde/pde_integration.cuh"

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <tuple>
#include <cmath>
#include <vector>

using namespace PDAC;
using namespace PDAC::test;

// Forward-declared in test_main.cu; scenarios register themselves at static init.
void register_scenario(const std::string& name,
    std::function<TestConfig(const std::string&)> builder);

namespace {

struct PersistStats {
    std::map<int, std::tuple<int,int,int>> prev_pos;    // agent -> last (x,y,z)
    std::map<int, std::tuple<int,int,int>> last_disp;   // agent -> last non-zero displacement
    std::map<int, double> cell_sum_cos;                 // per-cell running sum of cos(θ)
    std::map<int, int>    cell_n_pairs;                 // per-cell pair count
    double sum_cos = 0.0;
    int n_pairs = 0;
    int n_exact_match = 0;   // d_{t+1} == d_t exactly
    int n_still = 0;
    std::ofstream traj;
    std::ofstream summary;
};

static TestConfig build_persistence(const std::string& param_file) {
    TestConfig cfg;
    cfg.name = "persistence";
    // Sized so a 200-step persistent walk (RMS ~31 voxels with p=0.4) stays
    // well away from the walls: centered seed in 101³ → ≥40-voxel headroom.
    cfg.grid_size = 101;
    cfg.voxel_size = 20.0f;
    cfg.steps = 200;
    cfg.seed = 42;
    cfg.param_file = param_file;

    // --- Layer subset: pure movement ---
    cfg.layers.ecm_update = false;
    cfg.layers.recruitment = false;
    cfg.layers.occupancy = true;        // movement needs vol_used
    cfg.layers.movement = true;
    cfg.layers.neighbor_scan = false;   // keep adh_p_move at seed default 1.0
    cfg.layers.state_transition = false;
    cfg.layers.chemical_sources = false;
    cfg.layers.pde_solve = false;
    cfg.layers.pde_gradients = false;   // no gradient reads needed; field irrelevant
    cfg.layers.division = false;
    cfg.layers.qsp = false;
    cfg.layers.abm_export = false;

    // Force 1 move substep / ABM step so each observed (x,y,z) transition is
    // a single lattice move — lag-1 autocorrelation maps directly to PERSIST.
    cfg.int_env_overrides.push_back({"PARAM_TCELL_MOVE_STEPS", 1});

    // --- ECM: zero porosity barrier ---
    cfg.ecm.density   = uniform_field(0.0f);
    cfg.ecm.crosslink = uniform_field(0.0f);
    cfg.ecm.floor     = uniform_field(0.0f);

    // --- Seed 200 T_CELL_EFF on a 10x10x2 grid with 2-voxel spacing ---
    // Spacing-2 guarantees each cell has all 26 neighbor voxels free at step 0,
    // so blocking collisions (which fail persistence → contaminate measurement)
    // are rare. Center around (49, 49, 49) in a 101³ grid.
    for (int k = 0; k < 2; k++) {
        for (int j = 0; j < 10; j++) {
            for (int i = 0; i < 10; i++) {
                AgentSeed s;
                s.agent_type = AGENT_TCELL;
                s.x = 40 + 2 * i;   // 40, 42, ..., 58
                s.y = 40 + 2 * j;   // 40, 42, ..., 58
                s.z = 48 + 2 * k;   // 48, 50
                s.cell_state = T_CELL_EFF;
                s.int_vars.push_back({"life", 9999});
                s.int_vars.push_back({"divide_limit", 0});
                s.int_vars.push_back({"divide_flag", 0});
                s.int_vars.push_back({"divide_cd", 0});
                s.int_vars.push_back({"hypoxia_exposure", 0});
                s.float_vars.push_back({"hypoxia_kill_factor", 1.0f});
                s.float_vars.push_back({"adh_p_move", 1.0f});
                cfg.agents.push_back(s);
            }
        }
    }

    // --- Per-step callback: track displacements, accumulate autocorr stats ---
    auto stats = std::make_shared<PersistStats>();
    const std::string out_dir = "../test/scenarios/" + cfg.name + "/outputs";
    std::filesystem::create_directories(out_dir);
    stats->traj.open(out_dir + "/trajectories.csv");
    stats->traj << "step,id,x,y,z\n";
    stats->summary.open(out_dir + "/persist_running.csv");
    stats->summary << "step,n_pairs,n_exact_match,mean_cos\n";

    cfg.step_callback = [stats](flamegpu::CUDASimulation& sim,
                                flamegpu::ModelDescription& model,
                                unsigned int step)
    {
        flamegpu::AgentVector pop(model.Agent(AGENT_TCELL));
        sim.getPopulationData(pop);

        for (unsigned int i = 0; i < pop.size(); i++) {
            const int id = pop[i].getID();
            const int x = pop[i].getVariable<int>("x");
            const int y = pop[i].getVariable<int>("y");
            const int z = pop[i].getVariable<int>("z");

            stats->traj << step << "," << id << "," << x << "," << y << "," << z << "\n";

            auto it_pos = stats->prev_pos.find(id);
            if (it_pos != stats->prev_pos.end()) {
                int px, py, pz;
                std::tie(px, py, pz) = it_pos->second;
                const int dx = x - px, dy = y - py, dz = z - pz;
                if (dx == 0 && dy == 0 && dz == 0) {
                    // Cell stayed put (blocking / failed volume claim).
                    // Break the pair chain: next move is not "consecutive".
                    stats->n_still++;
                    stats->last_disp.erase(id);
                } else {
                    auto it_disp = stats->last_disp.find(id);
                    if (it_disp != stats->last_disp.end()) {
                        int pdx, pdy, pdz;
                        std::tie(pdx, pdy, pdz) = it_disp->second;
                        const float mag_p = std::sqrt(
                            static_cast<float>(pdx*pdx + pdy*pdy + pdz*pdz));
                        const float mag_c = std::sqrt(
                            static_cast<float>(dx*dx + dy*dy + dz*dz));
                        const float cos_a =
                            (pdx*dx + pdy*dy + pdz*dz) / (mag_p * mag_c);
                        stats->sum_cos += static_cast<double>(cos_a);
                        stats->cell_sum_cos[id] += static_cast<double>(cos_a);
                        stats->cell_n_pairs[id] += 1;
                        stats->n_pairs++;
                        if (pdx == dx && pdy == dy && pdz == dz) {
                            stats->n_exact_match++;
                        }
                    }
                    stats->last_disp[id] = {dx, dy, dz};
                }
            }
            stats->prev_pos[id] = {x, y, z};
        }

        const double mean_cos = stats->n_pairs > 0
            ? stats->sum_cos / stats->n_pairs : 0.0;
        stats->summary << step << "," << stats->n_pairs << ","
                       << stats->n_exact_match << "," << mean_cos << "\n";

        if (step == 0 || step == 49 || step == 99 || step == 149 || step == 199) {
            std::cout << "  [step " << step << "] pairs=" << stats->n_pairs
                      << " exact_match=" << stats->n_exact_match
                      << " still=" << stats->n_still
                      << " mean_cos=" << mean_cos << std::endl;
        }

        if (step + 1 == 200) {
            const float expected =
                model.Environment().getProperty<float>("PARAM_PERSIST_TCELL_EFF");
            const double empirical = stats->n_pairs > 0
                ? stats->sum_cos / stats->n_pairs : 0.0;
            const double exact_rate = stats->n_pairs > 0
                ? static_cast<double>(stats->n_exact_match) / stats->n_pairs : 0.0;
            const double err = std::abs(empirical - expected);
            const double tol = 0.03;

            // Dump per-cell mean cos(θ) for histogram.
            std::ofstream per_cell_out(
                "../test/scenarios/persistence/outputs/per_cell_autocorr.csv");
            per_cell_out << "id,n_pairs,mean_cos\n";
            for (const auto& [id, sum] : stats->cell_sum_cos) {
                int n = stats->cell_n_pairs[id];
                per_cell_out << id << "," << n << ","
                             << (n > 0 ? sum / n : 0.0) << "\n";
            }

            std::cout << "\n==================== Persistence result ====================\n"
                      << "  Expected (PARAM_PERSIST_TCELL_EFF):    " << expected << "\n"
                      << "  Empirical <cos(θ_t, θ_{t+1})>:          " << empirical << "\n"
                      << "  Exact-match rate (d_{t+1} == d_t):      " << exact_rate << "\n"
                      << "                                          "
                      << "  (theory: p + (1-p)/26 = "
                      << expected + (1 - expected) / 26.0 << ")\n"
                      << "  |err|:                                   " << err
                      << "   (tol " << tol << ")\n"
                      << "  Total pairs: " << stats->n_pairs
                      << "   Still: " << stats->n_still << "\n"
                      << "  Result: " << (err < tol ? "PASS" : "FAIL") << "\n"
                      << "============================================================\n"
                      << std::endl;
        }
    };

    return cfg;
}

// Static init: register scenario with the global registry.
static const bool persistence_registered = []() {
    register_scenario("persistence", build_persistence);
    return true;
}();

} // anonymous namespace
