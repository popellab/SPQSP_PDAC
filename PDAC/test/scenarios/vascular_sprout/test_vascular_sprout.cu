// Vascular sprouting — TIP chemotaxis + integrated network growth
//
// Validates two aspects of the vascular mechanism in isolation:
//
//   A. vascular_tip_chemotaxis — pure TIP CI under pinned +x VEGF gradient.
//      No state transitions or division; each ABM step = one TIP lattice move
//      (vascular_move runs once per step, no substep param). Measures
//      <cos θ> = <dx/|disp|> across all moves. On 26-Moore,
//      expected CI = PARAM_CHEMO_CI_VAS_TIP = 0.27. Isolates chemotaxis weight
//      math from sprouting/branching logic.
//
//   B. vascular_sprout_growth — single PHALANX at one end of a pinned VEGF
//      gradient. Full state_transition + division + movement. Observes the
//      integrated behavior: PHALANX sprouts new TIP (p_tip = VEGF/(VAS_50+VEGF),
//      MIN_NEIGHBOR box-empty check), TIP migrates +x via VEGF gradient,
//      TIP division each step leaves a PHALANX stalk trail. Known biology
//      limitation from CLAUDE.md: vascular network maturation rate is under-
//      tuned — this test exposes the trail-growth rate vs expected.
//
// Pass criteria (evaluated in make_figures.py):
//   A1. Empirical CI within 5% of PARAM_CHEMO_CI_VAS_TIP.
//   A2. n_still < 5% of (n_still + n_moves) — TIPs actually move.
//   B1. TIP max_x strictly increases with step (migration).
//   B2. PHALANX count monotonically non-decreasing; net growth by end of run.
//   B3. ≥1 new TIP spawned from source PHALANX in early steps at high VEGF.

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

void register_scenario(const std::string& name,
    std::function<TestConfig(const std::string&)> builder);

namespace {

// ============================================================================
// Scenario A — TIP chemotaxis CI
// ============================================================================

struct ChemoStats {
    std::map<int, std::tuple<int,int,int>> prev_pos;
    double sum_cos = 0.0;
    int n_moves = 0;
    int n_still = 0;
    std::ofstream traj;
    std::ofstream summary;
    std::ofstream params;
};

static TestConfig build_vascular_tip_chemotaxis(const std::string& param_file) {
    TestConfig cfg;
    cfg.name        = "vascular_tip_chemotaxis";
    cfg.grid_size   = 101;
    cfg.voxel_size  = 20.0f;
    cfg.steps       = 100;
    cfg.seed        = 42;
    cfg.param_file  = param_file;

    cfg.layers.ecm_update       = false;
    cfg.layers.recruitment      = false;
    cfg.layers.occupancy        = true;
    cfg.layers.movement         = true;
    cfg.layers.neighbor_scan    = false;
    cfg.layers.state_transition = false;  // no sprouting
    cfg.layers.chemical_sources = false;  // TIPs don't perturb pinned VEGF
    cfg.layers.pde_solve        = false;  // keep field pinned
    cfg.layers.pde_gradients    = true;   // still need gradient computed
    cfg.layers.division         = false;  // no stalk trail in CI-only test
    cfg.layers.qsp              = false;
    cfg.layers.abm_export       = false;

    // Pinned VEGF: linear +x gradient. Magnitude irrelevant for CI (move_cell
    // normalizes the gradient vector).
    cfg.pinned_fields.push_back({CHEM_VEGFA, linear_gradient(1.0f, 0.5f, /*x*/0)});

    cfg.ecm.density   = uniform_field(0.0f);
    cfg.ecm.crosslink = uniform_field(0.0f);
    cfg.ecm.floor     = uniform_field(0.0f);

    // Seed 100 TIP cells on a 10×10 slab at x=20 (80-voxel +x headroom).
    // Each TIP at a unique (y,z) so volume capacity never binds.
    unsigned int next_tip_id = 1;
    for (int j = 0; j < 10; j++) {
        for (int i = 0; i < 10; i++) {
            AgentSeed s;
            s.agent_type = AGENT_VASCULAR;
            s.x = 20;
            s.y = 45 + i;
            s.z = 45 + j;
            s.cell_state = VAS_TIP;
            s.int_vars.push_back({"is_dysfunctional", 0});
            s.int_vars.push_back({"maturity",         0});
            s.int_vars.push_back({"persist_dir_x",    0});
            s.int_vars.push_back({"persist_dir_y",    0});
            s.int_vars.push_back({"persist_dir_z",    0});
            // Give each TIP a unique tip_id so MIN_NEIGHBOR check would treat
            // them independently (shouldn't matter since sprouting is off).
            cfg.agents.push_back(s);
            next_tip_id++;
        }
    }

    auto stats = std::make_shared<ChemoStats>();
    const std::string out_dir = "../test/scenarios/vascular_sprout/outputs/" + cfg.name;
    std::filesystem::create_directories(out_dir);
    stats->traj.open(out_dir + "/trajectories.csv");
    stats->traj << "step,id,x,y,z\n";
    stats->summary.open(out_dir + "/ci_running.csv");
    stats->summary << "step,n_moves,n_still,ci_running\n";
    stats->params.open(out_dir + "/params.csv");
    stats->params << "key,value\n";

    cfg.step_callback = [stats, out_dir](flamegpu::CUDASimulation& sim,
                                         flamegpu::ModelDescription& model,
                                         unsigned int step)
    {
        if (step == 0) {
            auto env = model.Environment();
            stats->params << "PARAM_CHEMO_CI_VAS_TIP,"
                          << env.getProperty<float>("PARAM_CHEMO_CI_VAS_TIP") << "\n";
            stats->params << "PARAM_PERSIST_VAS_TIP,"
                          << env.getProperty<float>("PARAM_PERSIST_VAS_TIP") << "\n";
            stats->params << "PARAM_VAS_50,"
                          << env.getProperty<float>("PARAM_VAS_50") << "\n";
            stats->params << "grid_size," << env.getProperty<int>("grid_size_x") << "\n";
            stats->params.flush();
        }

        flamegpu::AgentVector pop(model.Agent(AGENT_VASCULAR));
        sim.getPopulationData(pop);
        for (unsigned int i = 0; i < pop.size(); i++) {
            const int id = pop[i].getID();
            const int x  = pop[i].getVariable<int>("x");
            const int y  = pop[i].getVariable<int>("y");
            const int z  = pop[i].getVariable<int>("z");
            stats->traj << step << "," << id << "," << x << "," << y << "," << z << "\n";

            auto it = stats->prev_pos.find(id);
            if (it != stats->prev_pos.end()) {
                int px, py, pz;
                std::tie(px, py, pz) = it->second;
                const int dx = x - px, dy = y - py, dz = z - pz;
                if (dx == 0 && dy == 0 && dz == 0) {
                    stats->n_still++;
                } else {
                    const float mag = std::sqrt(static_cast<float>(dx*dx + dy*dy + dz*dz));
                    stats->sum_cos += static_cast<double>(dx) / mag;
                    stats->n_moves++;
                }
            }
            stats->prev_pos[id] = {x, y, z};
        }

        const double ci_running = stats->n_moves > 0
            ? stats->sum_cos / stats->n_moves : 0.0;
        stats->summary << step << "," << stats->n_moves << ","
                       << stats->n_still << "," << ci_running << "\n";

        if (step == 0 || step + 1 == 100 || (step + 1) % 25 == 0) {
            std::cout << "  [step " << step << "] moves=" << stats->n_moves
                      << " still=" << stats->n_still
                      << " CI_running=" << ci_running << std::endl;
        }
    };

    return cfg;
}

// ============================================================================
// Scenario B — Integrated sprout growth
// ============================================================================

struct GrowthStats {
    std::ofstream time_series;   // step, n_tip, n_phalanx, max_x, frontier_x
    std::ofstream agents;        // step, id, state, x, y, z, tip_id
    std::ofstream params;
};

static TestConfig build_vascular_sprout_growth(const std::string& param_file) {
    TestConfig cfg;
    cfg.name        = "vascular_sprout_growth";
    cfg.grid_size   = 51;
    cfg.voxel_size  = 20.0f;
    cfg.steps       = 40;
    cfg.seed        = 42;
    cfg.param_file  = param_file;

    cfg.layers.ecm_update       = false;
    cfg.layers.recruitment      = false;
    cfg.layers.occupancy        = true;
    cfg.layers.movement         = true;
    cfg.layers.neighbor_scan    = false;
    cfg.layers.state_transition = true;   // PHALANX→TIP sprouting logic
    cfg.layers.chemical_sources = false;  // no source/uptake perturbs pinned field
    cfg.layers.pde_solve        = false;  // keep field pinned
    cfg.layers.pde_gradients    = true;   // ∇VEGF for TIP chemotaxis
    cfg.layers.division         = true;   // TIP division (stalk trail) + PHALANX sprout
    cfg.layers.qsp              = false;
    cfg.layers.abm_export       = false;

    // Pinned VEGF at saturating level 10 nM at x=0 ramping up to ~35 nM at x=50.
    // Source PHALANX at x=5 sees ~12.5 nM; far-end x=45 sees ~32.5 nM.
    // Well above any plausible PARAM_VAS_50 (typical QSP values ≪10 nM) so
    // p_tip ≈ 1 throughout and sprouting logic isn't rate-limited by VEGF.
    cfg.pinned_fields.push_back({CHEM_VEGFA, linear_gradient(10.0f, 0.5f, /*x*/0)});

    cfg.ecm.density   = uniform_field(0.0f);
    cfg.ecm.crosslink = uniform_field(0.0f);
    cfg.ecm.floor     = uniform_field(0.0f);

    // Single source PHALANX at one end, grid center y,z.
    AgentSeed src;
    src.agent_type = AGENT_VASCULAR;
    src.x = 5;
    src.y = 25;
    src.z = 25;
    src.cell_state = VAS_PHALANX;
    src.int_vars.push_back({"is_dysfunctional", 0});
    src.int_vars.push_back({"maturity",         100});  // mature — resist regression
    cfg.agents.push_back(src);

    auto stats = std::make_shared<GrowthStats>();
    const std::string out_dir = "../test/scenarios/vascular_sprout/outputs/" + cfg.name;
    std::filesystem::create_directories(out_dir);
    stats->time_series.open(out_dir + "/time_series.csv");
    stats->time_series << "step,n_tip,n_phalanx,n_total,max_x,mean_x_phalanx\n";
    stats->agents.open(out_dir + "/agents.csv");
    stats->agents << "step,id,state,x,y,z,tip_id\n";
    stats->params.open(out_dir + "/params.csv");
    stats->params << "key,value\n";

    cfg.step_callback = [stats, out_dir](flamegpu::CUDASimulation& sim,
                                         flamegpu::ModelDescription& model,
                                         unsigned int step)
    {
        if (step == 0) {
            auto env = model.Environment();
            stats->params << "PARAM_CHEMO_CI_VAS_TIP,"
                          << env.getProperty<float>("PARAM_CHEMO_CI_VAS_TIP") << "\n";
            stats->params << "PARAM_VAS_50,"
                          << env.getProperty<float>("PARAM_VAS_50") << "\n";
            stats->params << "PARAM_VAS_BRANCH_PROB,"
                          << env.getProperty<float>("PARAM_VAS_BRANCH_PROB") << "\n";
            stats->params << "PARAM_VAS_MIN_NEIGHBOR,"
                          << env.getProperty<float>("PARAM_VAS_MIN_NEIGHBOR") << "\n";
            stats->params << "PARAM_VAS_REGRESS_RATE,"
                          << env.getProperty<float>("PARAM_VAS_REGRESS_RATE") << "\n";
            stats->params << "grid_size," << env.getProperty<int>("grid_size_x") << "\n";
            stats->params << "source_x,5\n";
            stats->params << "source_y,25\n";
            stats->params << "source_z,25\n";
            stats->params.flush();
        }

        flamegpu::AgentVector pop(model.Agent(AGENT_VASCULAR));
        sim.getPopulationData(pop);

        int n_tip = 0, n_phalanx = 0;
        int max_x = 0;
        double sum_x_phalanx = 0.0;
        int n_x_phalanx = 0;

        for (unsigned int i = 0; i < pop.size(); i++) {
            const int id    = pop[i].getID();
            const int state = pop[i].getVariable<int>("cell_state");
            const int x     = pop[i].getVariable<int>("x");
            const int y     = pop[i].getVariable<int>("y");
            const int z     = pop[i].getVariable<int>("z");
            const unsigned int tid = pop[i].getVariable<unsigned int>("tip_id");

            stats->agents << step << "," << id << "," << state << ","
                          << x << "," << y << "," << z << "," << tid << "\n";

            if (state == VAS_TIP) {
                n_tip++;
                if (x > max_x) max_x = x;
            } else if (state == VAS_PHALANX) {
                n_phalanx++;
                sum_x_phalanx += x;
                n_x_phalanx++;
                if (x > max_x) max_x = x;
            }
        }

        const double mean_x_phalanx = n_x_phalanx > 0
            ? sum_x_phalanx / n_x_phalanx : 0.0;
        stats->time_series << step << "," << n_tip << "," << n_phalanx << ","
                           << (n_tip + n_phalanx) << "," << max_x << ","
                           << mean_x_phalanx << "\n";

        if (step < 5 || step % 5 == 0 || step + 1 == 40) {
            std::cout << "  [step " << step << "] tip=" << n_tip
                      << " phal=" << n_phalanx
                      << " max_x=" << max_x
                      << " mean_x_phal=" << mean_x_phalanx << std::endl;
        }
    };

    return cfg;
}

// ============================================================================
// Registration
// ============================================================================

static const bool tip_chemo_registered = []() {
    register_scenario("vascular_tip_chemotaxis", build_vascular_tip_chemotaxis);
    return true;
}();

static const bool sprout_growth_registered = []() {
    register_scenario("vascular_sprout_growth", build_vascular_sprout_growth);
    return true;
}();

} // anonymous namespace
