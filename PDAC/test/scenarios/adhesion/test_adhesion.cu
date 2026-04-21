// Test 3 — Adhesion (slope sweep)
//
// Verifies that the adhesion matrix produces the analytically expected
// per-neighbor reduction in move probability:
//
//   p_move = max(0, 1 - sum_j M[i][j] * n_j)
//
// Uses CANCER_STEM as the "focal" cell (the one we measure) and CANCER_PROG
// as "decoy" neighbors. XML gives M[cancer_stem][cancer_prog] = 0.15, so the
// sweep k = 0..7 yields eight well-separated, non-saturated points:
//
//     k |  p_move expected
//     --+------------------
//     0 |  1.00
//     1 |  0.85
//     2 |  0.70
//     3 |  0.55
//     4 |  0.40
//     5 |  0.25
//     6 |  0.10
//     7 |  0.00  (saturated)
//
// Measurement protocol
//   * Every focal and every decoy starts with adh_p_move = 0 so nothing
//     moves on step 0.
//   * The scan_neighbors layer runs at the END of step 0 and writes the
//     true adh_p_move from the neighbour counts into each agent.
//   * On step 1 the focal moves with its true p_move. We record whether
//     each focal changed voxel between its seed position and its step-1
//     position. That single binary outcome per focal is the observation.
//   * 80 independent pockets per k → 80 Bernoulli draws → σ ≈ 0.055 at
//     p = 0.5. Tolerance 0.10 is comfortable.
//
// Pocket layout
//   * 151³ grid with pockets on a cubic lattice spaced 7 voxels apart.
//     Moore-1 reach is 1 voxel, so pockets never contaminate each other.
//   * 8 k values × 80 pockets = 640 pockets. We scan through lattice
//     positions in row-major order and assign (k, trial) to each.
//   * The k decoys sit at the first k entries of a fixed Moore-26 offset
//     list (6 faces, then 12 edges, then 8 corners) relative to the focal.
//
// Layers: occupancy + movement + neighbor_scan only.

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
#include <set>
#include <tuple>
#include <cmath>
#include <vector>
#include <array>

using namespace PDAC;
using namespace PDAC::test;

void register_scenario(const std::string& name,
    std::function<TestConfig(const std::string&)> builder);

namespace {

constexpr int   GRID        = 151;
constexpr int   POCKET_STEP = 7;       // spacing between pocket centres (voxels)
constexpr int   MARGIN      = 4;       // leave headroom at grid edges
constexpr int   K_MAX       = 7;       // sweep k = 0 .. K_MAX inclusive
constexpr int   PER_K       = 80;      // pockets per k
constexpr float M_COEFF     = 0.15f;   // M[stem][prog] from XML
constexpr float TOL         = 0.10f;

// Moore-26 offsets: 6 faces, 12 edges, 8 corners (in that order, so k<=6 stays
// on face neighbours which are mutually non-Moore-adjacent).
static const std::array<std::array<int,3>, 26> MOORE_OFFSETS = {{
    // faces
    { 1, 0, 0}, {-1, 0, 0}, { 0, 1, 0}, { 0,-1, 0}, { 0, 0, 1}, { 0, 0,-1},
    // edges
    { 1, 1, 0}, { 1,-1, 0}, {-1, 1, 0}, {-1,-1, 0},
    { 1, 0, 1}, { 1, 0,-1}, {-1, 0, 1}, {-1, 0,-1},
    { 0, 1, 1}, { 0, 1,-1}, { 0,-1, 1}, { 0,-1,-1},
    // corners
    { 1, 1, 1}, { 1, 1,-1}, { 1,-1, 1}, { 1,-1,-1},
    {-1, 1, 1}, {-1, 1,-1}, {-1,-1, 1}, {-1,-1,-1},
}};

struct AdhesionStats {
    // Focal cells: position -> k
    std::map<std::tuple<int,int,int>, int> focal_k_at_seed;
    // Track each focal by ID once we see it after step 0.
    std::map<int, int> focal_k;                    // agent id -> k
    std::map<int, std::tuple<int,int,int>> seed_of; // agent id -> seed pos

    // Per-k counters (step-1 move rate)
    std::array<int, K_MAX + 1> step1_total = {};
    std::array<int, K_MAX + 1> step1_moved = {};

    std::ofstream result;
};

static TestConfig build_adhesion(const std::string& param_file) {
    TestConfig cfg;
    cfg.name        = "adhesion";
    cfg.grid_size   = GRID;
    cfg.voxel_size  = 20.0f;
    cfg.steps       = 2;        // step 0: scan populates adh_p_move; step 1: measure
    cfg.seed        = 42;
    cfg.param_file  = param_file;

    cfg.layers.ecm_update       = false;
    cfg.layers.recruitment      = false;
    cfg.layers.occupancy        = true;
    cfg.layers.movement         = true;
    cfg.layers.neighbor_scan    = true;
    cfg.layers.state_transition = false;
    cfg.layers.chemical_sources = false;
    cfg.layers.pde_solve        = false;
    cfg.layers.pde_gradients    = false;
    cfg.layers.division         = false;
    cfg.layers.qsp              = false;
    cfg.layers.abm_export       = false;

    // One move substep per ABM step -> at most 1 voxel displacement per step,
    // clean binary "moved vs stayed" signal.
    cfg.int_env_overrides.push_back({"PARAM_CANCER_MOVE_STEPS_STEM", 1});
    cfg.int_env_overrides.push_back({"PARAM_CANCER_MOVE_STEPS",      1});

    // Flat ECM so porosity = 1 everywhere.
    cfg.ecm.density   = uniform_field(0.0f);
    cfg.ecm.crosslink = uniform_field(0.0f);
    cfg.ecm.floor     = uniform_field(0.0f);

    auto stats = std::make_shared<AdhesionStats>();

    // Lay out pockets on a cubic lattice.
    const int slots_per_axis = (GRID - 2 * MARGIN) / POCKET_STEP;
    const int total_slots    = slots_per_axis * slots_per_axis * slots_per_axis;
    const int needed_pockets = (K_MAX + 1) * PER_K;
    if (total_slots < needed_pockets) {
        std::cerr << "[adhesion] Only " << total_slots
                  << " lattice slots for " << needed_pockets
                  << " pockets — increase grid or reduce PER_K.\n";
    }

    auto add_cell = [&](int x, int y, int z, int state) {
        AgentSeed s;
        s.agent_type = AGENT_CANCER_CELL;
        s.x = x; s.y = y; s.z = z;
        s.cell_state = state;
        // Freeze life-cycle so only adhesion/movement act on these cells.
        s.int_vars.push_back({"divideCountRemaining", 0});
        s.int_vars.push_back({"divideFlag", 0});
        s.int_vars.push_back({"divideCD", 9999});
        s.int_vars.push_back({"life", 9999});
        // No motion on step 0 — scan will set the real adh_p_move afterwards.
        s.float_vars.push_back({"adh_p_move", 0.0f});
        cfg.agents.push_back(s);
    };

    int placed = 0;
    for (int iz = 0; iz < slots_per_axis && placed < needed_pockets; iz++) {
        for (int iy = 0; iy < slots_per_axis && placed < needed_pockets; iy++) {
            for (int ix = 0; ix < slots_per_axis && placed < needed_pockets; ix++) {
                const int k = placed / PER_K;
                const int fx = MARGIN + POCKET_STEP * ix;
                const int fy = MARGIN + POCKET_STEP * iy;
                const int fz = MARGIN + POCKET_STEP * iz;

                // Focal cell (STEM).
                add_cell(fx, fy, fz, CANCER_STEM);
                stats->focal_k_at_seed[{fx, fy, fz}] = k;

                // k decoy neighbours (PROG).
                for (int j = 0; j < k; j++) {
                    const auto& off = MOORE_OFFSETS[j];
                    add_cell(fx + off[0], fy + off[1], fz + off[2], CANCER_PROGENITOR);
                }

                placed++;
            }
        }
    }

    std::cout << "[adhesion] Seeded " << placed << " pockets — "
              << PER_K << " each for k = 0 .. " << K_MAX << "\n";

    const std::string out_dir = "../test/scenarios/" + cfg.name + "/outputs";
    std::filesystem::create_directories(out_dir);
    stats->result.open(out_dir + "/slope_sweep.csv");
    stats->result << "k,expected_p,observed_p,n_focals,err\n";

    cfg.step_callback = [stats](flamegpu::CUDASimulation& sim,
                                flamegpu::ModelDescription& model,
                                unsigned int step)
    {
        flamegpu::AgentVector pop(model.Agent(AGENT_CANCER_CELL));
        sim.getPopulationData(pop);

        if (step == 0) {
            // Identify every focal cell by its seed position (it hasn't moved
            // yet because adh_p_move = 0 suppressed step-0 motion) and stash
            // its id + k for the step-1 measurement.
            for (unsigned int i = 0; i < pop.size(); i++) {
                const int cs = pop[i].getVariable<int>("cell_state");
                if (cs != CANCER_STEM) continue;
                const int x = pop[i].getVariable<int>("x");
                const int y = pop[i].getVariable<int>("y");
                const int z = pop[i].getVariable<int>("z");
                auto it = stats->focal_k_at_seed.find({x, y, z});
                if (it == stats->focal_k_at_seed.end()) continue;
                const int id = pop[i].getID();
                stats->focal_k[id] = it->second;
                stats->seed_of[id] = {x, y, z};
            }
            std::cout << "  [step 0] identified " << stats->focal_k.size()
                      << " focal cells (adh_p_move populated by scan)\n";
            return;
        }

        if (step == 1) {
            // Compare each focal's position to its seed. Since PARAM_CANCER_
            // MOVE_STEPS_STEM = 1 there is at most one move substep in step 1,
            // so any mismatch = moved exactly once this step.
            for (unsigned int i = 0; i < pop.size(); i++) {
                const int cs = pop[i].getVariable<int>("cell_state");
                if (cs != CANCER_STEM) continue;
                const int id = pop[i].getID();
                auto it = stats->focal_k.find(id);
                if (it == stats->focal_k.end()) continue;
                const int k = it->second;
                const auto [sx, sy, sz] = stats->seed_of[id];
                const int x = pop[i].getVariable<int>("x");
                const int y = pop[i].getVariable<int>("y");
                const int z = pop[i].getVariable<int>("z");
                const bool moved = (x != sx) || (y != sy) || (z != sz);
                stats->step1_total[k]++;
                if (moved) stats->step1_moved[k]++;
            }

            // Summary.
            std::cout << "\n==================== Adhesion slope sweep ==================\n"
                      << "  M[stem][prog] = " << M_COEFF
                      << "   tolerance = " << TOL << "\n"
                      << "    k  | expected   observed   n    |err|\n"
                      << "   ----+-----------------------------------\n";
            bool pass = true;
            for (int k = 0; k <= K_MAX; k++) {
                const int n = stats->step1_total[k];
                const float obs = (n > 0) ? float(stats->step1_moved[k]) / n : 0.0f;
                const float exp = std::max(0.0f, 1.0f - M_COEFF * k);
                const float err = std::abs(obs - exp);
                if (err > TOL) pass = false;
                std::cout << "    " << k << "  |   " << exp
                          << "      " << obs
                          << "    " << n
                          << "    " << err << "\n";
                stats->result << k << "," << exp << "," << obs
                              << "," << n << "," << err << "\n";
            }
            std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << "\n"
                      << "============================================================\n"
                      << std::endl;
            stats->result.flush();
        }
    };

    return cfg;
}

static const bool adhesion_registered = []() {
    register_scenario("adhesion", build_adhesion);
    return true;
}();

} // anonymous namespace
