// Test 7 — B Cell Lifecycle (Naive → Activated → Plasma)
//
// Validates the deterministic B cell state machine in bcell_state_step:
//   NAIVE → ACTIVATED:  requires has_antigen=1 AND (th_count + tfh_count) > 0
//   ACTIVATED → PLASMA: activation_timer >= PARAM_BCELL_ACTIVATION_TIMER * speedup
// Plus the plasma-cell secretion of ANTIBODY (CHEM=14).
//
// Setup: 31³ grid. Seed 25 BCELL_NAIVE on a 5×5 lattice at z=15 with
// has_antigen=1. Adjacent to each B cell (z=16), seed one T_CELL_EFF to
// provide th_count ≥ 1. Pre-set life=9999 on both to exclude apoptosis.
// Movement, division, ECM, recruitment are all OFF so positions stay fixed
// and state transitions are fully deterministic.
//
// Layers enabled:
//   neighbor_scan    — broadcast + scan (populates th_count on B cells)
//   state_transition — the actual state machine
//   chemical_sources — plasma cells secrete antibody
//   pde_solve        — antibody diffusion + decay
// All others OFF. In particular, division OFF so activated cells don't
// multiply — keeps cell count exactly 25 throughout and gives a clean
// count-vs-time profile.
//
// Expected timeline (deterministic given setup):
//   step 0:   NAIVE → ACTIVATED immediately (both preconditions met at scan)
//             activation_timer = 0 at end of step 0
//   step 1–39: ACTIVATED, timer counts 1..39
//   step 40:  ACTIVATED → PLASMA (timer=40 ≥ threshold=40, speedup=1.0 since
//             PARAM_BCELL_TLS_SPEEDUP=1.0 and no Tfh adjacent)
//   step 41+: PLASMA secretes antibody at PARAM_BCELL_ANTIBODY_RELEASE rate
//
// Pass criteria (evaluated in make_figures.py):
//   (1) At step 0 callback:   n_activated == 25, n_naive == 0, n_plasma == 0
//   (2) At step 39 callback:  n_activated == 25, n_plasma == 0
//   (3) At step 40 callback:  n_plasma == 25  (all differentiated this step)
//   (4) Monotonicity:         n_naive non-increasing, n_plasma non-decreasing
//   (5) Mass-balance on antibody (after step 40 settling):
//         Σ C_antibody ≈ N_plasma · S_per_cell / k_antibody
//       where S_per_cell = PARAM_BCELL_ANTIBODY_RELEASE / voxel_volume.
//
// Output files (in ../test/scenarios/bcell_lifecycle/outputs/):
//   params.csv         — runtime params (timer, rates, decay, voxel size)
//   state_counts.csv   — step, n_naive, n_activated, n_plasma, sum_antibody
//   per_cell.csv       — per-cell final state + activation_timer for debug

#include "../../test_harness.cuh"
#include "../../../core/common.cuh"
#include "../../../pde/pde_solver.cuh"
#include "../../../pde/pde_integration.cuh"

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <functional>
#include <memory>
#include <vector>
#include <string>

using namespace PDAC;
using namespace PDAC::test;

void register_scenario(const std::string& name,
    std::function<TestConfig(const std::string&)> builder);

namespace {

// Lattice geometry
constexpr int LATTICE_N     = 5;                 // 5×5 = 25 B-T pairs
constexpr int LATTICE_STEP  = 5;
constexpr int LATTICE_BASE  = 5;
constexpr int LATTICE_Z_B   = 15;                // B cells at z=15
constexpr int LATTICE_Z_T   = 16;                // T cells directly above
constexpr int N_PAIRS       = LATTICE_N * LATTICE_N;

constexpr int RUN_STEPS     = 60;                // enough past step 40

struct BCellStats {
    std::ofstream counts;
    std::ofstream params;
    std::ofstream per_cell;
};

static TestConfig build_bcell_lifecycle(const std::string& param_file) {
    TestConfig cfg;
    cfg.name = "bcell_lifecycle";
    cfg.grid_size = 31;
    cfg.voxel_size = 20.0f;
    cfg.steps = RUN_STEPS;
    cfg.seed = 42;
    cfg.param_file = param_file;

    // --- Layers: only what's needed for the state machine ---
    cfg.layers.ecm_update       = false;
    cfg.layers.recruitment      = false;
    cfg.layers.occupancy        = false;  // no movement → no occ grid needed
    cfg.layers.movement         = false;
    cfg.layers.neighbor_scan    = true;   // populates th_count for B cells
    cfg.layers.state_transition = true;   // the actual state machine
    cfg.layers.chemical_sources = true;   // plasma cells secrete antibody
    cfg.layers.pde_solve        = true;   // antibody diffusion + decay
    cfg.layers.pde_gradients    = false;  // no chemotaxis without movement
    cfg.layers.division         = false;  // keeps N_bcells == 25 throughout
    cfg.layers.qsp              = false;
    cfg.layers.abm_export       = false;

    // Disable the TLS plasma gate so the scenario tests the deterministic
    // timer-only path (no cluster, no Tfh — just antigen + Th help). A
    // separate scenario should cover the gated path end-to-end.
    cfg.bool_env_overrides.push_back({"PARAM_BCELL_PLASMA_REQUIRES_TLS", false});

    // --- Seed B cells (NAIVE, pre-loaded with antigen) + T cell helpers ---
    for (int j = 0; j < LATTICE_N; j++) {
        for (int i = 0; i < LATTICE_N; i++) {
            const int x = LATTICE_BASE + i * LATTICE_STEP;
            const int y = LATTICE_BASE + j * LATTICE_STEP;

            AgentSeed b;
            b.agent_type = AGENT_BCELL;
            b.x = x; b.y = y; b.z = LATTICE_Z_B;
            b.cell_state = BCELL_NAIVE;
            b.int_vars.push_back({"has_antigen", 1});
            b.int_vars.push_back({"life", 9999});
            b.int_vars.push_back({"activation_timer", 0});
            cfg.agents.push_back(b);

            AgentSeed t;
            t.agent_type = AGENT_TCELL;
            t.x = x; t.y = y; t.z = LATTICE_Z_T;
            t.cell_state = T_CELL_EFF;
            t.int_vars.push_back({"life", 9999});
            t.int_vars.push_back({"divide_cd", 9999});
            t.int_vars.push_back({"divide_limit", 0});
            cfg.agents.push_back(t);
        }
    }

    // --- Stats + CSV setup ---
    auto stats = std::make_shared<BCellStats>();
    const std::string out_dir = "../test/scenarios/" + cfg.name + "/outputs";
    std::filesystem::create_directories(out_dir);

    stats->params.open(out_dir + "/params.csv");
    stats->params << "key,value\n";

    stats->counts.open(out_dir + "/state_counts.csv");
    stats->counts << "step,n_naive,n_activated,n_plasma,sum_antibody\n";

    stats->per_cell.open(out_dir + "/per_cell.csv");
    stats->per_cell << "step,id,x,y,z,cell_state,activation_timer,has_antigen\n";

    cfg.step_callback = [stats, out_dir](flamegpu::CUDASimulation& sim,
                                         flamegpu::ModelDescription& model,
                                         unsigned int step)
    {
        const int gs = model.Environment().getProperty<int>("grid_size_x");
        const int V  = gs * gs * gs;

        // Dump runtime params once.
        if (step == 0) {
            auto env = model.Environment();
            auto put_f = [&](const char* k) {
                stats->params << k << "," << env.getProperty<float>(k) << "\n";
            };
            auto put_i = [&](const char* k) {
                stats->params << k << "," << env.getProperty<int>(k) << "\n";
            };
            put_f("PARAM_SEC_PER_SLICE");
            put_f("PARAM_VOXEL_SIZE_CM");
            put_i("PARAM_BCELL_ACTIVATION_TIMER");
            put_f("PARAM_BCELL_TLS_SPEEDUP");
            put_f("PARAM_BCELL_TFH_SPEEDUP");
            put_i("PARAM_BCELL_TLS_THRESHOLD");
            put_f("PARAM_BCELL_BREG_FRACTION");
            put_f("PARAM_BCELL_ANTIBODY_RELEASE");
            put_f("PARAM_BCELL_ACT_CXCL13_RELEASE");
            put_f("PARAM_BCELL_PLASMA_CXCL13_RELEASE");
            put_f("PARAM_BCELL_IL6_RELEASE");
            put_f("PARAM_ANTIBODY_DIFFUSIVITY");
            put_f("PARAM_ANTIBODY_DECAY_RATE");
            stats->params << "N_PAIRS," << N_PAIRS << "\n";
            stats->params << "LATTICE_Z_B," << LATTICE_Z_B << "\n";
            stats->params << "grid_size," << gs << "\n";
            stats->params.flush();
        }

        // Pull B cell population and count states.
        flamegpu::AgentVector pop(model.Agent(AGENT_BCELL));
        sim.getPopulationData(pop);

        int n_naive = 0, n_act = 0, n_plasma = 0;
        for (unsigned int i = 0; i < pop.size(); i++) {
            int cs = pop[i].getVariable<int>("cell_state");
            if (cs == BCELL_NAIVE)          n_naive++;
            else if (cs == BCELL_ACTIVATED) n_act++;
            else if (cs == BCELL_PLASMA)    n_plasma++;

            stats->per_cell << step << ","
                            << pop[i].getID() << ","
                            << pop[i].getVariable<int>("x") << ","
                            << pop[i].getVariable<int>("y") << ","
                            << pop[i].getVariable<int>("z") << ","
                            << cs << ","
                            << pop[i].getVariable<int>("activation_timer") << ","
                            << pop[i].getVariable<int>("has_antigen") << "\n";
        }

        // Sum antibody concentration across the full grid.
        std::vector<float> buf(V);
        PDAC::g_pde_solver->get_concentrations(buf.data(), CHEM_ANTIBODY);
        double sum_ab = 0.0;
        for (int i = 0; i < V; i++) sum_ab += buf[i];

        stats->counts << step << "," << n_naive << "," << n_act << ","
                      << n_plasma << "," << sum_ab << "\n";

        if (step == 0 || step == 1 || step == 39 || step == 40 ||
            step == 41 || step + 1 == RUN_STEPS) {
            std::cout << "  [step " << step << "] "
                      << "naive=" << n_naive
                      << " activated=" << n_act
                      << " plasma=" << n_plasma
                      << " Σantibody=" << sum_ab << std::endl;
        }

        if (step + 1 == RUN_STEPS) {
            stats->counts.flush();
            stats->per_cell.flush();
            std::cout << "\n==================== B cell lifecycle ====================\n"
                      << "  " << pop.size() << " B cells tracked over "
                      << RUN_STEPS << " steps\n"
                      << "  Outputs: " << out_dir << "/\n"
                      << "    params.csv        (BCell lifecycle params)\n"
                      << "    state_counts.csv  (n_naive, n_activated, n_plasma per step)\n"
                      << "    per_cell.csv      (per-cell state + timer per step)\n"
                      << "  Run make_figures.py for stacked-area + pass/fail summary.\n"
                      << "============================================================\n"
                      << std::endl;
        }
    };

    return cfg;
}

static const bool bcell_lifecycle_registered = []() {
    register_scenario("bcell_lifecycle", build_bcell_lifecycle);
    return true;
}();

} // anonymous namespace
