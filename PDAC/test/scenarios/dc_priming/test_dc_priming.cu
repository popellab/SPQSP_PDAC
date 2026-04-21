// Test 8 — DC Priming Cascade (naive CD8 → CYT)
//
// Validates the DC priming pathway in tcell_state_step (NAIVE branch):
//   p = 1 - exp(-k_CD8_act · H · Δt)
// where H = n_sites·dc_cdc1 / (n_sites·dc_cdc1 + n_local + PARAM_CELL)
// and dc_cdc1 is the Moore-neighborhood count of mature cDC1 DCs.
// On success: NAIVE → T_CELL_CYT, divide_limit += PARAM_PRIME_DIV_BURST,
// EVT_PRIME_CD8 incremented.
//
// These params (PARAM_DC_PRIME_K_CD8, PARAM_DC_N_SITES) were just wired
// through the QSP codegen pipeline — this test is also their first exercise.
//
// Setup: 31³ grid. 5×5 lattice = 25 (naive T, mature cDC1 DC) pairs.
// T cells at z=15, DCs at z=16 (directly above, Moore-adjacent). Both pre-set
// life=9999 and DC presentation_capacity=9999 so nobody dies or exhausts.
// Movement, division, ECM, recruitment, PDE all OFF → positions fixed,
// neighbor relationships deterministic.
//
// Layers enabled:
//   neighbor_scan    — populates neighbor_dc_cdc1_mature_count on T cells
//   state_transition — tcell_state_step priming branch; dc_state_step life/cap
// Everything else OFF.
//
// Expected timeline (k_CD8_act ≈ 2.66e-4/s from QSP, H ≈ 0.9, Δt = 21600s):
//   p_per_step ≈ 1 - exp(-5.2) ≈ 0.994  →  essentially instant on contact.
//   step 0: ~all 25 T cells prime NAIVE → CYT, EVT_PRIME_CD8 ≈ 25
//   step 1-9: any stragglers primed with same 99.4% prob
//
// Pass criteria (evaluated in make_figures.py):
//   (1) At step 0 callback: n_naive ≤ 2  (allow rare non-primed stragglers)
//   (2) At step 0 callback: n_cyt + n_naive == 25  (no spurious deaths yet)
//   (3) At step 0 callback: EVT_PRIME_CD8 == n_cyt  (one event per transition)
//   (4) DC count at step 0 == 25  (no DC deaths)
//   (5) divide_limit on primed T cells == PARAM_PRIME_DIV_BURST (burst applied)
//
// Note: Only step-0 counts are checked. On subsequent steps, ~40% of newly
// primed CYT cells die because the default PARAM_T_CELL_LIFE_MEAN_SLICE (~8)
// and PARAM_TCELL_LIFESPAN_SD_SLICE (~40) produce a normal draw that puts
// a large fraction of cells at life ≤ 1 — that's a lifespan-calibration
// issue (mean/SD mismatch), orthogonal to the priming mechanism this test
// is exercising.
//
// Output files (in ../test/scenarios/dc_priming/outputs/):
//   params.csv         — runtime params used by priming formula
//   state_counts.csv   — step, n_naive, n_cyt, n_dc, evt_prime_cd8
//   per_cell.csv       — per-T-cell final state for debug

#include "../../test_harness.cuh"
#include "../../../core/common.cuh"

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

constexpr int LATTICE_N     = 5;
constexpr int LATTICE_STEP  = 5;
constexpr int LATTICE_BASE  = 5;
constexpr int LATTICE_Z_T   = 15;
constexpr int LATTICE_Z_DC  = 16;
constexpr int N_PAIRS       = LATTICE_N * LATTICE_N;

constexpr int RUN_STEPS     = 2;

struct DCPrimingStats {
    std::ofstream counts;
    std::ofstream params;
    std::ofstream per_cell;
};

static TestConfig build_dc_priming(const std::string& param_file) {
    TestConfig cfg;
    cfg.name = "dc_priming";
    cfg.grid_size = 31;
    cfg.voxel_size = 20.0f;
    cfg.steps = RUN_STEPS;
    cfg.seed = 42;
    cfg.param_file = param_file;

    // Layers: only scan + state machine.
    cfg.layers.ecm_update       = false;
    cfg.layers.recruitment      = false;
    cfg.layers.occupancy        = false;
    cfg.layers.movement         = false;
    cfg.layers.neighbor_scan    = true;
    cfg.layers.state_transition = true;
    cfg.layers.chemical_sources = false;
    cfg.layers.pde_solve        = false;
    cfg.layers.pde_gradients    = false;
    cfg.layers.division         = false;
    cfg.layers.qsp              = false;
    cfg.layers.abm_export       = false;

    // Seed pairs: naive CD8 T at z=15, mature cDC1 DC at z=16 (directly above).
    for (int j = 0; j < LATTICE_N; j++) {
        for (int i = 0; i < LATTICE_N; i++) {
            const int x = LATTICE_BASE + i * LATTICE_STEP;
            const int y = LATTICE_BASE + j * LATTICE_STEP;

            AgentSeed t;
            t.agent_type = AGENT_TCELL;
            t.x = x; t.y = y; t.z = LATTICE_Z_T;
            t.cell_state = T_CELL_NAIVE;
            t.int_vars.push_back({"life", 9999});
            t.int_vars.push_back({"divide_cd", 9999});
            t.int_vars.push_back({"divide_limit", 0});
            cfg.agents.push_back(t);

            AgentSeed d;
            d.agent_type = AGENT_DC;
            d.x = x; d.y = y; d.z = LATTICE_Z_DC;
            d.cell_state = DC_MATURE;
            d.int_vars.push_back({"dc_subtype", DC_CDC1});
            d.int_vars.push_back({"life", 9999});
            d.int_vars.push_back({"presentation_capacity", 9999});
            cfg.agents.push_back(d);
        }
    }

    auto stats = std::make_shared<DCPrimingStats>();
    const std::string out_dir = "../test/scenarios/" + cfg.name + "/outputs";
    std::filesystem::create_directories(out_dir);

    stats->params.open(out_dir + "/params.csv");
    stats->params << "key,value\n";

    stats->counts.open(out_dir + "/state_counts.csv");
    stats->counts << "step,n_naive,n_cyt,n_eff,n_dc,evt_prime_cd8\n";

    stats->per_cell.open(out_dir + "/per_cell.csv");
    stats->per_cell << "step,id,x,y,z,cell_state,divide_limit\n";

    cfg.step_callback = [stats, out_dir](flamegpu::CUDASimulation& sim,
                                         flamegpu::ModelDescription& model,
                                         unsigned int step)
    {
        auto env = model.Environment();

        if (step == 0) {
            auto put_f = [&](const char* k) {
                stats->params << k << "," << env.getProperty<float>(k) << "\n";
            };
            auto put_i = [&](const char* k) {
                stats->params << k << "," << env.getProperty<int>(k) << "\n";
            };
            put_f("PARAM_SEC_PER_SLICE");
            put_f("PARAM_DC_PRIME_K_CD8");
            put_f("PARAM_DC_N_SITES");
            put_f("PARAM_CELL");
            put_i("PARAM_PRIME_DIV_BURST");
            stats->params << "N_PAIRS," << N_PAIRS << "\n";
            stats->params << "RUN_STEPS," << RUN_STEPS << "\n";
            stats->params.flush();
        }

        // Count T cell states.
        flamegpu::AgentVector tpop(model.Agent(AGENT_TCELL));
        sim.getPopulationData(tpop);
        int n_naive = 0, n_cyt = 0, n_eff = 0;
        for (unsigned int i = 0; i < tpop.size(); i++) {
            int cs = tpop[i].getVariable<int>("cell_state");
            if (cs == T_CELL_NAIVE)     n_naive++;
            else if (cs == T_CELL_CYT)  n_cyt++;
            else if (cs == T_CELL_EFF)  n_eff++;

            stats->per_cell << step << ","
                            << tpop[i].getID() << ","
                            << tpop[i].getVariable<int>("x") << ","
                            << tpop[i].getVariable<int>("y") << ","
                            << tpop[i].getVariable<int>("z") << ","
                            << cs << ","
                            << tpop[i].getVariable<int>("divide_limit") << "\n";
        }

        flamegpu::AgentVector dcpop(model.Agent(AGENT_DC));
        sim.getPopulationData(dcpop);
        const int n_dc = dcpop.size();

        // Pull EVT_PRIME_CD8 from device event counter.
        unsigned int* dev_evts = reinterpret_cast<unsigned int*>(
            env.getProperty<uint64_t>("event_counters_ptr"));
        unsigned int evt_buf[ABM_EVENT_COUNTER_SIZE];
        cudaMemcpy(evt_buf, dev_evts,
                   ABM_EVENT_COUNTER_SIZE * sizeof(unsigned int),
                   cudaMemcpyDeviceToHost);
        const unsigned int prime_cd8 = evt_buf[EVT_PRIME_CD8];

        stats->counts << step << "," << n_naive << "," << n_cyt << ","
                      << n_eff << "," << n_dc << "," << prime_cd8 << "\n";

        if (step == 0 || step == 1 || step + 1 == RUN_STEPS) {
            std::cout << "  [step " << step << "] "
                      << "naive=" << n_naive
                      << " cyt=" << n_cyt
                      << " DC=" << n_dc
                      << " Σprime=" << prime_cd8 << std::endl;
        }

        if (step + 1 == RUN_STEPS) {
            stats->counts.flush();
            stats->per_cell.flush();
            std::cout << "\n==================== DC priming cascade ====================\n"
                      << "  " << N_PAIRS << " (naive T, mature cDC1) pairs over "
                      << RUN_STEPS << " steps\n"
                      << "  Outputs: " << out_dir << "/\n"
                      << "    params.csv       (priming formula inputs)\n"
                      << "    state_counts.csv (naive/cyt + EVT_PRIME_CD8 per step)\n"
                      << "    per_cell.csv     (per-T-cell state + divide_limit)\n"
                      << "  Run make_figures.py for stacked-area + pass/fail.\n"
                      << "============================================================\n"
                      << std::endl;
        }
    };

    return cfg;
}

static const bool dc_priming_registered = []() {
    register_scenario("dc_priming", build_dc_priming);
    return true;
}();

} // anonymous namespace
