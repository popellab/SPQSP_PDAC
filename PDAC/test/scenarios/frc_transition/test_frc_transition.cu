// Test 10 — FRC Transition (Smoking Gun for New Code)
//
// Validates iCAF→FRC transition gating on LTβ-competent lymphocyte density
// with the Treg exclusion.
//
//   In fib_state_step (FIB_ICAF branch):
//     if (neighbor_ltb_lymph_count >= PARAM_FIB_FRC_LYMPH_THRESHOLD) dwell++
//     else                                                            dwell = 0
//     if (dwell >= PARAM_FIB_FRC_DWELL_STEPS) → state = FIB_FRC
//
//   neighbor_ltb_lymph_count (counted in fib_scan_neighbors):
//     LTβ⁺: BCELL_NAIVE/ACTIVATED, T_CELL_EFF/CYT, TCD4_TH/TFH
//     LTβ⁻: BCELL_PLASMA, T_CELL_NAIVE/SUPP, TCD4_TREG, TCD4_NAIVE
//
// Three sub-scenarios (all 21³, 1 FIB_ICAF at center, 20 steps):
//   frc_pos  — 10 BCELL_ACTIVATED + 10 T_CELL_EFF around iCAF (n_ltb=20 ≥ 8)
//              → expect state flips to FRC at step = DWELL_STEPS - 1 (= 3)
//   frc_treg — 20 TCD4_TREG packed around iCAF (n_ltb = 0)
//              → expect counter stays 0, state stays ICAF (Treg excluded)
//   frc_sub  — 3 BCELL_ACTIVATED + 3 T_CELL_EFF (n_ltb = 6 < 8)
//              → expect counter stays 0, state stays ICAF
//
// Layers: neighbor_scan + state_transition ON. Movement/division/PDE OFF.
// Also disable lifespan death by pre-seeding life=9999 on all agents.
//
// Output (per sub-scenario, in outputs_<name>/):
//   trajectory.csv — step, fib_state, frc_dwell_counter, n_ltb (post-scan)
//   params.csv     — FRC threshold + dwell steps

#include "../../test_harness.cuh"
#include "../../../core/common.cuh"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

using namespace PDAC;
using namespace PDAC::test;

void register_scenario(const std::string& name,
    std::function<TestConfig(const std::string&)> builder);

namespace {

constexpr int GRID       = 21;
constexpr int CENTER     = GRID / 2;  // = 10
constexpr int RUN_STEPS  = 20;

enum class Mode { POSITIVE, TREG, SUB };

// Moore-26 offsets around (0,0,0). We'll fill from the start up to N needed.
// Order: face neighbors first (6), then edge (12), then corner (8) — all within
// |dx|,|dy|,|dz| ≤ 1 and not all zero. fib_scan_neighbors filters Moore anyway.
static const int MOORE_OFFSETS[26][3] = {
    // Face (6)
    {-1, 0, 0}, { 1, 0, 0}, { 0,-1, 0}, { 0, 1, 0}, { 0, 0,-1}, { 0, 0, 1},
    // Edge (12)
    {-1,-1, 0}, {-1, 1, 0}, { 1,-1, 0}, { 1, 1, 0},
    {-1, 0,-1}, {-1, 0, 1}, { 1, 0,-1}, { 1, 0, 1},
    { 0,-1,-1}, { 0,-1, 1}, { 0, 1,-1}, { 0, 1, 1},
    // Corner (8)
    {-1,-1,-1}, {-1,-1, 1}, {-1, 1,-1}, {-1, 1, 1},
    { 1,-1,-1}, { 1,-1, 1}, { 1, 1,-1}, { 1, 1, 1},
};

static void seed_tcell_eff(TestConfig& cfg, int x, int y, int z) {
    AgentSeed s;
    s.agent_type = AGENT_TCELL;
    s.x = x; s.y = y; s.z = z;
    s.cell_state = T_CELL_EFF;
    s.int_vars.push_back({"life", 9999});
    s.int_vars.push_back({"divide_limit", 0});
    s.int_vars.push_back({"divide_flag", 0});
    s.int_vars.push_back({"divide_cd", 9999});
    s.float_vars.push_back({"adh_p_move", 1.0f});
    s.float_vars.push_back({"hypoxia_kill_factor", 1.0f});
    cfg.agents.push_back(s);
}

static void seed_bcell_act(TestConfig& cfg, int x, int y, int z) {
    AgentSeed s;
    s.agent_type = AGENT_BCELL;
    s.x = x; s.y = y; s.z = z;
    s.cell_state = BCELL_ACTIVATED;
    s.int_vars.push_back({"life", 9999});
    s.int_vars.push_back({"has_antigen", 1});
    s.int_vars.push_back({"divide_limit", 0});
    s.int_vars.push_back({"divide_flag", 0});
    s.int_vars.push_back({"divide_cd", 9999});
    s.float_vars.push_back({"adh_p_move", 1.0f});
    cfg.agents.push_back(s);
}

static void seed_treg(TestConfig& cfg, int x, int y, int z) {
    AgentSeed s;
    s.agent_type = AGENT_TREG;
    s.x = x; s.y = y; s.z = z;
    s.cell_state = TCD4_TREG;
    s.int_vars.push_back({"life", 9999});
    s.int_vars.push_back({"divide_limit", 0});
    s.int_vars.push_back({"divide_flag", 0});
    s.int_vars.push_back({"divide_cd", 9999});
    s.float_vars.push_back({"adh_p_move", 1.0f});
    cfg.agents.push_back(s);
}

struct FRCStats {
    std::ofstream traj;
    std::ofstream params;
};

static TestConfig build(const std::string& param_file,
                        const std::string& scenario_tag,
                        Mode mode)
{
    TestConfig cfg;
    cfg.name = "frc_" + scenario_tag;
    cfg.grid_size = GRID;
    cfg.voxel_size = 20.0f;
    cfg.steps = RUN_STEPS;
    cfg.seed = 42;
    cfg.param_file = param_file;

    // Minimal layer set: neighbor_scan (to populate neighbor_ltb_lymph_count)
    // + state_transition (to evaluate FRC gate). All else OFF so positions /
    // populations are frozen and fib_state_step is the only thing that can
    // flip the iCAF's cell_state.
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

    // --- Seed the iCAF at grid center -----------------------------------
    AgentSeed icaf;
    icaf.agent_type = AGENT_FIBROBLAST;
    icaf.x = CENTER; icaf.y = CENTER; icaf.z = CENTER;
    icaf.cell_state = FIB_ICAF;
    icaf.int_vars.push_back({"life", 9999});
    icaf.int_vars.push_back({"divide_cooldown", 9999});
    icaf.int_vars.push_back({"divide_count", 0});
    icaf.int_vars.push_back({"divide_flag", 0});
    icaf.int_vars.push_back({"frc_dwell_counter", 0});
    icaf.float_vars.push_back({"adh_p_move", 1.0f});
    cfg.agents.push_back(icaf);

    // --- Seed lymphocytes in Moore-26 neighbor voxels -------------------
    auto put_at = [&](int slot, const std::function<void(TestConfig&,int,int,int)>& fn) {
        const int ox = MOORE_OFFSETS[slot][0];
        const int oy = MOORE_OFFSETS[slot][1];
        const int oz = MOORE_OFFSETS[slot][2];
        fn(cfg, CENTER + ox, CENTER + oy, CENTER + oz);
    };

    if (mode == Mode::POSITIVE) {
        // 10 BCELL_ACTIVATED + 10 T_CELL_EFF in distinct Moore voxels.
        for (int s = 0; s < 10; s++)  put_at(s, seed_bcell_act);
        for (int s = 10; s < 20; s++) put_at(s, seed_tcell_eff);
    } else if (mode == Mode::TREG) {
        // 20 TCD4_TREG — LTβ⁻, should NOT satisfy the gate.
        for (int s = 0; s < 20; s++) put_at(s, seed_treg);
    } else { // SUB
        // 3 BCELL_ACTIVATED + 3 T_CELL_EFF — 6 < threshold (8).
        for (int s = 0; s < 3; s++)       put_at(s, seed_bcell_act);
        for (int s = 3; s < 6; s++)       put_at(s, seed_tcell_eff);
    }

    auto stats = std::make_shared<FRCStats>();
    const std::string out_dir = "../test/scenarios/frc_transition/outputs_" + scenario_tag;
    std::filesystem::create_directories(out_dir);

    stats->params.open(out_dir + "/params.csv");
    stats->params << "key,value\n";

    stats->traj.open(out_dir + "/trajectory.csv");
    stats->traj << "step,fib_state,frc_dwell_counter,neighbor_ltb_count\n";

    cfg.step_callback = [stats, out_dir](flamegpu::CUDASimulation& sim,
                                         flamegpu::ModelDescription& model,
                                         unsigned int step)
    {
        if (step == 0) {
            auto env = model.Environment();
            stats->params << "PARAM_FIB_FRC_LYMPH_THRESHOLD,"
                          << env.getProperty<float>("PARAM_FIB_FRC_LYMPH_THRESHOLD") << "\n";
            stats->params << "PARAM_FIB_FRC_DWELL_STEPS,"
                          << env.getProperty<float>("PARAM_FIB_FRC_DWELL_STEPS") << "\n";
            stats->params.flush();
        }

        flamegpu::AgentVector fpop(model.Agent(AGENT_FIBROBLAST));
        sim.getPopulationData(fpop);
        // Only one fibroblast — the iCAF at center. Grab state + counter.
        int fib_state = -1, dwell = -1, n_ltb = -1;
        for (unsigned int i = 0; i < fpop.size(); i++) {
            fib_state = fpop[i].getVariable<int>("cell_state");
            dwell     = fpop[i].getVariable<int>("frc_dwell_counter");
            n_ltb     = fpop[i].getVariable<int>("neighbor_ltb_lymph_count");
        }
        stats->traj << step << "," << fib_state << "," << dwell << ","
                    << n_ltb << "\n";

        if (step == 0 || step + 1 == RUN_STEPS || (step + 1) % 5 == 0) {
            std::cout << "  [step " << step << "] state=" << fib_state
                      << " dwell=" << dwell << " n_ltb=" << n_ltb << std::endl;
        }

        if (step + 1 == RUN_STEPS) stats->traj.flush();
    };

    return cfg;
}

static const bool frc_pos_registered = []() {
    register_scenario("frc_pos", [](const std::string& pf) {
        return build(pf, "pos", Mode::POSITIVE);
    });
    return true;
}();

static const bool frc_treg_registered = []() {
    register_scenario("frc_treg", [](const std::string& pf) {
        return build(pf, "treg", Mode::TREG);
    });
    return true;
}();

static const bool frc_sub_registered = []() {
    register_scenario("frc_sub", [](const std::string& pf) {
        return build(pf, "sub", Mode::SUB);
    });
    return true;
}();

} // anonymous namespace
