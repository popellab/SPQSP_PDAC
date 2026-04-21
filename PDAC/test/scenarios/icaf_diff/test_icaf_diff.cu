// Test 9 — iCAF / myCAF / Both Differentiation
//
// Validates the FIB_QUIESCENT → {myCAF, iCAF} activation logic in
// fib_state_step:
//   p_myCAF = k_mycaf · H_TGFb(TGFb_50_CAF_act)
//   p_iCAF  = k_icaf  · H_IL1(IL1_50, n_IL1) · (1 - H_TGFb(TGFb_50_IL1R1))
//                      · (1 + f_IL6 · H_IL6(IL6_50_iCAF))
// where H_X = Hill(X, EC50, n).
//
// Three sub-scenarios, each registers independently:
//   icaf_diff_il1   — IL1=100·EC50, TGFB=0    → expect ~all iCAF
//   icaf_diff_tgfb  — TGFB=100·EC50, IL1=0   → expect ~all myCAF
//   icaf_diff_both  — both high                → expect ~all myCAF
//                                                (TGFB fully suppresses iCAF)
//
// Setup (all three): 21³ grid, 100 FIB_QUIESCENT at z=10 on a 10×10 lattice.
// Movement/division OFF; neighbor_scan OFF; PDE solve OFF (fields pinned).
// 100 ABM steps each. With k_myCAF≈0.1/step and k_iCAF≈0.05/step at
// saturation, all three runs converge in ≤60 steps.
//
// Per-step counts of {quiescent, myCAF, iCAF} written to outputs/*/counts.csv;
// make_figures.py draws trajectories + final-state bar chart and evaluates
// the three biological predictions.

#include "../../test_harness.cuh"
#include "../../../core/common.cuh"
#include "../../../pde/pde_solver.cuh"

#include <cuda_runtime.h>
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

constexpr int GRID         = 21;
constexpr int LATTICE_N    = 10;
constexpr int LATTICE_BASE = 5;
constexpr int LATTICE_Z    = 10;
constexpr int N_FIBS       = LATTICE_N * LATTICE_N;
constexpr int RUN_STEPS    = 100;

// High = 100× EC50 (saturating). EC50 values (in nM) from the derived
// params block; conservative multiplier so Hill ≈ 1 even under noise.
// IL1_50 = P_IL1_50  * 1e6   ≈ 0.01 * 1e6 = 1e4  nM
// TGFb_50 = P_TGFb_50_CAF_act * 1e6 ≈ 0.07 * 1e6 = 7e4 nM
// These are large but the fib_state_step code reads nM from PDE directly, so
// a "saturating" value just needs to be large vs EC50. 1e7 works.
constexpr float SAT_CONC   = 1.0e7f;
constexpr float ZERO_CONC  = 0.0f;

struct ICAFStats {
    std::ofstream counts;
};

enum class Pinning { IL1_ONLY, TGFB_ONLY, BOTH };

static TestConfig build(const std::string& param_file,
                        const std::string& scenario_name,
                        Pinning pin)
{
    TestConfig cfg;
    cfg.name = scenario_name;
    cfg.grid_size = GRID;
    cfg.voxel_size = 20.0f;
    cfg.steps = RUN_STEPS;
    cfg.seed = 42;
    cfg.param_file = param_file;

    // Only run state_transition. Movement/division/neighbor_scan off.
    // PDE solve off so pinned fields persist exactly.
    cfg.layers.ecm_update       = false;
    cfg.layers.recruitment      = false;
    cfg.layers.occupancy        = false;
    cfg.layers.movement         = false;
    cfg.layers.neighbor_scan    = false;
    cfg.layers.state_transition = true;
    cfg.layers.chemical_sources = false;
    cfg.layers.pde_solve        = false;
    cfg.layers.pde_gradients    = false;
    cfg.layers.division         = false;
    cfg.layers.qsp              = false;
    cfg.layers.abm_export       = false;

    // Pin concentrations to the desired combination.
    const float il1  = (pin == Pinning::TGFB_ONLY) ? ZERO_CONC : SAT_CONC;
    const float tgfb = (pin == Pinning::IL1_ONLY)  ? ZERO_CONC : SAT_CONC;
    cfg.pinned_fields.push_back({CHEM_IL1,  uniform_field(il1)});
    cfg.pinned_fields.push_back({CHEM_TGFB, uniform_field(tgfb)});
    cfg.pinned_fields.push_back({CHEM_IL6,  uniform_field(0.0f)});

    // 10×10 lattice of FIB_QUIESCENT at z=LATTICE_Z, each at its own voxel
    // so no adhesion/occupancy effects come into play.
    for (int j = 0; j < LATTICE_N; j++) {
        for (int i = 0; i < LATTICE_N; i++) {
            AgentSeed f;
            f.agent_type = AGENT_FIBROBLAST;
            f.x = LATTICE_BASE + i;
            f.y = LATTICE_BASE + j;
            f.z = LATTICE_Z;
            f.cell_state = FIB_QUIESCENT;
            f.int_vars.push_back({"life", 9999});
            f.int_vars.push_back({"divide_cooldown", 9999});
            f.int_vars.push_back({"divide_count", 0});
            f.int_vars.push_back({"divide_flag", 0});
            f.int_vars.push_back({"frc_dwell_counter", 0});
            f.float_vars.push_back({"adh_p_move", 1.0f});
            cfg.agents.push_back(f);
        }
    }

    auto stats = std::make_shared<ICAFStats>();
    const std::string out_dir = "../test/scenarios/icaf_diff/outputs_" + scenario_name;
    std::filesystem::create_directories(out_dir);

    stats->counts.open(out_dir + "/counts.csv");
    stats->counts << "step,n_quiescent,n_mycaf,n_icaf,n_frc\n";

    cfg.step_callback = [stats, out_dir](flamegpu::CUDASimulation& sim,
                                         flamegpu::ModelDescription& model,
                                         unsigned int step)
    {
        flamegpu::AgentVector fpop(model.Agent(AGENT_FIBROBLAST));
        sim.getPopulationData(fpop);
        int n_q = 0, n_my = 0, n_i = 0, n_frc = 0;
        for (unsigned int i = 0; i < fpop.size(); i++) {
            int cs = fpop[i].getVariable<int>("cell_state");
            if      (cs == FIB_QUIESCENT) n_q++;
            else if (cs == FIB_MYCAF)     n_my++;
            else if (cs == FIB_ICAF)      n_i++;
            else if (cs == FIB_FRC)       n_frc++;
        }
        stats->counts << step << "," << n_q << "," << n_my << ","
                      << n_i << "," << n_frc << "\n";

        if (step == 0 || step + 1 == RUN_STEPS || (step + 1) % 25 == 0) {
            std::cout << "  [step " << step << "] "
                      << "q=" << n_q << " my=" << n_my
                      << " i=" << n_i << std::endl;
        }

        if (step + 1 == RUN_STEPS) {
            stats->counts.flush();
        }
    };

    return cfg;
}

static const bool icaf_il1_registered = []() {
    register_scenario("icaf_diff_il1", [](const std::string& pf) {
        return build(pf, "il1", Pinning::IL1_ONLY);
    });
    return true;
}();

static const bool icaf_tgfb_registered = []() {
    register_scenario("icaf_diff_tgfb", [](const std::string& pf) {
        return build(pf, "tgfb", Pinning::TGFB_ONLY);
    });
    return true;
}();

static const bool icaf_both_registered = []() {
    register_scenario("icaf_diff_both", [](const std::string& pf) {
        return build(pf, "both", Pinning::BOTH);
    });
    return true;
}();

} // anonymous namespace
