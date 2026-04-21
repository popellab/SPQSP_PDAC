// Vascular O2 — Saturation / implicit-split sanity
//
// Purpose: Validate the PHALANX O2 Krogh-cylinder source in isolation. With
// no tissue uptake (no cancer), the source + background decay λ=1e-5/s
// converge to a mass-balance steady state that is NOT simply C_blood:
//
//   Vessel voxel injects S·V = KvLv·(C_blood − C_local) per second,
//   every voxel decays at rate λ. At the (near-)uniform steady state:
//       KvLv·(C_blood − C_ss) ≈ λ · N_voxels · V · C_ss
//   → C_ss = C_blood / (1 + λ·N·V/KvLv)
//
//   For a 21³ grid at 20 µm with this vessel geometry, N·V=7.4e-5 cm³ and
//   the measured C_ss ≈ 1.23e-3 mM vs C_blood = 0.065 mM — consistent with
//   λ·N·V/KvLv ≈ 52 (source is mass-balance-limited by whole-grid decay).
//
// What this test actually validates:
//   (1) KvLv geometry term finite and positive (no NaN/inf).
//   (2) The `if (C_local < C_blood)` gate prevents overshoot past C_blood.
//   (3) Monotonic approach to a single steady-state asymptote (implicit split
//       does not oscillate or diverge).
//   (4) Near-uniform field at steady state (grid small relative to diffusion
//       length for background decay alone: L_bg=√(D/λ)≈660 voxels >> 21).
//
// Setup:
//   - 21³ grid, 20 µm voxels. Single AGENT_VASCULAR (VAS_PHALANX) at (10,10,10).
//   - No cancer. No ECM. Non-vessel agents set empty.
//   - Layers: occupancy + chemical_sources + pde_solve only.
//     (chemical_sources enables reset_pde_buffers; vascular_compute_chemical_sources
//      is the source kernel. state_transition OFF → no HEV/collapse/regression.)
//   - 60 steps (each ABM step = 21600 s → 360 hours real time, well past steady state).
//
// Pass criteria (evaluated in make_figures.py):
//   (P1) No overshoot: max_conc ≤ C_blood at every step (gate works).
//   (P2) Monotonic approach to asymptote: vessel_conc non-decreasing through
//        run (no oscillation from implicit split).
//   (P3) Near-uniform field at final step: (max − min)/mean < 1e-3 (whole-
//        grid mixing via diffusion is fast vs background decay on this grid).

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

using namespace PDAC;
using namespace PDAC::test;

void register_scenario(const std::string& name,
    std::function<TestConfig(const std::string&)> builder);

namespace {

struct SatStats {
    std::ofstream time_series;   // step, vessel_conc, min_conc, max_conc, mean_conc
    std::ofstream xline;         // step, x, conc (along y=cy, z=cz at snapshots)
    std::ofstream params;
    int cx = 0, cy = 0, cz = 0;
    int gs = 0;
};

static TestConfig build_vascular_o2_saturation(const std::string& param_file) {
    TestConfig cfg;
    cfg.name        = "vascular_o2_saturation";
    cfg.grid_size   = 21;
    cfg.voxel_size  = 20.0f;
    cfg.steps       = 60;
    cfg.seed        = 42;
    cfg.param_file  = param_file;

    // --- Layer subset ---
    cfg.layers.ecm_update       = false;
    cfg.layers.occupancy        = true;   // required so agents are placed
    cfg.layers.movement         = false;
    cfg.layers.neighbor_scan    = false;
    cfg.layers.state_transition = false;  // no HEV / collapse / regression
    cfg.layers.chemical_sources = true;   // vascular_compute_chemical_sources runs
    cfg.layers.pde_solve        = true;
    cfg.layers.pde_gradients    = false;
    cfg.layers.division         = false;
    cfg.layers.recruitment      = false;
    cfg.layers.qsp              = false;
    cfg.layers.abm_export       = false;

    // --- ECM: zero density and crosslink so vessel compression = 0 ---
    cfg.ecm.density   = uniform_field(0.0f);
    cfg.ecm.crosslink = uniform_field(0.0f);
    cfg.ecm.floor     = uniform_field(0.0f);

    // --- Single PHALANX vessel at grid center ---
    AgentSeed v;
    v.agent_type = AGENT_VASCULAR;
    v.x = cfg.grid_size / 2;
    v.y = cfg.grid_size / 2;
    v.z = cfg.grid_size / 2;
    v.cell_state = VAS_PHALANX;
    v.int_vars.push_back({"is_dysfunctional", 0});
    v.int_vars.push_back({"maturity",         0});
    cfg.agents.push_back(v);

    auto stats = std::make_shared<SatStats>();
    stats->cx = v.x; stats->cy = v.y; stats->cz = v.z;
    stats->gs = cfg.grid_size;
    const std::string out_dir = "../test/scenarios/" + cfg.name + "/outputs";
    std::filesystem::create_directories(out_dir);
    stats->time_series.open(out_dir + "/time_series.csv");
    stats->time_series << "step,vessel_conc,min_conc,max_conc,mean_conc\n";
    stats->xline.open(out_dir + "/xline.csv");
    stats->xline << "step,x,conc\n";
    stats->params.open(out_dir + "/params.csv");
    stats->params << "key,value\n";

    cfg.step_callback = [stats, out_dir](flamegpu::CUDASimulation& /*sim*/,
                                         flamegpu::ModelDescription& model,
                                         unsigned int step)
    {
        const int gs = stats->gs;
        const int V  = gs * gs * gs;
        const int cx = stats->cx, cy = stats->cy, cz = stats->cz;
        const int cidx = cz * gs * gs + cy * gs + cx;

        if (step == 0) {
            auto env = model.Environment();
            auto put_f = [&](const char* k) {
                stats->params << k << "," << env.getProperty<float>(k) << "\n";
            };
            auto put_i = [&](const char* k) {
                stats->params << k << "," << env.getProperty<int>(k) << "\n";
            };
            put_f("PARAM_VAS_O2_CONC");
            put_f("PARAM_VAS_RC");
            put_f("PARAM_VAS_SIGMA");
            put_f("PARAM_O2_DIFFUSIVITY");
            put_f("PARAM_O2_DECAY_RATE");
            put_f("PARAM_O2_UPTAKE");
            put_f("PARAM_VAS_ECM_COMPRESS_K");
            put_f("PARAM_VAS_MATURITY_RESISTANCE");
            put_f("PARAM_VAS_KVL_DYSFUNCTIONAL");
            put_f("PARAM_VOXEL_SIZE_CM");
            put_f("PARAM_SEC_PER_SLICE");
            put_i("PARAM_MOLECULAR_STEPS");
            stats->params << "grid_size," << gs << "\n";
            stats->params << "vessel_x," << cx << "\n";
            stats->params << "vessel_y," << cy << "\n";
            stats->params << "vessel_z," << cz << "\n";
            stats->params.flush();
        }

        std::vector<float> conc(V);
        PDAC::g_pde_solver->get_concentrations(conc.data(), CHEM_O2);

        float vmin =  1e30f, vmax = -1e30f, vsum = 0.0f;
        for (int i = 0; i < V; i++) {
            if (conc[i] < vmin) vmin = conc[i];
            if (conc[i] > vmax) vmax = conc[i];
            vsum += conc[i];
        }
        const float vmean = vsum / static_cast<float>(V);
        stats->time_series << step << "," << conc[cidx] << "," << vmin << ","
                           << vmax << "," << vmean << "\n";

        const bool snap = (step == 0 || step == 1 || step == 5 || step == 10
                        || step == 20 || step == 40 || step + 1 == 60);
        if (snap) {
            for (int x = 0; x < gs; x++) {
                const int idx = cz * gs * gs + cy * gs + x;
                stats->xline << step << "," << x << "," << conc[idx] << "\n";
            }
            std::cout << "  [step " << step
                      << "] vessel=" << conc[cidx]
                      << "  min=" << vmin << "  max=" << vmax
                      << "  mean=" << vmean << std::endl;
        }

        if (step + 1 == 60) {
            stats->time_series.flush();
            stats->xline.flush();
            std::cout << "\n======== Vascular O2 saturation test complete ========\n"
                      << "  Outputs: " << out_dir << "/\n"
                      << "    time_series.csv  (per-step vessel + min/max/mean)\n"
                      << "    xline.csv        (x-cross-section at snapshots)\n"
                      << "    params.csv       (runtime params)\n"
                      << "  Run make_figures.py for analytic comparison.\n"
                      << "======================================================\n"
                      << std::endl;
        }
    };

    return cfg;
}

static const bool vascular_o2_saturation_registered = []() {
    register_scenario("vascular_o2_saturation", build_vascular_o2_saturation);
    return true;
}();

} // anonymous namespace
