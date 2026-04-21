// Test 4 — ECM Density / Crosslink / MMP
//
// Validates the update_ecm_grid kernel (pde_integration.cu): Gaussian density
// deposition by a single myCAF, LOX crosslink accumulation toward saturation,
// and MMP-driven density degradation gated by crosslink resistance.
//
// Setup:
//   - 31³ grid. Single AGENT_FIBROBLAST (cell_state=FIB_MYCAF) at (15,15,15).
//   - TGFB pinned uniformly at 1000 nM (>> PARAM_ECM_TGFB_EC50 → H_TGFB ≈ 1).
//   - O2 pinned uniformly at 1.0 mM (>> PARAM_FIB_HYPOXIA_TH=0.01 → no HIF).
//   - ECM density preset to 0, crosslink preset to 0, floor preset to 0
//     (override the harness default that initializes density to CAP).
//   - 300 ABM steps.
//   - Layers: ecm_update + occupancy ONLY. Everything else disabled —
//     including pde_solve, so MMP stays exactly where the callback writes it.
//   - At step_callback == 149: cudaMemcpy MMP array to uniform 1e-3 (µM).
//     Kernel then reads this value every subsequent step (no PDE decay here).
//
// Observables (all dumped to CSVs, analytic comparisons done in make_figures.py):
//   1. Density Gaussian: radial profile through center vs. analytic
//        ρ_eq(r) ≈ fib(r) · (1+H_TGFB) · k_depo/3 · (1-sat) · yap / k_decay
//      where fib(r) = (1/(2πσ²)^1.5) · exp(-r²/2σ²), σ² = PARAM_FIB_ECM_VARIANCE.
//   2. Crosslink curve: center voxel crosslink(t) vs. analytic
//        c(n) = 1 - (1 - k_lox·dt)^n  (discrete forward-Euler of dc/dt=k_lox(1-c))
//   3. MMP degradation: from step 150 onward, density decays as
//        ρ(n) ≈ ρ(150) · exp(-k_mmp·MMP·n_steps·dt/(1+α·crosslink))
//
// This is a whitebox test: we validate the kernel math AS IMPLEMENTED,
// not against external biology. Empirical vs. analytic comparison uses the
// runtime parameter values (some are QSP-derived so not known at build time).

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
#include <cmath>

using namespace PDAC;
using namespace PDAC::test;

void register_scenario(const std::string& name,
    std::function<TestConfig(const std::string&)> builder);

namespace {

// MMP concentration written to the device array starting at step 149.
// Units: match whatever update_ecm_grid_kernel expects for mmp_conc.
// Chosen so k_mmp·MMP·dt gives a degradation rate that dominates the ongoing
// myCAF deposition (which at step 150 is ~1.2/step at center). At MMP=0.01 and
// crosslink≈0.96 (α=5), per-step loss ≈ k_mmp·MMP·dt/(1+α·c) · density ≈
// 1e-3·1e-2·21600/5.8 ≈ 3.7e-2 · density → ~11/step loss at density=300 —
// clearly visible net drop, settling into a new degradation-vs-deposition equilibrium.
constexpr float MMP_INJECT_VALUE = 1e-2f;
constexpr unsigned int MMP_INJECT_STEP = 149;  // effect visible at step 150 ecm_update

struct EcmStats {
    std::ofstream time_series;   // step, center_density, center_crosslink, max_density, mmp_value
    std::ofstream xline;         // step, x, density, crosslink   (y=cy, z=cz)
    std::ofstream params;        // dumps runtime param values once at step 0
    int cx = 0, cy = 0, cz = 0;  // center voxel (myCAF location)
    int gs = 0;                  // grid size
    bool mmp_injected = false;
};

static TestConfig build_ecm_density(const std::string& param_file) {
    TestConfig cfg;
    cfg.name = "ecm_density";
    cfg.grid_size = 31;
    cfg.voxel_size = 20.0f;
    cfg.steps = 300;
    cfg.seed = 42;
    cfg.param_file = param_file;

    // --- Layer subset: ECM update only ---
    cfg.layers.ecm_update       = true;   // zero_fib + build_density + update_ecm
    cfg.layers.occupancy        = true;   // required for any agent functions
    cfg.layers.movement         = false;  // myCAF stays put
    cfg.layers.neighbor_scan    = false;
    cfg.layers.state_transition = false;  // keep cell as FIB_MYCAF
    cfg.layers.chemical_sources = false;  // no PDE perturbation
    cfg.layers.pde_solve        = false;  // MMP stays at callback-written value
    cfg.layers.pde_gradients    = false;
    cfg.layers.division         = false;
    cfg.layers.recruitment      = false;
    cfg.layers.qsp              = false;
    cfg.layers.abm_export       = false;

    // --- Pinned chemical fields ---
    // TGFB saturating (H_TGFB → 1) so deposition rate is independent of exact value.
    cfg.pinned_fields.push_back({CHEM_TGFB, uniform_field(1000.0f)});
    // O2 well above PARAM_FIB_HYPOXIA_TH (0.01) so HIF boost is off → ecm_multiplier = 1.
    cfg.pinned_fields.push_back({CHEM_O2, uniform_field(1.0f)});
    // MMP is NOT pinned — starts at 0 (PDE init), callback cudaMemcpy's it to
    // MMP_INJECT_VALUE at step 149, and with pde_solve disabled the value
    // persists unchanged for the remainder of the run.

    // --- ECM preset: start with empty density + crosslink, no floor ---
    // The harness default initialize_ecm_to_saturation() fills density=CAP,
    // so we MUST override here to start from 0.
    cfg.ecm.density   = uniform_field(0.0f);
    cfg.ecm.crosslink = uniform_field(0.0f);
    cfg.ecm.floor     = uniform_field(0.0f);

    // --- Single myCAF at grid center ---
    AgentSeed s;
    s.agent_type = AGENT_FIBROBLAST;
    s.x = cfg.grid_size / 2;  // 15 for gs=31
    s.y = cfg.grid_size / 2;
    s.z = cfg.grid_size / 2;
    s.cell_state = FIB_MYCAF;
    cfg.agents.push_back(s);

    // --- Per-step callback: dump timeseries + xline; inject MMP at step 149 ---
    auto stats = std::make_shared<EcmStats>();
    stats->cx = s.x; stats->cy = s.y; stats->cz = s.z;
    stats->gs = cfg.grid_size;
    const std::string out_dir = "../test/scenarios/" + cfg.name + "/outputs";
    std::filesystem::create_directories(out_dir);
    stats->time_series.open(out_dir + "/ecm_time_series.csv");
    stats->time_series << "step,center_density,center_crosslink,max_density,mmp_value\n";
    stats->xline.open(out_dir + "/ecm_xline.csv");
    stats->xline << "step,x,density,crosslink\n";
    stats->params.open(out_dir + "/ecm_params.csv");
    stats->params << "key,value\n";

    cfg.step_callback = [stats, out_dir](flamegpu::CUDASimulation& /*sim*/,
                                         flamegpu::ModelDescription& model,
                                         unsigned int step)
    {
        const int gs = stats->gs;
        const int V = gs * gs * gs;
        const int cx = stats->cx, cy = stats->cy, cz = stats->cz;
        const int center_idx = cz * gs * gs + cy * gs + cx;

        // Dump runtime parameter values once at step 0 (includes QSP-derived rates).
        if (step == 0) {
            auto env = model.Environment();
            auto put = [&](const char* k, float v) {
                stats->params << k << "," << v << "\n";
            };
            put("PARAM_SEC_PER_SLICE",          env.getProperty<float>("PARAM_SEC_PER_SLICE"));
            put("PARAM_VOXEL_SIZE_CM",          env.getProperty<float>("PARAM_VOXEL_SIZE_CM"));
            put("PARAM_FIB_ECM_RADIUS",         env.getProperty<float>("PARAM_FIB_ECM_RADIUS"));
            put("PARAM_FIB_ECM_VARIANCE",       env.getProperty<float>("PARAM_FIB_ECM_VARIANCE"));
            put("PARAM_ECM_DECAY_RATE",         env.getProperty<float>("PARAM_ECM_DECAY_RATE"));
            put("PARAM_ECM_DEPOSITION_RATE",    env.getProperty<float>("PARAM_ECM_DEPOSITION_RATE"));
            put("PARAM_ECM_DENSITY_CAP",        env.getProperty<float>("PARAM_ECM_DENSITY_CAP"));
            put("PARAM_ECM_TGFB_EC50",          env.getProperty<float>("PARAM_ECM_TGFB_EC50"));
            put("PARAM_ECM_MMP_DEGRADE_RATE",   env.getProperty<float>("PARAM_ECM_MMP_DEGRADE_RATE"));
            put("PARAM_ECM_CROSSLINK_RATE",     env.getProperty<float>("PARAM_ECM_CROSSLINK_RATE"));
            put("PARAM_ECM_CROSSLINK_RESISTANCE", env.getProperty<float>("PARAM_ECM_CROSSLINK_RESISTANCE"));
            put("PARAM_ECM_YAP_EC50",           env.getProperty<float>("PARAM_ECM_YAP_EC50"));
            put("PARAM_ECM_BASELINE",           env.getProperty<float>("PARAM_ECM_BASELINE"));
            put("PARAM_FIB_HYPOXIA_TH",         env.getProperty<float>("PARAM_FIB_HYPOXIA_TH"));
            put("PARAM_FIB_HIF_ECM_BOOST",      env.getProperty<float>("PARAM_FIB_HIF_ECM_BOOST"));
            stats->params << "MMP_INJECT_VALUE," << MMP_INJECT_VALUE << "\n";
            stats->params << "MMP_INJECT_STEP,"  << MMP_INJECT_STEP  << "\n";
            stats->params << "center_x," << cx << "\ncenter_y," << cy << "\ncenter_z," << cz << "\n";
            stats->params << "grid_size," << gs << "\n";
            stats->params.flush();
        }

        // MMP injection: cudaMemcpy at step 149 so the step-150 ecm_update sees it.
        if (step == MMP_INJECT_STEP && !stats->mmp_injected) {
            std::vector<float> mmp_host(V, MMP_INJECT_VALUE);
            float* d_mmp = PDAC::g_pde_solver->get_device_concentration_ptr(CHEM_MMP);
            cudaMemcpy(d_mmp, mmp_host.data(), V * sizeof(float), cudaMemcpyHostToDevice);
            stats->mmp_injected = true;
            std::cout << "  [step " << step << "] MMP injected uniformly at "
                      << MMP_INJECT_VALUE << std::endl;
        }

        // Read ECM density + crosslink back to host for logging.
        std::vector<float> density(V), crosslink(V);
        cudaMemcpy(density.data(),   PDAC::get_ecm_density_device_ptr(),   V * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(crosslink.data(), PDAC::get_ecm_crosslink_device_ptr(), V * sizeof(float), cudaMemcpyDeviceToHost);

        // Time-series: center values + max density + current MMP level.
        float center_density = density[center_idx];
        float center_crosslink = crosslink[center_idx];
        float max_density = 0.0f;
        for (int i = 0; i < V; i++) if (density[i] > max_density) max_density = density[i];
        const float current_mmp = stats->mmp_injected ? MMP_INJECT_VALUE : 0.0f;
        stats->time_series << step << "," << center_density << ","
                           << center_crosslink << "," << max_density << ","
                           << current_mmp << "\n";

        // X-line (y=cy, z=cz) sampled at key snapshot steps for radial inspection.
        const bool snap = (step == 0 || step == 10 || step == 50 || step == 100
                        || step == 149 || step == 150 || step == 200 || step == 299);
        if (snap) {
            for (int x = 0; x < gs; x++) {
                const int idx = cz * gs * gs + cy * gs + x;
                stats->xline << step << "," << x << ","
                             << density[idx] << "," << crosslink[idx] << "\n";
            }
        }

        if (step == 0 || step == 50 || step == 100 || step == 149
            || step == 150 || step == 200 || step == 299) {
            std::cout << "  [step " << step
                      << "] center_density=" << center_density
                      << "  center_crosslink=" << center_crosslink
                      << "  max_density=" << max_density
                      << "  mmp=" << current_mmp << std::endl;
        }

        if (step + 1 == 300) {
            stats->time_series.flush();
            stats->xline.flush();
            std::cout << "\n==================== ECM density test complete ====================\n"
                      << "  Outputs: " << out_dir << "/\n"
                      << "    ecm_time_series.csv  (per-step center + max density)\n"
                      << "    ecm_xline.csv        (x-slice at snapshot steps)\n"
                      << "    ecm_params.csv       (runtime parameter values)\n"
                      << "  Run make_figures.py to generate validation plots.\n"
                      << "===================================================================\n"
                      << std::endl;
        }
    };

    return cfg;
}

static const bool ecm_density_registered = []() {
    register_scenario("ecm_density", build_ecm_density);
    return true;
}();

} // anonymous namespace
