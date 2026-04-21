// Vascular O2 — Krogh cylinder with tissue uptake (substep convergence sweep)
//
// Purpose: Verify that a vessel column surrounded by O2-consuming cells can
// produce the expected 2D Bessel K₀ decay profile, AND quantify how many PDE
// substeps the operator-split solver needs to resolve that profile at the
// literature uptake rate (k = 0.1/s).
//
// Operator-splitting artifact:
//   Per substep, apply_sources_uptakes_kernel applies C → C·exp(-k·dt_sub).
//   At k=0.1/s and the production default dt_sub = 21600/36 = 600s,
//   k·dt_sub = 60 → exp(-60) ≈ 0 — every cancer voxel drained to zero each
//   substep before LOD runs. LOD then spreads the vessel source across the
//   grid with nothing balancing diffusion, yielding a near-flat field.
//
//   The true continuous-time solution of D∇²C - k·C + S·δ = 0 is K₀(r/L) with
//   L = √(D/k) ≈ 6.6 voxels at 20 µm. To recover it, dt_sub must be small
//   compared to 1/k. We sweep PARAM_MOLECULAR_STEPS and show convergence.
//
// Substep variants registered:
//   vascular_o2_uptake_sub36   (dt_sub=600s, k·dt=60)   — production default, flat
//   vascular_o2_uptake_sub360  (dt_sub=60s,  k·dt=6)    — partial gradient
//   vascular_o2_uptake_sub3600 (dt_sub=6s,   k·dt=0.6)  — converged K₀
//
// Geometry:
//   Z-column of PHALANX at (cx, cy, z=0..gs-1). Every other voxel holds
//   a static CANCER_PROGENITOR (first-order O2 sink at PARAM_O2_UPTAKE).
//   z-invariant far from ends → treat as 2D cylindrical around column.
//
// Analytic (continuous-time):
//   C(r) = A · K₀(r/L),    L = √(D/k_eff),   k_eff = k_cancer + λ_bg
//
// Pass criterion (in make_figures.py, across all three runs):
//   (P1) Monotonic convergence: radial ratio C(r=1)/C(r=10) increases with
//        substep count.
//   (P2) High-substep run radial shape matches K₀(r/L) within 25% at
//        r = 3, 5, 7 voxels after log-rescale to match amplitudes.
//   (P3) Low-substep run radial ratio < 1.5 — documents the artifact.

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

constexpr int GRID_SIZE = 25;

// With 3600 substeps and dt_sub=6s, τ_diff_grid ≈ 143s and τ_uptake = 10s,
// so steady state is reached within a few ABM steps. 8 ABM steps is enough
// for all three substep counts.
constexpr unsigned int N_STEPS = 8;

struct UptakeStats {
    std::ofstream time_series;
    std::ofstream radial;
    std::ofstream xline;
    std::ofstream params;
    std::string variant;
    int substeps = 36;
    int cx = 0, cy = 0, cz = 0;
    int gs = 0;
};

static TestConfig build_variant(const std::string& name,
                                const std::string& param_file,
                                int substeps)
{
    TestConfig cfg;
    cfg.name        = name;
    cfg.grid_size   = GRID_SIZE;
    cfg.voxel_size  = 20.0f;
    cfg.steps       = N_STEPS;
    cfg.seed        = 42;
    cfg.param_file  = param_file;

    // Force PARAM_MOLECULAR_STEPS via env override. Applied before
    // defineTestLayers reads it, and before PDE init reads it from the
    // environment via PARAM_MOLECULAR_STEPS lookup.
    cfg.int_env_overrides.push_back({"PARAM_MOLECULAR_STEPS", substeps});

    cfg.layers.ecm_update       = false;
    cfg.layers.occupancy        = true;
    cfg.layers.movement         = false;
    cfg.layers.neighbor_scan    = false;
    cfg.layers.state_transition = false;
    cfg.layers.chemical_sources = true;
    cfg.layers.pde_solve        = true;
    cfg.layers.pde_gradients    = false;
    cfg.layers.division         = false;
    cfg.layers.recruitment      = false;
    cfg.layers.qsp              = false;
    cfg.layers.abm_export       = false;

    cfg.ecm.density   = uniform_field(0.0f);
    cfg.ecm.crosslink = uniform_field(0.0f);
    cfg.ecm.floor     = uniform_field(0.0f);

    const int cx = GRID_SIZE / 2;
    const int cy = GRID_SIZE / 2;
    const int cz = GRID_SIZE / 2;

    for (int z = 0; z < GRID_SIZE; z++) {
        AgentSeed v;
        v.agent_type = AGENT_VASCULAR;
        v.x = cx; v.y = cy; v.z = z;
        v.cell_state = VAS_PHALANX;
        v.int_vars.push_back({"is_dysfunctional", 0});
        v.int_vars.push_back({"maturity",         0});
        cfg.agents.push_back(v);
    }

    for (int z = 0; z < GRID_SIZE; z++) {
        for (int y = 0; y < GRID_SIZE; y++) {
            for (int x = 0; x < GRID_SIZE; x++) {
                if (x == cx && y == cy) continue;
                AgentSeed c;
                c.agent_type = AGENT_CANCER_CELL;
                c.x = x; c.y = y; c.z = z;
                c.cell_state = CANCER_PROGENITOR;
                c.int_vars.push_back({"divideCountRemaining", 0});
                c.int_vars.push_back({"divideFlag",           0});
                c.int_vars.push_back({"divideCD",             999999});
                c.int_vars.push_back({"life",                 999999});
                c.int_vars.push_back({"hypoxic",              0});
                cfg.agents.push_back(c);
            }
        }
    }

    auto stats = std::make_shared<UptakeStats>();
    stats->cx = cx; stats->cy = cy; stats->cz = cz;
    stats->gs = GRID_SIZE;
    stats->variant = name;
    stats->substeps = substeps;

    // All variants share one parent scenario folder; CSVs go into a
    // per-variant subfolder so make_figures.py can find them all.
    const std::string out_dir = "../test/scenarios/vascular_o2_cancer_uptake/outputs/" + name;
    std::filesystem::create_directories(out_dir);
    stats->time_series.open(out_dir + "/time_series.csv");
    stats->time_series << "step,vessel_conc,min_conc,max_conc,mean_conc\n";
    stats->radial.open(out_dir + "/radial.csv");
    stats->radial << "step,r_voxel,n_samples,mean_conc\n";
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
            put_f("PARAM_VOXEL_SIZE_CM");
            put_f("PARAM_SEC_PER_SLICE");
            put_i("PARAM_MOLECULAR_STEPS");
            stats->params << "grid_size," << gs << "\n";
            stats->params << "vessel_x," << cx << "\n";
            stats->params << "vessel_y," << cy << "\n";
            stats->params << "vessel_z," << cz << "\n";
            stats->params << "variant," << stats->variant << "\n";
            stats->params << "substeps_override," << stats->substeps << "\n";
            stats->params.flush();
        }

        std::vector<float> conc(V);
        PDAC::g_pde_solver->get_concentrations(conc.data(), CHEM_O2);

        float vmin = 1e30f, vmax = -1e30f, vsum = 0.0f;
        for (int i = 0; i < V; i++) {
            if (conc[i] < vmin) vmin = conc[i];
            if (conc[i] > vmax) vmax = conc[i];
            vsum += conc[i];
        }
        const float vmean = vsum / static_cast<float>(V);
        stats->time_series << step << "," << conc[cidx] << "," << vmin << ","
                           << vmax << "," << vmean << "\n";

        // Radial profile every step — cheap and lets make_figures.py plot
        // convergence in time for each variant.
        const int zmin = std::max(0, cz - 2);
        const int zmax = std::min(gs - 1, cz + 2);
        const int max_r = static_cast<int>(std::sqrt(2.0) * (gs / 2)) + 1;
        std::vector<double> rsum(max_r + 1, 0.0);
        std::vector<int>    rcnt(max_r + 1, 0);
        for (int z = zmin; z <= zmax; z++) {
            for (int y = 0; y < gs; y++) {
                for (int x = 0; x < gs; x++) {
                    const int dx = x - cx, dy = y - cy;
                    const float rf = std::sqrt(static_cast<float>(dx*dx + dy*dy));
                    const int rb = static_cast<int>(std::round(rf));
                    if (rb <= max_r) {
                        rsum[rb] += conc[z * gs * gs + y * gs + x];
                        rcnt[rb]++;
                    }
                }
            }
        }
        for (int r = 0; r <= max_r; r++) {
            if (rcnt[r] > 0) {
                stats->radial << step << "," << r << "," << rcnt[r] << ","
                              << (rsum[r] / rcnt[r]) << "\n";
            }
        }

        for (int x = 0; x < gs; x++) {
            const int idx = cz * gs * gs + cy * gs + x;
            stats->xline << step << "," << x << "," << conc[idx] << "\n";
        }

        std::cout << "  [" << stats->variant << " step " << step
                  << "] vessel=" << conc[cidx]
                  << "  min=" << vmin << "  max=" << vmax
                  << "  mean=" << vmean << std::endl;

        if (step + 1 == N_STEPS) {
            stats->time_series.flush();
            stats->radial.flush();
            stats->xline.flush();
            std::cout << "\n[" << stats->variant << "] complete. Output: "
                      << out_dir << "/\n" << std::endl;
        }
    };

    return cfg;
}

static const bool sub36_registered = []() {
    register_scenario("vascular_o2_uptake_sub36",
        [](const std::string& pf) {
            return build_variant("vascular_o2_uptake_sub36", pf, 36);
        });
    return true;
}();

static const bool sub360_registered = []() {
    register_scenario("vascular_o2_uptake_sub360",
        [](const std::string& pf) {
            return build_variant("vascular_o2_uptake_sub360", pf, 360);
        });
    return true;
}();

static const bool sub3600_registered = []() {
    register_scenario("vascular_o2_uptake_sub3600",
        [](const std::string& pf) {
            return build_variant("vascular_o2_uptake_sub3600", pf, 3600);
        });
    return true;
}();

} // anonymous namespace
