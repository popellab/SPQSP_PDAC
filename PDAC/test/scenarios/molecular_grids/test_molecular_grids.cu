// Test 6 — Molecular Grids (8 new substrates — meeting slide candidate)
//
// Validates every new PDE substrate reaches the correct steady state from a
// single-voxel point source. Catches: wrong diffusivity units, missing source
// registration, broken LOD Thomas pass, double-offset bugs, and decay-rate
// typos — all in one run.
//
// Chemicals exercised (IDs match ChemicalType enum):
//   CHEM_IL1, CHEM_IL6, CHEM_CXCL13, CHEM_MMP,
//   CHEM_ANTIBODY, CHEM_CCL21, CHEM_CXCL12, CHEM_CCL5
//
// Setup:
//   - 21³ grid, 20 µm voxels. Center at (10,10,10).
//   - Single inert AGENT_CANCER_CELL (CANCER_SENESCENT) at corner (1,1,1) —
//     required because FLAMEGPU needs ≥1 agent; chemical_sources layer is OFF
//     so this agent has no effect on the PDE.
//   - ALL layers off except `occupancy` and `pde_solve`. In particular:
//       chemical_sources = OFF  →  reset_pde_buffers does NOT fire
//       state_transition = OFF  →  "                                   "
//     So a source we write to `pde_source_ptr_N` PERSISTS across steps.
//   - Each step, after sim.step(), the callback cudaMemcpy's a unit point
//     source S = SOURCE_RATE [nM/s] into voxel (10,10,10) for each of the
//     eight chemicals. Idempotent — writing the same value every step is safe.
//     (Step 0's solve runs with zero source; step 1 onward sees the injected
//     source. We run 30 steps → ~29 effective; plenty for steady state given
//     decay rates 1e-5–5e-4 s⁻¹ and dt_abm = 21600 s.)
//
// Analytic (per chemical):
//   Open-space Krogh point-source steady state in 3D:
//     C(r) = (S · V_voxel) / (4πD·r) · exp(-r/λ),   λ = √(D/k)
//   The figure script overlays both this (informational) and an image-augmented
//   version that accounts for the 21³ Neumann box reflections — 7 of 8
//   chemicals have λ ≳ L (half-width = 10 vox), so open-space Krogh is NOT
//   the operating regime; CCL21 with λ ≈ 58 vox is effectively well-mixed
//   in this box. The quantitative pass criterion therefore uses mass balance,
//   which is regime-independent.
//
// Pass criterion (evaluated in make_figures.py, not CUDA):
//   (1) Mass balance: |Σ_i C_i - S/k| / (S/k) < 0.10   per chemical
//       Derivation: at steady state with Neumann BC, integrating the PDE
//         0 = D·∇²C − k·C + s   over the domain gives  S·V_vox = k·∫C dV,
//       i.e. Σ C_i = S/k. This is a single-number regime-independent check
//       on source injection, decay, and conservation.
//   (2) Steady state: |C_center(step 29) - C_center(step 24)| / C_center < 0.01.
//   Both must pass for every chemical.
//
// Output files (in ../test/scenarios/molecular_grids/outputs/):
//   params.csv         — runtime diffusivity/decay/voxel params
//   time_series.csv    — center voxel conc per step per chemical (convergence)
//   final_conc.csv     — full 3D field at final step: chem_id,x,y,z,conc

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

// -------------------------------------------------------------------------
// Chemicals under test. Order matters: index into ChemicalType enum.
// -------------------------------------------------------------------------
struct ChemSpec {
    int         chem_id;
    const char* name;
    const char* diff_param;
    const char* decay_param;
};
static const ChemSpec CHEMS[] = {
    { CHEM_IL1,      "IL1",      "PARAM_IL1_DIFFUSIVITY",      "PARAM_IL1_DECAY_RATE"      },
    { CHEM_IL6,      "IL6",      "PARAM_IL6_DIFFUSIVITY",      "PARAM_IL6_DECAY_RATE"      },
    { CHEM_CXCL13,   "CXCL13",   "PARAM_CXCL13_DIFFUSIVITY",   "PARAM_CXCL13_DECAY_RATE"   },
    { CHEM_MMP,      "MMP",      "PARAM_MMP_DIFFUSIVITY",      "PARAM_MMP_DECAY_RATE"      },
    { CHEM_ANTIBODY, "ANTIBODY", "PARAM_ANTIBODY_DIFFUSIVITY", "PARAM_ANTIBODY_DECAY_RATE" },
    { CHEM_CCL21,    "CCL21",    "PARAM_CCL21_DIFFUSIVITY",    "PARAM_CCL21_DECAY_RATE"    },
    { CHEM_CXCL12,   "CXCL12",   "PARAM_CXCL12_DIFFUSIVITY",   "PARAM_CXCL12_DECAY_RATE"   },
    { CHEM_CCL5,     "CCL5",     "PARAM_CCL5_DIFFUSIVITY",     "PARAM_CCL5_DECAY_RATE"     },
};
constexpr int N_CHEMS = sizeof(CHEMS) / sizeof(CHEMS[0]);

// Injection rate — uniform across all chemicals for simplicity.
// Units are [nM/s] because the source term s in the PDE is
// d[conc]/dt = ... + s, and substrate concentrations are [nM].
constexpr float SOURCE_RATE = 1.0f;

struct MolGridStats {
    std::ofstream time_series;
    std::ofstream params;
    std::ofstream final_conc;
    int cx = 0, cy = 0, cz = 0;
    int gs = 0;
};

static TestConfig build_molecular_grids(const std::string& param_file) {
    TestConfig cfg;
    cfg.name = "molecular_grids";
    cfg.grid_size = 21;
    cfg.voxel_size = 20.0f;
    cfg.steps = 30;
    cfg.seed = 42;
    cfg.param_file = param_file;

    // --- Layer subset: PDE solve only. No agent activity affects PDE. ---
    cfg.layers.ecm_update       = false;
    cfg.layers.occupancy        = true;   // minimal; needed for the dummy agent
    cfg.layers.movement         = false;
    cfg.layers.neighbor_scan    = false;
    cfg.layers.state_transition = false;  // do NOT fire reset_pde_buffers
    cfg.layers.chemical_sources = false;  // do NOT fire reset_pde_buffers
    cfg.layers.pde_solve        = true;
    cfg.layers.pde_gradients    = false;
    cfg.layers.division         = false;
    cfg.layers.recruitment      = false;
    cfg.layers.qsp              = false;
    cfg.layers.abm_export       = false;

    // --- Inert dummy agent at corner (needed so FLAMEGPU has ≥1 agent) ---
    AgentSeed s;
    s.agent_type = AGENT_CANCER_CELL;
    s.x = 1; s.y = 1; s.z = 1;
    s.cell_state = CANCER_SENESCENT;
    s.int_vars.push_back({"divideCountRemaining", 0});
    s.int_vars.push_back({"divideFlag", 0});
    s.int_vars.push_back({"divideCD", 999999});
    s.int_vars.push_back({"life", 999999});
    cfg.agents.push_back(s);

    // --- Stats + CSV setup ---
    auto stats = std::make_shared<MolGridStats>();
    stats->gs = cfg.grid_size;
    stats->cx = stats->cy = stats->cz = cfg.grid_size / 2;   // (10,10,10) for gs=21
    const std::string out_dir = "../test/scenarios/" + cfg.name + "/outputs";
    std::filesystem::create_directories(out_dir);

    stats->params.open(out_dir + "/params.csv");
    stats->params << "key,value\n";

    stats->time_series.open(out_dir + "/time_series.csv");
    stats->time_series << "step,chem,center_conc\n";

    stats->final_conc.open(out_dir + "/final_conc.csv");
    stats->final_conc << "chem,x,y,z,conc\n";

    cfg.step_callback = [stats, out_dir](flamegpu::CUDASimulation& /*sim*/,
                                         flamegpu::ModelDescription& model,
                                         unsigned int step)
    {
        const int gs = stats->gs;
        const int V  = gs * gs * gs;
        const int src_idx = stats->cz * gs * gs + stats->cy * gs + stats->cx;

        // Dump runtime params once at step 0 (before any injection).
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
            put_i("PARAM_MOLECULAR_STEPS");
            for (int c = 0; c < N_CHEMS; c++) {
                put_f(CHEMS[c].diff_param);
                put_f(CHEMS[c].decay_param);
            }
            stats->params << "SOURCE_RATE_nM_per_s," << SOURCE_RATE << "\n";
            stats->params << "center_x," << stats->cx << "\n";
            stats->params << "center_y," << stats->cy << "\n";
            stats->params << "center_z," << stats->cz << "\n";
            stats->params << "grid_size," << gs << "\n";
            stats->params.flush();
        }

        // Inject point source at center voxel for every chemical.
        // Idempotent: same value written every step. Persists because
        // reset_pde_buffers doesn't run (state_transition + chemical_sources
        // are both off).
        for (int c = 0; c < N_CHEMS; c++) {
            float* d_src = PDAC::g_pde_solver->get_device_source_ptr(CHEMS[c].chem_id);
            cudaMemcpy(d_src + src_idx, &SOURCE_RATE, sizeof(float),
                       cudaMemcpyHostToDevice);
        }

        // Log center concentration per chemical each step for convergence check.
        for (int c = 0; c < N_CHEMS; c++) {
            float val = PDAC::g_pde_solver->get_concentration_at_voxel(
                stats->cx, stats->cy, stats->cz, CHEMS[c].chem_id);
            stats->time_series << step << "," << CHEMS[c].name << ","
                               << val << "\n";
        }

        // On a few key steps, print center values for eyeball check.
        if (step == 0 || step == 9 || step == 19 || step + 1 == 30) {
            std::cout << "  [step " << step << "] center = {";
            for (int c = 0; c < N_CHEMS; c++) {
                float val = PDAC::g_pde_solver->get_concentration_at_voxel(
                    stats->cx, stats->cy, stats->cz, CHEMS[c].chem_id);
                std::cout << CHEMS[c].name << "=" << val;
                if (c + 1 < N_CHEMS) std::cout << ", ";
            }
            std::cout << "}" << std::endl;
        }

        // Final step: dump full 3D fields for all chemicals.
        if (step + 1 == 30) {
            std::vector<float> buf(V);
            for (int c = 0; c < N_CHEMS; c++) {
                PDAC::g_pde_solver->get_concentrations(buf.data(), CHEMS[c].chem_id);
                for (int z = 0; z < gs; z++) {
                    for (int y = 0; y < gs; y++) {
                        for (int x = 0; x < gs; x++) {
                            const int idx = z * gs * gs + y * gs + x;
                            stats->final_conc << CHEMS[c].name << "," << x << ","
                                              << y << "," << z << ","
                                              << buf[idx] << "\n";
                        }
                    }
                }
            }
            stats->final_conc.flush();
            stats->time_series.flush();

            std::cout << "\n==================== Molecular grids ====================\n"
                      << "  " << N_CHEMS << " chemicals × " << V << " voxels dumped\n"
                      << "  Outputs: " << out_dir << "/\n"
                      << "    params.csv       (diffusivity + decay, runtime)\n"
                      << "    time_series.csv  (center conc per step per chem)\n"
                      << "    final_conc.csv   (full 3D field per chem, final step)\n"
                      << "  Run make_figures.py for Krogh analytic comparison.\n"
                      << "=========================================================\n"
                      << std::endl;
        }
    };

    return cfg;
}

static const bool molecular_grids_registered = []() {
    register_scenario("molecular_grids", build_molecular_grids);
    return true;
}();

} // anonymous namespace
