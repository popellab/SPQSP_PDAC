// PDE substep sweep — non-O2 substrates
//
// Purpose: Quantify how much spatial gradient fidelity is lost across the
// operator-split PDE solver when we run fewer molecular substeps per ABM
// step. O2 is the known failure mode (k·dt_sub = 60 at 36 substeps, per
// `project_pde_substep_o2.md`). This test sweeps PARAM_MOLECULAR_STEPS for
// all 17 non-O2 substrates simultaneously and reports per-substrate
// convergence vs the highest-substep reference.
//
// Five scenarios: pde_substep_sub{6, 18, 36, 360, 3600}
//   - 6  & 18: below production, show what we lose if we go faster
//   - 36:     production default (in param_all_test.xml)
//   - 360:    10× refined (bridge)
//   - 3600:   100× refined (reference for convergence)
//
// Setup:
//   - 25³ grid, 20 µm voxels (5×10⁻² cm cube), 60 ABM steps.
//   - No agents. Layer config: occupancy OFF, movement OFF, state_transition
//     OFF, chemical_sources OFF, pde_solve ON, pde_gradients OFF.
//   - reset_pde_buffers is gated on state_transition||chemical_sources, so it
//     is NOT called — source buffer stays at 0 for all substrates. Fields
//     evolve purely via diffusion + decay (and any uptake rate set by
//     set_internal_params, but with no agents writing to uptake buffer it
//     also stays 0).
//   - step_callback pins C(center) = 1.0 for every non-O2 substrate at the
//     end of each ABM step. Next step's solve sees this as initial condition
//     and relaxes it outward under diffusion+decay.
//   - After many steps, we approach the Helmholtz steady state:
//         D ∇²C = k_eff · C,    C(center) = 1
//     whose spherical Green's function form is  C(r) = (sinh((R-r)/L) / r) /
//     (sinh((R-r_c)/L) / r_c)  with L = √(D/k_eff). For L >> grid the field
//     is ~uniform; for L ≤ grid the field decays radially.
//
// Output per scenario (in outputs/<scenario>/):
//   fields.csv   — step,substrate,x,conc  (line through center along x)
//   params.csv   — substep count, decay rates, diffusivities
//
// Pass criteria evaluated in make_figures.py (per-substrate):
//   P_conv: |C_subN(r=0) - C_sub3600(r=0)| / C_sub3600(r=0) < 5%  for all r
//           Flags substrates that deviate from converged reference under
//           reduced substep counts.

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
#include <string>
#include <vector>

using namespace PDAC;
using namespace PDAC::test;

void register_scenario(const std::string& name,
    std::function<TestConfig(const std::string&)> builder);

namespace {

// Substrates we sweep. CHEM_O2 is excluded — its artifact is already
// characterized in vascular_o2_cancer_uptake.
static const std::vector<std::pair<int, const char*>> NON_O2_SUBSTRATES = {
    {CHEM_IFN,      "IFN"},
    {CHEM_IL2,      "IL2"},
    {CHEM_IL10,     "IL10"},
    {CHEM_TGFB,     "TGFB"},
    {CHEM_CCL2,     "CCL2"},
    {CHEM_ARGI,     "ARGI"},
    {CHEM_NO,       "NO"},
    {CHEM_IL12,     "IL12"},
    {CHEM_VEGFA,    "VEGFA"},
    {CHEM_IL1,      "IL1"},
    {CHEM_IL6,      "IL6"},
    {CHEM_CXCL13,   "CXCL13"},
    {CHEM_MMP,      "MMP"},
    {CHEM_ANTIBODY, "ANTIBODY"},
    {CHEM_CCL21,    "CCL21"},
    {CHEM_CXCL12,   "CXCL12"},
    {CHEM_CCL5,     "CCL5"},
};

struct SweepStats {
    std::ofstream fields;
    std::ofstream params;
    int gs = 0;
    int cx = 0, cy = 0, cz = 0;
};

static TestConfig build_pde_substep(const std::string& scenario_name,
                                    int substep_count,
                                    const std::string& param_file)
{
    TestConfig cfg;
    cfg.name        = scenario_name;
    cfg.grid_size   = 25;
    cfg.voxel_size  = 20.0f;
    cfg.steps       = 60;
    cfg.seed        = 42;
    cfg.param_file  = param_file;

    // Force the substep count BEFORE PDE init reads it.
    cfg.int_env_overrides.push_back({"PARAM_MOLECULAR_STEPS", substep_count});

    cfg.layers.ecm_update       = false;
    cfg.layers.recruitment      = false;
    cfg.layers.occupancy        = false;
    cfg.layers.movement         = false;
    cfg.layers.neighbor_scan    = false;
    cfg.layers.state_transition = false;
    cfg.layers.chemical_sources = false;
    cfg.layers.pde_solve        = true;   // only thing we care about
    cfg.layers.pde_gradients    = false;
    cfg.layers.division         = false;
    cfg.layers.qsp              = false;
    cfg.layers.abm_export       = false;

    cfg.ecm.density   = uniform_field(0.0f);
    cfg.ecm.crosslink = uniform_field(0.0f);
    cfg.ecm.floor     = uniform_field(0.0f);

    auto stats = std::make_shared<SweepStats>();
    stats->gs = cfg.grid_size;
    stats->cx = cfg.grid_size / 2;
    stats->cy = cfg.grid_size / 2;
    stats->cz = cfg.grid_size / 2;

    const std::string out_dir =
        "../test/scenarios/pde_substep_sweep/outputs/" + cfg.name;
    std::filesystem::create_directories(out_dir);
    stats->fields.open(out_dir + "/fields.csv");
    stats->fields << "step,substrate,substrate_idx,x,conc\n";
    stats->params.open(out_dir + "/params.csv");
    stats->params << "key,value\n";

    cfg.step_callback = [stats, out_dir, substep_count](
        flamegpu::CUDASimulation& /*sim*/,
        flamegpu::ModelDescription& model,
        unsigned int step)
    {
        const int gs = stats->gs;
        const int cx = stats->cx, cy = stats->cy, cz = stats->cz;
        const int V  = gs * gs * gs;
        const int cidx = cz * gs * gs + cy * gs + cx;

        if (step == 0) {
            auto env = model.Environment();
            stats->params << "PARAM_MOLECULAR_STEPS," << substep_count << "\n";
            stats->params << "PARAM_SEC_PER_SLICE,"
                          << env.getProperty<float>("PARAM_SEC_PER_SLICE") << "\n";
            stats->params << "grid_size," << gs << "\n";
            stats->params << "voxel_size," << env.getProperty<float>("PARAM_VOXEL_SIZE_CM") << "\n";
            stats->params << "source_x," << cx << "\n";
            stats->params << "source_y," << cy << "\n";
            stats->params << "source_z," << cz << "\n";

            // Emit decay + diffusivity for each substrate so the figure
            // script can compute L=sqrt(D/k) and the analytical profile.
            auto dump_f = [&](const char* k) {
                stats->params << k << "," << env.getProperty<float>(k) << "\n";
            };
            dump_f("PARAM_IFNG_DIFFUSIVITY");   dump_f("PARAM_IFNG_DECAY_RATE");
            dump_f("PARAM_IL2_DIFFUSIVITY");    dump_f("PARAM_IL2_DECAY_RATE");
            dump_f("PARAM_IL10_DIFFUSIVITY");   dump_f("PARAM_IL10_DECAY_RATE");
            dump_f("PARAM_TGFB_DIFFUSIVITY");   dump_f("PARAM_TGFB_DECAY_RATE");
            dump_f("PARAM_CCL2_DIFFUSIVITY");   dump_f("PARAM_CCL2_DECAY_RATE");
            dump_f("PARAM_ARGI_DIFFUSIVITY");   dump_f("PARAM_ARGI_DECAY_RATE");
            dump_f("PARAM_NO_DIFFUSIVITY");     dump_f("PARAM_NO_DECAY_RATE");
            dump_f("PARAM_IL12_DIFFUSIVITY");   dump_f("PARAM_IL12_DECAY_RATE");
            dump_f("PARAM_VEGFA_DIFFUSIVITY");  dump_f("PARAM_VEGFA_DECAY_RATE");
            dump_f("PARAM_IL1_DIFFUSIVITY");    dump_f("PARAM_IL1_DECAY_RATE");
            dump_f("PARAM_IL6_DIFFUSIVITY");    dump_f("PARAM_IL6_DECAY_RATE");
            dump_f("PARAM_CXCL13_DIFFUSIVITY"); dump_f("PARAM_CXCL13_DECAY_RATE");
            dump_f("PARAM_MMP_DIFFUSIVITY");    dump_f("PARAM_MMP_DECAY_RATE");
            dump_f("PARAM_ANTIBODY_DIFFUSIVITY"); dump_f("PARAM_ANTIBODY_DECAY_RATE");
            dump_f("PARAM_CCL21_DIFFUSIVITY");  dump_f("PARAM_CCL21_DECAY_RATE");
            dump_f("PARAM_CXCL12_DIFFUSIVITY"); dump_f("PARAM_CXCL12_DECAY_RATE");
            dump_f("PARAM_CCL5_DIFFUSIVITY");   dump_f("PARAM_CCL5_DECAY_RATE");
            stats->params.flush();
        }

        // Dump x-line cross-section (through center) for each non-O2 substrate.
        std::vector<float> conc(V);
        for (const auto& [idx, name] : NON_O2_SUBSTRATES) {
            PDAC::g_pde_solver->get_concentrations(conc.data(), idx);
            for (int x = 0; x < gs; x++) {
                const int vidx = cz * gs * gs + cy * gs + x;
                stats->fields << step << "," << name << "," << idx << ","
                              << x << "," << conc[vidx] << "\n";
            }
        }

        // Re-pin center voxel to 1.0 on every substrate so next step's solve
        // sees a persistent Dirichlet-like boundary condition at center.
        const float unity = 1.0f;
        for (const auto& [idx, name] : NON_O2_SUBSTRATES) {
            float* d_conc = PDAC::g_pde_solver->get_device_concentration_ptr(idx);
            cudaMemcpy(&d_conc[cidx], &unity, sizeof(float),
                       cudaMemcpyHostToDevice);
        }

        if (step == 0 || step + 1 == 60 || (step + 1) % 15 == 0) {
            std::cout << "  [step " << step << "] written x-lines for "
                      << NON_O2_SUBSTRATES.size() << " substrates" << std::endl;
        }

        if (step + 1 == 60) {
            stats->fields.flush();
            std::cout << "\n======== PDE substep sweep (substeps="
                      << substep_count << ") complete ========\n"
                      << "  Outputs: " << out_dir << "/\n"
                      << "===========================================\n"
                      << std::endl;
        }
    };

    return cfg;
}

// Register five variants.
static const bool registered = []() {
    for (int substeps : {6, 18, 36, 360, 3600}) {
        const std::string name = "pde_substep_sub" + std::to_string(substeps);
        register_scenario(name,
            [substeps, name](const std::string& param_file) {
                return build_pde_substep(name, substeps, param_file);
            });
    }
    return true;
}();

} // anonymous namespace
