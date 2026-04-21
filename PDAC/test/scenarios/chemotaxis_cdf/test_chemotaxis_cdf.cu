// Test 1 — Chemotaxis CDF
//
// Verifies the gradient-weighted CDF movement helper produces the expected
// chemotactic index (CI). Seeds 200 T_CELL_EFF on a sparse slab around the
// grid center, pins CCL5 to a linear +x gradient, and measures the empirical
// CI over all cell movements.
//
// On the 26-neighbor Moore lattice, expected CI = bias/3 (see ci_to_bias
// docstring in common.cuh). move_cell() uses ci_to_bias(CI) = 3*CI, so the
// empirical <cos(theta_to_gradient)> should recover PARAM_CHEMO_CI_TCELL_EFF.
//
// Layers enabled: occupancy, movement, pde_gradients.
// Layers disabled: pde_solve (keep field pinned), neighbor_scan (keep default
// adh_p_move=1.0), state_transition (keep cells in EFF state), chemical_sources
// (T cells don't perturb field), division, ECM update, recruitment, QSP.

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
#include <tuple>
#include <cmath>
#include <vector>

using namespace PDAC;
using namespace PDAC::test;

// Forward-declared in test_main.cu; scenarios register themselves at static init.
void register_scenario(const std::string& name,
    std::function<TestConfig(const std::string&)> builder);

namespace {

struct ChemoStats {
    std::map<int, std::tuple<int,int,int>> prev_pos;  // agent_id -> last (x,y,z)
    double sum_cos = 0.0;       // sum of cos(angle_to_gradient) over all moves
    int n_moves = 0;
    int n_still = 0;            // cell-steps where cell did not move
    std::ofstream traj;         // per-step trajectory CSV
    std::ofstream summary;      // running CI summary
};

static TestConfig build_chemotaxis_cdf(const std::string& param_file) {
    TestConfig cfg;
    cfg.name = "chemotaxis_cdf";
    // Grid must be large enough that cells don't hit the +x wall within `steps`:
    // with 1 move substep/step and CI≈0.17, mean per-step drift is ~0.24 voxels;
    // over 200 steps cells drift ~48 voxels. Seed at x=10–19 gives ≥30-voxel
    // headroom to the +x wall at x=100.
    cfg.grid_size = 101;
    cfg.voxel_size = 20.0f;
    cfg.steps = 200;
    cfg.seed = 42;
    cfg.param_file = param_file;

    // --- Layer subset: isolate chemotaxis mechanism ---
    cfg.layers.ecm_update = false;
    cfg.layers.recruitment = false;
    cfg.layers.occupancy = true;        // movement needs vol_used grid
    cfg.layers.movement = true;
    cfg.layers.neighbor_scan = false;   // keeps adh_p_move at seed default 1.0
    cfg.layers.state_transition = false;// keep cells in EFF, no death/hypoxia
    cfg.layers.chemical_sources = false;// don't perturb pinned CCL5
    cfg.layers.pde_solve = false;       // keep field pinned
    cfg.layers.pde_gradients = true;    // compute ∇CCL5 each step
    cfg.layers.division = false;
    cfg.layers.qsp = false;
    cfg.layers.abm_export = false;

    // --- Force 1 move substep per ABM step ---
    // Default PARAM_TCELL_MOVE_STEPS=64 would advance each cell ~13 voxels/step
    // (bias-free ~8 voxels + bias ~5 voxels), quickly pinning cells at the +x
    // wall and confounding the aggregate CI. With 1 substep, every observed
    // ABM-step displacement IS a single lattice move, and <cos(theta)> directly
    // recovers CI = bias/3 per the 26-Moore analytical formula.
    cfg.int_env_overrides.push_back({"PARAM_TCELL_MOVE_STEPS", 1});

    // --- Pinned CCL5: linear gradient along +x ---
    // C(x,y,z) = 1.0 + 0.5 * x  [nM], so ∂C/∂x > 0 everywhere.
    // Magnitude doesn't matter for CI (move_cell normalizes gradient).
    cfg.pinned_fields.push_back({CHEM_CCL5, linear_gradient(1.0f, 0.5f, /*axis x*/0)});

    // --- ECM: zero density/crosslink so porosity=1 (no barrier) ---
    cfg.ecm.density   = uniform_field(0.0f);
    cfg.ecm.crosslink = uniform_field(0.0f);
    cfg.ecm.floor     = uniform_field(0.0f);

    // --- Seed 200 T_CELL_EFF on a 10x10x2 slab ---
    // Position at low-x end so cells have room to drift along the +x gradient
    // without hitting the far wall (grid=101, seed max x=19 → ~80-voxel headroom).
    for (int k = 0; k < 2; k++) {
        for (int j = 0; j < 10; j++) {
            for (int i = 0; i < 10; i++) {
                AgentSeed s;
                s.agent_type = AGENT_TCELL;
                s.x = 10 + i;
                s.y = 45 + j;
                s.z = 49 + k;
                s.cell_state = T_CELL_EFF;
                s.int_vars.push_back({"life", 9999});
                s.int_vars.push_back({"divide_limit", 0});
                s.int_vars.push_back({"divide_flag", 0});
                s.int_vars.push_back({"divide_cd", 0});
                s.int_vars.push_back({"hypoxia_exposure", 0});
                s.float_vars.push_back({"hypoxia_kill_factor", 1.0f});
                s.float_vars.push_back({"adh_p_move", 1.0f});
                cfg.agents.push_back(s);
            }
        }
    }

    // --- Per-step callback: track displacements, accumulate CI stats ---
    auto stats = std::make_shared<ChemoStats>();
    const std::string out_dir = "../test/scenarios/" + cfg.name + "/outputs";
    std::filesystem::create_directories(out_dir);
    stats->traj.open(out_dir + "/trajectories.csv");
    stats->traj << "step,id,x,y,z\n";
    stats->summary.open(out_dir + "/ci_running.csv");
    stats->summary << "step,n_moves,n_still,ci_running\n";

    cfg.step_callback = [stats, out_dir](flamegpu::CUDASimulation& sim,
                                         flamegpu::ModelDescription& model,
                                         unsigned int step)
    {
        // One-shot diagnostic on step 0: verify pinned CCL5 concentration and
        // computed gradient are present on device, then dump the z=49 slice
        // for Python to draw as a background heatmap behind agent snapshots.
        if (step == 0) {
            const int gs = model.Environment().getProperty<int>("grid_size_x");
            const int V = gs * gs * gs;
            std::vector<float> conc(V), gx(V);
            cudaMemcpy(conc.data(),
                       PDAC::g_pde_solver->get_device_concentration_ptr(CHEM_CCL5),
                       V * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(gx.data(),
                       PDAC::g_pde_solver->get_device_gradx_ptr(GRAD_CCL5),
                       V * sizeof(float), cudaMemcpyDeviceToHost);
            const int c = gs / 2;
            const int center_idx = c * gs * gs + c * gs + c;
            std::cout << "  [diag] CCL5 conc at (c,c,c)=" << conc[center_idx]
                      << "  at (0,c,c)=" << conc[c*gs*gs + c*gs + 0]
                      << "  at (gs-1,c,c)=" << conc[c*gs*gs + c*gs + gs-1] << "\n";
            std::cout << "  [diag] CCL5 grad_x at (c,c,c)=" << gx[center_idx]
                      << "  at (5,c,c)=" << gx[c*gs*gs + c*gs + 5]
                      << "  at (15,c,c)=" << gx[c*gs*gs + c*gs + 15] << "\n";

            // Field is pinned (never changes); dump the z=49 slice once.
            const int z_slice = 49;
            std::ofstream slice_out(out_dir + "/ccl5_slice_z49.csv");
            slice_out << "x,y,conc\n";
            for (int y = 0; y < gs; y++) {
                for (int x = 0; x < gs; x++) {
                    const int idx = z_slice * gs * gs + y * gs + x;
                    slice_out << x << "," << y << "," << conc[idx] << "\n";
                }
            }
        }

        flamegpu::AgentVector pop(model.Agent(AGENT_TCELL));
        sim.getPopulationData(pop);

        for (unsigned int i = 0; i < pop.size(); i++) {
            const int id = pop[i].getID();
            const int x = pop[i].getVariable<int>("x");
            const int y = pop[i].getVariable<int>("y");
            const int z = pop[i].getVariable<int>("z");

            stats->traj << step << "," << id << "," << x << "," << y << "," << z << "\n";

            auto it = stats->prev_pos.find(id);
            if (it != stats->prev_pos.end()) {
                int px, py, pz;
                std::tie(px, py, pz) = it->second;
                const int dx = x - px, dy = y - py, dz = z - pz;
                if (dx == 0 && dy == 0 && dz == 0) {
                    stats->n_still++;
                } else {
                    const float mag = std::sqrt(static_cast<float>(dx*dx + dy*dy + dz*dz));
                    // Gradient is along +x unit vector, so cos(theta) = dx / |disp|.
                    stats->sum_cos += static_cast<double>(dx) / mag;
                    stats->n_moves++;
                }
            }
            stats->prev_pos[id] = {x, y, z};
        }

        const double ci_running = stats->n_moves > 0
            ? stats->sum_cos / stats->n_moves : 0.0;
        stats->summary << step << "," << stats->n_moves << ","
                       << stats->n_still << "," << ci_running << "\n";

        if (step == 0 || step == 49 || step == 99 || step == 149 || step == 199) {
            std::cout << "  [step " << step << "] moves=" << stats->n_moves
                      << " still=" << stats->n_still
                      << " CI_running=" << ci_running << std::endl;
        }

        // Final report with pass/fail
        if (step + 1 == 200) {
            const float expected_CI =
                model.Environment().getProperty<float>("PARAM_CHEMO_CI_TCELL_EFF");
            const double empirical_CI = stats->n_moves > 0
                ? stats->sum_cos / stats->n_moves : 0.0;
            const double err = std::abs(empirical_CI - expected_CI);
            const double tol = 0.03;

            std::cout << "\n==================== Chemotaxis CDF result ====================\n"
                      << "  Expected CI (PARAM_CHEMO_CI_TCELL_EFF): " << expected_CI << "\n"
                      << "  Empirical CI:                           " << empirical_CI << "\n"
                      << "  |err|:                                   " << err
                      << "   (tol " << tol << ")\n"
                      << "  Total moves: " << stats->n_moves
                      << "   Still: " << stats->n_still << "\n"
                      << "  Result: " << (err < tol ? "PASS" : "FAIL") << "\n"
                      << "================================================================\n"
                      << std::endl;
        }
    };

    return cfg;
}

// Static init: register scenario with the global registry.
static const bool chemotaxis_cdf_registered = []() {
    register_scenario("chemotaxis_cdf", build_chemotaxis_cdf);
    return true;
}();

} // anonymous namespace
