// Test 5 — ECM Contact Guidance
//
// Validates the fiber-anisotropy branch of move_cell() against an analytic
// expectation on a uniformly-oriented ECM field. Two scenarios:
//
//   contact_guidance      : orient = (1,0,0), |orient| = 1 everywhere
//   contact_guidance_iso  : orient = (0,0,0) everywhere (control)
//
// Mechanism (cancer progenitor, bias=0, no chemotaxis):
//   apply_contact_guidance_persist picks sign from (persist_dir · fiber_hat).
//   Starting persist_dir=(0,0,0) → pdot=0 → sign=+1. Writes
//   grad = sign·fiber·w_cg, bias_strength = fiber_mag = 1.
//
// In the unified CDF (bias_strength=1, 26-Moore, fiber barrier at destination):
//   w_i = max(0, 1 + 1·cos(d_i, ghat)) * max(0, 1 - barrier · sin²_fiber)
// With barrier=0.55 and ghat=sign·(1,0,0):
//   Class           n   cos    w_grad   sin²    w_barrier    w_i
//   +x face         1    1      2       0       1            2.000
//   -x face         1   -1      0       0       1            0
//   ±y,±z face      4    0      1       1       0.45         0.450
//   +x edge xy/xz   4   +1/√2   1.707   0.5     0.725        1.237
//   -x edge xy/xz   4   -1/√2   0.293   0.5     0.725        0.212
//   yz edge         4    0      1       1       0.45         0.450
//   +x corner       4   +1/√3   1.577   2/3     0.633        0.999
//   -x corner       4   -1/√3   0.423   2/3     0.633        0.268
//   Σw = 16.464
//
// SINGLE-MODE expectations (conditional on sign=+1, i.e. ghat=+x):
//   <Δx>  = (2 + 4·1.237 + 4·0.999 - 4·0.212 - 4·0.268) / 16.464 = 0.548
//   <Δx²> = (2 + 4·1.237 + 4·0.212 + 4·0.999 + 4·0.268) / 16.464 = 0.781
//   <Δy²> = (2·0.45 + 2·1.237 + 2·0.212 + 4·0.45 + 4·0.999 + 4·0.268) / 16.464 = 0.648
//   Anisotropy <Δx²>/<Δy²> = 1.206
//
// POPULATION-MEAN expectation — sign-flip Markov mixing:
//   After each move, persist_dir := last move direction, and next step's
//   sign = (persist_dir · fiber_hat ≥ 0 ? +1 : -1). This flips the effective
//   gradient between +x and -x modes. Modes are symmetric (<Δx>_± = ±0.548)
//   and second moments are unchanged, but <Δx> is reduced by mixing.
//
//   Transition rates per step:
//     +x mode → -x mode: P(-x move) = (0 + 4·0.212 + 4·0.268) / 16.464 = 0.117
//     -x mode → +x mode: any move with pdot ≥ 0 (-x symmetric):
//                        = (non-(-x) weight in -x mode) / 16.464 = 5.520/16.464 = 0.335
//   Steady state: π₊/π₋ = 0.335/0.117 ≈ 2.87 → π₊ = 0.742, π₋ = 0.258.
//   Net <Δx> = 0.548 · (π₊ - π₋) = 0.548 · 0.484 = 0.265
//
// Note: because sign-flipping correlates consecutive moves, effective N is
// ~1/5 the raw count. For N=3625 moves, SE(<Δx>) ≈ 0.03 accounting for
// autocorrelation. Tolerance 0.05 is ~1.6σ.
//
// Per-move expectations (iso, bias=0, uniform over 26-Moore):
//   <Δx> = <Δy> = <Δz> = 0
//   <Δx²> = <Δy²> = <Δz²> = 18/26 = 0.6923
//
// Setup:
//   - 101³ grid, 30 ABM steps, 1 move substep/step (override CANCER_MOVE_STEPS=1).
//   - 125 cancer progenitors on a 5x5x5 spacing-3 lattice (span = 12 voxels)
//     seeded at x,y,z ∈ {10,13,16,19,22} × {44,47,50,53,56} × {44,47,50,53,56}.
//     Spacing >= 2 ensures no Moore-neighbor overlap → cells don't block each
//     other and the analytic single-cell drift prediction holds.
//     Cluster translates by ~22 voxels in +x over 30 steps → leading cell at
//     x~44, safe from 101-voxel wall.
//   - ECM density/crosslink/floor = 0 (porosity = 1, no barrier from density).
//   - Zero chemical gradients (no PDE solve).
//   - neighbor_scan DISABLED — no same-type contacts anyway with spacing 3.
//     adh_p_move seeded to 1.0 so no adhesion drag.

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

void register_scenario(const std::string& name,
    std::function<TestConfig(const std::string&)> builder);

namespace {

struct CGStats {
    // Per-cell last position (for per-step displacement deltas).
    std::map<int, std::tuple<int,int,int>> prev_pos;
    // Running sums across all cells, all moves (step > 0 only).
    double sum_dx = 0.0, sum_dy = 0.0, sum_dz = 0.0;
    double sum_dx2 = 0.0, sum_dy2 = 0.0, sum_dz2 = 0.0;
    long n_moves = 0;  // number of displacement observations
    long n_still = 0;  // cells that didn't move this step (blocked/adhesion)
    // Seeded positions (for total MSD from start).
    std::map<int, std::tuple<int,int,int>> seed_pos;
    std::ofstream traj;
    std::ofstream per_step;
    // Is this the aligned or iso scenario?
    bool aligned = true;
};

// Common builder body, parameterized by orient field setting.
static TestConfig build_common(const std::string& name,
                               const std::string& param_file,
                               bool aligned)
{
    TestConfig cfg;
    cfg.name = name;
    cfg.grid_size = 101;
    cfg.voxel_size = 20.0f;
    cfg.steps = 30;
    cfg.seed = 42;
    cfg.param_file = param_file;

    // --- Layer subset: movement + adhesion only ---
    cfg.layers.ecm_update = false;          // keep orient field fixed at preset
    cfg.layers.recruitment = false;
    cfg.layers.occupancy = true;
    cfg.layers.movement = true;
    cfg.layers.neighbor_scan = false;       // sparse seed → no same-type contacts
    cfg.layers.state_transition = false;
    cfg.layers.chemical_sources = false;
    cfg.layers.pde_solve = false;
    cfg.layers.pde_gradients = false;
    cfg.layers.division = false;
    cfg.layers.qsp = false;
    cfg.layers.abm_export = false;

    // One move substep / ABM step for clean per-move observation.
    cfg.int_env_overrides.push_back({"PARAM_CANCER_MOVE_STEPS", 1});

    // --- ECM: zero density/crosslink/floor → porosity = 1 everywhere ---
    cfg.ecm.density   = uniform_field(0.0f);
    cfg.ecm.crosslink = uniform_field(0.0f);
    cfg.ecm.floor     = uniform_field(0.0f);

    // --- ECM orientation ---
    if (aligned) {
        // Unit vector along +x everywhere. Magnitude 1 = full alignment.
        cfg.ecm.orient_x = uniform_field(1.0f);
        cfg.ecm.orient_y = uniform_field(0.0f);
        cfg.ecm.orient_z = uniform_field(0.0f);
    } else {
        // Isotropic: zero orientation → apply_contact_guidance_persist early-returns.
        cfg.ecm.orient_x = uniform_field(0.0f);
        cfg.ecm.orient_y = uniform_field(0.0f);
        cfg.ecm.orient_z = uniform_field(0.0f);
    }

    // --- Seed 200 cancer progenitors in a 10x10x2 block (aligned run:
    //     starts near -x wall so 30-step drift of ~22 voxels leaves headroom) ---
    // For iso, same seed for consistency (no drift anyway).
    auto stats = std::make_shared<CGStats>();
    stats->aligned = aligned;

    // 5x5x5 sparse lattice at spacing 3.
    for (int k = 0; k < 5; k++) {
        for (int j = 0; j < 5; j++) {
            for (int i = 0; i < 5; i++) {
                AgentSeed s;
                s.agent_type = AGENT_CANCER_CELL;
                s.x = 10 + 3 * i;   // 10, 13, 16, 19, 22
                s.y = 44 + 3 * j;   // 44, 47, 50, 53, 56
                s.z = 44 + 3 * k;   // 44, 47, 50, 53, 56
                s.cell_state = CANCER_PROGENITOR;
                s.int_vars.push_back({"divideCountRemaining", 0});
                s.int_vars.push_back({"divideFlag", 0});
                s.int_vars.push_back({"divideCD", 9999});
                s.int_vars.push_back({"life", 9999});
                s.float_vars.push_back({"adh_p_move", 1.0f});
                cfg.agents.push_back(s);
            }
        }
    }

    const std::string out_dir = "../test/scenarios/" + cfg.name + "/outputs";
    std::filesystem::create_directories(out_dir);
    stats->traj.open(out_dir + "/trajectories.csv");
    stats->traj << "step,id,x,y,z\n";
    stats->per_step.open(out_dir + "/per_step_stats.csv");
    stats->per_step << "step,n,mean_dx,mean_dy,mean_dz,"
                       "mean_dx2,mean_dy2,mean_dz2\n";

    cfg.step_callback = [stats](flamegpu::CUDASimulation& sim,
                                flamegpu::ModelDescription& model,
                                unsigned int step)
    {
        flamegpu::AgentVector pop(model.Agent(AGENT_CANCER_CELL));
        sim.getPopulationData(pop);

        // Per-step accumulators (this step's moves only).
        double step_sum_dx = 0, step_sum_dy = 0, step_sum_dz = 0;
        double step_sum_dx2 = 0, step_sum_dy2 = 0, step_sum_dz2 = 0;
        long step_n = 0;

        for (unsigned int i = 0; i < pop.size(); i++) {
            const int id = pop[i].getID();
            const int x = pop[i].getVariable<int>("x");
            const int y = pop[i].getVariable<int>("y");
            const int z = pop[i].getVariable<int>("z");

            stats->traj << step << "," << id << "," << x << "," << y << "," << z << "\n";

            if (step == 0) {
                stats->seed_pos[id] = {x, y, z};
            }

            auto it = stats->prev_pos.find(id);
            if (it != stats->prev_pos.end()) {
                int px, py, pz;
                std::tie(px, py, pz) = it->second;
                const int dx = x - px, dy = y - py, dz = z - pz;
                if (dx == 0 && dy == 0 && dz == 0) {
                    stats->n_still++;
                } else {
                    stats->sum_dx  += dx;   stats->sum_dy  += dy;   stats->sum_dz  += dz;
                    stats->sum_dx2 += dx*dx; stats->sum_dy2 += dy*dy; stats->sum_dz2 += dz*dz;
                    stats->n_moves++;
                    step_sum_dx += dx;  step_sum_dy += dy;  step_sum_dz += dz;
                    step_sum_dx2 += dx*dx; step_sum_dy2 += dy*dy; step_sum_dz2 += dz*dz;
                    step_n++;
                }
            }
            stats->prev_pos[id] = {x, y, z};
        }

        if (step > 0 && step_n > 0) {
            stats->per_step << step << "," << step_n << ","
                            << (step_sum_dx / step_n) << ","
                            << (step_sum_dy / step_n) << ","
                            << (step_sum_dz / step_n) << ","
                            << (step_sum_dx2 / step_n) << ","
                            << (step_sum_dy2 / step_n) << ","
                            << (step_sum_dz2 / step_n) << "\n";
        }

        if (step == 0 || step == 9 || step == 19 || step == 29) {
            const double mx = stats->n_moves > 0 ? stats->sum_dx / stats->n_moves : 0.0;
            const double mxs = stats->n_moves > 0 ? stats->sum_dx2 / stats->n_moves : 0.0;
            const double mys = stats->n_moves > 0 ? stats->sum_dy2 / stats->n_moves : 0.0;
            std::cout << "  [step " << step << "] n=" << stats->n_moves
                      << " <dx>=" << mx
                      << " <dx2>=" << mxs
                      << " <dy2>=" << mys
                      << " still=" << stats->n_still << std::endl;
        }

        if (step + 1 == 30) {
            const long N = stats->n_moves;
            const double mdx  = N > 0 ? stats->sum_dx  / N : 0.0;
            const double mdy  = N > 0 ? stats->sum_dy  / N : 0.0;
            const double mdz  = N > 0 ? stats->sum_dz  / N : 0.0;
            const double mdx2 = N > 0 ? stats->sum_dx2 / N : 0.0;
            const double mdy2 = N > 0 ? stats->sum_dy2 / N : 0.0;
            const double mdz2 = N > 0 ? stats->sum_dz2 / N : 0.0;
            const double aniso = mdy2 > 1e-12 ? mdx2 / mdy2 : 0.0;

            double exp_mdx, exp_mdx2, exp_mdy2, exp_aniso;
            if (stats->aligned) {
                // Sign-flip Markov mixing: see header comment.
                exp_mdx   = 0.265;
                exp_mdx2  = 0.781;
                exp_mdy2  = 0.648;
                exp_aniso = 1.206;
            } else {
                exp_mdx   = 0.0;
                exp_mdx2  = 18.0 / 26.0;  // 0.6923
                exp_mdy2  = 18.0 / 26.0;
                exp_aniso = 1.0;
            }

            // Tolerances:
            //  Aligned <Δx>: SE ≈ 0.03 w/ sign-flip autocorrelation at N=3625.
            //                tol 0.05 is ~1.6σ.
            //  Anisotropy:   tight since both moments track analytic well.
            //  Iso <Δx>:     ~0 by symmetry.
            const double tol_mean = 0.05;
            const double tol_aniso = stats->aligned ? 0.05 : 0.15;

            const double err_mdx   = std::abs(mdx - exp_mdx);
            const double err_aniso = std::abs(aniso - exp_aniso);
            const bool pass_mdx    = err_mdx < tol_mean;
            const bool pass_aniso  = err_aniso < tol_aniso;
            const bool pass_sym    = std::abs(mdy) < tol_mean && std::abs(mdz) < tol_mean;
            const bool pass = pass_mdx && pass_aniso && pass_sym;

            std::cout << "\n==================== Contact guidance ("
                      << (stats->aligned ? "aligned" : "iso") << ") ====================\n"
                      << "  Moves observed:  " << N
                      << "   Still:  " << stats->n_still << "\n"
                      << "  <Δx>/step:       " << mdx   << "   (exp " << exp_mdx
                      << ", |err|=" << err_mdx << ", tol " << tol_mean << ") "
                      << (pass_mdx ? "PASS" : "FAIL") << "\n"
                      << "  <Δy>/step:       " << mdy   << "   (exp 0, tol " << tol_mean << ") "
                      << (std::abs(mdy) < tol_mean ? "PASS" : "FAIL") << "\n"
                      << "  <Δz>/step:       " << mdz   << "   (exp 0, tol " << tol_mean << ") "
                      << (std::abs(mdz) < tol_mean ? "PASS" : "FAIL") << "\n"
                      << "  <Δx²>/step:      " << mdx2  << "   (exp " << exp_mdx2 << ")\n"
                      << "  <Δy²>/step:      " << mdy2  << "   (exp " << exp_mdy2 << ")\n"
                      << "  <Δz²>/step:      " << mdz2  << "   (exp " << exp_mdy2 << ")\n"
                      << "  Anisotropy <Δx²>/<Δy²>: " << aniso
                      << "   (exp " << exp_aniso << ", |err|=" << err_aniso
                      << ", tol " << tol_aniso << ") "
                      << (pass_aniso ? "PASS" : "FAIL") << "\n"
                      << "  Result: " << (pass ? "PASS" : "FAIL") << "\n"
                      << "==========================================================\n"
                      << std::endl;
        }
    };

    return cfg;
}

static TestConfig build_contact_guidance(const std::string& param_file) {
    return build_common("contact_guidance", param_file, /*aligned=*/true);
}

static TestConfig build_contact_guidance_iso(const std::string& param_file) {
    return build_common("contact_guidance_iso", param_file, /*aligned=*/false);
}

// Static init: register both scenarios with the global registry.
static const bool contact_guidance_registered = []() {
    register_scenario("contact_guidance",     build_contact_guidance);
    register_scenario("contact_guidance_iso", build_contact_guidance_iso);
    return true;
}();

} // anonymous namespace
