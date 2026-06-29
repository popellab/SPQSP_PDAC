#include "flamegpu/flamegpu.h"
#include <string>
#include <iostream>
#include <vector>
#include <set>

#include "../core/common.cuh"
#include "../pde/pde_integration.cuh"

// Build a set of evenly-spaced substep indices for an agent type with
// `n_moves` moves spread across `max_steps` total substeps.
// E.g. spread_steps(5, 53) → {0, 10, 21, 31, 42}
static std::set<int> spread_steps(int n_moves, int max_steps) {
    std::set<int> steps;
    if (n_moves <= 0 || max_steps <= 0) return steps;
    if (n_moves >= max_steps) {
        for (int i = 0; i < max_steps; i++) steps.insert(i);
        return steps;
    }
    for (int i = 0; i < n_moves; i++) {
        steps.insert(i * max_steps / n_moves);
    }
    return steps;
}

// From main.cu (global scope): prepares GPU buffer for ABM export
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER prepare_abm_export;

namespace PDAC {

// Extern declarations for host functions not declared in pde_integration.cuh
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER solve_pde_step;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER update_agent_counts;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER solve_qsp_step;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER zero_fib_density_field;
// fib_execute_divide removed: activation is now device-side (fib_activate)
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER aggregate_abm_events;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER copy_abm_counters_to_environment;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER reset_abm_event_counters;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER output_events;

void defineMainModelLayers(flamegpu::ModelDescription& model) {

    // ── Timing checkpoint: start of step ──
    {
        flamegpu::LayerDescription layer = model.newLayer("timing_step_start");
        layer.addHostFunction(timing_step_start);
    }

    // 1. ECM update: zero density field, scatter fibroblast Gaussian kernels, apply decay + secretion.
    //    Matches HCC update_ECM() which runs first in timeSlice before recruitment.
    {
        flamegpu::LayerDescription layer = model.newLayer("zero_fib_density_field");
        layer.addHostFunction(zero_fib_density_field);
    }
    {
        flamegpu::LayerDescription layer = model.newLayer("build_density_field");
        layer.addAgentFunction(AGENT_FIBROBLAST, "build_density_field");
    }
    {
        flamegpu::LayerDescription layer = model.newLayer("update_ecm");
        layer.addHostFunction(update_ecm_grid);
    }
    {
        flamegpu::LayerDescription layer = model.newLayer("decay_antigen");
        layer.addHostFunction(decay_antigen_grid);
    }
    {
        flamegpu::LayerDescription layer = model.newLayer("update_ecm_orientation");
        layer.addHostFunction(update_ecm_orientation);
    }
    {
        flamegpu::LayerDescription layer = model.newLayer("decay_stress_field");
        layer.addHostFunction(decay_stress_field);
    }

    // ── Timing checkpoint: after ECM ──
    {
        flamegpu::LayerDescription layer = model.newLayer("timing_after_ecm");
        layer.addHostFunction(timing_after_ecm);
    }

    // 2. Step bookkeeping: counts, event counter reset, recruitment source reset.
    {
        flamegpu::LayerDescription layer = model.newLayer("update_agent_counts");
        layer.addHostFunction(update_agent_counts);
    }
    {
        flamegpu::LayerDescription layer = model.newLayer("reset_abm_event_counters_start");
        layer.addHostFunction(reset_abm_event_counters);
    }
    {
        flamegpu::LayerDescription layer = model.newLayer("reset_recruitment_sources");
        layer.addHostFunction(reset_recruitment_sources);
    }
    {
        flamegpu::LayerDescription layer = model.newLayer("update_vasculature_count");
        layer.addHostFunction(update_vasculature_count);
    }

    // 3. Mark recruitment entry points from the Vvas field (media-2 Eq.4).
    //    Replaces per-vascular-agent marking: p_entry = Vvas * PARAM_ENTRY_ADHESION_SCALE,
    //    decoupled from vascular agent positions/counts (no vas_scaler). MDSC/MAC CCL2-gated.
    {
        flamegpu::LayerDescription layer = model.newLayer("mark_entry_points");
        layer.addHostFunction(mark_entry_points);
    }

    // 4. Recruitment: GPU kernel decides placement, thin host fn creates agents.
    //    Matches HCC time_slice_recruitment(). All occupancy checking runs on GPU.
    {
        flamegpu::LayerDescription layer = model.newLayer("recruit_gpu");
        layer.addHostFunction(recruit_gpu);
    }
    {
        flamegpu::LayerDescription layer = model.newLayer("place_recruited_agents");
        layer.addHostFunction(place_recruited_agents);
    }

    // ── Timing checkpoint: after recruitment ──
    {
        flamegpu::LayerDescription layer = model.newLayer("timing_after_recruit");
        layer.addHostFunction(timing_after_recruit);
    }

    // 5. Occupancy grid + movement.
    //    Matches HCC time_slice_movement(). Occ grid is built here so newly recruited
    //    cells block voxels before movement, and movement uses current positions.
    {
        flamegpu::LayerDescription layer = model.newLayer("zero_occ_grid");
        layer.addHostFunction(zero_occupancy_grid);
    }
    {
        flamegpu::LayerDescription layer = model.newLayer("write_to_occ_grid");
        layer.addAgentFunction(AGENT_CANCER_CELL, "write_to_occ_grid");
        layer.addAgentFunction(AGENT_TCELL,       "write_to_occ_grid");
        layer.addAgentFunction(AGENT_TREG,        "write_to_occ_grid");
        layer.addAgentFunction(AGENT_MDSC,        "write_to_occ_grid");
        layer.addAgentFunction(AGENT_MACROPHAGE,  "write_to_occ_grid");
        layer.addAgentFunction(AGENT_FIBROBLAST,  "write_to_occ_grid");
        layer.addAgentFunction(AGENT_VASCULAR,    "write_to_occ_grid");
        layer.addAgentFunction(AGENT_BCELL,       "write_to_occ_grid");
        layer.addAgentFunction(AGENT_DC,          "write_to_occ_grid");
    }
    // ── Timing checkpoint: after occupancy grid (zero + write) ──
    {
        flamegpu::LayerDescription layer = model.newLayer("timing_after_occ");
        layer.addHostFunction(timing_after_occ);
    }
    {
        flamegpu::LayerDescription layer = model.newLayer("reset_moves_cancer");
        layer.addAgentFunction(AGENT_CANCER_CELL, "reset_moves");
    }
    {
        // Cancer movement — Step-5 deterministic propose/commit, pulled OUT of the shared
        // batched move rounds below. Per round: reset owner → propose (pick + reserve_voxel)
        // → commit (unique winner moves). Owner reset each round (cheap on the SBI grid).
        const int cancer_steps = std::max(
            model.Environment().getProperty<int>("PARAM_CANCER_MOVE_STEPS"),
            model.Environment().getProperty<int>("PARAM_CANCER_MOVE_STEPS_STEM"));
        for (int r = 0; r < cancer_steps; r++) {
            const std::string rs = std::to_string(r);
            {
                flamegpu::LayerDescription layer = model.newLayer("reset_voxel_owner_cmove_" + rs);
                layer.addHostFunction(reset_voxel_owner);
            }
            {
                flamegpu::LayerDescription layer = model.newLayer("cancer_move_propose_" + rs);
                layer.addAgentFunction(AGENT_CANCER_CELL, "move_propose");
            }
            {
                flamegpu::LayerDescription layer = model.newLayer("cancer_move_commit_" + rs);
                layer.addAgentFunction(AGENT_CANCER_CELL, "move_commit");
            }
        }
    }
    {
        // Batched-round movement (PARAM_MOVE_BATCH = voxel-moves per agent per round).
        // Each round is ONE layer containing every mobile type; each agent's move kernel
        // does up to M voxel-moves in an internal loop (reading+writing the volume
        // occupancy on every move via run_move_batch), draining its moves_remaining budget
        // (reset in write_to_occ_grid / cancer_reset_moves). K = ceil(max move_steps / M)
        // round-layers REPLACE the old per-substep layers (71 → ~8), collapsing FLAMEGPU
        // per-layer overhead while keeping types interleaved to within M voxels per round.
        // Cancer is unbatched (1 voxel/call + stress deposit), so it occupies the first
        // `cancer_steps` rounds (1 move each).
        const int M = std::max(1, model.Environment().getProperty<int>("PARAM_MOVE_BATCH"));
        const int cancer_steps = std::max(
            model.Environment().getProperty<int>("PARAM_CANCER_MOVE_STEPS"),
            model.Environment().getProperty<int>("PARAM_CANCER_MOVE_STEPS_STEM"));
        const int tcell_steps  = model.Environment().getProperty<int>("PARAM_TCELL_MOVE_STEPS");
        const int mdsc_steps   = model.Environment().getProperty<int>("PARAM_MDSC_MOVE_STEPS");
        const int mac_steps    = model.Environment().getProperty<int>("PARAM_MAC_MOVE_STEPS");
        const int fib_steps    = model.Environment().getProperty<int>("PARAM_FIB_MOVE_STEPS");
        const int bcell_steps  = model.Environment().getProperty<int>("PARAM_BCELL_MOVE_STEPS");
        const int dc_steps     = model.Environment().getProperty<int>("PARAM_DC_MOVE_STEPS");

        // K = number of rounds (set by the fastest type / move_batch). Each type spreads its
        // moves PROPORTIONALLY across all K rounds (per_round = ceil(its_steps / K), computed
        // in the move fn from env "MOVE_K"), so slow types (e.g. fib, 8 steps) move 1/round
        // across all rounds rather than racing to their final position in round 0. All types
        // appear in every round and early-return once their moves_remaining is drained.
        const int max_steps = std::max({tcell_steps, mdsc_steps, mac_steps, fib_steps,
                                        bcell_steps, dc_steps, cancer_steps});
        const int K = std::max(1, (max_steps + M - 1) / M);  // ceil(max_steps / move_batch)
        model.Environment().newProperty<int>("MOVE_K", K);

        for (int r = 0; r < K; r++) {
            flamegpu::LayerDescription layer = model.newLayer("move_round_" + std::to_string(r));
            // Cancer moved deterministically above (propose/commit); not in the shared round.
            layer.addAgentFunction(AGENT_TCELL, "move");
            layer.addAgentFunction(AGENT_TREG, "move");
            layer.addAgentFunction(AGENT_MDSC, "move");
            layer.addAgentFunction(AGENT_MACROPHAGE, "move");
            layer.addAgentFunction(AGENT_FIBROBLAST, "move");
            layer.addAgentFunction(AGENT_BCELL, "move");
            layer.addAgentFunction(AGENT_DC, "move");
        }
        {
            flamegpu::LayerDescription layer = model.newLayer("move_vascular");
            layer.addAgentFunction(AGENT_VASCULAR, "move");
        }
    }

    // ── Timing checkpoint: after movement ──
    {
        flamegpu::LayerDescription layer = model.newLayer("timing_after_movement");
        layer.addHostFunction(timing_after_movement);
    }

    // 6. Neighbor scan: broadcast post-move positions then scan Moore neighborhood.
    //    Matches HCC agent_state_scan() inside time_slice_state_change(), which runs
    //    after movement so neighbor counts reflect current positions.
    {
        flamegpu::LayerDescription layer = model.newLayer("final_broadcast_cancer");
        layer.addAgentFunction(AGENT_CANCER_CELL, "broadcast_location");
    }
    {
        flamegpu::LayerDescription layer = model.newLayer("final_broadcast_tcell");
        layer.addAgentFunction(AGENT_TCELL, "broadcast_location");
    }
    {
        flamegpu::LayerDescription layer = model.newLayer("final_broadcast_treg");
        layer.addAgentFunction(AGENT_TREG, "broadcast_location");
    }
    {
        flamegpu::LayerDescription layer = model.newLayer("final_broadcast_mdsc");
        layer.addAgentFunction(AGENT_MDSC, "broadcast_location");
    }
    {
        flamegpu::LayerDescription layer = model.newLayer("final_broadcast_vascular");
        layer.addAgentFunction(AGENT_VASCULAR, "broadcast_location");
    }
    {
        flamegpu::LayerDescription layer = model.newLayer("final_broadcast_macrophage");
        layer.addAgentFunction(AGENT_MACROPHAGE, "broadcast_location");
    }
    {
        flamegpu::LayerDescription layer = model.newLayer("final_broadcast_fibroblast");
        layer.addAgentFunction(AGENT_FIBROBLAST, "broadcast_location");
    }
    {
        flamegpu::LayerDescription layer = model.newLayer("final_broadcast_bcell");
        layer.addAgentFunction(AGENT_BCELL, "broadcast_location");
    }
    {
        flamegpu::LayerDescription layer = model.newLayer("final_broadcast_dc");
        layer.addAgentFunction(AGENT_DC, "broadcast_location");
    }
    // ── Timing checkpoint: after the 9 broadcast-output layers, before the scan ──
    {
        flamegpu::LayerDescription layer = model.newLayer("timing_after_bcast_out");
        layer.addHostFunction(timing_after_bcast_out);
    }
    {
        flamegpu::LayerDescription layer = model.newLayer("final_scan_neighbors");
        layer.addAgentFunction(AGENT_CANCER_CELL, "count_neighbors");
        layer.addAgentFunction(AGENT_TCELL,       "scan_neighbors");
        layer.addAgentFunction(AGENT_TREG,        "scan_neighbors");
        layer.addAgentFunction(AGENT_MDSC,        "scan_neighbors");
        layer.addAgentFunction(AGENT_MACROPHAGE,  "scan_neighbors");
        layer.addAgentFunction(AGENT_FIBROBLAST,  "scan_neighbors");
        layer.addAgentFunction(AGENT_BCELL,       "scan_neighbors");
        layer.addAgentFunction(AGENT_DC,          "scan_neighbors");
    }

    // ── Timing checkpoint: after neighbor scan ──
    {
        flamegpu::LayerDescription layer = model.newLayer("timing_after_broadcast");
        layer.addHostFunction(timing_after_broadcast);
    }

    // 7. State transitions + chemical sources + division.
    //    Matches HCC time_slice_state_change(): agent_state_scan (done above) then
    //    agent_state_step (state + chemical sources + divide) in one pass. PDE buffers
    //    are reset first so agents atomicAdd fresh source/uptake rates for this step's solve.
    {
        flamegpu::LayerDescription layer = model.newLayer("reset_pde_buffers");
        layer.addHostFunction(reset_pde_buffers);
    }
    {
        flamegpu::LayerDescription layer = model.newLayer("state_transitions");
        layer.addAgentFunction(AGENT_CANCER_CELL, "state_step");
        layer.addAgentFunction(AGENT_TCELL,       "state_step");
        layer.addAgentFunction(AGENT_TREG,        "state_step");
        layer.addAgentFunction(AGENT_MDSC,        "state_step");
        layer.addAgentFunction(AGENT_MACROPHAGE,  "state_step");
        layer.addAgentFunction(AGENT_FIBROBLAST,  "state_step");
        layer.addAgentFunction(AGENT_VASCULAR,    "state_step");
        layer.addAgentFunction(AGENT_BCELL,       "state_step");
        layer.addAgentFunction(AGENT_DC,          "state_step");
    }
    {
        flamegpu::LayerDescription layer = model.newLayer("compute_chemical_sources");
        layer.addAgentFunction(AGENT_CANCER_CELL, "compute_chemical_sources");
        layer.addAgentFunction(AGENT_TCELL,       "compute_chemical_sources");
        layer.addAgentFunction(AGENT_TREG,        "compute_chemical_sources");
        layer.addAgentFunction(AGENT_MDSC,        "compute_chemical_sources");
        layer.addAgentFunction(AGENT_MACROPHAGE,  "compute_chemical_sources");
        layer.addAgentFunction(AGENT_FIBROBLAST,  "compute_chemical_sources");
        layer.addAgentFunction(AGENT_VASCULAR,    "compute_chemical_sources");
        layer.addAgentFunction(AGENT_BCELL,       "compute_chemical_sources");
        layer.addAgentFunction(AGENT_DC,          "compute_chemical_sources");
    }

    // ── Timing checkpoint: after state transitions + chemical sources ──
    {
        flamegpu::LayerDescription layer = model.newLayer("timing_after_sources");
        layer.addHostFunction(timing_after_sources);
    }

    // 7b. Prepare ABM export: reset counter and set flag
    {
        flamegpu::LayerDescription layer = model.newLayer("prepare_abm_export");
        layer.addHostFunction(prepare_abm_export);
    }

    // 7c. GPU-side agent packing for async ABM export (controlled by do_abm_export flag)
    {
        flamegpu::LayerDescription layer = model.newLayer("pack_for_export");
        layer.addAgentFunction(AGENT_CANCER_CELL, "pack_for_export");
        layer.addAgentFunction(AGENT_TCELL, "pack_for_export");
        layer.addAgentFunction(AGENT_TREG, "pack_for_export");
        layer.addAgentFunction(AGENT_MDSC, "pack_for_export");
        layer.addAgentFunction(AGENT_MACROPHAGE, "pack_for_export");
        layer.addAgentFunction(AGENT_FIBROBLAST, "pack_for_export");
        layer.addAgentFunction(AGENT_VASCULAR, "pack_for_export");
        layer.addAgentFunction(AGENT_BCELL, "pack_for_export");
        layer.addAgentFunction(AGENT_DC, "pack_for_export");
    }

    // 8. Wave-interleaved division (N_DIVIDE_WAVES rounds, cancer/tcell/treg interleaved).
    //    Matches HCC divide logic inside time_slice_state_change(). Wave assignment is
    //    set in state_step so each cell executes only in its assigned wave.
    {
        flamegpu::LayerDescription layer = model.newLayer("reset_divide_wave");
        layer.addHostFunction(reset_divide_wave);
    }
    // Cancer division — Step-5 deterministic reserve/confirm (priority claim into d_voxel_owner).
    // Pulled out of the wave loop (cancer no longer uses the wave gate). reset → reserve → confirm.
    {
        flamegpu::LayerDescription layer = model.newLayer("reset_voxel_owner_divide");
        layer.addHostFunction(reset_voxel_owner);
    }
    {
        flamegpu::LayerDescription layer = model.newLayer("divide_cancer_reserve");
        layer.addAgentFunction(AGENT_CANCER_CELL, "divide_reserve");
    }
    {
        flamegpu::LayerDescription layer = model.newLayer("divide_cancer_confirm");
        layer.addAgentFunction(AGENT_CANCER_CELL, "divide_confirm");
    }
    for (int w = 0; w < N_DIVIDE_WAVES; w++) {
        const std::string ws = std::to_string(w);
        {
            flamegpu::LayerDescription layer = model.newLayer("divide_tcell_w" + ws);
            layer.addAgentFunction(AGENT_TCELL, "divide");
        }
        {
            flamegpu::LayerDescription layer = model.newLayer("divide_treg_w" + ws);
            layer.addAgentFunction(AGENT_TREG, "divide");
        }
        {
            flamegpu::LayerDescription layer = model.newLayer("divide_bcell_w" + ws);
            layer.addAgentFunction(AGENT_BCELL, "divide");
        }
        {
            flamegpu::LayerDescription layer = model.newLayer("increment_divide_wave_" + ws);
            layer.addHostFunction(increment_divide_wave);
        }
    }
    {
        flamegpu::LayerDescription layer = model.newLayer("divide_vascular");
        layer.addAgentFunction(AGENT_VASCULAR, "vascular_divide");
    }
    {
        flamegpu::LayerDescription layer = model.newLayer("fib_divide");
        layer.addAgentFunction(AGENT_FIBROBLAST, "divide");
    }

    // ── Timing checkpoint: after division ──
    {
        flamegpu::LayerDescription layer = model.newLayer("timing_after_division");
        layer.addHostFunction(timing_after_division);
    }

    // 8b. Vvas field + Krogh O2 sourcing (media-2). Computes vascular volume fraction
    //     Vvas = Hill(VEGF)*(1-f_cancer_moore)*cabo, then adds O2 source/uptake from it.
    //     Runs after cancer O2-uptake (compute_chemical_sources) + after division, before solve.
    //     d_vvas_field persists for recruitment entry-point marking (Step 3).
    {
        flamegpu::LayerDescription layer = model.newLayer("compute_vvas_and_o2");
        layer.addHostFunction(compute_vvas_and_o2);
    }

    // 9. PDE solve + gradient computation.
    //    Matches HCC time_slice_molecular() which runs last, after all cellular events.
    //    Agents wrote sources/uptake in phase 7; solve advances concentrations one timestep.
    //    Gradients are computed here for use by chemotaxis in the next step's movement.
    {
        flamegpu::LayerDescription layer = model.newLayer("solve_pde");
        layer.addHostFunction(solve_pde_step);
    }
    // ── Timing checkpoint: after PDE solve ──
    {
        flamegpu::LayerDescription layer = model.newLayer("timing_after_pde");
        layer.addHostFunction(timing_after_pde);
    }
    {
        flamegpu::LayerDescription layer = model.newLayer("compute_pde_gradients");
        layer.addHostFunction(compute_pde_gradients);
    }
    // ── Timing checkpoint: after gradients ──
    {
        flamegpu::LayerDescription layer = model.newLayer("timing_after_gradients");
        layer.addHostFunction(timing_after_gradients);
    }

    // 10. QSP coupling: aggregate ABM events, advance ODE, export state.
    {
        flamegpu::LayerDescription layer = model.newLayer("aggregate_abm_events");
        layer.addHostFunction(aggregate_abm_events);
    }
    {
        flamegpu::LayerDescription layer = model.newLayer("copy_abm_counters_to_environment");
        layer.addHostFunction(copy_abm_counters_to_environment);
    }
    {
        flamegpu::LayerDescription layer = model.newLayer("solve_qsp");
        layer.addHostFunction(solve_qsp_step);
    }
    {
        flamegpu::LayerDescription layer = model.newLayer("reset_abm_event_counters");
        layer.addHostFunction(reset_abm_event_counters);
    }
}

} // namespace PDAC
