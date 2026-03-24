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

    // 3. Mark recruitment sources.
    //    Matches HCC update_vas() which marks T/MDSC/MAC source voxels before recruitment.
    {
        flamegpu::LayerDescription layer = model.newLayer("mark_vascular_t_sources");
        layer.addAgentFunction(AGENT_VASCULAR, "mark_t_sources");
    }
    {
        flamegpu::LayerDescription layer = model.newLayer("mark_mdsc_sources");
        layer.addHostFunction(mark_mdsc_sources);
    }
    {
        flamegpu::LayerDescription layer = model.newLayer("mark_mac_sources");
        layer.addHostFunction(mark_mac_sources);
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
    }
    {
        flamegpu::LayerDescription layer = model.newLayer("reset_moves_cancer");
        layer.addAgentFunction(AGENT_CANCER_CELL, "reset_moves");
    }
    {
        const int cancer_steps = model.Environment().getProperty<int>("PARAM_CANCER_MOVE_STEPS_STEM");
        const int tcell_steps  = model.Environment().getProperty<int>("PARAM_TCELL_MOVE_STEPS");
        const int treg_steps   = model.Environment().getProperty<int>("PARAM_TCELL_MOVE_STEPS");
        const int mdsc_steps   = model.Environment().getProperty<int>("PARAM_MDSC_MOVE_STEPS");
        const int mac_steps    = model.Environment().getProperty<int>("PARAM_MAC_MOVE_STEPS");
        const int fib_steps    = model.Environment().getProperty<int>("PARAM_FIB_MOVE_STEPS");

        // Interleaved movement: all mobile agent types compete for voxels in the
        // same layer each substep. Each type's moves are spread evenly across the
        // full substep range so slower types (cancer, MDSC, MAC) aren't front-loaded.
        // Cancer uses moves_remaining (stem=5, progenitor=1) and returns early when
        // exhausted; immune agents run for their full step count.
        const int max_steps = std::max({cancer_steps, tcell_steps, treg_steps, mdsc_steps, mac_steps, fib_steps});
        const auto cancer_on = spread_steps(cancer_steps, max_steps);
        const auto tcell_on  = spread_steps(tcell_steps,  max_steps);
        const auto treg_on   = spread_steps(treg_steps,   max_steps);
        const auto mdsc_on   = spread_steps(mdsc_steps,   max_steps);
        const auto mac_on    = spread_steps(mac_steps,     max_steps);
        const auto fib_on    = spread_steps(fib_steps,     max_steps);
        for (int i = 0; i < max_steps; i++) {
            flamegpu::LayerDescription layer = model.newLayer("move_interleaved_" + std::to_string(i));
            if (cancer_on.count(i)) layer.addAgentFunction(AGENT_CANCER_CELL, "move");
            if (tcell_on.count(i))  layer.addAgentFunction(AGENT_TCELL, "move");
            if (treg_on.count(i))   layer.addAgentFunction(AGENT_TREG, "move");
            if (mdsc_on.count(i))   layer.addAgentFunction(AGENT_MDSC, "move");
            if (mac_on.count(i))    layer.addAgentFunction(AGENT_MACROPHAGE, "move");
            if (fib_on.count(i))    layer.addAgentFunction(AGENT_FIBROBLAST, "move");
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
        flamegpu::LayerDescription layer = model.newLayer("final_scan_neighbors");
        layer.addAgentFunction(AGENT_CANCER_CELL, "count_neighbors");
        layer.addAgentFunction(AGENT_TCELL,       "scan_neighbors");
        layer.addAgentFunction(AGENT_TREG,        "scan_neighbors");
        layer.addAgentFunction(AGENT_MDSC,        "scan_neighbors");
        layer.addAgentFunction(AGENT_MACROPHAGE,  "scan_neighbors");
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
    }

    // 8. Wave-interleaved division (N_DIVIDE_WAVES rounds, cancer/tcell/treg interleaved).
    //    Matches HCC divide logic inside time_slice_state_change(). Wave assignment is
    //    set in state_step so each cell executes only in its assigned wave.
    {
        flamegpu::LayerDescription layer = model.newLayer("reset_divide_wave");
        layer.addHostFunction(reset_divide_wave);
    }
    for (int w = 0; w < N_DIVIDE_WAVES; w++) {
        const std::string ws = std::to_string(w);
        {
            flamegpu::LayerDescription layer = model.newLayer("divide_cancer_w" + ws);
            layer.addAgentFunction(AGENT_CANCER_CELL, "divide");
        }
        {
            flamegpu::LayerDescription layer = model.newLayer("divide_tcell_w" + ws);
            layer.addAgentFunction(AGENT_TCELL, "divide");
        }
        {
            flamegpu::LayerDescription layer = model.newLayer("divide_treg_w" + ws);
            layer.addAgentFunction(AGENT_TREG, "divide");
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
        flamegpu::LayerDescription layer = model.newLayer("fib_activate");
        layer.addAgentFunction(AGENT_FIBROBLAST, "activate");
    }

    // ── Timing checkpoint: after division ──
    {
        flamegpu::LayerDescription layer = model.newLayer("timing_after_division");
        layer.addHostFunction(timing_after_division);
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
