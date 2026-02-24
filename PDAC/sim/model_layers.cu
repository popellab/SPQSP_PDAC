#include "flamegpu/flamegpu.h"
#include <string>
#include <iostream>

#include "../core/common.cuh"
#include "../pde/pde_integration.cuh"

namespace PDAC {

// Debug host functions for tracing execution
FLAMEGPU_HOST_FUNCTION(debug_layer_start) {
    std::cout << "[DEBUG] Layer execution order check" << std::endl;
    std::cout.flush();
}

FLAMEGPU_HOST_FUNCTION(debug_before_broadcast) {
    std::cout << "[DEBUG] Before broadcast layers" << std::endl;
    std::cout.flush();
}

FLAMEGPU_HOST_FUNCTION(debug_before_state_transitions) {
    std::cout << "[DEBUG] Before state_transitions (agent functions with random)" << std::endl;
    std::cout.flush();
}

FLAMEGPU_HOST_FUNCTION(debug_before_movement) {
    std::cout << "[DEBUG] Before movement layers (most random calls)" << std::endl;
    std::cout.flush();
}

FLAMEGPU_HOST_FUNCTION(debug_before_division) {
    std::cout << "[DEBUG] Before division layers" << std::endl;
    std::cout.flush();
}

FLAMEGPU_HOST_FUNCTION(debug_before_divide_cancer) {
    std::cout << "[DEBUG-LAYER] About to execute divide_cancer" << std::endl;
    std::cout.flush();
}

FLAMEGPU_HOST_FUNCTION(debug_before_divide_tcell) {
    std::cout << "[DEBUG-LAYER] About to execute divide_tcell" << std::endl;
    std::cout.flush();
}

FLAMEGPU_HOST_FUNCTION(debug_before_divide_treg) {
    std::cout << "[DEBUG-LAYER] About to execute divide_treg" << std::endl;
    std::cout.flush();
}

FLAMEGPU_HOST_FUNCTION(debug_before_divide_vascular) {
    std::cout << "[DEBUG-LAYER] About to execute divide_vascular" << std::endl;
    std::cout.flush();
}

FLAMEGPU_HOST_FUNCTION(debug_before_fib_divide) {
    std::cout << "[DEBUG-LAYER] About to execute fib_execute_divide" << std::endl;
    std::cout.flush();
}

FLAMEGPU_HOST_FUNCTION(debug_after_fib_divide) {
    std::cout << "[DEBUG-LAYER] Completed fib_execute_divide" << std::endl;
    std::cout.flush();
}

FLAMEGPU_HOST_FUNCTION(debug_after_fib_check) {
    std::cout << "[DEBUG-LAYER] Completed execution after fib_execute_divide" << std::endl;
    std::cout.flush();
}

FLAMEGPU_HOST_FUNCTION(debug_before_aggregate) {
    std::cout << "[DEBUG-LAYER] About to execute aggregate_abm_events" << std::endl;
    std::cout.flush();
}

FLAMEGPU_HOST_FUNCTION(debug_after_aggregate) {
    std::cout << "[DEBUG-LAYER] Completed aggregate_abm_events" << std::endl;
    std::cout.flush();
}

FLAMEGPU_HOST_FUNCTION(debug_after_state_transitions) {
    std::cout << "[DEBUG] State transitions complete" << std::endl;
    std::cout.flush();
}

// Declare HostFunction objects from pde_integration.cu
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER update_agent_chemicals;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER collect_agent_sources;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER solve_pde_step;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER update_agent_counts;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER solve_qsp_step;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER zero_occupancy_grid;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER chk_after_zero_occ;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER chk_after_write_occ;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER chk_after_move_cancer;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER chk_after_move_tcell;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER chk_after_move_treg;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER chk_after_move_mdsc;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER chk_after_move_vas;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER chk_after_div_cancer;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER chk_after_div_tcell;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER chk_after_div_treg;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER chk_after_div_vas;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER mark_mac_sources;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER recruit_macrophages;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER zero_fib_density_field;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER update_ecm_grid;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER fib_execute_divide;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER aggregate_abm_events;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER copy_abm_counters_to_environment;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER reset_abm_event_counters;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER chk_start_step;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER chk_break;
// Define main model execution layers (state transitions and division)
void defineMainModelLayers(flamegpu::ModelDescription& model) {
    // Movement submodels loaded in first
    // After all movement submodels, do:
    // 1. Final broadcast and scan to get fresh neighbor data
    // 2. State transitions
    // 3. Division

    // { flamegpu::LayerDescription l = model.newLayer("chk_start_step"); l.addHostFunction(chk_start_step); }
    // 0. update agent counts
    {
        flamegpu::LayerDescription layer = model.newLayer("update_agent_counts");
        layer.addHostFunction(update_agent_counts);
    }
    // 0b. Recruitment system (following HCC Tumor::timeSlice order)
    // Reset ABM event counters (for this step's recruitment and death counts)
    {
        flamegpu::LayerDescription layer = model.newLayer("reset_abm_event_counters_start");
        layer.addHostFunction(reset_abm_event_counters);
    }
    // Reset recruitment sources
    {
        flamegpu::LayerDescription layer = model.newLayer("reset_recruitment_sources");
        layer.addHostFunction(reset_recruitment_sources);
    }
    // Mark vascular T cell sources (phalanx cells based on IFN-γ)
    {
       flamegpu::LayerDescription layer = model.newLayer("mark_vascular_t_sources");
       layer.addAgentFunction(AGENT_VASCULAR, "mark_t_sources");
    }
    
    // Mark MDSC sources (all voxels based on CCL2)
    {
        flamegpu::LayerDescription layer = model.newLayer("mark_mdsc_sources");
        layer.addHostFunction(mark_mdsc_sources);
    }
    // Recruit T cells at marked sources
    {
        flamegpu::LayerDescription layer = model.newLayer("recruit_t_cells");
        layer.addHostFunction(recruit_t_cells);
    }
    // Recruit MDSCs at marked sources
    {
        flamegpu::LayerDescription layer = model.newLayer("recruit_mdscs");
        layer.addHostFunction(recruit_mdscs);
    }
    // Mark macrophage sources (all voxels based on CCL2)
    {
        flamegpu::LayerDescription layer = model.newLayer("mark_mac_sources");
        layer.addHostFunction(mark_mac_sources);
    }
    // Recruit macrophages at marked sources
    {
        flamegpu::LayerDescription layer = model.newLayer("recruit_macrophages");
        layer.addHostFunction(recruit_macrophages);
    }
    // 1-4. Broadcast (separate layers required by FLAMEGPU2)
    // Messages accumulate across layers within the same step
    // TEMPORARILY DISABLED FOR DEBUGGING RANDOMMANAGER ERROR
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
    // 5. Scan neighbors
    {
        flamegpu::LayerDescription layer = model.newLayer("final_scan_neighbors");
        layer.addAgentFunction(AGENT_CANCER_CELL, "count_neighbors");
        layer.addAgentFunction(AGENT_TCELL, "scan_neighbors");
        layer.addAgentFunction(AGENT_TREG, "scan_neighbors");
        layer.addAgentFunction(AGENT_MDSC, "scan_neighbors");
        layer.addAgentFunction(AGENT_MACROPHAGE, "scan_neighbors");
    }

    // 6. READ chemicals from PDE to agents
    {
        flamegpu::LayerDescription layer = model.newLayer("read_chemicals_from_pde");
        layer.addHostFunction(update_agent_chemicals);
    }
    // 7. Agents update their chemical states (PDL1, activation, etc.)
    {
        flamegpu::LayerDescription layer = model.newLayer("update_chemical_states");
        layer.addAgentFunction(AGENT_CANCER_CELL, "update_chemicals");
        //layer.addAgentFunction(AGENT_TCELL, "update_chemicals"); // TCell states are updated in state step now
        //layer.addAgentFunction(AGENT_TREG, "update_chemicals"); // TReg states are updated in state step now
        layer.addAgentFunction(AGENT_MDSC, "update_chemicals");
        // layer.addAgentFunction(AGENT_MACROPHAGE, "update_chemicals"); // Macrophage states updated in state step
        layer.addAgentFunction(AGENT_VASCULAR, "update_chemicals");
    }
    // DEBUG: Before state transitions
    {
        flamegpu::LayerDescription layer = model.newLayer("debug_before_state_transitions_layer");
        layer.addHostFunction(debug_before_state_transitions);
    }
    // 8. Agent state transitions (killing, division decisions, etc.)
    {
        flamegpu::LayerDescription layer = model.newLayer("state_transitions");
        layer.addAgentFunction(AGENT_CANCER_CELL, "state_step");
        layer.addAgentFunction(AGENT_TCELL, "state_step");
        layer.addAgentFunction(AGENT_TREG, "state_step");
        layer.addAgentFunction(AGENT_MDSC, "state_step");
        layer.addAgentFunction(AGENT_MACROPHAGE, "state_step");
        layer.addAgentFunction(AGENT_FIBROBLAST, "state_step");
        layer.addAgentFunction(AGENT_VASCULAR, "state_step");
    }
    // DEBUG: After state transitions
    {
        flamegpu::LayerDescription layer = model.newLayer("debug_after_state_transitions_layer");
        layer.addHostFunction(debug_after_state_transitions);
    }
    // 9. Agents compute their chemical production/consumption rates
    // { flamegpu::LayerDescription l = model.newLayer("chk_break"); l.addHostFunction(chk_break); }
    {
        flamegpu::LayerDescription layer = model.newLayer("debug_before_compute_sources");
        layer.addHostFunction(debug_before_state_transitions); // Reuse same function, just for tracking
    }
    {
        flamegpu::LayerDescription layer = model.newLayer("compute_chemical_sources");
        layer.addAgentFunction(AGENT_CANCER_CELL, "compute_chemical_sources");
        layer.addAgentFunction(AGENT_TCELL, "compute_chemical_sources");
        layer.addAgentFunction(AGENT_TREG, "compute_chemical_sources");
        layer.addAgentFunction(AGENT_MDSC, "compute_chemical_sources");
        layer.addAgentFunction(AGENT_MACROPHAGE, "compute_chemical_sources");
        layer.addAgentFunction(AGENT_FIBROBLAST, "compute_chemical_sources");
        layer.addAgentFunction(AGENT_VASCULAR, "compute_chemical_sources");
    }

    // 10. WRITE agent sources to PDE
    {
        flamegpu::LayerDescription layer = model.newLayer("write_sources_to_pde");
        layer.addHostFunction(collect_agent_sources);
    }
    // 11. SOLVE PDE for one timestep
    {
        flamegpu::LayerDescription layer = model.newLayer("solve_pde");
        layer.addHostFunction(solve_pde_step);
    }
    // 11b. ECM deposition with Gaussian smoothing:
    // Step 1: Zero density field (reset from previous step)
    {
        flamegpu::LayerDescription layer = model.newLayer("zero_fib_density_field");
        layer.addHostFunction(zero_fib_density_field);
    }
    // Step 2: Fibroblasts scatter Gaussian kernels to density field
    {
        flamegpu::LayerDescription layer = model.newLayer("build_density_field");
        layer.addAgentFunction(AGENT_FIBROBLAST, "build_density_field");
    }
    // 11c. ECM grid update: apply decay, deposition from density field, and clamp to [baseline, saturation].
    {
        flamegpu::LayerDescription layer = model.newLayer("update_ecm");
        layer.addHostFunction(update_ecm_grid);
    }
    // 12. Occupancy grid: zero then populate with current live agent positions.
    // Must run after state_transitions (so dead agents are removed).
    {
        flamegpu::LayerDescription layer = model.newLayer("zero_occ_grid");
        layer.addHostFunction(zero_occupancy_grid);
    }
    { flamegpu::LayerDescription l = model.newLayer("chk_after_zero_occ"); l.addHostFunction(chk_after_zero_occ); }
    {
        // All agent types write their position atomically in parallel.
        flamegpu::LayerDescription layer = model.newLayer("write_to_occ_grid");
        layer.addAgentFunction(AGENT_CANCER_CELL, "write_to_occ_grid");
        layer.addAgentFunction(AGENT_TCELL,       "write_to_occ_grid");
        layer.addAgentFunction(AGENT_TREG,        "write_to_occ_grid");
        layer.addAgentFunction(AGENT_MDSC,        "write_to_occ_grid");
        layer.addAgentFunction(AGENT_MACROPHAGE,   "write_to_occ_grid");
        layer.addAgentFunction(AGENT_FIBROBLAST,   "write_to_occ_grid");
        layer.addAgentFunction(AGENT_VASCULAR,     "write_to_occ_grid");
    }
    { flamegpu::LayerDescription l = model.newLayer("chk_after_write_occ"); l.addHostFunction(chk_after_write_occ); }

    // 13. Single-phase movement via occ_grid (replaces per-cell-type submodels).
    // Each agent type gets N repeated move layers matching its XML move step count.
    // Agents CAS/atomicAdd to claim voxels; releases old voxel atomically on success.
    // Vascular tip cells use run-tumble (no occ_grid); stalk/phalanx don't move.

    // DEBUG: Before movement (this is where most random calls happen)
    {
        flamegpu::LayerDescription layer = model.newLayer("debug_before_movement_layer");
        layer.addHostFunction(debug_before_movement);
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

        for (int i = 0; i < cancer_steps; i++) {
            flamegpu::LayerDescription layer = model.newLayer("move_cancer_" + std::to_string(i));
            layer.addAgentFunction(AGENT_CANCER_CELL, "move");
        }
        { flamegpu::LayerDescription l = model.newLayer("chk_after_move_cancer"); l.addHostFunction(chk_after_move_cancer); }
        for (int i = 0; i < tcell_steps; i++) {
            flamegpu::LayerDescription layer = model.newLayer("move_tcell_" + std::to_string(i));
            layer.addAgentFunction(AGENT_TCELL, "move");
        }
        { flamegpu::LayerDescription l = model.newLayer("chk_after_move_tcell"); l.addHostFunction(chk_after_move_tcell); }
        for (int i = 0; i < treg_steps; i++) {
            flamegpu::LayerDescription layer = model.newLayer("move_treg_" + std::to_string(i));
            layer.addAgentFunction(AGENT_TREG, "move");
        }
        { flamegpu::LayerDescription l = model.newLayer("chk_after_move_treg"); l.addHostFunction(chk_after_move_treg); }
        for (int i = 0; i < mdsc_steps; i++) {
            flamegpu::LayerDescription layer = model.newLayer("move_mdsc_" + std::to_string(i));
            layer.addAgentFunction(AGENT_MDSC, "move");
        }
        { flamegpu::LayerDescription l = model.newLayer("chk_after_move_mdsc"); l.addHostFunction(chk_after_move_mdsc); }
        for (int i = 0; i < mac_steps; i++) {
            flamegpu::LayerDescription layer = model.newLayer("move_macrophage_" + std::to_string(i));
            layer.addAgentFunction(AGENT_MACROPHAGE, "move");
        }
        // Fibroblast chain movement: 6 layers per move cycle (supports chains up to 5 cells)
        // 1. Snapshot positions + reset fib_moved flags
        // 2. HEAD (sensor) runs TGFB chemotaxis
        // 3-6. Four propagation passes for follower movement through 5-cell chains
        // Layer i propagates through cells i steps behind HEAD
        for (int i = 0; i < fib_steps; i++) {
            {
                flamegpu::LayerDescription layer = model.newLayer("fib_write_pos_" + std::to_string(i));
                layer.addAgentFunction(AGENT_FIBROBLAST, "write_pos");
            }
            {
                flamegpu::LayerDescription layer = model.newLayer("fib_sensor_move_" + std::to_string(i));
                layer.addAgentFunction(AGENT_FIBROBLAST, "sensor_move");
            }
            // Four follow passes to propagate through 5-cell chains
            for (int pass = 0; pass < 4; pass++) {
                flamegpu::LayerDescription layer = model.newLayer("fib_follow_move_" + std::to_string(pass) + "_" + std::to_string(i));
                layer.addAgentFunction(AGENT_FIBROBLAST, "follow_move");
            }
        }
        {
            flamegpu::LayerDescription layer = model.newLayer("move_vascular");
            layer.addAgentFunction(AGENT_VASCULAR, "move");
        }
        { flamegpu::LayerDescription l = model.newLayer("chk_after_move_vas"); l.addHostFunction(chk_after_move_vas); }
    }

    // DEBUG: Before division
    {
        flamegpu::LayerDescription layer = model.newLayer("debug_before_division_layer");
        layer.addHostFunction(debug_before_division);
    }

    // 14. Single-phase division (atomicCAS replaces select → execute pair).
    // Each successful claim updates the grid immediately, so concurrent
    // threads in the same kernel see the updated occupancy.
    { flamegpu::LayerDescription l = model.newLayer("debug_div_cancer_pre"); l.addHostFunction(debug_before_divide_cancer); }
    {
        flamegpu::LayerDescription layer = model.newLayer("divide_cancer");
        layer.addAgentFunction(AGENT_CANCER_CELL, "divide");
    }
    { flamegpu::LayerDescription l = model.newLayer("chk_after_div_cancer"); l.addHostFunction(chk_after_div_cancer); }

    { flamegpu::LayerDescription l = model.newLayer("debug_div_tcell_pre"); l.addHostFunction(debug_before_divide_tcell); }
    {
        flamegpu::LayerDescription layer = model.newLayer("divide_tcell");
        layer.addAgentFunction(AGENT_TCELL, "divide");
    }
    { flamegpu::LayerDescription l = model.newLayer("chk_after_div_tcell"); l.addHostFunction(chk_after_div_tcell); }

    { flamegpu::LayerDescription l = model.newLayer("debug_div_treg_pre"); l.addHostFunction(debug_before_divide_treg); }
    {
        flamegpu::LayerDescription layer = model.newLayer("divide_treg");
        layer.addAgentFunction(AGENT_TREG, "divide");
    }
    { flamegpu::LayerDescription l = model.newLayer("chk_after_div_treg"); l.addHostFunction(chk_after_div_treg); }

    { flamegpu::LayerDescription l = model.newLayer("debug_div_vas_pre"); l.addHostFunction(debug_before_divide_vascular); }
    {
        flamegpu::LayerDescription layer = model.newLayer("divide_vascular");
        layer.addAgentFunction(AGENT_VASCULAR, "vascular_divide");
    }
    { flamegpu::LayerDescription l = model.newLayer("chk_after_div_vas"); l.addHostFunction(chk_after_div_vas); }

    // 14b. Fibroblast HEAD division execution (creates new cells, extends chain, converts to CAF)
    { flamegpu::LayerDescription l = model.newLayer("debug_fib_div_pre"); l.addHostFunction(debug_before_fib_divide); }
    {
        flamegpu::LayerDescription layer = model.newLayer("fib_execute_divide");
        layer.addHostFunction(fib_execute_divide);
    }
    { flamegpu::LayerDescription l = model.newLayer("debug_fib_div_post"); l.addHostFunction(debug_after_fib_divide); }

    // --- DEBUG LAYER AFTER FIB DIVIDE ---
    {
        flamegpu::LayerDescription l = model.newLayer("debug_after_fib_check");
        l.addHostFunction(debug_after_fib_check);
    }

    // 15a. Aggregate ABM events from agents (count deaths by cause)
    {
        flamegpu::LayerDescription l = model.newLayer("debug_before_aggregate");
        l.addHostFunction(debug_before_aggregate);
    }
    {
        flamegpu::LayerDescription layer = model.newLayer("aggregate_abm_events");
        layer.addHostFunction(aggregate_abm_events);
    }
    {
        flamegpu::LayerDescription l = model.newLayer("debug_after_aggregate");
        l.addHostFunction(debug_after_aggregate);
    }
    // 15b. Copy ABM event counters from MacroProperty to environment properties (for QSP to read)
    {
        flamegpu::LayerDescription layer = model.newLayer("copy_abm_counters_to_environment");
        layer.addHostFunction(copy_abm_counters_to_environment);
    }
    // 15c. Solve QSP model (reads ABM event counts from environment properties)
    {
        flamegpu::LayerDescription layer = model.newLayer("solve_qsp");
        layer.addHostFunction(solve_qsp_step);
    }
    // 15d. Reset ABM event counters for next step
    {
        flamegpu::LayerDescription layer = model.newLayer("reset_abm_event_counters");
        layer.addHostFunction(reset_abm_event_counters);
    }
}

} // namespace PDAC
