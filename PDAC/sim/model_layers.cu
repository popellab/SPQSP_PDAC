#include "flamegpu/flamegpu.h"
#include <string>
#include <iostream>

#include "../core/common.cuh"
#include "../pde/pde_integration.cuh"

namespace PDAC {

// Extern declarations for host functions not declared in pde_integration.cuh
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER solve_pde_step;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER update_agent_counts;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER solve_qsp_step;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER zero_fib_density_field;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER fib_execute_divide;
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

    // 0. Update agent counts for QSP
    {
        flamegpu::LayerDescription layer = model.newLayer("update_agent_counts");
        layer.addHostFunction(update_agent_counts);
    }
    // 0b. Reset ABM event counters for this step
    {
        flamegpu::LayerDescription layer = model.newLayer("reset_abm_event_counters_start");
        layer.addHostFunction(reset_abm_event_counters);
    }
    // 0c. Reset recruitment sources
    {
        flamegpu::LayerDescription layer = model.newLayer("reset_recruitment_sources");
        layer.addHostFunction(reset_recruitment_sources);
    }
    // 0d. Update vasculature count (used by mark_t_sources as denominator)
    {
        flamegpu::LayerDescription layer = model.newLayer("update_vasculature_count");
        layer.addHostFunction(update_vasculature_count);
    }
    // 0e. Mark vascular T cell sources (phalanx cells based on IFN-γ)
    {
       flamegpu::LayerDescription layer = model.newLayer("mark_vascular_t_sources");
       layer.addAgentFunction(AGENT_VASCULAR, "mark_t_sources");
    }
    // 0e. Mark MDSC/macrophage sources based on CCL2
    {
        flamegpu::LayerDescription layer = model.newLayer("mark_mdsc_sources");
        layer.addHostFunction(mark_mdsc_sources);
    }
    // 0f. Recruit immune cells at marked sources
    {
        flamegpu::LayerDescription layer = model.newLayer("recruit_t_cells");
        layer.addHostFunction(recruit_t_cells);
    }
    {
        flamegpu::LayerDescription layer = model.newLayer("recruit_mdscs");
        layer.addHostFunction(recruit_mdscs);
    }
    {
        flamegpu::LayerDescription layer = model.newLayer("mark_mac_sources");
        layer.addHostFunction(mark_mac_sources);
    }
    {
        flamegpu::LayerDescription layer = model.newLayer("recruit_macrophages");
        layer.addHostFunction(recruit_macrophages);
    }

    // ── Timing checkpoint: after Phase 0 (all recruitment) ──
    {
        flamegpu::LayerDescription layer = model.newLayer("timing_after_recruit");
        layer.addHostFunction(timing_after_recruit);
    }

    // 1-4. Broadcast locations (separate layers required by FLAMEGPU2)
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

    // ── Timing checkpoint: after Phase 1 (broadcast + neighbor scan) ──
    {
        flamegpu::LayerDescription layer = model.newLayer("timing_after_broadcast");
        layer.addHostFunction(timing_after_broadcast);
    }

    // 6. Reset PDE source/uptake buffers (agents will atomicAdd directly in compute_chemical_sources)
    {
        flamegpu::LayerDescription layer = model.newLayer("reset_pde_buffers");
        layer.addHostFunction(reset_pde_buffers);
    }

    // REMOVED - all chemical states tracked in state transitions layer now
    // 7. Agents update chemical states (PDL1, activation, etc.) — reads PDE directly via env ptr
    // {
    //     flamegpu::LayerDescription layer = model.newLayer("update_chemical_states");
    //     // layer.addAgentFunction(AGENT_CANCER_CELL, "update_chemicals");
    //     // layer.addAgentFunction(AGENT_MDSC, "update_chemicals");
    //     layer.addAgentFunction(AGENT_VASCULAR, "update_chemicals");
    // }

    // 8. Agent state transitions
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

    // 9. Agents compute chemical production/consumption rates (atomicAdd directly to PDE buffers)
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

    // ── Timing checkpoint: after state transitions + chemical sources ──
    {
        flamegpu::LayerDescription layer = model.newLayer("timing_after_sources");
        layer.addHostFunction(timing_after_sources);
    }

    // 10. Solve PDE for one timestep
    {
        flamegpu::LayerDescription layer = model.newLayer("solve_pde");
        layer.addHostFunction(solve_pde_step);
    }

    // ── Timing checkpoint: after PDE solve (wall time cross-check vs g_last_pde_ms) ──
    {
        flamegpu::LayerDescription layer = model.newLayer("timing_after_pde");
        layer.addHostFunction(timing_after_pde);
    }

    // 10a. Compute gradients for chemotaxis (IFN, TGFB, CCL2, VEGFA)
    {
        flamegpu::LayerDescription layer = model.newLayer("compute_pde_gradients");
        layer.addHostFunction(compute_pde_gradients);
    }

    // ── Timing checkpoint: after gradient computation ──
    {
        flamegpu::LayerDescription layer = model.newLayer("timing_after_gradients");
        layer.addHostFunction(timing_after_gradients);
    }

    // 10b. ECM deposition: zero density field, scatter Gaussian kernels, apply update
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

    // ── Timing checkpoint: after Phase 3 (ECM) ──
    {
        flamegpu::LayerDescription layer = model.newLayer("timing_after_ecm");
        layer.addHostFunction(timing_after_ecm);
    }

    // 11. Occupancy grid: zero then populate with current live agent positions
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
        layer.addAgentFunction(AGENT_MACROPHAGE,   "write_to_occ_grid");
        layer.addAgentFunction(AGENT_FIBROBLAST,   "write_to_occ_grid");
        layer.addAgentFunction(AGENT_VASCULAR,     "write_to_occ_grid");
    }

    // 12. Single-phase movement via occ_grid.
    // Each agent type gets N repeated move layers matching its XML move step count.
    // Agents CAS/atomicAdd to claim voxels; releases old voxel atomically on success.
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
        for (int i = 0; i < tcell_steps; i++) {
            flamegpu::LayerDescription layer = model.newLayer("move_tcell_" + std::to_string(i));
            layer.addAgentFunction(AGENT_TCELL, "move");
        }
        for (int i = 0; i < treg_steps; i++) {
            flamegpu::LayerDescription layer = model.newLayer("move_treg_" + std::to_string(i));
            layer.addAgentFunction(AGENT_TREG, "move");
        }
        for (int i = 0; i < mdsc_steps; i++) {
            flamegpu::LayerDescription layer = model.newLayer("move_mdsc_" + std::to_string(i));
            layer.addAgentFunction(AGENT_MDSC, "move");
        }
        for (int i = 0; i < mac_steps; i++) {
            flamegpu::LayerDescription layer = model.newLayer("move_macrophage_" + std::to_string(i));
            layer.addAgentFunction(AGENT_MACROPHAGE, "move");
        }
        // Fibroblast chain movement: 6 layers per move cycle (supports chains up to 5 cells)
        // 1. Snapshot positions + reset fib_moved flags
        // 2. HEAD (sensor) runs TGFB chemotaxis
        // 3-6. Four propagation passes for follower movement through 5-cell chains
        for (int i = 0; i < fib_steps; i++) {
            {
                flamegpu::LayerDescription layer = model.newLayer("fib_write_pos_" + std::to_string(i));
                layer.addAgentFunction(AGENT_FIBROBLAST, "write_pos");
            }
            {
                flamegpu::LayerDescription layer = model.newLayer("fib_sensor_move_" + std::to_string(i));
                layer.addAgentFunction(AGENT_FIBROBLAST, "sensor_move");
            }
            for (int pass = 0; pass < 4; pass++) {
                flamegpu::LayerDescription layer = model.newLayer("fib_follow_move_" + std::to_string(pass) + "_" + std::to_string(i));
                layer.addAgentFunction(AGENT_FIBROBLAST, "follow_move");
            }
        }
        {
            flamegpu::LayerDescription layer = model.newLayer("move_vascular");
            layer.addAgentFunction(AGENT_VASCULAR, "move");
        }
    }

    // ── Timing checkpoint: after Phase 4 (occ grid + all movement) ──
    {
        flamegpu::LayerDescription layer = model.newLayer("timing_after_movement");
        layer.addHostFunction(timing_after_movement);
    }

    // 13. Single-phase division (atomicCAS on occupancy grid)
    {
        flamegpu::LayerDescription layer = model.newLayer("divide_cancer");
        layer.addAgentFunction(AGENT_CANCER_CELL, "divide");
    }
    {
        flamegpu::LayerDescription layer = model.newLayer("divide_tcell");
        layer.addAgentFunction(AGENT_TCELL, "divide");
    }
    {
        flamegpu::LayerDescription layer = model.newLayer("divide_treg");
        layer.addAgentFunction(AGENT_TREG, "divide");
    }
    {
        flamegpu::LayerDescription layer = model.newLayer("divide_vascular");
        layer.addAgentFunction(AGENT_VASCULAR, "vascular_divide");
    }
    // Fibroblast HEAD division (host function: creates new agents, extends chain)
    {
        flamegpu::LayerDescription layer = model.newLayer("fib_execute_divide");
        layer.addHostFunction(fib_execute_divide);
    }

    // ── Timing checkpoint: after Phase 5 (all division) ──
    {
        flamegpu::LayerDescription layer = model.newLayer("timing_after_division");
        layer.addHostFunction(timing_after_division);
    }

    // 14. Aggregate ABM events and advance QSP
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
