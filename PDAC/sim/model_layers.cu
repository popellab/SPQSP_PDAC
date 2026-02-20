#include "flamegpu/flamegpu.h"
#include <string>

#include "../core/common.cuh"
#include "../pde/pde_integration.cuh"

namespace PDAC {

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
    // 0a. Recruitment system (following HCC Tumor::timeSlice order)
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
    // 9. Agents compute their chemical production/consumption rates
    // { flamegpu::LayerDescription l = model.newLayer("chk_break"); l.addHostFunction(chk_break); }
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
        for (int i = 0; i < fib_steps; i++) {
            flamegpu::LayerDescription layer = model.newLayer("move_fibroblast_" + std::to_string(i));
            layer.addAgentFunction(AGENT_FIBROBLAST, "move");
        }
        {
            flamegpu::LayerDescription layer = model.newLayer("move_vascular");
            layer.addAgentFunction(AGENT_VASCULAR, "move");
        }
        { flamegpu::LayerDescription l = model.newLayer("chk_after_move_vas"); l.addHostFunction(chk_after_move_vas); }
    }

    // 14. Single-phase division (atomicCAS replaces select → execute pair).
    // Each successful claim updates the grid immediately, so concurrent
    // threads in the same kernel see the updated occupancy.
    {
        flamegpu::LayerDescription layer = model.newLayer("divide_cancer");
        layer.addAgentFunction(AGENT_CANCER_CELL, "divide");
    }
    { flamegpu::LayerDescription l = model.newLayer("chk_after_div_cancer"); l.addHostFunction(chk_after_div_cancer); }
    {
        flamegpu::LayerDescription layer = model.newLayer("divide_tcell");
        layer.addAgentFunction(AGENT_TCELL, "divide");
    }
    { flamegpu::LayerDescription l = model.newLayer("chk_after_div_tcell"); l.addHostFunction(chk_after_div_tcell); }
    {
        flamegpu::LayerDescription layer = model.newLayer("divide_treg");
        layer.addAgentFunction(AGENT_TREG, "divide");
    }
    { flamegpu::LayerDescription l = model.newLayer("chk_after_div_treg"); l.addHostFunction(chk_after_div_treg); }
    {
        flamegpu::LayerDescription layer = model.newLayer("divide_vascular");
        layer.addAgentFunction(AGENT_VASCULAR, "vascular_divide");
    }
    { flamegpu::LayerDescription l = model.newLayer("chk_after_div_vas"); l.addHostFunction(chk_after_div_vas); }

    {
        flamegpu::LayerDescription layer = model.newLayer("solve_qsp");
        layer.addHostFunction(solve_qsp_step);
    }
}

} // namespace PDAC
