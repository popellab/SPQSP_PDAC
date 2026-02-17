#include "flamegpu/flamegpu.h"

#include "../core/common.cuh"
#include "../pde/pde_integration.cuh"

namespace PDAC {

// Declare HostFunction objects from pde_integration.cu
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER update_agent_chemicals;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER collect_agent_sources;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER solve_pde_step;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER update_agent_counts;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER solve_qsp_step;

// Define main model execution layers (state transitions and division)
void defineMainModelLayers(flamegpu::ModelDescription& model) {
    // Movement submodels loaded in first
    // After all movement submodels, do:
    // 1. Final broadcast and scan to get fresh neighbor data
    // 2. State transitions
    // 3. Division

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

    // 5. Scan neighbors
    {
        flamegpu::LayerDescription layer = model.newLayer("final_scan_neighbors");
        layer.addAgentFunction(AGENT_CANCER_CELL, "count_neighbors");
        layer.addAgentFunction(AGENT_TCELL, "scan_neighbors");
        layer.addAgentFunction(AGENT_TREG, "scan_neighbors");
        layer.addAgentFunction(AGENT_MDSC, "scan_neighbors");
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
        layer.addAgentFunction(AGENT_VASCULAR, "update_chemicals");
    }

    // 8. Agent state transitions (killing, division decisions, etc.)
    {
        flamegpu::LayerDescription layer = model.newLayer("state_transitions");
        layer.addAgentFunction(AGENT_CANCER_CELL, "state_step");
        layer.addAgentFunction(AGENT_TCELL, "state_step");
        layer.addAgentFunction(AGENT_TREG, "state_step");
        layer.addAgentFunction(AGENT_MDSC, "state_step");
        layer.addAgentFunction(AGENT_VASCULAR, "state_step");
    }

    // 9. Agents compute their chemical production/consumption rates
    {
        flamegpu::LayerDescription layer = model.newLayer("compute_chemical_sources");
        layer.addAgentFunction(AGENT_CANCER_CELL, "compute_chemical_sources");
        layer.addAgentFunction(AGENT_TCELL, "compute_chemical_sources");
        layer.addAgentFunction(AGENT_TREG, "compute_chemical_sources");
        layer.addAgentFunction(AGENT_MDSC, "compute_chemical_sources");
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

    // 11b. Vascular cell movement (tip cells only, run-tumble)
    {
        flamegpu::LayerDescription layer = model.newLayer("vascular_select_move");
        layer.addAgentFunction(AGENT_VASCULAR, "select_move_target");
    }
    {
        flamegpu::LayerDescription layer = model.newLayer("vascular_execute_move");
        layer.addAgentFunction(AGENT_VASCULAR, "execute_move");
    }

    // 12 - 14. Division layers for each cell type
    // Division: FLAMEGPU2 clears message lists when a new layer outputs to them.
    // Each cell type's execute must immediately follow its select to read the correct messages.
    // Note: FLAMEGPU2 doesn't allow multiple functions outputting to the same message in one layer.

    // Cancer cell division (select → execute)
    {
        flamegpu::LayerDescription layer = model.newLayer("select_divide_cancer");
        layer.addAgentFunction(AGENT_CANCER_CELL, "select_divide_target");
    }
    {
        flamegpu::LayerDescription layer = model.newLayer("execute_divide_cancer");
        layer.addAgentFunction(AGENT_CANCER_CELL, "execute_divide");
    }

    // Check for packing AFTER division
    {
        flamegpu::LayerDescription layer = model.newLayer("postdivision_broadcast_cancer");
        layer.addAgentFunction(AGENT_CANCER_CELL, "broadcast_location");
    }
    {
        flamegpu::LayerDescription layer = model.newLayer("postdivision_check_packing");
        layer.addAgentFunction(AGENT_CANCER_CELL, "check_voxel_packing");
    }

    // population won't expand until we include recruitment
    // T cell division (select → execute)
    {
        flamegpu::LayerDescription layer = model.newLayer("select_divide_tcell");
        layer.addAgentFunction(AGENT_TCELL, "select_divide_target");
    }
    {
        flamegpu::LayerDescription layer = model.newLayer("execute_divide_tcell");
        layer.addAgentFunction(AGENT_TCELL, "execute_divide");
    }

    // TReg division (select → execute)
    {
        flamegpu::LayerDescription layer = model.newLayer("select_divide_treg");
        layer.addAgentFunction(AGENT_TREG, "select_divide_target");
    }
    {
        flamegpu::LayerDescription layer = model.newLayer("execute_divide_treg");
        layer.addAgentFunction(AGENT_TREG, "execute_divide");
    }

    // Vascular cell division (select → execute)
    {
        flamegpu::LayerDescription layer = model.newLayer("select_divide_vascular");
        layer.addAgentFunction(AGENT_VASCULAR, "select_divide_target");
    }
    {
        flamegpu::LayerDescription layer = model.newLayer("execute_divide_vascular");
        layer.addAgentFunction(AGENT_VASCULAR, "execute_divide");
    }

    {
        flamegpu::LayerDescription layer = model.newLayer("solve_qsp");
        layer.addHostFunction(solve_qsp_step);
    }
}

} // namespace PDAC
