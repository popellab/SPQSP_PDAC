#include "flamegpu/flamegpu.h"
#include <memory>
#include <limits>

#include "../core/common.cuh"
#include "../agents/cancer_cell.cuh"
#include "../agents/t_cell.cuh"
#include "../agents/t_reg.cuh"
#include "../agents/mdsc.cuh"
#include "../agents/vascular_cell.cuh"

#include "../pde/pde_integration.cuh"
#include "gpu_param.h"
#include "../qsp/ode/ODE_system.h"
#include "../qsp/LymphCentral_wrapper.h"
#include "../qsp/ode/QSP_enum.h"

namespace PDAC {
// Declare HostFunction objects from pde_integration.cu
// These are defined using FLAMEGPU_HOST_FUNCTION macro which creates
// flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER global variables
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER update_agent_chemicals;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER collect_agent_sources;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER solve_pde_step;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER update_agent_counts;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER solve_qsp_step;

// Forward declarations
void defineCancerCellAgent(flamegpu::ModelDescription& model, bool include_state_divide);
void defineTCellAgent(flamegpu::ModelDescription& model, bool include_state_divide);
void defineTRegAgent(flamegpu::ModelDescription& model, bool include_state_divide);
void defineMDSCAgent(flamegpu::ModelDescription& model, bool include_state);
void defineVascularCellAgent(flamegpu::ModelDescription& model);

// Forward declaration (implemented in model_layers.cu)
void defineMainModelLayers(flamegpu::ModelDescription& model);

// Define the CancerCell agent and its variables
void defineCancerCellAgent(flamegpu::ModelDescription& model, bool include_state_divide) {
    flamegpu::AgentDescription cancer_cell = model.newAgent(AGENT_CANCER_CELL);

    // Identity: using FLAMEGPU's built-in ID system (FLAMEGPU->getID())
    // No manual "id" variable needed

    // Position (discrete voxel coordinates)
    cancer_cell.newVariable<int>("x");
    cancer_cell.newVariable<int>("y");
    cancer_cell.newVariable<int>("z");

    // State (CancerState enum)
    cancer_cell.newVariable<int>("cell_state", CANCER_STEM);

    // Division control
    cancer_cell.newVariable<int>("divideCD", 0);
    cancer_cell.newVariable<int>("divideFlag", 1);
    cancer_cell.newVariable<int>("divideCountRemaining", 0);
    cancer_cell.newVariable<unsigned int>("stemID", 0);

    // Movement control
    cancer_cell.newVariable<int>("moves_remaining", 0);

    cancer_cell.newVariable<float>("local_NO", 0.0f);
    cancer_cell.newVariable<float>("local_ArgI", 0.0f);
    cancer_cell.newVariable<float>("local_TGFB", 0.0f);
    cancer_cell.newVariable<float>("local_O2", 0.0f);
    cancer_cell.newVariable<float>("local_IFNg", 0.0f);

    // Molecular state (affects behavior)
    cancer_cell.newVariable<float>("PDL1_syn", 0.0f);
    cancer_cell.newVariable<int>("hypoxic", 0);                // Boolean: O2 below threshold

    // Drug effects (computed from local concentrations)
    cancer_cell.newVariable<float>("cabo_effect", 0.0f);       // VEGF inhibition level [0-1]

    // Neighbor counts (computed each step)
    cancer_cell.newVariable<int>("neighbor_Teff_count", 0);
    cancer_cell.newVariable<int>("neighbor_Treg_count", 0);
    cancer_cell.newVariable<int>("neighbor_cancer_count", 0);
    cancer_cell.newVariable<int>("neighbor_MDSC_count", 0);

    // Cached bitmask of available neighbor voxels (no cancer cell, in bounds)
    cancer_cell.newVariable<unsigned int>("available_neighbors", 0u);

    // Lifecycle
    cancer_cell.newVariable<int>("life", 0);
    cancer_cell.newVariable<int>("dead", 0);

    // Intent variables for two-phase conflict resolution
    cancer_cell.newVariable<int>("intent_action", INTENT_NONE);
    cancer_cell.newVariable<int>("target_x", -1);
    cancer_cell.newVariable<int>("target_y", -1);
    cancer_cell.newVariable<int>("target_z", -1);

    // Source/Sink rates
    cancer_cell.newVariable<float>("CCL2_release_rate", 0.0f);
    cancer_cell.newVariable<float>("TGFB_release_rate", 0.0f);
    cancer_cell.newVariable<float>("VEGFA_release_rate", 0.0f);
    cancer_cell.newVariable<float>("O2_uptake_rate", 0.0f);
    cancer_cell.newVariable<float>("IFNg_uptake_rate", 0.0f);

    // Define agent functions - movement functions always needed
    cancer_cell.newFunction("broadcast_location", cancer_broadcast_location)
        .setMessageOutput(MSG_CELL_LOCATION);

    cancer_cell.newFunction("count_neighbors", cancer_count_neighbors)
        .setMessageInput(MSG_CELL_LOCATION);

    cancer_cell.newFunction("check_voxel_packing", cancer_check_voxel_packing)
        .setMessageInput(MSG_CELL_LOCATION);

    cancer_cell.newFunction("update_chemicals", cancer_update_chemicals);

    cancer_cell.newFunction("compute_chemical_sources", cancer_compute_chemical_sources);

    cancer_cell.newFunction("reset_moves", cancer_reset_moves);

    cancer_cell.newFunction("select_move_target", cancer_select_move_target)
        .setMessageOutput(MSG_INTENT);

    cancer_cell.newFunction("execute_move", cancer_execute_move)
        .setMessageInput(MSG_INTENT);

    // Division and state functions only in main model
    if (include_state_divide) {
        cancer_cell.newFunction("state_step", cancer_cell_state_step)
            .setAllowAgentDeath(true);

        cancer_cell.newFunction("select_divide_target", cancer_select_divide_target)
            .setMessageOutput(MSG_INTENT);

        {
            flamegpu::AgentFunctionDescription fn = cancer_cell.newFunction("execute_divide", cancer_execute_divide);
            fn.setMessageInput(MSG_INTENT);
            fn.setAgentOutput(cancer_cell);
        }
    }


}

// Define the TCell agent and its variables
void defineTCellAgent(flamegpu::ModelDescription& model, bool include_state_divide) {
    flamegpu::AgentDescription tcell = model.newAgent(AGENT_TCELL);

    // Identity: using FLAMEGPU's built-in ID system

    // Position
    tcell.newVariable<int>("x");
    tcell.newVariable<int>("y");
    tcell.newVariable<int>("z");

    // State: T_CELL_EFF=0, T_CELL_CYT=1, T_CELL_SUPP=2
    tcell.newVariable<int>("cell_state", T_CELL_EFF);

    // Division control
    tcell.newVariable<int>("divide_flag", 0);
    tcell.newVariable<int>("divide_cd", 0);
    tcell.newVariable<int>("divide_limit", 0);

    // Molecular exposure
    tcell.newVariable<float>("local_IL2", 0.0f);

    // Molecular state (affects behavior)
    tcell.newVariable<float>("PDL1_syn", 0.0f);

    // Chemical production/release
    tcell.newVariable<float>("IFNg_release_rate", 0.0f);  // Current release rate (mol/s)
    tcell.newVariable<float>("IL2_release_rate", 0.0f);   // Current release rate (mol/s)
    tcell.newVariable<float>("IL2_release_remain", 0.0f); // Time remaining to release IL2 (s)
    tcell.newVariable<float>("IL2_uptake_rate", 0.0f);

    // Molecular exposure (cumulative for decisions)
    tcell.newVariable<float>("IL2_exposure", 0.0f);       // Cumulative IL2 exposure
    tcell.newVariable<float>("IL10_exposure", 0.0f);      // Cumulative suppression
    tcell.newVariable<float>("TGFB_exposure", 0.0f);      // Cumulative suppression

    // Drug effects
    tcell.newVariable<float>("PD1_occupancy", 0.0f);      // Fraction of PD1 blocked by Nivo [0-1]

    // Functional state (affected by chemicals)
    tcell.newVariable<float>("activation_level", 1.0f);   // Activity level [0-1]
    tcell.newVariable<int>("can_proliferate", 0);         // Boolean: IL2 above threshold

    // Neighbor counts (computed via messaging)
    tcell.newVariable<int>("neighbor_cancer_count", 0);
    tcell.newVariable<int>("neighbor_Treg_count", 0);
    tcell.newVariable<int>("neighbor_all_count", 0);
    tcell.newVariable<float>("max_neighbor_PDL1", 0.0f);
    tcell.newVariable<int>("found_progenitor", 0);

    // Cached bitmask of available neighbor voxels
    tcell.newVariable<unsigned int>("available_neighbors", 0u);

    // Lifecycle
    tcell.newVariable<int>("life", 0);
    tcell.newVariable<int>("dead", 0);

    // Intent variables for two-phase conflict resolution
    tcell.newVariable<int>("intent_action", INTENT_NONE);
    tcell.newVariable<int>("target_x", -1);
    tcell.newVariable<int>("target_y", -1);
    tcell.newVariable<int>("target_z", -1);

    // Define agent functions - movement functions always needed
    tcell.newFunction("broadcast_location", tcell_broadcast_location)
        .setMessageOutput(MSG_CELL_LOCATION);

    tcell.newFunction("scan_neighbors", tcell_scan_neighbors)
        .setMessageInput(MSG_CELL_LOCATION);

    tcell.newFunction("update_chemicals", tcell_update_chemicals);

    tcell.newFunction("compute_chemical_sources", tcell_compute_chemical_sources);

    tcell.newFunction("select_move_target", tcell_select_move_target)
        .setMessageOutput(MSG_INTENT);

    tcell.newFunction("execute_move", tcell_execute_move)
        .setMessageInput(MSG_INTENT);

    // Division and state functions only in main model
    if (include_state_divide) {
        tcell.newFunction("state_step", tcell_state_step)
            .setAllowAgentDeath(true);

        tcell.newFunction("select_divide_target", tcell_select_divide_target)
            .setMessageOutput(MSG_INTENT);

        {
            flamegpu::AgentFunctionDescription fn = tcell.newFunction("execute_divide", tcell_execute_divide);
            fn.setMessageInput(MSG_INTENT);
            fn.setAgentOutput(tcell);
        }
    }
}

// Define the TReg agent and its variables
void defineTRegAgent(flamegpu::ModelDescription& model, bool include_state_divide) {
    flamegpu::AgentDescription treg = model.newAgent(AGENT_TREG);

    // Identity: using FLAMEGPU's built-in ID system

    // Position
    treg.newVariable<int>("x");
    treg.newVariable<int>("y");
    treg.newVariable<int>("z");

    // State: TCD4_TREG=0, TCD4_TH=1
    treg.newVariable<int>("cell_state", TCD4_TREG);

    // Division control
    treg.newVariable<int>("divide_flag", 0);
    treg.newVariable<int>("divide_cd", 0);
    treg.newVariable<int>("divide_limit", 0);

     // Local chemical concentrations
    treg.newVariable<float>("local_TGFB", 0.0f);
    treg.newVariable<float>("local_IFNg", 0.0f); // might need to be gradient instead
    treg.newVariable<float>("local_ArgI", 0.0f);

    // Chemical production (Tregs are major source of IL10 and TGF-beta)
    treg.newVariable<float>("IL10_release_rate", 0.0f);
    treg.newVariable<float>("TGFB_release_rate", 0.0f);
    treg.newVariable<float>("IL2_release_rate",0.0f);

    treg.newVariable<float>("TGFB_release_remain", 0.0f);

    // Molecular state (affects behavior)
    treg.newVariable<float>("PDL1_syn", 0.0f);
    treg.newVariable<float>("CTLA4", 0.0f);

    // Molecular exposure
    treg.newVariable<float>("IL2_exposure", 0.0f);

    // Neighbor counts (computed via messaging)
    treg.newVariable<int>("neighbor_Tcell_count", 0);
    treg.newVariable<int>("neighbor_Treg_count", 0);
    treg.newVariable<int>("neighbor_cancer_count", 0);
    treg.newVariable<int>("neighbor_all_count", 0);
    treg.newVariable<int>("found_progenitor", 0);

    // Cached bitmask of available neighbor voxels
    treg.newVariable<unsigned int>("available_neighbors", 0u);

    // Lifecycle
    treg.newVariable<int>("life", 0);
    treg.newVariable<int>("dead", 0);

    // Intent variables for two-phase conflict resolution
    treg.newVariable<int>("intent_action", INTENT_NONE);
    treg.newVariable<int>("target_x", -1);
    treg.newVariable<int>("target_y", -1);
    treg.newVariable<int>("target_z", -1);

    // Define agent functions - movement functions always needed
    treg.newFunction("broadcast_location", treg_broadcast_location)
        .setMessageOutput(MSG_CELL_LOCATION);

    treg.newFunction("scan_neighbors", treg_scan_neighbors)
        .setMessageInput(MSG_CELL_LOCATION);

    treg.newFunction("update_chemicals", treg_update_chemicals);

    treg.newFunction("compute_chemical_sources", treg_compute_chemical_sources);

    treg.newFunction("select_move_target", treg_select_move_target)
        .setMessageOutput(MSG_INTENT);

    treg.newFunction("execute_move", treg_execute_move)
        .setMessageInput(MSG_INTENT);

    // Division and state functions only in main model
    if (include_state_divide) {
        treg.newFunction("state_step", treg_state_step)
            .setAllowAgentDeath(true);

        treg.newFunction("select_divide_target", treg_select_divide_target)
            .setMessageOutput(MSG_INTENT);

        {
            flamegpu::AgentFunctionDescription fn = treg.newFunction("execute_divide", treg_execute_divide);
            fn.setMessageInput(MSG_INTENT);
            fn.setAgentOutput(treg);
        }
    }
}

// Define the MDSC agent and its variables
// MDSCs are simpler than other cells: movement and life countdown only, no division
void defineMDSCAgent(flamegpu::ModelDescription& model, bool include_state) {
    flamegpu::AgentDescription mdsc = model.newAgent(AGENT_MDSC);

    // Identity: using FLAMEGPU's built-in ID system

    // Position
    mdsc.newVariable<int>("x");
    mdsc.newVariable<int>("y");
    mdsc.newVariable<int>("z");

    mdsc.newVariable<float>("local_IFNg", 0.0f);

    // Chemical production (MDSCs produce immunosuppressive factors)
    mdsc.newVariable<float>("NO_release_rate", 0.0f);    // Nitric oxide
    mdsc.newVariable<float>("ArgI_release_rate", 0.0f);  // Arginase I (immune suppression)
    mdsc.newVariable<float>("CCL2_uptake_rate", 0.0f);  // CCL2

    // Molecular state (affects behavior)
    mdsc.newVariable<float>("PDL1_syn", 0.0f);

    // Chemotaxis state (for directed migration)
    mdsc.newVariable<float>("CCL2_gradient_x", 0.0f);
    mdsc.newVariable<float>("CCL2_gradient_y", 0.0f);
    mdsc.newVariable<float>("CCL2_gradient_z", 0.0f);

    // Neighbor counts (computed via messaging)
    mdsc.newVariable<int>("neighbor_cancer_count", 0);
    mdsc.newVariable<int>("neighbor_Tcell_count", 0);
    mdsc.newVariable<int>("neighbor_Treg_count", 0);
    mdsc.newVariable<int>("neighbor_MDSC_count", 0);

    // Cached bitmask of available neighbor voxels (no MDSC)
    mdsc.newVariable<unsigned int>("available_neighbors", 0u);

    // Lifecycle
    mdsc.newVariable<int>("life", 0);
    mdsc.newVariable<int>("dead", 0);

    // Intent variables for two-phase conflict resolution
    mdsc.newVariable<int>("intent_action", INTENT_NONE);
    mdsc.newVariable<int>("target_x", -1);
    mdsc.newVariable<int>("target_y", -1);
    mdsc.newVariable<int>("target_z", -1);

    // Define agent functions - movement functions always needed
    mdsc.newFunction("broadcast_location", mdsc_broadcast_location)
        .setMessageOutput(MSG_CELL_LOCATION);

    mdsc.newFunction("scan_neighbors", mdsc_scan_neighbors)
        .setMessageInput(MSG_CELL_LOCATION);

    mdsc.newFunction("update_chemicals", mdsc_update_chemicals);

    mdsc.newFunction("compute_chemical_sources", mdsc_compute_chemical_sources);

    mdsc.newFunction("select_move_target", mdsc_select_move_target)
        .setMessageOutput(MSG_INTENT);

    mdsc.newFunction("execute_move", mdsc_execute_move)
        .setMessageInput(MSG_INTENT);

    // State step only in main model (MDSCs don't divide)
    if (include_state) {
        mdsc.newFunction("state_step", mdsc_state_step)
            .setAllowAgentDeath(true);
    }
}

// Define the VascularCell agent (Phase 1: Basic O2 secretion and VEGF-A uptake)
void defineVascularCellAgent(flamegpu::ModelDescription& model) {
    flamegpu::AgentDescription agent = model.newAgent(AGENT_VASCULAR);

    // Position (voxel coordinates)
    agent.newVariable<int>("x");
    agent.newVariable<int>("y");
    agent.newVariable<int>("z");

    // State (VAS_TIP, VAS_STALK, VAS_PHALANX)
    agent.newVariable<int>("vascular_state", VAS_PHALANX);  // Default: mature vessel

    // Chemical concentrations (read from PDE)
    agent.newVariable<float>("local_O2", 0.0f);
    agent.newVariable<float>("local_VEGFA", 0.0f);

    // VEGF-A gradient (read from PDE)
    agent.newVariable<float>("vegfa_grad_x", 0.0f);
    agent.newVariable<float>("vegfa_grad_y", 0.0f);
    agent.newVariable<float>("vegfa_grad_z", 0.0f);

    // Chemical source/sink rates (computed by agent)
    agent.newVariable<float>("O2_source", 0.0f);
    agent.newVariable<float>("VEGFA_sink", 0.0f);

    // Movement direction (for tip cells)
    agent.newVariable<float>("move_direction_x", 1.0f);
    agent.newVariable<float>("move_direction_y", 0.0f);
    agent.newVariable<float>("move_direction_z", 0.0f);

    // Run-tumble state
    agent.newVariable<int>("tumble", 1);  // Start in tumble phase

    // Tip ID (for tracking vessel lineages)
    agent.newVariable<unsigned int>("tip_id", 0);

    // Intent variables for two-phase conflict resolution
    agent.newVariable<int>("intent_action", INTENT_NONE);
    agent.newVariable<int>("target_x", -1);
    agent.newVariable<int>("target_y", -1);
    agent.newVariable<int>("target_z", -1);

    // State transition variables
    agent.newVariable<int>("mature_to_phalanx", 0);  // Anastomosis flag
    agent.newVariable<int>("branch", 0);  // Branch flag for phalanx cells

    // Register agent functions
    agent.newFunction("broadcast_location", vascular_broadcast_location)
        .setMessageOutput(MSG_CELL_LOCATION);

    agent.newFunction("update_chemicals", vascular_update_chemicals);

    agent.newFunction("compute_chemical_sources", vascular_compute_chemical_sources);

    // Recruitment source marking
    agent.newFunction("mark_t_sources", vascular_mark_t_sources);

    // State transitions and division
    agent.newFunction("state_step", vascular_state_step)
        .setMessageInput(MSG_CELL_LOCATION);

    agent.newFunction("select_divide_target", vascular_select_divide_target)
        .setMessageOutput(MSG_INTENT);

    {
        flamegpu::AgentFunctionDescription fn = agent.newFunction("execute_divide", vascular_execute_divide);
        fn.setMessageInput(MSG_INTENT);
        fn.setAllowAgentDeath(false);
        fn.setAgentOutput(agent);
    }

    // Movement functions (tip cells only)
    agent.newFunction("select_move_target", vascular_select_move_target)
        .setMessageOutput(MSG_INTENT);

    agent.newFunction("execute_move", vascular_execute_move)
        .setMessageInput(MSG_INTENT);
}

// Define the spatial message type for cell location broadcasting
void defineCellLocationMessage(flamegpu::ModelDescription& model, float voxel_size, int grid_max) {
    flamegpu::MessageSpatial3D::Description message = model.newMessage<flamegpu::MessageSpatial3D>(MSG_CELL_LOCATION);

    const float env_min = -voxel_size;
    const float env_max = (grid_max + 1) * voxel_size;

    message.setMin(env_min, env_min, env_min);
    message.setMax(env_max, env_max, env_max);
    message.setRadius(1.98f * voxel_size);

    // Message variables (shared by all agent types)
    message.newVariable<int>("agent_type");
    message.newVariable<int>("agent_id");
    message.newVariable<int>("cell_state");
    message.newVariable<float>("PDL1");
    message.newVariable<int>("voxel_x");
    message.newVariable<int>("voxel_y");
    message.newVariable<int>("voxel_z");
    message.newVariable<unsigned int>("tip_id");  // For vascular cells
}

// Define the spatial message type for intent broadcasting (two-phase conflict resolution)
void defineIntentMessage(flamegpu::ModelDescription& model, float voxel_size, int grid_max) {
    flamegpu::MessageSpatial3D::Description message = model.newMessage<flamegpu::MessageSpatial3D>(MSG_INTENT);

    const float env_min = -voxel_size;
    const float env_max = (grid_max + 1) * voxel_size;

    message.setMin(env_min, env_min, env_min);
    message.setMax(env_max, env_max, env_max);
    message.setRadius(1.98f * voxel_size);

    // Intent message variables
    message.newVariable<int>("agent_type");
    message.newVariable<unsigned int>("agent_id");
    message.newVariable<int>("intent_action");  // INTENT_NONE, INTENT_MOVE, INTENT_DIVIDE
    message.newVariable<int>("target_x");
    message.newVariable<int>("target_y");
    message.newVariable<int>("target_z");
    // Source position for conflict resolution when IDs are equal (e.g., new cells with id=0)
    message.newVariable<int>("source_x");
    message.newVariable<int>("source_y");
    message.newVariable<int>("source_z");
}

// Define environment properties (simulation parameters)
void defineEnvironment(flamegpu::ModelDescription& model,
                       int grid_x, int grid_y, int grid_z,
                       float voxel_size,
                       const PDAC::GPUParam& params) {

    flamegpu::EnvironmentDescription env = model.Environment();

    // Grid dimensions (from config, not XML)
    env.newProperty<int>(ENV_GRID_SIZE_X, grid_x);
    env.newProperty<int>(ENV_GRID_SIZE_Y, grid_y);
    env.newProperty<int>(ENV_GRID_SIZE_Z, grid_z);
    env.newProperty<float>(ENV_VOXEL_SIZE, voxel_size);

    // Simulation tracking
    env.newProperty<unsigned int>("current_step", 0u);
    env.newProperty<unsigned int>("next_agent_id", 1000u);

    // Step counter that starts at 0 when the main simulation (Phase 4) begins
    env.newProperty<unsigned int>("main_sim_step", 0u);

    // Agent count tracking (updated each timestep by host function)
    env.newProperty<unsigned int>("total_cancer_cells", 0u);
    env.newProperty<unsigned int>("total_tcells", 0u);
    env.newProperty<unsigned int>("total_tregs", 0u);
    env.newProperty<unsigned int>("total_mdscs", 0u);
    env.newProperty<unsigned int>("total_agents", 0u);  // Sum of all agents

    // Division tracking (for diagnostics)
    env.newProperty<unsigned int>("cancer_divide_attempts", 0u);
    env.newProperty<unsigned int>("cancer_divide_successes", 0u);
    env.newProperty<unsigned int>("tcell_divide_attempts", 0u);
    env.newProperty<unsigned int>("tcell_divide_successes", 0u);
    env.newProperty<unsigned int>("treg_divide_attempts", 0u);
    env.newProperty<unsigned int>("treg_divide_successes", 0u);

    // QSP state (updated by solve_qsp_step host function)
    env.newProperty<float>("qsp_teff_central", 0.0f);
    env.newProperty<float>("qsp_treg_central", 0.0f);
    env.newProperty<float>("qsp_th_central", 0.0f);

    env.newProperty<float>("qsp_teff_tumor", 0.0f);
    env.newProperty<float>("qsp_treg_tumor", 0.0f);
    env.newProperty<float>("qsp_th_tumor", 0.0f);
    env.newProperty<float>("qsp_mdsc_tumor", 0.0f);
    env.newProperty<float>("qsp_m1_tumor", 0.0f);
    env.newProperty<float>("qsp_m2_tumor", 0.0f);
    env.newProperty<float>("qsp_caf_tumor", 0.0f);

    env.newProperty<float>("qsp_nivo_tumor", 0.0f);   // Nivolumab concentration in tumor
    env.newProperty<float>("qsp_cabo_tumor", 0.0f);   // Cabozantinib concentration in tumor
    env.newProperty<float>("qsp_ipi_tumor", 0.0f);   // Ipilumab concentration in tumor

    env.newProperty<float>("qsp_cc_tumor", 0.0f); 
    env.newProperty<float>("qsp_cx_tumor", 0.0f);   
    env.newProperty<float>("qsp_t_exh_tumor", 0.0f);  
    env.newProperty<float>("qsp_tum_vol", 0.0f);  
    env.newProperty<float>("qsp_tum_cmax", 0.0f);   
    env.newProperty<float>("qsp_f_tum_cap", 0.0f);  

    // ABM state to QSP
    env.newProperty<int>("ABM_TEFF_REC", 0);  
    env.newProperty<int>("ABM_TREG_REC", 0);  
    env.newProperty<int>("ABM_TH_REC", 0);  
    env.newProperty<int>("ABM_MDSC_REC", 0);
    
    env.newProperty<int>("ABM_cc_death", 0);  
    env.newProperty<int>("ABM_cc_t_kill", 0);  

    // Populate ALL other parameters from XML
    params.populateFlameGPUEnvironment(env);
}

// Define environment properties for a movement submodel
void defineSubmodelEnvironment(flamegpu::ModelDescription& model,
                                int grid_x, int grid_y, int grid_z,
                                float voxel_size) {
    flamegpu::EnvironmentDescription env = model.Environment();

    // Grid dimensions (needed for bounds checking)
    env.newProperty<int>(ENV_GRID_SIZE_X, grid_x);
    env.newProperty<int>(ENV_GRID_SIZE_Y, grid_y);
    env.newProperty<int>(ENV_GRID_SIZE_Z, grid_z);
    env.newProperty<float>(ENV_VOXEL_SIZE, voxel_size);
}

void buildMovementSubmodel(
    flamegpu::ModelDescription& submodel,
    const std::string& moving_agent_name,
    const std::vector<std::string>& all_agent_names,
    int grid_x, int grid_y, int grid_z, float voxel_size)
{
    int grid_max = std::max({grid_x, grid_y, grid_z});

    // Define environment and messages
    defineSubmodelEnvironment(submodel, grid_x, grid_y, grid_z, voxel_size);
    defineCellLocationMessage(submodel, voxel_size, grid_max);
    defineIntentMessage(submodel, voxel_size, grid_max);

    // Define ALL agents (needed for broadcasting locations)
    // This uses your existing define functions with include_state_divide=false
    for (const auto& agent_name : all_agent_names) {
        if (agent_name == AGENT_CANCER_CELL) {
            defineCancerCellAgent(submodel, false);
        } else if (agent_name == AGENT_TCELL) {
            defineTCellAgent(submodel, false);
        } else if (agent_name == AGENT_TREG) {
            defineTRegAgent(submodel, false);
        } else if (agent_name == AGENT_MDSC) {
            defineMDSCAgent(submodel, false);
        }
        // Add new cell types here as else-if blocks
    }

    // Layer 1-4: ALL agents broadcast their locations (separate layers required by FLAMEGPU2)
    // Messages should accumulate across layers within the same step
    for (const auto& agent_name : all_agent_names) {
        flamegpu::LayerDescription layer = submodel.newLayer("broadcast_" + agent_name);
        layer.addAgentFunction(agent_name, "broadcast_location");
    }

    // Layer 2: Only the moving agent scans neighbors
    {
        flamegpu::LayerDescription layer = submodel.newLayer("scan");
        // Cancer cells use "count_neighbors", others use "scan_neighbors"
        if (moving_agent_name == AGENT_CANCER_CELL) {
            layer.addAgentFunction(moving_agent_name, "count_neighbors");
        } else {
            layer.addAgentFunction(moving_agent_name, "scan_neighbors");
        }
    }

    // Layer 3: Only the moving agent selects move target
    {
        flamegpu::LayerDescription layer = submodel.newLayer("select_move");
        layer.addAgentFunction(moving_agent_name, "select_move_target");
    }

    // Layer 4: Only the moving agent executes move
    {
        flamegpu::LayerDescription layer = submodel.newLayer("execute_move");
        layer.addAgentFunction(moving_agent_name, "execute_move");
    }
}

// Helper to create and configure a movement submodel in the main model
flamegpu::SubModelDescription createMovementSubmodel(
    flamegpu::ModelDescription& model,
    const std::string& submodel_name,
    const std::string& moving_agent_name,
    const std::vector<std::string>& all_agent_names,
    int max_steps,
    int grid_x, int grid_y, int grid_z, float voxel_size)
{
    // Create the submodel description
    flamegpu::ModelDescription submodelDesc(submodel_name);
    buildMovementSubmodel(submodelDesc, moving_agent_name, all_agent_names,
                          grid_x, grid_y, grid_z, voxel_size);

    // Add submodel to main model
    auto submodel = model.newSubModel(submodel_name, submodelDesc);

    // Bind ALL agents (so positions stay synchronized)
    for (const auto& agent_name : all_agent_names) {
        submodel.bindAgent(agent_name, agent_name, true, true);
    }

    // Set max steps for this agent type
    submodel.setMaxSteps(max_steps);

    return submodel;
}

// Build the complete model with per-cell-type movement submodels
std::unique_ptr<flamegpu::ModelDescription> buildModel(
    int grid_x, int grid_y, int grid_z, float voxel_size,
    const PDAC::GPUParam& gpu_params) {

    auto model = std::make_unique<flamegpu::ModelDescription>("PDAC_ABM_GPU");

    int grid_max = std::max({grid_x, grid_y, grid_z});

    // Define messages for main model
    defineCellLocationMessage(*model, voxel_size, grid_max);
    defineIntentMessage(*model, voxel_size, grid_max);

    // ========== AGENT CONFIGURATION ==========
    // To add a new agent type:
    // 1. Add to this list
    // 2. Add its define function call below
    // 3. Add its parameter name for move steps
    struct AgentConfig {
        std::string name;
        std::string move_steps_param;
    };

    const std::vector<AgentConfig> agent_configs = {
        {AGENT_CANCER_CELL, "PARAM_CANCER_MOVE_STEPS_STEM"},
        {AGENT_TCELL,       "PARAM_TCELL_MOVE_STEPS"},
        {AGENT_TREG,        "PARAM_TCELL_MOVE_STEPS"},  // or add PARAM_TREG_MOVE_STEPS
        {AGENT_MDSC,        "PARAM_MDSC_MOVE_STEPS"}
        // {AGENT_MACROPHAGE, "PARAM_MACROPHAGE_MOVE_STEPS"},
    };

    // Extract just agent names for convenience
    std::vector<std::string> all_agent_names;
    for (const auto& cfg : agent_configs) {
        all_agent_names.push_back(cfg.name);
    }

    // Define agents with all functions
    defineCancerCellAgent(*model, true);
    defineTCellAgent(*model, true);
    defineTRegAgent(*model, true);
    defineMDSCAgent(*model, true);
    defineVascularCellAgent(*model);

    // Define environment with GPU parameters loaded from XML
    defineEnvironment(*model, grid_x, grid_y, grid_z, voxel_size, gpu_params);

    // First reset cancer move steps
    {
        auto layer = model->newLayer("reset_cancer_moves");
        layer.addAgentFunction(AGENT_CANCER_CELL, "reset_moves");
    }                                                                                                                                                                                                            

    // ========== BUILD AND ADD MOVEMENT SUBMODELS ==========
    for (const auto& cfg : agent_configs) {
        int max_steps = model->Environment().getProperty<int>(cfg.move_steps_param);

        std::string submodel_name = cfg.name + "_movement";
        flamegpu::ModelDescription submodelDesc(submodel_name + "_desc");

        buildMovementSubmodel(submodelDesc, cfg.name, all_agent_names,
                              grid_x, grid_y, grid_z, voxel_size);

        auto submodel = model->newSubModel(submodel_name, submodelDesc);

        // Bind all agents
        for (const auto& agent_name : all_agent_names) {
            submodel.bindAgent(agent_name, agent_name, true, true);
        }
        submodel.setMaxSteps(max_steps);

        // Add layer for this submodel
        auto layer = model->newLayer(submodel_name + "_layer");
        layer.addSubModel(submodel);
    }

    // Define main model layers (state transitions, division)
    defineMainModelLayers(*model);

    return model;
}

} // namespace PDAC
