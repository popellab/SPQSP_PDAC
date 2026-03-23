#include "flamegpu/flamegpu.h"
#include <memory>
#include <limits>

#include "../core/common.cuh"
#include "../agents/cancer_cell.cuh"
#include "../agents/t_cell.cuh"
#include "../agents/t_reg.cuh"
#include "../agents/mdsc.cuh"
#include "../agents/macrophage.cuh"
#include "../agents/fibroblast.cuh"
#include "../agents/vascular_cell.cuh"
#include "../agents/pack_for_export.cuh"

#include "../pde/pde_integration.cuh"
#include "gpu_param.h"
#include "../qsp/ode/ODE_system.h"
#include "../qsp/LymphCentral_wrapper.h"
#include "../qsp/ode/QSP_enum.h"

namespace PDAC {
// Declare HostFunction objects from pde_integration.cu
// These are defined using FLAMEGPU_HOST_FUNCTION macro which creates
// flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER global variables
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER solve_pde_step;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER update_agent_counts;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER solve_qsp_step;
extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER zero_occupancy_grid;

// Forward declarations
void defineCancerCellAgent(flamegpu::ModelDescription& model, bool include_state_divide);
void defineTCellAgent(flamegpu::ModelDescription& model, bool include_state_divide);
void defineTRegAgent(flamegpu::ModelDescription& model, bool include_state_divide);
void defineMDSCAgent(flamegpu::ModelDescription& model, bool include_state);
void defineMacrophageAgent(flamegpu::ModelDescription& model, bool include_state);
void defineFibroblastAgent(flamegpu::ModelDescription& model, bool include_state);
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

    // Movement control
    cancer_cell.newVariable<int>("moves_remaining", 0);

    // Division control
    cancer_cell.newVariable<int>("divideCD", 0);
    cancer_cell.newVariable<int>("divideFlag", 1);
    cancer_cell.newVariable<int>("divideCountRemaining", 0);
    cancer_cell.newVariable<unsigned int>("stemID", 0);
    cancer_cell.newVariable<int>("divide_wave", 0);   // Wave assignment for interleaved division

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
    cancer_cell.newVariable<int>("neighbor_Mac1_count", 0);

    // Cached bitmask of available neighbor voxels (no cancer cell, in bounds)
    cancer_cell.newVariable<unsigned int>("available_neighbors", 0u);

    // Lifecycle
    cancer_cell.newVariable<int>("life", 0);
    cancer_cell.newVariable<int>("dead", 0);
    cancer_cell.newVariable<int>("death_reason", -1);  // 0=senescence, 1=T cell kill, 2=MAC kill, -1=alive
    cancer_cell.newVariable<int>("newborn", 0);  // 1 on first active step (compensating kill check)

    // Intent variables for two-phase conflict resolution
    cancer_cell.newVariable<int>("intent_action", INTENT_NONE);
    cancer_cell.newVariable<int>("target_x", -1);
    cancer_cell.newVariable<int>("target_y", -1);
    cancer_cell.newVariable<int>("target_z", -1);

    // Define agent functions - movement functions always needed
    cancer_cell.newFunction("broadcast_location", cancer_broadcast_location)
        .setMessageOutput(MSG_CELL_LOCATION);

    cancer_cell.newFunction("count_neighbors", cancer_count_neighbors)
        .setMessageInput(MSG_CELL_LOCATION);

    cancer_cell.newFunction("reset_moves", cancer_reset_moves);

    cancer_cell.newFunction("update_chemicals", cancer_update_chemicals);

    cancer_cell.newFunction("compute_chemical_sources", cancer_compute_chemical_sources);
    cancer_cell.newFunction("pack_for_export", pack_export_cancer);

    // Movement, state, division, and occupancy grid functions only in main model
    // (these access occ_grid which is not defined in submodel environments)
    if (include_state_divide) {
        cancer_cell.newFunction("write_to_occ_grid", cancer_write_to_occ_grid);
        cancer_cell.newFunction("move", cancer_move);

        cancer_cell.newFunction("state_step", cancer_cell_state_step)
            .setAllowAgentDeath(true);

        {
            flamegpu::AgentFunctionDescription fn = cancer_cell.newFunction("divide", cancer_divide);
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
    tcell.newVariable<int>("divide_wave", 0);   // Wave assignment for interleaved division

    // Movement state for run-tumble chemotaxis
    tcell.newVariable<float>("move_direction_x", 0.0f);
    tcell.newVariable<float>("move_direction_y", 0.0f);
    tcell.newVariable<float>("move_direction_z", 0.0f);
    tcell.newVariable<int>("tumble", 1);  // 0=running, 1=tumbling (default: tumble to pick initial direction)

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
    tcell.newFunction("pack_for_export", pack_export_tcell);

    // Movement, state, division, and occupancy grid functions only in main model
    if (include_state_divide) {
        tcell.newFunction("write_to_occ_grid", tcell_write_to_occ_grid);
        tcell.newFunction("move", tcell_move);

        tcell.newFunction("state_step", tcell_state_step)
            .setAllowAgentDeath(true);

        {
            flamegpu::AgentFunctionDescription fn = tcell.newFunction("divide", tcell_divide);
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

    // State: TCD4_TH=0, TCD4_TREG=1 (matches HCC convention: Th < TREG)
    treg.newVariable<int>("cell_state", TCD4_TREG);

    // Division control
    treg.newVariable<int>("divide_flag", 0);
    treg.newVariable<int>("divide_cd", 0);
    treg.newVariable<int>("divide_limit", 0);
    treg.newVariable<int>("divide_wave", 0);   // Wave assignment for interleaved division

    // Movement state for run-tumble chemotaxis
    treg.newVariable<float>("move_direction_x", 0.0f);
    treg.newVariable<float>("move_direction_y", 0.0f);
    treg.newVariable<float>("move_direction_z", 0.0f);
    treg.newVariable<int>("tumble", 1);  // 0=running, 1=tumbling (default: tumble to pick initial direction)

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
    treg.newFunction("pack_for_export", pack_export_treg);

    // Movement, state, division, and occupancy grid functions only in main model
    if (include_state_divide) {
        treg.newFunction("write_to_occ_grid", treg_write_to_occ_grid);
        treg.newFunction("move", treg_move);

        treg.newFunction("state_step", treg_state_step)
            .setAllowAgentDeath(true);

        {
            flamegpu::AgentFunctionDescription fn = treg.newFunction("divide", treg_divide);
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

    // Movement state for run-tumble chemotaxis
    mdsc.newVariable<float>("move_direction_x", 0.0f);
    mdsc.newVariable<float>("move_direction_y", 0.0f);
    mdsc.newVariable<float>("move_direction_z", 0.0f);
    mdsc.newVariable<int>("tumble", 1);  // 0=running, 1=tumbling (default: tumble to pick initial direction)

    // Molecular state (affects behavior)
    mdsc.newVariable<float>("PDL1_syn", 0.0f);

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
    mdsc.newFunction("pack_for_export", pack_export_mdsc);

    // Movement and state functions only in main model (MDSCs don't divide)
    if (include_state) {
        mdsc.newFunction("write_to_occ_grid", mdsc_write_to_occ_grid);
        mdsc.newFunction("move", mdsc_move);

        mdsc.newFunction("state_step", mdsc_state_step)
            .setAllowAgentDeath(true);
    }
}

// Define the Macrophage agent (M1/M2 polarization, CCL2 chemotaxis, cancer cell killing)
void defineMacrophageAgent(flamegpu::ModelDescription& model, bool include_state) {
    flamegpu::AgentDescription mac = model.newAgent(AGENT_MACROPHAGE);

    // Identity: using FLAMEGPU's built-in ID system

    // Position
    mac.newVariable<int>("x");
    mac.newVariable<int>("y");
    mac.newVariable<int>("z");

    // Macrophage state (0=M1, 1=M2)
    mac.newVariable<int>("cell_state", 1);  // Default: M2

    // Movement state for run-tumble chemotaxis
    mac.newVariable<float>("move_direction_x", 0.0f);
    mac.newVariable<float>("move_direction_y", 0.0f);
    mac.newVariable<float>("move_direction_z", 0.0f);
    mac.newVariable<int>("tumble", 1);  // 0=running, 1=tumbling (default: tumble to pick initial direction)

    // Molecular state (affects behavior)
    mac.newVariable<float>("PDL1_syn", 0.0f);

    // Movement control
    mac.newVariable<int>("moves_remaining", 0);

    // Neighbor counts (computed via messaging)
    mac.newVariable<int>("neighbor_cancer_count", 0);

    mac.newVariable<int>("ifng_active",0);

    // Lifecycle
    mac.newVariable<int>("life", 0);
    mac.newVariable<int>("dead", 0);

    // Define agent functions
    mac.newFunction("broadcast_location", mac_broadcast_location)
        .setMessageOutput(MSG_CELL_LOCATION);

    mac.newFunction("scan_neighbors", mac_scan_neighbors)
        .setMessageInput(MSG_CELL_LOCATION);

    mac.newFunction("update_chemicals", mac_update_chemicals);

    mac.newFunction("compute_chemical_sources", mac_compute_chemical_sources);
    mac.newFunction("pack_for_export", pack_export_mac);

    // Movement and state functions only in main model
    if (include_state) {
        mac.newFunction("write_to_occ_grid", mac_write_to_occ_grid);
        mac.newFunction("move", mac_move);

        mac.newFunction("state_step", mac_state_step)
            .setAllowAgentDeath(true);
    }
}

void defineFibroblastAgent(flamegpu::ModelDescription& model, bool include_state) {
    flamegpu::AgentDescription fib = model.newAgent(AGENT_FIBROBLAST);

    // Head position (segment 0 alias — used by broadcast_location and neighbor messaging)
    fib.newVariable<int>("x");
    fib.newVariable<int>("y");
    fib.newVariable<int>("z");

    // Multi-voxel chain: one agent occupies chain_len voxels (3=NORMAL, 5=CAF)
    // Segment 0 = head (chemotaxis), segment chain_len-1 = tail
    fib.newVariable<int, MAX_FIB_CHAIN_LENGTH>("seg_x", {0, 0, 0, 0, 0});
    fib.newVariable<int, MAX_FIB_CHAIN_LENGTH>("seg_y", {0, 0, 0, 0, 0});
    fib.newVariable<int, MAX_FIB_CHAIN_LENGTH>("seg_z", {0, 0, 0, 0, 0});
    fib.newVariable<int>("chain_len", 3);

    // Fibroblast state (0=NORMAL, 1=CAF)
    fib.newVariable<int>("cell_state", FIB_NORMAL);

    // Movement state for run-tumble chemotaxis (head segment)
    fib.newVariable<float>("move_direction_x", 0.0f);
    fib.newVariable<float>("move_direction_y", 0.0f);
    fib.newVariable<float>("move_direction_z", 0.0f);
    fib.newVariable<int>("tumble", 1);

    // Lifecycle
    fib.newVariable<int>("life", 0);

    // Activation flag (set by state_step, consumed by fib_activate)
    fib.newVariable<int>("divide_flag", 0);

    // Define agent functions
    fib.newFunction("broadcast_location", fib_broadcast_location)
        .setMessageOutput(MSG_CELL_LOCATION);

    fib.newFunction("compute_chemical_sources", fib_compute_chemical_sources);
    fib.newFunction("pack_for_export", pack_export_fib);

    if (include_state) {
        fib.newFunction("write_to_occ_grid", fib_write_to_occ_grid);
        fib.newFunction("move", fib_move);
        fib.newFunction("state_step", fib_state_step)
            .setAllowAgentDeath(true);
        fib.newFunction("activate", fib_activate);
        fib.newFunction("build_density_field", fib_build_density_field);
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
    agent.newVariable<int>("cell_state", VAS_PHALANX);  // Default: mature vessel

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
    agent.newFunction("pack_for_export", pack_export_vas);

    // Occupancy grid write function
    agent.newFunction("write_to_occ_grid", vascular_write_to_occ_grid);

    // Recruitment source marking
    agent.newFunction("mark_t_sources", vascular_mark_t_sources);

    // State transitions and division
    agent.newFunction("state_step", vascular_state_step);

    // Single-phase divide using occupancy grid
    {
        flamegpu::AgentFunctionDescription fn = agent.newFunction("vascular_divide", vascular_divide);
        fn.setAgentOutput(agent);
    }

    // Single-phase movement (tip cells only, run-tumble, no occ_grid interaction)
    agent.newFunction("move", vascular_move);
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

    // ABM → QSP event counters (copied from MacroProperty to these env properties each step, for QSP access)
    env.newProperty<int>("ABM_cc_death", 0);              // Total cancer cell deaths
    env.newProperty<int>("ABM_cc_death_t_kill", 0);       // Cancer deaths from T cell killing
    env.newProperty<int>("ABM_cc_death_mac_kill", 0);     // Cancer deaths from macrophage killing
    env.newProperty<int>("ABM_cc_death_natural", 0);      // Cancer deaths from senescence
    env.newProperty<int>("ABM_TEFF_REC", 0);              // T effector cells recruited to tumor
    env.newProperty<int>("ABM_TH_REC", 0);                // T helper cells recruited to tumor
    env.newProperty<int>("ABM_TREG_REC", 0);              // T regulatory cells recruited to tumor
    env.newProperty<int>("ABM_MDSC_REC", 0);              // MDSCs recruited to tumor
    env.newProperty<int>("ABM_MAC_REC", 0);               // Macrophages recruited to tumor

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

    // Drug properties
    env.newProperty<float>("R_cabo", 0.0f); // Resistance to Cabo

    // Vasculature count (updated each step by update_vasculature_count host function)
    env.newProperty<int>("n_vasculature_total", 1);

    // Wave-interleaved division: current wave index (0..N_DIVIDE_WAVES-1), managed by host fns
    env.newProperty<int>("divide_current_wave", 0);

    // Occupancy grid: stores per-voxel cell counts indexed by AgentType enum value.
    // Dimensions are compile-time constants; only [0..grid_size-1] are used at runtime.
    env.newMacroProperty<unsigned int,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX, NUM_OCC_TYPES>("occ_grid");

    // ECM grid and fibroblast density field are now plain CUDA device arrays (d_ecm_grid,
    // d_fib_density_field) allocated in initialize_pde_solver. Their uint64_t pointers
    // are registered as env properties "ecm_grid_ptr" and "fib_density_field_ptr" by
    // set_pde_pointers_in_environment. This eliminates MacroProperty D2H/H2D copies.

    // ABM event counters: atomic increment by agent functions, reset each step.
    // Indices defined in ABMEventCounterIndex enum (common.cuh).
    // Values: cancer deaths (by cause), immune cell recruitment counts.
    env.newMacroProperty<int, ABM_EVENT_COUNTER_SIZE>("abm_event_counters");

    // Single base-pointer props for per-step stats arrays (set by main.cu after CUDA alloc)
    // event_counters_ptr → device_event_counters[ABM_EVENT_COUNTER_SIZE]  (prolif + death + PDL1)
    // state_counters_ptr → device_state_counters[ABM_STATE_COUNTER_SIZE]  (per-state agent counts)
    env.newProperty<uint64_t>("event_counters_ptr", 0u);
    env.newProperty<uint64_t>("state_counters_ptr", 0u);

    // ABM export: GPU-side buffer and atomic counter (set at runtime by main.cu)
    env.newProperty<uint64_t>("abm_export_buf_ptr", 0u);
    env.newProperty<uint64_t>("abm_export_counter_ptr", 0u);
    env.newProperty<int>("do_abm_export", 0);

    // Populate ALL other parameters from XML
    params.populateFlameGPUEnvironment(env);
}

// Build the complete model
std::unique_ptr<flamegpu::ModelDescription> buildModel(
    int grid_x, int grid_y, int grid_z, float voxel_size,
    const PDAC::GPUParam& gpu_params) {

    auto model = std::make_unique<flamegpu::ModelDescription>("PDAC_ABM_GPU");

    int grid_max = std::max({grid_x, grid_y, grid_z});

    // MSG_CELL_LOCATION is still used by broadcast_location and scan_neighbors functions
    // (for state transitions and neighbor detection). MSG_INTENT has been removed;
    // all movement and division now use the occupancy grid (occ_grid) directly.
    defineCellLocationMessage(*model, voxel_size, grid_max);

    // Define agents with all functions
    defineCancerCellAgent(*model, true);
    defineTCellAgent(*model, true);
    defineTRegAgent(*model, true);
    defineMDSCAgent(*model, true);
    defineMacrophageAgent(*model, true);
    defineFibroblastAgent(*model, true);
    defineVascularCellAgent(*model);

    // Define environment with GPU parameters loaded from XML
    defineEnvironment(*model, grid_x, grid_y, grid_z, voxel_size, gpu_params);

    // Define all model layers (state transitions, PDE, movement, division, QSP)
    // Move step counts are read from the environment inside defineMainModelLayers.
    defineMainModelLayers(*model);

    return model;
}

} // namespace PDAC
