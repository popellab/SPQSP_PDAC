#include "initialization.cuh"
#include "../core/common.cuh"
#include "../abm/gpu_param.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <string>

namespace PDAC {

// ============================================================================
// SimulationConfig Implementation
// ============================================================================

SimulationConfig::SimulationConfig()
    : steps(200)
    , random_seed(12345)
    , init_method(0)
    , cluster_radius(5)
    , num_tcells(50)
    , num_tregs(10)
    , num_mdscs(5)
    , abm_out(true)
    , pde_out(false)
    , interval_out(1)
{
}

void SimulationConfig::parseCommandLine(int argc, const char** argv, const PDAC::GPUParam gpu_params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if ((arg == "--param-file" || arg == "-p") && i + 1 < argc);
        //param file already accounted for
        else if ((arg == "--initialization " || arg == "-i") && i + 1 < argc);
        //need to write still
        else if ((arg == "--steps" || arg == "-s") && i + 1 < argc) {
            steps = std::atoi(argv[++i]);
        } else if ((arg == "--out_abm" || arg == "-oa") && i + 1 < argc) {
            abm_out =  std::atoi(argv[++i]);
        } else if ((arg == "--out_pde" || arg == "-op") && i + 1 < argc) {
            pde_out =  std::atoi(argv[++i]);
        } else if ((arg == "--out_int" || arg == "-oi") && i + 1 < argc) {
            interval_out =  std::atoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            random_seed = std::atoi(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "\nOptions:\n"
                      << "  -p, --param-file FILE    Path to parameter XML file [default: param_all_test.xml]\n"
                      << "  -i, --initialization N   initialization type [default: 0] (not implemented)\n"
                      << "  -s, --steps N            Number of simulation steps [default: 200]\n"
                      << "  -oa, --out_abm Bool      Output ABM at interval frequency [default: true]\n"
                      << "  -op, --out_pde Bool      Output PDE at interval frequency [default: true]\n"
                      << "  -oi, --out_int N         Output interval frequency [default: 1]"
                      << "  --seed N                 Random seed [default: 12345]\n"
                      << "  -h, --help               Show this help\n";
            exit(0);
        }
    }

    grid_x     = gpu_params.getInt(PARAM_X_SIZE);
    grid_y     = gpu_params.getInt(PARAM_Y_SIZE);
    grid_z     = gpu_params.getInt(PARAM_Z_SIZE);
    voxel_size = gpu_params.getInt(PARAM_VOXEL_SIZE);

    cancer_move_steps = gpu_params.getInt(PARAM_CANCER_MOVE_STEPS);
    tcell_move_steps = gpu_params.getInt(PARAM_TCELL_MOVE_STEPS);
    treg_move_steps = gpu_params.getInt(PARAM_TCELL_MOVE_STEPS);
    mdsc_move_steps = gpu_params.getInt(PARAM_MDSC_MOVE_STEPS);
    dt_abm = gpu_params.getFloat(PARAM_SEC_PER_SLICE);
    molecular_steps = gpu_params.getInt(PARAM_MOLECULAR_STEPS);
}

void SimulationConfig::print() const {
    std::cout << "\n=== TNBC ABM-PDE GPU Simulation ===" << std::endl;
    std::cout << "Grid: " << grid_x << "x" << grid_y << "x" << grid_z << std::endl;
    std::cout << "Voxel size: " << voxel_size << " µm" << std::endl;
    std::cout << "Steps: " << steps << std::endl;
    std::cout << "Random seed: " << random_seed << std::endl;
    
    std::cout << "\nInitial Cell Populations:" << std::endl;
    std::cout << "  Tumor radius: " << cluster_radius << " voxels" << std::endl;
    std::cout << "  T cells: " << num_tcells << std::endl;
    std::cout << "  TRegs: " << num_tregs << std::endl;
    std::cout << "  MDSCs: " << num_mdscs << std::endl;
    
    std::cout << "\nPDE Integration:" << std::endl;
    std::cout << "  ABM timestep: " << dt_abm << " s (" << (dt_abm/60.0f) << " min)" << std::endl;
    std::cout << "  Molecular substeps: " << molecular_steps << std::endl;
    std::cout << "  PDE timestep: " << (dt_abm / molecular_steps) << " s" << std::endl;
    std::cout << "===================================\n" << std::endl;
}

// ============================================================================
// Cancer Cell Initialization
// ============================================================================

void initializeCancerCellCluster(
    flamegpu::AgentVector& cancer_agents,
    int grid_x, int grid_y, int grid_z,
    int cluster_radius,
    float stem_div_interval,
    float progenitor_div_interval,
    int progenitor_div_max)
{
    unsigned int count = 1;
    const int cx = grid_x / 2;
    const int cy = grid_y / 2;
    const int cz = grid_z / 2;

    for (int x = cx - cluster_radius; x <= cx + cluster_radius; x++) {
        for (int y = cy - cluster_radius; y <= cy + cluster_radius; y++) {
            for (int z = cz - cluster_radius; z <= cz + cluster_radius; z++) {
                const float dist = std::sqrt(
                    static_cast<float>((x - cx) * (x - cx) +
                    (y - cy) * (y - cy) +
                    (z - cz) * (z - cz))
                );

                if (dist > cluster_radius) continue;

                cancer_agents.push_back();
                flamegpu::AgentVector::Agent agent = cancer_agents.back();

                // Stem cells at center, progenitors at periphery
                bool is_stem = (dist < cluster_radius * 0.3f);
                int cell_state = is_stem ? CANCER_STEM : CANCER_PROGENITOR;
                int div_cd = is_stem ? 
                    static_cast<int>(stem_div_interval) : 
                    static_cast<int>(progenitor_div_interval);

                // Basic identity and state
                const int id = agent.getID();
                agent.setVariable<int>("x", x);
                agent.setVariable<int>("y", y);
                agent.setVariable<int>("z", z);
                agent.setVariable<int>("cell_state", cell_state);
                agent.setVariable<int>("divideCD", div_cd);
                agent.setVariable<int>("divideFlag", 1);
                agent.setVariable<int>("divideCountRemaining", progenitor_div_max);
                agent.setVariable<unsigned int>("stemID", is_stem ? id : 0);
                
                // Chemical state variables (from PDE coupling)
                agent.setVariable<float>("local_NO", 0.001f);
                agent.setVariable<float>("local_ArgI", 0.0f);
                agent.setVariable<float>("local_TGFB", 0.0f);
                agent.setVariable<float>("local_O2", 0.0f);
                agent.setVariable<float>("local_IFNg", 0.0f);

                agent.setVariable<float>("PDL1_surface", 0.1f);
                agent.setVariable<float>("PDL1_syn_rate", 0.0f);
                agent.setVariable<float>("PDL1_syn", 0.1f);
                agent.setVariable<float>("O2_uptake_rate", 0.0f);
                agent.setVariable<int>("hypoxic", 0);
                agent.setVariable<float>("cabo_effect", 0.0f);
                
                // Neighbor counts
                agent.setVariable<int>("neighbor_Teff_count", 0);
                agent.setVariable<int>("neighbor_Treg_count", 0);
                agent.setVariable<int>("neighbor_cancer_count", 0);
                agent.setVariable<int>("neighbor_MDSC_count", 0);
                agent.setVariable<unsigned int>("available_neighbors", 0u);
                
                // Lifecycle
                agent.setVariable<int>("life", 0);
                agent.setVariable<int>("dead", 0);
                
                // Intent
                agent.setVariable<int>("intent_action", INTENT_NONE);
                agent.setVariable<int>("target_x", x);
                agent.setVariable<int>("target_y", y);
                agent.setVariable<int>("target_z", z);

                count++;
            }
        }
    }

    // // TEMP: Initialize 7 cells in the grid center
    // std::vector<std::vector<int>> points = {{25,25,25},{25,25,26},{25,25,24},{25,26,25},{25,24,25},{26,25,25},{24,25,25}};
    // std::vector<int> ids = {1,2,3,4,5,6,7};
    // for (int i=0; i < 7; i++){
    //     cancer_agents.push_back();
    //     flamegpu::AgentVector::Agent agent = cancer_agents.back();

    //     // Stem cells at center, progenitors at periphery
    //     bool is_stem = false;
    //     int cell_state = is_stem ? CANCER_STEM : CANCER_PROGENITOR;
    //     int div_cd = is_stem ? 
    //         static_cast<int>(stem_div_interval) : 
    //         static_cast<int>(progenitor_div_interval);

    //     // Basic identity and state
    //     const int id = agent.getID();
    //     agent.setVariable<int>("x", points[i][0]);
    //     agent.setVariable<int>("y", points[i][1]);
    //     agent.setVariable<int>("z", points[i][2]);
    //     agent.setVariable<int>("cell_state", cell_state);
    //     agent.setVariable<int>("divideCD", div_cd);
    //     agent.setVariable<int>("divideFlag", 1);
    //     agent.setVariable<int>("divideCountRemaining", progenitor_div_max);
    //     agent.setVariable<unsigned int>("stemID", is_stem ? id : 0);
        
    //     // Chemical state variables (from PDE coupling)
    //     agent.setVariable<float>("local_NO", 0.001f);
    //     agent.setVariable<float>("local_ArgI", 0.0f);
    //     agent.setVariable<float>("local_TGFB", 0.0f);
    //     agent.setVariable<float>("local_O2", 0.0f);
    //     agent.setVariable<float>("local_IFNg", 0.0f);

    //     agent.setVariable<float>("PDL1_surface", 0.1f);
    //     agent.setVariable<float>("PDL1_syn_rate", 0.0f);
    //     agent.setVariable<float>("PDL1_syn", 0.1f);
    //     agent.setVariable<float>("O2_uptake_rate", 0.0f);
    //     agent.setVariable<int>("hypoxic", 0);
    //     agent.setVariable<float>("cabo_effect", 0.0f);
        
    //     // Neighbor counts
    //     agent.setVariable<int>("neighbor_Teff_count", 0);
    //     agent.setVariable<int>("neighbor_Treg_count", 0);
    //     agent.setVariable<int>("neighbor_cancer_count", 0);
    //     agent.setVariable<int>("neighbor_MDSC_count", 0);
    //     agent.setVariable<unsigned int>("available_neighbors", 0u);
        
    //     // Lifecycle
    //     agent.setVariable<int>("life", 0);
    //     agent.setVariable<int>("dead", 0);
        
    //     // Intent
    //     agent.setVariable<int>("intent_action", INTENT_NONE);
    //     agent.setVariable<int>("target_x", -1);
    //     agent.setVariable<int>("target_y", -1);
    //     agent.setVariable<int>("target_z", -1);

    //     count++;
    // }

    std::cout << "Initialized " << (count - 1) << " cancer cells in cluster" << std::endl;
}

// ============================================================================
// T Cell Initialization
// ============================================================================

void initializeTCells(
    flamegpu::AgentVector& tcell_agents,
    int grid_x, int grid_y, int grid_z,
    int tumor_radius, int num_tcells,
    float tcell_life_mean, int div_limit,
    float IL2_release_time, float IFNg_release_time)
{
    const int cx = grid_x / 2;
    const int cy = grid_y / 2;
    const int cz = grid_z / 2;

    // Place T cells in shell around tumor (invasive front)
    const float inner_radius = tumor_radius + 1;
    const float outer_radius = tumor_radius + 4;

    int placed = 0;
    int attempts = 0;
    const int max_attempts = num_tcells * 100;

    while (placed < num_tcells && attempts < max_attempts) {
        attempts++;

        // Random spherical coordinates
        float theta = static_cast<float>(rand()) / RAND_MAX * 2.0f * 3.14159f;
        float phi = std::acos(2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f);
        float r = inner_radius + static_cast<float>(rand()) / RAND_MAX * (outer_radius - inner_radius);

        int x = cx + static_cast<int>(r * std::sin(phi) * std::cos(theta));
        int y = cy + static_cast<int>(r * std::sin(phi) * std::sin(theta));
        int z = cz + static_cast<int>(r * std::cos(phi));

        // Bounds check
        if (x < 0 || x >= grid_x || y < 0 || y >= grid_y || z < 0 || z >= grid_z) {
            continue;
        }

        // Random life from exponential distribution
        float rnd = static_cast<float>(rand()) / RAND_MAX;
        int life = static_cast<int>(tcell_life_mean * std::log(1.0f / (rnd + 0.0001f)) + 0.5f);
        if (life < 1) life = 1;

        tcell_agents.push_back();
        flamegpu::AgentVector::Agent agent = tcell_agents.back();

        // Basic identity and state
        agent.setVariable<int>("x", x);
        agent.setVariable<int>("y", y);
        agent.setVariable<int>("z", z);
        agent.setVariable<int>("cell_state", T_CELL_EFF);
        agent.setVariable<int>("divide_flag", 0);
        agent.setVariable<int>("divide_cd", 0);
        agent.setVariable<int>("divide_limit", div_limit);
        
        // Chemical production/exposure
        agent.setVariable<float>("IL2_exposure", 0.0f);
        agent.setVariable<float>("IL2_release_remain", IL2_release_time);
        agent.setVariable<float>("IFNg_release_remain", IFNg_release_time);
        agent.setVariable<float>("IFNg_release_rate", 0.0f);
        agent.setVariable<float>("IL2_release_rate", 0.0f);
        
        // Local chemical concentrations (from PDE)
        agent.setVariable<float>("local_O2", 0.001f);
        agent.setVariable<float>("local_IFNg", 0.0f);
        agent.setVariable<float>("local_IL2", 0.0f);
        agent.setVariable<float>("local_IL10", 0.0f);
        agent.setVariable<float>("local_TGFB", 0.0f);

        // Cumulative exposures and functional state
        agent.setVariable<float>("IL10_exposure", 0.0f);
        agent.setVariable<float>("TGFB_exposure", 0.0f);
        agent.setVariable<float>("PD1_occupancy", 0.0f);
        agent.setVariable<float>("activation_level", 1.0f);
        agent.setVariable<int>("can_proliferate", 0);
        
        // Neighbor counts
        agent.setVariable<int>("neighbor_cancer_count", 0);
        agent.setVariable<int>("neighbor_Treg_count", 0);
        agent.setVariable<int>("neighbor_all_count", 0);
        agent.setVariable<float>("max_neighbor_PDL1", 0.0f);
        agent.setVariable<int>("found_progenitor", 0);
        agent.setVariable<unsigned int>("available_neighbors", 0u);
        
        // Lifecycle
        agent.setVariable<int>("life", life);
        agent.setVariable<int>("dead", 0);
        
        // Intent
        agent.setVariable<int>("intent_action", INTENT_NONE);
        agent.setVariable<int>("target_x", -1);
        agent.setVariable<int>("target_y", -1);
        agent.setVariable<int>("target_z", -1);

        placed++;
    }

    std::cout << "Initialized " << placed << " T cells around tumor margin" << std::endl;
}

// ============================================================================
// TReg Initialization
// ============================================================================

void initializeTRegs(
    flamegpu::AgentVector& treg_agents,
    int grid_x, int grid_y, int grid_z,
    int tumor_radius, int num_tregs,
    float treg_life_mean, int div_limit)
{
    const int cx = grid_x / 2;
    const int cy = grid_y / 2;
    const int cz = grid_z / 2;

    const float inner_radius = tumor_radius + 1;
    const float outer_radius = tumor_radius + 4;

    int placed = 0;
    int attempts = 0;
    const int max_attempts = num_tregs * 100;

    while (placed < num_tregs && attempts < max_attempts) {
        attempts++;

        float theta = static_cast<float>(rand()) / RAND_MAX * 2.0f * 3.14159f;
        float phi = std::acos(2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f);
        float r = inner_radius + static_cast<float>(rand()) / RAND_MAX * (outer_radius - inner_radius);

        int x = cx + static_cast<int>(r * std::sin(phi) * std::cos(theta));
        int y = cy + static_cast<int>(r * std::sin(phi) * std::sin(theta));
        int z = cz + static_cast<int>(r * std::cos(phi));

        if (x < 0 || x >= grid_x || y < 0 || y >= grid_y || z < 0 || z >= grid_z) {
            continue;
        }

        float rnd = static_cast<float>(rand()) / RAND_MAX;
        int life = static_cast<int>(treg_life_mean * std::log(1.0f / (rnd + 0.0001f)) + 0.5f);
        if (life < 1) life = 1;

        treg_agents.push_back();
        flamegpu::AgentVector::Agent agent = treg_agents.back();

        // Basic identity
        agent.setVariable<int>("x", x);
        agent.setVariable<int>("y", y);
        agent.setVariable<int>("z", z);
        agent.setVariable<int>("divide_flag", 0);
        agent.setVariable<int>("divide_cd", 0);
        agent.setVariable<int>("divide_limit", div_limit);
        
        // Chemical variables
        agent.setVariable<float>("local_O2", 0.001f);
        agent.setVariable<float>("local_IL2", 0.0f);
        agent.setVariable<float>("local_TGFB", 0.0f);
        agent.setVariable<float>("IL10_release_rate", 0.0f);
        agent.setVariable<float>("TGFB_release_rate", 0.0f);
        agent.setVariable<float>("IL2_consumption_rate", 0.0f);
        agent.setVariable<float>("IL2_exposure", 0.0f);
        agent.setVariable<float>("suppression_strength", 1.0f);
        agent.setVariable<int>("can_proliferate", 0);
        
        // Neighbor counts
        agent.setVariable<int>("neighbor_Tcell_count", 0);
        agent.setVariable<int>("neighbor_Treg_count", 0);
        agent.setVariable<int>("neighbor_cancer_count", 0);
        agent.setVariable<int>("neighbor_all_count", 0);
        agent.setVariable<unsigned int>("available_neighbors", 0u);
        
        // Lifecycle
        agent.setVariable<int>("life", life);
        agent.setVariable<int>("dead", 0);
        
        // Intent
        agent.setVariable<int>("intent_action", INTENT_NONE);
        agent.setVariable<int>("target_x", -1);
        agent.setVariable<int>("target_y", -1);
        agent.setVariable<int>("target_z", -1);

        placed++;
    }

    std::cout << "Initialized " << placed << " TReg cells around tumor margin" << std::endl;
}

// ============================================================================
// MDSC Initialization
// ============================================================================

void initializeMDSCs(
    flamegpu::AgentVector& mdsc_agents,
    int grid_x, int grid_y, int grid_z,
    int tumor_radius, int num_mdscs,
    float mdsc_life_mean)
{
    const int cx = grid_x / 2;
    const int cy = grid_y / 2;
    const int cz = grid_z / 2;

    const float inner_radius = tumor_radius + 1;
    const float outer_radius = tumor_radius + 5;

    int placed = 0;
    int attempts = 0;
    const int max_attempts = num_mdscs * 100;

    while (placed < num_mdscs && attempts < max_attempts) {
        attempts++;

        float theta = static_cast<float>(rand()) / RAND_MAX * 2.0f * 3.14159f;
        float phi = std::acos(2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f);
        float r = inner_radius + static_cast<float>(rand()) / RAND_MAX * (outer_radius - inner_radius);

        int x = cx + static_cast<int>(r * std::sin(phi) * std::cos(theta));
        int y = cy + static_cast<int>(r * std::sin(phi) * std::sin(theta));
        int z = cz + static_cast<int>(r * std::cos(phi));

        if (x < 0 || x >= grid_x || y < 0 || y >= grid_y || z < 0 || z >= grid_z) {
            continue;
        }

        float rnd = static_cast<float>(rand()) / RAND_MAX;
        int life = static_cast<int>(mdsc_life_mean * std::log(1.0f / (rnd + 0.0001f)) + 0.5f);
        if (life < 1) life = 1;

        mdsc_agents.push_back();
        flamegpu::AgentVector::Agent agent = mdsc_agents.back();

        // Basic identity
        agent.setVariable<int>("x", x);
        agent.setVariable<int>("y", y);
        agent.setVariable<int>("z", z);
        
        // Chemical variables
        agent.setVariable<float>("local_O2", 0.001f);
        agent.setVariable<float>("local_CCL2", 0.0f);
        agent.setVariable<float>("local_TGFB", 0.0f);
        agent.setVariable<float>("ROS_release_rate", 0.0f);
        agent.setVariable<float>("NO_release_rate", 0.0f);
        agent.setVariable<float>("suppression_radius", 1.0f);
        agent.setVariable<float>("activation_level", 1.0f);
        agent.setVariable<float>("CCL2_gradient_x", 0.0f);
        agent.setVariable<float>("CCL2_gradient_y", 0.0f);
        agent.setVariable<float>("CCL2_gradient_z", 0.0f);
        
        // Neighbor counts
        agent.setVariable<int>("neighbor_cancer_count", 0);
        agent.setVariable<int>("neighbor_Tcell_count", 0);
        agent.setVariable<int>("neighbor_Treg_count", 0);
        agent.setVariable<int>("neighbor_MDSC_count", 0);
        agent.setVariable<unsigned int>("available_neighbors", 0u);
        
        // Lifecycle
        agent.setVariable<int>("life", life);
        agent.setVariable<int>("dead", 0);
        
        // Intent
        agent.setVariable<int>("intent_action", INTENT_NONE);
        agent.setVariable<int>("target_x", -1);
        agent.setVariable<int>("target_y", -1);
        agent.setVariable<int>("target_z", -1);

        placed++;
    }

    std::cout << "Initialized " << placed << " MDSCs around tumor margin" << std::endl;
}

// ============================================================================
// Master Initialization Function
// ============================================================================

void initializeAllAgents(
    flamegpu::CUDASimulation& simulation,
    flamegpu::ModelDescription& model,
    const SimulationConfig& config)
{
    std::cout << "\n=== Initializing Agent Populations ===" << std::endl;
    
    // Get environment properties for agent initialization
    const float stem_div = model.Environment().getProperty<float>("PARAM_FLOAT_CANCER_CELL_STEM_DIV_INTERVAL_SLICE");
    const float prog_div = model.Environment().getProperty<float>("PARAM_FLOAT_CANCER_CELL_PROGENITOR_DIV_INTERVAL_SLICE");
    const int prog_max = model.Environment().getProperty<int>("PARAM_PROG_DIV_MAX");
    const float tcell_life = model.Environment().getProperty<float>("PARAM_T_CELL_LIFE_MEAN_SLICE");
    const int tcell_div_limit = model.Environment().getProperty<int>("PARAM_TCELL_DIV_LIMIT");
    const float IL2_release_time = model.Environment().getProperty<float>("PARAM_TCELL_IL2_RELEASE_TIME");
    const float IFN_release_time = model.Environment().getProperty<float>("PARAM_TCELL_IFNG_RELEASE_TIME");
    const float treg_life = model.Environment().getProperty<float>("PARAM_TCD4_LIFE_MEAN_SLICE");
    const int treg_div_limit = model.Environment().getProperty<int>("PARAM_TCD4_DIV_LIMIT");
    const float mdsc_life = model.Environment().getProperty<float>("PARAM_MDSC_LIFE_MEAN_SLICE");
    
    // Initialize cancer cells
    {
        flamegpu::AgentVector cancer_pop(model.Agent(AGENT_CANCER_CELL));
        initializeCancerCellCluster(
            cancer_pop, 
            config.grid_x, config.grid_y, config.grid_z,
            config.cluster_radius, stem_div, prog_div, prog_max);
        simulation.setPopulationData(cancer_pop);
    }
    
    // // Initialize T cells
    // if (config.num_tcells > 0) {
    //     flamegpu::AgentVector tcell_pop(model.Agent(AGENT_TCELL));
    //     initializeTCells(
    //         tcell_pop, 
    //         config.grid_x, config.grid_y, config.grid_z,
    //         config.cluster_radius, config.num_tcells, 
    //         tcell_life, tcell_div_limit,
    //         IL2_release_time, IFN_release_time);
    //     simulation.setPopulationData(tcell_pop);
    // }
    
    // // Initialize TRegs
    // if (config.num_tregs > 0) {
    //     flamegpu::AgentVector treg_pop(model.Agent(AGENT_TREG));
    //     initializeTRegs(
    //         treg_pop, 
    //         config.grid_x, config.grid_y, config.grid_z,
    //         config.cluster_radius, config.num_tregs, 
    //         treg_life, treg_div_limit);
    //     simulation.setPopulationData(treg_pop);
    // }
    
    // // Initialize MDSCs
    // if (config.num_mdscs > 0) {
    //     flamegpu::AgentVector mdsc_pop(model.Agent(AGENT_MDSC));
    //     initializeMDSCs(
    //         mdsc_pop, 
    //         config.grid_x, config.grid_y, config.grid_z,
    //         config.cluster_radius, config.num_mdscs, 
    //         mdsc_life);
    //     simulation.setPopulationData(mdsc_pop);
    // }
    
    std::cout << "Agent initialization complete\n" << std::endl;
}

} // namespace PDAC