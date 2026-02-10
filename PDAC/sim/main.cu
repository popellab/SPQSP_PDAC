#include "flamegpu/flamegpu.h"
#include <iostream>
#include <memory>
#include <fstream>
#include <sstream>
#include <vector>
#include <iomanip>

#include "../core/common.cuh"
#include "../pde/pde_integration.cuh"
#include "initialization.cuh"
#include "gpu_param.h"
#include "../qsp/LymphCentral_wrapper.h"
#include "../core/model_functions.cuh"

namespace PDAC {
    std::unique_ptr<flamegpu::ModelDescription> buildModel(
        int grid_x, int grid_y, int grid_z, float voxel_size,
        const PDAC::GPUParam& gpu_params);

    // void set_internal_params(flamegpu::ModelDescription& model, 
    //                          const LymphCentralWrapper& lymph);
}

// ============================================================================
// Simulation Monitoring Functions
// ============================================================================

FLAMEGPU_STEP_FUNCTION(exportPDEData) {
    unsigned int step = FLAMEGPU->environment.getProperty<unsigned int>("current_step");
    int interval = FLAMEGPU->environment.getProperty<int>("interval_out");

    // Only export every interval steps
    if (step % interval != 0) return;

    // Only export PDE data if solver is initialized
    if (!PDAC::g_pde_solver) return;

    // Get grid dimensions
    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int grid_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");

    // Create output directory if needed (should already exist)
    std::ostringstream filename;
    filename << "outputs/pde/pde_step_" << std::setw(6) << std::setfill('0') << step << ".csv";
    std::ofstream file(filename.str());

    // Write header
    file << "x,y,z,O2,IFN,IL2,IL10,TGFB,CCL2,ARGI,NO,IL12,VEGFA\n";

    // Allocate host buffer for concentration data
    const int total_voxels = grid_x * grid_y * grid_z;
    std::vector<float> concentrations(total_voxels);

    // Create a 2D array to store all substrate concentrations [NUM_SUBSTRATES][total_voxels]
    std::vector<std::vector<float>> all_concentrations(PDAC::NUM_SUBSTRATES);

    // Read all substrates from device
    for (int substrate_idx = 0; substrate_idx < PDAC::NUM_SUBSTRATES; substrate_idx++) {
        all_concentrations[substrate_idx].resize(total_voxels);
        PDAC::g_pde_solver->get_concentrations(all_concentrations[substrate_idx].data(), substrate_idx);
    }

    // Write data for each voxel
    for (int z = 0; z < grid_z; z++) {
        for (int y = 0; y < grid_y; y++) {
            for (int x = 0; x < grid_x; x++) {
                int voxel_idx = z * (grid_x * grid_y) + y * grid_x + x;

                file << x << "," << y << "," << z;

                // Write all substrate concentrations
                for (int substrate_idx = 0; substrate_idx < PDAC::NUM_SUBSTRATES; substrate_idx++) {
                    file << "," << all_concentrations[substrate_idx][voxel_idx];
                }

                file << "\n";
            }
        }
    }

    file.close();
}

FLAMEGPU_STEP_FUNCTION(exportABMData) {
    unsigned int step = FLAMEGPU->environment.getProperty<unsigned int>("current_step");
    int interval = FLAMEGPU->environment.getProperty<int>("interval_out");

    // Only export every interval steps
    if (step % interval != 0) return;

    std::ostringstream filename;
    filename << "outputs/abm/agents_step_" << std::setw(6) << std::setfill('0') << step << ".csv";
    std::ofstream file(filename.str());
    file << "agent_type,agent_id,x,y,z,cell_state,additional_info\n";
    // Cancer Cells
    {
        auto agent = FLAMEGPU->agent(PDAC::AGENT_CANCER_CELL);
        unsigned int count = agent.count();
        if (count > 0) {
            flamegpu::DeviceAgentVector cancer_pop = agent.getPopulationData();
            for (unsigned int i = 0; i < count; ++i) {
                unsigned int id = cancer_pop[i].getID();
                int x = cancer_pop[i].getVariable<int>("x");
                int y = cancer_pop[i].getVariable<int>("y");
                int z = cancer_pop[i].getVariable<int>("z");
                int state = cancer_pop[i].getVariable<int>("cell_state");
                int divideCD = cancer_pop[i].getVariable<int>("divideCD");
                int divideFlag = cancer_pop[i].getVariable<int>("divideFlag");
                
                std::string state_name;
                switch (state) {
                    case 0: state_name = "STEM"; break;
                    case 1: state_name = "PROGENITOR"; break;
                    case 2: state_name = "SENESCENT"; break;
                    default: state_name = "UNKNOWN"; break;
                }
                
                file << "CANCER," << id << "," << x << "," << y << "," << z << "," 
                     << state_name << ",divideCD=" << divideCD 
                     << ";divideFlag=" << divideFlag << "\n";
            }
        }
    }
    
    // T Cells
    {
        auto agent = FLAMEGPU->agent(PDAC::AGENT_TCELL);
        unsigned int count = agent.count();
        if (count > 0) {
            flamegpu::DeviceAgentVector tcell_pop = agent.getPopulationData();
            for (unsigned int i = 0; i < count; ++i) {
                unsigned int id = tcell_pop[i].getID();
                int x = tcell_pop[i].getVariable<int>("x");
                int y = tcell_pop[i].getVariable<int>("y");
                int z = tcell_pop[i].getVariable<int>("z");
                int state = tcell_pop[i].getVariable<int>("cell_state");
                int life = tcell_pop[i].getVariable<int>("life");
                
                std::string state_name;
                switch (state) {
                    case 0: state_name = "EFFECTOR"; break;
                    case 1: state_name = "CYTOTOXIC"; break;
                    case 2: state_name = "SUPPRESSED"; break;
                    default: state_name = "UNKNOWN"; break;
                }
                
                file << "TCELL," << id << "," << x << "," << y << "," << z << "," 
                     << state_name << ",life=" << life << "\n";
            }
        }
    }
    
    // TRegs
    {
        auto agent = FLAMEGPU->agent(PDAC::AGENT_TREG);
        unsigned int count = agent.count();
        if (count > 0) {
            flamegpu::DeviceAgentVector treg_pop = agent.getPopulationData();
            for (unsigned int i = 0; i < count; ++i) {
                unsigned int id = treg_pop[i].getID();
                int x = treg_pop[i].getVariable<int>("x");
                int y = treg_pop[i].getVariable<int>("y");
                int z = treg_pop[i].getVariable<int>("z");
                int life = treg_pop[i].getVariable<int>("life");
                
                file << "TREG," << id << "," << x << "," << y << "," << z << "," 
                     << "REGULATORY,life=" << life << "\n";
            }
        }
    }
    
    // MDSCs
    {
        auto agent = FLAMEGPU->agent(PDAC::AGENT_MDSC);
        unsigned int count = agent.count();
        if (count > 0) {
            flamegpu::DeviceAgentVector mdsc_pop = agent.getPopulationData();
            for (unsigned int i = 0; i < count; ++i) {
                unsigned int id = mdsc_pop[i].getID();
                int x = mdsc_pop[i].getVariable<int>("x");
                int y = mdsc_pop[i].getVariable<int>("y");
                int z = mdsc_pop[i].getVariable<int>("z");
                int life = mdsc_pop[i].getVariable<int>("life");
                
                file << "MDSC," << id << "," << x << "," << y << "," << z << "," 
                     << "MDSC,life=" << life << "\n";
            }
        }
    }
    file.close();
}

FLAMEGPU_STEP_FUNCTION(stepCounter) {
    unsigned int step = FLAMEGPU->environment.getProperty<unsigned int>("current_step");
    FLAMEGPU->environment.setProperty<unsigned int>("current_step", step + 1);

    if (step % 50 == 0) {
        const unsigned int cancer_count = FLAMEGPU->agent(PDAC::AGENT_CANCER_CELL).count();
        const unsigned int tcell_count = FLAMEGPU->agent(PDAC::AGENT_TCELL).count();
        const unsigned int treg_count = FLAMEGPU->agent(PDAC::AGENT_TREG).count();
        const unsigned int mdsc_count = FLAMEGPU->agent(PDAC::AGENT_MDSC).count();
        std::cout << "Step " << step
                  << ": Cancer=" << cancer_count
                  << ", T cells=" << tcell_count
                  << ", TRegs=" << treg_count
                  << ", MDSCs=" << mdsc_count << std::endl;
    }
}

FLAMEGPU_EXIT_CONDITION(checkSimulationEnd) {
    const unsigned int cancer_count = FLAMEGPU->agent(PDAC::AGENT_CANCER_CELL).count();
    if (cancer_count == 0) {
        std::cout << "\nAll cancer cells eliminated!" << std::endl;
        return flamegpu::EXIT;
    }
    return flamegpu::CONTINUE;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, const char** argv) {
    // Check for -p flag (XML path override)
    std::string param_file = "/home/chase/SPQSP/SPQSP_PDAC-main/PDAC/sim/resource/param_all_test.xml";
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "-p" && i + 1 < argc) {
            param_file = argv[++i];
            break;
        }
    }

    // Load XML parameters
    std::cout << "Loading parameters from: " << param_file << std::endl;
    PDAC::GPUParam gpu_params;
    try {
        gpu_params.initializeParams(param_file);
        std::cout << "Parameters loaded successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Failed to load parameters from XML: " << e.what() << std::endl;
        return 1;
    }

    // Parse configuration from command line
    PDAC::SimulationConfig config;
    config.parseCommandLine(argc, argv, gpu_params);
    config.print();

    // Seed random number generator
    srand(config.random_seed);
    
    // ========== BUILD MODEL ==========
    std::cout << "Building FLAME GPU 2 model..." << std::endl;
    auto model = PDAC::buildModel(
        config.grid_x, config.grid_y, config.grid_z,
        config.voxel_size,
        gpu_params);

    // Store output interval in environment for step functions
    model->Environment().newProperty<int>("interval_out", config.interval_out);

    // ========== INITIALIZE PDE SOLVER ==========
    std::cout << "Initializing PDE solver..." << std::endl;
    PDAC::initialize_pde_solver(
        config.grid_x, config.grid_y, config.grid_z, 
        config.voxel_size, config.dt_abm, config.molecular_steps,
         gpu_params);
    
    // Store PDE device pointers in model environment
    PDAC::set_pde_pointers_in_environment(*model);

    //     TODO
    // Process internal parameters from env params and new QSP params
    // ========== INITIALIZE QSP SOLVER ==========
    PDAC::LymphCentralWrapper _lymph;
    _lymph.initialize(param_file);
    PDAC::set_internal_params(*model, _lymph);

    // ========== ADD STEP FUNCTIONS ==========
    if (config.pde_out) {
        model->addStepFunction(exportPDEData);
    }
    if (config.abm_out) {
        model->addStepFunction(exportABMData);
    }
    model->addStepFunction(stepCounter);
    model->addExitCondition(checkSimulationEnd);
    
    // ========== CREATE SIMULATION ==========
    std::cout << "Creating CUDA simulation..." << std::endl;
    flamegpu::CUDASimulation simulation(*model);
    simulation.SimulationConfig().steps = config.steps;
    simulation.SimulationConfig().random_seed = config.random_seed;
    
    // ========== INITIALIZE AGENTS ==========
    if (config.init_method == 0) {
        std::cout << "Initializing agents with random distribution..." << std::endl;
        PDAC::initializeAllAgents(simulation, *model, config);
    } else { // do nothing, no other options right now
        std::cout << "Broken initialization" << std::endl;
        return 1;
    }
    
    // ========== RUN SIMULATION ==========
    std::cout << "\n=== Starting Simulation ===" << std::endl;
    simulation.simulate();
    
    // ========== REPORT RESULTS ==========
    std::cout << "\n=== Simulation Complete ===" << std::endl;
    
    flamegpu::AgentVector final_cancer(model->Agent(PDAC::AGENT_CANCER_CELL));
    flamegpu::AgentVector final_tcells(model->Agent(PDAC::AGENT_TCELL));
    flamegpu::AgentVector final_tregs(model->Agent(PDAC::AGENT_TREG));
    flamegpu::AgentVector final_mdscs(model->Agent(PDAC::AGENT_MDSC));
    
    simulation.getPopulationData(final_cancer);
    simulation.getPopulationData(final_tcells);
    simulation.getPopulationData(final_tregs);
    simulation.getPopulationData(final_mdscs);

    std::cout << "\nFinal Population Counts:" << std::endl;
    std::cout << "  Cancer cells: " << final_cancer.size() << std::endl;
    std::cout << "  T cells: " << final_tcells.size() << std::endl;
    std::cout << "  TRegs: " << final_tregs.size() << std::endl;
    std::cout << "  MDSCs: " << final_mdscs.size() << std::endl;

    // Count T cell states
    if (final_tcells.size() > 0) {
        int eff_count = 0, cyt_count = 0, supp_count = 0;
        for (unsigned int i = 0; i < final_tcells.size(); i++) {
            int state = final_tcells[i].getVariable<int>("cell_state");
            if (state == PDAC::T_CELL_EFF) eff_count++;
            else if (state == PDAC::T_CELL_CYT) cyt_count++;
            else if (state == PDAC::T_CELL_SUPP) supp_count++;
        }
        std::cout << "  T cell states - Effector: " << eff_count
                  << ", Cytotoxic: " << cyt_count
                  << ", Suppressed: " << supp_count << std::endl;
    }
    
    // ========== CLEANUP ==========
    PDAC::cleanup_pde_solver();
    
    std::cout << "\nSimulation finished successfully." << std::endl;
    
    return 0;
}