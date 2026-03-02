#include "flamegpu/flamegpu.h"
#include <iostream>
#include <memory>
#include <fstream>
#include <sstream>
#include <vector>
#include <iomanip>
#include <nvtx3/nvToolsExt.h>
#include <chrono>
#include <filesystem>

#include "../core/common.cuh"
#include "../core/layer_timing.h"
#include "../pde/pde_integration.cuh"
#include "initialization.cuh"
#include "gpu_param.h"
#include "../qsp/LymphCentral_wrapper.h"
#include "../core/model_functions.cuh"

// Exposed by qsp_integration.cu — true during Phase 3 pre-simulation
namespace PDAC {
extern bool is_presim_mode_active();
extern double get_last_pde_ms();
extern double get_last_qsp_ms();
}

// QSP CSV export step function (defined in qsp_integration.cu)
extern flamegpu::FLAMEGPU_STEP_FUNCTION_POINTER exportQSPData;

namespace PDAC {
    std::unique_ptr<flamegpu::ModelDescription> buildModel(
        int grid_x, int grid_y, int grid_z, float voxel_size,
        const PDAC::GPUParam& gpu_params);

    // void set_internal_params(flamegpu::ModelDescription& model, 
    //                          const LymphCentralWrapper& lymph);
}

// ============================================================================
// Output Directory Management
// ============================================================================

// Ensure output directories exist, creating them if necessary
void ensureOutputDirectories() {
    try {
        std::filesystem::create_directories("outputs/pde");
        std::filesystem::create_directories("outputs/abm");
    } catch (const std::exception& e) {
        std::cerr << "Warning: Could not create output directories: " << e.what() << std::endl;
    }
}

// ============================================================================
// Simulation Monitoring Functions
// ============================================================================

// Called manually from main() after presim completes — captures true day-0 PDE state.
void exportPDEData_step0(int grid_x, int grid_y, int grid_z) {
    ensureOutputDirectories();
    if (!PDAC::g_pde_solver) return;

    std::ostringstream filename;
    filename << "outputs/pde/pde_step_" << std::setw(6) << std::setfill('0') << 0 << ".csv";
    std::ofstream file(filename.str());
    file << "x,y,z,O2,IFN,IL2,IL10,TGFB,CCL2,ARGI,NO,IL12,VEGFA\n";

    const int total_voxels = grid_x * grid_y * grid_z;
    std::vector<std::vector<float>> all_concentrations(PDAC::NUM_SUBSTRATES);
    for (int s = 0; s < PDAC::NUM_SUBSTRATES; s++) {
        all_concentrations[s].resize(total_voxels);
        PDAC::g_pde_solver->get_concentrations(all_concentrations[s].data(), s);
    }

    for (int z = 0; z < grid_z; z++) {
        for (int y = 0; y < grid_y; y++) {
            for (int x = 0; x < grid_x; x++) {
                int idx = z * (grid_x * grid_y) + y * grid_x + x;
                file << x << "," << y << "," << z;
                for (int s = 0; s < PDAC::NUM_SUBSTRATES; s++)
                    file << "," << all_concentrations[s][idx];
                file << "\n";
            }
        }
    }
    file.close();
}

FLAMEGPU_STEP_FUNCTION(exportPDEData) {
    if (PDAC::is_presim_mode_active()) return;
    if (!PDAC::g_pde_solver) return;

    ensureOutputDirectories();

    const unsigned int main_step = FLAMEGPU->environment.getProperty<unsigned int>("main_sim_step");
    const int interval = FLAMEGPU->environment.getProperty<int>("interval_out");
    if (main_step % interval != 0) return;

    // Get grid dimensions
    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int grid_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");

    std::ostringstream filename;
    filename << "outputs/pde/pde_step_" << std::setw(6) << std::setfill('0') << main_step << ".csv";
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

// Called manually from main() after presim completes — captures true day-0 agent state.
void exportABMData_step0(flamegpu::CUDASimulation& sim, flamegpu::ModelDescription& model) {
    ensureOutputDirectories();

    std::ostringstream filename;
    filename << "outputs/abm/agents_step_" << std::setw(6) << std::setfill('0') << 0 << ".csv";
    std::ofstream file(filename.str());
    file << "agent_type,agent_id,x,y,z,cell_state,additional_info\n";

    // Cancer Cells
    {
        flamegpu::AgentVector cancer_pop(model.Agent(PDAC::AGENT_CANCER_CELL));
        sim.getPopulationData(cancer_pop);
        for (unsigned int i = 0; i < cancer_pop.size(); ++i) {
            unsigned int id = cancer_pop[i].getID();
            int x = cancer_pop[i].getVariable<int>("x");
            int y = cancer_pop[i].getVariable<int>("y");
            int z = cancer_pop[i].getVariable<int>("z");
            int state = cancer_pop[i].getVariable<int>("cell_state");
            int divideCD   = cancer_pop[i].getVariable<int>("divideCD");
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

    // T Cells
    {
        flamegpu::AgentVector tcell_pop(model.Agent(PDAC::AGENT_TCELL));
        sim.getPopulationData(tcell_pop);
        for (unsigned int i = 0; i < tcell_pop.size(); ++i) {
            unsigned int id = tcell_pop[i].getID();
            int x = tcell_pop[i].getVariable<int>("x");
            int y = tcell_pop[i].getVariable<int>("y");
            int z = tcell_pop[i].getVariable<int>("z");
            int state = tcell_pop[i].getVariable<int>("cell_state");
            int life  = tcell_pop[i].getVariable<int>("life");
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

    // TRegs
    {
        flamegpu::AgentVector treg_pop(model.Agent(PDAC::AGENT_TREG));
        sim.getPopulationData(treg_pop);
        for (unsigned int i = 0; i < treg_pop.size(); ++i) {
            unsigned int id = treg_pop[i].getID();
            int x = treg_pop[i].getVariable<int>("x");
            int y = treg_pop[i].getVariable<int>("y");
            int z = treg_pop[i].getVariable<int>("z");
            int state = treg_pop[i].getVariable<int>("cell_state");
            int life = treg_pop[i].getVariable<int>("life");
            std::string state_name;
            switch (state) {
                case 0: state_name = "TH"; break;
                case 1: state_name = "REGULATORY"; break;
                default: state_name = "UNKNOWN"; break;
            }
            file << "TREG," << id << "," << x << "," << y << "," << z << ","
                 << state_name << ",life=" << life << "\n";
        }
    }

    // MDSCs
    {
        flamegpu::AgentVector mdsc_pop(model.Agent(PDAC::AGENT_MDSC));
        sim.getPopulationData(mdsc_pop);
        for (unsigned int i = 0; i < mdsc_pop.size(); ++i) {
            unsigned int id = mdsc_pop[i].getID();
            int x = mdsc_pop[i].getVariable<int>("x");
            int y = mdsc_pop[i].getVariable<int>("y");
            int z = mdsc_pop[i].getVariable<int>("z");
            int life = mdsc_pop[i].getVariable<int>("life");
            file << "MDSC," << id << "," << x << "," << y << "," << z << ","
                 << "MDSC,life=" << life << "\n";
        }
    }
    // Macrophages
    {
        flamegpu::AgentVector mac_pop(model.Agent(PDAC::AGENT_MACROPHAGE));
        sim.getPopulationData(mac_pop);
        for (unsigned int i = 0; i < mac_pop.size(); ++i) {
            unsigned int id = mac_pop[i].getID();
            int x = mac_pop[i].getVariable<int>("x");
            int y = mac_pop[i].getVariable<int>("y");
            int z = mac_pop[i].getVariable<int>("z");
            int cell_state = mac_pop[i].getVariable<int>("cell_state");
            int life = mac_pop[i].getVariable<int>("life");
            file << "MAC," << id << "," << x << "," << y << "," << z << ","
                 << (cell_state == PDAC::MAC_M1 ? "M1" : "M2")
                 << ",life=" << life << "\n";
        }
    }
    // Fibroblasts
    {
        flamegpu::AgentVector fib_pop(model.Agent(PDAC::AGENT_FIBROBLAST));
        sim.getPopulationData(fib_pop);
        for (unsigned int i = 0; i < fib_pop.size(); ++i) {
            unsigned int id = fib_pop[i].getID();
            int x = fib_pop[i].getVariable<int>("x");
            int y = fib_pop[i].getVariable<int>("y");
            int z = fib_pop[i].getVariable<int>("z");
            int cell_state = fib_pop[i].getVariable<int>("cell_state");
            int life = fib_pop[i].getVariable<int>("life");
            file << "FIB," << id << "," << x << "," << y << "," << z << ","
                 << (cell_state == PDAC::FIB_CAF ? "CAF" : "NORMAL")
                 << ",life=" << life << "\n";
        }
    }

    // Vasculature — TIP and PHALANX only (matches HCC CPU output; STALK excluded)
    {
        flamegpu::AgentVector vas_pop(model.Agent(PDAC::AGENT_VASCULAR));
        sim.getPopulationData(vas_pop);
        for (unsigned int i = 0; i < vas_pop.size(); ++i) {
            int vas_state = vas_pop[i].getVariable<int>("cell_state");
            if (vas_state == 1) continue;  // Skip STALK
            unsigned int id = vas_pop[i].getID();
            int x = vas_pop[i].getVariable<int>("x");
            int y = vas_pop[i].getVariable<int>("y");
            int z = vas_pop[i].getVariable<int>("z");
            std::string state_name = (vas_state == 0) ? "TIP" : "PHALANX";
            file << "VAS," << id << "," << x << "," << y << "," << z << ","
                 << state_name
                 << ",life=" << 0 << "\n";
        }
    }
    file.close();
}

FLAMEGPU_STEP_FUNCTION(exportABMData) {
    if (PDAC::is_presim_mode_active()) return;

    ensureOutputDirectories();

    const unsigned int main_step = FLAMEGPU->environment.getProperty<unsigned int>("main_sim_step");
    const int interval = FLAMEGPU->environment.getProperty<int>("interval_out");
    if (main_step % interval != 0) return;

    std::ostringstream filename;
    filename << "outputs/abm/agents_step_" << std::setw(6) << std::setfill('0') << main_step << ".csv";
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
                int state = treg_pop[i].getVariable<int>("cell_state");
                int life = treg_pop[i].getVariable<int>("life");
                std::string state_name = (state == PDAC::TCD4_TH) ? "TH" : "REGULATORY";
                file << "TREG," << id << "," << x << "," << y << "," << z << ","
                     << state_name << ",life=" << life << "\n";
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
    // Macrophages
    {
        auto agent = FLAMEGPU->agent(PDAC::AGENT_MACROPHAGE);
        unsigned int count = agent.count();
        if (count > 0) {
            flamegpu::DeviceAgentVector mac_pop = agent.getPopulationData();
            for (unsigned int i = 0; i < count; ++i) {
                unsigned int id = mac_pop[i].getID();
                int x = mac_pop[i].getVariable<int>("x");
                int y = mac_pop[i].getVariable<int>("y");
                int z = mac_pop[i].getVariable<int>("z");
                int cell_state = mac_pop[i].getVariable<int>("cell_state");
                int life = mac_pop[i].getVariable<int>("life");

                file << "MAC," << id << "," << x << "," << y << "," << z << ","
                     << (cell_state == PDAC::MAC_M1 ? "M1" : "M2")
                     << ",life=" << life << "\n";
            }
        }
    }
    // Fibroblasts
    {
        auto agent = FLAMEGPU->agent(PDAC::AGENT_FIBROBLAST);
        unsigned int count = agent.count();
        if (count > 0) {
            flamegpu::DeviceAgentVector fib_pop = agent.getPopulationData();
            for (unsigned int i = 0; i < count; ++i) {
                unsigned int id = fib_pop[i].getID();
                int x = fib_pop[i].getVariable<int>("x");
                int y = fib_pop[i].getVariable<int>("y");
                int z = fib_pop[i].getVariable<int>("z");
                int cell_state = fib_pop[i].getVariable<int>("cell_state");
                int life = fib_pop[i].getVariable<int>("life");

                file << "FIB," << id << "," << x << "," << y << "," << z << ","
                     << (cell_state == PDAC::FIB_CAF ? "CAF" : "NORMAL")
                     << ",life=" << life << "\n";
            }
        }
    }

    // Vasculature — TIP and PHALANX only (matches HCC CPU output; STALK excluded)
    {
        auto agent = FLAMEGPU->agent(PDAC::AGENT_VASCULAR);
        unsigned int count = agent.count();
        if (count > 0) {
            flamegpu::DeviceAgentVector vas_pop = agent.getPopulationData();
            for (unsigned int i = 0; i < count; ++i) {
                int cell_state = vas_pop[i].getVariable<int>("cell_state");
                if (cell_state == 1) continue;  // Skip STALK
                unsigned int id = vas_pop[i].getID();
                int x = vas_pop[i].getVariable<int>("x");
                int y = vas_pop[i].getVariable<int>("y");
                int z = vas_pop[i].getVariable<int>("z");
                std::string state_name = (cell_state == 0) ? "TIP" : "PHALANX";

                file << "VAS," << id << "," << x << "," << y << "," << z << ","
                     << state_name
                     << ",life=" << 0 << "\n";
            }
        }
    }
    file.close();
}

FLAMEGPU_STEP_FUNCTION(stepCounter) {
    unsigned int step = FLAMEGPU->environment.getProperty<unsigned int>("current_step");
    FLAMEGPU->environment.setProperty<unsigned int>("current_step", step + 1);
    
    // Suppress output during Phase 3 pre-simulation warmup
    if (PDAC::is_presim_mode_active()) return;

    const unsigned int main_step = FLAMEGPU->environment.getProperty<unsigned int>("main_sim_step");

    // Compute treatment day from main_sim_step (0 = start of Phase 4)
    const float dt_abm  = FLAMEGPU->environment.getProperty<float>("PARAM_SEC_PER_SLICE");
    const float treat_day = static_cast<float>(main_step) * dt_abm / 86400.0f;

    // Agent counts
    const unsigned int cancer_count = FLAMEGPU->agent(PDAC::AGENT_CANCER_CELL).count();
    const unsigned int tcell_count  = FLAMEGPU->agent(PDAC::AGENT_TCELL).count();
    const unsigned int treg_count   = FLAMEGPU->agent(PDAC::AGENT_TREG).count();
    const unsigned int mdsc_count   = FLAMEGPU->agent(PDAC::AGENT_MDSC).count();
    const unsigned int mac_count   = FLAMEGPU->agent(PDAC::AGENT_MACROPHAGE).count();
    const unsigned int fib_count   = FLAMEGPU->agent(PDAC::AGENT_FIBROBLAST).count();
    const unsigned int vas_count   = FLAMEGPU->agent(PDAC::AGENT_VASCULAR).count();

    // QSP state (set by solve_qsp_step each step)
    const float tum_vol  = FLAMEGPU->environment.getProperty<float>("qsp_tum_vol");
    const float cc_tumor = FLAMEGPU->environment.getProperty<float>("qsp_cc_tumor");
    const float nivo     = FLAMEGPU->environment.getProperty<float>("qsp_nivo_tumor");
    const float cabo     = FLAMEGPU->environment.getProperty<float>("qsp_cabo_tumor");
    const float teff_t   = FLAMEGPU->environment.getProperty<float>("qsp_teff_tumor");
    const float treg_t   = FLAMEGPU->environment.getProperty<float>("qsp_treg_tumor");
    const float mdsc_t   = FLAMEGPU->environment.getProperty<float>("qsp_mdsc_tumor");

    std::cout << std::fixed << std::setprecision(2)
              << "[Day " << std::setw(7) << treat_day << "]" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "[ABM] CC=" << cancer_count 
              << "  TC=" << tcell_count
              << "  TR=" << treg_count
              << "  MD=" << mdsc_count
              << "  MAC=" << mac_count
              << "  FIB=" << fib_count
              << "  VAS=" << vas_count
              << std::endl;

    std::cout << std::scientific << std::setprecision(2) << "[QSP] vol=" << tum_vol
              << " cc=" << cc_tumor
              << " nivo=" << nivo
              << " cabo=" << cabo
              << " Teff=" << teff_t
              << " Treg=" << treg_t
              << " MDSC=" << mdsc_t
              << std::endl;

    FLAMEGPU->environment.setProperty<unsigned int>("main_sim_step", main_step + 1);
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
    auto start = std::chrono::high_resolution_clock::now();

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

    // ========== INITIALIZATION TIMING ==========
    std::ofstream init_file("outputs/init_timing.csv");
    init_file << "phase,ms\n";
    auto init_t0 = std::chrono::high_resolution_clock::now();
    auto init_lap = [&](const std::string& label) {
        auto init_t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(init_t1 - init_t0).count();
        init_file << label << "," << ms << "\n";
        init_file.flush();
        init_t0 = init_t1;
    };

    // ========== BUILD MODEL ==========
    std::cout << "Building FLAME GPU 2 model..." << std::endl;
    auto model = PDAC::buildModel(
        config.grid_x, config.grid_y, config.grid_z,
        config.voxel_size,
        gpu_params);

    // Store output interval in environment for step functions
    model->Environment().newProperty<int>("interval_out", config.interval_out);
    init_lap("build_model");

    // ========== INITIALIZE PDE SOLVER ==========
    std::cout << "Initializing PDE solver..." << std::endl;
    PDAC::initialize_pde_solver(
        config.grid_x, config.grid_y, config.grid_z,
        config.voxel_size, config.dt_abm, config.molecular_steps,
         gpu_params);

    // Store PDE device pointers in model environment
    PDAC::set_pde_pointers_in_environment(*model);
    init_lap("init_pde");

    // ========== GPU MEMORY QUERY (after PDE allocation) ==========
    size_t free_mem_1, total_mem_1;
    cudaMemGetInfo(&free_mem_1, &total_mem_1);
    size_t used_mem_1 = total_mem_1 - free_mem_1;
    std::cout << "[MEM] After PDE init: " << (used_mem_1 / (1024*1024)) << " MB used / "
              << (total_mem_1 / (1024*1024)) << " MB total" << std::endl;

    // Process internal parameters from env params and new QSP params
    // ========== INITIALIZE QSP SOLVER ==========
    PDAC::LymphCentralWrapper _lymph;
    _lymph.initialize(param_file);
    PDAC::set_internal_params(*model, _lymph);
    PDAC::set_lymph_pointer(&_lymph);  // Set global pointer for QSP host functions
    init_lap("init_qsp");

    // ========== ADD STEP FUNCTIONS ==========
    if (config.pde_out) {
        model->addStepFunction(exportPDEData);
    }
    if (config.abm_out) {
        model->addStepFunction(exportABMData);
    }
    model->addStepFunction(exportQSPData);
    model->addStepFunction(stepCounter);
    model->addExitCondition(checkSimulationEnd);

    // ========== ALLOCATE GPU MEMORY FOR EVENT COUNTERS ==========
    // Do this BEFORE creating CUDASimulation so environment properties are synced
    unsigned int* device_event_counters = nullptr;
    cudaMalloc(&device_event_counters, 5 * sizeof(unsigned int));
    cudaMemset(device_event_counters, 0, 5 * sizeof(unsigned int));

    // Store pointers to event counters in model environment (before CUDASimulation init)
    model->Environment().setProperty<uint64_t>("event_tcell_prolif_ptr",
        reinterpret_cast<uint64_t>(device_event_counters));
    model->Environment().setProperty<uint64_t>("event_tcell_recruit_ptr",
        reinterpret_cast<uint64_t>(device_event_counters + 1));
    model->Environment().setProperty<uint64_t>("event_th_prolif_ptr",
        reinterpret_cast<uint64_t>(device_event_counters + 2));
    model->Environment().setProperty<uint64_t>("event_th_recruit_ptr",
        reinterpret_cast<uint64_t>(device_event_counters + 3));
    model->Environment().setProperty<uint64_t>("event_treg_prolif_ptr",
        reinterpret_cast<uint64_t>(device_event_counters + 4));

    // ========== CREATE SIMULATION ==========
    // Increase CUDA per-thread stack size for complex kernels (default 1KB is too small
    // for cancer_cell_state_step with inlined Newton-Raphson double-precision math)
    cudaDeviceSetLimit(cudaLimitStackSize, 16384);  // 16KB per thread
    std::cout << "Creating CUDA simulation..." << std::endl;
    flamegpu::CUDASimulation simulation(*model);
    simulation.SimulationConfig().steps = config.steps;
    simulation.SimulationConfig().random_seed = config.random_seed;
    init_lap("cuda_sim_create");

    // ========== INITIALIZE AGENTS ==========
    if (config.init_method == 1) {
        std::cout << "Initializing agents from QSP steady-state (init_method=1)..." << std::endl;
        PDAC::initializeToQSP(simulation, *model, config, _lymph);
    } else {
        std::cout << "Initializing agents with default distribution (init_method=0)..." << std::endl;
        PDAC::initializeAllAgents(simulation, *model, config);
    }
    std::cout << "[DEBUG] Agent initialization complete" << std::endl;
    std::cout.flush();
    init_lap("init_agents");

    // ========== GPU MEMORY QUERY (after agent allocation) ==========
    size_t free_mem_2, total_mem_2;
    cudaMemGetInfo(&free_mem_2, &total_mem_2);
    size_t used_mem_2 = total_mem_2 - free_mem_2;
    std::cout << "[MEM] After agent init: " << (used_mem_2 / (1024*1024)) << " MB used / "
              << (total_mem_2 / (1024*1024)) << " MB total" << std::endl;

    // ========== PHASE 3: PRE-SIMULATION (QSP-seeded init only) ==========
    // Run ABM+QSP (no drugs) until QSP tumor volume reaches 1.0× target diameter.
    // This fills the gap between the 0.95× warmup and the treatment start.
    if (config.init_method == 1) {
        const double full_target_vol = _lymph.get_full_target_volume();
        double cur_vol = _lymph.get_tumor_volume();

        std::cout << "\n=== Phase 3: Pre-simulation (ABM+QSP, no drugs) ===" << std::endl;
        std::cout << "  Target volume (1.0x diam): " << full_target_vol << " cm^3" << std::endl;
        std::cout << "  Current QSP volume       : " << cur_vol         << " cm^3" << std::endl;

        _lymph.set_presimulation_mode(true);

        const unsigned int max_presim_steps = 100000;
        unsigned int presim_step = 0;

        while (cur_vol < full_target_vol && presim_step < max_presim_steps) {
            // std::cout << "[DEBUG] About to call simulation.step() for presim step " << presim_step << std::endl;
            // std::cout.flush();
            bool ok = simulation.step();
            // std::cout << "[DEBUG] Returned from simulation.step() successfully" << std::endl;
            // std::cout.flush();
            if (!ok) {
                std::cout << "  Pre-simulation: ABM terminated early (all cancer cells gone)" << std::endl;
                break;
            }
            cur_vol = _lymph.get_tumor_volume();
            presim_step++;

            if (presim_step % 50 == 0) {
                std::cout << "  Presim step " << presim_step
                          << ": QSP tum_vol=" << cur_vol
                          << " cm^3  (target=" << full_target_vol << ")" << std::endl;
            }
        }

        _lymph.set_presimulation_mode(false);

        std::cout << "  Pre-simulation complete: " << presim_step << " steps, "
                  << "QSP tum_vol=" << cur_vol << " cm^3" << std::endl;
        init_lap("presim");
    } else {
        init_lap("presim");  // Log presim time even if not run
    }
    init_file.close();

    // ========== EXPORT DAY-0 STATE (after presim, before first treatment step) ==========
    if (config.pde_out) exportPDEData_step0(config.grid_x, config.grid_y, config.grid_z);
    if (config.abm_out) exportABMData_step0(simulation, *model);

    // ========== RUN SIMULATION ==========
    std::cout << "\n=== Starting Simulation ===" << std::endl;

    // Open event output file for per-step event tracking
    std::ofstream event_file("outputs/event.csv");
    if (event_file.is_open()) {
        event_file << "Step,prolif.CD8.cytotoxic,recruit.CD8.effector,prolif.Th.default,recruit.Th.default,prolif.Treg.default\n";
    }

    // Open timing output file for per-step timing CSV
    std::ofstream timing_file("outputs/timing.csv");
    timing_file << "step,total_ms,pde_ms,qsp_ms,abm_ms\n";

    // Open per-layer timing CSV (long format: step, layer_name, time_ms)
    std::ofstream layer_file("outputs/layer_timing.csv");
    layer_file << "step,layer,ms\n";

    // Manual stepping loop with NVTX markers for profiling
    const unsigned int total_steps = simulation.SimulationConfig().steps;
    for (unsigned int i = 0; i < total_steps; i++) {
        nvtxRangePush("ABM Step");
        auto step_t0 = std::chrono::high_resolution_clock::now();
        bool continue_sim = simulation.step();
        auto step_t1 = std::chrono::high_resolution_clock::now();
        nvtxRangePop();

        // Capture high-level timing
        double step_ms = std::chrono::duration<double, std::milli>(step_t1 - step_t0).count();
        double pde_ms = PDAC::get_last_pde_ms();
        double qsp_ms = PDAC::get_last_qsp_ms();
        double abm_ms = step_ms - pde_ms - qsp_ms;
        timing_file << i << "," << step_ms << "," << pde_ms << "," << qsp_ms << "," << abm_ms << "\n";
        timing_file.flush();

        // Write per-layer timings collected by checkpoint host functions
        {
            // GPU memory snapshot after this step
            size_t free_m = 0, total_m = 0;
            cudaMemGetInfo(&free_m, &total_m);
            int gpu_used_mb = static_cast<int>((total_m - free_m) / (1024 * 1024));

            layer_file << i << ",gpu_mem_mb," << gpu_used_mb << "\n";
            layer_file << i << ",total_ms," << step_ms << "\n";
            layer_file << i << ",pde_solve_ms," << pde_ms << "\n";
            layer_file << i << ",qsp_solve_ms," << qsp_ms << "\n";

            // Checkpoint-recorded phases (filled by timing_after_* host functions)
            for (const auto& lt : PDAC::g_layer_timings) {
                layer_file << i << "," << lt.name << "," << lt.ms << "\n";
            }
            PDAC::g_layer_timings.clear();

            layer_file.flush();
        }

        // Read event counts from GPU and output
        if (event_file.is_open()) {
            unsigned int host_events[5];
            cudaMemcpy(host_events, device_event_counters, 5 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

            // Output: prolif.CD8.cytotoxic, recruit.CD8.effector, prolif.Th.default, recruit.Th.default, prolif.Treg.default
            event_file << i << "," << host_events[0] << "," << host_events[1] << ","
                       << host_events[2] << "," << host_events[3] << "," << host_events[4] << "\n";
            event_file.flush();

            // Reset counters for next step
            cudaMemset(device_event_counters, 0, 5 * sizeof(unsigned int));
        }

        if (!continue_sim) {
            std::cout << "Simulation terminated early at step " << i << std::endl;
            break;
        }
    }

    if (event_file.is_open()) {
        event_file.close();
        std::cout << "Created: outputs/event.csv" << std::endl;
    }

    if (timing_file.is_open()) {
        timing_file.close();
        std::cout << "Created: outputs/timing.csv" << std::endl;
    }

    if (layer_file.is_open()) {
        layer_file.close();
        std::cout << "Created: outputs/layer_timing.csv" << std::endl;
    }

    // ========== REPORT RESULTS ==========
    std::cout << "\n=== Simulation Complete ===" << std::endl;
    
    flamegpu::AgentVector final_cancer(model->Agent(PDAC::AGENT_CANCER_CELL));
    flamegpu::AgentVector final_tcells(model->Agent(PDAC::AGENT_TCELL));
    flamegpu::AgentVector final_tregs(model->Agent(PDAC::AGENT_TREG));
    flamegpu::AgentVector final_mdscs(model->Agent(PDAC::AGENT_MDSC));
    flamegpu::AgentVector final_macs(model->Agent(PDAC::AGENT_MACROPHAGE));
    flamegpu::AgentVector final_fibs(model->Agent(PDAC::AGENT_FIBROBLAST));

    simulation.getPopulationData(final_cancer);
    simulation.getPopulationData(final_tcells);
    simulation.getPopulationData(final_tregs);
    simulation.getPopulationData(final_mdscs);
    simulation.getPopulationData(final_macs);
    simulation.getPopulationData(final_fibs);

    std::cout << "\nFinal Population Counts:" << std::endl;
    std::cout << "  Cancer cells: " << final_cancer.size() << std::endl;
    std::cout << "  T cells: " << final_tcells.size() << std::endl;
    std::cout << "  TRegs: " << final_tregs.size() << std::endl;
    std::cout << "  MDSCs: " << final_mdscs.size() << std::endl;
    std::cout << "  Macrophages: " << final_macs.size() << std::endl;
    std::cout << "  Fibroblasts: " << final_fibs.size() << std::endl;

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

    // Free GPU event counter memory
    if (device_event_counters != nullptr) {
        cudaFree(device_event_counters);
    }
    
    std::cout << "\nSimulation finished successfully." << std::endl;

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    return 0;
}