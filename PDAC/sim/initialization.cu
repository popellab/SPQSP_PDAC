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
    , init_method(1)
    , cluster_radius(5)
    , num_tcells(50)
    , num_tregs(10)
    , num_mdscs(5)
    , num_macrophages(10)
    , num_fibroblasts(10)
    , vascular_mode("random")
    , vascular_xml_file("")
    , abm_out(true)
    , pde_out(true)
    , interval_out(1)
{
}

void SimulationConfig::parseCommandLine(int argc, const char** argv, const PDAC::GPUParam gpu_params) {
    // First, get defaults from XML
    grid_x     = gpu_params.getInt(PARAM_X_SIZE);
    grid_y     = gpu_params.getInt(PARAM_Y_SIZE);
    grid_z     = gpu_params.getInt(PARAM_Z_SIZE);
    voxel_size = gpu_params.getInt(PARAM_VOXEL_SIZE);

    // Then, parse command line to allow overrides
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if ((arg == "--param-file" || arg == "-p") && i + 1 < argc) {
            ++i;  // already handled before parseCommandLine
        } else if ((arg == "--initialization" || arg == "-i") && i + 1 < argc) {
            init_method = std::atoi(argv[++i]);
        } else if ((arg == "--grid-size" || arg == "-g") && i + 1 < argc) {
            int size = std::atoi(argv[++i]);
            grid_x = size;
            grid_y = size;
            grid_z = size;
        } else if ((arg == "--steps" || arg == "-s") && i + 1 < argc) {
            steps = std::atoi(argv[++i]);
        } else if ((arg == "--out_abm" || arg == "-oa") && i + 1 < argc) {
            abm_out =  std::atoi(argv[++i]);
        } else if ((arg == "--out_pde" || arg == "-op") && i + 1 < argc) {
            pde_out =  std::atoi(argv[++i]);
        } else if ((arg == "--out_int" || arg == "-oi") && i + 1 < argc) {
            interval_out =  std::atoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            random_seed = std::atoi(argv[++i]);
        } else if ((arg == "--vascular-mode" || arg == "-vm") && i + 1 < argc) {
            vascular_mode = argv[++i];
        } else if ((arg == "--vascular-xml" || arg == "-vx") && i + 1 < argc) {
            vascular_xml_file = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "\nOptions:\n"
                      << "  -p, --param-file FILE    Path to parameter XML file [default: param_all_test.xml]\n"
                      << "  -i, --initialization N   initialization type: 0=random, 1=QSP-seeded [default: 1]\n"
                      << "  -g, --grid-size N        Grid dimensions NxNxN [default: from XML]\n"
                      << "  -s, --steps N            Number of simulation steps [default: 200]\n"
                      << "  -oa, --out_abm Bool      Output ABM at interval frequency [default: true]\n"
                      << "  -op, --out_pde Bool      Output PDE at interval frequency [default: true]\n"
                      << "  -oi, --out_int N         Output interval frequency [default: 1]\n"
                      << "  --seed N                 Random seed [default: 12345]\n"
                      << "  -vm, --vascular-mode STR Vasculature initialization: random, xml, test [default: random]\n"
                      << "  -vx, --vascular-xml FILE XML file for vasculature (when mode=xml)\n"
                      << "  -h, --help               Show this help\n";
            exit(0);
        }
    }

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

                // Randomize division cooldown from uniform distribution [0.0×interval, 1.0×interval]
                float base_interval = is_stem ? stem_div_interval : progenitor_div_interval;
                float random_factor = (static_cast<float>(rand()) / RAND_MAX);  // Range: [0.5, 1.5]
                int div_cd = static_cast<int>((base_interval * random_factor) + 0.5);

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

                count++;
            }
        }
    }
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
    float IL2_release_time, int tcell_div_interval)
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

        float random_factor = (static_cast<float>(rand()) / RAND_MAX);  // Range: [0.0, 1.0]
        int div_cd = static_cast<int>((tcell_div_interval * random_factor) + 0.5);

        // Basic identity and state
        agent.setVariable<int>("x", x);
        agent.setVariable<int>("y", y);
        agent.setVariable<int>("z", z);
        agent.setVariable<int>("cell_state", T_CELL_EFF);
        agent.setVariable<int>("divide_flag", 0);
        agent.setVariable<int>("divide_cd", div_cd);
        agent.setVariable<int>("divide_limit", div_limit);
        
        // Chemical production/exposure
        agent.setVariable<float>("IL2_release_remain", IL2_release_time);
        
        // Lifecycle
        agent.setVariable<int>("life", life);

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
    float treg_life_mean, int div_limit, int treg_div_interval)
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

        float random_factor = (static_cast<float>(rand()) / RAND_MAX);  // Range: [0.0, 1.0]
        int div_cd = static_cast<int>((treg_div_interval * random_factor) + 0.5);

        // Basic identity
        agent.setVariable<int>("x", x);
        agent.setVariable<int>("y", y);
        agent.setVariable<int>("z", z);
        agent.setVariable<int>("divide_flag", 0);
        agent.setVariable<int>("divide_cd", div_cd);
        agent.setVariable<int>("divide_limit", div_limit);

        // Lifecycle
        agent.setVariable<int>("life", life);

        // State
        agent.setVariable<int>("cell_state", TCD4_TH); // initialize as T-Helper cells

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
        
        // Lifecycle
        agent.setVariable<int>("life", life);
        
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
// Macrophage Initialization
// ============================================================================
void initializeMacrophages(
    flamegpu::AgentVector& mac_agents,
    int grid_x, int grid_y, int grid_z,
    int tumor_radius, int num_macrophages,
    float mac_life_mean)
{
    const int cx = grid_x / 2;
    const int cy = grid_y / 2;
    const int cz = grid_z / 2;

    const float inner_radius = tumor_radius + 1;
    const float outer_radius = tumor_radius + 5;

    int placed = 0;
    int attempts = 0;
    const int max_attempts = num_macrophages * 100;

    while (placed < num_macrophages && attempts < max_attempts) {
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
        int life = static_cast<int>(mac_life_mean * std::log(1.0f / (rnd + 0.0001f)) + 0.5f);
        if (life < 1) life = 1;

        mac_agents.push_back();
        flamegpu::AgentVector::Agent agent = mac_agents.back();

        // Basic identity
        agent.setVariable<int>("x", x);
        agent.setVariable<int>("y", y);
        agent.setVariable<int>("z", z);

        // Macrophage state (1=M2 by default)
        agent.setVariable<int>("mac_state", 1);

        // Lifecycle
        agent.setVariable<int>("life", life);
        agent.setVariable<int>("dead", 0);

        // Movement state
        agent.setVariable<float>("move_direction_x", 0.0f);
        agent.setVariable<float>("move_direction_y", 0.0f);
        agent.setVariable<float>("move_direction_z", 0.0f);
        agent.setVariable<int>("tumble", 0);
        agent.setVariable<int>("moves_remaining", 0);

        // Initialize neighbor counts
        agent.setVariable<int>("neighbor_cancer_count", 0);

        placed++;
    }

    std::cout << "Initialized " << placed << " Macrophages (M2) around tumor margin" << std::endl;
}

// ============================================================================
// Vascular Cell Initialization
// ============================================================================

// Helper to set all vascular cell variables
inline void setVascularCellVariables(
    flamegpu::AgentVector::Agent& agent,
    int x, int y, int z,
    int state,
    float move_dir_x = 1.0f,
    float move_dir_y = 0.0f,
    float move_dir_z = 0.0f,
    unsigned int tip_id = 0)
{
    agent.setVariable<int>("x", x);
    agent.setVariable<int>("y", y);
    agent.setVariable<int>("z", z);
    agent.setVariable<int>("vascular_state", state);
    agent.setVariable<float>("move_direction_x", move_dir_x);
    agent.setVariable<float>("move_direction_y", move_dir_y);
    agent.setVariable<float>("move_direction_z", move_dir_z);
    agent.setVariable<int>("tumble", 1);  // Start in tumble phase
    agent.setVariable<int>("intent_action", 0);  // INTENT_NONE
    agent.setVariable<int>("target_x", -1);
    agent.setVariable<int>("target_y", -1);
    agent.setVariable<int>("target_z", -1);
    agent.setVariable<unsigned int>("tip_id", tip_id);
    agent.setVariable<int>("mature_to_phalanx", 0);
    agent.setVariable<int>("branch", 0);
}

void initializeVascularCellsRandom(
    flamegpu::AgentVector& vascular_agents,
    int grid_x, int grid_y, int grid_z,
    int tumor_radius,
    int num_segments)
{
    const int center_x = grid_x / 2;
    const int center_y = grid_y / 2;
    const int center_z = grid_z / 2;
    const double radius = tumor_radius;

    // Random number generator
    std::srand(12345);  // Use fixed seed for reproducibility
    auto rand_unif = []() { return static_cast<double>(std::rand()) / RAND_MAX; };

    int total_vessels = 0;

    for (int seg = 0; seg < num_segments; seg++) {
        int current_x, current_y, current_z;
        double dx = 0, dy = 0, dz = 0;
        int target_x, target_z;

        // Random starting position coefficients
        double z_coeff = 0.90 + (rand_unif() * 0.05);
        double x_coeff = 0.90 + (rand_unif() * 0.05);

        // Determine starting edge and direction based on segment number
        switch (seg % 4) {
            case 0:  // x=0 to x=xmax
                current_x = 0;
                current_y = center_y;
                current_z = static_cast<int>(z_coeff * (grid_z - 1));
                target_x = grid_x - 1;
                target_z = current_z;
                dx = 1;
                break;

            case 1:  // z=0 to z=zmax
                current_x = static_cast<int>(x_coeff * (grid_x - 1));
                current_y = center_y;
                current_z = 0;
                target_x = current_x;
                target_z = grid_z - 1;
                dz = 1;
                break;

            case 2:  // x=xmax to x=0
                current_x = grid_x - 1;
                current_y = center_y;
                current_z = static_cast<int>(z_coeff * (grid_z - 1));
                target_x = 0;
                target_z = current_z;
                dx = -1;
                break;

            case 3:  // z=zmax to z=0
                current_x = static_cast<int>(x_coeff * (grid_x - 1));
                current_y = center_y;
                current_z = grid_z - 1;
                target_x = current_x;
                target_z = 0;
                dz = -1;
                break;
        }

        // Add starting position if outside tumor
        double d_x_center = current_x - center_x;
        double d_y_center = current_y - center_y;
        double d_z_center = current_z - center_z;
        double dist_sq = d_x_center * d_x_center + d_y_center * d_y_center + d_z_center * d_z_center;

        if (dist_sq > radius * radius) {
            vascular_agents.push_back();
            flamegpu::AgentVector::Agent agent = vascular_agents.back();
            setVascularCellVariables(agent, current_x, current_y, current_z,
                                   2, 1.0f, 0.0f, 0.0f, seg);  // VAS_PHALANX, tip_id=seg
            total_vessels++;
        }

        // Random walk to target
        bool reached_end = false;
        int segment_length = 0;
        int max_length = 2 * std::max(grid_x, grid_z);

        while (!reached_end && segment_length < max_length) {
            // Directional persistence (80% keep direction, 20% change)
            double persistence = 0.2;
            if (rand_unif() > persistence) {
                if (seg % 4 == 0) {  // Moving in +x
                    dx = 1;
                    dz = (rand_unif() > 0.5) ? 1 : -1;
                }
                else if (seg % 4 == 1) {  // Moving in +z
                    dx = (rand_unif() > 0.5) ? 1 : -1;
                    dz = 1;
                }
                else if (seg % 4 == 2) {  // Moving in -x
                    dx = -1;
                    dz = (rand_unif() > 0.5) ? 1 : -1;
                }
                else {  // Moving in -z
                    dx = (rand_unif() > 0.5) ? 1 : -1;
                    dz = -1;
                }

                // Y direction changes
                double r = rand_unif();
                if (r < 0.333) {
                    dy = 1;
                } else if (r < 0.667) {
                    dy = -1;
                } else {
                    dy = 0;
                }
            }

            // Calculate next position
            int next_x = current_x + static_cast<int>(std::round(dx));
            int next_y = current_y + static_cast<int>(std::round(dy));
            int next_z = current_z + static_cast<int>(std::round(dz));

            // Boundary checking with reflection
            if (next_x < 0) {
                next_x = 0;
                dx = std::abs(dx);
            }
            if (next_x >= grid_x) {
                next_x = grid_x - 1;
                dx = -std::abs(dx);
            }

            if (next_y < 0) {
                next_y = 0;
                dy = 0;
            }
            if (next_y >= grid_y) {
                next_y = grid_y - 1;
                dy = 0;
            }

            // Keep z near target for x-moving segments
            if (seg % 4 == 0 || seg % 4 == 2) {
                double target_z_val = z_coeff * grid_z;
                if (std::abs(next_z - target_z_val) > 2) {
                    next_z = current_z + (next_z > target_z_val ? -1 : 1);
                }
            } else {
                if (next_z < 0) {
                    next_z = 0;
                    dz = std::abs(dz);
                }
                if (next_z >= grid_z) {
                    next_z = grid_z - 1;
                    dz = -std::abs(dz);
                }
            }

            // Keep x near target for z-moving segments
            if (seg % 4 == 1 || seg % 4 == 3) {
                double target_x_val = x_coeff * grid_x;
                if (std::abs(next_x - target_x_val) > 2) {
                    next_x = current_x + (next_x > target_x_val ? -1 : 1);
                }
            }

            // Update position
            current_x = next_x;
            current_y = next_y;
            current_z = next_z;

            // Check if reached target
            if ((seg % 4 == 0 && current_x >= target_x) ||
                (seg % 4 == 1 && current_z >= target_z) ||
                (seg % 4 == 2 && current_x <= target_x) ||
                (seg % 4 == 3 && current_z <= target_z)) {
                reached_end = true;
            }

            // Check distance from tumor center
            d_x_center = current_x - center_x;
            d_y_center = current_y - center_y;
            d_z_center = current_z - center_z;
            dist_sq = d_x_center * d_x_center + d_y_center * d_y_center + d_z_center * d_z_center;

            // Only add if outside tumor
            if (dist_sq > radius * radius) {
                vascular_agents.push_back();
                flamegpu::AgentVector::Agent agent = vascular_agents.back();
                setVascularCellVariables(agent, current_x, current_y, current_z,
                                       2, 1.0f, 0.0f, 0.0f, seg);  // VAS_PHALANX, tip_id=seg
                total_vessels++;
            }

            segment_length++;
        }
    }

    std::cout << "Initialized " << total_vessels << " vascular cells (random walk, "
              << num_segments << " segments)" << std::endl;
}

void initializeVascularCellsTest(
    flamegpu::AgentVector& vascular_agents,
    int grid_x, int grid_y, int grid_z)
{
    // Add 5 test vessels at tumor center (vertical column)
    const int cx = grid_x / 2;
    const int cy = grid_y / 2;
    const int cz = grid_z / 2;

    // Create 2 phalanx and 3 tip cells for testing
    for (int i = 0; i < 5; i++) {
        vascular_agents.push_back();
        flamegpu::AgentVector::Agent agent = vascular_agents.back();

        // Make first 2 phalanx, last 3 tip cells
        int state = (i < 2) ? 2 : 0;  // 2=PHALANX, 0=TIP
        setVascularCellVariables(agent,
                               cx, cy, cz + (i - 2) * 2,  // Vertical spacing
                               state, 1.0f, 0.0f, 0.0f, i);  // Start moving +x, tip_id=i
    }

    std::cout << "Initialized 5 test vascular cells (2 phalanx, 3 tip)" << std::endl;
}

// ============================================================================
// QSP Probability-Based Immune Cell Initialization
// ============================================================================

// Build flat occupancy grid (z-major, matching PDE voxel convention) from an
// existing AgentVector.  Agents at (x,y,z) mark index z*gx*gy + y*gx + x.
static void buildOccupancyGrid(
    const flamegpu::AgentVector& agents,
    std::vector<std::vector<int>>& occupied,
    int grid_x, int grid_y)
{
    for (unsigned int i = 0; i < agents.size(); i++) {
        int x = agents[i].getVariable<int>("x");
        int y = agents[i].getVariable<int>("y");
        int z = agents[i].getVariable<int>("z");
        occupied[z * grid_x * grid_y + y * grid_x + x][0] = 1;
    }
}

void initializeCancerCellsRandom(
    flamegpu::AgentVector& cancer_agents,
    int grid_x, int grid_y, int grid_z,
    int cluster_radius,
    float stem_div_interval,
    float progenitor_div_interval,
    int progenitor_div_max,
    std::vector<double> celltype_cdf)
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

                // Sample from CDF to determine Stem, Prog, or Sen
                double p = static_cast<float>(rand()) / RAND_MAX;
                int i = std::lower_bound(celltype_cdf.begin(), celltype_cdf.end(), p) - celltype_cdf.begin();
                int cell_state;
                int div_cd = 0;
                int div = 0;
                int divide_flag = 0;
                int is_stem = 0;
                if (i < progenitor_div_max + 1){
                    cancer_agents.push_back();
                    flamegpu::AgentVector::Agent agent = cancer_agents.back();
                    if (i==0) {
                        cell_state = CANCER_STEM;

                        float random_factor = (static_cast<float>(rand()) / RAND_MAX);
                        div_cd = static_cast<int>((stem_div_interval * random_factor) + 0.5);

                        divide_flag = 1;

                        is_stem = 1;

                    } else if (i == progenitor_div_max + 1){
                        cell_state = CANCER_SENESCENT;

                    } else {
                        cell_state = CANCER_PROGENITOR;
                        div = progenitor_div_max + 1 - i;

                        float random_factor = (static_cast<float>(rand()) / RAND_MAX);
                        div_cd = static_cast<int>((progenitor_div_interval * random_factor) + 0.5);

                        divide_flag = 1;
                    }

                    // Basic identity and state
                    const int id = agent.getID();
                    agent.setVariable<int>("x", x);
                    agent.setVariable<int>("y", y);
                    agent.setVariable<int>("z", z);
                    agent.setVariable<int>("cell_state", cell_state);
                    agent.setVariable<int>("divideCD", div_cd);
                    agent.setVariable<int>("divideFlag", divide_flag);
                    agent.setVariable<int>("divideCountRemaining", div);
                    agent.setVariable<unsigned int>("stemID", is_stem ? id : 0);

                    count++;
                }
            }
        }
    }
    std::cout << "Initialized " << (count - 1) << " cancer cells in cluster" << std::endl;
}

// Place T-helper cells (TCD4_TH) into the AGENT_TREG vector, testing
// probability p_th per empty voxel.
void initializeTHCellsFromQSP(
    flamegpu::AgentVector& treg_agents,
    int grid_x, int grid_y, int grid_z,
    double p_th,
    std::vector<std::vector<int>>& occupied,
    float life_mean, int div_limit, int div_interval)
{
    int placed = 0;
    for (int z = 0; z < grid_z; z++) {
        for (int y = 0; y < grid_y; y++) {
            for (int x = 0; x < grid_x; x++) {
                // const int idx = z * grid_x * grid_y + y * grid_x + x;

                const float rnd = static_cast<float>(rand()) / RAND_MAX;
                if (rnd >= static_cast<float>(p_th)) continue;

                float life_rnd = static_cast<float>(rand()) / RAND_MAX;
                int life = static_cast<int>(life_mean * std::log(1.0f / (life_rnd + 1e-4f)) + 0.5f);
                if (life < 1) life = 1;

                float div_rnd = static_cast<float>(rand()) / RAND_MAX;
                int div_cd = static_cast<int>(div_interval * div_rnd + 0.5f);

                treg_agents.push_back();
                flamegpu::AgentVector::Agent agent = treg_agents.back();
                agent.setVariable<int>("x", x);
                agent.setVariable<int>("y", y);
                agent.setVariable<int>("z", z);
                agent.setVariable<int>("cell_state", TCD4_TH);
                agent.setVariable<int>("divide_flag", 0);
                agent.setVariable<int>("divide_cd", div_cd);
                agent.setVariable<int>("divide_limit", div_limit);
                agent.setVariable<int>("life", life);

                agent.setVariable<float>("TGFB_release_remain", 0.0f);

                // Initialize molecular state
                agent.setVariable<float>("PDL1_syn", 0.0f);
                agent.setVariable<float>("CTLA4", 0.0f);
                agent.setVariable<float>("IL2_exposure", 0.0f);

                // Initialize neighbor counts
                agent.setVariable<int>("neighbor_Tcell_count", 0);
                agent.setVariable<int>("neighbor_Treg_count", 0);
                agent.setVariable<int>("neighbor_cancer_count", 0);
                agent.setVariable<int>("neighbor_all_count", 0);
                agent.setVariable<int>("found_progenitor", 0);
                agent.setVariable<unsigned int>("available_neighbors", 0u);

                // Initialize life/death
                agent.setVariable<int>("dead", 0);

                // Initialize intent
                agent.setVariable<int>("intent_action", 0);  // INTENT_NONE
                agent.setVariable<int>("target_x", -1);
                agent.setVariable<int>("target_y", -1);
                agent.setVariable<int>("target_z", -1);

                placed++;
            }
        }
    }
    std::cout << "  Placed " << placed << " TH cells (probability-based QSP)" << std::endl;
}

// Place regulatory T-cells (TCD4_TREG) into the AGENT_TREG vector, testing
// probability p_treg per empty voxel.
void initializeTRegCellsFromQSP(
    flamegpu::AgentVector& treg_agents,
    int grid_x, int grid_y, int grid_z,
    double p_treg,
    std::vector<std::vector<int>>& occupied,
    float life_mean, int div_limit, int div_interval)
{
    int placed = 0;
    for (int z = 0; z < grid_z; z++) {
        for (int y = 0; y < grid_y; y++) {
            for (int x = 0; x < grid_x; x++) {
                // const int idx = z * grid_x * grid_y + y * grid_x + x;

                const float rnd = static_cast<float>(rand()) / RAND_MAX;
                if (rnd >= static_cast<float>(p_treg)) continue;

                float life_rnd = static_cast<float>(rand()) / RAND_MAX;
                int life = static_cast<int>(life_mean * std::log(1.0f / (life_rnd + 1e-4f)) + 0.5f);
                if (life < 1) life = 1;

                float div_rnd = static_cast<float>(rand()) / RAND_MAX;
                int div_cd = static_cast<int>(div_interval * div_rnd + 0.5f);

                treg_agents.push_back();
                flamegpu::AgentVector::Agent agent = treg_agents.back();
                agent.setVariable<int>("x", x);
                agent.setVariable<int>("y", y);
                agent.setVariable<int>("z", z);
                agent.setVariable<int>("cell_state", TCD4_TREG);
                agent.setVariable<int>("divide_flag", 0);
                agent.setVariable<int>("divide_cd", div_cd);
                agent.setVariable<int>("divide_limit", div_limit);
                agent.setVariable<int>("life", life);

                placed++;
            }
        }
    }
    std::cout << "  Placed " << placed << " TReg cells (probability-based QSP)" << std::endl;
}

// Place MDSCs into the AGENT_MDSC vector, testing probability p_mdsc per
// empty voxel.
void initializeMDSCsFromQSP(
    flamegpu::AgentVector& mdsc_agents,
    int grid_x, int grid_y, int grid_z,
    double p_mdsc,
    std::vector<std::vector<int>>& occupied,
    float life_mean)
{
    int placed = 0;
    for (int z = 0; z < grid_z; z++) {
        for (int y = 0; y < grid_y; y++) {
            for (int x = 0; x < grid_x; x++) {
                // const int idx = z * grid_x * grid_y + y * grid_x + x;

                const float rnd = static_cast<float>(rand()) / RAND_MAX;
                if (rnd >= static_cast<float>(p_mdsc)) continue;

                float life_rnd = static_cast<float>(rand()) / RAND_MAX;
                int life = static_cast<int>(life_mean * std::log(1.0f / (life_rnd + 1e-4f)) + 0.5f);
                if (life < 1) life = 1;

                mdsc_agents.push_back();
                flamegpu::AgentVector::Agent agent = mdsc_agents.back();
                agent.setVariable<int>("x", x);
                agent.setVariable<int>("y", y);
                agent.setVariable<int>("z", z);
                agent.setVariable<int>("life", life);
                agent.setVariable<int>("intent_action", INTENT_NONE);
                agent.setVariable<int>("target_x", -1);
                agent.setVariable<int>("target_y", -1);
                agent.setVariable<int>("target_z", -1);

                placed++;
            }
        }
    }
    std::cout << "  Placed " << placed << " MDSCs (probability-based QSP)" << std::endl;
}

void initializeMacsFromQSP(
    flamegpu::AgentVector& mac_agents,
    int grid_x, int grid_y, int grid_z,
    double p_mac,
    std::vector<std::vector<int>>& occupied,
    float life_mean)
{
    int placed = 0;
    for (int z = 0; z < grid_z; z++) {
        for (int y = 0; y < grid_y; y++) {
            for (int x = 0; x < grid_x; x++) {
                // const int idx = z * grid_x * grid_y + y * grid_x + x;

                const float rnd = static_cast<float>(rand()) / RAND_MAX;
                if (rnd >= static_cast<float>(p_mac)) continue;

                float life_rnd = static_cast<float>(rand()) / RAND_MAX;
                int life = static_cast<int>(life_mean * std::log(1.0f / (life_rnd + 1e-4f)) + 0.5f);
                if (life < 1) life = 1;

                mac_agents.push_back();
                flamegpu::AgentVector::Agent agent = mac_agents.back();
                agent.setVariable<int>("x", x);
                agent.setVariable<int>("y", y);
                agent.setVariable<int>("z", z);
                agent.setVariable<int>("life", life);
                agent.setVariable<int>("mac_state", MAC_M2);

                placed++;
            }
        }
    }
    std::cout << "  Placed " << placed << " Macrophages (probability-based QSP)" << std::endl;
}

// ============================================================================
// Fibroblast Initialization
// ============================================================================

void initializeFibroblasts(
    flamegpu::AgentVector& fib_agents,
    int grid_x, int grid_y, int grid_z,
    int tumor_radius, int num_fibroblasts,
    float fib_life_mean)
{
    const int cx = grid_x / 2;
    const int cy = grid_y / 2;
    const int cz = grid_z / 2;

    const float inner_radius = tumor_radius + 1;
    const float outer_radius = tumor_radius + 5;

    int placed = 0;
    int attempts = 0;
    const int max_attempts = num_fibroblasts * 100;

    while (placed < num_fibroblasts && attempts < max_attempts) {
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
        int life = static_cast<int>(fib_life_mean * std::log(1.0f / (rnd + 0.0001f)) + 0.5f);
        if (life < 1) life = 1;

        fib_agents.push_back();
        flamegpu::AgentVector::Agent agent = fib_agents.back();
        agent.setVariable<int>("x", x);
        agent.setVariable<int>("y", y);
        agent.setVariable<int>("z", z);
        agent.setVariable<int>("fib_state", FIB_NORMAL);
        agent.setVariable<int>("life", life);

        placed++;
    }

    std::cout << "Initialized " << placed << " Fibroblasts (Normal) around tumor margin" << std::endl;
}

// Helper: find a free adjacent voxel for chain extension
// Returns true and sets nx,ny,nz if found; returns false otherwise
static bool findFreeAdjacent(int x, int y, int z, int grid_x, int grid_y, int grid_z,
                              std::vector<std::vector<int>>& occupied,
                              int& nx, int& ny, int& nz)
{
    // Try all 6 face-adjacent neighbors in random order
    int dx[6] = {1,-1,0,0,0,0};
    int dy[6] = {0,0,1,-1,0,0};
    int dz[6] = {0,0,0,0,1,-1};
    // Shuffle
    for (int i = 5; i > 0; i--) {
        int j = rand() % (i + 1);
        std::swap(dx[i], dx[j]);
        std::swap(dy[i], dy[j]);
        std::swap(dz[i], dz[j]);
    }
    for (int i = 0; i < 6; i++) {
        int cx = x + dx[i];
        int cy = y + dy[i];
        int cz = z + dz[i];
        if (cx < 0 || cx >= grid_x || cy < 0 || cy >= grid_y || cz < 0 || cz >= grid_z) continue;
        int idx = cx + cy * grid_x + cz * grid_x * grid_y;
        if (occupied[idx][0] == 0) {
            nx = cx; ny = cy; nz = cz;
            return true;
        }
    }
    return false;
}

void initializeFibroblastsFromQSP(
    flamegpu::AgentVector& fib_agents,
    int grid_x, int grid_y, int grid_z,
    double p_fib,
    std::vector<std::vector<int>>& occupied,
    float life_mean)
{
    int placed = 0;
    int slot_counter = 0;
    const int chain_len = MAX_FIB_CHAIN_LENGTH;  // 3: HEAD → MIDDLE → TAIL

    for (int z = 0; z < grid_z; z++) {
        for (int y = 0; y < grid_y; y++) {
            for (int x = 0; x < grid_x; x++) {
                const float rnd = static_cast<float>(rand()) / RAND_MAX;
                if (rnd >= static_cast<float>(p_fib)) continue;
                if (slot_counter + chain_len > MAX_FIB_SLOTS) break;

                // Check if starting voxel is free
                int idx0 = x + y * grid_x + z * grid_x * grid_y;
                if (occupied[idx0][0] != 0) continue;

                // Try to form a chain of chain_len cells starting here
                // Positions for each cell in the chain
                int cx[MAX_FIB_CHAIN_LENGTH], cy[MAX_FIB_CHAIN_LENGTH], cz[MAX_FIB_CHAIN_LENGTH];
                cx[0] = x; cy[0] = y; cz[0] = z;
                occupied[idx0][0] = 1;  // Mark HEAD occupied tentatively

                int actual_len = 1;
                for (int c = 1; c < chain_len; c++) {
                    if (!findFreeAdjacent(cx[c-1], cy[c-1], cz[c-1],
                                          grid_x, grid_y, grid_z, occupied,
                                          cx[c], cy[c], cz[c])) {
                        break;
                    }
                    int idxc = cx[c] + cy[c] * grid_x + cz[c] * grid_x * grid_y;
                    occupied[idxc][0] = 1;
                    actual_len++;
                }
                // Note: HEAD already marked occupied above, all chain cells marked tentatively.
                // actual_len is how many were placed (1 to chain_len).

                // Assign slots for this chain: [slot_counter .. slot_counter+actual_len-1]
                // Chain: cell[0]=HEAD (divides, future), cell[1]=MIDDLE, cell[actual_len-1]=TAIL (chemotaxis, leader_slot=-1)
                //        cell[2] (leader_slot=slot_counter+1), etc.
                if (slot_counter + actual_len > MAX_FIB_SLOTS) {
                    // Not enough slots left — undo occupancy and stop placing
                    for (int c = 0; c < actual_len; c++) {
                        int idxc = cx[c] + cy[c] * grid_x + cz[c] * grid_x * grid_y;
                        occupied[idxc][0] = 0;
                    }
                    break;
                }

                for (int c = 0; c < actual_len; c++) {
                    float life_rnd = static_cast<float>(rand()) / RAND_MAX;
                    int life = static_cast<int>(life_mean * std::log(1.0f / (life_rnd + 1e-4f)) + 0.5f);
                    if (life < 1) life = 1;

                    fib_agents.push_back();
                    flamegpu::AgentVector::Agent agent = fib_agents.back();
                    agent.setVariable<int>("x", cx[c]);
                    agent.setVariable<int>("y", cy[c]);
                    agent.setVariable<int>("z", cz[c]);
                    agent.setVariable<int>("fib_state", FIB_NORMAL);
                    agent.setVariable<int>("life", life);
                    agent.setVariable<int>("my_slot", slot_counter + c);
                    // TAIL (c==actual_len-1): leader_slot=-1 (moves via chemotaxis)
                    // Others: leader_slot = slot of cell directly toward tail (follows the tail's movement)
                    agent.setVariable<int>("leader_slot", (c == actual_len - 1) ? -1 : (slot_counter + c + 1));
                }

                slot_counter += actual_len;
                placed += actual_len;
            }
        }
    }
    std::cout << "  Placed " << placed << " Fibroblasts in chains (probability-based QSP), "
              << slot_counter << " slots used" << std::endl;
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
    const int tcell_div_interval = model.Environment().getProperty<int>("PARAM_TCELL_DIV_INTERNAL");
    const float treg_life = model.Environment().getProperty<float>("PARAM_TCD4_LIFE_MEAN_SLICE");
    const int treg_div_limit = model.Environment().getProperty<int>("PARAM_TCD4_DIV_LIMIT");
    const int treg_div_interval = model.Environment().getProperty<int>("PARAM_TCD4_DIV_INTERNAL");
    const float mdsc_life = model.Environment().getProperty<float>("PARAM_MDSC_LIFE_MEAN_SLICE");
    const float mac_life = model.Environment().getProperty<float>("PARAM_MAC_LIFE_MEAN");
    const float fib_life = model.Environment().getProperty<float>("PARAM_FIB_LIFE_MEAN");

    // Initialize cancer cells
    {
        flamegpu::AgentVector cancer_pop(model.Agent(AGENT_CANCER_CELL));
        initializeCancerCellCluster(
            cancer_pop, 
            config.grid_x, config.grid_y, config.grid_z,
            config.cluster_radius, stem_div, prog_div, prog_max);
        simulation.setPopulationData(cancer_pop);
    }
    
    // Initialize T cells
    if (config.num_tcells > 0) {
        flamegpu::AgentVector tcell_pop(model.Agent(AGENT_TCELL));
        initializeTCells(
            tcell_pop, 
            config.grid_x, config.grid_y, config.grid_z,
            config.cluster_radius, config.num_tcells, 
            tcell_life, tcell_div_limit,
            IL2_release_time, tcell_div_interval);
        simulation.setPopulationData(tcell_pop);
    }
    
    // Initialize TRegs
    if (config.num_tregs > 0) {
        flamegpu::AgentVector treg_pop(model.Agent(AGENT_TREG));
        initializeTRegs(
            treg_pop, 
            config.grid_x, config.grid_y, config.grid_z,
            config.cluster_radius, config.num_tregs, 
            treg_life, treg_div_limit, treg_div_interval);
        simulation.setPopulationData(treg_pop);
    }
    
    // Initialize MDSCs
    if (config.num_mdscs > 0) {
        flamegpu::AgentVector mdsc_pop(model.Agent(AGENT_MDSC));
        initializeMDSCs(
            mdsc_pop,
            config.grid_x, config.grid_y, config.grid_z,
            config.cluster_radius, config.num_mdscs,
            mdsc_life);
        simulation.setPopulationData(mdsc_pop);
    }

    // Initialize Macrophages
    if (config.num_macrophages > 0) {
        flamegpu::AgentVector mac_pop(model.Agent(AGENT_MACROPHAGE));
        initializeMacrophages(
            mac_pop,
            config.grid_x, config.grid_y, config.grid_z,
            config.cluster_radius, config.num_macrophages,
            mac_life);
        simulation.setPopulationData(mac_pop);
    }

    // Initialize Fibroblasts
    if (config.num_fibroblasts > 0) {
        flamegpu::AgentVector fib_pop(model.Agent(AGENT_FIBROBLAST));
        initializeFibroblasts(
            fib_pop,
            config.grid_x, config.grid_y, config.grid_z,
            config.cluster_radius, config.num_fibroblasts,
            fib_life);
        simulation.setPopulationData(fib_pop);
    }

    // === VASCULAR CELLS ===
    {
        flamegpu::AgentVector vascular_vec(model.Agent(AGENT_VASCULAR));

        if (config.vascular_mode == "random") {
            // Random walk initialization (HCC-style)
            int num_segments = 4;  // Default: 4 vessel segments
            initializeVascularCellsRandom(
                vascular_vec,
                config.grid_x, config.grid_y, config.grid_z,
                config.cluster_radius,
                num_segments);
        }
        else if (config.vascular_mode == "xml") {
            // XML-based initialization (Phase 2)
            std::cout << "WARNING: XML vasculature loading not yet implemented" << std::endl;
            std::cout << "  Falling back to test mode" << std::endl;
            initializeVascularCellsTest(vascular_vec, config.grid_x, config.grid_y, config.grid_z);
        }
        else if (config.vascular_mode == "test") {
            // Manual test pattern (5 vessels at center)
            initializeVascularCellsTest(vascular_vec, config.grid_x, config.grid_y, config.grid_z);
        }
        else {
            std::cerr << "ERROR: Unknown vascular mode '" << config.vascular_mode << "'" << std::endl;
            std::cerr << "  Valid modes: random, xml, test" << std::endl;
            std::cerr << "  Falling back to test mode" << std::endl;
            initializeVascularCellsTest(vascular_vec, config.grid_x, config.grid_y, config.grid_z);
        }

        simulation.setPopulationData(vascular_vec);
    }

    std::cout << "Agent initialization complete\n" << std::endl;
}
// ============================================================================
// Get cell type CDF
// ============================================================================
std::vector<double> get_celltype_cdf(flamegpu::ModelDescription& model){
    int dmax = model.Environment().getProperty<int>("PARAM_PROG_DIV_MAX");
    std::vector<double> celltype_cdf = std::vector<double>(dmax+3,0.0);

    double k, r, rs, rp, mu, l0, l1, l2;
    k = model.Environment().getProperty<float>("PARAM_ASYM_DIV_PROB");
    rs = model.Environment().getProperty<float>("PARAM_CSC_GROWTH_RATE");
    rp = model.Environment().getProperty<float>("PARAM_PROG_GROWTH_RATE");
    mu = model.Environment().getProperty<float>("PARAM_SEN_DEATH_RATE");
    r = rs * (1 - k);
	l0 = k*rs / (r + rp);
	l1 = 2 * rp / (r + rp);
	l2 = 2 * rp / (r + mu);
    double C;
    if (l1 == 1) {
        C = 1 + l0 + l0 * l2 * std::pow(l1, (dmax - 1));
    }
    else {
        C = 1 + l0 * (std::pow(l1, dmax) - 1) / (l1 - 1) + l0 * l2 * std::pow(l1, (dmax - 1));
    }
    double p;
    celltype_cdf[0] = p = 1 / C; // joint P
    p *= l0;
    celltype_cdf[1] = celltype_cdf[0] + p;
    for (size_t i = 2; i <= dmax; i++)
    {
        p *= l1;
        celltype_cdf[i] = celltype_cdf[i - 1] + p;
    }
    p *= l2;
    celltype_cdf[dmax + 1] = celltype_cdf[dmax] + p;
    celltype_cdf[dmax + 2] = 1.0;

    return celltype_cdf;
}

// ============================================================================
// QSP-Seeded Initialization
// ============================================================================

void initializeToQSP(
    flamegpu::CUDASimulation& simulation,
    flamegpu::ModelDescription& model,
    const SimulationConfig& config,
    const LymphCentralWrapper& lymph)
{
    std::cout << "\n=== Initializing Agents from QSP State ===" << std::endl;

    // Get QSP state after warmup
    QSPState qsp = lymph.get_state_for_abm();
    std::cout << "  QSP tumor volume  : " << qsp.tum_vol    << " cm^3" << std::endl;
    std::cout << "  QSP cc_tumor (raw): " << qsp.cc_tumor  << " molecules" << std::endl;
    std::cout << "  QSP Teff (SI)     : " << qsp.teff_tumor << std::endl;
    std::cout << "  QSP Treg (SI)     : " << qsp.treg_tumor << std::endl;
    std::cout << "  QSP MDSC (SI)     : " << qsp.mdsc_tumor << std::endl;

    // // -----------------------------------------------------------------------
    // // Compute cluster_radius in voxels from QSP tumor volume
    // //   Sphere volume: V = (4/3)π r³  →  r = cbrt(3V / 4π)
    // // -----------------------------------------------------------------------
    // const double voxel_size_cm = config.voxel_size * 1e-4;  // µm → cm
    // const double tum_radius_cm = std::cbrt(3.0 * qsp.tum_vol / (4.0 * M_PI));
    // int cluster_radius = static_cast<int>(std::round(tum_radius_cm / voxel_size_cm));

    // // Clamp to fit within grid (at least 1 voxel, at most grid_half - 2)
    // const int max_radius = std::min({config.grid_x, config.grid_y, config.grid_z}) / 2 - 2;
    // if (cluster_radius < 1) cluster_radius = 1;
    // if (cluster_radius > max_radius) cluster_radius = max_radius;

    // Use arbitrary scalar to initialize tumor radius
    int cluster_radius = static_cast<int>(0.44 * config.grid_x); 
    std::cout << "  cluster_radius  = " << cluster_radius << " voxels" << std::endl;

    // -----------------------------------------------------------------------
    // Immune cell placement probabilities (HCC pattern):
    //
    //   Teff: 0 at init (coefficient 0 in HCC)
    //   Treg: per-voxel probability (user addition; HCC had coefficient 0)
    //   TH  : p = 0.03 × (th   / total_immune_conc + ε)
    //   MDSC: p =         mdsc  / (total_immune_conc + ε)
    //
    //   total_immune_conc = Teff + TH + Treg + MDSC  (all in SI, mol/m³)
    //   CC is NOT in this denominator (it uses a separate fibroblast-style
    //   formula in HCC).  ε = 1e-30 prevents division by zero.
    // -----------------------------------------------------------------------
    const double avogadros = 6.022140857e23;  
    const double cc_SI     = qsp.cc_tumor / avogadros;
    const double eps = 1e-30;

    const double p_th   = 0.03 * qsp.th_tumor   / (qsp.th_tumor + cc_SI + eps);
    const double p_mdsc =        qsp.mdsc_tumor  / (qsp.mdsc_tumor + cc_SI + eps);
    const double p_treg = 0.0;
    const double p_mac  =        qsp.m2_tumor / (qsp.m2_tumor + cc_SI + eps);
    double p_fib  =        qsp.caf_tumor / (qsp.caf_tumor + cc_SI + eps);
    if (p_fib > 0.1) {
		p_fib = 0.1;
	} else if (p_fib < 0.001) {
        p_fib = 0.001;
    }

    std::cout << "  p_treg = " << p_treg   << std::endl;
    std::cout << "  p_th   = " << p_th   << std::endl;
    std::cout << "  p_mdsc = " << p_mdsc << std::endl;
    std::cout << "  p_mac  = " << p_mac  << std::endl;
    std::cout << "  p_fib  = " << p_fib  << std::endl;
    // -----------------------------------------------------------------------
    // Read cell-lifecycle parameters from model environment (same as initializeAllAgents)
    // -----------------------------------------------------------------------
    const float stem_div         = model.Environment().getProperty<float>("PARAM_FLOAT_CANCER_CELL_STEM_DIV_INTERVAL_SLICE");
    const float prog_div         = model.Environment().getProperty<float>("PARAM_FLOAT_CANCER_CELL_PROGENITOR_DIV_INTERVAL_SLICE");
    const int   prog_max         = model.Environment().getProperty<int>("PARAM_PROG_DIV_MAX");
    const float tcell_life       = model.Environment().getProperty<float>("PARAM_T_CELL_LIFE_MEAN_SLICE");
    const int   tcell_div_limit  = model.Environment().getProperty<int>("PARAM_TCELL_DIV_LIMIT");
    const float IL2_release_time = model.Environment().getProperty<float>("PARAM_TCELL_IL2_RELEASE_TIME");
    const int   tcell_div_interval = model.Environment().getProperty<int>("PARAM_TCELL_DIV_INTERNAL");
    const float treg_life        = model.Environment().getProperty<float>("PARAM_TCD4_LIFE_MEAN_SLICE");
    const int   treg_div_limit   = model.Environment().getProperty<int>("PARAM_TCD4_DIV_LIMIT");
    const int   treg_div_interval = model.Environment().getProperty<int>("PARAM_TCD4_DIV_INTERNAL");
    const float mdsc_life        = model.Environment().getProperty<float>("PARAM_MDSC_LIFE_MEAN_SLICE");
    const float mac_life        = model.Environment().getProperty<float>("PARAM_MAC_LIFE_MEAN");
    const float fib_life        = model.Environment().getProperty<float>("PARAM_FIB_LIFE_MEAN");
    // -----------------------------------------------------------------------
    // Get CDF for cancer cell population
    // -----------------------------------------------------------------------
    std::vector<double> celltype_cdf = get_celltype_cdf(model);
    // -----------------------------------------------------------------------
    // Initialize cancer cells (fills sphere of cluster_radius) and build
    // occupancy grid so immune cells don't overlap cancer cell positions.
    // -----------------------------------------------------------------------
    const int total_voxels = config.grid_x * config.grid_y * config.grid_z;
    // std::vector<std::vector<int>> occupied = std::vector<int>(total_voxels, std::vector<int>(3, 0));

    std::vector<std::vector<int>> occupied(total_voxels, std::vector<int>(3, 0));
    {
        flamegpu::AgentVector cancer_pop(model.Agent(AGENT_CANCER_CELL));
        initializeCancerCellsRandom(
            cancer_pop,
            config.grid_x, config.grid_y, config.grid_z,
            cluster_radius, stem_div, prog_div, prog_max, celltype_cdf);
        buildOccupancyGrid(cancer_pop, occupied, config.grid_x, config.grid_y);
        simulation.setPopulationData(cancer_pop);
    }

    // T cells: none at init — recruited during pre-simulation via QSP
    {
        std::cout << "[DEBUG] Setting T cells population..." << std::endl;
        flamegpu::AgentVector tcell_pop(model.Agent(AGENT_TCELL));
        simulation.setPopulationData(tcell_pop);  // empty
        std::cout << "[DEBUG] T cells population set" << std::endl;
    }

    // TH and TReg cells: probability-based placement across all voxels
    {
        std::cout << "[DEBUG] Initializing TH/TReg cells..." << std::endl;
        flamegpu::AgentVector treg_pop(model.Agent(AGENT_TREG));
        initializeTHCellsFromQSP(
            treg_pop,
            config.grid_x, config.grid_y, config.grid_z,
            p_th, occupied,
            treg_life, treg_div_limit, treg_div_interval);
        initializeTRegCellsFromQSP(
            treg_pop,
            config.grid_x, config.grid_y, config.grid_z,
            p_treg, occupied,
            treg_life, treg_div_limit, treg_div_interval);
        std::cout << "[DEBUG] Setting TReg population (" << treg_pop.size() << " agents)..." << std::endl;
        simulation.setPopulationData(treg_pop);
        std::cout << "[DEBUG] TReg population set" << std::endl;
    }

    // MDSCs: probability-based placement across all voxels
    {
        std::cout << "[DEBUG] Initializing MDSCs..." << std::endl;
        flamegpu::AgentVector mdsc_pop(model.Agent(AGENT_MDSC));
        initializeMDSCsFromQSP(
            mdsc_pop,
            config.grid_x, config.grid_y, config.grid_z,
            p_mdsc, occupied,
            mdsc_life);
        std::cout << "[DEBUG] Setting MDSC population (" << mdsc_pop.size() << " agents)..." << std::endl;
        simulation.setPopulationData(mdsc_pop);
        std::cout << "[DEBUG] MDSC population set" << std::endl;
    }

    // Macrophages: probability-based placement across all voxels
    {
        std::cout << "[DEBUG] Initializing Macrophages..." << std::endl;
        flamegpu::AgentVector mac_pop(model.Agent(AGENT_MACROPHAGE));
        initializeMacsFromQSP(
            mac_pop,
            config.grid_x, config.grid_y, config.grid_z,
            p_mac, occupied,
            mac_life);
        std::cout << "[DEBUG] Setting Macrophage population (" << mac_pop.size() << " agents)..." << std::endl;
        simulation.setPopulationData(mac_pop);
        std::cout << "[DEBUG] Macrophage population set" << std::endl;
    }

    // Fibroblasts: probability-based placement across all voxels
    {
        std::cout << "[DEBUG] Initializing Fibroblasts..." << std::endl;
        flamegpu::AgentVector fib_pop(model.Agent(AGENT_FIBROBLAST));
        initializeFibroblastsFromQSP(
            fib_pop,
            config.grid_x, config.grid_y, config.grid_z,
            p_fib, occupied,
            fib_life);
        std::cout << "[DEBUG] Setting Fibroblast population (" << fib_pop.size() << " agents)..." << std::endl;
        simulation.setPopulationData(fib_pop);
        std::cout << "[DEBUG] Fibroblast population set" << std::endl;
    }

    // Initialize vascular cells (same logic as initializeAllAgents)
    {
        std::cout << "[DEBUG] Initializing Vascular cells..." << std::endl;
        flamegpu::AgentVector vascular_vec(model.Agent(AGENT_VASCULAR));
        if (config.vascular_mode == "random") {
            initializeVascularCellsRandom(
                vascular_vec,
                config.grid_x, config.grid_y, config.grid_z,
                cluster_radius, /*num_segments=*/4);
        } else if (config.vascular_mode == "test") {
            initializeVascularCellsTest(vascular_vec, config.grid_x, config.grid_y, config.grid_z);
        } else {
            std::cerr << "WARNING: Unknown vascular mode '" << config.vascular_mode
                      << "', falling back to test mode" << std::endl;
            initializeVascularCellsTest(vascular_vec, config.grid_x, config.grid_y, config.grid_z);
        }
        std::cout << "[DEBUG] Setting Vascular population (" << vascular_vec.size() << " agents)..." << std::endl;
        simulation.setPopulationData(vascular_vec);
        std::cout << "[DEBUG] Vascular population set" << std::endl;
    }

    std::cout << "QSP-based agent initialization complete\n" << std::endl;
}

} // namespace PDAC