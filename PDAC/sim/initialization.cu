#include "initialization.cuh"
#include "../core/common.cuh"
#include "../abm/gpu_param.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <string>
#include <array>

namespace PDAC {

// ============================================================================
// SimulationConfig Implementation
// ============================================================================

SimulationConfig::SimulationConfig()
    : steps(200)
    , random_seed(12345)
    , init_method(0)
    , vascular_mode("random")
    , vascular_xml_file("")
    , grid_out(3)
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
            int val = std::atoi(argv[++i]);
            if (val != 0) {
                std::cerr << "WARNING: Only -i 0 (QSP-seeded) is supported. Ignoring -i " << val << std::endl;
            }
        } else if ((arg == "--grid-size" || arg == "-g") && i + 1 < argc) {
            int size = std::atoi(argv[++i]);
            grid_x = size;
            grid_y = size;
            grid_z = size;
        } else if ((arg == "--steps" || arg == "-s") && i + 1 < argc) {
            steps = std::atoi(argv[++i]);
        } else if ((arg == "--grid-output" || arg == "-G") && i + 1 < argc) {
            grid_out = std::atoi(argv[++i]);
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
                      << "  -i, --initialization N   initialization type: 0=QSP-seeded [default: 0]\n"
                      << "  -g, --grid-size N        Grid dimensions NxNxN [default: from XML]\n"
                      << "  -s, --steps N            Number of simulation steps [default: 200]\n"
                      << "  -G, --grid-output N      Grid output: 0=none, 1=ABM only, 2=PDE+ECM only, 3=both [default: 3]\n"
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
    std::cout << "\n=== PDAC ABM-PDE GPU Simulation ===" << std::endl;
    std::cout << "Grid: " << grid_x << "x" << grid_y << "x" << grid_z << std::endl;
    std::cout << "Voxel size: " << voxel_size << " µm" << std::endl;
    std::cout << "Steps: " << steps << std::endl;
    std::cout << "Random seed: " << random_seed << std::endl;
    std::cout << "Init: QSP-seeded" << std::endl;

    std::cout << "\nPDE Integration:" << std::endl;
    std::cout << "  ABM timestep: " << dt_abm << " s (" << (dt_abm/60.0f) << " min)" << std::endl;
    std::cout << "  Molecular substeps: " << molecular_steps << std::endl;
    std::cout << "  PDE timestep: " << (dt_abm / molecular_steps) << " s" << std::endl;
    std::cout << "===================================\n" << std::endl;
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
    unsigned int tip_id = 0,
    int branch = 0)
{
    agent.setVariable<int>("x", x);
    agent.setVariable<int>("y", y);
    agent.setVariable<int>("z", z);
    agent.setVariable<int>("cell_state", state);
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
    agent.setVariable<int>("branch", branch);
}

void initializeVascularCellsRandom(
    flamegpu::AgentVector& vascular_agents,
    int grid_x, int grid_y, int grid_z,
    int tumor_radius,
    int num_segments,
    float branch_prob,
    unsigned int seed)
{
    const int center_x = grid_x / 2;
    const int center_y = grid_y / 2;
    const int center_z = grid_z / 2;
    // Exclude vessels from within the tumor, not half the grid.
    // HCC's vascular graph includes vessels near/inside the tumor edge.
    const double radius = tumor_radius;

    // Random number generator
    std::srand(seed);
    auto rand_unif = []() { return static_cast<double>(std::rand()) / RAND_MAX; };

    // HCC Tumor.cpp line 1518: initial phalanx cells have p_branch = PARAM_VAS_BRANCH_PROB/5
    // chance to have branch=1 at init, allowing sprouting on step 1 before VEGFA exists
    const double p_branch_init = branch_prob / 5.0;

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
            int branch_flag = (rand_unif() < p_branch_init) ? 1 : 0;
            setVascularCellVariables(agent, current_x, current_y, current_z,
                                   2, static_cast<float>(dx), static_cast<float>(dy), static_cast<float>(dz), 0u, branch_flag);
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
                int branch_flag = (rand_unif() < p_branch_init) ? 1 : 0;
                setVascularCellVariables(agent, current_x, current_y, current_z,
                                       2, static_cast<float>(dx), static_cast<float>(dy), static_cast<float>(dz), 0u, branch_flag);
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
    float senescent_mean_life,
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
                if (i <= progenitor_div_max + 1){
                    cancer_agents.push_back();
                    flamegpu::AgentVector::Agent agent = cancer_agents.back();
                    if (i==0) {
                        cell_state = CANCER_STEM;

                        float random_factor = (static_cast<float>(rand()) / (RAND_MAX + 1.0f));
                        div_cd = static_cast<int>(stem_div_interval * random_factor) + 1;  // [1, interval]

                        divide_flag = 1;

                        is_stem = 1;

                    } else if (i == progenitor_div_max + 1){
                        cell_state = CANCER_SENESCENT;

                    } else {
                        cell_state = CANCER_PROGENITOR;
                        div = progenitor_div_max + 1 - i;

                        float random_factor = (static_cast<float>(rand()) / (RAND_MAX + 1.0f));
                        div_cd = static_cast<int>(progenitor_div_interval * random_factor) + 1;  // [1, interval]

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

                    // Senescent cells need a valid life countdown (exponential distribution)
                    if (cell_state == CANCER_SENESCENT) {
                        float r = static_cast<float>(rand()) / (RAND_MAX + 1.0f);
                        int sen_life = static_cast<int>(-senescent_mean_life * logf(r + 0.0001f) + 0.5f);
                        agent.setVariable<int>("life", sen_life > 0 ? sen_life : 1);
                    }

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
    float life_mean, float life_sd, int div_limit, int div_interval)
{
    int placed = 0;
    for (int z = 0; z < grid_z; z++) {
        for (int y = 0; y < grid_y; y++) {
            for (int x = 0; x < grid_x; x++) {
                // const int idx = z * grid_x * grid_y + y * grid_x + x;

                const float rnd = static_cast<float>(rand()) / RAND_MAX;
                if (rnd >= static_cast<float>(p_th)) continue;

                // Use normal distribution: lifeMean + gaussian * lifeSD (Box-Muller)
                float u1 = static_cast<float>(rand()) / RAND_MAX;
                float u2 = static_cast<float>(rand()) / RAND_MAX;
                if (u1 < 1e-6f) u1 = 1e-6f;
                float z0 = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * 3.14159f * u2);
                int life = static_cast<int>(life_mean + z0 * life_sd + 0.5f);
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
                // agent.setVariable<int>("divide_cd", div_cd);
                agent.setVariable<int>("divide_cd", div_interval);
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
    float life_mean, float life_sd, int div_limit, int div_interval)
{
    int placed = 0;
    for (int z = 0; z < grid_z; z++) {
        for (int y = 0; y < grid_y; y++) {
            for (int x = 0; x < grid_x; x++) {
                // const int idx = z * grid_x * grid_y + y * grid_x + x;

                const float rnd = static_cast<float>(rand()) / RAND_MAX;
                if (rnd >= static_cast<float>(p_treg)) continue;

                // Use normal distribution: lifeMean + gaussian * lifeSD (Box-Muller)
                float u1 = static_cast<float>(rand()) / RAND_MAX;
                float u2 = static_cast<float>(rand()) / RAND_MAX;
                if (u1 < 1e-6f) u1 = 1e-6f;
                float z0 = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * 3.14159f * u2);
                int life = static_cast<int>(life_mean + z0 * life_sd + 0.5f);
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
                agent.setVariable<int>("cell_state", MAC_M2);

                placed++;
            }
        }
    }
    std::cout << "  Placed " << placed << " Macrophages (probability-based QSP)" << std::endl;
}

// ============================================================================
// Fibroblast Initialization
// ============================================================================

void initializeFibroblastsFromQSP(
    flamegpu::AgentVector& fib_agents,
    int grid_x, int grid_y, int grid_z,
    double p_fib,
    std::vector<std::vector<int>>& occupied,
    float life_mean)
{
    int placed = 0;

    for (int z = 0; z < grid_z; z++) {
        for (int y = 0; y < grid_y; y++) {
            for (int x = 0; x < grid_x; x++) {
                const float rnd = static_cast<float>(rand()) / RAND_MAX;
                if (rnd >= static_cast<float>(p_fib)) continue;

                int idx0 = x + y * grid_x + z * grid_x * grid_y;
                if (occupied[idx0][0] != 0) continue;
                occupied[idx0][0] = 1;

                float life_rnd = static_cast<float>(rand()) / RAND_MAX;
                int life = static_cast<int>(life_mean * std::log(1.0f / (life_rnd + 1e-4f)) + 0.5f);
                if (life < 1) life = 1;

                fib_agents.push_back();
                flamegpu::AgentVector::Agent agent = fib_agents.back();
                agent.setVariable<int>("x", x);
                agent.setVariable<int>("y", y);
                agent.setVariable<int>("z", z);
                agent.setVariable<int>("cell_state", FIB_QUIESCENT);
                agent.setVariable<int>("life", life);
                agent.setVariable<int>("divide_cooldown", 0);
                agent.setVariable<int>("divide_count", 0);

                placed++;
            }
        }
    }
    std::cout << "  Placed " << placed << " Fibroblasts (quiescent, single-cell, probability-based QSP)" << std::endl;
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
    const float cancer_sen_life  = model.Environment().getProperty<float>("PARAM_CANCER_SENESCENT_MEAN_LIFE");
    const float tcell_life       = model.Environment().getProperty<float>("PARAM_T_CELL_LIFE_MEAN_SLICE");
    const int   tcell_div_limit  = model.Environment().getProperty<int>("PARAM_TCELL_DIV_LIMIT");
    const float IL2_release_time = model.Environment().getProperty<float>("PARAM_TCELL_IL2_RELEASE_TIME");
    const int   tcell_div_interval = model.Environment().getProperty<int>("PARAM_TCELL_DIV_INTERNAL");
    const float treg_life        = model.Environment().getProperty<float>("PARAM_TCD4_LIFE_MEAN_SLICE");
    const float treg_life_sd     = model.Environment().getProperty<float>("PARAM_TCELL_LIFESPAN_SD_SLICE");
    const int   treg_div_limit   = model.Environment().getProperty<int>("PARAM_TCD4_DIV_LIMIT");
    const int   treg_div_interval = model.Environment().getProperty<int>("PARAM_TCD4_DIV_INTERNAL");
    const float mdsc_life        = model.Environment().getProperty<float>("PARAM_MDSC_LIFE_MEAN_SLICE");
    const float mac_life        = model.Environment().getProperty<float>("PARAM_MAC_LIFE_MEAN");
    const float fib_life        = model.Environment().getProperty<float>("PARAM_FIB_LIFE_MEAN");
    const float vas_branch_prob = model.Environment().getProperty<float>("PARAM_VAS_BRANCH_PROB");
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
            cluster_radius, stem_div, prog_div, prog_max, cancer_sen_life, celltype_cdf);
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
            treg_life, treg_life_sd, treg_div_limit, treg_div_interval);
        initializeTRegCellsFromQSP(
            treg_pop,
            config.grid_x, config.grid_y, config.grid_z,
            p_treg, occupied,
            treg_life, treg_life_sd, treg_div_limit, treg_div_interval);
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
                cluster_radius, /*num_segments=*/1,
                vas_branch_prob,
                config.random_seed);
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