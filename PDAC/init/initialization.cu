#include "initialization.cuh"
#include "ductal_init.cuh"
#include "../core/common.cuh"
#include "../abm/gpu_param.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <string>
#include <array>
#include <algorithm>
#include <random>
#include <vector>
#include <queue>
#include <unordered_set>

namespace PDAC {

// ============================================================================
// SimulationConfig Implementation
// ============================================================================

SimulationConfig::SimulationConfig()
    : steps(200)
    , random_seed(12345)
    , cluster_radius(5)
    , vascular_mode("random")
    , vascular_xml_file("")
    , init_mode(0)
    , grid_out(0)
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
        } else if ((arg == "--init-mode" || arg == "-i") && i + 1 < argc) {
            init_mode = std::atoi(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "\nOptions:\n"
                      << "  -p, --param-file FILE    Path to parameter XML file [default: param_all_test.xml]\n"
                      << "  -g, --grid-size N        Grid dimensions NxNxN [default: from XML]\n"
                      << "  -s, --steps N            Number of simulation steps [default: 200]\n"
                      << "  -G, --grid-output N      Grid output: 0=none, 1=ABM only, 2=PDE+ECM only, 3=both [default: 0]\n"
                      << "  -oi, --out_int N         Output interval frequency [default: 1]\n"
                      << "  --seed N                 Random seed [default: 12345]\n"
                      << "  -i, --init-mode N        Init mode: 0=test (quick), 1=production (ductal network) [default: 0]\n"
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
    std::cout << "  Init mode: " << init_mode << (init_mode == 1 ? " (production — ductal network)" : " (test — quick)") << std::endl;
    std::cout << "  Tumor radius: " << cluster_radius << " voxels" << std::endl;

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
                                   2, static_cast<float>(dx), static_cast<float>(dy), static_cast<float>(dz), 1u, branch_flag);
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
                                       2, static_cast<float>(dx), static_cast<float>(dy), static_cast<float>(dz), 1u, branch_flag);
                total_vessels++;
            }

            segment_length++;
        }
    }

    std::cout << "Initialized " << total_vessels << " vascular cells (random walk, "
              << num_segments << " segments)" << std::endl;
}

// Sequentially assign initial TIP sprout intents to a subset of initialized PHALANX cells.
// Called after initializeVascularCellsRandom, before simulation starts.
// Shuffles PHALANX agents, then greedily assigns INTENT_DIVIDE to cells that are at least
// min_neighbor_range voxels away from any already-assigned cell — matching the runtime
// nearby-vessel exclusion logic but without the GPU race condition.
// Step 0 of vascular_state_step is skipped so these pre-set intents survive to vascular_divide.
void assignInitialVascularTips(
    flamegpu::AgentVector& vascular_agents,
    int grid_x, int grid_y, int grid_z,
    int min_neighbor_range,
    unsigned int seed)
{
    // Collect indices of all PHALANX agents
    std::vector<size_t> phalanx_indices;
    for (size_t i = 0; i < vascular_agents.size(); i++) {
        if (vascular_agents[i].getVariable<int>("cell_state") == VAS_PHALANX) {
            phalanx_indices.push_back(i);
        }
    }

    // Shuffle to avoid bias toward agents added first
    std::mt19937 rng(seed);
    std::shuffle(phalanx_indices.begin(), phalanx_indices.end(), rng);

    // Flat grid marking claimed positions (true = a tip source is here)
    std::vector<bool> claimed(grid_x * grid_y * grid_z, false);

    int num_assigned = 0;
    for (size_t idx : phalanx_indices) {
        auto agent = vascular_agents[idx];
        const int ax = agent.getVariable<int>("x");
        const int ay = agent.getVariable<int>("y");
        const int az = agent.getVariable<int>("z");

        // Check if any already-claimed position is within min_neighbor_range
        // Mirror HCC range: [-range, range) exclusive upper bound
        bool nearby = false;
        for (int dx = -min_neighbor_range; dx < min_neighbor_range && !nearby; dx++) {
            for (int dy = -min_neighbor_range; dy < min_neighbor_range && !nearby; dy++) {
                for (int dz = -min_neighbor_range; dz < min_neighbor_range && !nearby; dz++) {
                    if (dx == 0 && dy == 0 && dz == 0) continue;
                    const int cx = ax + dx;
                    const int cy = ay + dy;
                    const int cz = az + dz;
                    if (cx < 0 || cx >= grid_x || cy < 0 || cy >= grid_y || cz < 0 || cz >= grid_z) continue;
                    if (claimed[cz * grid_y * grid_x + cy * grid_x + cx]) {
                        nearby = true;
                    }
                }
            }
        }

        if (!nearby) {
            agent.setVariable<int>("intent_action", 2);
            claimed[az * grid_y * grid_x + ay * grid_x + ax] = true;
            num_assigned++;
        }
    }

    std::cout << "Assigned initial TIP sprout intents to " << num_assigned
              << " / " << phalanx_indices.size() << " PHALANX cells" << std::endl;
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
        if (occupied[idx][0] == 0 && occupied[idx][1] == 0) {
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
    const int init_chain_len = 3;

    for (int z = 0; z < grid_z; z++) {
        for (int y = 0; y < grid_y; y++) {
            for (int x = 0; x < grid_x; x++) {
                const float rnd = static_cast<float>(rand()) / RAND_MAX;
                if (rnd >= static_cast<float>(p_fib)) continue;

                // HCC algorithm: anchor (x,y,z) is NOT checked or marked.
                // Find all 3 chain positions first, then mark all at once.
                // c1 = free VN neighbor of anchor
                // c2 = free VN neighbor of c1
                // c3 = free VN neighbor of c2, with c3 != c1
                int c1x, c1y, c1z;
                if (!findFreeAdjacent(x, y, z, grid_x, grid_y, grid_z, occupied, c1x, c1y, c1z)) continue;
                int c2x, c2y, c2z;
                if (!findFreeAdjacent(c1x, c1y, c1z, grid_x, grid_y, grid_z, occupied, c2x, c2y, c2z)) continue;
                int c3x, c3y, c3z;
                if (!findFreeAdjacent(c2x, c2y, c2z, grid_x, grid_y, grid_z, occupied, c3x, c3y, c3z)) continue;
                if (c3x == c1x && c3y == c1y && c3z == c1z) continue;  // no loop

                // All 3 found — mark occupied and create chain
                int idx1 = c1x + c1y * grid_x + c1z * grid_x * grid_y;
                int idx2 = c2x + c2y * grid_x + c2z * grid_x * grid_y;
                int idx3 = c3x + c3y * grid_x + c3z * grid_x * grid_y;
                occupied[idx1][1] = 1;
                occupied[idx2][1] = 1;
                occupied[idx3][1] = 1;

                float life_rnd = static_cast<float>(rand()) / RAND_MAX;
                int life = static_cast<int>(life_mean * std::log(1.0f / (life_rnd + 1e-4f)) + 0.5f);
                if (life < 1) life = 1;

                int cx[MAX_FIB_CHAIN_LENGTH] = {c1x, c2x, c3x, 0, 0};
                int cy[MAX_FIB_CHAIN_LENGTH] = {c1y, c2y, c3y, 0, 0};
                int cz[MAX_FIB_CHAIN_LENGTH] = {c1z, c2z, c3z, 0, 0};

                fib_agents.push_back();
                flamegpu::AgentVector::Agent agent = fib_agents.back();
                agent.setVariable<int>("x", c1x);
                agent.setVariable<int>("y", c1y);
                agent.setVariable<int>("z", c1z);
                std::array<int, MAX_FIB_CHAIN_LENGTH> asx, asy, asz;
                for (int i = 0; i < MAX_FIB_CHAIN_LENGTH; i++) { asx[i] = cx[i]; asy[i] = cy[i]; asz[i] = cz[i]; }
                agent.setVariable<int, MAX_FIB_CHAIN_LENGTH>("seg_x", asx);
                agent.setVariable<int, MAX_FIB_CHAIN_LENGTH>("seg_y", asy);
                agent.setVariable<int, MAX_FIB_CHAIN_LENGTH>("seg_z", asz);
                agent.setVariable<int>("chain_len", init_chain_len);
                agent.setVariable<int>("cell_state", FIB_NORMAL);
                agent.setVariable<int>("life", life);

                placed++;
            }
        }
    }
    std::cout << "  Placed " << placed << " Fibroblast chains (3-segment, probability-based QSP)" << std::endl;
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

void initializeSphere(
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
    const int vas_min_neighbor = static_cast<int>(model.Environment().getProperty<float>("PARAM_VAS_MIN_NEIGHBOR"));
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
            // Match HCC: radius = 0.5 * grid_x, 1 segment
            int vas_radius = config.grid_x / 2;
            initializeVascularCellsRandom(
                vascular_vec,
                config.grid_x, config.grid_y, config.grid_z,
                vas_radius, /*num_segments=*/1,
                vas_branch_prob,
                config.random_seed);
            assignInitialVascularTips(
                vascular_vec,
                config.grid_x, config.grid_y, config.grid_z,
                vas_min_neighbor,
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

    std::cout << "Sphere-based agent initialization complete\n" << std::endl;
}

// ============================================================================
// Helper: find free adjacent voxel constrained to stroma
// ============================================================================
static bool findFreeAdjacentStroma(int x, int y, int z,
                                    int grid_x, int grid_y, int grid_z,
                                    std::vector<std::vector<int>>& occupied,
                                    const std::vector<uint8_t>& wall_mask,
                                    int& nx, int& ny, int& nz)
{
    int ddx[6] = {1,-1,0,0,0,0};
    int ddy[6] = {0,0,1,-1,0,0};
    int ddz[6] = {0,0,0,0,1,-1};
    for (int i = 5; i > 0; i--) {
        int j = rand() % (i + 1);
        std::swap(ddx[i], ddx[j]);
        std::swap(ddy[i], ddy[j]);
        std::swap(ddz[i], ddz[j]);
    }
    for (int i = 0; i < 6; i++) {
        int cx = x + ddx[i];
        int cy = y + ddy[i];
        int cz = z + ddz[i];
        if (cx < 0 || cx >= grid_x || cy < 0 || cy >= grid_y || cz < 0 || cz >= grid_z) continue;
        int idx = cx + cy * grid_x + cz * grid_x * grid_y;
        if (wall_mask[idx] != TISSUE_NONE) continue;
        if (occupied[idx][0] == 0 && occupied[idx][1] == 0) {
            nx = cx; ny = cy; nz = cz;
            return true;
        }
    }
    return false;
}

// ============================================================================
// Periductal Vascular Tree Generation
// ============================================================================
// Generates a vascular network in stroma with:
//   Phase 1: Trunk vessels from domain boundaries, random walk inward
//   Phase 2: Branches that increase near duct walls (periductal plexus)
//   Phase 3: Short capillary segments in lobular stroma (septum-weighted)
// All vessels are constrained to stroma voxels (wall_mask == TISSUE_NONE).

static void initializeVascularPeriductal(
    flamegpu::AgentVector& vascular_agents,
    int gx, int gy, int gz,
    const std::vector<uint8_t>& wall_mask,
    const std::vector<float>& dist_to_wall,
    const std::vector<float>& septum_density,
    float branch_prob,
    std::mt19937& rng)
{
    auto u01 = [&]() { return std::uniform_real_distribution<float>(0.0f, 1.0f)(rng); };
    auto vidx = [&](int x, int y, int z) { return x + y * gx + z * gx * gy; };
    auto in_bounds = [&](int x, int y, int z) {
        return x >= 0 && x < gx && y >= 0 && y < gy && z >= 0 && z < gz;
    };
    auto is_stroma = [&](int x, int y, int z) {
        return in_bounds(x, y, z) && wall_mask[vidx(x, y, z)] == TISSUE_NONE;
    };

    // Track occupied vascular positions to avoid doubling up
    std::unordered_set<int> vas_occupied;

    auto place_phalanx = [&](int x, int y, int z, float dx, float dy, float dz, int branch_flag) -> bool {
        int idx = vidx(x, y, z);
        if (vas_occupied.count(idx)) return false;
        vas_occupied.insert(idx);
        vascular_agents.push_back();
        auto agent = vascular_agents.back();
        setVascularCellVariables(agent, x, y, z, 2 /*VAS_PHALANX*/, dx, dy, dz, 1u, branch_flag);
        return true;
    };

    // ── Phase 1 & 2: Trunk vessels with periductal branching ──────────────
    //
    // Start N trunk vessels from random positions on the 6 grid faces.
    // Each trunk does a biased random walk inward with high persistence.
    // When near duct walls, branching probability increases.
    // Branches do shorter walks with lower persistence.

    const int n_entries = 10;              // number of trunk vessels
    const int trunk_max_len = gx + gy;    // max steps per trunk
    const int branch_max_len = 40;        // max steps per branch
    const int max_branch_depth = 3;       // trunk=0, branch=1, sub-branch=2, ...
    const float persistence_trunk = 0.85f;
    const float persistence_branch = 0.6f;
    const float p_branch_base = 0.02f;    // base branching probability per step
    const float p_branch_periductal = 0.15f;  // additional near duct walls
    const float periductal_decay = 5.0f;  // distance scale for periductal boost

    struct WalkState {
        int x, y, z;
        float dx, dy, dz;  // current direction (unit-ish)
        int steps_remaining;
        int depth;
    };

    std::vector<WalkState> walk_queue;

    // Seed trunk entry points on grid faces
    for (int e = 0; e < n_entries; e++) {
        int face = e % 6;
        int sx, sy, sz;
        float ddx = 0, ddy = 0, ddz = 0;

        // Pick random position on the chosen face, direction inward
        switch (face) {
            case 0: sx = 0;      sy = static_cast<int>(u01() * (gy-1)); sz = static_cast<int>(u01() * (gz-1)); ddx =  1; break;
            case 1: sx = gx - 1; sy = static_cast<int>(u01() * (gy-1)); sz = static_cast<int>(u01() * (gz-1)); ddx = -1; break;
            case 2: sy = 0;      sx = static_cast<int>(u01() * (gx-1)); sz = static_cast<int>(u01() * (gz-1)); ddy =  1; break;
            case 3: sy = gy - 1; sx = static_cast<int>(u01() * (gx-1)); sz = static_cast<int>(u01() * (gz-1)); ddy = -1; break;
            case 4: sz = 0;      sx = static_cast<int>(u01() * (gx-1)); sy = static_cast<int>(u01() * (gy-1)); ddz =  1; break;
            case 5: sz = gz - 1; sx = static_cast<int>(u01() * (gx-1)); sy = static_cast<int>(u01() * (gy-1)); ddz = -1; break;
        }

        // Find a stroma entry point (search inward if face voxel is blocked)
        for (int step = 0; step < 20; step++) {
            if (is_stroma(sx, sy, sz)) break;
            sx += static_cast<int>(ddx);
            sy += static_cast<int>(ddy);
            sz += static_cast<int>(ddz);
        }
        if (!is_stroma(sx, sy, sz)) continue;

        walk_queue.push_back({sx, sy, sz, ddx, ddy, ddz, trunk_max_len, 0});
    }

    // Process walk queue (BFS-like: trunks spawn branches, branches spawn sub-branches)
    int total_placed = 0;
    int branches_spawned = 0;

    while (!walk_queue.empty()) {
        WalkState ws = walk_queue.back();
        walk_queue.pop_back();

        float persistence = (ws.depth == 0) ? persistence_trunk : persistence_branch;
        float cur_dx = ws.dx, cur_dy = ws.dy, cur_dz = ws.dz;
        int cx = ws.x, cy = ws.y, cz = ws.z;

        for (int step = 0; step < ws.steps_remaining; step++) {
            // Place phalanx at current position
            if (is_stroma(cx, cy, cz)) {
                place_phalanx(cx, cy, cz, cur_dx, cur_dy, cur_dz, 0);
                total_placed++;
            }

            // Check branching
            if (ws.depth < max_branch_depth) {
                int cur_idx = vidx(cx, cy, cz);
                float dw = (cur_idx >= 0 && cur_idx < gx*gy*gz) ? dist_to_wall[cur_idx] : 30.0f;
                float p_branch = p_branch_base + p_branch_periductal * std::exp(-dw / periductal_decay);
                if (u01() < p_branch) {
                    // Pick a random perpendicular-ish direction for branch
                    float bx = u01() - 0.5f;
                    float by = u01() - 0.5f;
                    float bz = u01() - 0.5f;
                    // Reduce component along trunk direction to make branch roughly perpendicular
                    float dot = bx * cur_dx + by * cur_dy + bz * cur_dz;
                    bx -= dot * cur_dx * 0.7f;
                    by -= dot * cur_dy * 0.7f;
                    bz -= dot * cur_dz * 0.7f;
                    float len = std::sqrt(bx*bx + by*by + bz*bz);
                    if (len > 0.01f) { bx /= len; by /= len; bz /= len; }

                    int b_max = branch_max_len / (ws.depth + 1);  // shorter at deeper levels
                    walk_queue.push_back({cx, cy, cz, bx, by, bz, b_max, ws.depth + 1});
                    branches_spawned++;
                }
            }

            // Move: pick next position
            // With probability=persistence, keep current direction + small perturbation
            // Otherwise, pick a random 26-neighbor direction
            float nx, ny, nz;
            if (u01() < persistence) {
                // Perturb current direction slightly
                nx = cur_dx + (u01() - 0.5f) * 0.6f;
                ny = cur_dy + (u01() - 0.5f) * 0.6f;
                nz = cur_dz + (u01() - 0.5f) * 0.6f;
            } else {
                // Random tumble
                nx = u01() - 0.5f;
                ny = u01() - 0.5f;
                nz = u01() - 0.5f;
            }
            // Normalize
            float nlen = std::sqrt(nx*nx + ny*ny + nz*nz);
            if (nlen > 0.01f) { nx /= nlen; ny /= nlen; nz /= nlen; }
            cur_dx = nx; cur_dy = ny; cur_dz = nz;

            // Step to nearest voxel in direction
            int next_x = cx + static_cast<int>(std::round(cur_dx));
            int next_y = cy + static_cast<int>(std::round(cur_dy));
            int next_z = cz + static_cast<int>(std::round(cur_dz));

            // Boundary reflection
            if (next_x < 0) { next_x = 0; cur_dx = std::abs(cur_dx); }
            if (next_x >= gx) { next_x = gx - 1; cur_dx = -std::abs(cur_dx); }
            if (next_y < 0) { next_y = 0; cur_dy = std::abs(cur_dy); }
            if (next_y >= gy) { next_y = gy - 1; cur_dy = -std::abs(cur_dy); }
            if (next_z < 0) { next_z = 0; cur_dz = std::abs(cur_dz); }
            if (next_z >= gz) { next_z = gz - 1; cur_dz = -std::abs(cur_dz); }

            // If next position is not stroma, try up to 5 random neighbors
            if (!is_stroma(next_x, next_y, next_z)) {
                bool found = false;
                for (int attempt = 0; attempt < 5; attempt++) {
                    int rx = cx + (static_cast<int>(u01() * 3) - 1);
                    int ry = cy + (static_cast<int>(u01() * 3) - 1);
                    int rz = cz + (static_cast<int>(u01() * 3) - 1);
                    if (is_stroma(rx, ry, rz)) {
                        next_x = rx; next_y = ry; next_z = rz;
                        found = true;
                        break;
                    }
                }
                if (!found) break;  // walk terminates — stuck in non-stroma
            }

            cx = next_x; cy = next_y; cz = next_z;
        }
    }

    // ── Phase 3: Capillary fill in lobular stroma ────────────────────────
    // Place additional short random walks seeded from stroma voxels near
    // septum boundaries, weighted by septum_density.
    const int n_capillary_seeds = std::max(1, (gx * gy * gz) / 50000);  // ~160 for 200^3
    const int capillary_walk_len = 20;

    for (int cs = 0; cs < n_capillary_seeds; cs++) {
        // Pick a random stroma voxel, weighted by septum density
        int tries = 0;
        int sx, sy, sz;
        do {
            sx = static_cast<int>(u01() * gx) % gx;
            sy = static_cast<int>(u01() * gy) % gy;
            sz = static_cast<int>(u01() * gz) % gz;
            tries++;
        } while (tries < 100 &&
                 (!is_stroma(sx, sy, sz) ||
                  u01() > (0.2f + septum_density[vidx(sx, sy, sz)])));

        if (!is_stroma(sx, sy, sz)) continue;

        // Short random walk
        float dx = u01() - 0.5f, dy = u01() - 0.5f, dz = u01() - 0.5f;
        float len = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (len > 0.01f) { dx /= len; dy /= len; dz /= len; }

        int cx = sx, cy = sy, cz = sz;
        for (int step = 0; step < capillary_walk_len; step++) {
            if (is_stroma(cx, cy, cz)) {
                place_phalanx(cx, cy, cz, dx, dy, dz, 0);
                total_placed++;
            }

            // Low persistence walk
            if (u01() < 0.4f) {
                dx = u01() - 0.5f;
                dy = u01() - 0.5f;
                dz = u01() - 0.5f;
                float l = std::sqrt(dx*dx + dy*dy + dz*dz);
                if (l > 0.01f) { dx /= l; dy /= l; dz /= l; }
            }

            int nx = cx + static_cast<int>(std::round(dx));
            int ny = cy + static_cast<int>(std::round(dy));
            int nz = cz + static_cast<int>(std::round(dz));
            if (!is_stroma(nx, ny, nz)) break;
            cx = nx; cy = ny; cz = nz;
        }
    }

    std::cout << "  Periductal vasculature: " << total_placed << " PHALANX cells placed ("
              << n_entries << " trunks, " << branches_spawned << " branches, "
              << n_capillary_seeds << " capillary seeds)" << std::endl;
}

// ============================================================================
// Ductal-Aware Initialization (-i 1)
// ============================================================================

// Compute distance-to-wall field via multi-source BFS from TISSUE_WALL voxels.
// Returns float array [total_voxels], capped at max_dist. Non-stroma voxels get -1.
static std::vector<float> compute_dist_to_wall(
    const std::vector<uint8_t>& wall_mask,
    int gx, int gy, int gz, float max_dist = 30.0f)
{
    const int total = gx * gy * gz;
    std::vector<float> dist(total, max_dist + 1.0f);

    // Seed queue with all TISSUE_WALL voxels
    std::queue<int> q;
    for (int i = 0; i < total; i++) {
        if (wall_mask[i] == TISSUE_WALL) {
            dist[i] = 0.0f;
            q.push(i);
        } else if (wall_mask[i] == TISSUE_LUMEN) {
            dist[i] = -1.0f;  // mark lumen as invalid
        }
    }

    // BFS (6-connected) into stroma only
    const int offsets[6][3] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
    while (!q.empty()) {
        int idx = q.front(); q.pop();
        int z = idx / (gx * gy);
        int y = (idx - z * gx * gy) / gx;
        int x = idx - z * gx * gy - y * gx;
        float cur_d = dist[idx];
        if (cur_d >= max_dist) continue;

        for (int d = 0; d < 6; d++) {
            int nx = x + offsets[d][0];
            int ny = y + offsets[d][1];
            int nz = z + offsets[d][2];
            if (nx < 0 || nx >= gx || ny < 0 || ny >= gy || nz < 0 || nz >= gz) continue;
            int nidx = nx + ny * gx + nz * gx * gy;
            if (wall_mask[nidx] != TISSUE_NONE) continue;  // only propagate into stroma
            float new_d = cur_d + 1.0f;
            if (new_d < dist[nidx]) {
                dist[nidx] = new_d;
                q.push(nidx);
            }
        }
    }

    // Cap remaining stroma voxels that were unreached
    for (int i = 0; i < total; i++) {
        if (wall_mask[i] == TISSUE_NONE && dist[i] > max_dist) {
            dist[i] = max_dist;
        }
    }

    return dist;
}

// Find terminal ductules (nodes with no children)
static std::vector<int> find_terminal_nodes(const std::vector<DuctNode>& nodes) {
    std::unordered_set<int> parents;
    for (size_t i = 0; i < nodes.size(); i++) {
        if (nodes[i].parent >= 0) parents.insert(nodes[i].parent);
    }
    std::vector<int> terminals;
    for (size_t i = 0; i < nodes.size(); i++) {
        if (parents.find(static_cast<int>(i)) == parents.end()) {
            terminals.push_back(static_cast<int>(i));
        }
    }
    return terminals;
}

// Collect lumen voxels near a duct segment (between node and its parent)
static void collect_segment_lumen(const DuctNode& a, const DuctNode& b,
                                   const std::vector<uint8_t>& wall_mask,
                                   int gx, int gy, int gz,
                                   std::vector<int>& out_indices)
{
    float dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z;
    float seg_len = std::sqrt(dx*dx + dy*dy + dz*dz);
    if (seg_len < 0.01f) return;

    int n_samples = std::max(1, static_cast<int>(std::ceil(seg_len)));
    for (int s = 0; s <= n_samples; s++) {
        float t = static_cast<float>(s) / static_cast<float>(n_samples);
        float cx = a.x + dx * t;
        float cy = a.y + dy * t;
        float cz = a.z + dz * t;
        float r  = a.radius + (b.radius - a.radius) * t;

        int r_ceil = static_cast<int>(std::ceil(r)) + 1;
        int ix = static_cast<int>(std::round(cx));
        int iy = static_cast<int>(std::round(cy));
        int iz = static_cast<int>(std::round(cz));

        for (int vz = iz - r_ceil; vz <= iz + r_ceil; vz++) {
            for (int vy = iy - r_ceil; vy <= iy + r_ceil; vy++) {
                for (int vx = ix - r_ceil; vx <= ix + r_ceil; vx++) {
                    if (vx < 0 || vx >= gx || vy < 0 || vy >= gy || vz < 0 || vz >= gz) continue;
                    int idx = vx + vy * gx + vz * gx * gy;
                    if (wall_mask[idx] == TISSUE_LUMEN) {
                        out_indices.push_back(idx);
                    }
                }
            }
        }
    }
}

// Initialize cancer cells inside duct lumen, starting from a terminal ductule
// and walking up the tree until we hit the target cell count.
static int initializeCancerInDuct(
    flamegpu::AgentVector& cancer_agents,
    const DuctalNetwork& net,
    int target_cells,
    float stem_div_interval,
    float progenitor_div_interval,
    int progenitor_div_max,
    float senescent_mean_life,
    std::vector<double>& celltype_cdf,
    std::vector<std::vector<int>>& occupied,
    std::mt19937& rng,
    float& seed_cx, float& seed_cy, float& seed_cz)  // output: cancer seed center
{
    const int gx = net.grid_x, gy = net.grid_y, gz = net.grid_z;
    const auto& nodes = net.nodes;
    const auto& wall_mask = net.wall_mask;

    // Find terminal nodes and pick one randomly
    std::vector<int> terminals = find_terminal_nodes(nodes);
    if (terminals.empty()) {
        std::cerr << "[initializeCancerInDuct] ERROR: No terminal ductules found!" << std::endl;
        return 0;
    }
    std::uniform_int_distribution<int> term_dist(0, static_cast<int>(terminals.size()) - 1);
    int seed_node_idx = terminals[term_dist(rng)];

    seed_cx = nodes[seed_node_idx].x;
    seed_cy = nodes[seed_node_idx].y;
    seed_cz = nodes[seed_node_idx].z;

    std::cout << "  Cancer seed: terminal node " << seed_node_idx
              << " at (" << seed_cx << ", " << seed_cy
              << ", " << seed_cz << ") gen=" << nodes[seed_node_idx].generation
              << " radius=" << nodes[seed_node_idx].radius << std::endl;

    // Walk up tree from terminal, collecting lumen voxels for each segment
    // Use a set to deduplicate (segments overlap at shared radii)
    std::unordered_set<int> lumen_set;
    int cur = seed_node_idx;
    while (cur >= 0 && static_cast<int>(lumen_set.size()) < target_cells * 2) {
        if (nodes[cur].parent >= 0) {
            std::vector<int> seg_voxels;
            collect_segment_lumen(nodes[nodes[cur].parent], nodes[cur],
                                  wall_mask, gx, gy, gz, seg_voxels);
            for (int idx : seg_voxels) lumen_set.insert(idx);
        }
        // Also collect siblings of current node (fill branching junctions)
        cur = nodes[cur].parent;
    }

    // Convert to shuffled vector for random sampling
    std::vector<int> lumen_voxels(lumen_set.begin(), lumen_set.end());
    std::shuffle(lumen_voxels.begin(), lumen_voxels.end(), rng);

    int placed = 0;
    int max_place = std::min(target_cells, static_cast<int>(lumen_voxels.size()));

    std::cout << "  Available lumen voxels: " << lumen_voxels.size()
              << ", target: " << target_cells << ", will place: " << max_place << std::endl;

    for (int vi = 0; vi < max_place; vi++) {
        int idx = lumen_voxels[vi];
        int z = idx / (gx * gy);
        int y = (idx - z * gx * gy) / gx;
        int x = idx - z * gx * gy - y * gx;

        // Sample from CDF (same as initializeCancerCellsRandom)
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        double p = u01(rng);
        int i = static_cast<int>(std::lower_bound(celltype_cdf.begin(), celltype_cdf.end(), p) - celltype_cdf.begin());
        int cell_state;
        int div_cd = 0;
        int div = 0;
        int divide_flag = 0;
        int is_stem = 0;

        if (i > progenitor_div_max + 1) continue;  // skip overflow (shouldn't happen with CDF)

        cancer_agents.push_back();
        flamegpu::AgentVector::Agent agent = cancer_agents.back();

        if (i == 0) {
            cell_state = CANCER_STEM;
            float rf = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng);
            div_cd = static_cast<int>(stem_div_interval * rf) + 1;
            divide_flag = 1;
            is_stem = 1;
        } else if (i == progenitor_div_max + 1) {
            cell_state = CANCER_SENESCENT;
        } else {
            cell_state = CANCER_PROGENITOR;
            div = progenitor_div_max + 1 - i;
            float rf = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng);
            div_cd = static_cast<int>(progenitor_div_interval * rf) + 1;
            divide_flag = 1;
        }

        const int id = agent.getID();
        agent.setVariable<int>("x", x);
        agent.setVariable<int>("y", y);
        agent.setVariable<int>("z", z);
        agent.setVariable<int>("cell_state", cell_state);
        agent.setVariable<int>("divideCD", div_cd);
        agent.setVariable<int>("divideFlag", divide_flag);
        agent.setVariable<int>("divideCountRemaining", div);
        agent.setVariable<unsigned int>("stemID", is_stem ? id : 0);

        if (cell_state == CANCER_SENESCENT) {
            float r = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng);
            int sen_life = static_cast<int>(-senescent_mean_life * logf(r + 0.0001f) + 0.5f);
            agent.setVariable<int>("life", sen_life > 0 ? sen_life : 1);
        }

        occupied[idx][0] = 1;
        placed++;
    }

    return placed;
}

// ============================================================================
// initializeWholePDAC: Production initialization for -i 1
// ============================================================================
void initializeWholePDAC(
    flamegpu::CUDASimulation& simulation,
    flamegpu::ModelDescription& model,
    const SimulationConfig& config,
    const LymphCentralWrapper& lymph,
    const DuctalNetwork& ductal_network)
{
    std::cout << "\n=== Initializing Agents (Ductal Mode) ===" << std::endl;
    const int gx = config.grid_x, gy = config.grid_y, gz = config.grid_z;
    const int total_voxels = gx * gy * gz;
    const auto& wall_mask = ductal_network.wall_mask;

    // RNG seeded from config
    std::mt19937 rng(config.random_seed);

    // -----------------------------------------------------------------------
    // QSP state & parameters (same as initializeSphere)
    // -----------------------------------------------------------------------
    QSPState qsp = lymph.get_state_for_abm();
    std::cout << "  QSP tumor volume  : " << qsp.tum_vol    << " cm^3" << std::endl;

    const double avogadros = 6.022140857e23;
    const double cc_SI     = qsp.cc_tumor / avogadros;
    const double eps = 1e-30;

    const double p_th   = 0.03 * qsp.th_tumor   / (qsp.th_tumor + cc_SI + eps);
    const double p_mdsc =        qsp.mdsc_tumor  / (qsp.mdsc_tumor + cc_SI + eps);
    const double p_treg = 0.0;
    const double p_mac  =        qsp.m2_tumor / (qsp.m2_tumor + cc_SI + eps);
    double p_fib  =        qsp.caf_tumor / (qsp.caf_tumor + cc_SI + eps);
    if (p_fib > 0.1) p_fib = 0.1;
    else if (p_fib < 0.001) p_fib = 0.001;

    std::cout << "  p_th=" << p_th << "  p_mdsc=" << p_mdsc
              << "  p_mac=" << p_mac << "  p_fib=" << p_fib << std::endl;

    // Cell-lifecycle parameters
    const float stem_div         = model.Environment().getProperty<float>("PARAM_FLOAT_CANCER_CELL_STEM_DIV_INTERVAL_SLICE");
    const float prog_div         = model.Environment().getProperty<float>("PARAM_FLOAT_CANCER_CELL_PROGENITOR_DIV_INTERVAL_SLICE");
    const int   prog_max         = model.Environment().getProperty<int>("PARAM_PROG_DIV_MAX");
    const float cancer_sen_life  = model.Environment().getProperty<float>("PARAM_CANCER_SENESCENT_MEAN_LIFE");
    const float treg_life        = model.Environment().getProperty<float>("PARAM_TCD4_LIFE_MEAN_SLICE");
    const float treg_life_sd     = model.Environment().getProperty<float>("PARAM_TCELL_LIFESPAN_SD_SLICE");
    const int   treg_div_limit   = model.Environment().getProperty<int>("PARAM_TCD4_DIV_LIMIT");
    const int   treg_div_interval = model.Environment().getProperty<int>("PARAM_TCD4_DIV_INTERNAL");
    const float mdsc_life        = model.Environment().getProperty<float>("PARAM_MDSC_LIFE_MEAN_SLICE");
    const float mac_life         = model.Environment().getProperty<float>("PARAM_MAC_LIFE_MEAN");
    const float fib_life         = model.Environment().getProperty<float>("PARAM_FIB_LIFE_MEAN");
    const float vas_branch_prob  = model.Environment().getProperty<float>("PARAM_VAS_BRANCH_PROB");
    const int   vas_min_neighbor = static_cast<int>(model.Environment().getProperty<float>("PARAM_VAS_MIN_NEIGHBOR"));
    std::vector<double> celltype_cdf = get_celltype_cdf(model);

    // -----------------------------------------------------------------------
    // Precompute helper fields
    // -----------------------------------------------------------------------
    std::cout << "  Computing distance-to-wall field..." << std::endl;
    std::vector<float> dist_to_wall = compute_dist_to_wall(wall_mask, gx, gy, gz);

    // Count tissue types
    int n_lumen = 0, n_wall = 0, n_stroma = 0;
    for (int i = 0; i < total_voxels; i++) {
        if (wall_mask[i] == TISSUE_LUMEN) n_lumen++;
        else if (wall_mask[i] == TISSUE_WALL) n_wall++;
        else n_stroma++;
    }
    std::cout << "  Tissue: " << n_lumen << " lumen, " << n_wall << " wall, "
              << n_stroma << " stroma voxels" << std::endl;

    // Occupancy grid
    std::vector<std::vector<int>> occupied(total_voxels, std::vector<int>(3, 0));

    // -----------------------------------------------------------------------
    // 1. Cancer cells: fill a random duct branch lumen
    // -----------------------------------------------------------------------
    int target_cancer = 20000;  // default target; fill terminal branch + parents
    float tumor_cx = 0, tumor_cy = 0, tumor_cz = 0;  // filled by initializeCancerInDuct
    {
        flamegpu::AgentVector cancer_pop(model.Agent(AGENT_CANCER_CELL));
        int placed = initializeCancerInDuct(
            cancer_pop, ductal_network, target_cancer,
            stem_div, prog_div, prog_max, cancer_sen_life,
            celltype_cdf, occupied, rng,
            tumor_cx, tumor_cy, tumor_cz);
        std::cout << "  Cancer cells placed: " << placed << " (in duct lumen)" << std::endl;
        simulation.setPopulationData(cancer_pop);
    }

    // -----------------------------------------------------------------------
    // 1b. Break duct walls at tumor site
    //     Download face flags from GPU, clear wall faces at cancer-occupied
    //     lumen voxels (randomly, not every face), then re-upload.
    //     This represents tumor invasion through the duct wall.
    // -----------------------------------------------------------------------
    {
        const float p_break = 0.7f;  // probability of breaking each wall face at tumor
        std::vector<uint8_t> h_face_flags(total_voxels);
        cudaMemcpy(h_face_flags.data(), ductal_network.d_face_flags,
                   total_voxels * sizeof(uint8_t), cudaMemcpyDeviceToHost);

        // Face direction table: for each of 6 directions, the offset and the
        // bit on the source voxel and the complementary bit on the neighbor.
        struct FaceDir { int dx, dy, dz; uint8_t src_bit, dst_bit; };
        const FaceDir dirs[6] = {
            {-1,  0,  0, FACE_NEG_X, FACE_POS_X},
            { 1,  0,  0, FACE_POS_X, FACE_NEG_X},
            { 0, -1,  0, FACE_NEG_Y, FACE_POS_Y},
            { 0,  1,  0, FACE_POS_Y, FACE_NEG_Y},
            { 0,  0, -1, FACE_NEG_Z, FACE_POS_Z},
            { 0,  0,  1, FACE_POS_Z, FACE_NEG_Z},
        };

        int faces_broken = 0;
        for (int idx = 0; idx < total_voxels; idx++) {
            if (occupied[idx][0] == 0) continue;  // no cancer here
            if (h_face_flags[idx] == 0) continue;  // no wall faces to break

            int z = idx / (gx * gy);
            int y = (idx - z * gx * gy) / gx;
            int x = idx - z * gx * gy - y * gx;

            for (int d = 0; d < 6; d++) {
                if (!(h_face_flags[idx] & dirs[d].src_bit)) continue;  // no wall on this face

                // Randomly decide whether to break this face
                float r = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng);
                if (r >= p_break) continue;

                int nx = x + dirs[d].dx;
                int ny = y + dirs[d].dy;
                int nz = z + dirs[d].dz;

                // Clear source side
                h_face_flags[idx] &= ~dirs[d].src_bit;

                // Clear neighbor side (if in bounds)
                if (nx >= 0 && nx < gx && ny >= 0 && ny < gy && nz >= 0 && nz < gz) {
                    int nidx = nx + ny * gx + nz * gx * gy;
                    h_face_flags[nidx] &= ~dirs[d].dst_bit;
                }
                faces_broken++;
            }
        }

        // Re-upload modified face flags to GPU
        cudaMemcpy(ductal_network.d_face_flags, h_face_flags.data(),
                   total_voxels * sizeof(uint8_t), cudaMemcpyHostToDevice);

        std::cout << "  Duct walls broken at tumor: " << faces_broken << " faces cleared" << std::endl;
    }

    // -----------------------------------------------------------------------
    // 2. T cells: none at init (recruited during simulation)
    // -----------------------------------------------------------------------
    {
        flamegpu::AgentVector tcell_pop(model.Agent(AGENT_TCELL));
        simulation.setPopulationData(tcell_pop);
    }

    // -----------------------------------------------------------------------
    // 3. TH and TReg cells: stroma only, peritumoral enrichment
    // -----------------------------------------------------------------------
    // Peritumoral enrichment: immune cells are denser near the tumor
    const float peritumoral_radius = 30.0f;  // voxels (~600 um)
    const float peritumoral_boost = 3.0f;    // multiplier at tumor edge
    const float peritumoral_decay = 15.0f;   // distance decay scale
    auto peritumoral_factor = [&](int x, int y, int z) -> float {
        float dx = x - tumor_cx, dy = y - tumor_cy, dz = z - tumor_cz;
        float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (dist > peritumoral_radius * 2.0f) return 1.0f;
        return 1.0f + peritumoral_boost * std::exp(-dist / peritumoral_decay);
    };

    {
        flamegpu::AgentVector treg_pop(model.Agent(AGENT_TREG));

        // TH cells — stroma-filtered with peritumoral enrichment
        int th_placed = 0;
        for (int z = 0; z < gz; z++) {
            for (int y = 0; y < gy; y++) {
                for (int x = 0; x < gx; x++) {
                    int idx = x + y * gx + z * gx * gy;
                    if (wall_mask[idx] != TISSUE_NONE) continue;
                    float p_eff = static_cast<float>(p_th) * peritumoral_factor(x, y, z);
                    float rnd = static_cast<float>(rand()) / RAND_MAX;
                    if (rnd >= p_eff) continue;

                    // Box-Muller normal life
                    float u1 = static_cast<float>(rand()) / RAND_MAX;
                    float u2 = static_cast<float>(rand()) / RAND_MAX;
                    float z0 = std::sqrt(-2.0f * std::log(u1 + 1e-10f)) * std::cos(2.0f * M_PI * u2);
                    int life = static_cast<int>(treg_life + z0 * treg_life_sd + 0.5f);
                    if (life < 1) life = 1;

                    treg_pop.push_back();
                    auto agent = treg_pop.back();
                    agent.setVariable<int>("x", x);
                    agent.setVariable<int>("y", y);
                    agent.setVariable<int>("z", z);
                    agent.setVariable<int>("cell_state", TCD4_TH);
                    agent.setVariable<int>("life", life);
                    agent.setVariable<int>("divide_cd", rand() % std::max(1, treg_div_interval));
                    agent.setVariable<int>("divide_limit", treg_div_limit);
                    agent.setVariable<int>("divide_flag", 0);
                    agent.setVariable<int>("intent_action", 0);
                    th_placed++;
                }
            }
        }

        // TReg cells (p_treg = 0 currently, but loop is ready)
        int treg_placed = 0;
        for (int z = 0; z < gz; z++) {
            for (int y = 0; y < gy; y++) {
                for (int x = 0; x < gx; x++) {
                    int idx = x + y * gx + z * gx * gy;
                    if (wall_mask[idx] != TISSUE_NONE) continue;
                    float p_eff = static_cast<float>(p_treg) * peritumoral_factor(x, y, z);
                    float rnd = static_cast<float>(rand()) / RAND_MAX;
                    if (rnd >= p_eff) continue;

                    float u1 = static_cast<float>(rand()) / RAND_MAX;
                    float u2 = static_cast<float>(rand()) / RAND_MAX;
                    float z0 = std::sqrt(-2.0f * std::log(u1 + 1e-10f)) * std::cos(2.0f * M_PI * u2);
                    int life = static_cast<int>(treg_life + z0 * treg_life_sd + 0.5f);
                    if (life < 1) life = 1;

                    treg_pop.push_back();
                    auto agent = treg_pop.back();
                    agent.setVariable<int>("x", x);
                    agent.setVariable<int>("y", y);
                    agent.setVariable<int>("z", z);
                    agent.setVariable<int>("cell_state", TCD4_TREG);
                    agent.setVariable<int>("life", life);
                    agent.setVariable<int>("divide_cd", rand() % std::max(1, treg_div_interval));
                    agent.setVariable<int>("divide_limit", treg_div_limit);
                    agent.setVariable<int>("divide_flag", 0);
                    agent.setVariable<int>("intent_action", 0);
                    treg_placed++;
                }
            }
        }
        std::cout << "  TH cells placed: " << th_placed << "  TReg cells placed: " << treg_placed
                  << " (stroma only)" << std::endl;
        simulation.setPopulationData(treg_pop);
    }

    // -----------------------------------------------------------------------
    // 4. MDSCs: stroma only, peritumoral enrichment
    // -----------------------------------------------------------------------
    {
        flamegpu::AgentVector mdsc_pop(model.Agent(AGENT_MDSC));
        int mdsc_placed = 0;
        for (int z = 0; z < gz; z++) {
            for (int y = 0; y < gy; y++) {
                for (int x = 0; x < gx; x++) {
                    int idx = x + y * gx + z * gx * gy;
                    if (wall_mask[idx] != TISSUE_NONE) continue;
                    float p_eff = static_cast<float>(p_mdsc) * peritumoral_factor(x, y, z);
                    float rnd = static_cast<float>(rand()) / RAND_MAX;
                    if (rnd >= p_eff) continue;

                    float life_rnd = static_cast<float>(rand()) / RAND_MAX;
                    int life = static_cast<int>(mdsc_life * std::log(1.0f / (life_rnd + 1e-4f)) + 0.5f);
                    if (life < 1) life = 1;

                    mdsc_pop.push_back();
                    auto agent = mdsc_pop.back();
                    agent.setVariable<int>("x", x);
                    agent.setVariable<int>("y", y);
                    agent.setVariable<int>("z", z);
                    agent.setVariable<int>("life", life);
                    agent.setVariable<int>("intent_action", 0);
                    agent.setVariable<int>("target_x", -1);
                    agent.setVariable<int>("target_y", -1);
                    agent.setVariable<int>("target_z", -1);
                    mdsc_placed++;
                }
            }
        }
        std::cout << "  MDSCs placed: " << mdsc_placed << " (stroma only)" << std::endl;
        simulation.setPopulationData(mdsc_pop);
    }

    // -----------------------------------------------------------------------
    // 5. Macrophages: stroma only, M2, peritumoral enrichment
    // -----------------------------------------------------------------------
    {
        flamegpu::AgentVector mac_pop(model.Agent(AGENT_MACROPHAGE));
        int mac_placed = 0;
        for (int z = 0; z < gz; z++) {
            for (int y = 0; y < gy; y++) {
                for (int x = 0; x < gx; x++) {
                    int idx = x + y * gx + z * gx * gy;
                    if (wall_mask[idx] != TISSUE_NONE) continue;
                    float p_eff = static_cast<float>(p_mac) * peritumoral_factor(x, y, z);
                    float rnd = static_cast<float>(rand()) / RAND_MAX;
                    if (rnd >= p_eff) continue;

                    float life_rnd = static_cast<float>(rand()) / RAND_MAX;
                    int life = static_cast<int>(mac_life * std::log(1.0f / (life_rnd + 1e-4f)) + 0.5f);
                    if (life < 1) life = 1;

                    mac_pop.push_back();
                    auto agent = mac_pop.back();
                    agent.setVariable<int>("x", x);
                    agent.setVariable<int>("y", y);
                    agent.setVariable<int>("z", z);
                    agent.setVariable<int>("life", life);
                    agent.setVariable<int>("cell_state", MAC_M2);
                    mac_placed++;
                }
            }
        }
        std::cout << "  Macrophages placed: " << mac_placed << " (stroma only, M2)" << std::endl;
        simulation.setPopulationData(mac_pop);
    }

    // -----------------------------------------------------------------------
    // 6. Fibroblasts: stroma only, weighted by proximity to duct walls + septum
    // -----------------------------------------------------------------------
    {
        flamegpu::AgentVector fib_pop(model.Agent(AGENT_FIBROBLAST));
        int fib_placed = 0;
        const int init_chain_len = 3;
        const float fib_wall_boost = 3.0f;     // multiplier for near-wall density
        const float fib_wall_decay = 5.0f;     // distance scale (voxels)
        const float fib_septum_boost = 2.0f;   // multiplier for septum boundary density
        const auto& septum = ductal_network.septum_density;

        for (int z = 0; z < gz; z++) {
            for (int y = 0; y < gy; y++) {
                for (int x = 0; x < gx; x++) {
                    int idx = x + y * gx + z * gx * gy;
                    if (wall_mask[idx] != TISSUE_NONE) continue;

                    // Weighted probability: higher near walls and at septum boundaries
                    float dw = dist_to_wall[idx];
                    float wall_factor = 1.0f + fib_wall_boost * std::exp(-dw / fib_wall_decay);
                    float sept_factor = 1.0f + fib_septum_boost * septum[idx];
                    float p_eff = static_cast<float>(p_fib) * wall_factor * sept_factor;
                    if (p_eff > 0.5f) p_eff = 0.5f;  // cap

                    float rnd = static_cast<float>(rand()) / RAND_MAX;
                    if (rnd >= p_eff) continue;

                    // Find 3 chain positions in stroma
                    int c1x, c1y, c1z;
                    if (!findFreeAdjacentStroma(x, y, z, gx, gy, gz, occupied, wall_mask, c1x, c1y, c1z)) continue;
                    int c2x, c2y, c2z;
                    if (!findFreeAdjacentStroma(c1x, c1y, c1z, gx, gy, gz, occupied, wall_mask, c2x, c2y, c2z)) continue;
                    int c3x, c3y, c3z;
                    if (!findFreeAdjacentStroma(c2x, c2y, c2z, gx, gy, gz, occupied, wall_mask, c3x, c3y, c3z)) continue;
                    if (c3x == c1x && c3y == c1y && c3z == c1z) continue;

                    int idx1 = c1x + c1y * gx + c1z * gx * gy;
                    int idx2 = c2x + c2y * gx + c2z * gx * gy;
                    int idx3 = c3x + c3y * gx + c3z * gx * gy;
                    occupied[idx1][1] = 1;
                    occupied[idx2][1] = 1;
                    occupied[idx3][1] = 1;

                    float life_rnd = static_cast<float>(rand()) / RAND_MAX;
                    int life = static_cast<int>(fib_life * std::log(1.0f / (life_rnd + 1e-4f)) + 0.5f);
                    if (life < 1) life = 1;

                    int cx_arr[MAX_FIB_CHAIN_LENGTH] = {c1x, c2x, c3x, 0, 0};
                    int cy_arr[MAX_FIB_CHAIN_LENGTH] = {c1y, c2y, c3y, 0, 0};
                    int cz_arr[MAX_FIB_CHAIN_LENGTH] = {c1z, c2z, c3z, 0, 0};

                    fib_pop.push_back();
                    auto agent = fib_pop.back();
                    agent.setVariable<int>("x", c1x);
                    agent.setVariable<int>("y", c1y);
                    agent.setVariable<int>("z", c1z);
                    std::array<int, MAX_FIB_CHAIN_LENGTH> asx, asy, asz;
                    for (int j = 0; j < MAX_FIB_CHAIN_LENGTH; j++) { asx[j] = cx_arr[j]; asy[j] = cy_arr[j]; asz[j] = cz_arr[j]; }
                    agent.setVariable<int, MAX_FIB_CHAIN_LENGTH>("seg_x", asx);
                    agent.setVariable<int, MAX_FIB_CHAIN_LENGTH>("seg_y", asy);
                    agent.setVariable<int, MAX_FIB_CHAIN_LENGTH>("seg_z", asz);
                    agent.setVariable<int>("chain_len", init_chain_len);
                    agent.setVariable<int>("cell_state", FIB_NORMAL);
                    agent.setVariable<int>("life", life);
                    fib_placed++;
                }
            }
        }
        std::cout << "  Fibroblasts placed: " << fib_placed << " (stroma, wall+septum weighted)" << std::endl;
        simulation.setPopulationData(fib_pop);
    }

    // -----------------------------------------------------------------------
    // 7. Vasculature: periductal vascular tree
    // -----------------------------------------------------------------------
    {
        flamegpu::AgentVector vascular_vec(model.Agent(AGENT_VASCULAR));
        initializeVascularPeriductal(
            vascular_vec, gx, gy, gz,
            wall_mask, dist_to_wall, ductal_network.septum_density,
            vas_branch_prob, rng);
        assignInitialVascularTips(
            vascular_vec, gx, gy, gz,
            vas_min_neighbor, config.random_seed);
        simulation.setPopulationData(vascular_vec);
    }

    std::cout << "Ductal-mode agent initialization complete\n" << std::endl;
}

// ============================================================================
// Neighbor Scan Test Initialization (init_method=2)
// ============================================================================
// Creates exactly 3 agents in an 11^3 grid for testing neighbor detection:
// - M1 macrophage at (5,5,5)
// - PROGENITOR cancer at (5,5,6) [adjacent to mac]
// - EFFECTOR T cell at (5,5,7) [not adjacent to mac]
//
// Expected after correct broadcast/scan:
// - Cancer at (5,5,6): neighbor_Mac1_count = 1, neighbor_Teff_count = 0
// - Macrophage at (5,5,5): neighbor_cancer_count = 1
// - T cell at (5,5,7): neighbor_cancer_count = 1
void initializeNeighborTest(
    flamegpu::CUDASimulation& simulation,
    flamegpu::ModelDescription& model,
    const SimulationConfig& config)
{
    std::cout << "\n=== Initializing Neighbor Scan Test (init_method=2) ===" << std::endl;

    // 1. Create and place M1 macrophage at (5,5,5)
    {
        flamegpu::AgentVector mac_vec(model.Agent(AGENT_MACROPHAGE));
        mac_vec.push_back();
        auto agent = mac_vec.back();
        agent.setVariable<int>("x", 5);
        agent.setVariable<int>("y", 5);
        agent.setVariable<int>("z", 5);
        agent.setVariable<int>("cell_state", MAC_M1);
        agent.setVariable<int>("dead", 0);
        agent.setVariable<int>("life", 500);
        agent.setVariable<int>("neighbor_cancer_count", 0);
        agent.setVariable<int>("tumble", 0);
        agent.setVariable<float>("move_direction_x", 0.0f);
        agent.setVariable<float>("move_direction_y", 0.0f);
        agent.setVariable<float>("move_direction_z", 0.0f);
        simulation.setPopulationData(mac_vec);
    }
    std::cout << "[TEST] Placed M1 macrophage at (5,5,5)" << std::endl;

    // 2. Create and place TWO cancer cells:
    //    - PROGENITOR at (5,5,6) [adjacent to mac at (5,5,5)] - Moore neighborhood
    //    - PROGENITOR at (2,2,2) [FAR from mac] - Outside Moore neighborhood, tests radius limit
    {
        flamegpu::AgentVector cancer_vec(model.Agent(AGENT_CANCER_CELL));

        // Cancer 1: Adjacent (Moore neighbor)
        cancer_vec.push_back();
        auto agent = cancer_vec.back();
        agent.setVariable<int>("x", 5);
        agent.setVariable<int>("y", 5);
        agent.setVariable<int>("z", 6);
        agent.setVariable<int>("cell_state", CANCER_PROGENITOR);
        agent.setVariable<int>("dead", 0);
        agent.setVariable<float>("PDL1_syn", 0.0f);
        agent.setVariable<int>("neighbor_Teff_count", 0);
        agent.setVariable<int>("neighbor_Treg_count", 0);
        agent.setVariable<int>("neighbor_MDSC_count", 0);
        agent.setVariable<int>("neighbor_cancer_count", 0);
        agent.setVariable<int>("neighbor_Mac1_count", 0);
        agent.setVariable<int>("divideCD", 0);
        agent.setVariable<int>("divideFlag", 0);
        agent.setVariable<int>("divideCountRemaining", 0);
        agent.setVariable<unsigned int>("stemID", 0);

        // Cancer 2: Far away (outside spatial query radius)
        cancer_vec.push_back();
        {
            auto agent2 = cancer_vec.back();
            agent2.setVariable<int>("x", 2);
            agent2.setVariable<int>("y", 2);
            agent2.setVariable<int>("z", 2);
            agent2.setVariable<int>("cell_state", CANCER_PROGENITOR);
            agent2.setVariable<int>("dead", 0);
            agent2.setVariable<float>("PDL1_syn", 0.0f);
            agent2.setVariable<int>("neighbor_Teff_count", 0);
            agent2.setVariable<int>("neighbor_Treg_count", 0);
            agent2.setVariable<int>("neighbor_MDSC_count", 0);
            agent2.setVariable<int>("neighbor_cancer_count", 0);
            agent2.setVariable<int>("neighbor_Mac1_count", 0);
            agent2.setVariable<int>("divideCD", 0);
            agent2.setVariable<int>("divideFlag", 0);
            agent2.setVariable<int>("divideCountRemaining", 0);
            agent2.setVariable<unsigned int>("stemID", 0);
        }

        simulation.setPopulationData(cancer_vec);
    }
    std::cout << "[TEST] Placed PROGENITOR cancer at (5,5,6) [Moore neighbor to MAC]" << std::endl;
    std::cout << "[TEST] Placed PROGENITOR cancer at (2,2,2) [far from MAC, radius test]" << std::endl;

    // 3. Create and place EFFECTOR T cell at (5,5,7) [NOT adjacent to mac at (5,5,5)]
    {
        flamegpu::AgentVector tcell_vec(model.Agent(AGENT_TCELL));
        tcell_vec.push_back();
        auto agent = tcell_vec.back();
        agent.setVariable<int>("x", 5);
        agent.setVariable<int>("y", 5);
        agent.setVariable<int>("z", 7);
        agent.setVariable<int>("cell_state", T_CELL_EFF);
        agent.setVariable<int>("dead", 0);
        agent.setVariable<int>("life", 500);
        agent.setVariable<int>("neighbor_cancer_count", 0);
        agent.setVariable<int>("neighbor_Treg_count", 0);
        agent.setVariable<int>("divide_flag", 0);
        agent.setVariable<int>("divide_cd", 0);
        agent.setVariable<int>("divide_limit", 0);
        agent.setVariable<int>("tumble", 1);
        agent.setVariable<float>("IL2_release_remain", 0.0f);
        simulation.setPopulationData(tcell_vec);
    }
    std::cout << "[TEST] Placed EFFECTOR T cell at (5,5,7)" << std::endl;

    std::cout << "[TEST] Neighbor test initialization complete" << std::endl;
    std::cout << "[TEST] Expected results after neighbor scans:" << std::endl;
    std::cout << "  Cancer(5,5,6): neighbor_Mac1_count=1 (adjacent to MAC)" << std::endl;
    std::cout << "  Cancer(2,2,2): neighbor_Mac1_count=0 (too far from MAC, ~5.2 voxels away)" << std::endl;
    std::cout << "  MAC(5,5,5):    neighbor_cancer_count=1 (only sees (5,5,6), NOT (2,2,2))" << std::endl;
    std::cout << "  T(5,5,7):      neighbor_cancer_count=1 (only sees (5,5,6))" << std::endl;
}

} // namespace PDAC