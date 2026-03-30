#include "initialization.cuh"
#include "../core/common.cuh"
#include "../abm/gpu_param.h"
#include "../pde/pde_integration.cuh"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <string>
#include <array>
#include <algorithm>
#include <random>
#include <vector>

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
        } else if ((arg == "--initialization" || arg == "-i") && i + 1 < argc) {
            int val = std::atoi(argv[++i]);
            if (val == 0 || val == 1) {
                init_method = val;
            } else {
                std::cerr << "WARNING: Only -i 0 (simple) and -i 1 (structured) are supported. Ignoring -i " << val << std::endl;
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
                      << "  -G, --grid-output N      Grid output: 0=none, 1=ABM only, 2=PDE+ECM only, 3=both [default: 0]\n"
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
    std::cout << "Init: " << (init_method == 1 ? "Structured domain (-i 1)" : "QSP-seeded (-i 0)") << std::endl;

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
    int branch = 0,
    int initial_maturity = 100)
{
    agent.setVariable<int>("x", x);
    agent.setVariable<int>("y", y);
    agent.setVariable<int>("z", z);
    agent.setVariable<int>("cell_state", state);
    agent.setVariable<int>("persist_dir_x", static_cast<int>(move_dir_x));
    agent.setVariable<int>("persist_dir_y", static_cast<int>(move_dir_y));
    agent.setVariable<int>("persist_dir_z", static_cast<int>(move_dir_z));
    agent.setVariable<int>("intent_action", 0);  // INTENT_NONE
    agent.setVariable<int>("target_x", -1);
    agent.setVariable<int>("target_y", -1);
    agent.setVariable<int>("target_z", -1);
    agent.setVariable<unsigned int>("tip_id", tip_id);
    agent.setVariable<int>("mature_to_phalanx", 0);
    agent.setVariable<int>("branch", branch);
    agent.setVariable<int>("is_dysfunctional", 0);
    // Initial vessels get high maturity (pre-existing, stabilized)
    // TIP cells start at 0 since they're actively sprouting
    agent.setVariable<int>("maturity", (state == VAS_TIP) ? 0 : initial_maturity);
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

// ============================================================================
// Domain Structure Generation (Structured Init, -i 1)
// ============================================================================

std::vector<uint8_t> generate_domain_structure(
    int grid_x, int grid_y, int grid_z,
    float lobule_spacing, float septum_thickness,
    float tumor_radius_frac, float margin_thickness,
    unsigned int seed)
{
    const int total = grid_x * grid_y * grid_z;
    std::vector<uint8_t> voxel_type(total, VOXEL_STROMA);

    std::mt19937 rng(seed + 9999);  // offset seed for domain gen

    // -----------------------------------------------------------------------
    // Step 1: Poisson disk sampling for lobule centers in 3D
    // -----------------------------------------------------------------------
    struct Point3 { float x, y, z; };
    std::vector<Point3> centers;
    const float min_dist = lobule_spacing;
    const int max_attempts = 30;

    std::uniform_real_distribution<float> ux(0.0f, static_cast<float>(grid_x));
    std::uniform_real_distribution<float> uy(0.0f, static_cast<float>(grid_y));
    std::uniform_real_distribution<float> uz(0.0f, static_cast<float>(grid_z));

    // Seed first point
    centers.push_back({ux(rng), uy(rng), uz(rng)});

    // Active list for Bridson's algorithm
    std::vector<int> active;
    active.push_back(0);

    std::uniform_real_distribution<float> u_phi(0.0f, 2.0f * static_cast<float>(M_PI));
    std::uniform_real_distribution<float> u_cos(-1.0f, 1.0f);
    std::uniform_real_distribution<float> u_r(min_dist, 2.0f * min_dist);

    while (!active.empty()) {
        std::uniform_int_distribution<int> u_idx(0, static_cast<int>(active.size()) - 1);
        int ai = u_idx(rng);
        const Point3 p = centers[active[ai]];  // copy, not ref (vector may realloc)

        bool found = false;
        for (int attempt = 0; attempt < max_attempts; attempt++) {
            // Random point in spherical shell [min_dist, 2*min_dist]
            float r = u_r(rng);
            float cos_theta = u_cos(rng);
            float sin_theta = std::sqrt(1.0f - cos_theta * cos_theta);
            float phi = u_phi(rng);

            float nx = p.x + r * sin_theta * std::cos(phi);
            float ny = p.y + r * sin_theta * std::sin(phi);
            float nz = p.z + r * cos_theta;

            // Skip if outside grid
            if (nx < 0 || nx >= grid_x || ny < 0 || ny >= grid_y || nz < 0 || nz >= grid_z)
                continue;

            // Check distance to all existing centers (brute force — ~20 centers)
            bool too_close = false;
            for (const auto& c : centers) {
                float dx = nx - c.x, dy = ny - c.y, dz = nz - c.z;
                if (dx*dx + dy*dy + dz*dz < min_dist * min_dist) {
                    too_close = true;
                    break;
                }
            }

            if (!too_close) {
                int new_idx = static_cast<int>(centers.size());
                centers.push_back({nx, ny, nz});
                active.push_back(new_idx);
                found = true;
                break;
            }
        }

        if (!found) {
            active.erase(active.begin() + ai);
        }
    }

    std::cout << "  Domain: " << centers.size() << " lobule centers (spacing="
              << lobule_spacing << " voxels)" << std::endl;

    // -----------------------------------------------------------------------
    // Step 2: Voronoi tessellation — classify each voxel as LOBULE or SEPTUM
    //   A voxel is on a Voronoi boundary (septum) when the difference between
    //   its distance to the nearest and second-nearest center < septum_thickness.
    // -----------------------------------------------------------------------
    if (!centers.empty()) {
        for (int z = 0; z < grid_z; z++) {
            for (int y = 0; y < grid_y; y++) {
                for (int x = 0; x < grid_x; x++) {
                    float vx = x + 0.5f, vy = y + 0.5f, vz = z + 0.5f;
                    float d1 = 1e30f, d2 = 1e30f;  // nearest, second-nearest

                    for (const auto& c : centers) {
                        float dx = vx - c.x, dy = vy - c.y, dz = vz - c.z;
                        float d = std::sqrt(dx*dx + dy*dy + dz*dz);
                        if (d < d1) {
                            d2 = d1;
                            d1 = d;
                        } else if (d < d2) {
                            d2 = d;
                        }
                    }

                    int idx = x + y * grid_x + z * grid_x * grid_y;
                    if (d2 - d1 < septum_thickness) {
                        voxel_type[idx] = VOXEL_SEPTUM;
                    } else {
                        voxel_type[idx] = VOXEL_LOBULE;
                    }
                }
            }
        }
    }
    // If no centers were generated (grid too small), everything stays STROMA

    // -----------------------------------------------------------------------
    // Step 3: Overlay tumor hemisphere on x=0 face
    //   Center at (0, grid_y/2, grid_z/2). Tumor overwrites lobule/septum.
    //   Margin is the shell just outside the tumor surface.
    // -----------------------------------------------------------------------
    const float tumor_radius = tumor_radius_frac * grid_x;
    const float cy = grid_y / 2.0f;
    const float cz = grid_z / 2.0f;

    for (int z = 0; z < grid_z; z++) {
        for (int y = 0; y < grid_y; y++) {
            for (int x = 0; x < grid_x; x++) {
                float dx = x + 0.5f;  // distance from x=0 boundary
                float dy = (y + 0.5f) - cy;
                float dz = (z + 0.5f) - cz;
                float dist = std::sqrt(dx*dx + dy*dy + dz*dz);

                int idx = x + y * grid_x + z * grid_x * grid_y;

                if (dist <= tumor_radius) {
                    voxel_type[idx] = VOXEL_TUMOR;
                } else if (dist <= tumor_radius + margin_thickness) {
                    voxel_type[idx] = VOXEL_MARGIN;
                }
            }
        }
    }

    // Print summary
    int counts[5] = {};
    for (int i = 0; i < total; i++) counts[voxel_type[i]]++;
    std::cout << "  Voxel types: STROMA=" << counts[0] << " SEPTUM=" << counts[1]
              << " LOBULE=" << counts[2] << " TUMOR=" << counts[3]
              << " MARGIN=" << counts[4] << std::endl;

    return voxel_type;
}

// ============================================================================
// ECM Pre-Seeding by Voxel Type
// ============================================================================

void preseed_ecm_by_voxel_type(
    const std::vector<uint8_t>& voxel_type,
    const ECMInitParams& ecm,
    int total_voxels, unsigned int seed)
{
    std::mt19937 rng(seed + 7777);  // offset seed for ECM noise
    std::uniform_real_distribution<float> noise(-0.1f, 0.1f);  // ±10% perturbation

    std::vector<float> density(total_voxels);
    std::vector<float> crosslink(total_voxels);

    for (int i = 0; i < total_voxels; i++) {
        float d = 0.0f, c = 0.0f;
        switch (voxel_type[i]) {
            case VOXEL_STROMA:
                d = ecm.stroma_density;
                break;
            case VOXEL_SEPTUM:
                d = ecm.septum_density;
                c = ecm.septum_crosslink;
                break;
            case VOXEL_LOBULE:
                d = ecm.lobule_density;
                break;
            case VOXEL_TUMOR:
                d = ecm.tumor_density;
                break;
            case VOXEL_MARGIN:
                d = ecm.margin_density;
                c = ecm.margin_crosslink;
                break;
        }
        // Apply ±10% random perturbation for heterogeneity
        float pert = 1.0f + noise(rng);
        density[i]   = std::max(0.0f, d * pert);
        crosslink[i] = std::clamp(c * pert, 0.0f, 1.0f);
    }

    // Copy to GPU
    set_ecm_density_from_host(density.data(), total_voxels);
    set_ecm_crosslink_from_host(crosslink.data(), total_voxels);

    std::cout << "  ECM pre-seeded by voxel type" << std::endl;
}

// ============================================================================
// Master Initialization Function
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
            // Scale segments with grid volume to maintain ~3% vessel density.
            // Each segment places ~grid_size vessels along a line, so for a grid^3
            // domain we need 0.03 * grid^2 segments. Clamp to [4, 1000].
            int vas_radius = config.grid_x / 2;
            int num_seg = std::max(4, std::min(1000,
                static_cast<int>(0.03f * config.grid_x * config.grid_y)));
            initializeVascularCellsRandom(
                vascular_vec,
                config.grid_x, config.grid_y, config.grid_z,
                vas_radius, num_seg,
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

    std::cout << "QSP-based agent initialization complete\n" << std::endl;
}

// ============================================================================
// Structured Domain Initialization (-i 1)
// ============================================================================

void initializeStructuredDomain(
    flamegpu::CUDASimulation& simulation,
    flamegpu::ModelDescription& model,
    const SimulationConfig& config,
    const LymphCentralWrapper& lymph)
{
    std::cout << "\n=== Structured Domain Initialization ===" << std::endl;

    // -----------------------------------------------------------------------
    // Read DomainInit parameters from model environment
    // -----------------------------------------------------------------------
    const float lobule_spacing  = model.Environment().getProperty<float>("PARAM_DOMAIN_LOBULE_SPACING");
    const float septum_thick    = model.Environment().getProperty<float>("PARAM_DOMAIN_SEPTUM_THICKNESS");
    const float tumor_rad_frac  = model.Environment().getProperty<float>("PARAM_DOMAIN_TUMOR_RADIUS_FRAC");
    const float margin_thick    = model.Environment().getProperty<float>("PARAM_DOMAIN_MARGIN_THICKNESS");

    const int gx = config.grid_x, gy = config.grid_y, gz = config.grid_z;
    const int total_voxels = gx * gy * gz;

    // -----------------------------------------------------------------------
    // Step 1: Generate lobular structure + tumor hemisphere
    // -----------------------------------------------------------------------
    std::cout << "  Generating lobular structure..." << std::endl;
    std::vector<uint8_t> voxel_type = generate_domain_structure(
        gx, gy, gz, lobule_spacing, septum_thick,
        tumor_rad_frac, margin_thick, config.random_seed);

    // Copy to GPU
    set_voxel_type_from_host(voxel_type.data(), total_voxels);

    // -----------------------------------------------------------------------
    // Step 2: Pre-seed ECM by voxel type
    // -----------------------------------------------------------------------
    ECMInitParams ecm_params;
    ecm_params.septum_density    = model.Environment().getProperty<float>("PARAM_DOMAIN_ECM_SEPTUM_DENSITY");
    ecm_params.septum_crosslink  = model.Environment().getProperty<float>("PARAM_DOMAIN_ECM_SEPTUM_CROSSLINK");
    ecm_params.stroma_density    = model.Environment().getProperty<float>("PARAM_DOMAIN_ECM_STROMA_DENSITY");
    ecm_params.lobule_density    = model.Environment().getProperty<float>("PARAM_DOMAIN_ECM_LOBULE_DENSITY");
    ecm_params.margin_density    = model.Environment().getProperty<float>("PARAM_DOMAIN_ECM_MARGIN_DENSITY");
    ecm_params.margin_crosslink  = model.Environment().getProperty<float>("PARAM_DOMAIN_ECM_MARGIN_CROSSLINK");
    ecm_params.tumor_density     = model.Environment().getProperty<float>("PARAM_DOMAIN_ECM_TUMOR_DENSITY");

    preseed_ecm_by_voxel_type(voxel_type, ecm_params, total_voxels, config.random_seed);

    // -----------------------------------------------------------------------
    // Step 3: Read cell-lifecycle parameters
    // -----------------------------------------------------------------------
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

    std::vector<double> celltype_cdf = get_celltype_cdf(model);

    // Read region placement probabilities
    const float fib_p[5] = {
        model.Environment().getProperty<float>("PARAM_DOMAIN_FIB_P_STROMA"),   // STROMA
        model.Environment().getProperty<float>("PARAM_DOMAIN_FIB_P_SEPTUM"),   // SEPTUM
        model.Environment().getProperty<float>("PARAM_DOMAIN_FIB_P_LOBULE"),   // LOBULE
        model.Environment().getProperty<float>("PARAM_DOMAIN_FIB_P_TUMOR"),    // TUMOR
        model.Environment().getProperty<float>("PARAM_DOMAIN_FIB_P_MARGIN"),   // MARGIN
    };
    const float fib_margin_mycaf = model.Environment().getProperty<float>("PARAM_DOMAIN_FIB_MARGIN_MYCAF_FRAC");

    const float vas_p[5] = {
        model.Environment().getProperty<float>("PARAM_DOMAIN_VAS_P_STROMA"),
        model.Environment().getProperty<float>("PARAM_DOMAIN_VAS_P_SEPTUM"),
        model.Environment().getProperty<float>("PARAM_DOMAIN_VAS_P_LOBULE"),
        model.Environment().getProperty<float>("PARAM_DOMAIN_VAS_P_TUMOR"),
        model.Environment().getProperty<float>("PARAM_DOMAIN_VAS_P_MARGIN"),
    };
    const float vas_margin_tip = model.Environment().getProperty<float>("PARAM_DOMAIN_VAS_MARGIN_TIP_FRAC");

    const float mac_p[5] = {
        model.Environment().getProperty<float>("PARAM_DOMAIN_MAC_P_STROMA"),
        0.0f,  // SEPTUM: no macrophages
        0.0f,  // LOBULE: no macrophages
        model.Environment().getProperty<float>("PARAM_DOMAIN_MAC_P_TUMOR"),
        model.Environment().getProperty<float>("PARAM_DOMAIN_MAC_P_MARGIN"),
    };
    const float mac_margin_m1 = model.Environment().getProperty<float>("PARAM_DOMAIN_MAC_MARGIN_M1_FRAC");

    const float th_p_stroma = model.Environment().getProperty<float>("PARAM_DOMAIN_TH_P_STROMA");

    // Occupancy grid for exclusive cells
    std::vector<bool> occupied(total_voxels, false);

    // -----------------------------------------------------------------------
    // Step 4: Place cancer cells in VOXEL_TUMOR voxels
    // -----------------------------------------------------------------------
    {
        flamegpu::AgentVector cancer_pop(model.Agent(AGENT_CANCER_CELL));
        int placed = 0;

        for (int z = 0; z < gz; z++) {
            for (int y = 0; y < gy; y++) {
                for (int x = 0; x < gx; x++) {
                    int idx = x + y * gx + z * gx * gy;
                    if (voxel_type[idx] != VOXEL_TUMOR) continue;

                    // Sample from CDF
                    double p = static_cast<float>(rand()) / RAND_MAX;
                    int i = std::lower_bound(celltype_cdf.begin(), celltype_cdf.end(), p) - celltype_cdf.begin();
                    if (i > prog_max + 1) continue;

                    int cell_state, div_cd = 0, div = 0, divide_flag = 0, is_stem = 0;

                    cancer_pop.push_back();
                    flamegpu::AgentVector::Agent agent = cancer_pop.back();

                    if (i == 0) {
                        cell_state = CANCER_STEM;
                        float rf = static_cast<float>(rand()) / (RAND_MAX + 1.0f);
                        div_cd = static_cast<int>(stem_div * rf) + 1;
                        divide_flag = 1;
                        is_stem = 1;
                    } else if (i == prog_max + 1) {
                        cell_state = CANCER_SENESCENT;
                    } else {
                        cell_state = CANCER_PROGENITOR;
                        div = prog_max + 1 - i;
                        float rf = static_cast<float>(rand()) / (RAND_MAX + 1.0f);
                        div_cd = static_cast<int>(prog_div * rf) + 1;
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
                        float r = static_cast<float>(rand()) / (RAND_MAX + 1.0f);
                        int sen_life = static_cast<int>(-cancer_sen_life * logf(r + 0.0001f) + 0.5f);
                        agent.setVariable<int>("life", sen_life > 0 ? sen_life : 1);
                    }

                    occupied[idx] = true;
                    placed++;
                }
            }
        }
        simulation.setPopulationData(cancer_pop);
        std::cout << "  Placed " << placed << " cancer cells (VOXEL_TUMOR)" << std::endl;
    }

    // -----------------------------------------------------------------------
    // Step 5: T cells — none at init (recruited by QSP/vascular)
    // -----------------------------------------------------------------------
    {
        flamegpu::AgentVector tcell_pop(model.Agent(AGENT_TCELL));
        simulation.setPopulationData(tcell_pop);
    }

    // -----------------------------------------------------------------------
    // Step 6: TH cells — stroma only
    // -----------------------------------------------------------------------
    {
        flamegpu::AgentVector treg_pop(model.Agent(AGENT_TREG));
        int placed_th = 0;

        for (int z = 0; z < gz; z++) {
            for (int y = 0; y < gy; y++) {
                for (int x = 0; x < gx; x++) {
                    int idx = x + y * gx + z * gx * gy;
                    uint8_t vt = voxel_type[idx];
                    // TH in stroma and lobule (normal pancreatic tissue has resident T helpers)
                    float prob = (vt == VOXEL_STROMA || vt == VOXEL_LOBULE) ? th_p_stroma : 0.0f;
                    if (prob <= 0.0f) continue;

                    float rnd = static_cast<float>(rand()) / RAND_MAX;
                    if (rnd >= prob) continue;

                    // Life (normal distribution)
                    float u1 = static_cast<float>(rand()) / RAND_MAX;
                    float u2 = static_cast<float>(rand()) / RAND_MAX;
                    if (u1 < 1e-6f) u1 = 1e-6f;
                    float z0 = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * 3.14159f * u2);
                    int life = static_cast<int>(treg_life + z0 * treg_life_sd + 0.5f);
                    if (life < 1) life = 1;

                    treg_pop.push_back();
                    flamegpu::AgentVector::Agent agent = treg_pop.back();
                    agent.setVariable<int>("x", x);
                    agent.setVariable<int>("y", y);
                    agent.setVariable<int>("z", z);
                    agent.setVariable<int>("cell_state", TCD4_TH);
                    agent.setVariable<int>("divide_flag", 0);
                    agent.setVariable<int>("divide_cd", treg_div_interval);
                    agent.setVariable<int>("divide_limit", treg_div_limit);
                    agent.setVariable<int>("life", life);
                    agent.setVariable<float>("TGFB_release_remain", 0.0f);
                    agent.setVariable<float>("PDL1_syn", 0.0f);
                    agent.setVariable<float>("CTLA4", 0.0f);
                    agent.setVariable<float>("IL2_exposure", 0.0f);
                    agent.setVariable<int>("neighbor_Tcell_count", 0);
                    agent.setVariable<int>("neighbor_Treg_count", 0);
                    agent.setVariable<int>("neighbor_cancer_count", 0);
                    agent.setVariable<int>("neighbor_all_count", 0);
                    agent.setVariable<int>("found_progenitor", 0);
                    agent.setVariable<unsigned int>("available_neighbors", 0u);
                    agent.setVariable<int>("dead", 0);
                    agent.setVariable<int>("intent_action", 0);
                    agent.setVariable<int>("target_x", -1);
                    agent.setVariable<int>("target_y", -1);
                    agent.setVariable<int>("target_z", -1);
                    placed_th++;
                }
            }
        }
        simulation.setPopulationData(treg_pop);
        std::cout << "  Placed " << placed_th << " TH cells (stroma)" << std::endl;
    }

    // -----------------------------------------------------------------------
    // Step 7: MDSCs — none at structured init (recruited via CCL2)
    // -----------------------------------------------------------------------
    {
        flamegpu::AgentVector mdsc_pop(model.Agent(AGENT_MDSC));
        simulation.setPopulationData(mdsc_pop);
    }

    // -----------------------------------------------------------------------
    // Step 8: Macrophages — region-aware with M1/M2 polarization
    // -----------------------------------------------------------------------
    {
        flamegpu::AgentVector mac_pop(model.Agent(AGENT_MACROPHAGE));
        int placed = 0;

        for (int z = 0; z < gz; z++) {
            for (int y = 0; y < gy; y++) {
                for (int x = 0; x < gx; x++) {
                    int idx = x + y * gx + z * gx * gy;
                    uint8_t vt = voxel_type[idx];
                    float prob = mac_p[vt];
                    if (prob <= 0.0f) continue;

                    float rnd = static_cast<float>(rand()) / RAND_MAX;
                    if (rnd >= prob) continue;

                    float life_rnd = static_cast<float>(rand()) / RAND_MAX;
                    int life = static_cast<int>(mac_life * std::log(1.0f / (life_rnd + 1e-4f)) + 0.5f);
                    if (life < 1) life = 1;

                    // Polarization: margin has M1 fraction, rest are M2
                    int state = MAC_M2;
                    if (vt == VOXEL_MARGIN) {
                        float pol_rnd = static_cast<float>(rand()) / RAND_MAX;
                        if (pol_rnd < mac_margin_m1) state = MAC_M1;
                    }

                    mac_pop.push_back();
                    flamegpu::AgentVector::Agent agent = mac_pop.back();
                    agent.setVariable<int>("x", x);
                    agent.setVariable<int>("y", y);
                    agent.setVariable<int>("z", z);
                    agent.setVariable<int>("life", life);
                    agent.setVariable<int>("cell_state", state);
                    placed++;
                }
            }
        }
        simulation.setPopulationData(mac_pop);
        std::cout << "  Placed " << placed << " macrophages (region-aware)" << std::endl;
    }

    // -----------------------------------------------------------------------
    // Step 9: Fibroblasts — region-aware with myCAF pre-activation at margin
    // -----------------------------------------------------------------------
    {
        flamegpu::AgentVector fib_pop(model.Agent(AGENT_FIBROBLAST));
        int placed = 0;

        for (int z = 0; z < gz; z++) {
            for (int y = 0; y < gy; y++) {
                for (int x = 0; x < gx; x++) {
                    int idx = x + y * gx + z * gx * gy;
                    if (occupied[idx]) continue;  // skip cancer-occupied voxels

                    uint8_t vt = voxel_type[idx];
                    float prob = fib_p[vt];
                    if (prob <= 0.0f) continue;

                    float rnd = static_cast<float>(rand()) / RAND_MAX;
                    if (rnd >= prob) continue;

                    occupied[idx] = true;

                    float life_rnd = static_cast<float>(rand()) / RAND_MAX;
                    int life = static_cast<int>(fib_life * std::log(1.0f / (life_rnd + 1e-4f)) + 0.5f);
                    if (life < 1) life = 1;

                    // State: margin fibroblasts are pre-activated as myCAF
                    int state = FIB_QUIESCENT;
                    if (vt == VOXEL_MARGIN) {
                        float act_rnd = static_cast<float>(rand()) / RAND_MAX;
                        if (act_rnd < fib_margin_mycaf) state = FIB_MYCAF;
                    }

                    fib_pop.push_back();
                    flamegpu::AgentVector::Agent agent = fib_pop.back();
                    agent.setVariable<int>("x", x);
                    agent.setVariable<int>("y", y);
                    agent.setVariable<int>("z", z);
                    agent.setVariable<int>("cell_state", state);
                    agent.setVariable<int>("life", life);
                    agent.setVariable<int>("divide_cooldown", 0);
                    agent.setVariable<int>("divide_count", 0);
                    placed++;
                }
            }
        }
        simulation.setPopulationData(fib_pop);
        std::cout << "  Placed " << placed << " fibroblasts (region-aware)" << std::endl;
    }

    // -----------------------------------------------------------------------
    // Step 10: Vascular cells — region-aware with TIP fraction at margin
    // -----------------------------------------------------------------------
    {
        flamegpu::AgentVector vas_pop(model.Agent(AGENT_VASCULAR));
        int placed = 0;
        const double p_branch_init = vas_branch_prob / 5.0;

        for (int z = 0; z < gz; z++) {
            for (int y = 0; y < gy; y++) {
                for (int x = 0; x < gx; x++) {
                    int idx = x + y * gx + z * gx * gy;
                    uint8_t vt = voxel_type[idx];
                    float prob = vas_p[vt];
                    if (prob <= 0.0f) continue;

                    float rnd = static_cast<float>(rand()) / RAND_MAX;
                    if (rnd >= prob) continue;

                    // State: phalanx by default; margin has TIP fraction
                    int state = VAS_PHALANX;
                    if (vt == VOXEL_MARGIN) {
                        float tip_rnd = static_cast<float>(rand()) / RAND_MAX;
                        if (tip_rnd < vas_margin_tip) state = VAS_TIP;
                    }

                    int branch_flag = (static_cast<float>(rand()) / RAND_MAX < p_branch_init) ? 1 : 0;

                    vas_pop.push_back();
                    flamegpu::AgentVector::Agent agent = vas_pop.back();
                    setVascularCellVariables(agent, x, y, z, state,
                        1.0f, 0.0f, 0.0f, (state == VAS_TIP) ? static_cast<unsigned int>(placed + 1) : 0u, branch_flag);
                    placed++;
                }
            }
        }
        simulation.setPopulationData(vas_pop);
        std::cout << "  Placed " << placed << " vascular cells (region-aware)" << std::endl;
    }

    // -----------------------------------------------------------------------
    // Step 11: PDE warmup — diffusion+decay only (no agent sources)
    // -----------------------------------------------------------------------
    const int warmup_substeps = model.Environment().getProperty<int>("PARAM_DOMAIN_PDE_WARMUP_SUBSTEPS");
    run_pde_warmup(warmup_substeps);

    std::cout << "Structured domain initialization complete\n" << std::endl;
}

} // namespace PDAC