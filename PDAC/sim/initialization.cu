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
                // Explicit z bounds clamp (target_z may be close to edge)
                if (next_z < 0) { next_z = 0; }
                if (next_z >= grid_z) { next_z = grid_z - 1; }
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

DomainStructure generate_domain_structure(
    int grid_x, int grid_y, int grid_z,
    float lobule_spacing, float septum_thickness,
    float tumor_radius_frac,
    unsigned int seed)
{
    const int total = grid_x * grid_y * grid_z;
    DomainStructure domain;
    domain.is_septum.resize(total, false);
    domain.is_tumor.resize(total, false);

    std::mt19937 rng(seed + 9999);

    // -----------------------------------------------------------------------
    // Step 1: Poisson disk sampling for lobule centers in 3D (Bridson's)
    // -----------------------------------------------------------------------
    const float min_dist = lobule_spacing;
    const int max_attempts = 30;

    std::uniform_real_distribution<float> ux(0.0f, static_cast<float>(grid_x));
    std::uniform_real_distribution<float> uy(0.0f, static_cast<float>(grid_y));
    std::uniform_real_distribution<float> uz(0.0f, static_cast<float>(grid_z));

    domain.lobule_centers.push_back({ux(rng), uy(rng), uz(rng)});

    std::vector<int> active;
    active.push_back(0);

    std::uniform_real_distribution<float> u_phi(0.0f, 2.0f * static_cast<float>(M_PI));
    std::uniform_real_distribution<float> u_cos(-1.0f, 1.0f);
    std::uniform_real_distribution<float> u_r(min_dist, 2.0f * min_dist);

    while (!active.empty()) {
        std::uniform_int_distribution<int> u_idx(0, static_cast<int>(active.size()) - 1);
        int ai = u_idx(rng);
        const Point3 p = domain.lobule_centers[active[ai]];

        bool found = false;
        for (int attempt = 0; attempt < max_attempts; attempt++) {
            float r = u_r(rng);
            float cos_theta = u_cos(rng);
            float sin_theta = std::sqrt(1.0f - cos_theta * cos_theta);
            float phi = u_phi(rng);

            float nx = p.x + r * sin_theta * std::cos(phi);
            float ny = p.y + r * sin_theta * std::sin(phi);
            float nz = p.z + r * cos_theta;

            if (nx < 0 || nx >= grid_x || ny < 0 || ny >= grid_y || nz < 0 || nz >= grid_z)
                continue;

            bool too_close = false;
            for (const auto& c : domain.lobule_centers) {
                float dx = nx - c.x, dy = ny - c.y, dz = nz - c.z;
                if (dx*dx + dy*dy + dz*dz < min_dist * min_dist) {
                    too_close = true;
                    break;
                }
            }

            if (!too_close) {
                int new_idx = static_cast<int>(domain.lobule_centers.size());
                domain.lobule_centers.push_back({nx, ny, nz});
                active.push_back(new_idx);
                found = true;
                break;
            }
        }

        if (!found) {
            active.erase(active.begin() + ai);
        }
    }

    std::cout << "  Domain: " << domain.lobule_centers.size() << " lobule centers (spacing="
              << lobule_spacing << " voxels)" << std::endl;

    // -----------------------------------------------------------------------
    // Step 2: Voronoi tessellation — mark septum voxels
    // -----------------------------------------------------------------------
    if (!domain.lobule_centers.empty()) {
        for (int z = 0; z < grid_z; z++) {
            for (int y = 0; y < grid_y; y++) {
                for (int x = 0; x < grid_x; x++) {
                    float vx = x + 0.5f, vy = y + 0.5f, vz = z + 0.5f;
                    float d1 = 1e30f, d2 = 1e30f;

                    for (const auto& c : domain.lobule_centers) {
                        float dx = vx - c.x, dy = vy - c.y, dz = vz - c.z;
                        float d = std::sqrt(dx*dx + dy*dy + dz*dz);
                        if (d < d1) { d2 = d1; d1 = d; }
                        else if (d < d2) { d2 = d; }
                    }

                    int idx = x + y * grid_x + z * grid_x * grid_y;
                    if (d2 - d1 < septum_thickness) {
                        domain.is_septum[idx] = true;
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Step 3: Mark tumor hemisphere on x=0 face
    // -----------------------------------------------------------------------
    const float tumor_radius = tumor_radius_frac * grid_x;
    const float cy = grid_y / 2.0f;
    const float cz = grid_z / 2.0f;

    for (int z = 0; z < grid_z; z++) {
        for (int y = 0; y < grid_y; y++) {
            for (int x = 0; x < grid_x; x++) {
                float dx = x + 0.5f;
                float dy = (y + 0.5f) - cy;
                float dz = (z + 0.5f) - cz;
                float dist = std::sqrt(dx*dx + dy*dy + dz*dz);

                if (dist <= tumor_radius) {
                    int idx = x + y * grid_x + z * grid_x * grid_y;
                    domain.is_tumor[idx] = true;
                    domain.is_septum[idx] = false;  // tumor overrides septum
                }
            }
        }
    }

    // Print summary
    int n_septum = 0, n_tumor = 0, n_lobule = 0;
    for (int i = 0; i < total; i++) {
        if (domain.is_tumor[i]) n_tumor++;
        else if (domain.is_septum[i]) n_septum++;
        else n_lobule++;
    }
    std::cout << "  Regions: SEPTUM=" << n_septum << " LOBULE=" << n_lobule
              << " TUMOR=" << n_tumor << std::endl;

    return domain;
}

// ============================================================================
// ECM Pre-Seeding by Voxel Type
// ============================================================================

void preseed_ecm(
    const DomainStructure& domain,
    int grid_x, int grid_y, int grid_z,
    float septum_density, float septum_crosslink,
    float lobule_density, float tumor_density,
    float lobule_floor,
    unsigned int seed)
{
    const int total_voxels = grid_x * grid_y * grid_z;
    std::mt19937 rng(seed + 7777);
    std::uniform_real_distribution<float> noise(-0.1f, 0.1f);

    std::vector<float> density(total_voxels);
    std::vector<float> crosslink(total_voxels, 0.0f);
    std::vector<float> floor(total_voxels);
    std::vector<float> orient_x(total_voxels, 0.0f);
    std::vector<float> orient_y(total_voxels, 0.0f);
    std::vector<float> orient_z(total_voxels, 0.0f);

    for (int i = 0; i < total_voxels; i++) {
        float d, f;
        if (domain.is_tumor[i]) {
            d = tumor_density;
            f = lobule_floor;
        } else if (domain.is_septum[i]) {
            d = septum_density;
            f = septum_density;  // floor = init density (septa are persistent)
            crosslink[i] = septum_crosslink;
        } else {
            d = lobule_density;
            f = lobule_floor;
        }
        float pert = 1.0f + noise(rng);
        density[i] = std::max(0.0f, d * pert);
        floor[i] = f;
        crosslink[i] = std::clamp(crosslink[i] * (1.0f + noise(rng)), 0.0f, 1.0f);
    }

    // Compute septum fiber orientation (tangent to Voronoi boundary)
    if (!domain.lobule_centers.empty()) {
        const float ref_x = 0.0f, ref_y = 0.0f, ref_z = 1.0f;  // reference axis

        for (int z = 0; z < grid_z; z++) {
            for (int y = 0; y < grid_y; y++) {
                for (int x = 0; x < grid_x; x++) {
                    int idx = x + y * grid_x + z * grid_x * grid_y;
                    if (!domain.is_septum[idx]) continue;

                    float vx = x + 0.5f, vy = y + 0.5f, vz = z + 0.5f;

                    // Find two nearest lobule centers
                    float d1 = 1e30f, d2 = 1e30f;
                    int i1 = 0, i2 = 0;
                    for (int ci = 0; ci < (int)domain.lobule_centers.size(); ci++) {
                        const auto& c = domain.lobule_centers[ci];
                        float dx = vx - c.x, dy = vy - c.y, dz = vz - c.z;
                        float dd = std::sqrt(dx*dx + dy*dy + dz*dz);
                        if (dd < d1) { d2 = d1; i2 = i1; d1 = dd; i1 = ci; }
                        else if (dd < d2) { d2 = dd; i2 = ci; }
                    }

                    // Septum normal = direction between the two nearest centers
                    const auto& c1 = domain.lobule_centers[i1];
                    const auto& c2 = domain.lobule_centers[i2];
                    float nx = c2.x - c1.x, ny = c2.y - c1.y, nz_v = c2.z - c1.z;
                    float nmag = std::sqrt(nx*nx + ny*ny + nz_v*nz_v);
                    if (nmag < 1e-6f) continue;
                    nx /= nmag; ny /= nmag; nz_v /= nmag;

                    // Tangent = normal × reference axis
                    float tx = ny * ref_z - nz_v * ref_y;
                    float ty = nz_v * ref_x - nx * ref_z;
                    float tz = nx * ref_y - ny * ref_x;
                    float tmag = std::sqrt(tx*tx + ty*ty + tz*tz);

                    // If parallel to ref axis, use alternate
                    if (tmag < 1e-6f) {
                        tx = ny * 0.0f - nz_v * 1.0f;
                        ty = nz_v * 0.0f - nx * 0.0f;
                        tz = nx * 1.0f - ny * 0.0f;
                        tmag = std::sqrt(tx*tx + ty*ty + tz*tz);
                    }
                    if (tmag > 1e-6f) {
                        orient_x[idx] = tx / tmag;
                        orient_y[idx] = ty / tmag;
                        orient_z[idx] = tz / tmag;
                    }
                }
            }
        }
    }

    // Copy to GPU
    set_ecm_density_from_host(density.data(), total_voxels);
    set_ecm_crosslink_from_host(crosslink.data(), total_voxels);
    set_ecm_floor_from_host(floor.data(), total_voxels);
    set_ecm_orient_from_host(orient_x.data(), orient_y.data(), orient_z.data(), total_voxels);

    int n_sept = 0;
    for (int i = 0; i < total_voxels; i++) if (domain.is_septum[i]) n_sept++;
    std::cout << "  ECM pre-seeded: " << n_sept << " septum voxels with density="
              << septum_density << ", crosslink=" << septum_crosslink << std::endl;
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
    const int   treg_div_interval = model.Environment().getProperty<int>("PARAM_TREG_DIV_INTERVAL");
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
        flamegpu::AgentVector tcell_pop(model.Agent(AGENT_TCELL));
        simulation.setPopulationData(tcell_pop);  // empty
    }

    // TH and TReg cells: probability-based placement across all voxels
    {
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
        std::cout << "  Placed " << treg_pop.size() << " TH/TReg cells" << std::endl;
        simulation.setPopulationData(treg_pop);
    }

    // MDSCs: probability-based placement across all voxels
    {
        flamegpu::AgentVector mdsc_pop(model.Agent(AGENT_MDSC));
        initializeMDSCsFromQSP(
            mdsc_pop,
            config.grid_x, config.grid_y, config.grid_z,
            p_mdsc, occupied,
            mdsc_life);
        std::cout << "  Placed " << mdsc_pop.size() << " MDSCs" << std::endl;
        simulation.setPopulationData(mdsc_pop);
    }

    // Macrophages: probability-based placement across all voxels
    {
        flamegpu::AgentVector mac_pop(model.Agent(AGENT_MACROPHAGE));
        initializeMacsFromQSP(
            mac_pop,
            config.grid_x, config.grid_y, config.grid_z,
            p_mac, occupied,
            mac_life);
        std::cout << "  Placed " << mac_pop.size() << " Macrophages" << std::endl;
        simulation.setPopulationData(mac_pop);
    }

    // Fibroblasts: probability-based placement across all voxels
    {
        flamegpu::AgentVector fib_pop(model.Agent(AGENT_FIBROBLAST));
        initializeFibroblastsFromQSP(
            fib_pop,
            config.grid_x, config.grid_y, config.grid_z,
            p_fib, occupied,
            fib_life);
        std::cout << "  Placed " << fib_pop.size() << " Fibroblasts" << std::endl;
        simulation.setPopulationData(fib_pop);
    }

    // Initialize vascular cells (same logic as initializeAllAgents)
    {
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
        std::cout << "  Placed " << vascular_vec.size() << " Vascular cells" << std::endl;
        simulation.setPopulationData(vascular_vec);
    }

    // B cells — none at init (recruited via CXCL13)
    {
        flamegpu::AgentVector bcell_pop(model.Agent(AGENT_BCELL));
        simulation.setPopulationData(bcell_pop);
    }

    // DCs — none at init (recruited via CCL2)
    {
        flamegpu::AgentVector dc_pop(model.Agent(AGENT_DC));
        simulation.setPopulationData(dc_pop);
    }

    std::cout << "QSP-based agent initialization complete\n" << std::endl;
}

// ============================================================================
// Structured Domain Initialization (-i 1)
// ============================================================================

// ---------------------------------------------------------------------------
// Density-based cell placement helpers
// ---------------------------------------------------------------------------

// Sample N random positions from a list of valid voxel indices.
// If occupied != nullptr, marks sampled voxels as true (exclusive placement).
static std::vector<std::array<int,3>> sample_positions(
    const std::vector<int>& valid_indices,
    int N, int gx, int gy, std::mt19937& rng,
    std::vector<bool>* occupied = nullptr)
{
    // Shuffle a copy of the index list and take the first N (or fewer)
    std::vector<int> pool;
    if (occupied) {
        // Filter out already-occupied voxels
        pool.reserve(valid_indices.size());
        for (int idx : valid_indices) {
            if (!(*occupied)[idx]) pool.push_back(idx);
        }
    } else {
        pool = valid_indices;
    }

    std::shuffle(pool.begin(), pool.end(), rng);
    int count = std::min(N, static_cast<int>(pool.size()));

    std::vector<std::array<int,3>> positions(count);
    for (int i = 0; i < count; i++) {
        int idx = pool[i];
        int x = idx % gx;
        int y = (idx / gx) % gy;
        int z = idx / (gx * gy);
        positions[i] = {x, y, z};
        if (occupied) (*occupied)[idx] = true;
    }
    return positions;
}

// Exponential random life: life = -mean * ln(U), clamped to [1, inf)
static int random_exp_life(float mean, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float u = dist(rng);
    int life = static_cast<int>(-mean * logf(u + 1e-6f) + 0.5f);
    return life > 0 ? life : 1;
}

// Normal random life: life = mean + z*sd, clamped to [1, inf)
static int random_normal_life(float mean, float sd, std::mt19937& rng) {
    std::normal_distribution<float> dist(mean, sd);
    int life = static_cast<int>(dist(rng) + 0.5f);
    return life > 0 ? life : 1;
}

void initializeStructuredDomain(
    flamegpu::CUDASimulation& simulation,
    flamegpu::ModelDescription& model,
    const SimulationConfig& config,
    const LymphCentralWrapper& lymph)
{
    std::cout << "\n=== Structured Domain Initialization (density-based) ===" << std::endl;

    const int gx = config.grid_x, gy = config.grid_y, gz = config.grid_z;
    const int total_voxels = gx * gy * gz;
    const float voxel_um = config.voxel_size;

    // Domain volume in mm³: (grid_size * voxel_um / 1000)³
    const double domain_vol_mm3 = std::pow(gx * voxel_um / 1000.0, 3);
    std::cout << "  Domain: " << gx << "³ voxels @ " << voxel_um
              << " µm = " << domain_vol_mm3 << " mm³" << std::endl;

    std::mt19937 rng(config.random_seed);

    // -----------------------------------------------------------------------
    // Step 1: Generate lobular structure + tumor hemisphere
    // -----------------------------------------------------------------------
    const float lobule_spacing = model.Environment().getProperty<float>("PARAM_DOMAIN_LOBULE_SPACING");
    const float septum_thick   = model.Environment().getProperty<float>("PARAM_DOMAIN_SEPTUM_THICKNESS");
    const float tumor_rad_frac = model.Environment().getProperty<float>("PARAM_DOMAIN_TUMOR_RADIUS_FRAC");

    std::cout << "  Generating lobular structure..." << std::endl;
    DomainStructure domain = generate_domain_structure(
        gx, gy, gz, lobule_spacing, septum_thick,
        tumor_rad_frac, config.random_seed);

    // -----------------------------------------------------------------------
    // Step 2: Pre-seed ECM (density, crosslink, floor, fiber orientation)
    // -----------------------------------------------------------------------
    const float ecm_septum_density   = model.Environment().getProperty<float>("PARAM_DOMAIN_ECM_SEPTUM_DENSITY");
    const float ecm_septum_crosslink = model.Environment().getProperty<float>("PARAM_DOMAIN_ECM_SEPTUM_CROSSLINK");
    const float ecm_lobule_density   = model.Environment().getProperty<float>("PARAM_DOMAIN_ECM_LOBULE_DENSITY");
    const float ecm_tumor_density    = model.Environment().getProperty<float>("PARAM_DOMAIN_ECM_TUMOR_DENSITY");
    const float ecm_lobule_floor     = model.Environment().getProperty<float>("PARAM_ECM_BASELINE");

    std::cout << "  Pre-seeding ECM..." << std::endl;
    preseed_ecm(domain, gx, gy, gz,
                ecm_septum_density, ecm_septum_crosslink,
                ecm_lobule_density, ecm_tumor_density,
                ecm_lobule_floor, config.random_seed);

    // -----------------------------------------------------------------------
    // Step 3: Build voxel index lists for placement
    // -----------------------------------------------------------------------
    std::vector<int> tumor_voxels, nontumor_voxels;
    tumor_voxels.reserve(total_voxels / 10);
    nontumor_voxels.reserve(total_voxels);
    for (int idx = 0; idx < total_voxels; idx++) {
        if (domain.is_tumor[idx])
            tumor_voxels.push_back(idx);
        else
            nontumor_voxels.push_back(idx);
    }
    std::cout << "  Tumor voxels: " << tumor_voxels.size()
              << "  Non-tumor: " << nontumor_voxels.size() << std::endl;

    // Occupancy grid for exclusive cells (cancer, fibroblast, vascular)
    std::vector<bool> occupied(total_voxels, false);

    // -----------------------------------------------------------------------
    // Step 4: Read cell-lifecycle parameters
    // -----------------------------------------------------------------------
    const float stem_div         = model.Environment().getProperty<float>("PARAM_FLOAT_CANCER_CELL_STEM_DIV_INTERVAL_SLICE");
    const float prog_div         = model.Environment().getProperty<float>("PARAM_FLOAT_CANCER_CELL_PROGENITOR_DIV_INTERVAL_SLICE");
    const int   prog_max         = model.Environment().getProperty<int>("PARAM_PROG_DIV_MAX");
    const float cancer_sen_life  = model.Environment().getProperty<float>("PARAM_CANCER_SENESCENT_MEAN_LIFE");
    const float fib_life         = model.Environment().getProperty<float>("PARAM_FIB_LIFE_MEAN");
    const float mac_life         = model.Environment().getProperty<float>("PARAM_MAC_LIFE_MEAN");
    const float tcell_life       = model.Environment().getProperty<float>("PARAM_T_CELL_LIFE_MEAN_SLICE");
    const float tcell_life_sd    = model.Environment().getProperty<float>("PARAM_TCELL_LIFESPAN_SD_SLICE");
    const int   tcell_div_limit  = model.Environment().getProperty<int>("PARAM_TCELL_DIV_LIMIT");
    const float treg_life        = model.Environment().getProperty<float>("PARAM_TCD4_LIFE_MEAN_SLICE");
    const float treg_life_sd     = model.Environment().getProperty<float>("PARAM_TCELL_LIFESPAN_SD_SLICE");
    const int   treg_div_limit   = model.Environment().getProperty<int>("PARAM_TCD4_DIV_LIMIT");
    const int   treg_div_interval = model.Environment().getProperty<int>("PARAM_TREG_DIV_INTERVAL");
    const float bcell_life       = model.Environment().getProperty<float>("PARAM_BCELL_LIFE_MEAN");
    const float bcell_life_sd    = model.Environment().getProperty<float>("PARAM_BCELL_LIFE_SD");
    const float dc_life_imm      = model.Environment().getProperty<float>("PARAM_DC_LIFE_IMMATURE_MEAN");
    const float dc_life_imm_sd   = model.Environment().getProperty<float>("PARAM_DC_LIFE_IMMATURE_SD");

    // Cell densities (cells/mm³) from IMC data
    const float fib_density = model.Environment().getProperty<float>("PARAM_DOMAIN_FIB_DENSITY");
    const float vas_density = model.Environment().getProperty<float>("PARAM_DOMAIN_VAS_DENSITY");
    const float mac_density = model.Environment().getProperty<float>("PARAM_DOMAIN_MAC_DENSITY");
    const float cd8_density = model.Environment().getProperty<float>("PARAM_DOMAIN_CD8_DENSITY");
    const float cd4_density = model.Environment().getProperty<float>("PARAM_DOMAIN_CD4_DENSITY");
    const float dc_density  = model.Environment().getProperty<float>("PARAM_DOMAIN_DC_DENSITY");
    const float bcell_density = model.Environment().getProperty<float>("PARAM_DOMAIN_BCELL_DENSITY");

    // Non-tumor volume for healthy tissue cells
    const double nontumor_vol_mm3 = nontumor_voxels.size() * std::pow(voxel_um / 1000.0, 3);

    std::vector<double> celltype_cdf = get_celltype_cdf(model);

    // -----------------------------------------------------------------------
    // Step 5: Place cancer cells in tumor voxels (CDF-based, exclusive)
    // -----------------------------------------------------------------------
    {
        flamegpu::AgentVector cancer_pop(model.Agent(AGENT_CANCER_CELL));
        int placed = 0;

        // Shuffle tumor voxels
        std::vector<int> tvox = tumor_voxels;
        std::shuffle(tvox.begin(), tvox.end(), rng);

        for (int idx : tvox) {
            int x = idx % gx;
            int y = (idx / gx) % gy;
            int z = idx / (gx * gy);

            // Sample from CDF
            std::uniform_real_distribution<double> dist01(0.0, 1.0);
            double p = dist01(rng);
            int i = std::lower_bound(celltype_cdf.begin(), celltype_cdf.end(), p) - celltype_cdf.begin();
            if (i > prog_max + 1) continue;

            int cell_state, div_cd = 0, div = 0, divide_flag = 0, is_stem = 0;

            cancer_pop.push_back();
            flamegpu::AgentVector::Agent agent = cancer_pop.back();

            std::uniform_real_distribution<float> rf01(0.0f, 1.0f);
            if (i == 0) {
                cell_state = CANCER_STEM;
                div_cd = static_cast<int>(stem_div * rf01(rng)) + 1;
                divide_flag = 1;
                is_stem = 1;
            } else if (i == prog_max + 1) {
                cell_state = CANCER_SENESCENT;
            } else {
                cell_state = CANCER_PROGENITOR;
                div = prog_max + 1 - i;
                div_cd = static_cast<int>(prog_div * rf01(rng)) + 1;
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
                int sen_life = random_exp_life(cancer_sen_life, rng);
                agent.setVariable<int>("life", sen_life);
            }

            occupied[idx] = true;
            placed++;
        }
        simulation.setPopulationData(cancer_pop);
        std::cout << "  Cancer cells: " << placed << " (in " << tumor_voxels.size() << " tumor voxels)" << std::endl;
    }

    // -----------------------------------------------------------------------
    // Step 6: Fibroblasts — density-based in non-tumor, exclusive, FIB_QUIESCENT
    // -----------------------------------------------------------------------
    {
        int N_fib = static_cast<int>(std::round(fib_density * nontumor_vol_mm3));
        auto positions = sample_positions(nontumor_voxels, N_fib, gx, gy, rng, &occupied);

        flamegpu::AgentVector fib_pop(model.Agent(AGENT_FIBROBLAST));
        for (auto& pos : positions) {
            fib_pop.push_back();
            flamegpu::AgentVector::Agent agent = fib_pop.back();
            agent.setVariable<int>("x", pos[0]);
            agent.setVariable<int>("y", pos[1]);
            agent.setVariable<int>("z", pos[2]);
            agent.setVariable<int>("cell_state", FIB_QUIESCENT);
            agent.setVariable<int>("life", random_exp_life(fib_life, rng));
            agent.setVariable<int>("divide_cooldown", 0);
            agent.setVariable<int>("divide_count", 0);
        }
        simulation.setPopulationData(fib_pop);
        std::cout << "  Fibroblasts: " << positions.size() << " (target " << N_fib
                  << " @ " << fib_density << "/mm³)" << std::endl;
    }

    // -----------------------------------------------------------------------
    // Step 7: Vascular cells — density-based in non-tumor, exclusive, VAS_PHALANX
    // -----------------------------------------------------------------------
    {
        int N_vas = static_cast<int>(std::round(vas_density * nontumor_vol_mm3));
        auto positions = sample_positions(nontumor_voxels, N_vas, gx, gy, rng, &occupied);

        flamegpu::AgentVector vas_pop(model.Agent(AGENT_VASCULAR));
        for (size_t i = 0; i < positions.size(); i++) {
            vas_pop.push_back();
            flamegpu::AgentVector::Agent agent = vas_pop.back();
            setVascularCellVariables(agent, positions[i][0], positions[i][1], positions[i][2],
                                    VAS_PHALANX, 1.0f, 0.0f, 0.0f, 0u, 0);
        }

        // Assign initial TIP sprouts via spatial spreading
        const int vas_min_neighbor = static_cast<int>(model.Environment().getProperty<float>("PARAM_VAS_MIN_NEIGHBOR"));
        assignInitialVascularTips(vas_pop, gx, gy, gz,
            vas_min_neighbor,
            config.random_seed + 7);

        simulation.setPopulationData(vas_pop);
        std::cout << "  Vascular cells: " << positions.size() << " (target " << N_vas
                  << " @ " << vas_density << "/mm³)" << std::endl;
    }

    // -----------------------------------------------------------------------
    // Step 8: Macrophages — density-based, non-exclusive, MAC_M2
    // -----------------------------------------------------------------------
    {
        int N_mac = static_cast<int>(std::round(mac_density * nontumor_vol_mm3));
        auto positions = sample_positions(nontumor_voxels, N_mac, gx, gy, rng);

        flamegpu::AgentVector mac_pop(model.Agent(AGENT_MACROPHAGE));
        for (auto& pos : positions) {
            mac_pop.push_back();
            flamegpu::AgentVector::Agent agent = mac_pop.back();
            agent.setVariable<int>("x", pos[0]);
            agent.setVariable<int>("y", pos[1]);
            agent.setVariable<int>("z", pos[2]);
            agent.setVariable<int>("cell_state", MAC_M2);
            agent.setVariable<int>("life", random_exp_life(mac_life, rng));
        }
        simulation.setPopulationData(mac_pop);
        std::cout << "  Macrophages: " << positions.size() << " (target " << N_mac
                  << " @ " << mac_density << "/mm³, all M2)" << std::endl;
    }

    // -----------------------------------------------------------------------
    // Step 9: CD8 T cells — density-based, non-exclusive, T_CELL_EFF
    // -----------------------------------------------------------------------
    {
        int N_cd8 = static_cast<int>(std::round(cd8_density * nontumor_vol_mm3));
        auto positions = sample_positions(nontumor_voxels, N_cd8, gx, gy, rng);

        flamegpu::AgentVector tcell_pop(model.Agent(AGENT_TCELL));
        for (auto& pos : positions) {
            tcell_pop.push_back();
            flamegpu::AgentVector::Agent agent = tcell_pop.back();
            agent.setVariable<int>("x", pos[0]);
            agent.setVariable<int>("y", pos[1]);
            agent.setVariable<int>("z", pos[2]);
            agent.setVariable<int>("cell_state", T_CELL_EFF);
            agent.setVariable<int>("life", random_normal_life(tcell_life, tcell_life_sd, rng));
            agent.setVariable<int>("divide_limit", tcell_div_limit);
        }
        simulation.setPopulationData(tcell_pop);
        std::cout << "  CD8 T cells: " << positions.size() << " (target " << N_cd8
                  << " @ " << cd8_density << "/mm³)" << std::endl;
    }

    // -----------------------------------------------------------------------
    // Step 10: CD4 T cells (TH) — density-based, non-exclusive, TCD4_TH
    // -----------------------------------------------------------------------
    {
        int N_cd4 = static_cast<int>(std::round(cd4_density * nontumor_vol_mm3));
        auto positions = sample_positions(nontumor_voxels, N_cd4, gx, gy, rng);

        flamegpu::AgentVector treg_pop(model.Agent(AGENT_TREG));
        for (auto& pos : positions) {
            treg_pop.push_back();
            flamegpu::AgentVector::Agent agent = treg_pop.back();
            agent.setVariable<int>("x", pos[0]);
            agent.setVariable<int>("y", pos[1]);
            agent.setVariable<int>("z", pos[2]);
            agent.setVariable<int>("cell_state", TCD4_TH);
            agent.setVariable<int>("life", random_normal_life(treg_life, treg_life_sd, rng));
            agent.setVariable<int>("divide_flag", 0);
            agent.setVariable<int>("divide_cd", treg_div_interval);
            agent.setVariable<int>("divide_limit", treg_div_limit);
        }
        simulation.setPopulationData(treg_pop);
        std::cout << "  CD4 TH cells: " << positions.size() << " (target " << N_cd4
                  << " @ " << cd4_density << "/mm³)" << std::endl;
    }

    // -----------------------------------------------------------------------
    // Step 11: DCs — density-based, non-exclusive, DC_IMMATURE / DC_CDC1
    // -----------------------------------------------------------------------
    {
        int N_dc = static_cast<int>(std::round(dc_density * nontumor_vol_mm3));
        auto positions = sample_positions(nontumor_voxels, N_dc, gx, gy, rng);

        flamegpu::AgentVector dc_pop(model.Agent(AGENT_DC));
        for (auto& pos : positions) {
            dc_pop.push_back();
            flamegpu::AgentVector::Agent agent = dc_pop.back();
            agent.setVariable<int>("x", pos[0]);
            agent.setVariable<int>("y", pos[1]);
            agent.setVariable<int>("z", pos[2]);
            agent.setVariable<int>("cell_state", DC_IMMATURE);
            agent.setVariable<int>("dc_subtype", DC_CDC1);
            agent.setVariable<int>("life", random_normal_life(dc_life_imm, dc_life_imm_sd, rng));
        }
        simulation.setPopulationData(dc_pop);
        std::cout << "  DCs: " << positions.size() << " (target " << N_dc
                  << " @ " << dc_density << "/mm³, immature cDC1)" << std::endl;
    }

    // -----------------------------------------------------------------------
    // Step 12: B cells — density-based, non-exclusive, BCELL_NAIVE
    // -----------------------------------------------------------------------
    {
        int N_bcell = static_cast<int>(std::round(bcell_density * nontumor_vol_mm3));
        auto positions = sample_positions(nontumor_voxels, N_bcell, gx, gy, rng);

        flamegpu::AgentVector bcell_pop(model.Agent(AGENT_BCELL));
        for (auto& pos : positions) {
            bcell_pop.push_back();
            flamegpu::AgentVector::Agent agent = bcell_pop.back();
            agent.setVariable<int>("x", pos[0]);
            agent.setVariable<int>("y", pos[1]);
            agent.setVariable<int>("z", pos[2]);
            agent.setVariable<int>("cell_state", BCELL_NAIVE);
            agent.setVariable<int>("life", random_normal_life(bcell_life, bcell_life_sd, rng));
        }
        simulation.setPopulationData(bcell_pop);
        std::cout << "  B cells: " << positions.size() << " (target " << N_bcell
                  << " @ " << bcell_density << "/mm³, naive)" << std::endl;
    }

    // -----------------------------------------------------------------------
    // Step 13: MDSCs — none at init (pathological, recruited via CCL2)
    // -----------------------------------------------------------------------
    {
        flamegpu::AgentVector mdsc_pop(model.Agent(AGENT_MDSC));
        simulation.setPopulationData(mdsc_pop);
        std::cout << "  MDSCs: 0 (recruited dynamically)" << std::endl;
    }

    // -----------------------------------------------------------------------
    // Step 14: PDE warmup — diffusion+decay only (no agent sources)
    // -----------------------------------------------------------------------
    const int warmup_substeps = model.Environment().getProperty<int>("PARAM_DOMAIN_PDE_WARMUP_SUBSTEPS");
    run_pde_warmup(warmup_substeps);
    std::cout << "  PDE warmup: " << warmup_substeps << " substeps" << std::endl;

    std::cout << "Structured domain initialization complete\n" << std::endl;
}

} // namespace PDAC