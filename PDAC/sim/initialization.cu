#include "initialization.cuh"
#include "../core/common.cuh"
#include "../abm/gpu_param.h"
#include "../pde/pde_integration.cuh"
#include <iostream>
#include <fstream>
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
    : steps(500)
    , random_seed(12345)
    , init_method(0)  // 0 = single stem at center (new default)
    , scenario(Scenario::Default)
    , seed_position("x_low")
    , vascular_mode("none")  // ABM rebuild Step 3b: vascular agents removed; O2+entry via Vvas field
    , vascular_xml_file("")
    , placement_file("")     // §C6: empty = procedural init; set by --placement
    , qsp_onset_volume(-1.0) // §C6.1: <=0 = no standalone QSP warm-up
    , qsp_onset_dump("")
    , grid_out(0)
    , interval_out(1)
    , presim_qsp_enabled(true)
    , main_qsp_enabled(true)
    , presim_steps(-1)
    , xml_presim_steps(0)
    , presim_volume_target(-1.0)
    , presim_areal_target(-1.0)
    , presim_steps_from_cli(false)
    , presim_volume_from_cli(false)
    , presim_areal_from_cli(false)
    , main_steps_from_cli(false)
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
            if (val == 0 || val == 1 || val == 2) {
                init_method = val;
            } else {
                std::cerr << "WARNING: -i must be 0 (single_stem), 1 (cluster), or 2 (structured_deprecated). Ignoring -i " << val << std::endl;
            }
        } else if ((arg == "--grid-size" || arg == "-g") && i + 1 < argc) {
            int size = std::atoi(argv[++i]);
            grid_x = size;
            grid_y = size;
            grid_z = size;
        } else if ((arg == "--grid-x" || arg == "-gx") && i + 1 < argc) {
            grid_x = std::atoi(argv[++i]);
        } else if ((arg == "--grid-y" || arg == "-gy") && i + 1 < argc) {
            grid_y = std::atoi(argv[++i]);
        } else if ((arg == "--grid-z" || arg == "-gz") && i + 1 < argc) {
            grid_z = std::atoi(argv[++i]);
        } else if ((arg == "--steps" || arg == "-s") && i + 1 < argc) {
            steps = std::atoi(argv[++i]);
            main_steps_from_cli = true;
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
        } else if (arg == "--placement" && i + 1 < argc) {
            // §C6: initialize by data-driven placement. Overrides -i / --scenario.
            placement_file = argv[++i];
        } else if (arg == "--qsp-onset-volume" && i + 1 < argc) {
            qsp_onset_volume = std::atof(argv[++i]);
        } else if (arg == "--dump-qsp-onset" && i + 1 < argc) {
            qsp_onset_dump = argv[++i];
        } else if (arg == "--presim-mode" && i + 1 < argc) {
            std::string m = argv[++i];
            if (m == "qsp_abm")        presim_qsp_enabled = true;
            else if (m == "abm_only")  presim_qsp_enabled = false;
            else {
                std::cerr << "ERROR: --presim-mode must be qsp_abm or abm_only (got: " << m << ")" << std::endl;
                exit(1);
            }
        } else if (arg == "--main-mode" && i + 1 < argc) {
            std::string m = argv[++i];
            if (m == "qsp_abm")        main_qsp_enabled = true;
            else if (m == "abm_only")  main_qsp_enabled = false;
            else {
                std::cerr << "ERROR: --main-mode must be qsp_abm or abm_only (got: " << m << ")" << std::endl;
                exit(1);
            }
        } else if (arg == "--presim-steps" && i + 1 < argc) {
            presim_steps = std::atoi(argv[++i]);
            presim_steps_from_cli = true;
        } else if (arg == "--presim-volume" && i + 1 < argc) {
            presim_volume_target = std::atof(argv[++i]);
            presim_volume_from_cli = true;
        } else if (arg == "--presim-areal" && i + 1 < argc) {
            presim_areal_target = std::atof(argv[++i]);
            presim_areal_from_cli = true;
        } else if (arg == "--scenario" && i + 1 < argc) {
            std::string s = argv[++i];
            if (s == "default") {
                scenario = Scenario::Default;
            } else if (s == "single_stem_edge") {
                scenario = Scenario::SingleStemEdge;
            } else {
                std::cerr << "ERROR: --scenario must be 'default' or 'single_stem_edge' (got: " << s << ")" << std::endl;
                exit(1);
            }
        } else if (arg == "--seed-position" && i + 1 < argc) {
            std::string s = argv[++i];
            if (s != "x_low" && s != "x_high" && s != "y_low" &&
                s != "y_high" && s != "z_low" && s != "z_high") {
                std::cerr << "ERROR: --seed-position must be one of "
                             "x_low|x_high|y_low|y_high|z_low|z_high (got: " << s << ")" << std::endl;
                exit(1);
            }
            seed_position = s;
        } else if (arg == "--output-root" && i + 1 < argc) {
            // Consumed by main.cu pre-pass; skip the value so we don't error.
            ++i;
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "\nOptions:\n"
                      << "  -p, --param-file FILE    Path to parameter XML file [default: param_all_test.xml]\n"
                      << "  -i, --initialization N   Initialization type: 0=single_stem_center, 1=cluster_radius, 2=structured_deprecated [default: 0]\n"
                      << "  -g, --grid-size N        Grid dimensions NxNxN [default: from XML]\n"
                      << "  -gx, --grid-x N          Override grid X only (applied after -g)\n"
                      << "  -gy, --grid-y N          Override grid Y only (applied after -g)\n"
                      << "  -gz, --grid-z N          Override grid Z only (applied after -g)\n"
                      << "  -s, --steps N            Number of simulation steps [default: 200]\n"
                      << "  -G, --grid-output N      Grid output: 0=none, 1=ABM only, 2=PDE+ECM only, 3=both [default: 0]\n"
                      << "  -oi, --out_int N         Output interval frequency [default: 1]\n"
                      << "  --seed N                 Random seed [default: 12345]\n"
                      << "  -vm, --vascular-mode STR Vasculature initialization: none, random, xml, test [default: none]\n"
                      << "  -vx, --vascular-xml FILE XML file for vasculature (when mode=xml)\n"
                      << "  --qsp-onset-volume V     Warm the QSP alone to V (m^3) before init, independent of the presim\n"
                      << "  --dump-qsp-onset PATH    After the warm-up, write the onset JSON and exit (placement pass 1)\n"
                      << "  --presim-mode MODE       Presim stepping mode: qsp_abm | abm_only [default: qsp_abm]\n"
                      << "  --main-mode MODE         Main-sim stepping mode: qsp_abm | abm_only [default: qsp_abm]\n"
                      << "  --presim-areal F         Presim stop at cancer areal fraction F=N_cancer/(gx*gy*gz). Top precedence (section/SBI stopper).\n"
                      << "  --presim-steps N         Presim step count override (>=0). Wins over --presim-volume and XML.\n"
                      << "  --presim-volume V        Presim QSP tumor volume target (m^3). Used if --presim-steps absent.\n"
                      << "  --scenario NAME          Scenario bundle: default | single_stem_edge [default: default]\n"
                      << "                           single_stem_edge: resident vasc+PSC only, 1 stem cell at edge, ABM-only.\n"
                      << "  --seed-position FACE     For single_stem_edge: x_low|x_high|y_low|y_high|z_low|z_high [default: x_low]\n"
                      << "  --output-root PATH       Root directory for all output files [default: outputs]\n"
                      << "                           Auto-named under results/ when --scenario single_stem_edge is set.\n"
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
    std::cout << "Init: ";
    if      (init_method == 0) std::cout << "single_stem_center (-i 0)";
    else if (init_method == 1) std::cout << "cluster_radius (-i 1)";
    else if (init_method == 2) std::cout << "structured_deprecated (-i 2)";
    else                       std::cout << "unknown (-i " << init_method << ")";
    std::cout << std::endl;
    std::cout << "Scenario: "
              << (scenario == Scenario::SingleStemEdge ? "single_stem_edge" : "default");
    if (scenario == Scenario::SingleStemEdge) {
        std::cout << " (seed_position=" << seed_position << ")";
    }
    std::cout << std::endl;

    std::cout << "\nSimulation modes:" << std::endl;
    std::cout << "  Presim: " << (presim_qsp_enabled ? "qsp_abm" : "abm_only");
    if (presim_areal_from_cli)          std::cout << "  (stopper: cancer areal frac=" << presim_areal_target << ")";
    else if (presim_steps >= 0)         std::cout << "  (stopper: steps=" << presim_steps << ")";
    else if (presim_volume_target > 0)  std::cout << "  (stopper: tum_vol=" << presim_volume_target << " m^3)";
    else                                std::cout << "  (stopper: QSP-derived volume, resolved at runtime)";
    std::cout << std::endl;
    std::cout << "  Main:   " << (main_qsp_enabled ? "qsp_abm" : "abm_only") << std::endl;

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

void seedSingleStemAtCenter(
    flamegpu::AgentVector& cancer_pop,
    int grid_x, int grid_y, int grid_z,
    float stem_div_interval)
{
    cancer_pop.push_back();
    flamegpu::AgentVector::Agent agent = cancer_pop.back();
    const unsigned int id = agent.getID();
    const float rand_frac = static_cast<float>(rand()) / (RAND_MAX + 1.0f);
    const int   div_cd    = static_cast<int>(stem_div_interval * rand_frac) + 1;
    agent.setVariable<int>("x", grid_x / 2);
    agent.setVariable<int>("y", grid_y / 2);
    agent.setVariable<int>("z", grid_z / 2);
    agent.setVariable<int>("cell_state", CANCER_STEM);
    agent.setVariable<int>("divideCD", div_cd);
    agent.setVariable<int>("divideFlag", 1);
    agent.setVariable<int>("divideCountRemaining", 0);
    agent.setVariable<unsigned int>("stemID", id);
}

// Single-ledger cancer seeding (Jul 2026). At the Phase-C handoff the QSP tumor is already
// grown (~90% of the diameter target via the decoupled Phase-A warmup), but the ABM cancer,
// left to grow organically from 1 stem in the short coupled presim, never fills the box —
// leaving cancer-depleted, immune/fib-flooded windows. Instead, seed the ABM cancer to its
// geometric box-share N_seed = min(f·C1_QSP, box_capacity), f = V_box/V_T. The observable
// lumps stem+prog (→ cancer_prog) vs sen (→ cancer_sen), so the only split that matters is the
// senescent fraction (Ki67- fraction, an SBI param); stem fraction of the proliferating
// remainder is a fixed CSC-consistent constant. Placement is uniform-random across the box
// (dispersed cancer in interstitial stroma — PDAC's low-cellularity desmoplastic pattern);
// spatial nesting is a possible refinement.
void seedCancerToQSPShare(
    flamegpu::AgentVector& cancer_pop,
    int grid_x, int grid_y, int grid_z,
    double voxel_m, double V_T_m3, double C1_cells,
    float stem_div_interval, float prog_div_interval, int prog_max,
    float senescent_mean_life, float sen_frac)
{
    const long long total_voxels = static_cast<long long>(grid_x) * grid_y * grid_z;
    const double V_box = static_cast<double>(total_voxels) * voxel_m * voxel_m * voxel_m;
    const double f = (V_T_m3 > 0.0) ? std::min(1.0, V_box / V_T_m3) : 1.0;
    long long N_seed = static_cast<long long>(std::llround(f * C1_cells));
    if (N_seed < 1) N_seed = 1;
    if (N_seed > total_voxels) N_seed = total_voxels;   // cancer is exclusive (1 per voxel)
    const float STEM_FRAC_OF_PROLIF = 0.05f;            // CSC-consistent; fixed (observable lumps stem+prog)

    // Distinct random voxels: partial Fisher-Yates on the voxel-index list, take the first N_seed.
    std::vector<int> vidx(static_cast<size_t>(total_voxels));
    for (long long i = 0; i < total_voxels; ++i) vidx[static_cast<size_t>(i)] = static_cast<int>(i);
    for (long long i = total_voxels - 1; i > 0; --i) {
        long long j = static_cast<long long>(rand()) % (i + 1);
        std::swap(vidx[static_cast<size_t>(i)], vidx[static_cast<size_t>(j)]);
    }

    unsigned int dt = 0;
    long long n_stem = 0, n_prog = 0, n_sen = 0;
    const int gxy = grid_x * grid_y;
    for (long long k = 0; k < N_seed; ++k) {
        const int vi = vidx[static_cast<size_t>(k)];
        const int z = vi / gxy, rem = vi % gxy, y = rem / grid_x, x = rem % grid_x;
        cancer_pop.push_back();
        flamegpu::AgentVector::Agent agent = cancer_pop.back();
        const int id = agent.getID();
        agent.setVariable<int>("x", x);
        agent.setVariable<int>("y", y);
        agent.setVariable<int>("z", z);
        agent.setVariable<unsigned int>("det_tag", ++dt);   // unique lineage tag (deterministic ties)
        const float r = static_cast<float>(rand()) / (RAND_MAX + 1.0f);
        if (r < sen_frac) {
            agent.setVariable<int>("cell_state", CANCER_SENESCENT);
            agent.setVariable<int>("divideFlag", 0);
            agent.setVariable<int>("divideCD", 0);
            agent.setVariable<int>("divideCountRemaining", 0);
            agent.setVariable<unsigned int>("stemID", 0);
            const float ru = static_cast<float>(rand()) / (RAND_MAX + 1.0f);
            const int sl = static_cast<int>(-senescent_mean_life * logf(ru + 1e-4f) + 0.5f);
            agent.setVariable<int>("life", sl > 0 ? sl : 1);
            ++n_sen;
        } else {
            const float rs = static_cast<float>(rand()) / (RAND_MAX + 1.0f);
            const float rf = static_cast<float>(rand()) / (RAND_MAX + 1.0f);
            if (rs < STEM_FRAC_OF_PROLIF) {
                agent.setVariable<int>("cell_state", CANCER_STEM);
                agent.setVariable<int>("divideCD", static_cast<int>(stem_div_interval * rf) + 1);
                agent.setVariable<int>("divideFlag", 1);
                agent.setVariable<int>("divideCountRemaining", 0);
                agent.setVariable<unsigned int>("stemID", static_cast<unsigned int>(id));
                ++n_stem;
            } else {
                agent.setVariable<int>("cell_state", CANCER_PROGENITOR);
                agent.setVariable<int>("divideCD", static_cast<int>(prog_div_interval * rf) + 1);
                agent.setVariable<int>("divideFlag", 1);
                agent.setVariable<int>("divideCountRemaining", 1 + (rand() % (prog_max > 0 ? prog_max : 1)));
                agent.setVariable<unsigned int>("stemID", 0);
                ++n_prog;
            }
        }
    }
    std::cout << "  [single-ledger] cancer seeded to box-share: f=" << f
              << " C1_QSP=" << C1_cells << " cells -> N_seed=" << N_seed
              << "  (stem " << n_stem << " / prog " << n_prog << " / sen " << n_sen
              << ", sen_frac=" << sen_frac << ")" << std::endl;
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


// ============================================================================
// Fibroblast Initialization
// ============================================================================

void seedQuiescentPSCs(
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
                // Step-5 determinism: unique non-zero lineage tag per founder (one fib per voxel
                // here, so the voxel index is a unique key). Seeds det_rng + breaks co-located ties.
                agent.setVariable<unsigned int>("det_tag", static_cast<unsigned int>(idx0) + 1u);

                placed++;
            }
        }
    }
    std::cout << "  Placed " << placed << " quiescent PSCs (resident, density-based)" << std::endl;
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

    (void)seed;  // rng was only used by the silenced Poisson-disk pass

    // -----------------------------------------------------------------------
    // Lobular/septa structure is currently silenced (Apr 22, 2026).
    // Lobule centers stay empty, so the Voronoi septum pass below and the
    // septum fiber-orientation pass in preseed_ecm both skip. ECM collapses
    // to a uniform lobule baseline (plus tumor hemisphere). Unused silenced
    // params: PARAM_DOMAIN_LOBULE_SPACING, PARAM_DOMAIN_SEPTUM_THICKNESS,
    // PARAM_DOMAIN_ECM_SEPTUM_DENSITY, PARAM_DOMAIN_ECM_SEPTUM_CROSSLINK.
    // To re-enable: restore the Poisson disk + Voronoi blocks from git.
    // -----------------------------------------------------------------------
    (void)lobule_spacing;
    (void)septum_thickness;
    std::cout << "  Domain: lobular/septa structure disabled — uniform lobule baseline"
              << std::endl;

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

// CANONICAL initialization (Jul 2026 rewrite). Replaces the retired "initialize-to-QSP"
// scheme (which seeded the whole TME from QSP tumor-compartment counts evaluated at the
// seed-state QSP — meaningless at cancer≈1 cell, and it starved fibroblasts to ~9 cells).
// Under the single-ledger coupling the TME must EMERGE:
//   - cancer   : 1 stem cell at grid center (-i 0), or a cluster (-i 1)
//   - immune   : NONE at init — recruited during the run by the QSP-coupled gap-closing
//   - stroma   : resident quiescent PSCs at PARAM_DOMAIN_FIB_DENSITY (IMC density), which
//                activate (TGFβ/IL1) and proliferate into CAF — a real resident field,
//                not a QSP-derived probability.
//   - vascular : resident network (unchanged).
// The old initialize*FromQSP population-seeding functions have been retired.
void initializeSingleStem(
    flamegpu::CUDASimulation& simulation,
    flamegpu::ModelDescription& model,
    const SimulationConfig& config,
    const LymphCentralWrapper& lymph)
{
    std::cout << "\n=== Canonical Init (single-stem cancer + resident PSC stroma) ===" << std::endl;

    // Cluster radius for -i 1 (cluster_radius mode). Read from XML so the
    // user can tune the initial tumor size without rebuilding. For -i 0
    // (single_stem_center) this value is unused.
    const float radius_frac = model.Environment().getProperty<float>("PARAM_DOMAIN_TUMOR_RADIUS_FRAC");
    int cluster_radius = static_cast<int>(std::round(radius_frac * config.grid_x));
    std::cout << "  cluster_radius  = " << cluster_radius
              << " voxels (frac=" << radius_frac << ")" << std::endl;

    // Resident quiescent-PSC seeding probability from the IMC density (cells/mm³) × voxel volume.
    // This is a real resident stromal field (activates + proliferates into CAF), independent of
    // the QSP — it replaces the retired QSP-derived p_fib that floored at 0.001 (~9 cells).
    const double voxel_mm  = static_cast<double>(config.voxel_size) / 1000.0;
    const double voxel_mm3 = voxel_mm * voxel_mm * voxel_mm;
    const double fib_density = model.Environment().getProperty<float>("PARAM_DOMAIN_FIB_DENSITY"); // cells/mm³
    double p_fib = std::min(1.0, std::max(0.0, fib_density * voxel_mm3));
    std::cout << "  resident PSC p_fib = " << p_fib
              << " (density=" << fib_density << " cells/mm³)" << std::endl;
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
        if (config.init_method == 1) {
            // -i 1: cluster of cancer cells (legacy default)
            initializeCancerCellsRandom(
                cancer_pop,
                config.grid_x, config.grid_y, config.grid_z,
                cluster_radius, stem_div, prog_div, prog_max, cancer_sen_life, celltype_cdf);
            std::cout << "  Placed " << cancer_pop.size() << " cancer cells (cluster r="
                      << cluster_radius << ")" << std::endl;
        } else if (config.presim_qsp_enabled && config.presim_volume_target > 0.0
                   && lymph.get_tumor_cancer_cells() > 1.0) {
            // -i 0 + coupled single-ledger presim: the decoupled Phase-A warmup already grew the
            // QSP tumor, so seed the ABM cancer to its geometric box-share f·C1 (else it grows from
            // 1 stem in the short Phase C and the window ends cancer-depleted). See seedCancerToQSPShare.
            const double voxel_m   = static_cast<double>(config.voxel_size) * 1e-6;   // µm → m
            const double V_T_m3    = lymph.get_tumor_volume();
            const double C1_cells  = lymph.get_tumor_cancer_cells();
            const float  sen_frac  = model.Environment().getProperty<float>("PARAM_SEED_CANCER_SEN_FRAC");
            seedCancerToQSPShare(
                cancer_pop,
                config.grid_x, config.grid_y, config.grid_z,
                voxel_m, V_T_m3, C1_cells,
                stem_div, prog_div, prog_max, cancer_sen_life, sen_frac);
        } else {
            // -i 0 (default): single stem cell at grid center (uncoupled / abm_only → grows organically)
            seedSingleStemAtCenter(
                cancer_pop,
                config.grid_x, config.grid_y, config.grid_z,
                stem_div);
            std::cout << "  Placed " << cancer_pop.size() << " cancer stem cell (center)" << std::endl;
        }
        buildOccupancyGrid(cancer_pop, occupied, config.grid_x, config.grid_y);
        simulation.setPopulationData(cancer_pop);
    }

    // Immune cells (Teff, TH/TReg, MDSC, MAC): NONE at init — the QSP-coupled soft gap-closing
    // recruits them during the run to track f·X_QSP_tumor. FLAMEGPU still needs an (empty)
    // population set for every agent type.
    { flamegpu::AgentVector tcell_pop(model.Agent(AGENT_TCELL)); simulation.setPopulationData(tcell_pop); }
    { flamegpu::AgentVector treg_pop(model.Agent(AGENT_TREG));   simulation.setPopulationData(treg_pop); }
    { flamegpu::AgentVector mdsc_pop(model.Agent(AGENT_MDSC));   simulation.setPopulationData(mdsc_pop); }
    { flamegpu::AgentVector mac_pop(model.Agent(AGENT_MACROPHAGE)); simulation.setPopulationData(mac_pop); }

    // Stroma: resident quiescent PSCs at PARAM_DOMAIN_FIB_DENSITY (not QSP-derived). They activate
    // (TGFβ/IL1) and proliferate into CAF during the run.
    {
        flamegpu::AgentVector fib_pop(model.Agent(AGENT_FIBROBLAST));
        seedQuiescentPSCs(
            fib_pop,
            config.grid_x, config.grid_y, config.grid_z,
            p_fib, occupied,
            fib_life);
        std::cout << "  Placed " << fib_pop.size() << " resident quiescent PSCs" << std::endl;
        simulation.setPopulationData(fib_pop);
    }

    // Initialize vascular cells (same logic as initializeAllAgents)
    {
        flamegpu::AgentVector vascular_vec(model.Agent(AGENT_VASCULAR));
        if (config.vascular_mode == "none") {
            // ABM rebuild Step 3b: no vascular agents. O2 delivery + immune entry
            // points are now the Vvas field (compute_vvas_and_o2 / mark_entry_points).
            // All vascular agent layers no-op on the empty population.
        } else if (config.vascular_mode == "random") {
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

    std::cout << "Canonical agent initialization complete\n" << std::endl;
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

    // Seed antigen grid at tumor voxels so APC activation chains (DC maturation,
    // B cell capture) aren't strictly gated on the first cancer-death event.
    // Senescent cells still deposit P1_C1 on natural death — this is just a
    // bootstrap floor for step 0. Set to 0 to disable.
    {
        const float p1c1 = model.Environment().getProperty<float>("PARAM_ANTIGEN_DEPOSIT");
        const float seed_frac = model.Environment().getProperty<float>("PARAM_ANTIGEN_INIT_SEED_FRAC");
        const float seed_value = seed_frac * p1c1;
        seed_antigen_grid_voxels(tumor_voxels.data(),
                                 static_cast<int>(tumor_voxels.size()),
                                 seed_value);
        std::cout << "  Antigen seed: " << seed_value
                  << " molecules/voxel × " << tumor_voxels.size()
                  << " tumor voxels (= " << seed_frac
                  << " × PARAM_ANTIGEN_DEPOSIT)" << std::endl;
    }

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
        unsigned int mac_dt = 0x10000000u;  // Step-5: unique per-cell lineage tag (type-offset)
        for (auto& pos : positions) {
            mac_pop.push_back();
            flamegpu::AgentVector::Agent agent = mac_pop.back();
            agent.setVariable<int>("x", pos[0]);
            agent.setVariable<int>("y", pos[1]);
            agent.setVariable<int>("z", pos[2]);
            agent.setVariable<int>("cell_state", MAC_M2);
            agent.setVariable<int>("life", random_exp_life(mac_life, rng));
            agent.setVariable<unsigned int>("det_tag", ++mac_dt);
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
        unsigned int tcell_dt = 0x20000000u;  // Step-5: unique per-cell lineage tag (type-offset)
        for (auto& pos : positions) {
            tcell_pop.push_back();
            flamegpu::AgentVector::Agent agent = tcell_pop.back();
            agent.setVariable<int>("x", pos[0]);
            agent.setVariable<int>("y", pos[1]);
            agent.setVariable<int>("z", pos[2]);
            agent.setVariable<int>("cell_state", T_CELL_EFF);
            agent.setVariable<int>("life", random_normal_life(tcell_life, tcell_life_sd, rng));
            agent.setVariable<int>("divide_limit", tcell_div_limit);
            agent.setVariable<unsigned int>("det_tag", ++tcell_dt);
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
        unsigned int th_dt = 0x30000000u;  // Step-5: unique per-cell lineage tag (type-offset)
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
            agent.setVariable<unsigned int>("det_tag", ++th_dt);
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
        unsigned int dc_dt = 0x40000000u;  // Step-5: unique per-cell lineage tag (type-offset)
        for (auto& pos : positions) {
            dc_pop.push_back();
            flamegpu::AgentVector::Agent agent = dc_pop.back();
            agent.setVariable<int>("x", pos[0]);
            agent.setVariable<int>("y", pos[1]);
            agent.setVariable<int>("z", pos[2]);
            agent.setVariable<int>("cell_state", DC_IMMATURE);
            agent.setVariable<int>("dc_subtype", DC_CDC1);
            agent.setVariable<int>("life", random_normal_life(dc_life_imm, dc_life_imm_sd, rng));
            agent.setVariable<unsigned int>("det_tag", ++dc_dt);
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
        unsigned int bcell_dt = 0x50000000u;  // Step-5: unique per-cell lineage tag (type-offset)
        for (auto& pos : positions) {
            bcell_pop.push_back();
            flamegpu::AgentVector::Agent agent = bcell_pop.back();
            agent.setVariable<int>("x", pos[0]);
            agent.setVariable<int>("y", pos[1]);
            agent.setVariable<int>("z", pos[2]);
            agent.setVariable<int>("cell_state", BCELL_NAIVE);
            agent.setVariable<int>("life", random_exp_life(bcell_life, rng));
            agent.setVariable<unsigned int>("det_tag", ++bcell_dt);
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

// ============================================================================
// Single-Stem-Edge Diagnostic Init (--scenario single_stem_edge)
// ============================================================================
//
// Lay down a resident healthy domain (vasculature + quiescent PSCs only) and
// place exactly one cancer stem cell on the chosen face of the domain. No
// immune agents are placed at init — the scenario is designed to run with
// QSP frozen so recruitment from systemic pools does not occur.
//
// FLAMEGPU requires setPopulationData for every agent type, including the
// empty immune populations.
// ============================================================================
void initializeSingleStemEdge(
    flamegpu::CUDASimulation& simulation,
    flamegpu::ModelDescription& model,
    const SimulationConfig& config,
    const LymphCentralWrapper& /*lymph*/)
{
    std::cout << "\n=== Single-Stem-Edge Init ===" << std::endl;
    const int gx = config.grid_x;
    const int gy = config.grid_y;
    const int gz = config.grid_z;
    std::cout << "  Domain: " << gx << "x" << gy << "x" << gz << " voxels" << std::endl;

    // ---- Lifecycle params from model env ----
    const float stem_div        = model.Environment().getProperty<float>("PARAM_FLOAT_CANCER_CELL_STEM_DIV_INTERVAL_SLICE");
    const float fib_life        = model.Environment().getProperty<float>("PARAM_FIB_LIFE_MEAN");
    const float vas_branch_prob = model.Environment().getProperty<float>("PARAM_VAS_BRANCH_PROB");
    const int   vas_min_neighbor = static_cast<int>(model.Environment().getProperty<float>("PARAM_VAS_MIN_NEIGHBOR"));

    // ---- Vasculature: resident network, no central tumor exclusion ----
    // ABM rebuild Step 3b: skipped in "none" mode (default). O2 + immune entry
    // points are now the Vvas field; vascular agent layers no-op on empty pop.
    if (config.vascular_mode != "none") {
        flamegpu::AgentVector vascular_vec(model.Agent(AGENT_VASCULAR));
        int num_seg = std::max(4, std::min(1000,
            static_cast<int>(0.03f * gx * gy)));
        initializeVascularCellsRandom(
            vascular_vec,
            gx, gy, gz,
            /*tumor_radius=*/0,  // no exclusion zone — entire domain is healthy
            num_seg, vas_branch_prob,
            config.random_seed);
        assignInitialVascularTips(
            vascular_vec,
            gx, gy, gz,
            vas_min_neighbor,
            config.random_seed);
        std::cout << "  Placed " << vascular_vec.size() << " vascular cells" << std::endl;
        simulation.setPopulationData(vascular_vec);
    }

    // ---- Occupancy grid (cancer cluster builder uses this; we won't have a
    //      cluster, but PSC placement should still avoid the seed voxel below) ----
    const int total_voxels = gx * gy * gz;
    std::vector<std::vector<int>> occupied(total_voxels, std::vector<int>(3, 0));

    // ---- Quiescent PSCs at fixed density ----
    // The QSP-derived p_fib is meaningless without QSP coupling. Use a sane
    // fixed probability; matches the upper clamp in initializeSingleStem.
    {
        flamegpu::AgentVector fib_pop(model.Agent(AGENT_FIBROBLAST));
        const double p_fib_resident = 0.05;
        seedQuiescentPSCs(
            fib_pop,
            gx, gy, gz,
            p_fib_resident, occupied,
            fib_life);
        std::cout << "  Placed " << fib_pop.size() << " quiescent PSCs (p="
                  << p_fib_resident << ")" << std::endl;
        simulation.setPopulationData(fib_pop);
    }

    // ---- Resolve seed position on the requested face ----
    const int margin_x = std::max(2, gx / 30);
    const int margin_y = std::max(2, gy / 30);
    const int margin_z = std::max(2, gz / 30);
    int seed_x = margin_x, seed_y = gy / 2, seed_z = gz / 2;
    if      (config.seed_position == "x_low")  { seed_x = margin_x;            seed_y = gy / 2;            seed_z = gz / 2; }
    else if (config.seed_position == "x_high") { seed_x = gx - 1 - margin_x;   seed_y = gy / 2;            seed_z = gz / 2; }
    else if (config.seed_position == "y_low")  { seed_x = gx / 2;              seed_y = margin_y;          seed_z = gz / 2; }
    else if (config.seed_position == "y_high") { seed_x = gx / 2;              seed_y = gy - 1 - margin_y; seed_z = gz / 2; }
    else if (config.seed_position == "z_low")  { seed_x = gx / 2;              seed_y = gy / 2;            seed_z = margin_z; }
    else if (config.seed_position == "z_high") { seed_x = gx / 2;              seed_y = gy / 2;            seed_z = gz - 1 - margin_z; }
    std::cout << "  Seed cancer stem cell at (" << seed_x << "," << seed_y << "," << seed_z
              << ")  [face=" << config.seed_position << "]" << std::endl;

    // ---- Place exactly one cancer stem cell ----
    // Build by hand (not via initializeCancerCellsRandom + cluster_radius=0,
    // which samples a CDF and may produce a senescent or progenitor instead).
    {
        flamegpu::AgentVector cancer_pop(model.Agent(AGENT_CANCER_CELL));
        cancer_pop.push_back();
        flamegpu::AgentVector::Agent agent = cancer_pop.back();
        const unsigned int id = agent.getID();
        const float rand_frac = static_cast<float>(rand()) / (RAND_MAX + 1.0f);
        const int   div_cd    = static_cast<int>(stem_div * rand_frac) + 1;
        agent.setVariable<int>("x", seed_x);
        agent.setVariable<int>("y", seed_y);
        agent.setVariable<int>("z", seed_z);
        agent.setVariable<int>("cell_state", CANCER_STEM);
        agent.setVariable<int>("divideCD", div_cd);
        agent.setVariable<int>("divideFlag", 1);
        agent.setVariable<int>("divideCountRemaining", 0);
        agent.setVariable<unsigned int>("stemID", id);
        std::cout << "  Placed 1 cancer stem cell (id=" << id
                  << ", divideCD=" << div_cd << ")" << std::endl;
        simulation.setPopulationData(cancer_pop);
    }

    // ---- Empty populations for every other agent type ----
    // FLAMEGPU requires every agent type to receive setPopulationData even
    // when empty; otherwise the device-side state is uninitialized.
    { flamegpu::AgentVector pop(model.Agent(AGENT_TCELL));      simulation.setPopulationData(pop); }
    { flamegpu::AgentVector pop(model.Agent(AGENT_TREG));       simulation.setPopulationData(pop); }
    { flamegpu::AgentVector pop(model.Agent(AGENT_MDSC));       simulation.setPopulationData(pop); }
    { flamegpu::AgentVector pop(model.Agent(AGENT_MACROPHAGE)); simulation.setPopulationData(pop); }
    { flamegpu::AgentVector pop(model.Agent(AGENT_BCELL));      simulation.setPopulationData(pop); }
    { flamegpu::AgentVector pop(model.Agent(AGENT_DC));         simulation.setPopulationData(pop); }

    std::cout << "Single-stem-edge initialization complete\n" << std::endl;
}

// ============================================================================
// Initialization by data-driven PLACEMENT (workplan §C6)
// ============================================================================
// Reader for the .plc format written by abm-inference `placement_io.write_placement`.
// Keep this in sync with that module's docstring — PLC_VERSION is the contract.

namespace {

constexpr int PLC_VERSION_EXPECTED = 2;
constexpr int PLC_N_COLS = 6;

// Column order of the agent table: (type_id, x, y, z, cell_state, subtype).
enum PlcCol : int { PLC_TYPE = 0, PLC_X, PLC_Y, PLC_Z, PLC_STATE, PLC_SUBTYPE };

// Agent type ids — MUST match core/common.cuh CellType and outputs/abm_lz4_def.txt.
enum PlcType : int {
    PLC_CANCER = 0, PLC_TCELL, PLC_TREG, PLC_MDSC, PLC_MAC,
    PLC_FIB, PLC_VAS, PLC_BCELL, PLC_DC
};

inline float urand() { return static_cast<float>(rand()) / (RAND_MAX + 1.0f); }

// Lifespan draws. The recruitment path does NOT use one distribution for everything, so
// neither does this: MDSC/MAC/FIB/senescent-cancer are exponential, while T/TCD4/B/DC are
// normal(mean, sd) — see sample_exp_life_gpu / sample_normal_life_gpu in pde_integration.cu.
// Using the wrong one here would make placed cells differ from recruited ones in exactly the
// way this loader is supposed to avoid.
inline int draw_life_exp(float mean) {
    const float u = static_cast<float>(rand()) / RAND_MAX;
    const int life = static_cast<int>(mean * std::log(1.0f / (u + 1e-4f)) + 0.5f);
    return life > 0 ? life : 1;
}

inline int draw_life_normal(float mean, float sd) {
    const float u1 = urand() + 1e-10f;
    const float u2 = urand();
    const float z = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * 3.14159265f * u2);
    const int life = static_cast<int>(std::lround(mean + z * sd));
    return life > 0 ? life : 1;
}

}  // namespace

bool initializeFromPlacement(
    flamegpu::CUDASimulation& simulation,
    flamegpu::ModelDescription& model,
    const SimulationConfig& config,
    const LymphCentralWrapper& lymph)
{
    (void)lymph;  // Placement supplies arrangement; the QSP owns totals via the run-time
                  // gap-closing controller, so nothing is read from the QSP at init here.

    std::cout << "\n=== Init by PLACEMENT (§C6): " << config.placement_file
              << " ===" << std::endl;

    std::ifstream f(config.placement_file, std::ios::binary);
    if (!f) {
        std::cerr << "ERROR: cannot open placement file '" << config.placement_file << "'"
                  << std::endl;
        return false;
    }

    char magic[4] = {0};
    f.read(magic, 4);
    if (std::string(magic, 4) != "PLC1") {
        std::cerr << "ERROR: '" << config.placement_file << "' is not a PLC file" << std::endl;
        return false;
    }
    int header[8] = {0};
    f.read(reinterpret_cast<char*>(header), sizeof(header));
    const int version  = header[0];
    const int gx       = header[1];
    const int gy       = header[2];
    const int gz       = header[3];
    const int n_agents = header[4];
    const int n_cols   = header[5];
    const int has_vvas = header[6];
    const int has_ecm  = header[7];

    if (version != PLC_VERSION_EXPECTED) {
        std::cerr << "ERROR: placement version " << version << ", expected "
                  << PLC_VERSION_EXPECTED << " — regenerate with make_placement.py" << std::endl;
        return false;
    }
    if (n_cols != PLC_N_COLS) {
        std::cerr << "ERROR: placement has " << n_cols << " columns, expected "
                  << PLC_N_COLS << std::endl;
        return false;
    }
    // A grid mismatch would place every agent at the wrong voxel and silently produce a
    // plausible-looking but meaningless run, so it is fatal rather than a warning.
    if (gx != config.grid_x || gy != config.grid_y || gz != config.grid_z) {
        std::cerr << "ERROR: placement grid " << gx << "x" << gy << "x" << gz
                  << " does not match simulation grid " << config.grid_x << "x"
                  << config.grid_y << "x" << config.grid_z
                  << " — regenerate the placement with --shape, or run with -g " << gx
                  << std::endl;
        return false;
    }

    std::vector<int> table(static_cast<size_t>(n_agents) * PLC_N_COLS);
    f.read(reinterpret_cast<char*>(table.data()),
           static_cast<std::streamsize>(table.size() * sizeof(int)));
    if (!f) {
        std::cerr << "ERROR: placement truncated while reading " << n_agents << " agents"
                  << std::endl;
        return false;
    }

    // ── V_vas(t=0) ─────────────────────────────────────────────────────────────────────────
    // Uploaded BEFORE agents so that a failure here is reported before the (much noisier)
    // agent placement output. d_vvas_field is allocated by initialize_pde_solver, which main
    // runs well before agent init — but set_vvas_from_host checks rather than assuming.
    const int total_voxels = gx * gy * gz;
    if (has_vvas) {
        std::vector<float> vvas(static_cast<size_t>(total_voxels));
        f.read(reinterpret_cast<char*>(vvas.data()),
               static_cast<std::streamsize>(vvas.size() * sizeof(float)));
        if (!f) {
            std::cerr << "ERROR: placement truncated while reading the V_vas field" << std::endl;
            return false;
        }
        double sum = 0.0, vmax = 0.0;
        int n_zero = 0;
        for (float v : vvas) {
            sum += v;
            vmax = std::max(vmax, static_cast<double>(v));
            if (v <= 0.0f) ++n_zero;
        }
        if (!set_vvas_from_host(vvas.data(), total_voxels)) return false;
        std::cout << "  V_vas seeded: mean " << (sum / total_voxels)
                  << ", max " << vmax
                  << ", avascular " << (100.0 * n_zero / total_voxels) << "%" << std::endl;
        // The avascular fraction is the load-bearing number: the growth law is logistic in V,
        // so these voxels stay at zero and are what make tumour nests unperfused.
        if (n_zero == 0) {
            std::cout << "  NOTE: no voxel is exactly zero — the seed was written without a"
                         " floor, so the placed perfusion geometry will decay toward a uniform"
                         " field. Regenerate with --floor-frac > 0." << std::endl;
        }
    } else {
        std::cout << "  no V_vas in placement — falling back to uniform PARAM_VAS_INIT"
                  << std::endl;
    }

    // ── ECM(t=0) ───────────────────────────────────────────────────────────────────────────
    // Placement supersedes the presim, and the presim was the ONLY thing that ever built the
    // ECM field (both procedural ECM initializers are dead code — called from
    // set_internal_params before initialize_pde_solver allocates the arrays, so both
    // early-return). Without this seed a placement run has no desmoplastic compression at all:
    // measured at 21^3, ECM reached only 136 nmol/mL in 8 steps against a 700 collapse
    // threshold, V_vas grew unopposed 0.0074 -> 0.0213 and the O2 cv collapsed 0.28 -> 0.057.
    if (has_ecm) {
        std::vector<float> ecm(static_cast<size_t>(total_voxels));
        f.read(reinterpret_cast<char*>(ecm.data()),
               static_cast<std::streamsize>(ecm.size() * sizeof(float)));
        if (!f) {
            std::cerr << "ERROR: placement truncated while reading the ECM field" << std::endl;
            return false;
        }
        const float collapse_th = model.Environment().getProperty<float>("PARAM_VAS_COLLAPSE_THRESHOLD");
        const float recovery_th = model.Environment().getProperty<float>("PARAM_VAS_RECOVERY_THRESHOLD");
        double sum = 0.0;
        int n_above = 0, n_below = 0;
        for (float e : ecm) {
            sum += e;
            if (e > collapse_th) ++n_above;
            if (e < recovery_th) ++n_below;
        }
        set_ecm_density_from_host(ecm.data(), total_voxels);
        std::cout << "  ECM seeded: mean " << (sum / total_voxels)
                  << ", " << (100.0 * n_above / total_voxels) << "% above the collapse threshold ("
                  << collapse_th << "), "
                  << (100.0 * n_below / total_voxels) << "% below recovery (" << recovery_th << ")"
                  << std::endl;
        if (n_above == 0) {
            std::cout << "  NOTE: no voxel exceeds the collapse threshold — the ECM branch of"
                         " the occlusion term will be inert. Raise --ecm-p90." << std::endl;
        }
    } else {
        std::cout << "  no ECM in placement — compression will be inert until ECM accumulates;"
                     " with the presim skipped it never will." << std::endl;
    }

    // ── lifecycle parameters, read exactly as initializeSingleStem reads them ───────────────
    const float stem_div        = model.Environment().getProperty<float>("PARAM_FLOAT_CANCER_CELL_STEM_DIV_INTERVAL_SLICE");
    const float prog_div        = model.Environment().getProperty<float>("PARAM_FLOAT_CANCER_CELL_PROGENITOR_DIV_INTERVAL_SLICE");
    const int   prog_max        = model.Environment().getProperty<int>("PARAM_PROG_DIV_MAX");
    const float cancer_sen_life = model.Environment().getProperty<float>("PARAM_CANCER_SENESCENT_MEAN_LIFE");
    const float tcell_life      = model.Environment().getProperty<float>("PARAM_T_CELL_LIFE_MEAN_SLICE");
    const float tcell_life_sd   = model.Environment().getProperty<float>("PARAM_TCELL_LIFESPAN_SD");
    const int   tcell_div_limit = model.Environment().getProperty<int>("PARAM_TCELL_DIV_LIMIT");
    const float IL2_release_time = model.Environment().getProperty<float>("PARAM_TCELL_IL2_RELEASE_TIME");
    const int   tcell_div_interval = model.Environment().getProperty<int>("PARAM_TCELL_DIV_INTERNAL");
    const float th_life         = model.Environment().getProperty<float>("PARAM_TCD4_LIFE_MEAN_SLICE");
    const float treg_life       = model.Environment().getProperty<float>("PARAM_TREG_LIFE_MEAN_SLICE");
    const float treg_life_sd    = model.Environment().getProperty<float>("PARAM_TCD4_LIFESPAN_SD");
    const int   treg_div_limit  = model.Environment().getProperty<int>("PARAM_TCD4_DIV_LIMIT");
    const int   treg_div_interval = model.Environment().getProperty<int>("PARAM_TREG_DIV_INTERVAL");
    const int   th_div_interval   = model.Environment().getProperty<int>("PARAM_TH_DIV_INTERVAL");
    const float treg_tgfb_release = model.Environment().getProperty<float>("PARAM_TCD4_TGFB_RELEASE_TIME");
    const float ctla4_treg      = model.Environment().getProperty<float>("PARAM_CTLA4_TREG");
    const float mdsc_life       = model.Environment().getProperty<float>("PARAM_MDSC_LIFE_MEAN_SLICE");
    const float mac_life        = model.Environment().getProperty<float>("PARAM_MAC_LIFE_MEAN");
    const float fib_life        = model.Environment().getProperty<float>("PARAM_FIB_LIFE_MEAN");
    const float bcell_life      = model.Environment().getProperty<float>("PARAM_BCELL_LIFE_MEAN");
    const float bcell_life_plasma = model.Environment().getProperty<float>("PARAM_BCELL_LIFE_MEAN_PLASMA");
    const float dc_imm_life     = model.Environment().getProperty<float>("PARAM_DC_LIFE_IMMATURE_MEAN");
    const float dc_imm_life_sd  = model.Environment().getProperty<float>("PARAM_DC_LIFE_IMMATURE_SD");
    const float dc_mat_life     = model.Environment().getProperty<float>("PARAM_DC_LIFE_MATURE_MEAN");
    const float dc_mat_life_sd  = model.Environment().getProperty<float>("PARAM_DC_LIFE_MATURE_SD");

    flamegpu::AgentVector cancer_pop(model.Agent(AGENT_CANCER_CELL));
    flamegpu::AgentVector tcell_pop(model.Agent(AGENT_TCELL));
    flamegpu::AgentVector treg_pop(model.Agent(AGENT_TREG));
    flamegpu::AgentVector mdsc_pop(model.Agent(AGENT_MDSC));
    flamegpu::AgentVector mac_pop(model.Agent(AGENT_MACROPHAGE));
    flamegpu::AgentVector fib_pop(model.Agent(AGENT_FIBROBLAST));
    flamegpu::AgentVector vas_pop(model.Agent(AGENT_VASCULAR));
    flamegpu::AgentVector bcell_pop(model.Agent(AGENT_BCELL));
    flamegpu::AgentVector dc_pop(model.Agent(AGENT_DC));

    std::array<int, 9> placed{};
    int n_skipped_vas = 0, n_unknown = 0;

    for (int i = 0; i < n_agents; ++i) {
        const int* row = &table[static_cast<size_t>(i) * PLC_N_COLS];
        const int type  = row[PLC_TYPE];
        const int x     = row[PLC_X];
        const int y     = row[PLC_Y];
        const int z     = row[PLC_Z];
        const int state = row[PLC_STATE];
        const int sub   = row[PLC_SUBTYPE];
        if (x < 0 || x >= gx || y < 0 || y >= gy || z < 0 || z >= gz) continue;

        // det_tag: a unique non-zero lineage tag per placed cell. The high nibble carries the
        // type (as the recruitment path does) so tags are unique ACROSS types, which the
        // deterministic movement waves require; the low bits are the row index, which is
        // deterministic because the file order is fixed.
        const unsigned int det_tag =
            (static_cast<unsigned int>(type + 1) << 28) |
            ((static_cast<unsigned int>(i) + 1u) & 0x0FFFFFFFu);

        switch (type) {
        case PLC_CANCER: {
            cancer_pop.push_back();
            flamegpu::AgentVector::Agent a = cancer_pop.back();
            a.setVariable<int>("x", x); a.setVariable<int>("y", y); a.setVariable<int>("z", z);
            a.setVariable<int>("cell_state", state);
            a.setVariable<unsigned int>("det_tag", det_tag);
            if (state == CANCER_SENESCENT) {
                a.setVariable<int>("divideFlag", 0);
                a.setVariable<int>("divideCD", 0);
                a.setVariable<int>("divideCountRemaining", 0);
                a.setVariable<unsigned int>("stemID", 0);
                a.setVariable<int>("life", draw_life_exp(cancer_sen_life));
            } else if (state == CANCER_STEM) {
                a.setVariable<int>("divideCD", static_cast<int>(stem_div * urand()) + 1);
                a.setVariable<int>("divideFlag", 1);
                a.setVariable<int>("divideCountRemaining", 0);
                a.setVariable<unsigned int>("stemID", static_cast<unsigned int>(a.getID()));
            } else {  // CANCER_PROGENITOR
                a.setVariable<int>("divideCD", static_cast<int>(prog_div * urand()) + 1);
                a.setVariable<int>("divideFlag", 1);
                a.setVariable<int>("divideCountRemaining",
                                   1 + (rand() % (prog_max > 0 ? prog_max : 1)));
                a.setVariable<unsigned int>("stemID", 0);
            }
            ++placed[PLC_CANCER];
            break;
        }
        case PLC_TCELL: {
            tcell_pop.push_back();
            flamegpu::AgentVector::Agent a = tcell_pop.back();
            a.setVariable<int>("x", x); a.setVariable<int>("y", y); a.setVariable<int>("z", z);
            a.setVariable<int>("cell_state", state);
            a.setVariable<int>("life", draw_life_normal(tcell_life, tcell_life_sd));
            a.setVariable<int>("divide_cd", static_cast<int>(tcell_div_interval * urand()) + 1);
            a.setVariable<int>("divide_limit", tcell_div_limit);
            a.setVariable<float>("IL2_release_remain", IL2_release_time);
            a.setVariable<int>("persist_dir_x", 0);
            a.setVariable<int>("persist_dir_y", 0);
            a.setVariable<int>("persist_dir_z", 0);
            a.setVariable<int>("hypoxia_exposure", 0);
            a.setVariable<float>("hypoxia_kill_factor", 1.0f);
            a.setVariable<unsigned int>("det_tag", det_tag);
            ++placed[PLC_TCELL];
            break;
        }
        case PLC_TREG: {
            treg_pop.push_back();
            flamegpu::AgentVector::Agent a = treg_pop.back();
            a.setVariable<int>("x", x); a.setVariable<int>("y", y); a.setVariable<int>("z", z);
            a.setVariable<int>("cell_state", state);
            {
                const float life_m = (state == TCD4_TH || state == TCD4_TFH)
                                     ? th_life : treg_life;
                a.setVariable<int>("life", draw_life_normal(life_m, treg_life_sd));
                const int div_int = (state == TCD4_TH || state == TCD4_TFH)
                                    ? th_div_interval : treg_div_interval;
                a.setVariable<int>("divide_cd", static_cast<int>(div_int * urand()) + 1);
            }
            a.setVariable<int>("divide_limit", treg_div_limit);
            a.setVariable<float>("TGFB_release_remain", treg_tgfb_release);
            a.setVariable<float>("CTLA4", state == TCD4_TREG ? ctla4_treg : 0.0f);
            a.setVariable<int>("persist_dir_x", 0);
            a.setVariable<int>("persist_dir_y", 0);
            a.setVariable<int>("persist_dir_z", 0);
            a.setVariable<unsigned int>("det_tag", det_tag);
            ++placed[PLC_TREG];
            break;
        }
        case PLC_MDSC: {
            mdsc_pop.push_back();
            flamegpu::AgentVector::Agent a = mdsc_pop.back();
            a.setVariable<int>("x", x); a.setVariable<int>("y", y); a.setVariable<int>("z", z);
            a.setVariable<int>("life", draw_life_exp(mdsc_life));
            a.setVariable<int>("persist_dir_x", 0);
            a.setVariable<int>("persist_dir_y", 0);
            a.setVariable<int>("persist_dir_z", 0);
            a.setVariable<unsigned int>("det_tag", det_tag);
            ++placed[PLC_MDSC];
            break;
        }
        case PLC_MAC: {
            mac_pop.push_back();
            flamegpu::AgentVector::Agent a = mac_pop.back();
            a.setVariable<int>("x", x); a.setVariable<int>("y", y); a.setVariable<int>("z", z);
            a.setVariable<int>("cell_state", state);
            a.setVariable<int>("life", draw_life_exp(mac_life));
            a.setVariable<int>("persist_dir_x", 0);
            a.setVariable<int>("persist_dir_y", 0);
            a.setVariable<int>("persist_dir_z", 0);
            a.setVariable<unsigned int>("det_tag", det_tag);
            ++placed[PLC_MAC];
            break;
        }
        case PLC_FIB: {
            fib_pop.push_back();
            flamegpu::AgentVector::Agent a = fib_pop.back();
            a.setVariable<int>("x", x); a.setVariable<int>("y", y); a.setVariable<int>("z", z);
            a.setVariable<int>("cell_state", state);
            a.setVariable<int>("life", draw_life_exp(fib_life));
            a.setVariable<int>("divide_cooldown", 0);
            a.setVariable<int>("divide_count", 0);
            a.setVariable<unsigned int>("det_tag", det_tag);
            ++placed[PLC_FIB];
            break;
        }
        case PLC_VAS:
            // D-4: vasculature is a FIELD, not agents. The writer already excludes
            // vas_phalanx, so reaching here means the file disagrees with that decision.
            ++n_skipped_vas;
            break;
        case PLC_BCELL: {
            bcell_pop.push_back();
            flamegpu::AgentVector::Agent a = bcell_pop.back();
            a.setVariable<int>("x", x); a.setVariable<int>("y", y); a.setVariable<int>("z", z);
            a.setVariable<int>("cell_state", state);
            a.setVariable<int>("life", draw_life_exp(
                state == BCELL_PLASMA ? bcell_life_plasma : bcell_life));
            a.setVariable<int>("dead", 0);
            // An activated B cell by definition already met antigen; a naive one has not.
            a.setVariable<int>("has_antigen", state == BCELL_NAIVE ? 0 : 1);
            a.setVariable<int>("is_breg", 0);
            a.setVariable<int>("activation_timer", 0);
            a.setVariable<int>("divide_flag", 0);
            a.setVariable<int>("divide_cd", 0);
            a.setVariable<int>("divide_limit", 0);
            a.setVariable<int>("divide_wave", 0);
            a.setVariable<int>("persist_dir_x", 0);
            a.setVariable<int>("persist_dir_y", 0);
            a.setVariable<int>("persist_dir_z", 0);
            a.setVariable<int>("neighbor_cancer_count", 0);
            a.setVariable<int>("neighbor_th_count", 0);
            a.setVariable<int>("neighbor_bcell_count", 0);
            a.setVariable<int>("neighbor_fib_count", 0);
            a.setVariable<unsigned int>("det_tag", det_tag);
            ++placed[PLC_BCELL];
            break;
        }
        case PLC_DC: {
            dc_pop.push_back();
            flamegpu::AgentVector::Agent a = dc_pop.back();
            a.setVariable<int>("x", x); a.setVariable<int>("y", y); a.setVariable<int>("z", z);
            a.setVariable<int>("cell_state", state);
            a.setVariable<int>("dc_subtype", sub);
            a.setVariable<int>("life", state == DC_MATURE ? draw_life_normal(dc_mat_life, dc_mat_life_sd)
                                                          : draw_life_normal(dc_imm_life, dc_imm_life_sd));
            a.setVariable<int>("dead", 0);
            // A mature DC is one that has already taken up and is presenting antigen.
            a.setVariable<int>("presentation_capacity", state == DC_MATURE ? 1 : 0);
            a.setVariable<int>("persist_dir_x", 0);
            a.setVariable<int>("persist_dir_y", 0);
            a.setVariable<int>("persist_dir_z", 0);
            a.setVariable<int>("neighbor_tcell_count", 0);
            a.setVariable<int>("neighbor_bcell_count", 0);
            a.setVariable<unsigned int>("det_tag", det_tag);
            ++placed[PLC_DC];
            break;
        }
        default:
            ++n_unknown;
            break;
        }
    }

    // Every agent type must be submitted, including empty ones — FLAMEGPU requires it.
    simulation.setPopulationData(cancer_pop);
    simulation.setPopulationData(tcell_pop);
    simulation.setPopulationData(treg_pop);
    simulation.setPopulationData(mdsc_pop);
    simulation.setPopulationData(mac_pop);
    simulation.setPopulationData(fib_pop);
    simulation.setPopulationData(vas_pop);
    simulation.setPopulationData(bcell_pop);
    simulation.setPopulationData(dc_pop);

    std::cout << "  Placed " << n_agents << " agents:"
              << "  cancer " << placed[PLC_CANCER]
              << ", T " << placed[PLC_TCELL]
              << ", TReg/Th " << placed[PLC_TREG]
              << ", MDSC " << placed[PLC_MDSC]
              << ", MAC " << placed[PLC_MAC]
              << ", FIB " << placed[PLC_FIB]
              << ", BCELL " << placed[PLC_BCELL]
              << ", DC " << placed[PLC_DC] << std::endl;
    if (n_skipped_vas > 0) {
        std::cout << "  NOTE: skipped " << n_skipped_vas << " vascular agent rows — vasculature"
                     " is a field (D-4), so these should not have been written." << std::endl;
    }
    if (n_unknown > 0) {
        std::cerr << "  WARNING: " << n_unknown << " rows had an unrecognised type_id and were"
                     " dropped — the writer and this reader disagree." << std::endl;
    }
    std::cout << "Placement initialization complete\n" << std::endl;
    return true;
}

} // namespace PDAC