#ifndef FLAMEGPU_TNBC_CANCER_CELL_CUH
#define FLAMEGPU_TNBC_CANCER_CELL_CUH

#include "flamegpu/flamegpu.h"
#include "../core/common.cuh"

namespace PDAC {

// Helper function to get Moore neighborhood direction
// Returns direction offset for index 0-25
// Indices 0-5 are Von Neumann (face) neighbors, 6-25 are edge/corner neighbors
__device__ __forceinline__ void get_moore_direction(int idx, int& dx, int& dy, int& dz) {
    // Face neighbors (6): indices 0-5 - these are Von Neumann neighbors
    // Edge neighbors (12): indices 6-17
    // Corner neighbors (8): indices 18-25
    const int dirs[26][3] = {
        // Face neighbors (Von Neumann)
        {-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1},
        // Edge neighbors
        {-1, -1, 0}, {-1, 1, 0}, {1, -1, 0}, {1, 1, 0},
        {-1, 0, -1}, {-1, 0, 1}, {1, 0, -1}, {1, 0, 1},
        {0, -1, -1}, {0, -1, 1}, {0, 1, -1}, {0, 1, 1},
        // Corner neighbors
        {-1, -1, -1}, {-1, -1, 1}, {-1, 1, -1}, {-1, 1, 1},
        {1, -1, -1}, {1, -1, 1}, {1, 1, -1}, {1, 1, 1}
    };
    dx = dirs[idx][0];
    dy = dirs[idx][1];
    dz = dirs[idx][2];
}

// Von Neumann mask: only face neighbors (bits 0-5)
constexpr unsigned int VON_NEUMANN_MASK = 0x3Fu;  // binary: 00111111

// CancerCell agent function: Broadcast location to spatial message list
FLAMEGPU_AGENT_FUNCTION(cancer_broadcast_location, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const float voxel_size = FLAMEGPU->environment.getProperty<float>("voxel_size");

    FLAMEGPU->message_out.setVariable<int>("agent_type", CELL_TYPE_CANCER);
    FLAMEGPU->message_out.setVariable<int>("agent_id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setVariable<int>("cell_state", FLAMEGPU->getVariable<int>("cell_state"));
    FLAMEGPU->message_out.setVariable<float>("PDL1", FLAMEGPU->getVariable<float>("PDL1_syn"));
    FLAMEGPU->message_out.setVariable<int>("voxel_x", x);
    FLAMEGPU->message_out.setVariable<int>("voxel_y", y);
    FLAMEGPU->message_out.setVariable<int>("voxel_z", z);

    FLAMEGPU->message_out.setLocation(
        (x + 0.5f) * voxel_size,
        (y + 0.5f) * voxel_size,
        (z + 0.5f) * voxel_size
    );

    return flamegpu::ALIVE;
}

// CancerCell agent function: Count neighbors and cache available voxels
// Counts cytotoxic T cells (for killing), Tregs (for suppression), and other cancer cells
// Also builds a bitmask of which neighbor voxels have no cancer cell (available for move/divide)
FLAMEGPU_AGENT_FUNCTION(cancer_count_neighbors, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
    const int my_x = FLAMEGPU->getVariable<int>("x");
    const int my_y = FLAMEGPU->getVariable<int>("y");
    const int my_z = FLAMEGPU->getVariable<int>("z");
    const float voxel_size = FLAMEGPU->environment.getProperty<float>("voxel_size");
    const int size_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int size_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int size_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");

    const float my_pos_x = (my_x + 0.5f) * voxel_size;
    const float my_pos_y = (my_y + 0.5f) * voxel_size;
    const float my_pos_z = (my_z + 0.5f) * voxel_size;
    const unsigned int my_id = FLAMEGPU->getID();

    int tcyt_count = 0;    // Cytotoxic T cells (can kill)
    int treg_count = 0;    // Regulatory T cells
    int cancer_count = 0;  // Other cancer cells in neighborhood
    int mdsc_count = 0;    // MDSCs (suppress T cell killing)

    // Track which neighbor voxels have cancer cells (bit i = neighbor direction i)
    bool neighbor_blocked[26] = {false};
    int neighbor_tcells[26] = {0};

    for (const auto& msg : FLAMEGPU->message_in(my_pos_x, my_pos_y, my_pos_z)) {
        const int msg_x = msg.getVariable<int>("voxel_x");
        const int msg_y = msg.getVariable<int>("voxel_y");
        const int msg_z = msg.getVariable<int>("voxel_z");

        const int dx = msg_x - my_x;
        const int dy = msg_y - my_y;
        const int dz = msg_z - my_z;

        // Moore neighborhood (excluding self)
        if (abs(dx) <= 1 && abs(dy) <= 1 && abs(dz) <= 1 && !(dx == 0 && dy == 0 && dz == 0)) {
            const int agent_type = msg.getVariable<int>("agent_type");
            const int agent_cell_state = msg.getVariable<int>("cell_state");

            // Find direction index
            int dir_idx = -1;
            for (int i = 0; i < 26; i++) {
                int ddx, ddy, ddz;
                get_moore_direction(i, ddx, ddy, ddz);
                if (ddx == dx && ddy == dy && ddz == dz) {
                    dir_idx = i;
                    break;
                }
            }

            if (agent_type == CELL_TYPE_T) {
                if (agent_cell_state == T_CELL_CYT || agent_cell_state == T_CELL_EFF) {
                    tcyt_count++;
                }
                neighbor_tcells[dir_idx]++;
            } else if (agent_type == CELL_TYPE_TREG) {
                treg_count++;
                neighbor_tcells[dir_idx]++;
            } else if (agent_type == CELL_TYPE_MDSC) {
                mdsc_count++;
            } else if (agent_type == CELL_TYPE_CANCER) {
                cancer_count++;
                neighbor_blocked[dir_idx] = true;
            }
        }
    }

    // Build available_neighbors mask (voxels with no cancer, 1 or fewer tcells, no MDSCs AND in bounds)
    // Only scan Von Neumann neighbors for availability, can just skip other directions
    // Counts for interactions already calculated
    unsigned int available_neighbors = 0;
    for (int i = 0; i < 6; i++) {
        int dx, dy, dz;
        get_moore_direction(i, dx, dy, dz);
        int nx = my_x + dx;
        int ny = my_y + dy;
        int nz = my_z + dz;

        if (is_in_bounds(nx, ny, nz, size_x, size_y, size_z)) {
            // Cancer cells can move to voxel only if no other Cancer cell, MDSC, or more than 1 T cell
            if (!neighbor_blocked[i] && neighbor_tcells[i] <= 1) {
                available_neighbors |= (1u << i);
            }
        }
    }

    FLAMEGPU->setVariable<int>("neighbor_Teff_count", tcyt_count);
    FLAMEGPU->setVariable<int>("neighbor_Treg_count", treg_count);
    FLAMEGPU->setVariable<int>("neighbor_cancer_count", cancer_count);
    FLAMEGPU->setVariable<int>("neighbor_MDSC_count", mdsc_count);
    FLAMEGPU->setVariable<unsigned int>("available_neighbors", available_neighbors);

    return flamegpu::ALIVE;
}

// CancerCell agent function: Select movement target and broadcast intent
// Phase 1 of two-phase conflict resolution
// Uses cached available_neighbors mask from scan phase (no message re-iteration)
FLAMEGPU_AGENT_FUNCTION(cancer_select_move_target, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
    // Clear previous intent
    FLAMEGPU->setVariable<int>("intent_action", INTENT_NONE);
    FLAMEGPU->setVariable<int>("target_x", -1);
    FLAMEGPU->setVariable<int>("target_y", -1);
    FLAMEGPU->setVariable<int>("target_z", -1);

    const float voxel_size = FLAMEGPU->environment.getProperty<float>("voxel_size");
    const int my_x = FLAMEGPU->getVariable<int>("x");
    const int my_y = FLAMEGPU->getVariable<int>("y");
    const int my_z = FLAMEGPU->getVariable<int>("z");

    int moves_remaining = FLAMEGPU->getVariable<int>("moves_remaining");
    if (FLAMEGPU->getVariable<int>("dead") == 1 || moves_remaining <= 0) {
        // Output dummy message (required)
        FLAMEGPU->message_out.setVariable<int>("agent_type", CELL_TYPE_CANCER);
        FLAMEGPU->message_out.setVariable<unsigned int>("agent_id", FLAMEGPU->getID());
        FLAMEGPU->message_out.setVariable<int>("intent_action", INTENT_NONE);
        FLAMEGPU->message_out.setVariable<int>("target_x", -1);
        FLAMEGPU->message_out.setVariable<int>("target_y", -1);
        FLAMEGPU->message_out.setVariable<int>("target_z", -1);
        FLAMEGPU->message_out.setVariable<int>("source_x", my_x);
        FLAMEGPU->message_out.setVariable<int>("source_y", my_y);
        FLAMEGPU->message_out.setVariable<int>("source_z", my_z);
        FLAMEGPU->message_out.setLocation(-voxel_size, -voxel_size, -voxel_size);
        return flamegpu::ALIVE;
    }


    // float ECM_density = 1.0f;  // Placeholder for ECM density, should be read from environment or agent variable
    // float ECM_50 = 0.5f;  // Placeholder for ECM density at which move probability is halved, should be set in environment
    // float ECM_sat = ECM_density / (ECM_density + ECM_50);  // Saturation function to reduce move probability in dense ECM

    float ECM_sat = 0.2;

    // Density too high, do not move, but still broadcast intent for conflict resolution
    if (FLAMEGPU->random.uniform<float>() < ECM_sat) {
        // Output dummy message (required)
        FLAMEGPU->message_out.setVariable<int>("agent_type", CELL_TYPE_CANCER);
        FLAMEGPU->message_out.setVariable<unsigned int>("agent_id", FLAMEGPU->getID());
        FLAMEGPU->message_out.setVariable<int>("intent_action", INTENT_NONE);
        FLAMEGPU->message_out.setVariable<int>("target_x", -1);
        FLAMEGPU->message_out.setVariable<int>("target_y", -1);
        FLAMEGPU->message_out.setVariable<int>("target_z", -1);
        FLAMEGPU->message_out.setVariable<int>("source_x", my_x);
        FLAMEGPU->message_out.setVariable<int>("source_y", my_y);
        FLAMEGPU->message_out.setVariable<int>("source_z", my_z);
        FLAMEGPU->message_out.setLocation(-voxel_size, -voxel_size, -voxel_size);
        return flamegpu::ALIVE;
    }
    // Use cached available_neighbors mask, but only Von Neumann directions for movement
    const unsigned int available_all = FLAMEGPU->getVariable<unsigned int>("available_neighbors");
    const unsigned int available = available_all & VON_NEUMANN_MASK;  // Only face neighbors (6 directions)

    // Count available Von Neumann neighbors
    int num_available = __popc(available);  // Population count (number of set bits)

    int target_x = -1, target_y = -1, target_z = -1;
    int intent_action = INTENT_NONE;
    const unsigned int my_id = FLAMEGPU->getID();

    if (num_available > 0) {
        // Randomly select one of the available Von Neumann neighbors
        int selected = FLAMEGPU->random.uniform<int>(0, num_available - 1);
        int count = 0;
        for (int i = 0; i < 6; i++) {  // Only iterate through Von Neumann directions
            if (available & (1u << i)) {
                if (count == selected) {
                    int dx, dy, dz;
                    get_moore_direction(i, dx, dy, dz);
                    target_x = my_x + dx;
                    target_y = my_y + dy;
                    target_z = my_z + dz;
                    intent_action = INTENT_MOVE;
                    break;
                }
                count++;
            }
        }

        FLAMEGPU->setVariable<int>("intent_action", intent_action);
        FLAMEGPU->setVariable<int>("target_x", target_x);
        FLAMEGPU->setVariable<int>("target_y", target_y);
        FLAMEGPU->setVariable<int>("target_z", target_z);
    }
    

    // Broadcast intent message with source position for conflict resolution
    FLAMEGPU->message_out.setVariable<int>("agent_type", CELL_TYPE_CANCER);
    FLAMEGPU->message_out.setVariable<unsigned int>("agent_id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setVariable<int>("intent_action", intent_action);
    FLAMEGPU->message_out.setVariable<int>("target_x", target_x);
    FLAMEGPU->message_out.setVariable<int>("target_y", target_y);
    FLAMEGPU->message_out.setVariable<int>("target_z", target_z);
    FLAMEGPU->message_out.setVariable<int>("source_x", my_x);
    FLAMEGPU->message_out.setVariable<int>("source_y", my_y);
    FLAMEGPU->message_out.setVariable<int>("source_z", my_z);

    if (intent_action != INTENT_NONE) {
        FLAMEGPU->message_out.setLocation(
            (target_x + 0.5f) * voxel_size,
            (target_y + 0.5f) * voxel_size,
            (target_z + 0.5f) * voxel_size
        );
    } else {
        FLAMEGPU->message_out.setLocation(-voxel_size, -voxel_size, -voxel_size);
    }
    FLAMEGPU->setVariable<int>("moves_remaining", moves_remaining - 1);
    return flamegpu::ALIVE;
}

// Helper: Compare two agents for priority (lower wins)
// Returns true if agent1 has HIGHER priority (should win) over agent2
// Uses ID first, then source position as tiebreaker (lexicographic order)
__device__ __forceinline__ bool has_higher_priority(unsigned int id1, int sx1, int sy1, int sz1,
                                                     unsigned int id2, int sx2, int sy2, int sz2) {
    if (id1 != id2) return id1 < id2;
    if (sx1 != sx2) return sx1 < sx2;
    if (sy1 != sy2) return sy1 < sy2;
    return sz1 < sz2;
}

// CancerCell agent function: Execute movement if won conflict
// Phase 2 of two-phase conflict resolution
FLAMEGPU_AGENT_FUNCTION(cancer_execute_move, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
    const int intent_action = FLAMEGPU->getVariable<int>("intent_action");

    if (intent_action != INTENT_MOVE) {
        return flamegpu::ALIVE;
    }

    const int target_x = FLAMEGPU->getVariable<int>("target_x");
    const int target_y = FLAMEGPU->getVariable<int>("target_y");
    const int target_z = FLAMEGPU->getVariable<int>("target_z");
    const unsigned int my_id = FLAMEGPU->getID();
    const int my_x = FLAMEGPU->getVariable<int>("x");
    const int my_y = FLAMEGPU->getVariable<int>("y");
    const int my_z = FLAMEGPU->getVariable<int>("z");
    const float voxel_size = FLAMEGPU->environment.getProperty<float>("voxel_size");

    const float target_pos_x = (target_x + 0.5f) * voxel_size;
    const float target_pos_y = (target_y + 0.5f) * voxel_size;
    const float target_pos_z = (target_z + 0.5f) * voxel_size;

    // Check if any other agent with higher priority also wants this voxel
    bool can_move = true;
    for (const auto& msg : FLAMEGPU->message_in(target_pos_x, target_pos_y, target_pos_z)) {
        const int msg_target_x = msg.getVariable<int>("target_x");
        const int msg_target_y = msg.getVariable<int>("target_y");
        const int msg_target_z = msg.getVariable<int>("target_z");

        // Check if this message is for the same target voxel
        if (msg_target_x == target_x && msg_target_y == target_y && msg_target_z == target_z) {
            const int msg_agent_type = msg.getVariable<int>("agent_type");
            const unsigned int msg_id = msg.getVariable<unsigned int>("agent_id");
            const int msg_intent = msg.getVariable<int>("intent_action");
            const int msg_src_x = msg.getVariable<int>("source_x");
            const int msg_src_y = msg.getVariable<int>("source_y");
            const int msg_src_z = msg.getVariable<int>("source_z");

            // Skip self (same source position means it's our own message)
            if (msg_id == my_id) {
                continue;
            }

            // Skip if not a competing agent type
            if (!(msg_agent_type == CELL_TYPE_CANCER)) {
                continue;
            }

            // If another agent has higher priority, we lose
            if (has_higher_priority(msg_id, msg_src_x, msg_src_y, msg_src_z,
                                    my_id, my_x, my_y, my_z)) {
                can_move = false;
            }
        }
    }

    if (can_move) {
        FLAMEGPU->setVariable<int>("x", target_x);
        FLAMEGPU->setVariable<int>("y", target_y);
        FLAMEGPU->setVariable<int>("z", target_z);
    }

    // Clear intent
    FLAMEGPU->setVariable<int>("intent_action", INTENT_NONE);
    FLAMEGPU->setVariable<int>("target_x", -1);
    FLAMEGPU->setVariable<int>("target_y", -1);
    FLAMEGPU->setVariable<int>("target_z", -1);

    return flamegpu::ALIVE;
}

// Helper: Hill equation for PD1-PDL1 suppression
__device__ __forceinline__ float hill_equation_cancer(float x, float k50, float n) {
    if (x <= 0.0f) return 0.0f;
    const float xn = powf(x, n);
    const float kn = powf(k50, n);
    return xn / (kn + xn);
}

// Helper: Calculate T cell killing probability
__device__ __forceinline__ float get_kill_probability(
    float supp, float q, float kill_rate) {
	return 1 - std::pow(kill_rate, q*(1-supp));
}

__device__ __forceinline__ float get_PD1_PDL1(float PDL1, float Nivo,
     float T1, float k1, float k2, float k3) {
    
    double T2 = PDL1;

    double a = 1;
	double b = (Nivo*k2/k1*(2*k3/k1-1) - 2*T2 - T1 - 1/k1)/T2;
	double c = (Nivo*k2/k1 + 1/k1  +T2 + 2*T1 )/T2;
	double d = -T1/T2;

	//Newton_Raphson_root
	int max_iter = 20;
	double tol_rel = 1E-5;
	double root = 0;
	double res, root_new, f, f1;
	int i = 0;
	while (i < max_iter){
		f = a*std::pow(root, 3) + b*std::pow(root, 2)+ c*root + d;
		f1 = 3.0*a*std::pow(root, 2) + 2.0*b*root + c;
		root_new = root - f/f1;
		res = std::abs(root_new - root) / root_new;
		if (res > tol_rel){
			i++;
			root = root_new;
		}
		else{
			break;
		}
	}

	return T2*root;
}

__device__ __forceinline__ float get_PD1_supp(float conc, float n, float k50) {
    return hill_equation_cancer(conc, k50, n);
}

// CancerCell agent function: State update (Senesence updating, includes T cell killing and PDL1 dynamics)
FLAMEGPU_AGENT_FUNCTION(cancer_cell_state_step, flamegpu::MessageNone, flamegpu::MessageNone) {
    if (FLAMEGPU->getVariable<int>("dead") == 1) {
        return flamegpu::DEAD;
    }

    const int cell_state = FLAMEGPU->getVariable<int>("cell_state");

    // Senescent cells: countdown to death
    if (cell_state == CANCER_SENESCENT) {
        int life = FLAMEGPU->getVariable<int>("life");
        life--;
        if (life <= 0) {
            FLAMEGPU->setVariable<int>("dead", 1);
            return flamegpu::DEAD;
        }
        FLAMEGPU->setVariable<int>("life", life);
    }

    // === T CELL KILLING ===
    // Check if this cancer cell gets killed by neighboring cytotoxic T cells
    int neighbor_Teff = FLAMEGPU->getVariable<int>("neighbor_Teff_count");
    float total_cancer = static_cast<float>(FLAMEGPU->environment.getProperty<int>("total_cancer_cells"));
    float tumor_size = FLAMEGPU->environment.getProperty<float>("PARAM_GRID_SIZE");
    float cancer_ratio = total_cancer/tumor_size;

    if (neighbor_Teff > 0 && cancer_ratio > 0.1) {
        const int neighbor_cancer = FLAMEGPU->getVariable<int>("neighbor_cancer_count");

        const float PDL1 = FLAMEGPU->getVariable<float>("PDL1_syn");
        float nivo = FLAMEGPU->environment.getProperty<float>("qsp_nivo_tumor");
        float bond = get_PD1_PDL1(PDL1, nivo, 
                        FLAMEGPU->environment.getProperty<float>("PARAM_PD1_SYN"),
                        FLAMEGPU->environment.getProperty<float>("PARAM_PDL1_K1"),
                        FLAMEGPU->environment.getProperty<float>("PARAM_PDL1_K2"),
                        FLAMEGPU->environment.getProperty<float>("PARAM_PDL1_K3"));
        float supp = get_PD1_supp(bond, 
                        FLAMEGPU->environment.getProperty<float>("PARAM_N_PD1_PDL1"),
                        FLAMEGPU->environment.getProperty<float>("PARAM_PD1_PDL1_HALF"));

        float NO = FLAMEGPU->getVariable<float>("local_NO");
        float ArgI = FLAMEGPU->getVariable<float>("local_ArgI");
        float TGFB = FLAMEGPU->getVariable<float>("local_TGFB");

        float H_mdsc_c1 = 1 - (1 - (ArgI / (ArgI 
            + FLAMEGPU->environment.getProperty<float>("PARAM_MDSC_IC50_ArgI_CTL"))));
        
        float H_TGFB = (TGFB / (TGFB 
            + FLAMEGPU->environment.getProperty<float>("PARAM_TEFF_TGFB_EC50")));

        float q = (1 - H_mdsc_c1) * float(neighbor_Teff) 
            / (neighbor_Teff + neighbor_cancer 
                + FLAMEGPU->environment.getProperty<float>("PARAM_CELL")) * (1 - H_TGFB);

        float ECM_sat = 0.5f;

        float killing_scalar = FLAMEGPU->environment.getProperty<float>("PARAM_TKILL_SCALAR")
         * (1-supp);

        float p_kill = get_kill_probability(supp, q, 
            FLAMEGPU->environment.getProperty<float>("PARAM_ESCAPE_BASE"));

        // Probabilistic killing
        if (FLAMEGPU->random.uniform<float>() < p_kill) {
            FLAMEGPU->setVariable<int>("dead", 1);
            return flamegpu::DEAD;
        }
    }
    // TODO: repeat for Macrophages once added

    // === DIVISION COOLDOWN ===
    int divideCD = FLAMEGPU->getVariable<int>("divideCD");
    if (divideCD > 0) {
        divideCD--;
        FLAMEGPU->setVariable<int>("divideCD", divideCD);
    }

    // Hypoxia check
    if (cell_state == CANCER_PROGENITOR) {
        int hypoxic = FLAMEGPU->getVariable<int>("hypoxic");
        if (hypoxic == 1) {
            FLAMEGPU->setVariable("divideFlag",0);
        } else {
            FLAMEGPU->setVariable("divideFlag",1);
        }
    }

    // === PROGENITOR EXHAUSTION ===
    if (cell_state == CANCER_PROGENITOR) {
        const int divideCountRemaining = FLAMEGPU->getVariable<int>("divideCountRemaining");
        if (divideCountRemaining <= 0) {
            FLAMEGPU->setVariable<int>("cell_state", CANCER_SENESCENT);
            FLAMEGPU->setVariable<int>("divideCD", -1);
            FLAMEGPU->setVariable<int>("divideFlag", 0);

            const float mean_life = FLAMEGPU->environment.getProperty<float>("cancer_senescent_mean_life");
            const float rand_val = FLAMEGPU->random.uniform<float>();
            const int life = static_cast<int>(-mean_life * logf(rand_val + 0.0001f) + 0.5f);
            FLAMEGPU->setVariable<int>("life", life > 0 ? life : 1);
        }
    }

    return flamegpu::ALIVE;
}

// Diagnostic: Check for voxel packing after movement
FLAMEGPU_AGENT_FUNCTION(cancer_check_voxel_packing, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
    const int my_x = FLAMEGPU->getVariable<int>("x");
    const int my_y = FLAMEGPU->getVariable<int>("y");
    const int my_z = FLAMEGPU->getVariable<int>("z");
    const unsigned int my_id = FLAMEGPU->getID();
    const float voxel_size = FLAMEGPU->environment.getProperty<float>("voxel_size");

    const float my_pos_x = (my_x + 0.5f) * voxel_size;
    const float my_pos_y = (my_y + 0.5f) * voxel_size;
    const float my_pos_z = (my_z + 0.5f) * voxel_size;

    // Count cancer cells at MY exact voxel position and collect IDs
    int colocated_count = 0;
    unsigned int colocated_ids[10];  // Store up to 10 IDs
    for (const auto& msg : FLAMEGPU->message_in(my_pos_x, my_pos_y, my_pos_z)) {
        const unsigned int msg_id = msg.getVariable<unsigned int>("agent_id");
        if (msg_id == my_id){
            continue;
        }
        const int msg_x = msg.getVariable<int>("voxel_x");
        const int msg_y = msg.getVariable<int>("voxel_y");
        const int msg_z = msg.getVariable<int>("voxel_z");
        const int msg_type = msg.getVariable<int>("agent_type");

        if (msg_type == CELL_TYPE_CANCER &&
            msg_x == my_x && msg_y == my_y && msg_z == my_z) {
            if (colocated_count < 10) {
                colocated_ids[colocated_count] = msg.getVariable<int>("agent_id");
            }
            colocated_count++;
        }
    }

    // Report if multiple cells at same voxel (only print for one cell per location)
    // Use the cell with the LOWEST ID to avoid duplicate prints
    // colocated_count is OTHER cells (self already filtered), so > 0 means packing
    bool should_print = (colocated_count > 0 && my_id);
    if (should_print) {
        // Check if I have the lowest ID
        for (int i = 0; i < colocated_count && i < 10; i++) {
            if (colocated_ids[i] < my_id) {
                should_print = false;
                break;
            }
        }
    }

    if (should_print) {
        printf("[PACKING] Voxel (%d,%d,%d) has %d cancer cells. My ID=%u, Others: ",
               my_x, my_y, my_z, colocated_count + 1, my_id);
        for (int i = 0; i < colocated_count && i < 10; i++) {
            printf("%u%s", colocated_ids[i], (i < colocated_count-1 && i < 9) ? ", " : "");
        }
        printf("\n");
    }

    return flamegpu::ALIVE;
}

// CancerCell agent function: Select division target and broadcast intent
// Phase 1 of two-phase conflict resolution for division
// Uses cached available_neighbors mask from scan phase
FLAMEGPU_AGENT_FUNCTION(cancer_select_divide_target, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
    // Clear previous intent
    FLAMEGPU->setVariable<int>("intent_action", INTENT_NONE);
    FLAMEGPU->setVariable<int>("target_x", -1);
    FLAMEGPU->setVariable<int>("target_y", -1);
    FLAMEGPU->setVariable<int>("target_z", -1);

    const float voxel_size = FLAMEGPU->environment.getProperty<float>("voxel_size");
    const int my_x = FLAMEGPU->getVariable<int>("x");
    const int my_y = FLAMEGPU->getVariable<int>("y");
    const int my_z = FLAMEGPU->getVariable<int>("z");

    if (FLAMEGPU->getVariable<int>("dead") == 1) {
        FLAMEGPU->message_out.setVariable<int>("agent_type", CELL_TYPE_CANCER);
        FLAMEGPU->message_out.setVariable<unsigned int>("agent_id", FLAMEGPU->getID());
        FLAMEGPU->message_out.setVariable<int>("intent_action", INTENT_NONE);
        FLAMEGPU->message_out.setVariable<int>("target_x", -1);
        FLAMEGPU->message_out.setVariable<int>("target_y", -1);
        FLAMEGPU->message_out.setVariable<int>("target_z", -1);
        FLAMEGPU->message_out.setVariable<int>("source_x", my_x);
        FLAMEGPU->message_out.setVariable<int>("source_y", my_y);
        FLAMEGPU->message_out.setVariable<int>("source_z", my_z);
        FLAMEGPU->message_out.setLocation(-voxel_size, -voxel_size, -voxel_size);
        return flamegpu::ALIVE;
    }

    const int divideFlag = FLAMEGPU->getVariable<int>("divideFlag");
    const int divideCD = FLAMEGPU->getVariable<int>("divideCD");

    if (divideFlag == 0 || divideCD > 0) {
        FLAMEGPU->message_out.setVariable<int>("agent_type", CELL_TYPE_CANCER);
        FLAMEGPU->message_out.setVariable<unsigned int>("agent_id", FLAMEGPU->getID());
        FLAMEGPU->message_out.setVariable<int>("intent_action", INTENT_NONE);
        FLAMEGPU->message_out.setVariable<int>("target_x", -1);
        FLAMEGPU->message_out.setVariable<int>("target_y", -1);
        FLAMEGPU->message_out.setVariable<int>("target_z", -1);
        FLAMEGPU->message_out.setVariable<int>("source_x", my_x);
        FLAMEGPU->message_out.setVariable<int>("source_y", my_y);
        FLAMEGPU->message_out.setVariable<int>("source_z", my_z);
        FLAMEGPU->message_out.setLocation(-voxel_size, -voxel_size, -voxel_size);
        return flamegpu::ALIVE;
    }

    // Use cached available_neighbors mask, but only Von Neumann directions for movement
    const unsigned int available_all = FLAMEGPU->getVariable<unsigned int>("available_neighbors");
    const unsigned int available = available_all & VON_NEUMANN_MASK;  // Only face neighbors (6 directions)
    // Count available Von Neumann neighbors
    int num_available = __popc(available);  // Population count (number of set bits)
    int target_x = my_x, target_y = my_y, target_z = my_z;
    int intent_action = INTENT_NONE;

    if (num_available > 0) {
        int selected = FLAMEGPU->random.uniform<int>(0, num_available - 1);
        int count = 0;
        for (int i = 0; i < 6; i++) {
            if (available & (1u << i)) {
                if (count == selected) {
                    int dx, dy, dz;
                    get_moore_direction(i, dx, dy, dz);
                    target_x = my_x + dx;
                    target_y = my_y + dy;
                    target_z = my_z + dz;
                    intent_action = INTENT_DIVIDE;
                    break;
                }
                count++;
            }
        }

        FLAMEGPU->setVariable<int>("intent_action", intent_action);
        FLAMEGPU->setVariable<int>("target_x", target_x);
        FLAMEGPU->setVariable<int>("target_y", target_y);
        FLAMEGPU->setVariable<int>("target_z", target_z);

    }

    // Broadcast intent message with source position for conflict resolution
    FLAMEGPU->message_out.setVariable<int>("agent_type", CELL_TYPE_CANCER);
    FLAMEGPU->message_out.setVariable<unsigned int>("agent_id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setVariable<int>("intent_action", intent_action);
    FLAMEGPU->message_out.setVariable<int>("target_x", target_x);
    FLAMEGPU->message_out.setVariable<int>("target_y", target_y);
    FLAMEGPU->message_out.setVariable<int>("target_z", target_z);
    FLAMEGPU->message_out.setVariable<int>("source_x", my_x);
    FLAMEGPU->message_out.setVariable<int>("source_y", my_y);
    FLAMEGPU->message_out.setVariable<int>("source_z", my_z);

    if (intent_action != INTENT_NONE) {
        FLAMEGPU->message_out.setLocation(
            (target_x + 0.5f) * voxel_size,
            (target_y + 0.5f) * voxel_size,
            (target_z + 0.5f) * voxel_size
        );
    } else {
        FLAMEGPU->message_out.setLocation(-voxel_size, -voxel_size, -voxel_size);
    }

    return flamegpu::ALIVE;
}

// CancerCell agent function: Execute division if won conflict
// Phase 2 of two-phase conflict resolution for division
FLAMEGPU_AGENT_FUNCTION(cancer_execute_divide, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
    const float voxel_size = FLAMEGPU->environment.getProperty<float>("voxel_size");

    const int intent_action = FLAMEGPU->getVariable<int>("intent_action");
    if (intent_action != INTENT_DIVIDE) {
        return flamegpu::ALIVE;
    }

    const unsigned int my_id = FLAMEGPU->getID();
    const int target_x = FLAMEGPU->getVariable<int>("target_x");
    const int target_y = FLAMEGPU->getVariable<int>("target_y");
    const int target_z = FLAMEGPU->getVariable<int>("target_z");
    const int my_x = FLAMEGPU->getVariable<int>("x");
    const int my_y = FLAMEGPU->getVariable<int>("y");
    const int my_z = FLAMEGPU->getVariable<int>("z");
    const int cell_state = FLAMEGPU->getVariable<int>("cell_state");

    const float target_pos_x = (target_x + 0.5f) * voxel_size;
    const float target_pos_y = (target_y + 0.5f) * voxel_size;
    const float target_pos_z = (target_z + 0.5f) * voxel_size;

    // Check if any other cancer cell with higher priority also wants this voxel
    bool can_divide = true;
    for (const auto& msg : FLAMEGPU->message_in(target_pos_x, target_pos_y, target_pos_z)) {
        const int msg_target_x = msg.getVariable<int>("target_x");
        const int msg_target_y = msg.getVariable<int>("target_y");
        const int msg_target_z = msg.getVariable<int>("target_z");
        const int msg_src_x = msg.getVariable<int>("source_x");
        const int msg_src_y = msg.getVariable<int>("source_y");
        const int msg_src_z = msg.getVariable<int>("source_z");
        const int msg_agent_type = msg.getVariable<int>("agent_type");
        const unsigned int msg_id = msg.getVariable<unsigned int>("agent_id");
        const int msg_intent = msg.getVariable<int>("intent_action");

        // (Any cell whose source position equals our target means that voxel is occupied) 
        // This theoretically shouldn't come up but just checking
        if (msg_src_x == target_x && msg_src_y == target_y && msg_src_z == target_z) {
            // A cell already exists at our target - can't divide there
            // (unless it's a cell type that can coexist, but cancer can't share with cancer/MDSC)
            if (msg_agent_type == CELL_TYPE_CANCER || msg_agent_type == CELL_TYPE_MDSC) {
                can_divide = false;
                break;
            }
        }

        if (msg_target_x == target_x && msg_target_y == target_y && msg_target_z == target_z) {
            // Skip self (same source position means it's our own message)
            if (msg_id == my_id) {
                continue;
            }

            // Skip if not a competing agent type or not dividing
            if (!(msg_agent_type == CELL_TYPE_CANCER)) {
                continue;
            }

            // If another cancer cell has higher priority, we lose
            if (has_higher_priority(msg_id, msg_src_x, msg_src_y, msg_src_z,
                                    my_id, my_x, my_y, my_z)) {
                can_divide = false;
                break;
            }
        }
    }

    if (!can_divide) {
        // Clear intent and exit
        FLAMEGPU->setVariable<int>("intent_action", INTENT_NONE);
        FLAMEGPU->setVariable<int>("target_x", my_x);
        FLAMEGPU->setVariable<int>("target_y", my_y);
        FLAMEGPU->setVariable<int>("target_z", my_z);
        return flamegpu::ALIVE;
    }

    // Proceed with division
    const float asymmetric_div_prob = FLAMEGPU->environment.getProperty<float>("PARAM_ASYM_DIV_PROB");
    const int progenitor_div_max = FLAMEGPU->environment.getProperty<int>("PARAM_PROG_DIV_MAX");
    const int divMax = FLAMEGPU->environment.getProperty<int>("PARAM_PROG_DIV_MAX");
    const unsigned int stem_id = FLAMEGPU->getVariable<unsigned int>("stemID");

    // Generate unique daughter ID using FLAMEGPU's built-in ID generator
    // Don't set ID manually - FLAMEGPU auto-assigns unique IDs for new agents
    // const unsigned int daughter_id = ...; // Remove this - use auto-assigned ID

    if (cell_state == CANCER_STEM) {
        if (FLAMEGPU->random.uniform<float>() < asymmetric_div_prob) {
            float div_int = FLAMEGPU->environment.getProperty<float>(
                                "PARAM_FLOAT_CANCER_CELL_PROGENITOR_DIV_INTERVAL_SLICE");
            const float progenitor_div_interval = static_cast<int>(div_int + 0.5f);
            // Asymmetric: daughter is progenitor
            // ID auto-assigned by FLAMEGPU
            FLAMEGPU->agent_out.setVariable<int>("x", target_x);
            FLAMEGPU->agent_out.setVariable<int>("y", target_y);
            FLAMEGPU->agent_out.setVariable<int>("z", target_z);
            FLAMEGPU->agent_out.setVariable<int>("cell_state", CANCER_PROGENITOR);
            FLAMEGPU->agent_out.setVariable<int>("divideCD", progenitor_div_interval);
            FLAMEGPU->agent_out.setVariable<int>("divideFlag", 1);
            FLAMEGPU->agent_out.setVariable<int>("divideCountRemaining", divMax);
            FLAMEGPU->agent_out.setVariable<unsigned int>("stemID", stem_id);
        } else {
            float div_int = FLAMEGPU->environment.getProperty<float>(
                                "PARAM_FLOAT_CANCER_CELL_STEM_DIV_INTERVAL_SLICE");
            const float stem_div_interval = static_cast<int>(div_int + 0.5f);
            // Symmetric: daughter is stem
            // ID auto-assigned by FLAMEGPU
            FLAMEGPU->agent_out.setVariable<int>("x", target_x);
            FLAMEGPU->agent_out.setVariable<int>("y", target_y);
            FLAMEGPU->agent_out.setVariable<int>("z", target_z);
            FLAMEGPU->agent_out.setVariable<int>("cell_state", CANCER_STEM);
            FLAMEGPU->agent_out.setVariable<int>("divideCD", stem_div_interval);
            FLAMEGPU->agent_out.setVariable<int>("divideFlag", 1);
            FLAMEGPU->agent_out.setVariable<int>("divideCountRemaining", 0);
            FLAMEGPU->agent_out.setVariable<unsigned int>("stemID", stem_id);
        }

    } else if (cell_state == CANCER_PROGENITOR) {
        int divideCountRemaining = FLAMEGPU->getVariable<int>("divideCountRemaining");
        divideCountRemaining--;
        FLAMEGPU->setVariable<int>("divideCountRemaining",divideCountRemaining);
        float div_int = FLAMEGPU->environment.getProperty<float>(
                                "PARAM_FLOAT_CANCER_CELL_PROGENITOR_DIV_INTERVAL_SLICE");
        const float progenitor_div_interval = static_cast<int>(div_int + 0.5f);
        //TODO randomly sample divide count and divide CD for daughter
        // ID auto-assigned by FLAMEGPU
        FLAMEGPU->agent_out.setVariable<int>("x", target_x);
        FLAMEGPU->agent_out.setVariable<int>("y", target_y);
        FLAMEGPU->agent_out.setVariable<int>("z", target_z);
        FLAMEGPU->agent_out.setVariable<int>("cell_state", CANCER_PROGENITOR);
        FLAMEGPU->agent_out.setVariable<int>("divideCD", progenitor_div_interval);
        FLAMEGPU->agent_out.setVariable<int>("divideFlag", 1);
        FLAMEGPU->agent_out.setVariable<int>("divideCountRemaining", divideCountRemaining);
        FLAMEGPU->agent_out.setVariable<unsigned int>("stemID", stem_id);

        // Turn progenitor into senescent cell
        if (divideCountRemaining <= 0) {
            FLAMEGPU->setVariable<int>("cell_state", CANCER_SENESCENT);
            FLAMEGPU->setVariable<int>("divideCD", -1);
            FLAMEGPU->setVariable<int>("divideFlag", 0);

            const float mean_life = FLAMEGPU->environment.getProperty<float>(
                                                "PARAM_CANCER_SENESCENT_MEAN_LIFE");
            const float rand_val = FLAMEGPU->random.uniform<float>();
            const int life = static_cast<int>(-mean_life * logf(rand_val + 0.0001f) + 0.5f);
            FLAMEGPU->setVariable<int>("life", life > 0 ? life : 1);
        }
    }

    // Set common daughter variables
    // This should set to the default value if we don't manually call
    FLAMEGPU->agent_out.setVariable<int>("neighbor_Teff_count", 0);
    FLAMEGPU->agent_out.setVariable<int>("neighbor_Treg_count", 0);
    FLAMEGPU->agent_out.setVariable<int>("neighbor_cancer_count", 0);
    FLAMEGPU->agent_out.setVariable<int>("neighbor_MDSC_count", 0);
    FLAMEGPU->agent_out.setVariable<unsigned int>("available_neighbors", 0u);
    FLAMEGPU->agent_out.setVariable<int>("life", 0);
    FLAMEGPU->agent_out.setVariable<int>("dead", 0);
    FLAMEGPU->agent_out.setVariable<int>("intent_action", INTENT_NONE);
    FLAMEGPU->agent_out.setVariable<int>("target_x", -1);
    FLAMEGPU->agent_out.setVariable<int>("target_y", -1);
    FLAMEGPU->agent_out.setVariable<int>("target_z", -1);

    // Clear parent intent
    FLAMEGPU->setVariable<int>("intent_action", INTENT_NONE);
    FLAMEGPU->setVariable<int>("target_x", -1);
    FLAMEGPU->setVariable<int>("target_y", -1);
    FLAMEGPU->setVariable<int>("target_z", -1);

    // Reset parent divideCD to full interval (only if parent didn't become senescent)
    if (cell_state == CANCER_STEM) {
        float div_int = FLAMEGPU->environment.getProperty<float>(
                            "PARAM_FLOAT_CANCER_CELL_STEM_DIV_INTERVAL_SLICE");
        FLAMEGPU->setVariable<int>("divideCD", static_cast<int>(div_int + 0.5f));
    } else if (cell_state == CANCER_PROGENITOR &&
               FLAMEGPU->getVariable<int>("divideCountRemaining") > 0) {
        float div_int = FLAMEGPU->environment.getProperty<float>(
                            "PARAM_FLOAT_CANCER_CELL_PROGENITOR_DIV_INTERVAL_SLICE");
        FLAMEGPU->setVariable<int>("divideCD", static_cast<int>(div_int + 0.5f));
    }

    return flamegpu::ALIVE;
}

// Cancer Cell agent function: Update chemicals from PDE
// Reads local concentrations and computes molecular responses
FLAMEGPU_AGENT_FUNCTION(cancer_update_chemicals, flamegpu::MessageNone, flamegpu::MessageNone) {
    // ========== READ CHEMICAL CONCENTRATIONS FROM AGENT VARIABLES ==========
    // These were already set by the host function update_agent_chemicals in layer 6
    // No need to access PDE memory directly!

    float local_O2 = FLAMEGPU->getVariable<float>("local_O2");
    float local_IFNg = FLAMEGPU->getVariable<float>("local_IFNg");
    float local_TGFB = FLAMEGPU->getVariable<float>("local_TGFB");
    float local_ArgI = FLAMEGPU->getVariable<float>("local_ArgI");
    float local_NO = FLAMEGPU->getVariable<float>("local_NO");
    
    // ========== COMPUTE DERIVED MOLECULAR STATES ==========
    
    // 1. Hypoxia detection
    const float O2_hypoxia_threshold = FLAMEGPU->environment.getProperty<float>("PARAM_CANCER_HYPOXIA_TH");
    int hypoxic = (local_O2 < O2_hypoxia_threshold) ? 1 : 0;
    FLAMEGPU->setVariable<int>("hypoxic", hypoxic);
    
    // 2. PDL1 upregulation by IFN-gamma (Hill function)
    const float IFNg_PDL1_EC50 = FLAMEGPU->environment.getProperty<float>("PARAM_IFNG_PDL1_HALF");
    const float IFNg_PDL1_hill = FLAMEGPU->environment.getProperty<float>("PARAM_IFNG_PDL1_N");
    float H_IFNg = hill_equation_cancer(local_IFNg, IFNg_PDL1_EC50, IFNg_PDL1_hill);
    const float PDL1_syn_max = FLAMEGPU->environment.getProperty<float>("PARAM_PDL1_SYN_MAX");
    float minPDL1 = PDL1_syn_max * H_IFNg;

    float PDL1_current = FLAMEGPU->getVariable<float>("PDL1_syn");
    if (PDL1_current < minPDL1) {
        FLAMEGPU->setVariable<float>("PDL1_syn", minPDL1);
    }

    // 3. Cabozantinib effect (anti-angiogenic drug, managed by QSP model)
    // TODO: Get CABO concentration from QSP model via environment property
    float cabo_effect = 0.0f;  // Cabozantinib effect handled by CPU-side QSP model
    FLAMEGPU->setVariable<float>("cabo_effect", cabo_effect);

    return flamegpu::ALIVE;
}

// Cancer Cell agent function: Compute chemical source/sink rates
// Sets rates for PDE update based on cell state and environment
FLAMEGPU_AGENT_FUNCTION(cancer_compute_chemical_sources, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int cell_state = FLAMEGPU->getVariable<int>("cell_state");
    const int hypoxic = FLAMEGPU->getVariable<int>("hypoxic");
    const int dead = FLAMEGPU->getVariable<int>("dead");
    
    // Dead cells don't produce or consume
    if (dead == 1) {
        return flamegpu::ALIVE;
    }

    // Get base rates from environment
    float CCL2_release = FLAMEGPU->environment.getProperty<float>("PARAM_CCL2_RELEASE");
    float TGFB_release = 0.0f;
    float VEGFA_release = 0.0f;
    if (cell_state == CANCER_STEM){
        TGFB_release = FLAMEGPU->environment.getProperty<float>("PARAM_STEM_TGFB_RELEASE");
        VEGFA_release = FLAMEGPU->environment.getProperty<float>("PARAM_STEM_VEGFA_RELEASE");
    } else if (cell_state == CANCER_PROGENITOR){
        TGFB_release = FLAMEGPU->environment.getProperty<float>("PARAM_PROG_TGFB_RELEASE");
        VEGFA_release = FLAMEGPU->environment.getProperty<float>("PARAM_PROG_VEGFA_RELEASE");
    }
    float O2_uptake = FLAMEGPU->environment.getProperty<float>("PARAM_O2_UPTAKE");
    float IFNg_uptake = FLAMEGPU->environment.getProperty<float>("PARAM_CANCER_IFNG_UPTAKE");
    
    // ========== CCL2 PRODUCTION (immune cell recruitment) ==========
    if (hypoxic == 1) {
        CCL2_release *= 2.0f;  // Hypoxia-induced CCL2 upregulation
    }
    if (cell_state == CANCER_STEM) {
        CCL2_release *= 1.5f;
    }
    
    // ========== OXYGEN CONSUMPTION (NEGATIVE!) ==========
    if (cell_state == CANCER_PROGENITOR) {
        O2_uptake *= 1.5f;  // Proliferating cells consume more
    }

    if (cell_state == CANCER_SENESCENT){
        CCL2_release = 0.0f;
        TGFB_release = 0.0f;
        VEGFA_release = 0.0f;
    }
    // Sources
    FLAMEGPU->setVariable<float>("CCL2_release_rate", CCL2_release);
    FLAMEGPU->setVariable<float>("TGFB_release_rate", TGFB_release);
    FLAMEGPU->setVariable<float>("VEGFA_release_rate", VEGFA_release);
    // Sinks
    FLAMEGPU->setVariable<float>("O2_uptake_rate", -O2_uptake);  // NEGATIVE for consumption!
    FLAMEGPU->setVariable<float>("IFNg_uptake_rate", -IFNg_uptake);  // NEGATIVE for consumption!
    
    return flamegpu::ALIVE;
}

// Agent function to reset moves based on state
FLAMEGPU_AGENT_FUNCTION(cancer_reset_moves, flamegpu::MessageNone, flamegpu::MessageNone) {
    int cell_state = FLAMEGPU->getVariable<int>("cell_state");
    
    // Get max moves from environment based on state
    int max_moves;
    if (cell_state == CANCER_STEM) {
        max_moves = FLAMEGPU->environment.getProperty<int>("PARAM_CANCER_MOVE_STEPS_STEM");
    } else {
        max_moves = FLAMEGPU->environment.getProperty<int>("PARAM_CANCER_MOVE_STEPS");
    }
    
    FLAMEGPU->setVariable<int>("moves_remaining", max_moves);
    return flamegpu::ALIVE;
}


} // namespace PDAC

#endif // PDAC_CANCER_CELL_CUH
