#ifndef FLAMEGPU_TNBC_T_CELL_CUH
#define FLAMEGPU_TNBC_T_CELL_CUH

#include "flamegpu/flamegpu.h"
#include "../core/common.cuh"

namespace PDAC {

// Helper function to get Moore neighborhood direction (same as cancer_cell.cuh)
// Indices 0-5 are Von Neumann (face) neighbors
__device__ __forceinline__ void get_moore_direction_t(int idx, int& dx, int& dy, int& dz) {
    const int dirs[26][3] = {
        // Face neighbors (Von Neumann): indices 0-5
        {-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1},
        // Edge neighbors: indices 6-17
        {-1, -1, 0}, {-1, 1, 0}, {1, -1, 0}, {1, 1, 0},
        {-1, 0, -1}, {-1, 0, 1}, {1, 0, -1}, {1, 0, 1},
        {0, -1, -1}, {0, -1, 1}, {0, 1, -1}, {0, 1, 1},
        // Corner neighbors: indices 18-25
        {-1, -1, -1}, {-1, -1, 1}, {-1, 1, -1}, {-1, 1, 1},
        {1, -1, -1}, {1, -1, 1}, {1, 1, -1}, {1, 1, 1}
    };
    dx = dirs[idx][0];
    dy = dirs[idx][1];
    dz = dirs[idx][2];
}

// Von Neumann mask: only face neighbors (bits 0-5)
constexpr unsigned int VON_NEUMANN_MASK_T = 0x3Fu;  // binary: 00111111

// TCell agent function: Broadcast location
FLAMEGPU_AGENT_FUNCTION(tcell_broadcast_location, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const float voxel_size = FLAMEGPU->environment.getProperty<float>("voxel_size");

    FLAMEGPU->message_out.setVariable<int>("agent_type", CELL_TYPE_T);
    FLAMEGPU->message_out.setVariable<int>("agent_id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setVariable<int>("cell_state", FLAMEGPU->getVariable<int>("cell_state"));
    FLAMEGPU->message_out.setVariable<float>("PDL1", 0.0f);  // T cells don't have PDL1
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

// TCell agent function: Scan neighbors and cache available voxels
// Counts cancer cells, Tregs, tracks max PDL1, looks for progenitors
// Also builds bitmask of neighbor voxels with room for T cells
FLAMEGPU_AGENT_FUNCTION(tcell_scan_neighbors, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
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

    int cancer_count = 0;
    int treg_count = 0;
    int all_count = 0;
    float max_PDL1 = 0.0f;
    int found_progenitor = 0;

    // Per-neighbor counts: [direction][0=cancer, 1=tcell, 2=treg]
    int neighbor_counts[26][3] = {{0}};

    for (const auto& msg : FLAMEGPU->message_in(my_pos_x, my_pos_y, my_pos_z)) {
        const int msg_x = msg.getVariable<int>("voxel_x");
        const int msg_y = msg.getVariable<int>("voxel_y");
        const int msg_z = msg.getVariable<int>("voxel_z");

        const int dx = msg_x - my_x;
        const int dy = msg_y - my_y;
        const int dz = msg_z - my_z;

        // Moore neighborhood (excluding self)
        if (abs(dx) <= 1 && abs(dy) <= 1 && abs(dz) <= 1 && !(dx == 0 && dy == 0 && dz == 0)) {
            all_count++;
            const int agent_type = msg.getVariable<int>("agent_type");
            const int agent_state = msg.getVariable<int>("cell_state");
            const float PDL1 = msg.getVariable<float>("PDL1");

            if (PDL1 > max_PDL1) {
                max_PDL1 = PDL1;
            }

            // Find direction index
            int dir_idx = -1;
            for (int i = 0; i < 26; i++) {
                int ddx, ddy, ddz;
                get_moore_direction_t(i, ddx, ddy, ddz);
                if (ddx == dx && ddy == dy && ddz == dz) {
                    dir_idx = i;
                    break;
                }
            }

            if (dir_idx >= 0) {
                if (agent_type == CELL_TYPE_CANCER) {
                    cancer_count++;
                    neighbor_counts[dir_idx][0]++;
                    // TEMPORARY DEBUG: any cancer cell triggers T cell activation (test IFNg collection)
                    // Original: if (agent_state == CANCER_PROGENITOR)
                    if (agent_state == CANCER_STEM || agent_state == CANCER_PROGENITOR) {
                        found_progenitor = 1;
                    }
                } else if (agent_type == CELL_TYPE_T) {
                    neighbor_counts[dir_idx][1]++;
                } else if (agent_type == CELL_TYPE_TREG) {
                    treg_count++;
                    neighbor_counts[dir_idx][2]++;
                }
            }
        }
    }

    // Build available_neighbors mask (voxels with room for T cells)
    // Only scan Von Neumann neighbors for availability, can just skip other directions
    // Counts for interactions already calculated
    unsigned int available_neighbors = 0;
    for (int i = 0; i < 6; i++) {
        int dx, dy, dz;
        get_moore_direction_t(i, dx, dy, dz);
        int nx = my_x + dx;
        int ny = my_y + dy;
        int nz = my_z + dz;

        if (is_in_bounds(nx, ny, nz, size_x, size_y, size_z)) {
            bool has_cancer = (neighbor_counts[i][0] > 0);
            int t_count = neighbor_counts[i][1] + neighbor_counts[i][2];
            int max_cap = has_cancer ? MAX_T_PER_VOXEL_WITH_CANCER : MAX_T_PER_VOXEL;

            if ((t_count < max_cap)) {
                available_neighbors |= (1u << i);
            }
        }
    }

    FLAMEGPU->setVariable<int>("neighbor_cancer_count", cancer_count);
    FLAMEGPU->setVariable<int>("neighbor_Treg_count", treg_count);
    FLAMEGPU->setVariable<int>("neighbor_all_count", all_count);
    FLAMEGPU->setVariable<float>("max_neighbor_PDL1", max_PDL1);
    FLAMEGPU->setVariable<int>("found_progenitor", found_progenitor);
    FLAMEGPU->setVariable<unsigned int>("available_neighbors", available_neighbors);

    return flamegpu::ALIVE;
}

__device__ __forceinline__ float get_PD1_PDL1_TCELL(float PDL1, float Nivo,
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

// Helper: Hill equation for PD1-PDL1 suppression
__device__ __forceinline__ float hill_equation_TCELL(float x, float k50, float n) {
    if (x <= 0.0f) return 0.0f;
    const float xn = powf(x, n);
    const float kn = powf(k50, n);
    return xn / (kn + xn);
}

__device__ __forceinline__ float get_PD1_supp_TCELL(float conc, float n, float k50) {
    return hill_equation_TCELL(conc, k50, n);
}

// TCell agent function: State transitions
FLAMEGPU_AGENT_FUNCTION(tcell_state_step, flamegpu::MessageNone, flamegpu::MessageNone) {
    if (FLAMEGPU->getVariable<int>("dead") == 1) {
        return flamegpu::DEAD;
    }

    // TODO: need coupling from QSP to get this
    float nivo = FLAMEGPU->environment.getProperty<float>("qsp_nivo_tumor");

    int cell_state = FLAMEGPU->getVariable<int>("cell_state");

    // Life countdown
    int life = FLAMEGPU->getVariable<int>("life");
    life--;
    if (life <= 0) {
        FLAMEGPU->setVariable<int>("dead", 1);
        return flamegpu::DEAD;
    }
    FLAMEGPU->setVariable<int>("life", life);

    //Get cell specific parameters
    const int found_progenitor = FLAMEGPU->getVariable<int>("found_progenitor");

    // Get environment parameters
    const float sec_per_slice = FLAMEGPU->environment.getProperty<float>("PARAM_SEC_PER_SLICE");
    const float IFNg_release_rate = FLAMEGPU->environment.getProperty<float>("PARAM_IFNG_RELEASE");
    const float IL2_release_rate = FLAMEGPU->environment.getProperty<float>("PARAM_IL2_RELEASE");
    const float IL2_uptake_rate = FLAMEGPU->environment.getProperty<float>("PARAM_IL2_UPTAKE");
    const float param_cell = FLAMEGPU->environment.getProperty<float>("PARAM_CELL");
    const float exhaust_base_Treg = FLAMEGPU->environment.getProperty<float>("PARAM_EXHUAST_BASE_TREG");
    const float exhaust_base_PDL1 = FLAMEGPU->environment.getProperty<float>("PARAM_EXHUAST_BASE_PDL1");
    const float PD1_PDL1_half = FLAMEGPU->environment.getProperty<float>("PARAM_PD1_PDL1_HALF");
    const float n_PD1_PDL1 = FLAMEGPU->environment.getProperty<float>("PARAM_N_PD1_PDL1");

    // Update IL2 exposure — read directly from PDE
    const int nx_t = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny_t = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int ax_t = FLAMEGPU->getVariable<int>("x");
    const int ay_t = FLAMEGPU->getVariable<int>("y");
    const int az_t = FLAMEGPU->getVariable<int>("z");
    const int voxel_t = az_t * ny_t*nx_t + ay_t * nx_t + ax_t;
    float IL2 = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>("pde_concentration_ptr_2"))[voxel_t];
    float IL2_exposure = FLAMEGPU->getVariable<float>("IL2_exposure");
    IL2_exposure += IL2 * sec_per_slice;

    // Effector cells proliferate on IL2 exposure
    if (IL2_exposure > FLAMEGPU->environment.getProperty<float>("PARAM_TCELL_IL2_PROLIF_TH")) {
        FLAMEGPU->setVariable<int>("divide_flag", 1);
        FLAMEGPU->setVariable<float>("IL2_exposure", 0.0f);
    } else {
        FLAMEGPU->setVariable<float>("IL2_exposure",IL2_exposure);
    }

    // EFF -> CYT: Found cancer progenitor in neighborhood
    if (cell_state == T_CELL_EFF) {
        if (found_progenitor == 1) {
            FLAMEGPU->setVariable<int>("cell_state", T_CELL_CYT);
            FLAMEGPU->setVariable<int>("divide_flag", 1);
            FLAMEGPU->setVariable<float>("IFNg_release_rate", IFNg_release_rate);
            FLAMEGPU->setVariable<float>("IL2_release_rate", IL2_release_rate);
            FLAMEGPU->setVariable<float>("IL2_uptake_rate", -IL2_uptake_rate);
            cell_state = T_CELL_CYT;
        }
    }

    // CYT state: update release timers and check exhaustion
    if (cell_state == T_CELL_CYT) {
        if (found_progenitor == 0){
            FLAMEGPU->setVariable<float>("IFNg_release_rate", 0.0f);
        }
        // Decrement release timer for IL2
        float IL2_release_remain = FLAMEGPU->getVariable<float>("IL2_release_remain");
        if (IL2_release_remain > 0) {
            IL2_release_remain -= sec_per_slice;
            FLAMEGPU->setVariable<float>("IL2_release_remain", IL2_release_remain);
        } else {
            FLAMEGPU->setVariable<float>("IL2_release_rate", 0.0f);
        }

        // Check for exhaustion
        const int neighbor_Treg = FLAMEGPU->getVariable<int>("neighbor_Treg_count");
        const int neighbor_all = FLAMEGPU->getVariable<int>("neighbor_all_count");
        const float max_PDL1 = FLAMEGPU->getVariable<float>("max_neighbor_PDL1");


        bool exhausted = false;
        if (neighbor_Treg > 0){
            float denominator = neighbor_all + param_cell;
            const float q_exh = (denominator > 1e-12f) ? static_cast<float>(neighbor_Treg) / denominator : 1.0f;  // Prevent division by zero
            const float p_exhaust_treg = 1.0f - powf(exhaust_base_Treg, q_exh);

            if (FLAMEGPU->random.uniform<float>() < p_exhaust_treg){
                exhausted = true;
            }
        }
        else if (neighbor_all > 0){
            float bond = get_PD1_PDL1_TCELL(max_PDL1, nivo,
                        FLAMEGPU->environment.getProperty<float>("PARAM_PD1_SYN"),
                        FLAMEGPU->environment.getProperty<float>("PARAM_PDL1_K1"),
                        FLAMEGPU->environment.getProperty<float>("PARAM_PDL1_K2"),
                        FLAMEGPU->environment.getProperty<float>("PARAM_PDL1_K3"));
            float supp = get_PD1_supp_TCELL(bond, PD1_PDL1_half, n_PD1_PDL1);
            float p_exhaust_pdl1 = 1 - powf(exhaust_base_PDL1, supp);

            if (FLAMEGPU->random.uniform<float>() < p_exhaust_pdl1){
                exhausted = true;
            }
        }

        if (exhausted) {
                // Transition to suppressed
                FLAMEGPU->setVariable<int>("cell_state", T_CELL_SUPP);
                FLAMEGPU->setVariable<int>("divide_flag", 0);
                FLAMEGPU->setVariable<int>("divide_limit", 0);
                FLAMEGPU->setVariable<float>("IFNg_release_rate", 0.0f);
                FLAMEGPU->setVariable<float>("IL2_release_rate", 0.0f);
                FLAMEGPU->setVariable<float>("IL2_uptake_rate", 0.0f);
        }
    }

    // Division cooldown
    int divide_cd = FLAMEGPU->getVariable<int>("divide_cd");
    if (divide_cd > 0) {
        divide_cd--;
        FLAMEGPU->setVariable<int>("divide_cd", divide_cd);
    }

    return flamegpu::ALIVE;
}

// Helper: Compare two agents for priority (lower wins)
// Uses ID first, then source position as tiebreaker
__device__ __forceinline__ bool has_higher_priority_t(unsigned int id1, int sx1, int sy1, int sz1,
                                                       unsigned int id2, int sx2, int sy2, int sz2) {
    if (id1 != id2) return id1 < id2;
    if (sx1 != sx2) return sx1 < sx2;
    if (sy1 != sy2) return sy1 < sy2;
    return sz1 < sz2;
}

// TCell agent function: Select movement target and broadcast intent
// Phase 1 of two-phase conflict resolution
// Uses cached available_neighbors mask from scan phase
FLAMEGPU_AGENT_FUNCTION(tcell_select_move_target, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
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
        FLAMEGPU->message_out.setVariable<int>("agent_type", CELL_TYPE_T);
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

    const int cell_state = FLAMEGPU->getVariable<int>("cell_state");

    // Read ECM concentration from grid at current voxel
    auto ecm = FLAMEGPU->environment.getMacroProperty<float,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX>("ecm_grid");
    float ECM_density = ecm[my_x][my_y][my_z];  // ECM concentration (0.0-1.0)

    // If ECM density is high, cell has lower probability to move
    float ECM_sat = ECM_density;  // Direct use of ECM density as movement inhibition

    // High ECM density reduces move probability
    if (FLAMEGPU->random.uniform<float>() < ECM_sat) {
        // Output dummy message (required)
        FLAMEGPU->message_out.setVariable<int>("agent_type", CELL_TYPE_T);
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

    // TODO Gradient based chemotaxis/tumble implementation

    int target_x = -1, target_y = -1, target_z = -1;
    int intent_action = INTENT_NONE;

    // Use cached available_neighbors mask, but only Von Neumann directions for movement
    const unsigned int available_all = FLAMEGPU->getVariable<unsigned int>("available_neighbors");
    const unsigned int available = available_all & VON_NEUMANN_MASK_T;  // Only face neighbors (6 directions)
    int num_available = __popc(available);

    if (num_available > 0) {
        int selected = FLAMEGPU->random.uniform<int>(0, num_available - 1);
        int count = 0;
        for (int i = 0; i < 6; i++) {  // Only iterate through Von Neumann directions
            if (available & (1u << i)) {
                if (count == selected) {
                    int dx, dy, dz;
                    get_moore_direction_t(i, dx, dy, dz);
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
    FLAMEGPU->message_out.setVariable<int>("agent_type", CELL_TYPE_T);
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

// TCell agent function: Execute movement if won conflict
// Phase 2 of two-phase conflict resolution
// For T cells, multiple can move to same voxel (up to capacity)
// Higher priority agents win when capacity is exceeded
FLAMEGPU_AGENT_FUNCTION(tcell_execute_move, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
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

    // Count how many T cells/TRegs with higher priority also want this voxel
    int higher_priority_count = 0;
    for (const auto& msg : FLAMEGPU->message_in(target_pos_x, target_pos_y, target_pos_z)) {
        const int msg_target_x = msg.getVariable<int>("target_x");
        const int msg_target_y = msg.getVariable<int>("target_y");
        const int msg_target_z = msg.getVariable<int>("target_z");

        if (msg_target_x == target_x && msg_target_y == target_y && msg_target_z == target_z) {
            const int msg_agent_type = msg.getVariable<int>("agent_type");
            const unsigned int msg_id = msg.getVariable<unsigned int>("agent_id");
            const int msg_intent = msg.getVariable<int>("intent_action");
            const int msg_src_x = msg.getVariable<int>("source_x");
            const int msg_src_y = msg.getVariable<int>("source_y");
            const int msg_src_z = msg.getVariable<int>("source_z");

            if ((msg_agent_type == CELL_TYPE_T || msg_agent_type == CELL_TYPE_TREG) &&
                msg_intent == INTENT_MOVE) {
                // Skip self
                if (my_id == msg_id) {
                    continue;
                }
                // Count agents with higher priority
                if (has_higher_priority_t(msg_id, msg_src_x, msg_src_y, msg_src_z,
                                          my_id, my_x, my_y, my_z)) {
                    higher_priority_count++;
                }
            }
        }
    }

    // If fewer than MAX_T_PER_VOXEL agents have higher priority, we can move
    // But if its a cancer voxel, only 1 T Cell can go (we don't know if its a cancer voxel, just location)
    bool can_move = (higher_priority_count < MAX_T_PER_VOXEL);

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

// TCell agent function: Select division target and broadcast intent
// Uses cached available_neighbors mask from scan phase
FLAMEGPU_AGENT_FUNCTION(tcell_select_divide_target, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
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
        FLAMEGPU->message_out.setVariable<int>("agent_type", CELL_TYPE_T);
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

    const int divide_flag = FLAMEGPU->getVariable<int>("divide_flag");
    const int divide_cd = FLAMEGPU->getVariable<int>("divide_cd");
    const int divide_limit = FLAMEGPU->getVariable<int>("divide_limit");
    int cell_state = FLAMEGPU->getVariable<int>("cell_state");

    if (divide_flag == 0 || divide_cd > 0 || divide_limit <= 0 || cell_state != T_CELL_CYT) {
        FLAMEGPU->message_out.setVariable<int>("agent_type", CELL_TYPE_T);
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

    // Use cached available_neighbors mask
    const unsigned int available_all = FLAMEGPU->getVariable<unsigned int>("available_neighbors");
    const unsigned int available = available_all & VON_NEUMANN_MASK_T;  // Only face neighbors (6 directions)
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
                    get_moore_direction_t(i, dx, dy, dz);
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
    FLAMEGPU->message_out.setVariable<int>("agent_type", CELL_TYPE_T);
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

// TCell agent function: Execute division if won conflict
FLAMEGPU_AGENT_FUNCTION(tcell_execute_divide, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
    const int intent_action = FLAMEGPU->getVariable<int>("intent_action");

    if (intent_action != INTENT_DIVIDE) {
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

    // Count how many T cells/TRegs with higher priority also want this voxel
    int higher_priority_count = 0;
    for (const auto& msg : FLAMEGPU->message_in(target_pos_x, target_pos_y, target_pos_z)) {
        const int msg_target_x = msg.getVariable<int>("target_x");
        const int msg_target_y = msg.getVariable<int>("target_y");
        const int msg_target_z = msg.getVariable<int>("target_z");

        if (msg_target_x == target_x && msg_target_y == target_y && msg_target_z == target_z) {
            const int msg_agent_type = msg.getVariable<int>("agent_type");
            const unsigned int msg_id = msg.getVariable<unsigned int>("agent_id");
            const int msg_intent = msg.getVariable<int>("intent_action");
            const int msg_src_x = msg.getVariable<int>("source_x");
            const int msg_src_y = msg.getVariable<int>("source_y");
            const int msg_src_z = msg.getVariable<int>("source_z");

            if ((msg_agent_type == CELL_TYPE_T || msg_agent_type == CELL_TYPE_TREG) &&
                msg_intent == INTENT_DIVIDE) {
                // Skip self
                if (msg_id == my_id) {
                    continue;
                }
                // Count agents with higher priority
                if (has_higher_priority_t(msg_id, msg_src_x, msg_src_y, msg_src_z,
                                          my_id, my_x, my_y, my_z)) {
                    higher_priority_count++;
                }
            }
        }
    }

    // TCELL should not only check how many are trying to move there, but what is already there
    bool can_divide = (higher_priority_count < MAX_T_PER_VOXEL);

    if (!can_divide) {
        FLAMEGPU->setVariable<int>("intent_action", INTENT_NONE);
        FLAMEGPU->setVariable<int>("target_x", -1);
        FLAMEGPU->setVariable<int>("target_y", -1);
        FLAMEGPU->setVariable<int>("target_z", -1);
        return flamegpu::ALIVE;
    }

    // Proceed with division
    const int cell_state = FLAMEGPU->getVariable<int>("cell_state");
    const int divide_limit = FLAMEGPU->getVariable<int>("divide_limit");
    const float IL2_exposure = FLAMEGPU->getVariable<float>("IL2_exposure");
    
    const int div_interval = FLAMEGPU->environment.getProperty<int>("PARAM_TCELL_DIV_INTERNAL");
    const float IL2_release_time = FLAMEGPU->environment.getProperty<float>("PARAM_TCELL_IL2_RELEASE_TIME");
    const float tcell_life_mean = FLAMEGPU->environment.getProperty<float>("PARAM_T_CELL_LIFE_MEAN_SLICE");

    // Calculate daughter life (exponential distribution)
    const float rnd = FLAMEGPU->random.uniform<float>();
    const int daughter_life = static_cast<int>(tcell_life_mean * logf(1.0f / (rnd + 0.0001f)) + 0.5f);

    // Create daughter cell
    FLAMEGPU->agent_out.setVariable<int>("x", target_x);
    FLAMEGPU->agent_out.setVariable<int>("y", target_y);
    FLAMEGPU->agent_out.setVariable<int>("z", target_z);
    FLAMEGPU->agent_out.setVariable<int>("cell_state", cell_state);
    FLAMEGPU->agent_out.setVariable<int>("divide_flag", 0);
    FLAMEGPU->agent_out.setVariable<int>("divide_cd", div_interval);
    FLAMEGPU->agent_out.setVariable<int>("divide_limit", divide_limit - 1);
    FLAMEGPU->agent_out.setVariable<float>("IL2_exposure", IL2_exposure);
    FLAMEGPU->agent_out.setVariable<float>("IL2_release_remain", IL2_release_time);
    FLAMEGPU->agent_out.setVariable<int>("life", daughter_life > 0 ? daughter_life : 1);

    // Set release rates based on state
    // These will update on the next state step do not worry here
    // if (cell_state == T_CELL_CYT) {
    //     FLAMEGPU->agent_out.setVariable<float>("IFNg_release_rate",
    //         FLAMEGPU->environment.getProperty<float>("PARAM_TCELL_IFNG_RELEASE_RATE"));
    //     FLAMEGPU->agent_out.setVariable<float>("IL2_release_rate",
    //         FLAMEGPU->environment.getProperty<float>("PARAM_TCELL_IL2_RELEASE_RATE"));
    // } else {
    //     FLAMEGPU->agent_out.setVariable<float>("IFNg_release_rate", 0.0f);
    //     FLAMEGPU->agent_out.setVariable<float>("IL2_release_rate", 0.0f);
    // }

    // Update parent
    FLAMEGPU->setVariable<int>("divide_limit", divide_limit - 1);
    FLAMEGPU->setVariable<int>("divide_cd", div_interval);

    // Clear intent
    FLAMEGPU->setVariable<int>("intent_action", INTENT_NONE);
    FLAMEGPU->setVariable<int>("target_x", -1);
    FLAMEGPU->setVariable<int>("target_y", -1);
    FLAMEGPU->setVariable<int>("target_z", -1);

    return flamegpu::ALIVE;
}

// ============================================================
// Occupancy Grid Functions
// ============================================================

// Write this T cell's position to the occupancy grid.
FLAMEGPU_AGENT_FUNCTION(tcell_write_to_occ_grid, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    auto occ = FLAMEGPU->environment.getMacroProperty<unsigned int,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX, NUM_OCC_TYPES>("occ_grid");
    occ[x][y][z][CELL_TYPE_T] += 1u;
    return flamegpu::ALIVE;
}

// Single-phase T cell division using occupancy grid.
// T cells can share a voxel up to MAX_T_PER_VOXEL.
// Uses atomicAdd to claim a slot; reverts with atomicSub if over capacity.
FLAMEGPU_AGENT_FUNCTION(tcell_divide, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int divide_flag  = FLAMEGPU->getVariable<int>("divide_flag");
    const int divide_cd    = FLAMEGPU->getVariable<int>("divide_cd");
    const int divide_limit = FLAMEGPU->getVariable<int>("divide_limit");
    const int cell_state   = FLAMEGPU->getVariable<int>("cell_state");

    if (FLAMEGPU->getVariable<int>("dead") == 1 ||
        divide_flag == 0 || divide_cd > 0 || divide_limit <= 0 ||
        cell_state != T_CELL_CYT) {
        return flamegpu::ALIVE;
    }

    const int my_x  = FLAMEGPU->getVariable<int>("x");
    const int my_y  = FLAMEGPU->getVariable<int>("y");
    const int my_z  = FLAMEGPU->getVariable<int>("z");
    const int size_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int size_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int size_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");

    auto occ = FLAMEGPU->environment.getMacroProperty<unsigned int,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX, NUM_OCC_TYPES>("occ_grid");

    // Collect Von Neumann neighbors below T cell capacity
    int cand_x[6], cand_y[6], cand_z[6];
    int n_cands = 0;
    unsigned int max_cap[6];
    for (int i = 0; i < 6; i++) {
        int dx, dy, dz;
        get_moore_direction_t(i, dx, dy, dz);
        const int nx = my_x + dx, ny = my_y + dy, nz = my_z + dz;
        if (!is_in_bounds(nx, ny, nz, size_x, size_y, size_z)) continue;

        bool has_cancer = (occ[nx][ny][nz][CELL_TYPE_CANCER] == 0u);
        max_cap[n_cands] = static_cast<unsigned int>(has_cancer ? MAX_T_PER_VOXEL_WITH_CANCER : MAX_T_PER_VOXEL);

        if (occ[nx][ny][nz][CELL_TYPE_T] < max_cap[n_cands]) {
            cand_x[n_cands] = nx;
            cand_y[n_cands] = ny;
            cand_z[n_cands] = nz;
            n_cands++;
        }
    }

    if (n_cands == 0) return flamegpu::ALIVE;

    const int div_interval  = FLAMEGPU->environment.getProperty<int>("PARAM_TCELL_DIV_INTERNAL");
    const float IL2_release_time = FLAMEGPU->environment.getProperty<float>("PARAM_TCELL_IL2_RELEASE_TIME");
    const float tcell_life_mean  = FLAMEGPU->environment.getProperty<float>("PARAM_T_CELL_LIFE_MEAN_SLICE");
    const float IL2_exposure     = FLAMEGPU->getVariable<float>("IL2_exposure");

    // Fisher-Yates partial shuffle; try each candidate
    for (int i = 0; i < n_cands; i++) {
        const int j = i + static_cast<int>(FLAMEGPU->random.uniform<float>() * (n_cands - i));
        int tx = cand_x[i]; cand_x[i] = cand_x[j]; cand_x[j] = tx;
        int ty = cand_y[i]; cand_y[i] = cand_y[j]; cand_y[j] = ty;
        int tz = cand_z[i]; cand_z[i] = cand_z[j]; cand_z[j] = tz;
        unsigned int max_curr = max_cap[i]; max_cap[i] = max_cap[j]; max_cap[j] = max_curr;

        // Atomically increment; undo if this pushed count over capacity
        // operator+ performs atomicAdd and returns the OLD value
        const unsigned int old_count = occ[cand_x[i]][cand_y[i]][cand_z[i]][CELL_TYPE_T] + 1u;
        if (old_count >= max_curr) {
            occ[cand_x[i]][cand_y[i]][cand_z[i]][CELL_TYPE_T] -= 1u;  // undo
            continue;  // Voxel was full, try next
        }

        // Won a slot — create daughter
        const float rnd = FLAMEGPU->random.uniform<float>();
        const int daughter_life = static_cast<int>(tcell_life_mean * logf(1.0f / (rnd + 0.0001f)) + 0.5f);

        FLAMEGPU->agent_out.setVariable<int>("x", cand_x[i]);
        FLAMEGPU->agent_out.setVariable<int>("y", cand_y[i]);
        FLAMEGPU->agent_out.setVariable<int>("z", cand_z[i]);
        FLAMEGPU->agent_out.setVariable<int>("cell_state", cell_state);
        FLAMEGPU->agent_out.setVariable<int>("divide_flag", 0);
        FLAMEGPU->agent_out.setVariable<int>("divide_cd", div_interval);
        FLAMEGPU->agent_out.setVariable<int>("divide_limit", divide_limit - 1);
        FLAMEGPU->agent_out.setVariable<float>("IL2_exposure", IL2_exposure);
        FLAMEGPU->agent_out.setVariable<float>("IL2_release_remain", IL2_release_time);
        FLAMEGPU->agent_out.setVariable<int>("life", daughter_life > 0 ? daughter_life : 1);

        // Update parent
        FLAMEGPU->setVariable<int>("divide_limit", divide_limit - 1);
        FLAMEGPU->setVariable<int>("divide_cd", div_interval);

        break;  // Division done
    }

    // Clear parent intent
    FLAMEGPU->setVariable<int>("intent_action", INTENT_NONE);
    FLAMEGPU->setVariable<int>("target_x", -1);
    FLAMEGPU->setVariable<int>("target_y", -1);
    FLAMEGPU->setVariable<int>("target_z", -1);

    return flamegpu::ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(tcell_update_chemicals, flamegpu::MessageNone, flamegpu::MessageNone) {
    // ========== READ CHEMICAL CONCENTRATIONS DIRECTLY FROM PDE ==========
    const int nx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int ax = FLAMEGPU->getVariable<int>("x");
    const int ay = FLAMEGPU->getVariable<int>("y");
    const int az = FLAMEGPU->getVariable<int>("z");
    const int voxel = az * ny*nx + ay * nx + ax;

    float local_IL2 = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>("pde_concentration_ptr_2"))[voxel];

    // ========== COMPUTE DERIVED STATES ==========
    
    // 1. Update cumulative IL2 exposure ///// UPDATES IN STATE STEP NOW /////
    // float IL2_exposure = FLAMEGPU->getVariable<float>("IL2_exposure");
    // const float dt = FLAMEGPU->environment.getProperty<float>("PARAM_SEC_PER_SLICE");
    // IL2_exposure += local_IL2 * dt;  // Integrate over time
    // FLAMEGPU->setVariable<float>("IL2_exposure", IL2_exposure);
    
    return flamegpu::ALIVE;
}

// T Cell agent function: Compute chemical sources
// atomicAdds directly to PDE source/uptake arrays
FLAMEGPU_AGENT_FUNCTION(tcell_compute_chemical_sources, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int dead = FLAMEGPU->getVariable<int>("dead");

    // Dead cells don't produce
    if (dead == 1) {
        return flamegpu::DEAD;
    }

    // Compute voxel index
    const int nx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int ax = FLAMEGPU->getVariable<int>("x");
    const int ay = FLAMEGPU->getVariable<int>("y");
    const int az = FLAMEGPU->getVariable<int>("z");
    const int voxel = az * ny*nx + ay * nx + ax;

    const float vs_cm = FLAMEGPU->environment.getProperty<float>("voxel_size") * 1.0e-4f;
    const float voxel_volume = vs_cm * vs_cm * vs_cm;

    // Read rates set by state_step
    const float IFNg_rate = FLAMEGPU->getVariable<float>("IFNg_release_rate");
    const float IL2_rate  = FLAMEGPU->getVariable<float>("IL2_release_rate");
    // IL2_uptake_rate was stored as negative magnitude in state_step; get positive magnitude
    const float IL2_upt   = -FLAMEGPU->getVariable<float>("IL2_uptake_rate");  // negate to get positive [1/s]

    // IFN-γ secretion → src ptr 1 (IFN), divide by voxel_volume
    if (IFNg_rate > 0.0f) {
        atomicAdd(&reinterpret_cast<float*>(
            FLAMEGPU->environment.getProperty<uint64_t>("pde_source_ptr_1"))[voxel],
            IFNg_rate / voxel_volume);
    }

    // IL-2 secretion → src ptr 2 (IL2), divide by voxel_volume
    if (IL2_rate > 0.0f) {
        atomicAdd(&reinterpret_cast<float*>(
            FLAMEGPU->environment.getProperty<uint64_t>("pde_source_ptr_2"))[voxel],
            IL2_rate / voxel_volume);
    }

    // IL-2 uptake → upt ptr 2 (IL2), positive [1/s], no volume scaling
    if (IL2_upt > 0.0f) {
        atomicAdd(&reinterpret_cast<float*>(
            FLAMEGPU->environment.getProperty<uint64_t>("pde_uptake_ptr_2"))[voxel],
            IL2_upt);
    }

    return flamegpu::ALIVE;
}

// Single-phase T cell movement using occupancy grid.
// Replaces two-phase select_move_target + execute_move.
// T cells use atomicAdd+undo to allow up to MAX_T_PER_VOXEL per voxel.
// Fewer T cells are allowed in voxels occupied by cancer (MAX_T_PER_VOXEL_WITH_CANCER).
// T cell movement using run-tumble chemotaxis toward IFN-γ/CCL2
FLAMEGPU_AGENT_FUNCTION(tcell_move, flamegpu::MessageNone, flamegpu::MessageNone) {
    if (FLAMEGPU->getVariable<int>("dead") == 1) return flamegpu::ALIVE;

    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const int tumble = FLAMEGPU->getVariable<int>("tumble");
    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int grid_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");

    // ECM based movement probability
    // TEMPORARILY DISABLED for debugging
    // auto ecm = FLAMEGPU->environment.getMacroProperty<float,
    //     OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX>("ecm_grid");
    // float ECM_density = ecm[x][y][z];
    // double ECM_sat = ECM_density / (ECM_density + FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ECM_MOT_EC50"));
    // if (FLAMEGPU->random.uniform<float>() < ECM_sat) return flamegpu::ALIVE;

    // Use fixed ECM saturation temporarily
    if (FLAMEGPU->random.uniform<float>() < 0.2f) return flamegpu::ALIVE;

    const float move_dir_x = FLAMEGPU->getVariable<float>("move_direction_x");
    const float move_dir_y = FLAMEGPU->getVariable<float>("move_direction_y");
    const float move_dir_z = FLAMEGPU->getVariable<float>("move_direction_z");

    // Use IFN-γ gradient for chemotaxis (T cell primary attractant) — read directly from PDE
    const int nx_mv = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny_mv = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int voxel_mv = z * ny_mv*nx_mv + y * nx_mv + x;
    const float grad_x = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>("pde_grad_IFN_x"))[voxel_mv];
    const float grad_y = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>("pde_grad_IFN_y"))[voxel_mv];
    const float grad_z = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>("pde_grad_IFN_z"))[voxel_mv];

    auto occ = FLAMEGPU->environment.getMacroProperty<unsigned int,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX, NUM_OCC_TYPES>("occ_grid");

    int target_x = x;
    int target_y = y;
    int target_z = z;

    const float dt = FLAMEGPU->environment.getProperty<float>("PARAM_SEC_PER_SLICE");

    // === RUN PHASE (tumble == 0) ===
    if (tumble == 0) {
        float v_x = move_dir_x / dt;
        float v_y = move_dir_y / dt;
        float v_z = move_dir_z / dt;

        float norm_gradient = std::sqrt(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z);

        // Compute alignment to gradient
        float dot_product = v_x * grad_x + v_y * grad_y + v_z * grad_z;
        float norm_v = std::sqrt(v_x * v_x + v_y * v_y + v_z * v_z);
        float cos_theta = dot_product / (norm_v * norm_gradient);

        // Hill function: bias tumble rate based on gradient alignment and strength
        const float EC50_grad = 1.0f;
        float H_grad = norm_gradient / (norm_gradient + EC50_grad);
        if (cos_theta < 0) H_grad = -H_grad;

        const float lambda = 0.0000168;  // Base tumble rate
        float tumble_rate = (lambda / 2.0f) * (1.0f - cos_theta) * (1.0f - H_grad) * dt;
        float p_tumble = 1.0f - std::exp(-tumble_rate);

        if (FLAMEGPU->random.uniform<float>() < p_tumble) {
            FLAMEGPU->setVariable<int>("tumble", 1);
            return flamegpu::ALIVE;
        }

        // Continue running: move in current direction
        target_x = x + static_cast<int>(std::round(move_dir_x));
        target_y = y + static_cast<int>(std::round(move_dir_y));
        target_z = z + static_cast<int>(std::round(move_dir_z));

        // Check bounds
        if (target_x < 0 || target_x >= grid_x ||
            target_y < 0 || target_y >= grid_y ||
            target_z < 0 || target_z >= grid_z) {
            FLAMEGPU->setVariable<int>("tumble", 1);
            return flamegpu::ALIVE;
        }
    }
    // === TUMBLE PHASE (tumble == 1) ===
    else {
        const float sigma = 0.524f;
        float prob_sum = 0.0f;
        float probs[26];
        int dirs[26][3];
        int n_dirs = 0;

        float norm_movedir = std::sqrt(move_dir_x * move_dir_x +
                                      move_dir_y * move_dir_y +
                                      move_dir_z * move_dir_z);
        if (norm_movedir < 1e-6f) norm_movedir = 1.0f;

        // Sample all 26 Moore neighbors
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                for (int k = -1; k <= 1; k++) {
                    if (i == 0 && j == 0 && k == 0) continue;
                    float dot_product = i * move_dir_x + j * move_dir_y + k * move_dir_z;
                    float norm_dir = std::sqrt(static_cast<float>(i*i + j*j + k*k));
                    float cos_theta = dot_product / (norm_dir * norm_movedir);
                    if (cos_theta > 0) {
                        float rho = std::exp(cos_theta / (sigma * sigma)) / std::exp(1.0f / (sigma * sigma));
                        prob_sum += rho;
                        probs[n_dirs] = prob_sum;
                        dirs[n_dirs][0] = i;
                        dirs[n_dirs][1] = j;
                        dirs[n_dirs][2] = k;
                        n_dirs++;
                    }
                }
            }
        }

        if (n_dirs == 0) {
            FLAMEGPU->setVariable<int>("tumble", 0);
            return flamegpu::ALIVE;
        }

        for (int i = 0; i < n_dirs; i++) probs[i] /= prob_sum;

        // Weighted random selection of next direction
        float r = FLAMEGPU->random.uniform<float>();
        int selected_idx = 0;
        for (int i = 0; i < n_dirs; i++) {
            if (r < probs[i]) {
                selected_idx = i;
                break;
            }
        }

        // Set new movement direction
        FLAMEGPU->setVariable<float>("move_direction_x", static_cast<float>(dirs[selected_idx][0]));
        FLAMEGPU->setVariable<float>("move_direction_y", static_cast<float>(dirs[selected_idx][1]));
        FLAMEGPU->setVariable<float>("move_direction_z", static_cast<float>(dirs[selected_idx][2]));
        FLAMEGPU->setVariable<int>("tumble", 0);

        target_x = x + dirs[selected_idx][0];
        target_y = y + dirs[selected_idx][1];
        target_z = z + dirs[selected_idx][2];
    }

    // Try to move to target voxel (if valid and has capacity)
    if (target_x >= 0 && target_x < grid_x &&
        target_y >= 0 && target_y < grid_y &&
        target_z >= 0 && target_z < grid_z) {

        unsigned int max_t = (occ[target_x][target_y][target_z][CELL_TYPE_CANCER] > 0u)
            ? static_cast<unsigned int>(MAX_T_PER_VOXEL_WITH_CANCER)
            : static_cast<unsigned int>(MAX_T_PER_VOXEL);

        if (occ[target_x][target_y][target_z][CELL_TYPE_T] < max_t) {
            // Won: release old voxel and move
            occ[x][y][z][CELL_TYPE_T] -= 1u;
            FLAMEGPU->setVariable<int>("x", target_x);
            FLAMEGPU->setVariable<int>("y", target_y);
            FLAMEGPU->setVariable<int>("z", target_z);
        }
    }

    return flamegpu::ALIVE;
}

} // namespace PDAC

#endif // PDAC_T_CELL_CUH