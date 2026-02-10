#ifndef FLAMEGPU_TNBC_MDSC_CUH
#define FLAMEGPU_TNBC_MDSC_CUH

#include "flamegpu/flamegpu.h"
#include "../core/common.cuh"

namespace PDAC {

// Von Neumann mask: first 6 bits correspond to face neighbors (indices 0-5)
// Directions 0-5 are: {-1,0,0}, {1,0,0}, {0,-1,0}, {0,1,0}, {0,0,-1}, {0,0,1}
constexpr unsigned int VON_NEUMANN_MASK_MDSC = 0x3Fu;  // binary: 00111111

// Helper function to get Moore neighborhood direction for MDSC
__device__ __forceinline__ void get_moore_direction_mdsc(int idx, int& dx, int& dy, int& dz) {
    const int dirs[26][3] = {
        {-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1},
        {-1, -1, 0}, {-1, 1, 0}, {1, -1, 0}, {1, 1, 0},
        {-1, 0, -1}, {-1, 0, 1}, {1, 0, -1}, {1, 0, 1},
        {0, -1, -1}, {0, -1, 1}, {0, 1, -1}, {0, 1, 1},
        {-1, -1, -1}, {-1, -1, 1}, {-1, 1, -1}, {-1, 1, 1},
        {1, -1, -1}, {1, -1, 1}, {1, 1, -1}, {1, 1, 1}
    };
    dx = dirs[idx][0];
    dy = dirs[idx][1];
    dz = dirs[idx][2];
}

// MDSC agent function: Broadcast location
FLAMEGPU_AGENT_FUNCTION(mdsc_broadcast_location, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const float voxel_size = FLAMEGPU->environment.getProperty<float>("voxel_size");

    FLAMEGPU->message_out.setVariable<int>("agent_type", CELL_TYPE_MDSC);
    FLAMEGPU->message_out.setVariable<int>("agent_id", FLAMEGPU->getVariable<unsigned int>("id"));
    FLAMEGPU->message_out.setVariable<int>("cell_state", 0);  // MDSCs have single state
    FLAMEGPU->message_out.setVariable<float>("PDL1", 0.0f);   // MDSCs don't express PDL1
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

// MDSC agent function: Scan neighbors and cache available voxels
// MDSCs check for voxels without other MDSCs (1 MDSC per voxel max)
FLAMEGPU_AGENT_FUNCTION(mdsc_scan_neighbors, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
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
    int tcell_count = 0;
    int treg_count = 0;
    int mdsc_count = 0;

    // Track which neighbor voxels have MDSCs (for movement availability)
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

            // Find direction index
            int dir_idx = -1;
            for (int i = 0; i < 26; i++) {
                int ddx, ddy, ddz;
                get_moore_direction_mdsc(i, ddx, ddy, ddz);
                if (ddx == dx && ddy == dy && ddz == dz) {
                    dir_idx = i;
                    break;
                }
            }

            if (dir_idx >= 0) {
                if (agent_type == CELL_TYPE_CANCER) {
                    cancer_count++;
                    neighbor_blocked[dir_idx] = true;
                } else if (agent_type == CELL_TYPE_T) {
                    tcell_count++;
                    neighbor_tcells[dir_idx]++;
                } else if (agent_type == CELL_TYPE_TREG) {
                    treg_count++;
                    neighbor_tcells[dir_idx]++;
                } else if (agent_type == CELL_TYPE_MDSC) {
                    mdsc_count++;
                    neighbor_blocked[dir_idx] = true;
                }
            }
        }
    }

    // Build available_neighbors mask (voxels without MDSC - since 1 MDSC per voxel max)
    unsigned int available_neighbors = 0;
    for (int i = 0; i < 26; i++) {
        int dx, dy, dz;
        get_moore_direction_mdsc(i, dx, dy, dz);
        int nx = my_x + dx;
        int ny = my_y + dy;
        int nz = my_z + dz;

        if (is_in_bounds(nx, ny, nz, size_x, size_y, size_z)) {
            // MDSC can move to voxel only if no other MDSC is there
            if (!neighbor_blocked[i] && neighbor_tcells[i] == 0) {
                available_neighbors |= (1u << i);
            }
        }
    }

    FLAMEGPU->setVariable<int>("neighbor_cancer_count", cancer_count);
    FLAMEGPU->setVariable<int>("neighbor_Tcell_count", tcell_count);
    FLAMEGPU->setVariable<int>("neighbor_Treg_count", treg_count);
    FLAMEGPU->setVariable<int>("neighbor_MDSC_count", mdsc_count);
    FLAMEGPU->setVariable<unsigned int>("available_neighbors", available_neighbors);

    return flamegpu::ALIVE;
}

// MDSC agent function: State step (life countdown only - MDSCs don't divide)
FLAMEGPU_AGENT_FUNCTION(mdsc_state_step, flamegpu::MessageNone, flamegpu::MessageNone) {
    if (FLAMEGPU->getVariable<int>("dead") == 1) {
        return flamegpu::DEAD;
    }

    // Life countdown
    int life = FLAMEGPU->getVariable<int>("life");
    life--;
    if (life <= 0) {
        FLAMEGPU->setVariable<int>("dead", 1);
        return flamegpu::DEAD;
    }
    FLAMEGPU->setVariable<int>("life", life);

    return flamegpu::ALIVE;
}

// Helper: Compare two agents for priority (lower wins)
__device__ __forceinline__ bool has_higher_priority_mdsc(unsigned int id1, int sx1, int sy1, int sz1,
                                                          unsigned int id2, int sx2, int sy2, int sz2) {
    if (id1 != id2) return id1 < id2;
    if (sx1 != sx2) return sx1 < sx2;
    if (sy1 != sy2) return sy1 < sy2;
    return sz1 < sz2;
}

// MDSC agent function: Select movement target and broadcast intent
// Phase 1 of two-phase conflict resolution
// Uses cached available_neighbors mask from scan phase
FLAMEGPU_AGENT_FUNCTION(mdsc_select_move_target, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
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
        FLAMEGPU->message_out.setVariable<int>("agent_type", CELL_TYPE_MDSC);
        FLAMEGPU->message_out.setVariable<unsigned int>("agent_id", FLAMEGPU->getVariable<unsigned int>("id"));
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

    const float move_prob = FLAMEGPU->environment.getProperty<float>("PARAM_MDSC_MOVE_PROB");

    int target_x = -1, target_y = -1, target_z = -1;
    int intent_action = INTENT_NONE;

    if (FLAMEGPU->random.uniform<float>() < move_prob) {
        // Use cached available_neighbors mask, restricted to Von Neumann (6 face neighbors)
        const unsigned int available_all = FLAMEGPU->getVariable<unsigned int>("available_neighbors");
        const unsigned int available = available_all & VON_NEUMANN_MASK_MDSC;
        int num_available = __popc(available);

        if (num_available > 0) {
            int selected = FLAMEGPU->random.uniform<int>(0, num_available - 1);
            int count = 0;
            // Only iterate over Von Neumann directions (indices 0-5)
            for (int i = 0; i < 6; i++) {
                if (available & (1u << i)) {
                    if (count == selected) {
                        int dx, dy, dz;
                        get_moore_direction_mdsc(i, dx, dy, dz);
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
    }

    // Broadcast intent message with source position for conflict resolution
    FLAMEGPU->message_out.setVariable<int>("agent_type", CELL_TYPE_MDSC);
    FLAMEGPU->message_out.setVariable<unsigned int>("agent_id", FLAMEGPU->getVariable<unsigned int>("id"));
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

// MDSC agent function: Execute movement if won conflict
// Phase 2 of two-phase conflict resolution
// Since 1 MDSC per voxel, only highest priority agent wins
FLAMEGPU_AGENT_FUNCTION(mdsc_execute_move, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
    const int intent_action = FLAMEGPU->getVariable<int>("intent_action");

    if (intent_action != INTENT_MOVE) {
        return flamegpu::ALIVE;
    }

    const int target_x = FLAMEGPU->getVariable<int>("target_x");
    const int target_y = FLAMEGPU->getVariable<int>("target_y");
    const int target_z = FLAMEGPU->getVariable<int>("target_z");
    const unsigned int my_id = FLAMEGPU->getVariable<unsigned int>("id");
    const int my_x = FLAMEGPU->getVariable<int>("x");
    const int my_y = FLAMEGPU->getVariable<int>("y");
    const int my_z = FLAMEGPU->getVariable<int>("z");
    const float voxel_size = FLAMEGPU->environment.getProperty<float>("voxel_size");

    const float target_pos_x = (target_x + 0.5f) * voxel_size;
    const float target_pos_y = (target_y + 0.5f) * voxel_size;
    const float target_pos_z = (target_z + 0.5f) * voxel_size;

    // Check if any other MDSC/Cancer with higher priority also wants this voxel
    bool can_move = true;
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

            // Only 1 MDSC per voxel - higher priority wins
            if ((msg_agent_type == CELL_TYPE_MDSC || msg_agent_type == CELL_TYPE_CANCER) && msg_intent == INTENT_MOVE) {
                // Skip self
                if (msg_src_x == my_x && msg_src_y == my_y && msg_src_z == my_z) {
                    continue;
                }
                // Check if other agent has higher priority
                if (has_higher_priority_mdsc(msg_id, msg_src_x, msg_src_y, msg_src_z,
                                              my_id, my_x, my_y, my_z)) {
                    can_move = false;
                    break;
                }
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

// MDSC agent function: Update chemicals from PDE
FLAMEGPU_AGENT_FUNCTION(mdsc_update_chemicals, flamegpu::MessageNone, flamegpu::MessageNone) {
    // Get agent position
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    
    // Get grid dimensions
    const int grid_x = FLAMEGPU->environment.getProperty<int>("PARAM_GRID_SIZE_X");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("PARAM_GRID_SIZE_Y");
    const int grid_z = FLAMEGPU->environment.getProperty<int>("PARAM_GRID_SIZE_Z");
    
    // Calculate flat voxel index
    const int voxel_idx = z * (grid_x * grid_y) + y * grid_x + x;
    
    // ========== READ CHEMICAL CONCENTRATIONS FROM PDE ==========
    
    const float* d_O2 = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<unsigned long long>("pde_concentration_ptr_0"));
    const float* d_IFN = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<unsigned long long>("pde_concentration_ptr_1"));
    const float* d_IL2 = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<unsigned long long>("pde_concentration_ptr_2"));
    const float* d_IL10 = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<unsigned long long>("pde_concentration_ptr_3"));
    const float* d_TGFB = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<unsigned long long>("pde_concentration_ptr_4"));
    const float* d_CCL2 = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<unsigned long long>("pde_concentration_ptr_5"));
    
    float local_O2 = d_O2[voxel_idx];
    float local_IFNg = d_IFN[voxel_idx];
    float local_IL2 = d_IL2[voxel_idx];
    float local_IL10 = d_IL10[voxel_idx];
    float local_TGFB = d_TGFB[voxel_idx];
    float local_CCL2 = d_CCL2[voxel_idx];
    
    // Set local concentration variables
    FLAMEGPU->setVariable<float>("local_O2", local_O2);
    FLAMEGPU->setVariable<float>("local_IFNg", local_IFNg);
    FLAMEGPU->setVariable<float>("local_IL2", local_IL2);
    FLAMEGPU->setVariable<float>("local_IL10", local_IL10);
    FLAMEGPU->setVariable<float>("local_TGFB", local_TGFB);
    FLAMEGPU->setVariable<float>("local_CCL2", local_CCL2);
    
    // ========== COMPUTE DERIVED STATES ==========
    
    // 1. Activation level (enhanced by TGF-beta)
    float activation_level = 1.0f;
    if (local_TGFB > 0.0f) {
        const float TGFB_activate_EC50 = FLAMEGPU->environment.getProperty<float>("PARAM_TGFB_MDSC_ACTIVATE_EC50");
        activation_level = 1.0f + 0.3f * hill_equation(local_TGFB, TGFB_activate_EC50, 2.0f);
    }
    FLAMEGPU->setVariable<float>("activation_level", activation_level);
    
    // 2. Suppression radius
    const float base_suppression_radius = FLAMEGPU->environment.getProperty<float>("PARAM_MDSC_SUPPRESSION_RADIUS");
    float suppression_radius = base_suppression_radius * activation_level;
    FLAMEGPU->setVariable<float>("suppression_radius", suppression_radius);
    
    // 3. Compute CCL2 gradient for chemotaxis (simple finite difference)
    const float dx = FLAMEGPU->environment.getProperty<float>("voxel_size") * 1.0e-4f;  // to cm
    
    float CCL2_gradient_x = 0.0f;
    float CCL2_gradient_y = 0.0f;
    float CCL2_gradient_z = 0.0f;
    
    // X gradient
    if (x > 0 && x < grid_x - 1) {
        int idx_left = voxel_idx - 1;
        int idx_right = voxel_idx + 1;
        CCL2_gradient_x = (d_CCL2[idx_right] - d_CCL2[idx_left]) / (2.0f * dx);
    }
    
    // Y gradient
    if (y > 0 && y < grid_y - 1) {
        int idx_front = voxel_idx - grid_x;
        int idx_back = voxel_idx + grid_x;
        CCL2_gradient_y = (d_CCL2[idx_back] - d_CCL2[idx_front]) / (2.0f * dx);
    }
    
    // Z gradient
    if (z > 0 && z < grid_z - 1) {
        int idx_bottom = voxel_idx - grid_x * grid_y;
        int idx_top = voxel_idx + grid_x * grid_y;
        CCL2_gradient_z = (d_CCL2[idx_top] - d_CCL2[idx_bottom]) / (2.0f * dx);
    }
    
    FLAMEGPU->setVariable<float>("CCL2_gradient_x", CCL2_gradient_x);
    FLAMEGPU->setVariable<float>("CCL2_gradient_y", CCL2_gradient_y);
    FLAMEGPU->setVariable<float>("CCL2_gradient_z", CCL2_gradient_z);
    
    return flamegpu::ALIVE;
}

// MDSC agent function: Compute chemical sources
FLAMEGPU_AGENT_FUNCTION(mdsc_compute_chemical_sources, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int dead = FLAMEGPU->getVariable<int>("dead");
    const float activation_level = FLAMEGPU->getVariable<float>("activation_level");
    
    // Dead cells don't produce
    if (dead == 1) {
        FLAMEGPU->setVariable<float>("ROS_release_rate", 0.0f);
        FLAMEGPU->setVariable<float>("NO_release_rate", 0.0f);
        return flamegpu::ALIVE;
    }
    
    // ========== ROS (Reactive Oxygen Species) RELEASE ==========
    // MDSCs produce ROS which suppresses T cell function locally
    const float ROS_base = FLAMEGPU->environment.getProperty<float>("PARAM_ROS_RELEASE_RATE_BASE");
    float ROS_rate = ROS_base * activation_level;
    FLAMEGPU->setVariable<float>("ROS_release_rate", ROS_rate);
    
    // ========== NO (Nitric Oxide) RELEASE ==========
    // MDSCs produce NO which also suppresses T cells
    const float NO_base = FLAMEGPU->environment.getProperty<float>("PARAM_NO_RELEASE_RATE_BASE");
    float NO_rate = NO_base * activation_level;
    FLAMEGPU->setVariable<float>("NO_release_rate", NO_rate);
    
    return flamegpu::ALIVE;
}

} // namespace PDAC

#endif // PDAC_MDSC_CUH