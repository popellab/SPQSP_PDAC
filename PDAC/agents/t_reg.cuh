#ifndef FLAMEGPU_TNBC_T_REG_CUH
#define FLAMEGPU_TNBC_T_REG_CUH

#include "flamegpu/flamegpu.h"
#include "../core/common.cuh"

namespace PDAC {

// Helper function to get Moore neighborhood direction
// Indices 0-5 are Von Neumann (face) neighbors
__device__ __forceinline__ void get_moore_direction_treg(int idx, int& dx, int& dy, int& dz) {
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
constexpr unsigned int VON_NEUMANN_MASK_TREG = 0x3Fu;  // binary: 00111111

// TReg agent function: Broadcast location
FLAMEGPU_AGENT_FUNCTION(treg_broadcast_location, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const float voxel_size = FLAMEGPU->environment.getProperty<float>("voxel_size");

    FLAMEGPU->message_out.setVariable<int>("agent_type", CELL_TYPE_TREG);
    FLAMEGPU->message_out.setVariable<int>("agent_id", FLAMEGPU->getVariable<unsigned int>("id"));
    FLAMEGPU->message_out.setVariable<int>("cell_state", 0);  // TRegs have single state
    FLAMEGPU->message_out.setVariable<float>("PDL1", 0.0f);   // TRegs don't express PDL1
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

// TReg agent function: Scan neighbors and cache available voxels
// Counts T cells, TRegs, cancer cells and builds bitmask of available neighbor voxels
FLAMEGPU_AGENT_FUNCTION(treg_scan_neighbors, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
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

    int tcell_count = 0;
    int treg_count = 0;
    int cancer_count = 0;
    int all_count = 0;

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

            // Find direction index
            int dir_idx = -1;
            for (int i = 0; i < 26; i++) {
                int ddx, ddy, ddz;
                get_moore_direction_treg(i, ddx, ddy, ddz);
                if (ddx == dx && ddy == dy && ddz == dz) {
                    dir_idx = i;
                    break;
                }
            }

            if (dir_idx >= 0) {
                if (agent_type == CELL_TYPE_T) {
                    tcell_count++;
                    neighbor_counts[dir_idx][1]++;
                } else if (agent_type == CELL_TYPE_TREG) {
                    treg_count++;
                    neighbor_counts[dir_idx][2]++;
                } else if (agent_type == CELL_TYPE_CANCER) {
                    cancer_count++;
                    neighbor_counts[dir_idx][0]++;
                }
            }
        }
    }

    // Build available_neighbors mask (voxels with room for T cells/TRegs)
    unsigned int available_neighbors = 0;
    for (int i = 0; i < 26; i++) {
        int dx, dy, dz;
        get_moore_direction_treg(i, dx, dy, dz);
        int nx = my_x + dx;
        int ny = my_y + dy;
        int nz = my_z + dz;

        if (is_in_bounds(nx, ny, nz, size_x, size_y, size_z)) {
            bool has_cancer = (neighbor_counts[i][0] > 0);
            int t_count = neighbor_counts[i][1] + neighbor_counts[i][2];
            int max_cap = has_cancer ? MAX_T_PER_VOXEL_WITH_CANCER : MAX_T_PER_VOXEL;

            if (t_count < max_cap) {
                available_neighbors |= (1u << i);
            }
        }
    }

    FLAMEGPU->setVariable<int>("neighbor_Tcell_count", tcell_count);
    FLAMEGPU->setVariable<int>("neighbor_Treg_count", treg_count);
    FLAMEGPU->setVariable<int>("neighbor_cancer_count", cancer_count);
    FLAMEGPU->setVariable<int>("neighbor_all_count", all_count);
    FLAMEGPU->setVariable<unsigned int>("available_neighbors", available_neighbors);

    return flamegpu::ALIVE;
}

// TReg agent function: State transitions
// Handles life countdown and division triggering
FLAMEGPU_AGENT_FUNCTION(treg_state_step, flamegpu::MessageNone, flamegpu::MessageNone) {
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

    // Division cooldown
    int divide_cd = FLAMEGPU->getVariable<int>("divide_cd");
    if (divide_cd > 0) {
        divide_cd--;
        FLAMEGPU->setVariable<int>("divide_cd", divide_cd);
    }

    // Division logic (simplified - full version requires ArgI from molecular grid)
    // In the CPU version, division probability depends on:
    //   prob = (ArgI / (ArgI + EC50)) * (1 - Treg_count / (TumVol * TregMax)) / interval
    // For now, we use a simplified version with a base probability
    // TODO: Couple with molecular grid for ArgI-dependent division

    const int divide_limit = FLAMEGPU->getVariable<int>("divide_limit");
    const float treg_div_prob = FLAMEGPU->environment.getProperty<float>("PARAM_TREG_DIV_PROB");

    if (divide_cd == 0 && divide_limit > 0) {
        // Simplified: probabilistic division based on local conditions
        // In future, this should be modulated by ArgI concentration
        const int neighbor_treg = FLAMEGPU->getVariable<int>("neighbor_Treg_count");
        const float treg_density_factor = FLAMEGPU->environment.getProperty<float>("PARAM_TREG_DENSITY_FACTOR");

        // Reduce division probability when many Tregs nearby (density dependence)
        float effective_prob = treg_div_prob / (1.0f + neighbor_treg * treg_density_factor);

        if (FLAMEGPU->random.uniform<float>() < effective_prob) {
            FLAMEGPU->setVariable<int>("divide_flag", 1);
        }
    }

    return flamegpu::ALIVE;
}

// Helper: Compare two agents for priority (lower wins)
__device__ __forceinline__ bool has_higher_priority_treg(unsigned int id1, int sx1, int sy1, int sz1,
                                                          unsigned int id2, int sx2, int sy2, int sz2) {
    if (id1 != id2) return id1 < id2;
    if (sx1 != sx2) return sx1 < sx2;
    if (sy1 != sy2) return sy1 < sy2;
    return sz1 < sz2;
}

// TReg agent function: Select movement target and broadcast intent
// Phase 1 of two-phase conflict resolution
// Uses cached available_neighbors mask from scan phase
FLAMEGPU_AGENT_FUNCTION(treg_select_move_target, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
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
        FLAMEGPU->message_out.setVariable<int>("agent_type", CELL_TYPE_TREG);
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

    const float move_prob = FLAMEGPU->environment.getProperty<float>("treg_move_prob");

    int target_x = -1, target_y = -1, target_z = -1;
    int intent_action = INTENT_NONE;

    if (FLAMEGPU->random.uniform<float>() < move_prob) {
        // Use cached available_neighbors mask, but only Von Neumann directions for movement
        const unsigned int available_all = FLAMEGPU->getVariable<unsigned int>("available_neighbors");
        const unsigned int available = available_all & VON_NEUMANN_MASK_TREG;  // Only face neighbors (6 directions)
        int num_available = __popc(available);

        if (num_available > 0) {
            int selected = FLAMEGPU->random.uniform<int>(0, num_available - 1);
            int count = 0;
            for (int i = 0; i < 6; i++) {  // Only iterate through Von Neumann directions
                if (available & (1u << i)) {
                    if (count == selected) {
                        int dx, dy, dz;
                        get_moore_direction_treg(i, dx, dy, dz);
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
    FLAMEGPU->message_out.setVariable<int>("agent_type", CELL_TYPE_TREG);
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

// TReg agent function: Execute movement if won conflict
// Phase 2 of two-phase conflict resolution
FLAMEGPU_AGENT_FUNCTION(treg_execute_move, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
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
                if (msg_src_x == my_x && msg_src_y == my_y && msg_src_z == my_z) {
                    continue;
                }
                // Count agents with higher priority
                if (has_higher_priority_treg(msg_id, msg_src_x, msg_src_y, msg_src_z,
                                              my_id, my_x, my_y, my_z)) {
                    higher_priority_count++;
                }
            }
        }
    }

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

// TReg agent function: Select division target and broadcast intent
// Uses cached available_neighbors mask from scan phase
FLAMEGPU_AGENT_FUNCTION(treg_select_divide_target, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
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
        FLAMEGPU->message_out.setVariable<int>("agent_type", CELL_TYPE_TREG);
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

    const int divide_flag = FLAMEGPU->getVariable<int>("divide_flag");
    const int divide_cd = FLAMEGPU->getVariable<int>("divide_cd");
    const int divide_limit = FLAMEGPU->getVariable<int>("divide_limit");

    if (divide_flag == 0 || divide_cd > 0 || divide_limit <= 0) {
        FLAMEGPU->message_out.setVariable<int>("agent_type", CELL_TYPE_TREG);
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

    int target_x = -1, target_y = -1, target_z = -1;
    int intent_action = INTENT_NONE;

    // Use cached available_neighbors mask
    const unsigned int available = FLAMEGPU->getVariable<unsigned int>("available_neighbors");
    int num_available = __popc(available);

    if (num_available > 0) {
        int selected = FLAMEGPU->random.uniform<int>(0, num_available - 1);
        int count = 0;
        for (int i = 0; i < 26; i++) {
            if (available & (1u << i)) {
                if (count == selected) {
                    int dx, dy, dz;
                    get_moore_direction_treg(i, dx, dy, dz);
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
    FLAMEGPU->message_out.setVariable<int>("agent_type", CELL_TYPE_TREG);
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

// TReg agent function: Execute division if won conflict
FLAMEGPU_AGENT_FUNCTION(treg_execute_divide, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
    const int intent_action = FLAMEGPU->getVariable<int>("intent_action");

    if (intent_action != INTENT_DIVIDE) {
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
                if (msg_src_x == my_x && msg_src_y == my_y && msg_src_z == my_z) {
                    continue;
                }
                // Count agents with higher priority
                if (has_higher_priority_treg(msg_id, msg_src_x, msg_src_y, msg_src_z,
                                              my_id, my_x, my_y, my_z)) {
                    higher_priority_count++;
                }
            }
        }
    }

    bool can_divide = (higher_priority_count < MAX_T_PER_VOXEL);

    if (!can_divide) {
        FLAMEGPU->setVariable<int>("intent_action", INTENT_NONE);
        FLAMEGPU->setVariable<int>("target_x", -1);
        FLAMEGPU->setVariable<int>("target_y", -1);
        FLAMEGPU->setVariable<int>("target_z", -1);
        return flamegpu::ALIVE;
    }

    // Proceed with division
    const int divide_limit = FLAMEGPU->getVariable<int>("divide_limit");
    const int div_interval = FLAMEGPU->environment.getProperty<int>("PARAM_TREG_DIV_INTERVAL");
    const int div_limit_init = FLAMEGPU->environment.getProperty<int>("PARAM_TREG_DIV_LIMIT");
    const float treg_life_mean = FLAMEGPU->environment.getProperty<float>("PARAM_TREG_LIFE_MEAN");

    // Calculate daughter life (exponential distribution)
    const float rnd = FLAMEGPU->random.uniform<float>();
    const int daughter_life = static_cast<int>(treg_life_mean * logf(1.0f / (rnd + 0.0001f)) + 0.5f);

    // Create daughter cell
    FLAMEGPU->agent_out.setVariable<unsigned int>("id", 0u);
    FLAMEGPU->agent_out.setVariable<int>("x", target_x);
    FLAMEGPU->agent_out.setVariable<int>("y", target_y);
    FLAMEGPU->agent_out.setVariable<int>("z", target_z);
    FLAMEGPU->agent_out.setVariable<int>("divide_flag", 0);
    FLAMEGPU->agent_out.setVariable<int>("divide_cd", div_interval);
    FLAMEGPU->agent_out.setVariable<int>("divide_limit", divide_limit - 1);
    FLAMEGPU->agent_out.setVariable<int>("neighbor_Tcell_count", 0);
    FLAMEGPU->agent_out.setVariable<int>("neighbor_Treg_count", 0);
    FLAMEGPU->agent_out.setVariable<int>("neighbor_cancer_count", 0);
    FLAMEGPU->agent_out.setVariable<int>("neighbor_all_count", 0);
    FLAMEGPU->agent_out.setVariable<int>("life", daughter_life > 0 ? daughter_life : 1);
    FLAMEGPU->agent_out.setVariable<int>("dead", 0);
    FLAMEGPU->agent_out.setVariable<int>("intent_action", INTENT_NONE);
    FLAMEGPU->agent_out.setVariable<int>("target_x", -1);
    FLAMEGPU->agent_out.setVariable<int>("target_y", -1);
    FLAMEGPU->agent_out.setVariable<int>("target_z", -1);
    FLAMEGPU->agent_out.setVariable<unsigned int>("available_neighbors", 0u);

    // Update parent
    FLAMEGPU->setVariable<int>("divide_flag", 0);
    FLAMEGPU->setVariable<int>("divide_limit", divide_limit - 1);
    FLAMEGPU->setVariable<int>("divide_cd", div_interval);

    // Clear intent
    FLAMEGPU->setVariable<int>("intent_action", INTENT_NONE);
    FLAMEGPU->setVariable<int>("target_x", -1);
    FLAMEGPU->setVariable<int>("target_y", -1);
    FLAMEGPU->setVariable<int>("target_z", -1);

    return flamegpu::ALIVE;
}


// TReg agent function: Update chemicals from PDE
FLAMEGPU_AGENT_FUNCTION(treg_update_chemicals, flamegpu::MessageNone, flamegpu::MessageNone) {
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
    
    float local_O2 = d_O2[voxel_idx];
    float local_IFNg = d_IFN[voxel_idx];
    float local_IL2 = d_IL2[voxel_idx];
    float local_IL10 = d_IL10[voxel_idx];
    float local_TGFB = d_TGFB[voxel_idx];
    
    // Set local concentration variables
    FLAMEGPU->setVariable<float>("local_O2", local_O2);
    FLAMEGPU->setVariable<float>("local_IFNg", local_IFNg);
    FLAMEGPU->setVariable<float>("local_IL2", local_IL2);
    FLAMEGPU->setVariable<float>("local_IL10", local_IL10);
    FLAMEGPU->setVariable<float>("local_TGFB", local_TGFB);
    
    // ========== COMPUTE DERIVED STATES ==========
    
    // 1. Update cumulative IL2 exposure
    float IL2_exposure = FLAMEGPU->getVariable<float>("IL2_exposure");
    const float dt = FLAMEGPU->environment.getProperty<float>("PARAM_SEC_PER_SLICE");
    IL2_exposure += local_IL2 * dt;
    FLAMEGPU->setVariable<float>("IL2_exposure", IL2_exposure);
    
    // 2. Check proliferation threshold
    const float IL2_prolif_threshold = FLAMEGPU->environment.getProperty<float>("PARAM_TCELL_IL2_PROLIF_THRESHOLD");
    int can_proliferate = (IL2_exposure > IL2_prolif_threshold) ? 1 : 0;
    FLAMEGPU->setVariable<int>("can_proliferate", can_proliferate);
    
    // 3. Suppression strength (enhanced by TGF-beta)
    float suppression_strength = 1.0f;
    if (local_TGFB > 0.0f) {
        const float TGFB_suppress_EC50 = FLAMEGPU->environment.getProperty<float>("PARAM_TGFB_T_CELL_SUPPRESS_EC50");
        suppression_strength = 1.0f + 0.5f * hill_equation(local_TGFB, TGFB_suppress_EC50, 2.0f);
    }
    FLAMEGPU->setVariable<float>("suppression_strength", suppression_strength);
    
    return flamegpu::ALIVE;
}

// TReg agent function: Compute chemical sources
FLAMEGPU_AGENT_FUNCTION(treg_compute_chemical_sources, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int dead = FLAMEGPU->getVariable<int>("dead");
    const float suppression_strength = FLAMEGPU->getVariable<float>("suppression_strength");
    
    // Dead cells don't produce
    if (dead == 1) {
        FLAMEGPU->setVariable<float>("IL10_release_rate", 0.0f);
        FLAMEGPU->setVariable<float>("TGFB_release_rate", 0.0f);
        FLAMEGPU->setVariable<float>("IL2_consumption_rate", 0.0f);
        return flamegpu::ALIVE;
    }
    
    // Get base rates from environment
    const float IL10_base = FLAMEGPU->environment.getProperty<float>("PARAM_IL10_RELEASE_RATE_BASE");
    const float TGFB_base = FLAMEGPU->environment.getProperty<float>("PARAM_TGFB_RELEASE_RATE_BASE");
    const float IL2_consumption_base = FLAMEGPU->environment.getProperty<float>("PARAM_IL2_CONSUMPTION_TREG");
    
    // ========== IL-10 RELEASE ==========
    float IL10_rate = IL10_base * suppression_strength;
    FLAMEGPU->setVariable<float>("IL10_release_rate", IL10_rate);
    
    // ========== TGF-BETA RELEASE ==========
    float TGFB_rate = TGFB_base * suppression_strength;
    FLAMEGPU->setVariable<float>("TGFB_release_rate", TGFB_rate);
    
    // ========== IL-2 CONSUMPTION (NEGATIVE!) ==========
    float IL2_consumption = IL2_consumption_base;
    FLAMEGPU->setVariable<float>("IL2_consumption_rate", -IL2_consumption);  // NEGATIVE!
    
    return flamegpu::ALIVE;
}

} // namespace PDAC

#endif // PDAC_T_REG_CUH