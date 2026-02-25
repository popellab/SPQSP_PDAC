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

// Helper: Hill equation for PD1-PDL1 suppression
__device__ __forceinline__ float hill_equation_mdsc(float x, float k50, float n) {
    if (x <= 0.0f) return 0.0f;
    const float xn = powf(x, n);
    const float kn = powf(k50, n);
    return xn / (kn + xn);
}

// MDSC agent function: Broadcast location
FLAMEGPU_AGENT_FUNCTION(mdsc_broadcast_location, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const float voxel_size = FLAMEGPU->environment.getProperty<float>("voxel_size");

    FLAMEGPU->message_out.setVariable<int>("agent_type", CELL_TYPE_MDSC);
    FLAMEGPU->message_out.setVariable<int>("agent_id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setVariable<int>("cell_state", 0);  // MDSCs have single state
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
                } else if (agent_type == CELL_TYPE_T) {
                    tcell_count++;
                    neighbor_tcells[dir_idx]++;
                } else if (agent_type == CELL_TYPE_TREG) {
                    treg_count++;
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
            if (!neighbor_blocked[i]) {
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

    // Read ECM concentration from grid at current voxel
    auto ecm = FLAMEGPU->environment.getMacroProperty<float,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX>("ecm_grid");
    float ECM_density = ecm[my_x][my_y][my_z];  // ECM concentration (0.0-1.0)

    // If ECM density is high, cell has lower probability to move
    float ECM_sat = ECM_density;  // Direct use of ECM density as movement inhibition

    int target_x = -1, target_y = -1, target_z = -1;
    int intent_action = INTENT_NONE;

    if (FLAMEGPU->random.uniform<float>() < ECM_sat) {
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
    const unsigned int my_id = FLAMEGPU->getID();
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
            if ((msg_agent_type == CELL_TYPE_MDSC) && msg_intent == INTENT_MOVE) {
                // Skip self
                if (msg_id == my_id) {
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

// Occupancy Grid
FLAMEGPU_AGENT_FUNCTION(mdsc_write_to_occ_grid, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    auto occ = FLAMEGPU->environment.getMacroProperty<unsigned int,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX, NUM_OCC_TYPES>("occ_grid");
    occ[x][y][z][CELL_TYPE_MDSC].exchange(1u);  // Exclusive (MAX_MDSC_PER_VOXEL = 1)
    return flamegpu::ALIVE;
}

// MDSC agent function: Update chemicals from PDE
FLAMEGPU_AGENT_FUNCTION(mdsc_update_chemicals, flamegpu::MessageNone, flamegpu::MessageNone) {
    // ========== READ CHEMICAL CONCENTRATIONS DIRECTLY FROM PDE ==========
    const int nx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int ax = FLAMEGPU->getVariable<int>("x");
    const int ay = FLAMEGPU->getVariable<int>("y");
    const int az = FLAMEGPU->getVariable<int>("z");
    const int voxel = az * ny*nx + ay * nx + ax;
    float local_IFNg = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>("pde_concentration_ptr_1"))[voxel];
    
    // ========== COMPUTE DERIVED STATES ==========

    const float IFNg_PDL1_EC50 = FLAMEGPU->environment.getProperty<float>("PARAM_IFNG_PDL1_HALF");
    const float IFNg_PDL1_hill = FLAMEGPU->environment.getProperty<float>("PARAM_IFNG_PDL1_N");
    float H_IFNg = hill_equation_mdsc(local_IFNg, IFNg_PDL1_EC50, IFNg_PDL1_hill);
    const float PDL1_syn_max = FLAMEGPU->environment.getProperty<float>("PARAM_PDL1_SYN_MAX");
    float minPDL1 = PDL1_syn_max * H_IFNg;

    float PDL1_current = FLAMEGPU->getVariable<float>("PDL1_syn");
    if (PDL1_current < minPDL1) {
        FLAMEGPU->setVariable<float>("PDL1_syn", minPDL1);
    }
    
    return flamegpu::ALIVE;
}

// MDSC agent function: Compute chemical sources
// atomicAdds directly to PDE source/uptake arrays
FLAMEGPU_AGENT_FUNCTION(mdsc_compute_chemical_sources, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int dead = FLAMEGPU->getVariable<int>("dead");

    // Dead cells don't produce
    if (dead == 1) {
        return flamegpu::ALIVE;
    }

    // Compute voxel index and volume
    const int nx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int ax = FLAMEGPU->getVariable<int>("x");
    const int ay = FLAMEGPU->getVariable<int>("y");
    const int az = FLAMEGPU->getVariable<int>("z");
    const int voxel = az * ny*nx + ay * nx + ax;

    const float vs_cm = FLAMEGPU->environment.getProperty<float>("voxel_size") * 1.0e-4f;
    const float voxel_volume = vs_cm * vs_cm * vs_cm;

    // ArgI secretion → src ptr 6 (ARGI)
    const float ArgI_release = FLAMEGPU->environment.getProperty<float>("PARAM_ARGI_RELEASE");
    atomicAdd(&reinterpret_cast<float*>(
        FLAMEGPU->environment.getProperty<uint64_t>("pde_source_ptr_6"))[voxel],
        ArgI_release / voxel_volume);

    // NO secretion → src ptr 7 (NO)
    const float NO_release = FLAMEGPU->environment.getProperty<float>("PARAM_NO_RELEASE");
    atomicAdd(&reinterpret_cast<float*>(
        FLAMEGPU->environment.getProperty<uint64_t>("pde_source_ptr_7"))[voxel],
        NO_release / voxel_volume);

    // CCL2 uptake → upt ptr 5 (CCL2), positive [1/s], no volume scaling
    const float CCL2_uptake = FLAMEGPU->environment.getProperty<float>("PARAM_CCL2_UPTAKE");
    atomicAdd(&reinterpret_cast<float*>(
        FLAMEGPU->environment.getProperty<uint64_t>("pde_uptake_ptr_5"))[voxel],
        CCL2_uptake);

    return flamegpu::ALIVE;
}

// Single-phase MDSC movement using run-tumble chemotaxis.
// Replaces two-phase select_move_target + execute_move.
// MDSCs are exclusive per voxel (CAS) but can share voxels with cancer cells.
// Chemotaxis biased toward CCL2 gradient.
FLAMEGPU_AGENT_FUNCTION(mdsc_move, flamegpu::MessageNone, flamegpu::MessageNone) {

    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int grid_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");
    const int tumble = FLAMEGPU->getVariable<int>("tumble");

    // ECM based movement probability - TEMPORARILY DISABLED
    // auto ecm = FLAMEGPU->environment.getMacroProperty<float,
    //     OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX>("ecm_grid");
    // float ECM_density = ecm[x][y][z];
    // double ECM_sat = ECM_density / (ECM_density + FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ECM_MOT_EC50"));
    // if (FLAMEGPU->random.uniform<float>() < ECM_sat) return flamegpu::ALIVE;
    if (FLAMEGPU->random.uniform<float>() < 0.2f) return flamegpu::ALIVE;

    const float move_dir_x = FLAMEGPU->getVariable<float>("move_direction_x");
    const float move_dir_y = FLAMEGPU->getVariable<float>("move_direction_y");
    const float move_dir_z = FLAMEGPU->getVariable<float>("move_direction_z");

    // Use CCL2 gradient for chemotaxis — read directly from PDE
    const int nx_mv = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny_mv = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int voxel_mv = z * ny_mv*nx_mv + y * nx_mv + x;
    const float grad_x = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>("pde_grad_CCL2_x"))[voxel_mv];
    const float grad_y = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>("pde_grad_CCL2_y"))[voxel_mv];
    const float grad_z = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>("pde_grad_CCL2_z"))[voxel_mv];

    auto occ = FLAMEGPU->environment.getMacroProperty<unsigned int,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX, NUM_OCC_TYPES>("occ_grid");

    const float dt = FLAMEGPU->environment.getProperty<float>("PARAM_SEC_PER_SLICE");

    int target_x = x;
    int target_y = y;
    int target_z = z;

    // === RUN PHASE (tumble == 0) ===
    if (tumble == 0) {
        float v_x = move_dir_x / dt;
        float v_y = move_dir_y / dt;
        float v_z = move_dir_z / dt;

        float norm_gradient = std::sqrt(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z);

        float dot_product = v_x * grad_x + v_y * grad_y + v_z * grad_z;
        float norm_v = std::sqrt(v_x * v_x + v_y * v_y + v_z * v_z);
        float cos_theta = dot_product / (norm_v * norm_gradient);

        const float EC50_grad = 1.0f;
        float H_grad = norm_gradient / (norm_gradient + EC50_grad);
        if (cos_theta < 0) H_grad = -H_grad;

        const float lambda = 0.0000168f;
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

        float r = FLAMEGPU->random.uniform<float>();
        int selected_idx = 0;
        for (int i = 0; i < n_dirs; i++) {
            if (r < probs[i]) { selected_idx = i; break; }
        }

        FLAMEGPU->setVariable<float>("move_direction_x", static_cast<float>(dirs[selected_idx][0]));
        FLAMEGPU->setVariable<float>("move_direction_y", static_cast<float>(dirs[selected_idx][1]));
        FLAMEGPU->setVariable<float>("move_direction_z", static_cast<float>(dirs[selected_idx][2]));
        FLAMEGPU->setVariable<int>("tumble", 0);

        target_x = x + dirs[selected_idx][0];
        target_y = y + dirs[selected_idx][1];
        target_z = z + dirs[selected_idx][2];

        if (target_x < 0 || target_x >= grid_x ||
            target_y < 0 || target_y >= grid_y ||
            target_z < 0 || target_z >= grid_z) {
            return flamegpu::ALIVE;
        }
    }

    // Try to claim target voxel (CAS — MDSCs are exclusive per voxel)
    if (target_x != x || target_y != y || target_z != z) {
        if (occ[target_x][target_y][target_z][CELL_TYPE_MDSC].CAS(0u, 1u) == 0u) {
            occ[x][y][z][CELL_TYPE_MDSC].exchange(0u);
            FLAMEGPU->setVariable<int>("x", target_x);
            FLAMEGPU->setVariable<int>("y", target_y);
            FLAMEGPU->setVariable<int>("z", target_z);
        }
    }

    return flamegpu::ALIVE;
}

} // namespace PDAC

#endif // PDAC_MDSC_CUH