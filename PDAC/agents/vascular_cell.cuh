#ifndef FLAMEGPU_PDAC_VASCULAR_CELL_CUH
#define FLAMEGPU_PDAC_VASCULAR_CELL_CUH

#include "flamegpu/flamegpu.h"
#include "../core/common.cuh"

namespace PDAC {

// Vascular cell vascular_states
enum VascularCellState : int {
    VAS_TIP = 0,      // Actively sprouting tip cell
    VAS_STALK = 1,    // Connecting stalk cell
    VAS_PHALANX = 2   // Mature vessel (O2 secreting)
};

// Broadcast location
FLAMEGPU_AGENT_FUNCTION(vascular_broadcast_location, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const float voxel_size = FLAMEGPU->environment.getProperty<float>("voxel_size");

    FLAMEGPU->message_out.setVariable<int>("agent_type", CELL_TYPE_VASCULAR);
    FLAMEGPU->message_out.setVariable<int>("agent_id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setVariable<int>("cell_state", FLAMEGPU->getVariable<int>("vascular_state"));
    FLAMEGPU->message_out.setVariable<int>("voxel_x", x);
    FLAMEGPU->message_out.setVariable<int>("voxel_y", y);
    FLAMEGPU->message_out.setVariable<int>("voxel_z", z);
    FLAMEGPU->message_out.setVariable<unsigned int>("tip_id", FLAMEGPU->getVariable<unsigned int>("tip_id"));

    FLAMEGPU->message_out.setLocation(
        static_cast<float>(x) * voxel_size,
        static_cast<float>(y) * voxel_size,
        static_cast<float>(z) * voxel_size
    );

    return flamegpu::ALIVE;
}

// Read VEGF-A concentration and gradient from PDE (set by host function)
FLAMEGPU_AGENT_FUNCTION(vascular_update_chemicals, flamegpu::MessageNone, flamegpu::MessageNone) {
    // VEGF-A, O2, and gradients already written to agent variables by host function
    return flamegpu::ALIVE;
}

// Compute O2 source and VEGF-A sink rates
FLAMEGPU_AGENT_FUNCTION(vascular_compute_chemical_sources, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int vascular_state = FLAMEGPU->getVariable<int>("vascular_state");
    const float local_VEGFA = FLAMEGPU->getVariable<float>("local_VEGFA");
    const float local_O2 = FLAMEGPU->getVariable<float>("local_O2");

    // === O2 SECRETION (PHALANX ONLY) ===
    float O2_source = 0.0f;
    if (vascular_state == VAS_PHALANX) {
        float pi = 3.1415926f;
        const float sigma = FLAMEGPU->environment.getProperty<float>("PARAM_VAS_SIGMA");
        const float RC = FLAMEGPU->environment.getProperty<float>("PARAM_VAS_RC");
        const float voxel_volume_cm = std::pow(FLAMEGPU->environment.getProperty<float>("PARAM_VOXEL_SIZE_CM"), 3);
        float Lv = voxel_volume_cm / (std::pow(RC, 2) * pi);
        double Rt = 1 / std::pow(Lv * pi, 0.5);
        double w = RC / Rt;
        double lambda = 1 - w * w;
        double Kv = 2 * pi * FLAMEGPU->environment.getProperty<float>("PARAM_O2_DIFFUSIVITY")
                        * (lambda / (sigma * lambda - (2 + lambda) / 4 + (1 / lambda) * std::log(1 / w)));
        O2_source = Kv * Lv * (FLAMEGPU->environment.getProperty<float>("PARAM_VAS_O2_CONC") - local_O2);
        if (O2_source < 0.0f) O2_source = 0.0f;
    }

    // === VEGF-A UPTAKE (ALL STATES) ===
    const float VEGFA_uptake = FLAMEGPU->environment.getProperty<float>("PARAM_VEGFA_UPTAKE");
    const float VEGFA_sink = -VEGFA_uptake * local_VEGFA;

    // Store rates for collection by host function
    FLAMEGPU->setVariable<float>("O2_source", O2_source);
    FLAMEGPU->setVariable<float>("VEGFA_sink", VEGFA_sink);

    return flamegpu::ALIVE;
}

// Select movement target for tip cells (run-tumble algorithm)
FLAMEGPU_AGENT_FUNCTION(vascular_select_move_target, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
    const int vascular_state = FLAMEGPU->getVariable<int>("vascular_state");

    // Only tip cells move
    if (vascular_state != VAS_TIP) {
        FLAMEGPU->setVariable<int>("intent_action", INTENT_NONE);
        return flamegpu::ALIVE;
    }

    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const int tumble = FLAMEGPU->getVariable<int>("tumble");

    const float move_dir_x = FLAMEGPU->getVariable<float>("move_direction_x");
    const float move_dir_y = FLAMEGPU->getVariable<float>("move_direction_y");
    const float move_dir_z = FLAMEGPU->getVariable<float>("move_direction_z");

    const float vegfa_grad_x = FLAMEGPU->getVariable<float>("vegfa_grad_x");
    const float vegfa_grad_y = FLAMEGPU->getVariable<float>("vegfa_grad_y");
    const float vegfa_grad_z = FLAMEGPU->getVariable<float>("vegfa_grad_z");

    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int grid_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");
    const float voxel_size = FLAMEGPU->environment.getProperty<float>("voxel_size");
    const float dt = FLAMEGPU->environment.getProperty<float>("PARAM_SEC_PER_SLICE");

    int target_x = x;
    int target_y = y;
    int target_z = z;
    int new_tumble = tumble;

    // === RUN PHASE (tumble == 0) ===
    if (tumble == 0) {
        // Calculate velocity
        float v_x = move_dir_x / dt;
        float v_y = move_dir_y / dt;
        float v_z = move_dir_z / dt;

        // Gradient magnitude
        float norm_gradient = std::sqrt(vegfa_grad_x * vegfa_grad_x +
                                       vegfa_grad_y * vegfa_grad_y +
                                       vegfa_grad_z * vegfa_grad_z);

        // If no gradient, tumble
        if (norm_gradient < 1e-10f) {
            new_tumble = 1;
            FLAMEGPU->setVariable<int>("tumble", new_tumble);
            FLAMEGPU->setVariable<int>("intent_action", INTENT_NONE);
            return flamegpu::ALIVE;
        }

        // Dot product of velocity and gradient
        float dot_product = v_x * vegfa_grad_x + v_y * vegfa_grad_y + v_z * vegfa_grad_z;
        float norm_v = std::sqrt(v_x * v_x + v_y * v_y + v_z * v_z);
        float cos_theta = dot_product / (norm_v * norm_gradient);

        // Hill function of gradient magnitude
        const float EC50_grad = 1.0f;
        float H_grad = norm_gradient / (norm_gradient + EC50_grad);
        if (cos_theta < 0) {
            H_grad = -H_grad;
        }

        // Tumble probability (HCC formula)
        const float lambda = FLAMEGPU->environment.getProperty<float>("PARAM_VAS_TUMBLE");
        const float delta = FLAMEGPU->environment.getProperty<float>("PARAM_VAS_DELTA");
        float tumble_rate = (lambda / 2.0f) * (1.0f - cos_theta) * (1.0f - H_grad) * dt + delta;
        float p_tumble = 1.0f - std::exp(-tumble_rate);

        // Check if should tumble
        if (FLAMEGPU->random.uniform<float>() < p_tumble) {
            new_tumble = 1;
            FLAMEGPU->setVariable<int>("tumble", new_tumble);
            FLAMEGPU->setVariable<int>("intent_action", INTENT_NONE);
            return flamegpu::ALIVE;
        }

        // Continue running - move in current direction
        target_x = x + static_cast<int>(std::round(move_dir_x));
        target_y = y + static_cast<int>(std::round(move_dir_y));
        target_z = z + static_cast<int>(std::round(move_dir_z));

        // Boundary check
        if (target_x < 0 || target_x >= grid_x ||
            target_y < 0 || target_y >= grid_y ||
            target_z < 0 || target_z >= grid_z) {
            // Hit boundary, tumble
            new_tumble = 1;
            FLAMEGPU->setVariable<int>("tumble", new_tumble);
            FLAMEGPU->setVariable<int>("intent_action", INTENT_NONE);
            return flamegpu::ALIVE;
        }
    }
    // === TUMBLE PHASE (tumble == 1) ===
    else {
        // Sample new direction weighted by angle with previous direction
        // Build weighted distribution over 26 neighbors
        const float sigma = 0.524f;
        float prob_sum = 0.0f;
        float probs[26];
        int dirs[26][3];
        int n_dirs = 0;

        float norm_movedir = std::sqrt(move_dir_x * move_dir_x +
                                       move_dir_y * move_dir_y +
                                       move_dir_z * move_dir_z);
        if (norm_movedir < 1e-6f) {
            // No previous direction, set default
            norm_movedir = 1.0f;
        }

        // Enumerate 26 neighbors
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                for (int k = -1; k <= 1; k++) {
                    if (i == 0 && j == 0 && k == 0) continue;

                    // Check forward-facing (cos_theta > 0 with previous direction)
                    float dot_product = i * move_dir_x + j * move_dir_y + k * move_dir_z;
                    float norm_dir = std::sqrt(static_cast<float>(i*i + j*j + k*k));
                    float cos_theta = dot_product / (norm_dir * norm_movedir);

                    if (cos_theta > 0) {
                        // Weight by exponential of cos_theta
                        float rho = std::exp(cos_theta / (sigma * sigma)) / std::exp(1.0f / (sigma * sigma));
                        prob_sum += rho;
                        probs[n_dirs] = prob_sum;  // CDF
                        dirs[n_dirs][0] = i;
                        dirs[n_dirs][1] = j;
                        dirs[n_dirs][2] = k;
                        n_dirs++;
                    }
                }
            }
        }

        if (n_dirs == 0) {
            // No forward directions available, stay put
            new_tumble = 0;
            FLAMEGPU->setVariable<int>("tumble", new_tumble);
            FLAMEGPU->setVariable<int>("intent_action", INTENT_NONE);
            return flamegpu::ALIVE;
        }

        // Normalize probabilities
        for (int i = 0; i < n_dirs; i++) {
            probs[i] /= prob_sum;
        }

        // Sample direction
        float r = FLAMEGPU->random.uniform<float>();
        int selected_idx = 0;
        for (int i = 0; i < n_dirs; i++) {
            if (r < probs[i]) {
                selected_idx = i;
                break;
            }
        }

        // Set new direction
        int dx = dirs[selected_idx][0];
        int dy = dirs[selected_idx][1];
        int dz = dirs[selected_idx][2];

        target_x = x + dx;
        target_y = y + dy;
        target_z = z + dz;

        // Boundary check
        if (target_x < 0 || target_x >= grid_x ||
            target_y < 0 || target_y >= grid_y ||
            target_z < 0 || target_z >= grid_z) {
            // Can't move there, stay put
            new_tumble = 0;
            FLAMEGPU->setVariable<int>("tumble", new_tumble);
            FLAMEGPU->setVariable<int>("intent_action", INTENT_NONE);
            return flamegpu::ALIVE;
        }

        // Update move direction
        FLAMEGPU->setVariable<float>("move_direction_x", static_cast<float>(dx));
        FLAMEGPU->setVariable<float>("move_direction_y", static_cast<float>(dy));
        FLAMEGPU->setVariable<float>("move_direction_z", static_cast<float>(dz));

        // Switch to run phase
        new_tumble = 0;
        FLAMEGPU->setVariable<int>("tumble", new_tumble);
    }

    // Output move intent
    FLAMEGPU->message_out.setVariable<int>("agent_type", CELL_TYPE_VASCULAR);
    FLAMEGPU->message_out.setVariable<int>("agent_id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setVariable<int>("intent_action", INTENT_MOVE);
    FLAMEGPU->message_out.setVariable<int>("target_x", target_x);
    FLAMEGPU->message_out.setVariable<int>("target_y", target_y);
    FLAMEGPU->message_out.setVariable<int>("target_z", target_z);

    FLAMEGPU->message_out.setLocation(
        static_cast<float>(target_x) * voxel_size,
        static_cast<float>(target_y) * voxel_size,
        static_cast<float>(target_z) * voxel_size
    );

    FLAMEGPU->setVariable<int>("intent_action", INTENT_MOVE);
    FLAMEGPU->setVariable<int>("target_x", target_x);
    FLAMEGPU->setVariable<int>("target_y", target_y);
    FLAMEGPU->setVariable<int>("target_z", target_z);

    return flamegpu::ALIVE;
}

// Execute movement (two-phase conflict resolution)
FLAMEGPU_AGENT_FUNCTION(vascular_execute_move, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
    const int intent_action = FLAMEGPU->getVariable<int>("intent_action");

    if (intent_action != INTENT_MOVE) {
        return flamegpu::ALIVE;
    }

    const int target_x = FLAMEGPU->getVariable<int>("target_x");
    const int target_y = FLAMEGPU->getVariable<int>("target_y");
    const int target_z = FLAMEGPU->getVariable<int>("target_z");
    const flamegpu::id_t my_id = FLAMEGPU->getID();
    const float voxel_size = FLAMEGPU->environment.getProperty<float>("voxel_size");

    // Check for conflicts at target voxel
    bool conflict = false;

    for (const auto& msg : FLAMEGPU->message_in(
        static_cast<float>(target_x) * voxel_size,
        static_cast<float>(target_y) * voxel_size,
        static_cast<float>(target_z) * voxel_size)) {

        if (msg.getVariable<int>("intent_action") == INTENT_MOVE &&
            msg.getVariable<int>("target_x") == target_x &&
            msg.getVariable<int>("target_y") == target_y &&
            msg.getVariable<int>("target_z") == target_z &&
            msg.getVariable<int>("agent_id") < my_id) {
            // Another agent with lower ID also wants this voxel
            conflict = true;
            break;
        }
    }

    // If no conflict, move
    if (!conflict) {
        FLAMEGPU->setVariable<int>("x", target_x);
        FLAMEGPU->setVariable<int>("y", target_y);
        FLAMEGPU->setVariable<int>("z", target_z);
    }

    // Reset intent
    FLAMEGPU->setVariable<int>("intent_action", INTENT_NONE);

    return flamegpu::ALIVE;
}

// Mark T cell recruitment sources (phalanx cells based on IFN-γ)
FLAMEGPU_AGENT_FUNCTION(vascular_mark_t_sources, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int vascular_state = FLAMEGPU->getVariable<int>("vascular_state");

    // Only phalanx cells can be T cell sources
    if (vascular_state != VAS_PHALANX) {
        return flamegpu::ALIVE;
    }

    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int grid_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");

    // Get IFN-γ concentration at this voxel
    const float local_IFNg = FLAMEGPU->getVariable<float>("local_VEGFA");  // Will be set correctly when we add IFN read

    // Calculate Hill function (simplified from HCC - no tumor scaling for now)
    const float ec50_ifng = FLAMEGPU->environment.getProperty<int>("PARAM_TEFF_IFN_EC50");
    const float H_IFNg = local_IFNg / (local_IFNg + ec50_ifng);

    // Simple probability check
    const float p_entry = H_IFNg * 0.1f;  // Simplified (HCC uses tumor_scaler * vas_scaler)

    if (FLAMEGPU->random.uniform<float>() < p_entry) {
        // Get recruitment sources array pointer from environment
        unsigned long long ptr_val = FLAMEGPU->environment.getProperty<unsigned long long>("pde_recruitment_sources_ptr");
        int* d_recruitment_sources = reinterpret_cast<int*>(static_cast<uintptr_t>(ptr_val));

        // Mark this voxel as T cell source (bit 0)
        int idx = z * (grid_x * grid_y) + y * grid_x + x;
        atomicOr(&d_recruitment_sources[idx], 1);  // Set T cell bit
    }

    return flamegpu::ALIVE;
}

// State transitions and division decisions (following active HCC Vas.cpp - anastomosis commented out)
FLAMEGPU_AGENT_FUNCTION(vascular_state_step, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
    const int vascular_state = FLAMEGPU->getVariable<int>("vascular_state");
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const unsigned int my_tip_id = FLAMEGPU->getVariable<unsigned int>("tip_id");
    const float voxel_size = FLAMEGPU->environment.getProperty<float>("voxel_size");

    int divide_action = INTENT_NONE;  // 0=none, 1=tip_divide, 2=phalanx_sprout

    // VAS_TIP: Simple division if voxel is open (HCC lines 248-257)
    // Note: Anastomosis is commented out in HCC (lines 230-246)
    if (vascular_state == VAS_TIP) {
        // Check if current voxel allows division (not too crowded)
        int vascular_neighbor_count = 0;
        for (const auto& msg : FLAMEGPU->message_in(
            static_cast<float>(x) * voxel_size,
            static_cast<float>(y) * voxel_size,
            static_cast<float>(z) * voxel_size)) {
            if (msg.getVariable<int>("agent_type") == CELL_TYPE_VASCULAR) {
                vascular_neighbor_count++;
            }
        }

        // Allow division if not too many vascular neighbors (< 3 = voxel is "open")
        if (vascular_neighbor_count < 3) {
            divide_action = 1;  // INTENT_DIVIDE_TIP
        }
    }

    // VAS_STALK: Do nothing (HCC lines 261-264 are empty)
    else if (vascular_state == VAS_STALK) {
        // Empty - stalks do not divide or transition
    }

    // VAS_PHALANX: VEGF-dependent sprouting (HCC lines 266-319)
    else if (vascular_state == VAS_PHALANX) {
        const float local_VEGFA = FLAMEGPU->getVariable<float>("local_VEGFA");
        const float vas_50 = FLAMEGPU->environment.getProperty<float>("PARAM_VAS_50");

        // Probability of sprouting (VEGF-dependent) - HCC line 270
        const float p_tip = local_VEGFA / (vas_50 + local_VEGFA);

        const float rand_val = FLAMEGPU->random.uniform<float>();
        if (rand_val < p_tip) {
            // Check neighborhood for nearby vessels from different tip_id (HCC lines 276-308)
            const int min_neighbor_range = static_cast<int>(
                FLAMEGPU->environment.getProperty<float>("PARAM_VAS_MIN_NEIGHBOR"));

            bool nearby_vessel_exists = false;

            // Check expanded neighborhood (min_neighbor_range in each direction)
            for (int dx = -min_neighbor_range; dx <= min_neighbor_range; dx++) {
                for (int dy = -min_neighbor_range; dy <= min_neighbor_range; dy++) {
                    for (int dz = -min_neighbor_range; dz <= min_neighbor_range; dz++) {
                        if (dx == 0 && dy == 0 && dz == 0) continue;

                        const int check_x = x + dx;
                        const int check_y = y + dy;
                        const int check_z = z + dz;

                        for (const auto& msg : FLAMEGPU->message_in(
                            static_cast<float>(check_x) * voxel_size,
                            static_cast<float>(check_y) * voxel_size,
                            static_cast<float>(check_z) * voxel_size)) {

                            if (msg.getVariable<int>("agent_type") == CELL_TYPE_VASCULAR) {
                                const unsigned int neighbor_tip_id = msg.getVariable<unsigned int>("tip_id");
                                // HCC line 299: Check if different tip_id
                                if (neighbor_tip_id != my_tip_id) {
                                    nearby_vessel_exists = true;
                                    break;
                                }
                            }
                        }
                        if (nearby_vessel_exists) break;
                    }
                    if (nearby_vessel_exists) break;
                }
                if (nearby_vessel_exists) break;
            }

            // Only sprout if no nearby vessels from different networks (HCC lines 311-315)
            if (!nearby_vessel_exists) {
                divide_action = 2;  // INTENT_SPROUT_PHALANX
            }
        }
    }

    // Store division intent
    FLAMEGPU->setVariable<int>("intent_action", divide_action);

    return flamegpu::ALIVE;
}

// Select division target (two-phase division)
FLAMEGPU_AGENT_FUNCTION(vascular_select_divide_target, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
    const int divide_action = FLAMEGPU->getVariable<int>("intent_action");

    if (divide_action == 0) {
        return flamegpu::ALIVE;  // Not dividing
    }

    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const float voxel_size = FLAMEGPU->environment.getProperty<float>("voxel_size");

    // Output division intent message at current location (division happens in place)
    FLAMEGPU->message_out.setVariable<int>("agent_type", CELL_TYPE_VASCULAR);
    FLAMEGPU->message_out.setVariable<int>("agent_id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setVariable<int>("intent_action", divide_action);
    FLAMEGPU->message_out.setVariable<int>("target_x", x);
    FLAMEGPU->message_out.setVariable<int>("target_y", y);
    FLAMEGPU->message_out.setVariable<int>("target_z", z);

    FLAMEGPU->message_out.setLocation(
        static_cast<float>(x) * voxel_size,
        static_cast<float>(y) * voxel_size,
        static_cast<float>(z) * voxel_size
    );

    return flamegpu::ALIVE;
}

// Execute division (create daughter cell)
FLAMEGPU_AGENT_FUNCTION(vascular_execute_divide, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
    const int divide_action = FLAMEGPU->getVariable<int>("intent_action");

    if (divide_action == 0) {
        return flamegpu::ALIVE;  // Not dividing
    }

    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const float voxel_size = FLAMEGPU->environment.getProperty<float>("voxel_size");

    // Check for conflicts (multiple agents trying to divide at same location)
    bool has_conflict = false;
    const flamegpu::id_t my_id = FLAMEGPU->getID();

    for (const auto& msg : FLAMEGPU->message_in(
        static_cast<float>(x) * voxel_size,
        static_cast<float>(y) * voxel_size,
        static_cast<float>(z) * voxel_size)) {

        const int msg_action = msg.getVariable<int>("intent_action");
        if ((msg_action == 1 || msg_action == 2) &&  // Division intent
            msg.getVariable<int>("agent_id") < my_id) {
            has_conflict = true;
            break;
        }
    }

    if (!has_conflict) {
        // Perform division based on type
        if (divide_action == 1) {
            // TIP CELL DIVISION: Parent becomes STALK, daughter becomes TIP
            // Parent cell state change
            FLAMEGPU->setVariable<int>("vascular_state", VAS_STALK);

            // Create daughter as new TIP (inherits tip_id and move_direction)
            FLAMEGPU->agent_out.setVariable<int>("x", x);
            FLAMEGPU->agent_out.setVariable<int>("y", y);
            FLAMEGPU->agent_out.setVariable<int>("z", z);
            FLAMEGPU->agent_out.setVariable<int>("vascular_state", VAS_TIP);
            FLAMEGPU->agent_out.setVariable<unsigned int>("tip_id", FLAMEGPU->getVariable<unsigned int>("tip_id"));
            FLAMEGPU->agent_out.setVariable<float>("move_direction_x", FLAMEGPU->getVariable<float>("move_direction_x"));
            FLAMEGPU->agent_out.setVariable<float>("move_direction_y", FLAMEGPU->getVariable<float>("move_direction_y"));
            FLAMEGPU->agent_out.setVariable<float>("move_direction_z", FLAMEGPU->getVariable<float>("move_direction_z"));
            FLAMEGPU->agent_out.setVariable<int>("tumble", FLAMEGPU->getVariable<int>("tumble"));
            FLAMEGPU->agent_out.setVariable<int>("branch", 0);
            FLAMEGPU->agent_out.setVariable<float>("local_O2", FLAMEGPU->getVariable<float>("local_O2"));
            FLAMEGPU->agent_out.setVariable<float>("local_VEGFA", FLAMEGPU->getVariable<float>("local_VEGFA"));
            FLAMEGPU->agent_out.setVariable<float>("vegfa_grad_x", FLAMEGPU->getVariable<float>("vegfa_grad_x"));
            FLAMEGPU->agent_out.setVariable<float>("vegfa_grad_y", FLAMEGPU->getVariable<float>("vegfa_grad_y"));
            FLAMEGPU->agent_out.setVariable<float>("vegfa_grad_z", FLAMEGPU->getVariable<float>("vegfa_grad_z"));
            FLAMEGPU->agent_out.setVariable<float>("O2_source", 0.0f);
            FLAMEGPU->agent_out.setVariable<float>("VEGFA_sink", 0.0f);
            FLAMEGPU->agent_out.setVariable<int>("intent_action", INTENT_NONE);
            FLAMEGPU->agent_out.setVariable<int>("target_x", -1);
            FLAMEGPU->agent_out.setVariable<int>("target_y", -1);
            FLAMEGPU->agent_out.setVariable<int>("target_z", -1);
            FLAMEGPU->agent_out.setVariable<int>("mature_to_phalanx", 0);

        } else if (divide_action == 2) {
            // PHALANX SPROUTING: Parent stays PHALANX, daughter becomes new TIP with new tip_id
            // Parent stays as phalanx, just set branch flag
            FLAMEGPU->setVariable<int>("branch", 0);  // Clear branch flag after sprouting

            // Create daughter as new TIP with new unique tip_id
            // Use agent ID as new tip_id for uniqueness
            const unsigned int new_tip_id = static_cast<unsigned int>(my_id) + 1000000;  // Offset to avoid collision

            // Sample random initial direction
            const float theta = FLAMEGPU->random.uniform<float>() * 2.0f * 3.14159265f;
            const float phi = std::acos(2.0f * FLAMEGPU->random.uniform<float>() - 1.0f);
            const float dir_x = std::sin(phi) * std::cos(theta);
            const float dir_y = std::sin(phi) * std::sin(theta);
            const float dir_z = std::cos(phi);

            FLAMEGPU->agent_out.setVariable<int>("x", x);
            FLAMEGPU->agent_out.setVariable<int>("y", y);
            FLAMEGPU->agent_out.setVariable<int>("z", z);
            FLAMEGPU->agent_out.setVariable<int>("vascular_state", VAS_TIP);
            FLAMEGPU->agent_out.setVariable<unsigned int>("tip_id", new_tip_id);
            FLAMEGPU->agent_out.setVariable<float>("move_direction_x", dir_x);
            FLAMEGPU->agent_out.setVariable<float>("move_direction_y", dir_y);
            FLAMEGPU->agent_out.setVariable<float>("move_direction_z", dir_z);
            FLAMEGPU->agent_out.setVariable<int>("tumble", 1);  // Start in tumble phase
            FLAMEGPU->agent_out.setVariable<int>("branch", 0);
            FLAMEGPU->agent_out.setVariable<float>("local_O2", FLAMEGPU->getVariable<float>("local_O2"));
            FLAMEGPU->agent_out.setVariable<float>("local_VEGFA", FLAMEGPU->getVariable<float>("local_VEGFA"));
            FLAMEGPU->agent_out.setVariable<float>("vegfa_grad_x", 0.0f);
            FLAMEGPU->agent_out.setVariable<float>("vegfa_grad_y", 0.0f);
            FLAMEGPU->agent_out.setVariable<float>("vegfa_grad_z", 0.0f);
            FLAMEGPU->agent_out.setVariable<float>("O2_source", 0.0f);
            FLAMEGPU->agent_out.setVariable<float>("VEGFA_sink", 0.0f);
            FLAMEGPU->agent_out.setVariable<int>("intent_action", INTENT_NONE);
            FLAMEGPU->agent_out.setVariable<int>("target_x", -1);
            FLAMEGPU->agent_out.setVariable<int>("target_y", -1);
            FLAMEGPU->agent_out.setVariable<int>("target_z", -1);
            FLAMEGPU->agent_out.setVariable<int>("mature_to_phalanx", 0);
        }
    }

    // Reset division intent
    FLAMEGPU->setVariable<int>("intent_action", INTENT_NONE);

    return flamegpu::ALIVE;
}

} // namespace PDAC

#endif // FLAMEGPU_PDAC_VASCULAR_CELL_CUH
