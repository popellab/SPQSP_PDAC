#ifndef FLAMEGPU_PDAC_VASCULAR_CELL_CUH
#define FLAMEGPU_PDAC_VASCULAR_CELL_CUH

#include "flamegpu/flamegpu.h"
#include "../core/common.cuh"

namespace PDAC {

// Broadcast location
FLAMEGPU_AGENT_FUNCTION(vascular_broadcast_location, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const float voxel_size = FLAMEGPU->environment.getProperty<float>("voxel_size");

    FLAMEGPU->message_out.setVariable<int>("agent_type", CELL_TYPE_VASCULAR);
    FLAMEGPU->message_out.setVariable<int>("agent_id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setVariable<int>("cell_state", FLAMEGPU->getVariable<int>("cell_state"));
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

// Occupancy Grid
// Only STALK and PHALANX cells contribute — tip cells divide in place and
// the tip-divide check is whether any stalk/phalanx is already present.
FLAMEGPU_AGENT_FUNCTION(vascular_write_to_occ_grid, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int cell_state = FLAMEGPU->getVariable<int>("cell_state");
    if (cell_state == VAS_TIP) {
        return flamegpu::ALIVE;
    }
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    auto occ = FLAMEGPU->environment.getMacroProperty<unsigned int,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX, NUM_OCC_TYPES>("occ_grid");
    occ[x][y][z][CELL_TYPE_VASCULAR] += 1u;
    return flamegpu::ALIVE;
}

// Read VEGF-A concentration and gradient from PDE (set by host function)
FLAMEGPU_AGENT_FUNCTION(vascular_update_chemicals, flamegpu::MessageNone, flamegpu::MessageNone) {
    // Chemical concentrations are now read directly from PDE device pointers where needed.
    return flamegpu::ALIVE;
}

// Compute O2 source and VEGF-A sink rates — atomicAdd directly to PDE arrays
FLAMEGPU_AGENT_FUNCTION(vascular_compute_chemical_sources, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int cell_state = FLAMEGPU->getVariable<int>("cell_state");

    // Compute voxel index and volume
    const int nx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int ax = FLAMEGPU->getVariable<int>("x");
    const int ay = FLAMEGPU->getVariable<int>("y");
    const int az = FLAMEGPU->getVariable<int>("z");
    const int voxel = az * ny*nx + ay * nx + ax;

    const float vs_cm = FLAMEGPU->environment.getProperty<float>("voxel_size") * 1.0e-4f;
    const float voxel_volume = vs_cm * vs_cm * vs_cm;

    // === O2 SECRETION via Krogh cylinder model (PHALANX ONLY) ===
    // Formula identical to HCC Tumor.cpp lines 326-336 (Sharan et al.)
    // HCC applies this as a per-step BioFVM source with small substeps.
    // Here we split Kv*Lv*(C_blood - C_local) = Kv*Lv*C_blood - Kv*Lv*C_local
    // and put Kv*Lv into the uptake array (implicit in backward Euler) to avoid
    // oscillation at large dt=21600s.
    if (cell_state == VAS_PHALANX) {
        const float pi = 3.1415926f;
        const float sigma = FLAMEGPU->environment.getProperty<float>("PARAM_VAS_SIGMA");
        const float RC    = FLAMEGPU->environment.getProperty<float>("PARAM_VAS_RC");
        const float C_blood = FLAMEGPU->environment.getProperty<float>("PARAM_VAS_O2_CONC");

        // Identical to HCC Tumor.cpp
        float Lv     = voxel_volume / (RC * RC * pi);           // [cm]  (vessel length)
        float Rt     = 1.0f / std::sqrt(Lv * pi);              // [cm^-0.5] (Krogh cylinder radius)
        float w      = RC / Rt;
        float lambda = 1.0f - w * w;
        float Kv = 2.0f * pi * FLAMEGPU->environment.getProperty<float>("PARAM_O2_DIFFUSIVITY")
                   * (lambda / (sigma * lambda - (2.0f + lambda) / 4.0f
                                + (1.0f / lambda) * std::log(1.0f / w)));
        float KvLv = Kv * Lv;  // O2 transport coefficient [cm^3/s]

        // Split into implicit form for numerical stability at large dt:
        //   source  += KvLv * C_blood / voxel_volume   [conc/s, constant inflow]
        //   uptake  += KvLv / voxel_volume              [1/s, handled implicitly by solver]
        PDE_SECRETE(FLAMEGPU, PDE_SRC_O2, voxel, KvLv * C_blood / voxel_volume);
        PDE_UPTAKE(FLAMEGPU,  PDE_UPT_O2, voxel, KvLv / voxel_volume);
    }

    // === VEGF-A UPTAKE (ALL STATES) ===
    // VEGFA_uptake is a rate constant [1/s]; atomicAdd to uptake array (no volume scaling)
    const float VEGFA_uptake_rate = FLAMEGPU->environment.getProperty<float>("PARAM_VEGFA_UPTAKE");
    PDE_UPTAKE(FLAMEGPU, PDE_UPT_VEGFA, voxel, VEGFA_uptake_rate);

    return flamegpu::ALIVE;
}

// Mark T cell recruitment sources (phalanx cells based on IFN-γ)
FLAMEGPU_AGENT_FUNCTION(vascular_mark_t_sources, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int cell_state = FLAMEGPU->getVariable<int>("cell_state");
    if (cell_state != VAS_PHALANX) {
        return flamegpu::ALIVE;
    }

    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int grid_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");

    const int voxel_ts = z * (grid_x * grid_y) + y * grid_x + x;
    const float local_IFNg = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>(PDE_CONC_IFN))[voxel_ts];

    const float ec50_ifng = FLAMEGPU->environment.getProperty<float>("PARAM_TEFF_IFN_EC50");
    const float H_IFNg = local_IFNg / (local_IFNg + ec50_ifng);

    double max_cancer = grid_x * grid_y * grid_z;
    double tumor_scaler = std::sqrt(1e5 * max_cancer / (FLAMEGPU->environment.getProperty<float>("qsp_cc_tumor") * FLAMEGPU->environment.getProperty<float>("AVOGADROS")));
    double vas_scaler = 100.0 / 200.0;
    const float p_entry = H_IFNg * tumor_scaler * vas_scaler;

    if (FLAMEGPU->random.uniform<float>() < p_entry) {
        unsigned long long ptr_val = FLAMEGPU->environment.getProperty<unsigned long long>("pde_recruitment_sources_ptr");
        int* d_recruitment_sources = reinterpret_cast<int*>(static_cast<uintptr_t>(ptr_val));
        int idx = z * (grid_x * grid_y) + y * grid_x + x;
        atomicOr(&d_recruitment_sources[idx], 1);  // Set T cell bit
    }

    return flamegpu::ALIVE;
}

// State transitions and division decisions for vascular cells
FLAMEGPU_AGENT_FUNCTION(vascular_state_step, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
    const int cell_state = FLAMEGPU->getVariable<int>("cell_state");
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const unsigned int my_tip_id = FLAMEGPU->getVariable<unsigned int>("tip_id");
    const float voxel_size = FLAMEGPU->environment.getProperty<float>("voxel_size");

    int divide_action = INTENT_NONE;

    // VAS_TIP: simple division if voxel is not too crowded
    // Tip cells do not appear to care about crowding, just going to let them divide
    if (cell_state == VAS_TIP) {
        // int vascular_neighbor_count = 0;
        // for (const auto& msg : FLAMEGPU->message_in(
        //     static_cast<float>(x) * voxel_size,
        //     static_cast<float>(y) * voxel_size,
        //     static_cast<float>(z) * voxel_size)) {
        //     if (msg.getVariable<int>("agent_type") == CELL_TYPE_VASCULAR) {
        //         vascular_neighbor_count++;
        //     }
        // }
        // if (vascular_neighbor_count < 3) {
        //     divide_action = 1;  // INTENT_DIVIDE_TIP
        // }
        divide_action = 1;
    }
    // VAS_STALK: no transitions
    else if (cell_state == VAS_STALK) {
        // Empty
    }
    // VAS_PHALANX: VEGF-dependent sprouting
    else if (cell_state == VAS_PHALANX) {
        const int nx_ss = FLAMEGPU->environment.getProperty<int>("grid_size_x");
        const int ny_ss = FLAMEGPU->environment.getProperty<int>("grid_size_y");
        const int voxel_ss = z * ny_ss*nx_ss + y * nx_ss + x;
        const float local_VEGFA = PDE_READ(FLAMEGPU, PDE_CONC_VEGFA, voxel_ss);
        const float vas_50 = FLAMEGPU->environment.getProperty<float>("PARAM_VAS_50");
        const float p_tip = local_VEGFA / (vas_50 + local_VEGFA);

        if (FLAMEGPU->random.uniform<float>() < p_tip) {
            const int min_neighbor_range = static_cast<int>(
                FLAMEGPU->environment.getProperty<float>("PARAM_VAS_MIN_NEIGHBOR"));
            bool nearby_vessel_exists = false;

            for (int dx = -min_neighbor_range; dx <= min_neighbor_range && !nearby_vessel_exists; dx++) {
                for (int dy = -min_neighbor_range; dy <= min_neighbor_range && !nearby_vessel_exists; dy++) {
                    for (int dz = -min_neighbor_range; dz <= min_neighbor_range && !nearby_vessel_exists; dz++) {
                        if (dx == 0 && dy == 0 && dz == 0) continue;
                        for (const auto& msg : FLAMEGPU->message_in(
                            static_cast<float>(x+dx) * voxel_size,
                            static_cast<float>(y+dy) * voxel_size,
                            static_cast<float>(z+dz) * voxel_size)) {
                            if (msg.getVariable<int>("agent_type") == CELL_TYPE_VASCULAR &&
                                msg.getVariable<unsigned int>("tip_id") != my_tip_id) {
                                nearby_vessel_exists = true;
                                break;
                            }
                        }
                    }
                }
            }
            if (!nearby_vessel_exists) {
                divide_action = 2;  // INTENT_SPROUT_PHALANX
            }
        }
    }

    FLAMEGPU->setVariable<int>("intent_action", divide_action);
    return flamegpu::ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(vascular_divide, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int divide_action = FLAMEGPU->getVariable<int>("intent_action");
    if (divide_action == 0) {
        return flamegpu::ALIVE;
    }

    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const int cell_state = FLAMEGPU->getVariable<int>("cell_state");

    // ── TIP CELL DIVISION ──────────────────────────────────────────────────────
    // (HCC Vas.cpp agent_state_step lines 248-257 / Tumor.cpp lines 941-958)
    if (cell_state == VAS_TIP) {
        // Abort if any stalk or phalanx already occupies this voxel.
        // Tip cells are excluded from the occ_grid (write_to_occ_grid skips VAS_TIP).
        auto occ = FLAMEGPU->environment.getMacroProperty<unsigned int,
            OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX, NUM_OCC_TYPES>("occ_grid");
        if (occ[x][y][z][CELL_TYPE_VASCULAR] != 0u) {
            FLAMEGPU->setVariable<int>("intent_action", INTENT_NONE);
            return flamegpu::ALIVE;
        }

        // Parent TIP stays TIP (moves away next step, leaving phalanx behind).
        const unsigned int tip_id = FLAMEGPU->getVariable<unsigned int>("tip_id");

        FLAMEGPU->agent_out.setVariable<int>("x", x);
        FLAMEGPU->agent_out.setVariable<int>("y", y);
        FLAMEGPU->agent_out.setVariable<int>("z", z);
        FLAMEGPU->agent_out.setVariable<int>("cell_state", VAS_PHALANX);
        FLAMEGPU->agent_out.setVariable<unsigned int>("tip_id", tip_id);
        FLAMEGPU->agent_out.setVariable<float>("move_direction_x", 0.0f);
        FLAMEGPU->agent_out.setVariable<float>("move_direction_y", 0.0f);
        FLAMEGPU->agent_out.setVariable<float>("move_direction_z", 0.0f);
        FLAMEGPU->agent_out.setVariable<int>("tumble", 0);
        FLAMEGPU->agent_out.setVariable<int>("branch", 0);
        FLAMEGPU->agent_out.setVariable<int>("intent_action", INTENT_NONE);
        FLAMEGPU->agent_out.setVariable<int>("target_x", -1);
        FLAMEGPU->agent_out.setVariable<int>("target_y", -1);
        FLAMEGPU->agent_out.setVariable<int>("target_z", -1);
        FLAMEGPU->agent_out.setVariable<int>("mature_to_phalanx", 0);

    // ── PHALANX SPROUTING ──────────────────────────────────────────────────────
    // (HCC Vas.cpp lines 266-319 / Tumor.cpp lines 959-990)
    } else if (cell_state == VAS_PHALANX) {
        // No occupancy check — phalanx always sprouts (HCC ignores voxelIsOpen here).
        // New tip gets a unique tip_id so nearby-vessel check works correctly.
        const unsigned int new_tip_id =
            static_cast<unsigned int>(FLAMEGPU->getID()) + 1000000u;

        // Random initial direction; tip will orient via run-tumble on first step.
        const float theta = FLAMEGPU->random.uniform<float>() * 2.0f * 3.14159265f;
        const float phi   = acosf(2.0f * FLAMEGPU->random.uniform<float>() - 1.0f);

        FLAMEGPU->agent_out.setVariable<int>("x", x);
        FLAMEGPU->agent_out.setVariable<int>("y", y);
        FLAMEGPU->agent_out.setVariable<int>("z", z);
        FLAMEGPU->agent_out.setVariable<int>("cell_state", VAS_TIP);
        FLAMEGPU->agent_out.setVariable<unsigned int>("tip_id", new_tip_id);
        FLAMEGPU->agent_out.setVariable<float>("move_direction_x", sinf(phi) * cosf(theta));
        FLAMEGPU->agent_out.setVariable<float>("move_direction_y", sinf(phi) * sinf(theta));
        FLAMEGPU->agent_out.setVariable<float>("move_direction_z", cosf(phi));
        FLAMEGPU->agent_out.setVariable<int>("tumble", 1);
        FLAMEGPU->agent_out.setVariable<int>("branch", 0);
        FLAMEGPU->agent_out.setVariable<int>("intent_action", INTENT_NONE);
        FLAMEGPU->agent_out.setVariable<int>("target_x", -1);
        FLAMEGPU->agent_out.setVariable<int>("target_y", -1);
        FLAMEGPU->agent_out.setVariable<int>("target_z", -1);
        FLAMEGPU->agent_out.setVariable<int>("mature_to_phalanx", 0);

        // Parent phalanx stays phalanx; clear branch flag
        FLAMEGPU->setVariable<int>("branch", 0);
    }

    FLAMEGPU->setVariable<int>("intent_action", INTENT_NONE);
    return flamegpu::ALIVE;
}

// Single-phase vascular tip cell movement using run-tumble algorithm.
// Replaces two-phase vascular_select_move_target + vascular_execute_move.
// Only VAS_TIP cells move; STALK and PHALANX are stationary.
// Tip cells are not tracked in occ_grid so no CAS is needed for movement.
FLAMEGPU_AGENT_FUNCTION(vascular_move, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int cell_state = FLAMEGPU->getVariable<int>("cell_state");

    // Only tip cells move
    if (cell_state != VAS_TIP) {
        return flamegpu::ALIVE;
    }

    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const int tumble = FLAMEGPU->getVariable<int>("tumble");

    // ECM based movement probability
    auto ecm = FLAMEGPU->environment.getMacroProperty<float,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX>("ecm_grid");
    float ECM_density = ecm[x][y][z];
    double ECM_sat = ECM_density / (ECM_density + FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ECM_MOT_EC50"));
    if (FLAMEGPU->random.uniform<float>() < ECM_sat) return flamegpu::ALIVE;

    const float move_dir_x = FLAMEGPU->getVariable<float>("move_direction_x");
    const float move_dir_y = FLAMEGPU->getVariable<float>("move_direction_y");
    const float move_dir_z = FLAMEGPU->getVariable<float>("move_direction_z");

    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int grid_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");
    const float dt = FLAMEGPU->environment.getProperty<float>("PARAM_SEC_PER_SLICE");

    // VEGFA gradient — read directly from PDE
    const int voxel_mv = z * grid_y*grid_x + y * grid_x + x;
    const float vegfa_grad_x = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_VEGFA_X))[voxel_mv];
    const float vegfa_grad_y = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_VEGFA_Y))[voxel_mv];
    const float vegfa_grad_z = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_VEGFA_Z))[voxel_mv];

    int target_x = x;
    int target_y = y;
    int target_z = z;

    auto occ = FLAMEGPU->environment.getMacroProperty<unsigned int,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX, NUM_OCC_TYPES>("occ_grid");

    // === RUN PHASE (tumble == 0) ===
    if (tumble == 0) {
        float v_x = move_dir_x / dt;
        float v_y = move_dir_y / dt;
        float v_z = move_dir_z / dt;

        float norm_gradient = std::sqrt(vegfa_grad_x * vegfa_grad_x +
                                        vegfa_grad_y * vegfa_grad_y +
                                        vegfa_grad_z * vegfa_grad_z);

        float dot_product = v_x * vegfa_grad_x + v_y * vegfa_grad_y + v_z * vegfa_grad_z;
        float norm_v = std::sqrt(v_x * v_x + v_y * v_y + v_z * v_z);
        float cos_theta = dot_product / (norm_v * norm_gradient);

        const float EC50_grad = 1.0f;
        float H_grad = norm_gradient / (norm_gradient + EC50_grad);
        if (cos_theta < 0) H_grad = -H_grad;

        const float lambda = FLAMEGPU->environment.getProperty<float>("PARAM_VAS_TUMBLE");
        const float delta = FLAMEGPU->environment.getProperty<float>("PARAM_VAS_DELTA");
        float tumble_rate = (lambda / 2.0f) * (1.0f - cos_theta) * (1.0f - H_grad) * dt + delta;
        float p_tumble = 1.0f - std::exp(-tumble_rate);

        if (FLAMEGPU->random.uniform<float>() < p_tumble) {
            FLAMEGPU->setVariable<int>("tumble", 1);
            return flamegpu::ALIVE;
        }

        target_x = x + static_cast<int>(std::round(move_dir_x));
        target_y = y + static_cast<int>(std::round(move_dir_y));
        target_z = z + static_cast<int>(std::round(move_dir_z));

        if (target_x < 0 || target_x >= grid_x ||
            target_y < 0 || target_y >= grid_y ||
            target_z < 0 || target_z >= grid_z) {
            return flamegpu::ALIVE;
        }

        // Only move if target voxel is not occupied by a stationary vascular cell (HCC: voxelIsOpen)
        if (occ[target_x][target_y][target_z][CELL_TYPE_VASCULAR] != 0u) {
            FLAMEGPU->setVariable<int>("tumble", 1);
            return flamegpu::ALIVE;
        }
    }
    // === TUMBLE PHASE (tumble == 1) ===
    else {
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
                        // HCC formula: exp(cos_theta/sigma*sigma) / exp(1/sigma*sigma)
                    // Due to C++ left-to-right precedence this simplifies to exp(cos_theta - 1)
                    float rho = std::exp(cos_theta - 1.0f);
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

        int dx = dirs[selected_idx][0];
        int dy = dirs[selected_idx][1];
        int dz = dirs[selected_idx][2];

        target_x = x + dx;
        target_y = y + dy;
        target_z = z + dz;

        if (target_x < 0 || target_x >= grid_x ||
            target_y < 0 || target_y >= grid_y ||
            target_z < 0 || target_z >= grid_z) {
            FLAMEGPU->setVariable<int>("tumble", 1);
            return flamegpu::ALIVE;
        }

        FLAMEGPU->setVariable<float>("move_direction_x", static_cast<float>(dx));
        FLAMEGPU->setVariable<float>("move_direction_y", static_cast<float>(dy));
        FLAMEGPU->setVariable<float>("move_direction_z", static_cast<float>(dz));
        FLAMEGPU->setVariable<int>("tumble", 0);
    }

    // Apply movement directly (tip cells are not in occ_grid; no conflict resolution needed)
    FLAMEGPU->setVariable<int>("x", target_x);
    FLAMEGPU->setVariable<int>("y", target_y);
    FLAMEGPU->setVariable<int>("z", target_z);
    return flamegpu::ALIVE;
}

} // namespace PDAC

#endif // FLAMEGPU_PDAC_VASCULAR_CELL_CUH
