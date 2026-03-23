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
    const int vas_cs = FLAMEGPU->getVariable<int>("cell_state");
    FLAMEGPU->message_out.setVariable<int>("cell_state", vas_cs);
    FLAMEGPU->message_out.setVariable<int>("voxel_x", x);
    FLAMEGPU->message_out.setVariable<int>("voxel_y", y);
    FLAMEGPU->message_out.setVariable<int>("voxel_z", z);
    FLAMEGPU->message_out.setVariable<unsigned int>("tip_id", FLAMEGPU->getVariable<unsigned int>("tip_id"));

    FLAMEGPU->message_out.setLocation(
        static_cast<float>(x) * voxel_size,
        static_cast<float>(y) * voxel_size,
        static_cast<float>(z) * voxel_size
    );

    // Count this agent into per-state population snapshot
    auto* sc_vas = reinterpret_cast<unsigned int*>(FLAMEGPU->environment.getProperty<uint64_t>("state_counters_ptr"));
    atomicAdd(&sc_vas[vas_cs == VAS_TIP ? SC_VAS_TIP : SC_VAS_PHALANX], 1u);

    return flamegpu::ALIVE;
}

// Occupancy Grid
// STALK and PHALANX cells write to occ_grid (used for voxelIsOpen checks).
// TIP cells skip occ_grid but still write to vas_tip_id_grid so the
// nearby-vessel exclusion check (PHALANX sprouting) sees ALL vascular types,
// matching HCC's for_each_neighbor_ag which includes TIP/STALK/PHALANX.
FLAMEGPU_AGENT_FUNCTION(vascular_write_to_occ_grid, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int cell_state = FLAMEGPU->getVariable<int>("cell_state");
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");

    // Only stalk/phalanx count toward occupancy (TIP division checks this).
    if (cell_state != VAS_TIP) {
        auto occ = FLAMEGPU->environment.getMacroProperty<unsigned int,
            OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX, NUM_OCC_TYPES>("occ_grid");
        occ[x][y][z][CELL_TYPE_VASCULAR] += 1u;
    }

    // ALL vascular types write to tip_id grid for nearby-vessel exclusion.
    const int nx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int vidx = z * ny * nx + y * nx + x;
    unsigned int* vas_tip_id = reinterpret_cast<unsigned int*>(
        FLAMEGPU->environment.getProperty<uint64_t>("vas_tip_id_grid_ptr"));
    const unsigned int my_tip_id = FLAMEGPU->getVariable<unsigned int>("tip_id");
    atomicMax(&vas_tip_id[vidx], my_tip_id);

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
    // HCC uses: O2_transport = max(0, Kv*Lv*(C_blood - C_local))  — vessels only source.
    // We match this by applying the implicit split (stable at large dt) only when
    // C_local < C_blood.  When C_local >= C_blood the block is skipped entirely,
    // so vessels never act as O2 sinks — identical clamping behaviour to HCC.
    if (cell_state == VAS_PHALANX) {
        const float pi = 3.1415926f;
        const float sigma = FLAMEGPU->environment.getProperty<float>("PARAM_VAS_SIGMA");
        const float RC    = FLAMEGPU->environment.getProperty<float>("PARAM_VAS_RC");
        const float C_blood = FLAMEGPU->environment.getProperty<float>("PARAM_VAS_O2_CONC");

        const float C_local = PDE_READ(FLAMEGPU, PDE_CONC_O2, voxel);

        if (C_local < C_blood) {
            // Identical to HCC Tumor.cpp
            float Lv     = voxel_volume / (RC * RC * pi);           // [cm]  (vessel length)
            float Rt     = 1.0f / std::sqrt(Lv * pi);              // [cm^-0.5] (Krogh cylinder radius)
            float w      = RC / Rt;
            float lambda = 1.0f - w * w;
            float Kv = 2.0f * pi * FLAMEGPU->environment.getProperty<float>("PARAM_O2_DIFFUSIVITY")
                       * (lambda / (sigma * lambda - (2.0f + lambda) / 4.0f
                                    + (1.0f / lambda) * std::log(1.0f / w)));
            float KvLv = Kv * Lv;  // O2 transport coefficient [cm^3/s]

            // Implicit split: stable at large dt, drives C_local toward C_blood from below.
            // C_new cannot overshoot C_blood because the fixed point of the implicit scheme
            // is exactly C_blood, and backward Euler converges monotonically.
            //   source  += KvLv * C_blood / voxel_volume   [conc/s, constant inflow]
            //   uptake  += KvLv / voxel_volume              [1/s, handled implicitly by solver]
            PDE_SECRETE(FLAMEGPU, PDE_SRC_O2, voxel, KvLv * C_blood / voxel_volume);
            PDE_UPTAKE(FLAMEGPU,  PDE_UPT_O2, voxel, KvLv / voxel_volume);
        }
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

    // Check radius-3 BOX (7x7x7): skip marking if completely filled with cancer.
    // Matches HCC: 7x7x7 window_counts_inplace with local_cancer_ratio < 1 gate.
    const unsigned int* cancer_occ = reinterpret_cast<const unsigned int*>(
        FLAMEGPU->environment.getProperty<uint64_t>("cancer_occ_ptr"));
    int box_total = 0, box_cancer = 0;
    for (int dz = -3; dz <= 3; dz++) {
        for (int dy = -3; dy <= 3; dy++) {
            for (int dx = -3; dx <= 3; dx++) {
                int cx = x + dx, cy = y + dy, cz = z + dz;
                if (cx < 0 || cx >= grid_x || cy < 0 || cy >= grid_y || cz < 0 || cz >= grid_z) continue;
                box_total++;
                box_cancer += (cancer_occ[cz*(grid_x*grid_y) + cy*grid_x + cx] > 0u) ? 1 : 0;
            }
        }
    }
    if (box_total > 0 && box_cancer >= box_total) return flamegpu::ALIVE;

    const float ec50_ifng = FLAMEGPU->environment.getProperty<float>("PARAM_TEFF_IFN_EC50");
    const float H_IFNg = local_IFNg / (local_IFNg + ec50_ifng);

    double max_cancer = grid_x * grid_y * grid_z;
    double tumor_scaler = std::sqrt(1e5 * max_cancer / (FLAMEGPU->environment.getProperty<float>("qsp_cc_tumor") * FLAMEGPU->environment.getProperty<float>("AVOGADROS")));
    const int n_vas = FLAMEGPU->environment.getProperty<int>("n_vasculature_total");
    double vas_scaler = 100.0 / static_cast<double>(n_vas);
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
FLAMEGPU_AGENT_FUNCTION(vascular_state_step, flamegpu::MessageNone, flamegpu::MessageNone) {
    // Step 0: intent_action was pre-assigned by host initialization (assignInitialVascularTips).
    // Skip state transitions so the pre-set INTENT_DIVIDE flags are preserved for vascular_divide.
    if (FLAMEGPU->getStepCounter() == 0) {
        return flamegpu::ALIVE;
    }

    const int cell_state = FLAMEGPU->getVariable<int>("cell_state");
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const unsigned int my_tip_id = FLAMEGPU->getVariable<unsigned int>("tip_id");
    const float voxel_size = FLAMEGPU->environment.getProperty<float>("voxel_size");

    int divide_action = INTENT_NONE;

    // VAS_TIP: divide only if voxel is open for vascular type.
    // HCC Vas.cpp:248: if (_compartment->voxelIsOpen(getCoord(), getType()))
    // voxelIsOpen checks stalk+phalanx count < PARAM_VAS_MAX_PER_VOXEL (=1).
    // occ_grid[CELL_TYPE_VASCULAR] only counts stalk/phalanx (TIP skipped in write_to_occ_grid).
    if (cell_state == VAS_TIP) {
        auto occ = FLAMEGPU->environment.getMacroProperty<unsigned int,
            OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX, NUM_OCC_TYPES>("occ_grid");
        if (occ[x][y][z][CELL_TYPE_VASCULAR] == 0u) {
            divide_action = 1;  // INTENT_DIVIDE_TIP
        }
    }
    // VAS_STALK: no transitions
    else if (cell_state == VAS_STALK) {
        // Empty
    }
    // VAS_PHALANX: VEGF-dependent sprouting, or immediate branch if branch flag set
    else if (cell_state == VAS_PHALANX) {
        const int nx_ss = FLAMEGPU->environment.getProperty<int>("grid_size_x");
        const int ny_ss = FLAMEGPU->environment.getProperty<int>("grid_size_y");
        const int voxel_ss = z * ny_ss*nx_ss + y * nx_ss + x;
        const float local_VEGFA = PDE_READ(FLAMEGPU, PDE_CONC_VEGFA, voxel_ss);
        const float vas_50 = FLAMEGPU->environment.getProperty<float>("PARAM_VAS_50");
        const float p_tip = local_VEGFA / (vas_50 + local_VEGFA);

        if (FLAMEGPU->random.uniform<float>() < p_tip) {
            // Check for nearby vessels using the tip_id grid (written by write_to_occ_grid).
            // Direct array reads: O(range^3) lookups, no message overhead, correct tip_id filtering.
            const int range = static_cast<int>(
                FLAMEGPU->environment.getProperty<float>("PARAM_VAS_MIN_NEIGHBOR"));
            const int nx_g = FLAMEGPU->environment.getProperty<int>("grid_size_x");
            const int ny_g = FLAMEGPU->environment.getProperty<int>("grid_size_y");
            const int nz_g = FLAMEGPU->environment.getProperty<int>("grid_size_z");
            unsigned int* vas_tip_id = reinterpret_cast<unsigned int*>(
                FLAMEGPU->environment.getProperty<uint64_t>("vas_tip_id_grid_ptr"));

            bool nearby_vessel_exists = false;
            // HCC Vas.cpp:279 uses `i < num_neighbors` (exclusive), so range is [-15, 14].
            for (int dx = -range; dx < range && !nearby_vessel_exists; dx++) {
                for (int dy = -range; dy < range && !nearby_vessel_exists; dy++) {
                    for (int dz = -range; dz < range && !nearby_vessel_exists; dz++) {
                        if (dx == 0 && dy == 0 && dz == 0) continue;
                        const int cx = x + dx;
                        const int cy = y + dy;
                        const int cz = z + dz;
                        if (cx < 0 || cx >= nx_g || cy < 0 || cy >= ny_g || cz < 0 || cz >= nz_g) continue;
                        const unsigned int cell_tip_id = vas_tip_id[cz * ny_g * nx_g + cy * nx_g + cx];
                        if (cell_tip_id != 0u && cell_tip_id != my_tip_id) {
                            nearby_vessel_exists = true;
                        }
                    }
                }
            }
            if (!nearby_vessel_exists) {
                // Atomically mark this voxel with a unique sentinel so other PHALANX
                // cells in the same kernel see it during their nearby-vessel check.
                // In HCC, the new TIP is immediately visible (sequential processing);
                // this atomic write + threadfence emulates that on the GPU.
                const unsigned int sprout_marker =
                    static_cast<unsigned int>(FLAMEGPU->getID()) + 1000000u;
                atomicExch(&vas_tip_id[z * ny_g * nx_g + y * nx_g + x], sprout_marker);
                __threadfence();
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

        // Branch probability: newly-created phalanx may immediately flag for sprouting
        // (HCC Vas.cpp set_phalanx(): p_branch = PARAM_VAS_BRANCH_PROB * VEGFA / (vas_50 + VEGFA))
        const int size_x_d = FLAMEGPU->environment.getProperty<int>("grid_size_x");
        const int size_y_d = FLAMEGPU->environment.getProperty<int>("grid_size_y");
        const int voxel_d = z * size_y_d*size_x_d + y * size_x_d + x;
        const float local_VEGFA_d = PDE_READ(FLAMEGPU, PDE_CONC_VEGFA, voxel_d);
        const float vas_50_d = FLAMEGPU->environment.getProperty<float>("PARAM_VAS_50");
        const float branch_prob = FLAMEGPU->environment.getProperty<float>("PARAM_VAS_BRANCH_PROB");
        const float p_branch = branch_prob * local_VEGFA_d / (vas_50_d + local_VEGFA_d);
        const int new_branch = (FLAMEGPU->random.uniform<float>() < p_branch) ? 1 : 0;

        FLAMEGPU->agent_out.setVariable<int>("x", x);
        FLAMEGPU->agent_out.setVariable<int>("y", y);
        FLAMEGPU->agent_out.setVariable<int>("z", z);
        FLAMEGPU->agent_out.setVariable<int>("cell_state", VAS_PHALANX);
        FLAMEGPU->agent_out.setVariable<unsigned int>("tip_id", tip_id);
        // Inherit TIP's move_direction so it can be passed to future sprouted TIPs (HCC copy-ctor).
        FLAMEGPU->agent_out.setVariable<float>("move_direction_x", FLAMEGPU->getVariable<float>("move_direction_x"));
        FLAMEGPU->agent_out.setVariable<float>("move_direction_y", FLAMEGPU->getVariable<float>("move_direction_y"));
        FLAMEGPU->agent_out.setVariable<float>("move_direction_z", FLAMEGPU->getVariable<float>("move_direction_z"));
        FLAMEGPU->agent_out.setVariable<int>("tumble", 0);
        FLAMEGPU->agent_out.setVariable<int>("branch", new_branch);
        FLAMEGPU->agent_out.setVariable<int>("intent_action", INTENT_NONE);
        FLAMEGPU->agent_out.setVariable<int>("target_x", -1);
        FLAMEGPU->agent_out.setVariable<int>("target_y", -1);
        FLAMEGPU->agent_out.setVariable<int>("target_z", -1);
        FLAMEGPU->agent_out.setVariable<int>("mature_to_phalanx", 0);
        auto* evts_tip = reinterpret_cast<unsigned int*>(FLAMEGPU->environment.getProperty<uint64_t>("event_counters_ptr"));
        atomicAdd(&evts_tip[EVT_PROLIF_VAS_PHALANX], 1u);

    // ── PHALANX SPROUTING ──────────────────────────────────────────────────────
    // (HCC Vas.cpp lines 266-319 / Tumor.cpp lines 959-990)
    } else if (cell_state == VAS_PHALANX) {
        // No occupancy check — phalanx always sprouts (HCC ignores voxelIsOpen here).
        // New tip gets a unique tip_id so nearby-vessel check works correctly.
        const unsigned int new_tip_id =
            static_cast<unsigned int>(FLAMEGPU->getID()) + 1000000u;

        // Inherit phalanx's stored move_direction (HCC copy-ctor: _moveDirection = parent._moveDirection).
        // Phalanx carries the direction from the TIP that created it; new TIP inherits that direction
        // and starts in tumble mode so it immediately picks a biased direction on first move step.
        FLAMEGPU->agent_out.setVariable<int>("x", x);
        FLAMEGPU->agent_out.setVariable<int>("y", y);
        FLAMEGPU->agent_out.setVariable<int>("z", z);
        FLAMEGPU->agent_out.setVariable<int>("cell_state", VAS_TIP);
        FLAMEGPU->agent_out.setVariable<unsigned int>("tip_id", new_tip_id);
        FLAMEGPU->agent_out.setVariable<float>("move_direction_x", FLAMEGPU->getVariable<float>("move_direction_x"));
        FLAMEGPU->agent_out.setVariable<float>("move_direction_y", FLAMEGPU->getVariable<float>("move_direction_y"));
        FLAMEGPU->agent_out.setVariable<float>("move_direction_z", FLAMEGPU->getVariable<float>("move_direction_z"));
        FLAMEGPU->agent_out.setVariable<int>("tumble", 1);
        FLAMEGPU->agent_out.setVariable<int>("branch", 0);
        FLAMEGPU->agent_out.setVariable<int>("intent_action", INTENT_NONE);
        FLAMEGPU->agent_out.setVariable<int>("target_x", -1);
        FLAMEGPU->agent_out.setVariable<int>("target_y", -1);
        FLAMEGPU->agent_out.setVariable<int>("target_z", -1);
        FLAMEGPU->agent_out.setVariable<int>("mature_to_phalanx", 0);

        // Parent phalanx stays phalanx; clear branch flag and count new TIP
        FLAMEGPU->setVariable<int>("branch", 0);
        auto* evts_phalanx = reinterpret_cast<unsigned int*>(FLAMEGPU->environment.getProperty<uint64_t>("event_counters_ptr"));
        atomicAdd(&evts_phalanx[EVT_PROLIF_VAS_TIP], 1u);
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

    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int grid_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");

    // ECM based movement probability
    {
        const float* ecm_ptr = reinterpret_cast<const float*>(
            FLAMEGPU->environment.getProperty<uint64_t>("ecm_grid_ptr"));
        float ECM_density = ecm_ptr[z * (grid_x * grid_y) + y * grid_x + x];
        double ECM_sat = ECM_density / (ECM_density + FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ECM_MOT_EC50"));
        if (FLAMEGPU->random.uniform<float>() < ECM_sat) return flamegpu::ALIVE;
    }

    const float move_dir_x = FLAMEGPU->getVariable<float>("move_direction_x");
    const float move_dir_y = FLAMEGPU->getVariable<float>("move_direction_y");
    const float move_dir_z = FLAMEGPU->getVariable<float>("move_direction_z");
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
            // FLAMEGPU->setVariable<int>("tumble", 1);
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

        if (occ[target_x][target_y][target_z][CELL_TYPE_VASCULAR] != 0u) {
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
