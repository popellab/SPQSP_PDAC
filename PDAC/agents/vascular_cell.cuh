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
    FLAMEGPU->message_out.setVariable<float>("kill_factor", 0.0f);  // N/A for vascular

    FLAMEGPU->message_out.setLocation(
        static_cast<float>(x) * voxel_size,
        static_cast<float>(y) * voxel_size,
        static_cast<float>(z) * voxel_size
    );

    // Count this agent into per-state population snapshot
    auto* sc_vas = reinterpret_cast<unsigned int*>(FLAMEGPU->environment.getProperty<uint64_t>("state_counters_ptr"));
    const int sc_slot_vas = (vas_cs == VAS_TIP) ? SC_VAS_TIP :
                            (vas_cs == VAS_PHALANX_COLLAPSED) ? SC_VAS_COLLAPSED : SC_VAS_PHALANX;
    atomicAdd(&sc_vas[sc_slot_vas], 1u);

    return flamegpu::ALIVE;
}

// Volume occupancy + vascular tip_id grid.
// ALL vascular types write to vas_tip_id_grid for nearby-vessel exclusion
// and sprouting checks (stalk/phalanx presence = nonzero tip_id).
FLAMEGPU_AGENT_FUNCTION(vascular_write_to_occ_grid, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int cell_state = FLAMEGPU->getVariable<int>("cell_state");
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");

    const int nx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int nz = FLAMEGPU->environment.getProperty<int>("grid_size_z");
    if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz) {
        return flamegpu::ALIVE;
    }
    const int vidx = z * ny * nx + y * nx + x;

    // Tip_id grid for nearby-vessel exclusion + stalk/phalanx presence detection.
    unsigned int* vas_tip_id = reinterpret_cast<unsigned int*>(
        FLAMEGPU->environment.getProperty<uint64_t>("vas_tip_id_grid_ptr"));
    const unsigned int my_tip_id = FLAMEGPU->getVariable<unsigned int>("tip_id");
    atomicMax(&vas_tip_id[vidx], my_tip_id);

    // Volume-based occupancy
    float my_vol = (cell_state == VAS_TIP) ?
        FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_VAS_TIP") :
        FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_VAS_PHALANX");
    float* vol_used = VOL_PTR(FLAMEGPU);
    atomicAdd(&vol_used[vidx], my_vol);

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
    // Collapsed vessels produce no O2. Dysfunctional sprouts produce reduced O2.
    // Maturity modulates ECM compression resistance.
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

            // ECM compression: dense stroma compresses vessels → reduced O2 delivery
            // Mature vessels resist compression better than new sprouts
            const float* ecm_d_o2 = ECM_DENSITY_PTR(FLAMEGPU);
            float ecm_local_o2 = ecm_d_o2[voxel];
            float K_compress = FLAMEGPU->environment.getProperty<float>("PARAM_VAS_ECM_COMPRESS_K");
            const float mat_res = FLAMEGPU->environment.getProperty<float>("PARAM_VAS_MATURITY_RESISTANCE");
            const int mat_val = FLAMEGPU->getVariable<int>("maturity");
            float eff_compress_k = K_compress * (1.0f + mat_res * static_cast<float>(mat_val));
            float compression = ecm_local_o2 / (ecm_local_o2 + eff_compress_k + 1e-30f);
            KvLv *= (1.0f - compression);

            // Dysfunctional sprouts: permanently reduced O2 delivery
            if (FLAMEGPU->getVariable<int>("is_dysfunctional") == 1) {
                KvLv *= FLAMEGPU->environment.getProperty<float>("PARAM_VAS_KVL_DYSFUNCTIONAL");
            }

            // Implicit split: stable at large dt, drives C_local toward C_blood from below.
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
    // Only functional PHALANX cells mark recruitment sources (collapsed vessels excluded)
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

    int cell_state = FLAMEGPU->getVariable<int>("cell_state");
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const unsigned int my_tip_id = FLAMEGPU->getVariable<unsigned int>("tip_id");
    const float voxel_size = FLAMEGPU->environment.getProperty<float>("voxel_size");

    // --- Maturity: increment each step ---
    int maturity = FLAMEGPU->getVariable<int>("maturity");
    maturity++;
    FLAMEGPU->setVariable<int>("maturity", maturity);

    // --- Voxel index for ECM/PDE reads ---
    const int nx_v2 = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny_v2 = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int vidx_v2 = z * ny_v2 * nx_v2 + y * nx_v2 + x;
    const float* ecm_d = ECM_DENSITY_PTR(FLAMEGPU);
    const float ecm_local = ecm_d[vidx_v2];
    const float mat_resist = FLAMEGPU->environment.getProperty<float>("PARAM_VAS_MATURITY_RESISTANCE");

    // --- Vessel collapse: PHALANX → PHALANX_COLLAPSED ---
    if (cell_state == VAS_PHALANX) {
        const float collapse_th = FLAMEGPU->environment.getProperty<float>("PARAM_VAS_COLLAPSE_THRESHOLD");
        if (ecm_local > collapse_th) {
            const float collapse_ec50 = FLAMEGPU->environment.getProperty<float>("PARAM_VAS_COLLAPSE_EC50");
            float p_collapse = (ecm_local / (ecm_local + collapse_ec50))
                             / (1.0f + mat_resist * static_cast<float>(maturity));
            if (FLAMEGPU->random.uniform<float>() < p_collapse) {
                cell_state = VAS_PHALANX_COLLAPSED;
                FLAMEGPU->setVariable<int>("cell_state", VAS_PHALANX_COLLAPSED);
            }
        }
    }

    // --- Vessel recovery: PHALANX_COLLAPSED → PHALANX ---
    if (cell_state == VAS_PHALANX_COLLAPSED) {
        const float recovery_th = FLAMEGPU->environment.getProperty<float>("PARAM_VAS_RECOVERY_THRESHOLD");
        if (ecm_local < recovery_th) {
            const float recovery_rate = FLAMEGPU->environment.getProperty<float>("PARAM_VAS_RECOVERY_RATE");
            if (FLAMEGPU->random.uniform<float>() < recovery_rate) {
                cell_state = VAS_PHALANX;
                FLAMEGPU->setVariable<int>("cell_state", VAS_PHALANX);
            }
        }
    }

    // --- Vessel regression: low VEGF-A → death (PHALANX or COLLAPSED) ---
    if (cell_state == VAS_PHALANX || cell_state == VAS_PHALANX_COLLAPSED) {
        const float regress_rate = FLAMEGPU->environment.getProperty<float>("PARAM_VAS_REGRESS_RATE");
        const float vegfa_ec50 = FLAMEGPU->environment.getProperty<float>("PARAM_VAS_VEGFA_SURVIVAL_EC50");
        const float local_VEGFA_r = PDE_READ(FLAMEGPU, PDE_CONC_VEGFA, vidx_v2);
        // p_regress high when VEGF-A is low, reduced by maturity
        float p_regress = regress_rate * (1.0f - local_VEGFA_r / (local_VEGFA_r + vegfa_ec50))
                        / (1.0f + mat_resist * static_cast<float>(maturity));
        if (FLAMEGPU->random.uniform<float>() < p_regress) {
            auto* evts_reg = reinterpret_cast<unsigned int*>(
                FLAMEGPU->environment.getProperty<uint64_t>("event_counters_ptr"));
            const int ds = (cell_state == VAS_PHALANX_COLLAPSED) ? EVT_DEATH_VAS_COLLAPSED : EVT_DEATH_VAS_PHALANX;
            atomicAdd(&evts_reg[ds], 1u);
            return flamegpu::DEAD;
        }
    }

    int divide_action = INTENT_NONE;

    // VAS_TIP: divide only if voxel has no stalk/phalanx cell.
    // HCC Vas.cpp:248: if (_compartment->voxelIsOpen(getCoord(), getType()))
    // Stalk/phalanx cells write nonzero tip_id to vas_tip_id_grid; TIP cells
    // also write but we only gate on "is there already a vessel body here?"
    // which is equivalent since TIP is the one asking.
    if (cell_state == VAS_TIP) {
        const int nx_v = FLAMEGPU->environment.getProperty<int>("grid_size_x");
        const int ny_v = FLAMEGPU->environment.getProperty<int>("grid_size_y");
        const int vidx_v = z * ny_v * nx_v + y * nx_v + x;
        // Check volume capacity instead — if there's room, allow division
        float* vol_used = VOL_PTR(FLAMEGPU);
        float phalanx_vol = FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_VAS_PHALANX");
        float capacity = FLAMEGPU->environment.getProperty<float>("PARAM_VOXEL_CAPACITY");
        if (vol_used[vidx_v] + phalanx_vol <= capacity) {
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
    // TIP creates a PHALANX at current position. Check volume capacity.
    if (cell_state == VAS_TIP) {
        const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
        const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
        const int vidx = z * (grid_x * grid_y) + y * grid_x + x;

        float* vol_used = VOL_PTR(FLAMEGPU);
        const float capacity = FLAMEGPU->environment.getProperty<float>("PARAM_VOXEL_CAPACITY");
        float phalanx_vol = FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_VAS_PHALANX");

        if (!volume_try_claim(vol_used, vidx, phalanx_vol, capacity)) {
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
        // Inherit TIP's persist_dir so it can be passed to future sprouted TIPs.
        FLAMEGPU->agent_out.setVariable<int>("persist_dir_x", FLAMEGPU->getVariable<int>("persist_dir_x"));
        FLAMEGPU->agent_out.setVariable<int>("persist_dir_y", FLAMEGPU->getVariable<int>("persist_dir_y"));
        FLAMEGPU->agent_out.setVariable<int>("persist_dir_z", FLAMEGPU->getVariable<int>("persist_dir_z"));
        FLAMEGPU->agent_out.setVariable<int>("branch", new_branch);
        FLAMEGPU->agent_out.setVariable<int>("intent_action", INTENT_NONE);
        FLAMEGPU->agent_out.setVariable<int>("target_x", -1);
        FLAMEGPU->agent_out.setVariable<int>("target_y", -1);
        FLAMEGPU->agent_out.setVariable<int>("target_z", -1);
        FLAMEGPU->agent_out.setVariable<int>("mature_to_phalanx", 0);
        FLAMEGPU->agent_out.setVariable<int>("maturity", 0);
        // Dysfunctional if sprouting into hypoxic tissue
        const float local_O2_div = PDE_READ(FLAMEGPU, PDE_CONC_O2, voxel_d);
        const float vas_hyp_th = FLAMEGPU->environment.getProperty<float>("PARAM_VAS_HYPOXIA_TH");
        FLAMEGPU->agent_out.setVariable<int>("is_dysfunctional",
            (local_O2_div < vas_hyp_th) ? 1 : 0);
        auto* evts_tip = reinterpret_cast<unsigned int*>(FLAMEGPU->environment.getProperty<uint64_t>("event_counters_ptr"));
        atomicAdd(&evts_tip[EVT_PROLIF_VAS_PHALANX], 1u);

    // ── PHALANX SPROUTING ──────────────────────────────────────────────────────
    } else if (cell_state == VAS_PHALANX) {
        // Claim volume for the new TIP cell at this voxel
        const int grid_x_s = FLAMEGPU->environment.getProperty<int>("grid_size_x");
        const int grid_y_s = FLAMEGPU->environment.getProperty<int>("grid_size_y");
        const int vidx_s = z * (grid_x_s * grid_y_s) + y * grid_x_s + x;
        float* vol_used_s = VOL_PTR(FLAMEGPU);
        const float capacity_s = FLAMEGPU->environment.getProperty<float>("PARAM_VOXEL_CAPACITY");
        float tip_vol = FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_VAS_TIP");
        if (!volume_try_claim(vol_used_s, vidx_s, tip_vol, capacity_s)) {
            FLAMEGPU->setVariable<int>("intent_action", INTENT_NONE);
            return flamegpu::ALIVE;
        }

        // New tip gets a unique tip_id so nearby-vessel check works correctly.
        const unsigned int new_tip_id =
            static_cast<unsigned int>(FLAMEGPU->getID()) + 1000000u;

        // Inherit phalanx's stored persist_dir. New TIP starts with zero persist_dir
        // so it picks a gradient-biased direction on first move step.
        FLAMEGPU->agent_out.setVariable<int>("x", x);
        FLAMEGPU->agent_out.setVariable<int>("y", y);
        FLAMEGPU->agent_out.setVariable<int>("z", z);
        FLAMEGPU->agent_out.setVariable<int>("cell_state", VAS_TIP);
        FLAMEGPU->agent_out.setVariable<unsigned int>("tip_id", new_tip_id);
        FLAMEGPU->agent_out.setVariable<int>("persist_dir_x", 0);
        FLAMEGPU->agent_out.setVariable<int>("persist_dir_y", 0);
        FLAMEGPU->agent_out.setVariable<int>("persist_dir_z", 0);
        FLAMEGPU->agent_out.setVariable<int>("branch", 0);
        FLAMEGPU->agent_out.setVariable<int>("intent_action", INTENT_NONE);
        FLAMEGPU->agent_out.setVariable<int>("target_x", -1);
        FLAMEGPU->agent_out.setVariable<int>("target_y", -1);
        FLAMEGPU->agent_out.setVariable<int>("target_z", -1);
        FLAMEGPU->agent_out.setVariable<int>("mature_to_phalanx", 0);
        FLAMEGPU->agent_out.setVariable<int>("maturity", 0);
        FLAMEGPU->agent_out.setVariable<int>("is_dysfunctional", 0);

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
// Uses volume-based occupancy for movement collision detection.
// Vascular TIP cell movement via unified movement framework.
// Strong VEGF-A chemotaxis with high persistence. PHALANX cells don't move.
FLAMEGPU_AGENT_FUNCTION(vascular_move, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int cell_state = FLAMEGPU->getVariable<int>("cell_state");
    if (cell_state != VAS_TIP) return flamegpu::ALIVE;

    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");

    // Read VEGF-A gradient
    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int vidx = z * (grid_x * grid_y) + y * grid_x + x;
    const float gx = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_VEGFA_X))[vidx];
    const float gy = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_VEGFA_Y))[vidx];
    const float gz = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_VEGFA_Z))[vidx];

    MoveParams mp;
    mp.grid_x = grid_x;
    mp.grid_y = grid_y;
    mp.grid_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");
    mp.vol_used = VOL_PTR(FLAMEGPU);
    mp.my_vol = FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_VAS_TIP");
    mp.capacity = FLAMEGPU->environment.getProperty<float>("PARAM_VOXEL_CAPACITY");
    mp.ecm_density = ECM_DENSITY_PTR(FLAMEGPU);
    mp.ecm_crosslink = ECM_CROSSLINK_PTR(FLAMEGPU);
    mp.density_cap = FLAMEGPU->environment.getProperty<float>("PARAM_ECM_DENSITY_CAP");
    mp.min_porosity = FLAMEGPU->environment.getProperty<float>("PARAM_ECM_POROSITY_VAS_TIP");
    mp.p_move = 1.0f;
    mp.p_persist = FLAMEGPU->environment.getProperty<float>("PARAM_PERSIST_VAS_TIP");
    mp.bias_strength = FLAMEGPU->environment.getProperty<float>("PARAM_CHEMO_BIAS_VAS_TIP");
    mp.grad_x = gx; mp.grad_y = gy; mp.grad_z = gz;

    MoveResult r = move_cell(mp, x, y, z,
        FLAMEGPU->getVariable<int>("persist_dir_x"),
        FLAMEGPU->getVariable<int>("persist_dir_y"),
        FLAMEGPU->getVariable<int>("persist_dir_z"),
        FLAMEGPU->random.uniform<float>(),
        FLAMEGPU->random.uniform<float>(),
        FLAMEGPU->random.uniform<float>());

    if (r.moved) {
        FLAMEGPU->setVariable<int>("x", r.new_x);
        FLAMEGPU->setVariable<int>("y", r.new_y);
        FLAMEGPU->setVariable<int>("z", r.new_z);
        FLAMEGPU->setVariable<int>("persist_dir_x", r.persist_dx);
        FLAMEGPU->setVariable<int>("persist_dir_y", r.persist_dy);
        FLAMEGPU->setVariable<int>("persist_dir_z", r.persist_dz);
    }

    return flamegpu::ALIVE;
}

} // namespace PDAC

#endif // FLAMEGPU_PDAC_VASCULAR_CELL_CUH
