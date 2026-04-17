#ifndef FLAMEGPU_TNBC_CANCER_CELL_CUH
#define FLAMEGPU_TNBC_CANCER_CELL_CUH

#include "flamegpu/flamegpu.h"
#include "../core/common.cuh"

namespace PDAC {

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
    FLAMEGPU->message_out.setVariable<float>("kill_factor", 0.0f);  // N/A for cancer
    FLAMEGPU->message_out.setVariable<int>("dead", FLAMEGPU->getVariable<int>("dead"));  // Antigen source for B cells
    FLAMEGPU->message_out.setVariable<int>("voxel_x", x);
    FLAMEGPU->message_out.setVariable<int>("voxel_y", y);
    FLAMEGPU->message_out.setVariable<int>("voxel_z", z);

    FLAMEGPU->message_out.setLocation(
        (x + 0.5f) * voxel_size,
        (y + 0.5f) * voxel_size,
        (z + 0.5f) * voxel_size
    );

    // Count this agent into per-state population snapshot
    auto* sc = reinterpret_cast<unsigned int*>(FLAMEGPU->environment.getProperty<uint64_t>("state_counters_ptr"));
    const int cs = FLAMEGPU->getVariable<int>("cell_state");
    const int sc_slot = (cs == CANCER_STEM) ? SC_CANCER_STEM :
                        (cs == CANCER_PROGENITOR) ? SC_CANCER_PROG : SC_CANCER_SEN;
    atomicAdd(&sc[sc_slot], 1u);

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

    float tcyt_effective = 0.0f;  // Cytotoxic T cells weighted by hypoxia_kill_factor
    int treg_count = 0;    // Regulatory T cells
    int cancer_count = 0;  // Other cancer cells in neighborhood
    int mdsc_count = 0;    // MDSCs (suppress T cell killing)
    int mac_m1_count = 0;  // Mac M1 count
    int fib_count = 0;     // Fibroblasts

    // Adhesion: count all type+state neighbors for matrix dot product
    int adh_counts[ABM_STATE_COUNTER_SIZE] = {0};

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
            const float kill_factor = msg.getVariable<float>("kill_factor");

            // Adhesion: accumulate into type+state count vector
            adh_counts[msg_to_sc_idx(agent_type, agent_cell_state, kill_factor)]++;

            // Agent-specific interaction counts
            if (agent_type == CELL_TYPE_T) {
                if (agent_cell_state == T_CELL_CYT || agent_cell_state == T_CELL_EFF) {
                    tcyt_effective += kill_factor;
                }
            } else if (agent_type == CELL_TYPE_TREG) {
                treg_count++;
            } else if (agent_type == CELL_TYPE_MDSC) {
                mdsc_count++;
            } else if (agent_type == CELL_TYPE_CANCER) {
                cancer_count++;
            } else if (agent_type == CELL_TYPE_MAC) {
                if (agent_cell_state == MAC_M1) {
                    mac_m1_count++;
                }
            } else if (agent_type == CELL_TYPE_FIB) {
                fib_count++;
            }
        }
    }

    // Compute adhesion p_move from matrix
    const int my_sc = self_sc_idx(CELL_TYPE_CANCER, FLAMEGPU->getVariable<int>("cell_state"));
    const float adh_pmove = compute_adhesion_pmove(my_sc, adh_counts, ADH_MATRIX_PTR(FLAMEGPU));

    FLAMEGPU->setVariable<float>("neighbor_Teff_count", tcyt_effective);
    FLAMEGPU->setVariable<int>("neighbor_Treg_count", treg_count);
    FLAMEGPU->setVariable<int>("neighbor_cancer_count", cancer_count);
    FLAMEGPU->setVariable<int>("neighbor_MDSC_count", mdsc_count);
    FLAMEGPU->setVariable<int>("neighbor_Mac1_count", mac_m1_count);
    FLAMEGPU->setVariable<int>("neighbor_fib_count", fib_count);
    FLAMEGPU->setVariable<float>("adh_p_move", adh_pmove);

    return flamegpu::ALIVE;
}

// ============================================================
// Occupancy Grid Functions
// ============================================================

// Write this cancer cell's volume to the volume occupancy grid and mark
// the flat cancer presence array (used by recruitment density checks).
FLAMEGPU_AGENT_FUNCTION(cancer_write_to_occ_grid, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const int cell_state = FLAMEGPU->getVariable<int>("cell_state");

    const int gx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int gy = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int gz = FLAMEGPU->environment.getProperty<int>("grid_size_z");
    if (x < 0 || x >= gx || y < 0 || y >= gy || z < 0 || z >= gz) {
        return flamegpu::ALIVE;
    }
    const int vidx = z * (gx * gy) + y * gx + x;

    // Flat cancer presence array for recruitment density checks (is_tumor_dense_r3)
    unsigned int* cancer_occ = reinterpret_cast<unsigned int*>(
        FLAMEGPU->environment.getProperty<uint64_t>("cancer_occ_ptr"));
    atomicOr(&cancer_occ[vidx], 1u);

    // Volume-based occupancy
    float my_vol = (cell_state == CANCER_STEM) ?
        FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_CANCER_STEM") :
        (cell_state == CANCER_PROGENITOR) ?
        FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_CANCER_PROG") :
        FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_CANCER_SEN");
    float* vol_used = VOL_PTR(FLAMEGPU);
    atomicAdd(&vol_used[vidx], my_vol);

    return flamegpu::ALIVE;
}

// Single-phase cancer cell division using occupancy grid.
// Replaces the two-phase select_divide_target + execute_divide pair.
// Scans Von Neumann neighbors, shuffles candidates, tries atomicCAS until
// one succeeds or all are exhausted (reroll on contention).
FLAMEGPU_AGENT_FUNCTION(cancer_divide, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int divideFlag = FLAMEGPU->getVariable<int>("divideFlag");
    const int divideCD   = FLAMEGPU->getVariable<int>("divideCD");
    const int cell_state = FLAMEGPU->getVariable<int>("cell_state");

    if (FLAMEGPU->getVariable<int>("dead") == 1 ||
        divideFlag == 0 || divideCD > 0 || cell_state == CANCER_SENESCENT) {
        return flamegpu::ALIVE;
    }
    // Wave gate: only execute in the assigned wave round
    if (FLAMEGPU->getVariable<int>("divide_wave") !=
        FLAMEGPU->environment.getProperty<int>("divide_current_wave")) {
        return flamegpu::ALIVE;
    }

    const int my_x  = FLAMEGPU->getVariable<int>("x");
    const int my_y  = FLAMEGPU->getVariable<int>("y");
    const int my_z  = FLAMEGPU->getVariable<int>("z");
    const int size_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int size_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int size_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");

    // Volume-based occupancy: daughter volume for candidate pre-filter
    float* vol_used = VOL_PTR(FLAMEGPU);
    const float capacity = FLAMEGPU->environment.getProperty<float>("PARAM_VOXEL_CAPACITY");
    // Daughter volume depends on division outcome but use progenitor as conservative estimate
    float daughter_vol = FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_CANCER_PROG");

    // Collect Moore (26-direction) neighbors with enough volume capacity
    int cand_x[26], cand_y[26], cand_z[26];
    int n_cands = 0;
    for (int i = 0; i < 26; i++) {
        int dx, dy, dz;
        get_moore_direction(i, dx, dy, dz);
        const int nx = my_x + dx, ny = my_y + dy, nz = my_z + dz;
        if (!is_in_bounds(nx, ny, nz, size_x, size_y, size_z)) continue;
        int nvidx = nz * (size_x * size_y) + ny * size_x + nx;
        if (vol_used[nvidx] + daughter_vol <= capacity) {
            cand_x[n_cands] = nx;
            cand_y[n_cands] = ny;
            cand_z[n_cands] = nz;
            n_cands++;
        }
    }

    if (n_cands == 0) {
        return flamegpu::ALIVE;
    }

    // Fisher-Yates partial shuffle: try candidates in random order until volume claim wins.
    const float asymmetric_div_prob = FLAMEGPU->environment.getProperty<float>("PARAM_ASYM_DIV_PROB");
    const int divMax = FLAMEGPU->environment.getProperty<int>("PARAM_PROG_DIV_MAX");
    const unsigned int stem_id = FLAMEGPU->getVariable<unsigned int>("stemID");

    for (int i = 0; i < n_cands; i++) {
        // Swap with a random remaining candidate
        const int j = i + static_cast<int>(FLAMEGPU->random.uniform<float>() * (n_cands - i));
        int tx = cand_x[i]; cand_x[i] = cand_x[j]; cand_x[j] = tx;
        int ty = cand_y[i]; cand_y[i] = cand_y[j]; cand_y[j] = ty;
        int tz = cand_z[i]; cand_z[i] = cand_z[j]; cand_z[j] = tz;

        // Atomically claim volume for daughter cell
        int tvidx = cand_z[i] * (size_x * size_y) + cand_y[i] * size_x + cand_x[i];
        if (!volume_try_claim(vol_used, tvidx, daughter_vol, capacity)) continue;

        // Won the voxel — execute division
        const int target_x = cand_x[i];
        const int target_y = cand_y[i];
        const int target_z = cand_z[i];

        if (cell_state == CANCER_STEM) {
            float cabo = FLAMEGPU->environment.getProperty<float>("qsp_cabo_tumor");
            float R_cabo = FLAMEGPU->environment.getProperty<float>("R_cabo");
            float cabo_prolif_factor = 1 - (FLAMEGPU->environment.getProperty<float>("PARAM_LAMBDA_C_CABO") * 
                                            cabo / (cabo + FLAMEGPU->environment.getProperty<float>("PARAM_IC50_MET"))) * R_cabo;

            if (FLAMEGPU->random.uniform<float>() < asymmetric_div_prob) {
                // Asymmetric: daughter is progenitor
                const float div_int = FLAMEGPU->environment.getProperty<float>(
                    "PARAM_FLOAT_CANCER_CELL_PROGENITOR_DIV_INTERVAL_SLICE");
                FLAMEGPU->agent_out.setVariable<int>("x", target_x);
                FLAMEGPU->agent_out.setVariable<int>("y", target_y);
                FLAMEGPU->agent_out.setVariable<int>("z", target_z);
                FLAMEGPU->agent_out.setVariable<int>("cell_state", CANCER_PROGENITOR);
                FLAMEGPU->agent_out.setVariable<int>("divideCD", static_cast<int>(div_int + 0.5f));
                FLAMEGPU->agent_out.setVariable<int>("divideFlag", 1);
                FLAMEGPU->agent_out.setVariable<int>("divideCountRemaining", divMax);
                FLAMEGPU->agent_out.setVariable<unsigned int>("stemID", stem_id);
                FLAMEGPU->agent_out.setVariable<int>("newborn", 1);
            } else {
                // Symmetric: daughter is stem
                const float div_int = FLAMEGPU->environment.getProperty<float>(
                    "PARAM_FLOAT_CANCER_CELL_STEM_DIV_INTERVAL_SLICE");
                FLAMEGPU->agent_out.setVariable<int>("x", target_x);
                FLAMEGPU->agent_out.setVariable<int>("y", target_y);
                FLAMEGPU->agent_out.setVariable<int>("z", target_z);
                FLAMEGPU->agent_out.setVariable<int>("cell_state", CANCER_STEM);
                FLAMEGPU->agent_out.setVariable<int>("divideCD", static_cast<int>(div_int + 0.5f));
                FLAMEGPU->agent_out.setVariable<int>("divideFlag", 1);
                FLAMEGPU->agent_out.setVariable<int>("divideCountRemaining", 0);
                FLAMEGPU->agent_out.setVariable<unsigned int>("stemID", stem_id);
                FLAMEGPU->agent_out.setVariable<int>("newborn", 1);
            }
            // Reset parent divideCD — matches HCC deterministic reset
            FLAMEGPU->setVariable<int>("divideCD", static_cast<int>(
                (FLAMEGPU->environment.getProperty<float>("PARAM_FLOAT_CANCER_CELL_STEM_DIV_INTERVAL_SLICE")/cabo_prolif_factor) + 0.5f));

        } else if (cell_state == CANCER_PROGENITOR) {
            int divideCountRemaining = FLAMEGPU->getVariable<int>("divideCountRemaining");
            divideCountRemaining--;
            FLAMEGPU->setVariable<int>("divideCountRemaining", divideCountRemaining);

            const float div_int = FLAMEGPU->environment.getProperty<float>(
                "PARAM_FLOAT_CANCER_CELL_PROGENITOR_DIV_INTERVAL_SLICE");
            FLAMEGPU->agent_out.setVariable<int>("x", target_x);
            FLAMEGPU->agent_out.setVariable<int>("y", target_y);
            FLAMEGPU->agent_out.setVariable<int>("z", target_z);
            FLAMEGPU->agent_out.setVariable<int>("cell_state", CANCER_PROGENITOR);
            FLAMEGPU->agent_out.setVariable<int>("divideCD", static_cast<int>(div_int + 0.5f));
            FLAMEGPU->agent_out.setVariable<int>("divideFlag", 1);
            // Daughter inherits parent's (decremented) count — matches HCC behavior
            FLAMEGPU->agent_out.setVariable<int>("divideCountRemaining", divideCountRemaining);
            FLAMEGPU->agent_out.setVariable<unsigned int>("stemID", stem_id);
            FLAMEGPU->agent_out.setVariable<int>("newborn", 1);

            // H6 fix: Do NOT trigger senescence here. Match HCC behavior where
            // senescence is caught on the NEXT step by cancer_cell_state_step's
            // "PROGENITOR EXHAUSTION → SENESCENCE" check (lines ~480-491).
            // This gives a 1-step delay matching HCC's sequential processing.
            // Reset parent divideCD unconditionally — if divideCountRemaining <= 0,
            // state_step will catch it next step before any further division.
            FLAMEGPU->setVariable<int>("divideCD", static_cast<int>(div_int + 0.5f));
        }

        // Count successful division by parent state
        auto* evts = reinterpret_cast<unsigned int*>(FLAMEGPU->environment.getProperty<uint64_t>("event_counters_ptr"));
        atomicAdd(&evts[cell_state == CANCER_STEM ? EVT_PROLIF_CANCER_STEM : EVT_PROLIF_CANCER_PROG], 1u);

        break;  // Division done; stop trying candidates
    }

    return flamegpu::ALIVE;
}

// CancerCell agent function: State update — T cell/macrophage killing, division countdown, senescence
FLAMEGPU_AGENT_FUNCTION(cancer_cell_state_step, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int nx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int ax = FLAMEGPU->getVariable<int>("x");
    const int ay = FLAMEGPU->getVariable<int>("y");
    const int az = FLAMEGPU->getVariable<int>("z");
    const int voxel = az * ny*nx + ay * nx + ax;

    if (FLAMEGPU->getVariable<int>("dead") == 1) {
        return flamegpu::DEAD;
    }

    const int cell_state = FLAMEGPU->getVariable<int>("cell_state");

    // Antigen grid pointer (for depositing antigen on death)
    float* antigen_grid = ANTIGEN_GRID_PTR(FLAMEGPU);
    const float antigen_deposit = FLAMEGPU->environment.getProperty<float>("PARAM_ANTIGEN_DEPOSIT");

    // Senescent cells: countdown to death
    if (cell_state == CANCER_SENESCENT) {
        int life = FLAMEGPU->getVariable<int>("life");
        life--;
        if (life <= 0) {
            FLAMEGPU->setVariable<int>("dead", 1);
            FLAMEGPU->setVariable<int>("death_reason", 0);  // 0 = natural senescence
            auto* evts = reinterpret_cast<unsigned int*>(FLAMEGPU->environment.getProperty<uint64_t>("event_counters_ptr"));
            atomicAdd(&evts[EVT_DEATH_CANCER_SEN], 1u);
            atomicAdd(&antigen_grid[voxel], antigen_deposit);
            return flamegpu::DEAD;
        }
        FLAMEGPU->setVariable<int>("life", life);
    }

    // HIF-1α activation: binary switch at O2 threshold (all states)
    float local_O2 = PDE_READ(FLAMEGPU, PDE_CONC_O2, voxel);
    int hypoxic = (local_O2 < FLAMEGPU->environment.getProperty<float>("PARAM_CANCER_HYPOXIA_TH")) ? 1 : 0;
    FLAMEGPU->setVariable<int>("hypoxic", hypoxic);

    // Update PDL1: IFN-γ driven with exponential relaxation (both up and down)
    float local_IFNg = PDE_READ(FLAMEGPU, PDE_CONC_IFN, voxel);
    float PDL1 = update_PDL1(local_IFNg,
         FLAMEGPU->environment.getProperty<float>("PARAM_IFNG_PDL1_HALF"),
         FLAMEGPU->environment.getProperty<float>("PARAM_IFNG_PDL1_N"),
         FLAMEGPU->environment.getProperty<float>("PARAM_PDL1_SYN_MAX"),
         FLAMEGPU->getVariable<float>("PDL1_syn"),
         FLAMEGPU->environment.getProperty<float>("PARAM_PDL1_INTERNALIZATION_RATE"),
         FLAMEGPU->environment.getProperty<float>("PARAM_SEC_PER_SLICE"));
    if (hypoxic) {
        // HIF boost: additive fraction of PDL1_SYN_MAX
        PDL1 += FLAMEGPU->environment.getProperty<float>("PARAM_HIF_PDL1_BOOST")
              * FLAMEGPU->environment.getProperty<float>("PARAM_PDL1_SYN_MAX");
    }

    FLAMEGPU->setVariable<float>("PDL1_syn", PDL1);

    // Count PDL1-high cells (PDL1_syn > 0.5) for PDL1_frac computation
    if (PDL1 > 0.5f) {
        auto* evts = reinterpret_cast<unsigned int*>(FLAMEGPU->environment.getProperty<uint64_t>("event_counters_ptr"));
        atomicAdd(&evts[EVT_PDL1_COUNT], 1u);
    }

    // === T CELL KILLING ===
    float neighbor_Teff = FLAMEGPU->getVariable<float>("neighbor_Teff_count");  // Weighted by hypoxia_kill_factor
    if (neighbor_Teff > 0) {
        const int neighbor_cancer = FLAMEGPU->getVariable<int>("neighbor_cancer_count");
        const float PDL1 = FLAMEGPU->getVariable<float>("PDL1_syn");
        float nivo = FLAMEGPU->environment.getProperty<float>("qsp_nivo_tumor");
        float bond = get_PD1_PDL1(PDL1, nivo,
                        FLAMEGPU->environment.getProperty<float>("PARAM_PD1_SYN"),
                        FLAMEGPU->environment.getProperty<float>("PARAM_PDL1_K1"),
                        FLAMEGPU->environment.getProperty<float>("PARAM_PDL1_K2"),
                        FLAMEGPU->environment.getProperty<float>("PARAM_PDL1_K3"));
        float supp = hill_equation(bond,
                        FLAMEGPU->environment.getProperty<float>("PARAM_PD1_PDL1_HALF"),
                        FLAMEGPU->environment.getProperty<float>("PARAM_N_PD1_PDL1"));

        float NO   = PDE_READ(FLAMEGPU, PDE_CONC_NO, voxel);
        float ArgI = PDE_READ(FLAMEGPU, PDE_CONC_ARGI, voxel);
        float TGFB = PDE_READ(FLAMEGPU, PDE_CONC_TGFB, voxel);

        // MDSC suppression: ODE H_MDSC = 1-(1-H_NO)*(1-H_ArgI)
        float H_NO   = NO   / (NO   + FLAMEGPU->environment.getProperty<float>("PARAM_MDSC_IC50_NO_CTL"));
        float H_ArgI = ArgI / (ArgI + FLAMEGPU->environment.getProperty<float>("PARAM_MDSC_IC50_ArgI_CTL"));
        float one_minus_H_MDSC = (1.0f - H_NO) * (1.0f - H_ArgI);

        float H_TGFB = TGFB / (TGFB + FLAMEGPU->environment.getProperty<float>("PARAM_TEFF_TGFB_EC50"));

        // Density-based contact rate with Treg competitive suppression
        // ODE: CD8 / (CD8 + K_T_Treg*Tregs + cell); ABM adds cancer for target dilution
        const int neighbor_Treg = FLAMEGPU->getVariable<int>("neighbor_Treg_count");
        const float K_T_Treg = FLAMEGPU->environment.getProperty<float>("PARAM_K_T_TREG");
        float q = float(neighbor_Teff)
            / (neighbor_Teff + neighbor_cancer
                + K_T_Treg * neighbor_Treg
                + FLAMEGPU->environment.getProperty<float>("PARAM_CELL"));

        // Base kill probability from contact rate
        float p_kill = get_kill_probability(q,
            FLAMEGPU->environment.getProperty<float>("PARAM_ESCAPE_BASE"));

        // Suppression factors applied once each (matching ODE multiplicative structure)
        p_kill *= (1.0f - supp)      // PD1-PDL1 checkpoint
               *  (1.0f - H_TGFB)   // TGF-β suppression
               *  one_minus_H_MDSC   // MDSC (NO + ArgI)
               *  FLAMEGPU->environment.getProperty<float>("PARAM_TKILL_SCALAR");

        // HIF-active cancer cells downregulate MHC-I → reduced recognition
        if (hypoxic) {
            p_kill *= FLAMEGPU->environment.getProperty<float>("PARAM_HIF_MHC_REDUCTION");
        }

        if (FLAMEGPU->random.uniform<float>() < p_kill) {
            FLAMEGPU->setVariable<int>("dead", 1);
            FLAMEGPU->setVariable<int>("death_reason", 1);  // 1 = T cell killing
            auto* evts = reinterpret_cast<unsigned int*>(FLAMEGPU->environment.getProperty<uint64_t>("event_counters_ptr"));
            const int death_slot = (cell_state == CANCER_STEM) ? EVT_DEATH_CANCER_STEM :
                                   (cell_state == CANCER_PROGENITOR) ? EVT_DEATH_CANCER_PROG : EVT_DEATH_CANCER_SEN;
            atomicAdd(&evts[death_slot], 1u);
            atomicAdd(&antigen_grid[voxel], antigen_deposit);
            return flamegpu::DEAD;
        }
    }

    // === MACROPHAGE KILLING ===
    int neighbor_M1 = FLAMEGPU->getVariable<int>("neighbor_Mac1_count");
    if (neighbor_M1 > 0) {
        const int neighbor_cancer = FLAMEGPU->getVariable<int>("neighbor_cancer_count");
        const float PDL1 = FLAMEGPU->getVariable<float>("PDL1_syn");
        float nivo = FLAMEGPU->environment.getProperty<float>("qsp_nivo_tumor");
        float bond = get_PD1_PDL1(PDL1, nivo,
                        FLAMEGPU->environment.getProperty<float>("PARAM_MAC_PD1_SYN"),
                        FLAMEGPU->environment.getProperty<float>("PARAM_PDL1_K1"),
                        FLAMEGPU->environment.getProperty<float>("PARAM_PDL1_K2"),
                        FLAMEGPU->environment.getProperty<float>("PARAM_PDL1_K3"));

        float IL10 = PDE_READ(FLAMEGPU, PDE_CONC_IL10, voxel);
        float TGFB = PDE_READ(FLAMEGPU, PDE_CONC_TGFB, voxel);

        float A_SYN       = FLAMEGPU->environment.getProperty<float>("PARAM_A_SYN");
        float N_PD1_PDL1  = FLAMEGPU->environment.getProperty<float>("PARAM_N_PD1_PDL1");
        float PD1_PDL1_half = FLAMEGPU->environment.getProperty<float>("PARAM_PD1_PDL1_HALF");

        double H_PD1_M = hill_equation(bond / A_SYN, PD1_PDL1_half, N_PD1_PDL1);
        double kd = FLAMEGPU->environment.getProperty<float>("PARAM_KON_SIRPa_CD47")
                  / FLAMEGPU->environment.getProperty<float>("PARAM_KOFF_SIRPa_CD47");
        double a  = FLAMEGPU->environment.getProperty<float>("PARAM_C1_CD47_SYN");
        double b  = FLAMEGPU->environment.getProperty<float>("PARAM_MAC_SIRPa_SYN");
        double SIRPa_CD47_conc = ((a+b+1/kd) - std::sqrt((a+b+1/kd)*(a+b+1/kd) - 4*a*b)) / 2;
        double SIRPa_CD47_k50  = FLAMEGPU->environment.getProperty<float>("PARAM_MAC_SIRPa_HALF");
        double H_SIRPa_CD47_M  = hill_equation(SIRPa_CD47_conc, SIRPa_CD47_k50,
                                    FLAMEGPU->environment.getProperty<float>("PARAM_N_SIRPa_CD47"));
        double H_Mac_C = 1 - (1 - H_SIRPa_CD47_M) * (1 - H_PD1_M);
        double H_IL10_phago = IL10 / (IL10 + FLAMEGPU->environment.getProperty<float>("PARAM_MAC_IL_10_HALF_PHAGO"));
        double q = double(neighbor_M1) / (neighbor_M1 + neighbor_cancer
                   + FLAMEGPU->environment.getProperty<float>("PARAM_CELL"))
                   * (1 - H_Mac_C) * (1 - H_IL10_phago);
        double p_kill = get_kill_probability(q, FLAMEGPU->environment.getProperty<float>("PARAM_ESCAPE_MAC_BASE"));

        // HIF-active cancer cells downregulate MHC-I → reduced recognition
        if (hypoxic) {
            p_kill *= FLAMEGPU->environment.getProperty<float>("PARAM_HIF_MHC_REDUCTION");
        }

        // ADCC: antibody opsonization boosts macrophage killing
        {
            float ab_conc = PDE_READ(FLAMEGPU, PDE_CONC_ANTIBODY, voxel);
            float ab_ec50 = FLAMEGPU->environment.getProperty<float>("PARAM_ADCC_AB_EC50");
            float ab_n    = FLAMEGPU->environment.getProperty<float>("PARAM_ADCC_AB_HILL_N");
            float ab_max  = FLAMEGPU->environment.getProperty<float>("PARAM_ADCC_BOOST_MAX");
            float h_ab = ab_conc / (ab_conc + ab_ec50 + 1e-30f);  // Hill n=1 approx for speed
            if (ab_n != 1.0f) h_ab = powf(ab_conc, ab_n) / (powf(ab_conc, ab_n) + powf(ab_ec50, ab_n) + 1e-30f);
            p_kill *= (1.0 + ab_max * h_ab);  // boost: 1x (no Ab) to (1+ab_max)x (saturating Ab)
        }

        if (FLAMEGPU->random.uniform<float>() < p_kill) {
            FLAMEGPU->setVariable<int>("dead", 1);
            FLAMEGPU->setVariable<int>("death_reason", 2);  // 2 = macrophage killing
            auto* evts = reinterpret_cast<unsigned int*>(FLAMEGPU->environment.getProperty<uint64_t>("event_counters_ptr"));
            const int death_slot = (cell_state == CANCER_STEM) ? EVT_DEATH_CANCER_STEM :
                                   (cell_state == CANCER_PROGENITOR) ? EVT_DEATH_CANCER_PROG : EVT_DEATH_CANCER_SEN;
            atomicAdd(&evts[death_slot], 1u);
            atomicAdd(&antigen_grid[voxel], antigen_deposit);
            return flamegpu::DEAD;
        }
    }

    // === DIVISION COOLDOWN ===
    int divideCD = FLAMEGPU->getVariable<int>("divideCD");
    if (divideCD > 0) {
        divideCD--;
        FLAMEGPU->setVariable<int>("divideCD", divideCD);
    }

    // === WAVE ASSIGNMENT ===
    // Assign a random wave each step the cell is ready to divide so contention with
    // tcell/treg is interleaved rather than cancer always winning first pick.
    {
        const int divideFlag = FLAMEGPU->getVariable<int>("divideFlag");
        if (divideFlag == 1 && divideCD <= 0 && cell_state != CANCER_SENESCENT) {
            const int w = static_cast<int>(FLAMEGPU->random.uniform<float>() * N_DIVIDE_WAVES);
            FLAMEGPU->setVariable<int>("divide_wave", w < N_DIVIDE_WAVES ? w : N_DIVIDE_WAVES - 1);
        }
    }

    // === PROGENITOR EXHAUSTION → SENESCENCE ===
    // Matches HCC: senescence triggered in state_step (1-step delay after last division)
    if (cell_state == CANCER_PROGENITOR) {
        const int divideCountRemaining = FLAMEGPU->getVariable<int>("divideCountRemaining");
        if (divideCountRemaining <= 0) {
            FLAMEGPU->setVariable<int>("cell_state", CANCER_SENESCENT);
            FLAMEGPU->setVariable<int>("divideCD", -1);
            FLAMEGPU->setVariable<int>("divideFlag", 0);
            const float mean_life = FLAMEGPU->environment.getProperty<float>("PARAM_CANCER_SENESCENT_MEAN_LIFE");
            const float rand_val  = FLAMEGPU->random.uniform<float>();
            const int life = static_cast<int>(-mean_life * logf(rand_val + 0.0001f) + 0.5f);
            FLAMEGPU->setVariable<int>("life", life > 0 ? life : 1);
        }
    }

    return flamegpu::ALIVE;
}

// Cancer Cell agent function: Update chemicals from PDE
// Reads local concentrations and computes molecular responses
FLAMEGPU_AGENT_FUNCTION(cancer_update_chemicals, flamegpu::MessageNone, flamegpu::MessageNone) {
    //Everything in the statestep now
    return flamegpu::ALIVE;
}

// Cancer Cell agent function: Compute chemical source/sink rates
// Sets rates for PDE update based on cell state and environment
FLAMEGPU_AGENT_FUNCTION(cancer_compute_chemical_sources, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int cell_state = FLAMEGPU->getVariable<int>("cell_state");
    const int hypoxic = FLAMEGPU->getVariable<int>("hypoxic");
    const int dead = FLAMEGPU->getVariable<int>("dead");

    float O2_uptake = FLAMEGPU->environment.getProperty<float>("PARAM_O2_UPTAKE");

    // Get base rates from environment (per-state)
    float CCL2_release = 0.0f;
    float TGFB_release = 0.0f;
    float VEGFA_release = 0.0f;
    float IL1_release = 0.0f;
    float MMP_release = 0.0f;
    float CXCL12_release = 0.0f;
    float CCL5_release = 0.0f;
    if (cell_state == CANCER_STEM) {
        CCL2_release = FLAMEGPU->environment.getProperty<float>("PARAM_CCL2_RELEASE");
        TGFB_release = FLAMEGPU->environment.getProperty<float>("PARAM_STEM_TGFB_RELEASE");
        VEGFA_release = FLAMEGPU->environment.getProperty<float>("PARAM_STEM_VEGFA_RELEASE");
        IL1_release = FLAMEGPU->environment.getProperty<float>("PARAM_CANCER_IL1_RELEASE_STEM");
        MMP_release = FLAMEGPU->environment.getProperty<float>("PARAM_CANCER_MMP_RELEASE");
        CXCL12_release = FLAMEGPU->environment.getProperty<float>("PARAM_CXCL12_CANCER_RELEASE");
        CCL5_release = FLAMEGPU->environment.getProperty<float>("PARAM_CCL5_CANCER_RELEASE");
    } else if (cell_state == CANCER_PROGENITOR) {
        CCL2_release = FLAMEGPU->environment.getProperty<float>("PARAM_CCL2_RELEASE");
        TGFB_release = FLAMEGPU->environment.getProperty<float>("PARAM_PROG_TGFB_RELEASE");
        VEGFA_release = FLAMEGPU->environment.getProperty<float>("PARAM_PROG_VEGFA_RELEASE");
        IL1_release = FLAMEGPU->environment.getProperty<float>("PARAM_CANCER_IL1_RELEASE_PROG");
        MMP_release = FLAMEGPU->environment.getProperty<float>("PARAM_CANCER_MMP_RELEASE");
        CXCL12_release = FLAMEGPU->environment.getProperty<float>("PARAM_CXCL12_CANCER_RELEASE");
        CCL5_release = FLAMEGPU->environment.getProperty<float>("PARAM_CCL5_CANCER_RELEASE");
    } else if (cell_state == CANCER_SENESCENT) {
        IL1_release = FLAMEGPU->environment.getProperty<float>("PARAM_CANCER_IL1_RELEASE_SEN");
    }

    // HIF-1α boosts: upregulate VEGF-A and CCL2 under hypoxia
    if (hypoxic && dead == 0) {
        VEGFA_release *= FLAMEGPU->environment.getProperty<float>("PARAM_HIF_VEGF_BOOST");
        CCL2_release  *= FLAMEGPU->environment.getProperty<float>("PARAM_HIF_CCL2_BOOST");
    }

    // Dead cells don't produce or consume
    if (dead == 1) {
        CCL2_release = 0.0f;
        TGFB_release = 0.0f;
        VEGFA_release = 0.0f;
        IL1_release = 0.0f;
        MMP_release = 0.0f;
        CXCL12_release = 0.0f;
        CCL5_release = 0.0f;
        O2_uptake = 0.0f;
    }

    // Compute voxel index and volume for direct PDE writes
    const int nx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int ax = FLAMEGPU->getVariable<int>("x");
    const int ay = FLAMEGPU->getVariable<int>("y");
    const int az = FLAMEGPU->getVariable<int>("z");
    const int voxel = az * ny*nx + ay * nx + ax;

    const float vs_cm = FLAMEGPU->environment.getProperty<float>("voxel_size") * 1.0e-4f;
    const float voxel_volume = vs_cm * vs_cm * vs_cm;

    // Sources (secretion): atomicAdd to pde_source, divide by voxel_volume to get [conc/s]
    PDE_SECRETE(FLAMEGPU, PDE_SRC_CCL2, voxel, CCL2_release/voxel_volume);
    PDE_SECRETE(FLAMEGPU, PDE_SRC_TGFB, voxel, TGFB_release/voxel_volume);
    PDE_SECRETE(FLAMEGPU, PDE_SRC_VEGFA, voxel, VEGFA_release/voxel_volume);
    PDE_SECRETE(FLAMEGPU, PDE_SRC_IL1, voxel, IL1_release/voxel_volume);
    PDE_SECRETE(FLAMEGPU, PDE_SRC_MMP, voxel, MMP_release/voxel_volume);
    PDE_SECRETE(FLAMEGPU, PDE_SRC_CXCL12, voxel, CXCL12_release/voxel_volume);
    PDE_SECRETE(FLAMEGPU, PDE_SRC_CCL5, voxel, CCL5_release/voxel_volume);

    // Uptakes (consumption): atomicAdd to pde_uptake with positive magnitude [1/s]
    PDE_UPTAKE(FLAMEGPU, PDE_UPT_O2, voxel, O2_uptake);

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


// Cancer cell movement via unified movement framework.
// No persistence, no chemotaxis (bias=0, p_persist=0). Pure random walk.
FLAMEGPU_AGENT_FUNCTION(cancer_move, flamegpu::MessageNone, flamegpu::MessageNone) {
    if (FLAMEGPU->getVariable<int>("dead") == 1) return flamegpu::ALIVE;

    int moves_remaining = FLAMEGPU->getVariable<int>("moves_remaining");
    if (moves_remaining <= 0) return flamegpu::ALIVE;
    FLAMEGPU->setVariable<int>("moves_remaining", moves_remaining - 1);

    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const int cell_state = FLAMEGPU->getVariable<int>("cell_state");

    float my_vol = (cell_state == CANCER_STEM) ?
        FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_CANCER_STEM") :
        (cell_state == CANCER_PROGENITOR) ?
        FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_CANCER_PROG") :
        FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_CANCER_SEN");

    // Cancer stem gets weak O2 chemotaxis (deferred — CI=0 until O2 gradient computed)
    float bias = (cell_state == CANCER_STEM) ?
        ci_to_bias(FLAMEGPU->environment.getProperty<float>("PARAM_CHEMO_CI_CANCER_STEM")) : 0.0f;

    MoveParams mp;
    mp.grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    mp.grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    mp.grid_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");
    mp.vol_used = VOL_PTR(FLAMEGPU);
    mp.my_vol = my_vol;
    mp.capacity = FLAMEGPU->environment.getProperty<float>("PARAM_VOXEL_CAPACITY");
    mp.ecm_density = ECM_DENSITY_PTR(FLAMEGPU);
    mp.ecm_crosslink = ECM_CROSSLINK_PTR(FLAMEGPU);
    mp.density_cap = FLAMEGPU->environment.getProperty<float>("PARAM_ECM_DENSITY_CAP");
    mp.min_porosity = FLAMEGPU->environment.getProperty<float>("PARAM_ECM_POROSITY_CANCER");
    // Adhesion: pre-computed from matrix in count_neighbors
    mp.p_move = FLAMEGPU->getVariable<float>("adh_p_move");
    mp.p_persist = 0.0f;    // no persistence for cancer
    mp.bias_strength = bias;
    mp.grad_x = 0.0f; mp.grad_y = 0.0f; mp.grad_z = 0.0f;  // O2 gradient not yet computed
    mp.orient_x = ECM_ORIENT_X_PTR(FLAMEGPU);
    mp.orient_y = ECM_ORIENT_Y_PTR(FLAMEGPU);
    mp.orient_z = ECM_ORIENT_Z_PTR(FLAMEGPU);
    mp.barrier_strength = (cell_state == CANCER_STEM) ?
        FLAMEGPU->environment.getProperty<float>("PARAM_FIBER_BARRIER_CANCER_STEM") :
        FLAMEGPU->environment.getProperty<float>("PARAM_FIBER_BARRIER_CANCER_PROG");

    // Contact guidance: fiber orientation modifies effective gradient
    {
        const int vidx = z * (mp.grid_x * mp.grid_y) + y * mp.grid_x + x;
        float w_cg = (cell_state == CANCER_STEM) ?
            FLAMEGPU->environment.getProperty<float>("PARAM_CONTACT_GUIDANCE_CANCER_STEM") :
            FLAMEGPU->environment.getProperty<float>("PARAM_CONTACT_GUIDANCE_CANCER_PROG");
        float ox = ECM_ORIENT_X_PTR(FLAMEGPU)[vidx];
        float oy = ECM_ORIENT_Y_PTR(FLAMEGPU)[vidx];
        float oz = ECM_ORIENT_Z_PTR(FLAMEGPU)[vidx];
        if (bias > 0.0f) {
            apply_contact_guidance(mp.grad_x, mp.grad_y, mp.grad_z, ox, oy, oz, w_cg);
        } else {
            // No chemotaxis — use fiber + persistence for directional guidance
            apply_contact_guidance_persist(mp.grad_x, mp.grad_y, mp.grad_z, mp.bias_strength,
                ox, oy, oz, w_cg,
                FLAMEGPU->getVariable<int>("persist_dir_x"),
                FLAMEGPU->getVariable<int>("persist_dir_y"),
                FLAMEGPU->getVariable<int>("persist_dir_z"));
        }
    }

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

        // Deposit mechanical stress in movement direction (drives TACS-3 fiber reorientation)
        float stress_deposit = FLAMEGPU->environment.getProperty<float>("PARAM_ECM_STRESS_DEPOSIT");
        if (stress_deposit > 0.0f) {
            float* sx = STRESS_X_PTR(FLAMEGPU);
            float* sy = STRESS_Y_PTR(FLAMEGPU);
            float* sz = STRESS_Z_PTR(FLAMEGPU);
            float move_dx = static_cast<float>(r.persist_dx);
            float move_dy = static_cast<float>(r.persist_dy);
            float move_dz = static_cast<float>(r.persist_dz);
            float inv_len = rsqrtf(move_dx * move_dx + move_dy * move_dy + move_dz * move_dz + 1e-30f);
            move_dx *= inv_len * stress_deposit;
            move_dy *= inv_len * stress_deposit;
            move_dz *= inv_len * stress_deposit;
            int old_vidx = z * (mp.grid_x * mp.grid_y) + y * mp.grid_x + x;
            int new_vidx = r.new_z * (mp.grid_x * mp.grid_y) + r.new_y * mp.grid_x + r.new_x;
            atomicAdd(&sx[old_vidx], move_dx);
            atomicAdd(&sy[old_vidx], move_dy);
            atomicAdd(&sz[old_vidx], move_dz);
            atomicAdd(&sx[new_vidx], move_dx);
            atomicAdd(&sy[new_vidx], move_dy);
            atomicAdd(&sz[new_vidx], move_dz);
        }
    }

    return flamegpu::ALIVE;
}

} // namespace PDAC

#endif // PDAC_CANCER_CELL_CUH
