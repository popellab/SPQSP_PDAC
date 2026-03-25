#ifndef FLAMEGPU_TNBC_CANCER_CELL_CUH
#define FLAMEGPU_TNBC_CANCER_CELL_CUH

#include "flamegpu/flamegpu.h"
#include "../core/common.cuh"

namespace PDAC {

// ============================================================================
// Helper: Deposit radial outward stress kernel around a cancer cell position.
// Pushes ECM orientation away from the cancer cell (expansion pressure).
// Called on successful movement and division events.
// ============================================================================
__device__ void deposit_cancer_stress(
    float* reorient_x, float* reorient_y, float* reorient_z,
    int cx, int cy, int cz,
    int grid_x, int grid_y, int grid_z,
    float strength)
{
    const int stress_radius = 5;
    const float variance = 4.0f;  // sigma^2 = 2^2

    for (int dx = -stress_radius; dx <= stress_radius; dx++) {
        int nx = cx + dx;
        if (nx < 0 || nx >= grid_x) continue;
        for (int dy = -stress_radius; dy <= stress_radius; dy++) {
            int ny = cy + dy;
            if (ny < 0 || ny >= grid_y) continue;
            for (int dz = -stress_radius; dz <= stress_radius; dz++) {
                int nz = cz + dz;
                if (nz < 0 || nz >= grid_z) continue;
                float dist_sq = static_cast<float>(dx * dx + dy * dy + dz * dz);
                if (dist_sq < 1e-6f) continue;  // skip self
                float weight = strength * expf(-dist_sq / (2.0f * variance));
                float inv_dist = rsqrtf(dist_sq);
                int vidx = nz * (grid_x * grid_y) + ny * grid_x + nx;
                // Radial outward: direction from cancer cell to voxel = (dx, dy, dz)
                atomicAdd(&reorient_x[vidx], weight * dx * inv_dist);
                atomicAdd(&reorient_y[vidx], weight * dy * inv_dist);
                atomicAdd(&reorient_z[vidx], weight * dz * inv_dist);
            }
        }
    }
}

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

    int tcyt_count = 0;    // Cytotoxic T cells (can kill)
    int treg_count = 0;    // Regulatory T cells
    int cancer_count = 0;  // Other cancer cells in neighborhood
    int mdsc_count = 0;    // MDSCs (suppress T cell killing)
    int mac_m1_count = 0;    // Mac M1 count

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
            } else if (agent_type == CELL_TYPE_MAC) {
                if (agent_cell_state == MAC_M1){
                    mac_m1_count++;
                }
            }
        }
    }

    // // Build available_neighbors mask (voxels with no cancer, 1 or fewer tcells, no MDSCs AND in bounds)
    // // Only scan Von Neumann neighbors for availability, can just skip other directions
    // // Counts for interactions already calculated
    // unsigned int available_neighbors = 0;
    // for (int i = 0; i < 6; i++) {
    //     int dx, dy, dz;
    //     get_moore_direction(i, dx, dy, dz);
    //     int nx = my_x + dx;
    //     int ny = my_y + dy;
    //     int nz = my_z + dz;

    //     if (is_in_bounds(nx, ny, nz, size_x, size_y, size_z)) {
    //         // Cancer cells can move to voxel only if no other Cancer cell, MDSC, or more than 1 T cell
    //         if (!neighbor_blocked[i] && neighbor_tcells[i] <= 1) {
    //             available_neighbors |= (1u << i);
    //         }
    //     }
    // }

    FLAMEGPU->setVariable<int>("neighbor_Teff_count", tcyt_count);
    FLAMEGPU->setVariable<int>("neighbor_Treg_count", treg_count);
    FLAMEGPU->setVariable<int>("neighbor_cancer_count", cancer_count);
    FLAMEGPU->setVariable<int>("neighbor_MDSC_count", mdsc_count);
    FLAMEGPU->setVariable<int>("neighbor_Mac1_count", mac_m1_count);

    return flamegpu::ALIVE;
}

// ============================================================
// Occupancy Grid Functions
// ============================================================

// Write this cancer cell's position to the occupancy grid.
// Called each step after zero_occupancy_grid, before division.
// Encodes cell_state+1 so downstream code can distinguish state from empty (0).
FLAMEGPU_AGENT_FUNCTION(cancer_write_to_occ_grid, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const int cell_state = FLAMEGPU->getVariable<int>("cell_state");
    auto occ = FLAMEGPU->environment.getMacroProperty<unsigned int,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX, NUM_OCC_TYPES>("occ_grid");
    // Store cell_state+1 so 0 means empty and non-zero means occupied.
    // Cancer cells are exclusive (1 per voxel) so no atomic needed, but
    // atomicExchange ensures coherence if any races occur.
    occ[x][y][z][CELL_TYPE_CANCER].exchange(static_cast<unsigned int>(cell_state) + 1u);

    // Also write to flat cancer occupancy array used by recruitment density checks.
    const int gx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int gy = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int vidx = z * (gx * gy) + y * gx + x;
    unsigned int* cancer_occ = reinterpret_cast<unsigned int*>(
        FLAMEGPU->environment.getProperty<uint64_t>("cancer_occ_ptr"));
    atomicOr(&cancer_occ[vidx], 1u);

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

    auto occ = FLAMEGPU->environment.getMacroProperty<unsigned int,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX, NUM_OCC_TYPES>("occ_grid");

    const uint8_t* face_flags = reinterpret_cast<const uint8_t*>(
        FLAMEGPU->environment.getProperty<uint64_t>("face_flags_ptr"));

    // Collect Moore (26-direction) neighbors that appear empty for cancer and MDSC.
    int cand_x[26], cand_y[26], cand_z[26];
    int n_cands = 0;
    for (int i = 0; i < 26; i++) {
        int dx, dy, dz;
        get_moore_direction(i, dx, dy, dz);
        const int nx = my_x + dx, ny = my_y + dy, nz = my_z + dz;
        if (!is_in_bounds(nx, ny, nz, size_x, size_y, size_z)) continue;
        if (is_ductal_wall_blocked(face_flags, my_x, my_y, my_z, dx, dy, dz, size_x, size_y)) continue;
        // Empty for cancer AND not occupied by MDSC (matching execute_divide logic)
        if (occ[nx][ny][nz][CELL_TYPE_CANCER] == 0u &&
            occ[nx][ny][nz][CELL_TYPE_FIB] == 0u) {
            cand_x[n_cands] = nx;
            cand_y[n_cands] = ny;
            cand_z[n_cands] = nz;
            n_cands++;
        }
    }

    if (n_cands == 0) {
        return flamegpu::ALIVE;
    }

    // Fisher-Yates partial shuffle: try candidates in random order until CAS wins.
    const float asymmetric_div_prob = FLAMEGPU->environment.getProperty<float>("PARAM_ASYM_DIV_PROB");
    const int divMax = FLAMEGPU->environment.getProperty<int>("PARAM_PROG_DIV_MAX");
    const unsigned int stem_id = FLAMEGPU->getVariable<unsigned int>("stemID");

    for (int i = 0; i < n_cands; i++) {
        // Swap with a random remaining candidate
        const int j = i + static_cast<int>(FLAMEGPU->random.uniform<float>() * (n_cands - i));
        int tx = cand_x[i]; cand_x[i] = cand_x[j]; cand_x[j] = tx;
        int ty = cand_y[i]; cand_y[i] = cand_y[j]; cand_y[j] = ty;
        int tz = cand_z[i]; cand_z[i] = cand_z[j]; cand_z[j] = tz;

        // Atomically claim this voxel for cancer (0=empty → 1=claimed)
        const unsigned int prev = occ[cand_x[i]][cand_y[i]][cand_z[i]][CELL_TYPE_CANCER].CAS(0u, 1u);
        if (prev != 0u) continue;  // Lost to a concurrent thread, try next

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

    // Senescent cells: countdown to death
    if (cell_state == CANCER_SENESCENT) {
        int life = FLAMEGPU->getVariable<int>("life");
        life--;
        if (life <= 0) {
            FLAMEGPU->setVariable<int>("dead", 1);
            FLAMEGPU->setVariable<int>("death_reason", 0);  // 0 = natural senescence
            auto* evts = reinterpret_cast<unsigned int*>(FLAMEGPU->environment.getProperty<uint64_t>("event_counters_ptr"));
            atomicAdd(&evts[EVT_DEATH_CANCER_SEN], 1u);
            return flamegpu::DEAD;
        }
        FLAMEGPU->setVariable<int>("life", life);
    }

    // Hypoxia check for division
    // if (cell_state == CANCER_PROGENITOR){
    //     const float O2_hypoxia_threshold = FLAMEGPU->environment.getProperty<float>("PARAM_CANCER_HYPOXIA_TH");
    //     float O2 = PDE_READ(FLAMEGPU, PDE_CONC_O2, voxel);
    //     int hypoxic = (O2 < O2_hypoxia_threshold) ? 1 : 0;
    //     int divide_flag = (O2 < O2_hypoxia_threshold) ? 0 : 1;
    //     FLAMEGPU->setVariable<int>("hypoxic", hypoxic);
    //     FLAMEGPU->setVariable<int>("divideFlag", divide_flag);
    // }

    // Update PDL1
    float local_IFNg = PDE_READ(FLAMEGPU, PDE_CONC_IFN, voxel);
    float PDL1 = update_PDL1(local_IFNg,
         FLAMEGPU->environment.getProperty<float>("PARAM_IFNG_PDL1_HALF"),
         FLAMEGPU->environment.getProperty<float>("PARAM_IFNG_PDL1_N"),
         FLAMEGPU->environment.getProperty<float>("PARAM_PDL1_SYN_MAX"),
         FLAMEGPU->getVariable<float>("PDL1_syn"));

    FLAMEGPU->setVariable<float>("PDL1_syn", PDL1);

    // Count PDL1-high cells (PDL1_syn > 0.5) for PDL1_frac computation
    if (PDL1 > 0.5f) {
        auto* evts = reinterpret_cast<unsigned int*>(FLAMEGPU->environment.getProperty<uint64_t>("event_counters_ptr"));
        atomicAdd(&evts[EVT_PDL1_COUNT], 1u);
    }

    // === T CELL KILLING ===
    int neighbor_Teff = FLAMEGPU->getVariable<int>("neighbor_Teff_count");
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

        float H_mdsc_c1 = 1 - (1 - (ArgI / (ArgI
            + FLAMEGPU->environment.getProperty<float>("PARAM_MDSC_IC50_ArgI_CTL"))));
        float H_TGFB = (TGFB / (TGFB
            + FLAMEGPU->environment.getProperty<float>("PARAM_TEFF_TGFB_EC50")));
        float q = (1 - H_mdsc_c1) * float(neighbor_Teff)
            / (neighbor_Teff + neighbor_cancer
                + FLAMEGPU->environment.getProperty<float>("PARAM_CELL")) * (1 - H_TGFB);

        float p_kill = get_kill_probability_supp(supp, q,
            FLAMEGPU->environment.getProperty<float>("PARAM_ESCAPE_BASE"));

        p_kill *= FLAMEGPU->environment.getProperty<float>("PARAM_TKILL_SCALAR") * (1 - supp);

        if (FLAMEGPU->random.uniform<float>() < p_kill) {
            FLAMEGPU->setVariable<int>("dead", 1);
            FLAMEGPU->setVariable<int>("death_reason", 1);  // 1 = T cell killing
            auto* evts = reinterpret_cast<unsigned int*>(FLAMEGPU->environment.getProperty<uint64_t>("event_counters_ptr"));
            const int death_slot = (cell_state == CANCER_STEM) ? EVT_DEATH_CANCER_STEM :
                                   (cell_state == CANCER_PROGENITOR) ? EVT_DEATH_CANCER_PROG : EVT_DEATH_CANCER_SEN;
            atomicAdd(&evts[death_slot], 1u);
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


        if (FLAMEGPU->random.uniform<float>() < p_kill) {
            FLAMEGPU->setVariable<int>("dead", 1);
            FLAMEGPU->setVariable<int>("death_reason", 2);  // 2 = macrophage killing
            auto* evts = reinterpret_cast<unsigned int*>(FLAMEGPU->environment.getProperty<uint64_t>("event_counters_ptr"));
            const int death_slot = (cell_state == CANCER_STEM) ? EVT_DEATH_CANCER_STEM :
                                   (cell_state == CANCER_PROGENITOR) ? EVT_DEATH_CANCER_PROG : EVT_DEATH_CANCER_SEN;
            atomicAdd(&evts[death_slot], 1u);
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
    float IFNg_uptake = FLAMEGPU->environment.getProperty<float>("PARAM_CANCER_IFNG_UPTAKE");

    // Get base rates from environment
    float CCL2_release = 0.0f;
    float TGFB_release = 0.0f;
    float VEGFA_release = 0.0f;
    if (cell_state == CANCER_STEM){
        CCL2_release = FLAMEGPU->environment.getProperty<float>("PARAM_CCL2_RELEASE");
        TGFB_release = FLAMEGPU->environment.getProperty<float>("PARAM_STEM_TGFB_RELEASE");
        VEGFA_release = FLAMEGPU->environment.getProperty<float>("PARAM_STEM_VEGFA_RELEASE");
    } else if (cell_state == CANCER_PROGENITOR){
        CCL2_release = FLAMEGPU->environment.getProperty<float>("PARAM_CCL2_RELEASE");
        TGFB_release = FLAMEGPU->environment.getProperty<float>("PARAM_PROG_TGFB_RELEASE");
        VEGFA_release = FLAMEGPU->environment.getProperty<float>("PARAM_PROG_VEGFA_RELEASE");
    }

    // Dead cells don't produce or consume
    if (dead == 1) {
        CCL2_release = 0.0f;
        TGFB_release = 0.0f;
        VEGFA_release = 0.0f;
        O2_uptake = 0.0f;
        IFNg_uptake = 0.0f;
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

    // Uptakes (consumption): atomicAdd to pde_uptake with positive magnitude [1/s]
    PDE_UPTAKE(FLAMEGPU, PDE_UPT_O2, voxel, O2_uptake);
    PDE_UPTAKE(FLAMEGPU, PDE_UPT_IFN, voxel, IFNg_uptake);

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


// Single-phase cancer cell movement using occupancy grid.
// Replaces two-phase select_move_target + execute_move.
// Reads occ_grid to find open Moore neighbors (26 directions), claims atomically with CAS.
// Cancer cells require target voxel to have no cancer, no MDSC, and no fibroblast.
FLAMEGPU_AGENT_FUNCTION(cancer_move, flamegpu::MessageNone, flamegpu::MessageNone) {
    if (FLAMEGPU->getVariable<int>("dead") == 1) return flamegpu::ALIVE;

    int moves_remaining = FLAMEGPU->getVariable<int>("moves_remaining");
    if (moves_remaining <= 0) return flamegpu::ALIVE;

    // Always consume the move attempt (matches HCC loop behavior where each
    // iteration counts regardless of ECM block or no-candidate outcome)
    FLAMEGPU->setVariable<int>("moves_remaining", moves_remaining - 1);

    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const int cell_state = FLAMEGPU->getVariable<int>("cell_state");
    const int size_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int size_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int size_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");

    // ECM based movement probability: higher ECM → more likely to be blocked
    {
        const float* ecm_ptr = reinterpret_cast<const float*>(
            FLAMEGPU->environment.getProperty<uint64_t>("ecm_grid_ptr"));
        float ECM_density = ecm_ptr[z * (size_x * size_y) + y * size_x + x];
        float ECM_50 = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ECM_MOT_EC50");
        float ECM_sat = ECM_density / (ECM_density + ECM_50);
        if (FLAMEGPU->random.uniform<float>() < ECM_sat) return flamegpu::ALIVE;
    }

    auto occ = FLAMEGPU->environment.getMacroProperty<unsigned int,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX, NUM_OCC_TYPES>("occ_grid");

    const uint8_t* face_flags = reinterpret_cast<const uint8_t*>(
        FLAMEGPU->environment.getProperty<uint64_t>("face_flags_ptr"));

    // ECM orientation field for contact guidance
    const float* orient_x = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>("ecm_orient_x_ptr"));
    const float* orient_y = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>("ecm_orient_y_ptr"));
    const float* orient_z = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>("ecm_orient_z_ptr"));

    // Moore neighborhood offsets (26 directions)
    // Build candidate list: neighbors empty of cancer and Fibs, filtered by contact guidance
    int cands[26];
    int n_cands = 0;
    for (int i = 0; i < 26; i++) {
        int ddx, ddy, ddz;
        get_moore_direction(i, ddx, ddy, ddz);
        int nx = x + ddx;
        int ny = y + ddy;
        int nz = z + ddz;
        if (nx < 0 || nx >= size_x || ny < 0 || ny >= size_y || nz < 0 || nz >= size_z) continue;
        if (is_ductal_wall_blocked(face_flags, x, y, z, ddx, ddy, ddz, size_x, size_y)) continue;
        if (occ[nx][ny][nz][CELL_TYPE_CANCER] == 0u && occ[nx][ny][nz][CELL_TYPE_FIB] == 0u) {
            // Contact guidance: check alignment of move direction with fiber orientation
            // at target voxel. Accept with probability based on |cos(theta)| (0→1 normalized).
            int tidx = nz * (size_x * size_y) + ny * size_x + nx;
            float fx = orient_x[tidx], fy = orient_y[tidx], fz = orient_z[tidx];
            float move_len = sqrtf(static_cast<float>(ddx*ddx + ddy*ddy + ddz*ddz));
            float cos_theta = (ddx * fx + ddy * fy + ddz * fz) / move_len;
            // Use absolute value: fibers are undirected (moving along OR against is fine)
            float alignment = fabsf(cos_theta);  // 0 = perpendicular, 1 = aligned
            // Accept with probability = alignment (fully aligned → always accept,
            // perpendicular → never accept). Add floor so cells aren't completely stuck.
            float p_accept = 0.3f + 0.7f * alignment;
            if (FLAMEGPU->random.uniform<float>() < p_accept) {
                cands[n_cands++] = i;
            }
        }
    }
    if (n_cands == 0) return flamegpu::ALIVE;

    // Fisher-Yates shuffle for random candidate order
    for (int i = n_cands - 1; i > 0; i--) {
        int j = FLAMEGPU->random.uniform<int>(0, i);
        int tmp = cands[i]; cands[i] = cands[j]; cands[j] = tmp;
    }

    // Try candidates in shuffled order; CAS to atomically claim new voxel
    const unsigned int claim_val = static_cast<unsigned int>(cell_state) + 1u;
    for (int i = 0; i < n_cands; i++) {
        int ddx, ddy, ddz;
        get_moore_direction(cands[i], ddx, ddy, ddz);
        int nx = x + ddx;
        int ny = y + ddy;
        int nz = z + ddz;
        unsigned int old = occ[nx][ny][nz][CELL_TYPE_CANCER].CAS(0u, claim_val);
        if (old == 0u) {
            // Won the voxel — release old and update position
            occ[x][y][z][CELL_TYPE_CANCER].exchange(0u);
            FLAMEGPU->setVariable<int>("x", nx);
            FLAMEGPU->setVariable<int>("y", ny);
            FLAMEGPU->setVariable<int>("z", nz);

            // Deposit radial outward stress kernel at new position
            float* reorient_x = reinterpret_cast<float*>(
                FLAMEGPU->environment.getProperty<uint64_t>("ecm_reorient_x_ptr"));
            float* reorient_y = reinterpret_cast<float*>(
                FLAMEGPU->environment.getProperty<uint64_t>("ecm_reorient_y_ptr"));
            float* reorient_z = reinterpret_cast<float*>(
                FLAMEGPU->environment.getProperty<uint64_t>("ecm_reorient_z_ptr"));
            deposit_cancer_stress(reorient_x, reorient_y, reorient_z,
                nx, ny, nz, size_x, size_y, size_z, 1.5f);

            break;
        }
    }

    return flamegpu::ALIVE;
}

} // namespace PDAC

#endif // PDAC_CANCER_CELL_CUH
