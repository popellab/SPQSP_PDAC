#ifndef FLAMEGPU_TNBC_T_CELL_CUH
#define FLAMEGPU_TNBC_T_CELL_CUH

#include "flamegpu/flamegpu.h"
#include "../core/common.cuh"

namespace PDAC {

// ============================================================================
// EFF/CYT → SUPP costs the cell its remaining lifetime advantage.
// The ABM draws a CD8 lifespan once, at recruitment, from 1/k_CD8_death
// (PARAM_T_CELL_LIFE_MEAN_SLICE). The QSP removes EXHAUSTED CD8 at the separate,
// faster k_CD8_exh_death, so a cell that exhausts and keeps its original countdown
// lives k_CD8_exh_death/k_CD8_death times too long. Scaling the REMAINING life by
// the rate ratio gives the correct expected residual lifetime, which is what makes
// the ABM's SUPP pool comparable to SP_V_T_CD8_exh in the class audit.
// Applied at both exhaustion routes (hypoxic and PD-1/Treg); never reversed —
// SUPP is terminal in this model.
// ============================================================================
template<typename API>
__device__ __forceinline__ void scale_life_on_exhaustion(API FLAMEGPU) {
    const float s = FLAMEGPU->environment.template getProperty<float>("PARAM_CD8_EXH_LIFE_SCALE");
    const int life = FLAMEGPU->template getVariable<int>("life");
    const int scaled = static_cast<int>(life * s + 0.5f);
    FLAMEGPU->template setVariable<int>("life", scaled < 1 ? 1 : scaled);
}

// TCell agent function: Broadcast location
FLAMEGPU_AGENT_FUNCTION(tcell_broadcast_location, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const float voxel_size = FLAMEGPU->environment.getProperty<float>("voxel_size");

    FLAMEGPU->message_out.setVariable<int>("agent_type", CELL_TYPE_T);
    FLAMEGPU->message_out.setVariable<int>("agent_id", FLAMEGPU->getID());
    const int tcell_cs = FLAMEGPU->getVariable<int>("cell_state");
    FLAMEGPU->message_out.setVariable<int>("cell_state", tcell_cs);
    FLAMEGPU->message_out.setVariable<float>("PDL1", 0.0f);  // T cells don't have PDL1
    const float kf = (tcell_cs == T_CELL_NAIVE) ? 0.0f : 1.0f;
    FLAMEGPU->message_out.setVariable<float>("kill_factor", kf);
    FLAMEGPU->message_out.setVariable<int>("dead", 0);
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
    const int sc_slot = (tcell_cs == T_CELL_EFF)   ? SC_CD8_EFF :
                        (tcell_cs == T_CELL_CYT)    ? SC_CD8_CYT :
                        (tcell_cs == T_CELL_NAIVE)  ? SC_CD8_NAIVE : SC_CD8_SUP;
    atomicAdd(&sc[sc_slot], 1u);

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
    int dc_cdc1_mature_count = 0;

    // Adhesion: count all type+state neighbors
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
            all_count++;
            const int agent_type = msg.getVariable<int>("agent_type");
            const int agent_state = msg.getVariable<int>("cell_state");
            const float PDL1 = msg.getVariable<float>("PDL1");
            const float kill_factor = msg.getVariable<float>("kill_factor");

            // Adhesion: accumulate into type+state count vector
            adh_counts[msg_to_sc_idx(agent_type, agent_state, kill_factor)]++;

            if (PDL1 > max_PDL1) {
                max_PDL1 = PDL1;
            }

            // Agent-specific interaction counts
            if (agent_type == CELL_TYPE_CANCER) {
                cancer_count++;
                if (agent_state == CANCER_PROGENITOR) {
                    found_progenitor = 1;
                }
            } else if (agent_type == CELL_TYPE_TREG) {
                if (agent_state == TCD4_TREG) {
                    treg_count++;
                }
            } else if (agent_type == CELL_TYPE_DC) {
                // Detect mature cDC1 for naive CD8 priming
                if (agent_state == DC_MATURE) {
                    if (kill_factor < 0.5f) dc_cdc1_mature_count++;  // cDC1
                }
            }
        }
    }

    // Compute adhesion p_move from matrix
    const int my_sc = self_sc_idx(CELL_TYPE_T, FLAMEGPU->getVariable<int>("cell_state"));
    const float adh_pmove = compute_adhesion_pmove(my_sc, adh_counts, ADH_MATRIX_PTR(FLAMEGPU));

    FLAMEGPU->setVariable<int>("neighbor_cancer_count", cancer_count);
    FLAMEGPU->setVariable<int>("neighbor_Treg_count", treg_count);
    FLAMEGPU->setVariable<int>("neighbor_all_count", all_count);
    FLAMEGPU->setVariable<float>("max_neighbor_PDL1", max_PDL1);
    FLAMEGPU->setVariable<int>("found_progenitor", found_progenitor);
    FLAMEGPU->setVariable<int>("neighbor_dc_cdc1_mature_count", dc_cdc1_mature_count);
    FLAMEGPU->setVariable<float>("adh_p_move", adh_pmove);

    return flamegpu::ALIVE;
}

// TCell agent function: State transitions
FLAMEGPU_AGENT_FUNCTION(tcell_state_step, flamegpu::MessageNone, flamegpu::MessageNone) {
    if (FLAMEGPU->getVariable<int>("dead") == 1) {
        return flamegpu::DEAD;
    }

    // TODO: need coupling from QSP to get this
    float nivo = FLAMEGPU->environment.getProperty<float>("qsp_nivo_tumor");

    int cell_state = FLAMEGPU->getVariable<int>("cell_state");

    // Deterministic RNG for the state_step draws below (priming, lifespan, division/EFF->CYT,
    // exhaustion, wave). FLAMEGPU->random is nondeterministic for born/recruited cells; det_tag-
    // seeded streams (distinct salt per decision) are stable given lineage.
    const unsigned int sim_seed = FLAMEGPU->environment.getProperty<unsigned int>("sim_seed");
    const unsigned int det_tag  = FLAMEGPU->getVariable<unsigned int>("det_tag");
    const unsigned int cs_step  = FLAMEGPU->getStepCounter();

    // Life countdown
    int life = FLAMEGPU->getVariable<int>("life");
    life--;
    if (life <= 0) {
        FLAMEGPU->setVariable<int>("dead", 1);
        auto* evts = reinterpret_cast<unsigned int*>(FLAMEGPU->environment.getProperty<uint64_t>("event_counters_ptr"));
        const int ds = (cell_state == T_CELL_EFF)   ? EVT_DEATH_CD8_EFF :
                       (cell_state == T_CELL_CYT)    ? EVT_DEATH_CD8_CYT :
                       (cell_state == T_CELL_NAIVE)  ? EVT_DEATH_CD8_NAIVE : EVT_DEATH_CD8_SUP;
        atomicAdd(&evts[ds], 1u);
        return flamegpu::DEAD;
    }
    FLAMEGPU->setVariable<int>("life", life);

    // ── NAIVE state: DC priming only, skip all other logic ──
    if (cell_state == T_CELL_NAIVE) {
        const int dc_cdc1 = FLAMEGPU->getVariable<int>("neighbor_dc_cdc1_mature_count");
        if (dc_cdc1 > 0) {
            const float sec_per_slice = FLAMEGPU->environment.getProperty<float>("PARAM_SEC_PER_SLICE");
            const float n_sites = FLAMEGPU->environment.getProperty<float>("PARAM_DC_N_SITES");
            const int n_local = FLAMEGPU->getVariable<int>("neighbor_all_count");
            const float cell_p = FLAMEGPU->environment.getProperty<float>("PARAM_CELL");
            const float H = n_sites * dc_cdc1 / (n_sites * dc_cdc1 + n_local + cell_p);
            const float k = FLAMEGPU->environment.getProperty<float>("PARAM_DC_PRIME_K_CD8");
            const float p = 1.0f - expf(-k * H * sec_per_slice);
            unsigned int rs_prime = det_rng_seed(static_cast<int>(det_tag), cs_step, 0x7D01u, sim_seed);
            if (det_rng_uniform(rs_prime) < p) {
                // DC priming yields effector T cells (not cytotoxic). Cytotoxic
                // differentiation is the post-priming step, gated separately on
                // sustained progenitor contact via PARAM_TCELL_EFF_TO_CYT_RATE.
                FLAMEGPU->setVariable<int>("cell_state", T_CELL_EFF);
                const int N_prime = FLAMEGPU->environment.getProperty<int>("PARAM_PRIME_DIV_BURST");
                FLAMEGPU->setVariable<int>("divide_limit",
                    FLAMEGPU->getVariable<int>("divide_limit") + N_prime);
                FLAMEGPU->setVariable<int>("divide_cd", 0);
                FLAMEGPU->setVariable<int>("divide_flag", 1);
                // Reset IL2 release timer for newly primed cell
                FLAMEGPU->setVariable<float>("IL2_release_remain",
                    FLAMEGPU->environment.getProperty<float>("PARAM_TCELL_IL2_RELEASE_TIME"));
                // Extend lifespan: naive→CYT gets a fresh effector lifespan
                const float life_mean = FLAMEGPU->environment.getProperty<float>("PARAM_T_CELL_LIFE_MEAN_SLICE");
                const float life_sd = FLAMEGPU->environment.getProperty<float>("PARAM_TCELL_LIFESPAN_SD_SLICE");
                unsigned int rs_pl = det_rng_seed(static_cast<int>(det_tag), cs_step, 0x7D02u, sim_seed);
                const float u1 = det_rng_uniform(rs_pl), u2 = det_rng_uniform(rs_pl);
                const float nrm_pl = sqrtf(-2.0f * logf(u1 + 1e-7f)) * cosf(6.2831853f * u2);
                int new_life = static_cast<int>(life_mean + nrm_pl * life_sd + 0.5f);
                FLAMEGPU->setVariable<int>("life", new_life > 1 ? new_life : 1);
                auto* evts = reinterpret_cast<unsigned int*>(FLAMEGPU->environment.getProperty<uint64_t>("event_counters_ptr"));
                atomicAdd(&evts[EVT_PRIME_CD8], 1u);
            }
        }
        // NAIVE cells do nothing else — no killing, no exhaustion, no division, no chemokines
        return flamegpu::ALIVE;
    }

    //Get cell specific parameters
    const int found_progenitor = FLAMEGPU->getVariable<int>("found_progenitor");

    // Get environment parameters
    const float sec_per_slice = FLAMEGPU->environment.getProperty<float>("PARAM_SEC_PER_SLICE");
    // Update IL2 exposure — read directly from PDE
    const int nx_t = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny_t = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int ax_t = FLAMEGPU->getVariable<int>("x");
    const int ay_t = FLAMEGPU->getVariable<int>("y");
    const int az_t = FLAMEGPU->getVariable<int>("z");
    const int voxel_t = az_t * ny_t*nx_t + ay_t * nx_t + ax_t;

    float IL2 = PDE_READ(FLAMEGPU, PDE_CONC_IL2, voxel_t);
    float IL2_exposure = FLAMEGPU->getVariable<float>("IL2_exposure");
    IL2_exposure += IL2 * sec_per_slice;

    if (IL2_exposure > FLAMEGPU->environment.getProperty<float>("PARAM_TCELL_IL2_PROLIF_TH")) {
        FLAMEGPU->setVariable<int>("divide_flag", 1);
        FLAMEGPU->setVariable<float>("IL2_exposure", 0.0f);
    } else {
        FLAMEGPU->setVariable<float>("IL2_exposure", IL2_exposure);
    }

    // EFF -> CYT: probabilistic conversion when cancer progenitor in neighborhood.
    // Was deterministic (any contact → instant CYT) which over-converted; PDAC IMC
    // patients show ~0% cytotoxic. Rate parameterized by PARAM_TCELL_EFF_TO_CYT_RATE
    // so it can be tuned/swept; default ~1e-6/s gives ~12d timescale → ~2% per step.
    if (cell_state == T_CELL_EFF && found_progenitor == 1) {
        const float k_eff_cyt = FLAMEGPU->environment.getProperty<float>("PARAM_TCELL_EFF_TO_CYT_RATE");
        const float p_eff_cyt = 1.0f - expf(-k_eff_cyt * sec_per_slice);
        unsigned int rs_cyt = det_rng_seed(static_cast<int>(det_tag), cs_step, 0x7D04u, sim_seed);
        if (det_rng_uniform(rs_cyt) < p_eff_cyt) {
            cell_state = T_CELL_CYT;
            FLAMEGPU->setVariable<int>("cell_state", T_CELL_CYT);
            FLAMEGPU->setVariable<int>("divide_flag", 1);
        }
    }

    // CYT state: update release timers and check exhaustion
    if (cell_state == T_CELL_CYT) {
        // Decrement release timer for IL2
        float IL2_release_remain = FLAMEGPU->getVariable<float>("IL2_release_remain");
        if (IL2_release_remain > 0) {
            IL2_release_remain -= sec_per_slice;
            FLAMEGPU->setVariable<float>("IL2_release_remain", IL2_release_remain);
        }

        // Check for exhaustion — 4 independent pathways (matching ODE RF57/59/60/61)
        // Each pathway computes survival prob; combined as product
        const int neighbor_Treg = FLAMEGPU->getVariable<int>("neighbor_Treg_count");
        const int neighbor_all = FLAMEGPU->getVariable<int>("neighbor_all_count");
        const int neighbor_cancer = FLAMEGPU->getVariable<int>("neighbor_cancer_count");
        const float max_PDL1 = FLAMEGPU->getVariable<float>("max_neighbor_PDL1");
        const float param_cell = FLAMEGPU->environment.getProperty<float>("PARAM_CELL");

        float p_survive = 1.0f;

        // RF57 — Basal/antigen exhaustion: k_basal * C/(C+K_C_exh+cell)
        if (neighbor_cancer > 0) {
            const float K_C_exh = FLAMEGPU->environment.getProperty<float>("PARAM_K_C_EXH");
            float q_basal = static_cast<float>(neighbor_cancer)
                / (neighbor_cancer + K_C_exh + param_cell);
            p_survive *= powf(FLAMEGPU->environment.getProperty<float>("PARAM_EXHAUST_BASE_BASAL"), q_basal);
        }

        // RF60 — PD1-driven exhaustion: k_CD8 * C/(C+CD8+cell) * H_PD1
        if (neighbor_cancer > 0 && neighbor_all > 0) {
            float bond = get_PD1_PDL1(max_PDL1, nivo,
                        FLAMEGPU->environment.getProperty<float>("PARAM_PD1_SYN"),
                        FLAMEGPU->environment.getProperty<float>("PARAM_PDL1_K1"),
                        FLAMEGPU->environment.getProperty<float>("PARAM_PDL1_K2"),
                        FLAMEGPU->environment.getProperty<float>("PARAM_PDL1_K3"));
            float H_PD1 = hill_equation(bond,
                        FLAMEGPU->environment.getProperty<float>("PARAM_PD1_PDL1_HALF"),
                        FLAMEGPU->environment.getProperty<float>("PARAM_N_PD1_PDL1"));
            float q_pd1 = static_cast<float>(neighbor_cancer)
                / (neighbor_cancer + neighbor_all + param_cell) * H_PD1;
            p_survive *= powf(FLAMEGPU->environment.getProperty<float>("PARAM_EXHAUST_BASE_PD1"), q_pd1);
        }

        // RF59 — Treg exhaustion: k_Treg * Tregs/(CD8+Tregs+cell) * H_IL10
        if (neighbor_Treg > 0) {
            float IL10 = PDE_READ(FLAMEGPU, PDE_CONC_IL10, voxel_t);
            float IL10_ec50 = FLAMEGPU->environment.getProperty<float>("PARAM_EXHAUST_IL10_EC50");
            float H_IL10 = IL10 / (IL10 + IL10_ec50);
            float q_treg = static_cast<float>(neighbor_Treg)
                / (neighbor_all + param_cell) * H_IL10;
            p_survive *= powf(FLAMEGPU->environment.getProperty<float>("PARAM_EXHAUST_BASE_TREG"), q_treg);
        }

        // RF61 — Stiffness exhaustion: OMITTED — ABM handles spatially via ECM porosity
        // gating on movement, density-based speed reduction, and hypoxia from poor perfusion

        // Combined: roll against total exhaustion probability
        float p_exhaust = 1.0f - p_survive;
        unsigned int rs_exh = det_rng_seed(static_cast<int>(det_tag), cs_step, 0x7D05u, sim_seed);
        if (p_exhaust > 0.0f && det_rng_uniform(rs_exh) < p_exhaust) {
            FLAMEGPU->setVariable<int>("cell_state", T_CELL_SUPP);
            FLAMEGPU->setVariable<int>("divide_flag", 0);
            FLAMEGPU->setVariable<int>("divide_limit", 0);
            scale_life_on_exhaustion(FLAMEGPU);
        }
    }

    // Division cooldown
    int divide_cd = FLAMEGPU->getVariable<int>("divide_cd");
    if (divide_cd > 0) {
        divide_cd--;
        FLAMEGPU->setVariable<int>("divide_cd", divide_cd);
    }

    // === WAVE ASSIGNMENT ===
    {
        const int divide_flag_now = FLAMEGPU->getVariable<int>("divide_flag");
        const int divide_limit_now = FLAMEGPU->getVariable<int>("divide_limit");
        if (divide_flag_now == 1 && divide_cd <= 0 && divide_limit_now > 0) {
            unsigned int rs_wave = det_rng_seed(static_cast<int>(det_tag), cs_step, 0x7D06u, sim_seed);
            const int w = static_cast<int>(det_rng_uniform(rs_wave) * N_DIVIDE_WAVES);
            FLAMEGPU->setVariable<int>("divide_wave", w < N_DIVIDE_WAVES ? w : N_DIVIDE_WAVES - 1);
        }
    }

    return flamegpu::ALIVE;
}

// ============================================================
// Occupancy Grid Functions
// ============================================================

// Write this T cell's volume to the volume occupancy grid.
FLAMEGPU_AGENT_FUNCTION(tcell_write_to_occ_grid, flamegpu::MessageNone, flamegpu::MessageNone) {
    // Reset this step's move budget (batched-round movement; drained by tcell_move).
    FLAMEGPU->setVariable<int>("moves_remaining",
        FLAMEGPU->environment.getProperty<int>("PARAM_TCELL_MOVE_STEPS"));
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");

    const int gx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int gy = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int gz = FLAMEGPU->environment.getProperty<int>("grid_size_z");
    if (x < 0 || x >= gx || y < 0 || y >= gy || z < 0 || z >= gz) {
        return flamegpu::ALIVE;
    }
    const int vidx = z * (gx * gy) + y * gx + x;

    // Epsilon-kernel type count
    unsigned int* tc = TC_PTR(FLAMEGPU);
    const int n_voxels = gx * gy * gz;
    atomicAdd(&tc[OCC_TCELL * n_voxels + vidx], 1u);

    return flamegpu::ALIVE;
}

// T cell division — Step-5 deterministic reserve/confirm (mirrors cancer/fib). RESERVE picks one
// capacity-OK Moore neighbor (det_rng, not FLAMEGPU->random) and atomicMin-reserves it by
// (source-voxel, det_tag); CONFIRM commits the unique winner + births the daughter. Replaces the
// single-pass Fisher-Yates + volume_try_claim "first arrival wins" (thread-schedule dependent).
// Wave gate REMOVED — reserve/confirm resolves contention deterministically by priority, so this
// must be wired OUTSIDE the divide-wave loop (reset_owner -> reserve -> confirm), like cancer.
FLAMEGPU_AGENT_FUNCTION(tcell_divide_reserve, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->setVariable<int>("div_target_vidx", -1);  // reset reservation each step

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

    unsigned int* tc = TC_PTR(FLAMEGPU);
    const float* eps = EPS_PTR(FLAMEGPU);
    const int n_voxels = size_x * size_y * size_z;

    // Collect Moore (26-direction) neighbors with epsilon acceptance
    int cand_vidx[26];
    int n_cands = 0;
    for (int i = 0; i < 26; i++) {
        int dx, dy, dz;
        get_moore_direction(i, dx, dy, dz);
        const int nx = my_x + dx, ny = my_y + dy, nz = my_z + dz;
        if (!is_in_bounds(nx, ny, nz, size_x, size_y, size_z)) continue;
        const int nvidx = nz * (size_x * size_y) + ny * size_x + nx;
        if (epsilon_weight(tc, eps, OCC_TCELL, nvidx, n_voxels) > EPS_DIVIDE_THRESHOLD) cand_vidx[n_cands++] = nvidx;
    }
    if (n_cands == 0) return flamegpu::ALIVE;

    // Pick one candidate deterministically and reserve it by priority.
    const int my_vidx = my_z * (size_x * size_y) + my_y * size_x + my_x;
    const unsigned int sim_seed = FLAMEGPU->environment.getProperty<unsigned int>("sim_seed");
    const unsigned int det_tag  = FLAMEGPU->getVariable<unsigned int>("det_tag");
    unsigned int drs = det_rng_seed(static_cast<int>(det_tag), FLAMEGPU->getStepCounter(), 0x7D1Du, sim_seed);
    int pick = static_cast<int>(det_rng_uniform(drs) * n_cands);
    if (pick >= n_cands) pick = n_cands - 1;
    const int tvidx = cand_vidx[pick];
    unsigned long long* owner = reinterpret_cast<unsigned long long*>(
        FLAMEGPU->environment.getProperty<unsigned long long>("voxel_owner_ptr"));
    reserve_voxel(owner, tvidx, make_claim_priority(my_vidx, det_tag));
    FLAMEGPU->setVariable<int>("div_target_vidx", tvidx);
    return flamegpu::ALIVE;
}

// T cell division — CONFIRM pass. Commits the unique winner of each reserved voxel, then births
// the daughter (same variables as the original single-pass divide; daughter gets a hashed det_tag).
FLAMEGPU_AGENT_FUNCTION(tcell_divide_confirm, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int tvidx = FLAMEGPU->getVariable<int>("div_target_vidx");
    if (tvidx < 0) return flamegpu::ALIVE;

    const int cell_state = FLAMEGPU->getVariable<int>("cell_state");
    const int my_x  = FLAMEGPU->getVariable<int>("x");
    const int my_y  = FLAMEGPU->getVariable<int>("y");
    const int my_z  = FLAMEGPU->getVariable<int>("z");
    const int size_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int size_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");

    const unsigned int det_tag = FLAMEGPU->getVariable<unsigned int>("det_tag");
    const int my_vidx = my_z * (size_x * size_y) + my_y * size_x + my_x;
    unsigned long long* owner = reinterpret_cast<unsigned long long*>(
        FLAMEGPU->environment.getProperty<unsigned long long>("voxel_owner_ptr"));
    if (!won_voxel(owner, tvidx, make_claim_priority(my_vidx, det_tag))) {
        return flamegpu::ALIVE;  // lost the deterministic contention → don't divide this step
    }

    unsigned int* tc = TC_PTR(FLAMEGPU);
    const float* eps = EPS_PTR(FLAMEGPU);
    const int n_voxels = size_x * size_y * FLAMEGPU->environment.getProperty<int>("grid_size_z");
    if (epsilon_weight(tc, eps, OCC_TCELL, tvidx, n_voxels) <= EPS_DIVIDE_THRESHOLD) return flamegpu::ALIVE;
    type_count_claim(tc, OCC_TCELL, tvidx, n_voxels);

    const int divide_limit = FLAMEGPU->getVariable<int>("divide_limit");
    const int div_interval  = FLAMEGPU->environment.getProperty<int>("PARAM_TCELL_DIV_INTERNAL");
    const float IL2_release_time = FLAMEGPU->environment.getProperty<float>("PARAM_TCELL_IL2_RELEASE_TIME");
    const float tcell_life_mean  = FLAMEGPU->environment.getProperty<float>("PARAM_T_CELL_LIFE_MEAN_SLICE");
    const float tcell_life_sd    = FLAMEGPU->environment.getProperty<float>("PARAM_TCELL_LIFESPAN_SD_SLICE");
    const float IL2_exposure     = FLAMEGPU->getVariable<float>("IL2_exposure");

    const int target_x = tvidx % size_x;
    const int target_y = (tvidx / size_x) % size_y;
    const int target_z = tvidx / (size_x * size_y);

    // Daughter lifespan (deterministic normal via Box-Muller, not FLAMEGPU->random.normal)
    const unsigned int sim_seed = FLAMEGPU->environment.getProperty<unsigned int>("sim_seed");
    unsigned int crs = det_rng_seed(static_cast<int>(det_tag), FLAMEGPU->getStepCounter(), 0x7D2Du, sim_seed);
    const float u1 = det_rng_uniform(crs), u2 = det_rng_uniform(crs);
    const float nrm = sqrtf(-2.0f * logf(u1 + 1e-7f)) * cosf(6.2831853f * u2);
    int daughter_life = static_cast<int>(tcell_life_mean + nrm * tcell_life_sd + 0.5f);
    daughter_life = (daughter_life > 0) ? daughter_life : 1;

    FLAMEGPU->agent_out.setVariable<int>("x", target_x);
    FLAMEGPU->agent_out.setVariable<int>("y", target_y);
    FLAMEGPU->agent_out.setVariable<int>("z", target_z);
    FLAMEGPU->agent_out.setVariable<int>("cell_state", cell_state);
    FLAMEGPU->agent_out.setVariable<int>("divide_flag", 0);
    FLAMEGPU->agent_out.setVariable<int>("divide_cd", div_interval);
    FLAMEGPU->agent_out.setVariable<int>("divide_limit", divide_limit - 1);
    FLAMEGPU->agent_out.setVariable<float>("IL2_exposure", IL2_exposure);
    FLAMEGPU->agent_out.setVariable<float>("IL2_release_remain", IL2_release_time);
    FLAMEGPU->agent_out.setVariable<int>("life", daughter_life);
    FLAMEGPU->agent_out.setVariable<int>("persist_dir_x", 0);
    FLAMEGPU->agent_out.setVariable<int>("persist_dir_y", 0);
    FLAMEGPU->agent_out.setVariable<int>("persist_dir_z", 0);
    // Child lineage tag: deterministic + distinct per birth, independent of FLAMEGPU birth order.
    FLAMEGPU->agent_out.setVariable<unsigned int>("det_tag",
        det_rng_hash(det_tag ^ (static_cast<unsigned int>(tvidx) * 0x9E3779B1U)
                     ^ (static_cast<unsigned int>(FLAMEGPU->getStepCounter()) * 0x85EBCA77U)));

    // Update parent
    FLAMEGPU->setVariable<int>("divide_limit", divide_limit - 1);
    FLAMEGPU->setVariable<int>("divide_cd", div_interval);

    // Track T cell proliferation event
    {
        auto* evts = reinterpret_cast<unsigned int*>(FLAMEGPU->environment.getProperty<uint64_t>("event_counters_ptr"));
        const int ps = (cell_state == T_CELL_EFF) ? EVT_PROLIF_CD8_EFF :
                       (cell_state == T_CELL_CYT)  ? EVT_PROLIF_CD8_CYT : EVT_PROLIF_CD8_SUP;
        atomicAdd(&evts[ps], 1u);
    }

    // Clear parent intent
    FLAMEGPU->setVariable<int>("intent_action", INTENT_NONE);
    FLAMEGPU->setVariable<int>("target_x", -1);
    FLAMEGPU->setVariable<int>("target_y", -1);
    FLAMEGPU->setVariable<int>("target_z", -1);

    return flamegpu::ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(tcell_update_chemicals, flamegpu::MessageNone, flamegpu::MessageNone) {
    return flamegpu::ALIVE;
}

// T Cell agent function: Compute chemical sources
// atomicAdds directly to PDE source/uptake arrays
FLAMEGPU_AGENT_FUNCTION(tcell_compute_chemical_sources, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int dead = FLAMEGPU->getVariable<int>("dead");
    int cell_state = FLAMEGPU->getVariable<int>("cell_state");

    // O2 + TGF-β consumption (basal metabolism — all living T cells including naive)
    if (dead == 0) {
        const int nx0 = FLAMEGPU->environment.getProperty<int>("grid_size_x");
        const int ny0 = FLAMEGPU->environment.getProperty<int>("grid_size_y");
        const int voxel0 = FLAMEGPU->getVariable<int>("z") * ny0 * nx0
                         + FLAMEGPU->getVariable<int>("y") * nx0
                         + FLAMEGPU->getVariable<int>("x");
        const float O2_uptake = FLAMEGPU->environment.getProperty<float>("PARAM_O2_UPTAKE")
                              * FLAMEGPU->environment.getProperty<float>("PARAM_O2_NONTUMOR_UPTAKE_SCALAR");
        PDE_UPTAKE(FLAMEGPU, PDE_UPT_O2, voxel0, O2_uptake);
        const float TGFB_uptake = FLAMEGPU->environment.getProperty<float>("PARAM_TGFB_UPTAKE");
        PDE_UPTAKE(FLAMEGPU, PDE_UPT_TGFB, voxel0, TGFB_uptake);
    }

    float IFNg_release_rate = 0.0f;
    float IL2_release_rate = 0.0f;
    // Naive cells don't secrete any chemicals
    if (cell_state == T_CELL_NAIVE) return flamegpu::ALIVE;
    if (dead == 0){
        int found_progenitor = FLAMEGPU->getVariable<int>("found_progenitor");
        // EFF cells secrete IFN-γ only when in contact with cancer (matching HCC)
        // EFF cells do NOT secrete IL-2 — only CYT cells do, after activation
        if (cell_state == T_CELL_EFF && found_progenitor == 1){
            IFNg_release_rate = FLAMEGPU->environment.getProperty<float>("PARAM_IFNG_RELEASE");
        } else if (cell_state == T_CELL_CYT) {
            if (found_progenitor == 1){
                IFNg_release_rate = FLAMEGPU->environment.getProperty<float>("PARAM_IFNG_RELEASE");
            }
            if (FLAMEGPU->getVariable<float>("IL2_release_remain") > 0){
                IL2_release_rate = FLAMEGPU->environment.getProperty<float>("PARAM_IL2_RELEASE");
            }
        }

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

    // IFN-γ secretion → src ptr 1 (IFN), divide by voxel_volume
    PDE_SECRETE(FLAMEGPU, PDE_SRC_IFN, voxel, IFNg_release_rate / voxel_volume);

    // IL-2 secretion → src ptr 2 (IL2), divide by voxel_volume
    // No per-cell IL-2 uptake: PDE decay (k_IL2_deg) handles clearance;
    // ODE k_IL2_cons is LN-compartment consumption, not tumor tissue.
    PDE_SECRETE(FLAMEGPU, PDE_SRC_IL2, voxel, IL2_release_rate / voxel_volume);

    return flamegpu::ALIVE;
}

// Single-phase T cell movement using occupancy grid.
// Replaces two-phase select_move_target + execute_move.
// T cell movement uses epsilon-kernel occupancy (type_count_claim/type_count_release).
// T cell movement via unified movement framework.
// CCL5 chemotaxis toward tumor/iCAF regions (CCR5-mediated).
// T cell movement — Step-5 deterministic WAVE propose/commit (snapshot per wave). Mirrors
// mdsc_move_propose/commit: read-only walk of per_round steps (no claim) → reserve landing
// voxel by (source-voxel, det_tag) → commit the unique winner. Loser forfeits only this wave's
// per_round steps and retries next wave. CCL5 chemotaxis (CCR5) preserved.
FLAMEGPU_AGENT_FUNCTION(tcell_move_propose, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->setVariable<int>("move_target_vidx", -1);  // reset proposal each wave
    if (FLAMEGPU->getVariable<int>("dead") == 1) return flamegpu::ALIVE;

    int moves_remaining = FLAMEGPU->getVariable<int>("moves_remaining");
    if (moves_remaining <= 0) return flamegpu::ALIVE;
    const int move_K = FLAMEGPU->environment.getProperty<int>("MOVE_K");
    const int per_round = (FLAMEGPU->environment.getProperty<int>("PARAM_TCELL_MOVE_STEPS") + move_K - 1) / move_K;
    const int n_moves = min(per_round, moves_remaining);

    const int sx = FLAMEGPU->getVariable<int>("x");
    const int sy = FLAMEGPU->getVariable<int>("y");
    const int sz = FLAMEGPU->getVariable<int>("z");
    int x = sx, y = sy, z = sz;
    const int cell_state = FLAMEGPU->getVariable<int>("cell_state");

    float p_persist = (cell_state == T_CELL_EFF) ?
        FLAMEGPU->environment.getProperty<float>("PARAM_PERSIST_TCELL_EFF") :
        (cell_state == T_CELL_CYT) ?
        FLAMEGPU->environment.getProperty<float>("PARAM_PERSIST_TCELL_CYT") :
        FLAMEGPU->environment.getProperty<float>("PARAM_PERSIST_TCELL_SUP");

    MoveParams mp;
    mp.grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    mp.grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    mp.grid_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");
    mp.type_counts = TC_PTR(FLAMEGPU);
    mp.epsilon = EPS_PTR(FLAMEGPU);
    mp.n_voxels = mp.grid_x * mp.grid_y * mp.grid_z;
    mp.occ_class = OCC_TCELL;
    mp.ecm_density = ECM_DENSITY_PTR(FLAMEGPU);
    mp.ecm_crosslink = ECM_CROSSLINK_PTR(FLAMEGPU);
    mp.density_cap = FLAMEGPU->environment.getProperty<float>("PARAM_ECM_DENSITY_CAP");
    mp.min_porosity = FLAMEGPU->environment.getProperty<float>("PARAM_ECM_POROSITY_TCELL");
    // Adhesion: pre-computed from matrix in scan_neighbors
    mp.p_move = FLAMEGPU->getVariable<float>("adh_p_move");
    mp.p_persist = p_persist;

    // CCL5 chemotaxis: CD8+ T cells express CCR5, follow CCL5 toward tumor/iCAF
    float bias = ci_to_bias((cell_state == T_CELL_EFF) ?
        FLAMEGPU->environment.getProperty<float>("PARAM_CHEMO_CI_TCELL_EFF") :
        (cell_state == T_CELL_CYT) ?
        FLAMEGPU->environment.getProperty<float>("PARAM_CHEMO_CI_TCELL_CYT") :
        FLAMEGPU->environment.getProperty<float>("PARAM_CHEMO_CI_TCELL_SUP"));
    mp.bias_strength = bias;

    mp.orient_x = ECM_ORIENT_X_PTR(FLAMEGPU);
    mp.orient_y = ECM_ORIENT_Y_PTR(FLAMEGPU);
    mp.orient_z = ECM_ORIENT_Z_PTR(FLAMEGPU);
    mp.barrier_strength = FLAMEGPU->environment.getProperty<float>("PARAM_FIBER_BARRIER_TCELL");
    mp.pressure_grad_x = PRESSURE_GRAD_X_PTR(FLAMEGPU);
    mp.pressure_grad_y = PRESSURE_GRAD_Y_PTR(FLAMEGPU);
    mp.pressure_grad_z = PRESSURE_GRAD_Z_PTR(FLAMEGPU);
    mp.mechano_sensitivity = FLAMEGPU->environment.getProperty<float>("PARAM_MECHANO_TCELL");
    mp.debris = DEBRIS_PTR(FLAMEGPU);
    mp.debris_threshold = FLAMEGPU->environment.getProperty<float>("PARAM_DEBRIS_BLOCK_THRESHOLD");
    const float w_cg = FLAMEGPU->environment.getProperty<float>("PARAM_CONTACT_GUIDANCE_TCELL");

    const float* gax = reinterpret_cast<const float*>(FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_CCL5_X));
    const float* gay = reinterpret_cast<const float*>(FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_CCL5_Y));
    const float* gaz = reinterpret_cast<const float*>(FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_CCL5_Z));

    int pdx = FLAMEGPU->getVariable<int>("persist_dir_x");
    int pdy = FLAMEGPU->getVariable<int>("persist_dir_y");
    int pdz = FLAMEGPU->getVariable<int>("persist_dir_z");

    // Read-only snapshot walk (no claim) → wave landing voxel.
    run_move_batch_pick(FLAMEGPU, mp, n_moves, gax, gay, gaz, bias, w_cg, x, y, z, pdx, pdy, pdz);
    FLAMEGPU->setVariable<int>("moves_remaining", moves_remaining - n_moves);

    if (x != sx || y != sy || z != sz) {
        const int tvidx = z * (mp.grid_x * mp.grid_y) + y * mp.grid_x + x;
        const int my_vidx = sz * (mp.grid_x * mp.grid_y) + sy * mp.grid_x + sx;
        unsigned long long* owner = reinterpret_cast<unsigned long long*>(
            FLAMEGPU->environment.getProperty<unsigned long long>("voxel_owner_ptr"));
        reserve_voxel(owner, tvidx, make_claim_priority(my_vidx, FLAMEGPU->getVariable<unsigned int>("det_tag")));
        FLAMEGPU->setVariable<int>("move_target_vidx", tvidx);
    }
    return flamegpu::ALIVE;
}

// T cell movement — COMMIT pass. Unique winner moves (claim final, release source); losers stay.
FLAMEGPU_AGENT_FUNCTION(tcell_move_commit, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int tvidx = FLAMEGPU->getVariable<int>("move_target_vidx");
    if (tvidx < 0) return flamegpu::ALIVE;

    const int sx = FLAMEGPU->getVariable<int>("x");
    const int sy = FLAMEGPU->getVariable<int>("y");
    const int sz = FLAMEGPU->getVariable<int>("z");
    const int cell_state = FLAMEGPU->getVariable<int>("cell_state");
    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int my_vidx = sz * (grid_x * grid_y) + sy * grid_x + sx;
    const unsigned int det_tag = FLAMEGPU->getVariable<unsigned int>("det_tag");
    unsigned long long* owner = reinterpret_cast<unsigned long long*>(
        FLAMEGPU->environment.getProperty<unsigned long long>("voxel_owner_ptr"));
    if (!won_voxel(owner, tvidx, make_claim_priority(my_vidx, det_tag))) return flamegpu::ALIVE;

    unsigned int* tc = TC_PTR(FLAMEGPU);
    const int n_voxels = grid_x * grid_y * FLAMEGPU->environment.getProperty<int>("grid_size_z");
    type_count_claim(tc, OCC_TCELL, tvidx, n_voxels);
    type_count_release(tc, OCC_TCELL, my_vidx, n_voxels);

    const int tx = tvidx % grid_x;
    const int ty = (tvidx / grid_x) % grid_y;
    const int tz = tvidx / (grid_x * grid_y);
    FLAMEGPU->setVariable<int>("x", tx);
    FLAMEGPU->setVariable<int>("y", ty);
    FLAMEGPU->setVariable<int>("z", tz);
    FLAMEGPU->setVariable<int>("persist_dir_x", (tx > sx) - (tx < sx));
    FLAMEGPU->setVariable<int>("persist_dir_y", (ty > sy) - (ty < sy));
    FLAMEGPU->setVariable<int>("persist_dir_z", (tz > sz) - (tz < sz));
    return flamegpu::ALIVE;
}


// ── phi: CD8 is the KDE SOURCE for phi_1 (CD8xC1) and phi_3 (CD8xTreg). ──
// Uses is_qsp_CD8 so the population phi is measured on is byte-identical to the one the
// gap-closing audit compares against SP_V_T_CD8 — exhausted and naive CD8 are NOT CD8 here.
FLAMEGPU_AGENT_FUNCTION(tcell_phi_deposit, flamegpu::MessageNone, flamegpu::MessageNone) {
    if (is_qsp_CD8(FLAMEGPU->getVariable<int>("cell_state")))
        phi_deposit(FLAMEGPU, PHI_KDE_CD8);
    return flamegpu::ALIVE;
}

// ── phi: CD8 SAMPLES phi_6 (MDSC×CD8) and phi_7 (Bcell×CD8). ──
FLAMEGPU_AGENT_FUNCTION(tcell_phi_sample, flamegpu::MessageNone, flamegpu::MessageNone) {
    if (is_qsp_CD8(FLAMEGPU->getVariable<int>("cell_state"))) {
        phi_sample(FLAMEGPU, PHI_MDSC_CD8);
        phi_sample(FLAMEGPU, PHI_BCELL_CD8);
    }
    return flamegpu::ALIVE;
}

} // namespace PDAC

#endif // PDAC_T_CELL_CUH