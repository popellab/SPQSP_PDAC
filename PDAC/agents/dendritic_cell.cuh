// Dendritic Cell Agent Behavior Functions
// States: DC_IMMATURE, DC_MATURE
// Subtypes: DC_CDC1 (cross-present→CD8, IL-12), DC_CDC2 (MHC-II→CD4/Treg)
// Immature: homeostatic recruitment, probabilistic maturation via local antigen + IL-12
//           suppressed by IL-10 and TGF-β (ODE: k_APC_mat * H_signal * (1-H_IL10) * (1-H_TGFb))
// Mature: secretes IL-12 (cDC1 only) + CCL21, presents to T/B cells, then dies
// No division. Consumable APC.

#ifndef PDAC_DENDRITIC_CELL_CUH
#define PDAC_DENDRITIC_CELL_CUH

#include "flamegpu/flamegpu.h"
#include "../core/common.cuh"

namespace PDAC {

// ============================================================================
// DC: Broadcast location (spatial messaging)
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(dc_broadcast_location, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const float voxel_size = FLAMEGPU->environment.getProperty<float>("voxel_size");

    FLAMEGPU->message_out.setVariable<int>("agent_type", CELL_TYPE_DC);
    FLAMEGPU->message_out.setVariable<int>("agent_id", FLAMEGPU->getID());
    const int cs = FLAMEGPU->getVariable<int>("cell_state");
    FLAMEGPU->message_out.setVariable<int>("cell_state", cs);
    FLAMEGPU->message_out.setVariable<float>("PDL1", 0.0f);
    // Encode DC subtype in kill_factor: 0.0=cDC1, 1.0=cDC2 (read by T cells for priming)
    FLAMEGPU->message_out.setVariable<float>("kill_factor", (float)FLAMEGPU->getVariable<int>("dc_subtype"));
    FLAMEGPU->message_out.setVariable<int>("dead", 0);
    FLAMEGPU->message_out.setVariable<int>("voxel_x", x);
    FLAMEGPU->message_out.setVariable<int>("voxel_y", y);
    FLAMEGPU->message_out.setVariable<int>("voxel_z", z);

    FLAMEGPU->message_out.setLocation(
        (x + 0.5f) * voxel_size,
        (y + 0.5f) * voxel_size,
        (z + 0.5f) * voxel_size
    );

    // Per-state + subtype population snapshot
    auto* sc = reinterpret_cast<unsigned int*>(FLAMEGPU->environment.getProperty<uint64_t>("state_counters_ptr"));
    const int subtype = FLAMEGPU->getVariable<int>("dc_subtype");
    int sc_slot;
    if (subtype == DC_CDC1) {
        sc_slot = (cs == DC_IMMATURE) ? SC_DC_CDC1_IMMATURE : SC_DC_CDC1_MATURE;
    } else {
        sc_slot = (cs == DC_IMMATURE) ? SC_DC_CDC2_IMMATURE : SC_DC_CDC2_MATURE;
    }
    atomicAdd(&sc[sc_slot], 1u);

    return flamegpu::ALIVE;
}

// ============================================================================
// DC: Scan neighbors — T cells, B cells (for presentation capacity tracking)
// Antigen capture now uses persistent antigen grid (read in state_step).
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(dc_scan_neighbors, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
    const int my_x = FLAMEGPU->getVariable<int>("x");
    const int my_y = FLAMEGPU->getVariable<int>("y");
    const int my_z = FLAMEGPU->getVariable<int>("z");
    const float voxel_size = FLAMEGPU->environment.getProperty<float>("voxel_size");

    const float my_pos_x = (my_x + 0.5f) * voxel_size;
    const float my_pos_y = (my_y + 0.5f) * voxel_size;
    const float my_pos_z = (my_z + 0.5f) * voxel_size;

    int tcell_count = 0;
    int bcell_count = 0;

    // Adhesion: count all type+state neighbors
    int adh_counts[ABM_STATE_COUNTER_SIZE] = {0};

    for (const auto& msg : FLAMEGPU->message_in(my_pos_x, my_pos_y, my_pos_z)) {
        const int msg_x = msg.getVariable<int>("voxel_x");
        const int msg_y = msg.getVariable<int>("voxel_y");
        const int msg_z = msg.getVariable<int>("voxel_z");

        const int dx = msg_x - my_x;
        const int dy = msg_y - my_y;
        const int dz = msg_z - my_z;

        if (abs(dx) <= 1 && abs(dy) <= 1 && abs(dz) <= 1 && !(dx == 0 && dy == 0 && dz == 0)) {
            const int agent_type = msg.getVariable<int>("agent_type");
            const int agent_state = msg.getVariable<int>("cell_state");
            const float kill_factor = msg.getVariable<float>("kill_factor");

            // Adhesion: accumulate into type+state count vector
            adh_counts[msg_to_sc_idx(agent_type, agent_state, kill_factor)]++;

            if (agent_type == CELL_TYPE_T) {
                tcell_count++;
            } else if (agent_type == CELL_TYPE_BCELL) {
                bcell_count++;
            }
        }
    }

    // Compute adhesion p_move from matrix
    const int my_sc = dc_self_sc_idx(FLAMEGPU->getVariable<int>("cell_state"),
                                     FLAMEGPU->getVariable<int>("dc_subtype"));
    const float adh_pmove = compute_adhesion_pmove(my_sc, adh_counts, ADH_MATRIX_PTR(FLAMEGPU));

    FLAMEGPU->setVariable<int>("neighbor_tcell_count", tcell_count);
    FLAMEGPU->setVariable<int>("neighbor_bcell_count", bcell_count);
    FLAMEGPU->setVariable<float>("adh_p_move", adh_pmove);

    return flamegpu::ALIVE;
}

// ============================================================================
// DC: State step — antigen capture, maturation, presentation, death
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(dc_state_step, flamegpu::MessageNone, flamegpu::MessageNone) {
    if (FLAMEGPU->getVariable<int>("dead") == 1) {
        return flamegpu::DEAD;
    }

    auto* evts = reinterpret_cast<unsigned int*>(FLAMEGPU->environment.getProperty<uint64_t>("event_counters_ptr"));

    // Life countdown
    int life = FLAMEGPU->getVariable<int>("life");
    life--;
    if (life <= 0) {
        FLAMEGPU->setVariable<int>("dead", 1);
        const int cs = FLAMEGPU->getVariable<int>("cell_state");
        const int subtype = FLAMEGPU->getVariable<int>("dc_subtype");
        int death_slot;
        if (subtype == DC_CDC1) {
            death_slot = (cs == DC_IMMATURE) ? EVT_DEATH_DC_CDC1_IMMATURE : EVT_DEATH_DC_CDC1_MATURE;
        } else {
            death_slot = (cs == DC_IMMATURE) ? EVT_DEATH_DC_CDC2_IMMATURE : EVT_DEATH_DC_CDC2_MATURE;
        }
        atomicAdd(&evts[death_slot], 1u);
        return flamegpu::DEAD;
    }
    FLAMEGPU->setVariable<int>("life", life);

    const int cs = FLAMEGPU->getVariable<int>("cell_state");
    const int subtype = FLAMEGPU->getVariable<int>("dc_subtype");

    if (cs == DC_IMMATURE) {
        // Maturation: probabilistic per-step, driven by local antigen + cytokine milieu
        // ODE: k_APC_mat * cDC * [1-(1-H_DAMP)*(1-H_IL12)] * (1-H_IL10) * (1-H_TGFb_APC)
        // ABM: antigen grid serves as spatial DAMP analog

        const int nx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
        const int ny = FLAMEGPU->environment.getProperty<int>("grid_size_y");
        const int nz = FLAMEGPU->environment.getProperty<int>("grid_size_z");
        const int my_x = FLAMEGPU->getVariable<int>("x");
        const int my_y = FLAMEGPU->getVariable<int>("y");
        const int my_z = FLAMEGPU->getVariable<int>("z");
        const int voxel = my_z * ny * nx + my_y * nx + my_x;

        // Read local antigen as Moore-neighborhood mean (self + 26 neighbors)
        // so co-voxel placement isn't required for detection.
        const float* antigen_grid = reinterpret_cast<const float*>(
            FLAMEGPU->environment.getProperty<uint64_t>("antigen_grid_ptr"));
        const float local_antigen = read_grid_moore_avg(antigen_grid,
            my_x, my_y, my_z, nx, ny, nz);
        const float antigen_50 = FLAMEGPU->environment.getProperty<float>("PARAM_DC_ANTIGEN_50");

        // Read local cytokine concentrations [nM]
        const float local_il12 = PDE_READ(FLAMEGPU, PDE_CONC_IL12, voxel);
        const float local_il10 = PDE_READ(FLAMEGPU, PDE_CONC_IL10, voxel);
        const float local_tgfb = PDE_READ(FLAMEGPU, PDE_CONC_TGFB, voxel);

        // Hill functions
        const float H_antigen = local_antigen / (local_antigen + antigen_50 + 1e-30f);
        const float il12_50 = FLAMEGPU->environment.getProperty<float>("PARAM_DC_IL12_50");
        const float H_IL12 = local_il12 / (local_il12 + il12_50 + 1e-30f);
        const float il10_50 = FLAMEGPU->environment.getProperty<float>("PARAM_DC_IL10_50");
        const float H_IL10 = local_il10 / (local_il10 + il10_50 + 1e-30f);
        const float tgfb_50 = FLAMEGPU->environment.getProperty<float>("PARAM_DC_TGFB_50");
        const float H_TGFb = local_tgfb / (local_tgfb + tgfb_50 + 1e-30f);

        // Antibody-antigen immune complexes enhance DC cross-presentation via Fc receptors
        const float local_ab = PDE_READ(FLAMEGPU, PDE_CONC_ANTIBODY, voxel);
        const float ab_50 = FLAMEGPU->environment.getProperty<float>("PARAM_DC_ANTIBODY_50");
        const float H_ab = local_ab / (local_ab + ab_50 + 1e-30f);
        const float ab_boost = FLAMEGPU->environment.getProperty<float>("PARAM_DC_ANTIBODY_BOOST");
        // Effective antigen = antigen * (1 + boost * H_ab): immune complexes improve uptake
        const float H_antigen_eff = fminf(H_antigen * (1.0f + ab_boost * H_ab), 1.0f);

        // Combined activation signal: either antigen or IL-12 (or both) drive maturation
        const float H_signal = 1.0f - (1.0f - H_antigen_eff) * (1.0f - H_IL12);

        // Per-subtype maturation rate [prob/step]
        const float k_mat = (subtype == DC_CDC1)
            ? FLAMEGPU->environment.getProperty<float>("PARAM_DC_K_MAT_CDC1")
            : FLAMEGPU->environment.getProperty<float>("PARAM_DC_K_MAT_CDC2");

        // Maturation probability: rate × signal × suppression
        const float p_mature = k_mat * H_signal * (1.0f - H_IL10) * (1.0f - H_TGFb);

        if (p_mature > 0.0f && FLAMEGPU->random.uniform<float>() < p_mature) {
            FLAMEGPU->setVariable<int>("cell_state", DC_MATURE);

            // Reset life to mature lifespan
            const float mature_life_mean = FLAMEGPU->environment.getProperty<float>("PARAM_DC_LIFE_MATURE_MEAN");
            const float mature_life_sd = FLAMEGPU->environment.getProperty<float>("PARAM_DC_LIFE_MATURE_SD");
            float u1 = FLAMEGPU->random.uniform<float>() + 1e-10f;
            float u2 = FLAMEGPU->random.uniform<float>();
            float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
            int mature_life = __float2int_rn(mature_life_mean + z * mature_life_sd);
            if (mature_life < 1) mature_life = 1;
            FLAMEGPU->setVariable<int>("life", mature_life);

            // Set presentation capacity
            const int pres_cap = FLAMEGPU->environment.getProperty<int>("PARAM_DC_PRESENTATION_CAPACITY");
            FLAMEGPU->setVariable<int>("presentation_capacity", pres_cap);
        }
    }
    else if (cs == DC_MATURE) {
        // Presentation: each neighboring T/B cell contact costs 1 capacity
        int pres_cap = FLAMEGPU->getVariable<int>("presentation_capacity");
        const int tcell_count = FLAMEGPU->getVariable<int>("neighbor_tcell_count");
        const int bcell_count = FLAMEGPU->getVariable<int>("neighbor_bcell_count");
        int contacts = tcell_count + bcell_count;
        pres_cap -= contacts;
        if (pres_cap < 0) pres_cap = 0;
        FLAMEGPU->setVariable<int>("presentation_capacity", pres_cap);

        // Exhaustion death: presentation capacity depleted
        if (pres_cap <= 0) {
            FLAMEGPU->setVariable<int>("dead", 1);
            const int exhaust_slot = (subtype == DC_CDC1) ? EVT_DEATH_DC_CDC1_MATURE : EVT_DEATH_DC_CDC2_MATURE;
            atomicAdd(&evts[exhaust_slot], 1u);
            return flamegpu::DEAD;
        }
    }

    return flamegpu::ALIVE;
}

// ============================================================================
// DC: Write to occupancy grid (volume-based)
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(dc_write_to_occ_grid, flamegpu::MessageNone, flamegpu::MessageNone) {
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

    const int cs = FLAMEGPU->getVariable<int>("cell_state");
    float my_vol = (cs == DC_IMMATURE)
        ? FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_DC_IMMATURE")
        : FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_DC_MATURE");

    float* vol_used = VOL_PTR(FLAMEGPU);
    atomicAdd(&vol_used[vidx], my_vol);

    return flamegpu::ALIVE;
}

// ============================================================================
// DC: Compute chemical sources — Mature: secrete IL-12 + CCL21
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(dc_compute_chemical_sources, flamegpu::MessageNone, flamegpu::MessageNone) {
    if (FLAMEGPU->getVariable<int>("dead") == 1) return flamegpu::ALIVE;

    const int cs = FLAMEGPU->getVariable<int>("cell_state");
    if (cs != DC_MATURE) return flamegpu::ALIVE;

    const int nx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int ax = FLAMEGPU->getVariable<int>("x");
    const int ay = FLAMEGPU->getVariable<int>("y");
    const int az = FLAMEGPU->getVariable<int>("z");
    const int voxel = az * ny * nx + ay * nx + ax;

    const float vs_cm = FLAMEGPU->environment.getProperty<float>("voxel_size") * 1.0e-4f;
    const float voxel_volume = vs_cm * vs_cm * vs_cm;

    const int subtype = FLAMEGPU->getVariable<int>("dc_subtype");

    // cDC1 (mature): secrete IL-12 (pro-inflammatory, CD8 T cell activation)
    // ODE: k_IL12_sec * mcDC1 — only cDC1 produces IL-12
    if (subtype == DC_CDC1) {
        float il12_rate = FLAMEGPU->environment.getProperty<float>("PARAM_DC_IL12_RELEASE");
        PDE_SECRETE(FLAMEGPU, PDE_SRC_IL12, voxel, il12_rate / voxel_volume);
    }

    // CCL21 is produced by FRCs (fibroblastic reticular cells), not DCs.
    // Mature DCs still chemotax along the CCL21 gradient toward TLS T-zones, but they
    // do not establish the gradient. See FIB_FRC branch in fib_compute_chemical_sources.

    return flamegpu::ALIVE;
}

// ============================================================================
// DC: Movement — Immature: CCL2 chemotaxis; Mature: CCL21 chemotaxis
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(dc_move, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const int cs = FLAMEGPU->getVariable<int>("cell_state");

    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int vidx = z * (grid_x * grid_y) + y * grid_x + x;

    // State-dependent gradient: Immature→CCL2, Mature→CCL21
    float gx_val, gy_val, gz_val;
    if (cs == DC_IMMATURE) {
        gx_val = reinterpret_cast<const float*>(
            FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_CCL2_X))[vidx];
        gy_val = reinterpret_cast<const float*>(
            FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_CCL2_Y))[vidx];
        gz_val = reinterpret_cast<const float*>(
            FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_CCL2_Z))[vidx];
    } else {
        gx_val = reinterpret_cast<const float*>(
            FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_CCL21_X))[vidx];
        gy_val = reinterpret_cast<const float*>(
            FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_CCL21_Y))[vidx];
        gz_val = reinterpret_cast<const float*>(
            FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_CCL21_Z))[vidx];
    }

    // State-dependent volume
    float my_vol = (cs == DC_IMMATURE)
        ? FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_DC_IMMATURE")
        : FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_DC_MATURE");

    MoveParams mp;
    mp.grid_x = grid_x;
    mp.grid_y = grid_y;
    mp.grid_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");
    mp.vol_used = VOL_PTR(FLAMEGPU);
    mp.my_vol = my_vol;
    mp.capacity = FLAMEGPU->environment.getProperty<float>("PARAM_VOXEL_CAPACITY");
    mp.ecm_density = ECM_DENSITY_PTR(FLAMEGPU);
    mp.ecm_crosslink = ECM_CROSSLINK_PTR(FLAMEGPU);
    mp.density_cap = FLAMEGPU->environment.getProperty<float>("PARAM_ECM_DENSITY_CAP");
    mp.min_porosity = FLAMEGPU->environment.getProperty<float>("PARAM_ECM_POROSITY_DC");
    mp.grad_x = gx_val; mp.grad_y = gy_val; mp.grad_z = gz_val;

    // Adhesion: pre-computed from matrix in scan_neighbors
    mp.p_move = FLAMEGPU->getVariable<float>("adh_p_move");
    if (cs == DC_IMMATURE) {
        mp.p_persist = FLAMEGPU->environment.getProperty<float>("PARAM_PERSIST_DC_IMMATURE");
        mp.bias_strength = ci_to_bias(FLAMEGPU->environment.getProperty<float>("PARAM_CHEMO_CI_DC_IMMATURE"));
    } else {
        mp.p_persist = FLAMEGPU->environment.getProperty<float>("PARAM_PERSIST_DC_MATURE");
        mp.bias_strength = ci_to_bias(FLAMEGPU->environment.getProperty<float>("PARAM_CHEMO_CI_DC_MATURE"));
    }
    mp.orient_x = ECM_ORIENT_X_PTR(FLAMEGPU);
    mp.orient_y = ECM_ORIENT_Y_PTR(FLAMEGPU);
    mp.orient_z = ECM_ORIENT_Z_PTR(FLAMEGPU);
    mp.barrier_strength = FLAMEGPU->environment.getProperty<float>("PARAM_FIBER_BARRIER_DC");

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

#endif // PDAC_DENDRITIC_CELL_CUH
