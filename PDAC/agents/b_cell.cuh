// B Cell Agent Behavior Functions
// States: BCELL_NAIVE, BCELL_ACTIVATED, BCELL_PLASMA
// Naive → Activated: antigen capture + T cell help
// Activated → Plasma: time-dependent, accelerated in TLS clusters
// Plasma: secrete antibody (ADCC) or IL-10 (Breg)

#ifndef PDAC_B_CELL_CUH
#define PDAC_B_CELL_CUH

#include "flamegpu/flamegpu.h"
#include "../core/common.cuh"

namespace PDAC {

// ============================================================================
// BCell: Broadcast location (spatial messaging)
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(bcell_broadcast_location, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const float voxel_size = FLAMEGPU->environment.getProperty<float>("voxel_size");

    FLAMEGPU->message_out.setVariable<int>("agent_type", CELL_TYPE_BCELL);
    FLAMEGPU->message_out.setVariable<int>("agent_id", FLAMEGPU->getID());
    const int cs = FLAMEGPU->getVariable<int>("cell_state");
    FLAMEGPU->message_out.setVariable<int>("cell_state", cs);
    FLAMEGPU->message_out.setVariable<float>("PDL1", 0.0f);
    FLAMEGPU->message_out.setVariable<float>("kill_factor", 0.0f);
    FLAMEGPU->message_out.setVariable<int>("dead", 0);
    FLAMEGPU->message_out.setVariable<int>("voxel_x", x);
    FLAMEGPU->message_out.setVariable<int>("voxel_y", y);
    FLAMEGPU->message_out.setVariable<int>("voxel_z", z);

    FLAMEGPU->message_out.setLocation(
        (x + 0.5f) * voxel_size,
        (y + 0.5f) * voxel_size,
        (z + 0.5f) * voxel_size
    );

    // Per-state population snapshot
    auto* sc = reinterpret_cast<unsigned int*>(FLAMEGPU->environment.getProperty<uint64_t>("state_counters_ptr"));
    const int sc_slot = (cs == BCELL_NAIVE) ? SC_BCELL_NAIVE :
                        (cs == BCELL_ACTIVATED) ? SC_BCELL_ACT : SC_BCELL_PLASMA;
    atomicAdd(&sc[sc_slot], 1u);

    return flamegpu::ALIVE;
}

// ============================================================================
// BCell: Scan neighbors — antigen capture, T cell help, TLS clustering
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(bcell_scan_neighbors, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
    const int my_x = FLAMEGPU->getVariable<int>("x");
    const int my_y = FLAMEGPU->getVariable<int>("y");
    const int my_z = FLAMEGPU->getVariable<int>("z");
    const float voxel_size = FLAMEGPU->environment.getProperty<float>("voxel_size");

    const float my_pos_x = (my_x + 0.5f) * voxel_size;
    const float my_pos_y = (my_y + 0.5f) * voxel_size;
    const float my_pos_z = (my_z + 0.5f) * voxel_size;

    int cancer_count = 0;
    int th_count = 0;
    int bcell_count = 0;
    int fib_count = 0;

    for (const auto& msg : FLAMEGPU->message_in(my_pos_x, my_pos_y, my_pos_z)) {
        const int msg_x = msg.getVariable<int>("voxel_x");
        const int msg_y = msg.getVariable<int>("voxel_y");
        const int msg_z = msg.getVariable<int>("voxel_z");

        const int dx = msg_x - my_x;
        const int dy = msg_y - my_y;
        const int dz = msg_z - my_z;

        if (abs(dx) <= 1 && abs(dy) <= 1 && abs(dz) <= 1 && !(dx == 0 && dy == 0 && dz == 0)) {
            const int agent_type = msg.getVariable<int>("agent_type");

            if (agent_type == CELL_TYPE_CANCER) {
                cancer_count++;
            } else if (agent_type == CELL_TYPE_TREG) {
                // TH and Tfh cells provide T cell help for B cell activation
                int treg_state = msg.getVariable<int>("cell_state");
                if (treg_state == TCD4_TH || treg_state == TCD4_TFH) {
                    th_count++;
                }
            } else if (agent_type == CELL_TYPE_T) {
                // Effector T cells also provide help (simplified)
                th_count++;
            } else if (agent_type == CELL_TYPE_BCELL) {
                bcell_count++;
            } else if (agent_type == CELL_TYPE_FIB) {
                fib_count++;
            }
        }
    }

    FLAMEGPU->setVariable<int>("neighbor_cancer_count", cancer_count);
    FLAMEGPU->setVariable<int>("neighbor_th_count", th_count);
    FLAMEGPU->setVariable<int>("neighbor_bcell_count", bcell_count);
    FLAMEGPU->setVariable<int>("neighbor_fib_count", fib_count);

    return flamegpu::ALIVE;
}

// ============================================================================
// BCell: State step — state machine, antigen capture, division, death
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(bcell_state_step, flamegpu::MessageNone, flamegpu::MessageNone) {
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
        const int death_slot = (cs == BCELL_NAIVE) ? EVT_DEATH_BCELL_NAIVE :
                               (cs == BCELL_ACTIVATED) ? EVT_DEATH_BCELL_ACT : EVT_DEATH_BCELL_PLASMA;
        atomicAdd(&evts[death_slot], 1u);
        return flamegpu::DEAD;
    }
    FLAMEGPU->setVariable<int>("life", life);

    const int cs = FLAMEGPU->getVariable<int>("cell_state");
    int has_antigen = FLAMEGPU->getVariable<int>("has_antigen");
    const int th_count = FLAMEGPU->getVariable<int>("neighbor_th_count");
    const int bcell_count = FLAMEGPU->getVariable<int>("neighbor_bcell_count");

    // ── Antigen capture from persistent antigen grid ──
    if (has_antigen == 0) {
        const int nx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
        const int ny = FLAMEGPU->environment.getProperty<int>("grid_size_y");
        const int my_x = FLAMEGPU->getVariable<int>("x");
        const int my_y = FLAMEGPU->getVariable<int>("y");
        const int my_z = FLAMEGPU->getVariable<int>("z");
        const int voxel = my_z * ny * nx + my_y * nx + my_x;

        const float* antigen_grid = reinterpret_cast<const float*>(
            FLAMEGPU->environment.getProperty<uint64_t>("antigen_grid_ptr"));
        const float local_antigen = antigen_grid[voxel];
        const float capture_threshold = FLAMEGPU->environment.getProperty<float>("PARAM_ANTIGEN_CAPTURE_THRESHOLD");

        if (local_antigen > capture_threshold) {
            const float capture_prob = FLAMEGPU->environment.getProperty<float>("PARAM_BCELL_ANTIGEN_CAPTURE_PROB");
            float p_capture = capture_prob * local_antigen / (local_antigen + capture_threshold);
            if (FLAMEGPU->random.uniform<float>() < p_capture) {
                has_antigen = 1;
                FLAMEGPU->setVariable<int>("has_antigen", 1);
            }
        }
    }

    // ── State machine ──
    if (cs == BCELL_NAIVE) {
        // Naive → Activated: requires antigen + T cell help
        if (has_antigen == 1 && th_count > 0) {
            FLAMEGPU->setVariable<int>("cell_state", BCELL_ACTIVATED);
            FLAMEGPU->setVariable<int>("activation_timer", 0);

            // Roll Breg fate at activation
            const float breg_frac = FLAMEGPU->environment.getProperty<float>("PARAM_BCELL_BREG_FRACTION");
            if (FLAMEGPU->random.uniform<float>() < breg_frac) {
                FLAMEGPU->setVariable<int>("is_breg", 1);
            }

            // Initialize division capability
            FLAMEGPU->setVariable<int>("divide_cd", FLAMEGPU->environment.getProperty<int>("PARAM_BCELL_DIV_CD"));
            FLAMEGPU->setVariable<int>("divide_limit", FLAMEGPU->environment.getProperty<int>("PARAM_BCELL_DIV_LIMIT"));
        }
    }
    else if (cs == BCELL_ACTIVATED) {
        // Increment activation timer
        int timer = FLAMEGPU->getVariable<int>("activation_timer") + 1;
        FLAMEGPU->setVariable<int>("activation_timer", timer);

        // Compute differentiation threshold (TLS speedup)
        int base_timer = FLAMEGPU->environment.getProperty<int>("PARAM_BCELL_ACTIVATION_TIMER");
        int tls_th = FLAMEGPU->environment.getProperty<int>("PARAM_BCELL_TLS_THRESHOLD");
        float tls_speedup = FLAMEGPU->environment.getProperty<float>("PARAM_BCELL_TLS_SPEEDUP");

        float effective_threshold = (float)base_timer;
        if (bcell_count >= tls_th) {
            effective_threshold *= tls_speedup;  // Faster in TLS clusters
        }

        // Activated → Plasma: time-dependent differentiation
        if (timer >= (int)effective_threshold) {
            FLAMEGPU->setVariable<int>("cell_state", BCELL_PLASMA);
            FLAMEGPU->setVariable<int>("divide_flag", 0);
            FLAMEGPU->setVariable<int>("divide_cd", 0);
            FLAMEGPU->setVariable<int>("divide_limit", 0);
        } else {
            // Division logic for activated B cells
            int divide_cd = FLAMEGPU->getVariable<int>("divide_cd");
            int divide_limit = FLAMEGPU->getVariable<int>("divide_limit");
            if (divide_cd > 0) {
                divide_cd--;
                FLAMEGPU->setVariable<int>("divide_cd", divide_cd);
            }
            if (divide_cd <= 0 && divide_limit > 0 && FLAMEGPU->getVariable<int>("divide_flag") == 0) {
                FLAMEGPU->setVariable<int>("divide_flag", 1);
                const int w = static_cast<int>(FLAMEGPU->random.uniform<float>() * N_DIVIDE_WAVES);
                FLAMEGPU->setVariable<int>("divide_wave", w < N_DIVIDE_WAVES ? w : N_DIVIDE_WAVES - 1);
            }
        }
    }
    // BCELL_PLASMA: no state transitions, just lifespan countdown (handled above)

    return flamegpu::ALIVE;
}

// ============================================================================
// BCell: Write to occupancy grid (volume-based)
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(bcell_write_to_occ_grid, flamegpu::MessageNone, flamegpu::MessageNone) {
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
    float my_vol;
    if (cs == BCELL_NAIVE) my_vol = FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_BCELL_NAIVE");
    else if (cs == BCELL_ACTIVATED) my_vol = FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_BCELL_ACT");
    else my_vol = FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_BCELL_PLASMA");

    float* vol_used = VOL_PTR(FLAMEGPU);
    atomicAdd(&vol_used[vidx], my_vol);

    return flamegpu::ALIVE;
}

// ============================================================================
// BCell: Compute chemical sources
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(bcell_compute_chemical_sources, flamegpu::MessageNone, flamegpu::MessageNone) {
    if (FLAMEGPU->getVariable<int>("dead") == 1) return flamegpu::ALIVE;

    const int cs = FLAMEGPU->getVariable<int>("cell_state");
    const int is_breg = FLAMEGPU->getVariable<int>("is_breg");

    const int nx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int ax = FLAMEGPU->getVariable<int>("x");
    const int ay = FLAMEGPU->getVariable<int>("y");
    const int az = FLAMEGPU->getVariable<int>("z");
    const int voxel = az * ny * nx + ay * nx + ax;

    const float vs_cm = FLAMEGPU->environment.getProperty<float>("voxel_size") * 1.0e-4f;
    const float voxel_volume = vs_cm * vs_cm * vs_cm;

    if (cs == BCELL_PLASMA) {
        if (is_breg) {
            // Breg plasma: secrete IL-10 (immunosuppressive)
            float il10_rate = FLAMEGPU->environment.getProperty<float>("PARAM_BCELL_IL10_RELEASE");
            PDE_SECRETE(FLAMEGPU, PDE_SRC_IL10, voxel, il10_rate / voxel_volume);
        } else {
            // Normal plasma: secrete antibody (ADCC)
            float ab_rate = FLAMEGPU->environment.getProperty<float>("PARAM_BCELL_ANTIBODY_RELEASE");
            PDE_SECRETE(FLAMEGPU, PDE_SRC_ANTIBODY, voxel, ab_rate / voxel_volume);
        }
    }
    else if (cs == BCELL_ACTIVATED) {
        // Activated B cells secrete IL-6
        float il6_rate = FLAMEGPU->environment.getProperty<float>("PARAM_BCELL_IL6_RELEASE");
        PDE_SECRETE(FLAMEGPU, PDE_SRC_IL6, voxel, il6_rate / voxel_volume);

        // Breg activated cells also secrete IL-10
        if (is_breg) {
            float il10_rate = FLAMEGPU->environment.getProperty<float>("PARAM_BCELL_IL10_RELEASE");
            PDE_SECRETE(FLAMEGPU, PDE_SRC_IL10, voxel, il10_rate / voxel_volume);
        }
    }

    return flamegpu::ALIVE;
}

// ============================================================================
// BCell: Movement — CXCL13 chemotaxis (Naive/Activated), near-sessile (Plasma)
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(bcell_move, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const int cs = FLAMEGPU->getVariable<int>("cell_state");

    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int vidx = z * (grid_x * grid_y) + y * grid_x + x;

    // Read CXCL13 gradient
    const float gx = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_CXCL13_X))[vidx];
    const float gy = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_CXCL13_Y))[vidx];
    const float gz = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_CXCL13_Z))[vidx];

    // State-dependent volume
    float my_vol;
    if (cs == BCELL_NAIVE) my_vol = FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_BCELL_NAIVE");
    else if (cs == BCELL_ACTIVATED) my_vol = FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_BCELL_ACT");
    else my_vol = FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_BCELL_PLASMA");

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
    mp.min_porosity = FLAMEGPU->environment.getProperty<float>("PARAM_ECM_POROSITY_BCELL");
    mp.grad_x = gx; mp.grad_y = gy; mp.grad_z = gz;

    if (cs == BCELL_PLASMA) {
        // Near-sessile: very low movement probability
        mp.p_move = FLAMEGPU->environment.getProperty<float>("PARAM_BCELL_PLASMA_MOVE_PROB");
        mp.p_persist = 0.0f;
        mp.bias_strength = 0.0f;
    } else if (cs == BCELL_ACTIVATED) {
        // Reduced movement, some adhesion in clusters
        const int n_bcell = FLAMEGPU->getVariable<int>("neighbor_bcell_count");
        float adh_bcell = FLAMEGPU->environment.getProperty<float>("PARAM_ADH_BCELL_ACT_BCELL");
        float adh_ecm = FLAMEGPU->environment.getProperty<float>("PARAM_ADH_BCELL_ACT_ECM");
        float ecm_local = ECM_DENSITY_PTR(FLAMEGPU)[vidx];
        float ecm_th = FLAMEGPU->environment.getProperty<float>("PARAM_ADH_ECM_DENSITY_TH");
        float ecm_factor = (ecm_local > ecm_th) ? 1.0f : 0.0f;
        mp.p_move = fmaxf(0.0f, 1.0f - adh_bcell * (float)n_bcell / 26.0f - adh_ecm * ecm_factor);
        mp.p_persist = FLAMEGPU->environment.getProperty<float>("PARAM_PERSIST_BCELL_ACT");
        mp.bias_strength = FLAMEGPU->environment.getProperty<float>("PARAM_CHEMO_BIAS_BCELL_ACT");
    } else {
        // Naive: free-moving, CXCL13 chemotaxis
        mp.p_move = 1.0f;
        mp.p_persist = FLAMEGPU->environment.getProperty<float>("PARAM_PERSIST_BCELL_NAIVE");
        mp.bias_strength = FLAMEGPU->environment.getProperty<float>("PARAM_CHEMO_BIAS_BCELL_NAIVE");
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
    }

    return flamegpu::ALIVE;
}

// ============================================================================
// BCell: Division — Activated B cells only (clonal expansion)
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(bcell_divide, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int divide_flag = FLAMEGPU->getVariable<int>("divide_flag");
    if (divide_flag != 1) return flamegpu::ALIVE;

    // Wave gating
    if (FLAMEGPU->getVariable<int>("divide_wave") !=
        FLAMEGPU->environment.getProperty<int>("divide_current_wave")) {
        return flamegpu::ALIVE;
    }

    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int grid_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");
    float* vol_used = VOL_PTR(FLAMEGPU);
    const float capacity = FLAMEGPU->environment.getProperty<float>("PARAM_VOXEL_CAPACITY");
    const float my_vol = FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_BCELL_ACT");

    // Fisher-Yates shuffle 26 Moore neighbors for unbiased placement
    int offsets[26][3];
    int n = 0;
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                offsets[n][0] = dx; offsets[n][1] = dy; offsets[n][2] = dz;
                n++;
            }
        }
    }
    // Shuffle
    for (int i = n - 1; i > 0; i--) {
        int j = static_cast<int>(FLAMEGPU->random.uniform<float>() * (i + 1));
        if (j > i) j = i;
        int t0 = offsets[i][0]; offsets[i][0] = offsets[j][0]; offsets[j][0] = t0;
        int t1 = offsets[i][1]; offsets[i][1] = offsets[j][1]; offsets[j][1] = t1;
        int t2 = offsets[i][2]; offsets[i][2] = offsets[j][2]; offsets[j][2] = t2;
    }

    bool placed = false;
    for (int i = 0; i < n && !placed; i++) {
        int cx = x + offsets[i][0];
        int cy = y + offsets[i][1];
        int cz = z + offsets[i][2];
        if (cx < 0 || cx >= grid_x || cy < 0 || cy >= grid_y || cz < 0 || cz >= grid_z) continue;

        int vidx = cz * (grid_x * grid_y) + cy * grid_x + cx;
        float old_vol = atomicAdd(&vol_used[vidx], my_vol);
        if (old_vol + my_vol > capacity) {
            atomicAdd(&vol_used[vidx], -my_vol);
            continue;
        }

        // Create daughter cell
        FLAMEGPU->agent_out.setVariable<int>("x", cx);
        FLAMEGPU->agent_out.setVariable<int>("y", cy);
        FLAMEGPU->agent_out.setVariable<int>("z", cz);
        FLAMEGPU->agent_out.setVariable<int>("cell_state", BCELL_ACTIVATED);
        FLAMEGPU->agent_out.setVariable<int>("life", FLAMEGPU->getVariable<int>("life"));
        FLAMEGPU->agent_out.setVariable<int>("dead", 0);
        FLAMEGPU->agent_out.setVariable<int>("has_antigen", 1);  // Inherited from parent
        FLAMEGPU->agent_out.setVariable<int>("is_breg", FLAMEGPU->getVariable<int>("is_breg"));
        FLAMEGPU->agent_out.setVariable<int>("activation_timer", FLAMEGPU->getVariable<int>("activation_timer"));
        int new_limit = FLAMEGPU->getVariable<int>("divide_limit") - 1;
        int div_cd = FLAMEGPU->environment.getProperty<int>("PARAM_BCELL_DIV_CD");
        FLAMEGPU->agent_out.setVariable<int>("divide_flag", 0);
        FLAMEGPU->agent_out.setVariable<int>("divide_cd", div_cd);
        FLAMEGPU->agent_out.setVariable<int>("divide_limit", new_limit);
        FLAMEGPU->agent_out.setVariable<int>("divide_wave", 0);
        FLAMEGPU->agent_out.setVariable<int>("persist_dir_x", 0);
        FLAMEGPU->agent_out.setVariable<int>("persist_dir_y", 0);
        FLAMEGPU->agent_out.setVariable<int>("persist_dir_z", 0);
        FLAMEGPU->agent_out.setVariable<int>("neighbor_cancer_count", 0);
        FLAMEGPU->agent_out.setVariable<int>("neighbor_th_count", 0);
        FLAMEGPU->agent_out.setVariable<int>("neighbor_bcell_count", 0);
        FLAMEGPU->agent_out.setVariable<int>("neighbor_fib_count", 0);

        // Update parent
        FLAMEGPU->setVariable<int>("divide_flag", 0);
        FLAMEGPU->setVariable<int>("divide_cd", div_cd);
        FLAMEGPU->setVariable<int>("divide_limit", new_limit);

        auto* evts = reinterpret_cast<unsigned int*>(FLAMEGPU->environment.getProperty<uint64_t>("event_counters_ptr"));
        atomicAdd(&evts[EVT_PROLIF_BCELL_ACT], 1u);

        placed = true;
    }

    if (!placed) {
        FLAMEGPU->setVariable<int>("divide_flag", 0);
    }

    return flamegpu::ALIVE;
}

} // namespace PDAC

#endif // PDAC_B_CELL_CUH
