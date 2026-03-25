// Fibroblast / Cancer-Associated Fibroblast (CAF) Agent Behavior Functions
// States: FIB_NORMAL (quiescent fibroblast), FIB_CAF (activated CAF)
// Activation: TGFB-driven NORMAL->CAF transition
//
// Multi-voxel chain model: one agent occupies 3 voxels (NORMAL) or 5 (CAF).
//   seg_x/y/z[0] = head (chemotaxis via TGF-β run-tumble)
//   seg_x/y/z[chain_len-1] = tail
// Movement: head moves, segments shift (each takes previous position), tail releases.
//   All done atomically within a single agent function — no multi-pass synchronization.
// Activation: NORMAL→CAF extends chain from 3→5 by finding 2 free Von Neumann
//   neighbors off the tail. Both must be found or activation is skipped.

#ifndef FIBROBLAST_CUH
#define FIBROBLAST_CUH

#include "flamegpu/flamegpu.h"
#include "../core/common.cuh"

namespace PDAC {

// ============================================================================
// Fibroblast: Broadcast location (spatial messaging) — broadcasts head position
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(fib_broadcast_location, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
    const float voxel_size = FLAMEGPU->environment.getProperty<float>("voxel_size");
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const float fx = (x + 0.5f) * voxel_size;
    const float fy = (y + 0.5f) * voxel_size;
    const float fz = (z + 0.5f) * voxel_size;

    FLAMEGPU->message_out.setVariable<int>("agent_type", CELL_TYPE_FIB);
    FLAMEGPU->message_out.setVariable<unsigned int>("agent_id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setVariable<int>("x", x);
    FLAMEGPU->message_out.setVariable<int>("y", y);
    FLAMEGPU->message_out.setVariable<int>("z", z);
    FLAMEGPU->message_out.setLocation(fx, fy, fz);

    // Count chain segments (not just agents) into per-state population snapshot
    const int fib_cs  = FLAMEGPU->getVariable<int>("cell_state");
    const int clen    = FLAMEGPU->getVariable<int>("chain_len");
    auto* sc_fib = reinterpret_cast<unsigned int*>(FLAMEGPU->environment.getProperty<uint64_t>("state_counters_ptr"));
    atomicAdd(&sc_fib[fib_cs == FIB_NORMAL ? SC_FIB_NORM : SC_FIB_CAF], static_cast<unsigned int>(clen));

    return flamegpu::ALIVE;
}

// ============================================================================
// Fibroblast: Write all chain segments to occupancy grid
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(fib_write_to_occ_grid, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int chain_len = FLAMEGPU->getVariable<int>("chain_len");
    auto occ = FLAMEGPU->environment.getMacroProperty<unsigned int,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX, NUM_OCC_TYPES>("occ_grid");

    for (int i = 0; i < chain_len; i++) {
        const int sx = FLAMEGPU->getVariable<int, MAX_FIB_CHAIN_LENGTH>("seg_x", i);
        const int sy = FLAMEGPU->getVariable<int, MAX_FIB_CHAIN_LENGTH>("seg_y", i);
        const int sz = FLAMEGPU->getVariable<int, MAX_FIB_CHAIN_LENGTH>("seg_z", i);
        occ[sx][sy][sz][CELL_TYPE_FIB].exchange(1u);
    }

    return flamegpu::ALIVE;
}

// ============================================================================
// Fibroblast: Compute chemical sources (no-op — fibroblasts don't secrete)
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(fib_compute_chemical_sources, flamegpu::MessageNone, flamegpu::MessageNone) {
    return flamegpu::ALIVE;
}

// ============================================================================
// Fibroblast: Unified movement — head does TGF-β chemotaxis, chain shifts
// Head (seg 0) moves via run-tumble. On success: tail releases, segments shift,
// head claims new voxel. All in one pass — no cross-agent synchronization.
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(fib_move, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int chain_len = FLAMEGPU->getVariable<int>("chain_len");
    const int x = FLAMEGPU->getVariable<int>("x");  // head position
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int grid_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");
    const int tumble = FLAMEGPU->getVariable<int>("tumble");

    // ECM based movement probability: higher ECM → more likely to be blocked
    {
        const float* ecm_ptr = reinterpret_cast<const float*>(
            FLAMEGPU->environment.getProperty<uint64_t>("ecm_grid_ptr"));
        float ECM_density = ecm_ptr[z * (grid_x * grid_y) + y * grid_x + x];
        float ECM_50 = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ECM_MOT_EC50");
        float ECM_sat = ECM_density / (ECM_density + ECM_50);
        if (FLAMEGPU->random.uniform<float>() < ECM_sat) return flamegpu::ALIVE;
    }

    const float move_dir_x = FLAMEGPU->getVariable<float>("move_direction_x");
    const float move_dir_y = FLAMEGPU->getVariable<float>("move_direction_y");
    const float move_dir_z = FLAMEGPU->getVariable<float>("move_direction_z");

    // TGF-β gradient for chemotaxis
    const int voxel_mv = z * (grid_y * grid_x) + y * grid_x + x;
    const float grad_x = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_TGFB_X))[voxel_mv];
    const float grad_y = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_TGFB_Y))[voxel_mv];
    const float grad_z = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_TGFB_Z))[voxel_mv];

    auto occ = FLAMEGPU->environment.getMacroProperty<unsigned int,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX, NUM_OCC_TYPES>("occ_grid");

    const uint8_t* face_flags = reinterpret_cast<const uint8_t*>(
        FLAMEGPU->environment.getProperty<uint64_t>("face_flags_ptr"));

    const float EC50_grad = 1.0f;
    const float dt = FLAMEGPU->environment.getProperty<float>("PARAM_SEC_PER_SLICE");

    int new_hx = -1, new_hy = -1, new_hz = -1;  // candidate head position

    // === RUN PHASE (tumble == 0) ===
    if (tumble == 0) {
        float v_x = move_dir_x / dt;
        float v_y = move_dir_y / dt;
        float v_z = move_dir_z / dt;

        float dot_product = v_x * grad_x + v_y * grad_y + v_z * grad_z;

        // Lambda: same for NORMAL and CAF (shadowed-lambda bug from HCC preserved intentionally)
        float lambda = 0.0000168f;
        float norm_v = std::sqrt(v_x * v_x + v_y * v_y + v_z * v_z);
        float norm_gradient = std::sqrt(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z);

        float cos_theta = dot_product / (norm_v * norm_gradient);

        float H_grad = norm_gradient / (norm_gradient + EC50_grad);
        if (cos_theta < 0) H_grad = -H_grad;

        float p_tumble = (lambda / 2.0f) * (1.0f - cos_theta) * (1.0f - H_grad) * dt;
        p_tumble = 1.0f - std::exp(-p_tumble);

        if (FLAMEGPU->random.uniform<float>() < p_tumble) {
            FLAMEGPU->setVariable<int>("tumble", 1);
            return flamegpu::ALIVE;
        }

        int tx = x + static_cast<int>(std::round(move_dir_x));
        int ty = y + static_cast<int>(std::round(move_dir_y));
        int tz = z + static_cast<int>(std::round(move_dir_z));

        if (tx < 0 || tx >= grid_x || ty < 0 || ty >= grid_y || tz < 0 || tz >= grid_z) {
            return flamegpu::ALIVE;
        }

        // Ductal wall check
        if (is_ductal_wall_blocked(face_flags, x, y, z, tx-x, ty-y, tz-z, grid_x, grid_y)) {
            return flamegpu::ALIVE;
        }

        // Try CAS claim on head's target voxel
        if (occ[tx][ty][tz][CELL_TYPE_CANCER] == 0u &&
            occ[tx][ty][tz][CELL_TYPE_FIB].CAS(0u, 1u) == 0u) {
            new_hx = tx; new_hy = ty; new_hz = tz;
        }
    }
    // === TUMBLE PHASE (tumble == 1) ===
    else {
        int cand_x[26], cand_y[26], cand_z[26];
        int n_cands = 0;
        for (int di = -1; di <= 1; di++) for (int dj = -1; dj <= 1; dj++) for (int dk = -1; dk <= 1; dk++) {
            if (di == 0 && dj == 0 && dk == 0) continue;
            int nx = x + di, ny = y + dj, nz = z + dk;
            if (nx < 0 || nx >= grid_x || ny < 0 || ny >= grid_y || nz < 0 || nz >= grid_z) continue;
            if (is_ductal_wall_blocked(face_flags, x, y, z, di, dj, dk, grid_x, grid_y)) continue;
            if (occ[nx][ny][nz][CELL_TYPE_CANCER] != 0u) continue;
            if (occ[nx][ny][nz][CELL_TYPE_FIB] != 0u) continue;
            cand_x[n_cands] = nx; cand_y[n_cands] = ny; cand_z[n_cands] = nz;
            n_cands++;
        }
        if (n_cands == 0) return flamegpu::ALIVE;
        // Fisher-Yates shuffle
        for (int i = n_cands - 1; i > 0; i--) {
            int j = static_cast<int>(FLAMEGPU->random.uniform<float>() * (i + 1));
            if (j > i) j = i;
            int tx = cand_x[i]; cand_x[i] = cand_x[j]; cand_x[j] = tx;
            int ty = cand_y[i]; cand_y[i] = cand_y[j]; cand_y[j] = ty;
            int tz = cand_z[i]; cand_z[i] = cand_z[j]; cand_z[j] = tz;
        }
        for (int i = 0; i < n_cands; i++) {
            if (occ[cand_x[i]][cand_y[i]][cand_z[i]][CELL_TYPE_FIB].CAS(0u, 1u) == 0u) {
                new_hx = cand_x[i]; new_hy = cand_y[i]; new_hz = cand_z[i];
                FLAMEGPU->setVariable<float>("move_direction_x", static_cast<float>(new_hx - x));
                FLAMEGPU->setVariable<float>("move_direction_y", static_cast<float>(new_hy - y));
                FLAMEGPU->setVariable<float>("move_direction_z", static_cast<float>(new_hz - z));
                FLAMEGPU->setVariable<int>("tumble", 0);
                break;
            }
        }
    }

    // === CHAIN SHIFT (if head successfully claimed a new voxel) ===
    if (new_hx >= 0) {
        // Release tail voxel
        const int tail_idx = chain_len - 1;
        const int tail_x = FLAMEGPU->getVariable<int, MAX_FIB_CHAIN_LENGTH>("seg_x", tail_idx);
        const int tail_y = FLAMEGPU->getVariable<int, MAX_FIB_CHAIN_LENGTH>("seg_y", tail_idx);
        const int tail_z = FLAMEGPU->getVariable<int, MAX_FIB_CHAIN_LENGTH>("seg_z", tail_idx);
        occ[tail_x][tail_y][tail_z][CELL_TYPE_FIB].exchange(0u);

        // Shift segments: each takes the position of the one ahead of it
        for (int i = tail_idx; i >= 1; i--) {
            FLAMEGPU->setVariable<int, MAX_FIB_CHAIN_LENGTH>("seg_x", i,
                FLAMEGPU->getVariable<int, MAX_FIB_CHAIN_LENGTH>("seg_x", i - 1));
            FLAMEGPU->setVariable<int, MAX_FIB_CHAIN_LENGTH>("seg_y", i,
                FLAMEGPU->getVariable<int, MAX_FIB_CHAIN_LENGTH>("seg_y", i - 1));
            FLAMEGPU->setVariable<int, MAX_FIB_CHAIN_LENGTH>("seg_z", i,
                FLAMEGPU->getVariable<int, MAX_FIB_CHAIN_LENGTH>("seg_z", i - 1));
        }

        // Set new head position
        FLAMEGPU->setVariable<int, MAX_FIB_CHAIN_LENGTH>("seg_x", 0, new_hx);
        FLAMEGPU->setVariable<int, MAX_FIB_CHAIN_LENGTH>("seg_y", 0, new_hy);
        FLAMEGPU->setVariable<int, MAX_FIB_CHAIN_LENGTH>("seg_z", 0, new_hz);

        // Update head aliases
        FLAMEGPU->setVariable<int>("x", new_hx);
        FLAMEGPU->setVariable<int>("y", new_hy);
        FLAMEGPU->setVariable<int>("z", new_hz);
    }

    return flamegpu::ALIVE;
}

// ============================================================================
// Fibroblast: State step — TGFB-driven activation (NORMAL -> CAF) and lifespan
// Sets divide_flag for fib_activate to consume.
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(fib_state_step, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int cell_state = FLAMEGPU->getVariable<int>("cell_state");
    if (cell_state != FIB_NORMAL) return flamegpu::ALIVE;

    const int nx_ss = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny_ss = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int ax_ss = FLAMEGPU->getVariable<int>("x");
    const int ay_ss = FLAMEGPU->getVariable<int>("y");
    const int az_ss = FLAMEGPU->getVariable<int>("z");
    const int voxel_ss = az_ss * ny_ss * nx_ss + ay_ss * nx_ss + ax_ss;
    const float TGFB = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>(PDE_CONC_TGFB))[voxel_ss];
    const float ec50 = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_CAF_EC50");
    const float caf_act = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_CAF_ACTIVATION");
    const float activation = caf_act * 5.0f * (1.0f + TGFB / (TGFB + ec50));
    const float p_div = 1.0f - expf(-activation);

    if (FLAMEGPU->random.uniform<float>() < p_div) {
        FLAMEGPU->setVariable<int>("divide_flag", 1);
    }

    return flamegpu::ALIVE;
}

// ============================================================================
// Fibroblast: Activation — extend chain from 3→5, set state to CAF
// Finds 2 sequential free Von Neumann neighbors off the tail.
// Both must be found or activation is skipped entirely.
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(fib_activate, flamegpu::MessageNone, flamegpu::MessageNone) {
    if (FLAMEGPU->getVariable<int>("divide_flag") == 0) return flamegpu::ALIVE;
    if (FLAMEGPU->getVariable<int>("cell_state") != FIB_NORMAL) {
        FLAMEGPU->setVariable<int>("divide_flag", 0);
        return flamegpu::ALIVE;
    }

    // Clear flag regardless of outcome
    FLAMEGPU->setVariable<int>("divide_flag", 0);

    const int chain_len = FLAMEGPU->getVariable<int>("chain_len");
    if (chain_len >= MAX_FIB_CHAIN_LENGTH) return flamegpu::ALIVE;  // already max

    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int grid_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");

    auto occ = FLAMEGPU->environment.getMacroProperty<unsigned int,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX, NUM_OCC_TYPES>("occ_grid");

    // Current tail position
    const int tail_x = FLAMEGPU->getVariable<int, MAX_FIB_CHAIN_LENGTH>("seg_x", chain_len - 1);
    const int tail_y = FLAMEGPU->getVariable<int, MAX_FIB_CHAIN_LENGTH>("seg_y", chain_len - 1);
    const int tail_z = FLAMEGPU->getVariable<int, MAX_FIB_CHAIN_LENGTH>("seg_z", chain_len - 1);

    // Von Neumann (6 face neighbors)
    const int dx6[6] = {1, -1, 0, 0, 0, 0};
    const int dy6[6] = {0, 0, 1, -1, 0, 0};
    const int dz6[6] = {0, 0, 0, 0, 1, -1};

    // Find first new segment adjacent to tail
    int n1x = -1, n1y = -1, n1z = -1;
    for (int d = 0; d < 6; d++) {
        int nx = tail_x + dx6[d], ny = tail_y + dy6[d], nz = tail_z + dz6[d];
        if (nx < 0 || nx >= grid_x || ny < 0 || ny >= grid_y || nz < 0 || nz >= grid_z) continue;
        if (occ[nx][ny][nz][CELL_TYPE_CANCER] != 0u) continue;
        if (occ[nx][ny][nz][CELL_TYPE_FIB].CAS(0u, 1u) != 0u) continue;  // already occupied
        n1x = nx; n1y = ny; n1z = nz;
        break;
    }
    if (n1x < 0) return flamegpu::ALIVE;  // no space for first segment

    // Find second new segment adjacent to first new (not tail)
    int n2x = -1, n2y = -1, n2z = -1;
    for (int d = 0; d < 6; d++) {
        int nx = n1x + dx6[d], ny = n1y + dy6[d], nz = n1z + dz6[d];
        if (nx < 0 || nx >= grid_x || ny < 0 || ny >= grid_y || nz < 0 || nz >= grid_z) continue;
        // Skip the tail voxel itself
        if (nx == tail_x && ny == tail_y && nz == tail_z) continue;
        if (occ[nx][ny][nz][CELL_TYPE_CANCER] != 0u) continue;
        if (occ[nx][ny][nz][CELL_TYPE_FIB].CAS(0u, 1u) != 0u) continue;
        n2x = nx; n2y = ny; n2z = nz;
        break;
    }
    if (n2x < 0) {
        // Failed to find second — release first and abort
        occ[n1x][n1y][n1z][CELL_TYPE_FIB].exchange(0u);
        return flamegpu::ALIVE;
    }

    // Success: extend chain
    FLAMEGPU->setVariable<int, MAX_FIB_CHAIN_LENGTH>("seg_x", chain_len, n1x);
    FLAMEGPU->setVariable<int, MAX_FIB_CHAIN_LENGTH>("seg_y", chain_len, n1y);
    FLAMEGPU->setVariable<int, MAX_FIB_CHAIN_LENGTH>("seg_z", chain_len, n1z);
    FLAMEGPU->setVariable<int, MAX_FIB_CHAIN_LENGTH>("seg_x", chain_len + 1, n2x);
    FLAMEGPU->setVariable<int, MAX_FIB_CHAIN_LENGTH>("seg_y", chain_len + 1, n2y);
    FLAMEGPU->setVariable<int, MAX_FIB_CHAIN_LENGTH>("seg_z", chain_len + 1, n2z);
    FLAMEGPU->setVariable<int>("chain_len", chain_len + 2);
    FLAMEGPU->setVariable<int>("cell_state", FIB_CAF);

    return flamegpu::ALIVE;
}

// ============================================================================
// Fibroblast: Build Gaussian density field for ECM deposition
// Iterates over all chain_len segments, scattering a Gaussian kernel per segment.
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(fib_build_density_field, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int chain_len = FLAMEGPU->getVariable<int>("chain_len");
    const int cell_state = FLAMEGPU->getVariable<int>("cell_state");

    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int grid_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");

    const float scale = (cell_state == FIB_CAF) ? 1.0f : 0.5f;

    const int radius = 10;
    const float variance = 9.0f;  // sigma^2 = 3^2
    const float normalizer = 0.014784f;

    float* field_ptr = reinterpret_cast<float*>(
        FLAMEGPU->environment.getProperty<uint64_t>("fib_density_field_ptr"));

    // ECM reorientation accumulator: fibroblasts pull nearby ECM toward themselves
    float* reorient_x = reinterpret_cast<float*>(
        FLAMEGPU->environment.getProperty<uint64_t>("ecm_reorient_x_ptr"));
    float* reorient_y = reinterpret_cast<float*>(
        FLAMEGPU->environment.getProperty<uint64_t>("ecm_reorient_y_ptr"));
    float* reorient_z = reinterpret_cast<float*>(
        FLAMEGPU->environment.getProperty<uint64_t>("ecm_reorient_z_ptr"));

    // Traction strength: how strongly fibroblasts pull ECM toward themselves
    // CAF pulls harder than normal fibroblasts (matching ECM deposition scale)
    const float traction_strength = scale * 2.0f;

    for (int seg = 0; seg < chain_len; seg++) {
        const int cx = FLAMEGPU->getVariable<int, MAX_FIB_CHAIN_LENGTH>("seg_x", seg);
        const int cy = FLAMEGPU->getVariable<int, MAX_FIB_CHAIN_LENGTH>("seg_y", seg);
        const int cz = FLAMEGPU->getVariable<int, MAX_FIB_CHAIN_LENGTH>("seg_z", seg);

        for (int dx = -radius; dx <= radius; dx++) {
            const int nx_v = cx + dx;
            if (nx_v < 0 || nx_v >= grid_x) continue;
            for (int dy = -radius; dy <= radius; dy++) {
                const int ny_v = cy + dy;
                if (ny_v < 0 || ny_v >= grid_y) continue;
                for (int dz = -radius; dz <= radius; dz++) {
                    const int nz_v = cz + dz;
                    if (nz_v < 0 || nz_v >= grid_z) continue;
                    float dist_sq = static_cast<float>(dx * dx + dy * dy + dz * dz);
                    if (dist_sq < 1e-6f) continue;  // skip self-voxel
                    float kernel_val = scale * normalizer * expf(-dist_sq / (2.0f * variance));
                    int vidx = nz_v * (grid_x * grid_y) + ny_v * grid_x + nx_v;
                    atomicAdd(&field_ptr[vidx], kernel_val);

                    // Traction: vector from voxel toward fibroblast segment, Gaussian weighted
                    float inv_dist = rsqrtf(dist_sq);  // 1/sqrt(dist_sq)
                    float trac = traction_strength * kernel_val * inv_dist;
                    // Direction: (cx-nx_v, cy-ny_v, cz-nz_v) normalized * weight
                    // Note: dx = nx_v - cx, so toward fib = (-dx, -dy, -dz)
                    atomicAdd(&reorient_x[vidx], trac * static_cast<float>(-dx));
                    atomicAdd(&reorient_y[vidx], trac * static_cast<float>(-dy));
                    atomicAdd(&reorient_z[vidx], trac * static_cast<float>(-dz));
                }
            }
        }
    }

    return flamegpu::ALIVE;
}

}  // namespace PDAC

#endif  // FIBROBLAST_CUH
