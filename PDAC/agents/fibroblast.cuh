// Fibroblast Agent Behavior Functions
// States: FIB_QUIESCENT, FIB_MYCAF (myofibroblastic CAF), FIB_ICAF (inflammatory CAF)
// Single-cell, single-voxel agents. No chain model.
//
// Activation: Quiescent → myCAF (TGF-β driven), Quiescent → iCAF (IL-1 driven, TGF-β suppressed)
// ECM deposition: myCAF only (Gaussian kernel, radius from XML)
// Chemical sources: myCAF → TGF-β + CCL2; iCAF → IL-6 + CCL2 + CXCL12 + CCL5
// Division: activated fibroblasts can divide locally (cancer-neighbor PDGF proxy)

#ifndef FIBROBLAST_CUH
#define FIBROBLAST_CUH

#include "flamegpu/flamegpu.h"
#include "../core/common.cuh"

namespace PDAC {

// ============================================================================
// Fibroblast: Broadcast location (spatial messaging)
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
    FLAMEGPU->message_out.setVariable<int>("voxel_x", x);
    FLAMEGPU->message_out.setVariable<int>("voxel_y", y);
    FLAMEGPU->message_out.setVariable<int>("voxel_z", z);
    FLAMEGPU->message_out.setVariable<float>("kill_factor", 0.0f);  // N/A for fibroblast
    FLAMEGPU->message_out.setVariable<int>("dead", 0);
    FLAMEGPU->message_out.setLocation(fx, fy, fz);

    // Population count by state
    const int cs = FLAMEGPU->getVariable<int>("cell_state");
    auto* sc = reinterpret_cast<unsigned int*>(FLAMEGPU->environment.getProperty<uint64_t>("state_counters_ptr"));
    int sc_idx;
    if (cs == FIB_QUIESCENT)  sc_idx = SC_FIB_QUIESCENT;
    else if (cs == FIB_MYCAF) sc_idx = SC_FIB_MYCAF;
    else if (cs == FIB_ICAF)  sc_idx = SC_FIB_ICAF;
    else                       sc_idx = SC_FIB_FRC;
    atomicAdd(&sc[sc_idx], 1u);

    return flamegpu::ALIVE;
}

// ============================================================================
// Fibroblast: Scan neighbors (for adhesion)
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(fib_scan_neighbors, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
    const float voxel_size = FLAMEGPU->environment.getProperty<float>("voxel_size");
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const float wx = (x + 0.5f) * voxel_size;
    const float wy = (y + 0.5f) * voxel_size;
    const float wz = (z + 0.5f) * voxel_size;

    int cancer_count = 0;
    int fib_count = 0;
    int ltb_lymph_count = 0;  // LTβ-competent lymphocytes (drives iCAF→FRC)

    // Adhesion: count all type+state neighbors
    int adh_counts[ABM_STATE_COUNTER_SIZE] = {0};

    for (const auto& msg : FLAMEGPU->message_in(wx, wy, wz)) {
        const int mx = msg.getVariable<int>("voxel_x");
        const int my = msg.getVariable<int>("voxel_y");
        const int mz = msg.getVariable<int>("voxel_z");
        if (abs(mx - x) <= 1 && abs(my - y) <= 1 && abs(mz - z) <= 1
            && !(mx == x && my == y && mz == z)) {
            const int at = msg.getVariable<int>("agent_type");
            const int ms = msg.getVariable<int>("cell_state");
            const float kf = msg.getVariable<float>("kill_factor");

            // Adhesion: accumulate into type+state count vector
            adh_counts[msg_to_sc_idx(at, ms, kf)]++;

            if (at == CELL_TYPE_CANCER) { cancer_count++; continue; }
            if (at == CELL_TYPE_FIB) { fib_count++; continue; }
            // LTβ-competent lymphocyte filter (state-gated)
            // Ansel2000/Fütterer1998: naive+activated B cells express surface LTα1β2; plasma downregulates (Mebius2003)
            // Luther2002/Scheu2002: TCR-activated CD8 (eff+cyt) produce LTβ; naive/suppressed excluded
            // Furtado2007/Kumar2015: Th+Tfh are canonical LTβ producers; Tregs excluded (TLS forms at effector, not regulatory, sites)
            if (at == CELL_TYPE_BCELL) {
                if (ms == BCELL_NAIVE || ms == BCELL_ACTIVATED) ltb_lymph_count++;
            } else if (at == CELL_TYPE_T) {
                if (ms == T_CELL_EFF || ms == T_CELL_CYT) ltb_lymph_count++;
            } else if (at == CELL_TYPE_TREG) {
                if (ms == TCD4_TH || ms == TCD4_TFH) ltb_lymph_count++;
            }
        }
    }

    // Compute adhesion p_move from matrix
    const int my_sc = self_sc_idx(CELL_TYPE_FIB, FLAMEGPU->getVariable<int>("cell_state"));
    const float adh_pmove = compute_adhesion_pmove(my_sc, adh_counts, ADH_MATRIX_PTR(FLAMEGPU));

    FLAMEGPU->setVariable<int>("neighbor_cancer_count", cancer_count);
    FLAMEGPU->setVariable<int>("neighbor_fib_count", fib_count);
    FLAMEGPU->setVariable<int>("neighbor_ltb_lymph_count", ltb_lymph_count);
    FLAMEGPU->setVariable<float>("adh_p_move", adh_pmove);
    return flamegpu::ALIVE;
}

// ============================================================================
// Fibroblast: Write volume to occupancy grid
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(fib_write_to_occ_grid, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->setVariable<int>("moves_remaining",
        FLAMEGPU->environment.getProperty<int>("PARAM_FIB_MOVE_STEPS"));
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
    const int cell_state = FLAMEGPU->getVariable<int>("cell_state");
    float my_vol;
    if (cell_state == FIB_QUIESCENT) my_vol = FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_FIB_QUIESCENT");
    else if (cell_state == FIB_MYCAF) my_vol = FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_FIB_MYCAF");
    else if (cell_state == FIB_ICAF)  my_vol = FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_FIB_ICAF");
    else                               my_vol = FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_FIB_FRC");
    float* vol_used = VOL_PTR(FLAMEGPU);
    atomicAdd(&vol_used[vidx], my_vol);

    return flamegpu::ALIVE;
}

// ============================================================================
// Fibroblast: Compute chemical sources
// myCAF: TGF-β + CCL2 (HIF-boosted under hypoxia, reduced at severe hypoxia)
// iCAF:  IL-6 + CCL2 + CXCL12 + CCL5
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(fib_compute_chemical_sources, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int cs = FLAMEGPU->getVariable<int>("cell_state");
    if (cs == FIB_QUIESCENT) return flamegpu::ALIVE;

    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int voxel = z * (grid_x * grid_y) + y * grid_x + x;

    const float voxel_size_cm = FLAMEGPU->environment.getProperty<float>("voxel_size") * 1e-4f;
    const float voxel_volume = voxel_size_cm * voxel_size_cm * voxel_size_cm;

    // Hypoxia modulation for myCAF secretion
    float hif_boost = 1.0f;
    float metabolic_factor = 1.0f;
    if (cs == FIB_MYCAF) {
        const float local_O2 = PDE_READ(FLAMEGPU, PDE_CONC_O2, voxel);
        const float hyp_th = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_HYPOXIA_TH");
        if (local_O2 < hyp_th && hyp_th > 0.0f) {
            hif_boost = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_HIF_TGFB_BOOST");
            // PSC metabolic decline: at severe hypoxia (<50% of threshold), secretion drops
            const float severe_th = hyp_th * 0.5f;
            if (local_O2 < severe_th) {
                metabolic_factor = fmaxf(local_O2 / severe_th, 0.1f);
            }
        }
    }

    if (cs == FIB_MYCAF) {
        const float tgfb_rate  = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_MYCAF_TGFB_RELEASE");
        const float ccl2_rate  = FLAMEGPU->environment.getProperty<float>("PARAM_CCL2_RELEASE");
        const float vegfa_rate = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_MYCAF_VEGFA_RELEASE");
        float eff_boost = hif_boost * metabolic_factor;
        const float cxcl9_rate = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_CXCL9_RELEASE");
        PDE_SECRETE(FLAMEGPU, PDE_SRC_TGFB,  voxel, tgfb_rate  * eff_boost / voxel_volume);
        PDE_SECRETE(FLAMEGPU, PDE_SRC_CCL2,  voxel, ccl2_rate  * eff_boost / voxel_volume);
        PDE_SECRETE(FLAMEGPU, PDE_SRC_VEGFA, voxel, vegfa_rate * eff_boost / voxel_volume);
        // CXCL9/10 (CXCR3 ligand) — CAF-density-sourced effector recruitment cue (Step 4b)
        PDE_SECRETE(FLAMEGPU, PDE_SRC_CXCL9_10, voxel, cxcl9_rate * eff_boost / voxel_volume);
    } else if (cs == FIB_ICAF) {
        const float il6_rate    = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ICAF_IL6_RELEASE");
        const float ccl2_rate   = FLAMEGPU->environment.getProperty<float>("PARAM_CCL2_RELEASE");
        const float cxcl12_rate = FLAMEGPU->environment.getProperty<float>("PARAM_CXCL12_ICAF_RELEASE");
        const float ccl5_rate   = FLAMEGPU->environment.getProperty<float>("PARAM_CCL5_ICAF_RELEASE");
        const float cxcl9_rate = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_CXCL9_RELEASE");
        PDE_SECRETE(FLAMEGPU, PDE_SRC_IL6,    voxel, il6_rate / voxel_volume);
        PDE_SECRETE(FLAMEGPU, PDE_SRC_CCL2,   voxel, ccl2_rate / voxel_volume);
        PDE_SECRETE(FLAMEGPU, PDE_SRC_CXCL12, voxel, cxcl12_rate / voxel_volume);
        PDE_SECRETE(FLAMEGPU, PDE_SRC_CCL5,   voxel, ccl5_rate / voxel_volume);
        // CXCL9/10 (CXCR3 ligand) — CAF-density-sourced effector recruitment cue (Step 4b)
        PDE_SECRETE(FLAMEGPU, PDE_SRC_CXCL9_10, voxel, cxcl9_rate / voxel_volume);
    } else {  // FIB_FRC — TLS T-zone scaffold, CCL21 source (Luther2000, Link2007)
        const float ccl21_rate  = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_FRC_CCL21_RELEASE");
        const float cxcl12_rate = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_FRC_CXCL12_RELEASE");
        PDE_SECRETE(FLAMEGPU, PDE_SRC_CCL21,  voxel, ccl21_rate  / voxel_volume);
        PDE_SECRETE(FLAMEGPU, PDE_SRC_CXCL12, voxel, cxcl12_rate / voxel_volume);
    }

    return flamegpu::ALIVE;
}

// ============================================================================
// Fibroblast: Movement — single-cell random walk, state-dependent probability
// ============================================================================
// Fibroblast movement via unified movement framework.
// myCAF: TGF-β chemotaxis + persistence. iCAF: random walk + weak persistence.
// Quiescent: no movement. Activated: adhesion-gated via neighbor counts + ECM.
// Fibroblast movement — Step-5 deterministic propose/commit (mirrors cancer). One
// voxel-move per round: PROPOSE picks the target (det_rng, not FLAMEGPU->random which is
// nondeterministic for born cells) and atomicMin-reserves it by (source-voxel, det_tag);
// COMMIT moves the unique winner. Replaces the racy batched fib_move (volume_try_claim
// "first arrival wins" was thread-schedule dependent). Sessile states early-return.
FLAMEGPU_AGENT_FUNCTION(fib_move_propose, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->setVariable<int>("move_target_vidx", -1);  // reset proposal each round
    const int cs = FLAMEGPU->getVariable<int>("cell_state");
    // Quiescent + FRC fibroblasts are sessile (preserve tissue architecture / TLS scaffold)
    if (cs == FIB_QUIESCENT || cs == FIB_FRC) return flamegpu::ALIVE;

    int moves_remaining = FLAMEGPU->getVariable<int>("moves_remaining");
    if (moves_remaining <= 0) return flamegpu::ALIVE;
    FLAMEGPU->setVariable<int>("moves_remaining", moves_remaining - 1);

    int x = FLAMEGPU->getVariable<int>("x");
    int y = FLAMEGPU->getVariable<int>("y");
    int z = FLAMEGPU->getVariable<int>("z");

    float my_vol = (cs == FIB_MYCAF) ?
        FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_FIB_MYCAF") :
        FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_FIB_ICAF");
    float p_persist = (cs == FIB_MYCAF) ?
        FLAMEGPU->environment.getProperty<float>("PARAM_PERSIST_FIB_MYCAF") :
        FLAMEGPU->environment.getProperty<float>("PARAM_PERSIST_FIB_ICAF");
    float bias = (cs == FIB_MYCAF) ?
        ci_to_bias(FLAMEGPU->environment.getProperty<float>("PARAM_CHEMO_CI_FIB_MYCAF")) : 0.0f;

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
    mp.min_porosity = FLAMEGPU->environment.getProperty<float>("PARAM_ECM_POROSITY_FIB");
    mp.p_move = FLAMEGPU->getVariable<float>("adh_p_move");
    mp.p_persist = p_persist;
    mp.bias_strength = bias;
    mp.orient_x = ECM_ORIENT_X_PTR(FLAMEGPU);
    mp.orient_y = ECM_ORIENT_Y_PTR(FLAMEGPU);
    mp.orient_z = ECM_ORIENT_Z_PTR(FLAMEGPU);
    mp.barrier_strength = (cs == FIB_MYCAF) ?
        FLAMEGPU->environment.getProperty<float>("PARAM_FIBER_BARRIER_FIB_MYCAF") :
        FLAMEGPU->environment.getProperty<float>("PARAM_FIBER_BARRIER_FIB_ICAF");
    const float w_cg = (cs == FIB_MYCAF) ?
        FLAMEGPU->environment.getProperty<float>("PARAM_CONTACT_GUIDANCE_FIB_MYCAF") :
        FLAMEGPU->environment.getProperty<float>("PARAM_CONTACT_GUIDANCE_FIB_ICAF");

    int pdx = FLAMEGPU->getVariable<int>("persist_dir_x");
    int pdy = FLAMEGPU->getVariable<int>("persist_dir_y");
    int pdz = FLAMEGPU->getVariable<int>("persist_dir_z");

    // Read TGF-β gradient + blend with fiber orientation at the current voxel — the same
    // per-move setup run_move_batch did internally (single move here, not batched).
    const int GXY = mp.grid_x * mp.grid_y;
    const int vidx = z * GXY + y * mp.grid_x + x;
    float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f;
    if (bias > 0.0f) {
        g0 = reinterpret_cast<const float*>(FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_TGFB_X))[vidx];
        g1 = reinterpret_cast<const float*>(FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_TGFB_Y))[vidx];
        g2 = reinterpret_cast<const float*>(FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_TGFB_Z))[vidx];
    }
    const float ox = mp.orient_x[vidx], oy = mp.orient_y[vidx], oz = mp.orient_z[vidx];
    if (bias > 0.0f) apply_contact_guidance(g0, g1, g2, ox, oy, oz, w_cg);
    else             apply_contact_guidance_persist(g0, g1, g2, mp.bias_strength, ox, oy, oz, w_cg, pdx, pdy, pdz);
    mp.grad_x = g0; mp.grad_y = g1; mp.grad_z = g2;

    // Deterministic: lineage-tag-seeded RNG for the move draws, pick (no claim), then reserve.
    const unsigned int sim_seed = FLAMEGPU->environment.getProperty<unsigned int>("sim_seed");
    const unsigned int det_tag = FLAMEGPU->getVariable<unsigned int>("det_tag");
    unsigned int rs = det_rng_seed(static_cast<int>(det_tag), FLAMEGPU->getStepCounter(),
                                   static_cast<unsigned int>(moves_remaining), sim_seed);
    const float r_move = det_rng_uniform(rs);
    const float r_persist = det_rng_uniform(rs);
    const float r_dir = det_rng_uniform(rs);
    MoveResult r = move_cell_pick(mp, x, y, z, pdx, pdy, pdz, r_move, r_persist, r_dir);

    if (r.moved) {
        const int tvidx = r.new_z * GXY + r.new_y * mp.grid_x + r.new_x;
        unsigned long long* owner = reinterpret_cast<unsigned long long*>(
            FLAMEGPU->environment.getProperty<unsigned long long>("voxel_owner_ptr"));
        reserve_voxel(owner, tvidx, make_claim_priority(vidx, det_tag));
        FLAMEGPU->setVariable<int>("move_target_vidx", tvidx);
    }
    return flamegpu::ALIVE;
}

// Fibroblast movement — COMMIT pass. Moves the unique winner of each reserved voxel
// (lowest source-voxel/det_tag); losers stay. Race-free: one claimer per target voxel.
FLAMEGPU_AGENT_FUNCTION(fib_move_commit, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int tvidx = FLAMEGPU->getVariable<int>("move_target_vidx");
    if (tvidx < 0) return flamegpu::ALIVE;

    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const int cs = FLAMEGPU->getVariable<int>("cell_state");
    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");

    const unsigned int det_tag = FLAMEGPU->getVariable<unsigned int>("det_tag");
    const int my_vidx = z * (grid_x * grid_y) + y * grid_x + x;
    unsigned long long* owner = reinterpret_cast<unsigned long long*>(
        FLAMEGPU->environment.getProperty<unsigned long long>("voxel_owner_ptr"));
    if (!won_voxel(owner, tvidx, make_claim_priority(my_vidx, det_tag))) {
        return flamegpu::ALIVE;  // lost the deterministic contention → stay
    }

    float my_vol = (cs == FIB_MYCAF) ?
        FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_FIB_MYCAF") :
        FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_FIB_ICAF");
    float* vol_used = VOL_PTR(FLAMEGPU);

    // Unconditional claim: move_cell_pick verified snapshot capacity in propose + reserve gives a
    // unique winner per target → integer atomicAdd sums are order-independent, no capacity CHECK to
    // flip on a concurrent claim/release (target == another cell's source) → deterministic commit.
    atomicAdd(&vol_used[tvidx], my_vol);
    volume_release(vol_used, my_vidx, my_vol);

    const int target_x = tvidx % grid_x;
    const int target_y = (tvidx / grid_x) % grid_y;
    const int target_z = tvidx / (grid_x * grid_y);

    FLAMEGPU->setVariable<int>("x", target_x);
    FLAMEGPU->setVariable<int>("y", target_y);
    FLAMEGPU->setVariable<int>("z", target_z);
    FLAMEGPU->setVariable<int>("persist_dir_x", target_x - x);
    FLAMEGPU->setVariable<int>("persist_dir_y", target_y - y);
    FLAMEGPU->setVariable<int>("persist_dir_z", target_z - z);
    return flamegpu::ALIVE;
}

// ============================================================================
// Fibroblast: State step — activation, lifespan, division cooldown
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(fib_state_step, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int cs = FLAMEGPU->getVariable<int>("cell_state");

    // Lifespan check
    int life = FLAMEGPU->getVariable<int>("life");
    life--;
    FLAMEGPU->setVariable<int>("life", life);
    if (life <= 0) {
        auto* ec = reinterpret_cast<unsigned int*>(FLAMEGPU->environment.getProperty<uint64_t>("event_counters_ptr"));
        int death_idx;
        if (cs == FIB_QUIESCENT)  death_idx = EVT_DEATH_FIB_QUIESCENT;
        else if (cs == FIB_MYCAF) death_idx = EVT_DEATH_FIB_MYCAF;
        else if (cs == FIB_ICAF)  death_idx = EVT_DEATH_FIB_ICAF;
        else                       death_idx = EVT_DEATH_FIB_FRC;
        atomicAdd(&ec[death_idx], 1u);
        return flamegpu::DEAD;
    }

    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int ax = FLAMEGPU->getVariable<int>("x");
    const int ay = FLAMEGPU->getVariable<int>("y");
    const int az = FLAMEGPU->getVariable<int>("z");
    const int voxel = az * grid_y * grid_x + ay * grid_x + ax;

    // Read local concentrations for activation and interconversion
    const float TGFB = PDE_READ(FLAMEGPU, PDE_CONC_TGFB, voxel);
    const float IL1  = PDE_READ(FLAMEGPU, PDE_CONC_IL1, voxel);
    const float IL6  = PDE_READ(FLAMEGPU, PDE_CONC_IL6, voxel);
    const float caf_ec50 = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_CAF_EC50");
    const float H_TGFb = TGFB / (TGFB + caf_ec50 + 1e-30f);  // shared TGFb_50_CAF_act

    // IL-6 amplification for iCAF pathways: (1 + f_IL6 * H_IL6_iCAF)
    const float f_IL6  = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_IL6_ICAF_FACTOR");
    const float il6_ec50 = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_IL6_ICAF_EC50");
    const float H_IL6 = IL6 / (IL6 + il6_ec50 + 1e-30f);
    const float il6_boost = 1.0f + f_IL6 * H_IL6;

    // Deterministic RNG stream for the activation/interconversion draws below (FLAMEGPU->random
    // is nondeterministic for born cells; det_tag-seeded is stable given lineage). A cell is in
    // exactly one state, so only one branch draws — seed once, draw sequentially.
    const unsigned int sim_seed = FLAMEGPU->environment.getProperty<unsigned int>("sim_seed");
    const unsigned int det_tag = FLAMEGPU->getVariable<unsigned int>("det_tag");
    unsigned int srs = det_rng_seed(static_cast<int>(det_tag), FLAMEGPU->getStepCounter(), 0x57A7Eu, sim_seed);

    // Activation: only quiescent fibroblasts can activate
    if (cs == FIB_QUIESCENT) {
        // myCAF activation (ODE RF242): k_myCAF * H_TGFb(TGFb_50_CAF_act)
        const float k_mycaf = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_MYCAF_ACTIVATION");
        const float p_mycaf = k_mycaf * H_TGFb;

        if (det_rng_uniform(srs) < p_mycaf) {
            FLAMEGPU->setVariable<int>("cell_state", FIB_MYCAF);
            return flamegpu::ALIVE;
        }

        // iCAF activation (ODE RF243): k_iCAF * H_IL1_eff * (1 + f_IL6 * H_IL6_iCAF)
        // ABM: IL-1 Hill * (1 - TGF-β suppression) * IL-6 boost
        const float k_icaf   = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ICAF_ACTIVATION");
        const float icaf_ec50  = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ICAF_IL1_EC50");
        const float icaf_n     = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ICAF_IL1_HILL_N");
        const float supp_ec50  = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ICAF_TGFB_SUPPRESS_EC50");
        const float supp_n     = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ICAF_TGFB_SUPPRESS_N");
        const float p_icaf = k_icaf * hill_equation(IL1, icaf_ec50, icaf_n)
                            * (1.0f - hill_equation(TGFB, supp_ec50, supp_n))
                            * il6_boost;

        if (det_rng_uniform(srs) < p_icaf) {
            FLAMEGPU->setVariable<int>("cell_state", FIB_ICAF);
            return flamegpu::ALIVE;
        }
    }

    // iCAF ↔ myCAF interconversion (ODE RF245, RF246)
    if (cs == FIB_ICAF) {
        // iCAF → FRC: LTβ-competent lymphocyte density gating (takes priority over iCAF→myCAF)
        // Sustained threshold for PARAM_FIB_FRC_DWELL_STEPS → irreversible FRC commitment.
        // Refs: Bénézech2012 (FRC maturation kinetics), Drayton2003 (ectopic TLS cell density)
        const int n_ltb = FLAMEGPU->getVariable<int>("neighbor_ltb_lymph_count");
        const int frc_th = static_cast<int>(FLAMEGPU->environment.getProperty<float>("PARAM_FIB_FRC_LYMPH_THRESHOLD"));
        int dwell = FLAMEGPU->getVariable<int>("frc_dwell_counter");
        if (n_ltb >= frc_th) {
            dwell++;
        } else {
            dwell = 0;  // reset if threshold not sustained
        }
        const int frc_dwell_max = static_cast<int>(FLAMEGPU->environment.getProperty<float>("PARAM_FIB_FRC_DWELL_STEPS"));
        if (dwell >= frc_dwell_max) {
            FLAMEGPU->setVariable<int>("cell_state", FIB_FRC);
            FLAMEGPU->setVariable<int>("frc_dwell_counter", 0);
            return flamegpu::ALIVE;
        }
        FLAMEGPU->setVariable<int>("frc_dwell_counter", dwell);

        // iCAF → myCAF: k_iCAF_to_myCAF * H_TGFb(TGFb_50_CAF_act)
        const float k_i2m = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ICAF_TO_MYCAF");
        if (det_rng_uniform(srs) < k_i2m * H_TGFb) {
            FLAMEGPU->setVariable<int>("cell_state", FIB_MYCAF);
        }
    } else if (cs == FIB_MYCAF) {
        // myCAF → iCAF: k_myCAF_to_iCAF * H_IL1_eff * (1 + f_IL6 * H_IL6_iCAF)
        const float k_m2i = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_MYCAF_TO_ICAF");
        const float icaf_ec50  = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ICAF_IL1_EC50");
        const float icaf_n     = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ICAF_IL1_HILL_N");
        float H_IL1 = hill_equation(IL1, icaf_ec50, icaf_n);
        if (det_rng_uniform(srs) < k_m2i * H_IL1 * il6_boost) {
            FLAMEGPU->setVariable<int>("cell_state", FIB_ICAF);
        }
    }

    // Division cooldown decrement
    int cooldown = FLAMEGPU->getVariable<int>("divide_cooldown");
    if (cooldown > 0) {
        FLAMEGPU->setVariable<int>("divide_cooldown", cooldown - 1);
    }

    // Low-rate CAF proliferation: k_*CAF_prolif in XML is dialed down to the death
    // rate so births balance deaths (stable population with turnover). Kept the
    // divide_count cap as a safeguard; remove or raise once the QSP stroma
    // carrying-cap term (RF247/248) is mirrored on the ABM side.
    if ((cs == FIB_MYCAF || cs == FIB_ICAF) && cooldown <= 0) {
        const int div_count = FLAMEGPU->getVariable<int>("divide_count");
        const int div_max = static_cast<int>(FLAMEGPU->environment.getProperty<float>("PARAM_FIB_DIV_MAX"));
        if (div_count < div_max) {
            FLAMEGPU->setVariable<int>("divide_flag", 1);
        }
    }

    return flamegpu::ALIVE;
}

// ============================================================================
// Fibroblast: Division — Step-5 deterministic reserve/confirm (mirrors cancer).
// RESERVE picks one free Von Neumann neighbor (det_rng, not FLAMEGPU->random) and
// atomicMin-reserves it by (source-voxel, det_tag); CONFIRM commits the unique winner
// and births the daughter. Replaces the racy single-pass fib_divide (Fisher-Yates +
// volume_try_claim "first arrival wins" was thread-schedule dependent).
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(fib_divide_reserve, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->setVariable<int>("div_target_vidx", -1);  // reset reservation each step
    if (FLAMEGPU->getVariable<int>("divide_flag") == 0) return flamegpu::ALIVE;
    FLAMEGPU->setVariable<int>("divide_flag", 0);  // consume intent (re-set next step if eligible)

    const int cs = FLAMEGPU->getVariable<int>("cell_state");
    if (cs == FIB_QUIESCENT || cs == FIB_FRC) return flamegpu::ALIVE;

    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int grid_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");

    float* vol_used = VOL_PTR(FLAMEGPU);
    const float capacity = FLAMEGPU->environment.getProperty<float>("PARAM_VOXEL_CAPACITY");
    float daughter_vol = (cs == FIB_MYCAF) ?
        FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_FIB_MYCAF") :
        FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_FIB_ICAF");

    // Collect Von Neumann (6 face-adjacent) neighbors with enough volume capacity.
    const int dx6[6] = {1, -1, 0, 0, 0, 0};
    const int dy6[6] = {0, 0, 1, -1, 0, 0};
    const int dz6[6] = {0, 0, 0, 0, 1, -1};
    int cand_vidx[6];
    int n_cands = 0;
    for (int d = 0; d < 6; d++) {
        const int nx = x + dx6[d], ny = y + dy6[d], nz = z + dz6[d];
        if (nx < 0 || nx >= grid_x || ny < 0 || ny >= grid_y || nz < 0 || nz >= grid_z) continue;
        const int nvidx = nz * (grid_x * grid_y) + ny * grid_x + nx;
        if (vol_used[nvidx] + daughter_vol <= capacity) cand_vidx[n_cands++] = nvidx;
    }
    if (n_cands == 0) return flamegpu::ALIVE;

    // Pick one candidate deterministically and reserve it by priority.
    const int my_vidx = z * (grid_x * grid_y) + y * grid_x + x;
    const unsigned int sim_seed = FLAMEGPU->environment.getProperty<unsigned int>("sim_seed");
    const unsigned int det_tag = FLAMEGPU->getVariable<unsigned int>("det_tag");
    unsigned int drs = det_rng_seed(static_cast<int>(det_tag), FLAMEGPU->getStepCounter(), 0xF1Bu, sim_seed);
    int pick = static_cast<int>(det_rng_uniform(drs) * n_cands);
    if (pick >= n_cands) pick = n_cands - 1;
    const int tvidx = cand_vidx[pick];
    unsigned long long* owner = reinterpret_cast<unsigned long long*>(
        FLAMEGPU->environment.getProperty<unsigned long long>("voxel_owner_ptr"));
    reserve_voxel(owner, tvidx, make_claim_priority(my_vidx, det_tag));
    FLAMEGPU->setVariable<int>("div_target_vidx", tvidx);
    return flamegpu::ALIVE;
}

// Fibroblast division — CONFIRM pass. Commits the unique winner of each reserved voxel,
// then births the daughter (inherits parent state; gets a deterministic child det_tag).
FLAMEGPU_AGENT_FUNCTION(fib_divide_confirm, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int tvidx = FLAMEGPU->getVariable<int>("div_target_vidx");
    if (tvidx < 0) return flamegpu::ALIVE;

    const int cs = FLAMEGPU->getVariable<int>("cell_state");
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");

    const unsigned int det_tag = FLAMEGPU->getVariable<unsigned int>("det_tag");
    const int my_vidx = z * (grid_x * grid_y) + y * grid_x + x;
    unsigned long long* owner = reinterpret_cast<unsigned long long*>(
        FLAMEGPU->environment.getProperty<unsigned long long>("voxel_owner_ptr"));
    if (!won_voxel(owner, tvidx, make_claim_priority(my_vidx, det_tag))) {
        return flamegpu::ALIVE;  // lost the deterministic contention → don't divide this step
    }

    float* vol_used = VOL_PTR(FLAMEGPU);
    const float capacity = FLAMEGPU->environment.getProperty<float>("PARAM_VOXEL_CAPACITY");
    float daughter_vol = (cs == FIB_MYCAF) ?
        FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_FIB_MYCAF") :
        FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_FIB_ICAF");
    // Sole winner of this voxel → claim is race-free; guard against a pre-existing fill.
    if (!volume_try_claim(vol_used, tvidx, daughter_vol, capacity)) return flamegpu::ALIVE;

    const int nx = tvidx % grid_x;
    const int ny = (tvidx / grid_x) % grid_y;
    const int nz = tvidx / (grid_x * grid_y);

    // Per-subtype cooldown from QSP k_prolif
    const int cd = static_cast<int>((cs == FIB_MYCAF) ?
        FLAMEGPU->environment.getProperty<float>("PARAM_FIB_DIV_COOLDOWN_MYCAF") :
        FLAMEGPU->environment.getProperty<float>("PARAM_FIB_DIV_COOLDOWN_ICAF"));

    // Update parent
    FLAMEGPU->setVariable<int>("divide_cooldown", cd);
    FLAMEGPU->setVariable<int>("divide_count", FLAMEGPU->getVariable<int>("divide_count") + 1);

    // Create daughter agent
    auto daughter = FLAMEGPU->agent_out;
    daughter.setVariable<int>("x", nx);
    daughter.setVariable<int>("y", ny);
    daughter.setVariable<int>("z", nz);
    daughter.setVariable<int>("cell_state", cs);  // inherits parent state
    daughter.setVariable<int>("divide_flag", 0);
    daughter.setVariable<int>("divide_cooldown", cd);
    daughter.setVariable<int>("divide_count", 0);
    daughter.setVariable<int>("persist_dir_x", 0);
    daughter.setVariable<int>("persist_dir_y", 0);
    daughter.setVariable<int>("persist_dir_z", 0);
    // Child lineage tag: deterministic + distinct per birth, independent of FLAMEGPU birth order.
    const unsigned int child_tag =
        det_rng_hash(det_tag ^ (static_cast<unsigned int>(tvidx) * 0x9E3779B1U)
                     ^ (static_cast<unsigned int>(FLAMEGPU->getStepCounter()) * 0x85EBCA77U));
    daughter.setVariable<unsigned int>("det_tag", child_tag);
    // Daughter gets a FRESH exponential lifespan (mean PARAM_FIB_LIFE_MEAN), NOT the parent's
    // remaining life. Inheriting parent.life (regression from the determinism rewrite, 3d6a6849)
    // made each generation "born older" by the division cooldown, so lineages extinguished after
    // ~lifespan/cooldown divisions and the whole stroma collapsed to 0. Deterministic: seeded from
    // the child's det_tag. Matches the init lifespan sampling (life_mean·ln(1/U)).
    unsigned int life_rng = child_tag ^ 0x0F1B11FEu;
    const float fib_life_mean = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_LIFE_MEAN");
    int daughter_life = static_cast<int>(
        fib_life_mean * logf(1.0f / (det_rng_uniform(life_rng) + 1e-4f)) + 0.5f);
    daughter.setVariable<int>("life", daughter_life > 0 ? daughter_life : 1);

    // Record proliferation event
    auto* ec = reinterpret_cast<unsigned int*>(FLAMEGPU->environment.getProperty<uint64_t>("event_counters_ptr"));
    int prolif_idx = (cs == FIB_MYCAF) ? EVT_PROLIF_FIB_MYCAF : EVT_PROLIF_FIB_ICAF;
    atomicAdd(&ec[prolif_idx], 1u);
    return flamegpu::ALIVE;
}

// ============================================================================
// Fibroblast: Build Gaussian density field for ECM deposition (myCAF only)
// HIF-boosted under hypoxia, PSC metabolic decline at severe hypoxia.
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(fib_build_density_field, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int cs = FLAMEGPU->getVariable<int>("cell_state");
    if (cs != FIB_MYCAF) return flamegpu::ALIVE;  // only myCAF deposits ECM

    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int grid_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");
    const int voxel = z * (grid_x * grid_y) + y * grid_x + x;

    const int radius = static_cast<int>(FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ECM_RADIUS"));
    const float variance = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ECM_VARIANCE");

    // HIF ECM boost: hypoxic myCAFs produce more ECM, but severe hypoxia causes
    // metabolic decline (PSC self-limiting). Net effect = hif_boost * metabolic_factor.
    float ecm_multiplier = 1.0f;
    const float local_O2 = PDE_READ(FLAMEGPU, PDE_CONC_O2, voxel);
    const float hyp_th = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_HYPOXIA_TH");
    if (local_O2 < hyp_th && hyp_th > 0.0f) {
        ecm_multiplier = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_HIF_ECM_BOOST");
        // PSC metabolic decline at severe hypoxia (<50% threshold)
        const float severe_th = hyp_th * 0.5f;
        if (local_O2 < severe_th) {
            ecm_multiplier *= fmaxf(local_O2 / severe_th, 0.1f);
        }
    }

    // Normalizer for 3D Gaussian: 1 / ((2π σ²)^(3/2))
    const float normalizer = ecm_multiplier / (powf(2.0f * 3.14159265f * variance, 1.5f));

    // Fixed-point accumulation: the field is an unsigned-long-long buffer of scaled deposits,
    // so the atomicAdd sum is exact + order-independent (deterministic). Decoded back to float
    // in update_ecm_grid_kernel. See ecm_fp_encode/decode in common.cuh.
    unsigned long long* field_ptr = reinterpret_cast<unsigned long long*>(
        FLAMEGPU->environment.getProperty<uint64_t>("fib_density_field_ptr"));

    for (int dx = -radius; dx <= radius; dx++) {
        const int nx = x + dx;
        if (nx < 0 || nx >= grid_x) continue;
        for (int dy = -radius; dy <= radius; dy++) {
            const int ny = y + dy;
            if (ny < 0 || ny >= grid_y) continue;
            for (int dz = -radius; dz <= radius; dz++) {
                const int nz = z + dz;
                if (nz < 0 || nz >= grid_z) continue;
                float dist_sq = static_cast<float>(dx * dx + dy * dy + dz * dz);
                float kernel_val = normalizer * expf(-dist_sq / (2.0f * variance));
                atomicAdd(&field_ptr[nz * (grid_x * grid_y) + ny * grid_x + nx], ecm_fp_encode(kernel_val));
            }
        }
    }

    return flamegpu::ALIVE;
}

}  // namespace PDAC

#endif  // FIBROBLAST_CUH
