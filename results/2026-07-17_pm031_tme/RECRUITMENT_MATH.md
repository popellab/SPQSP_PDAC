# Recruitment rewrite — mathematical framework (pre-code, 2026-07-17)

Goal (user): the ABM should **track the QSP abundance trajectories** throughout the sim.
That means each ABM recruitment probability must match, mathematically, the QSP
central→tumor transport flux, integrated over the domain and the 6-hour timestep — with a
per-voxel probability that scales with voxel volume (finer/larger grid → smaller per-voxel
prob, same total). This note derives the target, audits the current code, and lists what to
settle before touching anything. **No code was changed.** Companion to `DIAGNOSIS.md`
(which found the immune deficit is domain saturation → Vvas≈0 → zero entry points).

## 1. Ground truth — QSP central→tumor transport law (CD8/Teff)

Reaction `V_C.CD8 -> V_T.CD8` (qspio-pdac `Teff_module.m:255`):
```
rate = q_CD8_T_in * V_T * V_C.CD8 * (C_total^2/(C_total^2 + K_C2_recruit)) * (1 - H_CXCL12_Texcl)
```
Φ_CD8 = q · V_T · N_C.CD8 · B · X   [cells/min], where
- `q_CD8_T_in` [1/(cm^3·min)] = k_CD8_mig·rho_adh·gamma_T  (`parameters_PDAC.m:320-324`)
- `V_T` tumour volume [cm^3];  `N_C.CD8` central amount [cells] (= moles × Avogadro)
- B = C_total^2/(C_total^2+K_C2_recruit)  (burden Hill ∈[0,1])
- X = 1 − H_CXCL12_Texcl  (CXCL12 exclusion ∈[0,1])
Species are in `'cell'` amount units; `getSpeciesVar(idx, false)` returns the raw solver
state = **moles** (hence the ×AVOGADROS in the ABM).

## 2. Target per-voxel probability

Per unit tumour volume (V_T CANCELS):
  φ = Φ/V_T = q · N_C.CD8 · B · X    [cells/(cm^3·min)]
Expected recruits into one voxel over dt, as a Bernoulli prob (valid while ≪1):

  **p_voxel = q_CD8_T_in · N_C.CD8 · B · X · v_vox · dt**

## 3. The domain-scaling IS the v_vox factor (matches the user's requirement)

- p_voxel ∝ v_vox → finer/larger grid (same physical volume) ⇒ smaller per-voxel prob.
- Total over domain = p_voxel·N_vox = φ·(N_vox·v_vox)·dt = φ·V_domain·dt = the QSP flux
  integrated over the represented volume and the 6h step. Resolution-invariant. ✓
- CAVEAT: holds only if p_voxel is applied at EVERY eligible voxel. If only N_ports voxels
  recruit, rescale p_port = p_voxel·(N_vox/N_ports) to conserve the total. The port/placement
  mechanism may change WHERE, never HOW MANY.

## 4. Audit — current ABM (pde_integration.cu)

Final per-voxel prob (`:969`):  p_final = p_entry · min(p_teff · eff_gate, 1)
- p_teff = N_C.CD8 · RECRUIT_K,  RECRUIT_K = q · dt · Avogadro · v_vox · REC_PORT
         ⇒ p_teff = q · N_C.CD8 · v_vox · dt · REC_PORT   (`derived_params.inc`)
- eff_gate = cxcl9_gate · (1 − cxcl12_inhib)
- p_entry  = min(Vvas · PARAM_ENTRY_ADHESION_SCALE, 1),  ENTRY_ADHESION_SCALE = 0.05

| target term | ABM | verdict |
|---|---|---|
| q_CD8_T_in | ✓ in RECRUIT_K | match |
| N_C.CD8 [cells] | ✓ central × Avogadro | match (moles confirmed) |
| v_vox | ✓ (voxel_size/1e6)^3 | match IF units consistent — VERIFY |
| dt | ✓ t_step_sec (21600) | ⚠ q is per-MINUTE, dt in SEC → ×60 unless q pre-converted |
| B (burden Hill) | ✗ MISSING | QSP recruits ↑ with burden; ABM omits it |
| X (CXCL12 excl) | ✓ (1−cxcl12_inhib) in eff_gate | match |
| — | REC_PORT extra | reconcile vs the N_vox/N_ports rescale |
| — | cxcl9_gate extra | rebuild addition, no QSP basis |
| — | **p_entry (Vvas·0.05) MULTIPLIER** | ✗✗ THE BREAK |

## 5. Core finding

The RATE correspondence is essentially ALREADY correct: p_teff = q·N_C·v_vox·dt·REC_PORT
matches the target up to REC_PORT + the missing burden Hill, and v_vox already encodes the
domain-scaling. **The break is STRUCTURAL:** p_final = p_entry · p_teff multiplies the
correct rate by an ABSOLUTE spatial throttle (Vvas·0.05) that has NO QSP counterpart. Vvas =
hill(VEGF)·(1−f_cancer) → 0 in cancer-dense voxels and ×0.05 elsewhere, so Σ p_final ≪ QSP
total. This is BOTH the under-recruitment AND (via 1−f_cancer→0) the saturation zero.

**Fix concept (makes the earlier instincts rigorous):** the spatial gate must be a
NORMALIZED placement weight w(voxel), ⟨w⟩=1 over eligible voxels, NOT an absolute multiplier:
  p_voxel = p_teff_rate · w(voxel),   ⟨w⟩ = 1
Then Σ p_voxel = QSP total (ABM tracks QSP abundance), while cells are placed preferentially
where w is high (vascular/margin) and less — but NEVER zero — in the dense core. The user's
"sub-linear (1−f_cancer)" is exactly w ∝ (1−f_cancer)^p renormalized to mean 1.

## 6. To settle BEFORE code (next session)

1. **What does the ABM domain represent?** V_T-cancels ⇒ a FIXED physical sub-volume
   (intensive p_voxel, total ∝ domain). The presim instead treats it as the whole tumour →
   the saturation bug. Same decision drives BOTH the recruitment normalization AND the
   presim-stop fix — they are one question. (Ties to the domain-size change already queued.)
2. **Verify unit factors** against the codegen'd QSPParam values (NOT the .m): q per-minute
   vs dt in seconds (×60); (voxel_size/1e6)^3 looks like m^3 while q is per-cm^3 (×1e6). If
   unreconciled, absolute rate is off ~1e7 and nothing downstream matters. Check first.
3. **Add or justify dropping B** (burden Hill). At grown steady state B≈1, so likely 2nd-order.
4. **Myeloid arm = separate derivation.** Mac/MDSC/DC don't read a QSP central pool — they use
   a CONSTANT RECRUIT_K, while the QSP grows V_T.Mac_M2 via its own reaction. Find that
   reaction's rate law and give it the same treatment (it's ALSO gated by the broken p_entry).
5. **Repeat §1–2 for Treg and Th** — same shape, different modulators (Treg uses CCL5 not
   CXCL12). Each needs its own w and gate mapping.

## 7. Related queued changes (from Jul-17 EOD, `DIAGNOSIS.md`)
- Soften `(1−f_cancer)` in `compute_vvas_kernel` (~pde_integration.cu:2149-2151) to a
  sub-linear roll-off — now formalized as the normalized weight w above.
- Domain thickness: the thin 5-voxel z forces lateral outgrowth → fast saturation. Ties to
  decision (1). NB grid geometry ripples to simulator.py grid + the observable mid-z slice.
