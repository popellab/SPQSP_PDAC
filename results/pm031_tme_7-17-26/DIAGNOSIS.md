# PM-031 presim-TME sweep — diagnosis of the immune deficit

**2026-07-17.** The sweep asked: does the ABM's grown pre-treatment TME ever reproduce
PM-031's measured immune abundances? Answer: no — the myeloid + regulatory compartment
is a systematic miss. This note localizes *where* the deficit is.

## The split: QSP vs. recruitment-to-ABM

The QSP→ABM coupling for T cells is 3 scalars (`V_C.CD8/Treg/Th` central → recruitment
probability); myeloid (mac/MDSC/DC) recruit at constant rate gated on CCL2/Vvas. To tell
whether the immune deficit is upstream (QSP) or in the ABM recruitment step, three
immune-empty grown thetas were re-run locally with the QSP trajectory retained (the
production run deleted `qsp_presim.csv` — a gap in `run_shard_pm031.py`'s
`keep_only_wanted`; QSP is CPU-cheap to regenerate, and the manifest is deterministic).

**The QSP is healthy — it grows a full immune infiltrate.** presim=stop QSP tumour pools
for three thetas that were ALL immune-empty in the ABM:

| theta | V_T.Mac_M2 | V_T.Mac_M1 | V_T.MDSC | V_T.Treg | V_T.cDC1 | V_C.CD8 (scalar ABM reads) |
|-------|-----------:|-----------:|---------:|---------:|---------:|---------------------------:|
| 2149  |         18 |        299 |      143 |       33 |      222 |                     221.4 |
| 495   |      1,251 |      3,837 |    1,907 |   10,400 |       79 |                      40.7 |
| 236   |    160,838 |    469,544 |   10,223 |  217,687 |      763 |                  69,260.9 |

Across a **4-order-of-magnitude** range of QSP immune content, the ABM recruited
essentially none.

**The chemical fields the recruitment gate reads are also healthy** (theta 495, saved
PDE presim=stop snapshot):

| field | role | mean | coverage |
|-------|------|-----:|---------:|
| CCL2  | myeloid recruitment gate | 78 | 100% of voxels |
| VEGFA | drives Vvas | 16,300 | 100% |
| CXCL9/10 (chan 18) | effector gate (CAF-derived) | 1,449 | 100% |
| CXCL12 | effector exclusion | 23 | 100% |

(Correction: there are 19 PDE substrates — `CXCL9_10` is index 18, added in the
recruitment rebuild. The "channel 18 = 1,449" value is CXCL9/10, **not** Vvas. Vvas is a
separate `d_vvas_field`, not part of the substrate array and not exported to the PDE
snapshot. An earlier draft mislabeled it and inferred a Vvas scale error — retracted.
The `compute_vvas_kernel` output is `hill(VEGF)·(1−f_cancer)·cabo`, correctly in [0,1].)

## Root cause — TUMOR SATURATES THE SECTION, so Vvas→0, so no entry points

The saved `stats_495.csv` (presim=stop) shows the mechanism directly:

| stat | value |
|------|------:|
| agentCount cancer (stem+prog+sen) | **80,230** on a 100×100×5 = 50,000-voxel section |
| agentCount MDSC / MAC-M2 / all others | 11 / 4 / 0 |
| **recruit.t_sources / mac_sources / mdsc_sources / dc_sources** | **0 / 0 / 0 / 0** |

The chain, confirmed by data:
1. The coupled presim grows one cancer stem to the QSP tumour-volume target (PM-031 =
   5 cm diameter). On the fixed 100×100×5 section that target is reached only when
   cancer **saturates the domain** (80k cancer cells >> 50k voxels).
2. `Vvas = hill(VEGF)·(1−f_cancer)·cabo`. With the Moore-n=2 cancer fraction `f_cancer ≈ 1`
   in every voxel of a saturated section, **Vvas ≈ 0 everywhere**.
3. `mark_entry_points` requires `Vvas > 0` (`pde_integration.cu:2252`), so **zero entry
   points are marked** (recruit.*_sources = 0).
4. No entry points → no immune recruitment. The 11 MDSC / 4 M2 still present were
   recruited EARLY, while the tumour was small and a non-cancer margin still existed;
   as cancer fills the section, entry shuts off and immune fraction decays to ~0.

`PARAM_ENTRY_ADHESION_SCALE = 0.05` compounds this at the margin (p_entry = min(Vvas·0.05,
1) — only ~2.5% per step even where Vvas≈0.5), but the dominant effect is the saturation:
there is nowhere left with Vvas>0.

**This is a coupling/geometry mismatch, not a broken kernel.** The presim grows a *solid*
cancer clone to a whole-tumour volume on a *tiny fixed section*, producing a cancer blob
where the real IMC section is a cancer/stroma/immune mixture (PM-031: 62% cancer, 10%
M2). The Vvas-gated recruitment is doing exactly what it's told; the problem is that the
presim's grow-one-stem-to-volume design leaves no room for the TME it is supposed to
reproduce.

Notes:
- Myeloid-first pattern (mac_M2, DC, MDSC biggest misses) is consistent: fibroblasts
  deposit locally and don't need a Vvas entry point, so they survive saturation;
  everything routed through `mark_entry_points` dies with it.
- `vas_phalanx` is separately a structural zero (vascular agents dropped in the rebuild).

## Fix directions (for discussion, not yet done)
- **Decouple the presim stop from full domain saturation.** Options: stop growth at a
  target cancer *density/occupancy* rather than the QSP whole-tumour volume; or seed the
  section as a representative sub-region (mixed) rather than growing one stem to fill it.
- **Reconsider the `(1−f_cancer)` gate for a section that is meant to be intratumoral.**
  A biopsy core deep in a tumour is mostly cancer yet still has interspersed immune;
  gating entry on the *absence* of cancer makes intratumoral immune structurally
  impossible once the core is cancer-dense. The immune-entry field may need to allow
  entry within cancer regions (perietal capillaries / interstitial entry), not only at
  the cancer-free margin.
- `PARAM_ENTRY_ADHESION_SCALE = 0.05` is a secondary throttle worth revisiting once the
  saturation issue is addressed.

## Reproduce
- QSP re-run: `scratchpad/qsp_rerun/theta_{495,236,2149}/qsp_presim_*.csv`
- picked thetas: immune-empty grown, from `composition.npz` (see `build_composition.py`)
- manifest regenerated deterministically: `make_manifest_pm031.py --n 10000 --seed 0`
