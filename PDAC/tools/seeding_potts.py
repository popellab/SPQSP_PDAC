"""D-9 arm A: placement by a latent neighbourhood-label (Potts) field.

The plan's rationale: "a latent neighborhood-label field (Potts-style) filled with
per-FunCN composition ... targets FunCN almost directly, since FunCN *is* a
neighborhood label." This module takes that literally.

Three steps
-----------
1. **Fit niches from the data.** Every IMC cell's stage-4 neighbourhood-composition
   vector IS its niche signature, so k-means over those vectors yields K niche
   archetypes: a composition ``comp[k]`` over ABM categories and a prevalence
   ``share[k]``. Nothing is invented — the niches are read off PASS01.
2. **Sample a 3D Potts field at finite temperature.** ``E = -β Σ_<ij> δ(L_i,L_j) -
   Σ_i h[L_i]``, with the field term ``h = log(share)`` pinning niche prevalence and β
   setting spatial coherence. **Sampled, never annealed** — §C6.3's "optimize → sample"
   is the change that matters, because virtual patients need a distribution of
   configurations and annealing collapses to one (that is exactly gate G2).
3. **Fill under the QSP ledger.** Cells are drawn from the local niche composition,
   then IPF-tilted and repaired so global counts equal the QSP box-share EXACTLY.
   C6.1 holds by construction: data sets arrangement, QSP sets totals, and G4 is an
   assertion rather than a hope.

β is calibrated, not chosen: ``calibrate_beta`` matches the sampled field's
same-label-adjacency to the value measured on the IMC niche labelling, so spatial
coherence comes from the data too.

Scored through ``seeding.py`` so both D-9 arms are judged identically — and remember
``seeding.TV_GRID_NOISE`` = 0.05: a G1 below that is fitting the voxel lattice.

Density is placed, not assumed uniform. Occupancy follows each niche's measured local
density (stage-4 ``nbhd_total_K``), so voxels are NOT required to be full and many are
empty. This is mechanistic: FunCN is a kernel-WEIGHTED sum, so a dense pure-cancer nest
dominates its own neighbourhood and pushes fractions toward 1. The target's upper tail
(cancer-in-cancer p90 ≈ 0.99) is therefore a density feature that uniform occupancy
cannot reach at any β or coarsening.
"""
from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .imc_mapping import SEED_DROP, TARGET_MAP
from .seeding import VOXEL_UM

# 20 µm voxel = 8000 µm³; PARAM_VOXEL_CAPACITY = 5920 µm³ of cell volume (0.74 FCC
# packing). At a ~1600 µm³ cell that is 3.7 → 3 cells per voxel. Structural constant.
VOXEL_CAPACITY_UM3 = 5920.0
CELL_VOLUME_UM3 = 1600.0
MAX_CELLS_PER_VOXEL = int(VOXEL_CAPACITY_UM3 // CELL_VOLUME_UM3)


@dataclass
class NicheModel:
    categories: list[str]
    comp: np.ndarray        # (K, n_cat) composition of each niche, rows sum to 1
    share: np.ndarray       # (K,) niche prevalence, sums to 1
    adjacency: float        # P(neighbouring cells share a niche label) in the data
    density: np.ndarray | None = None   # (K,) relative local cell density per niche
    # Per-niche EMPIRICAL composition vectors (the cells actually assigned to that
    # niche), not just the centroid. Virtual patients draw from these.
    pool: list[np.ndarray] | None = None
    beta: float | None = None

    @property
    def K(self) -> int:
        return len(self.share)


# ──────────────────────────────────────────────────────────────────────────
# 1. Fit niches from stage-4
# ──────────────────────────────────────────────────────────────────────────
def _cell_niche_vectors(pq: Path, col_to_abm: dict[str, str],
                        categories: list[str]) -> tuple[np.ndarray, np.ndarray]:
    d = pd.read_parquet(pq)
    keep = np.array([(a, b) not in SEED_DROP
                     for a, b in zip(d.cell_type, d.subtype)])
    keep &= ~d.nbhd_isolated.to_numpy()
    d = d[keep]
    if not len(d):
        return np.zeros((0, len(categories))), np.zeros((0, 2))
    M = np.stack([d[[k for k, v in col_to_abm.items() if v == c]].sum(axis=1).to_numpy()
                  for c in categories], axis=1)
    s = M.sum(axis=1, keepdims=True)
    M = np.divide(M, s, out=np.zeros_like(M), where=s > 0)   # placed_only
    # nbhd_total_K is the self-excluded kernel mass around each cell — stage 4's own
    # measure of LOCAL DENSITY, on exactly the kernel the targets were built with.
    return M, np.stack([d.x.to_numpy(), d.y.to_numpy()], axis=1), d.nbhd_total_K.to_numpy()


def fit_niches(stage4: Path, categories: list[str], col_to_abm: dict[str, str],
               K: int = 8, n_samples: int = 25, per_sample: int = 4000,
               seed: int = 0, pool_size: int = 3000) -> NicheModel:
    """K niche archetypes from the cells' own neighbourhood-composition vectors.

    K=8 is the knee: variance explained 0.61/0.68/0.72/0.76 at K=4/6/8/10, with the
    smallest niche still ~3% of cells at K=8 (it falls to 1.5% by K=16).
    """
    rng = np.random.default_rng(seed)
    files = sorted(glob.glob(str(stage4 / "*.parquet")))
    pick = rng.choice(files, min(n_samples, len(files)), replace=False)
    X, D, adj = [], [], []
    for f in pick:
        M, xy, kmass = _cell_niche_vectors(Path(f), col_to_abm, categories)
        if not len(M):
            continue
        idx = rng.choice(len(M), min(per_sample, len(M)), replace=False)
        X.append(M[idx])
        # Normalize per sample before pooling: absolute kernel mass varies with section
        # quality/cellularity, and only the BETWEEN-NICHE contrast should survive.
        km = kmass[idx]
        D.append(km / max(np.nanmedian(kmass), 1e-9))
        adj.append((M, xy))
    X = np.vstack(X)
    D = np.concatenate(D)
    km = KMeans(K, n_init=6, random_state=seed).fit(X)
    comp = np.clip(km.cluster_centers_, 0, None)
    comp /= comp.sum(axis=1, keepdims=True)
    share = np.bincount(km.labels_, minlength=K) / len(X)

    # Spatial coherence in the data: how often do two nearby cells share a niche?
    # Measured at one voxel (20 µm) so it matches the Potts lattice adjacency.
    from scipy.spatial import cKDTree
    same, tot = 0, 0
    for M, xy in adj[:8]:
        lab = km.predict(M)
        tree = cKDTree(xy)
        for i, j in tree.query_pairs(VOXEL_UM, output_type="ndarray")[:200000]:
            same += int(lab[i] == lab[j])
            tot += 1
    # Relative local density per niche — a tumour nest is packed, stroma is not, and
    # uniform occupancy cannot produce the dense pure-cancer neighbourhoods the target
    # tail is made of (FunCN is a kernel-weighted sum, so a dense nest dominates its
    # own neighbourhood and pushes fractions toward 1).
    dens = np.array([np.nanmean(D[km.labels_ == k]) if (km.labels_ == k).any() else 1.0
                     for k in range(K)])
    dens = np.clip(dens / np.nanmean(dens), 0.05, None)
    # Keep the within-niche spread, not only its mean. A centroid is an average, so
    # filling from it CAPS achievable neighbourhood purity: the target's cancer-in-
    # cancer p90 is 0.988 while the best K=8 centroid is 0.886, and no β, coarsening
    # or density setting can cross that ceiling. The empirical vectors span the real
    # range, so drawing from them lifts the cap without inflating K.
    pool = []
    for k in range(K):
        v = X[km.labels_ == k]
        if len(v) > pool_size:
            v = v[rng.choice(len(v), pool_size, replace=False)]
        pool.append(v if len(v) else comp[k][None, :])
    return NicheModel(categories, comp, share, same / max(tot, 1), density=dens,
                      pool=pool)


# ──────────────────────────────────────────────────────────────────────────
# 2. Sample the Potts field (finite temperature — never annealed)
# ──────────────────────────────────────────────────────────────────────────
def solve_niche_mixture(model: NicheModel, ratios: np.ndarray,
                        sum_weight: float = 50.0) -> np.ndarray:
    """Niche proportions π* whose mixture reproduces the QSP composition.

    **This is where C6.1 is enforced, and enforcing it here rather than per cell is
    the difference between the method working and not.** Tilting individual cell
    categories to hit global totals destroys the very structure being placed: measured
    on a first build, IPF at the cell level dragged niche 4's cancer_sen from its
    archetype 0.886 down to 0.664 and niche 6's from 0.520 to 0.177, flattening the
    contrast the target is made of. Solving for the MIXTURE instead leaves every niche
    interior intact — the QSP decides how much of each neighbourhood type exists, the
    data decides what each one looks like.

    NNLS on the simplex; the heavy last row pins Σπ = 1. Residual on PASS01: the raw
    data shares give ‖π·comp − ratios‖₁ = 0.118, π* gives 0.015.
    """
    from scipy.optimize import nnls

    A = np.vstack([model.comp.T, sum_weight * np.ones((1, model.K))])
    b = np.concatenate([np.asarray(ratios, float), [sum_weight]])
    pi, _ = nnls(A, b)
    pi = np.clip(pi, 0.0, None)
    s = pi.sum()
    return pi / s if s > 0 else np.full(model.K, 1.0 / model.K)


def sample_potts(shape: tuple[int, int, int], model: NicheModel, beta: float,
                 n_sweeps: int = 30, seed: int = 0,
                 init: np.ndarray | None = None,
                 share: np.ndarray | None = None) -> np.ndarray:
    """Heat-bath Gibbs sweeps over a 3D q-state Potts field. Returns (gx,gy,gz) int.

    Checkerboard updates: within one colour every site's neighbours are the other
    colour, so a whole half-lattice updates in parallel and each sweep is exact Gibbs.
    The temperature is fixed at 1 — the sampler stays in the equilibrium ensemble
    rather than descending to a mode, which is what makes between-draw spread (G2) a
    property of the model instead of an artifact of a cooling schedule.
    """
    rng = np.random.default_rng(seed)
    gx, gy, gz = shape
    K = model.K
    tgt_share = model.share if share is None else np.asarray(share, float)
    L = (rng.choice(K, size=shape, p=tgt_share) if init is None else init.copy())
    h = np.log(np.clip(tgt_share, 1e-12, None))

    ix, iy, iz = np.indices(shape)
    parity = (ix + iy + iz) % 2
    for sweep in range(n_sweeps):
        # Adaptive bias. h alone fixes only the EXPECTED label proportions; on a coarse
        # lattice (few sites, large domains) the realized proportions scatter badly —
        # a first build put 70% of cells in a niche whose share is 35%, which then
        # forced the ledger step to distort every niche. Nudging h by the observed
        # shortfall each sweep pins the realized mixture without freezing the field.
        if sweep:
            realized = np.bincount(L.reshape(-1), minlength=K) / L.size
            h += 0.5 * np.log(np.clip(tgt_share, 1e-12, None)
                              / np.clip(realized, 1e-12, None))
        for colour in (0, 1):
            m = parity == colour
            counts = np.zeros((K, gx, gy, gz), dtype=np.float32)
            for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)):
                nb = np.roll(L, sh, axis=ax)
                for k in range(K):
                    counts[k] += (nb == k)
            logits = beta * counts + h[:, None, None, None]
            logits -= logits.max(axis=0, keepdims=True)
            p = np.exp(logits)
            p /= p.sum(axis=0, keepdims=True)
            cdf = np.cumsum(p, axis=0)
            u = rng.random(shape)
            newL = (u[None] > cdf).sum(axis=0).clip(0, K - 1)
            L = np.where(m, newL, L)
    return L.astype(np.int16)


def site_ids(shape: tuple[int, int, int], coarsen: int) -> np.ndarray:
    """Coarse-site index per voxel — the unit a composition is drawn for.

    Drawing per SITE rather than per cell is what makes the empirical draw visible: a
    per-cell draw averages back to the centroid inside one kernel footprint, whereas a
    coarsen=8 site is 160 µm, comfortably larger than the ~100 µm kernel, so a drawn
    composition survives as the local neighbourhood a cell actually sees.
    """
    ix, iy, iz = np.indices(shape)
    c = max(1, coarsen)
    nx, ny = (shape[0] + c - 1) // c, (shape[1] + c - 1) // c
    return ((iz // c) * (nx * ny) + (iy // c) * nx + (ix // c)).astype(np.int64)


def sample_field(shape: tuple[int, int, int], model: NicheModel, beta: float,
                 coarsen: int = 1, n_sweeps: int = 25, seed: int = 0,
                 share: np.ndarray | None = None) -> np.ndarray:
    """Potts field on a coarsened lattice, upsampled to the voxel grid.

    **Coarsening is what gives domain size an independent knob, and without it the
    method does not work.** The FunCN kernel averages over a ~5-voxel radius (~121
    voxels), so nearest-neighbour Potts domains a few voxels across are averaged away
    and every cell's neighbourhood regresses to the global mean — measured: the
    cancer-in-cancer-neighbourhood median came out 0.50 against a target of 0.79, at
    every β from 0 to 1.2. β cannot fix it either: q=8 Potts in 3D goes from tiny
    domains straight to one system-spanning domain (adjacency 0.982 → 0.997 between
    β=0.9 and 1.2), so domain size is not independently controllable.

    Sampling on a lattice coarsened by ``coarsen`` makes one label domain ≥ the kernel
    footprint by construction: ``coarsen`` sets the scale (site = coarsen × 20 µm), β
    sets the roughness.
    """
    cs = tuple(max(2, s // coarsen) for s in shape)
    Lc = sample_potts(cs, model, beta, n_sweeps=n_sweeps, seed=seed, share=share)
    L = np.repeat(np.repeat(np.repeat(Lc, coarsen, 0), coarsen, 1), coarsen, 2)
    L = L[:shape[0], :shape[1], :shape[2]]
    pad = [(0, shape[i] - L.shape[i]) for i in range(3)]
    if any(p[1] > 0 for p in pad):
        L = np.pad(L, pad, mode="edge")
    return L


def field_adjacency(L: np.ndarray) -> float:
    """P(two lattice-adjacent sites share a label) — the statistic β is matched on."""
    s = t = 0
    for ax in (0, 1, 2):
        a = np.take(L, range(L.shape[ax] - 1), axis=ax)
        b = np.take(L, range(1, L.shape[ax]), axis=ax)
        s += int((a == b).sum())
        t += a.size
    return s / max(t, 1)


def calibrate_beta(shape: tuple[int, int, int], model: NicheModel,
                   grid: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.4),
                   n_sweeps: int = 20, seed: int = 0) -> tuple[float, pd.DataFrame]:
    """Pick β so the field's same-label adjacency matches the IMC value.

    β is a property of the tissue, not a tuning knob: coherence is measured on the
    data's own niche labelling and the field is matched to it.
    """
    rows = []
    for b in grid:
        L = sample_potts(shape, model, b, n_sweeps=n_sweeps, seed=seed)
        rows.append({"beta": b, "adjacency": field_adjacency(L)})
    df = pd.DataFrame(rows)
    best = float(df.iloc[(df.adjacency - model.adjacency).abs().argmin()].beta)
    return best, df


# ──────────────────────────────────────────────────────────────────────────
# 3. Fill under the QSP ledger
# ──────────────────────────────────────────────────────────────────────────
def fill_from_field(L: np.ndarray, model: NicheModel, counts: dict[str, int],
                    seed: int = 0, ipf_iters: int = 60,
                    sites: np.ndarray | None = None
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Place cells in the labelled field so global counts match ``counts`` EXACTLY.

    Slots are spread uniformly over voxels (capped at ``MAX_CELLS_PER_VOXEL``), each
    slot's category is drawn from its voxel's niche composition tilted by IPF to hit
    the global totals, and any residual is repaired by moving surplus cells into the
    voxels where the deficit category is most at home. Exact totals are what makes G4
    ("ABM totals equal f·X_QSP at t=0") an assertion rather than a check.
    """
    rng = np.random.default_rng(seed)
    cats = model.categories
    target = np.array([counts.get(c, 0) for c in cats], dtype=np.int64)
    N = int(target.sum())
    gx, gy, gz = L.shape
    n_vox = gx * gy * gz
    if N > n_vox * MAX_CELLS_PER_VOXEL:
        raise ValueError(f"{N} cells exceed capacity {n_vox * MAX_CELLS_PER_VOXEL} "
                         f"({MAX_CELLS_PER_VOXEL}/voxel over {n_vox} voxels)")

    flat = L.reshape(-1)
    # ── occupancy: NOT uniform, and voxels are NOT required to be full ──────
    # Cells are allocated to voxels in proportion to their niche's measured local
    # density, so packed tumour nests and sparse stroma both appear and many voxels
    # end up empty. This matters mechanistically, not cosmetically: FunCN is a
    # kernel-WEIGHTED sum, so a dense pure-cancer nest dominates its own neighbourhood
    # and drives fractions toward 1 — the target's upper tail (p90 ≈ 0.99) is a
    # density feature, and uniform occupancy cannot reach it at any β or coarsening.
    w = (np.ones(n_vox) if model.density is None else model.density[flat]).astype(float)
    w = w / w.sum()
    per_vox = rng.multinomial(N, w)
    # Respect the packing limit, spilling overflow into whatever room is left. Weighted
    # by w so the spill still prefers dense niches rather than smearing uniformly.
    for _ in range(50):
        over = per_vox - MAX_CELLS_PER_VOXEL
        excess = int(np.maximum(over, 0).sum())
        if excess == 0:
            break
        per_vox = np.minimum(per_vox, MAX_CELLS_PER_VOXEL)
        room = MAX_CELLS_PER_VOXEL - per_vox
        wr = w * (room > 0)
        if wr.sum() <= 0:
            raise ValueError("no room left to place cells at this capacity")
        add = rng.multinomial(excess, wr / wr.sum())
        per_vox = per_vox + np.minimum(add, room)
    slot_vox = np.repeat(np.arange(n_vox), per_vox)
    slot_lab = flat[slot_vox]

    # Propensities. With ``sites`` each coarse site draws ONE composition from its
    # niche's empirical pool, so domains differ from one another the way real regions
    # do; without it every domain of a niche is identical to the centroid.
    if sites is not None and model.pool is not None:
        site_of_slot = sites.reshape(-1)[slot_vox]
        P = np.empty((len(slot_vox), len(cats)))
        for sid in np.unique(site_of_slot):
            m = site_of_slot == sid
            k = int(slot_lab[m][0])
            pk = model.pool[k]
            P[m] = pk[rng.integers(len(pk))]
    else:
        P = model.comp[slot_lab]                   # (N, K_cat) propensities
    w = np.ones(len(cats))
    for _ in range(ipf_iters):
        Q = P * w
        Q /= Q.sum(axis=1, keepdims=True)
        col = Q.sum(axis=0)
        w *= np.where(col > 0, target / np.maximum(col, 1e-12), 1.0)
    Q = P * w
    Q /= Q.sum(axis=1, keepdims=True)
    draw = (rng.random(len(Q))[:, None] > np.cumsum(Q, axis=1)).sum(axis=1)
    draw = np.clip(draw, 0, len(cats) - 1)

    # Exact repair: move surplus slots to deficit categories, preferring slots whose
    # niche likes the deficit category most (so the repair follows the structure).
    for _ in range(200):
        have = np.bincount(draw, minlength=len(cats))
        diff = have - target
        if not diff.any():
            break
        over = int(np.argmax(diff))
        under = int(np.argmin(diff))
        n_move = int(min(diff[over], -diff[under]))
        if n_move <= 0:
            break
        idx = np.flatnonzero(draw == over)
        pref = model.comp[slot_lab[idx], under]
        draw[idx[np.argsort(-pref)[:n_move]]] = under

    pos = np.stack(np.unravel_index(slot_vox, L.shape), axis=1).astype(np.int64)
    return pos, np.array(cats, dtype=object)[draw]
