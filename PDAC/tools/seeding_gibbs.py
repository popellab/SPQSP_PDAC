"""D-9 arm B: placement by a multitype Gibbs point process fit to IMC pair correlations.

The counterpart to ``seeding_potts``. Where arm A models a latent neighbourhood LABEL and
fills it, arm B models the pairwise interactions between cell types directly:

    E = Σ_{i<j} φ_{t_i,t_j}(r_ij)

with φ fitted from the measured multitype pair-correlation functions and the configuration
sampled at finite temperature.

Why this is a genuinely different model, not a reparameterization of arm A: arm A can only
express structure that is a mixture of K neighbourhood archetypes, so it cannot represent an
arrangement that no single niche describes. Arm B has no niche vocabulary at all — every type
pair gets its own distance-resolved interaction. The bake-off is between "tissue is built from
a few recurring neighbourhood types" and "tissue is built from pairwise affinities".

Three things to keep honest
---------------------------
1. **PCF is not FunCN.** φ is fitted on pair correlations because that is the statistic a
   Gibbs process is naturally specified by, but the model is SCORED on the FunCN targets
   through ``seeding.py``, exactly as arm A is. Fitting and scoring statistics are kept
   separate on purpose — this is the "bridge" §C6.3 warns is needed.
2. **The potential is fitted in 2D and used in 3D.** IMC is a section, so the measured PCF is
   a 2D one; recovering the true 3D pair potential from it is an Abel-type inversion we do not
   attempt. Using the 2D-fitted φ as a 3D potential is an approximation, and it is precisely
   why the verdict comes from sectioning-then-scoring rather than from the fit residual.
3. **Counts are fixed, never sampled.** The QSP ledger sets how many cells of each type exist,
   so the sampler only ever MOVES type labels between occupied sites (a swap chain in the
   canonical ensemble). C6.1 and G4 therefore hold by construction here too — arm B cannot
   drift its composition even in principle.

The potential-of-mean-force inversion φ = -ln g is exact only in the dilute limit; in dense
tissue it over-states interaction strength. ``scale`` exists to soften it and is fitted by
scoring, not assumed.
"""
from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .imc_mapping import SEED_DROP, TARGET_MAP
from .seeding import VOXEL_UM, make_grid

# Interaction range. 200 µm = 10 voxels: beyond the FunCN kernel (100 µm) so the potential
# can express structure the target statistic can see, without an O(r³) cost blow-up.
R_MAX_VOX = 10
N_SHELL = 10


@dataclass
class PairPotential:
    categories: list[str]
    edges: np.ndarray            # (N_SHELL+1,) shell edges in voxels
    phi: np.ndarray              # (n_cat, n_cat, N_SHELL) pair potential
    g: np.ndarray                # (n_cat, n_cat, N_SHELL) measured PCF, for reporting

    @property
    def n_cat(self) -> int:
        return len(self.categories)


def _ring_kernels(edges: np.ndarray) -> list[np.ndarray]:
    """3D indicator kernels for each distance shell, as (2R+1)³ arrays."""
    R = int(edges[-1])
    ax = np.arange(-R, R + 1)
    dx, dy, dz = np.meshgrid(ax, ax, ax, indexing="ij")
    d = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    return [((d >= edges[i]) & (d < edges[i + 1])).astype(np.float32)
            for i in range(len(edges) - 1)]


def _ring_kernels_2d(edges: np.ndarray) -> list[np.ndarray]:
    R = int(edges[-1])
    ax = np.arange(-R, R + 1)
    dx, dy = np.meshgrid(ax, ax, indexing="ij")
    d = np.sqrt(dx ** 2 + dy ** 2)
    return [((d >= edges[i]) & (d < edges[i + 1])).astype(np.float32)
            for i in range(len(edges) - 1)]


def _conv2(field: np.ndarray, kern: np.ndarray) -> np.ndarray:
    gx, gy = field.shape
    R = kern.shape[0] // 2
    s = (gx + 2 * R, gy + 2 * R)
    out = np.real(np.fft.ifftn(np.fft.fftn(field, s=s) * np.fft.fftn(kern, s=s)))
    return out[R:R + gx, R:R + gy]


def fit_pair_potentials(stage4: Path, categories: list[str], col_to_abm: dict[str, str],
                        n_samples: int = 12, seed: int = 0,
                        scale: float = 1.0) -> PairPotential:
    """Measure multitype PCFs on the IMC sections and invert to a pair potential.

    g_AB(r) = (mean count of B in the shell around an A cell) / (ρ_B · shell area), pooled
    over samples, then φ = -scale·ln g. Shell counts come from FFT ring-convolutions of the
    per-type occupancy fields — the same machinery the rest of the pipeline uses.
    """
    rng = np.random.default_rng(seed)
    files = sorted(glob.glob(str(stage4 / "*.parquet")))
    pick = rng.choice(files, min(n_samples, len(files)), replace=False)
    edges = np.linspace(0.999, R_MAX_VOX, N_SHELL + 1)   # skip r=0 (self)
    kerns = _ring_kernels_2d(edges)
    n = len(categories)
    idx = {c: i for i, c in enumerate(categories)}

    obs = np.zeros((n, n, N_SHELL))     # summed observed counts
    exp = np.zeros((n, n, N_SHELL))     # summed expected-under-CSR counts
    for f in pick:
        d = pd.read_parquet(f, columns=["x", "y", "cell_type", "subtype"])
        keep = np.array([(a, b) not in SEED_DROP
                         for a, b in zip(d.cell_type, d.subtype)])
        d = d[keep]
        if len(d) < 500:
            continue
        cat = np.array([idx[TARGET_MAP[(a, b)]]
                        for a, b in zip(d.cell_type, d.subtype)])
        vx, vy, shape = make_grid(d.x.to_numpy(), d.y.to_numpy())
        fields = np.zeros((n, *shape), dtype=np.float32)
        np.add.at(fields, (cat, vx, vy), 1.0)
        area = shape[0] * shape[1]
        rho = fields.sum(axis=(1, 2)) / area
        for si, kern in enumerate(kerns):
            shell_area = kern.sum()
            conv = np.stack([_conv2(fields[j], kern) for j in range(n)])
            for i in range(n):
                m = cat == i
                if not m.any():
                    continue
                cnt = conv[:, vx[m], vy[m]].sum(axis=1)      # (n,) counts of each B
                obs[i, :, si] += cnt
                exp[i, :, si] += m.sum() * rho * shell_area
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(exp > 0, obs / np.maximum(exp, 1e-12), 1.0)
    g = np.clip(np.nan_to_num(g, nan=1.0), 1e-3, 1e3)
    phi = -scale * np.log(g)
    return PairPotential(categories, edges, phi.astype(np.float32), g)


def sample_gibbs(shape: tuple[int, int, int], pot: PairPotential,
                 counts: dict[str, int], n_sweeps: int = 12, seed: int = 0,
                 temperature: float = 1.0, refresh_every: int = 1,
                 max_cells_per_voxel: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-count multitype Gibbs sampler on the voxel lattice.

    Sites are chosen first (uniformly, capacity-capped); only TYPE LABELS move, by
    heat-bath resampling against the field

        U_t(v) = Σ_{t'} Σ_shells φ_{t,t'}(shell) · n_{t'}(shell around v)

    computed for every voxel at once by FFT — 17 forward transforms, 17² spectral
    multiplies and 17 inverse transforms per refresh, instead of an O(N·neighbours) loop.

    Counts are held exactly by resampling labels as a constrained assignment: each sweep
    proposes a large batch of PAIR SWAPS, which cannot change the composition. Finite
    temperature throughout — like arm A, this samples an ensemble rather than descending to
    a mode, because G2 is a gate and not an afterthought.
    """
    rng = np.random.default_rng(seed)
    cats = pot.categories
    n = len(cats)
    N = int(sum(counts.values()))
    gx, gy, gz = shape
    n_vox = gx * gy * gz
    if N > n_vox * max_cells_per_voxel:
        raise ValueError(f"{N} cells exceed capacity {n_vox * max_cells_per_voxel}")

    slot_vox = rng.permutation(np.repeat(np.arange(n_vox), max_cells_per_voxel))[:N]
    lab = np.repeat(np.arange(n), [counts.get(c, 0) for c in cats])
    lab = rng.permutation(lab)[:N] if len(lab) >= N else np.resize(lab, N)

    kerns = _ring_kernels(pot.edges)
    R = kerns[0].shape[0] // 2
    s = (gx + 2 * R, gy + 2 * R, gz + 2 * R)
    Kf = [np.fft.fftn(k, s=s) for k in kerns]
    vx, vy, vz = np.unravel_index(slot_vox, shape)

    for sweep in range(n_sweeps):
        if sweep % refresh_every == 0:
            occ = np.zeros((n, gx, gy, gz), dtype=np.float32)
            np.add.at(occ, (lab, vx, vy, vz), 1.0)
            Nf = [np.fft.fftn(occ[j], s=s) for j in range(n)]
            U = np.zeros((n, gx, gy, gz), dtype=np.float32)
            for si in range(N_SHELL):
                shell = np.stack([
                    np.real(np.fft.ifftn(Nf[j] * Kf[si]))[R:R + gx, R:R + gy, R:R + gz]
                    for j in range(n)])
                U += np.einsum("ij,jxyz->ixyz", pot.phi[:, :, si], shell)
            Ucell = U[:, vx, vy, vz]                      # (n_cat, N)

        # Batch of swap proposals — composition-preserving by construction.
        m = N // 2
        a = rng.integers(0, N, m)
        b = rng.integers(0, N, m)
        ta, tb = lab[a], lab[b]
        ok = ta != tb
        a, b, ta, tb = a[ok], b[ok], ta[ok], tb[ok]
        dE = (Ucell[tb, a] + Ucell[ta, b]) - (Ucell[ta, a] + Ucell[tb, b])
        acc = rng.random(len(a)) < np.exp(-np.clip(dE / temperature, -50, 50))
        a, b = a[acc], b[acc]
        # A slot must not take part in two swaps at once, or the pair bookkeeping (and
        # therefore the exact composition) breaks. Keep only proposals whose BOTH slots
        # appear exactly once in the accepted batch — vectorized; the sequential
        # first-occurrence version was the hot loop.
        touched = np.concatenate([a, b])
        uniq, cnt = np.unique(touched, return_counts=True)
        multi = np.zeros(N, dtype=bool)
        multi[uniq[cnt > 1]] = True
        keep = ~(multi[a] | multi[b])
        a, b = a[keep], b[keep]
        lab[a], lab[b] = lab[b], lab[a]

    pos = np.stack([vx, vy, vz], axis=1).astype(np.int64)
    return pos, np.array(cats, dtype=object)[lab]
