"""Shared scaffold for §C6 initialization-by-placement: targets, sectioning, gates.

Everything here is sampler-agnostic. The D-9 bake-off pits a Potts latent
neighbourhood-label field against a multitype Gibbs point process; both consume the
same targets, are scored by the same comparator, and are judged by the same G1/G2
metrics, so this module is not wasted work whichever wins.

Three decisions from §C6 are baked in here and are not the sampler's to revisit:

1. **QSP owns totals, data owns arrangement (C6.1).** ``counts_from_qsp`` turns
   stage-5 composition RATIOS into agent counts by multiplying the QSP box-share.
   No absolute density from the data ever reaches the ABM, so matching arrangement
   cannot fight the ledger.
2. **Match ``placed_only``, place no filler (user, 2026-08-07).** The ABM population
   is exactly the QSP ledger, so G4 holds by construction and no non-QSP agent
   population exists to double-specify totals. Accepted cost: the seeded tissue is
   ~1.37× denser in placed classes than real tissue, because the 27% of real cells the
   ABM cannot represent are not there to space them apart.
3. **Seed in 3D, virtually section, compare sections (C6.3).** IMC is a 2D section and
   the ABM a 3D volume; neighbourhood statistics do not transfer between them — a 2D
   cut through a 3D clustered field yields different neighbour counts than the 3D
   field. The stage-5 targets are section statistics, so ``virtual_section`` reduces a
   3D configuration the same way before anything is compared.

σ here is the PLACEMENT σ (30 µm, σ_vox 1.5) and is deliberately distinct from the
coupling φ-estimator σ — contract §4.4 separates the two statistics and they must not
collapse into one.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .funcn_core import build_kde_field, funcn_phi, k0_for

# Locked geometry (§C6.7). Voxel size is permanent; σ_vox 1.5 clears the §4.4
# artifact regime at σ_vox 0.5 where FunCN measures same-voxel co-occupancy.
VOXEL_UM = 20.0
SIGMA_UM = 30.0
SIGMA_VOX = SIGMA_UM / VOXEL_UM
BIN_W = 0.025
BINS = np.round(np.arange(0.0, 1.0 + BIN_W, BIN_W), 6)

NORM = "placed_only"          # decision 2 above
BACKGROUND = "background"

# ── The resolution floor on any FunCN comparison ────────────────────────────
# Re-binning the SAME cells onto a grid whose origin is shifted by a sub-voxel
# amount changes the target×source histograms by TV ≈ 0.046-0.057 (measured, 8
# random offsets, samples 100-JHU-016 / 100-NH-054). Between-PATIENT distance is
# TV ≈ 0.22-0.24. So the discretization artifact is ~25% of the real signal:
#
#   * G1 cannot be scored below this. A sampler reporting TV < 0.05 is fitting the
#     voxel lattice, not the tissue.
#   * The D-9 bake-off must not separate two samplers on differences of this size.
#   * It is also why §C6.7 puts 10 µm sub-voxel jitter on the ABM side — jitter
#     averages over exactly this offset dependence.
#
# Verified: with the grid convention below, the comparator reproduces the stage-5
# targets EXACTLY (TV = 0.000000) when fed a sample's own cells.
TV_GRID_NOISE = 0.05


def make_grid(x: np.ndarray, y: np.ndarray,
              voxel_um: float = VOXEL_UM) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Voxelize µm coordinates with the pipeline's grid convention.

    Origin at the min corner of the cells being gridded, ``floor`` binning, extent
    from the max index. Fixing this in one place matters: deriving the origin from a
    different cell subset (e.g. placed-only vs all-non-artifact) shifts every voxel
    index and moves the histograms by TV ≈ 0.05 — as large as a fifth of the
    between-patient signal, and silent.
    """
    vx = np.floor((np.asarray(x) - np.min(x)) / voxel_um).astype(np.int64)
    vy = np.floor((np.asarray(y) - np.min(y)) / voxel_um).astype(np.int64)
    return vx, vy, (int(vx.max()) + 1, int(vy.max()) + 1)


# ──────────────────────────────────────────────────────────────────────────
# Targets
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class SeedTarget:
    """Stage-5 placement targets for one patient (or the cohort)."""

    sample: str
    categories: list[str]                  # ABM categories, fixed order
    ratios: np.ndarray                     # (n_cat,) composition shares, sum 1
    hist: np.ndarray                       # (n_cat, n_cat, n_bin) target FunCN histograms
    hist_valid: np.ndarray                 # (n_cat, n_cat) bool — enough target cells
    phi: np.ndarray                        # (n_cat, n_cat) enrichment, diagnostic only

    @property
    def n_cat(self) -> int:
        return len(self.categories)


def load_targets(stage5: Path, sample: str | None = None,
                 categories: list[str] | None = None) -> SeedTarget:
    """Load one sample's targets, or the cohort pool when ``sample`` is None.

    Pairs with too few target cells are NaN in stage 5 (never zero) and come back
    ``hist_valid=False``: 3,689 such cells cohort-wide, concentrated in exactly the
    rare high-signal categories (bcell_naive, treg_treg, fib_iCAF, tcell_eff). The
    sampler MUST fall back to the cohort target for those rather than treating an
    absent histogram as "no neighbours of this class".
    """
    summ = pd.read_csv(stage5 / "funcn_target_summary.csv")
    hist = pd.read_parquet(stage5 / "funcn_histogram_bins.parquet")
    comp = pd.read_csv(stage5 / "composition_per_sample.csv")
    summ = summ[summ.norm == NORM]
    hist = hist[hist.norm == NORM]
    if sample is not None:
        summ, hist = summ[summ["sample"] == sample], hist[hist["sample"] == sample]
        comp = comp[comp["sample"] == sample]
        if not len(summ):
            raise KeyError(f"no stage-5 targets for sample {sample!r}")

    cats = categories or sorted(c.replace("ratio__", "")
                                for c in comp.columns if c.startswith("ratio__"))
    idx = {c: i for i, c in enumerate(cats)}
    n, nb = len(cats), len(BINS) - 1

    ratios = np.array([comp[f"ratio__{c}"].mean() for c in cats], dtype=np.float64)
    ratios = ratios / ratios.sum()

    H = np.zeros((n, n, nb))
    for t, s, bl, cnt in zip(hist.target, hist.source, hist.bin_left, hist["count"]):
        if t in idx and s in idx:
            H[idx[t], idx[s], min(int(round(bl / BIN_W)), nb - 1)] += cnt
    tot = H.sum(axis=2, keepdims=True)
    valid = tot[:, :, 0] > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        H = np.where(tot > 0, H / np.maximum(tot, 1), 0.0)

    P = np.full((n, n), np.nan)
    for t, g in summ.groupby("target"):
        if t not in idx:
            continue
        for s, gg in g.groupby("source"):
            if s in idx:
                P[idx[t], idx[s]] = gg.phi.mean()

    return SeedTarget(sample or "COHORT", cats, ratios, H, valid, P)


def sample_organization(stage5: Path, seed: int,
                        categories: list[str] | None = None) -> SeedTarget:
    """Draw a synthetic composition from the cohort's ALR-normal distribution.

    Instead of picking a discrete patient or using the cohort mean, this fits a
    multivariate normal in additive-log-ratio (ALR) space to the 95 PASS-01
    patients and draws one sample.  The draw is deterministic given ``seed``.

    The FunCN histograms and phi come from the cohort mean (same as
    ``load_targets(sample=None)``), since a synthetic patient has no real
    spatial data.  Only the composition ratios are drawn.
    """
    cohort = load_targets(stage5, sample=None, categories=categories)
    comp = pd.read_csv(stage5 / "composition_per_sample.csv")
    cats = cohort.categories

    R = np.column_stack([comp[f"ratio__{c}"].values for c in cats])
    R = np.clip(R, 1e-12, None)
    R = R / R.sum(axis=1, keepdims=True)

    # ALR with the largest-mean category as reference (numerically stable)
    ref = int(np.argmax(R.mean(axis=0)))
    X = np.log(R[:, np.arange(len(cats)) != ref] / R[:, ref:ref + 1])

    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)

    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal(mu, cov)

    # Invert ALR
    ratios_full = np.ones(len(cats))
    j = 0
    for i in range(len(cats)):
        if i == ref:
            continue
        ratios_full[i] = np.exp(z[j])
        j += 1
    ratios_full = ratios_full / ratios_full.sum()

    return SeedTarget(
        sample=f"alr_draw_seed={seed}",
        categories=cats,
        ratios=ratios_full,
        hist=cohort.hist,
        hist_valid=cohort.hist_valid,
        phi=cohort.phi,
    )


def counts_from_qsp(target: SeedTarget, qsp_totals: dict[str, float],
                    box_share: float) -> dict[str, int]:
    """C6.1: agent counts = QSP box-share totals, apportioned by data ratios.

    ``qsp_totals`` maps ABM category → whole-tumour cell count from the QSP;
    ``box_share`` is f = min(1, V_box/V_T). Categories the QSP does not supply fall
    back to the data ratio applied to the summed QSP total, so a missing QSP species
    cannot silently zero a category.
    """
    known = {c: qsp_totals[c] for c in target.categories if c in qsp_totals}
    out = {c: int(round(box_share * v)) for c, v in known.items()}
    missing = [c for c in target.categories if c not in known]
    if missing:
        pool = box_share * sum(known.values())
        for c in missing:
            r = target.ratios[target.categories.index(c)]
            out[c] = int(round(pool * r))
    return out


# ──────────────────────────────────────────────────────────────────────────
# 3D configuration → 2D virtual section → comparable statistics
# ──────────────────────────────────────────────────────────────────────────
def virtual_section(pos: np.ndarray, cat: np.ndarray, z: int,
                    thickness_vox: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Take a slab of a 3D configuration and flatten it to a 2D section.

    A 2D cut through a 3D clustered field has different neighbour statistics than the
    field itself, so the ABM must be sectioned before being compared to IMC. Cells
    within the slab are projected to z=0 — which is what a stained section does to
    everything within its physical thickness.
    """
    lo = z - thickness_vox // 2
    m = (pos[:, 2] >= lo) & (pos[:, 2] < lo + thickness_vox)
    p = pos[m].copy()
    p[:, 2] = 0
    return p, cat[m]


def section_funcn(pos2d: np.ndarray, cat: np.ndarray, categories: list[str],
                  shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell neighbourhood fractions for a 2D section — the stage-4 estimator.

    Returns ``(frac, keep)`` where ``frac`` is (n_kept, n_cat), self-excluded and
    renormalized so rows sum to 1, and ``keep`` is the boolean mask of cells that had
    any neighbour. **The mask is returned, not applied silently**: isolated cells are
    dropped from ``frac``, so a caller that pairs ``frac`` with the unmasked labels
    misaligns every row after the first isolated cell — a corruption that looks like a
    modest histogram mismatch rather than an error. Always index labels with ``keep``.

    Uses ``funcn_core`` with the placement σ so the simulated statistic is computed
    exactly the way the target was.
    """
    gx, gy = shape
    n = len(categories)
    if len(pos2d) == 0:
        return np.zeros((0, n)), np.zeros(0, dtype=bool)
    k0 = k0_for(SIGMA_VOX)
    num = np.zeros((len(pos2d), n))
    for i, c in enumerate(categories):
        m = cat == c
        fld = build_kde_field(pos2d[m] if m.any() else np.empty((0, 3), np.int64),
                              (gx, gy, 1), SIGMA_VOX)
        num[:, i] = fld[pos2d[:, 0], pos2d[:, 1], 0]
    idx = {c: i for i, c in enumerate(categories)}
    own = np.array([idx.get(c, -1) for c in cat])
    if (own < 0).any():
        raise ValueError("section_funcn: cell category outside `categories`")
    num[np.arange(len(pos2d)), own] -= k0
    denom = num.sum(axis=1)
    ok = denom > 1e-6
    out = np.zeros_like(num)
    out[ok] = np.clip(num[ok] / denom[ok, None], 0.0, 1.0)
    return out[ok], ok


def section_phi(pos2d: np.ndarray, cat: np.ndarray, categories: list[str],
                shape: tuple[int, int]) -> np.ndarray:
    """Permutation-based phi for a 2D virtual section.

    Calls ``funcn_phi`` with the placement σ. phi = T_obs / E_perm[T] where
    E_perm is the closed-form label-permutation expectation (see funcn_core.py).
    """
    gx, gy = shape
    if len(pos2d) == 0:
        return np.full((len(categories), len(categories)), np.nan)
    pos_by_cat = {}
    for c in categories:
        m = cat == c
        if m.any():
            pos_by_cat[c] = pos2d[m]
    return funcn_phi(pos_by_cat, categories, (gx, gy, 1), SIGMA_VOX)


def funcn_histograms(frac: np.ndarray, cat_of_cell: np.ndarray,
                     categories: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(n_cat, n_cat, n_bin) normalized histograms + (n_cat, n_cat) validity."""
    n, nb = len(categories), len(BINS) - 1
    H = np.zeros((n, n, nb))
    valid = np.zeros((n, n), dtype=bool)
    for i, t in enumerate(categories):
        m = cat_of_cell == t
        if m.sum() < 10:                       # same floor stage 5 used
            continue
        for j in range(n):
            cnt, _ = np.histogram(np.clip(frac[m, j], 0, 1), bins=BINS)
            s = cnt.sum()
            if s:
                H[i, j] = cnt / s
                valid[i, j] = True
    return H, valid


# ──────────────────────────────────────────────────────────────────────────
# Gates
# ──────────────────────────────────────────────────────────────────────────
def tv_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Total-variation (½·L1) distance per (target, source) pair. 0 = identical, 1 = disjoint."""
    return 0.5 * np.abs(a - b).sum(axis=-1)


def g1_fidelity(sim_h: np.ndarray, sim_valid: np.ndarray, target: SeedTarget,
                sim_counts: dict[str, int]) -> dict:
    """G1: does the placed state reproduce the target histograms and composition?

    Only pairs valid on BOTH sides are scored — an invalid pair is missing data, and
    scoring it as a perfect or a maximal mismatch would both be wrong.
    """
    both = sim_valid & target.hist_valid
    tv = tv_distance(sim_h, target.hist)
    n_tot = max(sum(sim_counts.values()), 1)
    sim_r = np.array([sim_counts.get(c, 0) / n_tot for c in target.categories])
    tv_mean = float(tv[both].mean()) if both.any() else np.nan
    return {
        "n_pairs_scored": int(both.sum()),
        "n_pairs_unscorable": int((~both).sum()),
        "tv_mean": tv_mean,
        "tv_p90": float(np.quantile(tv[both], 0.9)) if both.any() else np.nan,
        "tv_max": float(tv[both].max()) if both.any() else np.nan,
        "composition_l1": float(np.abs(sim_r - target.ratios).sum()),
        # Multiples of the discretization floor. <1 means the fit is inside the
        # binning artifact and is not evidence of anything.
        "tv_mean_over_noise": float(tv_mean / TV_GRID_NOISE),
        "at_noise_floor": bool(tv_mean <= TV_GRID_NOISE),
    }


def g2_variability(draw_hists: list[np.ndarray], draw_valid: list[np.ndarray],
                   patient_hists: list[np.ndarray],
                   patient_valid: list[np.ndarray]) -> dict:
    """G2: is between-DRAW spread comparable to between-PATIENT spread?

    The decisive gate for D-9. Annealing collapses to a point (ratio → 0); an
    over-loose sampler inflates it (ratio ≫ 1). Spread is measured as the mean
    pairwise TV distance within each set, over pairs valid everywhere in that set.
    """
    def spread(hs, vs):
        if len(hs) < 2:
            return np.nan
        ok = np.logical_and.reduce(vs)
        if not ok.any():
            return np.nan
        d = [float(tv_distance(hs[i], hs[j])[ok].mean())
             for i in range(len(hs)) for j in range(i + 1, len(hs))]
        return float(np.mean(d))

    dr, pt = spread(draw_hists, draw_valid), spread(patient_hists, patient_valid)
    return {"between_draw_spread": dr, "between_patient_spread": pt,
            "ratio": float(dr / pt) if pt and np.isfinite(pt) and pt > 0 else np.nan}
