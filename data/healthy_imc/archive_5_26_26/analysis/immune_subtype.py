"""
immune_subtype.py — Gate IMC immune cells into ABM-relevant subtypes.

For each cells.h5ad, adds obs['cell_subtype'] by gating the CD45+ population
using global GMM thresholds fit across all ROIs.

Subtypes (priority order):
  macrophage  — CD68+
  cd8_t       — CD3+ CD8+
  cd4_t       — CD3+ CD4+ CD8-
  b_cell      — CD20+ CD3-
  mdsc        — CD11b+ CD14+  (or CD14+ CD68- where no CD11b)
  other_immune — remaining CD45+

Non-immune cells retain their broad label from the pipeline.

Usage:
    python immune_subtype.py [--pancdb-imc DIR] [--dry-run]

Outputs:
    analysis/gating_thresholds.json   — GMM thresholds per marker
    analysis/subtype_counts.csv       — per-ROI cell counts by subtype
    (h5ad files updated in-place with obs['cell_subtype'])
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
from sklearn.mixture import GaussianMixture

# ── Marker lookup ──────────────────────────────────────────────────────────────

# Biological name → possible var_name suffixes (after the isotope_ prefix)
MARKER_SUFFIXES = {
    "CD45":  ["CD45"],
    "CD3":   ["CD3"],
    "CD4":   ["CD4"],
    "CD8":   ["CD8", "CD8a"],
    "CD20":  ["CD20"],
    "CD68":  ["CD68"],
    "CD11b": ["CD11b"],
    "CD14":  ["CD14"],
    "Foxp3": ["Foxp3"],
}

GATING_MARKERS = ["CD3", "CD4", "CD8", "CD20", "CD68", "CD11b", "CD14"]


def find_var(var_names: list[str], suffixes: list[str]) -> str | None:
    """Return the first var_name whose suffix matches one of the given suffixes."""
    for s in suffixes:
        for v in var_names:
            if v.endswith("_" + s):
                return v
    return None


# ── Threshold fitting ──────────────────────────────────────────────────────────

def fit_gmm_threshold(values: np.ndarray, marker: str) -> float:
    """Fit 2-component GMM, return crossing point between the two Gaussians."""
    values = values[np.isfinite(values)].reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=42, max_iter=200)
    gmm.fit(values)

    # Order components by mean
    order = np.argsort(gmm.means_.ravel())
    mu0, mu1 = gmm.means_.ravel()[order]
    sig0 = np.sqrt(gmm.covariances_.ravel()[order[0]])
    sig1 = np.sqrt(gmm.covariances_.ravel()[order[1]])

    # Find intersection: scan between means
    xs = np.linspace(mu0, mu1, 1000)
    d0 = gmm.weights_[order[0]] * _gauss(xs, mu0, sig0)
    d1 = gmm.weights_[order[1]] * _gauss(xs, mu1, sig1)
    cross_idx = np.argmin(np.abs(d0 - d1))
    threshold = float(xs[cross_idx])
    print(f"  {marker}: mu_neg={mu0:.3f} mu_pos={mu1:.3f} threshold={threshold:.3f}")
    return threshold


def _gauss(x, mu, sig):
    return np.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))


def fit_thresholds(h5ad_paths: list[Path]) -> dict[str, float]:
    """Pool all immune cells across ROIs and fit GMM thresholds per marker."""
    print(f"Fitting thresholds across {len(h5ad_paths)} ROIs...")
    pooled: dict[str, list[np.ndarray]] = {m: [] for m in GATING_MARKERS}

    for path in h5ad_paths:
        adata = ad.read_h5ad(path, backed="r")
        immune_mask = adata.obs["cell_type"] == "immune"
        if immune_mask.sum() < 10:
            continue
        X_immune = adata[immune_mask].X
        var_names = list(adata.var_names)

        for marker in GATING_MARKERS:
            var = find_var(var_names, MARKER_SUFFIXES[marker])
            if var is None:
                continue
            idx = var_names.index(var)
            pooled[marker].append(np.asarray(X_immune[:, idx]).ravel())

    thresholds = {}
    for marker, arrays in pooled.items():
        if not arrays:
            print(f"  {marker}: NO DATA — skipping")
            continue
        combined = np.concatenate(arrays)
        print(f"  {marker}: {len(combined):,} cells pooled")
        thresholds[marker] = fit_gmm_threshold(combined, marker)

    return thresholds


# ── Per-ROI gating ─────────────────────────────────────────────────────────────

def gate_roi(adata: ad.AnnData, thresholds: dict[str, float]) -> pd.Series:
    """
    Return a Series of cell_subtype labels for every cell in adata.
    Non-immune cells get their existing cell_type label.
    """
    var_names = list(adata.var_names)
    n = adata.n_obs
    subtypes = adata.obs["cell_type"].astype(str).copy()

    # Only work on immune cells
    immune_mask = (adata.obs["cell_type"] == "immune").values

    def expr(marker):
        """Return expression array for marker (zeros if not present)."""
        var = find_var(var_names, MARKER_SUFFIXES[marker])
        if var is None:
            return np.zeros(n)
        idx = var_names.index(var)
        return np.asarray(adata.X[:, idx]).ravel()

    def pos(marker):
        """Boolean mask: cells positive for marker."""
        thr = thresholds.get(marker)
        if thr is None:
            return np.zeros(n, dtype=bool)
        return expr(marker) >= thr

    cd68  = pos("CD68")
    cd3   = pos("CD3")
    cd4   = pos("CD4")
    cd8   = pos("CD8")
    cd20  = pos("CD20")
    cd11b = pos("CD11b")
    cd14  = pos("CD14")

    has_cd11b = find_var(var_names, MARKER_SUFFIXES["CD11b"]) is not None

    # Priority gating (within immune cells only)
    assigned = np.full(n, False)

    def assign(mask, label):
        sel = immune_mask & mask & ~assigned
        subtypes.iloc[np.where(sel)[0]] = label
        assigned[sel] = True

    assign(cd68,                             "macrophage")
    assign(cd3 & cd8,                        "cd8_t")
    assign(cd3 & cd4 & ~cd8,                 "cd4_t")
    assign(cd20 & ~cd3,                      "b_cell")
    if has_cd11b:
        assign(cd11b & cd14 & ~cd68,         "mdsc")
    else:
        # CD163-panel donors: use CD14+ CD68- as MDSC proxy
        assign(cd14 & ~cd68,                 "mdsc")

    # Remaining immune cells
    subtypes.iloc[np.where(immune_mask & ~assigned)[0]] = "other_immune"

    return subtypes


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pancdb-imc", default=None,
                        help="Path to pancdb_imc directory (default: auto-detect)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fit thresholds and print counts, don't write h5ad")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    imc_dir = Path(args.pancdb_imc) if args.pancdb_imc else script_dir.parent / "pancdb_imc"
    out_dir = script_dir

    h5ad_paths = sorted(imc_dir.rglob("cells.h5ad"))
    if not h5ad_paths:
        print(f"No cells.h5ad found under {imc_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(h5ad_paths)} ROIs")

    # Fit thresholds
    thresholds_path = out_dir / "gating_thresholds.json"
    if thresholds_path.exists():
        print(f"Loading existing thresholds from {thresholds_path}")
        with open(thresholds_path) as f:
            thresholds = json.load(f)
    else:
        thresholds = fit_thresholds(h5ad_paths)
        with open(thresholds_path, "w") as f:
            json.dump(thresholds, f, indent=2)
        print(f"Saved thresholds → {thresholds_path}")

    # Gate each ROI
    records = []
    for path in h5ad_paths:
        adata = ad.read_h5ad(path)
        donor = adata.obs["donor"].iloc[0]
        region = adata.obs["region"].iloc[0]
        roi = adata.obs["roi"].iloc[0]

        subtypes = gate_roi(adata, thresholds)
        counts = subtypes.value_counts().to_dict()

        if not args.dry_run:
            adata.obs["cell_subtype"] = subtypes.values
            adata.write_h5ad(path)

        record = {"donor": donor, "region": region, "roi": roi, **counts}
        records.append(record)
        print(f"  {donor}/{region}/{roi}: {counts}")

    df = pd.DataFrame(records).fillna(0)
    counts_path = out_dir / "subtype_counts.csv"
    df.to_csv(counts_path, index=False)
    print(f"\nSaved counts → {counts_path}")

    # Summary stats
    subtype_cols = [c for c in df.columns if c not in ("donor", "region", "roi")]
    print("\n=== Mean counts per ROI ===")
    print(df[subtype_cols].mean().round(1).to_string())


if __name__ == "__main__":
    main()
