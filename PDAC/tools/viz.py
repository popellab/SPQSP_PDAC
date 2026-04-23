"""Central-slice visualizations for ABM + ECM snapshots.

Two presets:
  plot_full_slice(df, ecm, ...)  — all cell types on ECM-density background
  plot_ecm_slice(df, ecm, ...)   — ECM density + fibroblasts + fiber directionality

Both operate on a (DataFrame, ecm-dict) pair. Loaders:
  df  = io.load_abm_lz4('outputs/abm/agents_step_000100.abm.lz4')
  ecm = viz.load_ecm('outputs/ecm/ecm_step_000100.ecm.lz4')

CLI:
  python -m PDAC.tools.viz --run-dir outputs --step 100 --mode full --out vis.png
  python -m PDAC.tools.viz --run-dir outputs --step 100 --mode ecm  --out ecm.png
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import lz4.block
import numpy as np
import pandas as pd

from . import io as _io

_ECM_CHANNELS = ["density", "fib_field", "orient_x", "orient_y", "orient_z"]

# Per-category colors. Cancer = reds, T = blues, TReg = purples, MDSC = brown,
# MAC = greens, FIB = oranges, VAS = pinks. Kept outside tab20 so neighboring
# types contrast in the legend.
_CATEGORY_COLORS = {
    "cancer_stem":          "#b30000",
    "cancer_prog":          "#e34a33",
    "cancer_sen":           "#fdbb84",
    "tcell_eff":            "#08519c",
    "tcell_cyt":            "#3182bd",
    "tcell_sup":            "#9ecae1",
    "tcell_naive":          "#c6dbef",
    "treg_th":              "#6a51a3",
    "treg_treg":            "#bcbddc",
    "treg_tfh":             "#54278f",
    "treg_tcd4_naive":      "#dadaeb",
    "mdsc":                 "#8c510a",
    "mac_M1":               "#238b45",
    "mac_M2":               "#a1d99b",
    "fib_quiescent":        "#fee391",
    "fib_iCAF":             "#fe9929",
    "fib_myCAF":            "#cc4c02",
    "fib_FRC":              "#662506",
    "vas_tip":              "#dd3497",
    "vas_phalanx":          "#f768a1",
    "vas_collapsed":        "#7a0177",
    "vas_hev":              "#ae017e",
    "bcell_naive":          "#fde725",
    "bcell_act":            "#b5de2b",
    "bcell_plasma":         "#6ece58",
    "dc_cDC1_immature":     "#762a83",
    "dc_cDC1_mature":       "#9970ab",
    "dc_cDC2_immature":     "#1b7837",
    "dc_cDC2_mature":       "#5aae61",
}
_UNKNOWN_COLOR = "#888888"


def load_ecm(path: str | Path) -> dict[str, np.ndarray]:
    """Decode a .ecm.lz4 snapshot into a dict of (gz, gy, gx) float32 arrays.

    Keys (when present): density, fib_field, orient_x, orient_y, orient_z.
    """
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"ECM1":
            raise ValueError(f"{path}: bad magic {magic!r}, expected b'ECM1'")
        gx, gy, gz, nc, raw_sz, _ = struct.unpack("<6i", f.read(24))
        data = np.frombuffer(lz4.block.decompress(f.read(), raw_sz), dtype=np.float32)
    arr = data.reshape(nc, gz, gy, gx)
    return {_ECM_CHANNELS[i]: arr[i] for i in range(min(nc, len(_ECM_CHANNELS)))}


def _resolve_slice(axis: str, slice_idx: int | None, grid_shape: tuple[int, int, int]) -> int:
    """Return a concrete slice index along `axis` ('x'|'y'|'z'), defaulting to center."""
    gz, gy, gx = grid_shape
    extents = {"x": gx, "y": gy, "z": gz}
    if axis not in extents:
        raise ValueError(f"slice_axis must be x|y|z, got {axis!r}")
    n = extents[axis]
    return n // 2 if slice_idx is None else int(slice_idx)


def _slice_ecm(field: np.ndarray, axis: str, idx: int, slab: int) -> np.ndarray:
    """Average an ECM field over ±slab voxels around `idx` along `axis`.

    Returns a 2D (row, col) array in the natural image orientation for that axis:
      axis='z' → (y, x), axis='y' → (z, x), axis='x' → (z, y).
    """
    lo, hi = max(0, idx - slab), min(field.shape["zyx".index(axis)], idx + slab + 1)
    if axis == "z":
        return field[lo:hi, :, :].mean(axis=0)       # (y, x)
    if axis == "y":
        return field[:, lo:hi, :].mean(axis=1)       # (z, x)
    return field[:, :, lo:hi].mean(axis=2)           # (z, y)


def _filter_cells_in_slab(df: pd.DataFrame, axis: str, idx: int, slab: int) -> pd.DataFrame:
    """Return cells whose coord along `axis` is within [idx-slab, idx+slab]."""
    col = axis  # df has columns x, y, z
    lo, hi = idx - slab, idx + slab
    return df[(df[col] >= lo) & (df[col] <= hi)].copy()


def _axis_columns(axis: str) -> tuple[str, str]:
    """Return the (horizontal, vertical) df columns for plotting on the given slice."""
    return {"z": ("x", "y"), "y": ("x", "z"), "x": ("y", "z")}[axis]


def _color_for(category: str) -> str:
    return _CATEGORY_COLORS.get(category, _UNKNOWN_COLOR)


def plot_full_slice(
    df: pd.DataFrame,
    ecm: dict[str, np.ndarray] | None = None,
    slice_axis: str = "z",
    slice_idx: int | None = None,
    slab: int = 1,
    category: str = "type_state",
    grid_shape: tuple[int, int, int] | None = None,
    types: list[int] | None = None,
    ax=None,
    cell_size: float = 6.0,
    ecm_cmap: str = "Greys",
    ecm_vmax: float | None = None,
    legend: bool = True,
    title: str | None = None,
):
    """All cell types on an ECM-density background for a central slice.

    Parameters
    ----------
    df : agent DataFrame (columns x,y,z,type,state,...).
    ecm : dict from load_ecm. Pass None to skip the background and plot cells only.
    slice_axis : 'x'|'y'|'z' — axis normal to the slice plane (default z).
    slice_idx : voxel index along slice_axis; None → center.
    slab : half-thickness in voxels (±slab). ECM is averaged, cells are filtered.
    category : passed to io.add_category for the color legend.
    grid_shape : (gz, gy, gx) fallback when ecm is None. If also None, inferred
                 from df coord maxima.
    types : if not None, keep only agents whose `type` is in this list (use
            io.TYPE_CANCER/TCELL/… constants). None → all types.
    cell_size : matplotlib scatter `s` parameter.
    ecm_cmap, ecm_vmax : background colormap + optional fixed max.
    """
    if types is not None:
        df = df[df["type"].isin(types)]
    import matplotlib.pyplot as plt

    # Resolve grid shape and slice index from whichever source is available.
    if ecm is not None and "density" in ecm:
        density = ecm["density"]
        shape = density.shape
    elif grid_shape is not None:
        density = None
        shape = tuple(int(v) for v in grid_shape)
    else:
        density = None
        gx = int(df["x"].max()) + 1 if len(df) else 1
        gy = int(df["y"].max()) + 1 if len(df) else 1
        gz = int(df["z"].max()) + 1 if len(df) else 1
        shape = (gz, gy, gx)
    idx = _resolve_slice(slice_axis, slice_idx, shape)

    # Dimensions of the 2D slice plane (rows, cols) in the same convention as _slice_ecm.
    slice_shape = {
        "z": (shape[1], shape[2]),  # (y, x)
        "y": (shape[0], shape[2]),  # (z, x)
        "x": (shape[0], shape[1]),  # (z, y)
    }[slice_axis]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    if density is not None:
        bg = _slice_ecm(density, slice_axis, idx, slab)
        vmax = ecm_vmax if ecm_vmax is not None else max(float(bg.max()), 1e-6)
        ax.imshow(bg, origin="lower", cmap=ecm_cmap, vmin=0.0, vmax=vmax,
                  extent=[0, bg.shape[1], 0, bg.shape[0]],
                  interpolation="nearest", zorder=0)

    df_cat = _io.add_category(df, mode=category)
    slab_df = _filter_cells_in_slab(df_cat, slice_axis, idx, slab)
    hcol, vcol = _axis_columns(slice_axis)

    # Plot per-category so the legend groups correctly.
    canonical = _io.category_list(mode=category)
    present = [c for c in canonical if (slab_df["category"] == c).any()]
    for cat in present:
        sub = slab_df[slab_df["category"] == cat]
        ax.scatter(sub[hcol] + 0.5, sub[vcol] + 0.5,
                   s=cell_size, c=_color_for(cat), label=cat,
                   edgecolors="none", alpha=0.9, zorder=2)

    ax.set_xlim(0, slice_shape[1]); ax.set_ylim(0, slice_shape[0])
    ax.set_xlabel(hcol); ax.set_ylabel(vcol)
    ax.set_aspect("equal")
    bg_tag = "" if density is not None else " (no ECM)"
    ax.set_title(title or f"Full slice — {slice_axis}={idx}, slab=±{slab}, "
                          f"n_cells={len(slab_df)}{bg_tag}")
    if legend and present:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                  fontsize=8, frameon=False)
    return ax.figure, ax


# Cell type groupings used by the cancer-only / immune-only presets.
_CANCER_TYPES = [_io.TYPE_CANCER]
_IMMUNE_TYPES = [_io.TYPE_TCELL, _io.TYPE_TREG, _io.TYPE_MDSC, _io.TYPE_MAC,
                 _io.TYPE_BCELL, _io.TYPE_DC]


def plot_cancer_slice(df, ecm=None, **kwargs):
    """Cancer-only view (stem/prog/sen) over an ECM-density background.

    Thin wrapper around plot_full_slice with types=[io.TYPE_CANCER].
    """
    kwargs.setdefault("title", None)
    if kwargs["title"] is None:
        kwargs["title"] = "Cancer slice"
    return plot_full_slice(df, ecm, types=_CANCER_TYPES, **kwargs)


def plot_immune_slice(df, ecm=None, **kwargs):
    """Immune-only view (T, TReg, MDSC, MAC) over an ECM-density background.

    Thin wrapper around plot_full_slice with the canonical immune type set.
    Fibroblasts and vascular cells are excluded.
    """
    kwargs.setdefault("title", None)
    if kwargs["title"] is None:
        kwargs["title"] = "Immune slice"
    return plot_full_slice(df, ecm, types=_IMMUNE_TYPES, **kwargs)


def plot_vascular_slice(
    df: pd.DataFrame,
    ecm: dict[str, np.ndarray] | None = None,
    slice_axis: str = "z",
    slice_idx: int | None = None,
    slab: int = 1,
    grid_shape: tuple[int, int, int] | None = None,
    ax=None,
    ecm_cmap: str = "Greys",
    ecm_vmax: float | None = None,
    cancer_size: float = 2.0,
    vas_size: float = 18.0,
    show_cancer: bool = True,
    title: str | None = None,
):
    """Vascular-focused view: phalanx + tip cells on ECM background.

    Cancer cells are drawn faintly as spatial context (VEGF source → drives
    sprouting); immune and fibroblast populations are omitted so the vascular
    network stands out.
    """
    import matplotlib.pyplot as plt

    # Resolve grid shape and slice index — mirrors plot_full_slice fallback chain.
    if ecm is not None and "density" in ecm:
        density = ecm["density"]
        shape = density.shape
    elif grid_shape is not None:
        density = None
        shape = tuple(int(v) for v in grid_shape)
    else:
        density = None
        gx = int(df["x"].max()) + 1 if len(df) else 1
        gy = int(df["y"].max()) + 1 if len(df) else 1
        gz = int(df["z"].max()) + 1 if len(df) else 1
        shape = (gz, gy, gx)
    idx = _resolve_slice(slice_axis, slice_idx, shape)
    slice_shape = {
        "z": (shape[1], shape[2]),
        "y": (shape[0], shape[2]),
        "x": (shape[0], shape[1]),
    }[slice_axis]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    if density is not None:
        bg = _slice_ecm(density, slice_axis, idx, slab)
        vmax = ecm_vmax if ecm_vmax is not None else max(float(bg.max()), 1e-6)
        ax.imshow(bg, origin="lower", cmap=ecm_cmap, vmin=0.0, vmax=vmax,
                  extent=[0, bg.shape[1], 0, bg.shape[0]],
                  interpolation="nearest", zorder=0)

    df_cat = _io.add_category(df, mode="type_state")
    slab_df = _filter_cells_in_slab(df_cat, slice_axis, idx, slab)
    hcol, vcol = _axis_columns(slice_axis)

    # Faint cancer context so vascular-tumor colocation is visible.
    if show_cancer:
        cancer = slab_df[slab_df["type"] == _io.TYPE_CANCER]
        if len(cancer):
            ax.scatter(cancer[hcol] + 0.5, cancer[vcol] + 0.5,
                       s=cancer_size, c="#b30000", label="cancer (context)",
                       alpha=0.25, edgecolors="none", zorder=1)

    vas = slab_df[slab_df["type"] == _io.TYPE_VAS]
    n_tip = n_phal = 0
    for cat, color, marker in [
        ("vas_phalanx", _CATEGORY_COLORS["vas_phalanx"], "s"),
        ("vas_tip",     _CATEGORY_COLORS["vas_tip"],     "*"),
    ]:
        sub = vas[vas["category"] == cat]
        if cat == "vas_tip":
            n_tip = len(sub)
        else:
            n_phal = len(sub)
        if len(sub):
            ax.scatter(sub[hcol] + 0.5, sub[vcol] + 0.5,
                       s=vas_size if cat == "vas_phalanx" else vas_size * 1.8,
                       c=color, marker=marker, label=cat,
                       edgecolors="black", linewidths=0.3, zorder=3)

    ax.set_xlim(0, slice_shape[1]); ax.set_ylim(0, slice_shape[0])
    ax.set_xlabel(hcol); ax.set_ylabel(vcol)
    ax.set_aspect("equal")
    bg_tag = "" if density is not None else " (no ECM)"
    ax.set_title(title or f"Vascular slice — {slice_axis}={idx}, slab=±{slab}, "
                          f"tip={n_tip}, phalanx={n_phal}{bg_tag}")
    if n_tip + n_phal > 0:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                  fontsize=8, frameon=False)
    return ax.figure, ax


def plot_ecm_slice(
    df: pd.DataFrame,
    ecm: dict[str, np.ndarray],
    slice_axis: str = "z",
    slice_idx: int | None = None,
    slab: int = 1,
    quiver_every: int = 4,
    quiver_scale: float = 25.0,
    ax=None,
    ecm_cmap: str = "YlOrBr",
    ecm_vmax: float | None = None,
    fib_size: float = 12.0,
    title: str | None = None,
):
    """ECM-focused view: density background + fibroblasts + fiber directionality.

    Directionality is drawn as a quiver of the two in-plane components of the
    (orient_x, orient_y, orient_z) field, sub-sampled every `quiver_every`
    voxels. Only voxels with nontrivial density are drawn.
    """
    import matplotlib.pyplot as plt

    if "density" not in ecm:
        raise ValueError("ecm dict must contain 'density'")
    density = ecm["density"]
    idx = _resolve_slice(slice_axis, slice_idx, density.shape)
    bg = _slice_ecm(density, slice_axis, idx, slab)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    vmax = ecm_vmax if ecm_vmax is not None else max(float(bg.max()), 1e-6)
    ax.imshow(bg, origin="lower", cmap=ecm_cmap, vmin=0.0, vmax=vmax,
              extent=[0, bg.shape[1], 0, bg.shape[0]], interpolation="nearest", zorder=0)

    # In-plane orient components for the quiver.
    # For axis=z: horizontal=x, vertical=y → use orient_x, orient_y.
    # For axis=y: horizontal=x, vertical=z → use orient_x, orient_z.
    # For axis=x: horizontal=y, vertical=z → use orient_y, orient_z.
    if slice_axis == "z":
        keys = ("orient_x", "orient_y")
    elif slice_axis == "y":
        keys = ("orient_x", "orient_z")
    else:
        keys = ("orient_y", "orient_z")

    if all(k in ecm for k in keys):
        ou = _slice_ecm(ecm[keys[0]], slice_axis, idx, slab)
        ov = _slice_ecm(ecm[keys[1]], slice_axis, idx, slab)
        step = max(1, int(quiver_every))
        hh, ww = bg.shape
        yy, xx = np.mgrid[0:hh:step, 0:ww:step]
        u = ou[::step, ::step]; v = ov[::step, ::step]
        # Scale by local density so empty regions don't show spurious arrows.
        scale_mask = bg[::step, ::step] / max(vmax, 1e-9)
        u = u * scale_mask; v = v * scale_mask
        ax.quiver(xx + 0.5, yy + 0.5, u, v,
                  color="black", angles="xy", scale_units="xy",
                  scale=1.0 / max(quiver_scale, 1e-6),
                  width=0.003, headwidth=3.0, headlength=3.5, zorder=2)

    # Fibroblasts on top.
    df_cat = _io.add_category(df, mode="type_state")
    fibs = _filter_cells_in_slab(df_cat, slice_axis, idx, slab)
    fibs = fibs[fibs["type"] == _io.TYPE_FIB]
    hcol, vcol = _axis_columns(slice_axis)
    for cat in ("fib_quiescent", "fib_iCAF", "fib_myCAF", "fib_FRC"):
        sub = fibs[fibs["category"] == cat]
        if len(sub):
            ax.scatter(sub[hcol] + 0.5, sub[vcol] + 0.5,
                       s=fib_size, c=_CATEGORY_COLORS[cat], label=cat,
                       edgecolors="black", linewidths=0.3, zorder=3)

    ax.set_xlim(0, bg.shape[1]); ax.set_ylim(0, bg.shape[0])
    ax.set_xlabel(hcol); ax.set_ylabel(vcol)
    ax.set_aspect("equal")
    ax.set_title(title or f"ECM slice — {slice_axis}={idx}, slab=±{slab}, "
                          f"n_fib={len(fibs)}")
    if len(fibs):
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                  fontsize=8, frameon=False)
    return ax.figure, ax


def _pair_paths(run_dir: Path, step: int, require_ecm: bool = True
                ) -> tuple[Path, Path | None]:
    """Locate the ABM (and optionally ECM) snapshot files under `run_dir`."""
    abm = run_dir / "abm" / f"agents_step_{step:06d}.abm.lz4"
    ecm = run_dir / "ecm" / f"ecm_step_{step:06d}.ecm.lz4"
    if not abm.exists():
        raise FileNotFoundError(f"Missing ABM snapshot: {abm}")
    if not ecm.exists():
        if require_ecm:
            raise FileNotFoundError(f"Missing ECM snapshot: {ecm}")
        ecm = None
    return abm, ecm


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path,
                    help="Outputs dir containing abm/ and ecm/ subdirs")
    ap.add_argument("--step",    type=int, help="Step index (paired files)")
    ap.add_argument("--abm",     type=Path, help="Direct path to agents_*.abm.lz4")
    ap.add_argument("--ecm",     type=Path, help="Direct path to ecm_*.ecm.lz4")
    ap.add_argument("--mode",    choices=["full", "ecm", "cancer", "immune", "vascular"],
                    required=True)
    ap.add_argument("--axis",    choices=["x", "y", "z"], default="z")
    ap.add_argument("--idx",     type=int, default=None)
    ap.add_argument("--slab",    type=int, default=1)
    ap.add_argument("--out",     type=Path, required=True)
    ap.add_argument("--dpi",     type=int, default=150)
    ap.add_argument("--quiver-every", type=int, default=4,
                    help="(ecm mode) sub-sample factor for directionality arrows")
    args = ap.parse_args(argv)

    # ECM is required only for the 'ecm' directionality mode.
    ecm_required = (args.mode == "ecm")
    if args.abm:
        abm_path = args.abm
        ecm_path = args.ecm  # may be None
        if ecm_required and ecm_path is None:
            ap.error("--ecm is required for --mode ecm")
    elif args.run_dir and args.step is not None:
        abm_path, ecm_path = _pair_paths(args.run_dir, args.step,
                                         require_ecm=ecm_required)
    else:
        ap.error("Provide either --abm (+ optional --ecm), or --run-dir and --step.")

    df = _io.load_abm_lz4(str(abm_path))
    ecm = load_ecm(ecm_path) if ecm_path is not None else None

    if args.mode == "full":
        fig, _ = plot_full_slice(df, ecm, slice_axis=args.axis,
                                 slice_idx=args.idx, slab=args.slab)
    elif args.mode == "cancer":
        fig, _ = plot_cancer_slice(df, ecm, slice_axis=args.axis,
                                   slice_idx=args.idx, slab=args.slab)
    elif args.mode == "immune":
        fig, _ = plot_immune_slice(df, ecm, slice_axis=args.axis,
                                   slice_idx=args.idx, slab=args.slab)
    elif args.mode == "vascular":
        fig, _ = plot_vascular_slice(df, ecm, slice_axis=args.axis,
                                     slice_idx=args.idx, slab=args.slab)
    else:
        fig, _ = plot_ecm_slice(df, ecm, slice_axis=args.axis,
                                slice_idx=args.idx, slab=args.slab,
                                quiver_every=args.quiver_every)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
