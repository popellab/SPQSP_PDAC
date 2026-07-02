"""Posterior-predictive trajectory plot for calibration targets.

For each calibration target the model was constrained against, produces a
panel showing how the observable evolves from healthy → diagnosis (or
treatment-end, for treatment scenarios) under the posterior. The
observed value with its 95% CI is overlaid at the right edge — the
calibration target's measurement time.

Inputs are pre-computed:

  - ``observable_df``: long-form ``[observable_id, sample_index,
    t_to_diagnosis_days, value]`` — produced by stacking outputs of
    ``qsp_inference.inference.trajectory_eval.evaluate_observable_over_trajectory``
    across cal targets.
  - ``calibration_targets``: list of objects exposing
    ``observable_id``, ``observable.units``, ``empirical_data.median``,
    ``empirical_data.ci95_lower``, ``empirical_data.ci95_upper``. Any
    duck-typed equivalent works (``maple.CalibrationTarget`` is the
    canonical case).

The pipeline upstream of this function (run posterior thetas through
``CppSimulator(evolve_trajectory_dir=...)``, assemble per-sim binaries
via ``qsp_hpc.cpp.evolve_trajectory.assemble_evolve_trajectory_long``,
evaluate each cal target's observable code) lives in the workflow
script ``run_calibration_trajectory_plot.py`` (TODO: not yet
integrated into ``workflows/sbi_runner``; see PR description).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def plot_calibration_trajectories(
    observable_df: pd.DataFrame,
    calibration_targets: Sequence[Any],
    *,
    output_path: Optional[str | Path] = None,
    n_cols: int = 3,
    ci_level: float = 0.95,
    ci_method: str = "pointwise",
    panel_size: tuple[float, float] = (4.5, 3.5),
    log_y: bool | str = False,
    log_y_ratio_threshold: float = 10.0,
    time_axis: str = "absolute",
    phase: str = "burn_in",
    scenario_per_target: Optional[Mapping[str, str]] = None,
    scalar_relative_std_threshold: float = 1e-3,
    title: Optional[str] = None,
) -> Figure:
    """Render one panel per cal target with posterior trajectory + observed CI.

    Args:
        observable_df: Long-form DataFrame with columns
            ``[observable_id, sample_index, t_to_diagnosis_days, value]``.
            Values are in the cal target's ``observable.units`` (the
            evaluator already converted). One ``observable_id`` per cal
            target plotted; targets without rows in this frame are skipped
            with a printed warning.
        calibration_targets: Iterable of cal-target objects exposing
            ``observable_id`` and an ``empirical_data`` block with
            ``median``, ``ci95_lower``, ``ci95_upper``, and ``units``.
            Plot order follows iteration order.
        output_path: If set, also save the figure to this path. Parent
            dirs are created. PNG is the canonical extension; matplotlib
            picks the format from the suffix.
        n_cols: Panels per row. ``n_rows = ceil(n_targets / n_cols)``.
        ci_level: Coverage of the trajectory band. ``0.95`` → q025/q975
            quantiles at each time. ``0.90`` → q05/q95.
        ci_method: ``"pointwise"`` (default) computes per-time quantiles
            independently. The label honestly reflects this — true
            simultaneous bands (Bonferroni / supremum bootstrap) are a
            future extension.
        panel_size: ``(width_in, height_in)`` per panel.
        log_y: ``True`` / ``False`` apply uniformly. ``"auto"`` decides
            per panel: log scale when the empirical CI spans more than
            ``log_y_ratio_threshold`` (i.e. ci95_upper / ci95_lower ≥ N),
            which is a clean signal for lognormal-ish targets where
            linear scale crushes the lower part of the band against
            zero. Cytokines, cell densities, and clinical-progression
            endpoints typically satisfy this.
        log_y_ratio_threshold: ``ci95_upper / ci95_lower`` cutoff used
            when ``log_y="auto"``. Default 10× — i.e. the empirical 95%
            CI must span at least one order of magnitude before log-y
            engages.
        time_axis: ``"absolute"`` (default) plots
            ``t_to_diagnosis_days`` directly (the values' sign + range
            depend on ``phase``). ``"fraction"`` rescales each sim's
            burn-in to ``[-1, 0]`` so variable-duration trajectories
            share a common axis — only meaningful for ``phase="burn_in"``;
            silently ignored otherwise.
        phase: ``"burn_in"`` (default) treats the time column as
            negative-leading-up-to-0 (= diagnosis), draws the observed
            marker at ``x=0``, and labels the axis as 'time to
            diagnosis'. ``"post_scenario"`` treats the time column as
            non-negative (0 = scenario start = diagnosis for evolve-to-
            diagnosis scenarios), looks up each cal target's observed
            marker x-position from
            ``empirical_data.index_values[0]`` (falling back to the
            rightmost data point), and labels the axis as 'time since
            diagnosis'. Used by sbi_runner to plot one figure per
            scenario phase.
        title: Optional figure-level suptitle.

    Returns:
        ``matplotlib.figure.Figure``. The caller may further annotate /
        save / close as needed; ``output_path`` is a convenience.
    """
    if ci_method != "pointwise":
        raise ValueError(
            f"ci_method={ci_method!r} not yet supported (only 'pointwise')"
        )
    if not (0 < ci_level < 1):
        raise ValueError(f"ci_level must be in (0, 1); got {ci_level}")
    if time_axis not in ("absolute", "fraction"):
        raise ValueError(
            f"time_axis must be 'absolute' or 'fraction'; got {time_axis!r}"
        )
    if phase not in ("burn_in", "post_scenario", "auto"):
        raise ValueError(
            f"phase must be 'burn_in', 'post_scenario', or 'auto'; got {phase!r}"
        )
    if phase == "post_scenario" and time_axis == "fraction":
        # Post-scenario sims share a fixed time grid; fraction
        # normalization is meaningless. Silently fall back.
        time_axis = "absolute"
    # Note: phase="auto" with time_axis="fraction" is allowed — the
    # fraction mode is applied per-panel (only when panel data is pure
    # burn-in). Post-scenario panels in the same figure get absolute.
    if isinstance(log_y, str) and log_y != "auto":
        raise ValueError(
            f"log_y string value must be 'auto'; got {log_y!r}"
        )

    required = {"observable_id", "sample_index", "t_to_diagnosis_days", "value"}
    missing = required - set(observable_df.columns)
    if missing:
        raise ValueError(f"observable_df missing required columns: {sorted(missing)}")

    targets = list(calibration_targets)
    n_targets = len(targets)
    if n_targets == 0:
        raise ValueError("calibration_targets is empty — nothing to plot")

    n_cols = max(1, min(n_cols, n_targets))
    n_rows = math.ceil(n_targets / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(panel_size[0] * n_cols, panel_size[1] * n_rows),
        squeeze=False,
    )
    flat_axes = axes.flatten()

    q_lo = (1 - ci_level) / 2
    q_hi = 1 - q_lo

    for i, target in enumerate(targets):
        ax = flat_axes[i]
        obs_id = (
            getattr(target, "calibration_target_id", None)
            or getattr(target, "observable_id", None)
            or getattr(target, "id", None)
        )
        if obs_id is None:
            raise ValueError(
                f"target index {i} has no id field "
                "(checked calibration_target_id, observable_id, id)"
            )
        sub = observable_df[observable_df["observable_id"] == obs_id]
        units = _extract_units(target)
        empirical = _extract_empirical(target)

        if sub.empty:
            print(f"  [plot] no posterior trajectory rows for {obs_id}; panel left blank")
            ax.set_title(f"{obs_id}\n(no posterior data)")
            ax.set_axis_off()
            continue

        # Per-panel log-y resolution.
        if log_y == "auto":
            panel_log_y = _auto_log_y_from_empirical(empirical, log_y_ratio_threshold)
        else:
            panel_log_y = bool(log_y)

        scenario = (
            (scenario_per_target or {}).get(obs_id) if scenario_per_target else None
        )
        # Detect "constant-over-time" scalar observables (e.g.,
        # tumor_doubling_time broadcasts a single VDT per sim across the
        # time vector). For those, the trajectory band is uninformative —
        # render a histogram of per-sim values instead.
        if _is_scalar_observable(sub, scalar_relative_std_threshold):
            _draw_scalar_panel(
                ax=ax,
                sub=sub,
                empirical=empirical,
                units=units,
                obs_id=obs_id,
                ci_level=ci_level,
                log_y=panel_log_y,
                scenario=scenario,
            )
        else:
            _draw_trajectory_panel(
                ax=ax,
                sub=sub,
                empirical=empirical,
                units=units,
                obs_id=obs_id,
                q_lo=q_lo,
                q_hi=q_hi,
                ci_level=ci_level,
                log_y=panel_log_y,
                time_axis=time_axis,
                phase=phase,
                scenario=scenario,
            )

    # Hide trailing empty axes if n_targets isn't a multiple of n_cols.
    for j in range(n_targets, len(flat_axes)):
        flat_axes[j].set_axis_off()

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  [plot] saved {out}")

    return fig


def _extract_units(target: Any) -> str:
    obs = getattr(target, "observable", None)
    if obs is not None and hasattr(obs, "units"):
        return obs.units
    emp = getattr(target, "empirical_data", None)
    if emp is not None and hasattr(emp, "units"):
        return emp.units
    return ""


def _extract_empirical(target: Any) -> dict:
    """Pull median + 95% CI + index_values off ``target.empirical_data``,
    supporting both flat (``median``, ``ci95_lower``, ``ci95_upper``) and
    maple's list schema (``median = [m]``, ``ci95 = [[lo, hi]]``).
    """
    emp = getattr(target, "empirical_data", None)
    if emp is None:
        return {}
    median = getattr(emp, "median", None)
    lo = getattr(emp, "ci95_lower", None)
    hi = getattr(emp, "ci95_upper", None)
    index_values = getattr(emp, "index_values", None)
    if median is not None and isinstance(median, (list, tuple)) and len(median) > 0:
        median = median[0]
    if lo is None and hi is None:
        ci95 = getattr(emp, "ci95", None)
        if isinstance(ci95, (list, tuple)) and len(ci95) > 0:
            pair = ci95[0]
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                lo, hi = pair
    return {
        "median": median,
        "ci95_lower": lo,
        "ci95_upper": hi,
        "index_values": index_values,
    }


def _auto_log_y_from_empirical(
    empirical: dict, ratio_threshold: float, log_symmetry_atol: float = 0.25
) -> bool:
    """Per-panel log-y heuristic.

    Two ways to qualify:

    1. **Log-symmetric CI95** (the canonical lognormal signature):
       ``log(hi/median) ≈ log(median/lo)``. Tighter CIs that are
       multiplicative around the median pass this test even when
       ``hi/lo`` is small (e.g. Liu 2015 CD8 density: median 63,
       CI [30, 132], ratio 4.4× — but log-arms are both 0.74, so it's
       cleanly lognormal). The check uses the relative arm asymmetry
       ``|log(hi/med) - log(med/lo)| / max(log(hi/med), log(med/lo))``;
       below ``log_symmetry_atol`` (default 0.25) → lognormal-ish.

    2. **Wide CI fallback**: ``hi/lo >= ratio_threshold`` (default 10×)
       still triggers log even if the symmetry test fails — keeps the
       old behavior for targets with very wide spans where linear
       scale would crush the band against zero.

    Returns False when CI bounds are missing or non-positive.
    """
    median = empirical.get("median")
    lo = empirical.get("ci95_lower")
    hi = empirical.get("ci95_upper")
    if lo is None or hi is None:
        return False
    try:
        lo_f = float(lo)
        hi_f = float(hi)
        med_f = float(median) if median is not None else None
    except (TypeError, ValueError):
        return False
    if lo_f <= 0 or hi_f <= 0 or lo_f > hi_f:
        return False
    # Wide-CI fallback (preserves old behavior for highly-uncertain targets).
    if (hi_f / lo_f) >= ratio_threshold:
        return True
    # Log-symmetry test — fires for tight lognormal CIs (e.g. Liu density)
    # that the ratio fallback misses.
    if med_f is None or med_f <= 0 or med_f < lo_f or med_f > hi_f:
        return False
    upper_arm = math.log(hi_f / med_f)
    lower_arm = math.log(med_f / lo_f)
    if upper_arm <= 0 or lower_arm <= 0:
        return False
    asym = abs(upper_arm - lower_arm) / max(upper_arm, lower_arm)
    # Also require a minimum log-span so a target with a very tight CI
    # (both arms tiny but symmetric) doesn't switch to log gratuitously.
    log_span = upper_arm + lower_arm  # log(hi/lo)
    return asym <= log_symmetry_atol and log_span >= math.log(2.0)


def _resolve_time_column(sub: pd.DataFrame, time_axis: str) -> tuple[pd.DataFrame, str]:
    """Add a ``t_fraction`` column when ``time_axis="fraction"`` and return
    (df, x_column_name). Each sim's burn-in is rescaled to ``[-1, 0]`` by
    dividing its ``t_to_diagnosis_days`` by its sim-specific
    ``t_diagnosis = -min(t_to_diagnosis_days)``."""
    if time_axis == "absolute":
        return sub, "t_to_diagnosis_days"
    sim_t_diag = sub.groupby("sample_index")["t_to_diagnosis_days"].transform(
        lambda s: -float(s.min())
    )
    # Guard against degenerate sims (only one timepoint at t=0).
    sim_t_diag = sim_t_diag.replace(0, np.nan)
    sub = sub.assign(t_fraction=sub["t_to_diagnosis_days"] / sim_t_diag)
    return sub, "t_fraction"


_FRACTION_GRID_POINTS = 100


_MIN_COVERAGE_FRACTION = 0.8


def _quantiles_per_time(
    sub: pd.DataFrame,
    t_col: str,
    q_lo: float,
    q_hi: float,
    time_axis: str,
    min_coverage: float = _MIN_COVERAGE_FRACTION,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(t_vals, lo, median, hi)`` for the panel.

    For ``time_axis="absolute"`` we group on the (already-shared)
    ``t_to_diagnosis_days`` column and quantile per group. For
    ``time_axis="fraction"`` each sim's normalized timepoints are
    sim-specific, so a naive groupby produces single-sample bins —
    instead, interpolate every sim onto a shared grid in [-1, 0] and
    quantile across sims at each grid point.

    In fraction mode, also drop grid points where fewer than
    ``min_coverage`` × n_sims contribute. The right edge naturally has
    lower coverage because each sim's last data row lands at
    ``(last_dump_t − t_diag) / t_diag < 0`` (the C++ side dumps at
    uniform intervals and doesn't capture the diagnosis row itself —
    a follow-up); without masking, those low-coverage points show up
    as visible transients in the median and band.
    """
    if time_axis == "absolute":
        # Even when sims nominally share a time grid (post-scenario
        # scenarios where t_end / dt is pool-level), CppBatchRunner stores
        # per-sim time arrays with ~1e-4 float drift across sims. A naive
        # groupby on ``t_to_diagnosis_days`` then yields one bin per
        # unique float ⇒ single-sample bins ⇒ ``fill_between`` collapses
        # to dense vertical slivers. Interpolate every sim onto a shared
        # grid (median of per-sim time arrays after rounding to a small
        # tolerance) and quantile across sims at each grid point.
        per_sim_times: list[np.ndarray] = []
        per_sim_vals: list[np.ndarray] = []
        for _, sim_df in sub.groupby("sample_index"):
            sim_df = sim_df.sort_values(t_col)
            x = sim_df[t_col].to_numpy(dtype=np.float64)
            y = sim_df["value"].to_numpy(dtype=np.float64)
            if len(x) >= 2:
                per_sim_times.append(x)
                per_sim_vals.append(y)
        if not per_sim_times:
            return np.array([]), np.array([]), np.array([]), np.array([])
        # Shared grid = sorted unique times rounded to 4 decimals
        # (~0.0001 days ~= 8.6s — finer than any QSP output cadence).
        all_t = np.concatenate(per_sim_times)
        grid = np.unique(np.round(all_t, 4))
        rows = []
        for x, y in zip(per_sim_times, per_sim_vals):
            in_range = (grid >= x.min()) & (grid <= x.max())
            row = np.full_like(grid, np.nan)
            row[in_range] = np.interp(grid[in_range], x, y)
            rows.append(row)
        arr = np.vstack(rows)  # (n_sims, n_grid)
        n_sims = arr.shape[0]
        coverage = np.sum(np.isfinite(arr), axis=0)
        keep = coverage >= max(2, int(np.ceil(min_coverage * n_sims)))
        if not keep.any():
            keep = coverage >= 2  # fall back to any 2-sim coverage
        median = np.nanmedian(arr[:, keep], axis=0)
        lo = np.nanquantile(arr[:, keep], q_lo, axis=0)
        hi = np.nanquantile(arr[:, keep], q_hi, axis=0)
        return grid[keep], lo, median, hi

    # fraction mode: interpolate per sim onto a shared grid.
    grid = np.linspace(-1.0, 0.0, _FRACTION_GRID_POINTS, dtype=np.float64)
    interpolated: list[np.ndarray] = []
    for _, sim_df in sub.groupby("sample_index"):
        sim_df = sim_df.sort_values(t_col)
        x = sim_df[t_col].to_numpy(dtype=np.float64)
        y = sim_df["value"].to_numpy(dtype=np.float64)
        if len(x) < 2:
            continue
        # Mask grid points outside the sim's actual fraction range so we
        # don't fabricate values where evolve never reached.
        in_range = (grid >= x.min()) & (grid <= x.max())
        row = np.full_like(grid, np.nan)
        row[in_range] = np.interp(grid[in_range], x, y)
        interpolated.append(row)
    if not interpolated:
        return np.array([]), np.array([]), np.array([]), np.array([])
    arr = np.vstack(interpolated)  # (n_sims, n_grid)
    n_sims = arr.shape[0]
    coverage = np.sum(np.isfinite(arr), axis=0)
    keep = coverage >= max(2, int(np.ceil(min_coverage * n_sims)))
    if not keep.any():
        # Pathological — fall back to "drop the rightmost half" and warn.
        keep[: len(grid) // 2] = True
    median = np.full(len(grid), np.nan)
    lo = np.full(len(grid), np.nan)
    hi = np.full(len(grid), np.nan)
    median[keep] = np.nanmedian(arr[:, keep], axis=0)
    lo[keep] = np.nanquantile(arr[:, keep], q_lo, axis=0)
    hi[keep] = np.nanquantile(arr[:, keep], q_hi, axis=0)
    return grid[keep], lo[keep], median[keep], hi[keep]


def _is_scalar_observable(sub: pd.DataFrame, rel_std_threshold: float) -> bool:
    """Detect cal targets that return a constant value broadcast over
    the time axis (e.g., ``tumor_doubling_time`` computes one VDT per
    sim and fills the time vector with that scalar).

    Per-sim ``std(values) / |mean(values)|`` close to 0 means flat. We
    require ALL sims to be flat (median across per-sim relative-std
    below the threshold) to avoid false positives on genuinely
    near-constant trajectories.
    """
    grouped = sub.groupby("sample_index")["value"]
    means = grouped.mean()
    stds = grouped.std(ddof=0)
    # Avoid div-by-zero; treat |mean|≈0 as scalar iff std≈0 too.
    abs_means = means.abs().replace(0, np.nan)
    relative = (stds / abs_means).fillna(stds)  # std-only when mean≈0
    if relative.empty:
        return False
    return float(relative.median()) < rel_std_threshold


def _draw_scalar_panel(
    *,
    ax: plt.Axes,
    sub: pd.DataFrame,
    empirical: dict,
    units: str,
    obs_id: str,
    ci_level: float,
    log_y: bool,
    scenario: Optional[str] = None,
) -> None:
    """Render a histogram of per-sim values for a constant-over-time
    cal target, plus the observed CI95 as a vertical shaded band.

    Useful for scalar derived observables (overall doubling time, etc.)
    where the trajectory band collapses to a flat stripe and you
    actually want the cohort distribution at a single value.
    """
    per_sim = sub.groupby("sample_index")["value"].mean().dropna().to_numpy()
    if len(per_sim) == 0:
        ax.set_title(f"{obs_id}\n(no data)")
        ax.set_axis_off()
        return

    if log_y:
        # Log-axis histogram — bin in log space, render on linear x with
        # log-spaced edges for readability.
        positive = per_sim[per_sim > 0]
        if len(positive) >= 2:
            edges = np.geomspace(positive.min(), positive.max(), 21)
            ax.hist(positive, bins=edges, color="C0", alpha=0.5, edgecolor="C0")
            ax.set_xscale("log")
        else:
            ax.hist(per_sim, bins=20, color="C0", alpha=0.5, edgecolor="C0")
    else:
        ax.hist(per_sim, bins=20, color="C0", alpha=0.5, edgecolor="C0")

    obs_median = empirical.get("median")
    obs_lo = empirical.get("ci95_lower")
    obs_hi = empirical.get("ci95_upper")
    if obs_lo is not None and obs_hi is not None:
        ax.axvspan(
            float(obs_lo),
            float(obs_hi),
            color="C3",
            alpha=0.15,
            label="observed 95% CI",
        )
    if obs_median is not None:
        ax.axvline(
            float(obs_median),
            color="C3",
            lw=2,
            label="observed median",
        )
    if units:
        ax.set_xlabel(units)
    ax.set_ylabel("posterior sim count")
    title = obs_id
    if scenario:
        title = f"{obs_id}\n[{scenario}]"
    ax.set_title(title, fontsize=10)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)


def _draw_trajectory_panel(
    *,
    ax: plt.Axes,
    sub: pd.DataFrame,
    empirical: dict,
    units: str,
    obs_id: str,
    q_lo: float,
    q_hi: float,
    ci_level: float,
    log_y: bool,
    time_axis: str = "absolute",
    phase: str = "burn_in",
    scenario: Optional[str] = None,
) -> None:
    """Draw one cal-target panel: trajectory band + median + observed marker."""
    # Resolve effective time_axis per panel: post-scenario panels in
    # mixed-phase figures fall back to absolute (fraction-norm is only
    # meaningful for variable-duration burn-in trajectories). We need
    # the panel-level phase decision before the time_axis choice, so
    # peek at the data range first.
    if phase == "auto":
        t_raw = sub["t_to_diagnosis_days"].to_numpy(dtype=np.float64)
        if len(t_raw) == 0 or t_raw.max() <= 0:
            panel_phase = "burn_in"
        else:
            panel_phase = "post_scenario"
        panel_time_axis = "absolute" if panel_phase == "post_scenario" else time_axis
    else:
        panel_phase = phase
        panel_time_axis = time_axis

    sub, t_col = _resolve_time_column(sub, panel_time_axis)
    t_vals, lo, median, hi = _quantiles_per_time(
        sub, t_col, q_lo, q_hi, panel_time_axis
    )

    band_label = f"posterior {int(round(ci_level * 100))}% (pointwise)"
    ax.fill_between(t_vals, lo, hi, color="C0", alpha=0.25, label=band_label, lw=0)
    ax.plot(t_vals, median, color="C0", lw=1.5, label="posterior median")

    # Observed marker x-position depends on resolved phase:
    #   burn_in       → 0.0 (diagnosis = end of burn-in)
    #   post_scenario → cal target's index_values[0] if set, else the
    #                   rightmost data point (end-of-scenario default).
    if panel_phase == "post_scenario":
        idx_vals = empirical.get("index_values")
        if isinstance(idx_vals, (list, tuple)) and idx_vals:
            try:
                marker_x = float(idx_vals[0])
            except (TypeError, ValueError):
                marker_x = float(t_vals[-1]) if len(t_vals) else 0.0
        else:
            marker_x = float(t_vals[-1]) if len(t_vals) else 0.0
    else:
        marker_x = 0.0

    obs_median = empirical.get("median")
    obs_lo = empirical.get("ci95_lower")
    obs_hi = empirical.get("ci95_upper")
    if obs_median is not None:
        yerr = None
        if obs_lo is not None and obs_hi is not None:
            yerr = np.array(
                [
                    [float(obs_median) - float(obs_lo)],
                    [float(obs_hi) - float(obs_median)],
                ]
            )
        ax.errorbar(
            x=[marker_x],
            y=[float(obs_median)],
            yerr=yerr,
            fmt="o",
            color="C3",
            ecolor="C3",
            elinewidth=2,
            capsize=5,
            markersize=7,
            label="observed (95% CI)",
            zorder=5,
        )
    ax.axvline(marker_x, color="grey", linestyle=":", lw=0.8, alpha=0.7)

    if log_y:
        ax.set_yscale("log")
    if panel_time_axis == "fraction":
        ax.set_xlabel("fraction of evolve duration (-1 = healthy IC)")
        ax.set_xlim(-1.05, 0.05)
    elif panel_phase == "post_scenario":
        ax.set_xlabel("time since diagnosis (days)")
    else:
        ax.set_xlabel("time to diagnosis (days)")
    if units:
        ax.set_ylabel(units)
    title = obs_id
    if scenario:
        title = f"{obs_id}\n[{scenario}]"
    ax.set_title(title, fontsize=10)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)


__all__ = ["plot_calibration_trajectories"]
