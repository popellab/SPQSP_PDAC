"""Coverage-style assertion helpers for qualitative-bounds tests (gh #57).

Replaces the binary "assert value satisfies bound" pattern with
"assert fraction of N draws satisfying bound >= coverage_threshold".
Failures report observed coverage and the offending percentile so a
regression is diagnosable without re-running the suite.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

DEFAULT_COVERAGE_THRESHOLD = 0.8


def coverage_fraction(
    values: np.ndarray,
    *,
    lower: Optional[float] = None,
    upper: Optional[float] = None,
) -> tuple[float, np.ndarray]:
    """Return ``(fraction_satisfying, satisfied_mask)`` for a values array.

    NaN entries are treated as "did not satisfy" — they show up as
    simulator failures or missing-column reads and shouldn't be silently
    excluded from the denominator.
    """
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    satisfied = np.zeros(arr.shape, dtype=bool)
    if lower is not None:
        satisfied[finite] = arr[finite] >= lower
    else:
        satisfied[finite] = True
    if upper is not None:
        satisfied[finite] &= arr[finite] <= upper
    return float(satisfied.mean()), satisfied


def assert_coverage(
    values: np.ndarray,
    *,
    lower: Optional[float] = None,
    upper: Optional[float] = None,
    threshold: float = DEFAULT_COVERAGE_THRESHOLD,
    name: str = "",
    units: str = "",
) -> None:
    """Assert that ``threshold`` fraction of values lie within ``[lower, upper]``.

    Failure message includes:
        - observed coverage and threshold
        - bound and units
        - sample percentiles (p05, p50, p95) and NaN count for diagnosis

    Args:
        values: 1-D array of observable values across N draws.
        lower, upper: Bounds (at least one required). Inclusive.
        threshold: Required satisfaction fraction in (0, 1].
        name: Bound identifier for the failure message.
        units: Units string for the failure message.
    """
    if lower is None and upper is None:
        raise ValueError(f"assert_coverage[{name}]: lower or upper must be set")
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise AssertionError(
            f"[{name}] coverage check called with empty values array — "
            "no valid draws survived simulation; check evolve_success rates"
        )

    frac, satisfied = coverage_fraction(arr, lower=lower, upper=upper)
    if frac >= threshold:
        return

    finite = np.isfinite(arr)
    n_finite = int(finite.sum())
    n_nan = int((~finite).sum())
    bound_str = []
    if lower is not None:
        bound_str.append(f">= {lower:g}")
    if upper is not None:
        bound_str.append(f"<= {upper:g}")
    bound_desc = " and ".join(bound_str)

    if n_finite > 0:
        p05 = float(np.nanpercentile(arr, 5))
        p50 = float(np.nanpercentile(arr, 50))
        p95 = float(np.nanpercentile(arr, 95))
        percentile_line = (
            f"  observed: p05={p05:.4g}, p50={p50:.4g}, p95={p95:.4g} {units}"
        )
    else:
        percentile_line = "  observed: all NaN"

    raise AssertionError(
        f"[{name}] coverage {frac*100:.1f}% < threshold {threshold*100:.0f}%\n"
        f"  bound: value {bound_desc} {units}\n"
        f"  N: {arr.size} draws ({n_finite} finite, {n_nan} NaN)\n"
        f"{percentile_line}\n"
        f"  satisfying: {int(satisfied.sum())}/{arr.size}"
    )


def assert_relational_coverage(
    a: np.ndarray,
    b: np.ndarray,
    *,
    op: str = ">",
    threshold: float = DEFAULT_COVERAGE_THRESHOLD,
    name: str = "",
) -> None:
    """Assert that ``threshold`` fraction of paired ``(a_i, b_i)`` satisfy
    a relational predicate (e.g. ``a > b`` for "CD8 expands under treatment").

    Used for cross-time and cross-scenario invariants where the bound is on
    a relationship between two arrays rather than a value range.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(
            f"[{name}] assert_relational_coverage: shape mismatch a={a.shape} b={b.shape}"
        )
    if a.size == 0:
        raise AssertionError(f"[{name}] relational coverage on empty arrays")

    finite = np.isfinite(a) & np.isfinite(b)
    if op == ">":
        satisfied = finite & (a > b)
    elif op == ">=":
        satisfied = finite & (a >= b)
    elif op == "<":
        satisfied = finite & (a < b)
    elif op == "<=":
        satisfied = finite & (a <= b)
    elif op == "==":
        satisfied = finite & np.isclose(a, b)
    else:
        raise ValueError(f"[{name}] unknown op {op!r}")

    frac = float(satisfied.mean())
    if frac >= threshold:
        return

    n_satisfying = int(satisfied.sum())
    p50_a = float(np.nanmedian(a))
    p50_b = float(np.nanmedian(b))
    raise AssertionError(
        f"[{name}] relational coverage {frac*100:.1f}% < threshold {threshold*100:.0f}%\n"
        f"  bound: a {op} b\n"
        f"  N: {a.size} ({int(finite.sum())} finite); satisfying: {n_satisfying}/{a.size}\n"
        f"  median a={p50_a:.4g}, median b={p50_b:.4g}"
    )
