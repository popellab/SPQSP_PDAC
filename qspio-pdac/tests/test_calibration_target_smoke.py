"""Order-of-magnitude smoke test for calibration-target compute_observable.

Catches the unit-conversion bug class (e.g., the 1000x collagen / TGFb
bugs found 2026-05-11) by plugging *typical* species values into each
target's compute_observable and asserting the result lands within ~2
orders of magnitude of the target's empirical_data.median. A wider gap
almost always indicates an author error in unit handling inside the
YAML's code block.

Typical species values live in tests/fixtures/typical_species.yaml.
When adding a new target whose species aren't in the fixture, add them
(a quick single-trace simulation median or a textbook estimate is fine).
Targets referencing un-fixtured species are skipped, not failed.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
TARGETS_DIR = REPO / "calibration_targets"
FIXTURE = REPO / "tests" / "fixtures" / "typical_species.yaml"

# Targets in these subtrees use a different schema (submodel, excluded)
# or require a full trajectory rather than a static t=0 species_dict
# (treatment-day-21 fold changes, longterm clinical endpoints, mechanistic
# peak-timing). The smoke only meaningfully evaluates diagnosis-state
# observables, where empirical_data.median lives in the same magnitude
# regime as compute_observable applied to typical species values.
SKIP_SUBTREES = {
    "submodel_targets",
    "excluded",
    "to-review",
    "predictions",
    "mechanistic",
    "gvax_neoadjuvant",
    "gvax_nivo_neoadjuvant",
    "clinical_progression",
    "cross_scenario",
}

# How far the smoke output may stray from empirical median before we
# call it a unit bug. 2.0 OOM catches the canonical 1000x bug class
# at a single typical θ point (the pre-fix collagen bug sits at log
# ratio ≈ -2.07 here — just inside the tolerance).
#
# Limitations:
#   - Cannot catch unit bugs that are masked by a compensating
#     magnitude error in the model output. Example: the pre-fix TGFb
#     code was 1000x too small as a formula but the model's V_T.TGFb
#     was ~100x larger than physiology, so the net single-θ gap was
#     only ~10x — well inside this tolerance. Such cases need the
#     dimensional (Pint) check or direct algebraic review.
#   - Targets where the prior-median model output is genuinely far
#     from the empirical target (real misspec, not a unit bug) will
#     flag here. Mark those with @pytest.mark.xfail and a comment so
#     the smoke remains a unit-bug detector, not a misspec dashboard.
OOM_TOLERANCE = 2.0

# Targets known to fail this OOM smoke for reasons other than unit bugs.
# Each entry is (path-fragment, reason). The smoke xfails these so the
# rest of the suite stays green; revisit when the underlying misspec is
# resolved or the fixture aux value is tightened to biology.
KNOWN_NON_UNIT_BUG_GAPS = {
    "il12_concentration_tumor_PDAC_deriv001": (
        "Model overproduces tumor IL-12 vs serum target (log ratio ~+2.1 with "
        "f_serum_to_tumor_IL12=1.0 fixture default). Real misspec signal already "
        "tracked in the misspec audit, not a units error."
    ),
}


def _iter_target_yamls():
    for p in TARGETS_DIR.rglob("*.yaml"):
        if any(part in SKIP_SUBTREES for part in p.parts):
            continue
        yield p


def _load_fixture():
    f = yaml.safe_load(FIXTURE.read_text())
    return f.get("species", {}), f.get("params", {}), f.get("aux", {})


def _compile_observable(code: str):
    ns: dict = {}
    exec(code, ns)
    fn = ns.get("compute_observable")
    if fn is None:
        raise RuntimeError("compute_observable function not defined in code block")
    return fn


def _gather_aux_param_names(target: dict) -> list[str]:
    """Aux params can be top-level or under observable; collect from both."""
    out: list[str] = []
    for src in (target.get("auxiliary_parameters") or [], (target.get("observable") or {}).get("auxiliary_parameters") or []):
        for ap in src:
            if isinstance(ap, dict) and ap.get("name"):
                out.append(ap["name"])
    return out


@pytest.fixture(scope="module")
def fixture_vals():
    return _load_fixture()


@pytest.mark.parametrize(
    "target_path",
    sorted(_iter_target_yamls()),
    ids=lambda p: str(p.relative_to(TARGETS_DIR)),
)
def test_observable_oom_smoke(target_path: Path, fixture_vals, request):
    species_vals, param_vals, aux_vals = fixture_vals
    for frag, reason in KNOWN_NON_UNIT_BUG_GAPS.items():
        if frag in str(target_path):
            request.applymarker(pytest.mark.xfail(reason=reason, strict=False))
            break
    target = yaml.safe_load(target_path.read_text())
    if not isinstance(target, dict):
        pytest.skip("not a target dict")
    obs = target.get("observable") or {}
    code = obs.get("code")
    if not code:
        pytest.skip("no compute_observable code block")

    # Resolve species_dict: the runtime loader merges species values with
    # any params the YAML references via species_dict, so we do the same.
    needed_species = list(obs.get("species") or [])
    species_dict: dict = {}
    missing: list[str] = []
    for s in needed_species:
        if s in species_vals:
            species_dict[s] = float(species_vals[s])
        elif s in param_vals:
            species_dict[s] = float(param_vals[s])
        else:
            missing.append(s)
    if missing:
        pytest.skip(f"typical values missing for: {missing} (add to tests/fixtures/typical_species.yaml)")

    # constants: raw values from the YAML (this matches the runtime
    # loader's behavior at trajectory_eval.py — units block is treated
    # as documentation, not enforced).
    constants = {c["name"]: float(c["value"]) for c in (obs.get("constants") or [])}

    # aux params: best-effort defaults (mostly dimensionless multipliers
    # near 1.0). The OOM tolerance absorbs minor mismatch here.
    for ap_name in _gather_aux_param_names(target):
        constants.setdefault(ap_name, float(aux_vals.get(ap_name, 1.0)))

    # Also expose common params that some observables look up via the
    # constants dict (e.g., Ve_T for stromal_fraction).
    for pname, pval in param_vals.items():
        constants.setdefault(pname, float(pval))

    fn = _compile_observable(code)
    try:
        result = fn(0.0, species_dict, constants)
    except KeyError as e:
        pytest.skip(f"compute_observable needs species/const {e} not in fixture")
    except Exception as e:
        pytest.fail(f"compute_observable raised: {e!r}")

    # Result may be scalar, list, or numpy array — squeeze to a single
    # representative number (median if iterable).
    try:
        if hasattr(result, "__len__"):
            import numpy as np

            arr = np.asarray(result, dtype=float)
            val = float(np.nanmedian(arr)) if arr.size else float("nan")
        else:
            val = float(result)
    except Exception as e:
        pytest.fail(f"could not coerce compute_observable result to float: {result!r} ({e!r})")

    if not math.isfinite(val):
        pytest.fail(f"compute_observable returned non-finite value: {val}")

    emp = (target.get("empirical_data") or {}).get("median")
    if emp is None:
        pytest.skip("no empirical_data.median to compare against")
    if isinstance(emp, list):
        emp = emp[0] if emp else None
    if emp is None:
        pytest.skip("empirical_data.median is empty list")
    emp = float(emp)

    # Both signs allowed; require magnitudes within OOM_TOLERANCE.
    if val == 0 and emp == 0:
        return
    if val == 0 or emp == 0:
        pytest.fail(f"one of val/emp is zero (val={val}, emp={emp})")

    log_ratio = math.log10(abs(val) / abs(emp))
    assert abs(log_ratio) <= OOM_TOLERANCE, (
        f"{target_path.relative_to(TARGETS_DIR)}: "
        f"compute_observable returned {val:.4g}, empirical median {emp:.4g} "
        f"(log10 ratio = {log_ratio:+.2f}, tolerance = ±{OOM_TOLERANCE}). "
        "Likely a units mismatch between the constants block and the "
        "compute_observable code — see tests/test_calibration_target_smoke.py docstring."
    )
