"""C++ replacement for test_qualitative_behaviors_helper.m.

Runs the qualitative-behavior scenarios at prior-median theta via
CppBatchRunner and produces the same ``sim_outputs`` dict layout that
the MATLAB helper emits through ``TEST_OUTPUT`` lines. The test bodies
in ``test_qualitative_behaviors.py`` consume this dict and do not need
to know which backend produced it.

Naming parity with the MATLAB helper:
    - V_T species land under bare names (``Mac_M1``, ``CD8``, ...)
    - V_ID species land under ``VID_<name>``
    - V_LN species land under ``VLN_<name>``
    - Compartment ``V_T`` lands under key ``V_T``
    - Aggregate rules (``C_total``, ``T_total``, ``M_total``, ``aPSC_total``)
      land under bare names
    - Hill functions land under their bare names under section ``hill`` /
      ``tx_d0_hill`` / ``tx_d21_hill``
    - ``vol_cell_mL`` is computed from the ``D_cell`` prior median
    - ``evolve_success`` is ``1`` iff the baseline sim's ``status == 0``
"""
from __future__ import annotations

import json
import math
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[2]


# Historical note: this fixture used to read a stage-2 promotion's
# run_config.json for things like cpp_binary_path/cpp_template_xml. After the
# in-repo C++ port (driver + resources landed under cpp/ and resources/cpp/),
# every consumed key is either overridden by _run_cfg below or has a default,
# so the stage-2 dependency was vestigial — removed so CI and fresh worktrees
# (which don't have a promoted symlink) can run the C++ qual fixture.


# V_T species → bare key (matches MATLAB helper's `strrep(name, 'V_T.', '')`)
VT_SPECIES: tuple[str, ...] = (
    "Mac_M1", "Mac_M2",
    "CD8", "CD8_exh",
    "Treg", "Th",
    "myCAF", "iCAF", "apCAF", "qPSC",
    "cDC1", "cDC2", "mcDC1", "mcDC2",
    "collagen", "C1",
    "CXCL12",
    "IL6", "IL10", "TGFb",
    "VEGF", "IFNg", "K", "MDSC",
)
# Aggregate rules (repeatedAssignment parameters, emitted bare in parquet)
AGG_RULES: tuple[str, ...] = ("C_total", "T_total", "M_total", "aPSC_total")
# Compartments that are also requested as scalars
COMPARTMENTS: tuple[str, ...] = ("V_T",)

# V_ID species → "VID_<name>" key
VID_SPECIES: tuple[str, ...] = ("GVAX_cells", "mAPC", "APC")
# V_LN species → "VLN_<name>" key
VLN_SPECIES: tuple[str, ...] = ("mcDC1", "mcDC2")

# Hill-function repeated-assignment rules to emit under a hill section.
HILL_RULES: tuple[str, ...] = (
    "H_TGFb", "H_TGFb_Teff", "H_TGFb_APC",
    "H_IL10",
    "H_IL12",
    "H_IL6_iCAF", "H_IL6_MDSC", "H_IL6_M2",
    "H_CCL5_Treg",
    "H_CXCL12_Texcl",
    "H_pore_APC", "H_pore_T", "H_stiff_Texh", "H_stiff_fib",
    "H_NO", "H_ArgI_Teff",
    "H_DAMP",
    "H_P1_apCAF",
)

# repeatedAssignment observables tracked at treatment timepoints
TX_OBSERVABLES: tuple[str, ...] = ("k_C_Tcell_eff",)


@dataclass(frozen=True)
class Snapshot:
    """A (scenario, time_index) pair to extract as a `sim_outputs` section."""

    key: str
    scenario_name: str
    at_time_days: float | None  # None → last timepoint


# Tier-1/2/3 snapshots (perturbation tests/tier-4 are deferred to Phase 2).
BASELINE_SCENARIO = "baseline_no_treatment"
PROGRESSION_SCENARIO = "clinical_progression"
TX_SCENARIO = "gvax_nivo_neoadjuvant_zheng2022"


def _priors_csv() -> Path:
    # Use the live priors CSV, not the stage-2 snapshot. The snapshot is
    # frozen at NPE-training time to keep the trained posterior valid for
    # PPC; the qual tests just want the current prior-median theta and
    # must stay in lockstep with the freshly regenerated cpp XML.
    return REPO_ROOT / "parameters/pdac_priors.csv"


def _run_cfg() -> dict:
    """Return the C++ fixture config.

    All four paths are repo-local: the three resource files live under
    ``resources/cpp/`` and the binary is built by ``make refresh-cpp``
    (or the CI workflow) into ``cpp/sim/build/qsp_sim``. Consumers treat
    a missing binary as a pytest skip via ``_require_qsp_sim``.
    """
    res = REPO_ROOT / "resources/cpp"
    return {
        "cpp_binary_path": str(REPO_ROOT / "cpp/sim/build/qsp_sim"),
        "cpp_template_xml": str(res / "param_all.xml"),
        "cpp_drug_metadata_yaml": str(res / "drug_metadata.yaml"),
        "cpp_healthy_state_yaml": str(res / "healthy_state.yaml"),
    }


def _require_qsp_sim() -> None:
    """Skip the current test if the qsp_sim binary hasn't been built."""
    binary = Path(_run_cfg()["cpp_binary_path"])
    if not binary.exists():
        import pytest
        pytest.skip(
            f"qsp_sim binary not built at {binary}. "
            "Run `make refresh-cpp` (or the CI build step) first."
        )


@lru_cache(maxsize=1)
def _median_theta() -> tuple[np.ndarray, list[str], dict[str, float]]:
    """Return (theta_row, param_names, param_value_by_name) at prior median.

    For lognormal rows the MATLAB helper uses ``exp(dist_param1)``; mirror
    that here so C++ and MATLAB start from the same point in parameter
    space. Non-lognormal rows fall back to the ``median`` column.
    """
    df = pd.read_csv(_priors_csv())
    names = df["name"].tolist()
    values = np.empty(len(names), dtype=np.float64)
    for i, row in df.iterrows():
        if row["distribution"] == "lognormal":
            values[i] = math.exp(float(row["dist_param1"]))
        else:
            values[i] = float(row["median"])

    # Fail fast if the priors drift past the regenerated XML — otherwise
    # every test in the suite crashes inside the XML renderer with an
    # opaque ParamNotFoundError. tests/test_cpp_xml_sync.py guards the
    # same invariant at collection time.
    import xml.etree.ElementTree as _ET
    xml_path = REPO_ROOT / "resources/cpp/param_all.xml"
    xml_names: set[str] = set()
    for e in _ET.parse(xml_path).iter():
        xml_names.add(e.tag)
        if "name" in e.attrib:
            xml_names.add(e.attrib["name"])
    missing = sorted(set(names) - xml_names)
    if missing:
        raise RuntimeError(
            f"{len(missing)} prior(s) not in {xml_path.relative_to(REPO_ROOT)}: "
            f"{missing[:10]}. Run `make refresh-cpp` or prune the CSV."
        )

    return values[None, :], names, dict(zip(names, values.tolist()))


def _scenario_yaml(scenario: str) -> Path:
    return REPO_ROOT / f"scenarios/{scenario}.yaml"


def _load_scenario_stop_time(scenario: str) -> float:
    import yaml

    data = yaml.safe_load(_scenario_yaml(scenario).read_text())
    sim_cfg = data.get("sim_config", {})
    return float(sim_cfg.get("stop_time", 1.0))


def _run_scenario(scenario: str, *, dt_days: float = 1.0) -> pd.DataFrame:
    """Run one scenario at median theta; return a single-row dataframe."""
    from qsp_hpc.cpp.batch_runner import CppBatchRunner

    theta, names, _ = _median_theta()
    cfg = _run_cfg()
    t_end = _load_scenario_stop_time(scenario)
    runner = CppBatchRunner(
        binary_path=cfg["cpp_binary_path"],
        template_path=cfg["cpp_template_xml"],
        subtree=cfg.get("cpp_subtree", "QSP"),
        scenario_yaml=_scenario_yaml(scenario),
        drug_metadata_yaml=cfg["cpp_drug_metadata_yaml"],
        healthy_state_yaml=cfg["cpp_healthy_state_yaml"],
    )
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / f"{scenario}.parquet"
        runner.run(
            theta_matrix=theta,
            param_names=names,
            t_end_days=t_end,
            dt_days=dt_days,
            output_path=out,
            scenario=scenario,
            seed=0,
        )
        return pq.read_table(str(out)).to_pandas()


def _time_index(time_arr: Sequence[float], at: float | None) -> int:
    if at is None:
        return len(time_arr) - 1
    for i, t in enumerate(time_arr):
        if t >= at:
            return i
    return len(time_arr) - 1


def _snapshot_values(row: pd.Series, idx: int) -> dict[str, float]:
    """Pull V_T / V_ID / V_LN species + aggregates + V_T compartment."""
    out: dict[str, float] = {}
    for sp in VT_SPECIES:
        col = f"V_T.{sp}"
        out[sp] = _at(row, col, idx)
    for sp in VID_SPECIES:
        out[f"VID_{sp}"] = _at(row, f"V_ID.{sp}", idx)
    for sp in VLN_SPECIES:
        out[f"VLN_{sp}"] = _at(row, f"V_LN.{sp}", idx)
    for name in AGG_RULES:
        out[name] = _at(row, name, idx)
    for name in COMPARTMENTS:
        out[name] = _at(row, name, idx)
    return out


def _hill_values(row: pd.Series, idx: int) -> dict[str, float]:
    return {name: _at(row, name, idx) for name in HILL_RULES}


def _at(row: pd.Series, col: str, idx: int) -> float:
    if col not in row.index:
        return float("nan")
    series = row[col]
    if series is None:
        return float("nan")
    try:
        return float(series[idx])
    except (IndexError, TypeError):
        return float("nan")


def _status_ok(df: pd.DataFrame) -> bool:
    return int(df["status"].iloc[0]) == 0


def build_sim_outputs(*, include_treatment: bool = True) -> dict:
    """Run scenarios at median theta; return the `sim_outputs` dict.

    Mirrors the shape test_qualitative_behaviors.py expects from the
    MATLAB helper: nested sections for `base`, `prog_dN`, `tx_dN`, plus
    flat scalars for `vol_cell_mL` and `evolve_success`.
    """
    _require_qsp_sim()
    _, _, param_map = _median_theta()
    D_cell_um = param_map.get("D_cell")
    if D_cell_um is None:
        # D_cell is a fixed model param, not in the priors CSV — read the
        # template default via a throwaway runner.
        from qsp_hpc.cpp.batch_runner import CppBatchRunner

        cfg = _run_cfg()
        probe = CppBatchRunner(
            binary_path=cfg["cpp_binary_path"],
            template_path=cfg["cpp_template_xml"],
            subtree=cfg.get("cpp_subtree", "QSP"),
        )
        D_cell_um = float(probe.template_defaults["D_cell"])
    vol_cell_mL = (4.0 / 3.0) * math.pi * (D_cell_um / 2.0 * 1e-4) ** 3

    out: dict = {"vol_cell_mL": vol_cell_mL}

    # --- Tier 1: baseline snapshot ---
    base_df = _run_scenario(BASELINE_SCENARIO, dt_days=0.5)
    base_ok = _status_ok(base_df)
    out["evolve_success"] = 1 if base_ok else 0
    if base_ok:
        row = base_df.iloc[0]
        time_arr = row["time"]
        idx0 = _time_index(time_arr, at=None)  # last (and typically only) point
        out["base"] = _snapshot_values(row, idx0)
        out["hill"] = _hill_values(row, idx0)

    # --- Tier 2: progression (0..90d) ---
    prog_df = _run_scenario(PROGRESSION_SCENARIO, dt_days=1.0)
    if _status_ok(prog_df):
        row = prog_df.iloc[0]
        time_arr = row["time"]
        for key, at in [
            ("prog_d0", 0.0),
            ("prog_d10", 10.0),
            ("prog_d30", 30.0),
            ("prog_d60", 60.0),
            ("prog_d90", 90.0),
        ]:
            idx = _time_index(time_arr, at)
            out[key] = _snapshot_values(row, idx)
            # Observables (bare repeatedAssignment rule names)
            if key == "prog_d0":
                for obs in TX_OBSERVABLES:
                    out[key][obs] = _at(row, obs, idx)
        # prog.C1_min for monotonicity test
        c1 = row["V_T.C1"]
        out["prog"] = {"C1_min": float(min(c1))}

    # --- Tier 3: GVAX+nivo ---
    if include_treatment:
        tx_df = _run_scenario(TX_SCENARIO, dt_days=1.0)
        if _status_ok(tx_df):
            row = tx_df.iloc[0]
            time_arr = row["time"]
            for key, at in [
                ("tx_d0", 0.0),
                ("tx_d3", 3.0),
                ("tx_d7", 7.0),
                ("tx_d14", 14.0),
                ("tx_d21", 21.0),
            ]:
                idx = _time_index(time_arr, at)
                out[key] = _snapshot_values(row, idx)
                for obs in TX_OBSERVABLES:
                    out[key][obs] = _at(row, obs, idx)
            # Hill function values at treatment endpoints
            out["tx_d0_hill"] = _hill_values(row, _time_index(time_arr, 0.0))
            out["tx_d21_hill"] = _hill_values(row, _time_index(time_arr, 21.0))

    return out


# ===========================================================================
# evolve_to_diagnosis metadata (test_evolve_to_diagnosis.py)
# ===========================================================================


def build_evolve_outputs() -> dict:
    """Mirror test_evolve_to_diagnosis_helper.m's TEST_OUTPUT keys.

    Returns the same scalar keys the MATLAB helper emits:
        D_cell_um, init_V_tumor_mL, init_CD8,
        evolution_success, time_to_diagnosis, V_T_diagnosis_mL,
        diam_diagnosis_cm, CD8_density, TAM_density, M1_M2_ratio,
        C1_density, collagen_diagnosis_mg,
        fallback_success, fallback_diam_cm, fallback_V_T_mL

    Runs the baseline scenario (which bundles evolve_to_diagnosis) via
    a :class:`CppEvolveCache` so the QSTH header carries
    ``t_diagnosis_days`` + ``vt_diameter_cm``. The "fallback" path has
    no C++ analogue (no MATLAB variants); we report the baseline values
    since the default C++ path already reads ``initial_tumour_diameter``
    from the template XML.
    """
    import tempfile

    from qsp_hpc.cpp.evolve_cache import CppEvolveCache, QsthHeader
    from qsp_hpc.cpp.param_xml import ParamXMLRenderer
    from qsp_hpc.cpp.runner import CppRunner
    import yaml

    _require_qsp_sim()
    out: dict = {}
    cfg = _run_cfg()

    # --- Template defaults (D_cell, CD8 density, default_tumor_cells) ---
    from qsp_hpc.cpp.batch_runner import CppBatchRunner

    probe_runner = CppBatchRunner(
        binary_path=cfg["cpp_binary_path"],
        template_path=cfg["cpp_template_xml"],
        subtree=cfg.get("cpp_subtree", "QSP"),
    )
    D_cell_um = float(probe_runner.template_defaults["D_cell"])
    vol_cell_mL = (4.0 / 3.0) * math.pi * (D_cell_um / 2.0 * 1e-4) ** 3

    hs_yaml = yaml.safe_load(Path(cfg["cpp_healthy_state_yaml"]).read_text())
    n_tumor = float(hs_yaml["default_tumor_cells"])
    cd8_density_per_mm3 = float(hs_yaml["cell_densities_per_mm3"]["CD8"])

    init_V_mL = n_tumor * vol_cell_mL
    init_V_mm3 = init_V_mL * 1000.0

    out["D_cell_um"] = D_cell_um
    out["init_V_tumor_mL"] = init_V_mL
    out["init_CD8"] = cd8_density_per_mm3 * init_V_mm3

    # --- Build the evolve blob to read t_diagnosis / vt_diameter ---
    theta_row, names, _ = _median_theta()
    params = {name: val for name, val in zip(names, theta_row[0].tolist())}

    runner = CppRunner(
        binary_path=cfg["cpp_binary_path"],
        template_path=cfg["cpp_template_xml"],
        subtree=cfg.get("cpp_subtree", "QSP"),
        scenario_yaml=_scenario_yaml(BASELINE_SCENARIO),
        drug_metadata_yaml=cfg["cpp_drug_metadata_yaml"],
        healthy_state_yaml=cfg["cpp_healthy_state_yaml"],
    )
    renderer = ParamXMLRenderer(
        template_path=cfg["cpp_template_xml"], subtree=cfg.get("cpp_subtree", "QSP")
    )

    with tempfile.TemporaryDirectory() as td:
        cache = CppEvolveCache(cache_root=Path(td) / "evolve", renderer=renderer, runner=runner)
        state_path, _ = cache.get_or_build(params=params, workdir=Path(td) / "work")
        header = QsthHeader.parse(state_path)

    out["evolution_success"] = 1
    out["time_to_diagnosis"] = header.t_diagnosis_days
    out["diam_diagnosis_cm"] = header.vt_diameter_cm
    out["V_T_diagnosis_mL"] = (math.pi / 6.0) * header.vt_diameter_cm ** 3

    # --- Post-evolve species densities via the baseline scenario ---
    base_df = _run_scenario(BASELINE_SCENARIO, dt_days=0.5)
    row = base_df.iloc[0]
    idx0 = _time_index(row["time"], at=None)
    V_T = _at(row, "V_T", idx0)
    V_T_mm3 = V_T * 1000.0 if V_T > 0 else float("nan")

    def _dens(count: float) -> float:
        if math.isnan(V_T_mm3):
            return float("nan")
        return count / V_T_mm3

    CD8 = _at(row, "V_T.CD8", idx0)
    Treg = _at(row, "V_T.Treg", idx0)
    M1 = _at(row, "V_T.Mac_M1", idx0)
    M2 = _at(row, "V_T.Mac_M2", idx0)
    C1 = _at(row, "V_T.C1", idx0)
    out["C1_diagnosis"] = C1
    out["C1_density"] = _dens(C1)
    out["CD8_diagnosis"] = CD8
    out["CD8_density"] = _dens(CD8)
    out["Treg_diagnosis"] = Treg
    out["Treg_density"] = _dens(Treg)
    out["TAM_diagnosis"] = M1 + M2
    out["TAM_density"] = _dens(M1 + M2)
    out["M1_M2_ratio"] = M1 / max(M2, 1.0)
    out["collagen_diagnosis_mg"] = _at(row, "V_T.collagen", idx0)

    # No C++ analogue for MATLAB-variant fallback; default C++ path reads
    # initial_tumour_diameter from the template, so fallback == baseline.
    out["fallback_success"] = 1
    out["fallback_diam_cm"] = out["diam_diagnosis_cm"]
    out["fallback_V_T_mL"] = out["V_T_diagnosis_mL"]

    return out


# ===========================================================================
# Coverage outputs (gh #57): run scenarios across N theta draws so qualitative
# bounds tests can be expressed as coverage checks rather than single-point
# assertions. Reuses the existing 177-feature restriction classifier from
# pdac-build-npe-sweep (drift-tolerant: top features are tumor-evolution
# params, the 10 newer priors live outside the dominant decision subspace).
# ===========================================================================


# DRIFT WATCH: this is a checked-in trained ML model and decays as priors
# evolve. Recall under the live prior must stay >= 0.95 — re-probe with
# tests/fixtures/restriction_classifier/probe_drift.py and regen per
# tests/fixtures/restriction_classifier/README.md if it slips.
# Trained on 100k draws. AUC=0.989, recall=0.982, precision=0.884 at the
# last drift probe (N=200, 2026-04-24).
_RESTRICTION_CLASSIFIER_DIR = REPO_ROOT / "tests/fixtures/restriction_classifier"
# Linear-space prior median for IFNg_50 (retired in c06e381). The classifier
# was trained when this prior existed; we fill it with its old median so the
# remaining 176 overlapping features carry the decision. Drift testing showed
# IFNg_50 is not in the top-15 features by permutation importance, so this
# fill barely matters quantitatively.
_IFNG50_FILL = 1000.0
_COVERAGE_CACHE_DIR = REPO_ROOT / "cache/qual_coverage"


@lru_cache(maxsize=1)
def _load_restriction_classifier():
    """Load the cached HistGBM classifier; returns ``(clf, feature_order)``."""
    from qsp_inference.inference.restriction import RestrictionClassifier

    clf = RestrictionClassifier.load(_RESTRICTION_CLASSIFIER_DIR)
    return clf, list(clf.feature_order)


def _project_to_classifier_features(
    theta_full: np.ndarray, full_names: list[str]
) -> np.ndarray:
    """Reorder/drop columns of ``theta_full`` to match the classifier's
    feature_order. Live-only params are dropped; classifier-only params (just
    ``IFNg_50`` today) are filled with their pre-retirement prior median."""
    _, clf_features = _load_restriction_classifier()
    name_to_col = {n: i for i, n in enumerate(full_names)}
    out = np.empty((theta_full.shape[0], len(clf_features)), dtype=np.float64)
    for j, feat in enumerate(clf_features):
        if feat in name_to_col:
            out[:, j] = theta_full[:, name_to_col[feat]]
        elif feat == "IFNg_50":
            out[:, j] = _IFNG50_FILL
        else:
            raise RuntimeError(
                f"Classifier feature {feat!r} not present in live priors and "
                "no fill value defined. Update _project_to_classifier_features."
            )
    return out


def _sample_filtered_thetas(
    n_target: int,
    *,
    seed: int = 2025,
    use_restriction: bool = True,
    oversample_factor: float = 1.6,
    max_attempts: int = 6,
    composite_prior_yaml: Path | str | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Sample ``n_target`` thetas from the live prior, optionally filtered
    through the restriction classifier so almost all draws will evolve to
    diagnosis successfully.

    If ``composite_prior_yaml`` is set, samples from the Stage-1-composed
    prior (base CSV + submodel-target posteriors via gaussian copula)
    instead of the base CSV prior. Use this to probe the prior actually fed
    into Stage-2 NPE training.

    Returns ``(theta, names)`` where ``theta`` has shape ``(n_target, P)``
    in the live priors-CSV column order.
    """
    import torch
    from qsp_inference.priors import load_composite_prior_log, load_prior

    csv = _priors_csv()
    full_names = pd.read_csv(csv)["name"].tolist()
    if composite_prior_yaml is not None:
        prior, _ = load_composite_prior_log(str(composite_prior_yaml), str(csv))
        # load_composite_prior_log returns log-space samples by design;
        # the C++ binary and restriction classifier consume raw-space rates.
        log_space = True
    else:
        prior = load_prior(csv)
        log_space = False

    def _draw(n: int, sd: int) -> np.ndarray:
        torch.manual_seed(sd)
        s = prior.sample((n,)).numpy().astype(np.float64)
        return np.exp(s) if log_space else s

    if not use_restriction:
        return _draw(n_target, seed), full_names

    clf, _ = _load_restriction_classifier()
    accepted: list[np.ndarray] = []
    n_accepted = 0
    factor = float(oversample_factor)
    for attempt in range(max_attempts):
        batch_n = max(int(factor * (n_target - n_accepted)), 64)
        batch_full = _draw(batch_n, seed + attempt)
        batch_clf = _project_to_classifier_features(batch_full, full_names)
        keep = clf.accept(batch_clf)
        if int(keep.sum()) > 0:
            accepted.append(batch_full[keep])
            n_accepted += int(keep.sum())
        if n_accepted >= n_target:
            break
        factor *= 1.5
    if n_accepted < n_target:
        raise RuntimeError(
            f"_sample_filtered_thetas: got {n_accepted}/{n_target} after "
            f"{max_attempts} attempts; classifier may be misaligned with priors"
        )
    theta = np.concatenate(accepted, axis=0)[:n_target]
    return theta, full_names


def _run_scenario_batch(
    scenario: str,
    theta: np.ndarray,
    names: list[str],
    *,
    evolve_cache_root: Path,
    dt_days: float = 1.0,
) -> pd.DataFrame:
    """Run ``scenario`` for N theta in a single batched CppBatchRunner call,
    sharing an evolve cache so evolve-to-diagnosis runs once per (theta,
    healthy-state-relevant-params) tuple instead of once per scenario."""
    from qsp_hpc.cpp.batch_runner import CppBatchRunner

    cfg = _run_cfg()
    t_end = _load_scenario_stop_time(scenario)
    runner = CppBatchRunner(
        binary_path=cfg["cpp_binary_path"],
        template_path=cfg["cpp_template_xml"],
        subtree=cfg.get("cpp_subtree", "QSP"),
        scenario_yaml=_scenario_yaml(scenario),
        drug_metadata_yaml=cfg["cpp_drug_metadata_yaml"],
        healthy_state_yaml=cfg["cpp_healthy_state_yaml"],
        evolve_cache_root=str(evolve_cache_root),
    )
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / f"{scenario}.parquet"
        runner.run(
            theta_matrix=theta,
            param_names=names,
            t_end_days=t_end,
            dt_days=dt_days,
            output_path=out,
            scenario=scenario,
            seed=0,
        )
        return pq.read_table(str(out)).to_pandas()


def _coverage_cache_key(theta: np.ndarray, *, include_treatment: bool) -> str:
    """Stable hash for caching the dict-of-arrays result on disk. Keyed by
    theta-bytes + qsp_sim mtime + scenario set."""
    import hashlib

    binary = Path(_run_cfg()["cpp_binary_path"])
    binary_mtime = binary.stat().st_mtime_ns if binary.exists() else 0
    h = hashlib.sha256()
    h.update(theta.tobytes())
    h.update(str(binary_mtime).encode())
    h.update(b"|tx" if include_treatment else b"|notx")
    return h.hexdigest()[:16]


def build_coverage_outputs(
    *,
    n_draws: int = 100,
    seed: int = 2025,
    use_restriction: bool = True,
    include_treatment: bool = True,
    cache: bool = True,
    composite_prior_yaml: Path | str | None = None,
    sampler_oversample_factor: float = 1.6,
    sampler_max_attempts: int = 6,
) -> dict:
    """N-draw analogue of ``build_sim_outputs``.

    Same key layout as ``build_sim_outputs`` but every leaf scalar becomes a
    1-D float array of length ``n_valid`` (only thetas where all requested
    scenarios returned ``status == 0``). Tests then compute coverage as
    ``(arr satisfies bound).mean()`` and assert ``>= threshold``.

    Top-level metadata keys:
        ``n_draws_requested``: ``n_draws``
        ``n_valid``: count of thetas surviving every scenario (so coverage
            denominators are trustworthy)
        ``vol_cell_mL``: same scalar as build_sim_outputs (D_cell is fixed)

    Caches the resulting dict-of-arrays at
    ``cache/qual_coverage/runs/<theta_hash>.npz`` so repeat sessions hit
    the cache in O(read).
    """
    _require_qsp_sim()

    theta, names = _sample_filtered_thetas(
        n_draws, seed=seed, use_restriction=use_restriction,
        composite_prior_yaml=composite_prior_yaml,
        oversample_factor=sampler_oversample_factor,
        max_attempts=sampler_max_attempts,
    )
    cache_key = _coverage_cache_key(theta, include_treatment=include_treatment)
    cache_path = _COVERAGE_CACHE_DIR / "runs" / f"{cache_key}.npz"
    if cache and cache_path.exists():
        loaded = np.load(cache_path, allow_pickle=True)
        return {k: loaded[k].item() if loaded[k].ndim == 0 else loaded[k]
                for k in loaded.files}

    # D_cell is a fixed model param (not in priors CSV) — read from template.
    from qsp_hpc.cpp.batch_runner import CppBatchRunner
    cfg = _run_cfg()
    probe = CppBatchRunner(
        binary_path=cfg["cpp_binary_path"],
        template_path=cfg["cpp_template_xml"],
        subtree=cfg.get("cpp_subtree", "QSP"),
    )
    D_cell_um = float(probe.template_defaults["D_cell"])
    vol_cell_mL = (4.0 / 3.0) * math.pi * (D_cell_um / 2.0 * 1e-4) ** 3

    evolve_cache_root = _COVERAGE_CACHE_DIR / "evolve"
    evolve_cache_root.mkdir(parents=True, exist_ok=True)

    # Run each scenario once for all N theta. Successful-row mask is the
    # intersection across scenarios so each output array represents thetas
    # that completed every scenario.
    base_df = _run_scenario_batch(
        BASELINE_SCENARIO, theta, names,
        evolve_cache_root=evolve_cache_root, dt_days=0.5,
    )
    prog_df = _run_scenario_batch(
        PROGRESSION_SCENARIO, theta, names,
        evolve_cache_root=evolve_cache_root, dt_days=1.0,
    )
    scenario_dfs = {"base": base_df, "prog": prog_df}
    if include_treatment:
        scenario_dfs["tx"] = _run_scenario_batch(
            TX_SCENARIO, theta, names,
            evolve_cache_root=evolve_cache_root, dt_days=1.0,
        )

    valid_mask = np.ones(len(theta), dtype=bool)
    for df in scenario_dfs.values():
        # status column has one int per row; align by sample_index.
        s = df.set_index("sample_index").reindex(np.arange(len(theta)))
        valid_mask &= (s["status"].fillna(-1).to_numpy() == 0)
    n_valid = int(valid_mask.sum())

    out: dict = {
        "n_draws_requested": np.asarray(n_draws),
        "n_valid": np.asarray(n_valid),
        "vol_cell_mL": np.asarray(vol_cell_mL),
        "theta_valid": theta[valid_mask].astype(np.float64),
        "param_names": np.asarray(names, dtype=object),
    }

    def _gather_snapshot_arrays(df: pd.DataFrame, at: float | None) -> dict[str, np.ndarray]:
        """Vectorized analogue of _snapshot_values across all valid rows."""
        df = df.set_index("sample_index").reindex(np.arange(len(theta)))
        df = df[valid_mask]
        n = len(df)
        snap: dict[str, np.ndarray] = {}
        # Helper: pull col, then index at `at` per row.
        def _take(col: str) -> np.ndarray:
            if col not in df.columns:
                return np.full(n, float("nan"))
            arr = np.empty(n, dtype=np.float64)
            for i, series in enumerate(df[col].to_numpy()):
                if series is None:
                    arr[i] = float("nan")
                    continue
                if at is None:
                    idx = len(series) - 1
                else:
                    # find first time >= at
                    times = df["time"].iloc[i]
                    idx = next((k for k, t in enumerate(times) if t >= at),
                               len(times) - 1)
                try:
                    arr[i] = float(series[idx])
                except (IndexError, TypeError):
                    arr[i] = float("nan")
            return arr

        for sp in VT_SPECIES:
            snap[sp] = _take(f"V_T.{sp}")
        for sp in VID_SPECIES:
            snap[f"VID_{sp}"] = _take(f"V_ID.{sp}")
        for sp in VLN_SPECIES:
            snap[f"VLN_{sp}"] = _take(f"V_LN.{sp}")
        for name in AGG_RULES:
            snap[name] = _take(name)
        for name in COMPARTMENTS:
            snap[name] = _take(name)
        return snap

    def _gather_hill_arrays(df: pd.DataFrame, at: float | None) -> dict[str, np.ndarray]:
        df = df.set_index("sample_index").reindex(np.arange(len(theta)))
        df = df[valid_mask]
        n = len(df)
        out_h: dict[str, np.ndarray] = {}
        for name in HILL_RULES:
            if name not in df.columns:
                out_h[name] = np.full(n, float("nan"))
                continue
            arr = np.empty(n, dtype=np.float64)
            for i, series in enumerate(df[name].to_numpy()):
                if series is None:
                    arr[i] = float("nan")
                    continue
                if at is None:
                    idx = len(series) - 1
                else:
                    times = df["time"].iloc[i]
                    idx = next((k for k, t in enumerate(times) if t >= at),
                               len(times) - 1)
                try:
                    arr[i] = float(series[idx])
                except (IndexError, TypeError):
                    arr[i] = float("nan")
            out_h[name] = arr
        return out_h

    # Tier 1: baseline diagnosis snapshot
    out["base"] = _gather_snapshot_arrays(base_df, at=None)
    out["hill"] = _gather_hill_arrays(base_df, at=None)

    # Tier 2: progression at d0/10/21/30/60/90.
    # prog_d21 lets GVAX-vs-untreated tests compare tx_d21 to a matched-time
    # untreated snapshot. prog_d0 also carries TX_OBSERVABLES so
    # checkpoint-fold tests can compare tx_d14 / prog_d0 (the legacy pattern
    # in build_sim_outputs).
    for key, at in [("prog_d0", 0.0), ("prog_d10", 10.0),
                    ("prog_d21", 21.0),
                    ("prog_d30", 30.0), ("prog_d60", 60.0),
                    ("prog_d90", 90.0)]:
        out[key] = _gather_snapshot_arrays(prog_df, at=at)
        if key == "prog_d0":
            sub = prog_df.set_index("sample_index").reindex(np.arange(len(theta)))
            sub = sub[valid_mask]
            for obs in TX_OBSERVABLES:
                if obs not in sub.columns:
                    out[key][obs] = np.full(int(valid_mask.sum()), float("nan"))
                    continue
                arr = np.empty(int(valid_mask.sum()), dtype=np.float64)
                for i, series in enumerate(sub[obs].to_numpy()):
                    if series is None:
                        arr[i] = float("nan")
                        continue
                    times = sub["time"].iloc[i]
                    idx = next((k for k, t in enumerate(times) if t >= at),
                               len(times) - 1)
                    try:
                        arr[i] = float(series[idx])
                    except (IndexError, TypeError):
                        arr[i] = float("nan")
                out[key][obs] = arr

    # progression-min C1 (monotonicity test)
    sub = prog_df.set_index("sample_index").reindex(np.arange(len(theta)))
    sub = sub[valid_mask]
    c1_mins = np.array(
        [float(min(s)) if s is not None else float("nan")
         for s in sub["V_T.C1"].to_numpy()]
    )
    out["prog"] = {"C1_min": c1_mins}

    # Tier 3: GVAX+nivo at d0/3/7/14/21
    if include_treatment:
        tx_df = scenario_dfs["tx"]
        for key, at in [("tx_d0", 0.0), ("tx_d3", 3.0), ("tx_d7", 7.0),
                        ("tx_d14", 14.0), ("tx_d21", 21.0)]:
            snap = _gather_snapshot_arrays(tx_df, at=at)
            for obs in TX_OBSERVABLES:
                # bare repeatedAssignment columns
                t_df = tx_df.set_index("sample_index").reindex(np.arange(len(theta)))
                t_df = t_df[valid_mask]
                arr = np.empty(int(valid_mask.sum()), dtype=np.float64)
                for i, series in enumerate(t_df[obs].to_numpy() if obs in t_df.columns else [None] * len(t_df)):
                    if series is None:
                        arr[i] = float("nan")
                        continue
                    times = t_df["time"].iloc[i]
                    idx = next((k for k, t in enumerate(times) if t >= at),
                               len(times) - 1)
                    try:
                        arr[i] = float(series[idx])
                    except (IndexError, TypeError):
                        arr[i] = float("nan")
                snap[obs] = arr
            out[key] = snap
        out["tx_d0_hill"] = _gather_hill_arrays(tx_df, at=0.0)
        out["tx_d21_hill"] = _gather_hill_arrays(tx_df, at=21.0)

    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # np.savez doesn't preserve nested dicts; save a flat namespace and
        # let the loader rehydrate via dotted keys on demand. For now we
        # pickle the whole dict via savez_compressed's allow_pickle path.
        np.savez_compressed(cache_path, **{
            k: np.array(v, dtype=object) if isinstance(v, dict) else np.asarray(v)
            for k, v in out.items()
        })
    return out
