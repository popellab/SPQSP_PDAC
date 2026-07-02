"""Schema + executability tests for the generated prediction YAMLs.

Checks that every file under ``calibration_targets/predictions/`` validates
against :class:`scripts.prediction_target.PredictionTarget`, that its
``observable.code`` compiles and defines ``compute_observable``, and that the
function produces a finite numpy array on a synthetic trajectory. The
generator script ``scripts/generate_prediction_yamls.py`` is exercised
end-to-end in a temp directory to catch drift between ``clinical_endpoints.py``
and the committed YAMLs.

Pintless interface (post-PR3 drop-Pint-from-calibration-targets): species_dict
values and the time vector are raw numpy arrays in canonical model_structure
units (cells in ``cell``, V_T in ``milliliter``, time in ``day``).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
PRED_DIR = ROOT / "calibration_targets" / "predictions"

sys.path.insert(0, str(SCRIPTS))

from prediction_target import PredictionTarget  # noqa: E402


@pytest.fixture(scope="module")
def synthetic_trajectory():
    """Slow-growing tumor: VDT=200d starting from 1e9 cells, 3y horizon.

    Returns raw-float arrays in canonical units (time=day, V_T.C1=cell,
    V_T=milliliter).
    """
    t = np.linspace(0, 1095, 50)
    c1 = 1e9 * np.exp(t * math.log(2) / 200)
    v_cm3 = c1 / 1.0e9  # canonical 1e9 cells/cm³ assumption
    diameters_cm = 2.0 * np.cbrt(3.0 * v_cm3 / (4.0 * math.pi))
    # Sphere volume in mL (cm³ ≡ mL)
    v_t_mL = (4.0 / 3.0) * math.pi * (diameters_cm / 2.0) ** 3
    return t, {"V_T.C1": c1, "V_T": v_t_mL}


def _yaml_paths():
    """All prediction YAMLs, recursing into per-scenario subdirs."""
    paths = sorted(PRED_DIR.rglob("*.yaml"))
    assert paths, f"No prediction YAMLs found in {PRED_DIR}"
    return paths


def _yaml_id(path: Path) -> str:
    """Test id: ``<scenario>/<endpoint>`` so parametrize names are unique."""
    return f"{path.parent.name}/{path.stem}"


class TestPredictionTargetSchema:
    @pytest.mark.parametrize("path", _yaml_paths(), ids=_yaml_id)
    def test_yaml_validates(self, path):
        doc = yaml.safe_load(path.read_text())
        PredictionTarget.model_validate(doc)

    @pytest.mark.parametrize("path", _yaml_paths(), ids=_yaml_id)
    def test_observable_code_compiles(self, path):
        doc = yaml.safe_load(path.read_text())
        code = doc["observable"]["code"]
        ns: dict = {}
        exec(code, ns)
        assert callable(ns.get("compute_observable"))

    def test_unique_ids_per_scenario(self):
        """(scenario, prediction_target_id) must be globally unique; same id may
        repeat across scenarios (e.g. ``os_at_12mo`` exists for each scenario)."""
        pairs = []
        for p in _yaml_paths():
            doc = yaml.safe_load(p.read_text())
            pairs.append((doc["scenario"], doc["prediction_target_id"]))
        assert len(pairs) == len(set(pairs)), f"Duplicate (scenario, id) pairs: {pairs}"


class TestObservableExecution:
    @pytest.mark.parametrize("path", _yaml_paths(), ids=_yaml_id)
    def test_compute_observable_returns_array(self, path, synthetic_trajectory):
        t, species = synthetic_trajectory
        doc = yaml.safe_load(path.read_text())
        code = doc["observable"]["code"]
        ns: dict = {}
        exec(code, ns)
        result = ns["compute_observable"](t, species, {})
        assert isinstance(result, np.ndarray)
        assert result.shape == t.shape


class TestEndpointSemantics:
    """Sanity checks on specific endpoints to catch unit / landmark bugs.

    These read from the ``clinical_progression_longterm`` scenario — the
    endpoint function bodies are identical across scenarios (only kwargs
    differ), so one scenario's worth of semantic checks is sufficient.
    """

    SCENARIO = "clinical_progression_longterm"

    def _run(self, species, t, endpoint: str) -> np.ndarray:
        path = PRED_DIR / self.SCENARIO / f"{endpoint}.yaml"
        doc = yaml.safe_load(path.read_text())
        ns: dict = {}
        exec(doc["observable"]["code"], ns)
        return ns["compute_observable"](t, species, {})

    def test_slow_growing_tumor_alive_at_all_os_landmarks(self, synthetic_trajectory):
        """VDT=200d starting at 1e9 cells does not reach 1e11 by day 1095."""
        t, sp = synthetic_trajectory
        for endpoint in ("os_at_12mo", "os_at_24mo", "os_at_36mo"):
            result = self._run(sp, t, endpoint)
            assert result[-1] == 1.0, f"{endpoint} should be alive (1) for slow-growing tumor"

    def test_fast_growing_tumor_dead_by_36mo(self):
        """VDT=30d starting at 1e10 cells reaches 1e11 within a year."""
        t = np.linspace(0, 1095, 100)
        c1 = 1e10 * np.exp(t * math.log(2) / 30)
        v_cm3 = c1 / 1e9
        diameters_cm = 2.0 * np.cbrt(3.0 * v_cm3 / (4.0 * math.pi))
        v_t_mL = (4.0 / 3.0) * math.pi * (diameters_cm / 2.0) ** 3
        sp = {"V_T.C1": c1, "V_T": v_t_mL}
        result_36 = self._run(sp, t, "os_at_36mo")
        assert result_36[-1] == 0.0


class TestCodegenDriftGuard:
    """Re-run the generator and compare to committed YAMLs — flags drift."""

    def test_generator_matches_committed(self, tmp_path, monkeypatch):
        import generate_prediction_yamls as gen

        # Redirect OUT_DIR to a temp path, regenerate, diff each file.
        fresh_dir = tmp_path / "predictions"
        monkeypatch.setattr(gen, "OUT_DIR", fresh_dir)
        rc = gen.main()
        assert rc == 0

        committed = {
            str(p.relative_to(PRED_DIR)): p.read_text() for p in _yaml_paths()
        }
        fresh = {
            str(p.relative_to(fresh_dir)): p.read_text()
            for p in sorted(fresh_dir.rglob("*.yaml"))
        }

        assert set(committed) == set(fresh), (
            f"Filename drift: committed={sorted(committed)} vs fresh={sorted(fresh)}"
        )
        for name in sorted(committed):
            assert committed[name] == fresh[name], (
                f"{name} differs from what the generator would emit. "
                "Run `python scripts/generate_prediction_yamls.py` and commit."
            )
