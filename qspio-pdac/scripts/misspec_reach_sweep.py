#!/usr/bin/env python3
"""Single-knob reach sweep for the misspec audit.

For each outside-CI observable, sweep its top driver around the posterior
median at ±k·σ_prior, run the local ``qsp_sim`` binary, evaluate the
calibration target's observable code, and check whether any sweep point
brings the observable inside the target CI95. Confirms (or refutes) the
``gap_log``-based MECHANISM_MISSING classification with a forward-sim
probe at a cost of ~5 sims × N_outside_targets × 0.2s/sim.

The sweep complements :mod:`misspec_audit`'s static analysis — same
diagnosis with explicit evidence:

- ``sweep_reaches_target = False`` → the top driver alone cannot lift the
  observable into target CI even at ±3σ; structural fix needed.
- ``sweep_reaches_target = True``  → the top driver *can* reach the
  target; the modal posterior is pulled away by competing constraints
  (true degeneracy / partner-tightening might help).

Skips:
- observables whose cal-target YAML defines empty ``index_values``
  (mechanistic peak-ordering targets — can't evaluate at a single time
  point).
- observables without a parseable scenario name in the prefix.
"""
from __future__ import annotations

import math
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent


# Default mapping scenario name → cal-target dirs. Mirrors
# workflows.run_config.DEFAULT_SCENARIOS so the sweep doesn't need a live
# import of that module (avoids importing torch / sbi for a forward-sim
# probe). Update if a new scenario lands in run_config.
DEFAULT_SCENARIO_CAL_DIRS = {
    "baseline_no_treatment": ["calibration_targets/baseline_no_treatment"],
    "clinical_progression": [
        "calibration_targets/clinical_progression",
        "calibration_targets/mechanistic/clinical_progression",
    ],
    "gvax_neoadjuvant_zheng2022": ["calibration_targets/gvax_neoadjuvant"],
    "gvax_nivo_neoadjuvant_zheng2022": [
        "calibration_targets/gvax_nivo_neoadjuvant",
        "calibration_targets/mechanistic/gvax_nivo_neoadjuvant",
    ],
}

# k-sigma sweep points around the posterior median (log space). Five points
# spans ±3σ_prior — enough to detect a monotonic dose-response in either
# direction without over-spending sims.
DEFAULT_K_SIGMA_PTS = (-3.0, -1.5, 0.0, 1.5, 3.0)


@dataclass
class SweepConfig:
    scenarios_dir: Path = ROOT / "scenarios"
    cal_target_root: Path = ROOT / "calibration_targets"
    drug_metadata: Path = ROOT / "resources" / "cpp" / "drug_metadata.yaml"
    healthy_state: Path = ROOT / "resources" / "cpp" / "healthy_state.yaml"
    param_xml: Path = ROOT / "resources" / "cpp" / "param_all.xml"
    submodel_priors_yaml: Path = ROOT / "notes" / "calibration" / "submodel_priors.yaml"
    priors_csv: Path = ROOT / "parameters" / "pdac_priors.csv"
    auxiliary_config: Path = ROOT / "auxiliary_config.yaml"
    qsp_sim_bin: Path = ROOT / "cpp" / "sim" / "build" / "qsp_sim"
    scenario_cal_dirs: dict[str, list[str]] = None  # type: ignore[assignment]
    k_sigma_pts: tuple[float, ...] = DEFAULT_K_SIGMA_PTS

    def __post_init__(self):
        if self.scenario_cal_dirs is None:
            self.scenario_cal_dirs = DEFAULT_SCENARIO_CAL_DIRS


# ---------------------------------------------------------------------------
# Base XML construction (submodel medians + posterior medians overlay)
# ---------------------------------------------------------------------------

def _load_submodel_medians(yaml_path: Path) -> dict[str, float]:
    """Return ``{param: median}``; recover underflowed medians from ``mu``.

    See ``feedback_submodel_priors_median_underflow.md`` — tiny lognormals
    serialize ``median: 0.0`` while ``mu`` retains full precision.
    """
    sm = yaml.safe_load(yaml_path.read_text())
    out: dict[str, float] = {}
    for entry in sm.get("parameters", []):
        name = entry["name"]
        marg = entry["marginal"]
        med = float(marg["median"])
        if med == 0.0 and "mu" in marg:
            med = float(math.exp(marg["mu"]))
        out[name] = med
    return out


def _load_prior_sigmas(csv_path: Path) -> dict[str, float]:
    """Return ``{param: sigma}`` from the priors CSV (lognormal sigma).

    Columns: ``name,median,units,distribution,dist_param1,dist_param2,...``;
    for lognormals ``dist_param2`` is σ (column index 5).
    """
    df = pd.read_csv(csv_path)
    name_col = "name" if "name" in df.columns else df.columns[0]
    sigma_col = df.columns[5]  # dist_param2 = lognormal sigma
    return dict(zip(df[name_col].astype(str), df[sigma_col].astype(float)))


def _load_auxiliary_param_names(yaml_path: Path) -> set[str]:
    """Return the set of auxiliary (nuisance) parameter names from
    ``auxiliary_config.yaml``.

    Auxiliary params don't appear in ``param_all.xml`` — they're applied
    downstream of the simulator to bridge model output to observed data.
    A reach sweep that varies them via XML overlay is a no-op, so callers
    skip these top drivers with an explicit note.

    Schema: ``groups: {group_name: {description, base_prior, members:
    {member_name: {...}}}}``. We discover member names by intersecting
    config keys with QSP posterior columns at call time, so the loader is
    forgiving about per-member schema variation.
    """
    if not yaml_path.exists():
        return set()
    try:
        doc = yaml.safe_load(yaml_path.read_text())
    except Exception:
        return set()
    names: set[str] = set()
    for group in (doc.get("groups") or {}).values():
        members = group.get("members") or {}
        if isinstance(members, dict):
            names.update(members.keys())
        elif isinstance(members, list):
            for m in members:
                if isinstance(m, dict) and m.get("name"):
                    names.add(m["name"])
    # Also include the group base — sometimes auxiliary draws are named
    # after the group key itself when the group is single-member.
    names.update((doc.get("groups") or {}).keys())
    return names


def _overlay_params(xml_text: str, overrides: dict[str, float]) -> str:
    """Replace ``<param>VALUE</param>`` entries; silently skip params absent
    from the XML (the audit can reference params codegen hasn't surfaced)."""
    out = xml_text
    for name, value in overrides.items():
        pattern = re.compile(rf"<{re.escape(name)}>[^<]*</{re.escape(name)}>")
        repl = f"<{name}>{value!r}</{name}>"
        out = pattern.sub(repl, out, count=1)
    return out


def build_base_xml(
    config: SweepConfig,
    posterior_medians: dict[str, float],
) -> str:
    """Compose the sweep base XML: factory defaults ← submodel medians ← posterior medians.

    Posterior medians overlay last so they win where SBI has actually
    contracted the prior; submodel medians fill in the rest.
    """
    xml = config.param_xml.read_text()
    sm = _load_submodel_medians(config.submodel_priors_yaml)
    xml = _overlay_params(xml, sm)
    xml = _overlay_params(xml, posterior_medians)
    return xml


# ---------------------------------------------------------------------------
# Cal target lookup + observable evaluation
# ---------------------------------------------------------------------------

def _parse_scenario_from_observable(obs_name: str) -> Optional[str]:
    """``<scenario>/<target_id>`` → scenario; returns None on bad input."""
    if "/" not in obs_name:
        return None
    return obs_name.split("/", 1)[0]


def find_cal_target_yaml(
    config: SweepConfig, observable: str
) -> Optional[Path]:
    """Locate the cal-target YAML for ``<scenario>/<target_id>``.

    Searches each declared cal-target dir for the scenario (mirrors
    qsp-hpc-tools' union-by-basename loader). Returns the first match.
    """
    scenario = _parse_scenario_from_observable(observable)
    if scenario is None:
        return None
    target_id = observable.split("/", 1)[1]
    for rel in config.scenario_cal_dirs.get(scenario, []):
        for path in (ROOT / rel).glob("*.yaml"):
            try:
                doc = yaml.safe_load(path.read_text())
            except Exception:
                continue
            if doc and doc.get("calibration_target_id") == target_id:
                return path
    return None


def _eval_observable_at_time(
    observable_code: str,
    constants: dict[str, float],
    csv_path: Path,
    target_time_days: float,
) -> float:
    """Run the cal target's ``compute_observable`` on a qsp_sim CSV and
    return the value at (or nearest to) ``target_time_days``."""
    df = pd.read_csv(csv_path)
    time = df["Time"].to_numpy(dtype=np.float64)
    species_dict = {
        col: df[col].to_numpy(dtype=np.float64)
        for col in df.columns
        if col != "Time"
    }
    ns: dict[str, object] = {"np": np, "numpy": np}
    exec(observable_code, ns)
    fn = ns.get("compute_observable")
    if fn is None:
        raise ValueError("observable code did not define compute_observable")
    values = fn(time, species_dict, constants)
    arr = np.asarray(values, dtype=np.float64)
    # Pick the row nearest to target_time
    idx = int(np.argmin(np.abs(time - target_time_days)))
    return float(arr[idx])


# ---------------------------------------------------------------------------
# qsp_sim runner
# ---------------------------------------------------------------------------

def _run_qsp_sim(
    config: SweepConfig,
    scenario_yaml: Path,
    param_xml_path: Path,
    csv_out: Path,
    t_end_days: float,
) -> bool:
    """Run one qsp_sim shot; return True on a clean exit."""
    cmd = [
        str(config.qsp_sim_bin),
        "--param", str(param_xml_path),
        "--scenario", str(scenario_yaml),
        "--drug-metadata", str(config.drug_metadata),
        "--evolve-to-diagnosis", str(config.healthy_state),
        "--csv-out", str(csv_out),
        "--t-end-days", str(t_end_days),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.returncode == 0 and csv_out.exists()


# ---------------------------------------------------------------------------
# Sweep entry
# ---------------------------------------------------------------------------

@dataclass
class SweepResult:
    observable: str
    top_driver: str
    posterior_median: float
    sigma: float
    sweep_values: list[float]
    sweep_obs_values: list[float]  # NaN for failed sims
    target_ci95_lo: float
    target_ci95_hi: float
    sweep_reaches_target: bool
    note: str = ""


def _posterior_medians(posterior_samples: pd.DataFrame, param_names: Iterable[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for p in param_names:
        if p in posterior_samples.columns:
            out[p] = float(np.median(posterior_samples[p].to_numpy()))
    return out


def _xml_param_names(xml_text: str) -> set[str]:
    """Top-level parameter tag names in ``param_all.xml`` (one level deep)."""
    return set(re.findall(r"<([A-Za-z_][A-Za-z0-9_]*)>[^<]*</\1>", xml_text))


def run_reach_sweep(
    diagnosis: pd.DataFrame,
    posterior_samples: pd.DataFrame,
    *,
    config: Optional[SweepConfig] = None,
    only_diagnoses: Iterable[str] = ("MECHANISM_MISSING", "DEGENERACY", "TOPOLOGY_CONFLICT", "NPE_UNDER_TRAINED"),
) -> pd.DataFrame:
    """Per outside-CI observable, sweep top driver and check target reach.

    Args:
        diagnosis: Output of :func:`misspec_audit.diagnose` — must have
            columns ``observable``, ``diagnosis``, ``top_driver``.
        posterior_samples: Posterior θ samples (raw linear units) from the
            run's ``posterior_samples.csv``.
        config: :class:`SweepConfig` overrides; defaults assume the
            current pdac-build layout.
        only_diagnoses: Sweep only these labels. Defaults cover the four
            non-OK outcomes; OK_INSIDE_CI is always skipped.

    Returns:
        DataFrame with one row per swept observable; never raises on
        per-target failure (sets ``note`` and leaves numeric columns NaN).
    """
    config = config or SweepConfig()
    prior_sigmas = _load_prior_sigmas(config.priors_csv)
    posterior_medians = _posterior_medians(posterior_samples, posterior_samples.columns)
    base_xml = build_base_xml(config, posterior_medians)
    # Auxiliary params live outside param_all.xml — varying them via XML
    # overlay is a no-op. Identify by exclusion from the XML param set.
    xml_param_set = _xml_param_names(config.param_xml.read_text())

    rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="reach_sweep_") as tmp_str:
        tmp = Path(tmp_str)
        for _, drow in diagnosis.iterrows():
            obs_name = drow["observable"]
            diag = drow["diagnosis"]
            top_driver = drow["top_driver"]
            if diag not in only_diagnoses or not top_driver:
                continue

            scenario = _parse_scenario_from_observable(obs_name)
            if scenario is None:
                rows.append(_skip_row(obs_name, top_driver, "no scenario prefix"))
                continue

            cal_path = find_cal_target_yaml(config, obs_name)
            if cal_path is None:
                rows.append(_skip_row(obs_name, top_driver, "cal target YAML not found"))
                continue

            try:
                cal = yaml.safe_load(cal_path.read_text())
                code = cal["observable"]["code"]
                constants_list = cal["observable"].get("constants") or []
                constants = {c["name"]: float(c["value"]) for c in constants_list}
                empirical = cal["empirical_data"]
                idx_vals = empirical.get("index_values") or []
                idx_type = empirical.get("index_type") or ""
                if not idx_vals:
                    # Convention per yaml_loader.py: empty index_values =>
                    # evaluate at t=0 (diagnosis snapshot). Matches the
                    # baseline_no_treatment scenario stop_time=1d behavior.
                    target_time = 0.0
                elif idx_type and idx_type != "time":
                    rows.append(_skip_row(
                        obs_name, top_driver,
                        f"non-temporal index_type={idx_type!r} (mechanistic ordering target)",
                    ))
                    continue
                else:
                    target_time = float(idx_vals[0])
                ci95 = empirical["ci95"]
                ci_lo, ci_hi = float(ci95[0][0]), float(ci95[0][1])
            except Exception as exc:
                rows.append(_skip_row(obs_name, top_driver, f"cal YAML parse failed: {exc}"))
                continue

            if top_driver not in xml_param_set:
                rows.append(_skip_row(
                    obs_name, top_driver,
                    "top_driver is auxiliary / not a QSP parameter (no XML overlay)",
                ))
                continue
            if top_driver not in prior_sigmas:
                rows.append(_skip_row(obs_name, top_driver, "top_driver missing prior sigma in CSV"))
                continue
            sigma = prior_sigmas[top_driver]
            post_med = posterior_medians.get(top_driver)
            if post_med is None or post_med <= 0:
                rows.append(_skip_row(obs_name, top_driver, "missing posterior median"))
                continue

            scenario_yaml = config.scenarios_dir / f"{scenario}.yaml"
            if not scenario_yaml.exists():
                rows.append(_skip_row(obs_name, top_driver, f"scenario YAML missing: {scenario_yaml}"))
                continue

            sweep_values: list[float] = []
            sweep_obs_values: list[float] = []
            for k in config.k_sigma_pts:
                val = post_med * math.exp(k * sigma)
                sweep_values.append(val)
                xml = _overlay_params(base_xml, {top_driver: val})
                xml_path = tmp / f"{_safe(obs_name)}_{k:+.1f}.xml"
                csv_path = tmp / f"{_safe(obs_name)}_{k:+.1f}.csv"
                xml_path.write_text(xml)
                ok = _run_qsp_sim(config, scenario_yaml, xml_path, csv_path, target_time + 1.0)
                if not ok:
                    sweep_obs_values.append(float("nan"))
                    continue
                try:
                    obs_val = _eval_observable_at_time(code, constants, csv_path, target_time)
                except Exception:
                    obs_val = float("nan")
                sweep_obs_values.append(obs_val)

            finite_obs = [v for v in sweep_obs_values if np.isfinite(v)]
            reaches = any(ci_lo <= v <= ci_hi for v in finite_obs)
            if not finite_obs:
                note = "all sweep sims failed"
            elif reaches:
                note = "at least one sweep point lands in target CI95"
            else:
                obs_lo, obs_hi = min(finite_obs), max(finite_obs)
                note = (
                    f"sweep range [{obs_lo:.3g}, {obs_hi:.3g}] vs target "
                    f"CI95 [{ci_lo:.3g}, {ci_hi:.3g}]"
                )

            rows.append({
                "observable": obs_name,
                "top_driver": top_driver,
                "posterior_median": post_med,
                "sigma_prior": sigma,
                "k_sigma_pts": list(config.k_sigma_pts),
                "sweep_values": sweep_values,
                "sweep_obs_values": sweep_obs_values,
                "target_ci95_lo": ci_lo,
                "target_ci95_hi": ci_hi,
                "sweep_reaches_target": reaches,
                "note": note,
            })

    return pd.DataFrame(rows)


def _skip_row(obs_name: str, top_driver: str, note: str) -> dict:
    return {
        "observable": obs_name,
        "top_driver": top_driver,
        "posterior_median": float("nan"),
        "sigma_prior": float("nan"),
        "k_sigma_pts": [],
        "sweep_values": [],
        "sweep_obs_values": [],
        "target_ci95_lo": float("nan"),
        "target_ci95_hi": float("nan"),
        "sweep_reaches_target": False,
        "note": note,
    }


def _safe(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", s)
