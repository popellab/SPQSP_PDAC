"""Phase 4 of OBED integration: one-shot posterior-predictive run over the
promoted stage-2 thetas, for each long-horizon scenario.

Writes ``figures/sbi_stage2_current/posterior_predictive_clinical.csv``
with one row per ``(sample_index, scenario, endpoint)`` — long format for
easy downstream groupby / pivot. The audit's OBED section (Phase 5) reads
this file and summarizes CI95 widths + Spearman correlations.

Usage:
    .venv/bin/python workflows/run_ppc_clinical.py [--n-thetas N]

Config is read from ``figures/sbi_stage2_current/run_config.json`` so the
C++ binary, template XML, healthy-state YAML, and drug-metadata YAML all
track whatever was used for the promoted run. The only overrides we apply
are t_end_days (bumped to 1095 for long-horizon endpoints) and the
scenario mapping (long-horizon scenario YAMLs + reused calibration
target dirs from the short-horizon arms).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qsp_hpc.simulation.cpp_simulator import CppSimulator

REPO_ROOT = Path(__file__).resolve().parent.parent

# Long-horizon scenarios for OBED Phase 4. Calibration targets are reused
# from the short-horizon arms (same biology — only stop_time differs) so
# the simulator has something to derive; prediction targets live alongside.
LONGHORIZON_SCENARIOS = {
    "clinical_progression_longterm": {
        "scenario_yaml": "scenarios/clinical_progression_longterm.yaml",
        "calibration_targets": "calibration_targets/clinical_progression",
        "prediction_targets": "calibration_targets/predictions/clinical_progression_longterm",
    },
    "gvax_neoadjuvant_longterm": {
        "scenario_yaml": "scenarios/gvax_neoadjuvant_longterm.yaml",
        "calibration_targets": "calibration_targets/gvax_neoadjuvant",
        "prediction_targets": "calibration_targets/predictions/gvax_neoadjuvant_longterm",
    },
    "gvax_nivo_neoadjuvant_longterm": {
        "scenario_yaml": "scenarios/gvax_nivo_neoadjuvant_longterm.yaml",
        "calibration_targets": "calibration_targets/gvax_nivo_neoadjuvant",
        "prediction_targets": "calibration_targets/predictions/gvax_nivo_neoadjuvant_longterm",
    },
}

T_END_DAYS = 1095.0  # 3-year horizon for OS/DFS/RECIST landmarks
DT_DAYS = 1.0


def align_theta_to_priors(
    posterior_df: pd.DataFrame, priors_df: pd.DataFrame
) -> tuple[np.ndarray, list[str]]:
    """Return ``(theta, priors_names)`` aligned to the training-time priors CSV.

    The snapshot priors CSV under the promoted run's ``delta_analysis/input/``
    carries the exact 177 parameter names the posterior was trained against —
    the live ``parameters/pdac_priors.csv`` has drifted forward (40 extra rows
    not yet in the C++ XML template, which would fail ``CppBatchRunner.run``
    validation). Using the snapshot keeps the PPC's simulator input space
    identical to what the NPE saw.
    """
    priors_names = priors_df["name"].tolist()
    posterior_cols = set(posterior_df.columns)
    missing = [n for n in priors_names if n not in posterior_cols]
    if missing:
        raise SystemExit(
            f"Posterior is missing {len(missing)} priors-CSV columns "
            f"(first 5: {missing[:5]}) — use the snapshot priors CSV from "
            "figures/sbi_stage2_current/delta_analysis/input/pdac_priors.csv "
            "via --priors-csv so the two agree."
        )
    theta = posterior_df[priors_names].to_numpy(dtype=np.float64)
    print(f"  theta: {theta.shape[0]} rows × {theta.shape[1]} priors-CSV cols")
    return theta, priors_names


def build_simulator(
    scen_name: str,
    scen_cfg: dict,
    run_cfg: dict,
    cache_dir: Path,
) -> CppSimulator:
    # Use the same snapshot priors CSV that the NPE was trained on, so the
    # simulator's param_names align with the theta columns. run_cfg points
    # at the live repo priors CSV which has drifted forward.
    priors_csv = (
        REPO_ROOT
        / "figures/sbi_stage2_current/delta_analysis/input/pdac_priors.csv"
    )
    return CppSimulator(
        priors_csv=priors_csv,
        binary_path=run_cfg["cpp_binary_path"],
        template_xml=run_cfg["cpp_template_xml"],
        subtree=run_cfg.get("cpp_subtree", "QSP"),
        model_version=run_cfg["model_version"],
        scenario=scen_name,
        submodel_priors_yaml=REPO_ROOT / run_cfg["submodel_priors_yaml"],
        scenario_yaml=REPO_ROOT / scen_cfg["scenario_yaml"],
        drug_metadata_yaml=run_cfg["cpp_drug_metadata_yaml"],
        healthy_state_yaml=run_cfg["cpp_healthy_state_yaml"],
        calibration_targets=REPO_ROOT / scen_cfg["calibration_targets"],
        model_structure_file=REPO_ROOT / run_cfg["model_structure_file"],
        t_end_days=T_END_DAYS,
        dt_days=DT_DAYS,
        seed=run_cfg.get("seed", 2025),
        cache_dir=cache_dir,
        verbose=True,
    )


def extract_prediction_columns(table, pred_ids: list[str]) -> pd.DataFrame:
    """Pull just the ``ts:<pred_id>`` columns + sample_index + status out
    of the simulator's output pa.Table. Calibration ``ts:*`` columns are
    intentionally dropped — the audit's OBED section only consumes
    prediction-target endpoints."""
    df = table.to_pandas()
    out = df[["sample_index", "status"]].copy()
    for pid in pred_ids:
        col = f"ts:{pid}"
        if col not in df.columns:
            raise RuntimeError(
                f"Expected prediction column {col!r} missing from simulator output — "
                "check that prediction_targets dir was passed to simulate_with_parameters."
            )
        out[pid] = df[col].to_numpy()
    return out


def list_prediction_ids(pred_dir: Path) -> list[str]:
    ids = []
    for p in sorted(pred_dir.glob("*.yaml")):
        import yaml

        with open(p) as f:
            data = yaml.safe_load(f)
        ids.append(data["prediction_target_id"])
    return ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--posterior",
        type=Path,
        default=REPO_ROOT / "figures/sbi_stage2_current/posterior_samples.csv",
    )
    ap.add_argument(
        "--run-config",
        type=Path,
        default=REPO_ROOT / "figures/sbi_stage2_current/run_config.json",
    )
    # Default to the training-time snapshot under the promoted run — matches
    # the NPE's parameter space exactly. Override with --priors-csv if the
    # stage-2 posterior is replaced by a newer run with a different snapshot.
    ap.add_argument(
        "--priors-csv",
        type=Path,
        default=REPO_ROOT
        / "figures/sbi_stage2_current/delta_analysis/input/pdac_priors.csv",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "figures/sbi_stage2_current/posterior_predictive_clinical.csv",
    )
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / "cache/ppc_clinical",
    )
    ap.add_argument(
        "--n-thetas",
        type=int,
        default=None,
        help="Truncate the posterior to the first N thetas (for smoke tests).",
    )
    ap.add_argument(
        "--scenarios",
        type=str,
        default=",".join(LONGHORIZON_SCENARIOS.keys()),
        help="Comma-separated subset of long-horizon scenarios to run.",
    )
    args = ap.parse_args()

    print("=" * 72)
    print("OBED Phase 4: clinical-endpoint posterior predictive")
    print("=" * 72)

    run_cfg = json.loads(args.run_config.read_text())
    posterior_df = pd.read_csv(args.posterior)
    priors_df = pd.read_csv(args.priors_csv)

    if args.n_thetas is not None:
        posterior_df = posterior_df.head(args.n_thetas)

    theta, priors_names = align_theta_to_priors(posterior_df, priors_df)
    print(f"  theta shape: {theta.shape}")

    scenarios_to_run = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    print(f"  scenarios: {scenarios_to_run}")

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for scen_name in scenarios_to_run:
        if scen_name not in LONGHORIZON_SCENARIOS:
            raise SystemExit(f"Unknown scenario: {scen_name!r}")
        scen_cfg = LONGHORIZON_SCENARIOS[scen_name]
        print()
        print(f"--- {scen_name} ---")

        pred_dir = REPO_ROOT / scen_cfg["prediction_targets"]
        pred_ids = list_prediction_ids(pred_dir)
        print(f"  prediction targets: {len(pred_ids)} — {pred_ids}")

        sim = build_simulator(scen_name, scen_cfg, run_cfg, args.cache_dir)
        _, table = sim.simulate_with_parameters(
            theta,
            prediction_targets=pred_dir,
        )

        preds = extract_prediction_columns(table, pred_ids)
        preds["scenario"] = scen_name

        # Long-form: one row per (sample_index, scenario, endpoint).
        long = preds.melt(
            id_vars=["sample_index", "status", "scenario"],
            value_vars=pred_ids,
            var_name="endpoint",
            value_name="value",
        )
        frames.append(long)
        n_ok = int((preds["status"] == 0).sum())
        print(f"  completed: {n_ok}/{len(preds)} thetas simulated successfully")

    out = pd.concat(frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print()
    print("=" * 72)
    print(f"wrote {len(out):,} rows → {args.output}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
