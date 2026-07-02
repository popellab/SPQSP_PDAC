"""End-to-end pipeline: per-sim burn-in trajectory binaries → posterior
predictive cal-target trajectory plot.

Glues together:
  - qsp_hpc.cpp.evolve_trajectory.assemble_evolve_trajectory_long
  - qsp_inference.inference.trajectory_eval.evaluate_calibration_target_over_trajectory
  - workflows.plot_calibration_trajectories

Intended to be called from sbi_runner step 7 (PPC) once trajectory
binaries have been dumped to disk by ``CppSimulator.simulate_with_parameters(
evolve_trajectory_dir=...)``. Standalone: pass already-loaded calibration
target objects, the trajectory directory, and the model schema bits.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

from matplotlib.figure import Figure


def run_calibration_trajectory_pipeline(
    *,
    calibration_targets: Sequence[Any],
    output_path: Optional[str | Path] = None,
    title: Optional[str] = None,
    columns_needed: Optional[Sequence[str]] = None,
    sample_indices: Optional[Sequence[int]] = None,
    n_cols: int = 3,
    log_y: bool | str = False,
    time_axis: str = "absolute",
    phase: str = "burn_in",
    scenario_name: Optional[str] = None,
    scenario_per_target: Optional[Mapping[str, str]] = None,
    # Burn-in inputs (phase="burn_in"):
    traj_dir: Optional[str | Path] = None,
    species_names: Optional[list[str]] = None,
    compartment_names: Optional[list[str]] = None,
    rule_names: Optional[list[str]] = None,
    # Post-scenario inputs (phase="post_scenario"):
    species_parquet: Optional[str | Path] = None,
    # Per-sim records (in ascending sample_index order) for observables
    # that depend on varied prior parameters declared under
    # observable.species (e.g. rho_collagen) or nuisance auxiliaries
    # declared under observable.auxiliary_parameters (e.g.
    # f_serum_to_tumor_X). Without these, such observables KeyError
    # inside the evaluator → on_error='nan' wipes the panel.
    parameter_records: Optional[Sequence[Mapping[str, float]]] = None,
    auxiliary_records: Optional[Sequence[Mapping[str, float]]] = None,
) -> Figure:
    """Assemble trajectories → evaluate cal targets → plot. Returns the Figure.

    Args:
        phase: ``"burn_in"`` (default) reads per-sim binary trajectories
            dumped during ``evolve_to_diagnosis`` (requires ``traj_dir``
            + ``species_names`` / ``compartment_names`` / ``rule_names``).
            ``"post_scenario"`` pivots a CppBatchRunner species parquet
            (requires ``species_parquet``) — use for treatment / clinical
            scenarios where observations happen at specific
            post-diagnosis times.
        traj_dir: Burn-in only. Directory of ``sim_<sample_index>.bin``.
        species_names / compartment_names / rule_names: Burn-in only.
            Trajectory column-order metadata read off the species_parquet
            schema (CppBatchRunner doesn't dump column names alongside
            the binaries).
        species_parquet: Post-scenario only. Path to the
            ``species_*.parquet`` written by CppBatchRunner.
        calibration_targets: Sequence of cal-target objects (must expose
            ``calibration_target_id``, ``observable.code``,
            ``observable.constants``, ``empirical_data.{median,
            ci95_lower, ci95_upper}``).
        output_path: Optional PNG to write the figure to. Parent dirs
            are created.
        title: Optional figure suptitle (e.g. the scenario name).
        columns_needed: Optional explicit list of columns to retain.
            ``None`` keeps all non-meta columns.
        sample_indices: Optional iterable of sample-index ints to keep.
        n_cols / log_y / time_axis: Forwarded to
            :func:`plot_calibration_trajectories`. Note: post-scenario
            phase silently forces ``time_axis="absolute"`` since
            fraction normalization assumes per-sim variable durations,
            which post-scenario sims don't have.

    Returns:
        The matplotlib ``Figure``. ``output_path`` is a save convenience.
    """
    from qsp_hpc.cpp.evolve_trajectory import (
        assemble_evolve_trajectory_wide_per_sim,
        assemble_post_scenario_trajectory_wide_per_sim,
    )
    from qsp_inference.inference.trajectory_eval import (
        evaluate_calibration_target_over_trajectory,
    )

    from workflows.plot_calibration_trajectories import plot_calibration_trajectories

    if phase not in ("burn_in", "post_scenario", "auto"):
        raise ValueError(
            f"phase must be 'burn_in', 'post_scenario', or 'auto'; got {phase!r}"
        )

    # 1. Assemble trajectories directly into wide-per-sim frames,
    # dispatching on phase. Skips the wide→long→wide round trip the
    # earlier long-form pipeline paid for.
    if phase == "burn_in":
        if traj_dir is None or species_names is None:
            raise ValueError(
                "phase='burn_in' requires traj_dir and species/comp/rule names"
            )
        wide_per_sim = assemble_evolve_trajectory_wide_per_sim(
            traj_dir,
            species_names=list(species_names),
            compartment_names=list(compartment_names) if compartment_names else [],
            rule_names=list(rule_names) if rule_names else [],
            columns=list(columns_needed) if columns_needed is not None else None,
            sample_indices=list(sample_indices) if sample_indices is not None else None,
        )
        if not wide_per_sim:
            raise ValueError(
                f"no per-sim trajectory binaries found in {traj_dir} — "
                "did the upstream PPC step set evolve_trajectory_dir?"
            )
    else:  # phase == "post_scenario"
        if species_parquet is None:
            raise ValueError(
                "phase='post_scenario' requires species_parquet (the "
                "CppBatchRunner output containing list-typed columns)"
            )
        wide_per_sim = assemble_post_scenario_trajectory_wide_per_sim(
            species_parquet,
            columns=list(columns_needed) if columns_needed is not None else None,
            sample_indices=list(sample_indices) if sample_indices is not None else None,
        )
        if not wide_per_sim:
            raise ValueError(
                f"no rows in {species_parquet} — did the upstream PPC step run?"
            )
        # Post-scenario sims share a fixed time grid → fraction
        # normalization is meaningless here; force absolute.
        time_axis = "absolute"

    obs_frames: list[pd.DataFrame] = []
    for target in calibration_targets:
        obs_id = (
            getattr(target, "calibration_target_id", None)
            or getattr(target, "observable_id", None)
            or getattr(target, "id", None)
        )
        if obs_id is None:
            raise ValueError(
                f"calibration target has no id field "
                "(checked calibration_target_id, observable_id, id)"
            )
        eval_df = evaluate_calibration_target_over_trajectory(
            target=target,
            wide_per_sim=wide_per_sim,
            parameter_records=parameter_records,
            auxiliary_records=auxiliary_records,
        )
        eval_df = eval_df.assign(observable_id=obs_id)
        obs_frames.append(eval_df)
    observable_df = pd.concat(obs_frames, ignore_index=True)

    # 3. Plot. If a scenario_name is given, broadcast it to every cal
    # target so panel titles show the source scenario.
    if scenario_per_target is None and scenario_name is not None:
        scenario_per_target = {
            (
                getattr(t, "calibration_target_id", None)
                or getattr(t, "observable_id", None)
                or getattr(t, "id", None)
            ): scenario_name
            for t in calibration_targets
        }
    fig = plot_calibration_trajectories(
        observable_df=observable_df,
        calibration_targets=list(calibration_targets),
        output_path=output_path,
        n_cols=n_cols,
        log_y=log_y,
        time_axis=time_axis,
        phase=phase,
        scenario_per_target=scenario_per_target,
        title=title,
    )
    return fig


def _dict_to_namespace(obj: Any) -> Any:
    """Recursively map ``dict`` → ``SimpleNamespace`` and ``list`` of dicts →
    list of namespaces, preserving leaf scalars. Yields duck-typed attribute
    access (``x.observable.code``, ``x.observable.constants[0].name``) without
    paying maple's pydantic validation cost (~17s cold, ~3s warm per call).
    """
    from types import SimpleNamespace

    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_dict_to_namespace(v) for v in obj]
    return obj


def load_cal_targets_from_dirs(
    cal_dirs: Sequence[str | Path],
    *,
    model_structure: Any = None,
) -> list[Any]:
    """Walk one or more directories of CalibrationTarget YAMLs → list of
    duck-typed namespace objects (``SimpleNamespace`` trees).

    Skips maple's pydantic ``CalibrationTarget.model_validate`` entirely —
    validation is appropriate at extraction/commit time, not on every
    trajectory-plot regen. Downstream consumers (the trajectory evaluator
    and the cal-traj plotter) only need attribute access to a small,
    known surface (``observable.code/constants/species/auxiliary_parameters``,
    ``empirical_data.median/ci95_*/index_values/units``,
    ``calibration_target_id``), all of which a recursive
    dict→SimpleNamespace map provides.

    Args:
        cal_dirs: Directories of ``*.yaml`` cal target files.
        model_structure: Accepted for backwards compatibility; ignored
            since validation no longer runs here.
    """
    import yaml as _yaml

    del model_structure  # validation step removed
    targets: list = []
    for d in cal_dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        for yml in sorted(d.glob("*.yaml")):
            with open(yml) as fh:
                data = _yaml.safe_load(fh)
            targets.append(_dict_to_namespace(data))
    return targets


def collect_required_trajectory_columns(
    calibration_targets: Sequence[Any],
) -> list[str]:
    """Union of ``observable.species`` across all cal targets — the
    minimal set of trajectory columns the cal-target observables need.

    Pass this as ``columns_needed`` to :func:`run_calibration_trajectory_pipeline`
    when the trajectory binaries dump every species (default) but the
    cal targets only reference a few.
    """
    cols: set[str] = set()
    for target in calibration_targets:
        obs = getattr(target, "observable", None)
        if obs is None:
            continue
        for col in getattr(obs, "species", []) or []:
            cols.add(col)
    return sorted(cols)


__all__ = [
    "collect_required_trajectory_columns",
    "load_cal_targets_from_dirs",
    "run_calibration_trajectory_pipeline",
]
