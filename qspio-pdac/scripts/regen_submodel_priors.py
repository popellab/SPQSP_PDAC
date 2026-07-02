#!/usr/bin/env python3
"""Re-run submodel inference + regenerate `notes/calibration/submodel_priors.yaml`.

Canonical fast path for the iterative debugging loop. Wraps the same
component-inference machinery `scripts/parameter_audit.py` uses, but skips the
audit's PPC simulations and Markdown report — those are the slow parts and
the debugging loop usually doesn't need them.

What this DOES:
    1. Re-runs MCMC for components containing any `--invalidate`d parameter
       (others reuse their `.compare_cache/` entries). When the cache is
       empty (first run, or after `rm -rf .compare_cache/`), `--rebuild-all`
       forces inference on every component.
    2. Re-aggregates the joint posterior cache and rewrites
       `notes/calibration/submodel_priors.yaml` (the canonical path read by
       `workflows/sbi_runner.py`).

What this DOES NOT do:
    - Posterior predictive checks
    - Markdown report (`notes/calibration/parameter_audit_report.md`)
    - SBI Stage-2 cross-checks

For the full audit (PPC + report + plots), use `scripts/parameter_audit.py`.

Usage:
    .venv/bin/python scripts/regen_submodel_priors.py
        Re-aggregate existing cache. Errors out if the cache is empty.

    .venv/bin/python scripts/regen_submodel_priors.py --invalidate APC0_cDC2_T
        Re-run MCMC for the component(s) containing APC0_cDC2_T, then
        re-aggregate. Other components reuse cached posteriors.

    .venv/bin/python scripts/regen_submodel_priors.py --invalidate q_CD8_T_in k_Mac_rec
        Multi-param invalidation.

    .venv/bin/python scripts/regen_submodel_priors.py --rebuild-all
        Run inference for every component (slow, ~minutes-to-hours
        depending on component count). Use after `rm -rf .compare_cache/`.

    .venv/bin/python scripts/regen_submodel_priors.py --output some/other/path.yaml
"""

import argparse
from pathlib import Path

from qsp_inference.audit.report import (
    AuditConfig,
    _run_inference,
    _write_submodel_priors,
    load_freshness_by_component,
    load_joint_samples,
    load_joint_samples_by_component,
    load_parameter_groups,
    load_submodel_targets,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-run submodel inference + regenerate submodel_priors.yaml."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root (default: cwd).",
    )
    parser.add_argument(
        "--invalidate",
        nargs="+",
        metavar="PARAM",
        default=None,
        help=(
            "Re-run MCMC for components containing these parameters. "
            "Other components reuse their cached posteriors. Pass nothing "
            "to skip re-inference and just re-aggregate the existing cache."
        ),
    )
    parser.add_argument(
        "--rebuild-all",
        action="store_true",
        help=(
            "Force a full inference pass: every component runs MCMC, "
            "ignoring any cached posteriors. Equivalent to `rm -rf "
            ".compare_cache/` followed by re-running inference. Use this "
            "when you've cleared the cache or want a clean slate. Slow."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output path (default: <project-root>/notes/calibration/submodel_priors.yaml — "
            "the canonical path read by workflows/sbi_runner.py)."
        ),
    )
    args = parser.parse_args()

    config = AuditConfig(project_root=args.project_root)
    output = args.output or args.project_root / "notes" / "calibration" / "submodel_priors.yaml"

    cache_empty = not config.compare_cache.exists() or not any(
        config.compare_cache.glob("comp_*.json")
    )

    if args.rebuild_all:
        # Wipe the cache and re-run inference for every component.
        if config.compare_cache.exists():
            import shutil

            shutil.rmtree(config.compare_cache)
            print(f"Wiped cache: {config.compare_cache}")
        _run_inference(config, invalidate_params=None)
    elif args.invalidate:
        # Re-run MCMC for components containing any of these params.
        # Cached components are skipped; this is the same machinery
        # `scripts/parameter_audit.py` uses, just without PPC + report.
        _run_inference(config, invalidate_params=args.invalidate)
    elif cache_empty:
        # No --invalidate, no --rebuild-all, and the cache is empty —
        # run inference once to populate it before aggregation.
        print(
            f"Cache at {config.compare_cache} is empty; running full "
            "inference to populate it (equivalent to --rebuild-all)."
        )
        _run_inference(config, invalidate_params=None)

    joint_samples = load_joint_samples(config.compare_cache)
    if not joint_samples:
        print(f"No cached joint samples in {config.compare_cache}.")
        print("Inference appears to have failed; check upstream logs.")
        return 1
    # qsp-inference PR #23: _write_submodel_priors now expects samples
    # grouped by component_id (dict[component_id, dict[param, samples]])
    # so the copula is built block-diagonally per inference component.
    joint_samples_by_component = load_joint_samples_by_component(config.compare_cache)
    if not joint_samples_by_component:
        print(f"No per-component samples in {config.compare_cache}.")
        return 1

    targets = load_submodel_targets(config.submodel_dir)
    groups = load_parameter_groups(config.param_groups)
    freshness_by_component = load_freshness_by_component(config.compare_cache)

    print(f"Cache: {config.compare_cache}")
    print(f"Loaded joint samples for {len(joint_samples)} parameters "
          f"across {len(joint_samples_by_component)} components.")
    if freshness_by_component:
        print(
            f"Stamping content fingerprints for "
            f"{len(freshness_by_component)} components into metadata.freshness."
        )
    else:
        print(
            "WARNING: no freshness manifests found in cache "
            "(comp_*.json written before fingerprinting was added). "
            "Re-run with --rebuild-all to add them."
        )
    print(f"Writing -> {output}")
    _write_submodel_priors(
        joint_samples_by_component,
        targets,
        groups,
        output,
        freshness_by_component=freshness_by_component,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
