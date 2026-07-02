# Post-Model-Change Checklist

Whenever you add/rename/remove a parameter, species, reaction, or rule in
`core/modules/` or `parameters/parameters_PDAC.m`, run this checklist before
kicking off Stage-1 (priors) or Stage-2 (joint NPE) calibration.

---

## What has to happen, and why

| # | Artifact | Why it matters | When to update |
|---|---|---|---|
| 1 | `parameters/parameters_PDAC.m` | MATLAB runtime defaults | New param value/units/Notes/derivedFrom |
| 2 | `parameters/pdac_priors.csv` | SBI prior sampling pool | New param OR median/range changed |
| 3 | `model_structure.json` | Unit validation, maple LLM context, PPC simulator XML | Any structural change (species, reactions, rules, params) |
| 4 | `model_defs.json` | LLM context for param definitions + reaction context | Any parameter added, or any reaction rate that references a param changed |
| 5 | `species_units.json` | Pint units in batch test-statistic code | New species |
| 6 | `notes/targets.csv` | maple batch-extraction wishlist (submodel calibration targets) | New param that could plausibly be pinned by a submodel target |
| 7 | `calibration_targets/submodel_targets/submodel_config.yaml` | Hierarchical groups / cascade cuts / inference overrides | New param joins an existing group OR defines a new one |
| 8 | `tests/qualitative_bounds.yaml` | Structural sanity checks | Change could violate or relax an existing bound |
| 9 | `DEVLOG.md` + `notes/model/pdac_changelog.md` | Human-readable narrative / per-module changelog | Every commit that touches the model |
| 10 | SBI simulation pools under `cache/sbi_simulations/` | Cached simulations keyed by config hash | Auto-invalidated — pool hash includes priors CSV + model scripts |
| 11 | C++ simulator XML template (under `delta_analysis/input/`) | HPC PPC runs via CppSimulator | Any new param has to be in the XML template before PPC can run |
| 12 | Stage-2 promotion symlink (`figures/sbi_stage2_current`) | Downstream analyses read this | Re-promote after a fresh Stage-2 run with new params |

---

## Mechanical commands

```bash
# 0. Verify the edit builds and the behavior audit still runs.
cd <worktree>
rm -f tests/scenarios/.sim_outputs.json
.venv/bin/pytest tests/scenarios/test_qualitative_behaviors.py -x -s

# 1. Regenerate model_structure.json (maple CLI).
.venv/bin/qsp-export-model \
    --matlab-model scripts/immune_oncology_model_PDAC.m \
    --output model_structure.json

# 2. Regenerate model_defs.json — same CLI with a different output target, or
#    run the MATLAB-side exporter under maple/src/maple/matlab/.
#    (TODO: confirm; current repo history shows these regenerated together.)

# 3. Validate priors CSV vs parameters_PDAC.m defaults.
.venv/bin/python scripts/parameter_audit.py

# 4. Regenerate species_units.json (if species changed).
#    batch_jobs/input_data/species_units.json is produced by export_model_metadata.m
#    -> run startup.m then `run scripts/export_model_metadata.m` in MATLAB.

# 5. Fast and slow test suites.
.venv/bin/pytest tests/ -m "not slow"
.venv/bin/pytest tests/ -m slow

# 6. Kick off Stage-1 (priors) SBI and Stage-2 (joint NPE).
python workflows/sbi_workflow.py             # per-scenario; honors pool cache
bash workflows/delta/run_delta_analysis.sh   # joint NPE sweep
python scripts/promote_sbi_stage2.py --scan <run_parent>

# 7. After promotion, re-run the audit and the clinical PPC.
python workflows/run_ppc_clinical.py
python scripts/parameter_audit.py
```

---

## Stage-1 → Stage-2 gate

Before running Stage-2 joint NPE on a new model version, confirm:

1. Stage-1 per-scenario SBI trained cleanly (look at the `diagnostics/summary.json` per-run).
2. `scripts/parameter_audit.py` reports all new params are present in the priors CSV and pass the PPC section.
3. `notes/model/post_change_checklist.md` (this file) is itself up to date if you changed the pipeline.

---

## Candidate automations (ordered by ROI)

A single Makefile target or a `scripts/refresh_model_artifacts.py` would
collapse most of this into one command. In order of estimated impact × effort:

1. **`make refresh-model-artifacts`** — chains `export-model`, `parameter_audit`,
   `pytest not slow`. Shortest path to "did I break anything" after a module
   edit. Medium effort: 1 Makefile, 1 wrapper script that reuses existing CLIs.
2. **Pre-commit hook** that diffs `parameters/parameters_PDAC.m` against
   `parameters/pdac_priors.csv` and fails when a param is defined but not in
   priors. Low effort (~20 lines of Python), catches the #1 most-common
   "I forgot to..." error.
3. **`scripts/refresh_structure_jsons.py`** — explicit, idempotent regenerator
   that always exports `model_structure.json` AND `model_defs.json` together,
   writing a provenance stamp with git SHA. Removes the current ambiguity
   where it's unclear which of the two maple entry points produced which file.
4. **Checklist-as-test** in `tests/test_post_change_sync.py`: parse
   `parameters_PDAC.m`, `pdac_priors.csv`, `model_structure.json` and assert
   their parameter sets agree. Runs in CI. Catches drift instantly.
5. **GitHub Action that re-exports structure JSONs on every push touching
   `core/modules/` or `parameters/`** — fully removes the manual step.
   Requires MATLAB on CI (licensing hassle); alternative is a check-only
   action that fails if the JSONs are out of date.

Minimum viable version: (1) + (2) + (4). Items (3) and (5) are optional polish.

---

## Where this checklist itself lives

Keep this note in sync with `CLAUDE.md`'s "Common Development Tasks" section.
If the toolchain shifts (e.g., maple splits `qsp-export-model` into two CLIs),
update BOTH the command block above and the CLAUDE.md pointer.
