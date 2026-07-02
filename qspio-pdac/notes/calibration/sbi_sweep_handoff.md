# SBI sweep handoff (2026-04-14)

Everything learned during the 20k-sim sweep debugging session. Read top-to-bottom
if picking up cold.

## TL;DR

- 20k simulation pool exists on Rockfish for all 4 scenarios (baseline_no_treatment,
  clinical_progression, gvax_neoadjuvant, gvax_nivo_neoadjuvant).
- 36-cell sweep completed (3 seeds × 3 flow sizes × 4 `n_train_subsample` values).
  Results at `sweep_runs/seed_flow_ntrain_20k/sweep_results_{long,summary}.csv` on Rockfish.
- **Main finding**: training data is the bottleneck, not flow capacity. 81% of
  sims drop in the joint-NaN filter. Highest-leverage next step: audit which
  test-stat observables are producing NaN and fix them.

## How the sweep system works

### Key files

- `workflows/sbi_runner.py` — `RunConfig` dataclass + `run_inference(cfg)` callable.
  The sole pipeline entry point.
- `workflows/sweep_submit.py` — reads a YAML grid spec, generates per-cell
  config JSONs + a single `sweep.sbatch` array job, submits.
- `workflows/sweep_run_one.py` — single-cell entrypoint; loads JSON → `RunConfig`
  → `run_inference`.
- `workflows/sweep_collect.py` — walks finished sweep dir, emits two CSVs
  (`sweep_results_long.csv` = one row per (cell, parameter); `sweep_results_summary.csv`
  = one row per cell).
- `sweep_specs/*.yaml` — declarative grid specs. Two exist:
  - `seed_flow_nsims.yaml` — original 3×3×3 at 2k/3.5k/5k sim budgets
  - `seed_flow_ntrain_20k.yaml` — 3×3×4 at 20k pool with n_train_subsample axis

### Run a sweep

From local laptop (spec + submit happen on the cluster):

```bash
# Local: make sure branch is pushed
git push

# Rockfish: pull + submit
ssh hpc "cd ~/pdac-build-git && git pull && \
  .venv/bin/python workflows/sweep_submit.py sweep_specs/<spec>.yaml"
```

`sweep_submit.py`:
- Expands the Cartesian product of axes into one config JSON per cell
- Writes one `sweep.sbatch` with `#SBATCH --array=0-(N-1)`
- Before submitting, each array task runs `uv pip install --upgrade qsp-hpc-tools@main qsp-inference@main` so cells pick up whatever's on main
- Submits with `sbatch`

Use `--overwrite` if the output dir exists, `--dry-run` to print cells without
writing, `--max-concurrent K` to throttle (rarely needed).

### Collect results

```bash
ssh hpc "cd ~/pdac-build-git && \
  .venv/bin/python workflows/sweep_collect.py sweep_runs/<sweep_name>"
# Then rsync CSVs back to local if you want to inspect
```

## Architecture decisions (and why)

### Parfor in `batch_worker.m`

`qsp-hpc-tools` ships `batch_worker.m` that runs each array task's patient loop.
We added `parfor` support:

- PR #6: main patient loop now `parfor (i, num_workers)` when `MATLAB_WORKERS>0`
  env var is set
- PR #7: same treatment for `extract_all_species_arrays.m` (was the bottleneck
  after #6 made the sim loop fast)
- PR #8: explicit `delete(gcp('nocreate'))` before batch_worker exits, so MATLAB
  doesn't hang on SLURM wall timeout waiting for parpool workers to drain
- PR #9: **auto-sizing** — `BatchConfig.max_cpus_per_account` caps `max_tasks` at
  `cap/cpus_per_task` so all tasks fit one wave. Also: when `matlab_workers>0`,
  `cpus_per_task` auto-derives to `W+2` and memory to `W*5+30 GB` — single knob.

Single-knob credentials on Rockfish (`~/.config/qsp-hpc/credentials.yaml` or
`$PROJECT/.qsp-hpc/credentials.yaml`):

```yaml
slurm:
  partition: shared
  time_limit: 02:00:00
  matlab_workers: 20           # everything derives from this
  max_cpus_per_account: 308    # Rockfish `shared` cap
```

### QSPResultLoader split (PR #11)

**The biggest foot-gun we hit**: `QSPSimulator` was designed for laptop→HPC flow
and always runs combine-on-HPC + SCP-to-local to read existing pool results.
When 9 sweep cells ran **on Rockfish** concurrently, they all truncated +
rewrote `combined_*.csv` at the same time → race → cells read partial files
(8136 rows instead of 20000).

Fix was to split concerns:
- `QSPSimulator` — orchestration (SSH, SCP, SLURM submit, remote combine)
- `QSPResultLoader` (new) — filesystem-only: given pool_root + same hashing
  inputs, reads `combined_params.csv` + `combined_test_stats.csv` directly.
  Raises `FileNotFoundError` or `ValueError` when files are missing/short.

`sbi_runner.py` tries `QSPResultLoader` first; falls back to `QSPSimulator`
only if loader can't find valid files. **Always prefer `QSPResultLoader` for
read-only loading from now on.** Don't reach for `QSPSimulator` when results
already exist — that's what caused the race.

### Automatic dep upgrade

Each sweep cell runs `uv pip install --upgrade qsp-hpc-tools@main qsp-inference@main`
in the sbatch script **before** Python starts (see `ARRAY_SBATCH_TEMPLATE` in
`sweep_submit.py`). This ensures cells always run merged-to-main code without
manual venv babysitting. Set `SKIP_SWEEP_UPGRADE=1` env var to disable.

Important: don't create editable installs on HPC. `ensure_hpc_venv()` inside
qsp-hpc-tools *also* auto-upgrades from `qsp_hpc_tools_source` on every
workflow submission — dueling installs used to clobber the editable links and
cause stale-code bugs. Let git@main be the source of truth on HPC.

## Gotchas we've already hit

### Test stats get wiped

`hpc_job_manager.py:922` fires `rm -rf test_stats/{hash}` when `n_derived !=
expected_n_sims`. If a sweep runs against a previously-populated pool where
chunks have been combined-and-cleaned, the chunk count is 0, the check
misfires, and the combined file gets nuked. Already-happened to
baseline_no_treatment and clinical_progression during our runs; we had to
re-derive by deleting local cache and rerunning `sbi_workflow.py`.

Open bug to fix (not done): `check_hpc_test_stats` should treat a valid
`combined_test_stats.csv` with matching row count as sufficient regardless of
chunk count. Didn't do it this session because we ran out of time.

### Staleness / hashes drift between laptop and Rockfish

Config hash depends on: priors CSV content + test stats (derived from
calibration target YAMLs) + model script + model version + scenario. If any
of those differ between laptop and Rockfish, hashes don't match and the
workflow thinks the pool is empty. **Always git pull on Rockfish before
submitting.** We now git clone on HPC rather than rsync (history: reference
`feedback_no_editable_on_hpc.md` memory).

### `EmptyDataError` on rare cells

One sweep cell failed with `pandas.errors.EmptyDataError: No columns to
parse`. Transient I/O flake on the shared filesystem. Didn't debug; just
retry the cell. If this becomes common, wrap the loader's `pd.read_csv` in
retry logic.

### sbi_runner uses Python 3.12+ f-string backslash

`workflows/sbi_runner.py` has f-strings like `f"{x!s}\t..."` that are invalid
in Python 3.11 (PEP 701 relaxed this in 3.12). The Rockfish venv at
`~/pdac-build-git/.venv` uses Python 3.12. Don't change this venv to 3.11.

## Performance numbers (current)

### Sim generation (20k sims × 4 scenarios, full pool)

- Pre-parfor: ~50 min per scenario (single-threaded MATLAB in 35 array tasks)
- Post all parfor PRs (#6–#9): **~5 min per scenario** (20 workers each,
  14-task wave saturating the 308-cpu account cap)
- Fixed overhead floor per task: ~90 s (parpool startup 40s + model accel 40s
  + parquet save 10s). Can't go below without deeper changes (see
  SimBiology deployment / `ExportedModel` ideas below).

### NPE training

Per sweep cell at 20k sims (≈3382 valid train rows after filter):
- n_train=1000: ~2–3 min, 25 epochs
- n_train=2000: ~4–6 min, 26 epochs
- n_train=full (3382): ~5–10 min (bigger flows longer), 30–49 epochs

### Sweep wall clock

36 cells × ~5 cpus × 4 cpus each = 144 cpus, single wave → **~15 min wall**.

## Key findings from the latest sweep

**See `sweep_runs/seed_flow_ntrain_20k/sweep_results_summary.csv` on Rockfish.**

1. Training data is the bottleneck. At every flow size, val loss drops ~25
   units going from 1000 → 3382 train rows. Still on the steep part of the curve.
2. Flow capacity barely matters. 128×7 vs 256×11 at any fixed n_train gives
   ~1–3 val loss units difference.
3. Shell contraction ≥50% shows up only at full data with 192×9 or 256×11
   (2 parameters identified per seed). Below full data: 0 identified params.
4. 19% retention from joint-NaN filter is the real ceiling. Each scenario
   produces ~30% NaN rate per sim; 4-scenario intersection → 0.7⁴ ≈ 24%.

## Next steps (in priority order)

### 1. Audit per-observable NaN rates (biggest leverage)

Add to `sbi_runner.py` after the raw sim load, before the joint filter:

```python
for scen_name in scenario_names:
    x = scenario_meta[scen_name]['x_raw']
    nan_by_col = np.isnan(x).mean(axis=0)
    for i, obs_name in enumerate(scenario_meta[scen_name]['observable_names']):
        if nan_by_col[i] > 0.1:
            print(f"  [{scen_name}] {obs_name}: {nan_by_col[i]:.1%} NaN")
```

Likely culprits: ratio observables where denominator can be 0 (e.g., CD8/Treg),
log-space observables where argument can be 0 or negative. Fix these in the
calibration target YAMLs' `code:` field (clip, epsilon, or return NaN only
when biologically meaningless).

If we lift retention 19% → 40%, we get ~8000 train rows → val loss likely
below 220, more params identifiable.

### 2. Fix the `rm -rf test_stats_dir` gotcha

`hpc_job_manager.check_hpc_test_stats` at qsp-hpc-tools. Add a check: if
`combined_test_stats.csv` exists with ≥ expected_n_sims rows, short-circuit
return True without touching chunks or the dir. Small patch; prevents future
sweeps from nuking test stats they're about to use.

### 3. Explore `ExportedModel` / SimBiology Compiler

`sbioaccelerate` takes 40s per task — 44% of the fixed overhead floor.
SimBiology's `ExportedModel` + `accelerate()` + save/load lets you pre-
compile once and load the compiled state on subsequent runs. Could drop
fixed-overhead floor from ~90s to ~50s. See
https://www.mathworks.com/help/simbio/deployment.html and our test at
`scripts/test_exported_model.m` (incomplete, work-in-progress; call signature
for `simFn = createSimFunction(model, param_names, [], {})` was being
debugged).

### 4. Consider more sims only after (1)

Generating 10k more sims is ~5 min of MATLAB time. But if retention stays at
19%, that's only ~1900 extra train rows. (1) is higher leverage per unit time.

## Tools to keep handy

### Check Rockfish queue

```bash
ssh hpc "squeue -u \$USER --format='%.12i %.12j %.8T %.10M' -h"
```

### Look at a cell's log

```bash
ssh hpc "tail -30 ~/pdac-build-git/sweep_runs/<sweep>/logs/<jobid>_<taskid>.out"
ssh hpc "tail -15 ~/pdac-build-git/sweep_runs/<sweep>/logs/<jobid>_<taskid>.err"
```

### Inspect pool state

```bash
ssh hpc "ls /home/jeliaso2/data-apopel1/jeliaso2/qsp_simulations/ | grep <scenario>"
# test stats availability:
ssh hpc "for s in baseline_no_treatment clinical_progression gvax_neoadjuvant_zheng2022 gvax_nivo_neoadjuvant_zheng2022; do \
  d=/home/jeliaso2/data-apopel1/jeliaso2/qsp_simulations/multi_scenario_v4_all_scenarios_54b90784_\$s/test_stats; \
  echo '==='\$s'==='; ls \$d 2>/dev/null; done"
```

### Kill a stuck sweep

```bash
ssh hpc "scancel <jobid>"
```

### Kick off `sbi_workflow.py` locally (when you need HPC-side test stats
re-derived)

```bash
rm cache/sbi_simulations/multi_scenario_v4_all_scenarios_*_<scenario>/*.mat
TS=$(date +%Y%m%d_%H%M)
nohup .venv/bin/python workflows/sbi_workflow.py --num-total-simulations 20000 \
    > logs/sbi_workflow_${TS}.log 2>&1 &
disown
# Tail progress
tail -f logs/sbi_workflow_${TS}.log
```

## Repo state

Branch: `fix/sbi-workflow-cleanup` (all work here, not merged to main yet).
qsp-hpc-tools: main (all parfor + loader PRs merged).
qsp-inference: main (Stage-2 audit + copula enhancements merged).

Open PRs: none.
