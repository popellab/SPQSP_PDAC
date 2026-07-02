# Local observable evaluation: Option (b) end-to-end

**Status:** Approved; rollout in progress (qsp-inference PR #36 merged 2026-05-07).
**Context:** Handoff `/tmp/handoff_20k-sbi-with-aux-cal-targets_2026-05-06.md`. The
runner gate at `workflows/sbi_runner.py:540` raises `NotImplementedError` because
auxiliary-using cal targets need per-sim aux draws to influence `x_train`, and
today's HPC worker has no place to accept them. This doc proposes the
architectural change that unblocks aux *and* collapses caching complexity as a
side effect.

## Motivation

Today's split:

- **HPC worker** (`derive_test_stats_worker`) reads `qsp_sim` output → loads
  `test_stats.csv` → calls each row's `compute_test_statistic(time, species,
  ureg)` → writes scalar parquet.
- **Runner** (`CppSimulator.run_hpc`) downloads scalar parquet, returns
  `(theta, x)`.
- **PPC step 7** uses `simulate_with_parameters` (separate suffix pool, same
  worker shape).
- **Trajectory plotter step 7f** reads species parquets locally and re-runs
  observable code via `evaluate_calibration_target_over_trajectory`.

Three caches (config-hash pool, test-stats sub-cache, PPC suffix pool), three
eval paths for the same operation. Aux draws can't enter the worker because
the worker bakes constants into wrappers at extraction time.

Option (b): move all observable evaluation onto the runner. Worker dumps
trajectories only. PPC and training and plotting share one eval path. Three
caches collapse to one. Aux is a kwarg on the eval helper.

## Non-goals

- Drop the trajectory cache entirely. Topping up 5k → 20k, τ_cohort sweeps,
  PRCC-tier sweeps, NPE arch sweeps, and posterior diagnostics all need it.
  Keeping a single trajectory pool per (binary, scenario) with internal
  training/PPC sub-pools (D1) is enough.
- Refactor submodel-priors fingerprinting. Different complexity tax,
  orthogonal to sim caching.
- Change the SBI training surface (`theta_train.pt`, `x_train.pt`). Only the
  *production* of `x_train` changes; downstream Stage-2/Delta consumers are
  unaffected.

## Layer 1: HPC worker (`qsp-hpc-tools`)

Today's worker (`derive_test_stats_worker`) earns its existence by doing two
things: (i) call `qsp_sim`, (ii) load `test_stats.csv` and evaluate each row's
wrapper to scalars. Under (b), (ii) is gone — observable eval moves to the
runner. What's left is "call `qsp_sim`, write something the runner can read."

`qsp_sim` already writes binary trajectories via `--binary-out` / `--csv-out`
(see runbook §3), and `qsp_hpc.cpp.runner.read_binary_trajectory` already
consumes them at `sbi_runner.py:2330`. So the worker shrinks to a
**thin trajectory → parquet converter, no eval logic.** Under D8 it dumps
all species; column projection happens at read time over `sshfs`.

```python
def dump_trajectory_worker(
    param_xml: Path,
    scenario_yaml: Path,
    drug_metadata: Path,
    healthy_state: Path,
    *,
    out_path: Path,                  # one parquet per sim chunk
    min_cadence_hours: float = 4.0,  # solver-native + cadence floor (D4)
    dump_burn_in: bool = False,      # if True, also dump pre-diagnosis
    burn_in_min_cadence_days: float = 14.0,
) -> Path
```

Per D8 the worker dumps *all* species; column projection happens at
read time. No `columns` kwarg.

`min_cadence_hours` defaults to 4 h (D4); scenario YAML may override
via `traj_min_cadence_hours`. qsp_sim runs CVODES in `CV_ONE_STEP`
mode and writes one row per internal step subject to the cadence
floor — coarse during steady-state, fine through dose transients,
capped at 6 dumps/day in the worst case. Read-time linear
interpolation lifts samples to cal-target `index_values`.

Output schema (long-form, two parquets per chunk; per Open Q5):

```
chunk_NNNN.trajectory.parquet
    sample_index | simulation_id | status | t_to_diagnosis_days |
    time | species (categorical) | value

chunk_NNNN.params.parquet
    sample_index | param:<name>_1 | param:<name>_2 | …
```

Splitting params from trajectory rows avoids replicating the 196-wide
theta vector across (n_timesteps × n_species) rows per sim. Joins on
`sample_index` at read time. Plain float values in canonical species
units (per `model_structure.json`); unit strings stored in
trajectory-parquet schema metadata as a `{species: unit_string}`
sidecar — informational, not a runtime dependency. No Pint, no CSV,
no scalar derivation on the worker.

### Read-path performance: column projection over `sshfs`

Spike test on 2026-05-06 against a real 90 MB / 2000-sim trajectory
parquet on JHPCE (`/home/jeliaso2/qsp-projects/cache/trajectory_grids/
gvax_nivo_neoadjuvant_zheng2022/trajectory_grid.parquet`, 8011 cols
wide-format, 252 species × 31 timepoints + theta/IDs):

| Operation | Time |
|---|---|
| scp full file (4.7 MB/s) | ~20 s |
| Local read, all 8011 cols | 0.56 s |
| Local read, 25 pruned cols | 0.15 s |
| sshfs connect (asyncssh) | 0.75 s |
| pyarrow over sshfs, all 8011 cols (cold) | 23.9 s |
| **pyarrow over sshfs, 25 cols (cold)** | **3.4 s** |
| pyarrow over sshfs, 25 cols (warm reuse) | 2.9 s |

Column projection over sshfs is **~7× faster than full sshfs read, and
faster than scp + local read combined** (3.4 s vs 20 s). Pyarrow's
small-byte-range access pattern is fine over asyncssh; the
naive-paramiko version of this same demo hung indefinitely because
paramiko's default SFTP doesn't pipeline — that *is* the wrong glue,
but `sshfs` (PyPI: `sshfs`, asyncssh-based) handles it.

Linear extrapolation to production scale:

| Scale | Pruned sshfs read | Full sshfs read |
|---|---|---|
| 5k × 3 scen | ~10 s | ~70 s |
| 20k × 3 scen | ~40 s | ~5 min |

20k × 3 pruned-projection is ~40 s — comfortably under the 4-min sim
time. No local disk staging required.

### Recommendation: SLURM writes parquet, runner column-projects via `sshfs`

SLURM runs `qsp_sim` per chunk and writes parquet on HPC (either
natively if `qsp_sim` grows a `--parquet-out` flag, or via a thin
shell-level binary→parquet conversion at the end of the SLURM script —
no Python "worker" needed). Runner uses `sshfs.SSHFileSystem(...)` +
`pyarrow.parquet.read_table(rf, columns=traj_columns)` (with predicate
pushdown on `species` per D8 long-form layout).

```python
ds = pq.ParquetDataset(remote_pool_dir, filesystem=ssh_fs)
traj_df = ds.read(
    columns=['sample_index', 'time', 'species', 'value'],
    filters=[('species', 'in', traj_columns)],
).to_pandas()
```

Why this is the chosen shape:

- **No re-sim on column-set change.** Per D8 the worker dumps every
  species; new cal targets are pure local recompute against the same
  trajectory pool. (Worker-side pruning would force the column set
  into `pool_id` and re-sim on changes — rejected.)
- **Simpler runner, simpler HPC.** No worker module to maintain.
- **Performance is proven sufficient.** Sshfs column-project + species
  filter ≈ 40 s at 20k × 3 scen, well under the 4-min sim phase.

ABM scale (much larger per-cell state) would flip this back toward
worker-side pruning. At QSP scale the simpler path wins.

## Layer 2: `CppSimulator` (`qsp-hpc-tools`)

```python
class CppSimulator:
    def run_hpc(
        self,
        n_simulations: int,
        *,
        traj_columns: Sequence[str],
        min_cadence_hours: float | None = None,  # None → scenario default (D4)
        return_burn_in: bool = False,
    ) -> SimulationBatch: ...

    def simulate_with_parameters(
        self,
        theta: np.ndarray,
        *,
        traj_columns: Sequence[str],
        min_cadence_hours: float | None = None,
        return_burn_in: bool = False,
    ) -> SimulationBatch: ...
```

`SimulationBatch`:

```python
@dataclass
class SimulationBatch:
    theta: np.ndarray              # (n_sims, n_params)
    sample_index: np.ndarray       # (n_sims,) int64 — primary row key (D1)
    traj_df: pd.DataFrame          # long-form: sample_index, time, species, value
                                   #   (matches on-disk layout; raw-data view)
    species_units: dict[str, str]  # canonical units per species (D3)
    burn_in_df: pd.DataFrame | None
    pool_id: str                   # for caching / provenance

# Observable code consumes wide-form ndarrays via species_dict
# (Open Q1, D3); the long→wide pivot lives inside evaluate_targets_to_x
# and is per-target projected, never materialized over all 252 species.
```

Both methods return the same shape — the only difference is the source of
`theta`. PPC stops being a special API; it submits new chunks into the
`ppc/` sub-pool of the same `pool_id` (D1), at a fresh `sample_index`
range owned by that sub-pool's manifest.

### Caching: one HPC pool per scenario, training/PPC sub-pools, `sample_index`-keyed rows (D1)

```
HPC ($SCRATCH/qsp_simulations/{pool_id}/)
    training/
        chunk_NNNN.trajectory.parquet  # long-form: sample_index, time, species, value
        chunk_NNNN.params.parquet      # one row per sample_index, param:* columns
        pool_manifest.json             # chunk → sample_index range, schema metadata,
                                       #   (prior, seed) snapshot for determinism contract
    ppc/
        chunk_NNNN.trajectory.parquet
        chunk_NNNN.params.parquet
        pool_manifest.json             # chunk → sample_index range,
                                       #   posterior-samples-file hash as provenance
```

The HPC pool dir is the only sim cache. Sweeps and audits read from it
via `sshfs`. Training and PPC share `pool_id` (same binary, same
scenario) but live in separate sub-pools — the determinism contract
("`(prior, seed)` → theta at sample_index K") only applies to the
training sub-pool. PPC theta comes from the inference machinery and
makes no such claim, so the sub-pool split keeps the contract clean for
cross-run training intersection while letting PPC share the same `pool_id`
namespace.

**Pool key:** `pool_id = sha256(binary_bytes | scenario_yaml_content)`.
Both inputs genuinely change `f(theta) → outputs`. Nothing else does.

**Row key:** `sample_index` within a pool. Theta vectors live in the
sidecar `chunk_NNNN.params.parquet` keyed on `sample_index` and join
to the trajectory table at read time. Theta is stored as data, not
indexed for content lookup — continuous-distribution prior draws
collide with probability ~0, so a `theta_hash → (chunk, row)` index
pays complexity cost for zero hit rate. Cross-scenario alignment,
τ_cohort / PRCC / NPE arch sweeps, and PPC all work fine with
`sample_index`-keyed rows + prior-and-seed-deterministic theta
sampling on the runner.

**PPC reads** as a fresh `sample_index` range in the `ppc/` sub-pool of
the same `pool_id`: posterior thetas are passed in, the runner submits
a fresh sbatch tagged with sample indices in the PPC sub-pool, and the
resulting chunks land under `{pool_id}/ppc/`. Training cross-run
intersection only consumes `{pool_id}/training/`; PPC consumers point
explicitly at `{pool_id}/ppc/`.

**Range allocator (Option A — session-global).**
`HPCSession.reserve_sample_index_range(n, kind)` allocates `[start, start+n)`
once at the session level and writes the same range into each
scenario's pool manifest under the requested sub-pool (`training` or
`ppc`). The starting point is `max(reservation high-watermark across
all scenario pools the session touches) + 1`, scanned at session start
and cached in-memory. Concurrent runners colliding on the same pool
flock per-pool during the reservation-append (same machinery as PR
#46/47). Consumers track which range they own and intersect within
it; cross-scenario alignment relies on the session having broadcast
the same range to every scenario's training sub-pool in a single
invocation.

```
Local (figures/sbi_runs/<timestamp_label>/)
    artifacts/
        theta_train.pt, x_train.pt, posterior_samples.csv
    plots/
        ...
```

Per-run derived artifacts — *not* a sim cache. Stage-2 / Delta /
parameter_audit consume these.

What goes away:

- `cache/sim_pools/`, `cache/sbi_simulations/` — drop. Read directly
  from HPC over sshfs.
- `compute_test_stats_hash` and `test_stats/<hash>/` subdirs.
- PPC suffix pool as a distinct `pool_id` — folded into the same
  `pool_id` namespace as a `ppc/` sub-pool (D1).
- Per-run "processed tensor cache" (`--cache-processed-tensors`) —
  questionable; conflicts with PPC today. Drop it; the OS page cache
  handles back-to-back re-reads in the same process.

What stays:

- HPC fan-out / sbatch / collection logic. Job orchestration, not caching.
- Pool-scoped uploads + flock (qsp-hpc-tools PRs #46/47/49/50).
- Per-run output dirs under `figures/sbi_runs/`.

If iteration cycles show repeat reads becoming annoying (40 s × N
adds up for sweeps), add a Phase-2 local mirror keyed on `pool_id`:

```
cache/sim_pools/{pool_id}/chunk_*.parquet  # only if measured-needed
```

Materialize on first read, serve from local on subsequent.
Invalidation = `rm -rf` the dir; `pool_id` captures everything. Don't
add until measured.

### Hash cleanup: drop everything but binary + scenario

`compute_pool_id_hash` today (`qsp_hpc/utils/hash_utils.py:16-60`) folds in:

| Input | Status | Action |
|---|---|---|
| `priors_csv` content | The prior is upstream sampling machinery, not a sim input. `sample_index → theta` reproducibility lives at the runner via `(prior, seed)` determinism, not in the pool key. | **Drop** |
| `submodel_priors_yaml` content | Same as above | **Drop** |
| `model_script` (string label, not content) | Cosmetic namespace, not a real input | **Drop** (or fold into dir name as suffix) |
| `binary_path` bytes | Real sim input — captures sim semantics. Byte-deterministic across worktrees since qsp-codegen `fdd81d5`. | Keep |
| `seed` | Determines theta-pool ordering on the runner; orthogonal to the pool's content. | **Drop** |

**Final hash:** `pool_id = sha256(binary_bytes | scenario_yaml_content)`.
That's it. Both inputs genuinely change `f(theta) → outputs`. Nothing
else does.

What this unifies (now there's no fragmenting hash component):

- **PPC stops being a separate `pool_id`.** Today PPC lives in a suffix
  pool keyed on a theta-hash; under the unified key, posterior-driven
  sims write into the `ppc/` sub-pool of the same `pool_id`. Training
  determinism contract stays clean (only `training/` claims it); PPC
  rows live alongside but don't pollute cross-run intersection.
- **Composite-vs-CSV-only mode merges.** Today they fragment into
  different `pool_id`s via the priors-CSV input; under the new key
  they share one pool.
- **τ_cohort, PRCC tier, NPE arch sweeps** — already share a pool
  today (no priors change), still share under the new scheme.
- **Cross-scenario `sample_index` intersection** at
  `sbi_runner.py:812-845` continues to work under D1 **for the
  training sub-pool**: same `(prior, seed)` produces the same theta at
  each `sample_index` in every scenario's `training/`, so intersecting
  on `sample_index` ≡ intersecting on theta. PPC sub-pools don't
  participate in this intersection (their theta comes from the
  inference machinery, not `(prior, seed)`). No rewrite needed beyond
  keeping the `(prior, seed)` → `theta_pool` contract on the runner.

**What this does NOT speed up.** Training-time prior iteration. Every
prior edit draws fresh continuous theta values; nothing reusable in the
old pool. The win is conceptual clarity (one key for one cache), not
faster training iteration.

**Implementation cost.** Now just "remove inputs from a hash + drop
test_stats subdir":

1. **`theta_pool` stays a runner-local sampler.** It computes
   prior-and-seed-deterministic theta sequences. Same role as today,
   just no longer part of the pool key.
2. **PPC routes through `simulate_with_parameters`** which now writes
   into the `ppc/` sub-pool of the same `pool_id` at a fresh
   `sample_index` range. No separate-`pool_id` scaffolding.
3. **Cross-scenario alignment** (`sbi_runner.py:812-845`) keeps using
   `sample_index` intersection, unchanged.
4. **`compute_test_stats_hash` and the `test_stats/<hash>/` subdir** —
   already gone under (b)'s observable-eval move.
5. **Determinism guardrail** (D7): `HPCSession` snapshots the priors-
   CSV content hash at `ensure_remote()` time and refuses to construct
   `theta_pool` later in the run if the on-disk hash has shifted.

The historical "binary bytes flip on every rebuild" pathology was
fixed in qsp-codegen `fdd81d5`; no action needed there.

**Things NOT to add to the hash:**

- `auxiliary_config.yaml` — aux is sampled at observable-eval time
  under (b), not at sim time. Trajectory parquets don't depend on it.
- `traj_columns` / `min_cadence_hours` — read-time / SLURM-request
  decisions under the no-worker design. Adding a cal target that
  touches a new species is a pure local recompute against the same
  trajectory pool; bumping the system-default cadence forces a hard
  cutover (D4/D6), per-scenario override forces a re-sim for that
  scenario only (cache miss, not invalidation).
- `test_stats.csv` — gone, eval is local.

## Layer 3: `qsp-inference`

Already has `evaluate_calibration_target_over_trajectory` with the right
signature (`auxiliary_records=` already supported). Add one helper:

```python
def evaluate_targets_to_x(
    targets: Sequence[CalibrationTarget],
    traj_df: pd.DataFrame,
    species_units: dict[str, str],     # canonical units, informational (D3)
    *,
    auxiliary_records: list[dict] | None = None,
    target_index_values: dict[str, Sequence[float]] | None = None,
) -> tuple[np.ndarray, list[str], EvalReport]:
    """Evaluate every target over traj_df, return (x, names, report).

    x has shape (n_sims, n_targets). NaN where observable.code raised or
    species/aux were missing.

    Inputs are plain-float ndarrays in canonical units (D3); Pint is not
    on the call path. species_units carries the unit string per species
    for assertion and human-readable provenance only.

    target_index_values supplies cal-target time points to interpolate
    to (linear, read-time, per D4). Defaults to each target's YAML
    index_values.

    auxiliary_records is filtered per-target inside; targets without
    auxiliary_parameters declared see no aux.

    Two observable-code signatures (per Open Q9):
      - Default (single-time / multi-time-via-index_values):
          code(time, species_dict, units_dict, aux_record)
        species_dict[name] is (n_sims,) at index_values[0] for
        single-time targets, or (n_sims, len(index_values)) for
        multi-point targets that enumerate all needed times in YAML.
      - Trajectory-access (opt-in via YAML wants_trajectory: true):
          code(time, species_dict, units_dict, aux_record, traj_slice)
        traj_slice exposes the per-sim wide-form view restricted to
        the species the target declared, with per-sim heterogeneous
        time axes (D4 solver-native cadence). Used for fold-change
        without enumerating endpoints, AUC over a window,
        time-to-event, etc. Greppable: rg "wants_trajectory: true"
        calibration_targets/ shows which targets opt in.

    EvalReport carries per-target counters (D7):
      Counter[target_name] -> (n_total, n_nan, n_code_raised,
                               n_species_missing, last_exception)
    Printed as a summary table at end of sbi_runner step 7.
    """
```

This is what `sbi_runner` step 7 calls. Aux is just an optional kwarg —
no special-casing per target outside this helper.

**Pint bridge during D3 migration:** during the transition window
each cal-target YAML carries an opt-in `pint_legacy: true` flag (default
`false`). The helper inspects this flag per target and, when true,
wraps that target's plain-float species values into Pint Quantities and
calls the legacy `(time, species, ureg)` signature. New cal targets are
authored with `pint_legacy: false` and the plain-float
`(time, species, units_dict)` signature. Existing observable code keeps
producing identical numerical outputs without rewrite. Once every cal
target's YAML drops `pint_legacy`, the bridge wrapper is deleted and
`pint` drops from the dependency tree.

The flag is self-documenting: a grep for `pint_legacy: true` shows
exactly which targets remain to be migrated, and the migration path is
"flip the flag, port the `code` to plain-float arithmetic, run unit
test, commit."

Promote `collect_required_trajectory_columns` from
`workflows/calibration_trajectory_pipeline.py` (pdac-build) into qsp-inference
so the runner can build `traj_columns` from cal targets without depending on
pdac-build.

## Layer 2.5: Factor out `HPCSession` (kill per-scenario setup tax)

Today each `CppSimulator` instance redoes its own setup work — binary
upload, venv check, priors upload, theta_pool generation — at
construction or first `run_hpc()`. With N scenarios that's N
redundant setups, each costing 30–60 s of SSH round-trips. Mitigation
in `sbi_runner.py:725-743` runs the first scenario "to populate the
evolve cache" and the rest sequentially, but the orchestration
overhead still repeats per scenario.

Extract once-per-run work into a shared session:

```python
session = HPCSession(
    binary_path=...,
    priors_csv=...,
    submodel_priors_yaml=...,
    seed=...,
    job_manager=job_manager,
)
session.ensure_remote()     # binary upload, venv check, priors upload,
                            # theta_pool generation — all once

# Per-scenario: only the cheap parts (sbatch submit + collect)
batches = {}
for scen, scen_cfg in SCENARIOS.items():
    batches[scen] = session.run_scenario(
        scenario_yaml=scen_cfg['yaml'],
        n_simulations=N,
        traj_columns=traj_cols_per_scen[scen],
    )
```

Saves: ~1–2 min per run on orchestration. Per-scenario sequential
submission still applies (JHPCE login-node sshd doesn't tolerate
concurrent SSH from one client) — but the *cost* of being sequential
is now just the sbatch+collect time, not redundant setup.

**Why we stop at Level 1 (don't fold scenarios into the worker
inner loop):**

- **Future scenarios.** Adding new scenarios down the road shouldn't
  reshape per-worker code; the LMDB evolve_cache (PR #75) makes the
  cross-scenario evolve_state reuse just-as-good as in-process reuse,
  with the additional benefit of persisting across runs and across
  workers.
- **Small per-job runtime is a feature.** Easier sbatch sizing,
  faster failure recovery, no scenario-level all-or-nothing
  failures inside a chunk.
- **Worker startup cost is empirically negligible** at our chunk
  sizes — the marginal saving from eliminating it doesn't pay for
  the structural change.

So `HPCSession` handles the orchestration redundancy; the LMDB
evolve_cache handles the simulation-data redundancy. Together they
cover both angles without changing the worker contract.

## Layer 4: `sbi_runner` step 2

Goes from ~150 lines (HPC fan-out + alignment + post-process) to roughly:

```python
# 1. Determine traj_columns + min_cadence per scenario (D4 default + per-scenario override)
traj_columns_per_scen, min_cadence_hours_per_scen = build_trajectory_request(SCENARIOS)

# 2. One HPC session for the whole run (binary upload, venv check,
#    priors upload, theta_pool gen all happen here, once).
#    HPCSession snapshots the priors-CSV content hash for the
#    determinism guardrail (D7).
session = HPCSession(
    binary_path=BINARY_PATH,
    priors_csv=PRIORS_CSV,
    submodel_priors_yaml=SUBMODEL_PRIORS_YAML,
    seed=cfg.seed,
    job_manager=job_manager,
)
session.ensure_remote()

# 3. Sample the theta pool once (locally; deterministic from prior+seed).
#    Same (prior, seed) → same theta at each sample_index in every scenario;
#    that's what makes sample_index intersection equivalent to theta intersection (D1).
theta_pool = session.sample_theta_pool(NUM_TOTAL_SIMULATIONS)

# 4. Per-scenario submit + collect (cheap parts only).
#    kind="training" routes writes into {pool_id}/training/ (D1);
#    PPC submissions later use kind="ppc" to write into {pool_id}/ppc/.
batches = {}
for scen, scen_cfg in SCENARIOS.items():
    batches[scen] = session.run_scenario(
        scenario_yaml=scen_cfg['yaml'],
        thetas=theta_pool,
        traj_columns=traj_columns_per_scen[scen],
        min_cadence_hours=min_cadence_hours_per_scen[scen],
        kind="training",
    )
    # batches[scen] is a SimulationBatch with theta + traj_df,
    # sample_index-keyed (D1). pool_id = sha256(binary | scenario_yaml).

# 5. Align by sample_index intersection across scenarios
#    (sbi_runner.py:812-845, semantics unchanged under D1)
shared_indices = intersect_by_sample_index(batches)

# 6. Sample aux records per-scenario (D2). Independent draws per cohort
#    by default; same cfg.seed across scenarios = future shared-patient mode.
aux_records_per_scen = {
    scen: (
        HierarchicalAuxiliaryPrior(aux_registry).sample_as_records(
            n_samples=len(shared_indices),
            seed=cfg.seed + scen_idx,   # scenario-specific
        )
        if not aux_registry.is_empty else None
    )
    for scen_idx, scen in enumerate(scenario_names)
}

# 7. Evaluate observables locally per scenario (plain-float, no Pint — D3)
eval_reports = {}
for scen in scenario_names:
    targets = load_cal_targets_from_dirs(SCENARIOS[scen]['calibration_targets'])
    traj_df_scen = batches[scen].slice_to_indices(shared_indices)
    x, names, report = evaluate_targets_to_x(
        targets,
        traj_df_scen,
        batches[scen].species_units,           # informational
        auxiliary_records=aux_records_per_scen[scen],
    )
    scenario_meta[scen]['x_raw'] = x
    scenario_meta[scen]['observable_names'] = names
    eval_reports[scen] = report

# 7b. Per-target NaN summary (D7) — three failure-mode counters distinguish
#     code bugs from column-projection bugs from sim-NaN cases.
log_eval_reports(eval_reports)

# 8. Joint NaN filter + concat (unchanged from today)
```

What disappears compared to today:

- N redundant `CppSimulator` setup paths (`sbi_runner.py:639-714`) →
  one `HPCSession.ensure_remote()`.
- The "first scenario sequential to populate evolve_cache, rest
  sequential because login-node sshd can't fan out" dance
  (`sbi_runner.py:721-743`) — still sequential per-scenario submit,
  but no per-scenario setup tax.
- The aux gate at `sbi_runner.py:540`.
- Cross-scenario timeseries `NotImplementedError` at `sbi_runner.py:875`.
- The shared-aux-across-scenarios assumption (D2) — per-scenario by
  default; shared mode opt-in via shared `cfg.seed` schedule.

## What goes away

- `qsp_hpc/batch/derive_test_stats_worker.py` (~400 LOC).
- `compute_test_stats_hash`, `_local_test_stats_path`, `_download_and_persist`'s
  test-stats branch.
- `simulate_with_parameters`'s separate-`pool_id` suffix-pool plumbing —
  it now writes to the `ppc/` sub-pool of the same `pool_id` as
  `run_hpc` (D1), and returns the same `SimulationBatch` shape.
- The 3-arg-vs-4-arg `compute_test_statistic` wrapper layer (CLAUDE.md gotcha).
- `scenarios/test_statistics/*.csv` loader scaffolding (CSVs already
  deprecated per CLAUDE.md).

## What's new

- `dump_trajectory_worker` replacing `derive_test_stats_worker`.
- `SimulationBatch` dataclass replacing `(theta, x)` tuple return.
- `evaluate_targets_to_x` helper in qsp-inference.
- `collect_required_trajectory_columns` promoted into qsp-inference.
- qsp_sim `--min-cadence-hours` flag (default 4.0) replacing
  `--dt-days`. Internal CVode loop switches from `CV_NORMAL` to
  `CV_ONE_STEP` (D4). Upstream change in qsp-codegen's `CVODEBase.cpp`.
  `--dt-days` is hard-removed (no alias — semantics aren't bit-equal,
  silent-translation would be a footgun); qsp_sim emits a targeted
  error if `--dt-days` is passed, pointing at `--min-cadence-hours` and
  the D4 behavior-change note. Generic argparse "unrecognized
  arguments" is too quiet for the half-merged-worktree case.

## Open API decisions

1. ~~**`SimulationBatch.traj_df` in-memory or lazy?**~~ Resolved by
   2026-05-07 benchmark (`/tmp/bench_traj_layouts.py`): in-memory is safe
   *if* we project at the parquet boundary. Full wide pandas frame at
   20k × 252 sp × 31 t ≈ 1.3 GB (borderline OOM); pruned to 25 species
   ≈ 124 MB (trivial). Contract: `evaluate_targets_to_x` reads only the
   species/time slices it needs, never materializes the full frame.
   **Layout split:** `SimulationBatch.traj_df` stays long-form
   (matches disk storage; long-form is the raw-data view). Observable
   code receives wide-form ndarrays via the `species_dict` API
   (`{name: np.ndarray}` per D3). `evaluate_targets_to_x` is the
   bridge — projects + pivots once per target, hands plain ndarrays to
   `code`. Observable authors never see long-form; the storage layer
   never sees wide-form materialized over all species.
2. **Burn-in default.** Default off; opt-in via `return_burn_in=True` for
   PPC trajectory plotting. Saves the bulk of bandwidth at training time.
3. ~~**Timepoint snapping.**~~ Resolved (D4): solver-native CV_ONE_STEP
   dumps with a 4 h minimum-cadence floor + read-time linear
   interpolation. `pool_id` is independent of cal-target `index_values`;
   adding a new cal-target time is a pure local recompute. Time axis
   is per-sim heterogeneous (each θ produces its own step sequence)
   but long-form schema and per-`sample_index` interp handle that.
4. **`pool_id` per-scenario or model-wide?** Per-scenario by construction
   under D1 — `scenario_yaml_content` is in the hash.
5. ~~**Worker dumps long-form or wide-form?**~~ Resolved by 2026-05-07
   benchmark: long-form wins on every metric. On a 2k-sim / 252-sp / 31-t
   parquet, long-form is **2.4× smaller on disk** (39 MB vs 93 MB —
   categorical species column compresses brilliantly with zstd) and
   fresh-process read is ~2× faster (0.20 s vs 0.37 s). Predicate pushdown
   on `('species', 'in', [...])` is effectively free (0.22 s for 5 species
   out of 252). Trajectory plotter (`run_calibration_trajectory_pipeline`)
   migrates to long-form input.

## Migration / risk

- **Existing pools become unusable.** Layout changes from
  `pool_dir/test_stats/<hash>/test_stats.parquet` (prior in pool_id,
  per-target scalar columns) to `pool_dir/chunk_*.parquet`
  (`sample_index`-keyed long-form trajectory, prior dropped from
  pool_id per D1). Hard cutover (D6). Re-sim cost is small for an
  iteration loop where priors flip anyway.
- **Four repos in flight** (qsp-codegen, qsp-hpc-tools, qsp-inference,
  pdac-build). Per `feedback_worktree_for_hpc_tools_features`, the
  qsp-hpc-tools and qsp-codegen changes each want their own worktree.
- **CPU-bound observable eval** moves to runner host. ~20k × ~25 obs ×
  3 scen = 1.5M plain-float dict evaluations (Pint dropped per D3).
  Expect well under a minute single-threaded; multiprocessing pool
  over theta chunks remains an option if needed. Smoke-test on a 1k
  slice before committing.
- **`sshfs` is a new runtime dependency.** Pure Python (asyncssh-backed),
  installs via `uv pip install sshfs`. Tested against JHPCE 2026-05-06:
  3.4 s for column-projected read of 90 MB / 8011-col parquet.

## PR ordering

Touches three repos. Suggested order, smallest-blast-radius first:

1. **qsp-inference** — `evaluate_targets_to_x` helper; promote
   `collect_required_trajectory_columns` from pdac-build's
   `workflows/calibration_trajectory_pipeline.py`. Standalone,
   testable, no consumer-side change required.
2. **qsp-codegen + qsp-hpc-tools** — biggest PR set. Splits into:
   - **qsp-codegen** `CVODEBase.cpp`: replace `CV_NORMAL`-driven
     `simOdeSample` with a `CV_ONE_STEP` loop honoring
     `min_cadence_hours` (D4). Add `--min-cadence-hours` CLI flag to
     qsp_sim_main; hard-remove `--dt-days` with a targeted error
     pointing at `--min-cadence-hours` and the D4 behavior-change note
     (no alias — semantics differ, silent translation would mislead).
     Test on the instrumented step-count baseline (D4 table) to
     confirm cap behavior.
   - **qsp-hpc-tools** `HPCSession` factor-out from `CppSimulator`
     (Layer 2.5).
   - Content-addressed pool layout: `pool_id = sha256(binary | scenario)`,
     `sample_index`-keyed rows, drop `compute_test_stats_hash` and the
     `test_stats/<hash>/` subdir.
   - SLURM job script writes parquet directly (no Python "worker"
     module — just `qsp_sim` + a shell-level binary→parquet step).
     Long-form schema (per the 2026-05-07 benchmark in Open Q5).
   - `sshfs`-based read path in `CppSimulator` / new `HPCSession`.
   - Update tests; drop `derive_test_stats_worker`.
3. **pdac-build** — `sbi_runner` step 2 rewrite per Layer 4 sketch.
   Delete aux gate (`sbi_runner.py:540`), delete test_statistics CSV
   loader, keep `sample_index` intersection (D1 — `(prior, seed)`
   determinism makes it equivalent to theta intersection). Migrate
   `run_calibration_trajectory_pipeline` to long-form input (Open Q5).
   Run prior-PC verification per the iterative-debugging runbook
   before kicking off 20k.

## Sequence to first 20k SBI run

Per the iterative-debugging runbook (notes/workflows/iterative_model_debugging.md):

1. Land qsp-inference helper PR (`evaluate_targets_to_x`,
   `collect_required_trajectory_columns`).
2. Spin up qsp-codegen worktree. Land `CV_ONE_STEP` + `min_cadence_hours`
   in `CVODEBase.cpp`; add `--min-cadence-hours` CLI to qsp_sim_main;
   verify against the D4 step-count baseline.
3. Spin up qsp-hpc-tools worktree. Land `HPCSession`, content-addressed
   pool layout, `sshfs` read path. Update tests.
4. Spin up pdac-build debug worktree (label `local-eval`). Wire step 2
   per Layer 4 sketch.
5. Trace pattern (per runbook §3): hand-edit a `param_all.xml`, run
   `qsp_sim` once, dump trajectory parquet, call
   `evaluate_targets_to_x` locally, compare the resulting **observable
   values** against today's HPC-derived `x` for the same theta and the
   same **non-aux** cal targets. Bit-equality target is at the
   observable layer (post-interp, post-`code`), not at the trajectory
   row layer: under CV_ONE_STEP (D4) internal step times depend on
   libm/SIMD path, so trajectory row counts and time values can drift
   across hosts even with bit-equal ODE evaluation at any given t.
   Linear interpolation to cal-target `index_values` absorbs that
   drift; observables are the actual contract. Aux-using cal targets
   are excluded from the comparison — aux records are freshly drawn
   locally, with no HPC-side reference to match against. If the
   non-aux observables agree to floating-point roundoff, the local
   eval path is semantically equivalent and the runner can flip over.
6. Submodel priors: `regen_submodel_priors.py` only if priors change
   in passing (this PR set shouldn't touch them).
7. **5k smoke run** from the worktree (per runbook §5):
   ```bash
   nohup .venv/bin/python -m workflows.sbi_runner \
     --scenarios baseline_no_treatment \
     --cpp-branch <branch> \
     --num-total-simulations 5000 \
     > /tmp/sbi_5k_local-eval.log 2>&1 &
   ```
   Watch:
   - **HPCSession setup time** — should be ~1 setup, not 3 (one per
     scenario). Confirms Layer 2.5 plumbing works end-to-end.
   - **Joint reachability + pairwise tension table** — compare to
     pre-change baseline log. Only the aux observables (TGFβ, IL-12,
     ArgI bridge group) should shift; non-aux observables should be
     within roundoff of the pre-change run.
8. **20k full run** with all scenarios (baseline_no_treatment,
   gvax_neoadjuvant_zheng2022, gvax_nivo_neoadjuvant_zheng2022) +
   aux targets included. Watch the same two diagnostics; the 4-min
   sim time + ~40 s `sshfs` species-filter read should keep wall-clock
   under ~6 min for the simulation phase.

## Decision log

Resolved during 2026-05-07 review.

### D1. Pool layout: `sample_index`-keyed rows, no theta-hash index

`pool_id = sha256(binary_bytes | scenario_yaml_content)`. Within a pool,
chunks are written in submission order; rows are addressed by
`sample_index`. Theta vectors are stored as `param:*` columns alongside
outputs (already true today, schema check 2026-05-07) but are *not*
indexed for content lookup.

Rationale: continuous-distribution prior draws collide with probability
~0, so a `theta_hash → (chunk, row)` index pays complexity cost for zero
hit rate. Cross-scenario alignment and τ_cohort / PRCC / NPE arch
sweeps work fine with `sample_index`-keyed rows in `training/` +
prior-and-seed-deterministic theta sampling on the runner. PPC also
uses `sample_index`-keyed rows but in the `ppc/` sub-pool, with theta
from the inference machinery — no determinism claim for those rows.

PPC submits new chunks into the `ppc/` sub-pool of the same `pool_id`,
with its own manifest and `sample_index` range. The `(prior, seed)`
determinism contract scopes to the `training/` sub-pool only — PPC
theta comes from the inference machinery and makes no determinism
claim. Cross-run training intersection consumes `training/` exclusively.

`HPCSession.reserve_sample_index_range(n, kind)` is session-global
(Option A): allocates one range, broadcasts it into each scenario's
manifest under the requested sub-pool, flock-protected per pool dir.
Starting watermark = `max(reservations across all scenario pools the
session touches) + 1`.

Scenario YAML hashing: whole-file content hash is fine. Scenarios change
rarely; cosmetic-edit-induced re-sims are an acceptable tax for
simplicity over canonicalization.

### D2. Aux records: per-scenario by default

```python
aux_records_per_scen = {
    scen: HierarchicalAuxiliaryPrior(aux_registry).sample_as_records(
        n_samples=len(shared_thetas),
        seed=cfg.seed + scen_idx,   # <-- scenario-specific by default
    )
    for scen_idx, scen in enumerate(scenario_names)
}
```

Rationale: PDAC cohorts are different patient populations measured by
different labs (Yu 2017 ≠ Zheng 2022). Cohort-translation σ,
measurement-error realization, and instrument calibration aux draws
should be statistically independent across scenarios. The
`tau_cohort`-default-0.5 random-effect machinery already encodes this
inter-cohort heterogeneity at the prior level.

**Future shared-patient mode** (counterfactual interpretation, same
patient across scenarios): override the seed schedule —

```python
aux_records_per_scen = {
    scen: HierarchicalAuxiliaryPrior(aux_registry).sample_as_records(
        n_samples=len(shared_thetas),
        seed=cfg.seed,   # <-- shared seed = shared aux per theta
    )
    for scen in scenario_names
}
```

Document this as a `--shared-aux-across-scenarios` runner flag once a
use case appears (e.g., a single virtual-patient prediction across
multiple treatment counterfactuals). Default stays per-scenario.

**Aux record persistence.** Aux records do not enter the trajectory
pool (they're observable-side, not sim-side). For training that's
fine — `theta_train.pt` and `x_train.pt` capture everything needed. For
PPC reproducibility, persist the realized aux records alongside other
per-run artifacts:

```
figures/sbi_runs/<timestamp_label>/artifacts/
    aux_train_{scenario}.parquet      # one row per sim, aux:* columns
    aux_ppc_{scenario}.parquet
```

Cheap (one row per sim per aux variable) and avoids the silent-drift
risk of "recompute aux at PPC time from the same seed schedule" if the
aux registry shape changes between training and PPC (analogous to the
prior-CSV drift D7 catches).

### D3. Drop Pint, minimize disruption

Replace `compute_test_statistic(time, species_dict, ureg)` with
`compute_test_statistic(time, species_dict, units_dict)` where:

- `species_dict[name]` is a plain `np.ndarray` of float values, already
  converted to the species' canonical unit by `evaluate_targets_to_x`
  at trajectory-load time (single multiplicative factor per species).
- `units_dict[name]` is the canonical unit *string* (e.g., `"nM"`,
  `"cell/microliter"`) — informational, for observable-code asserts and
  human readability. Not a registry.

Migration steps (minimal disruption):

1. `evaluate_targets_to_x` does the unit conversion at the trajectory
   boundary using a precomputed `{species: factor}` dict from
   `model_structure.json`. Per-target dispatch between the new
   (`species_dict: ndarray`, `units_dict: dict[str,str]`) and legacy
   (`species_dict: Pint Quantity`, `ureg`) signatures is driven by an
   explicit YAML flag `pint_legacy: true|false` (default `false`), per
   the bridge described in Layer 3. No runtime introspection, no
   exception-text dispatch.
2. New observable code uses plain floats. The Pint `* ureg.dimensionless`
   boilerplate goes away. `.to('uM').magnitude` patterns get replaced by
   either pre-conversion at load time or a hardcoded factor.
3. Add `tests/test_observables_unit_consistency.py` with two layers:
   (a) assert each cal target's declared `units` matches the canonical
   unit of the underlying species (post-conversion); (b) a generic
   harness that, for each cal target, builds a Pint-Quantity
   `species_dict` from `units_dict`, calls `code` with it, and asserts
   no `DimensionalityError`. (b) catches the
   `feedback_unit_harmonize_before_hill` class of bug — embedded unit
   constants in `code` (e.g., EC50 hardcoded in pM while species
   arrives in nM) — at PR time without runtime cost. Production stays
   plain-float; Pint is repurposed from runtime safety net to test-time
   guard, which is the role it was always best at. Authors write `code`
   using `+`, `-`, `*`, `/` and array-broadcast operations that work on
   both plain floats and Pint Quantities, so no per-target unit-test
   fixture is needed; one harness covers everything.
4. Migrate existing observable code to plain-float incrementally — no
   big-bang rewrite. Each migration is "flip `pint_legacy` to false,
   port the `code` to plain-float arithmetic, run unit test, commit."
   Migration progress is greppable: `rg "pint_legacy: true"
   calibration_targets/` shows what remains.
5. Once `rg` returns empty, delete the legacy bridge wrapper, drop
   `pint` from the dependency tree, and remove the 3-arg-vs-2-arg
   gotcha entry from CLAUDE.md.

The YAML-flag bridge keeps disruption low: existing cal targets keep
producing identical numerical outputs through the transition, the
migration state lives next to the target definition (where authors are
already looking), and there's no introspection-based dispatch to age
badly.

### D4. Trajectory grid: solver-native steps with a min-cadence cap

Replaces the earlier uniform-`dt_default` proposal. qsp_sim already
runs CVODES adaptive integration internally — coarsens during
steady-state, refines through dose transients — but throws the
adaptivity away by interpolating output back onto a user-fixed
`--dt-days` grid. Keep the adaptivity at the dump boundary.

**Scheme:** dump one row per CVode internal step, subject to a
**minimum-cadence floor** that suppresses dumps coming faster than a
configurable rate. Default floor: **4 hours** (`min_cadence_hours=4.0`).
Per-scenario override via scenario-YAML `traj_min_cadence_hours`.

Pseudocode (replaces the `CV_NORMAL` loop in `simOdeSample`):

```cpp
// CLI exposes hours; internal time axis is days. Convert once.
const double min_cadence_days = min_cadence_hours / 24.0;

// Switch to CV_ONE_STEP and accumulate dumps with a cadence floor
double t_last_dump = t_start;
while (t < t_end) {
    int flag = CVode(_cvode_mem, t_end, _y, &t, CV_ONE_STEP);
    check_flag(&flag, "CVode", 1);
    if (t - t_last_dump >= min_cadence_days || t >= t_end) {
        write_row_to_parquet(t);
        t_last_dump = t;
    }
}
```

**Why this beats both alternatives** (uniform fine grid; pure
solver-native):

- **Uniform `dt = 0.5 d`** wastes rows during long steady-state
  stretches in long-horizon scenarios (1095 d × 0.5 d = 2190 points
  per sim) and is too coarse during fast-PK transients (capping
  RECIST/PD-1 receptor dynamics). The 50-point cap-formula was both
  arbitrary and lossy.
- **Pure solver-native** explodes on stiff treatment scenarios.
  Empirical (2026-05-07): GVAX+nivo over 30 d takes 935 internal
  steps; over 1095 d, 2117 steps. At 20k sims that's 21 GB compressed
  per scenario — manageable but lumpy.
- **Cap at 4 h** gives the best of both: solver coarsens automatically
  during steady-state (clinical_progression 365 d → 25 dumps), and
  the cap bounds the worst case at 6 dumps/day regardless of
  internal stepping (gvax_nivo 30 d → ~180 dumps instead of 935).

**Empirical step counts** (instrumented `CVodeGetNumSteps`, default
rtol/atol on PDAC ODE; 2026-05-07):

| Scenario | t_end (d) | CVode steps | steps/day | Capped @ 4h |
|---|---|---|---|---|
| baseline_no_treatment | 1 | 4 | 4.0 | 4 |
| gvax_neoadj_zheng2022 | 21 | 326 | 15.5 | 126 |
| gvax_nivo_neoadj_zheng2022 | 30 | 935 | 31.2 | 180 |
| clinical_progression | 365 | 25 | 0.07 | 25 (uncapped) |
| clinical_progression_longterm | 1095 | 43 | 0.04 | 43 (uncapped) |
| gvax_neoadj_longterm | 1095 | 664 | 0.61 | ~660 (uncapped) |
| gvax_nivo_neoadj_longterm | 1095 | 2117 | 1.93 | ~700 (cap bites only at doses) |

The cap is mostly idle on slow scenarios — solver-native is already
coarser than the floor. It only bites on treatment scenarios where
the solver wants to resolve sub-hour PK transients we don't need to
record at full fidelity.

Read-time linear interpolation lifts samples to cal-target
`index_values`. Cal-target endpoints are typically integer days
(d=21, d=0); 4 h cadence gives ≤2 h interpolation distance, well
inside any cal-target CI budget. Add `tests/test_traj_interp.py`
covering known-shape (constant, linear, exponential decay) over
realistic step density.

Burn-in (pre-diagnosis) uses a coarser cadence floor
(`burn_in_min_cadence_days=14.0`, existing default in dump worker
proposal). Burn-in shape is for QA plotting only, not observable eval.

**`pool_id` independence:** internal step times depend on (binary,
scenario, theta) — already in the pool key. Different theta values
yield different step sequences in the same pool, which is *fine*:
long-form parquet stores `time` per row, and read-time interp doesn't
require a shared time axis. Each sim's row count varies; the pool
manifest reports min/max/median per chunk for diagnostics.

**System contract:** `min_cadence_hours` default sits in qsp_sim CLI
(`--min-cadence-hours 4.0`). Bumping the default = hard cutover (D6)
just like binary rebuilds; per-scenario YAML override is a normal
re-sim trigger. Add a unit test fingerprinting the default so a
silent edit trips CI.

### D5. sshfs: spike under realistic load before committing

Pre-commit verification (do this before merging the qsp-hpc-tools PR):

1. Mount `/home/jeliaso2/qsp-projects/cache/trajectory_grids/` over
   sshfs from a runner host while a normal sbatch submit/collect cycle
   is in flight against the same client.
2. Read a 90 MB / 8011-col trajectory parquet with column projection
   (~25 cols). Wall-clock budget: ≤5 s (3.4 s observed in isolation).
3. Repeat with two concurrent sshfs mounts from the same client (e.g.,
   simulating a future fan-out attempt). Watch for sshd `MaxStartups`
   throttling. If it kicks in, mount-and-unmount per-read instead of
   holding a long-running mount.
4. Fallback path if sshfs stalls under load: scp staged TMPDIR (~20 s
   for 90 MB at observed 4.7 MB/s; still well under sim time at all
   scales).

Mount lifecycle: open inside the read helper (`with sshfs.SSHFileSystem(...) as fs: ...`),
unmount immediately after the parquet read returns. No long-running
mount across the runner process lifetime — that's the policy-friendly
shape.

### D6. Hard cutover

Existing pools become unusable. No transitional reader. Re-sim cost is
small for an iteration loop where priors flip anyway.

### D7. Smaller things

- **`HPCSession.sample_theta_pool(N)` determinism contract:**
  `(priors_csv | submodel_priors_yaml | seed) → theta_pool`,
  scenario-independent. Scoped to the **training sub-pool only** — PPC
  theta comes from the inference machinery and is not regenerated from
  `(prior, seed)`. `HPCSession` snapshots both files' content hashes at
  `ensure_remote()` time and refuses to construct `theta_pool` if
  either drifted on disk later in the run, but does not gate writes
  into `ppc/` sub-pools. Covers both manual edits and
  `regen_submodel_priors.py` refreshes from a parallel agent loop
  (sanity guardrail; user has been bitten by silent prior shadowing
  per `feedback_csv_sigma_shadowed_by_submodel`).

- **Per-target NaN logger (eval step 7):** maintain
  `Counter[target_name] → (n_total, n_nan, n_code_raised,
  n_species_missing, last_exception)`. Print summary table at end of
  step 7. Distinguish (a) `code` raised — programming bug,
  (b) species missing from `traj_df` — column-projection bug,
  (c) NaN from the simulation itself — silent failure surface. Three
  different fix paths.

- **`--cache-processed-tensors`:** consumers unknown; grep before
  removing. Likely answer: only `sbi_runner` itself re-reads inside the
  same run, OS page cache covers it. Action: pre-removal, run
  `rg --type py "cache_processed_tensors|processed_tensor_cache"` across
  pdac-build + qsp-inference + qsp-hpc-tools. If the only hit is the
  flag's own definition, drop it. If `parameter_audit.py` or stage-2
  promotion reads from there, update those consumers first.

- **PPC-vs-training mixing:** resolved structurally by the sub-pool
  split (D1). Training rows live under `{pool_id}/training/`, PPC rows
  under `{pool_id}/ppc/`. Cross-run training intersection consumes
  `training/` exclusively; PPC consumers point at `ppc/`. No
  `range_filter` abstraction or kind-tagging at the row level.

### D8. Dump everything, project at read time (resolves traj_columns heterogeneity)

Followup to the 2026-05-07 long-vs-wide benchmark. Worker (or SLURM
parquet step) writes:

- **All species** present in the simulation output. ~252 columns at QSP
  scale; long-form storage with zstd-compressed categorical
  `species` column makes this cheap.
- **Solver-native time axis** with a 4 h minimum-cadence floor (D4).
  Step times vary per-sim; long-form schema doesn't care.

Read-time filtering handles all variation:

- New cal target touches a new species → predicate-push `species=...`,
  pure local recompute against existing chunks.
- New cal target wants a new `index_values=[t]` → linear interp from
  the per-sim solver-native rows, pure local recompute.

Why this matters: under per-write `traj_columns`, the pool manifest
would need per-chunk schema metadata, and a request for a species not
previously dumped would force a re-sim. Under D8, every chunk has every
species, so heterogeneity within a pool can't happen for the species
axis. The time axis *is* per-sim heterogeneous (solver-native) but
that's read-time-handled by interp on a per-`sample_index` basis.

**Storage cost** (long-form, zstd-compressed; conservative
extrapolation from the 2k × 252 sp × 31 t = 39 MB benchmark. The
benchmark's effective compression is ~2.5 B per long-form row
post-zstd+categorical; the table below uses a deliberately conservative
~4× headroom over that, so observed footprints will likely come in
smaller):

| Scenario | Capped output rows / sim | 20k sims, compressed |
|---|---|---|
| baseline (1 d) | 4 | 40 MB |
| gvax_neoadj (21 d) | ~126 | 1.3 GB |
| gvax_nivo (30 d) | ~180 | 1.8 GB |
| clinical (365 d) | 25 | 250 MB |
| clinical_longterm (1095 d) | 43 | 430 MB |
| gvax_neoadj_longterm (1095 d) | ~660 | 6.6 GB |
| gvax_nivo_longterm (1095 d) | ~700 | 7.0 GB |

Current SBI input (3 short scenarios × 20k sims): **~3.1 GB total**.
Long-horizon clinical 3-scenario × 20k: **~14 GB**. All comfortably
within JHPCE scratch budget; the cap keeps the worst case (uncapped
gvax_nivo 30 d) from ballooning to 9 GB.

Read cost (sshfs species-filter, extrapolated linearly from the
2026-05-06 spike at ~7 MB / s effective for filtered reads):

| Scale | Pruned sshfs read | Fraction of ~4-min sim phase |
|---|---|---|
| 5k × 3 short scen (~0.8 GB) | ~10 s | ~4% |
| 20k × 3 short scen (~3.1 GB) | ~40 s | ~17% |
| 20k × 3 long-horizon (~14 GB) | ~3 min | ~75% |

Long-form predicate pushdown is effectively free at the parquet level
(0.06 s for 5/252 species in the benchmark), so the read time is
dominated by SSH transport, not query cost. Per-sim time-axis variation
adds no read-time penalty beyond the interp itself.

**Caveat — system-contract drift:** if `min_cadence_hours` default
changes, existing pools' time-axis density shifts. Read-time interp
will still work (any cadence interpolates to a cal-target time), but
interpolation error scales with cadence. A 4 h → 12 h bump degrades
fast-PK observables silently. Treat the default as part of the system
contract; bumping it is a hard cutover (D6) just like binary rebuilds.
Per-scenario YAML override is a normal re-sim, not a cutover, because
each scenario's cadence lives downstream of the pool key.
