# SBI workflow on the C++ backend — handoff (2026-04-16)

Context for the next session: `workflows/sbi_runner.py` was switched from
the MATLAB `QSPSimulator` to the C++ `CppSimulator`. The
`sbi_workflow.py` entry point still works the same way — under the hood
it now invokes the qsp_sim binary on HPC instead of MATLAB.

## TL;DR

- Run `python workflows/sbi_workflow.py` (or `python -m workflows.sbi_runner ...`).
  No CLI changes; the C++ paths come from `RunConfig` defaults that point
  at the sibling SPQSP_PDAC checkout.
- First HPC submission rebuilds the qsp_sim binary on the cluster; later
  ones reuse the cached build (~zero overhead when source is unchanged).
- Local `qsp_sim` is built and ready at
  `/Users/joeleliason/Projects/SPQSP_PDAC/PDAC/qsp/sim/build/qsp_sim`.
  Rebuild if you pull SPQSP_PDAC changes
  (`cmake --build /path/build --target qsp_sim`).
- **PPC is not supported on the C++ backend yet** — leave
  `run_posterior_predictive=False` (the default). If you turn it on,
  `run_inference` raises `NotImplementedError` with a pointer to the
  missing `simulate_with_parameters` method.

## What changed in `sbi_runner.py`

Sole edit: the per-scenario simulator construction in section 2a (and
the corresponding fast-path block).

| Before | After |
|---|---|
| `QSPSimulator(model_script=..., max_tasks=..., project_root=...)` | `CppSimulator(binary_path=..., template_xml=..., scenario_yaml=..., drug_metadata_yaml=..., healthy_state_yaml=..., job_manager=HPCJobManager())` |
| `QSPResultLoader` fast path → `sim(N)` fallback | `sim.run_hpc(N)` (handles the same 3-tier walk internally) |
| `sim._test_stats_df.copy()` | `pd.read_csv(sim.test_stats_csv)` |

`RunConfig` gained 8 new fields with sensible defaults:

```python
scenarios_dir: str = 'scenarios'
cpp_binary_path: str = '<SPQSP_PDAC>/PDAC/qsp/sim/build/qsp_sim'
cpp_template_xml: str = '<SPQSP_PDAC>/PDAC/sim/resource/param_all.xml'
cpp_subtree: str = 'QSP'
cpp_drug_metadata_yaml: str = '<SPQSP_PDAC>/PDAC/sim/resource/drug_metadata.yaml'
cpp_healthy_state_yaml: str = '<SPQSP_PDAC>/PDAC/sim/resource/healthy_state.yaml'
cpp_t_end_days: float = 1.0
cpp_dt_days: float = 0.5
```

Per-scenario YAMLs auto-resolve as `f'{scenarios_dir}/{scen_name}.yaml'`
unless the scenario dict in `cfg.scenarios` sets `scenario_yaml`
explicitly. `cpp_t_end_days` is a fallback — the scenario YAML's
`sim_config.stop_time` overrides it inside `qsp_sim`.

## Prerequisites checklist

| Layer | State as of 2026-04-16 |
|---|---|
| Local `qsp_sim` binary | Built (922KB, links against the SPQSP_PDAC `cpp-sweep-binary-io` HEAD as of 461f87fd) |
| `qsp-hpc-tools` editable install in pdac-build venv | `cpp-simulation-backend` branch (verified `/Users/joeleliason/Projects/qsp-hpc-tools/qsp_hpc/__init__.py`) |
| `~/.config/qsp-hpc/credentials.yaml` `qsp_hpc_tools_source` | Pinned to `git+ssh://...@cpp-simulation-backend` |
| `~/.config/qsp-hpc/credentials.yaml` `cpp.branch` | Default `cpp-sweep-binary-io` |
| HPC `qsp-hpc-tools` venv | Auto-upgraded from the pinned branch on every submission |
| HPC `qsp_sim` build | Auto-rebuilt by `ensure_cpp_binary` on every submission |

## Running the workflow

```bash
# Default RunConfig (5000 sims × 4 scenarios)
python workflows/sbi_workflow.py

# Or via the sweep runner with custom config
python -m workflows.sbi_runner --num-total-simulations 1000 --seed 42

# Or via JSON config (full RunConfig override)
python -m workflows.sbi_runner --config-json sweep_specs/my_run.json
```

What you'll see:

1. **Per-scenario CppSimulator init** — logs binary path, config hash,
   pool dir.
2. **`run_hpc(N)` 3-tier walk** for each scenario:
   - **Tier 1** (local cache) — populated only after a previous run
     downloaded test stats; fresh slate misses.
   - **Tier 2** (HPC pre-derived test stats) — hits if a previous
     workflow already submitted this scenario; downloads
     `chunk_*_test_stats.csv` + `chunk_*_params.csv` (~MB scale).
   - **Tier 3** (HPC full sims, no derived stats) — submits a derivation
     job, waits, downloads.
   - **Tier 4** (no HPC pool yet) — submits a fresh C++ array of `N`
     sims with chained derivation (`afterok:<array_id>`), waits, downloads.
3. **NPE training + diagnostics** — unchanged from the MATLAB-era flow.

## What to expect on the first run

- HPC submission takes ~90-130s end-to-end (rsync + venv probe + git
  pull + cmake + sbatch). Subsequent runs are faster (cached venv,
  near-noop incremental cmake).
- ~13.6% NaN rate from `evolve_to_diagnosis` failures — this is the
  expected baseline for the `baseline_no_treatment` scenario at
  `t_end_days=1`. Per-scenario rates will differ for treatment scenarios.
- Pool dir on HPC:
  `{simulation_pool_path}/{model_version}_{config_hash[:8]}_{scenario}/`
  with `config_hash` covering: priors CSV, model_version,
  submodel_priors_yaml, qsp_sim binary SHA-256, template XML content,
  scenario_yaml + drug_metadata_yaml + healthy_state_yaml content.
  Rebuilding qsp_sim ⇒ new pool dir.

## Known gotchas / followups

1. **PPC not implemented.** `run_posterior_predictive=True` raises a
   clear `NotImplementedError` from sbi_runner section 7. The fix is
   ~50 LOC on `CppSimulator`: a `simulate_with_parameters(theta)`
   method that mirrors `QSPSimulator.simulate_with_parameters` (theta-
   hashed suffix pool + chained derivation). Not done because it
   touches an untested code path.
2. **`cpp_binary_path` must exist locally.** `CppSimulator.__init__`
   reads the binary bytes for the config-hash; missing binary →
   `FileNotFoundError`. Build with:
   ```bash
   cd /Users/joeleliason/Projects/SPQSP_PDAC/PDAC/qsp/sim
   mkdir -p build && cd build && cmake .. && cmake --build . --target qsp_sim -j 4
   ```
3. **Pool-id convention diverges from MATLAB.** C++ pool IDs include
   the binary SHA; MATLAB pool IDs use priors-only. Existing MATLAB
   pools on HPC are inaccessible from the C++ workflow (and vice
   versa). C++ pools are isolated by design.
4. **All model parameters now appear as `param:*` columns.** Cal-target
   functions can read any model parameter via `species_dict[name]`
   (sampled values from the prior; non-sampled values from the
   template default). This was the fix for the recurring "rho_collagen
   not in simulation data" failures.
5. **Status convention is now `0=success, 1=failure` in both backends.**
   Existing MATLAB-derived test stats (written under the old
   tri-state convention) will be misinterpreted by the worker. Re-run
   any MATLAB pools you still care about, or treat them as stale.

## PR / branch state at handoff time

| Repo | Branch | Pushed? | PR? |
|---|---|---|---|
| `qsp-hpc-tools` | `cpp-simulation-backend` | yes | not opened yet |
| `SPQSP_PDAC` | `cpp-sweep-binary-io` | yes (HEAD: `461f87fd`) | not opened yet |
| `pdac-build` | `fix/sbi-workflow-cleanup` | local edit, **uncommitted**: `workflows/sbi_runner.py` | n/a |

Recommended merge order if/when ready: SPQSP_PDAC → qsp-hpc-tools →
pdac-build (the binary-v2 format change is load-bearing for the
others).

## Diagnostic commands

```bash
# Check HPC qsp-hpc-tools branch:
ssh hpc 'cd ~/qsp-projects && cat .venv/hpc-qsp/lib/python3.11/site-packages/qsp_hpc-*.dist-info/direct_url.json 2>/dev/null'

# Check HPC SPQSP_PDAC branch + latest commit:
ssh hpc 'cd ~/SPQSP_PDAC && git log --oneline -1 && git rev-parse --abbrev-ref HEAD'

# Find the most recent C++ pool:
ssh hpc 'ls -lt /home/jeliaso2/data-apopel1/jeliaso2/qsp_simulations/ | head -5'

# Inspect a chunk's per-target NaN rate (replace HASH):
ssh hpc 'wc -l /path/to/pool/test_stats/HASH/chunk_*_test_stats.csv'
```

## Smoke validation that landed this work

`baseline_no_treatment_smoke_v5` (1k sims, scenario YAML + cal targets +
model_structure_file + chained derivation) showed all 22 cal targets
at exactly 13.6% NaN, matching the inherent `evolve_to_diagnosis`
failure rate. End-to-end ~2 min wall time. See `docs/CPP_SIMULATION_PLAN.md`
in qsp-hpc-tools for the full v1→v5 progression and the bug history
(binary v2 → status polarity → template defaults → model_structure
required).