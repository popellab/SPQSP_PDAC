# Iterative model debugging runbook

Reference for the recurring touchpoints we hit when adjusting parameters,
forward models, or calibration targets. Edit freely — this is the canonical
checklist linked from auto-memory.

> **Companion doc:** [`model_debugging_wisdom.md`](model_debugging_wisdom.md)
> captures the *reasoning* lessons from past sessions (which prior is actually
> binding, when traces mislead about SBI behavior, why structural additions
> sometimes don't help baseline, etc.). Skim it before proposing a fix.

The five touchpoints, in the order you usually need them:

1. [Spin up a debugging worktree](#1-spin-up-a-debugging-worktree)
2. [Rebuild the C++ binary via the Makefile](#2-rebuild-the-c-binary-via-the-makefile)
3. [Trace a single qsp-sim run with adjusted params](#3-trace-a-single-qsp-sim-run-with-adjusted-params)
4. [Refresh `submodel_priors.yaml`](#4-refresh-submodel_priorsyaml)
5. [Kick off an SBI verification run from the worktree](#5-kick-off-an-sbi-verification-run-from-the-worktree)

---

## 1. Spin up a debugging worktree

When the change spans more than a couple of files (calibration target rewrites,
prior re-anchoring, forward-model edits), do it in a worktree, not on a branch
in the primary checkout. See `feedback_worktree_for_hpc_tools_features.md`.

```bash
scripts/setup_debug_worktree.sh <short-label>            # forks from main
scripts/setup_debug_worktree.sh <short-label> <base>     # fork from another branch
```

Creates `feature/<short-label>` at `../pdac-build-<short-label>` and populates
the gitignored bits the rest of the loop expects:

- `cpp/sim/build/_deps/` seeded from primary (skips the 5-8 min SUNDIALS /
  yaml-cpp clone — deps are static, no pool-hash implications)
- `cpp/sim/build/qsp_sim` built via `make refresh-cpp` from the worktree
  (per-worktree binary; never symlinked from primary — the pool hash folds
  in binary bytes, see `feedback_cpp_local_binary_invalidates_hpc_cache.md`)
- `.compare_cache/` snapshot (worktree-owned after copy)
- `.qsp-hpc/` credentials override (so `sbi_runner` doesn't fall back to
  global creds pointing at legacy paths)
- `.env` for the OpenAI key (maple-driven extraction)
- `.venv` symlinked to primary (safe — absolute-path shebangs, no pool-hash
  implications)

The script is the source of truth for *how*; the rules table below explains
*why* each artifact has the policy it does. Total wall-clock: ~2-3 min.

**Per-worktree venv alternative** (~15-30s with cached wheels) when the
branch needs different sibling-repo source than primary, after running the
setup script:

```bash
cd "$WT" && rm .venv && uv venv && uv pip install -r requirements.txt \
  && uv pip install -e ../qsp-inference -e ../qsp-hpc-tools -e ../maple
```

For most debugging the symlink default is faster and lower drift risk.

Tear-down at end of session: see `feedback_editable_install_after_worktree_removal.md`
— if you `pip install -e` anything from a worktree, reinstall from the primary
before removing it.

### Cross-worktree artifact rules

| Artifact | Rule |
| --- | --- |
| `cpp/sim/build/` | **Always per-worktree.** Never symlink to primary. The pool hash folds in binary bytes, so a contaminated primary build silently corrupts pool caches across every worktree that symlinks to it. |
| `cpp/sim/build/` rebuilds | Only ever rebuild *into the worktree's own* `cpp/sim/build/`. Never run `make refresh-cpp` while a symlink to primary is in place. |
| `.compare_cache/` | **Snapshot-at-creation, then worktree-owned.** Copy once at worktree creation; updates from the worktree's audit/regen runs accumulate locally. Don't propagate back to primary. Cache is hash-keyed, so stale entries miss safely. |
| `.qsp-hpc/credentials.yaml` | **Copy at creation.** Gitignored; without it, the runner falls back to the global `~/.config/qsp-hpc/credentials.yaml`. Project-specific overrides (e.g., `cpp.repo_path`) silently disappear and you end up against whatever the global default points at. |
| `.env` | **Copy at creation.** Gitignored; holds the OpenAI API key used by maple-driven extraction. Without it, `scripts/staged_extraction.py` and the maple MCP tools fail at the LLM call. |
| `.venv/` | **Symlink at creation.** Gitignored; many Makefile targets and helper scripts hard-code `.venv/bin/<tool>` paths (qsp-codegen, qsp-refresh-param-xml, pytest, etc.). Symlink is safe — the venv contains tool installations with absolute-path shebangs, no pool-hash implications. Save the user from `make: .venv/bin/qsp-codegen: No such file or directory` half a workflow in. |
| `notes/calibration/submodel_priors.yaml` | **Tracked file** — git handles it. Don't hand-copy across worktrees. |
| `cache/sbi_simulations/` (HPC pool) | Hashed by config; multiple worktrees with the same priors / observable formulas hit the same HPC pool, different content yields different pools automatically. No cross-worktree action needed. |

Time budget per worktree spin-up: ~2–3 min for the cmake build + cache copy.
The earlier symlink shortcut (~30 sec) is faster but unsafe enough that the
trade isn't worth it.

---

## 2. Rebuild the C++ binary via the Makefile

When you've changed something that affects the simulator, use the Makefile —
don't hand-roll cmake invocations. `make help` lists the menu; the relevant
targets here:

| Target | When to run |
| --- | --- |
| `make refresh-cpp` | After editing `parameters/parameters_PDAC.m`, `core/modules/*.m`, or `scripts/immune_oncology_model_PDAC.m`. Regenerates SBML (MATLAB), runs codegen into `cpp/qsp/ode/`, refreshes `resources/cpp/param_all.xml`, and incrementally rebuilds `cpp/sim/build/qsp_sim`. **Requires MATLAB.** |
| `cmake --build cpp/sim/build --target qsp_sim -j4` | After editing only `cpp/qsp/*` or `cpp/sim/*` source directly (no MATLAB module touched). Skip the SBML/codegen path entirely. |
| `make parity` | **Always** run after MATLAB-side simbio changes (anything that flowed through `make refresh-cpp`). 365-day MATLAB ↔ C++ trajectory parity test catches codegen / SBML drift that the unit tests can't. Slow, MATLAB-dependent — but skipping it is how you discover later that the C++ binary disagrees with the MATLAB model on day-90 dynamics. |
| `make check-cpp-sync` | Quick sanity: priors CSV ↔ param XML ↔ species_units.json all reference the same parameter set. Run after `refresh-cpp` or any param-rename PR. |
| `make hpc-status` | Show which qsp-projects branch HPC is configured to pull, flag drift. Worth running before `--cpp-branch <feature>` SBI invocations to catch credential staleness. |
| ~~`make refresh-all`~~ | **User-only.** Don't run this from agent debug loops — it's a heavy pre-PR sweep including the full audit. The agent should compose the individual sub-targets it actually needs. |

Notes:

- **`git checkout` / `git worktree add` resets mtimes to "now."** Make's
  rebuild decisions are mtime-based, so right after a checkout or fresh
  worktree the generated artifacts (`model_structure.json`,
  `PDAC_model.sbml`) and their MATLAB sources may share an identical
  timestamp — make will declare everything up-to-date and skip work even
  if the artifacts on disk don't actually correspond to the current
  source. Symmetrically, a checkout that overwrites only a single source
  file can stamp it newer than the artifacts and trigger a rebuild that
  isn't logically needed. When in doubt after switching branches or
  spinning up a worktree, force: `make -B refresh-cpp` (or `-B
  model-structure` / `-B export-sbml` for just one). Cheap insurance.
  The `.PHONY` declarations on the alias targets don't help here — the
  alias has no recipe of its own; the file rule below it is what gates
  the work.
- **Don't manually `rm -rf` the build directory.** `refresh-cpp` is incremental
  (`cmake --build` only rebuilds changed targets) and a clean from scratch
  costs minutes. Per `feedback_cpp_local_binary_invalidates_hpc_cache.md`,
  rebuilding `qsp_sim` orphans every pool whose hash folded in the old
  binary bytes — treat clean rebuilds as a real cache invalidation, not a
  reflex.
- **No symlinks across worktrees.** Per Section 1a, every worktree owns its
  own `cpp/sim/build/`. Never run `refresh-cpp` while pointed at another
  worktree's build dir; the resulting binary will silently disagree with
  the source the worktree thinks it has.
- `make audit` and `make refresh-submodel-priors` exist in the Makefile but
  are user-facing. Don't reach for them from agent debugging loops — use
  `scripts/regen_submodel_priors.py [--invalidate PARAM ...]` instead
  (Section 4).

---

## 3. Trace a single qsp-sim run with adjusted params

Per `feedback_trace_before_proposing_fix.md`: run a forward sim before proposing
a parameter shift on analytical reasoning alone. The simplest path is to invoke
the `qsp_sim` binary directly with hand-edited param XMLs — no Python wrapping.

```text
Usage: qsp_sim --param <xml> [--csv-out <path>]
               [--scenario <scenario.yaml> --drug-metadata <drug_meta.yaml>]
               [--evolve-to-diagnosis <healthy_state.yaml>]
               [--t-end-days N] [--dt-days N]
               [--dump-state <path> | --initial-state <path>]
               [--params-hash <hex>]
```

### Pattern: parameter sweep over a single knob

```bash
TRACE=/tmp/<label>_trace
PRIMARY=/Users/joeleliason/Projects/pdac-build
mkdir -p "$TRACE"

# 1. Start from the canonical template (used by the SBI runner).
cp "$PRIMARY/resources/cpp/param_all.xml" "$TRACE/param_default.xml"

# 2. Make per-value copies, edit the knob inline.
for v in 0.025 0.050 0.075; do
  cp "$TRACE/param_default.xml" "$TRACE/param_${v}.xml"
  # XML edit — sed for one tag, or Python/xmllint for anything structural.
  python3 - <<PY
import re, pathlib
p = pathlib.Path("$TRACE/param_${v}.xml")
p.write_text(re.sub(r"<k_myCAF_prolif>.*?</k_myCAF_prolif>",
                    f"<k_myCAF_prolif>$v</k_myCAF_prolif>", p.read_text()))
PY
done

# 3. Run each (baseline scenario evolves to diagnosis first).
for v in 0.025 0.050 0.075; do
  "$PRIMARY/cpp/sim/build/qsp_sim" \
    --param         "$TRACE/param_${v}.xml" \
    --scenario      "$PRIMARY/scenarios/baseline_no_treatment.yaml" \
    --drug-metadata "$PRIMARY/resources/cpp/drug_metadata.yaml" \
    --evolve-to-diagnosis "$PRIMARY/resources/cpp/healthy_state.yaml" \
    --csv-out       "$TRACE/out_${v}.csv" \
    --t-end-days    1.0 --dt-days 0.5
done
```

Output is a CSV with one column per species; load in pandas and compute the
observable of interest. For repeated probes (e.g., k_myCAF_prolif sweep,
fs/ks combo grid), keep the trace XMLs around — you can re-run them after
forward-model edits to verify the candidate change actually closes the gap.

### Faster iteration: cache the post-evolve state

The baseline scenario spends most of its wall-clock evolving to diagnosis.
If you only care about the post-diagnosis dynamics, cache the evolve once
and re-use it across param sweeps:

```bash
"$PRIMARY/cpp/sim/build/qsp_sim" \
  --param "$TRACE/param_default.xml" \
  --evolve-to-diagnosis "$PRIMARY/resources/cpp/healthy_state.yaml" \
  --dump-state "$TRACE/diagnosis_state.qsth"

# Subsequent sweeps reuse the cached state.
"$PRIMARY/cpp/sim/build/qsp_sim" \
  --param "$TRACE/param_${v}.xml" \
  --scenario "$PRIMARY/scenarios/baseline_no_treatment.yaml" \
  --drug-metadata "$PRIMARY/resources/cpp/drug_metadata.yaml" \
  --initial-state "$TRACE/diagnosis_state.qsth" \
  --csv-out "$TRACE/out_${v}.csv"
```

The `--params-hash` flag stores a hex hash of the params at dump time and
verifies it on `--initial-state`, so swapping in a param XML that changes a
quantity used during evolve will trip a mismatch. Without `--params-hash`
the cache is reused silently — only valid if your sweep knob is a
post-evolve parameter.

### Schema validation: just run the test suite

After calibration target / submodel target YAML edits, the schema is checked
by the existing pytest suite — no need for a separate smoke script.

```bash
# Schema validation (Pydantic model_validate against model_structure.json):
.venv/bin/pytest tests/scenarios/test_calibration_targets.py

# Quick scope to one target while iterating:
.venv/bin/pytest tests/scenarios/test_calibration_targets.py -k "<target_stem>"
```

Use the qsp_sim binary trace pattern above for everything else (parameter
sensitivity, sanity checks, post-edit forward-sim verification).

### Gotcha: trace baseline ≠ SBI sampling baseline

`resources/cpp/param_all.xml` is populated from the CSV `value` column
(parameter defaults). SBI does *not* sample around those defaults — for any
parameter with an entry in `notes/calibration/submodel_priors.yaml`, the
workflow draws from the submodel posterior median (which can be shifted off
the CSV default) at the submodel posterior σ (which can be tighter or looser
than the CSV σ). If you want a trace that mirrors what SBI is actually
exploring, overlay submodel medians onto `param_all.xml` before sweeping:

```python
import yaml, pathlib, re
sm = yaml.safe_load(open("notes/calibration/submodel_priors.yaml"))
xml = pathlib.Path("resources/cpp/param_all.xml").read_text()
for entry in sm["parameters"]:
    name, med = entry["name"], entry["marginal"]["median"]
    xml = re.sub(rf"<{name}>.*?</{name}>", f"<{name}>{med}</{name}>", xml)
pathlib.Path("/tmp/<label>_trace/param_submodel_median.xml").write_text(xml)
```

Then sweep your knob from that XML, not from `param_all.xml`. A trace that
shows "k_X = 2.0 reaches the target" from the CSV-default baseline does
*not* imply SBI can reach the target — other parameters at submodel medians
may close the path that CSV defaults left open.

---

## 4. Refresh `submodel_priors.yaml`

### `scripts/regen_submodel_priors.py` — canonical debugging entry

Re-runs MCMC for any components containing the `--invalidate`d params (others
reuse their cache), then re-aggregates the joint posterior and rewrites
`notes/calibration/submodel_priors.yaml`. **Skips PPC and the Markdown
report** — that's what makes it usable in the tight debugging loop.

```bash
# Re-run MCMC for components touching APC0_cDC2_T (because its submodel
# target's forward model changed); refresh submodel_priors.yaml.
.venv/bin/python scripts/regen_submodel_priors.py --invalidate APC0_cDC2_T

# Multiple at once:
.venv/bin/python scripts/regen_submodel_priors.py --invalidate q_CD8_T_in k_Mac_rec

# Cache already fresh, just re-aggregate (e.g., after copying .compare_cache/
# into a new worktree, or after killing an audit mid-run):
.venv/bin/python scripts/regen_submodel_priors.py

# Full rebuild from scratch — wipe the cache and re-infer every component.
# Use after `rm -rf .compare_cache/` or when you suspect cached posteriors
# are stale (e.g., mixed sample counts, post-merge cache drift). Slow —
# ~minutes-to-hours depending on component count.
.venv/bin/python scripts/regen_submodel_priors.py --rebuild-all
```

Pass `--invalidate <param>` whenever a submodel target YAML for that param
changed (forward model, error model, inputs, or new target file). Otherwise
the cached posterior is stale and the regenerated YAML will lock in the
old result.

### Gotcha: `pdac_priors.csv` σ is shadowed for params with submodel entries

For any parameter that appears in `notes/calibration/submodel_priors.yaml`,
the SBI runner samples from the **submodel posterior** (its `marginal.mu`,
`marginal.sigma`) — the CSV `dist_param2` (σ) and `dist_param1` (μ) are
ignored entirely. Editing the CSV row changes `config_hash` (so the pool
key looks fresh) but `theta_train.pt` comes out bit-identical to the
pre-edit run. The smoking gun: identical md5 on `figures/.../artifacts/theta_train.pt`
across two runs whose CSV rows differ.

If you want to widen/shift the prior for a submodel-anchored parameter, the
options are:

- **Re-source the submodel target** (edit the YAML's `marginal` or its inputs)
  and `regen_submodel_priors.py --invalidate <param>` so the new posterior
  flows into `submodel_priors.yaml`.
- **Exclude the submodel target** (move to `excluded/`) so the CSV row takes
  effect again; only valid if the submodel evidence is genuinely double-counted
  by another observable (per `feedback_no_prior_tuning.md`, "different data
  sources" is *not* a reason to exclude).
- **Accept the constraint** and look for the structural fix elsewhere — if a
  realized-density target can't be reached at the submodel-anchored rate,
  another parameter (death rate, trafficking, recruitment) is the lever.

### `scripts/parameter_audit.py` — user-facing review tool

Same component-inference machinery as `regen_submodel_priors.py` plus
posterior-predictive checks, the Markdown audit report, and the Stage-2
cross-check section.

**This is a human-facing review tool, not part of the agent debugging loop.**
The agent should not invoke it during iterative work — `regen_submodel_priors.py`
covers everything inference-related that the loop needs. The audit report is
something the user reads when reviewing where calibration stands; the agent
producing it would just spend wall-clock without changing anything actionable.

```bash
# (User-driven only.)
.venv/bin/python scripts/parameter_audit.py --invalidate APC0_cDC2_T
```

### Before reading a misspec report, sanity-check for unit bugs

When a misspec audit flags a target with an extreme z-score (|z| > 10) or a
suspiciously round 1000×-ish ratio between `prior_med_raw` and the empirical
target, suspect a units bug in the YAML's `compute_observable` before
chasing model dynamics.

The 2026-05-11 collagen+TGFb bugs both came from `calibration_targets/*.yaml`
constants blocks: the runtime loader passes raw `c.value` to compute_observable
without honoring `c.units`, so any author who multiplied a `microgram/milligram`
or `kilogram/mole` value as if it were g/g or g/mol got a silent 1000× error.

Cheap sanity step before opening the misspec rabbit hole:

```bash
.venv/bin/pytest tests/test_calibration_target_smoke.py -k <target_slug>
```

The smoke plugs typical species values into each diagnosis-state target's
`compute_observable` and fails when the result is >2 OOM from the empirical
median. It catches the canonical 1000× class but cannot catch unit bugs that
are masked by compensating magnitude errors in the model output — for those,
manually check that the units multiplied in the code's math chain produce the
declared `observable.units`. The test docstring lists the limitation in full.

---

## 5. Kick off an SBI verification run from the worktree

After model/calibration edits, the standard 5k baseline prior-PC run.

```bash
cd "$WT"
TS=$(date +%Y%m%d_%H%M%S)
LABEL=<short-label>

nohup "$PRIMARY/.venv/bin/python" -m workflows.sbi_runner \
  --scenarios baseline_no_treatment \
  --cpp-branch "$NAME" \
  --num-total-simulations 5000 \
  > /tmp/sbi_run_${LABEL}_${TS}.log 2>&1 &

echo "PID=$!"
echo "Log: /tmp/sbi_run_${LABEL}_${TS}.log"
```

Use `python -m workflows.sbi_runner`, not `workflows/sbi_workflow.py`. The
latter is a thin backward-compat entry; `sbi_runner` is the canonical module
and exposes the full CLI (sweep flags like `--seed`, `--hidden-features`,
`--num-total-simulations`, etc.).

Common flags worth setting explicitly even when they match the default —
keeps the launch line self-documenting and makes the assumptions auditable
in the log:

- **`--scenarios <list>`** — comma-separated. Most debugging runs are
  baseline-only. Add `clinical_progression`, `gvax_neoadjuvant_zheng2022`,
  `gvax_nivo_neoadjuvant_zheng2022` for a full multi-scenario run. The pool
  cache key is keyed off this list.
- **`--cpp-branch <feature-branch>`** — explicit even when `cpp/` is
  unchanged; keeps HPC builds traceable to the branch under test.
- **`--with-submodel-sync` / `--skip-submodel-sync`** — composite prior
  (loads `submodel_priors.yaml` and composes per-component posteriors) is
  **on by default since 2026-05-04**. `--with-submodel-sync` is now a
  no-op kept for self-documenting launch lines. Pass `--skip-submodel-sync`
  to fall back to CSV-only independent normals — useful for smoke tests
  against deliberately wide priors that shouldn't be tightened by the
  static composite, or when iterating on stage-2 wiring.
- **`--no-posterior-predictive`** — section 7 (PPC sims + misspec verdict)
  is **on by default since 2026-05-04**. Pass `--no-posterior-predictive`
  for fast smoke checks where only training-time diagnostics are needed
  (saves ~1000 posterior-draw HPC sims). Note `--cache-processed-tensors`
  requires this flag since it's incompatible with PPC.
- **`--save-evolve-trajectories`** — opt-in. Adds a step 7f that
  produces per-scenario `cal_target_trajectories_<scen>.png` (burn-in)
  and `cal_target_trajectories_<scen>_post.png` (post-scenario, when
  `t_end_days > 7`) under the run dir. Each panel is a posterior
  trajectory band ending at the observed value's CI95 marker, with
  per-panel auto log-y, fraction-of-evolve x-axis for burn-in,
  absolute-time x-axis for post-scenario. Scalar-derived endpoints
  (e.g. `tumor_doubling_time` — one VDT per sim broadcast over time)
  are auto-detected and rendered as histograms instead of bands. Costs
  one extra round of fresh PPC sims (the suffix-pool cache is bypassed
  for trajectory-enabled calls; cached test-stat parquets have no
  trajectory data). Optional `--evolve-trajectory-dt-days N` controls
  burn-in dump cadence; default 14 d (biweekly). Worth running once
  per debugging cycle to eyeball whether posterior trajectories
  actually converge to observations — see
  `workflows/calibration_trajectory_pipeline.py` for the assembler →
  evaluator → plotter chain.
- **`--num-total-simulations N`** — default 5000. Set lower (e.g. 500)
  for fast smoke checks of the pipeline / new observable formulas; full
  prior-PC verification needs the full 5k.
- **Restriction classifier is on by default** — the theta pool is
  rejection-sampled against `tests/fixtures/restriction_classifier/`
  (the committed structural-validity classifier). Don't disable this
  unless you have a reason; the unrestricted pool produces a high NaN
  rate. Override with `--restriction-classifier-dir <other-dir>` if
  you've trained a new classifier (via `workflows/restrict_prior_train.py`),
  or pass `--no-restriction-classifier` to opt out for a sanity check.
- **`--tau-cohort <float>`** — cohort random-effect SD in copula-Gaussian
  space (default 0.0 = off). Adds a per-cohort additive shift at training
  time so the likelihood absorbs inter-cohort heterogeneity (e.g.,
  Liudahl vs Golesworthy systematic offsets in PDAC TIL/Treg counts) as
  nuisance noise rather than forcing a single θ to compromise. The
  shift is post-load (cohort_id = `primary_data_source.source_tag`), so
  the existing pool is reused for any τ — no re-simulation. See
  `notes/calibration/cohort_random_effect_sweep_2026-04-30.md` for
  empirical guidance:
    • **τ=0.5** for default inference — minimum top-pair |φ|, k_eff
      already at 11/22, posterior usefully wide.
    • **τ=1.0** for *diagnostic mode* — surfaces structural-vs-cohort
      decomposition, e.g. swaps Cluster B (`cdc1×th_exh`) into the top
      pairwise tension as Cluster A's cohort component is absorbed.
      Posterior too diffuse to ship; use only to rank structural fixes.
    • τ=0.0 ships the historical "cohort-blind" fit — sims are then
      forced to compromise across genuinely-different patient cohorts.

Other flags exist (`--seed`, `--cache-processed-tensors`, `--prcc-tier`,
NSF architecture knobs, `--pretrained-dir` / `--sample-on-delta` for Delta
offload) but aren't part of the day-to-day debugging loop. See
`python -m workflows.sbi_runner --help` if you need them.

Notes:

- **cwd = worktree** so the local Python reads the worktree's calibration target
  YAMLs, the worktree's `submodel_priors.yaml`, and the priors CSV.
- **`--cpp-branch <feature-branch>`** is explicit even when the C++ source is
  unchanged — keeps HPC builds traceable to the branch under test.
- **5000 sims is the default** (`RunConfig.num_total_simulations`); add
  `--num-total-simulations N` to override.
- **Pool `config_hash` flips** when `submodel_priors.yaml` or `pdac_priors.csv`
  changes — that means a fresh HPC pool, full 5k re-simulated. (Per CLAUDE.md.)
- **`test_stats_hash` flips** when calibration target observables change —
  cheap recompute against the existing pool when only formulas changed.
- Multi-scenario: `--scenarios baseline_no_treatment,gvax_neoadjuvant_zheng2022,...`.
- **The local C++ binary is required** even when sims run on HPC — the runner
  probes it on init. The symlink from step 1a covers this.

What to watch for in the log (once HPC returns):

```
3. PRIOR PREDICTIVE CHECK
  Per-observable prior predictive coverage:
    OUTSIDE: <obs>: obs=..., 90%CI=[...]

  Joint reachability (prior, sim within target empirical CI95):
    Per-sample count: median=12/22, max=20, p95=15
    Least-jointly-reachable targets (bottom 10):
       4.27% (...) stromal_fraction
      10.71% (...) cd8_density_baseline
      ...

  Pairwise tension (prior, top 15 by negative phi, both marginals ≥ 5%):
      phi    P_i    P_j   P_ij  target_i / target_j
    -0.528  37.6%  59.4%   9.8%  cdc1_density_baseline
                                    th_exhausted_fraction
```

**Joint reachability is the primary debugging signal.** Per-observable
`OUTSIDE` flags only tell you whether the prior median is well-centered;
the actual structural problem is when the *intersection* of CI95s across
multiple targets is empty. Read the log in this order:

1. **Per-sample reachability count** (`median=N/M`) — how many of the M
   targets a typical prior sample satisfies. If median is well below M,
   no theta in the prior can satisfy all targets simultaneously and
   the model has structural tension somewhere.
2. **Bottom-10 least-jointly-reachable targets** — these are the
   bottlenecks. < 5% reachability means the obs is unreachable under
   any prior draw → structural fix needed (forward-model bug, missing
   mechanism, or genuinely incompatible target). 10–30% means the
   posterior can pull but the prior is poorly placed.
3. **Pairwise tension table** — the smoking gun for *why* a target is
   bottom-10. Negative `phi` with `P_ij ≪ P_i × P_j` means the two
   targets are mutually exclusive under the prior — sims that satisfy
   one rarely satisfy the other. This is where you discover that
   `cdc1_density` and `th_exhausted_fraction` are fighting over the
   same parameters, or that `stromal_fraction` and a CAF-density
   target can't both be high. Fixing one of the priors / formulas
   without addressing the conflict won't move joint reachability.

Per `feedback_reachability_not_prior_coverage.md`: prior-PC misses are
OK if the obs is reachable in the tail (posterior pulls). Real failures
are obs unreachable + tension with another target — that's the loop you
want to close.

Compare these three sections to the pre-change baseline log to attribute
movement to your specific edit.

---

## 6. Null-shuffle + skip-sync diagnostics — separating real signal from prior-copula transfer

Sometimes the posterior of a parameter shifts or contracts when there's no
mechanistic reason for it to. The 2026-05-01 GMCSF investigation tracked
this down to a specific failure mode in the composite prior, not in NPE
itself.

`EC50_GMCSF` is structurally absent from the baseline likelihood — it
only enters via a Hill function on `V_ID.GMCSF`, which is identically zero
when no GVAX is dosed. Its posterior should equal the prior at baseline.
But three observations suggested otherwise:

- With `--with-submodel-sync`: posterior shifted **+0.78σ** rightward,
  σ tightened 40%.
- Posterior correlations to `rho_Treg_max`, `k_IL2_sec`, `k_IL12_sec`,
  `k_APC_mat_cDC1`, `n_CD8_clones`, etc. all > 0.9.
- A null-shuffle run (training data permuted, no likelihood signal at
  all) reproduced the **same > 0.9 correlation structure** — confirming
  the correlations were not from the data.
- A `--skip-submodel-sync` run (CSV-only prior, no composite copula)
  collapsed shift to **+0.07σ**, σ ratio to 0.97, top correlation to
  0.11. Recovery of the prior, almost exactly as predicted.

The mechanism is **prior copula transfer**, not NPE flow artifact:

1. The composite prior in `submodel_priors.yaml` has a Gaussian copula
   block. When inspected, EC50_GMCSF's row had 38 entries with |r| > 0.3,
   and several at |r| > 0.99 (rho_Treg_max 0.999, k_IL2_sec 0.998, etc.).
2. Baseline data identifies the partners (rho_Treg_max etc. via real
   likelihood channels), and the near-deterministic copula correlation
   then drags EC50_GMCSF's posterior along. By prior decree, not by data.
3. The 0.99 correlations themselves are an upstream bug in
   `qsp-inference` (`audit/report.py:_write_submodel_priors`). The audit
   buckets per-component posterior samples by `len(samples)` as a proxy
   for component membership and fits a Gaussian copula to row-aligned
   columns. When components share RNG state across NPE samplings, row
   alignment becomes comonotonic coupling and the recovered copula goes
   to r → 1.

This is a literature-distinct failure mode from the standard NPE
overconfidence pattern (Hermans et al. TMLR 2022) — it's a *prior*
problem, not a posterior approximation problem. NPE is faithfully
propagating bad prior structure.

Two diagnostics now in the toolkit:

**(a) `--skip-submodel-sync` confirmation run.** The single most informative
test for "is this posterior shift driven by data or by the composite prior?"
Drops `submodel_priors.yaml` from the pool key (so it triggers a fresh 5k
sim — not free, but cheap relative to the diagnostic value). Trains NPE
with the CSV-only independent-Normal prior, no copula. If a param's shift
collapses to ≈ 0 under skip-sync, the apparent signal was prior-copula
transfer, not data. **This is the canonical first test.**

**(b) `--null-shuffle` companion run.** Same pipeline, same hyperparameters,
same training data, except `x_train` rows are permuted relative to
`theta_train` just before `NPE.append_simulations(...)`. By construction
the trained NPE has zero likelihood signal. Useful as a secondary test
that isolates *NPE flow artifact* from prior-copula transfer when run
under `--skip-submodel-sync` (no copula in either run). Per-param
`contraction_local` from the null run is the architectural artifact floor
at this N / hyperparameters; subtract from real to estimate genuine
signal contraction.

(Note: running `--null-shuffle` *with* `--with-submodel-sync` does NOT
isolate NPE flow artifact, because the composite copula prior is still in
play and will produce non-trivial shifts even under shuffled data. If you
want to characterize NPE bias specifically, pair `--null-shuffle` with
`--skip-submodel-sync`.)

```bash
cd "$WT"
TS=$(date +%Y%m%d_%H%M%S)
LABEL=<short-label>

# Pair with the real run from §5 — same flags, just append --null-shuffle.
nohup "$PRIMARY/.venv/bin/python" -m workflows.sbi_runner \
  --scenarios baseline_no_treatment \
  --cpp-branch "$NAME" \
  --with-submodel-sync \
  --num-total-simulations 5000 \
  --null-shuffle \
  > /tmp/sbi_run_${LABEL}_null_${TS}.log 2>&1 &
```

`--null-shuffle` auto-suffixes the save dir with `_null` so the real run
isn't clobbered (so for the example above outputs land in
`figures/sbi_baseline_no_treatment_null/`). The simulator pool key is
unchanged — null-shuffle reuses the existing pool, no re-simulation. Only
the post-load `x_train` tensor is permuted. Cohort-RE (if `--tau-cohort >
0`) is applied per-scenario *before* the shuffle, so cohort effects are
also decorrelated. Posterior is still evaluated at the real `x_obs` —
that's correct: you want to see what a trained-on-noise NPE does at the
real observation point.

Compare the two `local_calibration.csv` files:

```python
import pandas as pd
real = pd.read_csv('figures/sbi_baseline_no_treatment/local_calibration.csv').set_index('parameter')
null = pd.read_csv('figures/sbi_baseline_no_treatment_null/local_calibration.csv').set_index('parameter')
df = real[['contraction_local']].join(null[['contraction_local']], rsuffix='_null')
df['contraction_real'] = df['contraction_local'] - df['contraction_local_null']
df.sort_values('contraction_real', ascending=False)
```

Interpretation:

- `contraction_real ≳ 0.2` — likely genuine identification.
- `contraction_real ≈ 0` (real ≈ null) — the apparent contraction is a
  per-architecture artifact, not signal. Don't tighten the prior or
  interpret the marginal shift; the parameter is structurally unidentified
  by the data you have.
- `contraction_local_null > 0.1` for any param flags that the architecture
  has a meaningful artifact floor — useful as a sanity check on the run
  as a whole. At 5k sims this is normal; at higher N it should shrink.

Companion check, no extra training: posterior correlations to other
identified params. Any param with high `contraction_local` *and*
`max |corr to identified param| > 0.7` is a phantom-contraction
candidate, and the null-shuffle run is the cheapest way to confirm.

When to run these diagnostics:

- Marginal posterior shift on a param that shouldn't be touchable by the
  scenario (e.g., therapy param in baseline-only). Run `--skip-submodel-sync`
  first. If shift collapses, it's copula transfer.
- Suspiciously high posterior correlations between an unidentified param
  and an identified one. Same — `--skip-submodel-sync` is the cheapest
  isolation test.
- Sanity check before publishing or making a parameter-update decision
  off a marginal.

When *not* to bother:
- Multi-scenario fits where the param has a real likelihood channel from
  at least one scenario, *and* its top copula partners are < 0.5. Real
  signal will dominate.
- After you've already confirmed via `--skip-submodel-sync` that the
  shift is data-driven.

Cost:
- `--skip-submodel-sync`: triggers 5k fresh sims on HPC (different pool
  key). ~15-30 min depending on queue.
- `--null-shuffle`: reuses existing pool, no new sims. One extra training
  run, ~5-10 min local.

Use surgically, not as a default companion to every run.

### Composite prior copula health check

Before trusting any param's posterior under `--with-submodel-sync`, check
its row in the copula:

```python
import yaml, numpy as np
d = yaml.safe_load(open('notes/calibration/submodel_priors.yaml'))
cop = d['copula']
params = cop['parameters']
corr = np.array(cop['correlation'])
i = params.index('YOUR_PARAM')
row = np.abs(corr[i])
big = [(params[j], corr[i,j]) for j in np.argsort(-row)[1:11] if abs(corr[i,j]) > 0.5]
for p, r in big: print(f'  {p:30s}  {r:+.3f}')
```

If you see entries above |r| ≈ 0.95 to params that aren't biologically
related (e.g., GVAX param vs T cell rate), assume it's the audit's
length-bucket bug (`qsp-inference` `audit/report.py:_write_submodel_priors`)
and treat any posterior shift on this param as suspect until verified
under `--skip-submodel-sync`.

---

## Pitfalls / things we keep re-learning

- **Don't tune priors to match data.** See `feedback_no_prior_tuning.md`.
  When the model can't match a target, the legitimate responses are:
  1. **Add a submodel/calibration target** that directly constrains the
     parameter(s) the offending observable is sensitive to (so the prior
     gets pulled by independent literature, not the target you can't fit).
  2. **Fix or exclude an existing target** that's pointing the wrong way —
     wrong-indication source, wrong forward model, wrong unit conversion,
     misread snippet. Move it to `excluded/` with a README note.
  3. **Fix the QSP model** — forward-model bug, missing mechanism, or
     a structural assumption that breaks under this scenario.
  4. **Widen σ on the prior** if the literature genuinely allows the
     range and the current σ is too tight.
  What's *not* on this list: shifting μ to observed values. That replaces
  Bayesian update with hand-fitting and silently locks in whatever bias
  the target had.
- **Reachability ≠ prior PC.** Per `feedback_reachability_not_prior_coverage.md`,
  prior-PC misses are OK if reachable in the tail. Real failures are when
  observed data is unreachable under any prior draw.
- **Trace before recommending a fix.** Per `feedback_trace_before_proposing_fix.md`.
  Especially for analytical SS arguments — sim it first.
- **Worktree binary symlink invalidates pool caches if you rebuild.** The
  pool key folds in binary bytes; rebuilding `qsp_sim` in `cpp/sim/build/`
  orphans every pool that depended on the old binary across all worktrees.
  See issue #107: this also means each fresh worktree gets a different
  binary (path-cascade through `__FILE__`-based string section deltas)
  and forces a fresh 5k pool, even on identical source. Tax to be aware
  of, not a bug to chase from inside a debugging loop.
- **`--tau-cohort` does NOT change the prior-PC pairwise φ table.** That
  table reads from raw simulator outputs (joint reachability of empirical
  CI95s), which is a property of prior+simulator+target CIs — none of
  which τ_cohort touches. Cohort-RE acts on the *likelihood* (post-copula
  training space). To see the cohort-RE effect on tensions, read the
  **posterior** pairwise tension table from section 7 (PPC is on by
  default; only opt out with `--no-posterior-predictive` if you need a
  fast smoke check) and the LOO misfit-driver influences. Don't expect
  the prior-PC tension table to move when sweeping τ — if it did, the
  prototype would be doing the wrong thing.
- **`.compare_cache/` is gitignored** and not copied by `git worktree add` —
  copy it manually (step 1b) or accept that the next audit will re-run all
  components.
