# Delta GPU pipeline for NPE (train + analyze) — 2026-04-19

Everything from NPE training through posterior analysis runs on Delta. Only summaries + plots come back to the laptop.

## Components

Under `pdac-build/workflows/delta/`:

| file | role |
|---|---|
| `train_npe_delta.py` | loads `theta_train.pt`/`x_train.pt`/`npe_config.json`, trains NSF, writes `density_estimator.pt` + `learning_curve.csv` + `metrics.json` |
| `train_npe.sbatch` | `gpuA40x4-interactive`, 1 GPU, 30 min |
| `run_delta_training.sh [tag]` | upload artifacts → sbatch → poll → rsync into `figures/sbi_multi/delta_out/<tag>/` |
| `analyze_npe_delta.py` | samples `(200, n_test)` + `(2000, 1)` chunked, runs `sbi_recovery`, `sbi_z_score_contraction`, `sbi_calibration_ecdf` (global + local), posterior-vs-prior plot, writes summaries |
| `analyze_npe.sbatch` | same partition, 45 min |
| `run_delta_analysis.sh <stage_dir> [tag]` | upload bundle → sbatch → poll → rsync output into `<stage_dir>/output/` |

Integrated into `pdac-build/workflows/sbi_runner.py`:

- `--pretrained-dir <dir>` — skips local training, loads `density_estimator.pt` + `learning_curve.csv` + `metrics.json`.
- `--sample-on-delta` — requires `--pretrained-dir`. Offloads **all of steps 5/5b/6** to Delta via `_analyze_on_delta`. Downloads only `recovery.png`, `sbc_ecdf_{global,local}.png`, `posterior_vs_prior.png`, `local_calibration.csv`, `posterior_samples.csv`, `posterior_summary.csv`, `run_metrics.json`. Total ~15 MB.

## One-time Delta setup

- ACCESS Delta account, Kerberos + Duo 2FA (no SSH keys accepted).
- `~/.ssh/config` Host `delta` with ControlMaster/ControlPersist 24h (already set).
- `~/sbi-env` = venv created with `--system-site-packages` on top of `pytorch-conda/2.8`, with `sbi`, `qsp-inference`, `maple` installed.
- Since `qsp-inference` GitHub repo is private and the Delta login node has no SSH-key for GitHub, install from rsync'd source:
  ```
  rsync -az --exclude .git --exclude .venv --exclude build --exclude __pycache__ \
     ~/Projects/maple/          delta:maple/
  rsync -az --exclude .git --exclude .venv --exclude build --exclude __pycache__ \
     ~/Projects/qsp-inference/  delta:qsp-inference/
  ssh delta '~/sbi-env/bin/pip install ./maple && ~/sbi-env/bin/pip install --no-deps ./qsp-inference'
  ```
  (`--no-deps` on the second install dodges the pinned git-URL conflict in `qsp-inference/pyproject.toml`; `./maple` already satisfied.)

## End-to-end run (single config)

1. **Stage training tensors** — first run of `sbi_runner.py` writes `figures/sbi_multi/artifacts/{theta_train.pt, x_train.pt, npe_config.json}` in step 4. If you already have those from a prior run, skip this.

2. **Train on Delta**
   ```
   cd ~/Projects/pdac-build
   workflows/delta/run_delta_training.sh my_run_tag
   # downloads -> figures/sbi_multi/delta_out/my_run_tag/{density_estimator.pt, learning_curve.csv, metrics.json}
   ```

3. **Run full analysis** (reloads cached sims, skips training, offloads 5/5b/6)
   ```
   .venv/bin/python -u workflows/sbi_runner.py \
     --cpp-jobs-per-chunk 400 --num-total-simulations 100000 \
     --cpp-branch cpp-sweep-binary-io --tools-branch cpp-backend-v1 \
     --pretrained-dir figures/sbi_multi/delta_out/my_run_tag \
     --sample-on-delta
   ```

## Grid of training settings

The knobs in `npe_config.json`: `hidden_features`, `num_transforms`, `num_bins`, `batch_size`, `max_epochs`. Today they live in `figures/sbi_multi/artifacts/npe_config.json`, which is what `train_npe_delta.py` reads.

Minimal sweep loop (write yourself — not in the tree today):

```bash
# sweep.sh — laptop side
for cfg in configs/grid_*.json; do
  tag="$(basename "$cfg" .json)"
  # swap in this config
  cp "$cfg" figures/sbi_multi/artifacts/npe_config.json
  # train (blocks ~4-8 min on A40)
  workflows/delta/run_delta_training.sh "$tag"
  # full analysis (another ~4 min: rehydrate sims + 1 min Delta round-trip)
  .venv/bin/python -u workflows/sbi_runner.py \
    --cpp-jobs-per-chunk 400 --num-total-simulations 100000 \
    --cpp-branch cpp-sweep-binary-io --tools-branch cpp-backend-v1 \
    --pretrained-dir figures/sbi_multi/delta_out/$tag --sample-on-delta \
    --save-dir figures/sbi_multi/runs/$tag
done
# collate
jq -s 'map({tag: input_filename, median_r2_global, median_contraction_local, n_miscal_local})' \
   figures/sbi_multi/runs/*/run_metrics.json > sweep_summary.json
```

Known friction for a real sweep, fix before running many configs:

- **Duo re-auth** every time the macOS ControlMaster socket dies (lid close, network flap). Either (a) jump-host with ControlPersist on an always-on box, or (b) run the entire sweep inside one interactive Delta session with tmux. Without this, every job resume prompts Duo.
- **Sim rehydration** (~3-4 min) happens on every `sbi_runner.py` invocation. If we cached the post–copula `theta_{train,test}.pt`/`x_{train,test}.pt` and shortcut past step 2 when `--pretrained-dir` is given, the analysis pass could drop to ~1 min per config. Not built yet.
- **Parallelism**: today each config is a separate sbatch job → no reason they can't run concurrently. Easy upgrade: fire all `run_delta_training.sh` jobs in the background in a loop (each prints its JID), poll a list, then analyze. But watch Delta's queue concurrency limits and the gpuA40x4-interactive partition size.
- `npe_config.json` is under `figures/sbi_multi/artifacts/` — which is shared by `sbi_runner.py`'s own step-4 write. A sweep that overwrites it races with any other local run. Put grid configs in a dedicated dir (`figures/sbi_multi/sweep/configs/`) and copy into place per iteration.

## What the last run produced (2026-04-19)

100k sims, bs=2048, 52 epochs, best val 174.70. Median R² ~0, median contraction_local 0.04. Top-8 identified (contraction_local > 0.5): `k_M2_pol`, `APC0_cDC2_T`, `k_cell_clear`, `k_C_CD8_exh`, `f_stroma_max`, `k_Mac_rec`, `APC0_cDC1_T`, `k_Th_pro_tumor`. 20 params > 0.3, 41 > 0.1. Artifacts at `pdac-build/figures/sbi_multi/`.

---

## NPE training-settings sweep — implemented 2026-04-19

Lives in `workflows/sweep/`:

| file | role |
|---|---|
| `grid.yaml` | 3-axis spec: `hidden_features` × `num_transforms` × `n_train_subsample`. 12 cells total. |
| `generate_configs.py` | Expands `grid.yaml` → `configs/{h{hf}_t{nt}_s{sub}}.json` + `configs/_tags.txt`. |
| `run_sweep.sh` | Full driver: serial stage → parallel train → parallel analyze → collate. |
| `collate_sweep.py` | Reads per-cell `local_calibration.csv` + `run_metrics.json` → `sweep_by_param.csv` (long form) + `sweep_summary.csv` (per-cell aggregates, gain/loss vs reference cell). |

Three new `sbi_runner.py` flags built for the sweep:

- `--cache-processed-tensors` — caches post-step-2 state (train/test tensors, copula quantiles, target_df, …) to `cache/sbi_processed/{key}.pkl`. Key = `{model_version}__{scenarios}__tier{n}__sub{N}__split{n}__s{n}`. Second hit = ~10s instead of 3–4 min. **Incompatible with** `--run-posterior-predictive` or `misspec_detection_mode` (scenario_meta['simulator'] isn't cached).
- `--stage-artifacts-only` — runs through step 2 + artifact writing, then exits. Used by the sweep driver so Delta does the actual training.
- (Already existed) `--pretrained-dir`, `--sample-on-delta`.

Artifact-writing moved to end of step 2 (was after training) so `--stage-artifacts-only` can skip everything after.

Per-param metrics in `local_calibration.csv` now include `zscore_mean_local`, `zscore_std_local`, `coverage95_local`, `post_sd_log_local` — from `qsp_inference.inference.compute_per_param_calibration` (added on `qsp-inference` branch `npe/per-param-calibration`, 4 pytest cases).

`run_delta_training.sh` now honors `ARTIFACTS` / `LOCAL_OUT` env vars so parallel cells don't race on `figures/sbi_multi/artifacts/`.

### Running the sweep

1. Merge `qsp-inference` `npe/per-param-calibration` → main and reinstall locally + rsync + reinstall on Delta.
2. `workflows/sweep/generate_configs.py` (already run — regenerate if grid changes).
3. `PY=/path/to/venv/bin/python bash workflows/sweep/run_sweep.sh`
4. Outputs in `figures/sbi_multi/sweep/`:
   - `runs/{tag}/` per cell (local_calibration.csv, recovery.png, posterior_samples.csv, …)
   - `delta_out/{tag}/` per cell (density_estimator.pt, learning_curve.csv, metrics.json)
   - `logs/train_{tag}.log`, `logs/analyze_{tag}.log`
   - `sweep_by_param.csv`, `sweep_summary.csv` (top-level)

Stage is serial (shares the processed-tensor cache — 2 unique keys across 12 cells, first hit ~4 min, rest cache-free). Train + analyze run 12 concurrent `run_delta_training.sh` / `sbi_runner` invocations; SLURM queues them; ControlMaster keeps SSH warm.

### Collation primary columns (`sweep_by_param.csv`)

`cell_tag, hidden_features, num_transforms, n_train_subsample, parameter, r2_global, r2_local, contraction_global, contraction_local, ks_pval_global, ks_pval_local, zscore_mean_local, zscore_std_local, coverage95_local, post_sd_log_local, is_miscal_local, is_identified_local`

Per-cell summary carries `n_identified_local`, `n_calibrated_local`, plus `n_newly_identified_vs_ref` / `n_lost_identified_vs_ref` / `n_newly_calibrated_vs_ref` / `n_broke_calibration_vs_ref` / `n_contraction_gain_0p1_vs_ref` computed against `ref_tag` (default: first row of `_tags.txt`, i.e. smallest cell).

---

## Sweep 1 completed (2026-04-20)

24 cells: `{128,256,512} × {5,9} × {10k,25k,50k,100k}`. Bugs hit + fixed during the run: multi-line `cmd=` in `run_sweep.sh` was eating CLI args (→ sbi_runner used defaults, 5000 sims); MaxSessions (~10) saturated on first 24-parallel fan-out (16 cells failed `Session open refused by peer`). Fixed both: driver uses bash arrays + `MAX_PARALLEL=6` throttle + `ONLY_TAGS` filter for retrying failures.

**Findings (24/24 collated in `figures/sbi_multi/sweep/sweep_summary.csv`):**

| sub | identified (mean, range) | calibrated | val_loss |
|---|---|---|---|
| 10k | 0.7 (0–2) | 142 | 194 |
| 25k | 5.0 (3–6) | 148 | 182 |
| 50k | 8.3 (7–9) | 153 | 173 |
| 100k | 8.3 (7–9) | 153 | 173 |

Data size dominates. 50k ≈ 100k because the joint-NaN-filter cap is ~48,880 survivors (54% of 100k). Capacity axes (width/depth) barely move the needle; high-capacity + low-data combinations OVERFIT (0 identified). **9 params total ever cross `contraction_local > 0.5`**, at most. Bottleneck is theta yield / parameter degeneracy, not network size.

Identified-when-enough-data: `k_M2_pol`, `k_cell_clear`, `k_C_CD8_exh`, `f_stroma_max`, `k_Mac_rec`, `APC0_cDC2_T`, `k_Th_pro_tumor`, `APC0_cDC1_T`, `k_MMP_M2`.

---

## Classifier-restricted prior (merged 2026-04-20, launch blocked)

Built to address the "yield ceiling" — 46% of prior draws fail `evolve_to_diagnosis`, so 100k sims → only 54k usable. Train a classifier on `(theta, is_valid_joint_across_scenarios)` from the existing pool, reject low-`p(valid)` thetas before simulating.

### Components (merged to main)

- **`qsp-inference`** — `compute_per_param_calibration` (PR #12), `RestrictionClassifier` + `train_restriction_classifier` + `sample_restricted` (PR #13). `scikit-learn` promoted to core dep.
- **`qsp-hpc-tools`** — PR #44 (theta_pool.py rejection-sampling against a classifier) + PR #45 (fix `_compute_config_hash` to include classifier bytes + threshold so restricted/unrestricted pools get distinct on-disk dirs).
- **`pdac-build`** (on `sweep/npe-training-grid` branch) — `workflows/restrict_prior_train.py` writes `cache/restriction_classifier/{classifier.pkl,metadata.json}`; `sbi_runner.py` CLI gains `--restriction-classifier-dir` / `--restriction-threshold`; passes `theta_pool_size=max(N_total,100k)` through to CppSimulator to avoid `IndexError` when `N_total > 100_000`. Also: parallel scenario fan-out in step 2 — scenario 1 serial (populates evolve cache), scenarios 2–N parallel via ThreadPoolExecutor.

### Classifier quality (CV AUC 0.989)

At τ=0.5: accept 54.8% of prior, 94.6% of accepted survive → 0.52 usable per prior draw (vs 0.54 baseline). At τ=0.8: accept 45%, 99% survive.

Needed sims for 50k usable: baseline 92k → τ=0.5 53k (**43% fewer sims**). Top drivers of failure: `k_stroma_support`, `f_stroma_max`, `k_C1_death`, `k_C1_growth`, `k_vas_growth`, `k_vas_decay`. Expected — all are evolve-to-diagnosis gating.

### Launch attempt died to lid-close — PICK UP HERE

Ran the first launch at 2026-04-20 02:45. Local theta-pool generation succeeded (110k × 177 accepted thetas at `cache/sbi_simulations/theta_pools/theta_pool_96eb89c0ae22759b_n110000_restricted.npy`). HPC pool dirs created with restricted config hashes: `d986ecc8`, `3e28df15`, `112b4dbb`, `2c632a12`. qsp_sim binary rebuilt on Rockfish at 02:46:17. Then user closed the laptop lid — ControlMaster to `hpc` died, subsequent SSH calls hung waiting for Duo re-auth, sbatch never submitted. **Not a bug**, just the laptop going to sleep.

The stuck process was killed, all prepared artifacts are intact. Relaunch should pick up instantly because the theta pool + HPC scenarios dirs already exist.

### Step-by-step relaunch

```
# 0. Sanity: ControlMaster to Rockfish alive + keyed, Duo push accepted
ssh hpc "squeue -u \$USER -o '%.10i %.14j %.8T %.10M' | head -5"

# 1. (Already done this session) Train classifier — skip unless retraining
#    python workflows/restrict_prior_train.py
#    → cache/restriction_classifier/{classifier.pkl,metadata.json}, CV AUC 0.989

# 2. LAUNCH the restricted 110k sim job. Use the pdac-build primary venv;
#    this worktree (sweep/npe-training-grid) has no .venv of its own.
#    Run from repo root (pdac-build-npe-sweep/).
/Users/joeleliason/Projects/pdac-build/.venv/bin/python -u workflows/sbi_runner.py \
  --cpp-jobs-per-chunk 400 --num-total-simulations 110000 \
  --cpp-branch cpp-sweep-binary-io \
  --restriction-classifier-dir cache/restriction_classifier \
  --restriction-threshold 0.5 \
  --cache-processed-tensors --stage-artifacts-only \
  --save-dir figures/sbi_multi/sweep/restricted_stage \
  2>&1 | tee figures/sbi_multi/sweep/restricted_stage/sim.log
```

Run with `run_in_background: true` + a Monitor on the log filtering for `^  --- Scenario|pool id:|HPC request|No pre-derived|submitting fresh|job_id|Submitted|Joint NaN filter|Saved processed bundle|STAGE-ARTIFACTS|Traceback|refused by peer`.

Expected flow (no changes needed to any code):

1. Classifier-restricted theta pool loads from disk cache (instant).
2. **Scenario 1 (`baseline_no_treatment`) runs sbatch synchronously** on Rockfish (~275 array tasks × ~400 sims each → 110k sims total). Populates the `evolve_cache` on HPC as it runs.
3. **Scenarios 2-4 (`clinical_progression`, `gvax_neoadjuvant_zheng2022`, `gvax_nivo_neoadjuvant_zheng2022`) launch in parallel** via `ThreadPoolExecutor` after scenario 1 completes. They submit their own sbatch arrays concurrently. Should hit the warm evolve cache so each per-sim is ~20× faster.
4. After all 4 scenarios download: joint NaN filter → ~100k survivors.
5. Processed bundle written to `cache/sbi_processed/..._restr2adbb9b075.pkl` (restriction-classifier-keyed — fixed 2026-04-20; see `_processed_cache_path` in `sbi_runner.py`).
6. Staged artifacts written to `figures/sbi_multi/sweep/restricted_stage/artifacts/`.
7. Script exits (stage-artifacts-only).

Wall time estimate: ~2-4 hours for the HPC sims + ~5 min for download + processing. Dominated by scenario 1.

### After the sims finish — sweep against the restricted pool

The existing `run_sweep.sh` driver doesn't know about the classifier. Two options:

- **Option A (quick)**: add `--restriction-classifier-dir cache/restriction_classifier --restriction-threshold 0.5` to both sbi_runner invocations in `workflows/sweep/run_sweep.sh` (the stage call at L56 and the analyze call at L127). Regenerate `workflows/sweep/configs/*.json` if the grid needs updating, otherwise reuse the 24-cell grid. Run with throttle: `MAX_PARALLEL=6 PY=/…/.venv/bin/python bash workflows/sweep/run_sweep.sh`.
- **Option B (cleaner)**: add a `restriction_classifier_dir` field to `grid.yaml`, have `generate_configs.py` emit it into the per-cell JSONs, and have `run_sweep.sh` read it out. ~20 line change across three files, but the grid spec becomes self-describing.

Either way, after the sweep completes, `sweep_by_param.csv` / `sweep_summary.csv` will show the identified-param distribution at the new effective training size of ~95k (up from 48k). If the 9-param ceiling rises, parameter identifiability really was data-starved. If it doesn't rise, the true bottleneck is observation degeneracy (same obs signature → different thetas) and next move is more / better observables or sequential SNPE.

### Known open issues / future work

- **Carry-over from unrestricted pool** — we decided against (the 54k existing valid sims are "wasted" by doing a fresh 100k restricted pool). If HPC cost matters, revisit Option 2 from the session (multi-pool support in sbi_runner) — ~60-80 lines in step 2 to concat two pools per scenario.
- **sbi-based truncated SNPE** — sequential rounds + restriction is the principled next step once we see what the single-round restricted gains. `workflows/sequential_sbi_workflow.py` is the existing skeleton.

### File map

- `cache/restriction_classifier/{classifier.pkl,metadata.json}` — trained classifier, CV AUC 0.989.
- `cache/sbi_simulations/theta_pools/theta_pool_96eb89c0ae22759b_n110000_restricted.npy` — rejection-sampled 110k pool ready for upload.
- `cache/sbi_simulations/multi_scenario_v4_all_scenarios_{d986ecc8,3e28df15,112b4dbb,2c632a12}_*` — empty pool dirs awaiting sim output.
- `workflows/restrict_prior_train.py` — retrains the classifier from the unrestricted pool's parquets. Run once per model_version or priors update.
- `workflows/restrict_prior_experiment.py` — standalone yield-diagnosis (threshold vs accept_frac vs survival table).
- `figures/sbi_multi/sweep/restricted_stage/sim.log` — 112-line log of the hung launch, useful for diffing against the next attempt.

---

## 2026-04-20 PM — concurrency fixes landed + 120k sim remainder

Today's session unblocked the restricted 110k run and fixed four qsp-hpc-tools bugs that the scenario fan-out exposed. We're left with **~120k HPC sim work** across two scenarios, all serialized. Next session: just relaunch.

### qsp-hpc-tools PRs merged today (all on main)

| # | title | what it does |
|---|---|---|
| 46 | bump derive SLURM time limit 15m → 45m | 110k-pool derive was timing out at the 15m wall |
| 47 | derive: hoist Pint unit parsing + swap loop order | ~10× speedup on derive (~23 min → ~5 min for 110k). Plan phase pre-parses units once per batch, pre-wraps template_defaults, materialises sim_df cols as numpy arrays. Sim outer, test-stat inner. |
| 49 | pool-scope HPC upload paths + flock ensure_cpp_binary | `batch_jobs/input/<pool_id>/*.{json,yaml,csv}` instead of shared paths; `qsp_cpp_batch_job_<pool_id>.sh`; `flock -x -w 600 ~/.qsp_ensure_cpp_binary_<md5>.lock` around git-pull + cmake. Closes #48. |
| 50 | drop setup_remote_directories call that wiped pool inputs | Follow-up to #49 — the `rm -rf batch_jobs/input && mkdir -p` at submit start was a TOCTOU that erased pool-scoped subdirs sibling threads had just uploaded to. |

HPC venv on Rockfish is now at commit `b77166f` (main with all four PRs).

### Laptop-side changes in pdac-build (working tree, not committed)

- `workflows/sbi_runner.py` fan-out **reverted to sequential** (see comment block around L543): even with #49/#50 eliminating the HPC upload race, Rockfish's login-node sshd rate-limits three concurrent SSH exec streams from the laptop (observed: `rc=255 Connection closed by 162.129.223.61 port 22` + `kex_exchange_identification: Connection reset by peer` retry storms). Warm evolve cache keeps each serial scenario at ~3-5 min of sim time, so the loss is ~5-10 min wall vs a risky parallel path. Follow-up if you want parallelism back: serialize just `ensure_cpp_binary` via a Python-side `threading.Lock` in qsp-hpc-tools (or switch to a ControlMaster multiplexed SSH connection).
- `workflows/smoke_parallel_hpc_fanout.py` (new): minimal 2-scenario `ThreadPoolExecutor` reproducer that was used to verify pool-scoped uploads land distinctly. Pass/fail evidence lives in `figures/sbi_multi/smoke/parallel_fanout.log`.

### Current HPC + local state (2026-04-20 late PM)

| scenario | pool id | HPC state | local cache | work remaining |
|---|---|---|---|---|
| baseline_no_treatment | `d986ecc8` | 275 chunks, derive done | full 110k test stats | **none — local cache hit** |
| clinical_progression | `3e28df15` | 250 chunks (~100k sims, 25 tasks failed) | empty | **10k top-up + derive** |
| gvax_neoadjuvant_zheng2022 | `112b4dbb` | empty pool dir (failed array) | empty | **fresh 110k + derive** |
| gvax_nivo_neoadjuvant_zheng2022 | `2c632a12` | 2 batch dirs (800 smoke + 110k), derive done | partial 800 | **download + combine only** |

`count_pool_simulations` correctly reports `100000` for clinical from the laptop — so `run_hpc` will take the partial-pool branch and submit only 10k new sims (indices 100_000..109_999). No counter-bug fix needed.

### Residual failure we did NOT root-cause — quota on apopel1

The 25 failed clinical array tasks all died with:

```
File "qsp_hpc/cpp/param_xml.py", line 181, in render_to_file
    out.write_bytes(self.render(params))
  File ".../pathlib.py", line 1067, in write_bytes
OSError: [Errno 122] Disk quota exceeded
```

writing a small XML file to the per-task `.workdir_chunk_NNN/` under the pool dir. Standard `quota` / `df -h` / `df -i` show no limit hit (`/data` is 74% full with 1.4 PB free, user usage 260 GB, inodes at 31%). Likely a GPFS fileset-level quota on `/data/apopel1` that's only visible through Rockfish-specific tooling (`mmlsquota`, admin tooling). Could also have been a transient spike from 276 tasks × 400 sims all writing XML concurrently. **Probably worth a quick check with the cluster's quota tool before relaunching** — otherwise we'll likely see the same ~9% task-failure rate on the next clinical submission.

### Next-session relaunch

```
# 0. Verify Rockfish SSH is alive
ssh hpc "squeue -u \$USER -o '%.10i %.14j %.8T %.10M' | head -5"

# 1. (Optional) Sanity-check apopel1 fileset quota with Rockfish's quota tool
#    (mmlsquota, mmrepquota, or whatever your admin docs point to)

# 2. Relaunch — same command as before; sequential fan-out is now in-tree
/Users/joeleliason/Projects/pdac-build/.venv/bin/python -u workflows/sbi_runner.py \
  --cpp-jobs-per-chunk 400 --num-total-simulations 110000 \
  --cpp-branch cpp-sweep-binary-io \
  --restriction-classifier-dir cache/restriction_classifier \
  --restriction-threshold 0.5 \
  --cache-processed-tensors --stage-artifacts-only \
  --save-dir figures/sbi_multi/sweep/restricted_stage \
  2>&1 | tee figures/sbi_multi/sweep/restricted_stage/sim.log
```

Expected flow:

1. **baseline**: local test-stats cache hit (110k) — instant return (~3s).
2. **clinical_progression**: pool has 100k → top-up 10k → sbatch array (~25 tasks × 400 sims) → derive (~2m with refactor) → download. ~6-8 min total.
3. **gvax_neoadjuvant_zheng2022**: pool empty → fresh 110k → sbatch array (~275 tasks, warm evolve cache → ~3-5 min sim) → derive (~5 min) → download. ~12-15 min total.
4. **gvax_nivo_neoadjuvant_zheng2022**: HPC pool has 110k + 800 smoke sims + derive already done (from earlier in today's session) → just download + combine. ~1-2 min.
5. Joint NaN filter + processed bundle + staged artifacts → exit.

**Total estimate: ~25 min wall.** Dominated by gvax_neoadjuvant's fresh 110k.

### If the quota bug bites again

If clinical or gvax_neoadjuvant come back with `Errno 122 Disk quota exceeded` task failures:

- Array job will show some `COMPLETED`, some `FAILED` in `sacct -j <id> --format=State --noheader | sort -u`.
- Derive (`afterok:<array>`) will stick in PENDING forever because afterok requires all tasks to succeed.
- Scan/cancel the stuck derive and kill the orchestrator Python process.
- The `retry_missing_chunks` kwarg on `submit_cpp_jobs` exists (in qsp-hpc-tools `hpc_job_manager.py`) for submitting a sparse array of just the failed task indices — but sbi_runner doesn't plumb it through yet (default 0). Adding a CLI flag `--retry-missing-chunks N` and threading it down through `CppSimulator.run_hpc` → `job_manager.submit_cpp_jobs(retry_missing_chunks=N)` is a ~10-line change if we want automatic retry.

### Residual counter bug (benign, can ignore)

`count_pool_simulations` in qsp-hpc-tools' `result_collector.py` triggers a `ValueError: invalid literal for int() with base 10: ''` warning when it's asked about a **truly empty** pool dir — the shell falls through to the MAT fallback which `awk`s over zero input and emits `N_SIMS:` with no value. The error is caught and `n_available` defaults to 0, which is the correct answer for an empty pool, so nothing downstream is broken. Harmless noise in the log. Could be cleaned up by making the shell's MAT fallback print `N_SIMS:0` explicitly, but not blocking.

### File map additions from today

- `cache/sbi_simulations/multi_scenario_v4_all_scenarios_2c632a12_gvax_nivo_neoadjuvant_zheng2022/` — has the 800-smoke batch + 110k-restricted batch + completed derive.
- `cache/sbi_simulations/multi_scenario_v4_all_scenarios_3e28df15_clinical_progression/batch_20260420_143721_028430_clinical_progression_seed2025/` — 250 good chunks from the PM run; 10k top-up will add chunks 250..274.
- `/Users/joeleliason/Projects/qsp-hpc-tools` — now on main with PRs #46/47/49/50 merged; commit `b77166f` installed on HPC venv.
- `~/.qsp_ensure_cpp_binary_<md5>.lock` on HPC — new flock lockfile, md5 of repo path (`/home/jeliaso2/SPQSP_PDAC`).
- `workflows/smoke_parallel_hpc_fanout.py` — kept around for future concurrency regression tests.

---

## 2026-04-21 — restricted sweep v1+v2 landed; fold-change targets merged; scenario split

Picked up the `sweep/npe-training-grid` branch and drove the restricted-pool workflow end-to-end twice (once on the original 38-observable set, once after pulling in 4 new fold-change targets from main). The answer is pretty clear: **the `>0.5 contraction_local` ceiling is ~11 identified params, robust across 75k–100k training size and across the fold-change additions**. Next bottleneck is observable degeneracy, not data.

### What was done

1. **Freed HPC space.** `/data/apopel1` was filling up. `du` was too slow on GPFS so we inventoried `qsp_simulations/` by dir name + mtime and nuked 117 stale pool dirs (everything not in `{d986ecc8, 3e28df15, 112b4dbb, 2c632a12}` + `evolve_cache` + `theta_pools`). Confirmed with user before `rm -rf`.
2. **Relaunched the restricted 110k stage.** Hit a `ValueError: cannot reindex on an axis with duplicate labels` in `sbi_runner.py` step 2c: clinical had 3600 duplicate sample_indices (top-up batch re-ran indices that the Apr 20 batch had partially written), and gvax_nivo had 800 (residual smoke batch). Fixed by deduping `pos.index.duplicated(keep='first')` before reindexing, mirroring the set-based shared_arr collapse. Committed as `e67c4fa`. Also cleaned up the 800-sim smoke batch + stale derived test_stats for gvax_nivo on HPC before re-deriving.
3. **Sweep v1 (38 obs, no fold-change).** 5-cell 1D sweep over `n_train_subsample ∈ {10k, 25k, 50k, 75k, 100k}` with fixed `hidden_features=256, num_transforms=5`. Ceiling at 11 identified at `>0.5` (s75k = s100k). Grid YAML + driver thread `--restriction-classifier-dir` / `--restriction-threshold` via new `RESTRICT_DIR` / `RESTRICT_THRESHOLD` env vars. Committed as `94c1d85`.
4. **Pulled main → sweep branch.** Brought in 4 new fold-change calibration targets (cd8_fc_nonLA + cd8gzmb_fc_nonLA for both gvax_neoadjuvant and gvax_nivo), 2 new submodel targets (CXCL9_50_Trec, k_NK_kill), OBED Phase 2/4 machinery (predictions dir, longterm scenarios, run_ppc_clinical.py), and the PR #51 qsp-hpc-tools update (simulate_with_parameters). Old `cd8_fold_increase_gvax_nivo_d21` target moved to `excluded/`.
5. **Scenario split.** The merge bumped `scenarios/clinical_progression.yaml` stop_time 365 → 1095, which mutated its pool config hash from `3e28df15` → `67512b03` and orphaned the 110k sims on disk. Reverted clinical_progression.yaml back to 365, moved the 1095-day version to a new `scenarios/clinical_progression_longterm.yaml` (mirroring the `gvax_*_longterm` pattern), renamed `calibration_targets/predictions/clinical_progression → .../clinical_progression_longterm/`, updated scenario field inside each of the 12 prediction YAMLs, and updated `scripts/generate_prediction_yamls.py` + `scripts/prediction_target.py` + `tests/test_prediction_targets.py` + `workflows/run_ppc_clinical.py` to the new name. 148/148 prediction-target tests pass. Committed as `46da8c2`.
6. **Sweep v2 (41 obs, with fold-change).** Same grid, fresh processed bundle after merge. Results basically overlap v1.

### Sweep v1 (38 obs) — threshold table

| cell | n_train | >0.1 | >0.2 | >0.3 | >0.4 | >0.5 | >0.6 | >0.7 |
|---|---|---|---|---|---|---|---|---|
| s10k | 10k | 31 | 9 | 3 | 1 | 0 | 0 | 0 |
| s25k | 25k | 37 | 23 | 15 | 8 | 5 | 2 | 0 |
| s50k | 50k | 42 | 28 | 22 | 14 | 9 | 7 | 3 |
| s75k | 75k | 44 | 28 | 24 | 17 | **11** | 7 | 5 |
| s100k | 100k | 44 | 28 | 24 | 17 | **11** | 7 | 6 |

### Sweep v2 (41 obs, +3 fold-change) — threshold table

| cell | n_train | >0.1 | >0.2 | >0.3 | >0.4 | >0.5 | >0.6 | >0.7 |
|---|---|---|---|---|---|---|---|---|
| s10k | 10k | 29 | 9 | 3 | 1 | 0 | 0 | 0 |
| s25k | 25k | 38 | 23 | 15 | 8 | 6 | 3 | 2 |
| s50k | 50k | 45 | 29 | 21 | 13 | 8 | 7 | 3 |
| s75k | 75k | 46 | 30 | 23 | 18 | **11** | 7 | 6 |
| s100k | 100k | 46 | 29 | 23 | 18 | **11** | 8 | 6 |

Delta vs v1: `>0.5` ceiling identical; `>0.4` up by +1; `>0.7` at s25k up from 0 → 2 (fold-change helps a bit at small n). Best val loss at s100k improved 163.7 → 162.5.

### Pool state after today

All four `multi_scenario_v4_all_scenarios_*` pool dirs on HPC hold 110k restricted sims. Clinical pool (`3e28df15`) now also carries a 2026-04-21 AM top-up batch (25 chunks) on top of the original 250 — total 275 chunks — but 9 chunks from the first batch wrote sample_indices that overlap with the top-up, producing 3600 duplicate sample_indices when derived. Dedupe patch in `sbi_runner.py` handles this on read; no plan to clean up on HPC.

Local `cache/sbi_processed/*.pkl` is now valid for the 41-obs bundle (`...__restr2adbb9b075.pkl`). Delta artifacts for sweep v2 at `figures/sbi_multi/sweep/{delta_out,runs,logs}/s{10,25,50,75,100}k/`, collated to `sweep_summary.csv` + `sweep_by_param.csv` at `figures/sbi_multi/sweep/`.

### Friction notes

- Delta SSH ControlMaster died twice during the session (macOS lid close or 6–8h Duo re-auth window). Each time all 5 training jobs failed with `ssh_askpass: ... No such file`, `Permission denied (gssapi-with-mic,password)`. Workaround: re-auth manually via `ssh delta` + Duo push, then relaunch. Before relaunching, **wipe stale `density_estimator.pt` / `learning_curve.csv` / `metrics.json` out of `figures/sbi_multi/sweep/delta_out/s*/`** — otherwise the analyze step falls through to the old estimator (wrong obs dim if the bundle changed) and silently runs to completion against the wrong flow.
- `count_pool_simulations` in qsp-hpc-tools reads file-metadata correctly (verified manually), so the "HPC pool empty" branch today was purely a scenario-YAML-hash drift, not a pool-counting bug.
- The 5-minute ControlMaster persist doesn't cover the whole ~15 min sweep window. User mentioned simplifying qsp-hpc-tools — a standing followup would be a single tmux-on-jumphost pattern or a persistent ControlMaster on an always-on box.
- Processed-bundle cache key (`restr2adbb9b075` suffix) fingerprints priors + pool config, NOT calibration target YAMLs. So changing YAMLs doesn't invalidate the processed cache — have to `rm -f cache/sbi_processed/*.pkl` manually. This bit us once after the merge; kept as-is because the downstream per-scenario `test_stats_hash` DOES include target content (so HPC re-derives correctly) and the processed bundle is cheap to rebuild once the per-scenario caches repopulate.

### Commits on `sweep/npe-training-grid` today

- `e67c4fa` sbi_runner: dedupe sample_index in cross-scenario alignment
- `94c1d85` sweep: 1D over n_train_subsample; fix capacity; add restriction flags
- `eacf434` Merge branch 'main' into sweep/npe-training-grid
- `46da8c2` scenarios: split clinical_progression into short (365d) + longterm (1095d)

### Suggested next steps

1. **Publish the scenario split.** The `clinical_progression → clinical_progression_longterm` rename + predictions dir move is branch-local right now. It's a pure refactor (tests pass, no new sims needed for short-horizon) and would unblock anyone who pulls main + wants to run sweeps without the same revert dance. PR opportunity.
2. **Identifiability is observation-degenerate, not data-starved.** The >0.5 ceiling at 11 across 75k vs 100k vs +3 fold-change targets argues the remaining params are correlated under the current observables. Options: (a) sequential SNPE-C multi-round focused inference (existing `workflows/sequential_sbi_workflow.py` skeleton), (b) new non-redundant observable extraction (spatial mIHC patterning, TCR repertoire overlap, flow-sorted cytokine panels) — consult the priority list in `notes/calibration/calibration_strategy.md`. (c) Drop non-identifiable params from the SBI pool and treat them as nuisance — but that doesn't help the calibrated-count metric.
3. **Stage 2 of the param audit.** PPC audit was at 240/282 = 85% after the fix branch (see memory). With the restricted+fold-change bundle we probably want to re-score. Not urgent.
4. **Longterm scenarios.** `scenarios/{clinical_progression,gvax_neoadjuvant,gvax_nivo_neoadjuvant}_longterm.yaml` are now wired to the prediction-target YAMLs. Next session could run the OBED PPC workflow on a trained posterior — `workflows/run_ppc_clinical.py` is the entry point, needs the posterior .pt + the longterm sim pools (which don't exist yet; would be ~15–25 min of HPC fresh sims × 3 scenarios).

---

## 2026-04-21 PM — capacity sweep at fixed n=100k (observable-degeneracy confirmation)

PR #24 merged the scenario split to main. Then ran a dedicated NN-capacity grid at
fixed `n_train_subsample=100000` against the restricted + fold-change bundle to test
whether a larger flow could still extract gains from the current 41-observable set.

Grid: `hidden_features ∈ {256, 512, 1024}` × `num_transforms ∈ {5, 9, 13}` (9 cells,
`batch_size=2048`, `max_epochs=80`, `num_bins=10`). Configs at
`workflows/sweep/configs_capacity/`, outputs at `figures/sbi_multi/sweep_capacity/`.
Wall: ~40 min end-to-end (stage hit processed-tensor cache → instant; 2 waves of Delta
training; 9 parallel analyze jobs).

### Results (`sweep_summary.csv`)

| cell | n_identified (>0.5) | n_calibrated | best_val_loss | nn_params |
|---|---|---|---|---|
| **h256_t5** (ref) | **11** | 134 | **162.48** | 5.0M |
| h256_t9 | 11 | 133 | 165.55 | 9.1M |
| h256_t13 | 11 | 133 | 167.85 | 13.1M |
| h512_t5 | 11 | 136 | 164.80 | 12.6M |
| h512_t9 | 10 | 139 | 167.23 | 22.6M |
| h512_t13 | 10 | 139 | 169.09 | 32.6M |
| h1024_t5 | 8 | 139 | 166.12 | 35.4M |
| h1024_t9 | 9 | 137 | 168.74 | 63.7M |
| h1024_t13 | 10 | 125 | 170.68 | 92.1M |

### Takeaways

- **Capacity axis is flat / slightly harmful.** Scaling NN up to 92M params (18×
  baseline) does not raise the `contraction_local > 0.5` count above 11, and both
  `val_loss` and identified-count trend monotonically worse with width (clear overfit
  signal — early-stopped epochs drop from 81 at h256_t5 to 34 at h1024_t13).
- **`h256_t5` is Pareto-best** on (identification, val_loss, epochs_used). Keep it as
  the default training config going forward.
- **Confirms the sweep v2 conclusion**: the bottleneck is observable degeneracy, not
  network expressivity. Identifiability of the remaining ~160 params is observation-
  limited under the current 41-observable bundle.

### Implications for next steps

1. Stop investing in NN capacity. 256/5 is sufficient.
2. The real moves are (a) sequential SNPE-C (focuses density near `x_obs`, may convert
   some of the currently-degenerate params), or (b) extracting non-redundant
   observables (spatial mIHC, TCR overlap, cytokine panels).
3. If pursuing (a), `workflows/sequential_sbi_workflow.py` exists as a skeleton but
   needs to be wired to the restriction-classifier pool + fold-change obs set. Expect
   the effective gain cap from sequential rounds alone to be small (~2–4 more params)
   given how degenerate the current Jacobian rows are.
4. The `calibrated` count drops ~10 between the ref and the two 1024-wide cells with
   t≥9, but rises on the smaller-depth 1024 / 512 cells. The calibration metric
   tolerates some width at low depth; it's the deep+wide combos that miscalibrate. Not
   worth chasing — same net message.

### Commits on `sweep/npe-training-grid`

- `sweep: 3x3 capacity grid at fixed n_train=100k` (configs + `generate_configs.py`
  arg flags). Main-side PR not planned — this is research tooling local to the branch.

### File map additions

- `workflows/sweep/grid_capacity.yaml`, `workflows/sweep/configs_capacity/*.json`
- `figures/sbi_multi/sweep_capacity/{sweep_summary.csv, sweep_by_param.csv, runs/,
  delta_out/, logs/, stage/}`

---

## Sequential SNPE-C — kickoff plan (2026-04-21)

Decision: capacity is exhausted; the next lever is focusing density around `x_obs`
via SNPE-C (APT; Greenberg et al., 2019). Goal: lift the `contraction_local > 0.5`
count from 11 toward 15–20 on the current 41-observable bundle by concentrating
simulation effort where the posterior actually lives.

**Budget prior.** The remaining ~160 unidentified parameters have degenerate rows
in the observation Jacobian, so don't expect sequential to move the ceiling by more
than ~3–6 params on its own. If it does less, the observation degeneracy argument
is fully closed and the only remaining move is new observables. Either outcome is
a useful answer.

### Reuse strategy: start from round 1 already done

Round 1 of an SNPE-C run is "sample prior → simulate → train NPE" — which is
exactly what we already have. So:

- **Round-1 training data** = the 90,919-row processed bundle at
  `cache/sbi_processed/multi_scenario_v4_all_scenarios__…__restr2adbb9b075.pkl`
  (4 scenarios concatenated, joint NaN-filtered, copula-transformed, post-
  restriction-classifier).
- **Round-1 NPE** = the trained flow at
  `figures/sbi_multi/sweep_capacity/delta_out/h256_t5/density_estimator.pt`
  (h256_t5 — Pareto-best from the capacity sweep, 5.0M params, val_loss 162.48).
- **Round-1 copula quantiles + x_obs** = pulled from the same processed bundle
  (`obs_quantiles`, `obs_transformed`, `obs_tensor` in sbi_runner's step 2).

That cuts the first ~25 minutes off every sequential run. Rounds 2+ are the new
work.

### What has to be built

The existing skeleton at `workflows/sequential_sbi_workflow.py` (683 lines) is
~6 months stale and is **not** a drop-in starting point. Substantive gaps:

| gap | what it assumes | what we need |
|---|---|---|
| single scenario | `SCENARIO = 'baseline_no_treatment'` | 4-scenario concat w/ joint NaN filter, mirroring `sbi_runner.py` step 2 |
| no restriction classifier | raw prior sampler | round-2+ theta proposals should be rejection-sampled against the classifier before sim — posterior may still propose thetas that fail `evolve_to_diagnosis` |
| no fold-change targets | `calibration_targets/baseline_no_treatment/` only | 41-obs bundle including the 4 `cd8_fc_nonLA` / `cd8gzmb_fc_nonLA` targets |
| QSPSimulator (MATLAB) | `sim(n)` → MATLAB local | `CppSimulator` + `simulate_with_parameters(theta)` (PR #51 added this) for HPC dispatch of round-2+ batches |
| stale hyperparams | `hidden_features=50, max_epochs=500, batch_size=128` | match `h256_t5`: hf=256, t=5, bins=10, max_epochs=80, batch=2048 |
| no Delta offload | trains NPE locally | piggyback on `workflows/delta/run_delta_training.sh` each round (training tensors ship, estimator ships back) |

Cleanest path: **write a new `workflows/sequential_runner.py`** modeled on
`sbi_runner.py`, rather than retrofit the skeleton. Keep
`sequential_sbi_workflow.py` as reference / delete later once the new runner is
validated.

### Round-2+ loop — concrete

Per round (after loading round-1 state):

1. **Sample proposals.** `theta_log = posterior.sample((N,), x=x_obs_tensor)`;
   `theta_orig = exp(theta_log)`.
2. **Reject via classifier.** Score with
   `cache/restriction_classifier/classifier.pkl` at τ=0.5; keep accepted.
   Expect ~95% acceptance if round-1 posterior is already well-constrained.
3. **HPC fan-out.** For each of the 4 scenarios, call
   `CppSimulator.simulate_with_parameters(accepted_theta)` — submits a fresh
   sbatch array keyed by a new theta-hash suffix pool
   (`cache/sbi_simulations/posterior_predictive_<hash>/`). This DOES NOT reuse
   the restricted 110k pool — the whole point is new samples near the posterior.
   Expected: ~500 sims × 4 scenarios at ~20× warm-cache speedup ≈ 10–15 min/round.
4. **Concat + NaN filter** across the 4 scenarios, same as sbi_runner step 2.
5. **Copula-transform** using the *round-1* quantiles (don't re-fit the copula
   — rounds must share a common latent space or the density estimator breaks).
6. **Append + retrain** with atomic loss:
   `inference.append_simulations(theta_log, x_transformed, proposal=posterior)
    .train(...)`. Retraining from scratch per round is the sbi convention for
   SNPE-C; ship training tensors to Delta and reuse
   `run_delta_training.sh`.
7. **Rebuild posterior** conditioned on `x_obs`, compute Wasserstein vs previous
   round's posterior samples. Stop when max-axis W < 0.01 σ_prior OR round ≥ 10.

### Budget

- Round 1: 0 min (reuse).
- Rounds 2–N: ~15 min HPC + ~8 min Delta train = ~25 min/round.
- Plus ~5 min overhead per round (upload/download, retrain warmup).
- 5 rounds total ≈ 2.5 hours wall. 10 rounds ≈ 5 hours.

Worth front-loading simulations in round 2 (e.g., N=1000 or 2000) to stabilize
the posterior fast, then taper (500/round). Greenberg 2019 shows diminishing
returns past ~5 rounds.

### Things to decide before launching

1. **Round budget.** Start with 5 rounds × 500 sims? Or 3 rounds × 2000?
2. **Classifier threshold for round-2+.** Keep τ=0.5, tighten to 0.8, or drop
   entirely? Classifier is prior-conditional — may over-reject posterior-region
   thetas. Worth a quick sanity check: score h256_t5's posterior samples and
   see what fraction would be rejected.
3. **Training-from-scratch vs warm-start.** sbi's `NPE.train(retrain_from_scratch=True)`
   is the safe default. Warm-starting from round-1 weights could speed up each
   round but isn't standard SNPE-C.
4. **Where the round-2+ HPC pools live.** Under
   `cache/sbi_simulations/posterior_predictive_<theta_hash>/` (PR #51 convention)
   per-scenario, with the simulator's existing caching. Each round produces its
   own hash; no manual cleanup needed.

### Known risks

- **Proposal collapse.** If round-1 posterior is already too broad (median
  contraction was 0.028 on h256_t5), round-2 proposals spread nearly back to
  prior and sequential gives nothing. This would confirm observation-degeneracy.
- **Copula drift.** If round-2+ theta lives in a region where the round-1
  copula fit is poor (marginal quantile tails), x_transformed becomes
  ill-conditioned. Mitigation: sanity-check that round-2 x_copula values stay
  within round-1's observed range; if they drift, refit copula globally and
  retrain round 1 — more expensive but safer.
- **HPC queue contention.** Sequential has hard round-to-round dependencies —
  can't parallelize across rounds the way the capacity sweep could. Worst case
  is a stalled sbatch job blocking the whole run.

### File map — anticipated

- `workflows/sequential_runner.py` (new)
- `cache/sbi_simulations/posterior_predictive_<hash>_*` (per-round, per-scenario)
- `figures/sbi_sequential/rounds/<round_idx>/{density_estimator.pt,
  posterior_samples.csv, local_calibration.csv, ...}`
- `figures/sbi_sequential/convergence.csv` (Wasserstein + R²/cell trajectory
  across rounds)

---

## 2026-04-21 PM — sequential runner built + tested end-to-end

### What landed on this branch

- `workflows/sequential_runner.py` — round-loop orchestrator.
- `workflows/delta/{train_npe_sequential.py, train_npe_sequential.sbatch,
  run_delta_sequential_training.sh}` — multi-round SNPE-C trainer + driver.
- Warm-start mode (`cfg.warm_start_round1=True` by default). Round 1 is
  NOT appended to sbi; the h256_t5 estimator is loaded as initial
  weights, and round K trains atomic-loss only on round-K's ~300 rows
  with `retrain_from_scratch=False`. Cuts ~50 min/round → ~1–2 min/round.
- `cfg.npe_num_atoms=5` default (sbi default 10, overkill for small
  round sizes).

Upstream dependencies (both merged to `main`):

- **qsp-hpc-tools PR #52** — `CppSimulator.simulate_with_parameters(backend="local"|"hpc")`. HPC submits the user theta as its own suffix-pool, chains derivation, downloads, reshapes to the same pa.Table schema the local path emits.
- **qsp-hpc-tools PR #53** — race-recovery in `submit_derivation_job`:
  small arrays (N=300 at 20/task = 15 tasks, ~20s wall) complete
  before the laptop submits chained derivation. sbatch rejects the
  `afterok:<jid>` dependency with "Job dependency problem" even though
  the parent succeeded. Fix: when sacct confirms the parent is
  COMPLETED, retry derivation without the dependency. Tight scope —
  only afterok:, only that exact error message, only all-COMPLETED.

### What was validated in this session

- Full end-to-end run with N=300 thetas × 4 scenarios (HPC backend):
  classifier kept 295/300 (98%, mean p_valid=0.934 — posterior in the
  valid region vs prior baseline ~55%). All 4 HPC sims ran ~2.5 min
  each. Joint NaN filter: 294/295 survived (≥99%).
- Delta sequential trainer **cold mode** ran round 1 (391s on 90,919
  rows, best_val=162.48 — matches the reference h256_t5 exactly, so
  the bundle replays correctly) and started round 2 at ~40s/epoch
  (that's 91,213 rows × 11 atoms per epoch — expected but painful).
  Cancelled before finishing.
- Warm-start smoke on synthetic data: round 1 skipped (epochs=0,
  wall=0s), round 2 atomic-loss runs in 0.3s on 100 rows.

### What will need re-running after model updates

If any of the following change in pdac-build:

- `parameters/pdac_priors.csv` (rows, columns, or distribution params)
- `scenarios/*.yaml` (stop_time, initialization function, dosing, etc.)
- `calibration_targets/**` (adding/removing observables, unit conversions,
  code changes in observable functions)
- `model_structure.json` (species units)
- `SPQSP_PDAC` C++ source (model equations, parameter XML template)

...then:

1. **Priors / pool-config changes** → bump `MODEL_VERSION` in
   `sbi_runner.py` (currently `multi_scenario_v4_all_scenarios`).
   This invalidates the processed bundle + all HPC pools for the new
   version, forcing a fresh round-1 simulation pass.
2. **Scenario YAML changes** → pool config hashes shift (the `_____`
   hex prefix on the pool dirs). Drop the affected `cache/sbi_simulations/*`
   dirs or let them age out on HPC. The orphaned suffix pools on the
   laptop don't hurt; just consume disk.
3. **Calibration target changes** → `test_stats_hash` changes. HPC
   derivation reruns automatically, but the **round-1 processed
   bundle** at `cache/sbi_processed/*__restr2adbb9b075.pkl` is NOT
   invalidated by cal-target edits (keyed on priors+pool+sub+seed,
   not on target content). Manually `rm -f cache/sbi_processed/*.pkl`
   to rebuild.
4. **Observable count changes** → the h256_t5 estimator is tied to
   41 observables. Changing the bundle's obs dim invalidates the
   warm-start estimator. Need to re-run the capacity sweep (or at
   least a fresh single-cell sbi_runner on the new obs set) to get
   a new round-1 estimator before sequential makes sense.
5. **C++ binary rebuild** (any SPQSP_PDAC branch change that rebuilds
   `PDAC/qsp/sim/build/qsp_sim`) → CppSimulator's `config_hash` changes
   because it hashes the local binary bytes, invalidating ALL
   `cache/sbi_simulations/*` suffix pools even though the HPC backend
   never executes the local binary. This bit us on 2026-04-21 when a
   side cmake build at 20:35 invalidated the sequential-round-2 sim
   caches. No fix yet (see "Known quirks" below).

### Known quirks

- **Local-binary hash over-invalidates HPC-backend caches.** CppSimulator
  doesn't know which backend will be used, so it hashes the local binary
  into `config_hash`. For HPC-only sequential runs, the local binary
  bytes are irrelevant — but rebuilding invalidates everything. Possible
  fix: add `hash_local_binary=False` flag to CppSimulator, or split
  config_hash into local-only + remote-only components. Not blocking.
- **Delta ControlMaster TTL is 6–8h (Duo re-auth window).** Mid-run
  deaths require a manual `ssh delta` + Duo accept, then relaunching
  the runner with the **same** `--run-tag` (the theta hash is
  deterministic; local suffix-pool caches hit in <1s per scenario on
  resume). Using a new tag regenerates the `sequential_round{r}_<tag>`
  suffix-pool dir name and misses the cache.
- **Cumulative-retrain SNPE-C is wrong for asymmetric rounds.** Don't
  flip `warm_start_round1=False` unless you truly want to retrain
  round 1 from scratch each iteration. At 300x data asymmetry the
  atomic-loss per epoch is 11× the original NPE cost for no good
  reason — the round-K gradient gets drowned out.

### Suggested relaunch after model updates

```bash
# After bumping MODEL_VERSION / updating priors / etc., re-run the
# capacity-sweep reference cell to produce a fresh h256_t5 estimator
# + processed bundle. This is the one expensive step (~6 hrs wall
# for 100k fresh sims + ~7 min Delta train).
#
# Then:
.venv/bin/python -u workflows/sequential_runner.py \
  --num-rounds 3 --n-per-round 500 --backend hpc \
  --cpp-branch cpp-sweep-binary-io \
  --run-tag seq_$(date +%Y%m%d_%H%M%S) \
  2>&1 | tee figures/sbi_sequential/runs/$(ls -t figures/sbi_sequential/runs/ | head -1)/run.log
```

Per-round wall with warm-start + N=500: ~12 min HPC sims + ~2 min
Delta atomic-loss train = ~14 min/round. 3 rounds ≈ 45 min wall.

### Files to keep in mind

- `cache/sbi_processed/multi_scenario_v4_all_scenarios__...__restr2adbb9b075.pkl` — round-1 bundle (tied to model_version + priors + restriction classifier).
- `figures/sbi_multi/sweep_capacity/delta_out/h256_t5/density_estimator.pt` — round-1 warm-start source.
- `cache/restriction_classifier/{classifier.pkl, metadata.json}` — CV AUC 0.989 (tied to priors CSV; retrain via `workflows/restrict_prior_train.py` after prior changes).
- `cache/sbi_simulations/multi_scenario_v4_all_scenarios_{9cfdff18, 03246360, 4fb91adf, 4dae3a3e}_*` — round-1 HPC pool results (tied to MODEL_VERSION).
