# Codegen decoupling + cross-repo simplification plan

**Status:** planning, 2026-04-21 (updated 2026-04-22: gating step 1 resolved)
**Related:** `notes/calibration/gvax_cascade_fixes_session_handoff.md` (§1 RESOLVED: C++ equivalence failures were a test-harness misconfig in `test_ode_vs_matlab.py`, not a codegen bug — hard-coded `~/Projects/pdac-build/` path compared main-branch MATLAB against worktree C++. Fixed by env-var override with fallback. Remaining `PDL1_total`/`PDL2_total` divergence is pre-existing and deferred — not codegen-related.)

## Goal

Separate the SBML → C++ ODE codegen from SPQSP_PDAC so it can be a standalone
tool the ODE consumers pull in, and simplify the cross-repo plumbing that
accreted around the current in-SPQSP layout. Option A (codegen-only tool repo,
consumers own generated artifacts) was chosen over alternatives that also pull
the standalone `qsp_sim` simulator and CVODE wrapper into a new repo — simpler
boundary, consumers already have refresh infrastructure (Makefile,
`hpc_status.py`, `check_priors_sync.py`).

## Current crosstalk (audit)

1. **Codegen lives in the consumer, not next to its inputs.** `make
   refresh-cpp` in pdac-build invokes `cd $SPQSP && PDAC_BUILD_DIR=$CURDIR
   python PDAC/codegen/qsp_codegen.py`. Tool lives in SPQSP_PDAC, reads
   pdac-build inputs via env-var hack, writes back into SPQSP_PDAC.
   Direction of information flow is inverted relative to filesystem layout.
   Worktrees only work because `PDAC_BUILD_DIR` overrides the default.
2. **7-step cross-repo dance** in `refresh-cpp`: `cd $SPQSP; reexport_sbml.sh`
   → `qsp_codegen.py` → `refresh_param_xml.py` → `cmake --build` →
   `check_sync.py`. All with `PDAC_BUILD_DIR=$CURDIR` re-exported each step.
3. **SBML export lives in pdac-build's `.m` scripts but is invoked from a
   shell script in SPQSP's `PDAC/codegen/reexport_sbml.sh`**. SBML authoring
   crosses the repo boundary.
4. **`ensure_cpp_binary()` in qsp-hpc-tools hardcodes the
   `{repo}/PDAC/qsp/sim/build/qsp_sim` layout** (hpc_job_manager.py:709–721)
   and derives `cpp.repo_path` by chopping 5 path components. Brittle.
5. `hpc_status.py` imports `HPCJobManager` just to read `credentials.yaml`'s
   `cpp.branch`, then issues ad-hoc inline SSH bash. Not a blocker.
6. 5-repo drift check (soon 6 with new `qsp-codegen` repo) is unavoidable —
   each has independent drift potential. Constant cost.
7. **HPC auto-rebuild logic is unaffected by the split.** `ensure_cpp_binary`
   does `git pull` + `cmake build` in SPQSP_PDAC; it does NOT run codegen
   (generated files are checked in). After the split, HPC logic doesn't
   change: run `qsp-codegen` locally, commit generated C++ into SPQSP_PDAC,
   push, HPC auto-pulls and rebuilds.

## Simplifications

| # | Change | Effort | Payoff |
|---|---|---|---|
| 1 | Standalone `qsp-codegen` CLI with `--sbml`/`--out` args | M | Removes `PDAC_BUILD_DIR` env-var hack; worktrees just work; codegen testable in isolation |
| 2 | Move SBML export into pdac-build (`make export-sbml`); drop `reexport_sbml.sh` from SPQSP | S | Authoring and storage colocated; removes cross-repo shell script |
| 3 | Collapse `refresh-cpp` target to 2 steps | S (falls out of 1+2) | Easier to reason about, easier to document |
| 4 | `check_sync.py` stays in SPQSP's own Makefile (not invoked by pdac-build) | S | Consumer owns its invariants |
| 5 | Require `cpp.repo_path` explicit in `credentials.yaml`, drop derivation | XS | Removes brittle path-chopping in qsp-hpc-tools |

## Scope of new `qsp-codegen` repo

**In:**
- `qsp_codegen.py` (2292 LOC, SBML → ODE C++ generator)
- `refresh_param_xml.py` (74 LOC, param XML regeneration from `parameters_PDAC.m`)
- Any Jinja/templates they use (to confirm during extraction)
- CLI entry point: `qsp-codegen --sbml <path> --out-dir <path>` (and
  `qsp-refresh-param-xml --params <.m> --out <xml>`)
- `pyproject.toml` with pinned version so consumers pin it
- Unit tests + small fixture SBML for regression testing in isolation

**Out (stays in SPQSP_PDAC):**
- `abm_param_codegen.py` + `bootstrap_abm_xml.py` — ABM-specific, tied to
  SPQSP's `param_all.xml` layout
- `check_sync.py` + `sync_checks.py` — consumer-side invariants between
  generated files and hand-written ABM code
- Generated `ode/*.cpp|h|xml` — checked in by each consumer

**Moves to pdac-build:**
- SBML export: the MATLAB-side of `reexport_sbml.sh` becomes `make export-sbml`
  in pdac-build's Makefile, writing `PDAC_model.sbml` to the repo root.
- `reexport_sbml.sh` itself is deleted.

## Execution order

1. ~~**Fix C++ equivalence bug in SPQSP_PDAC**~~ — **RESOLVED 2026-04-22.**
   Diagnosis was not a codegen forward-ref bug as hypothesized; it was
   `test_ode_vs_matlab.py` hard-coding `~/Projects/pdac-build/` so the
   worktree's `make cpp-equivalence` compared main-branch MATLAB to
   worktree C++. Fix landed on `fix/gvax-cascade-bottleneck` (env-var
   override with sibling fallback). Orthogonal to the decoupling plan —
   clears the gate.
2. **`make export-sbml` in pdac-build**; delete `reexport_sbml.sh` from
   SPQSP. Trivial.
3. **Extract `qsp-codegen`** repo with CLI. Port `qsp_codegen.py` +
   `refresh_param_xml.py`. Pin version in pdac-build's Makefile. Update
   `refresh-cpp` target to `qsp-codegen --sbml ... --out ...` + `cmake
   --build`. Move `check_sync.py` invocation into SPQSP's own Makefile.
4. **Drop `cpp.repo_path` derivation in `ensure_cpp_binary`** —
   qsp-hpc-tools one-liner.

## Non-goals

- Moving the standalone `qsp_sim` simulator or CVODE wrapper out of
  SPQSP_PDAC. These are ODE consumers, not codegen.
- Moving the full hybrid SPQSP ABM+ODE simulator.
- Changing the `cpp.branch` HPC tracking model. That stays as-is.
- Regenerating `submodel_priors.yaml` (MCMC, hours).

## Related artifacts

- `Makefile` (pdac-build) — `refresh-cpp`, `hpc-status` targets
- `scripts/hpc_status.py` (pdac-build)
- `SPQSP_PDAC/PDAC/codegen/` — source being split
- `qsp-hpc-tools/qsp_hpc/batch/hpc_job_manager.py` — `ensure_cpp_binary`
- User memory: `reference_refresh_workflow.md`
