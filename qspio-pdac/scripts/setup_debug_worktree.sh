#!/usr/bin/env bash
# Spin up a debugging worktree with all the gitignored bits the agent loop
# expects. Mechanical companion to Section 1 of
# notes/workflows/iterative_model_debugging.md — keep in sync if rules change.
#
# Usage:
#   scripts/setup_debug_worktree.sh <short-label> [<base-branch>]
#
# Creates:
#   - branch:   feature/<short-label>     (forked from <base-branch>, default main)
#   - worktree: ../pdac-build-<short-label>
# Populates:
#   - cpp/sim/build/_deps  (seeded from primary if present, saves SUNDIALS clone)
#   - cpp/sim/build/qsp_sim (built via `make refresh-cpp`)
#   - .compare_cache/ snapshot
#   - .qsp-hpc/ credentials override
#   - .env (OpenAI key)
#   - .venv symlink to primary's venv
#
# See notes/workflows/iterative_model_debugging.md for rationale on each step
# (per-worktree cpp build, snapshot vs symlink rules, etc.).

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 <short-label> [<base-branch>]" >&2
  exit 2
fi

LABEL="$1"
BASE_BRANCH="${2:-main}"

PRIMARY="$(git rev-parse --show-toplevel)"
NAME="feature/${LABEL}"
WT="$(dirname "$PRIMARY")/pdac-build-${LABEL}"

if [ -e "$WT" ]; then
  echo "error: worktree path already exists: $WT" >&2
  exit 1
fi

echo ">> Creating worktree at $WT (branch $NAME from $BASE_BRANCH)..."
git -C "$PRIMARY" worktree add -b "$NAME" "$WT" "$BASE_BRANCH"

echo ">> Seeding cpp/sim/build/_deps/*-src from primary (skips network re-fetch of SUNDIALS / yaml-cpp)..."
mkdir -p "$WT/cpp/sim/build/_deps"
if [ -d "$PRIMARY/cpp/sim/build/_deps" ]; then
  # ONLY *-src/ is path-portable. *-build/ contains nested cmake try_compile
  # projects (e.g. _deps/sundials-build/POSIX_TIMER_TEST/CMakeCache.txt) and
  # *-subbuild/ contains the FetchContent wrapper-project, both of which have
  # the primary's absolute path hard-coded in CMakeCache. cmake refuses to
  # reuse them at the new worktree path and bails. Saving the src copy is
  # what actually skips the slow part (the git clone over network); the
  # *-build compile (~3-5 min) reruns per worktree and is unavoidable.
  for srcdir in "$PRIMARY"/cpp/sim/build/_deps/*-src; do
    [ -d "$srcdir" ] || continue
    cp -r "$srcdir" "$WT/cpp/sim/build/_deps/"
  done
else
  echo "   (primary has no cpp/sim/build/_deps yet; first build will fetch)"
fi

echo ">> Snapshotting .compare_cache/ (worktree owns it after copy)..."
SRC_CACHE="$PRIMARY/calibration_targets/submodel_targets/.compare_cache"
if [ -d "$SRC_CACHE" ]; then
  cp -r "$SRC_CACHE" "$WT/calibration_targets/submodel_targets/.compare_cache"
else
  echo "   (no .compare_cache/ in primary; next audit will rebuild)"
fi

echo ">> Copying .qsp-hpc/ credentials override..."
if [ -d "$PRIMARY/.qsp-hpc" ]; then
  cp -r "$PRIMARY/.qsp-hpc" "$WT/.qsp-hpc"
else
  echo "   WARNING: no .qsp-hpc/ in primary — sbi_runner will fall back to global creds" >&2
fi

echo ">> Copying .env (OpenAI key for maple extraction)..."
if [ -f "$PRIMARY/.env" ]; then
  cp "$PRIMARY/.env" "$WT/.env"
else
  echo "   WARNING: no .env in primary — staged_extraction.py will fail at LLM call" >&2
fi

echo ">> Symlinking .venv -> primary..."
ln -s "$PRIMARY/.venv" "$WT/.venv"

echo ">> Building qsp_sim via \`make -B refresh-cpp\` (forces SBML + model_structure.json regen)..."
# -B is required because git worktree add resets all mtimes to "now," so make
# would otherwise see source and generated files as in-sync and skip the
# regen. Force ensures the worktree starts with artifacts that actually
# match its current cpp/qsp/ source. Needs MATLAB.
( cd "$WT" && make -B refresh-cpp )

cat <<EOF

== Worktree ready ==
   path:   $WT
   branch: $NAME (from $BASE_BRANCH)

Next:
   cd $WT
   # ... edit, then:
   python -m workflows.sbi_runner --scenarios baseline_no_treatment \\
     --cpp-branch $NAME --num-total-simulations 5000

Tear-down:
   git worktree remove $WT
   (reinstall any worktree-side editable installs from primary first —
    see feedback_editable_install_after_worktree_removal.md)
EOF
