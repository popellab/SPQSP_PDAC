# QSPIO-PDAC — the model, and post-model-change helpers.
# `make help` for the menu. See notes/model/post_change_checklist.md for rationale.
#
# The corpus, the priors and the campaign moved to pdac-paper on 2026-08-19, and
# the targets that compared the two sides went with them: run `make
# check-priors-sync`, `check-cpp-sync` and `check-observables-sync` from there.
# PAPER_ROOT is exported for run_median_simulation.m, which reads the priors and
# the submodel posterior and is the one thing here that looks the other way.

SHELL        := /bin/bash
PAPER_ROOT   ?= ../pdac-paper
export PAPER_ROOT
PY           := .venv/bin/python
PYTEST       := .venv/bin/pytest

MATLAB_MODEL := scripts/immune_oncology_model_PDAC.m
STRUCTURE    := model_structure.json

# In-repo C++ layout. The port from SPQSP_PDAC completed: the simulator builds
# from cpp/ in this repo, and SPQSP_PDAC exists on no machine we use.
CPP_ODE_DIR   := cpp/qsp/ode
CPP_PARAM_XML := resources/cpp/param_all.xml
CPP_SIM_DIR   := cpp/sim
CPP_BUILD_DIR := $(CPP_SIM_DIR)/build
# ccache: cache the unchanging translation units (qsp_sim_main, QSPParam, SUNDIALS
# glue) across rebuilds so only the regenerated ODE_system.cpp recompiles. Guarded
# so it's a no-op where ccache isn't installed (e.g. HPC).
CCACHE        := $(shell command -v ccache 2>/dev/null)
CCACHE_FLAGS  := $(if $(CCACHE),-DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache,)

.PHONY: help refresh-all refresh-model-artifacts \
        model-structure test-fast test-slow \
        export-sbml refresh-cpp parity hpc-status \
        schematic

help:
	@echo "PDAC post-model-change targets"
	@echo
	@echo "  Orchestrators"
	@echo "    make refresh-all              -- full sweep (Python + submodel + C++) with reminders"
	@echo "    make refresh-model-artifacts  -- structure + the model's own tests"
	@echo
	@echo "  Layer 1 (Python / MATLAB)"
	@echo "    make model-structure          -- regenerate model_structure.json (needs MATLAB)"
	@echo "    make test-fast                -- pytest -m 'not slow'"
	@echo "    make test-slow                -- pytest -m slow"
	@echo
	@echo "  Layer 2 (C++ / HPC)"
	@echo "    make export-sbml              -- regenerate PDAC_model.sbml (MATLAB) in this repo"
	@echo "    make refresh-cpp              -- SBML+codegen+param XML+build (in-repo cpp/)"
	@echo "    make parity                   -- 365-day MATLAB <-> C++ trajectory parity (needs MATLAB)"
	@echo "    make hpc-status               -- list repos/branches HPC points at, flag drift"
	@echo
	@echo
	@echo "  In pdac-paper, not here (they compare the two sides)"
	@echo "    check-priors-sync, check-submodel-sync, check-cpp-sync,"
	@echo "    refresh-observables, check-observables-sync"
	@echo
	@echo "  Utility"
	@echo "    make schematic                -- re-export qsp_schematic.{pdf,png} from drawio source"

# ---------------- Orchestrators ----------------

refresh-model-artifacts: model-structure test-fast
	@echo
	@echo "== Python-side refresh complete =="

refresh-all: refresh-model-artifacts refresh-cpp hpc-status
	@echo
	@echo "=============================================================="
	@echo "== FULL REFRESH COMPLETE =="
	@echo "=============================================================="
	@echo "Reminders before running Stage-2 SBI:"
	@echo "  1. If you changed core/modules/*.m, commit+push pdac-build."
	@echo "  2. If refresh-cpp regenerated SBML or C++: commit+push pdac-build"
	@echo "     (generated files under cpp/qsp/ode/ + resources/cpp/param_all.xml)."
	@echo "     ensure_cpp_binary() on HPC pulls the configured branch, so"
	@echo "     unpushed commits are invisible to the cluster."
	@echo "  3. Verify make hpc-status shows no FAILs."
	@echo "  4. If new params were added, the priors and submodel YAMLs are in"
	@echo "     pdac-paper: run its check-priors-sync and check-submodel-sync."

# ---------------- Layer 1 ----------------

model-structure: $(STRUCTURE)

$(STRUCTURE): $(MATLAB_MODEL) scripts/export_model_structure.m $(wildcard core/modules/*.m) parameters/parameters_PDAC.m
	@echo ">> Regenerating $(STRUCTURE) via MATLAB..."
	matlab -batch "run('scripts/export_model_structure.m')"
	@echo ">> $(STRUCTURE) refreshed ($$(wc -l < $(STRUCTURE)) lines)."

test-fast:
	$(PYTEST) tests/ -m "not slow"

test-slow:
	$(PYTEST) tests/ -m slow

export-sbml: PDAC_model.sbml

PDAC_model.sbml: $(MATLAB_MODEL) $(wildcard core/modules/*.m) parameters/parameters_PDAC.m
	@echo ">> Regenerating PDAC_model.sbml via MATLAB..."
	matlab -batch "run('scripts/export_sbml.m')"
	@echo ">> PDAC_model.sbml refreshed."

QSP_CODEGEN    := .venv/bin/qsp-codegen
QSP_REFRESH_XML := .venv/bin/qsp-refresh-param-xml

refresh-cpp: export-sbml model-structure
	@# SBML lives in pdac-build (producer); qsp-codegen reads it from here
	@# and writes generated C++ into the in-repo $(CPP_ODE_DIR). qsp-codegen
	@# is a pinned sibling tool (pip install -e if editing).
	@echo ">> qsp-codegen: $(CURDIR)/PDAC_model.sbml -> $(CPP_ODE_DIR) ..."
	$(QSP_CODEGEN) --sbml $(CURDIR)/PDAC_model.sbml --out-dir $(CPP_ODE_DIR)
	@echo ">> qsp-refresh-param-xml: $(CPP_ODE_DIR)/qsp_params_xml_snippet.xml -> $(CPP_PARAM_XML) ..."
	$(QSP_REFRESH_XML) \
	    --snippet $(CPP_ODE_DIR)/qsp_params_xml_snippet.xml \
	    --xml $(CPP_PARAM_XML)
	@echo ">> configure+build qsp_sim (CPU ODE target)..."
	@# Always reconfigure — cmake is idempotent and a no-op when nothing
	@# changed. The previous CMakeCache.txt-existence guard could skip
	@# configure when CMakeCache.txt existed but Makefile did not (partial
	@# build, copied artifacts, etc.), leaving the build to fail with
	@# "Makefile: No such file or directory".
	@# Pin Python3_EXECUTABLE to the project venv so cmake's
	@# `python -m qsp_codegen.cmake --prefix` lookup hits the qsp-codegen
	@# wheel installed there, not whatever system python3 is first on PATH.
	cmake -S $(CPP_SIM_DIR) -B $(CPP_BUILD_DIR) -DCMAKE_BUILD_TYPE=Release \
	      $(CCACHE_FLAGS) \
	      -DPython3_EXECUTABLE=$(CURDIR)/.venv/bin/python
	cmake --build $(CPP_BUILD_DIR) --target qsp_sim -j 4
	@echo
	@echo "== refresh-cpp done. Binary: $(CPP_BUILD_DIR)/qsp_sim"
	@echo "== REMINDER: commit generated files under $(CPP_ODE_DIR)/ and $(CPP_PARAM_XML)."

# ---------------- Layer 2 (C++ / HPC) ----------------

parity:
	$(PYTEST) tests/test_cpp_matlab_parity.py -v -m slow

hpc-status:
	@$(PY) scripts/hpc_status.py

# ---------------- Utility ----------------

# Re-render the auto-layout schematic used in the Lustgarten doc from its D2 source via the
# custom Node + elkjs renderer in core/visualization/elk-renderer/. That renderer (not d2's
# own layout, and not drawio) is what produced the canonical figure in the ICI grant
# materials. Pipeline: D2 -> custom ELK layout -> SVG -> rsvg-convert -> PDF + PNG.
SCHEMATIC_D2     := core/visualization/qsp_schematic.d2
SCHEMATIC_RENDER := core/visualization/elk-renderer/render_schematic.mjs
# figures/ is gitignored; it used to be notes/paper/figures/, and notes/ went
# on 2026-08-19. The rendered schematic is a build product either way.
SCHEMATIC_SVG    := figures/qsp_schematic.svg
SCHEMATIC_PDF    := figures/qsp_schematic.pdf
SCHEMATIC_PNG    := figures/qsp_schematic.png

schematic:
	@if ! command -v node >/dev/null 2>&1; then \
	  echo "node not found. Install with: brew install node"; exit 1; \
	fi
	@if ! command -v rsvg-convert >/dev/null 2>&1; then \
	  echo "rsvg-convert not found. Install with: brew install librsvg"; exit 1; \
	fi
	@if [ ! -d core/visualization/elk-renderer/node_modules ]; then \
	  echo "Installing elk-renderer node_modules..."; \
	  (cd core/visualization/elk-renderer && npm install >/dev/null); \
	fi
	@mkdir -p $(dir $(SCHEMATIC_SVG))
	node $(SCHEMATIC_RENDER) $(SCHEMATIC_SVG)
	rsvg-convert -f pdf -o $(SCHEMATIC_PDF) $(SCHEMATIC_SVG)
	rsvg-convert -f png -z 2 -o $(SCHEMATIC_PNG) $(SCHEMATIC_SVG)
	@echo "Re-rendered $(SCHEMATIC_PDF) and $(SCHEMATIC_PNG) from $(SCHEMATIC_D2)."
