# PDAC QSP — post-model-change helpers.
# `make help` for the menu. See notes/model/post_change_checklist.md for rationale.

SHELL        := /bin/bash
PY           := .venv/bin/python
PYTEST       := .venv/bin/pytest
EXPORT_MODEL := .venv/bin/qsp-export-model

MATLAB_MODEL := scripts/immune_oncology_model_PDAC.m
STRUCTURE    := model_structure.json

# In-repo C++ layout (port from SPQSP_PDAC, see notes/calibration/cpp_qual_slow_tests_handoff.md).
CPP_ODE_DIR   := cpp/qsp/ode
CPP_PARAM_XML := resources/cpp/param_all.xml
CPP_SIM_DIR   := cpp/sim
CPP_BUILD_DIR := $(CPP_SIM_DIR)/build

.PHONY: help refresh-all refresh-model-artifacts \
        model-structure audit test-fast test-slow \
        check-priors-sync check-cpp-sync check-submodel-sync \
        refresh-submodel-priors validate-cal-targets \
        export-sbml refresh-cpp parity hpc-status \
        sim-cache-sentinel

help:
	@echo "PDAC post-model-change targets"
	@echo
	@echo "  Orchestrators"
	@echo "    make refresh-all              -- full sweep (Python + submodel + C++) with reminders"
	@echo "    make refresh-model-artifacts  -- Python-side only: structure + priors + audit + tests"
	@echo
	@echo "  Layer 1 (Python / MATLAB)"
	@echo "    make model-structure          -- regenerate model_structure.json (needs MATLAB)"
	@echo "    make check-priors-sync        -- diff parameters_PDAC.m vs pdac_priors.csv"
	@echo "    make audit                    -- scripts/parameter_audit.py"
	@echo "    make test-fast                -- pytest -m 'not slow'"
	@echo "    make test-slow                -- pytest -m slow"
	@echo
	@echo "  Layer 2 (submodel calibration)"
	@echo "    make check-submodel-sync      -- submodel YAMLs + config reference valid params"
	@echo "    make refresh-submodel-priors  -- regen submodel_priors.yaml (heavyweight, MCMC)"
	@echo "    make validate-cal-targets     -- pytest tests/scenarios/test_calibration_targets.py"
	@echo
	@echo "  Layer 3 (C++ / HPC)"
	@echo "    make export-sbml              -- regenerate PDAC_model.sbml (MATLAB) in this repo"
	@echo "    make refresh-cpp              -- SBML+codegen+param XML+build (in-repo cpp/)"
	@echo "    make check-cpp-sync           -- verify priors CSV <-> cpp XML <-> species_units.json"
	@echo "    make parity                   -- 365-day MATLAB <-> C++ trajectory parity (needs MATLAB)"
	@echo "    make hpc-status               -- list repos/branches HPC points at, flag drift"
	@echo
	@echo "  Utility"
	@echo "    make sim-cache-sentinel       -- list SBI pools"

# ---------------- Orchestrators ----------------

refresh-model-artifacts: check-priors-sync model-structure audit test-fast
	@echo
	@echo "== Python-side refresh complete =="

refresh-all: refresh-model-artifacts check-submodel-sync validate-cal-targets \
             refresh-cpp check-cpp-sync hpc-status
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
	@echo "  4. If new params were added: consider make refresh-submodel-priors"
	@echo "     (skipped above — it runs MCMC and takes hours)."

# ---------------- Layer 1 ----------------

model-structure: $(STRUCTURE)

$(STRUCTURE): $(MATLAB_MODEL) $(wildcard core/modules/*.m) parameters/parameters_PDAC.m
	@echo ">> Regenerating $(STRUCTURE) via qsp-export-model (requires MATLAB)..."
	$(EXPORT_MODEL) --matlab-model $(MATLAB_MODEL) --output $(STRUCTURE)
	@echo ">> $(STRUCTURE) refreshed ($$(wc -l < $(STRUCTURE)) lines)."

check-priors-sync:
	@$(PY) scripts/check_priors_sync.py
	@echo
	@echo ">> test_prior_consistency (median==exp(mu), CSV/.m default + unit match)..."
	@$(PYTEST) tests/scenarios/test_prior_consistency.py -q

audit:
	$(PY) scripts/parameter_audit.py

test-fast:
	$(PYTEST) tests/ -m "not slow"

test-slow:
	$(PYTEST) tests/ -m slow

# ---------------- Layer 2 ----------------

check-submodel-sync:
	@$(PY) scripts/check_submodel_sync.py

refresh-submodel-priors:
	@echo "submodel_priors.yaml regen (MCMC, ~hours)."
	@echo "Not yet wired via Makefile — invoke the qsp-inference submodel pipeline"
	@echo "directly once the canonical CLI is exposed. In the meantime:"
	@echo "  $(PY) -c 'from qsp_inference.submodel.parameterizer import write_priors_yaml ...'"
	@echo
	@echo "Or see tests/integration/test_submodel_inference_pipeline.py for the"
	@echo "orchestration pattern."
	@exit 0

validate-cal-targets: $(STRUCTURE)
	$(PYTEST) tests/scenarios/test_calibration_targets.py -m slow

# ---------------- Layer 3 ----------------

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
	      -DPython3_EXECUTABLE=$(CURDIR)/.venv/bin/python
	cmake --build $(CPP_BUILD_DIR) --target qsp_sim -j 4
	@echo
	@echo "== refresh-cpp done. Binary: $(CPP_BUILD_DIR)/qsp_sim"
	@echo "== REMINDER: commit generated files under $(CPP_ODE_DIR)/ and $(CPP_PARAM_XML)."

check-cpp-sync:
	$(PYTEST) tests/test_cpp_xml_sync.py -q

parity:
	$(PYTEST) tests/test_cpp_matlab_parity.py -v -m slow

hpc-status:
	@$(PY) scripts/hpc_status.py

# ---------------- Utility ----------------

sim-cache-sentinel:
	@ls cache/sbi_simulations/ 2>/dev/null | sort
	@echo "(pools keyed by content hash of priors + test stats + model scripts + MODEL_VERSION)"
