# SPQSP_PDAC — project-level build and sync targets.
#
# The QSP model source of truth is qspio-pdac/ (MATLAB → SBML → C++ codegen).
# The ABM in PDAC/ embeds a copy of the generated C++ ODE code plus a merged
# param XML. These targets keep the two sides in sync.

PY ?= python3
QSPIO := qspio-pdac
ABM_ODE := PDAC/qsp/ode
SRC_ODE := $(QSPIO)/cpp/qsp/ode

ODE_FILES := ODE_system.cpp ODE_system.h QSPParam.cpp QSPParam.h QSP_enum.h

.PHONY: help sync-qsp check-qsp-sync check-sync

help:
	@echo "SPQSP_PDAC targets"
	@echo ""
	@echo "  make sync-qsp        -- copy ODE files from qspio-pdac → PDAC/qsp/ode/"
	@echo "  make check-qsp-sync  -- verify ODE files + params are in sync (read-only)"
	@echo "  make check-sync      -- alias for check-qsp-sync"
	@echo ""
	@echo "QSP codegen (SBML → C++):"
	@echo "  cd qspio-pdac && make refresh-cpp   -- regenerate C++ from SBML"
	@echo "  make sync-qsp                       -- then copy into PDAC/"

sync-qsp:
	@echo "Syncing ODE files: $(SRC_ODE) → $(ABM_ODE)"
	@for f in $(ODE_FILES); do \
		cp -v "$(SRC_ODE)/$$f" "$(ABM_ODE)/$$f"; \
	done
	@if [ -f "$(SRC_ODE)/qsp_params_xml_snippet.xml" ]; then \
		cp -v "$(SRC_ODE)/qsp_params_xml_snippet.xml" "$(ABM_ODE)/qsp_params_xml_snippet.xml"; \
	fi
	@echo "Sync complete. Rebuild the ABM: cd PDAC/sim && ./build.sh"

check-qsp-sync:
	$(PY) PDAC/codegen/check_sync.py

check-sync: check-qsp-sync
