# SPQSP_PDAC consumer-side invariants.
#
# Codegen (SBML -> C++) lives upstream in pdac-build via `make refresh-cpp`,
# which drives the standalone `qsp-codegen` tool and writes into this repo's
# PDAC/qsp/ode/ directory. This Makefile only holds checks that verify the
# current working tree is internally consistent.

PY ?= python3

.PHONY: help check-sync

help:
	@echo "SPQSP_PDAC targets"
	@echo "  make check-sync   -- run PDAC/codegen/check_sync.py (SBML/ODE/binary/XML drift)"
	@echo
	@echo "Codegen itself is driven from pdac-build: `cd ../pdac-build && make refresh-cpp`"

check-sync:
	$(PY) PDAC/codegen/check_sync.py
