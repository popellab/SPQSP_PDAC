.PHONY: build test test-build test-gpu clean

BUILD_DIR := build
BINARY := $(BUILD_DIR)/bin/pdac
CMAKE_ARGS := -DCMAKE_BUILD_TYPE=Release
NPROC := $(shell nproc 2>/dev/null || echo 4)

ifdef CUDA_ARCH
CMAKE_ARGS += -DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH)
endif

ifdef SUNDIALS_DIR
CMAKE_ARGS += -DSUNDIALS_DIR=$(SUNDIALS_DIR)
endif

build:
	cmake -S . -B $(BUILD_DIR) $(CMAKE_ARGS)
	cmake --build $(BUILD_DIR) --parallel $(NPROC)

test: test-build test-gpu

test-build:
	uv run --extra test pytest tests/test_build.py -v

test-gpu: build
	uv run --extra test pytest tests/ -v

clean:
	rm -rf $(BUILD_DIR)
