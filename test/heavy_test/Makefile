# Makefile for Heavy MNIST Test Suite
# This runs comprehensive validation on the full MNIST dataset

# Paths
RTL_DIR = ../../rtl
TEST_DIR = .
COCOTB_DIR = ../cocotb_tests

# Verilator settings
TOPLEVEL_LANG = verilog
SIM = verilator

# DUT
TOPLEVEL = tinyml_accelerator_top
MODULE = test_full_mnist

# Verilator flags
EXTRA_ARGS += --trace --trace-structs
EXTRA_ARGS += -Wno-WIDTH -Wno-CASEINCOMPLETE -Wno-CASEX -Wno-TIMESCALEMOD
EXTRA_ARGS += -Wno-UNOPTFLAT -Wno-PINMISSING -Wno-UNUSED -Wno-BLKANDNBLK
EXTRA_ARGS += --timing  # Enable timing support
EXTRA_ARGS += -CFLAGS "-std=c++14"

# Source files - include all RTL modules
VERILOG_SOURCES = $(RTL_DIR)/tinyml_accelerator_top.sv
VERILOG_SOURCES += $(RTL_DIR)/fetch_unit.sv
VERILOG_SOURCES += $(RTL_DIR)/i_decoder.sv
VERILOG_SOURCES += $(RTL_DIR)/simple_memory.sv
VERILOG_SOURCES += $(RTL_DIR)/load_v.sv
VERILOG_SOURCES += $(RTL_DIR)/load_m.sv
VERILOG_SOURCES += $(RTL_DIR)/store.sv
VERILOG_SOURCES += $(RTL_DIR)/buffer_file.sv
VERILOG_SOURCES += $(RTL_DIR)/pe.sv
VERILOG_SOURCES += $(RTL_DIR)/top_gemv.sv
VERILOG_SOURCES += $(RTL_DIR)/quantization.sv
VERILOG_SOURCES += $(RTL_DIR)/quantizer_pipeline.sv
VERILOG_SOURCES += $(RTL_DIR)/scale_calculator.sv
VERILOG_SOURCES += $(RTL_DIR)/wallace_32x32.sv
VERILOG_SOURCES += $(RTL_DIR)/compressor_3to2.sv
VERILOG_SOURCES += $(RTL_DIR)/relu.sv

# Execution unit modules
VERILOG_SOURCES += $(RTL_DIR)/execution_unit/modular_execution_unit.sv
VERILOG_SOURCES += $(RTL_DIR)/execution_unit/buffer_controller.sv
VERILOG_SOURCES += $(RTL_DIR)/execution_unit/load_execution.sv
VERILOG_SOURCES += $(RTL_DIR)/execution_unit/gemv_execution.sv
VERILOG_SOURCES += $(RTL_DIR)/execution_unit/relu_execution.sv
VERILOG_SOURCES += $(RTL_DIR)/execution_unit/store_execution.sv

# Python path setup
export PYTHONPATH := $(COCOTB_DIR)/utils:$(PYTHONPATH)

# Test configuration via environment variables
# Usage examples:
#   make run_test                          # Run all 10,000 images
#   make run_test NUM_IMAGES=100           # Test first 100 images
#   make run_test STOP_ON_FIRST_FAIL=1     # Stop on first failure for debugging
#   make run_test VERBOSE=1                # Enable verbose output for all tests

# Default: test all images
NUM_IMAGES ?= 10000
export NUM_IMAGES

# Stop on first failure (useful for debugging)
STOP_ON_FIRST_FAIL ?= 0
export STOP_ON_FIRST_FAIL

# Verbose mode
VERBOSE ?= 0
export VERBOSE

# Include cocotb makefile
include $(shell cocotb-config --makefiles)/Makefile.sim

# Custom targets
.PHONY: run_test clean_sim quick_test debug_test

# Run full test suite
run_test: sim

# Quick test - first 100 images
quick_test:
	$(MAKE) run_test NUM_IMAGES=100

# Debug test - stop on first failure with verbose output
debug_test:
	$(MAKE) run_test NUM_IMAGES=100 STOP_ON_FIRST_FAIL=1 VERBOSE=1

# Clean simulation artifacts
clean_sim:
	rm -rf sim_build
	rm -rf __pycache__
	rm -rf *.vcd
	rm -rf results.xml
	rm -rf .pytest_cache

# Help
help:
	@echo "Heavy MNIST Test Suite"
	@echo "======================="
	@echo ""
	@echo "Targets:"
	@echo "  run_test       - Run full test suite (default: all 10,000 images)"
	@echo "  quick_test     - Run quick test (first 100 images)"
	@echo "  debug_test     - Run debug test (100 images, stop on first fail, verbose)"
	@echo "  clean_sim      - Clean simulation artifacts"
	@echo "  help           - Show this help"
	@echo ""
	@echo "Configuration (environment variables):"
	@echo "  NUM_IMAGES=N           - Test first N images (default: 10000)"
	@echo "  STOP_ON_FIRST_FAIL=1   - Stop on first failure (default: 0)"
	@echo "  VERBOSE=1              - Enable verbose output (default: 0)"
	@echo ""
	@echo "Examples:"
	@echo "  make run_test NUM_IMAGES=1000"
	@echo "  make debug_test"
	@echo "  make run_test STOP_ON_FIRST_FAIL=1 VERBOSE=1"
