# Cocotb Makefile for TinyML Accelerator Golden Model Verification
# This makefile runs the complete accelerator with golden model comparison

# Simulator
SIM ?= verilator

# Test target selection - MUST be set before any includes
TEST_TARGET ?= full_accelerator
# Options: full_accelerator, top_gemv

# Export for nested makes
export TEST_TARGET

ifeq ($(TEST_TARGET),top_gemv)
    TOPLEVEL = top_gemv
    MODULE = test_top_gemv
else
    # Default: full_accelerator
    TOPLEVEL = tinyml_accelerator_top
    MODULE = test_golden_comparison
endif

# Export cocotb variables
export TOPLEVEL
export MODULE

# Toplevel RTL module
TOPLEVEL_LANG = verilog

# Export language
export TOPLEVEL_LANG

# Source files - selected based on TEST_TARGET
ifeq ($(TEST_TARGET),top_gemv)
    # For top_gemv testbench, include only necessary supporting modules
    VERILOG_SOURCES += $(shell pwd)/../../rtl/top_gemv.sv
    VERILOG_SOURCES += $(shell pwd)/../../rtl/pe.sv
    VERILOG_SOURCES += $(shell pwd)/../../rtl/quantization.sv
    VERILOG_SOURCES += $(shell pwd)/../../rtl/scale_calculator.sv
    VERILOG_SOURCES += $(shell pwd)/../../rtl/quantizer_pipeline.sv
    VERILOG_SOURCES += $(shell pwd)/../../rtl/wallace_32x32.sv
    VERILOG_SOURCES += $(shell pwd)/../../rtl/compressor_3to2.sv
else
    # For full accelerator testbench
    VERILOG_SOURCES += $(shell pwd)/../../rtl/tinyml_accelerator_top.sv
    VERILOG_SOURCES += $(shell pwd)/../../rtl/fetch_unit.sv
    VERILOG_SOURCES += $(shell pwd)/../../rtl/i_decoder.sv
    VERILOG_SOURCES += $(shell pwd)/../../rtl/simple_memory.sv

    # Execution unit files
    VERILOG_SOURCES += $(shell pwd)/../../rtl/execution_unit/modular_execution_unit.sv
    VERILOG_SOURCES += $(shell pwd)/../../rtl/execution_unit/buffer_controller.sv
    VERILOG_SOURCES += $(shell pwd)/../../rtl/execution_unit/load_execution.sv
    VERILOG_SOURCES += $(shell pwd)/../../rtl/execution_unit/gemv_execution.sv
    VERILOG_SOURCES += $(shell pwd)/../../rtl/execution_unit/relu_execution.sv
    VERILOG_SOURCES += $(shell pwd)/../../rtl/execution_unit/store_execution.sv

    # Supporting modules
    VERILOG_SOURCES += $(shell pwd)/../../rtl/buffer_file.sv
    VERILOG_SOURCES += $(shell pwd)/../../rtl/load_v.sv
    VERILOG_SOURCES += $(shell pwd)/../../rtl/load_m.sv
    VERILOG_SOURCES += $(shell pwd)/../../rtl/store.sv
    VERILOG_SOURCES += $(shell pwd)/../../rtl/top_gemv.sv
    VERILOG_SOURCES += $(shell pwd)/../../rtl/pe.sv
    VERILOG_SOURCES += $(shell pwd)/../../rtl/relu.sv
    VERILOG_SOURCES += $(shell pwd)/../../rtl/quantization.sv
    VERILOG_SOURCES += $(shell pwd)/../../rtl/scale_calculator.sv
    VERILOG_SOURCES += $(shell pwd)/../../rtl/quantizer_pipeline.sv
    VERILOG_SOURCES += $(shell pwd)/../../rtl/wallace_32x32.sv
    VERILOG_SOURCES += $(shell pwd)/../../rtl/compressor_3to2.sv
endif

# Verilator specific flags
ifeq ($(SIM),verilator)
    EXTRA_ARGS += --trace --trace-structs
    EXTRA_ARGS += -Wno-WIDTHTRUNC -Wno-WIDTHEXPAND
    EXTRA_ARGS += -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM
    EXTRA_ARGS += -Wno-PINCONNECTEMPTY
    COMPILE_ARGS += -Wno-WIDTHTRUNC -Wno-WIDTHEXPAND
    COMPILE_ARGS += -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM
endif

# Include cocotb makefiles
include $(shell cocotb-config --makefiles)/Makefile.sim

# Custom targets
.PHONY: clean_all prepare run_test

# Clean everything including generated files
clean_all: clean
	rm -rf __pycache__
	rm -f results.xml
	rm -f dump.vcd

# Prepare: ensure dram.hex exists
prepare:
	@echo "Checking for dram.hex..."
	@if [ ! -f ../../compiler/dram.hex ]; then \
		echo "dram.hex not found. Running compiler to generate it..."; \
		cd ../../compiler && python3 main.py; \
	else \
		echo "dram.hex found."; \
	fi

# Run test with preparation
run_test: prepare clean
	$(MAKE) TEST_TARGET=$(TEST_TARGET) SIM=$(SIM) MODULE=$(MODULE) TOPLEVEL=$(TOPLEVEL) 

# Help target
help:
	@echo "TinyML Accelerator Cocotb Test Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make              - Run the default cocotb test"
	@echo "  make prepare      - Ensure dram.hex is generated"
	@echo "  make run_test     - Prepare and run test"
	@echo "  make clean        - Clean cocotb generated files"
	@echo "  make clean_all    - Clean all generated files"
	@echo "  make help         - Show this help message"
	@echo ""
	@echo "Test Targets (TEST_TARGET variable):"
	@echo "  TEST_TARGET=full_accelerator  - Test complete accelerator (default)"
	@echo "  TEST_TARGET=top_gemv          - Test top_gemv module only"
	@echo ""
	@echo "Simulators (SIM variable):"
	@echo "  SIM=verilator     - Use Verilator (default)"
	@echo "  SIM=icarus        - Use Icarus Verilog"
	@echo ""
	@echo "Examples:"
	@echo "  make run_test                              - Run complete accelerator test"
	@echo "  make TEST_TARGET=top_gemv run_test         - Run top_gemv test"
	@echo "  make SIM=verilator TEST_TARGET=top_gemv    - Run top_gemv with Verilator"
