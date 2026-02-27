#!/bin/bash
set -e

echo "=== Running New Unit Tests ==="

# Ensure we are in the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Modular Execution Unit Integration Test
echo "----------------------------------------------------------------"
echo "Building modular_execution_unit_tb..."
echo "----------------------------------------------------------------"
rm -rf obj_dir
verilator --cc --exe --trace --build -j 0 -Wno-fatal \
  --top-module modular_execution_unit \
  ../../rtl/fpga_modules/modular_execution.sv \
  ../../rtl/fpga_modules/buffer_controller.sv \
  ../../rtl/fpga_modules/buffer_file.sv \
  ../../rtl/fpga_modules/load_execution.sv \
  ../../rtl/load_v.sv \
  ../../rtl/load_m.sv \
  ../../rtl/fpga_modules/gemv_execution.sv \
  ../../rtl/fpga_modules/gemv_unit_core.sv \
  ../../rtl/fpga_modules/relu_execution.sv \
  ../../rtl/relu.sv \
  ../../rtl/fpga_modules/store_execution.sv \
  ../../rtl/store.sv \
  ../../rtl/pe.sv \
  ../../rtl/scale_calculator.sv \
  ../../rtl/quantizer_pipeline.sv \
  ../../rtl/quantizer_pipeline.sv \
  ../../rtl/fpga_modules/Gowin_RAM16SDP_Mock.sv \
  modular_execution_unit_tb.cpp \
  -CFLAGS "-std=c++17" \
  -o modular_execution_test > build.log 2>&1

echo "Running modular_execution_test..."
./obj_dir/modular_execution_test

# 2. FPGA GEMV Execution Unit Test
echo "----------------------------------------------------------------"
echo "Building fpga_gemv_execution_tb..."
echo "----------------------------------------------------------------"
rm -rf obj_dir
verilator --cc --exe --trace --build -j 0 -Wno-fatal \
  --top-module gemv_execution \
  ../../rtl/fpga_modules/gemv_execution.sv \
  ../../rtl/fpga_modules/gemv_unit_core.sv \
  ../../rtl/pe.sv \
  ../../rtl/scale_calculator.sv \
  ../../rtl/quantizer_pipeline.sv \
  ../../rtl/quantizer_pipeline.sv \
  ../../rtl/fpga_modules/Gowin_RAM16SDP_Mock.sv \
  fpga_gemv_execution_tb.cpp \
  -CFLAGS "-std=c++17" \
  -o fpga_gemv_execution_test > build.log 2>&1

echo "Running fpga_gemv_execution_test..."
./obj_dir/fpga_gemv_execution_test

# 3. Store Execution Unit Test
echo "----------------------------------------------------------------"
echo "Building store_execution_tb..."
echo "----------------------------------------------------------------"
rm -rf obj_dir
verilator --cc --exe --trace --build -j 0 -Wno-fatal \
  --top-module store_execution \
  ../../rtl/fpga_modules/store_execution.sv \
  ../../rtl/store.sv \
  store_execution_tb.cpp \
  -CFLAGS "-std=c++17" \
  -o store_execution_test > build.log 2>&1

echo "Running store_execution_test..."
./obj_dir/store_execution_test

# 4. Neural Network Integration Test
echo "----------------------------------------------------------------"
echo "Building neural_network_tb..."
echo "----------------------------------------------------------------"
rm -rf obj_dir
verilator --cc --exe --trace --build -j 0 -Wno-fatal \
  --top-module modular_execution_unit \
  ../../rtl/fpga_modules/modular_execution.sv \
  ../../rtl/fpga_modules/buffer_controller.sv \
  ../../rtl/fpga_modules/buffer_file.sv \
  ../../rtl/fpga_modules/load_execution.sv \
  ../../rtl/load_v.sv \
  ../../rtl/load_m.sv \
  ../../rtl/fpga_modules/gemv_execution.sv \
  ../../rtl/fpga_modules/gemv_unit_core.sv \
  ../../rtl/fpga_modules/relu_execution.sv \
  ../../rtl/relu.sv \
  ../../rtl/fpga_modules/store_execution.sv \
  ../../rtl/store.sv \
  ../../rtl/pe.sv \
  ../../rtl/scale_calculator.sv \
  ../../rtl/quantizer_pipeline.sv \
  ../../rtl/fpga_modules/Gowin_RAM16SDP_Mock.sv \
  neural_network_tb.cpp \
  -CFLAGS "-std=c++17" \
  -o neural_network_test > build.log 2>&1

echo "Running neural_network_test..."
./obj_dir/neural_network_test

echo "----------------------------------------------------------------"
echo "=== All Working Unit Tests PASSED ==="
echo "----------------------------------------------------------------"
