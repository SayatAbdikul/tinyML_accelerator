#!/bin/bash
# Quick runner for execution unit tests
# Usage: ./run_execution_unit_tests.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Execution Unit Testbench Runner"
echo "=========================================="
echo ""

# Check dependencies
if ! command -v cocotb-config &> /dev/null; then
    echo "❌ Error: cocotb not found"
    echo "   Install: pip install cocotb"
    exit 1
fi

if ! command -v verilator &> /dev/null; then
    echo "❌ Error: verilator not found"
    echo "   macOS: brew install verilator"
    echo "   Linux: sudo apt-get install verilator"
    exit 1
fi

echo "✅ Dependencies OK"
echo ""

# Check for dram.hex
if [ ! -f "../../compiler/dram.hex" ]; then
    echo "⚠️  dram.hex not found. Generating..."
    cd ../../compiler
    python3 main.py
    cd -
fi

echo "✅ dram.hex found"
echo ""

# Activate venv if it exists
if [ -f ".venv/bin/activate" ]; then
    source ".venv/bin/activate"
    echo "✅ Virtual environment activated"
else
    echo "⚠️  No venv found at .venv/"
fi

echo ""
echo "Running execution unit tests..."
echo "  - test_neural_network_complete (13 instructions)"
echo "  - test_single_load_v"
echo "  - test_single_gemv"
echo ""

# Run tests
make TEST_TARGET=execution_unit run_test

# Check results
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ All tests PASSED!"
    echo "=========================================="
    echo ""
    echo "Waveform: dump.vcd"
    echo "View with: gtkwave dump.vcd"
else
    echo ""
    echo "=========================================="
    echo "❌ Tests FAILED"
    echo "=========================================="
    echo ""
    echo "Check sim_build/results.xml for details"
    exit 1
fi
