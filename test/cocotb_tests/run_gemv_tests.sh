#!/bin/bash
# Quick start script for GEMV testbench
# Usage: ./run_gemv_tests.sh

set -e  # Exit on any error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "TinyML Accelerator - GEMV Testbench"
echo "=========================================="
echo ""

# Check for cocotb
if ! command -v cocotb-run &> /dev/null; then
    echo "❌ Error: cocotb not found. Please install:"
    echo "   pip install cocotb"
    exit 1
fi

# Check for verilator
if ! command -v verilator &> /dev/null; then
    echo "❌ Error: verilator not found. Please install:"
    echo "   brew install verilator  # macOS"
    echo "   sudo apt-get install verilator  # Ubuntu/Debian"
    exit 1
fi

# Check for numpy
python3 -c "import numpy" 2>/dev/null || {
    echo "❌ Error: numpy not found. Please install:"
    echo "   pip install numpy"
    exit 1
}

echo "✅ All dependencies found"
echo ""

# Clean previous runs (optional, uncomment to enable)
# echo "Cleaning previous builds..."
# make clean_all > /dev/null 2>&1

# Run GEMV tests
echo "Running GEMV testbench with 3 test cases..."
echo "  1. test_top_gemv_small (4×8 matrix)"
echo "  2. test_top_gemv_medium (16×32 matrix)"
echo "  3. test_top_gemv_with_quantization_check (2×4 fixed)"
echo ""

make TEST_TARGET=top_gemv run_test

# Check results
if [ -f "sim_build/results.xml" ]; then
    echo ""
    echo "=========================================="
    echo "✅ Tests completed!"
    echo "=========================================="
    echo ""
    echo "Waveform saved to: dump.vcd"
    echo "View with:        gtkwave dump.vcd"
    echo ""
    
    # Try to parse results
    if grep -q "failures=\"0\"" sim_build/results.xml; then
        echo "All tests PASSED ✅"
    else
        echo "Some tests may have FAILED ⚠️"
        echo "Check sim_build/results.xml for details"
    fi
else
    echo "❌ Tests failed - no results.xml generated"
    exit 1
fi
