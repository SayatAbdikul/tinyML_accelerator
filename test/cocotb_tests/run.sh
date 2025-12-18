#!/bin/bash
# Quick test script for TinyML Accelerator Cocotb verification
# This script runs the complete verification flow

echo "========================================================================"
echo "  TinyML Accelerator - Golden Model Verification"
echo "========================================================================"
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Check if cocotb is installed
if ! python3 -c "import cocotb" 2>/dev/null; then
    echo "❌ Cocotb not installed. Installing..."
    pip3 install -r requirements.txt
fi

# Check if dram.hex exists
if [ ! -f "../../compiler/dram.hex" ]; then
    echo "⚠️  dram.hex not found. Generating..."
    cd ../../compiler
    python3 main.py
    cd -
fi

echo ""
echo "Starting verification test..."
echo ""

# Run the test
make run_test

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "  ✅ VERIFICATION SUCCESSFUL"
    echo "========================================================================"
    echo ""
    echo "Generated files:"
    echo "  - Waveforms: dump.vcd (view with: gtkwave dump.vcd)"
    echo "  - Results: results.xml"
    echo ""
else
    echo ""
    echo "========================================================================"
    echo "  ❌ VERIFICATION FAILED"
    echo "========================================================================"
    echo ""
    echo "Check the output above for error details."
    echo ""
    exit 1
fi
