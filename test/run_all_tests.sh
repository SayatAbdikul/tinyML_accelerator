#!/bin/bash
# Run both test suites in sequence
# Usage: ./run_all_tests.sh [quick|full]

set -e  # Exit on error

MODE=${1:-quick}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "================================================================================"
echo "TinyML Accelerator - Comprehensive Test Suite"
echo "================================================================================"
echo "Mode: $MODE"
echo ""

# Track results
BASIC_RESULT=0
HEAVY_RESULT=0

# Run basic tests
echo "================================================================================"
echo "PHASE 1: Running Basic Tests (cocotb_tests)"
echo "================================================================================"
cd "$SCRIPT_DIR/cocotb_tests"
if make TEST_TARGET=golden_comparison run_test; then
    echo "✓ Basic tests PASSED"
    BASIC_RESULT=0
else
    echo "✗ Basic tests FAILED"
    BASIC_RESULT=1
fi
echo ""

# Run heavy tests based on mode
echo "================================================================================"
echo "PHASE 2: Running Heavy Tests (heavy_test)"
echo "================================================================================"
cd "$SCRIPT_DIR/heavy_test"

if [ "$MODE" = "full" ]; then
    echo "Running FULL heavy test (10,000 images - this will take 30-60 minutes)..."
    if make run_test; then
        echo "✓ Heavy tests PASSED"
        HEAVY_RESULT=0
    else
        echo "✗ Heavy tests FAILED"
        HEAVY_RESULT=1
    fi
elif [ "$MODE" = "quick" ]; then
    echo "Running QUICK heavy test (100 images)..."
    if make quick_test; then
        echo "✓ Quick heavy tests PASSED"
        HEAVY_RESULT=0
    else
        echo "✗ Quick heavy tests FAILED"
        HEAVY_RESULT=1
    fi
else
    echo "Unknown mode: $MODE"
    echo "Usage: $0 [quick|full]"
    exit 1
fi
echo ""

# Summary
echo "================================================================================"
echo "TEST SUITE SUMMARY"
echo "================================================================================"
echo ""
if [ $BASIC_RESULT -eq 0 ]; then
    echo "✓ Basic Tests: PASSED"
else
    echo "✗ Basic Tests: FAILED"
fi

if [ $HEAVY_RESULT -eq 0 ]; then
    echo "✓ Heavy Tests: PASSED"
else
    echo "✗ Heavy Tests: FAILED"
fi
echo ""

if [ $BASIC_RESULT -eq 0 ] && [ $HEAVY_RESULT -eq 0 ]; then
    echo "================================================================================"
    echo "✓✓✓ ALL TESTS PASSED ✓✓✓"
    echo "================================================================================"
    echo "The RTL implementation is validated and ready for production."
    exit 0
else
    echo "================================================================================"
    echo "✗✗✗ SOME TESTS FAILED ✗✗✗"
    echo "================================================================================"
    echo "Please review the test output above and fix the failures."
    exit 1
fi
