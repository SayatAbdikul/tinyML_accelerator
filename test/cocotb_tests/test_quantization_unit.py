"""
Cocotb Testbench for Quantization Module
==========================================

Compares RTL quantization module against Python golden model using random test inputs.

Test Flow:
1. Generate random int32 test values
2. For each test value and random scale/zero_point:
   a. Apply Python quantize_int32_to_int8 function
   b. Apply RTL quantization module
   c. Compare results
3. Report statistics on matches and mismatches
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import numpy as np
import sys
import os

# Add compiler directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../compiler'))
from helper_functions import quantize_int32_to_int8


class QuantizationTester:
    """Helper class for quantization module testing"""
    
    def __init__(self, dut):
        self.dut = dut
        self.clock_period = 10  # 10ns = 100MHz
        
    async def reset(self):
        """Apply reset to the module"""
        self.dut.reset_n.value = 0
        await RisingEdge(self.dut.clk)
        await RisingEdge(self.dut.clk)
        self.dut.reset_n.value = 1
        await RisingEdge(self.dut.clk)
        cocotb.log.info("Reset complete")
    
    async def calibrate(self, max_abs_value, timeout_cycles=1000):
        """
        Perform calibration with given max_abs value.
        Returns True if calibration succeeds, False on timeout.
        """
        self.dut.start_calib.value = 1
        self.dut.max_abs.value = max_abs_value
        await RisingEdge(self.dut.clk)
        self.dut.start_calib.value = 0
        
        # Wait for calibration to complete
        cycle_count = 0
        while cycle_count < timeout_cycles:
            await RisingEdge(self.dut.clk)
            if self.dut.calib_ready.value == 1:
                cocotb.log.info(f"Calibration complete after {cycle_count} cycles (max_abs={max_abs_value})")
                return True
            cycle_count += 1
        
        cocotb.log.error(f"Calibration timeout after {timeout_cycles} cycles")
        return False
    
    async def quantize_value(self, int32_value, timeout_cycles=100):
        """
        Quantize a single int32 value.
        Returns the int8 result or None on timeout.
        """
        # Ensure it's in int32 range
        int32_value = int(np.int32(int32_value))
        
        self.dut.data_in.value = int32_value
        self.dut.data_valid.value = 1
        await RisingEdge(self.dut.clk)
        self.dut.data_valid.value = 0
        
        # Wait for result
        cycle_count = 0
        while cycle_count < timeout_cycles:
            await RisingEdge(self.dut.clk)
            if self.dut.data_valid_out.value == 1:
                # RTL data_out is already signed int8
                result = self.dut.data_out.value.signed_integer
                cocotb.log.debug(f"Quantization complete after {cycle_count} cycles: {int32_value} -> {result}")
                return result
            cycle_count += 1
        
        cocotb.log.error(f"Quantization timeout after {timeout_cycles} cycles for input {int32_value}")
        return None
    
    def python_quantize(self, x_int32, scale, zero_point):
        """Apply Python quantization function"""
        # Convert to numpy array if needed
        x_arr = np.array([x_int32], dtype=np.int32)
        result = quantize_int32_to_int8(x_arr, scale, zero_point)
        return result[0]


@cocotb.test()
async def test_quantization_random_inputs(dut):
    """
    Main test: Compare RTL vs Python quantization on random inputs
    
    Tests:
    - Random int32 values (full range)
    - Fixed scales and zero points
    - Verify exact match between RTL and Python
    """
    
    tester = QuantizationTester(dut)
    
    # Start clock
    cocotb.log.info("Starting 100MHz clock")
    clock = Clock(dut.clk, tester.clock_period, units="ns")
    cocotb.start_soon(clock.start())
    
    await tester.reset()
    
    # Test parameters
    num_tests = 500
    seed = 42
    np.random.seed(seed)
    
    cocotb.log.info("\n" + "=" * 70)
    cocotb.log.info("QUANTIZATION MODULE VERIFICATION")
    cocotb.log.info("=" * 70)
    cocotb.log.info(f"Testing with {num_tests} random inputs")
    
    # Generate test cases with different scales
    test_cases = []
    scales = [1.0, 10.0, 100.0, 0.1, 0.01]
    zero_points = [0, 1, -1, 10, -10]
    
    # For each scale, generate test values
    for scale in scales:
        for zero_point in zero_points:
            max_abs = int(128 * scale)
            test_cases.append({
                'scale': scale,
                'zero_point': zero_point,
                'max_abs': max_abs,
                'count': num_tests // len(scales) // len(zero_points)
            })
    
    total_tests = 0
    matches = 0
    mismatches = 0
    mismatch_details = []
    
    for test_case_idx, test_case in enumerate(test_cases):
        scale = test_case['scale']
        zero_point = test_case['zero_point']
        max_abs = test_case['max_abs']
        num_values = test_case['count']
        
        cocotb.log.info(f"\n--- Test Case {test_case_idx + 1}: scale={scale}, zero_point={zero_point}, max_abs={max_abs} ---")
        
        # Calibrate with this scale
        if not await tester.calibrate(int(max_abs), timeout_cycles=2000):
            cocotb.log.error(f"Calibration failed for scale {scale}")
            continue
        
        # Wait a few cycles for ready
        for _ in range(10):
            await RisingEdge(dut.clk)
        
        # Generate and test random values
        for val_idx in range(num_values):
            # Generate random int32 value
            int32_val = np.random.randint(-2**31, 2**31-1, dtype=np.int32)
            
            # Python quantization
            python_result = tester.python_quantize(int32_val, scale, zero_point)
            
            # RTL quantization
            rtl_result = await tester.quantize_value(int32_val, timeout_cycles=200)
            
            total_tests += 1
            
            if rtl_result is None:
                cocotb.log.error(f"RTL returned None for input {int32_val}")
                mismatches += 1
                continue
            
            # Compare
            if rtl_result == python_result:
                matches += 1
                cocotb.log.debug(f"✓ Match: {int32_val} -> RTL:{rtl_result} == Python:{python_result}")
            else:
                mismatches += 1
                diff = rtl_result - python_result
                cocotb.log.warning(
                    f"✗ MISMATCH: {int32_val} -> RTL:{rtl_result} != Python:{python_result} (diff={diff})"
                )
                mismatch_details.append({
                    'input': int32_val,
                    'scale': scale,
                    'zero_point': zero_point,
                    'rtl_result': rtl_result,
                    'python_result': python_result,
                    'diff': diff
                })
    
    # Report results
    cocotb.log.info("\n" + "=" * 70)
    cocotb.log.info("TEST RESULTS")
    cocotb.log.info("=" * 70)
    cocotb.log.info(f"Total tests:  {total_tests}")
    cocotb.log.info(f"Matches:      {matches} ({matches/total_tests*100:.1f}%)")
    cocotb.log.info(f"Mismatches:   {mismatches} ({mismatches/total_tests*100:.1f}%)")
    
    if mismatch_details:
        cocotb.log.warning("\n" + "=" * 70)
        cocotb.log.warning("MISMATCH DETAILS")
        cocotb.log.warning("=" * 70)
        for detail in mismatch_details[:10]:  # Show first 10
            cocotb.log.warning(
                f"Input: {detail['input']:12d} | Scale: {detail['scale']:8.2f} | "
                f"Zero: {detail['zero_point']:3d} | RTL: {detail['rtl_result']:4d} | "
                f"Python: {detail['python_result']:4d} | Diff: {detail['diff']:4d}"
            )
        if len(mismatch_details) > 10:
            cocotb.log.warning(f"... and {len(mismatch_details) - 10} more mismatches")
    
    # Final assertion
    if matches == total_tests:
        cocotb.log.info("\n✅ PASS: All tests matched!")
        assert True
    else:
        cocotb.log.error(f"\n❌ FAIL: {mismatches} mismatches found")
        assert False, f"Quantization mismatch: {mismatches}/{total_tests} failures"


@cocotb.test()
async def test_quantization_edge_cases(dut):
    """
    Test edge cases: min/max values, boundaries, etc.
    """
    
    tester = QuantizationTester(dut)
    
    # Start clock
    clock = Clock(dut.clk, tester.clock_period, units="ns")
    cocotb.start_soon(clock.start())
    
    await tester.reset()
    
    cocotb.log.info("\n" + "=" * 70)
    cocotb.log.info("EDGE CASE TESTS")
    cocotb.log.info("=" * 70)
    
    # Edge case test values
    edge_cases = [
        # (value, scale, zero_point, description)
        (0, 1.0, 0, "Zero input"),
        (127, 1.0, 0, "Max positive int8"),
        (-128, 1.0, 0, "Min negative int8"),
        (2**31 - 1, 1000.0, 0, "Max int32"),
        (-(2**31), 1000.0, 0, "Min int32"),
        (100, 10.0, 0, "Small scale"),
        (100, 0.1, 0, "Very small scale"),
        (100, 1.0, 10, "Positive zero_point"),
        (100, 1.0, -10, "Negative zero_point"),
    ]
    
    matches = 0
    mismatches = 0
    
    for test_idx, (value, scale, zero_point, description) in enumerate(edge_cases):
        cocotb.log.info(f"\nTest {test_idx + 1}: {description}")
        cocotb.log.info(f"  Input: {value}, Scale: {scale}, Zero-Point: {zero_point}")
        
        max_abs = int(128 * scale)
        if not await tester.calibrate(max_abs, timeout_cycles=2000):
            cocotb.log.error(f"Calibration failed")
            continue
        
        for _ in range(10):
            await RisingEdge(dut.clk)
        
        # Python result
        python_result = tester.python_quantize(value, scale, zero_point)
        
        # RTL result
        rtl_result = await tester.quantize_value(value, timeout_cycles=200)
        
        if rtl_result is None:
            cocotb.log.error("RTL timeout")
            mismatches += 1
            continue
        
        if rtl_result == python_result:
            matches += 1
            cocotb.log.info(f"  ✓ MATCH: {value} -> {rtl_result}")
        else:
            mismatches += 1
            cocotb.log.warning(f"  ✗ MISMATCH: {value} -> RTL: {rtl_result}, Python: {python_result}")
    
    cocotb.log.info("\n" + "=" * 70)
    cocotb.log.info(f"Edge Case Results: {matches} matches, {mismatches} mismatches")
    cocotb.log.info("=" * 70)
    
    if mismatches == 0:
        cocotb.log.info("✅ PASS: All edge cases passed!")
    else:
        cocotb.log.error(f"❌ FAIL: {mismatches} edge cases failed")
        assert False, f"Edge case failures: {mismatches}/{len(edge_cases)}"
