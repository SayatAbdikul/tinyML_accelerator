"""
Minimal cocotb testbench for top_gemv quantization debug
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer
import numpy as np
import sys
import os

# Add compiler path for golden model imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../compiler'))
from helper_functions import quantize_int32_to_int8


@cocotb.test()
async def test_top_gemv_quantization(dut):
    """
    Test top_gemv quantization with simple vectors:
    - x = [1, 2, 3, 4, 5, ...]
    - w_row = [1, 1, 1, 1, 1, ...] (all ones)
    - bias = [10]
    - Expected: sum(x) + bias = (1+2+3+...+32) + 10 = 528 + 10 = 538 (before quant)
    """
    
    cocotb.log.info("=== Test: top_gemv Quantization Debug ===")
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start(start_high=False))
    
    # Reset
    dut.rst.value = 1
    dut.start.value = 0
    await FallingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.rst.value = 0
    await FallingEdge(dut.clk)
    
    # Setup inputs
    # Create simple test vectors
    x_vec = np.array([i + 1 for i in range(32)], dtype=np.int8)  # [1, 2, 3, ..., 32]
    w_tile = np.array([1 for _ in range(32)], dtype=np.int8)     # All ones
    bias_vec = np.array([10], dtype=np.int8)                     # Bias = 10
    
    # Set inputs
    for i in range(32):
        dut.x[i].value = int(x_vec[i])
    for i in range(32):
        dut.w_tile_row_in[i].value = int(w_tile[i])
    dut.bias[0].value = int(bias_vec[0])
    
    dut.rows.value = 1
    dut.cols.value = 32
    
    cocotb.log.info(f"Input x: {x_vec[:8]}... (sum={np.sum(x_vec)})")
    cocotb.log.info(f"Weight row: all 1s")
    cocotb.log.info(f"Bias: {bias_vec[0]}")
    
    # Compute expected result (golden)
    # Raw accumulation: sum(x * w) + bias = sum(1*i) + 10 = 528 + 10 = 538
    raw_sum = int(np.sum(x_vec.astype(np.int32) * w_tile.astype(np.int32))) + int(bias_vec[0])
    cocotb.log.info(f"Golden model: raw_sum = {raw_sum}")
    
    # Find max absolute value
    max_abs_val = abs(raw_sum)
    cocotb.log.info(f"Max abs value: {max_abs_val}")
    
    # Quantize using golden quantizer
    int32_array = np.array([raw_sum], dtype=np.int32)
    scale = max_abs_val / 127
    golden_quantized = quantize_int32_to_int8(int32_array, scale, 0)
    cocotb.log.info(f"Golden quantized (scale={scale:.4f}): {golden_quantized[0]}")
    
    # Start GEMV
    dut.start.value = 1
    await FallingEdge(dut.clk)
    dut.start.value = 0
    
    # Wait for first weight
    for _ in range(100):
        if dut.w_ready.value:
            cocotb.log.info("✓ W_ready asserted")
            break
        await FallingEdge(dut.clk)
    
    # Send weight tile
    dut.w_valid.value = 1
    await FallingEdge(dut.clk)
    dut.w_valid.value = 0
    
    # Wait for done
    for cycle in range(10000):
        if dut.done.value:
            cocotb.log.info(f"✓ Done asserted after {cycle} cycles")
            break
        await FallingEdge(dut.clk)
    else:
        cocotb.log.error("❌ GEMV timed out!")
        assert False, "GEMV timed out"
    
    # Read output
    await FallingEdge(dut.clk)
    rtl_output = int(dut.y[0].value)
    if rtl_output & 0x80:  # Handle signed conversion
        rtl_output = rtl_output - 256
    
    cocotb.log.info(f"\n📊 Results:")
    cocotb.log.info(f"  Golden quantized: {golden_quantized[0]}")
    cocotb.log.info(f"  RTL output:       {rtl_output}")
    cocotb.log.info(f"  Error:            {abs(rtl_output - golden_quantized[0])}")
    
    # Check match
    if abs(rtl_output - golden_quantized[0]) <= 2:
        cocotb.log.info("✅ TEST PASSED!")
    else:
        cocotb.log.error("❌ TEST FAILED - Output mismatch")
        assert False, f"Mismatch: RTL={rtl_output}, Golden={golden_quantized[0]}"

