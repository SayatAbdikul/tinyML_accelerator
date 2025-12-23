"""
Cocotb Testbench for top_gemv Module
=====================================

This testbench validates the top_gemv RTL module against the Python golden model
by running equivalent GEMV operations and comparing quantized outputs.

Test Flow:
1. Generate random int8 weights, inputs, and biases.
2. Execute golden model GEMV (compute int32, quantize to int8).
3. Drive RTL top_gemv with the same inputs (tile-by-tile).
4. Read RTL quantized output.
5. Compare against golden model output (exact match expected).
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer, FallingEdge
import numpy as np
import sys
import os

# Add compiler path for golden model imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../compiler'))
from helper_functions import quantize_int32_to_int8


class GoldenGEMV:
    """Python reference implementation matching golden_model.py GEMV logic."""
    
    @staticmethod
    def gemv(weights, x, bias, rows, cols):
        """
        Perform GEMV: y = W @ x + bias
        
        Args:
            weights: int8 array of shape (rows, cols)
            x: int8 array of shape (cols,)
            bias: int8 array of shape (rows,)
            rows: number of rows
            cols: number of columns
            
        Returns:
            y_quantized: int8 array of shape (rows,)
        """
        # Ensure inputs are numpy arrays with correct types
        W = np.array(weights, dtype=np.int8).reshape(rows, cols)
        X = np.array(x, dtype=np.int8)
        B = np.array(bias, dtype=np.int8)
        
        # Compute GEMV in int32
        y_int32 = np.zeros(rows, dtype=np.int32)
        for i in range(rows):
            y_int32[i] = np.int32(0)
            for j in range(cols):
                y_int32[i] += np.int32(W[i, j]) * np.int32(X[j])
            y_int32[i] += np.int32(B[i])
        
        # Quantize to int8 (matching golden model's store behavior)
        # First quantization in GEMV: scale by max abs value
        max_abs = np.max(np.abs(y_int32))
        if max_abs == 0:
            max_abs = 1  # avoid div by zero
        scale = max_abs / 127.0
        
        y_quantized = quantize_int32_to_int8(y_int32, scale, 0)
        
        return y_int32, y_quantized, scale


@cocotb.test()
async def test_top_gemv_small(dut):
    """Test top_gemv with a small 4x8 matrix."""
    
    cocotb.log.info("=== Test: top_gemv_small (4x8 matrix) ===")
    
    # Test parameters
    rows = 4
    cols = 8
    tile_size = 32
    
    # Generate random inputs (int8)
    np.random.seed(42)
    weights_flat = np.random.randint(-128, 127, rows * cols, dtype=np.int8)
    x_input = np.random.randint(-128, 127, cols, dtype=np.int8)
    bias = np.random.randint(-128, 127, rows, dtype=np.int8)
    
    cocotb.log.info(f"Input shape: rows={rows}, cols={cols}")
    cocotb.log.info(f"Weights sample: {weights_flat[:8]}")
    cocotb.log.info(f"X input: {x_input}")
    cocotb.log.info(f"Bias: {bias}")
    
    # === Golden Model ===
    golden = GoldenGEMV()
    y_int32_golden, y_quantized_golden, scale_golden = golden.gemv(
        weights_flat, x_input, bias, rows, cols
    )
    
    cocotb.log.info(f"Golden model int32 output: {y_int32_golden}")
    cocotb.log.info(f"Golden model scale: {scale_golden:.6f}")
    cocotb.log.info(f"Golden model quantized output: {y_quantized_golden}")
    
    # === RTL Setup ===
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start(start_high=False))
    
    # Reset
    dut.rst.value = 1
    dut.start.value = 0
    dut.w_valid.value = 0
    await FallingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.rst.value = 0
    await FallingEdge(dut.clk)
    
    # Set input dimensions
    dut.rows.value = rows
    dut.cols.value = cols
    
    # Load x and bias into RTL inputs
    for i in range(cols):
        dut.x[i].value = int(x_input[i])
    for i in range(rows):
        dut.bias[i].value = int(bias[i])
    
    cocotb.log.info("Inputs loaded to RTL")
    
    # === Feed weights tile by tile ===
    await FallingEdge(dut.clk)
    dut.start.value = 1
    await FallingEdge(dut.clk)
    dut.start.value = 0
    
    # Process weights row-by-row (each row is a tile of size cols)
    # top_gemv expects weights in TILE_SIZE-sized chunks
    tile_count = 0
    for row_idx in range(rows):
        # Wait for w_ready
        for _ in range(100):
            if dut.w_ready.value:
                break
            await FallingEdge(dut.clk)
        else:
            cocotb.log.error(f"Timeout waiting for w_ready at row {row_idx}")
            assert False, "w_ready not asserted"
        
        # Prepare weight tile (one row of the matrix)
        weight_tile = weights_flat[row_idx * cols:(row_idx + 1) * cols]
        
        # Pad to TILE_SIZE if necessary
        weight_tile_padded = np.zeros(tile_size, dtype=np.int8)
        weight_tile_padded[:cols] = weight_tile
        
        # Load into RTL
        for i in range(tile_size):
            dut.w_tile_row_in[i].value = int(weight_tile_padded[i])
        
        dut.w_valid.value = 1
        await FallingEdge(dut.clk)
        dut.w_valid.value = 0
        
        cocotb.log.info(f"Weight tile {tile_count} loaded (row {row_idx})")
        tile_count += 1
    
    # === Wait for RTL to complete ===
    timeout = 50000
    for cycle in range(timeout):
        if dut.done.value:
            cocotb.log.info(f"RTL done asserted at cycle {cycle}")
            break
        await FallingEdge(dut.clk)
    else:
        cocotb.log.error(f"Timeout waiting for done signal after {timeout} cycles")
        assert False, "Timeout"
    
    # === Read RTL outputs ===
    await FallingEdge(dut.clk)
    y_rtl = np.zeros(rows, dtype=np.int8)
    for i in range(rows):
        val = int(dut.y[i].value)
        # Convert from 8-bit signed logic to Python int
        if val & 0x80:  # negative bit set
            val = val - 256
        y_rtl[i] = val
    
    cocotb.log.info(f"RTL output: {y_rtl}")
    
    # === Compare ===
    max_error = np.max(np.abs(y_rtl.astype(np.int32) - y_quantized_golden.astype(np.int32)))
    cocotb.log.info(f"Max error: {max_error}")
    
    if max_error == 0:
        cocotb.log.info("✅ Test PASSED: Outputs match (within tolerance)")
    else:
        cocotb.log.error(f"❌ Test FAILED: Max error {max_error} > 0")
        cocotb.log.error(f"Expected: {y_quantized_golden}")
        cocotb.log.error(f"Got:      {y_rtl}")
        assert False, f"Output mismatch: max error {max_error}"


@cocotb.test()
async def test_top_gemv_medium(dut):
    """Test top_gemv with a larger 16x32 matrix."""
    
    cocotb.log.info("=== Test: top_gemv_medium (16x32 matrix) ===")
    
    # Test parameters
    rows = 16
    cols = 32
    tile_size = 32
    
    # Generate random inputs (int8)
    np.random.seed(123)
    weights_flat = np.random.randint(-128, 127, rows * cols, dtype=np.int8)
    x_input = np.random.randint(-128, 127, cols, dtype=np.int8)
    bias = np.random.randint(-128, 127, rows, dtype=np.int8)
    
    cocotb.log.info(f"Input shape: rows={rows}, cols={cols}")
    
    # === Golden Model ===
    golden = GoldenGEMV()
    y_int32_golden, y_quantized_golden, scale_golden = golden.gemv(
        weights_flat, x_input, bias, rows, cols
    )
    
    cocotb.log.info(f"Golden model scale: {scale_golden:.6f}")
    cocotb.log.info(f"Golden model quantized output (first 8): {y_quantized_golden[:8]}")
    
    # === RTL Setup ===
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start(start_high=False))
    
    # Reset
    dut.rst.value = 1
    dut.start.value = 0
    dut.w_valid.value = 0
    await FallingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.rst.value = 0
    await FallingEdge(dut.clk)
    
    # Set input dimensions
    dut.rows.value = rows
    dut.cols.value = cols
    
    # Load x and bias into RTL inputs
    for i in range(cols):
        dut.x[i].value = int(x_input[i])
    for i in range(rows):
        dut.bias[i].value = int(bias[i])
    
    # === Feed weights row by row ===
    await FallingEdge(dut.clk)
    dut.start.value = 1
    await FallingEdge(dut.clk)
    dut.start.value = 0
    
    for row_idx in range(rows):
        # Wait for w_ready
        for _ in range(100):
            if dut.w_ready.value:
                break
            await FallingEdge(dut.clk)
        else:
            cocotb.log.error(f"Timeout waiting for w_ready at row {row_idx}")
            assert False, "w_ready not asserted"
        
        # Prepare weight tile
        weight_tile = weights_flat[row_idx * cols:(row_idx + 1) * cols]
        weight_tile_padded = np.zeros(tile_size, dtype=np.int8)
        weight_tile_padded[:cols] = weight_tile
        
        # Load into RTL
        for i in range(tile_size):
            dut.w_tile_row_in[i].value = int(weight_tile_padded[i])
        
        dut.w_valid.value = 1
        await FallingEdge(dut.clk)
        dut.w_valid.value = 0
    
    # === Wait for RTL to complete ===
    timeout = 100000
    for cycle in range(timeout):
        if dut.done.value:
            cocotb.log.info(f"RTL done asserted at cycle {cycle}")
            break
        await FallingEdge(dut.clk)
    else:
        cocotb.log.error(f"Timeout after {timeout} cycles")
        assert False, "Timeout"
    
    # === Read RTL outputs ===
    await FallingEdge(dut.clk)
    y_rtl = np.zeros(rows, dtype=np.int8)
    for i in range(rows):
        val = int(dut.y[i].value)
        if val & 0x80:
            val = val - 256
        y_rtl[i] = val
    
    cocotb.log.info(f"RTL quantized output (first 8): {y_rtl[:8]}")
    
    # === Compare ===
    max_error = np.max(np.abs(y_rtl.astype(np.int32) - y_quantized_golden.astype(np.int32)))
    cocotb.log.info(f"Max error: {max_error}")
    
    if max_error == 0:
        cocotb.log.info("✅ Test PASSED")
    else:
        cocotb.log.error(f"❌ Test FAILED: Max error {max_error} > 0")
        assert False, f"Output mismatch: max error {max_error}"


@cocotb.test()
async def test_top_gemv_with_quantization_check(dut):
    """Test top_gemv and explicitly verify quantization behavior."""
    
    cocotb.log.info("=== Test: top_gemv with quantization verification ===")
    
    # Smaller test for detailed inspection
    rows = 2
    cols = 4
    tile_size = 32
    
    # Use fixed values for reproducibility
    weights_flat = np.array([10, 20, 30, 40, -50, -60, 70, -80], dtype=np.int8)
    x_input = np.array([5, 10, 15, 20], dtype=np.int8)
    bias = np.array([100, -100], dtype=np.int8)
    
    cocotb.log.info(f"Fixed test case:")
    cocotb.log.info(f"  Weights (2x4): {weights_flat.reshape(2, 4)}")
    cocotb.log.info(f"  X: {x_input}")
    cocotb.log.info(f"  Bias: {bias}")
    
    # === Golden Model ===
    golden = GoldenGEMV()
    y_int32_golden, y_quantized_golden, scale_golden = golden.gemv(
        weights_flat, x_input, bias, rows, cols
    )
    
    cocotb.log.info(f"Golden int32 result: {y_int32_golden}")
    cocotb.log.info(f"Golden scale: {scale_golden:.6f}")
    cocotb.log.info(f"Golden quantized: {y_quantized_golden}")
    
    # === RTL ===
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start(start_high=False))
    
    dut.rst.value = 1
    dut.start.value = 0
    dut.w_valid.value = 0
    await FallingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.rst.value = 0
    await FallingEdge(dut.clk)
    
    dut.rows.value = rows
    dut.cols.value = cols
    
    for i in range(cols):
        dut.x[i].value = int(x_input[i])
    for i in range(rows):
        dut.bias[i].value = int(bias[i])
    
    await FallingEdge(dut.clk)
    dut.start.value = 1
    await FallingEdge(dut.clk)
    dut.start.value = 0
    
    for row_idx in range(rows):
        for _ in range(100):
            if dut.w_ready.value:
                break
            await FallingEdge(dut.clk)
        
        weight_tile = weights_flat[row_idx * cols:(row_idx + 1) * cols]
        weight_tile_padded = np.zeros(tile_size, dtype=np.int8)
        weight_tile_padded[:cols] = weight_tile
        
        for i in range(tile_size):
            dut.w_tile_row_in[i].value = int(weight_tile_padded[i])
        
        dut.w_valid.value = 1
        await FallingEdge(dut.clk)
        dut.w_valid.value = 0
    
    timeout = 100000
    for cycle in range(timeout):
        if dut.done.value:
            break
        await FallingEdge(dut.clk)
    else:
        assert False, "Timeout"
    
    await FallingEdge(dut.clk)
    y_rtl = np.zeros(rows, dtype=np.int8)
    for i in range(rows):
        val = int(dut.y[i].value)
        if val & 0x80:
            val = val - 256
        y_rtl[i] = val
    
    cocotb.log.info(f"RTL quantized: {y_rtl}")
    
    # Detailed comparison
    for i in range(rows):
        error = abs(int(y_rtl[i]) - int(y_quantized_golden[i]))
        cocotb.log.info(f"  Row {i}: RTL={y_rtl[i]}, Golden={y_quantized_golden[i]}, Error={error}")
    
    max_error = np.max(np.abs(y_rtl.astype(np.int32) - y_quantized_golden.astype(np.int32)))
    
    if max_error == 0:
        cocotb.log.info("✅ Test PASSED")
    else:
        cocotb.log.error(f"❌ Test FAILED: Max error {max_error}")
        assert False
