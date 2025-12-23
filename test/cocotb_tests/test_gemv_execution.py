"""
Cocotb Testbench for gemv_execution Module
==========================================

This testbench validates the gemv_execution RTL module by comparing its output
against the Python golden model's GEMV function.

The gemv_execution module:
1. Reads x vector tiles from buffer
2. Reads bias vector tiles from buffer  
3. Reads weight matrix tiles and streams to GEMV unit
4. Waits for GEMV computation to complete
5. Writes results back to destination buffer

The module interfaces with buffer_controller for data access.

Test Cases:
1. Small GEMV (4x4 matrix)
2. Medium GEMV (12x32 - like Layer 2)
3. Large GEMV (12x784 - like Layer 1)
4. Edge cases with partial tiles
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer
import numpy as np
import sys
import os

# Add compiler path for golden model imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../compiler'))


class GEMVExecutionTester:
    """Helper class for gemv_execution module testing with buffer_controller."""
    
    TILE_SIZE = 32  # Elements per tile (256 bits / 8 bits)
    MAX_ROWS = 1024
    MAX_COLS = 1024
    
    def __init__(self, dut):
        self.dut = dut
        self.memory = None
        
        # Track buffer contents for verification
        self.buffers = {}
        
    def load_dram(self, hex_file):
        """Load DRAM contents from hex file into numpy int8 array."""
        if not os.path.exists(hex_file):
            raise FileNotFoundError(f"DRAM hex file not found: {hex_file}")
        with open(hex_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        # Convert unsigned hex bytes to signed int8
        vals = []
        for line in lines:
            b = int(line, 16)
            if b >= 128:
                b = b - 256
            vals.append(b)
        self.memory = np.array(vals, dtype=np.int8)
        cocotb.log.info(f"Loaded DRAM from {hex_file}: {len(self.memory)} bytes")
        
    async def reset(self):
        """Reset the DUT."""
        self.dut.rst.value = 1
        self.dut.start.value = 0
        self.dut.dest_buffer_id.value = 0
        self.dut.w_buffer_id.value = 0
        self.dut.x_buffer_id.value = 0
        self.dut.b_buffer_id.value = 0
        self.dut.cols.value = 0
        self.dut.rows.value = 0
        
        # Reset buffer controller inputs
        self.dut.vec_read_valid.value = 0
        self.dut.mat_read_valid.value = 0
        
        for i in range(self.TILE_SIZE):
            self.dut.vec_read_tile[i].value = 0
            self.dut.mat_read_tile[i].value = 0
        
        await FallingEdge(self.dut.clk)
        await FallingEdge(self.dut.clk)
        await FallingEdge(self.dut.clk)
        
        self.dut.rst.value = 0
        await FallingEdge(self.dut.clk)
        
        # Clear buffer tracking
        self.buffers = {}
        
    def load_buffer(self, buffer_id, data):
        """Load data into simulated buffer for test."""
        self.buffers[buffer_id] = np.array(data, dtype=np.int8)
        
    def get_tiles_from_buffer(self, buffer_id, num_elements):
        """Get tile data from simulated buffer."""
        if buffer_id not in self.buffers:
            return []
        data = self.buffers[buffer_id]
        tiles = []
        num_tiles = (num_elements + self.TILE_SIZE - 1) // self.TILE_SIZE
        for t in range(num_tiles):
            start = t * self.TILE_SIZE
            end = min(start + self.TILE_SIZE, len(data))
            tile = np.zeros(self.TILE_SIZE, dtype=np.int8)
            tile[:end-start] = data[start:end]
            tiles.append(tile)
        return tiles
        
    async def execute_gemv(self, dest_id, w_id, x_id, b_id, rows, cols, timeout=50000):
        """
        Execute a GEMV operation by acting as the buffer controller.
        
        This simulates the buffer controller providing data to gemv_execution.
        
        Returns:
            result: The result array from RTL
        """
        # Set up GEMV parameters
        self.dut.dest_buffer_id.value = dest_id
        self.dut.w_buffer_id.value = w_id
        self.dut.x_buffer_id.value = x_id
        self.dut.b_buffer_id.value = b_id
        self.dut.rows.value = rows
        self.dut.cols.value = cols
        
        # Calculate expected tiles - RTL uses row-major tiling
        # Each row is tiled separately: total_tiles = rows * tiles_per_row
        x_tiles = self.get_tiles_from_buffer(x_id, cols)
        b_tiles = self.get_tiles_from_buffer(b_id, rows)
        
        # For weight matrix, tiles are organized per row
        # Each row of the matrix (cols elements) is tiled, then next row
        tiles_per_row = (cols + self.TILE_SIZE - 1) // self.TILE_SIZE
        w_tiles = []
        w_data = self.buffers.get(w_id, np.array([], dtype=np.int8))
        for r in range(rows):
            for t in range(tiles_per_row):
                start = r * cols + t * self.TILE_SIZE
                end = min(start + self.TILE_SIZE, r * cols + cols)
                tile = np.zeros(self.TILE_SIZE, dtype=np.int8)
                tile_len = min(self.TILE_SIZE, end - start)
                if tile_len > 0 and start < len(w_data):
                    tile[:tile_len] = w_data[start:start+tile_len]
                w_tiles.append(tile)
        
        x_tile_idx = 0
        b_tile_idx = 0
        w_tile_idx = 0
        
        # Tracking state
        x_tiles_sent = 0
        b_tiles_sent = 0
        w_tiles_sent = 0
        total_x_tiles = len(x_tiles)
        total_b_tiles = len(b_tiles)
        total_w_tiles = len(w_tiles)
        
        cocotb.log.info(f"Starting GEMV: {rows}x{cols}, x_tiles={total_x_tiles}, b_tiles={total_b_tiles}, w_tiles={total_w_tiles}")
        
        # Start the operation
        await FallingEdge(self.dut.clk)
        self.dut.start.value = 1
        await FallingEdge(self.dut.clk)
        self.dut.start.value = 0
        
        # CRITICAL: The RTL sets vec_read_enable=1 on start, but it goes back to 0 on the next cycle
        # We need to detect the first read request RIGHT NOW (before entering the loop)
        # The vec_read_enable is already high at this point (after the rising edge where start was seen)
        vec_read_pipeline = [0, 0]  # 2-stage pipeline for vec reads
        mat_read_pipeline = [0, 0]  # 2-stage pipeline for mat reads
        
        # Track last enable state to detect edges
        last_vec_enable = 0
        last_mat_enable = 0
        
        # Capture the initial vec_read_enable that's already high
        if int(self.dut.vec_read_enable.value) == 1:
            vec_pending_buf_id = int(self.dut.vec_read_buffer_id.value)
            vec_read_pipeline[0] = 1  # Queue the first read
            last_vec_enable = 1
            cocotb.log.info(f"Initial vec_read_enable captured, buf_id={vec_pending_buf_id}")
        else:
            vec_pending_buf_id = 0
            
        result_tiles = []
        
        for cycle in range(timeout):
            await FallingEdge(self.dut.clk)
            
            # Sample current enable states
            curr_vec_enable = int(self.dut.vec_read_enable.value)
            curr_mat_enable = int(self.dut.mat_read_enable.value)
            
            # Shift pipeline first
            vec_read_valid_now = vec_read_pipeline[1]
            vec_read_pipeline[1] = vec_read_pipeline[0]
            vec_read_pipeline[0] = 0
            
            mat_read_valid_now = mat_read_pipeline[1]
            mat_read_pipeline[1] = mat_read_pipeline[0]
            mat_read_pipeline[0] = 0
            
            # Default: no valid
            self.dut.vec_read_valid.value = 0
            self.dut.mat_read_valid.value = 0
            
            # Provide vector data when pipeline says so (BEFORE checking for new requests)
            if vec_read_valid_now:
                if vec_pending_buf_id == x_id and x_tile_idx < total_x_tiles:
                    tile_data = x_tiles[x_tile_idx]
                    x_tile_idx += 1
                    x_tiles_sent += 1
                elif vec_pending_buf_id == b_id and b_tile_idx < total_b_tiles:
                    tile_data = b_tiles[b_tile_idx]
                    b_tile_idx += 1
                    b_tiles_sent += 1
                else:
                    tile_data = np.zeros(self.TILE_SIZE, dtype=np.int8)
                
                # Set tile data on DUT
                for i in range(self.TILE_SIZE):
                    self.dut.vec_read_tile[i].value = int(tile_data[i]) & 0xFF
                
                self.dut.vec_read_valid.value = 1
                if x_tiles_sent + b_tiles_sent <= 5:
                    cocotb.log.info(f"Cycle {cycle}: Sending vec_read_valid, buf={vec_pending_buf_id}, tile[0]={tile_data[0]}, x_sent={x_tiles_sent}, b_sent={b_tiles_sent}")
            
            # Provide matrix data when pipeline says so
            if mat_read_valid_now:
                if w_tile_idx < total_w_tiles:
                    tile_data = w_tiles[w_tile_idx]
                    w_tile_idx += 1
                    w_tiles_sent += 1
                else:
                    tile_data = np.zeros(self.TILE_SIZE, dtype=np.int8)
                    
                # Set tile data on DUT
                for i in range(self.TILE_SIZE):
                    self.dut.mat_read_tile[i].value = int(tile_data[i]) & 0xFF
                    
                self.dut.mat_read_valid.value = 1
                if w_tiles_sent <= 5:
                    cocotb.log.info(f"Cycle {cycle}: Sending mat_read_valid, w_sent={w_tiles_sent}")
            
            # Check for RISING EDGE of vector read request - only queue on 0->1 transition
            if curr_vec_enable == 1 and last_vec_enable == 0:
                vec_read_pipeline[0] = 1
                vec_pending_buf_id = int(self.dut.vec_read_buffer_id.value)
                if x_tiles_sent + b_tiles_sent <= 5:
                    cocotb.log.info(f"Cycle {cycle}: vec_read_enable rising edge, buf_id={vec_pending_buf_id}")
                
            # Check for RISING EDGE of matrix read request - only queue on 0->1 transition
            if curr_mat_enable == 1 and last_mat_enable == 0:
                mat_read_pipeline[0] = 1
                if w_tiles_sent <= 5:
                    cocotb.log.info(f"Cycle {cycle}: mat_read_enable rising edge")
            
            # Update last enable states
            last_vec_enable = curr_vec_enable
            last_mat_enable = curr_mat_enable
            
            # Capture write tile data
            if int(self.dut.vec_write_enable.value) == 1:
                tile_data = []
                for i in range(self.TILE_SIZE):
                    val = int(self.dut.vec_write_tile[i].value)
                    if val & 0x80:
                        val = val - 256
                    tile_data.append(val)
                result_tiles.append(tile_data)
                cocotb.log.info(f"Cycle {cycle}: Write tile captured")
                
            # Check for completion
            if int(self.dut.done.value) == 1:
                cocotb.log.info(f"✅ GEMV complete after {cycle} cycles")
                cocotb.log.info(f"   Tiles sent: x={x_tiles_sent}, b={b_tiles_sent}, w={w_tiles_sent}")
                cocotb.log.info(f"   Result tiles: {len(result_tiles)}")
                break
        else:
            cocotb.log.error(f"❌ GEMV timeout after {timeout} cycles")
            cocotb.log.error(f"   Tiles sent: x={x_tiles_sent}/{total_x_tiles}, b={b_tiles_sent}/{total_b_tiles}, w={w_tiles_sent}/{total_w_tiles}")
            return None
            
        # Extract result from result output
        result = []
        for i in range(rows):
            val = int(self.dut.result[i].value)
            if val & 0x80:
                val = val - 256
            result.append(val)
            
        return np.array(result, dtype=np.int8)
        
    def golden_gemv(self, w_data, x_data, b_data, rows, cols):
        """
        Compute GEMV using golden model algorithm.
        
        y[i] = sum(W[i,j] * x[j]) + b[i], then quantize to int8
        """
        # Ensure data is int8
        w = np.array(w_data, dtype=np.int8)
        x = np.array(x_data, dtype=np.int8)
        b = np.array(b_data, dtype=np.int8)
        
        # Compute int32 accumulation
        result = np.zeros(rows, dtype=np.int32)
        for i in range(rows):
            acc = np.int32(0)
            for j in range(cols):
                acc += np.int32(w[i * cols + j]) * np.int32(x[j])
            acc += np.int32(b[i])
            result[i] = acc
            
        # Quantize to int8 (same as golden model)
        max_abs = np.max(np.abs(result))
        if max_abs == 0:
            return np.zeros(rows, dtype=np.int8)
        scale = max_abs / 127.0
        quantized = np.clip(np.round(result / scale), -128, 127).astype(np.int8)
        
        return quantized


@cocotb.test()
async def test_gemv_execution_small(dut):
    """Test small GEMV: 4x4 matrix with known values."""
    
    cocotb.log.info("=" * 60)
    cocotb.log.info("TEST: gemv_execution small (4x4)")
    cocotb.log.info("=" * 60)
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Create tester
    tester = GEMVExecutionTester(dut)
    await tester.reset()
    
    rows, cols = 4, 4
    
    # Create simple test data
    # W = identity-like matrix
    w_data = np.array([
        1, 0, 0, 0,  # Row 0
        0, 1, 0, 0,  # Row 1
        0, 0, 1, 0,  # Row 2
        0, 0, 0, 1,  # Row 3
    ], dtype=np.int8)
    
    # x = [1, 2, 3, 4]
    x_data = np.array([1, 2, 3, 4], dtype=np.int8)
    
    # b = [10, 20, 30, 40]
    b_data = np.array([10, 20, 30, 40], dtype=np.int8)
    
    # Expected: y = W*x + b = [1+10, 2+20, 3+30, 4+40] = [11, 22, 33, 44]
    # After quantization: scale = 44/127 ≈ 0.346, so quantized ≈ [32, 64, 95, 127]
    
    # Load buffers
    tester.load_buffer(1, w_data)  # W in buffer 1
    tester.load_buffer(2, x_data)  # x in buffer 2  
    tester.load_buffer(3, b_data)  # b in buffer 3
    
    # Execute GEMV: dest=0, w=1, x=2, b=3
    result = await tester.execute_gemv(
        dest_id=0, w_id=1, x_id=2, b_id=3,
        rows=rows, cols=cols
    )
    
    if result is None:
        raise cocotb.result.TestFailure("GEMV execution timed out")
        
    # Compute golden
    golden = tester.golden_gemv(w_data, x_data, b_data, rows, cols)
    
    cocotb.log.info(f"RTL result: {list(result)}")
    cocotb.log.info(f"Golden result: {list(golden)}")
    
    # Compare with tolerance (quantization can cause ±1-2 differences)
    max_error = np.max(np.abs(result.astype(np.int32) - golden.astype(np.int32)))
    if max_error <= 2:  # Allow small quantization differences
        cocotb.log.info(f"Small GEMV: ✅ PASSED - {rows}x{cols} matches (max_error={max_error})")
    else:
        cocotb.log.error(f"Small GEMV: ❌ FAILED - max_error={max_error}")
        raise cocotb.result.TestFailure(f"GEMV mismatch: max_error={max_error}")


@cocotb.test()
async def test_gemv_execution_zeros(dut):
    """Test GEMV with zero input: output should equal bias (quantized)."""
    
    cocotb.log.info("=" * 60)
    cocotb.log.info("TEST: gemv_execution zeros input")
    cocotb.log.info("=" * 60)
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Create tester
    tester = GEMVExecutionTester(dut)
    await tester.reset()
    
    rows, cols = 12, 32
    
    # W = random weights
    np.random.seed(42)
    w_data = np.random.randint(-128, 127, size=rows*cols, dtype=np.int8)
    
    # x = all zeros
    x_data = np.zeros(cols, dtype=np.int8)
    
    # b = known bias values
    b_data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120], dtype=np.int8)
    
    # Expected: y = W*0 + b = b, then quantized
    
    # Load buffers
    tester.load_buffer(1, w_data)
    tester.load_buffer(2, x_data)
    tester.load_buffer(3, b_data)
    
    # Execute GEMV
    result = await tester.execute_gemv(
        dest_id=0, w_id=1, x_id=2, b_id=3,
        rows=rows, cols=cols
    )
    
    if result is None:
        raise cocotb.result.TestFailure("GEMV execution timed out")
        
    # Compute golden
    golden = tester.golden_gemv(w_data, x_data, b_data, rows, cols)
    
    cocotb.log.info(f"RTL result: {list(result)}")
    cocotb.log.info(f"Golden result: {list(golden)}")
    
    # Compare with tolerance (quantization can cause ±1-2 differences)
    max_error = np.max(np.abs(result.astype(np.int32) - golden.astype(np.int32)))
    if max_error <= 2:  # Allow small quantization differences
        cocotb.log.info(f"Zeros input: ✅ PASSED (max_error={max_error})")
    else:
        cocotb.log.error(f"Zeros input: ❌ FAILED - max_error={max_error}")
        raise cocotb.result.TestFailure(f"GEMV mismatch: max_error={max_error}")


@cocotb.test()
async def test_gemv_execution_layer2_size(dut):
    """Test Layer 2 GEMV size: 12x32 matrix."""
    
    cocotb.log.info("=" * 60)
    cocotb.log.info("TEST: gemv_execution Layer 2 size (12x32)")
    cocotb.log.info("=" * 60)
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Create tester
    tester = GEMVExecutionTester(dut)
    
    # Load actual data from dram.hex
    hex_file = os.path.join(os.path.dirname(__file__), '../../compiler/dram.hex')
    tester.load_dram(hex_file)
    
    await tester.reset()
    
    rows, cols = 12, 32
    
    # Use actual Layer 2 parameters (from model_assembly.asm)
    # W2 at 0x12BC0 (76736), 32x12 = 384 elements
    # Note: In GEMV, we compute y = W * x + b where W is rows x cols
    # So for 12 output rows and 32 input cols, W has 12*32=384 elements
    
    # Create synthetic but realistic test data
    np.random.seed(123)
    w_data = np.random.randint(-50, 50, size=rows*cols, dtype=np.int8)
    x_data = np.random.randint(-30, 30, size=cols, dtype=np.int8)
    b_data = np.random.randint(-100, 100, size=rows, dtype=np.int8)
    
    # Load buffers
    tester.load_buffer(1, w_data)
    tester.load_buffer(2, x_data)
    tester.load_buffer(3, b_data)
    
    # Execute GEMV
    result = await tester.execute_gemv(
        dest_id=0, w_id=1, x_id=2, b_id=3,
        rows=rows, cols=cols
    )
    
    if result is None:
        raise cocotb.result.TestFailure("GEMV execution timed out")
        
    # Compute golden
    golden = tester.golden_gemv(w_data, x_data, b_data, rows, cols)
    
    cocotb.log.info(f"RTL result: {list(result)}")
    cocotb.log.info(f"Golden result: {list(golden)}")
    
    # Compare
    if np.array_equal(result, golden):
        cocotb.log.info(f"Layer 2 size: ✅ PASSED - {rows}x{cols} matches")
    else:
        max_error = np.max(np.abs(result.astype(np.int32) - golden.astype(np.int32)))
        cocotb.log.error(f"Layer 2 size: ❌ FAILED - max_error={max_error}")
        for i in range(rows):
            if result[i] != golden[i]:
                cocotb.log.error(f"  Row {i}: RTL={result[i]}, Golden={golden[i]}, diff={result[i]-golden[i]}")
        raise cocotb.result.TestFailure(f"GEMV mismatch: max_error={max_error}")


@cocotb.test()
async def test_gemv_execution_layer1_size(dut):
    """Test Layer 1 GEMV size: 12x784 matrix (the actual problematic case)."""
    
    cocotb.log.info("=" * 60)
    cocotb.log.info("TEST: gemv_execution Layer 1 size (12x784)")
    cocotb.log.info("=" * 60)
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Create tester
    tester = GEMVExecutionTester(dut)
    
    # Load actual data from dram.hex
    hex_file = os.path.join(os.path.dirname(__file__), '../../compiler/dram.hex')
    tester.load_dram(hex_file)
    
    await tester.reset()
    
    rows, cols = 12, 784
    
    # Load actual data from DRAM
    # Input vector at 0x700 (1792), 784 elements
    # W1 at 0x10700 (67328), 12*784 = 9408 elements  
    # B1 at 0x13001 (77825), 12 elements
    
    x_addr = 0x700
    w_addr = 0x10700
    b_addr = 0x13001
    
    x_data = tester.memory[x_addr:x_addr+cols].copy()
    w_data = tester.memory[w_addr:w_addr+rows*cols].copy()
    b_data = tester.memory[b_addr:b_addr+rows].copy()
    
    cocotb.log.info(f"Loaded x from 0x{x_addr:06X}: {cols} elements")
    cocotb.log.info(f"  x[0:10] = {list(x_data[:10])}")
    cocotb.log.info(f"  x non-zero: {np.count_nonzero(x_data)}")
    
    cocotb.log.info(f"Loaded W from 0x{w_addr:06X}: {rows*cols} elements")
    cocotb.log.info(f"  W[0:10] = {list(w_data[:10])}")
    
    cocotb.log.info(f"Loaded b from 0x{b_addr:06X}: {rows} elements")
    cocotb.log.info(f"  b = {list(b_data)}")
    
    # Load buffers
    tester.load_buffer(1, w_data)
    tester.load_buffer(2, x_data)
    tester.load_buffer(3, b_data)
    
    # Execute GEMV
    result = await tester.execute_gemv(
        dest_id=0, w_id=1, x_id=2, b_id=3,
        rows=rows, cols=cols,
        timeout=100000  # Longer timeout for large matrix
    )
    
    if result is None:
        raise cocotb.result.TestFailure("GEMV execution timed out")
        
    # Compute golden
    golden = tester.golden_gemv(w_data, x_data, b_data, rows, cols)
    
    cocotb.log.info(f"RTL result: {list(result)}")
    cocotb.log.info(f"Golden result: {list(golden)}")
    
    # Compare
    if np.array_equal(result, golden):
        cocotb.log.info(f"Layer 1 size: ✅ PASSED - {rows}x{cols} matches")
    else:
        max_error = np.max(np.abs(result.astype(np.int32) - golden.astype(np.int32)))
        cocotb.log.error(f"Layer 1 size: ❌ FAILED - max_error={max_error}")
        for i in range(rows):
            if result[i] != golden[i]:
                cocotb.log.error(f"  Row {i}: RTL={result[i]}, Golden={golden[i]}, diff={int(result[i])-int(golden[i])}")
        raise cocotb.result.TestFailure(f"GEMV mismatch: max_error={max_error}")


@cocotb.test()
async def test_gemv_execution_partial_tiles(dut):
    """Test GEMV with dimensions not aligned to tile size (32)."""
    
    cocotb.log.info("=" * 60)
    cocotb.log.info("TEST: gemv_execution partial tiles (5x17)")
    cocotb.log.info("=" * 60)
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Create tester
    tester = GEMVExecutionTester(dut)
    await tester.reset()
    
    rows, cols = 5, 17  # Not aligned to 32
    
    # Create test data
    np.random.seed(789)
    w_data = np.random.randint(-50, 50, size=rows*cols, dtype=np.int8)
    x_data = np.random.randint(-30, 30, size=cols, dtype=np.int8)
    b_data = np.random.randint(-100, 100, size=rows, dtype=np.int8)
    
    # Load buffers
    tester.load_buffer(1, w_data)
    tester.load_buffer(2, x_data)
    tester.load_buffer(3, b_data)
    
    # Execute GEMV
    result = await tester.execute_gemv(
        dest_id=0, w_id=1, x_id=2, b_id=3,
        rows=rows, cols=cols
    )
    
    if result is None:
        raise cocotb.result.TestFailure("GEMV execution timed out")
        
    # Compute golden
    golden = tester.golden_gemv(w_data, x_data, b_data, rows, cols)
    
    cocotb.log.info(f"RTL result: {list(result)}")
    cocotb.log.info(f"Golden result: {list(golden)}")
    
    # Compare
    if np.array_equal(result, golden):
        cocotb.log.info(f"Partial tiles: ✅ PASSED - {rows}x{cols} matches")
    else:
        max_error = np.max(np.abs(result.astype(np.int32) - golden.astype(np.int32)))
        cocotb.log.error(f"Partial tiles: ❌ FAILED - max_error={max_error}")
        raise cocotb.result.TestFailure(f"GEMV mismatch: max_error={max_error}")


@cocotb.test()
async def test_gemv_execution_single_row(dut):
    """Test GEMV with single output row."""
    
    cocotb.log.info("=" * 60)
    cocotb.log.info("TEST: gemv_execution single row (1x64)")
    cocotb.log.info("=" * 60)
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Create tester
    tester = GEMVExecutionTester(dut)
    await tester.reset()
    
    rows, cols = 1, 64
    
    # Create test data - simple pattern
    w_data = np.ones(rows*cols, dtype=np.int8)  # All ones
    x_data = np.arange(cols, dtype=np.int8)  # 0, 1, 2, ... 63
    b_data = np.array([10], dtype=np.int8)
    
    # Expected: sum(0..63) + 10 = 2016 + 10 = 2026
    # After quantization: scale = 2026/127, quantized = 127
    
    # Load buffers
    tester.load_buffer(1, w_data)
    tester.load_buffer(2, x_data)
    tester.load_buffer(3, b_data)
    
    # Execute GEMV
    result = await tester.execute_gemv(
        dest_id=0, w_id=1, x_id=2, b_id=3,
        rows=rows, cols=cols
    )
    
    if result is None:
        raise cocotb.result.TestFailure("GEMV execution timed out")
        
    # Compute golden
    golden = tester.golden_gemv(w_data, x_data, b_data, rows, cols)
    
    cocotb.log.info(f"RTL result: {list(result)}")
    cocotb.log.info(f"Golden result: {list(golden)}")
    
    # Compare
    if np.array_equal(result, golden):
        cocotb.log.info(f"Single row: ✅ PASSED")
    else:
        max_error = np.max(np.abs(result.astype(np.int32) - golden.astype(np.int32)))
        cocotb.log.error(f"Single row: ❌ FAILED - max_error={max_error}")
        raise cocotb.result.TestFailure(f"GEMV mismatch: max_error={max_error}")
