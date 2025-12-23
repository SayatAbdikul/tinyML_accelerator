"""
Cocotb Testbench for load_m Module
==================================

This testbench validates the load_m RTL module by comparing its output
against the Python golden model's load_m function.

The load_m module loads matrix data from DRAM memory into tile-sized chunks,
streaming them out one tile at a time. Unlike load_v which outputs an unpacked
array, load_m outputs a packed 256-bit bus.

Test Cases:
1. Small matrix (32 elements - exactly 1 tile)
2. Medium matrix (64 elements - 2 tiles)
3. Weight matrix layer 1 (12x784 = 9408 elements)
4. Weight matrix layer 2 (32x12 = 384 elements)
5. Weight matrix layer 3 (10x32 = 320 elements)
6. Partial tile (50 elements)
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer
import numpy as np
import sys
import os

# Add compiler path for golden model imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../compiler'))


class LoadMTester:
    """Helper class for load_m module testing."""
    
    TILE_SIZE = 32  # Elements per tile (256 bits / 8 bits)
    TILE_WIDTH = 256  # Bits per tile
    
    def __init__(self, dut):
        self.dut = dut
        self.memory = None
        
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
        self.dut.valid_in.value = 0
        self.dut.dram_addr.value = 0
        self.dut.rows.value = 0
        self.dut.cols.value = 0
        
        await FallingEdge(self.dut.clk)
        await FallingEdge(self.dut.clk)
        await FallingEdge(self.dut.clk)
        
        self.dut.rst.value = 0
        await FallingEdge(self.dut.clk)
        
    def unpack_tile(self, packed_value):
        """
        Unpack a 256-bit packed tile into 32 int8 values.
        
        The RTL stores data with bytes in little-endian order:
        first byte at bits[7:0], second byte at bits[15:8], etc.
        """
        packed_int = int(packed_value)
        elements = []
        for i in range(self.TILE_SIZE):
            # Extract each 8-bit byte, starting from lowest bits
            byte_val = (packed_int >> (i * 8)) & 0xFF
            # Convert to signed
            if byte_val >= 128:
                byte_val = byte_val - 256
            elements.append(byte_val)
        # No reversal needed - RTL uses little-endian order
        return elements
        
    async def load_matrix(self, addr, rows, cols, timeout=50000):
        """
        Execute a load_m operation and collect all output tiles.
        
        Args:
            addr: DRAM address
            rows: Number of rows in matrix
            cols: Number of columns per row
            
        Returns:
            List of all elements loaded (concatenated from tiles), with row padding
        """
        # Set inputs
        self.dut.dram_addr.value = addr
        self.dut.rows.value = rows
        self.dut.cols.value = cols
        
        # Pulse valid_in
        await FallingEdge(self.dut.clk)
        self.dut.valid_in.value = 1
        await FallingEdge(self.dut.clk)
        self.dut.valid_in.value = 0
        
        # Collect tiles
        all_data = []
        tiles_collected = 0
        tiles_per_row = (cols + self.TILE_SIZE - 1) // self.TILE_SIZE
        expected_tiles = rows * tiles_per_row
        
        for cycle in range(timeout):
            await FallingEdge(self.dut.clk)
            
            if self.dut.tile_out.value:
                # Unpack tile data from 256-bit bus
                tile_data = self.unpack_tile(self.dut.data_out.value)
                all_data.extend(tile_data)
                tiles_collected += 1
                if tiles_collected <= 5 or tiles_collected == expected_tiles:
                    cocotb.log.debug(f"  Tile {tiles_collected}/{expected_tiles} captured")
                
            if self.dut.valid_out.value:
                cocotb.log.info(f"✅ Load complete after {cycle} cycles, {tiles_collected} tiles")
                break
        else:
            raise TimeoutError(f"load_m timed out after {timeout} cycles")
        
        # Return all data including padding (tests will validate per-row)
        return all_data
    
    def golden_load_m(self, addr, rows, cols):
        """
        Execute golden model load_m and return the result with row padding.
        
        The RTL stores each row aligned to tile boundaries. For a matrix with
        cols=784, each row uses 25 tiles (800 bytes), with 16 bytes of padding.
        """
        if self.memory is None:
            raise RuntimeError("DRAM not loaded")
        
        result = []
        tiles_per_row = (cols + self.TILE_SIZE - 1) // self.TILE_SIZE
        bytes_per_padded_row = tiles_per_row * self.TILE_SIZE
        
        for row in range(rows):
            row_start = addr + row * cols
            row_data = list(self.memory[row_start:row_start + cols])
            # Pad to tile boundary
            padding = bytes_per_padded_row - cols
            row_data.extend([0] * padding)
            result.extend(row_data)
        
        return result
    
    def compare_results(self, rtl_data, golden_data, name=""):
        """Compare RTL output with golden model."""
        if len(rtl_data) != len(golden_data):
            cocotb.log.error(f"{name}: Length mismatch - RTL={len(rtl_data)}, Golden={len(golden_data)}")
            return False
        
        rtl_array = np.array(rtl_data, dtype=np.int8)
        golden_array = np.array(golden_data, dtype=np.int8)
        
        if not np.array_equal(rtl_array, golden_array):
            diff_indices = np.where(rtl_array != golden_array)[0]
            cocotb.log.error(f"{name}: Data mismatch at {len(diff_indices)} positions")
            cocotb.log.error(f"  First 10 mismatches at indices: {diff_indices[:10]}")
            for idx in diff_indices[:10]:
                cocotb.log.error(f"    Index {idx}: RTL={rtl_data[idx]}, Golden={golden_data[idx]}")
            return False
        
        cocotb.log.info(f"{name}: ✅ PASSED - {len(rtl_data)} elements match")
        return True


@cocotb.test()
async def test_load_m_single_tile(dut):
    """Test loading exactly 1 tile (32 elements) - 1 row x 32 cols."""
    cocotb.log.info("="*60)
    cocotb.log.info("TEST: load_m single tile (1x32 = 32 elements)")
    cocotb.log.info("="*60)
    
    tester = LoadMTester(dut)
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start(start_high=False))
    
    # Reset
    await tester.reset()
    
    # Load DRAM
    dram_path = os.path.join(os.path.dirname(__file__), '../../compiler/dram.hex')
    tester.load_dram(dram_path)
    
    # Test: 1 row x 32 columns = exactly 1 tile
    addr = 0x10700  # Weight matrix address
    rows = 1
    cols = 32
    
    cocotb.log.info(f"Loading {rows}x{cols} = {rows*cols} elements from address 0x{addr:06X}")
    
    # Execute RTL
    rtl_result = await tester.load_matrix(addr, rows, cols)
    
    # Get golden result
    golden_result = tester.golden_load_m(addr, rows, cols)
    
    cocotb.log.info(f"RTL result: {rtl_result}")
    cocotb.log.info(f"Golden result: {golden_result}")
    
    # Compare
    success = tester.compare_results(rtl_result, golden_result, "Single tile")
    assert success, "Single tile test failed"


@cocotb.test()
async def test_load_m_two_tiles(dut):
    """Test loading 2 tiles (1 row x 64 columns)."""
    cocotb.log.info("="*60)
    cocotb.log.info("TEST: load_m two tiles (1x64 = 64 elements)")
    cocotb.log.info("="*60)
    
    tester = LoadMTester(dut)
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start(start_high=False))
    
    # Reset
    await tester.reset()
    
    # Load DRAM
    dram_path = os.path.join(os.path.dirname(__file__), '../../compiler/dram.hex')
    tester.load_dram(dram_path)
    
    # Test: 1 row x 64 columns = 2 tiles
    addr = 0x10700
    rows = 1
    cols = 64
    
    cocotb.log.info(f"Loading {rows}x{cols} = {rows*cols} elements from address 0x{addr:06X}")
    
    # Execute RTL
    rtl_result = await tester.load_matrix(addr, rows, cols)
    
    # Get golden result
    golden_result = tester.golden_load_m(addr, rows, cols)
    
    cocotb.log.info(f"RTL first 10: {rtl_result[:10]}")
    cocotb.log.info(f"Golden first 10: {golden_result[:10]}")
    
    # Compare
    success = tester.compare_results(rtl_result, golden_result, "Two tiles")
    assert success, "Two tiles test failed"


@cocotb.test()
async def test_load_m_weight_matrix_layer2(dut):
    """Test loading Layer 2 weight matrix (32x12 = 384 elements)."""
    cocotb.log.info("="*60)
    cocotb.log.info("TEST: load_m Layer 2 weights (32x12 = 384 elements)")
    cocotb.log.info("="*60)
    
    tester = LoadMTester(dut)
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start(start_high=False))
    
    # Reset
    await tester.reset()
    
    # Load DRAM
    dram_path = os.path.join(os.path.dirname(__file__), '../../compiler/dram.hex')
    tester.load_dram(dram_path)
    
    # Test: Layer 2 weights at 0x12bc0, 32 rows x 12 cols = 384 elements
    # From model_assembly.asm: LOAD_M 2, 0x12bc0, 32, 12
    addr = 0x12bc0
    rows = 32
    cols = 12
    
    tiles_per_row = (cols + 31) // 32  # 1 tile per row
    total_tiles = rows * tiles_per_row  # 32 tiles
    
    cocotb.log.info(f"Loading weight matrix: {rows}x{cols} = {rows*cols} elements from address 0x{addr:06X}")
    cocotb.log.info(f"Expected tiles: {total_tiles} ({tiles_per_row} per row x {rows} rows)")
    
    # Execute RTL
    rtl_result = await tester.load_matrix(addr, rows, cols, timeout=5000)
    
    # Get golden result
    golden_result = tester.golden_load_m(addr, rows, cols)
    
    # Stats
    rtl_nonzero = sum(1 for x in rtl_result if x != 0)
    golden_nonzero = sum(1 for x in golden_result if x != 0)
    cocotb.log.info(f"RTL non-zero elements: {rtl_nonzero}")
    cocotb.log.info(f"Golden non-zero elements: {golden_nonzero}")
    
    # Compare
    success = tester.compare_results(rtl_result, golden_result, "Layer 2 weights")
    assert success, "Layer 2 weights test failed"


@cocotb.test()
async def test_load_m_weight_matrix_layer3(dut):
    """Test loading Layer 3 weight matrix (10x32 = 320 elements)."""
    cocotb.log.info("="*60)
    cocotb.log.info("TEST: load_m Layer 3 weights (10x32 = 320 elements)")
    cocotb.log.info("="*60)
    
    tester = LoadMTester(dut)
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start(start_high=False))
    
    # Reset
    await tester.reset()
    
    # Load DRAM
    dram_path = os.path.join(os.path.dirname(__file__), '../../compiler/dram.hex')
    tester.load_dram(dram_path)
    
    # Test: Layer 3 weights at 0x12d40, 10 rows x 32 cols = 320 elements
    # From model_assembly.asm: LOAD_M 1, 0x12d40, 10, 32
    addr = 0x12d40
    rows = 10
    cols = 32
    
    tiles_per_row = (cols + 31) // 32  # 1 tile per row
    total_tiles = rows * tiles_per_row  # 10 tiles
    
    cocotb.log.info(f"Loading weight matrix: {rows}x{cols} = {rows*cols} elements from address 0x{addr:06X}")
    cocotb.log.info(f"Expected tiles: {total_tiles}")
    
    # Execute RTL
    rtl_result = await tester.load_matrix(addr, rows, cols, timeout=5000)
    
    # Get golden result
    golden_result = tester.golden_load_m(addr, rows, cols)
    
    cocotb.log.info(f"RTL first 10: {rtl_result[:10]}")
    cocotb.log.info(f"Golden first 10: {golden_result[:10]}")
    
    # Compare
    success = tester.compare_results(rtl_result, golden_result, "Layer 3 weights")
    assert success, "Layer 3 weights test failed"


@cocotb.test()
async def test_load_m_weight_matrix_layer1(dut):
    """Test loading Layer 1 weight matrix (12x784 = 9408 elements)."""
    cocotb.log.info("="*60)
    cocotb.log.info("TEST: load_m Layer 1 weights (12x784 = 9408 elements)")
    cocotb.log.info("="*60)
    
    tester = LoadMTester(dut)
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start(start_high=False))
    
    # Reset
    await tester.reset()
    
    # Load DRAM
    dram_path = os.path.join(os.path.dirname(__file__), '../../compiler/dram.hex')
    tester.load_dram(dram_path)
    
    # Test: Layer 1 weights at 0x10700, 12 rows x 784 cols = 9408 elements
    # From model_assembly.asm: LOAD_M 1, 0x10700, 12, 784
    addr = 0x10700
    rows = 12
    cols = 784
    
    tiles_per_row = (cols + 31) // 32  # 25 tiles per row
    total_tiles = rows * tiles_per_row  # 300 tiles
    
    cocotb.log.info(f"Loading weight matrix: {rows}x{cols} = {rows*cols} elements from address 0x{addr:06X}")
    cocotb.log.info(f"Expected tiles: {total_tiles} ({tiles_per_row} per row x {rows} rows)")
    
    # Execute RTL
    rtl_result = await tester.load_matrix(addr, rows, cols, timeout=100000)
    
    # Get golden result
    golden_result = tester.golden_load_m(addr, rows, cols)
    
    # Stats
    total_elements = rows * tiles_per_row * 32  # Padded size
    rtl_nonzero = sum(1 for x in rtl_result if x != 0)
    golden_nonzero = sum(1 for x in golden_result if x != 0)
    cocotb.log.info(f"RTL non-zero elements: {rtl_nonzero}/{total_elements}")
    cocotb.log.info(f"Golden non-zero elements: {golden_nonzero}/{total_elements}")
    cocotb.log.info(f"RTL first 20: {rtl_result[:20]}")
    cocotb.log.info(f"Golden first 20: {golden_result[:20]}")
    
    # Compare
    success = tester.compare_results(rtl_result, golden_result, "Layer 1 weights")
    assert success, "Layer 1 weights test failed"


@cocotb.test()
async def test_load_m_partial_tile(dut):
    """Test loading a partial tile (1 row x 50 columns = 2 tiles)."""
    cocotb.log.info("="*60)
    cocotb.log.info("TEST: load_m partial tile (1x50 = 50 elements, 2 tiles)")
    cocotb.log.info("="*60)
    
    tester = LoadMTester(dut)
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start(start_high=False))
    
    # Reset
    await tester.reset()
    
    # Load DRAM
    dram_path = os.path.join(os.path.dirname(__file__), '../../compiler/dram.hex')
    tester.load_dram(dram_path)
    
    # Test: 1 row x 50 columns = 2 tiles, second is partial
    addr = 0x10700
    rows = 1
    cols = 50
    
    cocotb.log.info(f"Loading {rows}x{cols} = {rows*cols} elements from address 0x{addr:06X}")
    
    # Execute RTL
    rtl_result = await tester.load_matrix(addr, rows, cols)
    
    # Get golden result
    golden_result = tester.golden_load_m(addr, rows, cols)
    
    cocotb.log.info(f"RTL first 10: {rtl_result[:10]}")
    cocotb.log.info(f"RTL positions 30-50 (partial): {rtl_result[30:50]}")
    cocotb.log.info(f"RTL positions 50-64 (padding): {rtl_result[50:64]}")
    cocotb.log.info(f"Golden first 10: {golden_result[:10]}")
    cocotb.log.info(f"Golden positions 30-50 (partial): {golden_result[30:50]}")
    cocotb.log.info(f"Golden positions 50-64 (padding): {golden_result[50:64]}")
    
    # Compare
    success = tester.compare_results(rtl_result, golden_result, "Partial tile")
    assert success, "Partial tile test failed"
