"""
Cocotb Testbench for load_v Module
==================================

This testbench validates the load_v RTL module by comparing its output
against the Python golden model's load_v function.

The load_v module loads vector data from DRAM memory into tile-sized chunks,
streaming them out one tile at a time.

Test Cases:
1. Small vector (32 elements - exactly 1 tile)
2. Medium vector (64 elements - 2 tiles)
3. Large vector (784 elements - multiple tiles, like MNIST input)
4. Partial tile (50 elements - tests zero-padding)
5. Edge case: minimum length (1 element)
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer
import numpy as np
import sys
import os

# Add compiler path for golden model imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../compiler'))


class LoadVTester:
    """Helper class for load_v module testing."""
    
    TILE_SIZE = 32  # Elements per tile (256 bits / 8 bits)
    
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
        self.dut.length.value = 0
        
        await FallingEdge(self.dut.clk)
        await FallingEdge(self.dut.clk)
        await FallingEdge(self.dut.clk)
        
        self.dut.rst.value = 0
        await FallingEdge(self.dut.clk)
        
    async def load_vector(self, addr, length, timeout=5000):
        """
        Execute a load_v operation and collect all output tiles.
        
        Returns:
            List of all elements loaded (concatenated from tiles)
        """
        # Set inputs
        self.dut.dram_addr.value = addr
        self.dut.length.value = length
        
        # Pulse valid_in
        await FallingEdge(self.dut.clk)
        self.dut.valid_in.value = 1
        await FallingEdge(self.dut.clk)
        self.dut.valid_in.value = 0
        
        # Collect tiles
        all_data = []
        tiles_collected = 0
        expected_tiles = (length + self.TILE_SIZE - 1) // self.TILE_SIZE
        
        for cycle in range(timeout):
            await FallingEdge(self.dut.clk)
            
            if self.dut.tile_out.value:
                # Collect tile data
                tile_data = []
                for i in range(self.TILE_SIZE):
                    val = int(self.dut.data_out[i].value)
                    if val & 0x80:  # Sign extend
                        val = val - 256
                    tile_data.append(val)
                all_data.extend(tile_data)
                tiles_collected += 1
                cocotb.log.debug(f"  Tile {tiles_collected}/{expected_tiles} captured")
                
            if self.dut.valid_out.value:
                cocotb.log.info(f"✅ Load complete after {cycle} cycles, {tiles_collected} tiles")
                break
        else:
            raise TimeoutError(f"load_v timed out after {timeout} cycles")
        
        # Trim to requested length
        return all_data[:length]
    
    def golden_load_v(self, addr, length):
        """Execute golden model load_v and return the result."""
        if self.memory is None:
            raise RuntimeError("DRAM not loaded")
        return list(self.memory[addr:addr + length])
    
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
            cocotb.log.error(f"  First 5 mismatches at indices: {diff_indices[:5]}")
            for idx in diff_indices[:5]:
                cocotb.log.error(f"    Index {idx}: RTL={rtl_data[idx]}, Golden={golden_data[idx]}")
            return False
        
        cocotb.log.info(f"{name}: ✅ PASSED - {len(rtl_data)} elements match")
        return True


@cocotb.test()
async def test_load_v_single_tile(dut):
    """Test loading exactly 1 tile (32 elements)."""
    cocotb.log.info("="*60)
    cocotb.log.info("TEST: load_v single tile (32 elements)")
    cocotb.log.info("="*60)
    
    tester = LoadVTester(dut)
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start(start_high=False))
    
    # Reset
    await tester.reset()
    
    # Load DRAM
    dram_path = os.path.join(os.path.dirname(__file__), '../../compiler/dram.hex')
    tester.load_dram(dram_path)
    
    # Test address and length
    addr = 0x700
    length = 32  # Exactly 1 tile
    
    cocotb.log.info(f"Loading {length} elements from address 0x{addr:06X}")
    
    # Execute RTL
    rtl_result = await tester.load_vector(addr, length)
    
    # Get golden result
    golden_result = tester.golden_load_v(addr, length)
    
    # Compare
    success = tester.compare_results(rtl_result, golden_result, "Single tile")
    assert success, "Single tile test failed"


@cocotb.test()
async def test_load_v_two_tiles(dut):
    """Test loading 2 tiles (64 elements)."""
    cocotb.log.info("="*60)
    cocotb.log.info("TEST: load_v two tiles (64 elements)")
    cocotb.log.info("="*60)
    
    tester = LoadVTester(dut)
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start(start_high=False))
    
    # Reset
    await tester.reset()
    
    # Load DRAM
    dram_path = os.path.join(os.path.dirname(__file__), '../../compiler/dram.hex')
    tester.load_dram(dram_path)
    
    # Test: Load 64 elements (2 tiles)
    addr = 0x10700  # Weight matrix address (non-zero data)
    length = 64
    
    cocotb.log.info(f"Loading {length} elements from address 0x{addr:06X}")
    
    # Execute RTL
    rtl_result = await tester.load_vector(addr, length)
    
    # Get golden result
    golden_result = tester.golden_load_v(addr, length)
    
    cocotb.log.info(f"RTL first 10: {rtl_result[:10]}")
    cocotb.log.info(f"Golden first 10: {golden_result[:10]}")
    
    # Compare
    success = tester.compare_results(rtl_result, golden_result, "Two tiles")
    assert success, "Two tiles test failed"


@cocotb.test()
async def test_load_v_large_vector(dut):
    """Test loading a large vector (784 elements - MNIST input size)."""
    cocotb.log.info("="*60)
    cocotb.log.info("TEST: load_v large vector (784 elements)")
    cocotb.log.info("="*60)
    
    tester = LoadVTester(dut)
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start(start_high=False))
    
    # Reset
    await tester.reset()
    
    # Load DRAM
    dram_path = os.path.join(os.path.dirname(__file__), '../../compiler/dram.hex')
    tester.load_dram(dram_path)
    
    # Test: Load 784 elements (25 tiles, like MNIST input)
    addr = 0x700
    length = 784
    
    cocotb.log.info(f"Loading {length} elements from address 0x{addr:06X}")
    cocotb.log.info(f"Expected tiles: {(length + 31) // 32}")
    
    # Execute RTL
    rtl_result = await tester.load_vector(addr, length, timeout=10000)
    
    # Get golden result
    golden_result = tester.golden_load_v(addr, length)
    
    # Find non-zero elements for logging
    rtl_nonzero = sum(1 for x in rtl_result if x != 0)
    golden_nonzero = sum(1 for x in golden_result if x != 0)
    cocotb.log.info(f"RTL non-zero elements: {rtl_nonzero}")
    cocotb.log.info(f"Golden non-zero elements: {golden_nonzero}")
    
    # Compare
    success = tester.compare_results(rtl_result, golden_result, "Large vector (784)")
    assert success, "Large vector test failed"


@cocotb.test()
async def test_load_v_partial_tile(dut):
    """Test loading a partial tile (50 elements - not a multiple of 32)."""
    cocotb.log.info("="*60)
    cocotb.log.info("TEST: load_v partial tile (50 elements)")
    cocotb.log.info("="*60)
    
    tester = LoadVTester(dut)
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start(start_high=False))
    
    # Reset
    await tester.reset()
    
    # Load DRAM
    dram_path = os.path.join(os.path.dirname(__file__), '../../compiler/dram.hex')
    tester.load_dram(dram_path)
    
    # Test: Load 50 elements (2 tiles, second is partial)
    addr = 0x10700
    length = 50
    
    cocotb.log.info(f"Loading {length} elements from address 0x{addr:06X}")
    
    # Execute RTL
    rtl_result = await tester.load_vector(addr, length)
    
    # Get golden result
    golden_result = tester.golden_load_v(addr, length)
    
    cocotb.log.info(f"RTL first 10: {rtl_result[:10]}")
    cocotb.log.info(f"RTL last 10: {rtl_result[-10:]}")
    cocotb.log.info(f"Golden first 10: {golden_result[:10]}")
    cocotb.log.info(f"Golden last 10: {golden_result[-10:]}")
    
    # Compare
    success = tester.compare_results(rtl_result, golden_result, "Partial tile")
    assert success, "Partial tile test failed"


@cocotb.test()
async def test_load_v_bias_vector(dut):
    """Test loading bias vector (12 elements - from actual assembly)."""
    cocotb.log.info("="*60)
    cocotb.log.info("TEST: load_v bias vector (12 elements)")
    cocotb.log.info("="*60)
    
    tester = LoadVTester(dut)
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start(start_high=False))
    
    # Reset
    await tester.reset()
    
    # Load DRAM
    dram_path = os.path.join(os.path.dirname(__file__), '../../compiler/dram.hex')
    tester.load_dram(dram_path)
    
    # Test: Load bias vector from model_assembly.asm: LOAD_V 4, 0x13001, 12
    addr = 0x13001
    length = 12
    
    cocotb.log.info(f"Loading bias vector: {length} elements from address 0x{addr:06X}")
    
    # Execute RTL
    rtl_result = await tester.load_vector(addr, length)
    
    # Get golden result
    golden_result = tester.golden_load_v(addr, length)
    
    cocotb.log.info(f"RTL bias: {rtl_result}")
    cocotb.log.info(f"Golden bias: {golden_result}")
    
    # Compare
    success = tester.compare_results(rtl_result, golden_result, "Bias vector")
    assert success, "Bias vector test failed"


@cocotb.test()
async def test_load_v_minimum_length(dut):
    """Test loading minimum length (1 element)."""
    cocotb.log.info("="*60)
    cocotb.log.info("TEST: load_v minimum length (1 element)")
    cocotb.log.info("="*60)
    
    tester = LoadVTester(dut)
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start(start_high=False))
    
    # Reset
    await tester.reset()
    
    # Load DRAM
    dram_path = os.path.join(os.path.dirname(__file__), '../../compiler/dram.hex')
    tester.load_dram(dram_path)
    
    # Test: Load just 1 element
    addr = 0x10700
    length = 1
    
    cocotb.log.info(f"Loading {length} element from address 0x{addr:06X}")
    
    # Execute RTL
    rtl_result = await tester.load_vector(addr, length)
    
    # Get golden result
    golden_result = tester.golden_load_v(addr, length)
    
    cocotb.log.info(f"RTL result: {rtl_result}")
    cocotb.log.info(f"Golden result: {golden_result}")
    
    # Compare
    success = tester.compare_results(rtl_result, golden_result, "Minimum length")
    assert success, "Minimum length test failed"
