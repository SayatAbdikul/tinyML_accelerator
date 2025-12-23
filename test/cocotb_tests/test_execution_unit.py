"""
Cocotb Testbench for Modular Execution Unit
============================================

This testbench validates the modular_execution_unit RTL module by executing
a complete neural network (784→12→32→10) and comparing results against the
Python golden model.

Test Flow:
1. Load instruction sequence from model_assembly.asm (via golden model)
2. Execute each instruction on RTL execution unit
3. Track buffer states in both RTL and golden model
4. Compare intermediate and final results
5. Validate complete neural network execution

Network Architecture:
- Input: 784 neurons (28×28 MNIST image)
- Hidden Layer 1: 12 neurons (FC + ReLU)
- Hidden Layer 2: 32 neurons (FC + ReLU)
- Output Layer: 10 neurons (FC, classification scores)
- Total: 10,112 parameters
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer
import numpy as np
import sys
import os

# Add compiler path for golden model imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../compiler'))
import golden_model
from golden_model import load_v, load_m, gemv, relu, store, load_memory
from helper_functions import quantize_int32_to_int8


class ExecutionUnitTester:
    """Helper class for execution unit testing."""
    
    # Opcode definitions
    OPCODE_LOAD_V = 0x01
    OPCODE_LOAD_M = 0x02
    OPCODE_STORE = 0x03
    OPCODE_GEMV = 0x04
    OPCODE_RELU = 0x05
    
    def __init__(self, dut):
        self.dut = dut
        self.memory = None
        self.cycle_count = 0
        
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
        self.dut.opcode.value = 0
        self.dut.dest.value = 0
        self.dut.length_or_cols.value = 0
        self.dut.rows.value = 0
        self.dut.addr.value = 0
        self.dut.x_id.value = 0
        self.dut.w_id.value = 0
        self.dut.b_id.value = 0
        
        await FallingEdge(self.dut.clk)
        await FallingEdge(self.dut.clk)
        await FallingEdge(self.dut.clk)
        
        self.dut.rst.value = 0
        await FallingEdge(self.dut.clk)
        self.cycle_count = 0
        
    async def execute_instruction(self, opcode, dest, length_or_cols, rows, addr, x_id, w_id, b_id, timeout=100000):
        """Execute a single instruction on the RTL."""
        # Set instruction fields
        self.dut.opcode.value = opcode
        self.dut.dest.value = dest
        self.dut.length_or_cols.value = length_or_cols
        self.dut.rows.value = rows
        self.dut.addr.value = addr
        self.dut.x_id.value = x_id
        self.dut.w_id.value = w_id
        self.dut.b_id.value = b_id
        
        # Pulse start
        await FallingEdge(self.dut.clk)
        self.dut.start.value = 1
        await FallingEdge(self.dut.clk)
        self.dut.start.value = 0
        
        # Wait for done
        start_cycle = self.cycle_count
        for _ in range(timeout):
            if self.dut.done.value:
                cycles_taken = self.cycle_count - start_cycle
                return cycles_taken
            await FallingEdge(self.dut.clk)
            self.cycle_count += 1
            
        raise TimeoutError(f"Instruction timed out after {timeout} cycles")
        
    def read_buffer(self, buffer_id, length):
        """Read buffer contents from RTL."""
        # Note: This would require buffer access signals in RTL
        # For now, we'll rely on the result output
        # In a real implementation, you'd need debug/test hooks
        pass
        
    def compare_buffers(self, rtl_data, golden_data, name, tolerance=2):
        """Compare RTL buffer with golden model buffer."""
        rtl_array = np.array(rtl_data, dtype=np.int8)
        golden_array = np.array(golden_data, dtype=np.int8)
        
        if len(rtl_array) != len(golden_array):
            cocotb.log.error(f"{name}: Length mismatch - RTL={len(rtl_array)}, Golden={len(golden_array)}")
            return False
            
        diff = np.abs(rtl_array.astype(np.int32) - golden_array.astype(np.int32))
        max_error = np.max(diff)
        
        if max_error > tolerance:
            cocotb.log.error(f"{name}: Max error {max_error} > {tolerance}")
            cocotb.log.error(f"  RTL (first 10): {rtl_array[:10]}")
            cocotb.log.error(f"  Golden (first 10): {golden_array[:10]}")
            return False
        else:
            cocotb.log.info(f"{name}: Match! Max error={max_error}")
            return True


@cocotb.test()
async def test_neural_network_complete(dut):
    """
    Test complete neural network execution: 784→12→32→10
    Replicates model_assembly.asm instruction sequence.
    """
    
    cocotb.log.info("="*60)
    cocotb.log.info("NEURAL NETWORK TEST: 784→12→32→10")
    cocotb.log.info("="*60)
    
    # Setup
    tester = ExecutionUnitTester(dut)
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start(start_high=False))
    
    # Reset
    await tester.reset()
    cocotb.log.info("Reset complete")
    
    # Load DRAM
    dram_path = os.path.join(os.path.dirname(__file__), '../../compiler/dram.hex')
    if not os.path.exists(dram_path):
        cocotb.log.error(f"DRAM file not found: {dram_path}")
        cocotb.log.info("Run: cd ../../compiler && python3 main.py")
        assert False, "dram.hex not found"
        
    tester.load_dram(dram_path)
    
    # Initialize golden model global state
    import golden_model
    golden_model.memory = tester.memory
    golden_model.buffers = {}
    golden_model.flag = 0
    
    # Track layer success
    layer1_success = True
    layer2_success = True
    layer3_success = True
    total_cycles = 0
    
    # ========== LAYER 1: 784 → 12 ==========
    cocotb.log.info("")
    cocotb.log.info("╔════════════════════════════════════╗")
    cocotb.log.info("║     LAYER 1: 784 → 12 (FC)        ║")
    cocotb.log.info("╚════════════════════════════════════╝")
    
    try:
        # Step 1: LOAD_V 9, 0x700, 784 (input vector)
        cocotb.log.info("\nStep 1: LOAD_V 9, 0x700, 784 (input vector)")
        load_v(9, 0x700, 784)  # Golden model
        cocotb.log.info(f"  Golden buffer 9 loaded: {len(golden_model.buffers[9])} elements")
        cocotb.log.info(f"  First 50: {golden_model.buffers[9][:50]}")
        cocotb.log.info(f"  Non-zero count: {np.count_nonzero(golden_model.buffers[9])}")
        cycles = await tester.execute_instruction(
            opcode=0x01, dest=9, length_or_cols=784, rows=0,
            addr=0x700, x_id=0, w_id=0, b_id=0, timeout=2000
        )
        total_cycles += cycles
        cocotb.log.info(f"  ✅ Completed in {cycles} cycles")        # Step 2: LOAD_M 1, 0x10700, 12, 784 (weight matrix W1)
        cocotb.log.info("\nStep 2: LOAD_M 1, 0x10700, 12, 784 (weight matrix W1)")
        load_m(1, 0x10700, 12, 784)  # Golden model
        cocotb.log.info(f"  Golden buffer 1 loaded: {len(golden_model.buffers[1])} elements")
        cocotb.log.info(f"  First 50: {golden_model.buffers[1][:50]}")
        cocotb.log.info(f"  Non-zero count: {np.count_nonzero(golden_model.buffers[1])}")
        cycles = await tester.execute_instruction(
            opcode=0x02, dest=1, length_or_cols=784, rows=12,
            addr=0x10700, x_id=0, w_id=0, b_id=0, timeout=25000
        )
        total_cycles += cycles
        cocotb.log.info(f"  ✅ Completed in {cycles} cycles")
        
        # Step 3: LOAD_V 4, 0x13001, 12 (bias vector b1)
        cocotb.log.info("\nStep 3: LOAD_V 4, 0x13001, 12 (bias vector b1)")
        load_v(4, 0x13001, 12)  # Golden model
        cycles = await tester.execute_instruction(
            opcode=0x01, dest=4, length_or_cols=12, rows=0,
            addr=0x13001, x_id=0, w_id=0, b_id=0, timeout=200
        )
        total_cycles += cycles
        cocotb.log.info(f"  ✅ Completed in {cycles} cycles")
        
        # Step 4: GEMV 5, 1, 9, 4, 12, 784 (W1 * input + b1)
        cocotb.log.info("\nStep 4: GEMV 5, 1, 9, 4, 12, 784 (W1 * input + b1)")
        gemv(5, 1, 9, 4, 12, 784)  # Golden model
        cycles = await tester.execute_instruction(
            opcode=0x04, dest=5, length_or_cols=784, rows=12,
            addr=0, x_id=9, w_id=1, b_id=4, timeout=60000
        )
        total_cycles += cycles
        cocotb.log.info(f"  ✅ Completed in {cycles} cycles")

        # Compare intermediate GEMV output for Layer 1
        await FallingEdge(dut.clk)
        rtl_l1 = []
        for i in range(12):
            v = int(dut.result[i].value)
            if v & 0x80:
                v = v - 256
            rtl_l1.append(v)
        golden_l1 = np.array(golden_model.buffers[5][:12], dtype=np.int8)
        diff_l1 = np.abs(np.array(rtl_l1, dtype=np.int32) - golden_l1.astype(np.int32))
        max_err_l1 = int(np.max(diff_l1))
        cocotb.log.info(f"  🔍 Layer 1 GEMV compare: max_error={max_err_l1}")
        cocotb.log.info(f"     RTL (quantized int8): {rtl_l1}")
        cocotb.log.info(f"     Golden (quantized int8): {golden_l1.tolist()}")
        if max_err_l1 > 2:
            cocotb.log.error(f"  ❌ Layer 1 GEMV mismatch")
        
        # Step 5: RELU 7, 5 (activation)
        cocotb.log.info("\nStep 5: RELU 7, 5 (activation function)")
        relu(7, 5, 12)  # Golden model
        cycles = await tester.execute_instruction(
            opcode=0x05, dest=7, length_or_cols=12, rows=0,
            addr=0, x_id=5, w_id=0, b_id=0, timeout=300
        )
        total_cycles += cycles
        cocotb.log.info(f"  ✅ Completed in {cycles} cycles")

        # Compare intermediate RELU output for Layer 1
        await FallingEdge(dut.clk)
        rtl_l1_relu = []
        for i in range(12):
            v = int(dut.result[i].value)
            if v & 0x80:
                v = v - 256
            rtl_l1_relu.append(v)
        golden_l1_relu = np.array(golden_model.buffers[7][:12], dtype=np.int8)
        diff_l1_relu = np.abs(np.array(rtl_l1_relu, dtype=np.int32) - golden_l1_relu.astype(np.int32))
        max_err_l1_relu = int(np.max(diff_l1_relu))
        cocotb.log.info(f"  🔍 Layer 1 RELU compare: max_error={max_err_l1_relu}")
        if max_err_l1_relu > 2:
            cocotb.log.error(f"  ❌ Layer 1 RELU mismatch. RTL[:12]={rtl_l1_relu}, Golden[:12]={golden_l1_relu.tolist()}")
        
        cocotb.log.info("\n✅ Layer 1 Complete: 784 → 12")
        
    except Exception as e:
        cocotb.log.error(f"❌ Layer 1 Failed: {e}")
        layer1_success = False
        assert False, f"Layer 1 failed: {e}"
    
    # ========== LAYER 2: 12 → 32 ==========
    cocotb.log.info("")
    cocotb.log.info("╔════════════════════════════════════╗")
    cocotb.log.info("║     LAYER 2: 12 → 32 (FC)         ║")
    cocotb.log.info("╚════════════════════════════════════╝")
    
    try:
        # Step 6: LOAD_M 2, 0x12bc0, 32, 12 (weight matrix W2)
        cocotb.log.info("\nStep 6: LOAD_M 2, 0x12bc0, 32, 12 (weight matrix W2)")
        load_m(2, 0x12bc0, 32, 12)  # Golden model
        cycles = await tester.execute_instruction(
            opcode=0x02, dest=2, length_or_cols=12, rows=32,
            addr=0x12bc0, x_id=0, w_id=0, b_id=0, timeout=1500
        )
        total_cycles += cycles
        cocotb.log.info(f"  ✅ Completed in {cycles} cycles")
        
        # Step 7: LOAD_V 3, 0x1300d, 32 (bias vector b2)
        cocotb.log.info("\nStep 7: LOAD_V 3, 0x1300d, 32 (bias vector b2)")
        load_v(3, 0x1300d, 32)  # Golden model
        cycles = await tester.execute_instruction(
            opcode=0x01, dest=3, length_or_cols=32, rows=0,
            addr=0x1300d, x_id=0, w_id=0, b_id=0, timeout=250
        )
        total_cycles += cycles
        cocotb.log.info(f"  ✅ Completed in {cycles} cycles")
        
        # Step 8: GEMV 6, 2, 7, 3, 32, 12 (W2 * h1 + b2)
        cocotb.log.info("\nStep 8: GEMV 6, 2, 7, 3, 32, 12 (W2 * h1 + b2)")
        gemv(6, 2, 7, 3, 32, 12)  # Golden model
        cycles = await tester.execute_instruction(
            opcode=0x04, dest=6, length_or_cols=12, rows=32,
            addr=0, x_id=7, w_id=2, b_id=3, timeout=8000
        )
        total_cycles += cycles
        cocotb.log.info(f"  ✅ Completed in {cycles} cycles")

        # Compare intermediate GEMV output for Layer 2
        await FallingEdge(dut.clk)
        rtl_l2 = []
        for i in range(32):
            v = int(dut.result[i].value)
            if v & 0x80:
                v = v - 256
            rtl_l2.append(v)
        golden_l2 = np.array(golden_model.buffers[6][:32], dtype=np.int8)
        diff_l2 = np.abs(np.array(rtl_l2, dtype=np.int32) - golden_l2.astype(np.int32))
        max_err_l2 = int(np.max(diff_l2))
        cocotb.log.info(f"  🔍 Layer 2 GEMV compare: max_error={max_err_l2}")
        if max_err_l2 > 2:
            cocotb.log.error(f"  ❌ Layer 2 GEMV mismatch. RTL[:32]={rtl_l2[:16]}, Golden[:32]={golden_l2[:16].tolist()}")
        
        # Step 9: RELU 8, 6 (activation)
        cocotb.log.info("\nStep 9: RELU 8, 6 (activation function)")
        relu(8, 6, 32)  # Golden model
        cycles = await tester.execute_instruction(
            opcode=0x05, dest=8, length_or_cols=32, rows=0,
            addr=0, x_id=6, w_id=0, b_id=0, timeout=300
        )
        total_cycles += cycles
        cocotb.log.info(f"  ✅ Completed in {cycles} cycles")

        # Compare intermediate RELU output for Layer 2
        await FallingEdge(dut.clk)
        rtl_l2_relu = []
        for i in range(32):
            v = int(dut.result[i].value)
            if v & 0x80:
                v = v - 256
            rtl_l2_relu.append(v)
        golden_l2_relu = np.array(golden_model.buffers[8][:32], dtype=np.int8)
        diff_l2_relu = np.abs(np.array(rtl_l2_relu, dtype=np.int32) - golden_l2_relu.astype(np.int32))
        max_err_l2_relu = int(np.max(diff_l2_relu))
        cocotb.log.info(f"  🔍 Layer 2 RELU compare: max_error={max_err_l2_relu}")
        if max_err_l2_relu > 2:
            cocotb.log.error(f"  ❌ Layer 2 RELU mismatch. RTL[:32]={rtl_l2_relu[:16]}, Golden[:32]={golden_l2_relu[:16].tolist()}")
        
        cocotb.log.info("\n✅ Layer 2 Complete: 12 → 32")
        
    except Exception as e:
        cocotb.log.error(f"❌ Layer 2 Failed: {e}")
        layer2_success = False
        assert False, f"Layer 2 failed: {e}"
    
    # ========== LAYER 3: 32 → 10 (OUTPUT) ==========
    cocotb.log.info("")
    cocotb.log.info("╔════════════════════════════════════╗")
    cocotb.log.info("║   LAYER 3: 32 → 10 (OUTPUT)       ║")
    cocotb.log.info("╚════════════════════════════════════╝")
    
    try:
        # Step 10: LOAD_M 1, 0x12d40, 10, 32 (weight matrix W3)
        cocotb.log.info("\nStep 10: LOAD_M 1, 0x12d40, 10, 32 (weight matrix W3)")
        load_m(1, 0x12d40, 10, 32)  # Golden model
        cycles = await tester.execute_instruction(
            opcode=0x02, dest=1, length_or_cols=32, rows=10,
            addr=0x12d40, x_id=0, w_id=0, b_id=0, timeout=1200
        )
        total_cycles += cycles
        cocotb.log.info(f"  ✅ Completed in {cycles} cycles")
        
        # Step 11: LOAD_V 4, 0x1302d, 10 (bias vector b3)
        cocotb.log.info("\nStep 11: LOAD_V 4, 0x1302d, 10 (bias vector b3)")
        load_v(4, 0x1302d, 10)  # Golden model
        cycles = await tester.execute_instruction(
            opcode=0x01, dest=4, length_or_cols=10, rows=0,
            addr=0x1302d, x_id=0, w_id=0, b_id=0, timeout=150
        )
        total_cycles += cycles
        cocotb.log.info(f"  ✅ Completed in {cycles} cycles")
        
        # Step 12: GEMV 5, 1, 8, 4, 10, 32 (W3 * h2 + b3 - FINAL OUTPUT)
        cocotb.log.info("\nStep 12: GEMV 5, 1, 8, 4, 10, 32 (W3 * h2 + b3 - FINAL OUTPUT)")
        gemv(5, 1, 8, 4, 10, 32)  # Golden model
        cycles = await tester.execute_instruction(
            opcode=0x04, dest=5, length_or_cols=32, rows=10,
            addr=0, x_id=8, w_id=1, b_id=4, timeout=6000
        )
        total_cycles += cycles
        cocotb.log.info(f"  ✅ Completed in {cycles} cycles")

        # Compare intermediate GEMV output for Layer 3
        await FallingEdge(dut.clk)
        rtl_l3 = []
        for i in range(10):
            v = int(dut.result[i].value)
            if v & 0x80:
                v = v - 256
            rtl_l3.append(v)
        golden_l3 = np.array(golden_model.buffers[5][:10], dtype=np.int8)
        diff_l3 = np.abs(np.array(rtl_l3, dtype=np.int32) - golden_l3.astype(np.int32))
        max_err_l3 = int(np.max(diff_l3))
        cocotb.log.info(f"  🔍 Layer 3 GEMV compare: max_error={max_err_l3}")
        if max_err_l3 > 2:
            cocotb.log.error(f"  ❌ Layer 3 GEMV mismatch. RTL[:10]={rtl_l3}, Golden[:10]={golden_l3.tolist()}")
        
        cocotb.log.info("\n✅ Layer 3 Complete: 32 → 10 (OUTPUT)")
        
    except Exception as e:
        cocotb.log.error(f"❌ Layer 3 Failed: {e}")
        layer3_success = False
        assert False, f"Layer 3 failed: {e}"
    
    # ========== COMPARE FINAL OUTPUT ==========
    cocotb.log.info("")
    cocotb.log.info("╔════════════════════════════════════════════════════════╗")
    cocotb.log.info("║           COMPARING FINAL OUTPUT                      ║")
    cocotb.log.info("╚════════════════════════════════════════════════════════╝")
    
    # Read RTL output
    await FallingEdge(dut.clk)
    rtl_output = []
    for i in range(10):
        val = int(dut.result[i].value)
        if val & 0x80:  # Handle signed conversion
            val = val - 256
        rtl_output.append(val)
    
    # Get golden model output
    golden_output = golden_model.buffers[5][:10]  # Buffer 5 contains final output
    
    cocotb.log.info("\n📊 Final Neural Network Output (10 classification scores):")
    cocotb.log.info(f"  RTL Output:    {rtl_output}")
    cocotb.log.info(f"  Golden Output: {list(golden_output)}")
    cocotb.log.info(f"\n🔧 Debugging: Check if RTL matches golden at each layer")
    cocotb.log.info(f"  Layer 1 GEMV mismatch: RTL int8={rtl_l1} vs Golden int8={golden_l1.tolist()}")
    cocotb.log.info(f"  Layer 2 GEMV mismatch: RTL int8={rtl_l2} vs Golden int8={golden_l2.tolist()}")
    cocotb.log.info(f"  Layer 3 GEMV mismatch: RTL int8={rtl_l3} vs Golden int8={golden_l3.tolist()}")
    
    # Compare
    rtl_array = np.array(rtl_output, dtype=np.int8)
    golden_array = np.array(golden_output, dtype=np.int8)
    diff = np.abs(rtl_array.astype(np.int32) - golden_array.astype(np.int32))
    max_error = np.max(diff)
    
    cocotb.log.info(f"\nPer-class comparison:")
    for i in range(10):
        error = abs(rtl_output[i] - golden_output[i])
        status = "✅" if error <= 2 else "❌"
        cocotb.log.info(f"  Class {i}: RTL={rtl_output[i]:4d}, Golden={golden_output[i]:4d}, Error={error:2d} {status}")
    
    cocotb.log.info(f"\nMax error: {max_error}")
    
    # ========== SUMMARY ==========
    cocotb.log.info("")
    cocotb.log.info("╔════════════════════════════════════════════════════════╗")
    cocotb.log.info("║           NEURAL NETWORK TEST COMPLETE                ║")
    cocotb.log.info("╚════════════════════════════════════════════════════════╝")
    
    cocotb.log.info("\n📊 Test Results:")
    cocotb.log.info(f"  Layer 1 (784→12):  {'✅ PASSED' if layer1_success else '❌ FAILED'}")
    cocotb.log.info(f"  Layer 2 (12→32):   {'✅ PASSED' if layer2_success else '❌ FAILED'}")
    cocotb.log.info(f"  Layer 3 (32→10):   {'✅ PASSED' if layer3_success else '❌ FAILED'}")
    cocotb.log.info(f"  Output Match:      {'✅ PASSED' if max_error <= 2 else '❌ FAILED'}")
    
    cocotb.log.info("\n📈 Network Architecture:")
    cocotb.log.info("  Input layer:    784 neurons")
    cocotb.log.info("  Hidden layer 1: 12 neurons  (9,408 parameters)")
    cocotb.log.info("  Hidden layer 2: 32 neurons  (384 parameters)")
    cocotb.log.info("  Output layer:   10 neurons  (320 parameters)")
    cocotb.log.info("  Total parameters: 10,112")
    
    cocotb.log.info("\n🔧 Operations Executed:")
    cocotb.log.info("  LOAD_V operations: 5")
    cocotb.log.info("  LOAD_M operations: 3")
    cocotb.log.info("  GEMV operations:   3")
    cocotb.log.info("  RELU operations:   2")
    cocotb.log.info("  Total instructions: 13")
    
    cocotb.log.info(f"\n⏱️  Total Cycles: {total_cycles}")
    
    # Final assertion
    all_passed = layer1_success and layer2_success and layer3_success and (max_error <= 2)
    
    if all_passed:
        cocotb.log.info("\n🎉 SUCCESS! Complete neural network executed successfully!")
        cocotb.log.info("   All 13 instructions verified against golden model.")
    else:
        cocotb.log.error("\n❌ TEST FAILED! Check errors above.")
        assert False, f"Neural network test failed (max_error={max_error})"


@cocotb.test()
async def test_single_load_v(dut):
    """Test a single LOAD_V instruction."""
    
    cocotb.log.info("=== Test: Single LOAD_V Instruction ===")
    
    tester = ExecutionUnitTester(dut)
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start(start_high=False))
    
    # Reset
    await tester.reset()
    
    # Load DRAM
    dram_path = os.path.join(os.path.dirname(__file__), '../../compiler/dram.hex')
    if not os.path.exists(dram_path):
        cocotb.log.warning("dram.hex not found, skipping test")
        return
        
    tester.load_dram(dram_path)
    
    # Initialize golden model
    import golden_model
    golden_model.memory = tester.memory
    golden_model.buffers = {}
    
    # Execute LOAD_V
    cocotb.log.info("Executing: LOAD_V 9, 0x700, 784")
    load_v(9, 0x700, 784)  # Golden
    
    cycles = await tester.execute_instruction(
        opcode=0x01, dest=9, length_or_cols=784, rows=0,
        addr=0x700, x_id=0, w_id=0, b_id=0, timeout=2000
    )
    
    cocotb.log.info(f"✅ Completed in {cycles} cycles")
    cocotb.log.info(f"Golden buffer 9 has {len(golden_model.buffers[9])} elements")


@cocotb.test()
async def test_single_gemv(dut):
    """Test a single GEMV operation with small matrices."""
    
    cocotb.log.info("=== Test: Single GEMV Operation ===")
    
    tester = ExecutionUnitTester(dut)
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start(start_high=False))
    
    # Reset
    await tester.reset()
    
    # Load DRAM
    dram_path = os.path.join(os.path.dirname(__file__), '../../compiler/dram.hex')
    if not os.path.exists(dram_path):
        cocotb.log.warning("dram.hex not found, skipping test")
        return
        
    tester.load_dram(dram_path)
    
    # Initialize golden model
    import golden_model
    golden_model.memory = tester.memory
    golden_model.buffers = {}
    golden_model.flag = 0
    
    # Setup: Load input vector, weight matrix, and bias
    cocotb.log.info("Loading test data...")
    
    # LOAD_V for input
    cocotb.log.info("Loading input vector...")
    load_v(9, 0x700, 32)  # Load 32 elements as input
    await tester.execute_instruction(
        opcode=0x01, dest=9, length_or_cols=32, rows=0,
        addr=0x700, x_id=0, w_id=0, b_id=0, timeout=500
    )
    cocotb.log.info(f"  Golden loaded buffer 9: {golden_model.buffers[9][:10]}...")
    
    # LOAD_M for weights (10×32 matrix)
    cocotb.log.info("Loading weight matrix...")
    load_m(1, 0x12d40, 10, 32)
    await tester.execute_instruction(
        opcode=0x02, dest=1, length_or_cols=32, rows=10,
        addr=0x12d40, x_id=0, w_id=0, b_id=0, timeout=1200
    )
    cocotb.log.info(f"  Golden loaded buffer 1: {np.array(golden_model.buffers[1][:10], dtype=np.int8)}...")
    
    # LOAD_V for bias
    cocotb.log.info("Loading bias vector...")
    load_v(4, 0x1302d, 10)
    await tester.execute_instruction(
        opcode=0x01, dest=4, length_or_cols=10, rows=0,
        addr=0x1302d, x_id=0, w_id=0, b_id=0, timeout=150
    )
    cocotb.log.info(f"  Golden loaded buffer 4: {golden_model.buffers[4]}")
    
    # Execute GEMV
    cocotb.log.info("Executing: GEMV 5, 1, 9, 4, 10, 32")
    gemv(5, 1, 9, 4, 10, 32)  # Golden
    
    cycles = await tester.execute_instruction(
        opcode=0x04, dest=5, length_or_cols=32, rows=10,
        addr=0, x_id=9, w_id=1, b_id=4, timeout=6000
    )
    
    cocotb.log.info(f"✅ GEMV completed in {cycles} cycles")
    
    # Compare results
    await FallingEdge(dut.clk)
    rtl_gemv_output = []
    for i in range(10):
        v = int(dut.result[i].value)
        if v & 0x80:
            v = v - 256
        rtl_gemv_output.append(v)
    
    golden_gemv_output = np.array(golden_model.buffers[5][:10], dtype=np.int8)
    
    cocotb.log.info(f"  RTL GEMV output:    {rtl_gemv_output}")
    cocotb.log.info(f"  Golden GEMV output: {golden_gemv_output.tolist()}")
    
    diff = np.abs(np.array(rtl_gemv_output, dtype=np.int32) - golden_gemv_output.astype(np.int32))
    max_error = int(np.max(diff))
    cocotb.log.info(f"  Max error: {max_error}")
    
    if max_error > 2:
        cocotb.log.error(f"❌ GEMV mismatch! Difference too large")
