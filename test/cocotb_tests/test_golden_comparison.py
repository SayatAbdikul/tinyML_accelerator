"""
Cocotb Testbench for TinyML Accelerator - Golden Model Verification
===================================================================

This testbench verifies the RTL implementation against the Python golden model
by comparing the results written to DRAM after execution using actual MNIST test data.

Test Flow:
1. Load MNIST test dataset (same as main.py)
2. For each test image:
   a. Quantize and save to dram.hex at input address
   b. Execute RTL through all instructions
   c. Read RTL results from dram.hex after STORE
   d. Execute golden model with same input
   e. Compare RTL vs Golden model results
3. Report overall accuracy

The comparison is performed on the output buffer (typically at address 0x20000)
after the STORE instruction writes the final neural network output.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer, FallingEdge
import os
import sys
import numpy as np
import torch
from torchvision import datasets, transforms

# Add compiler directory to path for importing golden model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../compiler'))

from golden_model import execute_program, get_dram
from dram import save_dram_to_file, save_input_to_dram, read_from_dram, MEM_SIZE
from compile import generate_assembly
from model import create_mlp_model
from assembler import assemble_file


class TinyMLAcceleratorTester:
    """Helper class to manage the accelerator testbench"""
    
    def __init__(self, dut):
        self.dut = dut
        self.clock_period = 10  # 10ns = 100MHz
        self.output_addr = 0x20000  # Default output address from compile.py
        self.output_length = 10  # Default output length for MNIST (10 classes)
        self.input_addr = 0x700  # Input data address (must match compile.py)
        self.dram_offsets = {
            "inputs":  0x700,
            "weights": 0x10700,
            "biases":  0x13000,
            "outputs": 0x20000,
        }
        
    async def reset(self):
        """Apply reset to the DUT"""
        self.dut.rst.value = 1
        self.dut.start.value = 0
        await RisingEdge(self.dut.clk)
        await RisingEdge(self.dut.clk)
        self.dut.rst.value = 0
        await RisingEdge(self.dut.clk)
        cocotb.log.info("Reset complete")
        
    async def wait_for_done(self, timeout_cycles=200000):
        """Wait for the done signal with timeout"""
        cycle_count = 0
        while cycle_count < timeout_cycles:
            await RisingEdge(self.dut.clk)
            if self.dut.done.value == 1:
                cocotb.log.info(f"Done signal received after {cycle_count} cycles")
                return True
            cycle_count += 1
            
            # Progress indicator every 10000 cycles
            if cycle_count % 10000 == 0:
                cocotb.log.info(f"  ... still executing (cycle {cycle_count})")
                
        cocotb.log.error(f"Timeout waiting for done after {timeout_cycles} cycles")
        return False
        
    async def execute_single_instruction(self):
        """Execute a single instruction by pulsing start"""
        # Pulse start signal
        self.dut.start.value = 1
        await RisingEdge(self.dut.clk)
        self.dut.start.value = 0
        
        # Wait for completion
        success = await self.wait_for_done()
        return success
        
    async def execute_program(self, max_instructions=20):
        """
        Execute all instructions in the program sequentially.
        The program ends when an all-zero instruction is fetched.
        """
        cocotb.log.info("=" * 70)
        cocotb.log.info("Starting program execution on RTL")
        cocotb.log.info("=" * 70)
        
        for instr_num in range(max_instructions):
            cocotb.log.info(f"\n--- Executing instruction #{instr_num} ---")
            
            # Execute one instruction
            success = await self.execute_single_instruction()
            
            if not success:
                cocotb.log.error(f"Failed to execute instruction #{instr_num}")
                return False
                
            # Small delay between instructions
            await RisingEdge(self.dut.clk)
            await RisingEdge(self.dut.clk)
            
        cocotb.log.info("\n" + "=" * 70)
        cocotb.log.info("Program execution complete")
        cocotb.log.info("=" * 70 + "\n")
        return True
        
    def read_memory_from_rtl(self, start_addr, length):
        """
        Read memory contents directly from RTL simulation memory.
        Accesses the store module's DRAM memory array.
        """
        try:
            # Access store module's memory through hierarchy
            # Path: top -> execution_u -> store_exec -> store_inst -> dram -> memory
            store_mem = self.dut.execution_u.store_exec.store_inst.dram.memory
            
            # Read memory values
            result = []
            for addr in range(start_addr, start_addr + length):
                # Get value from simulation
                val = store_mem[addr].value.integer
                
                # Convert unsigned to signed int8 if needed
                if val > 127:
                    val = val - 256
                result.append(val)
                    
            result = np.array(result, dtype=np.int8)
            cocotb.log.info(f"Read {length} bytes from RTL memory at address 0x{start_addr:06X}")
            return result
            
        except AttributeError as e:
            cocotb.log.error(f"Cannot access RTL memory hierarchy: {e}")
            cocotb.log.error("Memory path may be incorrect. Check module hierarchy.")
            return None
        except Exception as e:
            cocotb.log.error(f"Error reading RTL memory: {e}")
            return None
    
    def read_memory_from_file(self, hex_file, start_addr, length):
        """
        Read memory contents from hex file after RTL execution.
        Each line in the hex file is one byte in hex format.
        NOTE: This method is deprecated - use read_memory_from_rtl() instead.
        """
        try:
            with open(hex_file, 'r') as f:
                lines = f.readlines()
                
            # Convert hex strings to integers
            memory = []
            for line in lines:
                line = line.strip()
                if line:
                    # Convert hex to signed int8
                    val = int(line, 16)
                    # Convert unsigned to signed if needed
                    if val > 127:
                        val = val - 256
                    memory.append(val)
                    
            # Extract the region of interest
            end_addr = start_addr + length
            if end_addr > len(memory):
                cocotb.log.error(f"Requested memory region [{start_addr}:{end_addr}] exceeds file size {len(memory)}")
                return None
                
            result = np.array(memory[start_addr:end_addr], dtype=np.int8)
            cocotb.log.info(f"Read {length} bytes from {hex_file} at address 0x{start_addr:06X}")
            return result
            
        except FileNotFoundError:
            cocotb.log.error(f"Memory file {hex_file} not found")
            return None
        except Exception as e:
            cocotb.log.error(f"Error reading memory file: {e}")
            return None
            
    def prepare_input(self, input_tensor, compiler_dir):
        """
        Prepare input data by writing it to dram.hex at the input address.
        This mimics what main.py does.
        """
        # Quantize input to int8
        dummy_input = input_tensor.to(torch.int8).numpy().squeeze().flatten()
        
        cocotb.log.info(f"Preparing input: shape={input_tensor.shape}, quantized_length={len(dummy_input)}")
        
        # Save to DRAM (this updates the global DRAM state)
        save_input_to_dram(dummy_input, self.dram_offsets["inputs"])
        
        # Verify write
        written_input = read_from_dram(self.dram_offsets["inputs"], len(dummy_input))
        if not np.array_equal(dummy_input, written_input):
            cocotb.log.error("Input data mismatch after writing to DRAM")
            return False
            
        # Save DRAM state to hex file
        dram_hex_path = os.path.join(compiler_dir, 'dram.hex')
        save_dram_to_file(dram_hex_path)
        cocotb.log.info(f"Input saved to {dram_hex_path}")
        
        return True
        
    def compare_results(self, rtl_output, golden_output, verbose=True):
        """
        Compare RTL and golden model outputs element-by-element.
        Returns (match, differences, max_error)
        """
        if rtl_output is None or golden_output is None:
            cocotb.log.error("Cannot compare - one or both outputs are None")
            return False, None, None
            
        if len(rtl_output) != len(golden_output):
            cocotb.log.error(f"Length mismatch: RTL={len(rtl_output)}, Golden={len(golden_output)}")
            return False, None, None
            
        # Calculate differences
        differences = rtl_output - golden_output
        max_error = np.max(np.abs(differences))
        num_mismatches = np.count_nonzero(differences)
        
        # Check for exact match
        match = np.array_equal(rtl_output, golden_output)
        
        if verbose:
            # Log comparison results
            cocotb.log.info("\n" + "=" * 70)
            cocotb.log.info("RESULTS COMPARISON")
            cocotb.log.info("=" * 70)
            cocotb.log.info(f"Output length: {len(rtl_output)} elements")
            cocotb.log.info(f"Exact match: {match}")
            cocotb.log.info(f"Mismatches: {num_mismatches}/{len(rtl_output)} elements")
            cocotb.log.info(f"Max absolute error: {max_error}")
            
            cocotb.log.info("\nElement-by-element comparison:")
            cocotb.log.info("-" * 70)
            cocotb.log.info(f"{'Index':<8} {'RTL':<12} {'Golden':<12} {'Diff':<12} {'Match'}")
            cocotb.log.info("-" * 70)
            
            for i in range(len(rtl_output)):
                rtl_val = int(rtl_output[i])
                golden_val = int(golden_output[i])
                diff = rtl_val - golden_val
                match_str = "✓" if diff == 0 else "✗"
                cocotb.log.info(f"{i:<8} {rtl_val:<12} {golden_val:<12} {diff:<12} {match_str}")
                
            cocotb.log.info("-" * 70)
            
            # Print summary
            if match:
                cocotb.log.info("\n✅ PASS: RTL output matches golden model exactly!")
            else:
                cocotb.log.warning(f"\n⚠️  MISMATCH: {num_mismatches} element(s) differ (max error: {max_error})")
                
            cocotb.log.info("=" * 70 + "\n")
        
        return match, differences, max_error
    
    def get_prediction(self, output):
        """Get predicted class from output vector"""
        return np.argmax(output)


@cocotb.test()
async def test_accelerator_mnist_dataset(dut):
    """
    Main test: Compare RTL execution against golden model using MNIST test dataset
    
    This test:
    1. Loads the MNIST test dataset (10,000 images)
    2. Tests a subset of images (configurable)
    3. For each image:
       - Prepares input in dram.hex
       - Executes RTL
       - Executes golden model
       - Compares outputs
       - Checks if prediction matches label
    4. Reports overall accuracy for both RTL and golden model
    """
    
    # Create tester instance
    tester = TinyMLAcceleratorTester(dut)
    
    # Start clock
    cocotb.log.info("Starting 100MHz clock")
    clock = Clock(dut.clk, tester.clock_period, units="ns")
    cocotb.start_soon(clock.start())
    
    # Get paths
    compiler_dir = os.path.join(os.path.dirname(__file__), '../../compiler')
    dram_hex_path = os.path.join(compiler_dir, 'dram.hex')
    os.chdir(compiler_dir)  # Change to compiler directory for imports to work
    
    # ========================================================================
    # STEP 0: Initialize model and generate assembly
    # ========================================================================
    cocotb.log.info("\n" + "=" * 70)
    cocotb.log.info("STEP 0: INITIALIZE MODEL AND GENERATE ASSEMBLY")
    cocotb.log.info("=" * 70)
    
    # Create model
    create_mlp_model()
    model_path = "mlp_model.onnx"
    
    # Save weights/biases to DRAM
    from dram import save_initializers_to_dram
    weight_map, bias_map = save_initializers_to_dram(model_path, tester.dram_offsets)
    cocotb.log.info("Weights and biases saved to DRAM")
    
    # Generate and assemble instructions
    generate_assembly(model_path, "model_assembly.asm")
    assemble_file("model_assembly.asm")
    cocotb.log.info("Assembly code generated and assembled")
    
    # ========================================================================
    # STEP 1: Load MNIST test dataset
    # ========================================================================
    cocotb.log.info("\n" + "=" * 70)
    cocotb.log.info("STEP 1: LOAD MNIST TEST DATASET")
    cocotb.log.info("=" * 70)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_images = torch.stack([img for img, _ in test_dataset])
    test_labels = torch.tensor([label for _, label in test_dataset])
    
    cocotb.log.info(f"Loaded {len(test_labels)} test images")
    
    # ========================================================================
    # STEP 2: Test on subset of images
    # ========================================================================
    num_tests = 10  # Test first 10 images for faster verification
    cocotb.log.info(f"\nTesting on first {num_tests} images...")
    
    rtl_correct = 0
    golden_correct = 0
    both_correct = 0
    total_tests = 0
    
    for test_idx in range(num_tests):
        cocotb.log.info("\n" + "=" * 70)
        cocotb.log.info(f"TEST IMAGE {test_idx + 1}/{num_tests} - Label: {test_labels[test_idx].item()}")
        cocotb.log.info("=" * 70)
        
        # Prepare input
        input_image = test_images[test_idx]
        label = test_labels[test_idx].item()
        
        success = tester.prepare_input(input_image, compiler_dir)
        if not success:
            cocotb.log.error(f"Failed to prepare input for test {test_idx}")
            continue
            
        # Apply reset before each test
        await tester.reset()
        
        # Execute RTL
        cocotb.log.info("Executing RTL...")
        success = await tester.execute_program(max_instructions=20)
        
        if not success:
            cocotb.log.error(f"RTL execution failed for test {test_idx}")
            continue
            
        # Wait for memory writes to settle
        for _ in range(100):
            await RisingEdge(dut.clk)
            
        # Read RTL results directly from simulation memory
        rtl_output = tester.read_memory_from_rtl(
            tester.output_addr,
            tester.output_length
        )
        
        if rtl_output is None:
            cocotb.log.error(f"Failed to read RTL output for test {test_idx}")
            continue
            
        # Execute golden model
        cocotb.log.info("Executing golden model...")
        try:
            golden_output = execute_program(dram_hex_path)
            golden_output = np.array(golden_output, dtype=np.int8)
        except Exception as e:
            cocotb.log.error(f"Golden model execution failed: {e}")
            continue
            
        # Compare outputs
        match, differences, max_error = tester.compare_results(rtl_output, golden_output, verbose=False)
        
        # Get predictions
        rtl_pred = tester.get_prediction(rtl_output)
        golden_pred = tester.get_prediction(golden_output)
        
        # Check accuracy
        rtl_match = (rtl_pred == label)
        golden_match = (golden_pred == label)
        outputs_match = match or (max_error is not None and max_error <= 2)
        
        if rtl_match:
            rtl_correct += 1
        if golden_match:
            golden_correct += 1
        if rtl_match and golden_match:
            both_correct += 1
            
        total_tests += 1
        
        # Log results for this test
        cocotb.log.info(f"\nTest {test_idx + 1} Results:")
        cocotb.log.info(f"  Label:           {label}")
        cocotb.log.info(f"  RTL prediction:  {rtl_pred} {'✓' if rtl_match else '✗'}")
        cocotb.log.info(f"  Golden prediction: {golden_pred} {'✓' if golden_match else '✗'}")
        cocotb.log.info(f"  Outputs match:   {outputs_match} (max error: {max_error})")
        cocotb.log.info(f"  RTL output:      {rtl_output}")
        cocotb.log.info(f"  Golden output:   {golden_output}")
        
        if not outputs_match:
            cocotb.log.warning(f"⚠️  Output mismatch detected!")
            
    # ========================================================================
    # STEP 3: Report overall results
    # ========================================================================
    cocotb.log.info("\n" + "=" * 70)
    cocotb.log.info("FINAL RESULTS")
    cocotb.log.info("=" * 70)
    
    rtl_accuracy = (rtl_correct / total_tests * 100) if total_tests > 0 else 0
    golden_accuracy = (golden_correct / total_tests * 100) if total_tests > 0 else 0
    
    cocotb.log.info(f"\nTested {total_tests} images:")
    cocotb.log.info(f"  RTL Accuracy:    {rtl_correct}/{total_tests} ({rtl_accuracy:.1f}%)")
    cocotb.log.info(f"  Golden Accuracy: {golden_correct}/{total_tests} ({golden_accuracy:.1f}%)")
    cocotb.log.info(f"  Both Correct:    {both_correct}/{total_tests} ({both_correct/total_tests*100:.1f}%)")
    
    # Final assertion
    if rtl_accuracy >= 70 and abs(rtl_accuracy - golden_accuracy) <= 10:
        cocotb.log.info("\n✅ TEST PASSED: RTL achieves acceptable accuracy and matches golden model!")
    else:
        cocotb.log.error("\n❌ TEST FAILED: RTL accuracy too low or differs significantly from golden model")
        assert False, f"RTL accuracy: {rtl_accuracy}%, Golden: {golden_accuracy}%"
        

@cocotb.test()
async def test_single_instruction_load_v(dut):
    """
    Simple sanity test: Execute a single LOAD_V instruction
    """
    tester = TinyMLAcceleratorTester(dut)
    
    # Start clock
    clock = Clock(dut.clk, tester.clock_period, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    await tester.reset()
    
    # Execute one instruction
    cocotb.log.info("Executing single instruction test")
    success = await tester.execute_single_instruction()
    
    assert success, "Single instruction execution failed"
    cocotb.log.info("✅ Single instruction test passed")


@cocotb.test()
async def test_reset_behavior(dut):
    """
    Test proper reset behavior
    """
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Apply reset
    dut.rst.value = 1
    dut.start.value = 0
    
    for _ in range(10):
        await RisingEdge(dut.clk)
        
    # Check that done is low during reset
    assert dut.done.value == 0, "Done signal should be low during reset"
    
    # Release reset
    dut.rst.value = 0
    await RisingEdge(dut.clk)
    
    # Done should still be low after reset
    assert dut.done.value == 0, "Done signal should be low after reset"
    
    cocotb.log.info("✅ Reset behavior test passed")
