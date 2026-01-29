""" dram.py - A module for managing DRAM in a custom architecture.
This module provides functions to read and write data to a simulated DRAM,
save initializers from an ONNX model, and manage memory operations.
It includes functions to handle quantization of tensors, write to DRAM,
and read from DRAM."""

import numpy as np
import onnx
from onnx import numpy_helper
from helper_functions import quantize_tensor_f32_int8
from top_sort import topological_sort
from accelerator_config import AcceleratorConfig

MEM_SIZE = AcceleratorConfig.MEM_SIZE  # Total memory size (Reduced to 60KB for FPGA fit)
dram = np.zeros(MEM_SIZE, dtype=np.int8)

def write_to_dram(array, start_addr):
    end_addr = start_addr + len(array)
    # Check for overflow but allow overwriting (warning optional or removed for repeated runs)
    if end_addr > len(dram):
        print(f"DRAM overflow: trying to write {len(array)} bytes at address {hex(start_addr)}")
        raise ValueError("DRAM overflow")
        
    dram[start_addr:end_addr] = array
    # print(f"Written {len(array)} bytes to DRAM at address {hex(start_addr)}")
    return end_addr  # Return next free address

def read_from_dram(start_addr, length):
    end_addr = start_addr + length
    if end_addr > len(dram):
        print(f"DRAM overflow: trying to read {length} bytes from address {hex(start_addr)}")
        raise ValueError("DRAM overflow")
    data = dram[start_addr:end_addr]
    # print(f"Read {length} bytes from DRAM at address {hex(start_addr)}")
    return np.array(dram[start_addr:end_addr], dtype=np.int8)

def get_dram():
    return dram.copy()  # Return a copy to avoid external modifications

def save_initializers_to_dram(model_path, dram_offsets):
    """Saves the initializers (weights and biases) from an ONNX model to DRAM.
    Quantizes the tensors to int8 format and writes them to specified DRAM addresses.
    Ensures that writing follows the topological order of graph execution."""
    global dram
    dram = np.zeros(MEM_SIZE, dtype=np.int8)
    model = onnx.load(model_path)
    graph = model.graph

    weight_ptr = dram_offsets["weights"]
    bias_ptr = dram_offsets["biases"]

    weight_map = {}
    bias_map = {}
    
    # Pre-process initializers into a map for easy lookup
    initializer_data = {}
    for init in graph.initializer:
        initializer_data[init.name] = init

    # Use existing helper or topological sort to traverse in execution order
    ordered_nodes = topological_sort(graph)
    visited_initializers = set()

    # Traverse nodes in topological order
    for node in ordered_nodes:
        # mirror compile.py logic: skip Reshape nodes
        if node.op_type == "Reshape":
            continue

        for input_name in node.input:
            if input_name in initializer_data and input_name not in visited_initializers:
                visited_initializers.add(input_name)
                init = initializer_data[input_name]
                array = numpy_helper.to_array(init)

                # Choose scale/zero_point per tensor or globally
                scale = np.max(np.abs(array)) / 127 if np.max(np.abs(array)) > 0 else 1.0

                if len(array.shape) > 1:  # weight
                    rows, cols = array.shape
                    TILE_WIDTH = AcceleratorConfig.TILE_ELEMS
                    padded_cols = ((cols + TILE_WIDTH - 1) // TILE_WIDTH) * TILE_WIDTH
                    
                    # Pad rows to TILE_WIDTH
                    padded_array = np.zeros((rows, padded_cols), dtype=np.int8)
                    q_array = quantize_tensor_f32_int8(array, scale)
                    padded_array[:, :cols] = q_array
                    
                    # Verify padding
                    padding_elements = padded_array[:, cols:].flatten()
                    non_zero_padding = np.count_nonzero(padding_elements)
                    if non_zero_padding > 0:
                        print(f"ERROR: Padding contains {non_zero_padding} non-zero elements for {input_name}")
                    else:
                        # print(f"DEBUG_DRAM: Padding verified zero for {input_name}. Cols={cols}, Padded={padded_cols}")
                        # print(f"DEBUG_DRAM: Last 10 elements of row 0: {padded_array[0, cols-10:cols]}")
                        # print(f"DEBUG_DRAM: First 10 elements of padding row 0: {padded_array[0, cols:cols+10]}")
                        pass

                    quant_array = padded_array.flatten()
                    
                    weight_map[input_name] = weight_ptr
                    weight_ptr = write_to_dram(quant_array, weight_ptr)
                else:  # bias
                    quant_array = quantize_tensor_f32_int8(array, scale).flatten()
                    bias_map[input_name] = bias_ptr
                    bias_ptr = write_to_dram(quant_array, bias_ptr)

    # Some initializers might not be inputs to any node in the graph (e.g. unused)
    # We ignore them as compile.py would also ignore them.

    return weight_map, bias_map

def save_input_to_dram(input_tensor, addr):
    # print(f"Saving input tensor to DRAM at address {hex(addr)} with shape {input_tensor.shape}")
    write_to_dram(input_tensor, addr)

def save_dram_to_file(filename="dram.hex"):
    """Saves the current state of DRAM to a hex file."""
    counter = 0
    # May be commented out to avoid overwriting to file in this example on each input
    with open(filename, "w") as f:
        for byte in dram:
            # Convert signed int8 to unsigned for hex
            val = np.uint8(byte)
            f.write(f"{val:02X}\n")
            counter = counter + 1
