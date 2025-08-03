""" dram.py - A module for managing DRAM in a custom architecture.
This module provides functions to read and write data to a simulated DRAM,
save initializers from an ONNX model, and manage memory operations.
It includes functions to handle quantization of tensors, write to DRAM,
and read from DRAM."""

import numpy as np
import onnx
from onnx import numpy_helper
from helper_functions import quantize_tensor_f32_int8

MEM_SIZE = 0x100BB8  # Total memory size
dram = np.zeros(MEM_SIZE, dtype=np.int8)

def write_to_dram(array, start_addr):
    end_addr = start_addr + len(array)
    if dram[start_addr] != 0:
        print(f"DRAM address {hex(start_addr)} is already occupied")
        raise ValueError("DRAM address already occupied")
    if end_addr > len(dram):
        print(f"DRAM overflow: trying to write {len(array)} bytes at address {hex(start_addr)}")
        raise ValueError("DRAM overflow")
    dram[start_addr:end_addr] = array
    # print(f"Written {len(array)} bytes to DRAM at address {hex(start_addr)}")
    # if len(array) == 100:
    #     print(f"Data: {array}... (total {len(array)} bytes)")
    # print(f"DRAM state: {dram[start_addr:end_addr]}")
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
    Quantizes the tensors to int8 format and writes them to specified DRAM addresses."""
    global dram
    dram = np.zeros(MEM_SIZE, dtype=np.int8)
    model = onnx.load(model_path)
    graph = model.graph

    weight_ptr = dram_offsets["weights"]
    bias_ptr = dram_offsets["biases"]

    weight_map = {}
    bias_map = {}

    for init in graph.initializer:
        name = init.name
        array = numpy_helper.to_array(init)

        # Choose scale/zero_point per tensor or globally
        scale = np.max(np.abs(array)) / 127 if np.max(np.abs(array)) > 0 else 1.0

        quant_array = quantize_tensor_f32_int8(array, scale).flatten()
        # print("quant_array is: ", quant_array)
        # print("original array is: ", array)
        if len(array.shape) > 1:  # weight
            
            weight_map[name] = weight_ptr
            weight_ptr = write_to_dram(quant_array, weight_ptr)
        else:  # bias
            bias_map[name] = bias_ptr
            bias_ptr = write_to_dram(quant_array, bias_ptr)

    return weight_map, bias_map

def save_input_to_dram(input_tensor, addr):
    # print(f"Saving input tensor to DRAM at address {hex(addr)} with shape {input_tensor.shape}")
    write_to_dram(input_tensor, addr)

def save_dram_to_file(filename="dram.hex"):
    """Saves the current state of DRAM to a hex file."""
    # May be commented out to avoid overwriting to file in this example on each input
    with open(filename, "w") as f:
        for byte in dram:
            # Convert signed int8 to unsigned for hex
            val = np.uint8(byte)
            f.write(f"{val:02X}\n")