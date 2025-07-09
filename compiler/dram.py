import numpy as np
import onnx
from onnx import numpy_helper
from helper_functions import quantize_tensor

MEM_SIZE = 0x50000  # Total memory size
dram = np.zeros(MEM_SIZE, dtype=np.uint8)

def write_to_dram(array, start_addr):
    end_addr = start_addr + len(array)
    if dram[start_addr] != 0:
        print(f"DRAM address {hex(start_addr)} is already occupied")
        raise ValueError("DRAM address already occupied")
    if end_addr > len(dram):
        print(f"DRAM overflow: trying to write {len(array)} bytes at address {hex(start_addr)}")
        raise ValueError("DRAM overflow")
    dram[start_addr:end_addr] = array
    print(f"Written {len(array)} bytes to DRAM at address {hex(start_addr)}")
    # print(f"DRAM state: {dram[start_addr:end_addr]}")
    return end_addr  # Return next free address

def save_initializers_to_dram(model_path, dram_offsets):
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
        scale = np.max(np.abs(array)) / 127.0 if np.max(np.abs(array)) > 0 else 1.0
        zero_point = 128  # for symmetric quantization

        quant_array = quantize_tensor(array, scale, zero_point).flatten()

        if len(array.shape) > 1:  # weight
            weight_map[name] = weight_ptr
            weight_ptr = write_to_dram(quant_array, weight_ptr)
        else:  # bias
            bias_map[name] = bias_ptr
            bias_ptr = write_to_dram(quant_array, bias_ptr)

    return weight_map, bias_map

def save_input_to_dram(input_tensor, addr):
    quant_input = quantize_tensor(input_tensor, scale=0.02, zero_point=128)
    write_to_dram(quant_input.flatten(), addr)

def save_dram_to_file(filename="dram.hex"):
    with open(filename, "w") as f:
        for byte in dram:
            f.write(f"{byte:02X}\n")
