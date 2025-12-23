""" 
Golden model of the accelerator.
Emulates all the instructions.
"""
import os
import numpy as np
from dram import get_dram
from helper_functions import quantize_int32_to_int8
buffers = {}
output_length = 10
quantized_output_scale = 0.1
quantized_output_zero_point = 0
output_buffer = 0
def load_memory(dram_file):
    """Load memory from a hex file."""
    # In testing purposes, we don't use hex file here
    return get_dram()  # Use the global DRAM state


def i_decoder(instruction):
    opcode = instruction & 0x1F  # last 5 bits
    
    if opcode == 1: # LOAD_V
        dest = instruction >> 5 & 0x1F
        len = instruction >> 10 & 0x3FFFF # next 18 bits
        addr = instruction >> 40 & 0xFFFFFF # next 24 bits 
        load_v(dest, addr, len)
        # print(f"LOAD_V: dest={dest}, addr={addr:#010x}, len={len}")

    elif opcode == 2: # LOAD_M
        dest = instruction >> 5 & 0x1F
        cols = instruction >> 10 & 0x3FF # next 10 bits
        rows = instruction >> 20 & 0x3FF # next 10 bits
        addr = instruction >> 40 & 0xFFFFFF # next 24 bits
        load_m(dest, addr, rows, cols)

    elif opcode == 3: # STORE
        buf_id = instruction >> 5 & 0x1F
        len = instruction >> 10 & 0x3FFFF # next 18 bits
        addr = instruction >> 40 & 0xFFFFFF # next 24 bits
        store(buf_id, addr, len)

    elif opcode == 4: # GEMV
        dest = instruction >> 5 & 0x1F
        cols = instruction >> 10 & 0x3FF
        rows = instruction >> 20 & 0x3FF
        b = instruction >> 30 & 0x1F
        x = instruction >> 35 & 0x1F
        w = instruction >> 40 & 0x1F
        gemv(dest, w, x, b, rows, cols)
    elif opcode == 5: # RELU
        dest = instruction >> 5 & 0x1F
        x = instruction >> 10 & 0x1F
        length = instruction >> 20 & 0x3FF  # length is at bits 20-29 (10 bits)
        relu(dest, x, length)
    else:
        return f"UNKNOWN_OPCODE: {opcode}"

def load_v(dest, addr, length):
    """Load vector from memory to buffer."""
    # if addr == 0x100000:  # Print the input
    #   print(f"INPUT: LOAD_V on buffer {dest}: the array is {memory[addr:addr + length]} at address {addr:#010x} with length {length}")
    buffers[dest] = memory[addr:addr + length]  # Load vector from memory to buffer

def load_m(dest, addr, rows, cols):
    buffers[dest] = memory[addr:addr + rows * cols]  
    # if addr == 0x20000:  # Print the weights
    #     print(f"LOAD_M on buffer {dest}: the matrix is {buffers[dest]} at address {addr:#010x} with rows={rows}, cols={cols}")
    # print
    # print("The number of unique values in the output buffer of matrix is:", np.unique(buffers[dest]).size)


def store(buf_id, addr, length):
    """Store buffer to memory."""
    # output_length = length
    # quantized_output_scale = np.max(np.abs(buffers[buf_id])) / 127
    # buffers[buf_id] = quantize_int32_to_int8(np.array(buffers[buf_id], dtype=np.int32), quantized_output_scale, quantized_output_zero_point)
    # print(f"STORE: buf_id={buf_id}, addr={addr:#010x}, length={length}, output_length={output_length}")
    for i in range(length):
        memory[addr + i] = buffers[buf_id][i]  # Extract byte from buffer
    global output_buffer
    output_buffer = buf_id  # Update the output buffer

flag = 0
def gemv(dest, w, x, b, rows, cols):
    """Perform GEMV operation."""
    global flag
    buffers[dest] = [0] * rows  # Initialize destination buffer with zeros
    # print(f"GEMV: dest={dest}, w={w}, x={x}, b={b}, rows={rows}, cols={cols}")
    for i in range(rows):
        sum = np.int32(0)  # Initialize sum as int32 to avoid overflow
        for j in range(cols):
            sum += np.int32(buffers[w][i * cols + j]) * np.int32(buffers[x][j])
            # Matrix-vector multiplication
            # if flag == 2:  # Print only once
            #     print(f"Multiplying w[{i * cols + j}]={buffers[w][i * cols + j]} with x[{j}]={buffers[x][j]} to sum={sum}")

        sum += buffers[b][i]
        buffers[dest][i] = np.int32(sum)  # Store the result in the destination buffer
    # print("The number of unique values in the output buffer is:", np.unique(buffers[dest]).size)
    # print("The output buffer is:", buffers[dest])
    flag += 1  # Set flag to indicate GEMV has been executed
    quantized_output_scale = np.max(np.abs(buffers[dest])) / 127
    buffers[dest] = quantize_int32_to_int8(np.array(buffers[dest], dtype=np.int32), quantized_output_scale, quantized_output_zero_point)

def relu(dest, x, length):
    """Apply ReLU activation to specified number of elements."""
    buffers[dest] = [max(0, val) for val in buffers[x][:length]]  # Apply ReLU to first 'length' elements

def execute_program(hex_file):
    """Execute the program from a hex file."""
    # Clear global state for fresh execution
    global buffers, output_buffer, flag
    buffers = {}
    output_buffer = 0
    flag = 0
    
    with open(hex_file, 'r') as file:
        lines = [line.strip() for _, line in zip(range(0x700), file)]
        instructions = [''.join(lines[i:i+8]) for i in range(0, len(lines), 8)]
        # print(f"Instructions are: {instructions[0:13]}")
        instructions = [int(instruction, 16) for instruction in instructions if instruction]  # Convert hex to int
        # print(f"Instructions are: {instructions[0:13]}")
    global memory
    memory = load_memory('dram.hex')
    for instruction in instructions:
        i_decoder(instruction)

    return buffers[output_buffer][0:output_length]  # Return the final output buffer (assuming it's always buffer 6)
    