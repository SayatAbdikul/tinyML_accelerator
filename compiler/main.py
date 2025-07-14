import numpy as np
import onnx
from onnx import numpy_helper
import torch
# Your functions/modules
from compile import generate_assembly
from helper_functions import print_weights_in_order, quantize_tensor_f32_int8
from model import create_mlp_model, run_model
from silver_model import execute_program
from dram import save_initializers_to_dram, save_input_to_dram, save_dram_to_file, read_from_dram

def main(seed):
    # 1. Create and save the model
    create_mlp_model()
    model_path = "mlp_model.onnx"
    model = onnx.load(model_path)

    # 2. DRAM configuration

    dram_offsets = {
        "inputs":  0x10000,
        "weights": 0x20000,
        "biases":  0x30000,
        "outputs": 0x40000,
    }

    # 3. Save weights/biases to DRAM
    weight_map, bias_map = save_initializers_to_dram(model_path, dram_offsets)
    torch.manual_seed(seed)
    # 4. Save dummy input to DRAM
    torch_input = torch.randint(low=-128, high=128, size=(1, 1, 10, 10), dtype=torch.int8)
    torch_input = torch_input.to(torch.float32)  # Convert to int8
    dummy_input = torch_input.to(torch.int8).numpy().squeeze().flatten()
    save_input_to_dram(dummy_input, dram_offsets["inputs"])
    written_input = read_from_dram(dram_offsets["inputs"], len(dummy_input))
    if not np.array_equal(dummy_input, written_input):
        print("The length of the input tensor is", len(dummy_input))
        print("The input data is: ", dummy_input)
        print("The written input data is: ", written_input)
        raise ValueError("Input data mismatch after writing to DRAM")

    # 5. Save DRAM to hex file
    save_dram_to_file("dram.hex")

    # 6. Generate assembly using same model
    generate_assembly(model_path, "model_assembly.asm")

    # # 7. Optional: print the ordered weights and biases
    # print_weights_in_order(model_path)

    # 8. Assemble the model to a hex file
    from assembler import assemble_file
    assemble_file("model_assembly.asm", "program.hex")

    output_design = execute_program("program.hex")
    # Execute the program from the hex file
    output_model = run_model(torch_input).numpy().squeeze().flatten()

    # Quantize the model output
    scale = 0.1
    zero_point = 0
    quantized_output_model = quantize_tensor_f32_int8(output_model, scale, zero_point)
    return output_design, quantized_output_model

def calculate_difference(output_model, quantized_output_model):
    """
    Calculate the difference between the original output and the quantized output.
    """
    diff = output_model.astype(np.int16) - quantized_output_model.astype(np.int16)
    return diff

if __name__ == "__main__":
    sum = 0
    for i in range(100):
        output_design, quantized_output_model = main(i)
        sum += np.sum(calculate_difference(output_design, quantized_output_model))
        print("The design output: ", output_design)
        print("The model output: ", quantized_output_model)
    sum /= 1
    print("Average difference over 100 runs:", sum)
    # You can add more checks or validations here if needed