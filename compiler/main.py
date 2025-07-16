import numpy as np
import onnx
from onnx import numpy_helper
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader 
# Your functions/modules
from compile import generate_assembly
from helper_functions import print_weights_in_order, quantize_tensor_f32_int8
from model import create_mlp_model, run_model
from silver_model import execute_program
from dram import save_initializers_to_dram, save_input_to_dram, save_dram_to_file, read_from_dram

def evaluate_design(seed, torch_input, label):
    # 1. Create and save the model
    create_mlp_model()
    model_path = "mlp_model.onnx"

    # 2. DRAM configuration

    dram_offsets = {
        "inputs":  0x100000,
        "weights": 0x200000,
        "biases":  0x300000,
        "outputs": 0x400000,
    }

    # 3. Save weights/biases to DRAM
    weight_map, bias_map = save_initializers_to_dram(model_path, dram_offsets)
    torch.manual_seed(seed)
    # 4. Save dummy input to DRAM
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
    max_index = np.argmax(output_design)
    print("Output from the design:", output_design)
    print("Expected label:", label)
    print("Max index from the design:", max_index)
    return max_index == label.item()


if __name__ == "__main__":
    sum = 0
    total_elements = 0
    # 1. Define transformations (e.g., convert to tensor and normalize)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts PIL image to Tensor [0,1]
        transforms.Normalize((0.1307,), (0.3081,))  # Mean and std from MNIST dataset
    ])

    # 2. Load the training and test datasets
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # 3. Wrap them in DataLoader for batching and shuffling
    test_images = torch.stack([img for img, _ in test_dataset])  # shape: [10000, 1, 28, 28]
    test_labels = torch.tensor([label for _, label in test_dataset])  # shape: [10000]
    for i in range(len(test_labels)):
        output_design = evaluate_design(i, test_images[i], test_labels[i])  # Random input tensor and label
        sum += output_design
        if i % 10 == 0:
            print(f"{i+1} runs completed, current accuracy: {sum / (i + 1) * 100}%")

    accuracy = sum / len(test_labels) * 100  # Convert to percentage
    print("Average accuracy over all runs:", accuracy)