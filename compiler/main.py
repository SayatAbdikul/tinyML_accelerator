import numpy as np
import onnx
from onnx import numpy_helper

# Your functions/modules
from compile import generate_assembly
from helper_functions import print_weights_in_order
from model import create_mlp_model
from dram import save_initializers_to_dram, save_input_to_dram, save_dram_to_file

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

# 4. Save dummy input to DRAM
dummy_input = np.random.randn(1, 1, 28, 28).astype(np.float32)
save_input_to_dram(dummy_input, dram_offsets["inputs"])

# 5. Save DRAM to hex file
save_dram_to_file("dram.hex")

# 6. Generate assembly using same model
generate_assembly(model_path, "model_assembly.asm")

# 7. Optional: print the ordered weights and biases
print_weights_in_order(model_path)
