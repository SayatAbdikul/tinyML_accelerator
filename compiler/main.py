import onnx
from onnx import numpy_helper
from compile import generate_assembly
model_path = "mlp_model.onnx"
model = onnx.load(model_path)

generate_assembly(model_path, "model_assembly.asm")