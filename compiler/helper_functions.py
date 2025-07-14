import onnx
from onnx import numpy_helper
from top_sort import topological_sort
import numpy as np

def load_initializers(graph):
    init_map = {}
    for initializer in graph.initializer:
        array = numpy_helper.to_array(initializer)
        init_map[initializer.name] = {
            "array": array,
            "shape": array.shape,
            "dtype": str(array.dtype)
        }
    return init_map

def extract_shape(tensor_type): # Extracts the shape from a tensor type
    shape = []
    for dim in tensor_type.shape.dim:
        if dim.HasField('dim_value'):
            shape.append(dim.dim_value)
        elif dim.HasField('dim_param'):
            shape.append(dim.dim_param)
        else:
            shape.append("?")
    return shape # for example, [1, 28, 28] for a 2D image input

def build_tensor_shape_map(model): # Builds a map of tensor names to their shapes
    shape_map = {}
    for vi in model.graph.input:
        shape_map[vi.name] = extract_shape(vi.type.tensor_type)
    for vi in model.graph.output:
        shape_map[vi.name] = extract_shape(vi.type.tensor_type)
    for vi in model.graph.value_info:
        shape_map[vi.name] = extract_shape(vi.type.tensor_type)
    return shape_map # For example: {'input': [1, 784], 'fc1_output': [1, 128], 'output': [1, 10]}

def build_initializer_map(graph): # Builds a map of initializer names to their data, shape, and dtype
    init_map = {}
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        quantized_array = quantize_tensor_f32_int8(arr, scale=0.1, zero_point=0)
        init_map[init.name] = {
            "data": quantized_array,
            "shape": quantized_array.shape,
            "dtype": str(quantized_array.dtype),
            "type": "weight" if len(quantized_array.shape) > 1 else "bias"
        }
        
    return init_map

# Helper to calculate tensor size
def tensor_size(shape):
    size = 1
    for dim in shape:
        if isinstance(dim, int):
            size *= dim
        elif isinstance(dim, str) and dim.isdigit():
            size *= int(dim)
        elif dim == "?":
            size = 0  # Unknown size
    return size

def quantize_tensor_f32_int8(tensor, scale, zero_point = 0):
    return np.clip(np.round(tensor / scale + zero_point), -128, 127).astype(np.int8)

def quantize_int32_to_int8(x_int32, scale, zero_point):
    x_fp32 = x_int32.astype(np.float32) * scale
    x_rounded = np.round(x_fp32)
    x_quantized = x_rounded + zero_point
    x_clamped = np.clip(x_quantized, -128, 127)
    return x_clamped.astype(np.int8)

def print_weights_in_order(model_path):
    model = onnx.load(model_path)
    graph = model.graph

    sorted_nodes = topological_sort(graph)
    init_map = load_initializers(graph)

    print("📦 Weights and Biases in Topological Execution Order:\n")
    for idx, node in enumerate(sorted_nodes):
        print(f"{idx+1:>2}. 🧩 OpType: {node.op_type}")
        for input_name in node.input:
            if input_name in init_map:
                entry = init_map[input_name]
                kind = "Bias" if len(entry["shape"]) == 1 else "Weight"
                print(f"    └── {kind}: {input_name}")
                print(f"        Shape: {entry['shape']}, Dtype: {entry['dtype']}")
                print(f"        First values: {entry['array'].flatten()[:5]}")
        print()