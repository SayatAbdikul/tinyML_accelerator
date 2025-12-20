""" Creates an assembly code from ONNX file """
import onnx
from onnx import shape_inference
import numpy as np
from helper_functions import build_tensor_shape_map, build_initializer_map, topological_sort, tensor_size



def generate_assembly(model_path, output_file):
    model = shape_inference.infer_shapes(onnx.load(model_path)) 
    graph = model.graph
    
    shape_map = build_tensor_shape_map(model)
    initializer_map = build_initializer_map(graph)
    ordered_nodes = topological_sort(graph)
    
    # Memory allocation tracking
    mat_buf = 1 # can be 1 or 2
    bias_vector_buf = 3 # can be 3 or 4
    gemv_buf = 5 # can be 5 or 6
    relu_buf = 7 # can be 7 or 8
    input_buf = 9 # always 9 for input tensor
    tensor_buffer_map = {}
    tensor_size_map = {}  # Track output sizes for RELU length
    asm_instructions = []
    weight_counter = 0
    bias_counter = 0
    skip_nodes = set()  # Store node names instead of objects
    
    # Memory address simulation
    dram_addresses = {
        "inputs":  0x700, # giving space for 223 instructions before the inputs
        "weights": 0x10700, # 128*128=16384 inputs can be saved
        "biases":  0x13000, # ... weights can be saved
        "outputs": 0x20000, # ... biases can be saved
        # 1000 outputs, total 0x100BB8 values in dram
    }
    
    
    # Process each node and generate assembly
    for i, node in enumerate(ordered_nodes):
        node_name = node.name if node.name else f"node_{i}"
        # print(f"Processing node: {node_name} (OpType: {node.op_type})")
        # Skip nodes that have been marked for skipping
        if node_name in skip_nodes:
            continue
            
        # Handle Reshape operations
        if node.op_type == "Reshape":
            # We have only 1 input and 1 output for Reshape
            input_name = node.input[0]
            output_name = node.output[0]
            
            # If input is already in tensor_buffer_map, reuse it
            if input_name in tensor_buffer_map:
                tensor_buffer_map[output_name] = tensor_buffer_map[input_name]
            else: # Create a new buffer for the Reshape output
                # We have only 1 input, and we put it in buffer 0
                tensor_buffer_map[input_name] = 0
                size = tensor_size(shape_map.get(input_name, []))
                asm_instructions.append(f"LOAD_V {input_buf}, {hex(dram_addresses['inputs'])}, {size}")
                # print("The length of the input tensor is", size)
                tensor_buffer_map[output_name] = 0
            continue
        
        # Process node inputs (weights/biases)
        for input_name in node.input:
            # not possible otherwise, but why not check?
            if input_name in initializer_map and input_name not in tensor_buffer_map:
                tensor_data = initializer_map[input_name]
                tensor_type = tensor_data["type"]
                
                if tensor_type == "weight":
                    # Matrix load
                    if len(tensor_data["shape"]) == 2:
                        rows, cols = tensor_data["shape"]
                    else:
                        # Handle higher-dimensional weights: we assume last dimension is columns
                        # and all previous dimensions are rows
                        # e.g., for 3D tensor [batch, rows, cols], we flatten
                        # the first dimensions to get total rows
                        rows = np.prod(tensor_data["shape"][:-1])
                        cols = tensor_data["shape"][-1]
                    size = rows * cols
                    tensor_buffer_map[input_name] = mat_buf
                    asm_instructions.append(f"LOAD_M {mat_buf}, {hex(dram_addresses['weights'] + weight_counter)}, {rows}, {cols}")
                    weight_counter += size
                    # ping-pong the weight buffer between 1 and 2
                    mat_buf = 2 if mat_buf == 1 else 1
                elif tensor_type == "bias":
                    # Vector load
                    size = tensor_size(tensor_data["shape"])
                    tensor_buffer_map[input_name] = bias_vector_buf
                    asm_instructions.append(f"LOAD_V {bias_vector_buf}, {hex(dram_addresses['biases'] + bias_counter)}, {size}")
                    # print(f"Bias: LOAD_V {bias_vector_buf}, {hex(dram_addresses['biases'] + bias_counter)}, {size}")
                    bias_counter += size
                    # ping-pong the bias buffer between 3 and 4
                    bias_vector_buf = 4 if bias_vector_buf == 3 else 3

        # Generate operation instructions
        if node.op_type in ["Gemm", "MatMul"]:
            # Find inputs - assume format: [input, weight, (optional bias)]
            input_buf = tensor_buffer_map.get(node.input[0], "?") # will be ? if not found
            input_buf = 9 if input_buf == 0 else input_buf  # default to 0 if not found
            weight_buf = tensor_buffer_map.get(node.input[1], "?")
            bias_buf = 4 if bias_vector_buf == 3 else 3  # ping-pong bias buffer
            
            # Get matrix dimensions
            if node.input[1] in initializer_map:
                weight_shape = initializer_map[node.input[1]]["shape"]
                if len(weight_shape) == 2:
                    rows, cols = weight_shape
                else:
                    rows = np.prod(weight_shape[:-1])
                    cols = weight_shape[-1]
            else:
                # Fallback to shape map
                shape = shape_map.get(node.input[1], ["?", "?"])
                rows, cols = shape[0], shape[1]
            
            asm_instructions.append(f"GEMV {gemv_buf}, {weight_buf}, {input_buf}, {bias_buf}, {rows}, {cols}")
            
            tensor_buffer_map[node.output[0]] = gemv_buf
            # Track output size for subsequent RELU
            tensor_size_map[node.output[0]] = rows
            gemv_buf = 6 if gemv_buf == 5 else 5  # ping-pong GEMV buffer
        
        elif node.op_type == "Add":
            # Skip if already processed as part of GEMV
            continue
        
        elif node.op_type == "Relu":
            input_buf = tensor_buffer_map.get(node.input[0], "?")
            # Get the length from the input tensor's tracked size
            relu_length = tensor_size_map.get(node.input[0], 0)
            asm_instructions.append(f"RELU {relu_buf}, {input_buf}, {relu_length}")
            tensor_buffer_map[node.output[0]] = relu_buf
            tensor_size_map[node.output[0]] = relu_length  # Pass through the size
            relu_buf = 8 if relu_buf == 7 else 7  # ping-pong ReLU buffer

        # Handle final output storage
        if node.output[0] in [o.name for o in graph.output]:
            size = tensor_size(shape_map.get(node.output[0], []))
            out_buf = tensor_buffer_map.get(node.output[0], "?")
            asm_instructions.append(f"STORE {out_buf}, {hex(dram_addresses['outputs'])}, {size}")

    # Write assembly to file
    with open(output_file, "w") as f:
        f.write("; Custom Architecture Assembly Code\n")
        f.write("; Generated from ONNX model\n\n")
        f.write("\n".join(asm_instructions))
    
    # print(f"Generated assembly saved to {output_file}")

if __name__ == "__main__":
    # Example usage
    model_path = "mlp_model.onnx"
    output_file = "assembly_code.asm"
    generate_assembly(model_path, output_file)
    print(f"Assembly code generated and saved to {output_file}")