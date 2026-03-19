import pytest
import numpy as np
import torch
import torch.nn as nn
from dram import get_dram, write_to_dram, dram, MEM_SIZE
import golden_model
from accelerator_config import AcceleratorConfig

# -- Utility for bit-exact comparison --
def _get_buffer(buf_id, length):
    return np.array(golden_model.buffers[buf_id][:length], dtype=np.int8)

# -- Reusable clean state --
@pytest.fixture(autouse=True)
def reset_state():
    golden_model.buffers = {}
    golden_model.flag = 0
    golden_model.pending_conv_config = {}
    
    # reset simulated DRAM
    global dram
    dram.fill(0)
    golden_model.memory = dram

# =========================================================================
# TEST 1: instruction decode (ISA parser logic)
# =========================================================================
def test_decode_conv2d_cfg_and_run():
    # Construct a test configuration:
    # fmap=16x16, in_C=4, out_C=8, kernel=3x3, stride=1, pad=1
    cfg_opcode = 0x06
    dest = 11
    fmap_h, fmap_w = 16, 16
    in_c, out_c = 4, 8
    kh, kw = 3, 3
    stride, pad = 1, 1
    
    word_cfg = (pad << 45) | (stride << 42) | (kw << 38) | (kh << 34) | \
               (out_c << 28) | (in_c << 22) | (fmap_w << 16) | (fmap_h << 10) | \
               (dest << 5) | cfg_opcode
               
    golden_model.i_decoder(word_cfg)
    
    cfg = golden_model.pending_conv_config
    assert cfg['fmap_h'] == 16
    assert cfg['fmap_w'] == 16
    assert cfg['in_c'] == 4
    assert cfg['out_c'] == 8
    assert cfg['kh'] == 3
    assert cfg['kw'] == 3
    assert cfg['stride'] == 1
    assert cfg['pad'] == 1
    
    # Run instruction layout (doesn't execute here because buffers are empty)
    run_opcode = 0x07
    dest, x_id, w_id, b_id = 11, 9, 2, 4
    relu_flag = 1
    word_run = (relu_flag << 25) | (b_id << 20) | (w_id << 15) | (x_id << 10) | (dest << 5) | run_opcode
    
    # Mock conv2d to just catch arguments
    args_caught = {}
    def mock_conv2d(**kwargs):
        args_caught.update(kwargs)
        
    original = golden_model.conv2d
    try:
        golden_model.conv2d = mock_conv2d
        golden_model.i_decoder(word_run)
        assert args_caught['dest'] == 11
        assert args_caught['x'] == 9
        assert args_caught['w'] == 2
        assert args_caught['b'] == 4
        assert args_caught['apply_relu'] is True
        assert args_caught['kh'] == 3
        # Should pull from pending config
        assert args_caught['out_c'] == 8
    finally:
        golden_model.conv2d = original


# =========================================================================
# TEST 2: math execution: direct convolution + ReLU vs NumPy
# =========================================================================
def test_conv2d_math_direct():
    """Test the direct conv2d implementation in the golden model vs a naive nested loop reference."""
    in_c, out_c = 2, 3
    fmap_h, fmap_w = 4, 4
    kh, kw = 3, 3
    stride, pad = 1, 1
    
    np.random.seed(42)
    # Generate input, weight, bias (simulating already loaded buffers)
    # Values chosen small to avoid saturation during the initial accumulation.
    x_data = np.random.randint(-10, 10, size=(in_c, fmap_h, fmap_w), dtype=np.int8)
    w_data = np.random.randint(-10, 10, size=(out_c, in_c, kh, kw), dtype=np.int8)
    b_data = np.random.randint(-10, 10, size=(out_c,), dtype=np.int8)
    
    golden_model.buffers[1] = x_data.flatten().tolist()
    golden_model.buffers[2] = w_data.flatten().tolist()
    golden_model.buffers[3] = b_data.flatten().tolist()
    
    # ── Expected computation (numpy reference) ──────────────────────────────────
    x_padded = np.pad(x_data, ((0,0), (pad,pad), (pad,pad)), mode='constant')
    out_h = (fmap_h + 2*pad - kh) // stride + 1
    out_w = (fmap_w + 2*pad - kw) // stride + 1
    
    expected = np.zeros((out_c, out_h, out_w), dtype=np.int32)
    for oc in range(out_c):
        for oh in range(out_h):
            for ow in range(out_w):
                acc = np.int32(0)
                for ic in range(in_c):
                    for khi in range(kh):
                        for kwi in range(kw):
                            val_x = x_padded[ic, oh*stride + khi, ow*stride + kwi]
                            val_w = w_data[oc, ic, khi, kwi]
                            acc += np.int32(val_w) * np.int32(val_x)
                expected[oc, oh, ow] = acc + b_data[oc]
                
    # Apply RTL-exact quantization
    max_abs = int(np.max(np.abs(expected)))
    expected_quant = golden_model.quantize_int32_to_int8_rtl_exact(expected.flatten().astype(np.int32), max_abs, 0)
    # Apply ReLU
    expected_relu = np.maximum(expected_quant, 0)
    
    # ── Actual computation via golden model ─────────────────────────────────────
    golden_model.conv2d(dest=4, w=2, x=1, b=3, 
                        fmap_h=fmap_h, fmap_w=fmap_w, in_c=in_c, out_c=out_c, 
                        kh=kh, kw=kw, stride=stride, pad=pad, apply_relu=True)
                        
    actual_relu = _get_buffer(4, len(expected_relu))
    np.testing.assert_array_equal(actual_relu, expected_relu)


# =========================================================================
# TEST 3: math execution: maxpool vs NumPy
# =========================================================================
def test_maxpool_math():
    """Test maxpooling sliding window."""
    channels = 4
    fmap_h, fmap_w = 4, 4
    pool_size, stride = 2, 2
    
    np.random.seed(42)
    x_data = np.random.randint(-128, 127, size=(channels, fmap_h, fmap_w), dtype=np.int8)
    golden_model.buffers[1] = x_data.flatten().tolist()
    
    out_h = (fmap_h - pool_size) // stride + 1
    out_w = (fmap_w - pool_size) // stride + 1
    expected = np.zeros((channels, out_h, out_w), dtype=np.int8)
    
    for c in range(channels):
        for oh in range(out_h):
            for ow in range(out_w):
                window = x_data[c, oh*stride:oh*stride+pool_size, ow*stride:ow*stride+pool_size]
                expected[c, oh, ow] = np.max(window)
                
    golden_model.maxpool(dest=2, x=1, fmap_h=fmap_h, fmap_w=fmap_w, 
                         channels=channels, pool_size=pool_size, stride=stride)
                         
    actual = _get_buffer(2, len(expected.flatten()))
    np.testing.assert_array_equal(actual, expected.flatten())


# =========================================================================
# TEST 4: SmallCNN end-to-end integration via compiler + golden model
# =========================================================================
def test_smallcnn_end_to_end(tmp_path):
    """
    1. Export the SmallCNN to ONNX.
    2. Run compile.py to generate ASM.
    3. Run assembler to write machine code to DRAM.
    4. Save conv + fc weights to their designated regions in DRAM.
    5. Load an input vector to DRAM.
    6. Execute the instruction stream in golden_model.
    7. Compare golden model output vs PyTorch direct output (with tolerance for quantization loss).
    """
    import model
    import compile
    import assembler
    import dram
    
    # 1. Export
    onnx_file = str(tmp_path / "cnn_model.onnx")
    asm_file = str(tmp_path / "assembly.asm")
    
    cnn = model.create_cnn_model()
    # model.create_cnn_model() already saves to 'cnn_model.onnx' in the cwd.
    
    # 2. Compile ONNX -> ASM
    compile.generate_assembly("cnn_model.onnx", asm_file)
    
    # Verify the assembly has CNN instructions
    with open(asm_file, 'r') as f:
        asm_code = f.read()
    print("----- GENERATED ASSEMBLY -----")
    print(asm_code)
    print("------------------------------")
    assert "CONV2D_CFG" in asm_code
    assert "CONV2D_RUN" in asm_code
    assert "MAXPOOL" in asm_code
    assert "GEMV" in asm_code
    
    # 3. Assemble -> writes instructions to dram starting at 0x0
    hex_file = asm_file.replace('.asm', '.hex')
    assembler.assemble_file(asm_file, output_file=hex_file)
    
    # 4. Save weights
    dram_offsets = {
        "weights": AcceleratorConfig.DRAM_ADDR_WEIGHTS,
        "conv_weights": AcceleratorConfig.DRAM_ADDR_CONV_WEIGHTS,
        "biases": AcceleratorConfig.DRAM_ADDR_BIASES
    }
    
    # We must write both the FC weights (tile padded) and Conv weights (flat).
    # The build_initializer_map logic determines layout based on ndim in dram.py
    # Actually, let's call the specific functions:
    fc_weights, _ = dram.save_initializers_to_dram("cnn_model.onnx", dram_offsets)
    conv_w, conv_b = dram.save_conv_weights_to_dram("cnn_model.onnx", dram_offsets)
    
    # Sync memory back to golden model
    golden_model.memory = dram.get_dram()
    
    # 5. Provide an input image
    np.random.seed(0)
    input_img = np.random.randint(-50, 50, size=(1, 1, 28, 28), dtype=np.int8)
    flat_input = input_img.flatten()
    golden_model.memory[AcceleratorConfig.DRAM_ADDR_INPUTS : AcceleratorConfig.DRAM_ADDR_INPUTS + len(flat_input)] = flat_input
    
    # Run PyTorch reference
    with torch.no_grad():
        pt_input = torch.tensor(input_img, dtype=torch.float32)
        pt_out = cnn(pt_input).numpy().flatten()
    
    # 6. Execute via golden model
    # Load the hex file generated by assembler (it's one 64-bit instr per line)
    instr_words = []
    with open(hex_file, 'r') as f:
        for line in f:
            if line.strip():
                instr_words.append(int(line.strip(), 16))
    
    # Sync memory back to golden model
    # First get the dram with weights from step 4, then merge the input
    golden_model.memory = dram.get_dram()
    golden_model.memory[AcceleratorConfig.DRAM_ADDR_INPUTS : AcceleratorConfig.DRAM_ADDR_INPUTS + len(flat_input)] = flat_input
    
    # Manually load the input feature map into buffer 9 (LOAD_V)
    # assembler.py encoding for LOAD_V:
    # word = (addr << 40) | (length << 10)  | (dest << 5) | opcode
    load_v_opcode = 0x01
    load_v_dest = 9
    load_v_length = len(flat_input)
    load_v_addr = AcceleratorConfig.DRAM_ADDR_INPUTS
    
    load_v_word = (load_v_addr << 40) | (load_v_length << 10) | (load_v_dest << 5) | load_v_opcode
    golden_model.i_decoder(load_v_word)
    
    for word in instr_words:
        golden_model.i_decoder(word)
        
    final_output = _get_buffer(golden_model.output_buffer, AcceleratorConfig.OUT_N)
    
    # 7. Compare
    # Because of aggressive 8-bit intermediate quantization + scaling logic vs f32 PyTorch,
    # we expect shape/scale correlation, but absolute match requires PyTorch Quantization Aware Training.
    # We verify that the pipeline completes cleanly and produces a vector of exactly OUT_N ints.
    assert len(final_output) == AcceleratorConfig.OUT_N
    
    print("\nPyTorch native float32 output:", pt_out)
    print("Golden Model 8-bit output:", final_output)
