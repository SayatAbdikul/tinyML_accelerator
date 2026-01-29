import numpy as np
from accelerator_config import AcceleratorConfig

def quantize_int32_to_int8_rtl_exact(x_int32, max_abs, zero_point=0):
    """
    Bit-exact simulation of RTL quantization.
    Logic matches scale_calculator.sv and quantizer_pipeline.sv.
    """
    if max_abs == 0:
        return np.zeros_like(x_int32, dtype=np.int8)
    
    # 1. Simulate Scale Calculator (scale_calculator.sv)
    # reciprocal_scale = (127 << 24) // max_abs
    max_val = (1 << (AcceleratorConfig.DATA_WIDTH - 1)) - 1
    divider = max_val << 24
    reciprocal_scale = int(divider // max_abs)
    
    print(f"Max abs: {max_abs}")
    print(f"Reciprocal scale: {reciprocal_scale}")
    
    # 2. Simulate Multiplier (quantizer_pipeline.sv)
    # Use int64 to prevent overflow during multiplication
    # RTL: mult_pipe[0] <= $signed(stage1_value) * $signed({1'b0, stage1_scale});
    products = x_int32.astype(np.int64) * reciprocal_scale
    
    print(f"Input: {x_int32[0]}")
    print(f"Product: {products[0]}")

    # 3. Simulate Rounding (quantizer_pipeline.sv)
    # (product + (1 << 23)) >> 24
    rounded = (products + (1 << 23)) >> 24
    
    print(f"Rounded: {rounded[0]}")

    # 4. Clamp to int8
    clipped = np.clip(rounded, -128, 127)
    return clipped.astype(np.int8)

# Test case from the failing run
# RTL Output: -66 (index 0)
# Golden Output: -39 (index 0)
# Input is unknown, but we can reverse engineer or just use the golden model's logic to see where RTL deviates if we assume input is same.
# WAIT! Max Absolute Error is 65.
# Let's check a specific value from the logs if possible.
# In the log:
# DEBUG_GOLDEN_PRE_QUANT: max_abs=239194, scale=8907
# DEBUG_GOLDEN_PRE_QUANT: res[0] = 4952
# DEBUG_GOLDEN_PRE_QUANT: res[1] = 147731

# Let's trace res[0] = 4952
x_val = np.array([4952], dtype=np.int32)
max_abs = 239194
res = quantize_int32_to_int8_rtl_exact(x_val, max_abs)
print(f"Result: {res[0]}")

# Let's trace res[1] = 147731
x_val = np.array([147731], dtype=np.int32)
res = quantize_int32_to_int8_rtl_exact(x_val, max_abs)
print(f"Result: {res[0]}")

