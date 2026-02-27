# Accelerator Configuration - FPGA variant simulation (heavy_test_fpga)
# Matches the parameters in test/heavy_test_fpga/tinyml_accelerator_top.sv exactly.
#
# IMPORTANT: This file must be imported BEFORE any compiler modules (dram.py,
# compile.py, etc.) so that DRAM padding uses TILE_ELEMS=8, matching the RTL.
# test_full_mnist.py inserts the local directory first in sys.path to ensure this.
#
# With TILE_ELEMS=8 the weight layout is:
#   fc1 (12×784): 784 divisible by 8 → no padding; 12 × 784 = 9408 bytes
#   fc2 (32×12):  12 not divisible by 8 → padded to 16; 32 × 16 = 512 bytes
#   fc3 (10×32):  32 divisible by 8 → no padding; 10 × 32 = 320 bytes
#   Total weights: 10240 bytes → max address = 2368 + 10240 = 12608 < 32768 (2^15)

class AcceleratorConfig:
    DATA_WIDTH = 8
    ADDR_WIDTH = 15              # matches tinyml_accelerator_top.sv

    TILE_ELEMS = 8               # Must match TILE_WIDTH / DATA_WIDTH in RTL (64 / 8 = 8)
    TILE_WIDTH = 64              # matches tinyml_accelerator_top.sv

    MEM_SIZE = 32768             # 32KB = 2^15 (matches ADDR_WIDTH=15)

    VECTOR_BUFFER_WIDTH = 8192   # 32 tiles × 256 bits = 1024 elements (≥784 needed)
    MATRIX_BUFFER_WIDTH = 131072 # 512 tiles × 256 bits = 16384 elements (≥9600 for fc1)

    MAX_ROWS = 784               # matches src/fpga_top.sv
    MAX_COLS = 784               # matches src/fpga_top.sv
    OUT_N = 10

    # Memory Map (same as compiler/accelerator_config.py)
    DRAM_ADDR_INPUTS  = 192    # 0x0C0
    DRAM_ADDR_BIASES  = 1216   # 0x4C0
    DRAM_ADDR_OUTPUTS = 2240   # 0x8C0
    DRAM_ADDR_WEIGHTS = 2368   # 0x940
