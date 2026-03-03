# Accelerator Configuration Module

class AcceleratorConfig:
    DATA_WIDTH = 8
    ADDR_WIDTH = 16

    # TILE_WIDTH=64 matches the FPGA RTL (src/tinyml_accelerator_top_fpga.sv).
    # TILE_ELEMS = TILE_WIDTH / DATA_WIDTH = 64 / 8 = 8.
    # This controls how weight rows are padded in the DRAM image:
    #   fc1 (12×784): 784 divisible by 8 → no padding; 12×784 = 9408 bytes
    #   fc2 (32×12):  12 not divisible by 8 → padded to 16; 32×16 = 512 bytes
    #   fc3 (10×32):  32 divisible by 8 → no padding; 10×32 = 320 bytes
    #   Total weights: 10240 bytes → max address = 2368 + 10240 = 12608 < 65536
    TILE_ELEMS = 8
    TILE_WIDTH = 64

    MEM_SIZE = 32768  # 32KB

    VECTOR_BUFFER_WIDTH = 8192
    MATRIX_BUFFER_WIDTH = 131072

    MAX_ROWS = 1024
    MAX_COLS = 1024
    OUT_N = 10

    # Memory Map
    DRAM_ADDR_INPUTS = 192
    DRAM_ADDR_BIASES = 1216
    DRAM_ADDR_OUTPUTS = 2240
    DRAM_ADDR_WEIGHTS = 2368
