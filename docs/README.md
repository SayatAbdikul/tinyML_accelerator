# TinyML Accelerator Documentation

Comprehensive documentation for the TinyML Accelerator hardware design.

## Overview

The TinyML Accelerator is a specialized hardware accelerator for neural network inference with quantized 8-bit integer arithmetic. It implements a custom 5-instruction ISA optimized for MLP operations. The design targets Gowin GW2AR-18 FPGA (Tang Nano 20K) and is validated on MNIST digit classification.

## Documentation Files

### Architecture Documentation
- **[RTL_ARCHITECTURE.md](RTL_ARCHITECTURE.md)** - Detailed RTL architecture documentation
  - Module hierarchy and descriptions
  - FSM state diagrams
  - Signal flow diagrams

### Visual Diagrams
- **[diagrams/](diagrams/)** - Visual architecture diagrams
  - System architecture
  - Module hierarchy tree
  - GEMV pipeline
  - Memory system

### Test Documentation
- **[../test/TEST_GUIDE.md](../test/TEST_GUIDE.md)** - Test suite overview and workflow
- **[../test/TESTBENCH_COMPARISON.md](../test/TESTBENCH_COMPARISON.md)** - Testbench comparison
- **[../test/heavy_test/README.md](../test/heavy_test/README.md)** - Full validation guide

## Module Summary

| Category | Modules | Purpose |
|----------|---------|---------|
| **Top Level** | `fpga_top` / `tinyml_accelerator_top` | System coordinator |
| **Control** | `fetch_unit`, `i_decoder` | Instruction fetch and decode |
| **Execution** | `modular_execution_unit` + 5 sub-modules | Operation dispatch |
| **Memory** | `simple_memory` (1 unified instance) | 32 KB DRAM (Gowin_SP BRAM on FPGA) |
| **Buffers** | `buffer_file` (vector + matrix instances) | Tile-indexed temporary storage |
| **Computation** | `gemv_unit_core`, 8x `pe` | Tiled matrix-vector multiply |
| **Quantization** | `quantizer_pipeline`, `scale_calculator` | INT32 to INT8 conversion |
| **Arithmetic** | `wallace_32x32`, `compressor_3to2` | 32-bit multiplication |
| **Activation** | `relu` | ReLU activation |
| **Data Movement** | `load_v`, `load_m`, `store` | DRAM to/from buffer transfers |

## Design Hierarchy

```
Level 1: System
  └─ fpga_top (FPGA) / tinyml_accelerator_top (Sim)

Level 2: Subsystems
  ├─ fetch_unit (+ fetch_unit_fpga for FPGA)
  ├─ i_decoder
  ├─ simple_memory (unified, 32 KB)
  └─ modular_execution_unit

Level 3: Execution Modules
  ├─ buffer_controller
  │   └─ buffer_file (vector + matrix)
  ├─ load_execution
  │   ├─ load_v
  │   └─ load_m
  ├─ gemv_execution
  │   └─ gemv_unit_core
  │       ├─ pe[0:7] (8 processing elements)
  │       ├─ Gowin_SDPB_32 (x-vector BSRAM, packed 4:1)
  │       ├─ Gowin_SDPB_32 (accumulator BSRAM)
  │       ├─ scale_calculator
  │       │   └─ wallace_32x32
  │       │       └─ compressor_3to2
  │       └─ quantizer_pipeline
  ├─ relu_execution
  │   └─ relu
  └─ store_execution
      └─ store

Level 4: IP Blocks (FPGA)
  ├─ Gowin_SP (BRAM for unified memory)
  ├─ Gowin_SDPB_32 (dual-port BSRAM for x_mem and res_mem)
  └─ DSP blocks (MULT36X36, MULTADDALU18X18)
```

## Two RTL Trees

The project maintains two parallel RTL trees:

| Tree | Purpose | Top Module | Key Differences |
|------|---------|------------|-----------------|
| `src/` | FPGA synthesis (Gowin EDA) | `fpga_top.sv` | Gowin IP primitives, UART I/O |
| `rtl/` + `rtl/fpga_modules/` | Simulation (Verilator/cocotb) | `tinyml_accelerator_top.sv` | Mock IP, `$readmemh` memory |

**`src/` is the synthesis source of truth.** After modifying simulation files in `rtl/fpga_modules/`, sync to `src/` (e.g., `cp rtl/fpga_modules/gemv_unit_core.sv src/top_gemv.sv`).

### Simulation Mocks
- `Gowin_RAM16SDP_Mock.sv` — Simulates Gowin LUTRAM (async read)
- `Gowin_SDPB_32.sv` (in rtl/fpga_modules/) — Simulates Gowin BSRAM (1-cycle synchronous read)

## DRAM Memory Map

The design uses a unified 32 KB memory with the following layout:

| Region | Address | Size | Contents |
|--------|---------|------|----------|
| Instructions | 0x000 | ~192 B | Program code (fetched by fetch_unit) |
| Inputs | 0x0C0 | ~784 B | Input vector (loaded per image) |
| Biases | 0x4C0 | ~54 B | Layer biases (fc1, fc2, fc3) |
| Outputs | 0x8C0 | ~10 B | Inference results |
| Weights | 0x940 | ~10 KB | Weight matrices (fc1: 12x784, fc2: 32x12, fc3: 10x32) |

Addresses are configured in `compiler/accelerator_config.py`. The compiler generates `dram.hex` which is loaded via UART on FPGA or `$readmemh` in simulation.

## Key Configuration Parameters

| Parameter | FPGA Value | Simulation Value | Notes |
|-----------|-----------|-----------------|-------|
| TILE_ELEMS | 8 | 8 (fpga_modules) / 32 (execution_unit) | Elements per tile |
| TILE_WIDTH | 64 | 64 / 256 | Bits per tile |
| ADDR_WIDTH | 16 | 16 / 24 | Memory address bits |
| MAX_ROWS | 784 | 784 / 1024 | Max vector/matrix dimension |
| GEMV_TILE_SIZE | 8 | 8 | Must equal TILE_ELEMS |

## GEMV Pipeline (gemv_unit_core.sv)

The GEMV core is the most complex module, implementing a multi-stage pipelined FSM:

### Phase 1: Data Loading
1. **LOAD_X** / **STORE_X** — Receive x-vector tiles, pack 4 int8 per 32-bit BSRAM word (B1)
2. **LOAD_BIAS** / **STORE_BIAS** — Receive bias tiles, write to accumulator BSRAM

### Phase 2: Weight Processing (per weight tile)
```
WAIT_TILE -> WAIT_PE -> SUM_PARTIAL -> READ_ACCUM -> PREP_ACCUM -> ACCUMULATE -> WAIT_NEXT
     |                                                                              |
     +<-------- (B2: next x-tile prefetched during PREP_ACCUM/ACCUMULATE) <--------+
```

### Phase 3: Post-Processing
1. **FIND_MAX** — Scan accumulator for max absolute value
2. **COMPUTE_SCALE** — Calculate reciprocal scale (iterative division)
3. **QUANTIZE** — Apply scale, round, saturate to int8
4. **OUTPUT_Y** — Stream quantized results back as tiles

### Key Optimizations
- **B1**: x-vector BSRAM packed 4:1 (2 reads per tile instead of 8)
- **B2**: Next x-tile prefetched during accumulate pipeline (zero-overhead tile transitions)
- **A2**: Pipelined adder tree (SUM_PARTIAL stage halves logic depth)
- **A3**: Registered accumulator write-back (breaks res_dout+sum carry chain)
- **A6**: x_mem moved from LUTRAM to BSRAM (eliminates 6-level MUX cascade)

## Performance

### FPGA Synthesis (Gowin GW2AR-18)
- **Fmax**: 89.201 MHz
- **Cycles/image**: 25,470
- **Latency**: 0.286 ms/image (~3,500 images/sec)
- **Logic**: 42% utilization
- **BSRAM**: 94% (43/46)

### Critical Path
The critical path is in `buffer_controller -> vector_buffer_inst` (opcode decode -> buffer tile index), not in the GEMV core.

### Validation
- **MNIST Accuracy**: 95% (10,000 images)
- **Exact Match**: 100% vs golden model
- **Max Error**: 0

## Compiler Toolchain

```
model.py          — Define MLP architecture (784->12->32->10)
     |
compile.py        — ONNX model -> assembly instructions
     |
assembler.py      — Assembly -> binary machine code
     |
dram.py           — Pack instructions + weights + biases into dram.hex
     |
golden_model.py   — Python reference for RTL verification
```

Configuration: `compiler/accelerator_config.py` defines TILE_ELEMS, DRAM addresses, and memory layout.

## FPGA Workflow

```bash
# 1. Compile model
cd compiler && python3 main.py

# 2. Synthesize with Gowin EDA (src/ directory)
# 3. Program FPGA bitstream

# 4. Load DRAM via UART
cd memory_tools && ./uart_load_hex ../compiler/dram.hex /dev/ttyUSB1

# 5. Run inference (S1 button) and read results
./uart_read_max /dev/ttyUSB1
```

## Test Suites

| Test | Location | Framework | Images | Purpose |
|------|----------|-----------|--------|---------|
| FPGA Simulation | `test/heavy_test_fpga/` | cocotb + Verilator | 20 | **Primary** — tests FPGA modules |
| Full Validation | `test/heavy_test/` | cocotb + Verilator | 10,000 | Production validation (sim RTL) |
| Component Tests | `test/cocotb_tests/` | cocotb | varies | Per-module validation |
| Unit Tests | `test/new_unit_tests/` | Verilator C++ | N/A | Low-level module tests |

## Future Enhancements

- **B3**: Stream weights directly from DRAM during GEMV (bypass buffer controller)
- **B5**: Wider tiles (TILE_SIZE=16) for 2x throughput with minimal BSRAM increase
- **Buffer controller pipelining**: Register the opcode decode path to improve Fmax

---

| Version | Date | Description |
|---------|------|-------------|
| 2.0 | Mar 2026 | Updated for FPGA deployment, unified memory, B1/B2 optimizations |
| 1.0 | Dec 2025 | Initial documentation |
