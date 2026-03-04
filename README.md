# TinyML Accelerator

A hardware accelerator for neural network inference with quantized 8-bit integer arithmetic. Implements a custom 5-instruction ISA optimized for MLP operations. Validated on MNIST digit classification and deployed on Gowin GW2AR-18 FPGA (Tang Nano 20K).

## Key Features

- **8 Parallel Processing Elements (PEs)**: 8x8 signed multipliers operating in parallel
- **Tile-Based Computation**: 8-element tiles with BSRAM-backed storage
- **Dynamic Quantization**: Automatic 32-bit to 8-bit conversion with max-abs scaling
- **Custom ISA**: LOAD_V, LOAD_M, GEMV, RELU, STORE instructions
- **Unified Memory**: Single 32 KB DRAM with UART loading
- **FPGA Validated**: 89 MHz Fmax on Gowin GW2AR-18, 25,470 cycles/image (~3,500 images/sec)
- **Accuracy**: 95% on MNIST (10,000 test images), 100% exact match vs golden model

## Quick Start

### Run Tests
```bash
# FPGA simulation test (primary, 20 images)
cd test/heavy_test_fpga
make run_test NUM_IMAGES=1

# Simulation-only test (10,000 images, uses large register arrays)
cd test/heavy_test
make run_test

# Component-level tests
cd test/cocotb_tests
make TEST_TARGET=golden_comparison run_test
```

### Compile Neural Network Model
```bash
cd compiler
python3 compile.py  # Generates assembly from ONNX model
python3 main.py     # Compiles to machine code (dram.hex)
```

## Project Structure

```
tinyML_accelerator/
├── src/                        # FPGA synthesis source (Gowin EDA)
│   ├── fpga_top.sv            # FPGA top-level (UART + accelerator)
│   ├── tinyml_accelerator_top_fpga.sv
│   ├── top_gemv.sv            # GEMV core (BSRAM-optimized)
│   ├── simple_memory.sv       # Gowin_SP BRAM + UART loader
│   ├── Gowin_SDPB_32.sv       # BSRAM wrapper (block RAM)
│   ├── uart_rx.sv, uart_tx.sv # UART interface
│   └── ...                    # All execution modules
├── rtl/                        # Simulation RTL
│   ├── tinyml_accelerator_top.sv
│   ├── fpga_modules/          # FPGA-adapted modules (simulation mocks)
│   │   ├── gemv_unit_core.sv  # GEMV core with BSRAM
│   │   ├── Gowin_SDPB_32.sv   # BSRAM simulation mock
│   │   ├── Gowin_RAM16SDP_Mock.sv
│   │   └── ...
│   └── execution_unit/        # Original simulation execution modules
├── compiler/                   # Python toolchain
│   ├── compile.py             # ONNX to assembly compiler
│   ├── assembler.py           # Assembly to machine code
│   ├── dram.py                # DRAM hex generator
│   ├── golden_model.py        # Python reference implementation
│   ├── accelerator_config.py  # Tile size / address configuration
│   └── dram.hex               # Generated memory image
├── test/
│   ├── heavy_test_fpga/       # Primary FPGA simulation test (cocotb)
│   ├── heavy_test/            # Full 10K image validation (simulation RTL)
│   ├── cocotb_tests/          # Component-level cocotb tests
│   └── new_unit_tests/        # Verilator C++ unit tests
├── memory_tools/               # UART communication tools
└── docs/                       # Architecture documentation
```

## Two RTL Trees

| Tree | Purpose | Top Module | Memory |
|------|---------|------------|--------|
| `src/` | FPGA synthesis (Gowin EDA) | `fpga_top.sv` | Gowin_SP BRAM |
| `rtl/` + `rtl/fpga_modules/` | Simulation (Verilator/cocotb) | `tinyml_accelerator_top.sv` | Register array |

`rtl/fpga_modules/` mirrors `src/` with simulation-compatible mocks for Gowin IP blocks. After modifying `rtl/fpga_modules/gemv_unit_core.sv`, sync to `src/top_gemv.sv`.

## Architecture Overview

### Instruction Set

| Opcode | Instruction | Description |
|--------|-------------|-------------|
| 0x01 | LOAD_V | Load vector from DRAM to buffer |
| 0x02 | LOAD_M | Load matrix from DRAM to buffer |
| 0x04 | GEMV | Matrix-vector multiply + quantization |
| 0x05 | RELU | Apply ReLU activation |
| 0x03 | STORE | Write buffer to DRAM |

### Module Hierarchy

```
fpga_top.sv (FPGA) / tinyml_accelerator_top.sv (Sim)
├── fetch_unit.sv              — Instruction fetch from DRAM
├── i_decoder.sv               — Decode 5-instruction ISA
├── simple_memory.sv           — Unified 32 KB DRAM
└── modular_execution_unit.sv
    ├── buffer_controller.sv   — Vector/matrix buffer management
    │   └── buffer_file.sv     — Tile-indexed buffer storage
    ├── load_execution.sv      — LOAD_V / LOAD_M orchestration
    │   ├── load_v.sv          — Vector loading from DRAM
    │   └── load_m.sv          — Matrix loading from DRAM
    ├── gemv_execution.sv      — GEMV tile streaming
    │   └── gemv_unit_core.sv  — Core GEMV FSM
    │       ├── pe.sv (x8)     — 8-bit signed multiply
    │       ├── Gowin_SDPB_32  — x-vector BSRAM (packed 4:1)
    │       ├── Gowin_SDPB_32  — Accumulator BSRAM
    │       ├── scale_calculator.sv
    │       └── quantizer_pipeline.sv
    ├── relu_execution.sv      — ReLU activation
    └── store_execution.sv     — Buffer to DRAM write-back
```

### Key Specifications
- **Data Width**: 8-bit signed integers (int8)
- **Tile Size**: 8 elements (TILE_ELEMS=8, TILE_WIDTH=64 bits)
- **PEs**: 8 parallel multiply-accumulate units
- **Memory**: Unified 32 KB DRAM (FPGA: Gowin_SP BRAM + UART loader)
- **Accumulator**: 32-bit (BSRAM-backed, Gowin_SDPB_32)
- **Throughput**: 8 MACs/cycle during GEMV

### DRAM Memory Map

| Region | Address | Size | Contents |
|--------|---------|------|----------|
| Instructions | 0x000 | ~192 B | Program code |
| Inputs | 0x0C0 | ~784 B | Input vector (per image) |
| Biases | 0x4C0 | ~54 B | Layer biases |
| Outputs | 0x8C0 | ~10 B | Inference results |
| Weights | 0x940 | ~10 KB | Weight matrices |

## Performance

### FPGA Synthesis (Gowin GW2AR-18)
- **Fmax**: 89.201 MHz (11 logic levels)
- **Cycles/image**: 25,470
- **Latency**: 0.286 ms/image (~3,500 images/sec)
- **Logic**: 42% (8,640 / 20,736)
- **BSRAM**: 94% (43 / 46)
- **DSP**: 5 blocks (MULT36X36 + 4x MULTADDALU18X18)

### Validation Results
- **MNIST Test Accuracy**: 95% (10,000 images)
- **Exact Match Rate**: 100% outputs match golden model
- **Max Quantization Error**: 0

### Optimizations Applied
| Optimization | Cycles | Fmax | Effect |
|-------------|--------|------|--------|
| Baseline (BSRAM accum) | 43,996 | 67 MHz | Initial FPGA design |
| FIND_MAX/CLEAR fix | 34,942 | -- | -20.6% cycles |
| A1-A4 (pipeline stages) | 36,221 | 84 MHz | +25% Fmax |
| A6 (x_mem BSRAM) | 37,501 | 91 MHz | +36% Fmax |
| B1 (pack x_mem 4:1) | 29,301 | 86 MHz | -21.9% cycles |
| B2 (prefetch x tile) | 25,470 | 89 MHz | -13.1% cycles |

## FPGA Deployment

The design runs on Gowin Tang Nano 20K (GW2AR-LV18QN88C8/I7):

1. **Synthesize** with Gowin EDA using files from `src/`
2. **Load DRAM** via UART: `memory_tools/uart_load_hex compiler/dram.hex`
3. **Run inference**: Press S1 button, read results via `memory_tools/uart_read_max`

### UART Workflow
```bash
# Load compiled model to FPGA memory
cd memory_tools && ./uart_load_hex ../compiler/dram.hex /dev/ttyUSB1

# Read inference results
./uart_read_max /dev/ttyUSB1
```

## Design Highlights

### GEMV Pipeline (gemv_unit_core.sv)
The GEMV core uses a multi-stage pipelined FSM:
1. **Load x-vector** into packed BSRAM (4 int8 per 32-bit word)
2. **Load biases** into accumulator BSRAM
3. **Per-tile loop**: WAIT_TILE -> WAIT_PE -> SUM_PARTIAL -> READ_ACCUM -> PREP_ACCUM -> ACCUMULATE -> WAIT_NEXT
4. **B2 Prefetch**: Next x-tile loaded during accumulate pipeline (zero overhead)
5. **Quantization**: FIND_MAX -> COMPUTE_SCALE -> QUANTIZE -> OUTPUT_Y

### Quantization Pipeline
1. **Calibration**: Find max absolute value across all accumulator rows
2. **Scale Calculation**: Compute `reciprocal = 2^23 / max_abs` (iterative division)
3. **Quantization**: 32x32 Wallace tree multiply -> round -> saturate to int8

### Tiling Strategy
- **8-element tiles** match PE count
- x-vector stored in BSRAM with 4:1 packing (B1 optimization)
- Next tile prefetched during accumulate pipeline (B2 optimization)

## Documentation

- **[docs/RTL_ARCHITECTURE.md](docs/RTL_ARCHITECTURE.md)** - Detailed RTL architecture
- **[docs/README.md](docs/README.md)** - Documentation index
- **[test/TEST_GUIDE.md](test/TEST_GUIDE.md)** - Test suite overview
- **[test/heavy_test/README.md](test/heavy_test/README.md)** - Validation guide
- **[rtl/execution_unit/README.md](rtl/execution_unit/README.md)** - Execution unit architecture
- **[memory_tools/README.md](memory_tools/README.md)** - UART tools

---

**Status**: Validated in simulation (Verilator + cocotb) and synthesized for Gowin GW2AR-18 FPGA at 89 MHz.
