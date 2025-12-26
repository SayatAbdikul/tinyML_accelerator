# TinyML Accelerator

A hardware accelerator for neural network inference with quantized 8-bit integer arithmetic. Implements a custom 5-instruction ISA optimized for Multi Layer Perceptron(MLP) operations. The design was tested on the claassification model on MNIST dataset. 

## Key Features

- **32 Parallel Processing Elements (PEs)**: 8×8 multipliers operating in parallel
- **Tile-Based Computation**: 32-element tiles for efficient memory bandwidth
- **Dynamic Quantization**: Automatic 32-bit → 8-bit conversion with scaling
- **Custom ISA**: LOAD_V, LOAD_M, GEMV, RELU, STORE instructions
- **Modular Architecture**: 22 SystemVerilog modules, ~3,300 lines of code
- **Validated**: 90%+ accuracy on MNIST (10,000 test images)

## Quick Start

### Run Tests
```bash
# Quick validation (100 images)
cd test/heavy_test
make quick_test

# Full validation (10,000 images
make run_test

# Component-level tests (fast)
cd ../cocotb_tests
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
├── rtl/                    # SystemVerilog RTL (22 modules)
│   ├── tinyml_accelerator_top.sv
│   ├── execution_unit/     # Execution modules
│   ├── top_gemv.sv        # Matrix-vector multiply
│   └── ...
├── compiler/               # Python toolchain
│   ├── compile.py         # ONNX → assembly compiler
│   ├── golden_model.py    # Reference implementation
│   └── dram.hex           # Memory
├── test/                  # Verification in C++ and Python using Verilator and CocoTB
│   ├── heavy_test/        # Full 10K image validation
│   └── cocotb_tests/      # Component-level tests
└── docs/                   # Architecture documentation
    ├── RTL_ARCHITECTURE.md
    └── diagrams/
```

## Documentation

### Architecture
- **[docs/RTL_ARCHITECTURE.md](docs/RTL_ARCHITECTURE.md)** - Complete RTL documentation (all 22 modules, signal flows, FSMs)
- **[docs/diagrams/](docs/diagrams/)** - Visual architecture diagrams (system, GEMV pipeline, memory map, FSMs)
- **[docs/README.md](docs/README.md)** - Documentation index and quick reference

### Testing
- **[test/TEST_GUIDE.md](test/TEST_GUIDE.md)** - Test suite overview and workflow
- **[test/heavy_test/README.md](test/heavy_test/README.md)** - Comprehensive validation guide
- **[test/TESTBENCH_COMPARISON.md](test/TESTBENCH_COMPARISON.md)** - Golden model vs RTL comparison

### Module Details
- **[rtl/execution_unit/README.md](rtl/execution_unit/README.md)** - Execution unit architecture

## Architecture Overview

### Instruction Set

| Opcode | Instruction | Description |
|--------|-------------|-------------|
| 0x01 | LOAD_V | Load vector from DRAM to buffer |
| 0x02 | LOAD_M | Load matrix from DRAM to buffer |
| 0x04 | GEMV | Matrix-vector multiply + quantization |
| 0x05 | RELU | Apply ReLU activation |
| 0x03 | STORE | Write buffer to DRAM |

### System Flow
```
┌─────────┐    ┌─────────┐    ┌──────────────────┐
│  Fetch  │ -> │ Decode  │ -> │ Execute (GEMV)   │
│  Unit   │    │         │    │ 32 PEs + Quant   │
└─────────┘    └─────────┘    └──────────────────┘
                                      |
                                      v
                              ┌──────────────┐
                              │ Memory (4×)  │
                              │ Buffers      │
                              └──────────────┘
```

### Key Specifications
- **Data Width**: 8-bit signed integers
- **Max Matrix**: 1024×1024 elements
- **Buffer Memory**: 16 vector buffers (8 KB each) + 2 matrix buffers (64 KB each)
- **Memory**: 4 separate 16 MB DRAM instances (simulated; needs external memory for FPGA)
- **Throughput**: 32 MACs/cycle during GEMV

## Performance

### Validation Results
- **MNIST Test Accuracy**: >90% (10,000 images)
- **Exact Match Rate**: >95% outputs match golden model
- **Max Quantization Error**: ≤3 units

### Latency (approximate cycles)
- **LOAD_V**: `⌈length / 32⌉`
- **GEMV**: `rows × ⌈cols / 32⌉ + quantization_overhead`
- **RELU**: `length` (element-wise)


## FPGA Deployment

⚠️ **Current design requires modifications for FPGA**:

1. **Memory Interface**: Replace 4× `simple_memory` (64 MB) with external DDR controller + arbiter
2. **Synthesis Directives**: Add `(* ram_style = "block" *)` for large arrays, `(* use_dsp = "yes" *)` for multipliers
3. **Clock Constraints**: Define timing constraints for target FPGA

See architecture docs for detailed FPGA adaptation guide.


## Design Highlights

### Modular Execution Unit
- **Hierarchical FSMs**: Top-level → Execution → Operation-specific state machines
- **Buffer Controller**: Unified vector/matrix buffer management
- **Separated Concerns**: Load, GEMV, ReLU, Store as independent modules

### Quantization Pipeline
1. **Calibration**: Find max absolute value
2. **Scale Calculation**: Compute `scale = 127 / max_abs` (32-cycle division)
3. **Quantization**: 32×32 Wallace tree multiplier → round → saturate to 8-bit

### Tiling Strategy
- **32-element tiles** match PE count
- Reduces memory bandwidth (32 bytes/cycle)
- Enables pipelined matrix-row streaming


---

**Status**: Validated in simulation (Verilator). FPGA port requires external memory controller integration.
