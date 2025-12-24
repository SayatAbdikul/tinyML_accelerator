# TinyML Accelerator Documentation

Comprehensive documentation for the TinyML Accelerator hardware design.

## Overview

The TinyML Accelerator is a specialized hardware accelerator for neural network inference with quantized 8-bit integer arithmetic. It implements a custom instruction set architecture (ISA) with 5 instructions optimized for deep learning operations.

## Documentation Files

### Architecture Documentation
- **[RTL_ARCHITECTURE.md](RTL_ARCHITECTURE.md)** - Complete RTL architecture documentation
  - System overview
  - Module hierarchy (22 modules)
  - Detailed descriptions
  - Signal flow diagrams
  - Performance characteristics
  - Design patterns

### Visual Diagrams
- **[diagrams/](diagrams/)** - Visual architecture diagrams
  - System architecture
  - Module hierarchy tree
  - Execution unit details
  - GEMV pipeline
  - Memory system
  - FSM state machines

### Test Documentation
- **[../test/TESTBENCH_COMPARISON.md](../test/TESTBENCH_COMPARISON.md)** - Testbench comparison
  - Golden model vs RTL tests
  - Heavy test vs Cocotb tests
  - Test methodology comparison

## Quick Start

### View Architecture
```bash
# Read architecture documentation
open docs/RTL_ARCHITECTURE.md

# Generate and view diagrams
cd docs/diagrams
./generate_diagrams.sh
open system_architecture.png
```

### Module Summary

| Category | Modules | Purpose |
|----------|---------|---------|
| **Top Level** | tinyml_accelerator_top | Main coordinator |
| **Control** | fetch_unit, i_decoder | Instruction fetch & decode |
| **Execution** | modular_execution_unit + 5 sub-modules | Operation execution |
| **Memory** | simple_memory (4 instances) | Data storage |
| **Buffers** | buffer_file (2 instances) | Temporary storage |
| **Computation** | top_gemv, 32× pe | Matrix-vector multiply |
| **Quantization** | quantization, scale_calculator | 32→8 bit conversion |
| **Arithmetic** | wallace_32x32, compressor_3to2 | Fast multiplication |
| **Activation** | relu | ReLU activation |
| **Data Movement** | load_v, load_m, store | DRAM ↔ Buffer |

**Total: 22 unique modules** across ~2500 lines of SystemVerilog

## Key Features

### Instruction Set
1. **LOAD_V** - Load vector from DRAM to buffer
2. **LOAD_M** - Load matrix from DRAM to buffer
3. **GEMV** - General matrix-vector multiplication (with quantization)
4. **RELU** - Apply ReLU activation function
5. **STORE** - Store buffer to DRAM

### Architecture Highlights
- **Tiled Computation**: 32-element tiles for efficient memory bandwidth
- **Parallel Processing**: 32 PEs operating in parallel
- **Pipelined Quantization**: Multi-stage 32-bit → 8-bit conversion
- **Modular Design**: Clean separation of concerns
- **FSM-Based Control**: Hierarchical state machines

### Memory System
- **4 Separate Memories**: Instructions, Load_V, Load_M, Store
- **Memory Map**:
  - 0x000000-0x000700: Instructions (1792 bytes)
  - 0x000700-0x010700: Input vectors (64KB)
  - 0x010700-0x013000: Weight matrices (11KB)
  - 0x013000-0x020000: Bias vectors (52KB)
  - 0x020000-0x030000: Output buffers (64KB)

### Buffer System
- **Vector Buffers**: 32 buffers × 1024 elements (configurable)
- **Matrix Buffers**: 32 buffers × variable size
- **Tile-Based Access**: 32 bytes per tile

## Performance

### Throughput
- **32 MACs/cycle** during GEMV accumulation
- **32 bytes/cycle** memory bandwidth (tile-based)

### Latency (approximate)
- **LOAD_V**: ceil(length / 32) cycles
- **LOAD_M**: ceil(rows × cols / 32) cycles
- **GEMV**: rows × ceil(cols / 32) + quantization_overhead
- **RELU**: length cycles (element-wise)
- **STORE**: ceil(length / 32) cycles

### Resource Utilization
- **32 × 8×8 Multipliers** (PEs)
- **~64KB Buffer Memory** (configurable)
- **4 × 16MB DRAM** (64MB total, configurable)

## Design Hierarchy

```
Level 1: System
  └─ tinyml_accelerator_top

Level 2: Subsystems
  ├─ fetch_unit
  ├─ i_decoder
  └─ modular_execution_unit

Level 3: Execution Modules
  ├─ buffer_controller
  ├─ load_execution
  ├─ gemv_execution
  ├─ relu_execution
  └─ store_execution

Level 4: Computational Units
  ├─ top_gemv
  │  ├─ pe[0:31]
  │  └─ quantization
  ├─ load_v
  ├─ load_m
  ├─ relu
  └─ store

Level 5: Arithmetic Primitives
  ├─ wallace_32x32
  ├─ scale_calculator
  └─ compressor_3to2
```

## Abstraction Levels

### Level 1: Behavioral
```
Input → [ TinyML Accelerator ] → Output
```

### Level 2: Architectural
```
Fetch → Decode → Execute → Writeback
         ↓         ↓         ↓
      [Instruction Decoder]
                   ↓
      [Execution Unit with Buffers]
                   ↓
            [Memory System]
```

### Level 3: Microarchitectural
```
fetch_unit → i_decoder → modular_execution_unit
                              ├─ buffer_controller
                              ├─ load_execution
                              ├─ gemv_execution (32 PEs)
                              ├─ relu_execution
                              └─ store_execution
                                      ↓
                              [4× simple_memory]
```

### Level 4: RTL Implementation
See [RTL_ARCHITECTURE.md](RTL_ARCHITECTURE.md) for detailed RTL structure

## Design Patterns

### 1. Modular Decomposition
Each major function is a separate, reusable module

### 2. Hierarchical FSMs
Multi-level state machines for complex control flow

### 3. Handshake Protocol
Standardized start/done signals for module coordination

### 4. Parameterization
Configurable tile sizes, data widths, buffer counts

### 5. Tiled Processing
Efficient memory bandwidth through tile-based computation

## Verification

### Test Suites
1. **Golden Model** (Python) - Algorithm reference
2. **Cocotb Tests** (20 images) - Basic RTL validation
3. **Heavy Test** (10,000 images) - Production validation

See [../test/TESTBENCH_COMPARISON.md](../test/TESTBENCH_COMPARISON.md) for details.

### Pass Criteria
- **RTL Accuracy**: ≥85% on MNIST
- **Exact Match Rate**: ≥95% outputs match golden model
- **Max Error**: ≤3 quantization units

## File Organization

```
docs/
├── README.md                    # This file
├── RTL_ARCHITECTURE.md          # Complete architecture doc
└── diagrams/                    # Visual diagrams
    ├── README.md
    ├── generate_diagrams.sh
    ├── *.dot                    # DOT source files
    └── *.png                    # Generated diagrams

rtl/
├── tinyml_accelerator_top.sv   # Top level
├── fetch_unit.sv
├── i_decoder.sv
├── buffer_file.sv
├── simple_memory.sv
├── top_gemv.sv
├── pe.sv
├── quantization.sv
├── wallace_32x32.sv
├── compressor_3to2.sv
├── load_v.sv
├── load_m.sv
├── store.sv
├── relu.sv
└── execution_unit/             # Execution modules
    ├── modular_execution_unit.sv
    ├── buffer_controller.sv
    ├── load_execution.sv
    ├── gemv_execution.sv
    ├── relu_execution.sv
    └── store_execution.sv

test/
├── cocotb_tests/               # Basic validation
├── heavy_test/                 # Comprehensive validation
└── TESTBENCH_COMPARISON.md

compiler/
├── golden_model.py             # Python reference
├── compile.py                  # Compiler
└── model.py                    # Neural network model
```

## Getting Started

### 1. Review Architecture
```bash
# Read architecture documentation
cat docs/RTL_ARCHITECTURE.md

# Generate visual diagrams
cd docs/diagrams
./generate_diagrams.sh
```

### 2. Explore RTL
```bash
# View top-level module
cat rtl/tinyml_accelerator_top.sv

# View execution unit
cat rtl/execution_unit/modular_execution_unit.sv

# View GEMV pipeline
cat rtl/top_gemv.sv
```

### 3. Run Tests
```bash
# Quick test (20 images)
cd test/cocotb_tests
make run_test

# Heavy test (10,000 images)
cd test/heavy_test
make quick_test  # First 100 images
make run_test    # Full dataset
```

## Key Insights

### Memory Architecture
⚠️ **Critical**: The design uses **4 separate memory instances** with no hardware interconnect. Testbenches must manually synchronize all instances when updating memory during simulation.

### Quantization Strategy
GEMV automatically quantizes its 32-bit outputs to 8-bit using dynamic scaling:
1. Find max absolute value across all outputs
2. Compute scale = 127 / max_abs
3. Apply scale to each output with saturation

### Tiling Strategy
- **Tile Size**: 32 elements (matches PE count)
- **Benefits**: Reduces memory bandwidth, enables parallelism
- **Trade-off**: Increased control complexity

### Performance Bottlenecks
1. **GEMV Operation**: Dominates execution time
2. **Memory Bandwidth**: 32 bytes/cycle may limit large matrices
3. **Quantization**: Multi-cycle latency for scale computation

## Future Enhancements

Potential improvements:
1. **Pipeline GEMV**: Overlap computation and data loading
2. **Unified Memory**: Single memory with arbiter instead of 4 separate instances
3. **Dynamic Precision**: Support multiple quantization bit-widths
4. **Sparse Matrix Support**: Skip zero multiplications
5. **Multi-Core**: Parallel execution units for batch processing

## References

- [SystemVerilog IEEE 1800-2017](https://ieeexplore.ieee.org/document/8299595)
- [Cocotb Documentation](https://docs.cocotb.org/)
- [Verilator Documentation](https://verilator.org/guide/latest/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

## License

[Project License Information]

## Contributors

[Project Contributors]

## Revision History

| Version | Date | Description |
|---------|------|-------------|
| 1.0 | Dec 24, 2025 | Initial comprehensive documentation |

---

For questions or issues, please refer to the detailed documentation files or contact the development team.
