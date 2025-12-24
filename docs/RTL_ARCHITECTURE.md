# TinyML Accelerator RTL Architecture Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Module Hierarchy](#module-hierarchy)
3. [Top-Level Architecture](#top-level-architecture)
4. [Execution Unit Architecture](#execution-unit-architecture)
5. [Memory Subsystem](#memory-subsystem)
6. [Computational Units](#computational-units)
7. [Module Descriptions](#module-descriptions)
8. [Signal Flow Diagrams](#signal-flow-diagrams)

---

## System Overview

The TinyML Accelerator is a hardware accelerator for neural network inference with the following key features:

- **Instruction Set Architecture**: 5 instructions (LOAD_V, LOAD_M, GEMV, RELU, STORE)
- **Data Path**: 8-bit quantized integer arithmetic
- **Memory**: Shared DRAM with 4 separate memory instances
- **Computation**: Tiled GEMV with 32 PEs, pipelined quantization
- **Control**: FSM-based execution with modular design

---

## Module Hierarchy

```
tinyml_accelerator_top (Top Level)
├── fetch_unit (Instruction Fetch)
│   └── simple_memory (Instruction Memory)
│
├── i_decoder (Instruction Decoder)
│
└── modular_execution_unit (Execution Engine)
    ├── buffer_controller (Buffer Management)
    │   ├── buffer_file (Vector Buffers)
    │   └── buffer_file (Matrix Buffers)
    │
    ├── load_execution (Memory Load Operations)
    │   ├── load_v (Vector Load)
    │   │   └── simple_memory (Data Memory)
    │   └── load_m (Matrix Load)
    │       └── simple_memory (Data Memory)
    │
    ├── gemv_execution (GEMV Orchestration)
    │   └── top_gemv (GEMV Computation)
    │       ├── pe[0:31] (Processing Elements)
    │       ├── quantization (Quantization Unit)
    │       │   ├── quantizer_pipeline (Pipeline Stage)
    │       │   └── scale_calculator (Scale Computation)
    │       │       └── wallace_32x32 (32-bit Multiplier)
    │       │           └── compressor_3to2 (Wallace Tree)
    │       └── (Accumulation & Control Logic)
    │
    ├── relu_execution (ReLU Activation)
    │   └── relu (ReLU Module)
    │
    └── store_execution (Memory Store Operations)
        └── store (Store Module)
            └── simple_memory (Data Memory)
```

---

## Top-Level Architecture

### Block Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                     tinyml_accelerator_top                          │
│                                                                      │
│  ┌────────────┐      ┌────────────┐      ┌──────────────────────┐ │
│  │            │      │            │      │                      │ │
│  │  Fetch     │─────▶│ Instruction│─────▶│  Modular Execution  │ │
│  │  Unit      │      │  Decoder   │      │       Unit          │ │
│  │            │      │            │      │                      │ │
│  └────────────┘      └────────────┘      └──────────────────────┘ │
│       │                    │                       │               │
│       │                    │                       │               │
│       ▼                    ▼                       ▼               │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │              Shared Memory System (4 instances)             │  │
│  │  - Fetch Memory  - Load_V Memory                            │  │
│  │  - Load_M Memory - Store Memory                             │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  Control Signals:                                                   │
│    clk, rst, start ──────────────────────────────────▶             │
│    done ◀────────────────────────────────────────────              │
│                                                                      │
│  Output:                                                             │
│    y[0:9] ◀─────────────────────────────────────────               │
└─────────────────────────────────────────────────────────────────────┘
```

### FSM State Diagram

```
        ┌─────────┐
        │  IDLE   │◀─────────────────────────────────────┐
        └────┬────┘                                      │
             │ start=1                                   │
             ▼                                           │
        ┌─────────┐                                      │
        │  FETCH  │                                      │
        └────┬────┘                                      │
             │ fetch_en=1                                │
             ▼                                           │
     ┌──────────────┐                                    │
     │ WAIT_FETCH   │                                    │
     └──────┬───────┘                                    │
            │ fetch_done=1                               │
            ▼                                            │
     ┌──────────────┐                                    │
     │   DECODE     │                                    │
     └──────┬───────┘                                    │
            │ (latch instruction)                        │
            ▼                                            │
  ┌──────────────────┐                                   │
  │ EXECUTE_START    │                                   │
  └────────┬─────────┘                                   │
           │ exec_start=1                                │
           ▼                                             │
  ┌──────────────────┐                                   │
  │ EXECUTE_WAIT     │                                   │
  └────────┬─────────┘                                   │
           │ exec_done=1                                 │
           ▼                                             │
        ┌──────┐                                         │
        │ DONE │─────────────────────────────────────────┘
        └──────┘ (done pulse, check for zero instr)
```

---

## Execution Unit Architecture

### Modular Execution Unit Block Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│                  modular_execution_unit                            │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                 Main FSM Controller                          │ │
│  │  IDLE → DISPATCH → WAIT_* → COMPLETE                         │ │
│  └───────────────┬──────────────────────────────────────────────┘ │
│                  │                                                 │
│  ┌───────────────┴──────────────────────────────────────────────┐ │
│  │                                                               │ │
│  │  ┌─────────────────┐    ┌──────────────────┐                │ │
│  │  │ Buffer          │    │ Load Execution   │                │ │
│  │  │ Controller      │◀───│   - load_v       │                │ │
│  │  │   - Vec Buffers │    │   - load_m       │                │ │
│  │  │   - Mat Buffers │    └──────────────────┘                │ │
│  │  └───────┬─────────┘                                         │ │
│  │          │                                                    │ │
│  │          │         ┌──────────────────┐                      │ │
│  │          ├────────▶│ GEMV Execution   │                      │ │
│  │          │         │   - top_gemv     │                      │ │
│  │          │         │   - 32 PEs       │                      │ │
│  │          │         │   - Quantization │                      │ │
│  │          │         └──────────────────┘                      │ │
│  │          │                                                    │ │
│  │          │         ┌──────────────────┐                      │ │
│  │          ├────────▶│ ReLU Execution   │                      │ │
│  │          │         │   - relu module  │                      │ │
│  │          │         └──────────────────┘                      │ │
│  │          │                                                    │ │
│  │          │         ┌──────────────────┐                      │ │
│  │          └────────▶│ Store Execution  │                      │ │
│  │                    │   - store module │                      │ │
│  │                    └──────────────────┘                      │ │
│  │                                                               │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Execution Unit FSM

```
    ┌──────┐
    │ IDLE │◀──────────────────────────────────────────┐
    └───┬──┘                                           │
        │ start=1                                      │
        ▼                                              │
  ┌──────────┐                                         │
  │ DISPATCH │──┐                                      │
  └──────────┘  │                                      │
                │                                      │
     ┌──────────┼─────────────┬───────────────┐       │
     │          │             │               │       │
     │ LOAD_V   │ LOAD_M      │ GEMV          │ RELU │ STORE
     │          │             │               │       │
     ▼          ▼             ▼               ▼       ▼
┌─────────┐┌─────────┐ ┌──────────┐  ┌──────────┐┌──────────┐
│WAIT_LOAD││WAIT_LOAD│ │WAIT_GEMV │  │WAIT_RELU ││WAIT_STORE│
└────┬────┘└────┬────┘ └─────┬────┘  └─────┬────┘└─────┬────┘
     │          │            │             │            │
     └──────────┴────────────┴─────────────┴────────────┘
                            │
                            ▼
                     ┌───────────┐
                     │ COMPLETE  │───────────────────────┘
                     └───────────┘ done=1
```

---

## Memory Subsystem

### Memory Architecture

The accelerator uses **4 separate simple_memory instances**:

```
┌──────────────────────────────────────────────────────────────┐
│                  Memory System                               │
│                                                               │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ Fetch Memory     │  │ Load_V Memory    │                │
│  │ (Instructions)   │  │ (Vector Data)    │                │
│  │ 0x000000-0x000700│  │ 0x000700-0x020000│                │
│  └──────────────────┘  └──────────────────┘                │
│                                                               │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ Load_M Memory    │  │ Store Memory     │                │
│  │ (Matrix Data)    │  │ (Output Data)    │                │
│  │ 0x010700-0x020000│  │ 0x020000-0x030000│                │
│  └──────────────────┘  └──────────────────┘                │
│                                                               │
│  All initialized from: dram.hex                              │
└──────────────────────────────────────────────────────────────┘

Memory Map:
  0x000000 - 0x000700: Instructions (1792 bytes)
  0x000700 - 0x010700: Input vectors (64KB)
  0x010700 - 0x013000: Weight matrices (11KB)
  0x013000 - 0x020000: Bias vectors (52KB)
  0x020000 - 0x030000: Output buffers (64KB)
```

### Buffer System

```
┌──────────────────────────────────────────────────────────────┐
│                  buffer_controller                           │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Vector Buffer File                                    │ │
│  │  - 32 buffers x 1024 elements (configurable)          │ │
│  │  - Tile-based access (32 elements per tile)           │ │
│  │  - Separate read/write pointers per buffer            │ │
│  │  - Read: element-wise output                          │ │
│  │  - Write: tile-based input                            │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Matrix Buffer File                                    │ │
│  │  - 32 buffers x configurable size                     │ │
│  │  - Tile-based access (256 bits = 32 bytes per tile)   │ │
│  │  - Used for weight matrices                           │ │
│  │  - Row-major storage                                  │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

---

## Computational Units

### GEMV Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                      top_gemv Module                            │
│                                                                  │
│  Input Buffers:                                                 │
│    w_tile_row[32] ──┐                                          │
│    x[1024]          │                                          │
│    bias[1024]       │                                          │
│                     │                                          │
│  ┌──────────────────▼───────────────────────────────────────┐ │
│  │  Processing Element Array (32 PEs)                       │ │
│  │                                                           │ │
│  │   PE[0]: w[0] × x[col+0] → pe_out[0]                    │ │
│  │   PE[1]: w[1] × x[col+1] → pe_out[1]                    │ │
│  │   ...                                                     │ │
│  │   PE[31]: w[31] × x[col+31] → pe_out[31]                │ │
│  │                                                           │ │
│  └───────────────────┬───────────────────────────────────────┘ │
│                      │                                          │
│  ┌──────────────────▼───────────────────────────────────────┐ │
│  │  Accumulation Logic                                      │ │
│  │  - Sum PE outputs (tree adder)                           │ │
│  │  - Accumulate across tiles                               │ │
│  │  - Add bias after all tiles processed                    │ │
│  │  Result: res[row] = Σ(w[row,*] × x[*]) + bias[row]      │ │
│  │  Format: 32-bit signed integer                           │ │
│  └───────────────────┬───────────────────────────────────────┘ │
│                      │                                          │
│  ┌──────────────────▼───────────────────────────────────────┐ │
│  │  Quantization Pipeline                                   │ │
│  │                                                           │ │
│  │  Stage 1: Find Max Absolute Value                        │ │
│  │    max_abs = max(|res[i]|) for all rows                 │ │
│  │                                                           │ │
│  │  Stage 2: Compute Scale                                  │ │
│  │    scale = max_abs / 127                                 │ │
│  │    (uses scale_calculator with wallace_32x32)           │ │
│  │                                                           │ │
│  │  Stage 3: Quantize Each Element                          │ │
│  │    y[i] = saturate(round(res[i] / scale))               │ │
│  │    Range: [-128, 127] (int8)                            │ │
│  │                                                           │ │
│  └───────────────────┬───────────────────────────────────────┘ │
│                      │                                          │
│                      ▼                                          │
│                 Output: y[1024]                                 │
│                 (8-bit quantized)                               │
└─────────────────────────────────────────────────────────────────┘
```

### Processing Element (PE)

```
┌──────────────────────┐
│         PE           │
│                      │
│  w (8-bit)  ─────┐   │
│                  │   │
│  x (8-bit)  ─────┼──▶│  Multiplier
│                  │   │  (8×8=16 bit)
│                  │   │
│                  └──▶│  y = w × x
│                      │  (16-bit output)
│                      │
└──────────────────────┘

32 PEs operate in parallel per tile
```

### Quantization Unit

```
┌──────────────────────────────────────────────────────────────┐
│                    quantization Module                       │
│                                                               │
│  FSM: IDLE → CALIB → READY → (process data)                │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Calibration Phase                                     │ │
│  │  Input: max_abs (32-bit unsigned)                     │ │
│  │  Compute: scale = (127 << 24) / max_abs              │ │
│  │  Format: Q8.24 fixed-point                            │ │
│  └────────────────┬───────────────────────────────────────┘ │
│                   │                                          │
│  ┌────────────────▼───────────────────────────────────────┐ │
│  │  scale_calculator                                      │ │
│  │  - Computes reciprocal using division                 │ │
│  │  - Uses wallace_32x32 for multiplication              │ │
│  │  - Pipelined for throughput                           │ │
│  └────────────────┬───────────────────────────────────────┘ │
│                   │                                          │
│  ┌────────────────▼───────────────────────────────────────┐ │
│  │  Quantization Phase                                    │ │
│  │  For each data_in (32-bit signed):                    │ │
│  │    1. Multiply by scale                               │ │
│  │    2. Shift right 24 bits                             │ │
│  │    3. Saturate to [-128, 127]                         │ │
│  │    Output: data_out (8-bit signed)                    │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

### Wallace Tree Multiplier

```
┌──────────────────────────────────────────────────────────────┐
│                  wallace_32x32 Multiplier                    │
│                                                               │
│  Input: a[31:0], b[31:0] (32-bit operands)                  │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Partial Product Generation                            │ │
│  │  - 32×32 = 1024 partial products                       │ │
│  │  - Organized in rows (Booth encoding possible)        │ │
│  └────────────────┬───────────────────────────────────────┘ │
│                   │                                          │
│  ┌────────────────▼───────────────────────────────────────┐ │
│  │  Wallace Tree Reduction                                │ │
│  │  - Multiple layers of 3:2 compressors                 │ │
│  │  - Reduces partial products to 2 operands             │ │
│  │  - Irregular tree structure for efficiency            │ │
│  └────────────────┬───────────────────────────────────────┘ │
│                   │                                          │
│  ┌────────────────▼───────────────────────────────────────┐ │
│  │  compressor_3to2 (used multiple times)                │ │
│  │  - Full Adder: sum = a ⊕ b ⊕ cin                      │ │
│  │  - Carry out: cout = (a&b) | (b&cin) | (a&cin)       │ │
│  └────────────────┬───────────────────────────────────────┘ │
│                   │                                          │
│  ┌────────────────▼───────────────────────────────────────┐ │
│  │  Final Carry-Propagate Adder                          │ │
│  │  - Adds final two operands                            │ │
│  │  - Produces 64-bit result                             │ │
│  └────────────────────────────────────────────────────────┘ │
│                   │                                          │
│                   ▼                                          │
│            Output: product[63:0]                             │
└──────────────────────────────────────────────────────────────┘
```

---

## Module Descriptions

### Core Modules

| Module | Lines | Purpose | Key Features |
|--------|-------|---------|--------------|
| **tinyml_accelerator_top** | 220 | Top-level coordinator | FSM control, instruction flow |
| **fetch_unit** | 97 | Instruction fetch | Sequential instruction reading |
| **i_decoder** | 46 | Instruction decoder | Combinational decode logic |
| **modular_execution_unit** | 482 | Execution coordinator | Dispatch, orchestration |

### Memory Modules

| Module | Lines | Purpose | Key Features |
|--------|-------|---------|--------------|
| **simple_memory** | ~50 | Memory array | $readmemh, byte-addressable |
| **buffer_file** | 113 | Multi-buffer storage | Tile-based I/O, per-buffer indices |

### Execution Modules

| Module | Lines | Purpose | Key Features |
|--------|-------|---------|--------------|
| **buffer_controller** | ~200 | Buffer management | Dual buffer files, multiplexing |
| **load_execution** | ~200 | Load orchestration | LOAD_V and LOAD_M coordination |
| **load_v** | ~150 | Vector loading | Tile-based DRAM→Buffer |
| **load_m** | ~150 | Matrix loading | Tile-based DRAM→Buffer |
| **gemv_execution** | ~200 | GEMV orchestration | top_gemv coordination |
| **top_gemv** | 366 | GEMV computation | PE array, accumulation, quantization |
| **relu_execution** | ~150 | ReLU orchestration | Activation coordination |
| **relu** | ~100 | ReLU computation | Element-wise max(0, x) |
| **store_execution** | ~150 | Store orchestration | Buffer→DRAM writes |
| **store** | ~150 | Memory write | Tile-based output storage |

### Computational Modules

| Module | Lines | Purpose | Key Features |
|--------|-------|---------|--------------|
| **pe** | 25 | Processing element | 8×8→16 bit multiply |
| **quantization** | 107 | Quantization unit | Scale computation, saturation |
| **quantizer_pipeline** | ~100 | Pipeline stage | Pipelined quantization |
| **scale_calculator** | ~150 | Scale computation | Division, reciprocal |
| **wallace_32x32** | ~400 | 32-bit multiplier | Wallace tree implementation |
| **compressor_3to2** | 15 | Full adder | 3:2 compression |

---

## Signal Flow Diagrams

### Instruction Execution Flow

```
Time ──────────────────────────────────────────────────────────▶

Cycle:  0    1    2    3    4    5    ...    N    N+1
        │    │    │    │    │    │           │    │
Top:   IDLE FETCH WAIT DECODE EXEC EXEC ... EXEC DONE
        │    │    │    │    │    │           │    │
Fetch:      ─────┬────                            
                 │                                  
         pc=0 ──┴──▶ instr ─────────▶              
                                                    
Decoder:                  ────┬────                
                              │                    
                     fields ──┴─────▶              
                                                    
Exec:                              ───────...────┬─
                                                 │
                                        result ──┴─▶
```

### GEMV Tile Processing

```
For rows × cols matrix-vector multiply with TILE_SIZE = 32:

Time: t0        t1        t2        t3        ...  tN
      │         │         │         │              │
Row 0:│ Tile 0  │ Tile 1  │ Tile 2  │ ... │Bias+Q │→ y[0]
      │ (cols   │ (cols   │ (cols   │     │       │
      │  0-31)  │ 32-63)  │ 64-95)  │     │       │
      │         │         │         │              │
Row 1:│         │ Tile 0  │ Tile 1  │ ... │Bias+Q │→ y[1]
      │         │         │         │              │
...   │         │         │         │              │
      │         │         │         │              │
Row N:│         │         │         │ ... │Bias+Q │→ y[N]

Each tile:
  1. Load w_tile (32 weights)
  2. Multiply w_tile × x[col:col+31] (32 PEs)
  3. Accumulate to res[row]
  4. Repeat for next tile

After all tiles for all rows:
  1. Find max_abs across all res[]
  2. Compute scale
  3. Quantize each res[row] → y[row]
```

### Memory Access Pattern

```
LOAD_V Instruction:
  DRAM[addr] ────┬────────────────────────────┐
                 │ (tile by tile)             │
                 ├─▶ tile[0:31]               │
                 ├─▶ tile[32:63]              │
                 └─▶ ...                      │
                                              │
  Buffer[dest] ◀────────────────────────────────
               (accumulates tiles)

LOAD_M Instruction:
  DRAM[addr] ────┬────────────────────────────┐
                 │ (rows × cols, tile-based)  │
                 ├─▶ row0_tile[0:31]          │
                 ├─▶ row0_tile[32:63]         │
                 ├─▶ row1_tile[0:31]          │
                 └─▶ ...                      │
                                              │
  Buffer[dest] ◀────────────────────────────────
               (row-major storage)

STORE Instruction:
  Buffer[src] ────┬────────────────────────────┐
                  │ (tile by tile)             │
                  ├─▶ tile[0:31]               │
                  └─▶ ...                      │
                                               │
  DRAM[addr] ◀──────────────────────────────────
```

---

## Abstraction Levels

### Level 1: System Level
```
        Input ──▶ [ TinyML Accelerator ] ──▶ Output
                          │
                    Instructions
                    (LOAD/GEMV/RELU/STORE)
```

### Level 2: Subsystem Level
```
[ Fetch & Decode ] ──▶ [ Execution Unit ] ──▶ [ Memory ]
                              │
                      ┌───────┴────────┐
                      │                │
                   [Buffers]      [Compute]
```

### Level 3: Module Level
```
Fetch + Decoder + Execution(Load + GEMV + ReLU + Store)
                            │
                    ┌───────┴────────┐
                    │                │
            [Buffer Control]    [top_gemv]
                                     │
                            ┌────────┴─────────┐
                            │                  │
                        [32 PEs]        [Quantization]
```

### Level 4: Gate Level
```
PE: Multiplier (Booth/Wallace)
Quantization: Divider, Multiplier, Saturator
Memory: SRAM arrays
Buffers: Register files
```

---

## Critical Design Features

### 1. Memory Synchronization
- **4 separate memory instances** must be synchronized
- All initialized from same `dram.hex` at time 0
- Testbench must manually sync updates during runtime
- No hardware interconnect between memories

### 2. Buffer Management
- **Dual buffer files**: Vector (element access) and Matrix (tile access)
- Each buffer has independent read/write pointers
- Tile-based access for efficient memory bandwidth
- Reset capability for reusing buffers

### 3. Tiled Computation
- GEMV uses 32-element tiles (matching 32 PEs)
- Reduces memory bandwidth requirements
- Enables pipelining and parallel processing
- Handles variable matrix sizes

### 4. Quantization Pipeline
- **Two-phase**: Calibration then quantization
- Calibration: Find max, compute scale
- Quantization: Apply scale, saturate
- Fixed-point arithmetic (Q8.24 format)

### 5. FSM-Based Control
- **Top level**: 7-state FSM for instruction execution
- **Execution unit**: 7-state FSM for operation dispatch
- **Sub-modules**: Individual FSMs for each operation
- Handshake signals (start/done) for coordination

---

## Design Patterns

### 1. Modular Decomposition
Each major function is a separate module with well-defined interface

### 2. Hierarchical FSMs
Multi-level state machines for complex control flow

### 3. Pipelining
Quantization and multiplication use pipelined stages

### 4. Parameterization
Configurable tile sizes, data widths, buffer counts

### 5. Handshake Protocol
start/done signals for module coordination

---

## Performance Characteristics

### Latency (approximate cycles)
- **LOAD_V**: `ceil(length / TILE_SIZE)` cycles
- **LOAD_M**: `ceil(rows * cols / TILE_SIZE)` cycles  
- **GEMV**: `rows * ceil(cols / TILE_SIZE) + quantization_latency`
- **RELU**: `length` cycles (element-wise)
- **STORE**: `ceil(length / TILE_SIZE)` cycles

### Throughput
- **PE Array**: 32 MACs per cycle (during accumulation)
- **Memory**: 32 bytes per cycle (tile-based)
- **Overall**: Limited by slowest operation (typically GEMV)

### Resource Utilization
- **32 PEs**: 32 × (8×8 multipliers)
- **Buffers**: Configurable, typically 32 buffers × 1024 elements
- **Memory**: 4 instances × 16MB = 64MB total (configurable)

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Dec 24, 2025 | Initial comprehensive documentation |

