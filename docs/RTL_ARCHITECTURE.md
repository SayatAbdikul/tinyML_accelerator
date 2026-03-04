# TinyML Accelerator RTL Architecture Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Module Hierarchy](#module-hierarchy)
3. [Top-Level Architecture](#top-level-architecture)
4. [Execution Unit Architecture](#execution-unit-architecture)
5. [Memory Subsystem](#memory-subsystem)
6. [GEMV Core Pipeline](#gemv-core-pipeline)
7. [Quantization Pipeline](#quantization-pipeline)
8. [Module Descriptions](#module-descriptions)
9. [Signal Flow Diagrams](#signal-flow-diagrams)
10. [Optimizations](#optimizations)

---

## System Overview

The TinyML Accelerator is a hardware accelerator for neural network inference targeting the Gowin GW2AR-18 FPGA (Tang Nano 20K):

- **ISA**: 5 instructions (LOAD_V, LOAD_M, GEMV, RELU, STORE)
- **Data Path**: 8-bit signed integer (int8) arithmetic
- **Memory**: Unified 32 KB DRAM (single Gowin_SP BRAM instance)
- **Computation**: Tiled GEMV with 8 PEs, BSRAM-backed accumulator and x-vector
- **Tiles**: 8 elements (64 bits) per tile
- **Control**: Hierarchical FSM-based execution
- **Performance**: 89 MHz Fmax, 25,470 cycles/inference (0.286 ms, ~3,500 inferences/sec) on 10KB classification model.

---

## Module Hierarchy

```
fpga_top.sv (FPGA) / tinyml_accelerator_top.sv (Sim)
├── fetch_unit.sv                  — Instruction fetch from unified DRAM
│   └── (fetch_unit_fpga.sv on FPGA — adds FETCH_PRIME for BRAM latency)
│
├── i_decoder.sv                   — Decode 5-instruction ISA
│
├── simple_memory.sv               — Unified 32 KB DRAM
│   └── (FPGA: Gowin_SP BRAM + UART loader)
│   └── (Sim: register array + $readmemh)
│
└── modular_execution_unit.sv
    ├── buffer_controller.sv       — Vector/matrix buffer management
    │   ├── buffer_file.sv         — Vector buffers (tile-indexed)
    │   └── buffer_file.sv         — Matrix buffers (tile-indexed)
    │
    ├── load_execution.sv          — LOAD_V / LOAD_M orchestration
    │   ├── load_v.sv              — Vector loading (DRAM → buffer)
    │   └── load_m.sv              — Matrix loading (DRAM → buffer, row-aware)
    │
    ├── gemv_execution.sv          — GEMV tile bridging
    │   └── gemv_unit_core.sv      — Core GEMV FSM
    │       ├── pe.sv (×8)         — 8-bit signed multiply (1-cycle latency)
    │       ├── Gowin_SDPB_32      — x-vector BSRAM (packed 4:1, B1)
    │       ├── Gowin_SDPB_32      — Accumulator BSRAM (32-bit)
    │       ├── scale_calculator.sv
    │       │   └── wallace_32x32.sv
    │       │       └── compressor_3to2.sv
    │       └── quantizer_pipeline.sv
    │
    ├── relu_execution.sv          — ReLU activation (tile-streamed)
    │   └── relu.sv                — Element-wise max(0, x)
    │
    └── store_execution.sv         — Buffer → DRAM write-back
        └── store.sv               — Tile-based memory write
```

---

## Top-Level Architecture

### Block Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│              tinyml_accelerator_top (Sim) / fpga_top (FPGA)      │
│                                                                    │
│  ┌────────────┐     ┌────────────┐     ┌──────────────────────┐ │
│  │  Fetch     │────▶│ Instruction│────▶│  Modular Execution   │ │
│  │  Unit      │     │  Decoder   │     │       Unit           │ │
│  └─────┬──────┘     └────────────┘     └──────────┬───────────┘ │
│        │                                           │             │
│        │         ┌─────────────────────┐           │             │
│        └────────▶│  Unified Memory     │◀──────────┘             │
│                  │  (32 KB DRAM)       │                         │
│                  │  Gowin_SP BRAM      │                         │
│                  └─────────────────────┘                         │
│                                                                    │
│  Controls: clk, rst, start ──▶         ◀── done                 │
│  Output:   y[0:9] (inference result)                             │
└──────────────────────────────────────────────────────────────────┘
```

### Top-Level FSM

```
        ┌─────────┐
        │  IDLE   │◀──────────────────────────────────┐
        └────┬────┘                                   │
             │ start=1                                │
             ▼                                        │
        ┌─────────┐                                   │
        │  FETCH  │                                   │
        └────┬────┘                                   │
             │ fetch_en=1                             │
             ▼                                        │
     ┌──────────────┐                                 │
     │ WAIT_FETCH   │                                 │
     └──────┬───────┘                                 │
            │ fetch_done=1                            │
            ▼                                         │
     ┌──────────────┐                                 │
     │   DECODE     │                                 │
     └──────┬───────┘                                 │
            │ (latch opcode, operands, address)       │
            ▼                                         │
  ┌──────────────────┐                                │
  │ EXECUTE_START    │                                │
  └────────┬─────────┘                                │
           │ exec_start=1                             │
           ▼                                          │
  ┌──────────────────┐                                │
  │ EXECUTE_WAIT     │                                │
  └────────┬─────────┘                                │
           │ exec_done=1                              │
           ▼                                          │
        ┌──────┐                                      │
        │ DONE │──────────────────────────────────────┘
        └──────┘ (done pulse; loops if instr != 0)
```

The top module fetches instructions sequentially from DRAM, decodes them, and dispatches to the execution unit. When the zero instruction is fetched (end of program), the accelerator halts.

### Instruction Format (64-bit)

```
[63:59]  opcode     — 5-bit operation code
[58:54]  dest       — destination buffer ID
[53:44]  rows       — row count (GEMV, LOAD_M)
[43:34]  length/cols — vector length or column count
[33:29]  b_id       — bias buffer ID (GEMV)
[28:24]  x_id       — x-vector buffer ID (GEMV)
[23:19]  w_id       — weight buffer ID (GEMV)
[18:3]   addr       — DRAM address (LOAD/STORE)
[2:0]    reserved
```

---

## Execution Unit Architecture

### Block Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                  modular_execution_unit                          │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                 Main FSM Controller                        │ │
│  │  IDLE → DISPATCH → WAIT_* → COMPLETE                       │ │
│  └─────────────────────┬──────────────────────────────────────┘ │
│                        │                                         │
│  ┌─────────────────────┴──────────────────────────────────────┐ │
│  │                                                             │ │
│  │  ┌─────────────────┐    ┌──────────────────┐              │ │
│  │  │ Buffer          │    │ Load Execution   │              │ │
│  │  │ Controller      │◀───│   - load_v       │              │ │
│  │  │  • Vec Buffers  │    │   - load_m       │              │ │
│  │  │  • Mat Buffers  │    └──────────────────┘              │ │
│  │  └───────┬─────────┘                                       │ │
│  │          │                                                  │ │
│  │          │         ┌──────────────────┐                    │ │
│  │          ├────────▶│ GEMV Execution   │                    │ │
│  │          │         │  → gemv_unit_core│                    │ │
│  │          │         │    • 8 PEs       │                    │ │
│  │          │         │    • x_mem BSRAM │                    │ │
│  │          │         │    • res_mem BSRAM│                   │ │
│  │          │         │    • Quantization│                    │ │
│  │          │         └──────────────────┘                    │ │
│  │          │                                                  │ │
│  │          │         ┌──────────────────┐                    │ │
│  │          ├────────▶│ ReLU Execution   │                    │ │
│  │          │         └──────────────────┘                    │ │
│  │          │                                                  │ │
│  │          │         ┌──────────────────┐                    │ │
│  │          └────────▶│ Store Execution  │                    │ │
│  │                    └──────────────────┘                    │ │
│  │                                                             │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│           ┌───────────────────────┐                              │
│           │ Unified Memory Port   │                              │
│           │ (arbitrated by FSM)   │                              │
│           └───────────────────────┘                              │
└──────────────────────────────────────────────────────────────────┘
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
                │ (based on opcode)                    │
     ┌──────────┼─────────────┬───────────────┐       │
     │          │             │               │       │
     ▼          ▼             ▼               ▼       │
 WAIT_LOAD  WAIT_GEMV    WAIT_RELU     WAIT_STORE    │
     │          │             │               │       │
     └──────────┴─────────────┴───────────────┘       │
                        │                              │
                        ▼                              │
                  ┌───────────┐                        │
                  │ COMPLETE  │────────────────────────┘
                  └───────────┘ done=1
```

The execution unit uses a single unified memory port. During LOAD operations, load_v or load_m drive the memory interface. During STORE, the store module drives it. GEMV and RELU operate on buffer data only (no direct DRAM access).

---

## Memory Subsystem

### Unified DRAM (32 KB)

The accelerator uses a **single unified memory** instance:

```
┌──────────────────────────────────────────────────────┐
│              Unified Memory (32 KB)                   │
│              simple_memory.sv                         │
│                                                        │
│  FPGA: Gowin_SP BRAM + UART loader                   │
│  Sim:  Register array + $readmemh(dram.hex)          │
│                                                        │
│  Interface:                                           │
│    mem_addr [ADDR_WIDTH-1:0]  — byte address          │
│    mem_rdata [7:0]            — read data (1 byte)    │
│    mem_wdata [7:0]            — write data (1 byte)   │
│    mem_we                     — write enable           │
│    mem_valid                  — read data valid        │
│                                                        │
│  FPGA: 1-cycle synchronous read (Gowin_SP, OCE=0)    │
│  Sim:  Combinational read (register array)            │
└──────────────────────────────────────────────────────┘
```

### DRAM Memory Map

| Region | Address | Size | Contents |
|--------|---------|------|----------|
| Instructions | 0x000 | ~192 B | Program code (fetched by fetch_unit) |
| Inputs | 0x0C0 | ~784 B | Input vector (loaded per image) |
| Biases | 0x4C0 | ~54 B | Layer biases (fc1, fc2, fc3) |
| Outputs | 0x8C0 | ~10 B | Inference results |
| Weights | 0x940 | ~10 KB | Weight matrices (fc1: 12x784, fc2: 32x12, fc3: 10x32) |

Addresses are configured in `compiler/accelerator_config.py`.

### Buffer System

```
┌──────────────────────────────────────────────────────────────┐
│                    buffer_controller.sv                       │
│                                                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Vector Buffer File (buffer_file.sv)                  │   │
│  │  - 16 buffers (configurable via VECTOR_BUFFER_COUNT) │   │
│  │  - Tile-based access (8 elements = 64 bits per tile) │   │
│  │  - Separate read/write tile pointers per buffer       │   │
│  │  - Stored as flat register array, indexed by tile     │   │
│  │  - Used for: x-vectors, biases, y-results            │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Matrix Buffer File (buffer_file.sv)                  │   │
│  │  - 2 buffers (weight storage)                         │   │
│  │  - Tile-based access (64 bits per tile)               │   │
│  │  - Row-major weight storage                           │   │
│  │  - Cleared between GEMV invocations (clr_cache)      │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                                │
│  Arbitration: buffer_controller routes read/write requests   │
│  from load, gemv, relu, and store execution modules          │
└──────────────────────────────────────────────────────────────┘
```

---

## GEMV Core Pipeline

The GEMV core (`gemv_unit_core.sv`) is the most complex module, implementing a multi-stage pipelined FSM with BSRAM-backed storage.

### Internal Storage

| Resource | Implementation | Size | Purpose |
|----------|---------------|------|---------|
| **x_mem** | Gowin_SDPB_32 (BSRAM) | 1024×32 | x-vector storage, packed 4:1 (B1) |
| **res_mem** | Gowin_SDPB_32 (BSRAM) | 1024×32 | 32-bit accumulator per output row |

Both use synchronous (registered) reads with 1-cycle latency. READ_xxx states in the FSM prime the address one cycle before the consuming state.

### Phase 1: Data Loading

```
IDLE ──▶ LOAD_X ──▶ STORE_X ──▶ ... ──▶ LOAD_BIAS ──▶ STORE_BIAS ──▶ ...
```

1. **LOAD_X / STORE_X** — Receive x-vector tiles from gemv_execution, pack into x_mem BSRAM.
   - B1: 4 int8 elements per 32-bit word → 2 writes per tile (was 8)
   - x_mem depth: 784/4 = 196 words (fits in 1024×32 BSRAM)
2. **LOAD_BIAS / STORE_BIAS** — Receive bias tiles, sign-extend int8→int32, write to res_mem as initial accumulator values.

### Phase 2: Weight Processing (per weight tile)

```
     ┌──────────────────────────────────────────────────────────────┐
     │                                                              │
     ▼                                                              │
 READ_X_TILE ──▶ LOAD_X_TILE ──┐  (cold start only, first tile)   │
                                │                                   │
 WAIT_TILE ◀────────────────────┘                                   │
     │                                                              │
     │ w_valid=1                                                    │
     ▼                                                              │
 WAIT_PE ──▶ SUM_PARTIAL ──▶ READ_ACCUM ──▶ PREP_ACCUM ──▶ ACCUMULATE
                                                                │
                                         ┌──────────────────────┤
                                         │ (overflow)           │ (no overflow)
                                         ▼                      ▼
                                    READ_ACCUM_2 ──▶ ACCUMULATE_2 ──▶ WAIT_NEXT
                                                                          │
                                                     ┌────────────────────┤
                                                     │ last_in_row        │ !last_in_row
                                                     ▼                    ▼
                                              (next row or         WAIT_TILE ◀─┘
                                               READ_MAX)           (B2: tile prefetched)
```

**Per-tile cycle budget (steady state with B1+B2):**

| State | Cycles | Purpose |
|-------|--------|---------|
| WAIT_TILE | 1 | Handshake weight tile from gemv_execution |
| WAIT_PE | 1 | PE multiply latency |
| SUM_PARTIAL | 1 | A2: pairwise PE output sums (4 pairs) |
| READ_ACCUM | 1 | A2 stage-2 sum + prime res_mem BSRAM read |
| PREP_ACCUM | 1 | A3: register res_dout + sum; B2: prime x_mem word 0 |
| ACCUMULATE | 1 | Write accumulator; B2: capture x_mem word 0, prime word 1 |
| WAIT_NEXT | 1 | Update counters; B2: capture x_mem word 1 |
| **Total** | **7** | **(+3 for overflow tiles)** |

**B2 Prefetch:** During PREP_ACCUM/ACCUMULATE/WAIT_NEXT, x_mem is idle from PE computation. B2 uses these cycles to read the next tile's two packed words from x_mem, eliminating READ_X_TILE + LOAD_X_TILE from the steady-state loop (saves 3 cycles/tile).

### Phase 3: Post-Processing

```
READ_MAX ──▶ PREP_MAX ──▶ FIND_MAX ──▶ ... ──▶ COMPUTE_SCALE ──▶ READ_QUANTIZE ──▶ QUANTIZE ──▶ ... ──▶ READ_OUTPUT_Y ──▶ OUTPUT_Y ──▶ DONE
```

1. **FIND_MAX** — Scan res_mem for max absolute value across all rows. Uses A4: registered abs isolates BSRAM→abs→compare chain.
2. **COMPUTE_SCALE** — Calculate `reciprocal = 2^23 / max_abs` (iterative division via `scale_calculator`).
3. **QUANTIZE** — Apply scale via 32×32 Wallace tree multiply, round, saturate to int8. Results written back to res_mem.
4. **OUTPUT_Y** — Stream quantized results as 8-element tiles back to gemv_execution.

### Processing Element (PE)

```
┌──────────────────────┐
│         PE           │
│                      │
│  w (int8)  ──┐       │
│              ├──▶ y = w × x (int16)
│  x (int8)  ──┘       │
│                      │
│  1-cycle latency     │
└──────────────────────┘

8 PEs operate in parallel per tile.
Throughput: 8 MACs/cycle during accumulation.
```

### Tile Count Budget(on the tested model)

| Layer | Rows × Tiles/row | Total tiles |
|-------|-------------------|-------------|
| fc1 | 12 × 98 | 1,176 |
| fc2 | 32 × 2 | 64 |
| fc3 | 10 × 4 | 40 |
| **Total** | | **1,280** |

---

## Quantization Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                Quantization (within gemv_unit_core)           │
│                                                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  1. Calibration: FIND_MAX                             │   │
│  │     max_abs = max(|res_mem[i]|) for i in 0..rows-1   │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  2. Scale: COMPUTE_SCALE (scale_calculator.sv)        │   │
│  │     reciprocal_scale = (127 << 16) / max_abs          │   │
│  │     Uses iterative division (shift-subtract)          │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  3. Quantize: QUANTIZE (quantizer_pipeline.sv)        │   │
│  │     For each res_mem[i]:                               │   │
│  │       product = int32_value × reciprocal_scale         │   │
│  │                 (wallace_32x32 multiplier)             │   │
│  │       shifted = product >> 23                          │   │
│  │       result  = saturate(shifted, -128, 127)          │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  4. Output: OUTPUT_Y                                   │   │
│  │     Stream quantized int8 values as 8-element tiles   │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

The Wallace tree multiplier (`wallace_32x32.sv`) uses `compressor_3to2` full adders to reduce partial products, producing a 64-bit result from two 32-bit operands.

---

## Module Descriptions

### Core Modules

| Module | File | Purpose |
|--------|------|---------|
| **tinyml_accelerator_top** | `rtl/tinyml_accelerator_top.sv` | Top-level FSM, instruction sequencing |
| **fpga_top** | `src/fpga_top.sv` | FPGA top with UART + button I/O |
| **fetch_unit** | `rtl/fetch_unit.sv` | Sequential instruction fetch from DRAM |
| **fetch_unit_fpga** | `src/fetch_unit_fpga.sv` | FPGA variant with FETCH_PRIME for BRAM latency |
| **i_decoder** | `rtl/i_decoder.sv` | Combinational instruction decode |
| **simple_memory** | `rtl/simple_memory.sv` | Unified 32 KB DRAM (sim: reg array) |

### Execution Modules

| Module | File | Purpose |
|--------|------|---------|
| **modular_execution_unit** | `rtl/fpga_modules/modular_execution.sv` | Execution coordinator, memory arbitration |
| **buffer_controller** | `rtl/fpga_modules/buffer_controller.sv` | Dual buffer file management |
| **buffer_file** | `rtl/fpga_modules/buffer_file.sv` | Tile-indexed register file storage |
| **load_execution** | `rtl/fpga_modules/load_execution.sv` | LOAD_V / LOAD_M coordination |
| **load_v** | `rtl/load_v.sv` | Vector load FSM (DRAM → buffer) |
| **load_m** | `rtl/load_m.sv` | Matrix load FSM (DRAM → buffer, row-aware) |
| **gemv_execution** | `rtl/fpga_modules/gemv_execution.sv` | GEMV tile bridging (buffer ↔ core) |
| **gemv_unit_core** | `rtl/fpga_modules/gemv_unit_core.sv` | Core GEMV FSM (PE array + quantization) |
| **relu_execution** | `rtl/fpga_modules/relu_execution.sv` | ReLU tile-streaming orchestration |
| **relu** | `rtl/relu.sv` | Element-wise max(0, x) |
| **store_execution** | `rtl/fpga_modules/store_execution.sv` | Buffer → DRAM write-back |
| **store** | `rtl/store.sv` | Tile-based memory write FSM |

### Computational Modules

| Module | File | Purpose |
|--------|------|---------|
| **pe** | `rtl/fpga_modules/pe.sv` | 8×8→16 bit signed multiply (1-cycle) |
| **scale_calculator** | `rtl/fpga_modules/scale_calculator.sv` | Iterative division for reciprocal scale |
| **quantizer_pipeline** | `rtl/fpga_modules/quantizer_pipeline.sv` | Pipelined int32→int8 quantization |
| **wallace_32x32** | `rtl/fpga_modules/wallace_32x32.sv` | 32-bit Wallace tree multiplier |
| **compressor_3to2** | `rtl/fpga_modules/compressor_3to2.sv` | Full adder (3:2 compression) |

### IP Blocks / Simulation Mocks

| Module | File | Purpose |
|--------|------|---------|
| **Gowin_SDPB_32** | `rtl/fpga_modules/Gowin_SDPB_32.sv` | BSRAM mock (1-cycle synchronous read) |
| **Gowin_RAM16SDP_Mock** | `rtl/fpga_modules/Gowin_RAM16SDP_Mock.sv` | LUTRAM mock (async read) |
| **Gowin_SP** | (Gowin IP, synthesis only) | Single-port BRAM for unified DRAM |

---

## Optimizations

### Applied Optimizations (chronological)

| ID | Name | Effect | Description |
|----|------|--------|-------------|
| — | FIND_MAX loop fix | −2,300 cy | Scan only `rows` entries, not `MAX_ROWS` |
| — | CLEAR_REMAINING removal | −6,700 cy | Stop zeroing unused accumulator rows |
| A1 | PE valid removal | −1,300 cy | Zero x_current_tile for invalid elements instead of gating adder tree |
| A2 | Pipelined adder tree | +1,280 cy, +Fmax | SUM_PARTIAL stage: pairwise sums halve logic depth |
| A3 | Registered write-back | +1,280 cy, +Fmax | PREP_ACCUM: register res_dout+sum before BSRAM write |
| A4 | Registered abs | +54 cy, +Fmax | PREP_MAX: register abs(res_dout) before FIND_MAX comparison |
| A6 | x_mem to BSRAM | +1,280 cy, +Fmax | Replace 128× LUTRAM with Gowin_SDPB_32 (eliminates 6-level MUX) |
| B1 | Pack x_mem 4:1 | −8,200 cy | 4 int8 per 32-bit word → 2 reads per tile (was 8) |
| B2 | Prefetch x tile | −3,831 cy | Load next tile during PREP_ACCUM/ACCUMULATE/WAIT_NEXT |

### Performance History

| State | Cycles | Fmax (MHz) | Latency (ms) |
|-------|--------|------------|--------------|
| Baseline (BSRAM accum) | 43,996 | 67 | — |
| +FIND_MAX/CLEAR fix | 34,942 | — | — |
| +A1–A4 | 36,221 | 84 | — |
| +A6 (x_mem BSRAM) | 37,501 | 91 | 0.411 |
| +B1 (pack x_mem 4:1) | 29,301 | 86 | 0.343 |
| **+B2 (prefetch x tile)** | **25,470** | **89** | **0.286** |

### Key Design Decisions

1. **BSRAM over LUTRAM** — x_mem and res_mem use Gowin_SDPB_32 (block RAM) instead of Gowin_RAM16SDP (LUTRAM). LUTRAM created deep MUX cascades that limited Fmax. BSRAM has 1-cycle read latency, requiring READ_xxx pipeline states, but eliminates the MUX critical path.

2. **Packed x_mem (B1)** — 4 int8 values per 32-bit BSRAM word. Reduces LOAD_X_TILE from 8 reads to 2, saving 6 cycles per weight tile across 1,280 tiles.

3. **Prefetch during accumulate (B2)** — x_mem is idle during WAIT_PE through WAIT_NEXT (the accumulate pipeline). B2 uses PREP_ACCUM and ACCUMULATE to prime and capture x_mem reads for the NEXT tile, eliminating READ_X_TILE + LOAD_X_TILE from steady-state.

4. **Unconditional adder tree (A1)** — Instead of gating PE outputs with validity masks (expensive MUX logic), invalid x elements are zeroed in LOAD_X_TILE. pe_out = w × 0 = 0 naturally, so the adder tree sum is correct without gating.

### Critical Path

The critical path is in `buffer_controller → vector_buffer_inst` (opcode decode → load execution → buffer controller → tile index), not in the GEMV core. Further Fmax improvements require pipelining the buffer controller's opcode decode path.

---

## Two RTL Trees

| Tree | Purpose | Top Module | Memory | IP Blocks |
|------|---------|------------|--------|-----------|
| `src/` | FPGA synthesis (Gowin EDA) | `fpga_top.sv` | Gowin_SP BRAM + UART | Native Gowin IP |
| `rtl/` + `rtl/fpga_modules/` | Simulation (Verilator/cocotb) | `tinyml_accelerator_top.sv` | Register array + `$readmemh` | Mock modules |

`rtl/fpga_modules/` mirrors `src/` with simulation-compatible mocks:
- `Gowin_SDPB_32.sv` — Mock for Gowin BSRAM (registered read, 1-cycle latency)
- `Gowin_RAM16SDP_Mock.sv` — Mock for Gowin LUTRAM (async read)

After modifying `rtl/fpga_modules/gemv_unit_core.sv`, sync to `src/top_gemv.sv`.

### FPGA-Only Differences
- `src/fpga_top.sv` — Adds UART RX/TX, button debounce, LED output
- `src/fetch_unit_fpga.sv` — Adds FETCH_PRIME state for Gowin_SP 1-cycle read latency
- `src/simple_memory.sv` — Gowin_SP BRAM with UART write port (OCE must be 0)

---

## FPGA Synthesis Results (Gowin GW2AR-18)

- **Fmax**: 89.201 MHz (11 logic levels)
- **Cycles/image**: 25,470
- **Latency**: 0.286 ms/image (~3,500 images/sec)
- **Logic**: 42% (8,640 / 20,736)
- **BSRAM**: 94% (43 / 46)
- **DSP**: 5 blocks (1× MULT36X36 + 4× MULTADDALU18X18)
- **MNIST Accuracy**: 95% (10,000 images), 100% exact match vs golden model

---

| Version | Date | Changes |
|---------|------|---------|
| 2.0 | Mar 2026 | Complete rewrite for FPGA architecture: unified memory, 8 PEs, BSRAM, B1/B2 optimizations |
| 1.0 | Dec 2025 | Initial documentation |
