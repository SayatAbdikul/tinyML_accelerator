gtkwave dump.vcd
# TinyML Accelerator - Cocotb Golden Model Verification

Single cocotb testbench that runs the TinyML accelerator RTL end-to-end and compares against the Python golden model. The RTL is started once, runs through all instructions until the zero instruction is fetched, and the outputs are checked element-by-element with a small quantization tolerance.

## Overview

Main test: `test_golden_comparison.py`

- Generates/uses `dram.hex` (instructions, weights, biases, inputs) and syncs it into all RTL memories.
- Pulses `start` once, waits for `done` when the zero instruction is hit.
- Reads RTL outputs from the store memory at `0x20000` (10 bytes) and also from `y[]` for debug.
- Runs the Python golden model on the same `dram.hex` and compares outputs (accepts max error ≤ 2).
- Uses a small MNIST subset (first 10 test images) for faster runs.

## Prerequisites

- Verilator
  - macOS: `brew install verilator`
  - Ubuntu/Debian: `sudo apt-get install verilator`
- Python deps: from this directory you can use a venv (recommended)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start (from `test/cocotb_tests`)

```bash
# Recommended
python3 run_test.py --clean

# Or
make run_test

# Or
./run.sh
```

These will prepare `dram.hex` if missing, build with Verilator, run the cocotb test, and write `dump.vcd` and `results.xml` here.

## Manual Preparation (optional)

```bash
cd ../../compiler
python3 main.py   # generates dram.hex
cd ../test/cocotb_tests
make              # runs the testbench
```

## Test Files

- `test_golden_comparison.py` — main testbench with:
  - `test_accelerator_mnist_dataset` (end-to-end, MNIST subset)
  - `test_single_instruction_load_v` (sanity)
  - `test_reset_behavior` (reset check)
- `Makefile` — cocotb + Verilator build; `make run_test` prepares and runs.
- `run_test.py` — convenience wrapper (dependency check, `dram.hex` generation, run make).
- `run.sh` — quick wrapper.

## Memory Map (must match compiler)

- Instructions: `0x00000`
- Input: `0x00700`
- Weights: `0x10700`
- Biases: `0x13000`
- Output: `0x20000` (10 bytes)

## Debugging

- Waveform: `dump.vcd` (view with `gtkwave dump.vcd`).
- Increase timeout in `execute_all(timeout_cycles=...)` if you change program length.
- Watch signals: `tinyml_accelerator_top.t_state`, `tinyml_accelerator_top.instr`, `tinyml_accelerator_top.done`, `execution_u.state`, memory interfaces.

## Extending

- When RTL behavior changes, update `compiler/golden_model.py`, `assembler.py`, and `test_golden_comparison.py` together.
- Keep address constants aligned with `compiler/compile.py`.
- If you add instructions, update decoding/execution in both RTL and golden model and extend the test.
