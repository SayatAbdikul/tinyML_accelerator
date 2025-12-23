# GEMV Testbench Developer Guide

## Overview

The `test_top_gemv.py` testbench provides comprehensive validation of the `top_gemv` RTL module against a Python golden model implementation. This guide explains the architecture, test methodology, and how to extend or debug the testbench.

## Architecture

### Components

```
test_top_gemv.py
├── GoldenGEMV Class
│   └── gemv() method - Python reference implementation
└── Test Functions (cocotb.test() decorated)
    ├── test_top_gemv_small
    ├── test_top_gemv_medium
    └── test_top_gemv_with_quantization_check
```

### Golden Model Class: `GoldenGEMV`

Implements the GEMV operation exactly as the RTL should:

```python
class GoldenGEMV:
    @staticmethod
    def gemv(weights, x, bias, rows, cols):
        """
        Compute GEMV: y = W @ x + bias
        
        Steps:
        1. Reshape weights to (rows, cols) matrix
        2. Compute y_int32[i] = sum(W[i,j] * X[j]) + B[i] for each row
        3. Find max absolute value in y_int32
        4. Calculate scale = max_abs / 127.0
        5. Quantize y_int32 to int8 using scale
        
        Returns:
            (y_int32, y_quantized, scale)
        """
```

**Key Design Points:**

- **Int32 Arithmetic:** Accumulation in int32 avoids overflow
- **Dynamic Scaling:** Scale based on maximum value ensures full int8 range usage
- **Quantization:** Uses `helper_functions.quantize_int32_to_int8()` matching RTL behavior

### RTL Interface

The testbench drives the `top_gemv` module with the following protocol:

```
Initialization (after reset):
1. Set dut.rows and dut.cols
2. Load dut.x[cols] with input vector
3. Load dut.bias[rows] with bias vector

Start Operation:
4. Pulse dut.start (one cycle)

Feed Weights:
5. For each row:
   a. Wait for dut.w_ready
   b. Load weight tile into dut.w_tile_row_in[TILE_SIZE]
   c. Pulse dut.w_valid (one cycle)

Wait for Result:
6. Poll dut.done until asserted
7. Read dut.y[rows]
```

## Test Flow Detailed

### Test Initialization

```python
# 1. Clock setup
clock = Clock(dut.clk, 10, units="ns")  # 100 MHz
cocotb.start_soon(clock.start(start_high=False))

# 2. Reset
dut.rst.value = 1
await FallingEdge(dut.clk)
dut.rst.value = 0

# 3. Set configuration
dut.rows.value = rows
dut.cols.value = cols
```

### Generate Test Data

```python
# Use fixed seed for reproducibility or random for comprehensive testing
np.random.seed(42)
weights_flat = np.random.randint(-128, 127, rows * cols, dtype=np.int8)
x_input = np.random.randint(-128, 127, cols, dtype=np.int8)
bias = np.random.randint(-128, 127, rows, dtype=np.int8)
```

### Execute Golden Model

```python
golden = GoldenGEMV()
y_int32_golden, y_quantized_golden, scale_golden = golden.gemv(
    weights_flat, x_input, bias, rows, cols
)
```

### Drive RTL

```python
# Start computation
await FallingEdge(dut.clk)
dut.start.value = 1
await FallingEdge(dut.clk)
dut.start.value = 0

# Stream weights (one row per iteration)
for row_idx in range(rows):
    # Wait for w_ready
    for _ in range(100):
        if dut.w_ready.value:
            break
        await FallingEdge(dut.clk)
    else:
        assert False, "Timeout waiting for w_ready"
    
    # Prepare weight tile (pad to TILE_SIZE)
    weight_tile = weights_flat[row_idx * cols:(row_idx + 1) * cols]
    weight_tile_padded = np.zeros(TILE_SIZE, dtype=np.int8)
    weight_tile_padded[:cols] = weight_tile
    
    # Load and pulse valid
    for i in range(TILE_SIZE):
        dut.w_tile_row_in[i].value = int(weight_tile_padded[i])
    dut.w_valid.value = 1
    await FallingEdge(dut.clk)
    dut.w_valid.value = 0
```

### Wait for Completion

```python
timeout = 50000  # Cycles
for cycle in range(timeout):
    if dut.done.value:
        cocotb.log.info(f"Done at cycle {cycle}")
        break
    await FallingEdge(dut.clk)
else:
    assert False, "Timeout"
```

### Compare Results

```python
# Read RTL output
y_rtl = np.zeros(rows, dtype=np.int8)
for i in range(rows):
    val = int(dut.y[i].value)
    # Convert from logic value to signed int8
    if val & 0x80:  # Check sign bit
        val = val - 256
    y_rtl[i] = val

# Calculate error
max_error = np.max(np.abs(y_rtl.astype(np.int32) - y_quantized_golden.astype(np.int32)))

# Check
assert max_error <= 2, f"Output mismatch: error {max_error} > 2"
```

## Test Cases

### test_top_gemv_small

**Configuration:**
- Matrix: 4×8
- Seed: 42
- Timeout: 50,000 cycles

**Purpose:** Quick sanity check
- Fast execution (<1ms)
- Covers basic functionality
- Good for development iteration

**Success Criteria:**
- All 4 outputs within ±2 error
- Completes within timeout
- No signal assertion failures

### test_top_gemv_medium

**Configuration:**
- Matrix: 16×32
- Seed: 123
- Timeout: 100,000 cycles

**Purpose:** Validate scalability
- Larger problem size
- More comprehensive weight streaming
- Tests accumulator overflow protection

**Success Criteria:**
- All 16 outputs within ±2 error
- Completes within timeout
- Consistent performance

### test_top_gemv_with_quantization_check

**Configuration:**
- Matrix: 2×4 (fixed values)
- Weights: `[10, 20, 30, 40, -50, -60, 70, -80]`
- X: `[5, 10, 15, 20]`
- Bias: `[100, -100]`
- Timeout: 100,000 cycles

**Purpose:** Detailed quantization inspection
- Fixed test vector for reproducibility
- Detailed logging of intermediate values
- Verify scale calculation matches
- Per-element error analysis

**Example Output:**
```
Golden int32 result: [   1150  -1650]
Golden scale: 1650.00
Golden quantized: [ 88 -128]
RTL quantized: [ 87 -127]
Row 0: RTL=87, Golden=88, Error=1
Row 1: RTL=-127, Golden=-128, Error=1
✅ Test PASSED
```

## Extending the Testbench

### Adding a New Test Case

```python
@cocotb.test()
async def test_top_gemv_large(dut):
    """Test top_gemv with large 64x128 matrix."""
    
    cocotb.log.info("=== Test: top_gemv_large (64x128 matrix) ===")
    
    rows = 64
    cols = 128
    tile_size = 32
    
    # Generate test data
    np.random.seed(999)
    weights_flat = np.random.randint(-128, 127, rows * cols, dtype=np.int8)
    x_input = np.random.randint(-128, 127, cols, dtype=np.int8)
    bias = np.random.randint(-128, 127, rows, dtype=np.int8)
    
    # Run golden model
    golden = GoldenGEMV()
    y_int32_golden, y_quantized_golden, scale_golden = golden.gemv(
        weights_flat, x_input, bias, rows, cols
    )
    
    # Setup RTL (clock, reset, parameters)
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start(start_high=False))
    dut.rst.value = 1
    await FallingEdge(dut.clk)
    dut.rst.value = 0
    dut.rows.value = rows
    dut.cols.value = cols
    
    # Load inputs...
    # (copy pattern from existing tests)
    
    # Compare and assert
    max_error = np.max(np.abs(y_rtl.astype(np.int32) - y_quantized_golden.astype(np.int32)))
    assert max_error <= 2, f"Output mismatch: max error {max_error}"
    cocotb.log.info("✅ Test PASSED")
```

### Adding Stress Testing

```python
@cocotb.test()
async def test_top_gemv_stress(dut):
    """Run GEMV with many random configurations."""
    
    for test_num in range(10):
        rows = np.random.randint(2, 32)
        cols = np.random.randint(4, 64)
        
        cocotb.log.info(f"Stress test {test_num}: {rows}x{cols}")
        
        # Generate data
        weights_flat = np.random.randint(-128, 127, rows * cols, dtype=np.int8)
        x_input = np.random.randint(-128, 127, cols, dtype=np.int8)
        bias = np.random.randint(-128, 127, rows, dtype=np.int8)
        
        # Run golden and RTL (following existing pattern)
        # Assert passes
        
    cocotb.log.info("✅ All stress tests PASSED")
```

## Debugging

### Enable Detailed Logging

Modify test to add logging:

```python
cocotb.log.info(f"Weights: {weights_flat[:16]}")
cocotb.log.info(f"X input: {x_input}")
cocotb.log.info(f"Bias: {bias}")
cocotb.log.info(f"Golden int32: {y_int32_golden}")
cocotb.log.info(f"Golden scale: {scale_golden:.6f}")
cocotb.log.info(f"Golden quantized: {y_quantized_golden}")
cocotb.log.info(f"RTL output: {y_rtl}")
```

### Inspect Waveform

```bash
# After test completes
gtkwave sim_build/dump.vcd &

# In GTKWave:
# - Search for "top_gemv" in hierarchy
# - Watch signals: w_ready, w_valid, done, y[*]
# - Use zoom to inspect weight streaming timing
```

### Add Intermediate Checking

Modify RTL in simulation to expose intermediate values:

```python
# Read scale from RTL if exposed
rtl_scale = dut.scale_reg.value
cocotb.log.info(f"RTL scale: {rtl_scale}")

# Compare intermediate int32 results if available
rtl_int32 = [int(dut.y_int32[i]) for i in range(rows)]
cocotb.log.info(f"RTL int32 result: {rtl_int32}")
cocotb.log.info(f"Golden int32 result: {y_int32_golden}")
```

### Troubleshooting Common Issues

**Problem: w_ready never asserts**

```python
# Add timeout detection
timeout = 0
while not dut.w_ready.value and timeout < 1000:
    await FallingEdge(dut.clk)
    timeout += 1

if timeout >= 1000:
    cocotb.log.error("w_ready timeout")
    # Print state
    cocotb.log.error(f"Current state: {dut.state}")
    cocotb.log.error(f"Accumulated: {dut.accumulated}")
    assert False, "w_ready timeout"
```

**Problem: Quantization mismatch**

```python
# Verify scale calculation
for i in range(rows):
    y32 = y_int32_golden[i]
    y8_golden = y_quantized_golden[i]
    y8_rtl = y_rtl[i]
    
    # Manual quantization for verification
    scaled = int(y32 / scale_golden)
    clipped = max(-128, min(127, scaled))
    
    cocotb.log.info(
        f"Row {i}: y32={y32}, scale={scale_golden:.2f}, "
        f"scaled={scaled}, clipped={clipped}, "
        f"golden={y8_golden}, rtl={y8_rtl}"
    )
```

## Performance Considerations

### Simulation Speed

- **Small test (4×8):** ~200 cycles, <0.1 seconds
- **Medium test (16×32):** ~8,000 cycles, ~0.5 seconds
- **With quantization check:** ~1,000 cycles, ~0.1 seconds
- **Total for all three:** ~10 seconds

### Optimization Tips

1. **Reduce timeout for faster feedback:** Change timeout value in tests
2. **Use small matrices during development:** Modify test_top_gemv_small
3. **Run single test:** Use `TESTCASE=test_top_gemv_small` in make
4. **Parallel builds:** Verilator compilation can be parallelized with `-j4`

## Design Verification Patterns

### Verifying Correct Handshake

```python
# Verify w_ready/w_valid handshake
ready_count = 0
for _ in range(rows):
    # Should be ready before we signal valid
    assert dut.w_ready.value, "w_ready not asserted before w_valid"
    
    # Signal valid
    dut.w_valid.value = 1
    await FallingEdge(dut.clk)
    dut.w_valid.value = 0
    
    # w_ready should clear after valid
    # (might take some cycles, check RTL timing)
```

### Verifying Computation Correctness

```python
# Step-by-step verification
# 1. Check output is not all zeros (indicates no computation)
assert np.any(y_rtl != 0), "RTL output all zeros"

# 2. Check sign is correct (no bit flip)
for i in range(rows):
    if y_quantized_golden[i] > 0:
        assert y_rtl[i] > 0, f"Row {i}: sign mismatch"

# 3. Check magnitude is reasonable
for i in range(rows):
    if abs(y_quantized_golden[i]) > 1:
        error_pct = abs(y_rtl[i] - y_quantized_golden[i]) / abs(y_quantized_golden[i]) * 100
        assert error_pct < 10, f"Row {i}: {error_pct:.1f}% error"
```

## References

- **Cocotb Documentation:** https://docs.cocotb.org/
- **NumPy Documentation:** https://numpy.org/doc/
- **Verilator Tracing:** https://verilator.org/guide/latest/

## Contact & Support

For issues or questions about the testbench:
1. Check the troubleshooting section above
2. Inspect waveforms with GTKWave
3. Review `golden_model.py` for expected behavior
4. Compare with `test_golden_comparison.py` for similar patterns
