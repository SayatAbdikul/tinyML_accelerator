// Verilator testbench for gemv_execution.sv
// Tests the tile-bridging wrapper that converts 32-element buffer tiles
// to 6-element GEMV unit tiles and back.
//
// The testbench simulates a buffer controller:
//  - Provides X, bias, and weight data via vec_read_tile / mat_read_tile
//  - Captures results from vec_write_tile
//  - Compares HW output against a software golden model

#include "Vgemv_execution.h"
#include "verilated.h"
#include "verilated_vcd_c.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <vector>

// ----- Test parameters -----
#define ROWS 40
#define COLS 40
#define TILE_ELEMS 32
#define DATA_WIDTH 8
#define MAX_WAIT 200000

// Buffer IDs used in the test
#define X_BUF_ID 1
#define B_BUF_ID 2
#define W_BUF_ID 3
#define DST_BUF_ID 4

Vgemv_execution *dut;
vluint64_t main_time = 0;
double sc_time_stamp() { return main_time; }

void tick(VerilatedVcdC *tfp) {
  dut->clk = 0;
  dut->eval();
  if (tfp)
    tfp->dump(main_time++);
  dut->clk = 1;
  dut->eval();
  if (tfp)
    tfp->dump(main_time++);
}

// ----- Test data storage -----
static int8_t x_vec[COLS];
static int8_t bias_vec[ROWS];
static int8_t W_mat[ROWS][COLS]; // W[row][col]

// Buffer tile storage: data is served in 32-element tiles
// X vector tiles
static int x_buf_tiles;
static int8_t x_tiles[32][TILE_ELEMS]; // max 32 buffer tiles

// Bias vector tiles
static int b_buf_tiles;
static int8_t b_tiles[32][TILE_ELEMS];

// Weight matrix tiles: stored row-major, each row split into 32-elem tiles
static int w_buf_tiles_per_row;
static int8_t w_tiles[800][TILE_ELEMS]; // row * tiles_per_row indexed

// Track which tile to serve next for each buffer
static int x_tile_ptr = 0;
static int b_tile_ptr = 0;
static int w_tile_ptr = 0;

// Result collection
static int8_t hw_result[ROWS];
static int result_idx = 0;

// ----- Generate random test data -----
void generate_data() {
  srand(42);
  for (int i = 0; i < COLS; i++)
    x_vec[i] = (int8_t)(rand() % 201 - 100); // -100..100
  for (int i = 0; i < ROWS; i++)
    bias_vec[i] = (int8_t)(rand() % 21 - 10); // -10..10
  for (int r = 0; r < ROWS; r++)
    for (int c = 0; c < COLS; c++)
      W_mat[r][c] = (int8_t)(rand() % 11 - 5); // -5..5

  // Pack X into 32-element buffer tiles
  x_buf_tiles = (COLS + TILE_ELEMS - 1) / TILE_ELEMS;
  memset(x_tiles, 0, sizeof(x_tiles));
  for (int t = 0; t < x_buf_tiles; t++)
    for (int i = 0; i < TILE_ELEMS; i++) {
      int idx = t * TILE_ELEMS + i;
      x_tiles[t][i] = (idx < COLS) ? x_vec[idx] : 0;
    }

  // Pack bias into 32-element buffer tiles
  b_buf_tiles = (ROWS + TILE_ELEMS - 1) / TILE_ELEMS;
  memset(b_tiles, 0, sizeof(b_tiles));
  for (int t = 0; t < b_buf_tiles; t++)
    for (int i = 0; i < TILE_ELEMS; i++) {
      int idx = t * TILE_ELEMS + i;
      b_tiles[t][i] = (idx < ROWS) ? bias_vec[idx] : 0;
    }

  // Pack weight matrix into 32-element buffer tiles (row-major, per row)
  w_buf_tiles_per_row = (COLS + TILE_ELEMS - 1) / TILE_ELEMS;
  memset(w_tiles, 0, sizeof(w_tiles));
  for (int r = 0; r < ROWS; r++)
    for (int t = 0; t < w_buf_tiles_per_row; t++)
      for (int i = 0; i < TILE_ELEMS; i++) {
        int col = t * TILE_ELEMS + i;
        w_tiles[r * w_buf_tiles_per_row + t][i] =
            (col < COLS) ? W_mat[r][col] : 0;
      }
}

// ----- Software golden model -----
// Performs GEMV with quantization (matches gemv_unit_core behavior)
void sw_golden(int8_t *y_out) {
  // 1. Compute int32 accumulator: y32[r] = bias[r] + sum(W[r][c]*x[c])
  int32_t y32[ROWS];
  for (int r = 0; r < ROWS; r++) {
    y32[r] = (int32_t)bias_vec[r];
    for (int c = 0; c < COLS; c++)
      y32[r] += (int32_t)W_mat[r][c] * (int32_t)x_vec[c];
  }

  // 2. Find max absolute value
  int32_t max_abs = 0;
  for (int r = 0; r < ROWS; r++) {
    int32_t av = (y32[r] >= 0) ? y32[r] : -y32[r];
    if (av > max_abs)
      max_abs = av;
  }
  if (max_abs == 0)
    max_abs = 1;

  // 3. Compute reciprocal scale in Q8.24 format (matches scale_calculator.sv)
  // HW computes: (127 << 24) / max_abs using 32-cycle binary long division
  int64_t reciprocal_scale = ((int64_t)127 << 24) / max_abs;

  // 4. Quantize each element (matches quantizer_pipeline.sv)
  // HW: signed multiply (int32 * Q8.24), then round-shift by 24
  for (int r = 0; r < ROWS; r++) {
    int64_t prod =
        (int64_t)y32[r] * reciprocal_scale; // signed * unsigned-as-signed
    int64_t shifted =
        (prod + ((int64_t)1 << 23)) >> 24; // round + arithmetic right shift
    if (shifted > 127)
      shifted = 127; // clamp
    if (shifted < -128)
      shifted = -128;
    y_out[r] = (int8_t)shifted;
  }
}

// ----- Buffer controller simulation -----
// The gemv_execution module asserts vec_read_enable/mat_read_enable for one
// cycle, then waits in a LOAD_*_BUF state polling
// vec_read_valid/mat_read_valid. We respond with 1-cycle latency: latch the
// request at the rising edge, provide the data on the NEXT rising edge.

static bool vec_read_pending = false;
static bool mat_read_pending = false;
static int vec_pending_buf_id = 0;

// Called AFTER tick() so DUT outputs reflect the latest clock edge.
void serve_buffer_controller() {
  // ---- Capture write results first (before clearing valid) ----
  if (dut->vec_write_enable) {
    for (int i = 0; i < TILE_ELEMS; i++) {
      if (result_idx + i < ROWS)
        hw_result[result_idx + i] = (int8_t)dut->vec_write_tile[i];
    }
    result_idx += TILE_ELEMS;
  }

  // ---- Latch new read requests (will be served on next tick) ----
  if (dut->vec_read_enable && !vec_read_pending) {
    vec_read_pending = true;
    vec_pending_buf_id = dut->vec_read_buffer_id;
  }
  if (dut->mat_read_enable && !mat_read_pending) {
    mat_read_pending = true;
  }
}

// Called BEFORE tick() to set inputs for the next rising edge.
void drive_buffer_inputs() {
  // Default: no valid data
  dut->vec_read_valid = 0;
  dut->mat_read_valid = 0;

  // Serve pending vector read
  if (vec_read_pending) {
    dut->vec_read_valid = 1;
    vec_read_pending = false;

    int8_t *src = nullptr;
    if (vec_pending_buf_id == X_BUF_ID) {
      src = x_tiles[x_tile_ptr++];
    } else if (vec_pending_buf_id == B_BUF_ID) {
      src = b_tiles[b_tile_ptr++];
    } else {
      static int8_t zeros[TILE_ELEMS] = {};
      src = zeros;
    }
    for (int i = 0; i < TILE_ELEMS; i++)
      dut->vec_read_tile[i] = src[i];
  }

  // Serve pending matrix read
  if (mat_read_pending) {
    dut->mat_read_valid = 1;
    mat_read_pending = false;
    int8_t *src = w_tiles[w_tile_ptr++];
    for (int i = 0; i < TILE_ELEMS; i++)
      dut->mat_read_tile[i] = src[i];
  }
}

// ----- Main -----
int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  Verilated::traceEverOn(true);

  dut = new Vgemv_execution;
  VerilatedVcdC *tfp = new VerilatedVcdC;
  dut->trace(tfp, 99);
  tfp->open("gemv_execution.vcd");

  generate_data();

  std::cout << "Test parameters: ROWS=" << ROWS << " COLS=" << COLS
            << " TILE_ELEMS=" << TILE_ELEMS << "\n";
  std::cout << "Buffer tiles: X=" << x_buf_tiles << " B=" << b_buf_tiles
            << " W_per_row=" << w_buf_tiles_per_row << "\n";

  // ---- Reset ----
  dut->rst = 1;
  dut->start = 0;
  dut->vec_read_valid = 0;
  dut->mat_read_valid = 0;
  for (int i = 0; i < TILE_ELEMS; i++) {
    dut->vec_read_tile[i] = 0;
    dut->mat_read_tile[i] = 0;
  }
  for (int i = 0; i < 5; i++)
    tick(tfp);
  dut->rst = 0;
  tick(tfp);

  // ---- Start GEMV execution ----
  dut->start = 1;
  dut->dest_buffer_id = DST_BUF_ID;
  dut->w_buffer_id = W_BUF_ID;
  dut->x_buffer_id = X_BUF_ID;
  dut->b_buffer_id = B_BUF_ID;
  dut->cols = COLS;
  dut->rows = ROWS;
  tick(tfp);
  dut->start = 0;

  std::cout << "[TB] Start pulse sent.\n";

  // ---- Run simulation until done or timeout ----
  int cycle = 0;
  bool timed_out = true;
  for (cycle = 0; cycle < MAX_WAIT; cycle++) {
    // 1. After previous tick, read DUT outputs and latch requests
    serve_buffer_controller();

    // 2. Drive inputs for next clock edge
    drive_buffer_inputs();

    // 3. Advance clock
    tick(tfp);

    if (dut->done) {
      // Capture any final write that happens on the done cycle
      serve_buffer_controller();
      std::cout << "[TB] DUT signaled done at cycle " << cycle << ".\n";
      timed_out = false;
      break;
    }

    // Progress reporting
    if (cycle > 0 && cycle % 10000 == 0)
      std::cout << "[TB] Cycle " << cycle << "...\n";
  }

  if (timed_out) {
    std::cerr << "TIMEOUT after " << MAX_WAIT << " cycles!\n";
    tfp->close();
    delete dut;
    return 1;
  }

  // ---- Compare with golden model ----
  int8_t sw_result[ROWS];
  sw_golden(sw_result);

  int mismatches = 0;
  for (int r = 0; r < ROWS; r++) {
    if (hw_result[r] != sw_result[r]) {
      mismatches++;
      if (mismatches <= 20)
        std::cout << "Mismatch Row " << r << ": HW=" << (int)hw_result[r]
                  << " SW=" << (int)sw_result[r] << "\n";
    }
  }

  if (mismatches == 0)
    std::cout << "PASSED! All " << ROWS << " outputs match.\n";
  else
    std::cout << "FAILED with " << mismatches << " mismatches.\n";

  std::cout << "Clock cycles: " << cycle << "\n";

  tfp->close();
  delete dut;
  return mismatches ? 1 : 0;
}
