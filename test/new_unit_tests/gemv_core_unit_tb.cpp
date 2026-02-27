#include "Vgemv_unit_core.h"
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

#define ROWS 32
#define COLS 32
#define TILE 6
#define DWIDTH 8
#define MAX_WAIT 10000

Vgemv_unit_core *dut;

vluint64_t main_time = 0;
double sc_time_stamp() { return main_time; }

void tick(VerilatedVcdC *tfp) {
  dut->clk = 0;
  dut->eval();
  tfp->dump(main_time++);
  dut->clk = 1;
  dut->eval();
  tfp->dump(main_time++);
}

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  dut = new Vgemv_unit_core;
  VerilatedVcdC *tfp = new VerilatedVcdC;
  Verilated::traceEverOn(true);
  dut->trace(tfp, 99);
  tfp->open("dump.vcd");

  std::srand(std::time(0));

  int8_t w[ROWS][COLS];
  int8_t x[COLS];
  int8_t bias[ROWS];
  int32_t y_expected[ROWS] = {0};

  // 1. Randomization and Golden Model Computation
  for (int i = 0; i < ROWS; i++) {
    bias[i] = rand() % 256 - 128;
    for (int j = 0; j < COLS; j++) {
      w[i][j] = rand() % 256 - 128;
    }
  }

  for (int j = 0; j < COLS; j++) {
    x[j] = rand() % 256 - 128;
  }

  // Golden Model
  for (int i = 0; i < ROWS; i++) {
    int32_t sum = 0;
    for (int j = 0; j < COLS; j++) {
      sum += w[i][j] * x[j];
    }
    y_expected[i] = sum + bias[i];
  }

  // Calculate Hardware Parameters (Quantization)
  int32_t max_abs = 0;
  for (int i = 0; i < ROWS; i++) {
    if (std::abs(y_expected[i]) > max_abs)
      max_abs = std::abs(y_expected[i]);
  }
  if (max_abs == 0)
    max_abs = 1;

  uint32_t reciprocal_scale = (static_cast<uint32_t>(127) << 24) / max_abs;

  std::vector<int8_t> y_quantized_sw;
  for (int i = 0; i < ROWS; i++) {
    int64_t product = static_cast<int64_t>(y_expected[i]) * reciprocal_scale;
    int32_t q = static_cast<int32_t>((product + (1 << 23)) >> 24);
    if (q > 127)
      q = 127;
    if (q < -128)
      q = -128;
    y_quantized_sw.push_back(static_cast<int8_t>(q));
  }

  std::cout << "Test parameters: ROWS=" << ROWS << " COLS=" << COLS
            << " TILE=" << TILE << "\n";

  // Pre-declare all variables used after goto labels
  int tiles_x = (COLS + TILE - 1) / TILE;
  int tiles_bias = (ROWS + TILE - 1) / TILE;
  int tiles_per_row = (COLS + TILE - 1) / TILE;
  int errors = 0;
  int wait_cycles = 0;
  std::vector<int8_t> y_hw_streamed(ROWS, 0);

  // 2. Initialize DUT
  dut->rst = 1;
  dut->start = 0;
  dut->w_valid = 0;
  dut->x_tile_valid = 0;
  dut->bias_tile_valid = 0;
  dut->y_tile_ready = 1;
  dut->rows = ROWS;
  dut->cols = COLS;

  tick(tfp);
  dut->rst = 0;
  tick(tfp);

  dut->start = 1;
  tick(tfp);
  dut->start = 0;
  std::cout << "[TB] Start pulse sent. Streaming X tiles (" << tiles_x
            << ")...\n";

  // 3. Stream X Tiles
  for (int t = 0; t < tiles_x; t++) {
    wait_cycles = 0;
    while (!dut->x_tile_ready) {
      tick(tfp);
      if (++wait_cycles > MAX_WAIT) {
        std::cerr << "[TB] STUCK waiting for x_tile_ready, tile " << t << "\n";
        goto cleanup;
      }
    }
    dut->x_tile_valid = 1;
    dut->x_tile_idx = t;
    for (int k = 0; k < TILE; k++) {
      int idx = t * TILE + k;
      dut->x_tile_in[k] = (idx < COLS) ? x[idx] : 0;
    }
    tick(tfp);
    dut->x_tile_valid = 0;
  }
  std::cout << "[TB] X tiles loaded.\n";
  // Allow FSM to transition from LOAD_X to LOAD_BIAS
  tick(tfp);
  tick(tfp);

  // 4. Stream Bias Tiles
  for (int t = 0; t < tiles_bias; t++) {
    wait_cycles = 0;
    while (!dut->bias_tile_ready) {
      tick(tfp);
      if (++wait_cycles > MAX_WAIT) {
        std::cerr << "[TB] STUCK waiting for bias_tile_ready, tile " << t
                  << "\n";
        goto cleanup;
      }
    }
    dut->bias_tile_valid = 1;
    dut->bias_tile_idx = t;
    for (int k = 0; k < TILE; k++) {
      int idx = t * TILE + k;
      dut->bias_tile_in[k] = (idx < ROWS) ? bias[idx] : 0;
    }
    tick(tfp);
    dut->bias_tile_valid = 0;
  }
  std::cout << "[TB] Bias tiles loaded.\n";
  // Allow FSM to transition from LOAD_BIAS to CLEAR_MEM and clear all rows
  // CLEAR_MEM takes MAX_ROWS cycles (784), so we need to tick enough
  for (int i = 0; i < 800; i++)
    tick(tfp);

  // 5. Stream Weights (Row by Row, Tile by Tile)
  std::cout << "[TB] Streaming weight tiles...\n";
  for (int r = 0; r < ROWS; r++) {
    for (int t = 0; t < tiles_per_row; t++) {
      wait_cycles = 0;
      while (!dut->w_ready) {
        tick(tfp);
        if (++wait_cycles > MAX_WAIT) {
          std::cerr << "[TB] STUCK waiting for w_ready, row=" << r
                    << " tile=" << t << "\n";
          goto cleanup;
        }
      }

      dut->w_valid = 1;
      for (int k = 0; k < TILE; k++) {
        int c = t * TILE + k;
        dut->w_tile_row_in[k] = (c < COLS) ? w[r][c] : 0;
      }
      tick(tfp);
      dut->w_valid = 0;

      wait_cycles = 0;
      while (!dut->tile_done && !dut->done) {
        tick(tfp);
        if (++wait_cycles > MAX_WAIT) {
          std::cerr << "[TB] STUCK waiting for tile_done, row=" << r
                    << " tile=" << t << "\n";
          goto cleanup;
        }
      }
      if (dut->done)
        break;
    }
    if (r % 20 == 0)
      std::cout << "[TB] Weight row " << r << "/" << ROWS << " done.\n";
    if (dut->done)
      break;
  }
  std::cout
      << "[TB] All weight tiles streamed. Waiting for post-processing...\n";

  // 6. Capture Output Streaming
  // OUTPUT_Y outputs tiles before DONE_STATE, so we tick and capture
  // y_tile_valid
  while (!dut->done) {
    if (dut->y_tile_valid && dut->y_tile_ready) {
      int tidx = dut->y_tile_idx;
      for (int k = 0; k < TILE; k++) {
        int r = tidx * TILE + k;
        if (r < ROWS) {
          y_hw_streamed[r] = static_cast<int8_t>(dut->y_tile_out[k]);
        }
      }
    }
    tick(tfp);
    if (main_time > 4000000) {
      std::cerr << "[TB] Timeout waiting for DONE\n";
      break;
    }
  }

  // 7. Verify Results
  for (int i = 0; i < ROWS; i++) {
    if (y_hw_streamed[i] != y_quantized_sw[i]) {
      if (errors < 20) {
        std::cerr << "Mismatch Row " << i << ": HW=" << (int)y_hw_streamed[i]
                  << " SW=" << (int)y_quantized_sw[i] << "\n";
      }
      errors++;
    }
  }

  if (errors) {
    std::cout << "FAILED with " << errors << " mismatches.\n";
  } else {
    std::cout << "PASSED! All " << ROWS << " outputs match.\n";
  }
  std::cout << "Clock cycles: " << main_time / 2 << "\n";

cleanup:
  dut->final();
  tfp->close();
  delete dut;
  delete tfp;
  return errors ? 1 : 0;
}
