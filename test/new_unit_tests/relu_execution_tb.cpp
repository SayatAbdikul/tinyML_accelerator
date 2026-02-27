// Testbench for ReLU Execution Module
// Tests ReLU activation with proper source/destination buffer handling
// Simulates buffer controller for both reads and writes

#include "Vrelu_execution.h"
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <map>
#include <vector>
#include <verilated.h>
#include <verilated_vcd_c.h>

const int MAX_CYCLES = 2000;
const int TILE_ELEMS = 32;

class ReluExecutionTB {
private:
  Vrelu_execution *dut;
  VerilatedVcdC *trace;
  uint64_t time_counter;

  // Mock Buffer Storage
  // key: buffer_id, value: vector of data
  std::map<int, std::vector<int8_t>> buffer_data;
  // Read pointer for each buffer to simulate sequential access
  std::map<int, int> buffer_read_ptrs;

  // Write storage to verify results
  std::map<int, std::vector<int8_t>> buffer_writes;

public:
  ReluExecutionTB() {
    dut = new Vrelu_execution;
    Verilated::traceEverOn(true);
    trace = new VerilatedVcdC;
    dut->trace(trace, 99);
    trace->open("relu_execution.vcd");
    time_counter = 0;
  }

  ~ReluExecutionTB() {
    trace->close();
    delete trace;
    delete dut;
  }

  void tick() {
    time_counter++;
    dut->clk = 0;
    dut->eval();

    // --- Mock Buffer Controller Read Logic ---
    static bool req_pending = false;
    static int req_buffer_id = 0;

    // Reset valid signal by default
    dut->vec_read_valid = 0;
    bool serving_now = false;

    // Check if we need to respond to a previous request
    if (req_pending) {
      dut->vec_read_valid = 1;
      serving_now = true;

      // Get current read pointer for this buffer (default to 0)
      int current_offset = buffer_read_ptrs[req_buffer_id];

      // Provide Data
      if (buffer_data.find(req_buffer_id) != buffer_data.end()) {
        const auto &data = buffer_data[req_buffer_id];
        for (int i = 0; i < TILE_ELEMS; i++) {
          if ((current_offset + i) < data.size()) {
            dut->vec_read_tile[i] = data[current_offset + i];
          } else {
            dut->vec_read_tile[i] = 0;
          }
        }
      } else {
        // Default pattern if buffer not initialized
        for (int i = 0; i < TILE_ELEMS; i++)
          dut->vec_read_tile[i] = 0;
      }

      // Auto-increment read pointer
      buffer_read_ptrs[req_buffer_id] = current_offset + TILE_ELEMS;

      req_pending = false;
    }

    // Capture new request (to be served NEXT cycle)
    // Suppress capture if we just served (handshake overlap)
    if (dut->vec_read_enable && !serving_now) {
      req_pending = true;
      req_buffer_id = dut->vec_read_buffer_id;
    }

    // --- Mock Buffer Controller Write Logic ---
    if (dut->vec_write_enable) {
      int buf_id = dut->vec_write_buffer_id;
      for (int i = 0; i < TILE_ELEMS; i++) {
        // We don't have an exact offset from the DUT (it just streams tiles)
        // So we just append to the vector for that buffer
        buffer_writes[buf_id].push_back((int8_t)dut->vec_write_tile[i]);
      }
    }

    trace->dump(time_counter * 10);

    dut->clk = 1;
    dut->eval();
    trace->dump(time_counter * 10 + 5);
  }

  void reset() {
    printf("=== ReLU Execution Module Testbench ===\n");
    printf("Applying reset...\n");

    dut->rst = 1;
    dut->start = 0;
    dut->dest_buffer_id = 0;
    dut->x_buffer_id = 0;
    dut->length = 0;
    dut->vec_read_valid = 0;

    buffer_read_ptrs.clear();
    buffer_writes.clear();

    for (int i = 0; i < 5; i++)
      tick();

    dut->rst = 0;
    tick();
    printf("Reset released\n");
  }

  void set_buffer_data(int id, const std::vector<int8_t> &data) {
    buffer_data[id] = data;
    buffer_read_ptrs[id] = 0;
  }

  bool verify_buffer_data(int id, const std::vector<int8_t> &expected) {
    if (buffer_writes.find(id) == buffer_writes.end()) {
      printf("❌ No writes recorded for buffer %d\n", id);
      return false;
    }

    const auto &actual = buffer_writes[id];
    bool pass = true;

    // Only compare up to expected size (ignore potential padding in last tile)
    if (actual.size() < expected.size()) {
      printf("❌ Data size mismatch: Expected %lu, Got %lu\n", expected.size(),
             actual.size());
      pass = false;
    }

    for (size_t i = 0; i < expected.size(); i++) {
      if (actual[i] != expected[i]) {
        printf("❌ Mismatch at index %lu: Expected %d, Got %d\n", i, expected[i],
               actual[i]);
        pass = false;
        if (i > 10)
          break; // Don't spam
      }
    }

    return pass;
  }

  bool wait_for_done(int max_cycles = MAX_CYCLES) {
    int cycle = 0;
    while (!dut->done && cycle < max_cycles) {
      tick();
      cycle++;
    }

    if (dut->done) {
      printf("✅ Operation completed in %d cycles\n", cycle);
      return true;
    } else {
      printf("❌ Operation timed out after %d cycles\n", max_cycles);
      return false;
    }
  }

  void test_relu_single_tile() {
    printf("\n--- Test ReLU Single Tile ---\n");
    printf("Testing ReLU: buffer 5 -> buffer 10, length=32\n");

    std::vector<int8_t> input_data;
    std::vector<int8_t> expected_data;

    for (int i = 0; i < 32; i++) {
      int8_t val = (i % 2 == 0) ? (i - 16) : (i + 10);
      input_data.push_back(val);
      expected_data.push_back(val < 0 ? 0 : val);
    }

    set_buffer_data(5, input_data);
    buffer_writes.clear();

    dut->dest_buffer_id = 10;
    dut->x_buffer_id = 5;
    dut->length = 32;
    dut->start = 1;

    tick();
    dut->start = 0;

    if (wait_for_done()) {
      if (verify_buffer_data(10, expected_data)) {
        printf("✅ ReLU computation correct\n");
      } else {
        printf("❌ ReLU computation failed\n");
      }
    }

    tick();
    tick();
  }

  void test_relu_multiple_tiles() {
    printf("\n--- Test ReLU Multiple Tiles ---\n");
    printf("Testing ReLU: buffer 3 -> buffer 7, length=96 (3 tiles)\n");

    std::vector<int8_t> input_data;
    std::vector<int8_t> expected_data;

    for (int i = 0; i < 96; i++) {
      int8_t val = (i % 2 == 0) ? (i - 50) : (i - 20);
      input_data.push_back(val);
      expected_data.push_back(val < 0 ? 0 : val);
    }

    set_buffer_data(3, input_data);
    buffer_writes.clear();

    dut->dest_buffer_id = 7;
    dut->x_buffer_id = 3;
    dut->length = 96;
    dut->start = 1;

    tick();
    dut->start = 0;

    if (wait_for_done()) {
      if (verify_buffer_data(7, expected_data)) {
        printf("✅ ReLU multiple tiles computation correct\n");
      } else {
        printf("❌ ReLU multiple tiles computation failed\n");
      }
    }

    tick();
    tick();
  }

  void test_relu_partial_tile() {
    printf("\n--- Test ReLU Partial Tile ---\n");
    printf("Testing ReLU: buffer 1 -> buffer 2, length=10\n");

    std::vector<int8_t> input_data;
    std::vector<int8_t> expected_data;

    // Create 10 elements
    for (int i = 0; i < 10; i++) {
      int8_t val = (i - 5) * 10; // -50, -40, ..., 40
      input_data.push_back(val);
      expected_data.push_back(val < 0 ? 0 : val);
    }

    set_buffer_data(1, input_data);
    buffer_writes.clear();

    dut->dest_buffer_id = 2;
    dut->x_buffer_id = 1;
    dut->length = 10;
    dut->start = 1;

    tick();
    dut->start = 0;

    if (wait_for_done()) {
      if (verify_buffer_data(2, expected_data)) {
        printf("✅ ReLU partial tile computation correct\n");
      } else {
        printf("❌ ReLU partial tile computation failed\n");
      }

      // Check zero padding (crucial for partial tiles)
      const auto &actual = buffer_writes[2];
      bool padding_ok = true;
      for (size_t i = 10; i < 32; i++) { // Check remaining elements in the tile
        if (actual[i] != 0) {
          printf("❌ Non-zero padding at index %lu: %d\n", i, actual[i]);
          padding_ok = false;
        }
      }
      if (padding_ok)
        printf("✅ Zero padding verified\n");
    }

    tick();
    tick();
  }

  void run_all_tests() {
    reset();
    test_relu_single_tile();
    test_relu_multiple_tiles();
    test_relu_partial_tile();

    printf("\n=== ReLU Execution Tests Complete ===\n");
    printf("Total simulation time: %llu cycles\n", time_counter);
  }
};

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  ReluExecutionTB tb;
  tb.run_all_tests();
  return 0;
}
