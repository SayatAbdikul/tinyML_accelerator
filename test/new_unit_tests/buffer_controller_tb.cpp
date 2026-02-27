// Testbench for Buffer Controller Module
// Tests vector and matrix buffer read/write operations

#include "Vbuffer_controller.h"
#include <cstdint>
#include <cstdio>
#include <verilated.h>
#include <verilated_vcd_c.h>

const int MAX_CYCLES = 1000;
const int TILE_ELEMS = 32;

class BufferControllerTB {
private:
  Vbuffer_controller *dut;
  VerilatedVcdC *trace;
  uint64_t time_counter;

public:
  BufferControllerTB() {
    dut = new Vbuffer_controller;
    Verilated::traceEverOn(true);
    trace = new VerilatedVcdC;
    dut->trace(trace, 99);
    trace->open("buffer_controller.vcd");
    time_counter = 0;
  }

  ~BufferControllerTB() {
    trace->close();
    delete trace;
    delete dut;
  }

  void tick() {
    time_counter++;
    dut->clk = 0;
    dut->eval();
    trace->dump(time_counter * 10); // to have better visibility in waveform

    dut->clk = 1;
    dut->eval();
    trace->dump(time_counter * 10 + 5);
  }

  void reset() {
    printf("=== Buffer Controller Testbench ===\n");
    printf("Applying reset...\n");

    dut->rst = 1;
    dut->vec_write_enable = 0;
    dut->vec_read_enable = 0;
    dut->mat_write_enable = 0;
    dut->mat_read_enable = 0;
    dut->clr_cache = 0;

    for (int i = 0; i < 5; i++)
      tick();

    dut->rst = 0;
    tick();
    printf("Reset released\n");
  }

  void test_vector_buffer_write_read() {
    printf("\n--- Test Vector Buffer Write/Read ---\n");

    // Write test data to vector buffer 5
    printf("Writing tile to vector buffer 5...\n");
    dut->vec_write_buffer_id = 5;
    dut->vec_write_enable = 1;

    // Fill tile with test pattern
    for (int i = 0; i < TILE_ELEMS; i++) {
      dut->vec_write_tile[i] = i + 10;
    }

    tick();
    dut->vec_write_enable = 0;
    tick();
    tick();

    // Read back from same buffer
    printf("Reading tile from vector buffer 5...\n");
    dut->vec_read_buffer_id = 5;
    dut->vec_read_enable = 1;
    tick();
    dut->vec_read_enable = 0;
    tick(); // Wait for read latency
    tick();

    // Verify data
    bool pass = true;
    for (int i = 0; i < TILE_ELEMS; i++) {
      int8_t expected = i + 10;
      int8_t actual = (int8_t)dut->vec_read_tile[i];
      if (actual != expected) {
        printf("❌ Mismatch at [%d]: expected %d, got %d\n", i, expected,
               actual);
        pass = false;
      }
    }

    if (pass) {
      printf("✅ Vector buffer write/read test PASSED\n");
    } else {
      printf("❌ Vector buffer write/read test FAILED\n");
    }
  }

  void test_matrix_buffer_write_read() {
    printf("\n--- Test Matrix Buffer Write/Read ---\n");

    // Write test data to matrix buffer 3
    printf("Writing tile to matrix buffer 3...\n");
    dut->mat_write_buffer_id = 3;
    dut->mat_write_enable = 1;

    // Create test pattern (packed 256-bit tile)
    uint32_t test_pattern[8]; // 256 bits = 8 x 32-bit words
    for (int i = 0; i < 8; i++) {
      test_pattern[i] = 0x01020304 + i;
    }

    // Copy to DUT (Verilator represents wide signals as arrays)
    for (int i = 0; i < 8; i++) {
      dut->mat_write_tile[i] = test_pattern[i];
    }

    tick();
    dut->mat_write_enable = 0;
    tick();
    tick();

    // Read back from same buffer
    printf("Reading tile from matrix buffer 3...\n");
    dut->mat_read_buffer_id = 3;
    dut->mat_read_enable = 1;
    tick();
    dut->mat_read_enable = 0;
    tick(); // Wait for read latency
    tick();

    // Verify data by unpacking mat_read_tile into bytes and comparing
    bool pass = true;

    // Matrix buffer outputs as unpacked array of 32 elements (same as vector)
    // We need to verify against our original test pattern
    // Original pattern was 8 x 32-bit words, now read back as 32 x 8-bit
    // elements
    for (int i = 0; i < TILE_ELEMS; i++) {
      // Reconstruct expected byte from test_pattern
      int word_idx = i / 4;     // Which 32-bit word (0-7)
      int byte_in_word = i % 4; // Which byte within that word (0-3)

      // Extract expected byte (little-endian)
      uint8_t expected = (test_pattern[word_idx] >> (byte_in_word * 8)) & 0xFF;
      uint8_t actual = (uint8_t)dut->mat_read_tile[i];

      if (actual != expected) {
        printf("❌ Mismatch at element [%d]: expected 0x%02X, got 0x%02X\n", i,
               expected, actual);
        pass = false;
      }
    }

    if (pass) {
      printf("✅ Matrix buffer write/read test PASSED\n");
      printf("   All 32 bytes verified correctly\n");
    } else {
      printf("❌ Matrix buffer write/read test FAILED\n");
    }
  }

  void test_multiple_buffers() {
    printf("\n--- Test Multiple Buffer IDs ---\n");

    // Write different patterns to buffers 0, 1, 2
    for (int buf_id = 0; buf_id < 3; buf_id++) {
      dut->vec_write_buffer_id = buf_id;
      dut->vec_write_enable = 1;

      for (int i = 0; i < TILE_ELEMS; i++) {
        dut->vec_write_tile[i] = (buf_id + 1) * 10 + i;
      }

      tick();
      dut->vec_write_enable = 0;
      tick();
    }

    // Read back and verify each buffer
    bool pass = true;
    for (int buf_id = 0; buf_id < 3; buf_id++) {
      dut->vec_read_buffer_id = buf_id;
      dut->vec_read_enable = 1;
      tick();
      dut->vec_read_enable = 0;
      tick();
      tick();

      for (int i = 0; i < TILE_ELEMS; i++) {
        int8_t expected = (buf_id + 1) * 10 + i;
        int8_t actual = (int8_t)dut->vec_read_tile[i];
        if (actual != expected) {
          printf("❌ Buffer %d mismatch at [%d]: expected %d, got %d\n", buf_id,
                 i, expected, actual);
          pass = false;
        }
      }
    }

    if (pass) {
      printf("✅ Multiple buffer test PASSED\n");
    } else {
      printf("❌ Multiple buffer test FAILED\n");
    }
  }

  void run_all_tests() {
    reset();
    test_vector_buffer_write_read();
    test_matrix_buffer_write_read();
    test_multiple_buffers();

    printf("\n=== Buffer Controller Tests Complete ===\n");
    printf("Total simulation time: %llu cycles\n", time_counter);
  }
};

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  BufferControllerTB tb;
  tb.run_all_tests();
  return 0;
}
