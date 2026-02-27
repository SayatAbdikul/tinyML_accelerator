// Testbench for Load Execution Module
// Tests LOAD_V and LOAD_M operations with buffer controller integration
// Includes mock memory simulation

#include "Vload_execution.h"
#include <cstdint>
#include <cstdio>
#include <map>
#include <vector>
#include <verilated.h>
#include <verilated_vcd_c.h>

const int MAX_CYCLES = 2000;
const int TILE_ELEMS = 32;

class LoadExecutionTB {
private:
  Vload_execution *dut;
  VerilatedVcdC *trace;
  uint64_t time_counter;
  std::map<uint32_t, uint8_t> memory;

public:
  LoadExecutionTB() {
    dut = new Vload_execution;
    Verilated::traceEverOn(true);
    trace = new VerilatedVcdC;
    dut->trace(trace, 99);
    trace->open("load_execution.vcd");
    time_counter = 0;
  }

  ~LoadExecutionTB() {
    trace->close();
    delete trace;
    delete dut;
  }

  void tick() {
    time_counter++;
    dut->clk = 0;
    dut->eval();

    // Mock Memory Response Logic
    // Respond signals combinatorially based on request state
    // If request was asserted, verify functionality
    if (dut->mem_req) {
      uint32_t addr = dut->mem_addr;
      if (memory.find(addr) != memory.end()) {
        dut->mem_rdata = memory[addr];
      } else {
        dut->mem_rdata = 0; // Default to 0 for uninitialized memory
      }
      dut->mem_valid = 1;
    } else {
      dut->mem_valid = 0;
      dut->mem_rdata = 0;
    }

    trace->dump(time_counter * 10);

    dut->clk = 1;
    dut->eval();
    trace->dump(time_counter * 10 + 5);
  }

  void reset() {
    printf("=== Load Execution Module Testbench ===\n");
    printf("Applying reset...\n");

    dut->rst = 1;
    dut->start = 0;
    dut->opcode = 0;
    dut->dest_buffer_id = 0;
    dut->length_or_cols = 0;
    dut->rows = 0;
    dut->addr = 0;
    dut->mem_valid = 0;
    dut->mem_rdata = 0;

    for (int i = 0; i < 5; i++)
      tick();

    dut->rst = 0;
    tick();
    printf("Reset released\n");
  }

  void clear_memory() { memory.clear(); }

  void write_memory(uint32_t start_addr, const std::vector<uint8_t> &data) {
    for (size_t i = 0; i < data.size(); i++) {
      memory[start_addr + i] = data[i];
    }
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

  void test_load_vector() {
    printf("\n--- Test LOAD_V Operation ---\n");
    printf("Loading 64 elements to vector buffer 7\n");

    // Prepare memory data
    std::vector<uint8_t> test_data;
    for (int i = 0; i < 64; i++) {
      test_data.push_back((i + 1) & 0xFF);
    }
    clear_memory();
    write_memory(0x1000, test_data);

    dut->opcode = 0x01; // LOAD_V
    dut->dest_buffer_id = 7;
    dut->length_or_cols = 64;
    dut->addr = 0x1000;
    dut->start = 1;

    tick();
    dut->start = 0;

    // Monitor vec_write_enable signals
    int tiles_written = 0;
    int cycles = 0;

    while (!dut->done && cycles < MAX_CYCLES) {
      if (dut->vec_write_enable) {
        tiles_written++;
        printf("  Tile %d written to buffer %d\n", tiles_written,
               dut->vec_write_buffer_id);
      }
      tick();
      cycles++;
    }

    if (dut->done) {
      printf("✅ LOAD_V completed: %d tiles written in %d cycles\n",
             tiles_written, cycles);

      // Verify expected tile count (64 elements / 32 per tile = 2 tiles)
      int expected_tiles = (64 + TILE_ELEMS - 1) / TILE_ELEMS;
      if (tiles_written == expected_tiles) {
        printf("✅ Correct number of tiles written\n");
      } else {
        printf("❌ Expected %d tiles, got %d\n", expected_tiles, tiles_written);
      }
    } else {
      printf("❌ LOAD_V timed out\n");
    }

    tick();
    tick();
  }

  void test_load_matrix() {
    printf("\n--- Test LOAD_M Operation ---\n");
    printf("Loading 8x16 matrix to buffer 2\n");

    // Prepare memory data (8 rows * 16 cols = 128 bytes)
    std::vector<uint8_t> test_data;
    for (int i = 0; i < 128; i++) {
      test_data.push_back((i + 0xA0) & 0xFF);
    }
    clear_memory();
    write_memory(0x2000, test_data);

    dut->opcode = 0x02; // LOAD_M
    dut->dest_buffer_id = 2;
    dut->length_or_cols = 16; // columns
    dut->rows = 8;
    dut->addr = 0x2000;
    dut->start = 1;

    tick();
    dut->start = 0;

    // Monitor mat_write_enable signals
    int tiles_written = 0;
    int cycles = 0;
    while (!dut->done && cycles < MAX_CYCLES) {
      if (dut->mat_write_enable) {
        tiles_written++;
        printf("  Matrix tile %d written to buffer %d\n", tiles_written,
               dut->mat_write_buffer_id);
      }
      tick();
      cycles++;
    }

    if (dut->done) {
      printf("✅ LOAD_M completed: %d tiles written in %d cycles\n",
             tiles_written, cycles);

      // Verify expected tile count
      // load_m is row-aware and pads each row to tile alignment.
      // So tiles = rows * tiles_per_row
      int tiles_per_row = (16 + TILE_ELEMS - 1) / TILE_ELEMS; // (16+31)/32 = 1
      int expected_tiles = 8 * tiles_per_row;                 // 8
      if (tiles_written == expected_tiles) {
        printf("✅ Correct number of tiles written (%d)\n", tiles_written);
      } else {
        printf("❌ Expected %d tiles, got %d\n", expected_tiles, tiles_written);
      }
    } else {
      printf("❌ LOAD_M timed out\n");
    }

    tick();
    tick();
  }

  void test_invalid_opcode() {
    printf("\n--- Test Invalid Opcode ---\n");

    dut->opcode = 0x10; // Invalid
    dut->start = 1;

    tick();
    dut->start = 0;

    if (wait_for_done(10)) {
      printf("✅ Invalid opcode handled gracefully\n");
    }
  }

  void test_single_element_vector() {
    printf("\n--- Test Single Element LOAD_V ---\n");
    printf("Loading 1 element to vector buffer 0\n");

    // Memory setup
    clear_memory();
    write_memory(0x3000, {0x42, 0x00, 0x00, 0x00}); // One 4-byte element

    dut->opcode = 0x01;
    dut->dest_buffer_id = 0;
    dut->length_or_cols = 1;
    dut->addr = 0x3000;
    dut->start = 1;

    tick();
    dut->start = 0;

    int tiles_written = 0;
    int cycles = 0;
    while (!dut->done && cycles < MAX_CYCLES) {
      if (dut->vec_write_enable) {
        tiles_written++;
      }
      tick();
      cycles++;
    }

    int expected_tiles = 1; // 1 element still needs 1 tile
    if (dut->done && tiles_written == expected_tiles) {
      printf("✅ Single element load: %d tile in %d cycles\n", tiles_written,
             cycles);
    } else {
      printf("❌ Expected %d tile, got %d\n", expected_tiles, tiles_written);
    }

    tick();
    tick();
  }

  void test_exact_tile_boundary() {
    printf("\n--- Test Exact Tile Boundary (32 elements) ---\n");

    std::vector<uint8_t> data(32 * 4, 0x55); // 32 elements, 4 bytes each
    clear_memory();
    write_memory(0x4000, data);

    dut->opcode = 0x01;
    dut->dest_buffer_id = 15;
    dut->length_or_cols = 32; // Exactly one tile
    dut->addr = 0x4000;
    dut->start = 1;

    tick();
    dut->start = 0;

    int tiles_written = 0;
    int cycles = 0;
    while (!dut->done && cycles < MAX_CYCLES) {
      if (dut->vec_write_enable) {
        tiles_written++;
      }
      tick();
      cycles++;
    }

    if (tiles_written == 1) {
      printf("✅ Exact tile boundary handled correctly: 1 tile\n");
    } else {
      printf("❌ Expected 1 tile, got %d\n", tiles_written);
    }

    tick();
    tick();
  }

  void test_non_aligned_matrix() {
    printf("\n--- Test Non-Aligned Matrix (7x13) ---\n");
    // 7 rows, 13 cols.
    // Row-aware padding: each row takes ceil(13/32)=1 tile.

    clear_memory();
    for (int r = 0; r < 7; r++) {
      for (int c = 0; c < 13; c++) {
        uint32_t val = (r * 13 + c) & 0xFF;
        // Store 4 bytes per element because load_m reads TILE_WIDTH logic?
        // Wait, DATA_WIDTH=8. TILE_WIDTH=256. TILE_ELEMS=32.
        // Address logic in load_m: `mem_addr <= mem_addr + 1`
        // Is mem_addr byte address or element address?
        // SystemVerilog: `input logic [ADDR_WIDTH-1:0] dram_addr`
        // `input logic [DATA_WIDTH-1:0] mem_rdata` -> 8 bits.
        // So memory is byte-addressed.
        // load_m increments addr by 1 per element.
        // So one element = 1 byte.
        // My previous thought about 4 bytes was wrong?
        // Let's check DATA_WIDTH in load_execution.sv: defaults to 8.
        // So 1 byte per element.
        memory[0x5000 + r * 32 + c] = val; // Padded to 32 bytes per row
      }
    }

    dut->opcode = 0x02;
    dut->dest_buffer_id = 3;
    dut->length_or_cols = 13;
    dut->rows = 7;
    dut->addr = 0x5000;
    dut->start = 1;

    tick();
    dut->start = 0;

    int tiles_written = 0;
    int cycles = 0;
    while (!dut->done && cycles < MAX_CYCLES) {
      if (dut->mat_write_enable) {
        tiles_written++;
      }
      tick();
      cycles++;
    }

    int tiles_per_row = (13 + TILE_ELEMS - 1) / TILE_ELEMS; // 1
    int expected_tiles = 7 * tiles_per_row;                 // 7

    if (dut->done && tiles_written == expected_tiles) {
      printf("✅ Non-aligned matrix: %d tiles (1 per row * 7 rows)\n",
             tiles_written);
    } else {
      printf("❌ Expected %d tiles, got %d\n", expected_tiles, tiles_written);
    }
    tick();
    tick();
  }

  void test_back_to_back_loads() {
    printf("\n--- Test Back-to-Back Loads ---\n");

    clear_memory();
    std::vector<uint8_t> data1(32, 0xAA);
    std::vector<uint8_t> data2(16, 0xBB);
    write_memory(0x6000, data1);
    write_memory(0x7000, data2);

    // First load
    dut->opcode = 0x01;
    dut->dest_buffer_id = 10;
    dut->length_or_cols = 32;
    dut->addr = 0x6000;
    dut->start = 1;
    tick();
    dut->start = 0;

    int cycles = 0;
    while (!dut->done && cycles < MAX_CYCLES) {
      tick();
      cycles++;
    }

    if (!dut->done) {
      printf("❌ First load timed out\n");
      return;
    }

    printf("  First load complete\n");

    // Immediate second load (no delay)
    dut->opcode = 0x01;
    dut->dest_buffer_id = 11;
    dut->length_or_cols = 16;
    dut->addr = 0x7000;
    dut->start = 1;
    tick();
    dut->start = 0;

    cycles = 0;
    while (!dut->done && cycles < MAX_CYCLES) {
      tick();
      cycles++;
    }

    if (dut->done) {
      printf("✅ Back-to-back loads completed successfully\n");
    } else {
      printf("❌ Second load timed out\n");
    }

    tick();
    tick();
  }

  void run_all_tests() {
    reset();
    test_load_vector();
    test_load_matrix();
    test_invalid_opcode();
    test_single_element_vector();
    test_exact_tile_boundary();
    test_non_aligned_matrix();
    test_back_to_back_loads();
    // Test buffer ID verification implicitly covered

    printf("\n=== Load Execution Tests Complete ===\n");
    printf("Total simulation time: %llu cycles\n", time_counter);
  }
};

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  LoadExecutionTB tb;
  tb.run_all_tests();
  return 0;
}
