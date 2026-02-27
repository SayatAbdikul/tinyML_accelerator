// Testbench for Store Execution Module
// Tests STORE operations from buffer to memory
// Simulates buffer controller (source) and memory (destination)

#include "Vstore_execution.h"
#include <cstdint>
#include <cstdio>
#include <map>
#include <vector>
#include <verilated.h>
#include <verilated_vcd_c.h>

const int MAX_CYCLES = 2000;
const int TILE_ELEMS = 32;
const int DATA_WIDTH = 8;

class StoreExecutionTB {
private:
  Vstore_execution *dut;
  VerilatedVcdC *trace;
  uint64_t time_counter;
  std::map<uint32_t, uint8_t> memory;

  // Mock Buffer Storage
  std::map<int, std::vector<int8_t>> buffer_data;
  // Read pointer for each buffer to simulate sequential access
  std::map<int, int> buffer_read_ptrs;

public:
  StoreExecutionTB() {
    dut = new Vstore_execution;
    Verilated::traceEverOn(true);
    trace = new VerilatedVcdC;
    dut->trace(trace, 99);
    trace->open("store_execution.vcd");
    time_counter = 0;
  }

  ~StoreExecutionTB() {
    trace->close();
    delete trace;
    delete dut;
  }

  void tick() {
    time_counter++;
    dut->clk = 0;
    dut->eval();

    // --- Mock Buffer Controller Response Logic ---
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
          dut->vec_read_tile[i] = i + req_buffer_id;
      }

      // Auto-increment read pointer (simulating buffer_file behavior)
      buffer_read_ptrs[req_buffer_id] = current_offset + TILE_ELEMS;

      req_pending = false;
    }

    // Capture new request (to be served NEXT cycle)
    if (dut->vec_read_enable && !serving_now) {
      req_pending = true;
      req_buffer_id = dut->vec_read_buffer_id;
    }

    // --- Mock Memory Write Logic ---
    // The DUT asserts mem_we and provides address/data.
    if (dut->mem_we) {
      memory[dut->mem_addr] = dut->mem_wdata;
      // printf("Mem Write: Addr 0x%04x = 0x%02x\n", dut->mem_addr,
      // dut->mem_wdata);
    }

    // Always ready
    dut->mem_ready = 1;

    trace->dump(time_counter * 10);

    dut->clk = 1;
    dut->eval();
    trace->dump(time_counter * 10 + 5);
  }

  void reset() {
    printf("=== Store Execution Module Testbench ===\n");
    printf("Applying reset...\n");

    dut->rst = 1;
    dut->start = 0;
    dut->src_buffer_id = 0;
    dut->length = 0;
    dut->addr = 0;
    dut->vec_read_valid = 0;
    dut->mem_ready = 0;

    buffer_read_ptrs.clear();

    for (int i = 0; i < 5; i++)
      tick();

    dut->rst = 0;
    tick();
    printf("Reset released\n");
  }

  void clear_memory() { memory.clear(); }

  void set_buffer_data(int id, const std::vector<int8_t> &data) {
    buffer_data[id] = data;
    buffer_read_ptrs[id] = 0; // Reset pointer when setting data
  }

  bool verify_memory(uint32_t start_addr,
                     const std::vector<uint8_t> &expected) {
    bool pass = true;
    for (size_t i = 0; i < expected.size(); i++) {
      uint32_t addr = start_addr + i;
      if (memory.find(addr) == memory.end()) {
        printf("❌ Missing write at 0x%04x\n", addr);
        pass = false;
      } else if (memory[addr] != expected[i]) {
        printf("❌ Mismatch at 0x%04x: Expected 0x%02x, Got 0x%02x\n", addr,
               expected[i], memory[addr]);
        pass = false;
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

  void test_store_vector() {
    printf("\n--- Test STORE Operation ---\n");
    printf("Storing 64 elements from buffer 5 to DRAM 0x2000\n");

    // Setup Mock Buffer Data
    std::vector<int8_t> test_data;
    std::vector<uint8_t> expected_mem;
    for (int i = 0; i < 64; i++) {
      test_data.push_back(i + 10);
      expected_mem.push_back(i + 10);
    }
    set_buffer_data(5, test_data);
    clear_memory();

    dut->src_buffer_id = 5;
    dut->length = 64;
    dut->addr = 0x2000;
    dut->start = 1;

    tick();
    dut->start = 0;

    if (wait_for_done()) {
      if (verify_memory(0x2000, expected_mem)) {
        printf("✅ Data verification PASSED\n");
      } else {
        printf("❌ Data verification FAILED\n");
      }
    }

    tick();
    tick();
  }

  void test_store_partial_tile() {
    printf("\n--- Test Store Partial Tile ---\n");
    printf("Storing 13 elements from buffer 2 to DRAM 0x3000\n");

    std::vector<int8_t> test_data;
    std::vector<uint8_t> expected_mem;
    for (int i = 0; i < 13; i++) {
      test_data.push_back(0xA0 + i);
      expected_mem.push_back(0xA0 + i);
    }
    set_buffer_data(2, test_data);
    clear_memory();

    dut->src_buffer_id = 2;
    dut->length = 13;
    dut->addr = 0x3000;
    dut->start = 1;

    tick();
    dut->start = 0;

    if (wait_for_done()) {
      if (verify_memory(0x3000, expected_mem)) {
        printf("✅ Data verification PASSED\n");
      } else {
        printf("❌ Data verification FAILED\n");
      }

      // Ensure no extra bytes written
      if (memory.size() == 13) {
        printf("✅ Exact count written (13 bytes)\n");
      } else {
        printf("❌ Extra writes detected. Memory size: %lu\n", memory.size());
      }
    }

    tick();
    tick();
  }

  void run_all_tests() {
    reset();
    test_store_vector();
    test_store_partial_tile();

    printf("\n=== Store Execution Tests Complete ===\n");
    printf("Total simulation time: %llu cycles\n", time_counter);
  }
};

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  StoreExecutionTB tb;
  tb.run_all_tests();
  return 0;
}
