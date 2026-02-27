// Integration Testbench for Modular Execution Unit
// Tests complete operation sequences with all modules integrated
// Includes Mock DRAM for Load/Store operations

#include "Vmodular_execution_unit.h"
#include <cstdint>
#include <cstdio>
#include <map>
#include <vector>
#include <verilated.h>
#include <verilated_vcd_c.h>

const int MAX_CYCLES = 50000;

class ModularExecutionUnitTB {
private:
  Vmodular_execution_unit *dut;
  VerilatedVcdC *trace;
  uint64_t time_counter;

  // Mock Memory
  std::map<uint32_t, uint8_t> memory;

  // Memory latency simulation
  int mem_read_latency_counter = 0;
  bool mem_read_pending = false;
  uint8_t mem_read_data_buffer = 0;

public:
  ModularExecutionUnitTB() {
    dut = new Vmodular_execution_unit;
    Verilated::traceEverOn(true);
    trace = new VerilatedVcdC;
    dut->trace(trace, 99);
    trace->open("modular_execution_unit.vcd");
    time_counter = 0;
  }

  ~ModularExecutionUnitTB() {
    trace->close();
    delete trace;
    delete dut;
  }

  void tick() {
    time_counter++;
    dut->clk = 0;
    dut->eval();

    // --- Mock Memory Logic ---

    // Default responses
    dut->mem_valid = 0;

    // Handle Write Request (Immediate)
    if (dut->mem_req && dut->mem_we) {
      memory[dut->mem_addr] = dut->mem_wdata;
      // printf("  Mem Write: [0x%x] = 0x%02x\n", dut->mem_addr,
      // dut->mem_wdata);
    }

    // Handle Read Request
    // If free to accept new read
    if (dut->mem_req && !dut->mem_we && !mem_read_pending) {
      mem_read_pending = true;
      mem_read_latency_counter = 4; // Simulate 4 cycle latency

      if (memory.find(dut->mem_addr) != memory.end()) {
        mem_read_data_buffer = memory[dut->mem_addr];
      } else {
        mem_read_data_buffer = 0; // Default 0
      }
      // printf("  Mem Read Req: [0x%x]\n", dut->mem_addr);
    }

    // Handle Read Response Latency
    if (mem_read_pending) {
      if (mem_read_latency_counter > 0) {
        mem_read_latency_counter--;
      } else {
        dut->mem_valid = 1;
        dut->mem_rdata = mem_read_data_buffer;
        mem_read_pending = false;
        // printf("  Mem Read Data: 0x%02x\n", mem_read_data_buffer);
      }
    }

    trace->dump(time_counter * 10);

    dut->clk = 1;
    dut->eval();
    trace->dump(time_counter * 10 + 5);
  }

  void reset() {
    printf("=== Modular Execution Unit Integration Testbench ===\n");
    printf("Applying reset...\n");

    dut->rst = 1;
    dut->start = 0;
    dut->opcode = 0;
    dut->dest = 0;
    dut->length_or_cols = 0;
    dut->rows = 0;
    dut->addr = 0;
    dut->b_id = 0;
    dut->x_id = 0;
    dut->w_id = 0;

    dut->mem_valid = 0;
    dut->mem_rdata = 0;

    mem_read_pending = false;

    for (int i = 0; i < 5; i++)
      tick();

    dut->rst = 0;
    tick();
    printf("Reset released\n\n");
  }

  // Helper to write data to mock memory
  void write_mock_memory(uint32_t start_addr,
                         const std::vector<uint8_t> &data) {
    for (size_t i = 0; i < data.size(); i++) {
      memory[start_addr + i] = data[i];
    }
  }

  // Helper to verify mock memory content
  bool verify_mock_memory(uint32_t start_addr,
                          const std::vector<uint8_t> &expected) {
    bool pass = true;
    for (size_t i = 0; i < expected.size(); i++) {
      uint32_t addr = start_addr + i;
      if (memory.find(addr) == memory.end()) {
        printf("❌ Missing memory at 0x%x\n", addr);
        pass = false;
      } else if (memory[addr] != expected[i]) {
        printf("❌ Mismatch at 0x%x: Expected 0x%02x, Got 0x%02x\n", addr,
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

  void start_operation(uint8_t op, uint8_t dest_reg = 0, uint16_t cols = 10,
                       uint16_t rows_param = 10, uint32_t address = 0x1000,
                       uint8_t b = 0, uint8_t w = 0, uint8_t x = 0) {
    dut->opcode = op;
    dut->dest = dest_reg;
    dut->length_or_cols = cols;
    dut->rows = rows_param;
    dut->addr = address;
    dut->b_id = b;
    dut->w_id = w;
    dut->x_id = x;
    dut->start = 1;

    tick();
    dut->start = 0;
  }

  void test_full_flow() {
    printf("=== Test: Full Pipeline (Load -> GEMV -> ReLU -> Store) ===\n");
    printf("1. Load Input (32 elements) -> Buffer 1\n");
    printf("2. Load Weights (32x32 matrix) -> Buffer 2\n");
    printf("3. GEMV (32x32 * 32x1) -> Buffer 3\n");
    printf("4. ReLU (Buffer 3) -> Buffer 4\n");
    printf("5. Store (Buffer 4) -> Memory 0x4000\n\n");

    // Setup Data
    std::vector<uint8_t> input_vec(32);
    std::vector<uint8_t> weights(1024); // 32x32

    // Initialize with simple values
    for (int i = 0; i < 32; i++)
      input_vec[i] = 1; // Input = all 1s
    for (int i = 0; i < 1024; i++)
      weights[i] = 1; // Identity-like or simple sum

    // Write to mock memory
    write_mock_memory(0x1000, input_vec);
    write_mock_memory(0x2000, weights);

    // 1. Load Input
    printf("[1] Loading Input Vector...\n");
    start_operation(0x01, 1, 32, 0, 0x1000); // LOAD_V to Buf 1
    if (!wait_for_done(200))
      return;

    // 2. Load Weights
    printf("[2] Loading Weight Matrix...\n");
    start_operation(0x02, 2, 32, 32,
                    0x2000); // LOAD_M to Buf 2 (32 cols, 32 rows)
    if (!wait_for_done(10000))
      return; // Loads take time (1024 bytes * 4 cycles latency ~= 4096 cycles)

    // 3. GEMV
    // Result should be: Each output = Row * Input = sum(1*1 for 32 elems) = 32
    // However, GEMV unit applies dynamic quantization, scaling max value (32)
    // to ~127. So we expect 127 (0x7F).
    printf("[3] Performing GEMV...\n");
    start_operation(0x04, 3, 32, 32, 0, 0, 2,
                    1); // GEMV: Dest=3, W=2, X=1, B=0(unused)
    if (!wait_for_done(20000))
      return;

    // 4. ReLU
    // Input is 32 (positive), so output should remain 32.
    // Let's verify isolation: if we used -32 somehow, it should become 0.
    printf("[4] Performing ReLU...\n");
    start_operation(0x05, 4, 32, 0, 0, 0, 0, 3); // ReLU: Dest=4, Src=3
    if (!wait_for_done(200))
      return;

    // 5. Store
    printf("[5] Storing Result...\n");
    start_operation(0x03, 4, 32, 0, 0x4000); // STORE: Src=4, Addr=0x4000
    if (!wait_for_done(200))
      return;

    // Verify Memory
    // Expected: 32 elements. Each value 127 (0x7F) due to dynamic quantization.
    std::vector<uint8_t> expected_result(32, 127);
    printf("Verifying Memory at 0x4000 (Expecting 0x7F)...\n");
    if (verify_mock_memory(0x4000, expected_result)) {
      printf("✅ Full Pipeline Test Verification PASSED!\n");
    } else {
      printf("❌ Full Pipeline Test Verification FAILED!\n");
    }
  }

  void run_all_tests() {
    reset();
    test_full_flow();

    printf("\n");
    printf("═══════════════════════════════════════════════════\n");
    printf("  Modular Execution Unit Integration Tests Complete\n");
    printf("═══════════════════════════════════════════════════\n");
    printf("Total simulation time: %llu cycles\n", time_counter);
  }
};

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  ModularExecutionUnitTB tb;
  tb.run_all_tests();
  return 0;
}
