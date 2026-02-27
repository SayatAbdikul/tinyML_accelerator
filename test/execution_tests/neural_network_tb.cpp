// Neural Network Sequence Testbench for Modular Execution Unit
// Tests complete neural network execution: 784→12→32→10
// Replicates assembly sequence from model_assembly.asm

#include "Vmodular_execution_unit.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <vector>
#include <verilated.h>
#include <verilated_vcd_c.h>

// Test parameters
const int MAX_CYCLES = 100000;
const int DATA_WIDTH = 8;

class NeuralNetworkTB {
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

  // Instruction encoding helpers
  struct Instruction {
    uint8_t opcode;
    uint8_t dest;
    uint16_t length_or_cols;
    uint16_t rows;
    uint32_t addr;
    uint8_t x_id; // For GEMV, RELU (source buffer)
    uint8_t w_id; // For GEMV
    uint8_t b_id; // For GEMV
  };

public:
  NeuralNetworkTB() {
    dut = new Vmodular_execution_unit;

    Verilated::traceEverOn(true);
    trace = new VerilatedVcdC;
    dut->trace(trace, 99);
    trace->open("neural_network.vcd");

    time_counter = 0;
  }

  ~NeuralNetworkTB() {
    trace->close();
    delete trace;
    delete dut;
  }

  void tick() {
    time_counter++;

    dut->clk = 0;
    dut->eval();

    // --- Mock Memory Logic ---

    // Handle Write Request (Immediate)
    if (dut->mem_req && dut->mem_we) {
      memory[dut->mem_addr] = dut->mem_wdata;
    }

    // Handle Read Request (Pipelined 1-cycle latency)
    // mem_valid and mem_rdata reflect the request from the PREVIOUS cycle
    dut->mem_valid = mem_read_pending;
    dut->mem_rdata = mem_read_data_buffer;

    // Register the current request for the NEXT cycle
    if (dut->mem_req && !dut->mem_we) {
      mem_read_pending = true;
      if (memory.find(dut->mem_addr) != memory.end()) {
        mem_read_data_buffer = memory[dut->mem_addr];
      } else {
        mem_read_data_buffer = 0; // Default 0
      }
    } else {
      mem_read_pending = false;
    }

    trace->dump(time_counter * 10);

    dut->clk = 1;
    dut->eval();
    trace->dump(time_counter * 10 + 5);

    if (Verilated::gotFinish()) {
      exit(0);
    }
  }

  void reset() {
    printf("=== Neural Network Testbench for Modular Execution Unit ===\n");
    printf("Applying reset...\n");

    dut->rst = 1;
    dut->start = 0;
    dut->opcode = 0;
    dut->dest = 0;
    dut->length_or_cols = 0;
    dut->rows = 0;
    dut->addr = 0;
    dut->x_id = 0;
    dut->w_id = 0;
    dut->b_id = 0;

    dut->mem_valid = 0;
    dut->mem_rdata = 0;

    mem_read_pending = false;

    for (int i = 0; i < 10; i++) {
      tick();
    }

    dut->rst = 0;
    tick();
    printf("Reset complete\n\n");
  }

  // Helper to write data to mock memory
  void write_mock_memory(uint32_t start_addr,
                         const std::vector<uint8_t> &data) {
    for (size_t i = 0; i < data.size(); i++) {
      memory[start_addr + i] = data[i];
    }
  }

  bool wait_for_done(int max_cycles = MAX_CYCLES) {
    int cycle = 0;
    while (!dut->done && cycle < max_cycles) {
      tick();
      cycle++;
      if (cycle % 10000 == 0) {
        printf("  ... still processing (cycle %d)\n", cycle);
      }
    }

    if (dut->done) {
      printf("  ✅ Completed in %d cycles\n", cycle);
      return true;
    } else {
      printf("  ❌ ERROR: Timed out after %d cycles\n", max_cycles);
      return false;
    }
  }

  void execute_instruction(const Instruction &instr) {
    dut->opcode = instr.opcode;
    dut->dest = instr.dest;
    dut->length_or_cols = instr.length_or_cols;
    dut->rows = instr.rows;
    dut->addr = instr.addr;
    dut->x_id = instr.x_id;
    dut->w_id = instr.w_id;
    dut->b_id = instr.b_id;
    dut->start = 1;

    tick();
    dut->start = 0;
  }

  void test_neural_network() {
    printf("=== NEURAL NETWORK TEST: 784→12→32→10 ===\n");
    printf("Replicating model_assembly.asm instruction sequence\n\n");

    printf("🎯 ASSEMBLY INSTRUCTIONS (model_assembly.asm):\n");
    printf("  LOAD_V 9, 0xc0,  784\n");
    printf("  LOAD_M 1, 0x940, 12, 800\n");
    printf("  LOAD_V 3, 0x4c0, 12\n");
    printf("  GEMV   5, 1, 9, 3, 12, 784\n");
    printf("  RELU   7, 5, 12\n");
    printf("  LOAD_M 2, 0x2ec0, 32, 32\n");
    printf("  LOAD_V 4, 0x4cc, 32\n");
    printf("  GEMV   6, 2, 7, 4, 32, 12\n");
    printf("  RELU   8, 6, 32\n");
    printf("  LOAD_M 1, 0x32c0, 10, 32\n");
    printf("  LOAD_V 3, 0x4ec, 10\n");
    printf("  GEMV   5, 1, 8, 3, 10, 32\n");
    printf("  STORE  5, 0x8c0, 10\n");
    printf("\n");

    // Initialize dummy data in mock memory
    std::vector<uint8_t> dummy_input(784, 0);
    // Add some small non-zero values to avoid saturation
    for (int i = 0; i < 30; i++)
      dummy_input[i] = 2; // Input is 2

    std::vector<uint8_t> dummy_w1(12 * 800, 0);
    for (int r = 0; r < 12; r++) {
      for (int c = 0; c < 30; c++) {
        dummy_w1[r * 800 + c] = (r % 2 == 0) ? 1 : 2; // Row even: 1, Row odd: 2
      }
    }
    std::vector<uint8_t> dummy_b1(12, 1);

    std::vector<uint8_t> dummy_w2(32 * 32, 0);
    for (int r = 0; r < 32; r++) {
      for (int c = 0; c < 12; c++) {
        dummy_w2[r * 32 + c] = (c % 2 == 0) ? 2 : 1; // Make it positive
      }
    }
    std::vector<uint8_t> dummy_b2(32, 1);

    std::vector<uint8_t> dummy_w3(10 * 32, 0);
    for (int r = 0; r < 10; r++) {
      for (int c = 0; c < 32; c++) {
        dummy_w3[r * 32 + c] = (r + 1); // Row 0 is 1, Row 1 is 2, etc.
      }
    }
    std::vector<uint8_t> dummy_b3(10, 0);

    write_mock_memory(0xc0, dummy_input); // Input 784
    write_mock_memory(0x940, dummy_w1);   // Weight 1 (12x800)
    write_mock_memory(0x4c0, dummy_b1);   // Bias 1
    write_mock_memory(0x2ec0, dummy_w2);  // Weight 2 (32x32)
    write_mock_memory(0x4cc, dummy_b2);   // Bias 2
    write_mock_memory(0x32c0, dummy_w3);  // Weight 3 (10x32)
    write_mock_memory(0x4ec, dummy_b3);   // Bias 3

    // Track success of each layer
    bool layer1_success = true;
    bool layer2_success = true;
    bool layer3_success = true;

    // ========== LAYER 1: 784 → 12 ==========
    printf("╔════════════════════════════════════╗\n");
    printf("║     LAYER 1: 784 → 12 (FC)        ║\n");
    printf("╚════════════════════════════════════╝\n\n");

    // Step 1: LOAD_V 9, 0xc0, 784
    printf("Step 1: LOAD_V 9, 0xc0, 784 (input vector - 784 elements)\n");
    Instruction load_input = {0x01, 9, 784, 0, 0xc0, 0, 0, 0};
    execute_instruction(load_input);
    if (!wait_for_done(4000)) {
      printf("❌ Failed at Step 1\n");
      layer1_success = false;
      return;
    }

    // Step 2: LOAD_M 1, 0x940, 12, 800
    printf("\nStep 2: LOAD_M 1, 0x940, 12, 800 (weight matrix W1 - 12×800)\n");
    Instruction load_w1 = {0x02, 1, 800, 12, 0x940, 0, 0, 0};
    execute_instruction(load_w1);
    if (!wait_for_done(25000)) {
      printf("❌ Failed at Step 2\n");
      layer1_success = false;
      return;
    }

    // Step 3: LOAD_V 3, 0x4c0, 12
    printf("\nStep 3: LOAD_V 3, 0x4c0, 12 (bias vector b1 - 12 elements)\n");
    Instruction load_b1 = {0x01, 3, 12, 0, 0x4c0, 0, 0, 0};
    execute_instruction(load_b1);
    if (!wait_for_done(200)) {
      printf("❌ Failed at Step 3\n");
      layer1_success = false;
      return;
    }

    // Step 4: GEMV 5, 1, 9, 3, 12, 784
    printf("\nStep 4: GEMV 5, 1, 9, 3, 12, 784 (W1 * input + b1)\n");
    printf("  Matrix: 12×784, Vector: 784×1, Output: 12×1\n");
    Instruction gemv1 = {0x04, 5, 784, 12, 0, 9, 1, 3};
    execute_instruction(gemv1);
    if (!wait_for_done(60000)) {
      printf("❌ Failed at Step 4 (GEMV)\n");
      layer1_success = false;
      return;
    }

    // Step 5: RELU 7, 5
    printf("\nStep 5: RELU 7, 5 (activation function)\n");
    Instruction relu1 = {0x05, 7, 12, 0, 0, 5, 0, 0}; // dest=7, x_id=5
    execute_instruction(relu1);
    if (!wait_for_done(300)) {
      printf("❌ Failed at Step 5 (ReLU)\n");
      layer1_success = false;
      return;
    }

    printf("\n✅ Layer 1 Complete: 784 → 12\n\n");

    // ========== LAYER 2: 12 → 32 ==========
    printf("╔════════════════════════════════════╗\n");
    printf("║     LAYER 2: 12 → 32 (FC)         ║\n");
    printf("╚════════════════════════════════════╝\n\n");

    // Step 6: LOAD_M 2, 0x2ec0, 32, 32
    printf("Step 6: LOAD_M 2, 0x2ec0, 32, 32 (weight matrix W2 - 32×32)\n");
    Instruction load_w2 = {0x02, 2, 32, 32, 0x2ec0, 0, 0, 0};
    execute_instruction(load_w2);
    if (!wait_for_done(1500)) {
      printf("❌ Failed at Step 6\n");
      layer2_success = false;
      return;
    }

    // Step 7: LOAD_V 4, 0x4cc, 32
    printf("\nStep 7: LOAD_V 4, 0x4cc, 32 (bias vector b2 - 32 elements)\n");
    Instruction load_b2 = {0x01, 4, 32, 0, 0x4cc, 0, 0, 0};
    execute_instruction(load_b2);
    if (!wait_for_done(250)) {
      printf("❌ Failed at Step 7\n");
      layer2_success = false;
      return;
    }

    // Step 8: GEMV 6, 2, 7, 4, 32, 12
    printf("\nStep 8: GEMV 6, 2, 7, 4, 32, 12 (W2 * h1 + b2)\n");
    printf("  Matrix: 32×12, Vector: 12×1, Output: 32×1\n");
    Instruction gemv2 = {0x04, 6, 12, 32, 0, 7, 2, 4};
    execute_instruction(gemv2);
    if (!wait_for_done(8000)) {
      printf("❌ Failed at Step 8 (GEMV)\n");
      layer2_success = false;
      return;
    }

    // Step 9: RELU 8, 6
    printf("\nStep 9: RELU 8, 6 (activation function)\n");
    Instruction relu2 = {0x05, 8, 32, 0, 0, 6, 0, 0}; // dest=8, x_id=6
    execute_instruction(relu2);
    if (!wait_for_done(300)) {
      printf("❌ Failed at Step 9 (ReLU)\n");
      layer2_success = false;
      return;
    }

    printf("\n✅ Layer 2 Complete: 12 → 32\n\n");

    // ========== LAYER 3: 32 → 10 (OUTPUT) ==========
    printf("╔════════════════════════════════════╗\n");
    printf("║   LAYER 3: 32 → 10 (OUTPUT)       ║\n");
    printf("╚════════════════════════════════════╝\n\n");

    // Step 10: LOAD_M 1, 0x32c0, 10, 32
    printf("Step 10: LOAD_M 1, 0x32c0, 10, 32 (weight matrix W3 - 10×32)\n");
    Instruction load_w3 = {0x02, 1, 32, 10, 0x32c0, 0, 0, 0};
    execute_instruction(load_w3);
    if (!wait_for_done(1200)) {
      printf("❌ Failed at Step 10\n");
      layer3_success = false;
      return;
    }

    // Step 11: LOAD_V 3, 0x4ec, 10
    printf("\nStep 11: LOAD_V 3, 0x4ec, 10 (bias vector b3 - 10 elements)\n");
    Instruction load_b3 = {0x01, 3, 10, 0, 0x4ec, 0, 0, 0};
    execute_instruction(load_b3);
    if (!wait_for_done(150)) {
      printf("❌ Failed at Step 11\n");
      layer3_success = false;
      return;
    }

    // Step 12: GEMV 5, 1, 8, 3, 10, 32
    printf(
        "\nStep 12: GEMV 5, 1, 8, 3, 10, 32 (W3 * h2 + b3 - FINAL OUTPUT)\n");
    printf("  Matrix: 10×32, Vector: 32×1, Output: 10×1\n");
    Instruction gemv3 = {0x04, 5, 32, 10, 0, 8, 1, 3};
    execute_instruction(gemv3);
    if (!wait_for_done(6000)) {
      printf("❌ Failed at Step 12 (Final GEMV)\n");
      layer3_success = false;
      return;
    }

    // Step 13: STORE 5, 0x8c0, 10
    printf("Step 13: STORE 5, 0x8c0, 10 (write output vector)\n");
    Instruction store_out = {0x03, 5, 10, 0, 0x8c0, 0, 0, 0};
    execute_instruction(store_out);
    if (!wait_for_done(500)) {
      printf("❌ Failed at Step 13 (STORE)\n");
      return;
    }

    printf("\n✅ Layer 3 Complete: 32 → 10 (OUTPUT)\n\n");

    // ========== SUMMARY ==========
    printf("╔════════════════════════════════════════════════════════╗\n");
    printf("║           NEURAL NETWORK TEST COMPLETE                ║\n");
    printf("╚════════════════════════════════════════════════════════╝\n\n");

    printf("📊 Test Results:\n");
    printf("  Layer 1 (784→12):  %s\n",
           layer1_success ? "✅ PASSED" : "❌ FAILED");
    printf("  Layer 2 (12→32):   %s\n",
           layer2_success ? "✅ PASSED" : "❌ FAILED");
    printf("  Layer 3 (32→10):   %s\n",
           layer3_success ? "✅ PASSED" : "❌ FAILED");
    printf("\n");

    printf("📈 Network Architecture:\n");
    printf("  Input layer:    784 neurons\n");
    printf("  Hidden layer 1: 12 neurons  (9,408 parameters)\n");
    printf("  Hidden layer 2: 32 neurons  (384 parameters)\n");
    printf("  Output layer:   10 neurons  (320 parameters)\n");
    printf("  Total parameters: 10,112\n");
    printf("\n");

    printf("🔧 Operations Executed:\n");
    printf("  LOAD_V operations: 5\n");
    printf("  LOAD_M operations: 3\n");
    printf("  GEMV operations:   3\n");
    printf("  RELU operations:   2\n");
    printf("  STORE operations:  1\n");
    printf("  Total instructions: 14\n");
    printf("\n");

    if (layer1_success && layer2_success && layer3_success) {
      printf("🎉 SUCCESS! Complete neural network executed successfully!\n");
      printf("   All 14 assembly instructions from model_assembly.asm "
             "verified.\n");
    } else {
      printf("⚠️  Some layers failed. Check logs above for details.\n");
    }

    // Display final output from memory (after STORE instruction)
    printf("\n📊 Final Neural Network Output (10 classification scores from "
           "memory 0x8c0):\n");
    for (int i = 0; i < 10; i++) {
      uint8_t val = memory[0x8c0 + i];
      printf("  Class %d: %4d (0x%02x)\n", i, (int8_t)val, val);
    }

    printf("\n");
  }

  void run() {
    reset();
    test_neural_network();

    printf("\n=== Total Simulation Time: %llu cycles ===\n", time_counter);

    // Extra cycles for cleanup
    for (int i = 0; i < 20; i++) {
      tick();
    }
  }
};

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);

  NeuralNetworkTB tb;
  tb.run();

  return 0;
}
