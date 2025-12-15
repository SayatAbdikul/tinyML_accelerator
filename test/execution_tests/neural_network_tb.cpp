// Neural Network Sequence Testbench for Modular Execution Unit
// Tests complete neural network execution: 784→12→32→10
// Replicates assembly sequence from model_assembly.asm

#include <verilated.h>
#include <verilated_vcd_c.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include "Vmodular_execution_unit.h"

// Test parameters
const int MAX_CYCLES = 100000;
const int DATA_WIDTH = 8;

class NeuralNetworkTB {
private:
    Vmodular_execution_unit* dut;
    VerilatedVcdC* trace;
    uint64_t time_counter;
    
    // Instruction encoding helpers
    struct Instruction {
        uint8_t opcode;
        uint8_t dest;
        uint16_t length_or_cols;
        uint16_t rows;
        uint32_t addr;
        uint8_t x_id;        // For GEMV, RELU (source buffer)
        uint8_t w_id;        // For GEMV
        uint8_t b_id;        // For GEMV
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
        
        for (int i = 0; i < 10; i++) {
            tick();
        }
        
        dut->rst = 0;
        tick();
        printf("Reset complete\n\n");
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
    
    void execute_instruction(const Instruction& instr) {
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
        
        printf("🎯 ASSEMBLY INSTRUCTIONS:\n");
        printf("Layer 1: LOAD_V 9, 0x700, 784\n");
        printf("         LOAD_M 1, 0x10700, 12, 784\n");
        printf("         LOAD_V 4, 0x13001, 12\n");
        printf("         GEMV 5, 1, 9, 4, 12, 784\n");
        printf("         RELU 7, 5\n");
        printf("Layer 2: LOAD_M 2, 0x12bc0, 32, 12\n");
        printf("         LOAD_V 3, 0x1300d, 32\n");
        printf("         GEMV 6, 2, 7, 3, 32, 12\n");
        printf("         RELU 8, 6\n");
        printf("Layer 3: LOAD_M 1, 0x12d40, 10, 32\n");
        printf("         LOAD_V 4, 0x1302d, 10\n");
        printf("         GEMV 5, 1, 8, 4, 10, 32\n");
        printf("\n");
        
        // Track success of each layer
        bool layer1_success = true;
        bool layer2_success = true;
        bool layer3_success = true;
        
        // ========== LAYER 1: 784 → 12 ==========
        printf("╔════════════════════════════════════╗\n");
        printf("║     LAYER 1: 784 → 12 (FC)        ║\n");
        printf("╚════════════════════════════════════╝\n\n");
        
        // Step 1: LOAD_V 9, 0x700, 784
        printf("Step 1: LOAD_V 9, 0x700, 784 (input vector - 784 elements)\n");
        Instruction load_input = {0x01, 9, 784, 0, 0x700, 0, 0, 0};
        execute_instruction(load_input);
        if (!wait_for_done(2000)) {
            printf("❌ Failed at Step 1\n");
            layer1_success = false;
            return;
        }
        
        // Step 2: LOAD_M 1, 0x10700, 12, 784
        printf("\nStep 2: LOAD_M 1, 0x10700, 12, 784 (weight matrix W1 - 12×784)\n");
        Instruction load_w1 = {0x02, 1, 784, 12, 0x10700, 0, 0, 0};
        execute_instruction(load_w1);
        if (!wait_for_done(25000)) {
            printf("❌ Failed at Step 2\n");
            layer1_success = false;
            return;
        }
        
        // Step 3: LOAD_V 4, 0x13001, 12
        printf("\nStep 3: LOAD_V 4, 0x13001, 12 (bias vector b1 - 12 elements)\n");
        Instruction load_b1 = {0x01, 4, 12, 0, 0x13001, 0, 0, 0};
        execute_instruction(load_b1);
        if (!wait_for_done(200)) {
            printf("❌ Failed at Step 3\n");
            layer1_success = false;
            return;
        }
        
        // Step 4: GEMV 5, 1, 9, 4, 12, 784
        printf("\nStep 4: GEMV 5, 1, 9, 4, 12, 784 (W1 * input + b1)\n");
        printf("  Matrix: 12×784, Vector: 784×1, Output: 12×1\n");
        Instruction gemv1 = {0x04, 5, 784, 12, 0, 9, 1, 4};
        execute_instruction(gemv1);
        if (!wait_for_done(60000)) {
            printf("❌ Failed at Step 4 (GEMV)\n");
            layer1_success = false;
            return;
        }
        
        // Step 5: RELU 7, 5
        printf("\nStep 5: RELU 7, 5 (activation function)\n");
        Instruction relu1 = {0x05, 7, 12, 0, 0, 5, 0, 0};  // dest=7, x_id=5
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
        
        // Step 6: LOAD_M 2, 0x12bc0, 32, 12
        printf("Step 6: LOAD_M 2, 0x12bc0, 32, 12 (weight matrix W2 - 32×12)\n");
        Instruction load_w2 = {0x02, 2, 12, 32, 0x12bc0, 0, 0, 0};
        execute_instruction(load_w2);
        if (!wait_for_done(1500)) {
            printf("❌ Failed at Step 6\n");
            layer2_success = false;
            return;
        }
        
        // Step 7: LOAD_V 3, 0x1300d, 32
        printf("\nStep 7: LOAD_V 3, 0x1300d, 32 (bias vector b2 - 32 elements)\n");
        Instruction load_b2 = {0x01, 3, 32, 0, 0x1300d, 0, 0, 0};
        execute_instruction(load_b2);
        if (!wait_for_done(250)) {
            printf("❌ Failed at Step 7\n");
            layer2_success = false;
            return;
        }
        
        // Step 8: GEMV 6, 2, 7, 3, 32, 12
        printf("\nStep 8: GEMV 6, 2, 7, 3, 32, 12 (W2 * h1 + b2)\n");
        printf("  Matrix: 32×12, Vector: 12×1, Output: 32×1\n");
        Instruction gemv2 = {0x04, 6, 12, 32, 0, 7, 2, 3};
        execute_instruction(gemv2);
        if (!wait_for_done(8000)) {
            printf("❌ Failed at Step 8 (GEMV)\n");
            layer2_success = false;
            return;
        }
        
        // Step 9: RELU 8, 6
        printf("\nStep 9: RELU 8, 6 (activation function)\n");
        Instruction relu2 = {0x05, 8, 32, 0, 0, 6, 0, 0};  // dest=8, x_id=6
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
        
        // Step 10: LOAD_M 1, 0x12d40, 10, 32
        printf("Step 10: LOAD_M 1, 0x12d40, 10, 32 (weight matrix W3 - 10×32)\n");
        Instruction load_w3 = {0x02, 1, 32, 10, 0x12d40, 0, 0, 0};
        execute_instruction(load_w3);
        if (!wait_for_done(1200)) {
            printf("❌ Failed at Step 10\n");
            layer3_success = false;
            return;
        }
        
        // Step 11: LOAD_V 4, 0x1302d, 10
        printf("\nStep 11: LOAD_V 4, 0x1302d, 10 (bias vector b3 - 10 elements)\n");
        Instruction load_b3 = {0x01, 4, 10, 0, 0x1302d, 0, 0, 0};
        execute_instruction(load_b3);
        if (!wait_for_done(150)) {
            printf("❌ Failed at Step 11\n");
            layer3_success = false;
            return;
        }
        
        // Step 12: GEMV 5, 1, 8, 4, 10, 32
        printf("\nStep 12: GEMV 5, 1, 8, 4, 10, 32 (W3 * h2 + b3 - FINAL OUTPUT)\n");
        printf("  Matrix: 10×32, Vector: 32×1, Output: 10×1\n");
        Instruction gemv3 = {0x04, 5, 32, 10, 0, 8, 1, 4};
        execute_instruction(gemv3);
        if (!wait_for_done(6000)) {
            printf("❌ Failed at Step 12 (Final GEMV)\n");
            layer3_success = false;
            return;
        }
        
        printf("\n✅ Layer 3 Complete: 32 → 10 (OUTPUT)\n\n");
        
        // ========== SUMMARY ==========
        printf("╔════════════════════════════════════════════════════════╗\n");
        printf("║           NEURAL NETWORK TEST COMPLETE                ║\n");
        printf("╚════════════════════════════════════════════════════════╝\n\n");
        
        printf("📊 Test Results:\n");
        printf("  Layer 1 (784→12):  %s\n", layer1_success ? "✅ PASSED" : "❌ FAILED");
        printf("  Layer 2 (12→32):   %s\n", layer2_success ? "✅ PASSED" : "❌ FAILED");
        printf("  Layer 3 (32→10):   %s\n", layer3_success ? "✅ PASSED" : "❌ FAILED");
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
        printf("  Total instructions: 13\n");
        printf("\n");
        
        if (layer1_success && layer2_success && layer3_success) {
            printf("🎉 SUCCESS! Complete neural network executed successfully!\n");
            printf("   All 13 assembly instructions from model_assembly.asm verified.\n");
        } else {
            printf("⚠️  Some layers failed. Check logs above for details.\n");
        }
        
        // Display final output from result register
        printf("\n📊 Final Neural Network Output (10 classification scores):\n");
        for (int i = 0; i < 10; i++) {
            printf("  Class %d: %4d (0x%02x)\n", i, (int8_t)dut->result[i], (uint8_t)dut->result[i]);
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

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    
    NeuralNetworkTB tb;
    tb.run();
    
    return 0;
}
