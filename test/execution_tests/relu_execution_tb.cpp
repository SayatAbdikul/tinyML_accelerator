// Testbench for ReLU Execution Module
// Tests ReLU activation with proper source/destination buffer handling

#include <verilated.h>
#include <verilated_vcd_c.h>
#include <cstdio>
#include <cstdint>
#include "Vrelu_execution.h"

const int MAX_CYCLES = 500;
const int TILE_ELEMS = 32;

class ReluExecutionTB {
private:
    Vrelu_execution* dut;
    VerilatedVcdC* trace;
    uint64_t time_counter;
    
    // Simulated buffer for testing
    int8_t test_buffer[TILE_ELEMS];
    
public:
    ReluExecutionTB() {
        dut = new Vrelu_execution;
        Verilated::traceEverOn(true);
        trace = new VerilatedVcdC;
        dut->trace(trace, 99);
        trace->open("relu_execution.vcd");
        time_counter = 0;
        
        // Initialize test buffer with mix of positive and negative values
        for (int i = 0; i < TILE_ELEMS; i++) {
            test_buffer[i] = (i % 2 == 0) ? (i - 16) : (i + 10);
        }
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
        
        for (int i = 0; i < 5; i++) tick();
        
        dut->rst = 0;
        tick();
        printf("Reset released\n");
    }
    
    void test_relu_single_tile() {
        printf("\n--- Test ReLU Single Tile ---\n");
        printf("Testing ReLU: buffer 5 -> buffer 10, length=32\n");
        
        // Start ReLU operation
        dut->dest_buffer_id = 10;
        dut->x_buffer_id = 5;  // Read from buffer 5
        dut->length = 32;
        dut->start = 1;
        
        tick();
        dut->start = 0;
        
        // Simulate buffer read response
        int read_count = 0;
        int write_count = 0;
        int8_t written_data[TILE_ELEMS];
        
        for (int cycle = 0; cycle < MAX_CYCLES && !dut->done; cycle++) {
            // When ReLU requests read, provide data after 1 cycle delay
            if (dut->vec_read_enable) {
                printf("  ReLU requesting read from buffer %d\n", dut->vec_read_buffer_id);
                
                // Verify it's reading from correct buffer (x_buffer_id, not dest)
                if (dut->vec_read_buffer_id != 5) {
                    printf("❌ ERROR: Reading from wrong buffer! Expected 5, got %d\n",
                           dut->vec_read_buffer_id);
                }
                
                read_count++;
                
                // Provide test data on next cycle
                tick();
                dut->vec_read_valid = 1;
                for (int i = 0; i < TILE_ELEMS; i++) {
                    dut->vec_read_tile[i] = test_buffer[i];
                }
                tick();
                dut->vec_read_valid = 0;
                cycle += 2;
            } else {
                tick();
            }
            
            // Capture written data
            if (dut->vec_write_enable) {
                printf("  ReLU writing to buffer %d\n", dut->vec_write_buffer_id);
                
                // Verify it's writing to correct buffer
                if (dut->vec_write_buffer_id != 10) {
                    printf("❌ ERROR: Writing to wrong buffer! Expected 10, got %d\n",
                           dut->vec_write_buffer_id);
                }
                
                for (int i = 0; i < TILE_ELEMS; i++) {
                    written_data[i] = (int8_t)dut->vec_write_tile[i];
                }
                write_count++;
            }
        }
        
        if (dut->done) {
            printf("✅ ReLU completed: %d reads, %d writes\n", read_count, write_count);
            
            // Verify ReLU correctness
            bool pass = true;
            for (int i = 0; i < TILE_ELEMS; i++) {
                int8_t expected = (test_buffer[i] < 0) ? 0 : test_buffer[i];
                if (written_data[i] != expected) {
                    printf("❌ ReLU error at [%d]: input=%d, expected=%d, got=%d\n",
                           i, test_buffer[i], expected, written_data[i]);
                    pass = false;
                }
            }
            
            if (pass) {
                printf("✅ ReLU computation correct\n");
            } else {
                printf("❌ ReLU computation has errors\n");
            }
            
            // Print sample results
            printf("\nSample ReLU results:\n");
            for (int i = 0; i < 8; i++) {
                printf("  [%2d] %4d -> %4d\n", i, test_buffer[i], written_data[i]);
            }
            
        } else {
            printf("❌ ReLU timed out\n");
        }
    }
    
    void test_relu_multiple_tiles() {
        printf("\n--- Test ReLU Multiple Tiles ---\n");
        printf("Testing ReLU: buffer 3 -> buffer 7, length=96 (3 tiles)\n");
        
        dut->dest_buffer_id = 7;
        dut->x_buffer_id = 3;
        dut->length = 96;
        dut->start = 1;
        
        tick();
        dut->start = 0;
        
        int tiles_processed = 0;
        
        for (int cycle = 0; cycle < MAX_CYCLES && !dut->done; cycle++) {
            if (dut->vec_read_enable) {
                // Provide data after delay
                tick();
                dut->vec_read_valid = 1;
                for (int i = 0; i < TILE_ELEMS; i++) {
                    dut->vec_read_tile[i] = (int8_t)((tiles_processed * TILE_ELEMS + i) - 48);
                }
                tick();
                dut->vec_read_valid = 0;
                cycle += 2;
            } else {
                tick();
            }
            
            if (dut->vec_write_enable) {
                tiles_processed++;
                printf("  Processed tile %d\n", tiles_processed);
            }
        }
        
        if (dut->done) {
            printf("✅ ReLU completed: %d tiles processed\n", tiles_processed);
            
            if (tiles_processed == 3) {
                printf("✅ Correct number of tiles\n");
            } else {
                printf("❌ Expected 3 tiles, got %d\n", tiles_processed);
            }
        } else {
            printf("❌ ReLU timed out\n");
        }
    }
    
    void run_all_tests() {
        reset();
        test_relu_single_tile();
        
        reset();
        test_relu_multiple_tiles();
        
        printf("\n=== ReLU Execution Tests Complete ===\n");
        printf("Total simulation time: %llu cycles\n", time_counter);
    }
};

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    ReluExecutionTB tb;
    tb.run_all_tests();
    return 0;
}
