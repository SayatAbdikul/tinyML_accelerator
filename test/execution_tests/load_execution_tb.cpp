// Testbench for Load Execution Module
// Tests LOAD_V and LOAD_M operations with buffer controller integration

#include <verilated.h>
#include <verilated_vcd_c.h>
#include <cstdio>
#include <cstdint>
#include "Vload_execution.h"

const int MAX_CYCLES = 1000;
const int TILE_ELEMS = 32;

class LoadExecutionTB {
private:
    Vload_execution* dut;
    VerilatedVcdC* trace;
    uint64_t time_counter;
    
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
        
        for (int i = 0; i < 5; i++) tick();
        
        dut->rst = 0;
        tick();
        printf("Reset released\n");
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
        
        dut->opcode = 0x01;  // LOAD_V
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
                printf("  Tile %d written to buffer %d\n",
                       tiles_written, dut->vec_write_buffer_id);
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
        
        dut->opcode = 0x02;  // LOAD_M
        dut->dest_buffer_id = 2;
        dut->length_or_cols = 16;  // columns
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
                printf("  Matrix tile %d written to buffer %d\n",
                       tiles_written, dut->mat_write_buffer_id);
            }
            tick();
            cycles++;
        }
        
        if (dut->done) {
            printf("✅ LOAD_M completed: %d tiles written in %d cycles\n",
                   tiles_written, cycles);
            
            // Verify expected tile count (8*16 = 128 elements = 4 tiles)
            int total_elements = 8 * 16;
            int expected_tiles = (total_elements + TILE_ELEMS - 1) / TILE_ELEMS;
            if (tiles_written == expected_tiles) {
                printf("✅ Correct number of tiles written\n");
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
        
        dut->opcode = 0x10;  // Invalid
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
            printf("✅ Single element load: %d tile in %d cycles\n", tiles_written, cycles);
        } else {
            printf("❌ Expected %d tile, got %d\n", expected_tiles, tiles_written);
        }
        
        tick();
        tick();
    }
    
    void test_exact_tile_boundary() {
        printf("\n--- Test Exact Tile Boundary (32 elements) ---\n");
        
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
        
        // 7*13 = 91 elements, ceil(91/32) = 3 tiles
        int expected_tiles = (7 * 13 + TILE_ELEMS - 1) / TILE_ELEMS;
        if (dut->done && tiles_written == expected_tiles) {
            printf("✅ Non-aligned matrix: %d tiles for 91 elements\n", tiles_written);
        } else {
            printf("❌ Expected %d tiles, got %d\n", expected_tiles, tiles_written);
        }
        
        tick();
        tick();
    }
    
    void test_back_to_back_loads() {
        printf("\n--- Test Back-to-Back Loads ---\n");
        
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
    
    void test_buffer_id_verification() {
        printf("\n--- Test Buffer ID Verification ---\n");
        
        bool all_passed = true;
        
        // Test max buffer ID (31)
        dut->opcode = 0x01;
        dut->dest_buffer_id = 31;
        dut->length_or_cols = 10;
        dut->addr = 0x8000;
        dut->start = 1;
        tick();
        dut->start = 0;
        
        int cycles = 0;
        uint8_t observed_buffer_id = 0;
        while (!dut->done && cycles < MAX_CYCLES) {
            if (dut->vec_write_enable) {
                observed_buffer_id = dut->vec_write_buffer_id;
            }
            tick();
            cycles++;
        }
        
        if (observed_buffer_id == 31) {
            printf("  ✅ Buffer ID 31 correctly used\n");
        } else {
            printf("  ❌ Expected buffer 31, got %d\n", observed_buffer_id);
            all_passed = false;
        }
        
        tick();
        tick();
        
        // Test buffer ID 0
        dut->opcode = 0x01;
        dut->dest_buffer_id = 0;
        dut->length_or_cols = 10;
        dut->addr = 0x9000;
        dut->start = 1;
        tick();
        dut->start = 0;
        
        cycles = 0;
        observed_buffer_id = 255; // Invalid value
        while (!dut->done && cycles < MAX_CYCLES) {
            if (dut->vec_write_enable) {
                observed_buffer_id = dut->vec_write_buffer_id;
            }
            tick();
            cycles++;
        }
        
        if (observed_buffer_id == 0) {
            printf("  ✅ Buffer ID 0 correctly used\n");
        } else {
            printf("  ❌ Expected buffer 0, got %d\n", observed_buffer_id);
            all_passed = false;
        }
        
        if (all_passed) {
            printf("✅ All buffer IDs verified correctly\n");
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
        test_buffer_id_verification();
        
        printf("\n=== Load Execution Tests Complete ===\n");
        printf("Total simulation time: %llu cycles\n", time_counter);
    }
};

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    LoadExecutionTB tb;
    tb.run_all_tests();
    return 0;
}
