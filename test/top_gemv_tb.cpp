#include "Vtop_gemv.h"
#include "verilated.h"
#include "verilated_vcd_c.h"
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <type_traits>
#include <cstring>

#define ROWS 64
#define COLS 92
#define TILE 32
#define DWIDTH 8

// Helper to drive w_tile_row_in for both unpacked (CData[TILE]) and packed (WData[...]) ports


Vtop_gemv *dut;

vluint64_t main_time = 0;
double sc_time_stamp() { return main_time; }

void tick(VerilatedVcdC* tfp) {
    dut->clk = 0; dut->eval(); tfp->dump(main_time++);
    dut->clk = 1; dut->eval(); tfp->dump(main_time++);
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vtop_gemv;
    VerilatedVcdC* tfp = new VerilatedVcdC;
    Verilated::traceEverOn(true);
    dut->trace(tfp, 99);
    tfp->open("dump.vcd");

    std::srand(std::time(0));

    int8_t w[ROWS][COLS], x[COLS], bias[ROWS]= {0};
    int y_expected[ROWS] = {0};

    // Random weights and input vector
    for (int i = 0; i < ROWS; i++) {
        bias[i] = rand() % 256 - 128; // Random bias in range [-128, 127]
        for (int j = 0; j < COLS; j++) {
            w[i][j] = rand() % 256 - 128;
        }
    }

    for (int j = 0; j < COLS; j++) {
        x[j] = rand() % 256 - 128;
    }

    // Compute expected result in software
    std::vector<int32_t> y_int32;

    for (int i = 0; i < ROWS; i++) {
        int sum = 0;
        for (int j = 0; j < COLS; j++) {
            sum += w[i][j] * x[j];
        }
        int y = sum + bias[i];
        //std::cout << "Row " << i << ": sum = " << sum << ", bias = " << static_cast<int>(bias[i])
        //          << ", y = " << y << "\n";
        y_expected[i] = y;
        y_int32.push_back(y);
    }
    // 1. Compute max absolute value (same as testbench)
    int32_t max_abs = 0;
    for (int y_val : y_expected) {
        int32_t abs_val = std::abs(y_val);
        if (abs_val > max_abs) max_abs = abs_val;
    }

    // 2. Avoid division by zero
    if (max_abs == 0) max_abs = 1;

    // 3. Compute reciprocal scale AS HARDWARE DOES
    uint32_t reciprocal_scale = (static_cast<uint32_t>(127) << 24) / max_abs;

    // 4. Quantize using HARDWARE'S METHOD
    std::vector<int8_t> y_quantized;
    for (int y_val : y_expected) {
        int64_t product = static_cast<int64_t>(y_val) * reciprocal_scale;
        int32_t quantized = static_cast<int32_t> ((product + (1 << 23)) >> 24);  // Truncate lower 24 bits
        
        // Clip to int8 range
        if (quantized > 127) quantized = 127;
        if (quantized < -128) quantized = -128;
        
        y_quantized.push_back(static_cast<int8_t>(quantized));
    }
    std::cout<<"The software reciprocal scale is "<<2130706432 / max_abs<<" with the max abs "<<max_abs<<"\n";
    int32_t zero_point = 0; // symmetric
    
    // Assign inputs to DUT

    for (int i = 0; i < ROWS; i++) {
        dut->bias[i] = bias[i];
    }

    for (int j = 0; j < COLS; j++) {
        dut->x[j] = x[j];
    }
    // Apply reset
    dut->rst = 1;
    tick(tfp);
    dut->rst = 0;
    tick(tfp);
    
    // Initialize weights in tiles
    for(int i=0; i<TILE; i++) dut->w_tile_row_in[i] = 0; // Initialize to zero
    dut->w_valid = 0;

    std::cout<<"Testbench started with "<<ROWS<<" rows and "<<COLS<<" columns.\n";
    // start the eval
    dut->rows = ROWS;
    dut->cols = COLS;
    dut->start = 1;
    tick(tfp);
    dut->start = 0;

    // Calculate tiles per row
    int tiles_per_row = (COLS + TILE - 1) / TILE;

    int tiles_sent = 0;
    int total_tiles = ROWS * tiles_per_row;
    int wx = 0, wy = 0, t_cnt = 0;
    for(int i=0; i<TILE; i++){
        dut->w_tile_row_in[i] = 0; // Initialize to zero
    }
    
    for(int i=0; i<ROWS; i++){
        t_cnt = 0;
        for(int j=0; j<COLS; j++){
            dut->w_tile_row_in[t_cnt] = w[i][j];
            // if(t_cnt == TILE - 1) {
            //     std::cout<<"Software w_tile_row_in[31] = "<<static_cast<int>(dut->w_tile_row_in[t_cnt])<<"\n";
            // }
            t_cnt++;
            if(t_cnt == TILE) {
                // Wait for w_ready
                while (dut->w_ready == 0) {
                    tick(tfp);
                }
                // Send tile
                dut->w_valid = 1;
                tick(tfp);  // Clock to latch on posedge
                dut->w_valid = 0;

                // Wait for tile_done
                while (dut->tile_done == 0 && dut->done == 0) {
                    tick(tfp);
                }

                tiles_sent++;
                t_cnt = 0; // Reset tile counter
                // Clear for next tile
                for (int k = 0; k < TILE; ++k) dut->w_tile_row_in[k] = 0;
            }
        }
        // Flush a partial tile at end of row (zero-padded)
        if (t_cnt != 0) {
            for (int k = t_cnt; k < TILE; ++k) dut->w_tile_row_in[k] = 0;
            while (dut->w_ready == 0) {
                tick(tfp);
            }
            dut->w_valid = 1;
            tick(tfp);
            dut->w_valid = 0;
            while (dut->tile_done == 0 && dut->done == 0) {
                tick(tfp);
            }
            tiles_sent++;
            t_cnt = 0;
            for (int k = 0; k < TILE; ++k) dut->w_tile_row_in[k] = 0;
        }
    }

    // Wait for final done (quantization, etc.)
    while (dut->done == 0) {
        tick(tfp);
    }

    // Verify quantized output
    int errors = 0;
    for (int i = 0; i < ROWS; i++) {
        int y_hw = static_cast<int8_t>(dut->y[i]); // assume y is int8_t
        int y_sw = y_quantized[i];

        if (y_hw != y_sw) {
            std::cerr << "Mismatch at row " << i
                      << ": expected=" << static_cast<int>(y_sw)
                      << ", got=" << static_cast<int>(y_hw) << std::endl;
            errors++;
        }
    }
    std::cout<<"The clock cycles passed: "<<main_time/2<<"\n";
    if (errors == 0) {
        std::cout << "✅ GEMV passed successfully!" << std::endl;
    } else {
        std::cerr << "❌ GEMV failed with " << errors << " errors." << std::endl;
    }

    dut->final();
    tfp->close();
    delete tfp;
    delete dut;
    return 0;
}

