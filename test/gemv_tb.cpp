#include "Vgemv.h"
#include "verilated.h"
#include "verilated_vcd_c.h"
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cmath>


#define ROWS 128
#define COLS 128
#define TILE 8
#define DWIDTH 8

Vgemv *dut;

vluint64_t main_time = 0;
double sc_time_stamp() { return main_time; }

void tick(VerilatedVcdC* tfp) {
    dut->clk = 0; dut->eval(); tfp->dump(main_time++);
    dut->clk = 1; dut->eval(); tfp->dump(main_time++);
}


int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vgemv;
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
        y_expected[i] = y;
        y_int32.push_back(y);
        //std::cout << "Row " << i << ": Expected y = " << y << std::endl;
    }
    // 1. Compute max absolute value (same as testbench)
    int32_t max_abs = 0;
    for (int y_val : y_expected) {
        int32_t abs_val = std::abs(y_val);
        // std::cout<<"The sw abs_val is "<<abs_val<<"\n";
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

    // Quantize software results

    // Assign inputs to DUT
    for (int i = 0; i < ROWS; i++) {
        dut->bias[i] = bias[i];
        for (int j = 0; j < COLS; j++) {
            dut->w[i][j] = w[i][j];
        }
    }

    for (int j = 0; j < COLS; j++) {
        dut->x[j] = x[j];
    }

    // Apply reset
    dut->rst = 1;
    tick(tfp);
    dut->rst = 0;
    tick(tfp);

    // Run simulation
    std::cout << "Running GEMV..." << std::endl;
    bool done = false;
    for (int t = 0; t < 10000; t++) {
        tick(tfp);
        if (dut->done) {
            done = true;
            tick(tfp);
            break;
        }
    }

    if (!done) {
        std::cerr << "ERROR: Timeout waiting for done signal\n";
        return 1;
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

