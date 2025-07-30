#include "Vgemv.h"
#include "verilated.h"
#include <iostream>
#include <cstdlib>
#include <ctime>

#define ROWS 128
#define COLS 128
#define TILE 8
#define DWIDTH 8

Vgemv *dut;

vluint64_t main_time = 0;
double sc_time_stamp() { return main_time; }

void tick() {
    dut->clk = 0; dut->eval(); main_time++;
    dut->clk = 1; dut->eval(); main_time++;
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vgemv;

    std::srand(std::time(0));

    // Prepare inputs BEFORE reset
    int w[ROWS][COLS], x[COLS], bias[ROWS], y_expected[ROWS] = {0};

    // Random weights and input vector
    for (int i = 0; i < ROWS; i++) {
        bias[i] = rand() % 400 - 200;
        for (int j = 0; j < COLS; j++) {
            w[i][j] = rand() % 400 - 200;
        }
    }

    for (int j = 0; j < COLS; j++) {
        x[j] = rand() % 400 - 200;
    }

    // Compute expected result in software
    for (int i = 0; i < ROWS; i++) {
        int sum = 0;
        for (int j = 0; j < COLS; j++) {
            sum += w[i][j] * x[j];
        }
        y_expected[i] = sum + bias[i];
    }

    // Assign values to DUT BEFORE starting simulation
    for (int i = 0; i < ROWS; i++) {
        dut->bias[i] = bias[i];
        for (int j = 0; j < COLS; j++) {
            dut->w[i][j] = w[i][j];
        }
    }

    for (int j = 0; j < COLS; j++) {
        dut->x[j] = x[j];
    }

    // Apply reset with inputs stable
    dut->rst = 1;
    tick();  // Apply reset
    dut->rst = 0;
    tick();  // Release reset

    // Run simulation
    std::cout << "Running GEMV..." << std::endl;
    bool done = false;
    for (int t = 0; t < 10000; t++) {
        tick();
        // std::cout << "The results of the GEMV operation at time " << t << ":\n";
        // for (int i = 0; i < ROWS; i++) {
        //     std::cout << "y[" << i << "] = " << static_cast<int>(dut->y[i]) << std::endl;
        // }
        if (dut->done) {  // DONE state = 4
            done = true;
            tick();
            break;
        }
    }

    if (!done) {
        std::cerr << "ERROR: Timeout waiting for done signal\n";
        return 1;
    }

    // Verify output
    int errors = 0;
    for (int i = 0; i < ROWS; i++) {
        int y_hw = static_cast<int8_t>(dut->y[i]);
        int y_sw = y_expected[i];
        y_sw = static_cast<int8_t>(y_sw); // getting lower 8 bits with sign
        if (y_hw != y_sw) {
            std::cerr << "Mismatch at row " << i
                      << ": expected=" << y_sw
                      << ", got=" << y_hw << std::endl;
            errors++;
        }
    }

    if (errors == 0) {
        std::cout << "✅ GEMV passed successfully!" << std::endl;
    } else {
        std::cerr << "❌ GEMV failed with " << errors << " errors." << std::endl;
    }

    dut->final();
    delete dut;
    return 0;
}
