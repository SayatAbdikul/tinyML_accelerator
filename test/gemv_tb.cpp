#include "Vgemv.h"
#include "verilated.h"
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

void tick() {
    dut->clk = 0; dut->eval(); main_time++;
    dut->clk = 1; dut->eval(); main_time++;
}
std::vector<int8_t> quantize_int32_to_int8(const std::vector<int32_t>& x_int32, float scale, int32_t zero_point) {
    std::vector<int8_t> result;
    result.reserve(x_int32.size());

    for (int32_t x : x_int32) {
        // 1. Scale down to quantization grid
        float x_scaled = static_cast<float>(x) / scale;

        // 2. Apply zero-point
        float x_quantized = x_scaled + static_cast<float>(zero_point);

        // 3. Round to nearest
        int32_t x_rounded = static_cast<int32_t>(std::round(x_quantized));

        // 4. Clip to int8 range [-128, 127]
        int8_t x_clipped = static_cast<int8_t>(std::min(std::max(x_rounded, -128), 127));

        result.push_back(x_clipped);
    }

    return result;
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vgemv;

    std::srand(std::time(0));

    int w[ROWS][COLS], x[COLS], bias[ROWS], y_expected[ROWS] = {0};

    // Random weights and input vector
    for (int i = 0; i < ROWS; i++) {
        bias[i] = rand() % 4;
        for (int j = 0; j < COLS; j++) {
            w[i][j] = rand() % 4;
        }
    }

    for (int j = 0; j < COLS; j++) {
        x[j] = rand() % 4;
    }

    // Compute expected result in software
    std::vector<int32_t> y_int32;
    int32_t max_abs = 0;

    for (int i = 0; i < ROWS; i++) {
        int sum = 0;
        for (int j = 0; j < COLS; j++) {
            sum += w[i][j] * x[j];
        }
        int y = sum + bias[i];
        y_expected[i] = y;
        y_int32.push_back(y);
        max_abs = std::max(max_abs, std::abs(y));
    }

    // Compute scale
    float scale = static_cast<float>(max_abs) / 127.0f;
    int32_t zero_point = 0; // symmetric

    // Quantize software results
    std::vector<int8_t> y_quantized = quantize_int32_to_int8(y_int32, scale, zero_point);

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
    tick();
    dut->rst = 0;
    tick();

    // Run simulation
    std::cout << "Running GEMV..." << std::endl;
    bool done = false;
    for (int t = 0; t < 10000; t++) {
        tick();
        if (dut->done) {
            done = true;
            tick();
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

    if (errors == 0) {
        std::cout << "✅ GEMV passed successfully!" << std::endl;
    } else {
        std::cerr << "❌ GEMV failed with " << errors << " errors." << std::endl;
    }

    dut->final();
    delete dut;
    return 0;
}
