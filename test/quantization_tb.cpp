#include "Vquantization.h"
#include "verilated.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <cstdlib> // For std::rand() and std::srand()

vluint64_t main_time = 0;
double sc_time_stamp() { return main_time; }

// Helper function to compute expected quantized value
int8_t compute_expected(int32_t input, uint32_t scale) {
    int64_t product = static_cast<int64_t>(input) * static_cast<int64_t>(scale);
    int64_t rounded = (product + (1 << 23)) >> 24;  // Round to nearest integer
    if (rounded > 127) return 127;
    if (rounded < -128) return -128;
    return static_cast<int8_t>(rounded);
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vquantization* dut = new Vquantization;

    // Initialize
    dut->clk = 0;
    dut->reset_n = 0;
    dut->start_calib = 0;
    dut->max_abs = 0;
    dut->data_in = 0;
    dut->data_valid = 0;

    // Reset sequence (2 cycles)
    for (int i = 0; i < 4; i++) {
        dut->clk = !dut->clk;
        dut->eval();
        main_time++;
    }
    dut->reset_n = 1;

    // Randomized test data
    std::vector<std::pair<uint32_t, std::vector<std::pair<int32_t, int8_t>>>> tests;
    for (int t = 0; t < 1000; ++t) {
        uint32_t max_abs = std::rand() % 1000000; // Random max_abs in [0, 999999]
        std::vector<std::pair<int32_t, int8_t>> vecs;
        for (int i = 0; i < 8; ++i) {
            int32_t input = (std::rand() % (2 * max_abs + 1)) - static_cast<int32_t>(max_abs); // [-max_abs, max_abs]
            uint32_t scale = (127UL << 24) / max_abs;
            int8_t expected = compute_expected(input, scale);
            vecs.push_back({input, expected});
        }
        tests.push_back({max_abs, vecs});
    }
    int errors = 0;
    for (auto& test : tests) {
        uint32_t max_abs = test.first;
        auto& vectors = test.second;

        // Start calibration
        //std::cout << "Starting calibration for max_abs=" << max_abs << "\n";
        dut->start_calib = 1;
        dut->max_abs = max_abs;
        
        // Toggle clock once
        dut->clk = 1; dut->eval(); main_time++;
        dut->clk = 0; dut->eval(); main_time++;
        dut->start_calib = 0;

        // Wait for calibration to complete
        int timeout = 0;
        while (!dut->calib_ready && timeout < 100) {
            dut->clk = 1; dut->eval(); main_time++;
            dut->clk = 0; dut->eval(); main_time++;
            timeout++;
        }
        
        if (!dut->calib_ready) {
            std::cerr << "Calibration timeout!\n";
            break;
        }

        // Send test vectors
        std::vector<int8_t> expected;
        int out_idx = 0;
        for(int i = 0; i < vectors.size() + 10 + vectors.size() * 3; i++){
            if(i < vectors.size()){
                dut->data_in = vectors[i].first;
                dut->data_valid = 1;
                

                
                expected.push_back(vectors[i].second);
            }
            if (dut->data_valid_out) {
                if (!expected.empty()) {
                    int8_t exp_val = expected.front();
                    int8_t act_val = dut->data_out;
                    
                    std::cout << "Input: " << std::setw(4) << vectors.front().first
                              << " | Expected: " << std::setw(4) << static_cast<int>(exp_val)
                              << " | Actual: " << std::setw(4) << static_cast<int>(act_val);
                    
                    if (exp_val == act_val) {
                        std::cout << " [PASS]\n";
                    } else {
                        std::cout << " [FAIL]\n";
                        errors++;
                    }
                    
                    // Remove this expected value
                    expected.erase(expected.begin());
                    vectors.erase(vectors.begin());
                }
            }
            // Toggle clock
                dut->clk = 1; dut->eval(); main_time++;
                dut->clk = 0; dut->eval(); main_time++;
        }

        dut->data_valid = 0;

        
        
        // Clear any remaining expected values
        while (!expected.empty()) {
            std::cerr << "Missing output for input: " << vectors.front().first << "\n";
            expected.erase(expected.begin());
            vectors.erase(vectors.begin());
        }
    }

    dut->final();
    delete dut;
    std::cout << "Tests completed, the number of errors: " << errors << "\n";
    return 0;
}