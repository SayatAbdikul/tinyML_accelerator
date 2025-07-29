#include "Vwallace_32x32.h"
#include "verilated.h"
#include "verilated_vcd_c.h"
#include <iostream>
#include <random>
#include <cassert>
#include <iomanip>

vluint64_t main_time = 0;
const int CLOCK_PERIOD = 10;

double sc_time_stamp() { return main_time; }

int main(int argc, char **argv, char **env) {
    Verilated::commandArgs(argc, argv);
    Vwallace_32x32* top = new Vwallace_32x32;
    VerilatedVcdC* tfp = new VerilatedVcdC;
    Verilated::traceEverOn(true);
    top->trace(tfp, 99);
    tfp->open("dump.vcd");

    // Constants
    const int PIPELINE_STAGES = 3;
    const int TEST_CYCLES = 100;
    const int TOTAL_HALF_CYCLES = 2 * (TEST_CYCLES + PIPELINE_STAGES + 20);  // Increased simulation length

    // Clocking and reset
    top->clk = 0;
    top->rst_n = 0;
    top->valid_in = 0;

    // Initialize random number generator
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int32_t> dist(-2147483648, 2147483647);
    
    // Test vectors: [a, b, expected]
    int64_t test_vectors[TEST_CYCLES][3];

    // Generate test vectors with corner cases
    test_vectors[0][0] = 0; test_vectors[0][1] = 0;
    test_vectors[1][0] = 1; test_vectors[1][1] = 1;
    test_vectors[2][0] = -1; test_vectors[2][1] = -1;
    test_vectors[3][0] = 2147483647; test_vectors[3][1] = 2147483647;
    test_vectors[4][0] = -2147483648; test_vectors[4][1] = 1;
    test_vectors[5][0] = -2147483648; test_vectors[5][1] = -1;
    test_vectors[6][0] = 123456789; test_vectors[6][1] = 987654321;
    test_vectors[7][0] = -123456789; test_vectors[7][1] = 987654321;
    test_vectors[8][0] = 123456789; test_vectors[8][1] = -987654321;
    test_vectors[9][0] = -123456789; test_vectors[9][1] = -987654321;

    // Calculate expected products for all vectors
    for (int i = 0; i < TEST_CYCLES; i++) {
        if (i >= 10) {
            // Generate random values for remaining vectors
            test_vectors[i][0] = dist(rng);
            test_vectors[i][1] = dist(rng);
        }
        // Calculate expected product (correctly handles 64-bit multiplication)
        test_vectors[i][2] = static_cast<int64_t>(test_vectors[i][0]) * 
                             static_cast<int64_t>(test_vectors[i][1]);
    }

    // Reset sequence
    for (int i = 0; i < 5; i++) {
        top->clk = !top->clk;
        top->rst_n = (i > 3) ? 1 : 0;
        top->eval();
        tfp->dump(main_time);
        main_time += CLOCK_PERIOD / 2;
    }

    int input_ptr = 0;
    int output_ptr = 0;
    int error_count = 0;

    // Main simulation loop - using half-cycles
    for (int half_cycle = 0; half_cycle < TOTAL_HALF_CYCLES; half_cycle++) {
        // Toggle clock
        top->clk = !top->clk;
        
        // Drive inputs on falling edge
        if (top->clk == 0) {
            if (input_ptr < TEST_CYCLES) {
                top->a = test_vectors[input_ptr][0];
                top->b = test_vectors[input_ptr][1];
                top->valid_in = 1;
                input_ptr++;
            } else {
                top->valid_in = 0;
            }
        }

        // Evaluate design
        top->eval();
        tfp->dump(main_time);
        main_time += CLOCK_PERIOD / 2;

        // Check outputs on rising edge when valid_out is high
        if (top->clk == 1 && top->valid_out) {
            if (output_ptr < TEST_CYCLES) {
                int64_t expected = test_vectors[output_ptr][2];
                uint64_t raw_actual = top->prod;
                int64_t actual;
                memcpy(&actual, &raw_actual, sizeof(actual));
                
                if (actual != expected) {
                    std::cerr << "Half-cycle " << half_cycle << " (Output #" << output_ptr << "):\n"
                              << "  a = 0x" << std::hex << std::setw(8) << std::setfill('0') 
                              << test_vectors[output_ptr][0] << " (" << std::dec 
                              << test_vectors[output_ptr][0] << ")\n"
                              << "  b = 0x" << std::hex << std::setw(8) << std::setfill('0') 
                              << test_vectors[output_ptr][1] << " (" << std::dec 
                              << test_vectors[output_ptr][1] << ")\n"
                              << "  Expected: 0x" << std::hex << std::setw(16) << std::setfill('0') 
                              << expected << " (" << std::dec << expected << ")\n"
                              << "  Actual:   0x" << std::hex << std::setw(16) << std::setfill('0') 
                              << actual << " (" << std::dec << actual << ")\n"
                              << "  Difference: " << (expected - actual) << "\n";
                    error_count++;
                }
                output_ptr++;
            }
        }
    }

    // Summary
    std::cout << "Test completed: ";
    std::cout << TEST_CYCLES << " vectors, ";
    std::cout << error_count << " errors\n";
    std::cout << "Captured outputs: " << output_ptr << "/" << TEST_CYCLES << "\n";

    // Cleanup
    tfp->close();
    delete tfp;
    top->final();
    delete top;
    
    return error_count ? 1 : 0;
}