#include "Vwallace_32x32.h"
#include "verilated.h"
#include <iostream>
#include <random>
#include <cassert>

vluint64_t main_time = 0;

double sc_time_stamp() { return main_time; }

int main(int argc, char **argv, char **env) {
    Verilated::commandArgs(argc, argv);

    Vwallace_32x32* top = new Vwallace_32x32;

    // Clocking and control
    const int cycles = 100;
    top->clk = 0;
    top->rst_n = 0;

    // Reset the design
    for (int i = 0; i < 5; ++i) {
        top->clk = !top->clk;
        top->eval();
        main_time++;
    }
    top->rst_n = 1;

    std::mt19937 rng(12345);  // Fixed seed for reproducibility
    std::uniform_int_distribution<uint32_t> dist(0, 0xFFFFFFFF);

    uint32_t a_vals[cycles] = {0};
    uint32_t b_vals[cycles] = {0};
    uint64_t expected[cycles] = {0};
    top->valid_in = 1;  // Initially valid

    // Drive inputs & simulate
    for (int i = 0; i < cycles + 3; ++i) {
        top->clk = 0;


        if(i < cycles) {
            a_vals[i] = dist(rng);
            b_vals[i] = dist(rng);
            top->a = a_vals[i];
            top->b = b_vals[i];
            expected[i] = (uint64_t)a_vals[i] * b_vals[i];
        } else {
            top->a = 0;
            top->b = 0;
            top->valid_in = 0;  // No more valid inputs
        }

        top->eval();
        top->clk = 1;
        top->eval();


        if (top->valid_out) {
            static int out_cycle = 0;
            std::cout << "[Cycle " << i << "] ";
            std::cout << "a = " << a_vals[out_cycle] << ", ";
            std::cout << "b = " << b_vals[out_cycle] << ", ";
            std::cout << "Expected = " << expected[out_cycle] << ", ";
            std::cout << "Got = " << top->prod;
            if (top->prod == expected[out_cycle]) {
                std::cout << " [PASS]\n";
            } else {
                std::cout << " [FAIL]\n";
                assert(false && "Mismatch in product!");
            }
            out_cycle++;
        }

        main_time++;
    }
    

    std::cout << "Testbench completed successfully.\n";

    top->final();
    delete top;
    return 0;
}
