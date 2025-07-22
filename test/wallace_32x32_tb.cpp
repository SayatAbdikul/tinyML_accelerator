#include "Vwallace_32x32.h"
#include "verilated.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <iomanip>

vluint64_t main_time = 0;
double sc_time_stamp() { return main_time; }

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vwallace_32x32* dut = new Vwallace_32x32;

    // Reset (if needed)
    // dut->reset = 1;
    // for (int i = 0; i < 2; ++i) {
    //     dut->clk = !dut->clk;
    //     dut->eval();
    //     main_time++;
    // }
    // dut->reset = 0;

    std::srand(42);
    int errors = 0;
    for (int i = 0; i < 20; ++i) {
        uint32_t a = std::rand();
        uint32_t b = std::rand();
        uint64_t expected = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);

        dut->a = a;
        dut->b = b;
        dut->eval();
        uint64_t actual = ((uint64_t)dut->prod);

        std::cout << "a: " << std::setw(10) << a
                  << " b: " << std::setw(10) << b
                  << " | Expected: " << std::setw(20) << expected
                  << " | Actual: " << std::setw(20) << actual;

        if (actual == expected) {
            std::cout << " [PASS]\n";
        } else {
            std::cout << " [FAIL]\n";
            errors++;
        }
    }

    delete dut;
    std::cout << "Test completed. Errors: " << errors << std::endl;
    return errors ? 1 : 0;
}