#include "Vscale_calculator.h"
#include "verilated.h"
#include <iostream>
#include <iomanip>
#include <cstdint>

vluint64_t main_time = 0;
double sc_time_stamp() { return main_time; }

void tick(Vscale_calculator* dut) {
    dut->clk = 0; dut->eval(); main_time++;
    dut->clk = 1; dut->eval(); main_time++;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vscale_calculator* dut = new Vscale_calculator;

    // Initialize all inputs
    dut->reset_n = 0;
    dut->start = 0;
    dut->max_abs = 0;

    // Reset sequence
    tick(dut);
    tick(dut);
    dut->reset_n = 1;

    // Test input (handle div-by-zero case)
    uint32_t max_abs = 255;
    if (max_abs == 0) {
        std::cerr << "Error: max_abs cannot be zero!" << std::endl;
        return 1;
    }
    dut->max_abs = max_abs;
    dut->start = 1;
    tick(dut);
    dut->start = 0;

    // Wait for ready with sufficient timeout
    bool ready_received = false;
    for (int i = 0; i < 50; i++) {  // Worst-case ~32 cycles + margin
        tick(dut);
        if (dut->ready) {
            ready_received = true;
            break;
        }
    }

    // Handle results
    if (ready_received) {
        uint32_t result = dut->reciprocal_scale;
        uint32_t expected = (static_cast<uint64_t>(127) << 24) / max_abs;
        
        std::cout << "Result:   0x" << std::hex << std::setw(8) << std::setfill('0') << result << "\n";
        std::cout << "Expected: 0x" << std::hex << std::setw(8) << std::setfill('0') << expected << "\n";
        
        if (result != expected) {
            std::cerr << "ERROR: Mismatch!" << std::endl;
        }

        float scale_f = static_cast<float>(result) / (1 << 24);
        scale_f = 1/scale_f;
        std::cout << "Float: " << std::dec << scale_f << std::endl;
    } else {
        std::cerr << "Error: Timeout!" << std::endl;
    }

    dut->final();
    delete dut;
    return 0;
}