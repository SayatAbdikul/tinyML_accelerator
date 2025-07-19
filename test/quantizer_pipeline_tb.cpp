#include "Vquantizer_pipeline.h"
#include "verilated.h"
#include <iostream>
#include <cstdint>
#include <vector>
#include <iomanip>

vluint64_t main_time = 0; // Simulation time
double sc_time_stamp() { return main_time; }

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    Vquantizer_pipeline* dut = new Vquantizer_pipeline;

    dut->clk = 0;
    dut->reset_n = 0;

    // Clock generation and reset deassertion
    for (int i = 0; i < 4; ++i) {
        dut->clk = !dut->clk;
        dut->eval();
        main_time++;
    }
    dut->reset_n = 1;

    // Structure to hold input and expected test vectors
    struct TestVec {
        int32_t input_value;
        uint32_t reciprocal_scale; // Q8.24 fixed-point
    };

    std::vector<TestVec> test_inputs = {
        {1000, 1 << 24},     // scale = 1.0 -> result = 1000 -> clipped to 127
        {-1000, 1 << 24},    // result = -1000 -> clipped to -128
        {127, 1 << 24},      // result = 127 -> no clip
        {50, (uint32_t)(0.5 * (1 << 24))}, // scale = 0.5 -> result = 25
        {200, (uint32_t)(0.25 * (1 << 24))}, // scale = 0.25 -> result = 50
        {-64, (uint32_t)(0.5 * (1 << 24))},  // result = -32
    };

    const int PIPELINE_LATENCY = 4;
    std::vector<int8_t> golden_outputs;

    // Feed inputs
    for (size_t i = 0; i < test_inputs.size() + PIPELINE_LATENCY; ++i) {
        dut->clk = 0;
        dut->eval();

        if (i < test_inputs.size()) {
            dut->int32_value = test_inputs[i].input_value;
            dut->reciprocal_scale = test_inputs[i].reciprocal_scale;
            dut->valid_in = 1;
        } else {
            dut->valid_in = 0;
            dut->int32_value = 0;
            dut->reciprocal_scale = 0;
        }

        dut->clk = 1;
        dut->eval();
        main_time++;

        // Monitor output
        if (dut->valid_out) {
            int8_t value = static_cast<int8_t>(dut->int8_value);
            std::cout << "Cycle " << std::setw(2) << main_time
                    << " | Output = " << std::setw(4) << static_cast<int>(value)
                    << std::endl;
        }
    }

    delete dut;
    return 0;
}
