#include <verilated.h>
#include "Vrelu.h"

#include <cstdint>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <string>

static constexpr int LENGTH = 128; // Must match relu.sv default LENGTH

static void drive_inputs(Vrelu* top, const std::vector<int8_t>& in) {
    for (int i = 0; i < LENGTH; ++i) {
        // Assign as raw 8-bit two's complement
        top->in_vec[i] = static_cast<uint8_t>(in[i]);
    }
}

static void sample_and_check(Vrelu* top, const std::vector<int8_t>& in, const std::string& name) {
    top->eval();

    for (int i = 0; i < LENGTH; ++i) {
        const int exp = (in[i] < 0) ? 0 : static_cast<int>(in[i]);
        const int got = static_cast<int>(static_cast<int8_t>(top->out_vec[i]));
        if (got != exp) {
            std::cerr << "ReLU mismatch in test '" << name << "' at index " << i
                      << ": in=" << static_cast<int>(in[i])
                      << " exp=" << exp
                      << " got=" << got << std::endl;
            std::exit(1);
        }
    }
    std::cout << "PASS: " << name << std::endl;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vrelu* top = new Vrelu;

    // Test 1: all zeros
    {
        std::vector<int8_t> vec(LENGTH, 0);
        drive_inputs(top, vec);
        sample_and_check(top, vec, "all_zeros");
    }

    // Test 2: all -1
    {
        std::vector<int8_t> vec(LENGTH, -1);
        drive_inputs(top, vec);
        sample_and_check(top, vec, "all_minus_one");
    }

    // Test 3: all min (-128)
    {
        std::vector<int8_t> vec(LENGTH, static_cast<int8_t>(-128));
        drive_inputs(top, vec);
        sample_and_check(top, vec, "all_min");
    }

    // Test 4: mixed sawtooth [-32, 31] repeated
    {
        std::vector<int8_t> vec(LENGTH);
        for (int i = 0; i < LENGTH; ++i) vec[i] = static_cast<int8_t>((i % 64) - 32);
        drive_inputs(top, vec);
        sample_and_check(top, vec, "sawtooth_pm32");
    }

    // Test 5: boundary walk including negatives and positives
    {
        std::vector<int8_t> vec(LENGTH);
        for (int i = 0; i < LENGTH; ++i) {
            int v = -64 + (i % 128); // cycles -64..63
            vec[i] = static_cast<int8_t>(v);
        }
        drive_inputs(top, vec);
        sample_and_check(top, vec, "boundary_walk");
    }

    // Test 6: pseudo-random
    {
        std::srand(1);
        std::vector<int8_t> vec(LENGTH);
        for (int i = 0; i < LENGTH; ++i) {
            int r = std::rand() % 256;             // 0..255
            vec[i] = static_cast<int8_t>(r - 128); // -128..127
        }
        drive_inputs(top, vec);
        sample_and_check(top, vec, "random");
    }

    delete top;
    return 0;
}
