#include <verilated.h>                  // Verilator main header (simulation engine API)
#include <iostream>                     // For std::cout / std::cerr
#include <iomanip>                      // For std::setw / std::setfill formatting
#include "Vfetch_unit.h"                // Generated model header for fetch_unit

static vluint64_t main_time = 0;        // Global simulation time counter
double sc_time_stamp() { return main_time; } // Required by some tracing interfaces

void tick(Vfetch_unit* top) {
    top->clk = 0;                       // Drive clock low
    top->eval();                        // Evaluate design at falling edge / low phase
    main_time++;                        // Increment time
    top->clk = 1;                       // Drive clock high
    top->eval();                        // Evaluate design at rising edge
    main_time++;                        // Increment time
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv); // Pass command-line args to Verilator (e.g., +trace)
    auto* top = new Vfetch_unit;        // Allocate DUT instance

    // Init / reset sequence
    top->clk = 0;                       // Initialize clock
    top->rst_n = 0;                     // Assert active‑low reset
    top->fetch_en_i = 0;                // Deassert fetch enable
    tick(top);                          // Cycle 1 under reset
    tick(top);                          // Cycle 2 under reset
    top->rst_n = 1;                     // Release reset (design now active)

    auto fetch_instruction = [&](int idx) {     // Lambda to run one instruction fetch
        top->fetch_en_i = 1;             // Pulse fetch enable (start fetch FSM)
        tick(top);                       // Apply pulse for one cycle
        top->fetch_en_i = 0;             // Remove enable

        const int MAX_CYC = 200;         // Safety timeout bound
        bool got = false;                // Flag when done pulse seen
        for (int i = 0; i < MAX_CYC; ++i) {
            tick(top);                   // Advance one cycle
            if (top->done) {             // Check for fetch completion pulse
                uint64_t instr = top->instr_o; // Read 64‑bit instruction output
                std::cout << "Instruction " << idx      // Report which fetch
                          << " @cycle " << main_time     // Current sim time
                          << " PC(after fetch)=" << top->pc_o // PC after bytes consumed
                          << " instr=0x" << std::setw(16) << std::setfill('0') << std::hex << instr // Hex format
                          << std::dec << std::endl;      // Restore decimal formatting
                got = true;                // Mark success
                break;                     // Exit wait loop
            }
        }
        if (!got) {                        // If timeout exceeded
            std::cerr << "Timeout waiting for instruction " << idx << std::endl;
            exit(1);                       // Abort with error
        }
    };

    fetch_instruction(0);                  // Fetch first 64‑bit instruction
    fetch_instruction(1);                  // Fetch second 64‑bit instruction
    fetch_instruction(2);                  // Fetch third 64‑bit instruction
    fetch_instruction(3);                  // Fetch fourth 64‑bit instruction
    std::cout << "Test completed." << std::endl; // Indicate success

    top->final();                          // Final model cleanup (Verilator hook)
    delete top;                            // Free allocated DUT
    return 0;                              // Exit normally
}
