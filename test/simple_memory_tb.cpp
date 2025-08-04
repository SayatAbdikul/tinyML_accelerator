#include <verilated.h>
#include <verilated_vcd_c.h>
#include "Vsimple_memory.h"
#include <fstream>
#include <iostream>
#include <iomanip>

int main(int argc, char** argv) {
    // Initialize Verilator
    Verilated::commandArgs(argc, argv);

    // Create instance and trace
    Vsimple_memory* dut = new Vsimple_memory;
    // Initialize signals
    dut->clk = 1;
    dut->we = 0;
    dut->addr = 0;
    dut->din = 0;

    // Run simulation for 20 clock cycles
    for (int i = 0; i < 20; i++) {
        // Toggle clock
        dut->clk = !dut->clk;
        
        // Change address on falling edge
        if (dut->clk == 0) {
            // Read different addresses at different times
            if (i == 2) dut->addr = 0;      // Read addr 0
            else if (i == 4) dut->addr = 1; // Read addr 1
            else if (i == 6) dut->addr = 2; // Read addr 2
            else if (i == 8) dut->addr = 3; // Read addr 3
            else if (i == 10) dut->addr = 9; // Read addr 9
            else if (i == 12) dut->addr = 11; // Read addr 11 (0xFF)
            else if (i == 14) {
                // Write operation
                dut->we = 1;
                dut->addr = 20;
                dut->din = 0x55;
            }
            else if (i == 16) {
                // Read back written value
                dut->we = 0;
                dut->addr = 20;
            }
        }

        // Evaluate model
        dut->eval();

        // Print values on rising edge
        if (dut->clk == 1) {
            if (i > 1 && i < 18) {  // Skip initial unstable states
                std::cout << "Cycle " << std::setw(2) << i/2
                          << ": Addr = 0x" << std::hex << std::setw(2) 
                          << static_cast<int>(dut->addr)
                          << " | Data = 0x" << std::hex << std::setw(2)
                          << static_cast<int>(dut->dout)
                          << std::endl;
            }
        }
    }

    delete dut;
    
    std::cout << "\nSimulation completed!" << std::endl;
    return 0;
}
