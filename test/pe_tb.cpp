#include <verilated.h>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include "Vpe.h"  // Include the Verilator-generated header file

int main(int argc, char **argv) {
    // Initialize Verilator's simulation
    Verilated::commandArgs(argc, argv);

    // Create an instance of the module
    Vpe* top = new Vpe;

    // Set the initial values for the inputs
    top->clk = 0;
    top->rst = 1;
    top->eval();
    top->rst = 0;
    int w, x;
    top->clk = 1;  // Initialize weight input
    // Simulate for 10 cycles
    for (int cycle = 0; cycle < 100; cycle++) {
        // Apply reset for the first 2 cycles
        w = rand() % 256 - 128;
        x = rand() % 256 - 128;
        top->w = w;  // Example weight input
        top->x = x; // Example activation input
        // Toggle the clock
        top->clk = !top->clk;
        top->eval();
        top->clk = !top->clk;
        
        // Evaluate the module at the current time step
        top->eval();
        int actual = static_cast<int16_t>(top->y);
        // Print the outputs at each clock cycle
        if (w*x != actual) {
            std::cout << "Cycle " << cycle << ": " 
                      << "w = " << std::setw(4) << w
                      << ", x = " << std::setw(4) << x 
                      << ", y = " << std::setw(4) << actual 
                      << std::endl;
        }
    }

    // Clean up and exit
    delete top;
    return 0;
}
