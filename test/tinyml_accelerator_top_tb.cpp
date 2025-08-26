#include <verilated.h>
#include <verilated_vcd_c.h>
#include "Vtinyml_accelerator_top.h"
#include <iostream>
#include <iomanip>

vluint64_t main_time = 0;
double sc_time_stamp() { 
    return main_time; 
}

void tick(VerilatedVcdC* tfp, Vtinyml_accelerator_top* dut) {
    dut->clk = 0;
    dut->eval();
    if (tfp) tfp->dump(main_time);
    main_time++;
    
    dut->clk = 1;
    dut->eval();
    if (tfp) tfp->dump(main_time);
    main_time++;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    
    // Create DUT instance
    Vtinyml_accelerator_top* dut = new Vtinyml_accelerator_top;
    
    // Create trace
    VerilatedVcdC* tfp = nullptr;
    if (Verilated::gotFinish() == false) {
        Verilated::traceEverOn(true);
        tfp = new VerilatedVcdC;
        dut->trace(tfp, 99);
        tfp->open("tinyml_accelerator_top.vcd");
    }
    
    std::cout << "=== tinyML Accelerator Top Level Test ===" << std::endl;
    
    // Initialize signals
    dut->clk = 0;
    dut->rst = 1;
    dut->start = 0;
    
    // Reset sequence
    std::cout << "Applying reset..." << std::endl;
    for (int i = 0; i < 5; i++) {
        tick(tfp, dut);
    }
    
    dut->rst = 0;
    std::cout << "Released reset" << std::endl;
    
    // Wait a few cycles
    for (int i = 0; i < 3; i++) {
        tick(tfp, dut);
    }
    
    // Start processing
    std::cout << "Starting accelerator..." << std::endl;
    dut->start = 1;
    tick(tfp, dut);
    dut->start = 0;
    
    // Run until done or timeout
    int cycle_count = 0;
    const int TIMEOUT = 10000;  // Generous timeout
    
    while (!dut->done && cycle_count < TIMEOUT) {
        tick(tfp, dut);
        cycle_count++;
        
        // Print progress every 100 cycles
        if (cycle_count % 100 == 0) {
            std::cout << "Cycle " << cycle_count << " - Still processing..." << std::endl;
        }
    }
    
    if (dut->done) {
        std::cout << "\\n=== Processing Complete! ===" << std::endl;
        std::cout << "Total cycles: " << cycle_count << std::endl;
        
        // Display results
        std::cout << "\\nOutput Results:" << std::endl;
        for (int i = 0; i < 10; i++) {
            int8_t result = static_cast<int8_t>(dut->y[i]);
            std::cout << "y[" << i << "] = " << std::setw(4) 
                      << static_cast<int>(result) << std::endl;
        }
        
        std::cout << "\\n✅ Test completed successfully!" << std::endl;
    } else {
        std::cout << "\\n❌ Test timed out after " << TIMEOUT << " cycles!" << std::endl;
        return 1;
    }
    
    // Run a few more cycles to see final state
    for (int i = 0; i < 5; i++) {
        tick(tfp, dut);
    }
    
    // Cleanup
    if (tfp) {
        tfp->close();
        delete tfp;
    }
    delete dut;
    
    return 0;
}
