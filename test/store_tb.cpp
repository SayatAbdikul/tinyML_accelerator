#include <verilated.h>
#include "Vstore.h"
#include "Vstore___024root.h"
#include <iostream>
#include <cstdint>

static vluint64_t main_time = 0;
double sc_time_stamp() { return main_time; }

static void tick(Vstore* top) {
    top->clk = 0; top->eval(); main_time += 5;
    top->clk = 1; top->eval(); main_time += 5;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vstore* top = new Vstore("store");

    // Reset
    top->clk = 0; top->rst = 1; top->start = 0;
    for (int i = 0; i < 4; ++i) tick(top);
    top->rst = 0;

    // Program STORE: write 10 bytes from buffer id 3 to DRAM at 0x20000
    const uint32_t base_addr = 0x20000;
    const int length = 10; // elements
    top->dram_addr = base_addr;
    top->length = length;
    top->buf_id = 3;

    // Prepare a tile of data to feed when requested
    uint8_t tile_vals[32];
    for (int i = 0; i < 32; ++i) tile_vals[i] = (uint8_t)(i + 1); // 1,2,...

    // Start
    top->start = 1; tick(top); top->start = 0;

    int guard = 10000;
    bool fed_tile = false;

    while (!Verilated::gotFinish() && guard--) {
        // If DUT requests a tile, present data and pulse read_done
        if (top->buf_read_en && !fed_tile) {
            for (int i = 0; i < 32; ++i) top->buf_read_data[i] = tile_vals[i];
            top->buf_read_done = 1;
            fed_tile = true;
        } else {
            top->buf_read_done = 0;
        }
        tick(top);
        if (top->done) break;
    }

    std::cout << "Done=" << (int)top->done << " fed_tile=" << fed_tile << std::endl;

    // Validate DRAM contents written by store module
    int mismatches = 0;
    for (int i = 0; i < length; ++i) {
        // Access public_flat memory via rootp (name per Vstore___024root.h)
        // I don't know how it works, it was done by Copilot.
        uint8_t got = static_cast<uint8_t>(top->rootp->store__DOT__dram__DOT__memory[base_addr + i]);
        uint8_t exp = tile_vals[i];
        if (got != exp) {
            std::cout << "Mismatch at +" << i << ": got=" << (int)got << " exp=" << (int)exp << std::endl;
            mismatches++;
        }
    }

    if (mismatches == 0) {
        std::cout << "DRAM write verification PASSED" << std::endl;
    } else {
        std::cout << "DRAM write verification FAILED, mismatches=" << mismatches << std::endl;
    }

    delete top;
    return (top->done && mismatches == 0) ? 0 : 1;
}
