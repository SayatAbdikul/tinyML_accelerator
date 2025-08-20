#include "Vload_v.h"
#include "verilated.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdint>

vluint64_t main_time = 0;
double sc_time_stamp() { return main_time; }

void tick(Vload_v* top) {
    top->clk = 0; top->eval(); main_time++;
    top->clk = 1; top->eval(); main_time++;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vload_v* top = new Vload_v;

    // Test parameters
    const int TILE_WIDTH = 256;
    const int TILE_BYTES = TILE_WIDTH / 8;  // 32 bytes/tile
    const int TEST_DATA_BYTES = 64;         // 2 tiles (64 bytes)
    const int DRAM_ADDR = 0x000000;
    const int WORDS = (TILE_WIDTH + 31) / 32;  // 8 words for 256 bits

    // Reset sequence
    top->rst = 1;
    top->valid_in = 0;
    tick(top);
    top->rst = 0;

    // Apply test stimulus
    top->dram_addr = DRAM_ADDR;
    top->length = TEST_DATA_BYTES * 8;  // Convert bytes to bits
    top->valid_in = 1;
    tick(top);
    top->valid_in = 0;

    int tiles_seen = 0;

    while (tiles_seen < 3 && !Verilated::gotFinish()) {
        tick(top);

        if (top->tile_out) {
            std::cout << "[Tile #" << tiles_seen << "] Data: ";
            // Print bytes in ascending index (address) order
            for (int i = 0; i < TILE_BYTES; ++i) {
                uint8_t byte_val = static_cast<uint8_t>(top->data_out[i]);
                std::cout << std::hex << std::setw(2) << std::setfill('0')
                          << static_cast<int>(byte_val);
            }
            std::cout << std::dec << "\n";
            tiles_seen++;
        }

        if (top->valid_out) {
            std::cout << "[DONE] All tiles processed\n";
            break;
        }
    }

    std::cout << "Total tiles: " << tiles_seen << std::endl;

    delete top;
    return 0;
}