#include <verilated.h>
#include "Vstore.h"
#include "Vstore___024root.h"
#include <iostream>
#include <cstdint>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
static vluint64_t main_time = 0;
double sc_time_stamp() { return main_time; }

static void tick(Vstore* top) {
    top->clk = 0; top->eval(); main_time += 5;
    top->clk = 1; top->eval(); main_time += 5;
}

void fill_dram_with_zeros() {
    const size_t start = 131070;
    const size_t end   = 131090; // exclusive

    // Step 1: read all hex bytes from file
    std::ifstream infile("/Users/sayat/Documents/GitHub/tinyML_accelerator/rtl/dram.hex");
    if (!infile) {
        std::cerr << "Error: cannot open dram.hex for reading.\n";
        return;
    }

    std::vector<unsigned char> bytes;
    std::string token;
    while (infile >> token) {
        unsigned int byte;
        std::stringstream ss;
        ss << std::hex << token;
        ss >> byte;
        bytes.push_back(static_cast<unsigned char>(byte));
    }
    infile.close();

    if (bytes.empty()) {
        std::cerr << "Error: dram.hex is empty or invalid format.\n";
        return;
    }

    // Step 2: ensure the file is large enough
    if (bytes.size() < end) {
        bytes.resize(end, 0); // extend with zeros if needed
    }

    // Step 3: overwrite range with zeros
    for (size_t i = start; i < end; ++i)
        bytes[i] = 0;

    // Step 4: rewrite the modified data
    std::ofstream outfile("/Users/sayat/Documents/GitHub/tinyML_accelerator/rtl/dram.hex");
    if (!outfile) {
        std::cerr << "Error: cannot open dram.hex for writing.\n";
        return;
    }

    outfile << std::uppercase << std::setfill('0');
    for (size_t i = 0; i < bytes.size(); ++i) {
        outfile << std::setw(2) << std::hex << (int)bytes[i] << "\n";
    }
    outfile.close();

    std::cout << "Overwritten dram.hex bytes [" << start << ", " << end << ") with zeros.\n";
}

std::vector<uint8_t> get_values(int len, int start, const std::string& dram) {
    std::ifstream file(dram);
    std::vector<uint8_t> result;

    if (!file) {
        std::cerr << "Error: cannot open file " << dram << "\n";
        return result;
    }

    std::vector<unsigned char> bytes;
    std::string token;

    // Read all hex tokens into memory
    while (file >> token) {
        unsigned int byte;
        std::stringstream ss;
        ss << std::hex << token;
        ss >> byte;
        bytes.push_back(static_cast<unsigned char>(byte));
    }
    file.close();

    // If file smaller than requested range, pad with zeros
    if (bytes.size() < static_cast<size_t>(start + len)) {
        bytes.resize(start + len, 0);
    }

    // Clip start if too large
    if (start >= static_cast<int>(bytes.size())) {
        std::cerr << "Start index out of range: " << start << "\n";
        return result;
    }

    // Extract range
    for (int i = start; i < start + len; ++i)
        result.push_back(bytes[i]);

    return result;
}



int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vstore* top = new Vstore("store");
    fill_dram_with_zeros();
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
        if (top->done) {
            // Allow final writes to complete
            for (int j = 0; j < 1; ++j) tick(top);
            break;
        }
    }

    std::cout << "Done=" << (int)top->done << " fed_tile=" << fed_tile << std::endl;

    // Validate DRAM contents written by store module
    int mismatches = 0;
    std::vector<uint8_t> values_got = get_values(length+5, 0x20000, "/Users/sayat/Documents/GitHub/tinyML_accelerator/rtl/dram.hex");
    for (int i = 0; i < length; ++i) {
        // Access public_flat memory via rootp (name per Vstore___024root.h)
        // I don't know how it works, it was done by Copilot.
        uint8_t got = values_got[i];
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
    return 0;
}
