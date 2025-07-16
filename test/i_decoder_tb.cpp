#include "Vi_decoder.h"
#include "verilated.h"
#include <iostream>
using namespace std;

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    Vi_decoder* dut = new Vi_decoder;

    // Instruction set (uint64_t instead of int)
    uint64_t instructions[] = {
        0x10000000000C4121,
        0x20000000080C4022,
        0x3000000000020061,
        0x00000148C80C40A4,
        0x00000000000014E5,
        0x2188000004020042,
        0x3000800000010081,
        0x00000239040200C4,
        0x0000000000001905,
        0x21A8000000A10022,
        0x3000C00000002861,
        0x00000140C0A100A4,
        0x40000000000028A3
    };

    for (int i = 0; i < 13; i++) {
        dut->instr = instructions[i];  // Correct signal name is likely 'instr'
        dut->eval();

        cout << "Instruction: 0x" << hex << instructions[i] << dec << "\n";
        cout << "Opcode     : " << int(dut->opcode) << "\n";
        cout << "Dest       : " << int(dut->dest) << "\n";
        cout << "Cols/Len   : " << int(dut->length_or_cols) << "\n";
        cout << "Rows       : " << int(dut->rows) << "\n";
        cout << "Addr       : 0x" << hex << int(dut->addr) << dec << "\n";
        cout << "W, X, B    : " << int(dut->w) << ", " << int(dut->x) << ", " << int(dut->b) << "\n";
        cout << "-----------\n";
    }

    dut->final();
    delete dut;
    return 0;
}
