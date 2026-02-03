module simple_memory #(
    parameter ADDR_WIDTH = 16,        // Address bus width (determines memory depth)
    parameter DATA_WIDTH = 8,        // Data bus width
    parameter HEX_FILE = "/Users/sayat/Documents/GitHub/tinyML_accelerator/compiler/dram.hex"  // Memory initialization file
)(
    input  logic                    clk,     // Clock
    input  logic                    we,      // Write enable
    input  logic [ADDR_WIDTH-1:0]   addr,    // Address input
    input  logic [DATA_WIDTH-1:0]   din,     // Data input (for writes)
    output logic [DATA_WIDTH-1:0]   dout,    // Data output (for reads)
    input  logic                    dump     // When high for a cycle, dump memory to HEX_FILE
);

// Calculate memory depth
localparam MEM_DEPTH = 2**ADDR_WIDTH;

// Declare memory array 
logic [DATA_WIDTH-1:0] memory [0:MEM_DEPTH-1] /*verilator public_flat*/;

// Initialize memory from hex file
initial begin
    // $display("Initializing memory from %s", HEX_FILE);
    $readmemh(HEX_FILE, memory);
end

// Memory operation
always_ff @(posedge clk) begin
    // Read-before-write behavior
    dout <= memory[addr];
    // Write operation
    if (we) begin
        memory[addr] <= din;
    end
    // Dump on request
    if (dump) begin
        $writememh(HEX_FILE, memory);
        //$display("[simple_memory] Dumped memory to %s", HEX_FILE);
    end
end

endmodule
