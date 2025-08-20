module simple_memory #(
    parameter ADDR_WIDTH = 24,        // Address bus width (determines memory depth)
    parameter DATA_WIDTH = 8,        // Data bus width
    parameter HEX_FILE = "/Users/sayat/Documents/GitHub/tinyML_accelerator/rtl/dram.hex"  // Memory initialization file
)(
    input  logic                    clk,     // Clock
    input  logic                    we,      // Write enable
    input  logic [ADDR_WIDTH-1:0]   addr,    // Address input
    input  logic [DATA_WIDTH-1:0]   din,     // Data input (for writes)
    output logic [DATA_WIDTH-1:0]   dout     // Data output (for reads)
);

// Calculate memory depth
localparam MEM_DEPTH = 2**ADDR_WIDTH;

// Declare memory array 
logic [DATA_WIDTH-1:0] memory [0:MEM_DEPTH-1];

// Initialize memory from hex file
initial begin
    $display("Initializing memory from %s", HEX_FILE);
    $readmemh(HEX_FILE, memory);
end

// Memory operation
always_ff @(posedge clk) begin
    // Read-before-write behavior
    dout <= memory[addr];
    //$display("Memory read at address %0h: %0h", addr, memory[addr]);
    // Write operation
    if (we) begin
        memory[addr] <= din;
        //$display("Memory write at address %0h: %0h", addr, memory[addr]);
    end
end

endmodule
