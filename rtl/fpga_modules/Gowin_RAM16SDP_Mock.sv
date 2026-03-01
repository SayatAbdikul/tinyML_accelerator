// Mock of Gowin_RAM16SDP — matches the port interface used in src/top_gemv.sv
// Simple dual-port RAM: async read, sync write
// 1024 x 32-bit (10-bit address)
module Gowin_RAM16SDP (
    output logic [31:0] dout,
    input  logic        clk,
    input  logic        wre,
    input  logic [9:0]  wad,
    input  logic [31:0] di,
    input  logic [9:0]  rad
);

    logic [31:0] mem [0:1023];

    // Asynchronous read
    assign dout = mem[rad];

    // Synchronous write
    always_ff @(posedge clk) begin
        if (wre)
            mem[wad] <= di;
    end

endmodule
