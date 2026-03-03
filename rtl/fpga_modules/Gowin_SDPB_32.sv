// Gowin_SDPB_32 — simulation behavioural model for BSRAM (SDPB) 1024 x 32-bit
// Synchronous read with 1-cycle output latency (registered output).
// Drop-in replacement for Gowin_RAM16SDP on the res_mem (accumulator) port.
// Port interface kept identical to Gowin_RAM16SDP so gemv_unit_core.sv only
// needs the module-name change and the FSM READ_xxx pipeline states.
//
// For synthesis (Gowin EDA): see src/Gowin_SDPB_32.sv which uses (* ram_style = "block" *)
// to infer SDPB BSRAM blocks instead of LUTRAM, eliminating the 6-level MUX2
// output cascade that was the dominant term in the critical path.

module Gowin_SDPB_32 (
    output logic [31:0] dout,   // Registered output: reflects mem[rad] from previous clk edge
    input  logic        clk,
    input  logic        wre,    // Write enable (synchronous)
    input  logic [9:0]  wad,    // Write address
    input  logic [31:0] di,     // Write data
    input  logic [9:0]  rad     // Read address
);

    logic [31:0] mem [0:1023];

    // Synchronous read — 1-cycle latency (matches SDPB registered output).
    // res_dout in gemv_unit_core reflects mem[rad] presented on the PREVIOUS clock edge.
    always_ff @(posedge clk) begin
        dout <= mem[rad];
    end

    // Synchronous write
    always_ff @(posedge clk) begin
        if (wre)
            mem[wad] <= di;
    end

endmodule
