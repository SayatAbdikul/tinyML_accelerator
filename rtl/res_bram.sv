// Result BRAM Module for top_gemv
// Explicitly instantiated as dual-port BRAM to avoid DFF inference
// This module wraps the result accumulator array

module res_bram #(
    parameter DATA_WIDTH = 32,      // 4*8 = 32 bits per accumulator
    parameter DEPTH = accelerator_config_pkg::MAX_ROWS,         // MAX_ROWS entries
    parameter ADDR_WIDTH = $clog2(accelerator_config_pkg::MAX_ROWS)       // $clog2(MAX_ROWS)
)(
    input  logic                    clk,
    input  logic                    rst,
    
    // Write port
    input  logic                    wr_en,
    input  logic [ADDR_WIDTH-1:0]   wr_addr,
    input  logic signed [DATA_WIDTH-1:0] wr_data,
    
    // Read port
    input  logic                    rd_en,
    input  logic [ADDR_WIDTH-1:0]   rd_addr,
    output logic signed [DATA_WIDTH-1:0] rd_data
);

    // Explicit BRAM declaration with Gowin-compatible attributes
    // Using simple dual-port RAM pattern that Gowin recognizes
    (* ram_style = "block" *)
    logic signed [DATA_WIDTH-1:0] mem [0:DEPTH-1];
    
    // Write port - synchronous write
    always_ff @(posedge clk) begin
        if (wr_en) begin
            mem[wr_addr] <= wr_data;
        end
    end
    
    // Read port - synchronous read with output register
    always_ff @(posedge clk) begin
        if (rd_en) begin
            rd_data <= mem[rd_addr];
        end
    end

endmodule
