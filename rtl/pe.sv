module pe #(
    parameter int DATA_WIDTH = 8
)(
    input  logic clk,
    input  logic rst,
    input  logic signed [DATA_WIDTH-1:0] w,   // Weight input
    input  logic signed [DATA_WIDTH-1:0] x,   // Activation input
    output logic signed [2*DATA_WIDTH-1:0] y    // Output (w * x)
);

    logic signed [2*DATA_WIDTH-1:0] mult_result;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            mult_result <= '0;
        end else begin
            mult_result <= w * x;
        end
    end

    assign y = mult_result;  // Truncate or keep full bits as needed

endmodule
