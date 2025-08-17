module relu #(
    parameter int DATA_WIDTH = 8,
    parameter int LENGTH     = 128
)(
    input  logic signed [DATA_WIDTH-1:0] in_vec  [0:LENGTH-1],
    output logic signed [DATA_WIDTH-1:0] out_vec [0:LENGTH-1]
);
    // Element-wise ReLU: out = max(0, in)
    always_comb begin
        for (int i = 0; i < LENGTH; i++) begin
            out_vec[i] = in_vec[i][DATA_WIDTH-1] ? '0 : in_vec[i];
        end
    end
endmodule
