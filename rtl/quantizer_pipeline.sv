module quantizer_pipeline (
    input clk,
    input reset_n,
    input signed [31:0] int32_value,
    input [31:0] reciprocal_scale, // Q8.24 format
    input valid_in,
    output reg signed [7:0] int8_value,
    output reg valid_out
);

    // Stage 1: Register inputs
    reg signed [31:0] stage1_value;
    reg [31:0] stage1_scale;
    reg stage1_valid;

    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            stage1_valid <= 0;
        end else begin
            stage1_value <= int32_value;
            stage1_scale <= reciprocal_scale;
            stage1_valid <= valid_in;
        end
    end

    // Stage 2: 32x32 multiplier replacement (Native * operator)
    logic signed [63:0] stage2_product;
    logic stage2_valid;

    reg signed [63:0] mult_pipe [0:3];
    reg [3:0] valid_pipe;
    
    always_ff @(posedge clk or negedge reset_n) begin
        if(!reset_n) begin
             valid_pipe <= 0;
             mult_pipe[0] <= 0;
             mult_pipe[1] <= 0; 
             mult_pipe[2] <= 0; 
             mult_pipe[3] <= 0;
        end else begin
             valid_pipe <= {valid_pipe[2:0], stage1_valid};
             // Signed × Unsigned multiplication: int32 × Q8.24 reciprocal_scale
             mult_pipe[0] <= $signed(stage1_value) * $signed({1'b0, stage1_scale});
             mult_pipe[1] <= mult_pipe[0];
             mult_pipe[2] <= mult_pipe[1];
             mult_pipe[3] <= mult_pipe[2];
        end
    end
    
    assign stage2_product = mult_pipe[3];
    assign stage2_valid = valid_pipe[3];


    // Stage 3: Rounding
    reg signed [63:0] stage3_rounded;
    reg stage3_valid;

    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            stage3_valid <= 0;
        end else begin
            stage3_rounded <= (stage2_product + (1 << 23)) >>> 24;
            stage3_valid <= stage2_valid;
        end
    end

    // Stage 4: Clamp to int8
    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            int8_value <= 0;
            valid_out <= 0;
        end else begin
            valid_out <= stage3_valid;
            if (stage3_valid) begin
                if (stage3_rounded > 127)
                    int8_value <= 127;
                else if (stage3_rounded <= -128)
                    int8_value <= -128;
                else
                    int8_value <= $signed(stage3_rounded[7:0]);
                    // $display("Quantized value: %d", int8_value);
            end
        end
    end
endmodule
