// ===========================================================================
//  Pipelined Quantizer
// ===========================================================================
module quantizer_pipeline (
    input clk,
    input reset_n,
    input signed [31:0] int32_value,
    input [31:0] reciprocal_scale, // Q8.24 format
    input valid_in,
    output reg signed [7:0] int8_value,
    output reg valid_out
);

// Pipeline stages
reg signed [31:0] stage1_value;
reg [31:0] stage1_scale;  // Now used in stage 2
reg stage1_valid;

reg signed [63:0] stage2_product;  // Full 64-bit product
reg stage2_valid;

reg signed [63:0] stage3_rounded;  // 64-bit rounded value
reg stage3_valid;

always @(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
        // Clear pipeline
        stage1_valid <= 0;
        stage2_valid <= 0;
        stage3_valid <= 0;
        valid_out <= 0;
        int8_value <= 0;
    end else begin
        // Stage 1: Input registration
        stage1_value <= int32_value;
        stage1_scale <= reciprocal_scale;  // Now used in stage 2
        stage1_valid <= valid_in;
        
        // Stage 2: Signed multiplication using REGISTERED scale
        stage2_product <= $signed(stage1_value) * $signed(stage1_scale);
        stage2_valid <= stage1_valid;
        
        // Stage 3: Rounding (add 0.5 in Q24)
        stage3_rounded <= (stage2_product + (1 << 23)) >>> 24;
        stage3_valid <= stage2_valid;
        
        // Stage 4: Clamping and output
        valid_out <= stage3_valid;
        if (stage3_valid) begin
            // Proper clamping to [-128, 127]
            if (stage3_rounded > 127) begin
                int8_value <= 127;
            end else if (stage3_rounded < -128) begin
                int8_value <= -128;
            end else begin
                int8_value <= stage3_rounded[7:0];
            end
        end
    end
end

endmodule
