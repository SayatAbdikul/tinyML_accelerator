module wallace_32x32 (
    input clk,
    input rst_n,
    input valid_in,
    input signed [31:0] a,
    input signed [31:0] b,
    output reg signed [63:0] prod,
    output reg valid_out
);

// Pipeline registers
reg [63:0] stage3_reg [0:9];  // After stage 3 reduction
reg [63:0] stage8_reg [0:1];  // After stage 8 reduction
reg [63:0] prod_reg;          // Final product

// Pipeline validity flags
reg valid_stage1;
reg valid_stage2;
reg valid_stage3;

// Signed multiplication with corrected Baugh-Wooley
wire [63:0] pp [0:31];
wire [63:0] a_extended = {{32{a[31]}}, a};  // Sign-extended multiplicand
genvar gi;
generate
    for (gi = 0; gi < 32; gi = gi + 1) begin : pp_gen
        if (gi < 31) begin: normal_pps
            // Sign-extended shift for bits 0-30
            assign pp[gi] = b[gi] ? (a_extended << gi) : 64'b0;
        end
        else begin: msb_pp
            // Corrected MSB handling: two's complement of sign-extended a
            assign pp[31] = b[31] ? - (a_extended << 31) : 64'b0;
        end
    end
endgenerate

// Reduction stages 1-3 (32->10 rows)
wire [63:0] stage1 [0:21];
generate
    for (gi = 0; gi < 10; gi = gi + 1) begin : stage1_group
        compressor_3to2 comp1_inst (
            .a(pp[3*gi]),
            .b(pp[3*gi+1]),
            .c(pp[3*gi+2]),
            .sum(stage1[2*gi]),
            .carry(stage1[2*gi+1])
        );
    end
    assign stage1[20] = pp[30];
    assign stage1[21] = pp[31];
endgenerate

wire [63:0] stage2 [0:14];
generate
    for (gi = 0; gi < 7; gi = gi + 1) begin : stage2_group
        compressor_3to2 comp2_inst (
            .a(stage1[3*gi]),
            .b(stage1[3*gi+1]),
            .c(stage1[3*gi+2]),
            .sum(stage2[2*gi]),
            .carry(stage2[2*gi+1])
        );
    end
    assign stage2[14] = stage1[21];
endgenerate

wire [63:0] stage3 [0:9];
generate
    for (gi = 0; gi < 5; gi = gi + 1) begin : stage3_group
        compressor_3to2 comp3_inst (
            .a(stage2[3*gi]),
            .b(stage2[3*gi+1]),
            .c(stage2[3*gi+2]),
            .sum(stage3[2*gi]),
            .carry(stage3[2*gi+1])
        );
    end
endgenerate

// Pipeline Stage 1: Register stage3 results
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        valid_stage1 <= 0;
        for (int j = 0; j < 10; j++) stage3_reg[j] <= '0;
    end
    else if (valid_in) begin
        valid_stage1 <= 1;
        for (int j = 0; j < 10; j++) stage3_reg[j] <= stage3[j];
    end
    else begin
        valid_stage1 <= 0;
        for (int j = 0; j < 10; j++) stage3_reg[j] <= '0;
    end
end

// Reduction stages 4-8 (10->2 rows)
wire [63:0] stage4 [0:6];
generate
    for (gi = 0; gi < 3; gi = gi + 1) begin : stage4_group
        compressor_3to2 comp4_inst (
            .a(stage3_reg[3*gi]),
            .b(stage3_reg[3*gi+1]),
            .c(stage3_reg[3*gi+2]),
            .sum(stage4[2*gi]),
            .carry(stage4[2*gi+1])
        );
    end
    assign stage4[6] = stage3_reg[9];
endgenerate

wire [63:0] stage5 [0:4];
generate
    for (gi = 0; gi < 2; gi = gi + 1) begin : stage5_group
        compressor_3to2 comp5_inst (
            .a(stage4[3*gi]),
            .b(stage4[3*gi+1]),
            .c(stage4[3*gi+2]),
            .sum(stage5[2*gi]),
            .carry(stage5[2*gi+1])
        );
    end
    assign stage5[4] = stage4[6];
endgenerate

wire [63:0] stage6 [0:3];
compressor_3to2 comp6_0_inst (
    .a(stage5[0]),
    .b(stage5[1]),
    .c(stage5[2]),
    .sum(stage6[0]),
    .carry(stage6[1])
);
assign stage6[2] = stage5[3];
assign stage6[3] = stage5[4];

wire [63:0] stage7 [0:2];
compressor_3to2 comp7_0_inst (
    .a(stage6[0]),
    .b(stage6[1]),
    .c(stage6[2]),
    .sum(stage7[0]),
    .carry(stage7[1])
);
assign stage7[2] = stage6[3];

wire [63:0] stage8 [0:1];
compressor_3to2 comp8_0_inst (
    .a(stage7[0]),
    .b(stage7[1]),
    .c(stage7[2]),
    .sum(stage8[0]),
    .carry(stage8[1])
);

// Pipeline Stage 2: Register stage8 results
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        valid_stage2 <= 0;
        stage8_reg[0] <= '0;
        stage8_reg[1] <= '0;
    end
    else if (valid_stage1) begin
        valid_stage2 <= 1;
        stage8_reg[0] <= stage8[0];
        stage8_reg[1] <= stage8[1];
    end
    else begin
        valid_stage2 <= 0;
        stage8_reg[0] <= '0;
        stage8_reg[1] <= '0;
    end
end

// Final addition
wire signed [63:0] final_sum = stage8_reg[0] + stage8_reg[1];

// Pipeline Stage 3: Register final result
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        valid_stage3 <= 0;
        prod_reg <= '0;
    end
    else if (valid_stage2) begin
        valid_stage3 <= 1;
        prod_reg <= final_sum;
    end
    else begin
        valid_stage3 <= 0;
        prod_reg <= '0;
    end
end

assign prod = prod_reg;
assign valid_out = valid_stage3;

endmodule
