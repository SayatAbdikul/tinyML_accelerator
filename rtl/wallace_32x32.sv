module wallace_32x32 (
    input clk,
    input rst_n,
    input [31:0] a,
    input [31:0] b,
    output reg [63:0] prod,
    output reg valid_out
);

// Stage 1: Partial Product Generation + Initial Reduction
reg [63:0] stage3_reg [0:9];  // 10 rows after stage 3

// Stage 2: Intermediate Reduction
reg [63:0] stage8_reg [0:1];  // Final 2 rows

// Stage 3: Final Addition
reg [63:0] prod_reg;

// Pipeline stage validity flags
reg valid_stage1;
reg valid_stage2;
reg valid_stage3;


// ------------------------------------------------------------------------
// Combinational logic between pipeline stages
// ------------------------------------------------------------------------

// Partial products
wire [63:0] pp [0:31];
generate
    genvar i;
    for (i = 0; i < 32; i = i + 1) begin : pp_gen
        assign pp[i] = b[i] ? ({{32{1'b0}}, a} << i) : 64'b0;
    end
endgenerate

// Stage 1: 32->22 rows
wire [63:0] stage1 [0:21];
generate
    for (i = 0; i < 10; i = i + 1) begin : stage1_group
        compressor_3to2 comp1 (
            .a(pp[3*i]),
            .b(pp[3*i+1]),
            .c(pp[3*i+2]),
            .sum(stage1[2*i]),
            .carry(stage1[2*i+1])
        );
    end
    assign stage1[20] = pp[30];
    assign stage1[21] = pp[31];
endgenerate

// Stage 2: 22->15 rows
wire [63:0] stage2 [0:14];
generate
    for (i = 0; i < 7; i = i + 1) begin : stage2_group
        compressor_3to2 comp2 (
            .a(stage1[3*i]),
            .b(stage1[3*i+1]),
            .c(stage1[3*i+2]),
            .sum(stage2[2*i]),
            .carry(stage2[2*i+1])
        );
    end
    assign stage2[14] = stage1[21];
endgenerate

// Stage 3: 15->10 rows
wire [63:0] stage3 [0:9];
generate
    for (i = 0; i < 5; i = i + 1) begin : stage3_group
        compressor_3to2 comp3 (
            .a(stage2[3*i]),
            .b(stage2[3*i+1]),
            .c(stage2[3*i+2]),
            .sum(stage3[2*i]),
            .carry(stage3[2*i+1])
        );
    end
endgenerate

// ------------------------------------------------------------------------
// Pipeline Stage 1 Register
// ------------------------------------------------------------------------
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        valid_stage1 <= 0;  // Reset stage 1 validity
        for (int j = 0; j < 10; j++) stage3_reg[j] <= 64'b0;
    end
    else begin
        valid_stage1 <= 1;  // Mark stage 1 as valid
        for (int j = 0; j < 10; j++) stage3_reg[j] <= stage3[j];
    end
end

// ------------------------------------------------------------------------
// Stage 4-8: Combinational Reduction (10 rows -> 2 rows)
// ------------------------------------------------------------------------
wire [63:0] stage4 [0:6];   // Stage 4: 10->7 rows
wire [63:0] stage5 [0:4];   // Stage 5: 7->5 rows
wire [63:0] stage6 [0:3];   // Stage 6: 5->4 rows
wire [63:0] stage7 [0:2];   // Stage 7: 4->3 rows
wire [63:0] stage8 [0:1];   // Stage 8: 3->2 rows

// Stage 4: Reduce 10 rows to 7 rows
generate
    for (i = 0; i < 3; i = i + 1) begin : stage4_group
        compressor_3to2 comp4 (
            .a(stage3_reg[3*i]),
            .b(stage3_reg[3*i+1]),
            .c(stage3_reg[3*i+2]),
            .sum(stage4[2*i]),
            .carry(stage4[2*i+1])
        );
    end
    // Remaining 1 row
    assign stage4[6] = stage3_reg[9];
endgenerate

// Stage 5: Reduce 7 rows to 5 rows
generate
    for (i = 0; i < 2; i = i + 1) begin : stage5_group
        compressor_3to2 comp5 (
            .a(stage4[3*i]),
            .b(stage4[3*i+1]),
            .c(stage4[3*i+2]),
            .sum(stage5[2*i]),
            .carry(stage5[2*i+1])
        );
    end
    // Remaining 1 row
    assign stage5[4] = stage4[6];
endgenerate

// Stage 6: Reduce 5 rows to 4 rows
generate
    compressor_3to2 comp6_0 (
        .a(stage5[0]),
        .b(stage5[1]),
        .c(stage5[2]),
        .sum(stage6[0]),
        .carry(stage6[1])
    );
    assign stage6[2] = stage5[3];
    assign stage6[3] = stage5[4];
endgenerate

// Stage 7: Reduce 4 rows to 3 rows
generate
    compressor_3to2 comp7_0 (
        .a(stage6[0]),
        .b(stage6[1]),
        .c(stage6[2]),
        .sum(stage7[0]),
        .carry(stage7[1])
    );
    assign stage7[2] = stage6[3];
endgenerate

// Stage 8: Reduce 3 rows to 2 rows
generate
    compressor_3to2 comp8_0 (
        .a(stage7[0]),
        .b(stage7[1]),
        .c(stage7[2]),
        .sum(stage8[0]),
        .carry(stage8[1])
    );
endgenerate

// ------------------------------------------------------------------------
// Pipeline Stage 2 Register
// ------------------------------------------------------------------------
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        valid_stage2 <= 0;  // Reset stage 2 validity
        stage8_reg[0] <= 64'b0;
        stage8_reg[1] <= 64'b0;
    end
    else begin
        valid_stage2 <= valid_stage1;  // Mark stage 2 as valid
        stage8_reg[0] <= stage8[0];
        stage8_reg[1] <= stage8[1];
    end
end

// ------------------------------------------------------------------------
// Stage 3: Final Addition
// ------------------------------------------------------------------------
wire [63:0] final_sum;

assign final_sum = stage8_reg[0] + stage8_reg[1];

// ------------------------------------------------------------------------
// Pipeline Stage 3 Register
// ------------------------------------------------------------------------
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        valid_stage3 <= 0;  // Reset stage 3 validity
        prod_reg <= 64'b0;
    end
    else begin
        valid_stage3 <= valid_stage2;  // Mark stage 3 as valid
        prod_reg <= final_sum;
    end
end

assign prod = prod_reg;
assign valid_out = valid_stage3;  // Output valid signal for the final product

endmodule
