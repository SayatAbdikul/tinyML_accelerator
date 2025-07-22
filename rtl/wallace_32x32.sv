module wallace_32x32 (
    input [31:0] a,
    input [31:0] b,
    output [63:0] prod
);

// Generate partial products
wire [63:0] pp [0:31];
generate
    genvar i;
    for (i = 0; i < 32; i = i + 1) begin : pp_gen
        assign pp[i] = b[i] ? ({{32{1'b0}}, a} << i) : 64'b0;
    end
endgenerate

// Reduction stages
wire [63:0] stage1 [0:21];  // Stage 1: 32->22
wire [63:0] stage2 [0:14];  // Stage 2: 22->15
wire [63:0] stage3 [0:9];   // Stage 3: 15->10
wire [63:0] stage4 [0:6];   // Stage 4: 10->7
wire [63:0] stage5 [0:4];   // Stage 5: 7->5
wire [63:0] stage6 [0:3];   // Stage 6: 5->4
wire [63:0] stage7 [0:2];   // Stage 7: 4->3
wire [63:0] stage8 [0:1];   // Stage 8: 3->2

// ---------------------------------------------------------------
// Stage 1: Reduce 32 partial products to 22 rows
// ---------------------------------------------------------------
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

// ---------------------------------------------------------------
// Stage 2: Reduce 22 rows to 15 rows
// ---------------------------------------------------------------
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
    // Remaining 1 row
    assign stage2[14] = stage1[21];
endgenerate

// ---------------------------------------------------------------
// Stage 3: Reduce 15 rows to 10 rows
// ---------------------------------------------------------------
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

// ---------------------------------------------------------------
// Stage 4: Reduce 10 rows to 7 rows
// ---------------------------------------------------------------
generate
    for (i = 0; i < 3; i = i + 1) begin : stage4_group
        compressor_3to2 comp4 (
            .a(stage3[3*i]),
            .b(stage3[3*i+1]),
            .c(stage3[3*i+2]),
            .sum(stage4[2*i]),
            .carry(stage4[2*i+1])
        );
    end
    // Remaining 1 row
    assign stage4[6] = stage3[9];
endgenerate

// ---------------------------------------------------------------
// Stage 5: Reduce 7 rows to 5 rows
// ---------------------------------------------------------------
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

// ---------------------------------------------------------------
// Stage 6: Reduce 5 rows to 4 rows
// ---------------------------------------------------------------
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

// ---------------------------------------------------------------
// Stage 7: Reduce 4 rows to 3 rows
// ---------------------------------------------------------------
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

// ---------------------------------------------------------------
// Stage 8: Reduce 3 rows to 2 rows
// ---------------------------------------------------------------
generate
    compressor_3to2 comp8_0 (
        .a(stage7[0]),
        .b(stage7[1]),
        .c(stage7[2]),
        .sum(stage8[0]),
        .carry(stage8[1])
    );
endgenerate

// ---------------------------------------------------------------
// Final addition
// ---------------------------------------------------------------
assign prod = stage8[0] + stage8[1];

endmodule
