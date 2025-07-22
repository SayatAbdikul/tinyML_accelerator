// 3:2 Compressor (Carry-Save Adder)
module compressor_3to2 (
    input [63:0] a,
    input [63:0] b,
    input [63:0] c,
    output [63:0] sum,
    output [63:0] carry
);
    assign sum = a ^ b ^ c;
    assign carry = ((a & b) | (b & c) | (a & c)) << 1;
endmodule
