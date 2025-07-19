module scale_calculator (
    input clk,
    input reset_n,
    input [31:0] max_abs,
    input start,
    output reg [31:0] reciprocal_scale,
    output reg ready
);

localparam DIVIDEND = 32'd2130706432;  // 127 << 24
localparam MAX_CYCLES = 32;

reg [5:0] cycle_count;
reg [63:0] rem_quot;   // {remainder, quotient}
reg [31:0] divisor;
reg active;

always @(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
        ready <= 0;
        reciprocal_scale <= 0;
        cycle_count <= 0;
        active <= 0;
        rem_quot <= 0;
        divisor <= 0;
    end else begin
        if (start && !active) begin
            if (max_abs == 0) begin
                reciprocal_scale <= 0;
                ready <= 1;
            end else begin
                divisor <= max_abs;
                rem_quot <= {32'b0, DIVIDEND};
                cycle_count <= 0;
                active <= 1;
                ready <= 0;
            end
        end else if (active) begin
            // Calculate next state COMBINATORIALLY
            reg [63:0] next_rem_quot = rem_quot << 1;
            
            if (next_rem_quot[63:32] >= divisor) begin
                next_rem_quot[63:32] = next_rem_quot[63:32] - divisor;
                next_rem_quot[0] = 1'b1;  // Set LSB of quotient
            end
            
            // Update registers
            rem_quot <= next_rem_quot;
            
            if (cycle_count == MAX_CYCLES-1) begin
                reciprocal_scale <= next_rem_quot[31:0];
                ready <= 1;
                active <= 0;
            end else begin
                cycle_count <= cycle_count + 1;
            end
        end else begin
            ready <= 0;
        end
    end
end

endmodule
