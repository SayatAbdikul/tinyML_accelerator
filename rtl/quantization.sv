module quantization (
    input clk,
    input reset_n,          // Active-low reset
    // Control interface
    input start_calib,      // Pulse to start calibration
    input [31:0] max_abs,   // Max absolute value (unsigned)
    // Data interface
    input signed [31:0] data_in,
    input data_valid,
    output signed [7:0] data_out,
    output data_valid_out,
    // Status signals
    output calib_busy, 
    output calib_ready,
    output data_ready       // System ready for new data
);

// ===========================================================================
//  FSM States & Control Signals
// ===========================================================================
localparam STATE_IDLE    = 2'b00;
localparam STATE_CALIB   = 2'b01;
localparam STATE_READY   = 2'b10;

reg [1:0] state;
reg [31:0] stored_max_abs;
reg [31:0] scale_reg;      // Q8.24 format
reg calib_ready_reg;

// FSM Control
always @(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
        state <= STATE_IDLE;
        stored_max_abs <= 0;
        scale_reg <= 0;
        calib_ready_reg <= 0;
    end else begin
        case (state)
            STATE_IDLE: begin
                if (start_calib) begin
                    stored_max_abs <= max_abs;
                    state <= STATE_CALIB;
                    calib_ready_reg <= 0;
                end
            end
            
            STATE_CALIB: begin
                // Wait for calibration completion
                if (div_calib_ready) begin
                    scale_reg <= reciprocal_scale;
                    //$display("Calibration complete: scale = %0d", reciprocal_scale);
                    state <= STATE_READY;
                    calib_ready_reg <= 1;
                end
            end
            
            STATE_READY: begin
                // Restart calibration if requested
                if (start_calib) begin
                    stored_max_abs <= max_abs;
                    state <= STATE_CALIB;
                    calib_ready_reg <= 0;
                end
            end
            default: begin
                
            end
        endcase
    end
end

// ===========================================================================
//  Scale Calibration Module (Reciprocal Calculation)
// ===========================================================================
wire div_calib_ready;
wire [31:0] reciprocal_scale;

scale_calculator u_scale_calculator (
    .clk(clk),
    .reset_n(reset_n), 
    .max_abs(stored_max_abs),
    .start(state == STATE_CALIB && !div_calib_ready),  // Continuous start
    .reciprocal_scale(reciprocal_scale),
    .ready(div_calib_ready)
);

// ===========================================================================
//  Quantization Pipeline
// ===========================================================================
quantizer_pipeline u_quantizer (
    .clk(clk),
    .reset_n(reset_n),
    .int32_value(data_in),
    .reciprocal_scale(scale_reg),
    .valid_in(data_valid),
    .int8_value(data_out),
    .valid_out(data_valid_out)
);

// ===========================================================================
//  Status Outputs
// ===========================================================================
assign calib_busy = (state == STATE_CALIB);
assign calib_ready = calib_ready_reg;
assign data_ready = (state == STATE_READY);
endmodule
