module gemv #(parameter DATA_WIDTH = 8, parameter ROWS = 128, parameter COLUMNS = 128, parameter TILE_SIZE = 8)
(
  input logic clk,
  input logic rst,
  input logic signed [DATA_WIDTH-1:0] w [0:ROWS-1][0:COLUMNS-1],
  input logic signed [DATA_WIDTH-1:0] x [0:COLUMNS-1],
  input logic signed [DATA_WIDTH-1:0] bias [0:ROWS-1],
  output logic signed [DATA_WIDTH-1:0] y [0:ROWS-1],
  output logic done
);
    typedef enum logic [3:0] {
        IDLE, PRESENT_TILE, ACCUMULATE, NEXT_ROW, BIAS, FIND_MAX, COMPUTE_SCALE, QUANTIZE, DONE
    } state_t;
    
    state_t state;

    localparam ROW_IDX_WIDTH  = $clog2(ROWS);
    localparam TILE_IDX_WIDTH = $clog2(COLUMNS / TILE_SIZE);

    logic signed [4*DATA_WIDTH-1:0] res [0:ROWS-1];
    logic [ROW_IDX_WIDTH-1:0] row_idx;
    logic [TILE_IDX_WIDTH-1:0] tile_idx;
    
    logic signed [DATA_WIDTH-1:0] w_in [0:TILE_SIZE-1];
    logic signed [DATA_WIDTH-1:0] x_in [0:TILE_SIZE-1];
    logic signed [2*DATA_WIDTH-1:0] pe_out [0:TILE_SIZE-1];
    logic signed [4*DATA_WIDTH-1:0] temp_sum;
    // Scale computation values
    logic signed [4*DATA_WIDTH-1:0] reciprocal_scale; // Placeholder for scale computation
    logic scale_ready;
    logic signed [4*DATA_WIDTH-1:0] max_abs_reg, current_abs;
    //logic signed [4*DATA_WIDTH-1:0] abs_val;
    logic [$clog2(ROWS)-1:0] max_idx;

    // Quantization values
    logic signed [4*DATA_WIDTH-1:0] int32_value;
    logic signed [DATA_WIDTH-1:0] int8_value;
    logic [ROW_IDX_WIDTH-1:0] quant_in_idx, quant_out_idx;
    
    logic quant_valid_in, quant_valid_out;

    // Input selection
    always_comb begin
        for (int i = 0; i < TILE_SIZE; i++) begin
            int idx = tile_idx * TILE_SIZE + i;
            w_in[i] = (idx < COLUMNS) ? w[row_idx][idx] : '0;
            x_in[i] = (idx < COLUMNS) ? x[idx] : '0;
        end
    end

    // PE array
    genvar i;
    generate for (i = 0; i < TILE_SIZE; i++) begin
        pe #(DATA_WIDTH) pe_inst (
            .clk(clk),
            .rst(rst),
            .w(w_in[i]),
            .x(x_in[i]),
            .y(pe_out[i])
        );
    end 
    endgenerate

    // Sum PE outputs
    always_comb begin
        temp_sum = '0;
        for (int j = 0; j < TILE_SIZE; j++) begin
            temp_sum = temp_sum + {{16{pe_out[j][15]}}, pe_out[j]}; // the last successful signed number fix was here
        end
    end

    // FSM controller
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            row_idx <= '0;
            tile_idx <= '0;
            for (int j = 0; j < ROWS; j++) begin
                res[j] <= '0;
            end
        end else begin
            case (state)
                IDLE: begin
                    row_idx <= '0;
                    tile_idx <= '0;
                    for (int j = 0; j < ROWS; j++) begin
                        res[j] <= '0;
                    end
                    state <= PRESENT_TILE;
                end
                
                PRESENT_TILE: begin
                    // Present tile to PEs
                    state <= ACCUMULATE;  // Wait for PE results
                end
                
                ACCUMULATE: begin
                    // Accumulate results from PEs
                    res[row_idx] <= res[row_idx] + temp_sum;
                    
                    // Check if last tile in row
                    if ({4'b0, tile_idx} < (COLUMNS/TILE_SIZE)-1) begin
                        tile_idx <= tile_idx + 1;
                        state <= PRESENT_TILE;  // Process next tile
                    end else begin
                        tile_idx <= '0;
                        state <= NEXT_ROW;  // Move to next row
                    end
                end
                
                NEXT_ROW: begin
                    if ({4'b0, row_idx} < ROWS-1) begin
                        row_idx <= row_idx + 1;
                        state <= PRESENT_TILE;  // Process next row
                    end else begin
                        state <= BIAS;  // All rows processed
                    end
                end
                
                BIAS: begin
                    // Add bias to all rows
                    for (int j = 0; j < ROWS; j++) begin
                        res[j] <= res[j] + {{24{bias[j][7]}}, bias[j]};
                        //$display("Row %0d: Result after bias = %0d", j, res[j]+bias[j]);
                    end
                    state <= FIND_MAX;  // Prepare for quantization
                end
                FIND_MAX: begin
                    // Compare and update max value
                    if (current_abs > max_abs_reg) begin
                        // $display("current_abs is %d", $signed(current_abs));
                        // $display("and the res value for it is %d", res[max_idx]);
                        max_abs_reg <= current_abs;
                    end
                    
                    // Move to next row
                    if ({1'b0, max_idx} < ROWS-1) begin
                        max_idx <= max_idx + 1;
                    end else begin
                        state <= COMPUTE_SCALE;
                        // Handle division by zero
                        if (max_abs_reg == 0) max_abs_reg <= 1;
                    end
                end
                COMPUTE_SCALE: begin
                    if(scale_ready) begin
                        state <= QUANTIZE;  // Move to quantization stage   
                        //$display("The max abs value is %d", max_abs_reg);                     
                    end
                end
                QUANTIZE: begin
                    // Apply quantization using the computed scale
                    if({1'b0, quant_in_idx} < ROWS) begin
                        int32_value <= res[quant_in_idx];
                        quant_in_idx <= quant_in_idx + 1;
                        quant_valid_in <= 1;
                    end else begin
                        quant_valid_in <= 0;  // Reset valid signal
                    end
                    if (quant_valid_out) begin
                        res[quant_out_idx] <= {{24{int8_value[7]}}, int8_value};  // Q8.24 to Q8.0
                        quant_out_idx <= quant_out_idx + 1;
                        if ({1'b0, quant_out_idx} == ROWS - 1) begin
                            state <= DONE;  // All rows quantized
                        end
                    end
                end
                DONE: begin
                    state <= IDLE;  // Computation complete
                end
                
                default: state <= IDLE;
            endcase
        end
    end
    assign current_abs = (res[max_idx] >= 0) ? res[max_idx] : $signed(res[max_idx])*(-1);
    
    // always_comb begin
    //     abs_val = '0;
    //     if(state == COMPUTE_SCALE) begin
    //         max_abs_reg = '0;
    //         for (int j = 0; j < ROWS; j++) begin
    //             abs_val = (res[j] >= 0) ? res[j] : -res[j];
    //             if (abs_val > max_abs_reg) max_abs_reg = abs_val;
    //         end
    //     end else begin
    //         max_abs_reg = '0;  // Reset when not in FIND_MAX state
    //     end
    // end
    scale_calculator inst (
        .clk(clk),
        .reset_n(~rst),
        .max_abs(max_abs_reg),
        .start(state == COMPUTE_SCALE),  // Start scale calculation
        .reciprocal_scale(reciprocal_scale),
        .ready(scale_ready)
    );


    quantizer_pipeline quantize(
        .clk(clk),
        .reset_n(~rst),
        .int32_value(int32_value),
        .reciprocal_scale(reciprocal_scale),
        .valid_in(quant_valid_in),
        .int8_value(int8_value),
        .valid_out(quant_valid_out)
    );


    // Output handling
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            for (int j = 0; j < ROWS; j++) begin
                y[j] <= '0;
            end
        end else if (state == DONE) begin
            for (int j = 0; j < ROWS; j++) begin
                y[j] <= res[j][DATA_WIDTH-1:0];  // Truncate to DATA_WIDTH bits
            end
        end
    end
    
    // Done signal generation
    assign done = (state == DONE);

endmodule
