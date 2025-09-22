module top_gemv #(
    parameter DATA_WIDTH = 8,
    parameter MAX_ROWS = 1024,
    parameter MAX_COLUMNS = 1024,
    parameter TILE_SIZE = 32
) (
    input logic clk,
    input logic rst,

    // Control Signals
    input logic start,
    output logic w_ready,

    // Data Inputs
    input logic w_valid,
    input logic signed [DATA_WIDTH-1:0] w_tile_row_in [0:TILE_SIZE-1],
    input logic signed [DATA_WIDTH-1:0] x [0:MAX_COLUMNS-1],
    input logic signed [DATA_WIDTH-1:0] bias [0:MAX_ROWS-1],
    input logic [9:0] rows,
    input logic [9:0] cols,

    // Data Outputs
    output logic signed [DATA_WIDTH-1:0] y [0:MAX_ROWS-1],
    output logic tile_done,
    output logic done
);

    // FSM States
    enum int unsigned { IDLE = 0, WAIT_TILE = 1, ACCUMULATE = 2, WAIT_NEXT = 3, BIAS = 4, 
    FIND_MAX = 5, COMPUTE_SCALE = 6, QUANTIZE = 7, DONE = 8, WAIT_PE = 9 } state, next_state;

    // Index widths
    localparam ROW_IDX_WIDTH  = $clog2(MAX_ROWS);
    localparam COL_IDX_WIDTH  = $clog2(MAX_COLUMNS);
    localparam TILE_IDX_WIDTH = $clog2(MAX_COLUMNS/TILE_SIZE + 1);

    // Internal registers
    logic signed [4*DATA_WIDTH-1:0] res [0:MAX_ROWS-1];
    logic [ROW_IDX_WIDTH-1:0] row_idx;
    logic [TILE_IDX_WIDTH-1:0] tile_idx;
    logic signed [DATA_WIDTH-1:0] w_latched [0:TILE_SIZE-1];
    
    // PE connections
    logic signed [DATA_WIDTH-1:0] x_in [0:TILE_SIZE-1];
    logic signed [2*DATA_WIDTH-1:0] pe_out [0:TILE_SIZE-1];
    
    // Tile boundary detection
    logic last_in_row;
    logic row_overflow;
    logic [COL_IDX_WIDTH-1:0] col_start;
    logic [COL_IDX_WIDTH-1:0] num_current_row;
    
    // Split accumulations
    logic signed [4*DATA_WIDTH-1:0] sum_current_row;
    logic signed [4*DATA_WIDTH-1:0] sum_next_row;
    
    // Scale computation
    logic signed [4*DATA_WIDTH-1:0] reciprocal_scale;
    logic scale_ready;
    logic signed [4*DATA_WIDTH-1:0] max_abs_reg, current_abs;
    logic [ROW_IDX_WIDTH-1:0] max_idx;

    // Quantization
    logic signed [4*DATA_WIDTH-1:0] int32_value;
    logic signed [DATA_WIDTH-1:0] int8_value;
    logic [ROW_IDX_WIDTH-1:0] quant_in_idx, quant_out_idx;
    logic quant_valid_in, quant_valid_out;

    // ================= Combinational Logic =================
    
    // Calculate starting column and valid elements
    assign col_start = {tile_idx * TILE_SIZE}[COL_IDX_WIDTH-1:0];
    assign num_current_row = (col_start < cols[COL_IDX_WIDTH-1:0]) ? 
                           ((int'(col_start) + TILE_SIZE <= cols[COL_IDX_WIDTH-1:0]) ? TILE_SIZE[9:0] : cols[COL_IDX_WIDTH-1:0] - col_start) : 
                           0; // strange behaviour in verilator warning without int()
    
    // x input selection with zero-padding
    always_comb begin
        for (int i = 0; i < TILE_SIZE; i++) begin
            // Current row elements
            if (i < num_current_row) begin
                x_in[i] = x[int'(col_start) + i];
                // if(x[int'(col_start) + i]!= 0)
                //     $display("Using nonzero x[%0d] = %0d for row %0d", int'(col_start) + i, x[int'(col_start) + i], row_idx);
            end 
            // Next row elements (if spanning)
            else if (row_overflow) begin
                x_in[i] = x[i - int'(num_current_row)];
            end 
            // Beyond matrix columns
            else begin
                x_in[i] = '0;
            end
        end
    end

    // Split PE outputs into current and next row accumulations
    always_comb begin
        sum_current_row = '0;
        sum_next_row = '0;
        
        for (int j = 0; j < TILE_SIZE; j++) begin
            logic signed [4*DATA_WIDTH-1:0] extended_out;
            extended_out = {{(2*DATA_WIDTH){pe_out[j][2*DATA_WIDTH-1]}}, pe_out[j]};
            // if(extended_out != 0)
            //     $display("PE output[%0d] = %0d for row %0d", j, extended_out, row_idx);
            if (j < num_current_row) begin
                sum_current_row += extended_out;
            end
            else if (row_overflow) begin
                sum_next_row += extended_out;
            end
        end
    end

    // Tile boundary conditions
    assign last_in_row = (int'(col_start) + TILE_SIZE >= cols[COL_IDX_WIDTH-1:0]);
    assign row_overflow = (col_start < cols[COL_IDX_WIDTH-1:0]) && (int'(col_start) + TILE_SIZE > cols[COL_IDX_WIDTH-1:0]);

    // Absolute value for max calculation
    assign current_abs = (res[max_idx] >= 0) ? res[max_idx] : -res[max_idx];
    
    // ================ Next State Logic ================
    always_comb begin : next_state_logic
        next_state = IDLE;
        case (state)
            IDLE: next_state = start ? WAIT_TILE : IDLE;
            WAIT_TILE: next_state = w_valid ? WAIT_PE : WAIT_TILE;
            WAIT_PE: next_state = ACCUMULATE;
            ACCUMULATE: next_state = WAIT_NEXT;
            WAIT_NEXT: begin
                if (last_in_row) begin
                    if (row_idx < rows[ROW_IDX_WIDTH-1:0] - 1) begin
                        next_state = WAIT_TILE;
                    end else begin
                        next_state = BIAS;
                    end
                end else begin
                    next_state = WAIT_TILE;
                end
            end
            BIAS: next_state = FIND_MAX;
            FIND_MAX: next_state = int'(max_idx) < MAX_ROWS - 1 ? FIND_MAX : COMPUTE_SCALE;
            COMPUTE_SCALE: next_state = scale_ready ? QUANTIZE : COMPUTE_SCALE;
            QUANTIZE: next_state = quant_valid_out ? (quant_out_idx < rows[ROW_IDX_WIDTH-1:0] - 1 ? QUANTIZE : DONE) : QUANTIZE;
            DONE: next_state = IDLE;
            default: next_state = IDLE; // Handle unexpected states
        endcase
    end
    always_ff@(posedge clk or posedge rst) begin
        if(rst)
            state <= IDLE;
        else
            state <= next_state;
    end
    // ================= Sequential Logic =================
    always_ff @(posedge clk) begin
        case (state)
                IDLE: begin
                    w_ready <= 0;
                    tile_done <= 0;
                    done <= 0;
                    quant_valid_in <= 0;
                    //$display("IDLE state, waiting for start signal");
                    if (start) begin
                        // Initialize counters and registers
                        row_idx <= '0;
                        tile_idx <= '0;
                        max_idx <= '0;
                        max_abs_reg <= '0;
                        quant_in_idx <= '0;
                        quant_out_idx <= '0;
                        
                        // Clear result registers
                        for (int j = 0; j < MAX_ROWS; j++) begin
                            res[j] <= '0;
                        end
                        
                        //$display("Starting GEMV operation with %0d rows and %0d columns", rows, cols);
                    end
                end

                WAIT_TILE: begin
                    w_ready <= 1;  // Signal ready for weights
                    tile_done <= 0;
                    //$display("Waiting for tile %0d for row %0d, state is %d", tile_idx, row_idx, state);
                    if (w_valid) begin
                        // Latch incoming weights (unpacked array)
                        for (int i = 0; i < TILE_SIZE; i++) begin
                            w_latched[i] <= w_tile_row_in[i];
                            // if(w_tile_row_in[i] != 0) 
                            //     $display("Received nonzero weight[%0d] = %0d for row %0d", i, w_tile_row_in[i], row_idx);
                        end
                        w_ready <= 0;
                        //$display("Received tile %0d for row %0d", tile_idx, row_idx);
                    end
                end

                WAIT_PE: begin
                    // Allow one cycle for PE outputs to settle (in case PE is registered)
                    tile_done <= 0;
                end

                ACCUMULATE: begin
                    res[row_idx] <= res[row_idx] + sum_current_row;
                    // if(sum_current_row != 0)
                    //     $display("Row %0d, Tile %0d: Accumulated tile %0d for row %0d, sum_current_row = %0d", row_idx, tile_idx, tile_idx, row_idx, sum_current_row);
                    if (row_overflow) begin
                        res[row_idx+1] <= res[row_idx+1] + sum_next_row;
                        //$display("Row overflow: accumulated tile %0d for row %0d and next row %0d", tile_idx, row_idx, row_idx+1);
                    end
                    tile_done <= 1;
                    //$display("Row %0d, Tile %0d: Accumulated result = %0d", row_idx, tile_idx, res[row_idx]);            
                end
                WAIT_NEXT: begin
                    tile_done <= 0;
                    if (last_in_row) begin
                        tile_idx <= '0;
                        //$display("Last tile in row: tile_idx = %d, cols = %d", tile_idx, cols);
                        if (row_idx < rows[ROW_IDX_WIDTH-1:0] - 1) begin
                            // Move to next row (skip next if already accumulated)
                            row_idx <= row_idx + 1;
                        end
                    end else begin
                        tile_idx <= tile_idx + 1;
                    end
                end
                BIAS: begin
                    tile_done <= 0;
                    // Add bias to all rows
                    for (int j = 0; j < MAX_ROWS; j++) begin
                        if (j < rows) begin
                            res[j] <= res[j] + {{(3*DATA_WIDTH){bias[j][DATA_WIDTH-1]}}, bias[j]};
                            //$display("Bias applied to row %0d: %0d", j, res[j]);
                        end
                        else begin
                            res[j] <= '0; // Zero padding for unused rows
                        end
                    end
                    max_idx <= 0;
                    max_abs_reg <= 0;
                end

                FIND_MAX: begin
                    tile_done <= 0;
                    // Find maximum absolute value in result vector
                    if (current_abs > max_abs_reg) begin
                        max_abs_reg <= current_abs;
                    end

                    if (int'(max_idx) < MAX_ROWS-1) begin
                        max_idx <= max_idx + 1;
                    end else begin
                        // Avoid division by zero
                        if (max_abs_reg == 0) max_abs_reg <= 1;
                    end
                end

                COMPUTE_SCALE: begin
                    tile_done <= 0;
                    // Wait for scale computation to complete
                    if (scale_ready) begin
                        quant_in_idx <= 0;
                        quant_out_idx <= 0;
                    end
                end

                QUANTIZE: begin
                    tile_done <= 0;
                    // Pipeline quantization process
                    quant_valid_in <= 0;
                    if (int'(quant_in_idx) < MAX_ROWS) begin
                        int32_value <= res[quant_in_idx];
                        quant_in_idx <= quant_in_idx + 1;
                        quant_valid_in <= (quant_in_idx < rows[ROW_IDX_WIDTH-1:0]);
                    end

                    if (quant_valid_out) begin
                        if (int'(quant_out_idx) < MAX_ROWS) begin
                            res[quant_out_idx] <= {{(3*DATA_WIDTH){int8_value[DATA_WIDTH-1]}}, int8_value};
                            quant_out_idx <= quant_out_idx + 1;
                        end
                        
                    end
                end

                DONE: begin
                    tile_done <= 0;
                    done <= 1;
                    $display("hardware results are: ");
                    // for (int j = 0; j < rows; j++) begin
                    //     $display("y[%0d] = %0d", j, res[j]);
                    // end
                    //$display("GEMV operation completed. Results ready.");
                end

                default: begin
                    tile_done <= 0;
                    done <= 0;
                    //$display("Unexpected state: %0d", state);
                end
        endcase
    end

    // ================= PE Array Instantiation =================
    generate
        for (genvar i = 0; i < TILE_SIZE; i++) begin : pe_array
            pe #(
                .DATA_WIDTH(DATA_WIDTH)
            ) pe_inst (
                .clk(clk),
                .rst(rst),
                .w(w_latched[i]),
                .x(x_in[i]),
                .y(pe_out[i])
            );
        end
    endgenerate

    // ================= Sub-module Instantiations =================
    
    // Scale calculator
    scale_calculator scale_inst (
        .clk(clk),
        .reset_n(~rst),
        .max_abs(max_abs_reg),
        .start(state == COMPUTE_SCALE),
        .reciprocal_scale(reciprocal_scale),
        .ready(scale_ready)
    );

    // Quantization pipeline
    quantizer_pipeline quantize_inst (
        .clk(clk),
        .reset_n(~rst),
        .int32_value(int32_value),
        .reciprocal_scale(reciprocal_scale),
        .valid_in(quant_valid_in),
        .int8_value(int8_value),
        .valid_out(quant_valid_out)
    );

    // ================= Output Assignment =================
    always_ff @(posedge clk) begin
        if (state == DONE) begin
            for (int j = 0; j < MAX_ROWS; j++) begin
                if (j < rows) begin
                    y[j] <= res[j][DATA_WIDTH-1:0];
                end else begin
                    y[j] <= '0;
                end
            end
        end
    end

endmodule
