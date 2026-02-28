

module gemv_unit_core #(
    parameter DATA_WIDTH = 8,
    parameter MAX_ROWS = 784,
    parameter MAX_COLUMNS = 784,
    parameter TILE_SIZE = 32
) (
    input logic clk,
    input logic rst,

    // Control Signals
    input logic start,
    output logic w_ready,

    // Weight tile input (streamed per row)
    input logic w_valid,
    input logic signed [DATA_WIDTH-1:0] w_tile_row_in [0:TILE_SIZE-1],
    
    // X vector tile input (streamed)
    input logic x_tile_valid,
    input logic signed [DATA_WIDTH-1:0] x_tile_in [0:TILE_SIZE-1],
    output logic x_tile_ready,
    input logic [9:0] x_tile_idx,
    
    // Bias tile input (streamed)
    input logic bias_tile_valid,
    input logic signed [DATA_WIDTH-1:0] bias_tile_in [0:TILE_SIZE-1],
    output logic bias_tile_ready,
    input logic [9:0] bias_tile_idx,
    
    // Y tile output (streamed)
    output logic y_tile_valid,
    output logic signed [DATA_WIDTH-1:0] y_tile_out [0:TILE_SIZE-1],
    input logic y_tile_ready,
    output logic [9:0] y_tile_idx,
    
    input logic [9:0] rows,
    input logic [9:0] cols,

    output logic tile_done,
    output logic done
);

    // FSM States
    enum int unsigned { 
        IDLE            = 0, 
        LOAD_X          = 1,
        STORE_X         = 2,     // Write x tile elements to x_mem one at a time
        LOAD_BIAS       = 3,
        STORE_BIAS      = 4,     // Write bias elements to res_mem (accumulator)
        CLEAR_REMAINING = 5, // Clear accumulator rows beyond num_rows
        READ_X_TILE     = 6,
        LOAD_X_TILE     = 7,     // Read x tile from x_mem for current tile_idx
        WAIT_TILE       = 8, 
        WAIT_PE         = 9,
        READ_ACCUM      = 10,
        ACCUMULATE      = 11, 
        READ_ACCUM_2    = 12,
        ACCUMULATE_2    = 13, 
        WAIT_NEXT       = 14, 
        READ_MAX        = 15,
        FIND_MAX        = 16, 
        COMPUTE_SCALE   = 17, 
        READ_QUANTIZE   = 18,
        QUANTIZE        = 19, 
        READ_OUTPUT_Y   = 20,
        OUTPUT_Y        = 21,
        DONE_STATE      = 22
    } state, next_state;

    // Index widths
    localparam ROW_IDX_WIDTH  = $clog2(MAX_ROWS);
    localparam COL_IDX_WIDTH  = $clog2(MAX_COLUMNS);
    localparam TILE_IDX_WIDTH = $clog2(MAX_COLUMNS/TILE_SIZE + 1);

    // Internal registers
    logic [ROW_IDX_WIDTH-1:0] row_idx;
    logic [TILE_IDX_WIDTH-1:0] tile_idx;
    logic signed [DATA_WIDTH-1:0] w_latched [0:TILE_SIZE-1];
    
    // Current x tile for PE computation
    logic signed [DATA_WIDTH-1:0] x_current_tile [0:TILE_SIZE-1];
    
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
    logic signed [4*DATA_WIDTH-1:0] max_abs_reg;
    logic signed [4*DATA_WIDTH-1:0] current_abs;
    logic [ROW_IDX_WIDTH-1:0] max_idx;

    // Quantization
    logic signed [4*DATA_WIDTH-1:0] int32_value;
    logic signed [DATA_WIDTH-1:0] int8_value;
    logic [ROW_IDX_WIDTH-1:0] quant_in_idx, quant_out_idx;
    logic quant_valid_in, quant_valid_out;
    logic [ROW_IDX_WIDTH-1:0] output_y_idx;

    // Accumulator BRAM Interface
    logic [31:0] res_dout, res_din;
    logic [ROW_IDX_WIDTH-1:0] res_rad, res_wad;
    logic res_wre;
    
    // X-Vector RAM Interface
    logic [31:0] x_mem_dout, x_mem_din;
    logic [9:0] x_mem_wad, x_mem_rad;
    logic x_mem_wre;

    // X loading/storing registers
    logic signed [DATA_WIDTH-1:0] x_latched [0:TILE_SIZE-1];
    logic [9:0] x_store_idx;      // Global element index during STORE_X
    logic [2:0] x_store_elem;     // Element within tile during STORE_X
    logic [2:0] x_load_elem;      // Element counter during LOAD_X_TILE

    // Bias storing registers
    logic signed [DATA_WIDTH-1:0] bias_latched [0:TILE_SIZE-1];
    logic [ROW_IDX_WIDTH-1:0] bias_store_idx;   // Row index during STORE_BIAS
    logic [2:0] bias_store_elem;                 // Element within tile during STORE_BIAS
    
    // X/Bias loading counters
    logic [9:0] x_load_tile_count;
    logic [9:0] bias_load_tile_count;
    logic [9:0] total_x_tiles;
    logic [9:0] total_bias_tiles;
    
    // Y output counters
    logic [9:0] y_output_tile_count;
    logic [9:0] total_y_tiles;
    logic [2:0] y_elem_idx;

    // ================= Combinational Logic =================
    
    // Calculate starting column and valid elements
    localparam COL_START_WIDTH = TILE_IDX_WIDTH + $clog2(TILE_SIZE);
    logic [COL_START_WIDTH-1:0] col_start_full;
    assign col_start_full = tile_idx * TILE_SIZE;
    assign col_start = col_start_full[COL_IDX_WIDTH-1:0];
    assign num_current_row = (col_start < cols[COL_IDX_WIDTH-1:0]) ? 
                           ((int'(col_start) + TILE_SIZE <= cols[COL_IDX_WIDTH-1:0]) ? TILE_SIZE[9:0] : cols[COL_IDX_WIDTH-1:0] - col_start) : 
                           0; 
    
    // x input selection - use current tile
    always_comb begin
        for (int i = 0; i < TILE_SIZE; i++) begin
            x_in[i] = x_current_tile[i];
        end
    end

    // Split PE outputs
    always_comb begin
        sum_current_row = '0;
        sum_next_row = '0;
        
        for (int j = 0; j < TILE_SIZE; j++) begin
            logic signed [4*DATA_WIDTH-1:0] extended_out;
            extended_out = {{(2*DATA_WIDTH){pe_out[j][2*DATA_WIDTH-1]}}, pe_out[j]};
            if (j < num_current_row) begin
                sum_current_row += extended_out;
            end
            else if (row_overflow) begin
                sum_next_row += extended_out;
            end
        end
    end

    assign last_in_row = (int'(col_start) + TILE_SIZE >= cols[COL_IDX_WIDTH-1:0]);
    assign row_overflow = (col_start < cols[COL_IDX_WIDTH-1:0]) && (int'(col_start) + TILE_SIZE > cols[COL_IDX_WIDTH-1:0]);

    // Absolute value logic (using BRAM output)
    assign current_abs = ($signed(res_dout) >= 0) ? $signed(res_dout) : -$signed(res_dout);
    
    // X-Vector RAM (stores x values, 1 element per 32-bit word, sign-extended)
    // Instantiating Mock Gowin RAM to match physical FPGA implementation later
    Gowin_RAM16SDP_Mock x_ram (
        .dout(x_mem_dout), //output [31:0] dout
        .clka(clk), //input clka
        .cea(x_mem_wre), //input cea
        .reseta(~rst), //input reseta
        .clkb(clk), //input clkb
        .ceb(1'b1), //input ceb
        .resetb(~rst), //input resetb
        .oce(1'b1), //input oce
        .ada({2'b0, x_mem_wad}), //input [11:0] ada
        .din(x_mem_din), //input [31:0] din
        .adb({2'b0, x_mem_rad}) //input [11:0] adb
    );

    // Accumulator RAM (Shadow RAM) - also stores bias as initial values
    // Instantiating Mock Gowin RAM to match physical FPGA implementation later
    Gowin_RAM16SDP_Mock res_ram (
        .dout(res_dout), //output [31:0] dout
        .clka(clk), //input clka
        .cea(res_wre), //input cea
        .reseta(~rst), //input reseta
        .clkb(clk), //input clkb
        .ceb(1'b1), //input ceb
        .resetb(~rst), //input resetb
        .oce(1'b1), //input oce
        .ada({2'b0, res_wad}), //input [11:0] ada
        .din(res_din), //input [31:0] din
        .adb({2'b0, res_rad}) //input [11:0] adb
    );

    // X-Vector RAM Control Muxing
    always_comb begin
        x_mem_wre = 0;
        x_mem_wad = 0;
        x_mem_din = 0;
        x_mem_rad = 0;

        case(state)
            STORE_X: begin
                x_mem_wad = x_store_idx;
                x_mem_wre = 1;
                x_mem_din = {{24{x_latched[x_store_elem][DATA_WIDTH-1]}}, x_latched[x_store_elem]};
            end
            READ_X_TILE: begin
                x_mem_rad = {3'b0, tile_idx} * TILE_SIZE[9:0];
            end
            LOAD_X_TILE: begin
                x_mem_rad = {3'b0, tile_idx} * TILE_SIZE[9:0] + {7'b0, x_load_elem} + 1;
            end
            default: ;
        endcase
    end

    // Accumulator RAM Control Muxing
    // NOTE: Gowin_RAM16SDP has asynchronous read — res_dout reflects res_rad
    // in the SAME cycle. Each state that uses res_dout must set res_rad itself.
    always_comb begin
        res_wre = 0;
        res_wad = 0;
        res_din = 0;
        res_rad = 0;

        case(state)
            STORE_BIAS: begin
                res_wad = bias_store_idx;
                res_wre = 1;
                // Sign-extend int8 bias to int32 and store as initial accumulator value
                res_din = {{(3*DATA_WIDTH){bias_latched[bias_store_elem][DATA_WIDTH-1]}}, bias_latched[bias_store_elem]};
            end
            CLEAR_REMAINING: begin
                res_wad = row_idx;
                res_wre = 1;
                res_din = '0;
            end
            READ_ACCUM: begin
                res_rad = row_idx;     // 1-cycle latency pipeline read address
            end
            ACCUMULATE: begin
                res_wad = row_idx;
                res_wre = 1;
                // Bias is already pre-loaded as initial value — just accumulate MAC result
                res_din = $signed(res_dout) + sum_current_row;
                // DEBUG TRACE

            end
            READ_ACCUM_2: begin
                res_rad = row_idx + 1; // 1-cycle latency pipeline read address
            end
            ACCUMULATE_2: begin
                res_wad = row_idx + 1;
                res_wre = 1;
                res_din = $signed(res_dout) + sum_next_row;
            end
            READ_MAX: begin
                res_rad = max_idx;
            end
            FIND_MAX: begin
                res_rad = max_idx + 1;
            end
            READ_QUANTIZE: begin
                res_rad = quant_in_idx;
            end
            QUANTIZE: begin
                res_rad = quant_in_idx + 1;
                if (quant_valid_out) begin
                    res_wre = 1;
                    res_wad = quant_out_idx;
                    res_din = {{(3*DATA_WIDTH){int8_value[DATA_WIDTH-1]}}, int8_value};
                end
            end
            READ_OUTPUT_Y: begin
                res_rad = output_y_idx;
            end
            OUTPUT_Y: begin
                if (y_tile_ready || !y_tile_valid)
                    res_rad = output_y_idx + 1; // Prepare next early if pipelining
                else
                    res_rad = output_y_idx; // Stalled: re-read current element
            end
            default: ;
        endcase
    end

    
    // ================ Next State Logic ================

    always_comb begin : next_state_logic
        next_state = IDLE;
        case (state)
            IDLE: next_state = start ? LOAD_X : IDLE;

            LOAD_X: next_state = x_tile_valid ? STORE_X : LOAD_X;

            STORE_X: begin
                if (x_store_elem == TILE_SIZE - 1) begin
                    if (x_load_tile_count + 1 >= total_x_tiles)
                        next_state = LOAD_BIAS;
                    else
                        next_state = LOAD_X;
                end else
                    next_state = STORE_X;
            end

            LOAD_BIAS: next_state = bias_tile_valid ? STORE_BIAS : LOAD_BIAS;

            STORE_BIAS: begin
                if (bias_store_elem == TILE_SIZE - 1) begin
                    if (bias_load_tile_count + 1 >= total_bias_tiles)
                        next_state = CLEAR_REMAINING;
                    else
                        next_state = LOAD_BIAS;
                end else
                    next_state = STORE_BIAS;
            end

            CLEAR_REMAINING: begin
                next_state = (int'(row_idx) >= MAX_ROWS - 1) ? READ_X_TILE : CLEAR_REMAINING;
            end

            READ_X_TILE: begin
                next_state = LOAD_X_TILE;
            end

            LOAD_X_TILE: begin
                next_state = (x_load_elem == TILE_SIZE - 1) ? WAIT_TILE : LOAD_X_TILE;
            end

            WAIT_TILE: next_state = w_valid ? WAIT_PE : WAIT_TILE;
            WAIT_PE: next_state = READ_ACCUM;
            
            READ_ACCUM: next_state = ACCUMULATE;
            ACCUMULATE: next_state = row_overflow ? READ_ACCUM_2 : WAIT_NEXT;
            
            READ_ACCUM_2: next_state = ACCUMULATE_2;
            ACCUMULATE_2: next_state = WAIT_NEXT;

            WAIT_NEXT: begin
                if (last_in_row) begin
                    if (row_idx < rows[ROW_IDX_WIDTH-1:0] - 1)
                        next_state = READ_X_TILE; // Next row, reload x tile 0
                    else
                        next_state = READ_MAX;
                end else
                    next_state = READ_X_TILE; // Same row, next x tile
            end

            READ_MAX: next_state = FIND_MAX;
            FIND_MAX: next_state = int'(max_idx) < MAX_ROWS - 1 ? READ_MAX : COMPUTE_SCALE;
            
            COMPUTE_SCALE: next_state = scale_ready ? READ_QUANTIZE : COMPUTE_SCALE;
            
            READ_QUANTIZE: next_state = QUANTIZE;
            QUANTIZE: next_state = quant_valid_out ? (quant_out_idx < rows[ROW_IDX_WIDTH-1:0] - 1 ? QUANTIZE : READ_OUTPUT_Y) : QUANTIZE;
            
            READ_OUTPUT_Y: next_state = OUTPUT_Y;
            OUTPUT_Y: next_state = (y_output_tile_count >= total_y_tiles && y_tile_ready) ? DONE_STATE : OUTPUT_Y;
            
            DONE_STATE: next_state = IDLE;
            default: next_state = IDLE;
        endcase
    end
    
    always_ff@(posedge clk or posedge rst) begin
        if(rst) state <= IDLE;
        else begin
            state <= next_state;
        end
    end

    // ================= Sequential Logic =================
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            w_ready <= 0;
            tile_done <= 0;
            done <= 0;
            row_idx <= 0;
            tile_idx <= 0;
            max_idx <= 0;
            max_abs_reg <= 0;
            quant_in_idx <= 0;
            quant_out_idx <= 0;
            output_y_idx <= 0;
            x_tile_ready <= 0;
            bias_tile_ready <= 0;
            y_tile_valid <= 0;
            y_tile_idx <= 0;
            x_load_tile_count <= 0;
            bias_load_tile_count <= 0;
            y_output_tile_count <= 0;
            total_x_tiles <= 0;
            total_bias_tiles <= 0;
            total_y_tiles <= 0;
            y_elem_idx <= 0;
            quant_valid_in <= 0;
            int32_value <= 0;
            x_store_idx <= 0;
            x_store_elem <= 0;
            x_load_elem <= 0;
            bias_store_idx <= 0;
            bias_store_elem <= 0;
            for (int i = 0; i < TILE_SIZE; i++) begin
                x_current_tile[i] <= 0;
                w_latched[i] <= 0;
                y_tile_out[i] <= 0;
                x_latched[i] <= 0;
                bias_latched[i] <= 0;
            end
        end else begin
            case (state)
                IDLE: begin
                    w_ready <= 0;
                    tile_done <= 0;
                    done <= 0;
                    row_idx <= 0;
                    if (start) begin
                        total_x_tiles <= (cols + TILE_SIZE - 1) / TILE_SIZE;
                        total_bias_tiles <= (rows + TILE_SIZE - 1) / TILE_SIZE;
                        total_y_tiles <= (rows + TILE_SIZE - 1) / TILE_SIZE;
                        x_load_tile_count <= 0;
                        bias_load_tile_count <= 0;
                        y_output_tile_count <= 0;
                        x_store_idx <= 0;
                        x_store_elem <= 0;
                        bias_store_idx <= 0;
                        bias_store_elem <= 0;
                        output_y_idx <= 0;
                        y_elem_idx <= 0;
                        quant_in_idx <= 0;
                        quant_out_idx <= 0;
                        x_tile_ready <= 1;
                    end
                end
                
                // ---- X Loading: receive tile, then store elements to x_mem ----
                LOAD_X: begin
                    x_tile_ready <= 1;
                    if (x_tile_valid) begin
                        // Latch the tile data for element-by-element writing
                        for (int i = 0; i < TILE_SIZE; i++)
                            x_latched[i] <= x_tile_in[i];
                        x_store_elem <= 0;
                        x_tile_ready <= 0; // Stop accepting until stored
                    end
                end

                STORE_X: begin
                    // Write one element per cycle to x_mem
                    x_store_idx <= x_store_idx + 1;
                    x_store_elem <= x_store_elem + 1;
                    if (x_store_elem == TILE_SIZE - 1) begin
                        x_load_tile_count <= x_load_tile_count + 1;
                        if (x_load_tile_count + 1 >= total_x_tiles) begin
                            x_tile_ready <= 0;
                            bias_tile_ready <= 1;
                        end else begin
                            x_tile_ready <= 1; // Ready for next tile
                        end
                    end
                end
                
                // ---- Bias Loading: receive tile, then store to accumulator RAM ----
                LOAD_BIAS: begin
                    bias_tile_ready <= 1;
                    if (bias_tile_valid) begin
                        // Latch the bias tile for element-by-element writing
                        for (int i = 0; i < TILE_SIZE; i++)
                            bias_latched[i] <= bias_tile_in[i];
                        bias_store_elem <= 0;
                        bias_tile_ready <= 0; // Stop accepting until stored
                    end
                end

                STORE_BIAS: begin
                    // Write one bias element per cycle to res_mem (accumulator)
                    // Each element is sign-extended int8 -> int32
                    if (int'(bias_store_idx) < MAX_ROWS - 1)
                        bias_store_idx <= bias_store_idx + 1;
                    
                    bias_store_elem <= bias_store_elem + 1;
                    if (bias_store_elem == TILE_SIZE - 1) begin
                        bias_store_elem <= 0;
                        bias_load_tile_count <= bias_load_tile_count + 1;
                        if (bias_load_tile_count + 1 >= total_bias_tiles) begin
                            bias_tile_ready <= 0;
                            // Start clearing remaining rows
                            // bias_store_idx now points to first row after bias data
                            row_idx <= rows[ROW_IDX_WIDTH-1:0]; // Changed from bias_store_idx + 1
                        end else begin
                            bias_tile_ready <= 1; // Ready for next tile
                        end
                    end
                end
                
                // ---- Clear accumulator rows beyond actual bias data ----
                CLEAR_REMAINING: begin
                    if (int'(row_idx) < MAX_ROWS - 1)
                        row_idx <= row_idx + 1;
                    else begin 
                        row_idx <= 0;
                        tile_idx <= '0;
                        x_load_elem <= 0;
                        max_idx <= '0;
                        max_abs_reg <= '0;
                        quant_in_idx <= '0;
                        quant_out_idx <= '0;
                        output_y_idx <= '0;
                    end
                end

                READ_X_TILE: begin
                    x_load_elem <= 0;
                end

                // ---- Load x tile from x_mem for current tile_idx ----
                LOAD_X_TILE: begin
                    x_current_tile[x_load_elem] <= x_mem_dout[DATA_WIDTH-1:0];
                    x_load_elem <= x_load_elem + 1;
                    if (x_load_elem == TILE_SIZE - 1) begin
                        x_load_elem <= 0;
                    end
                end

                // ---- Weight processing ----
                WAIT_TILE: begin
                    w_ready <= 1; 
                    tile_done <= 0;
                    if (w_valid) begin
                        for (int i = 0; i < TILE_SIZE; i++) w_latched[i] <= w_tile_row_in[i];
                        w_ready <= 0;
                    end
                end

                WAIT_PE: tile_done <= 0;

                READ_ACCUM: tile_done <= 0;
                
                ACCUMULATE: begin
                    tile_done <= !row_overflow;
                    // Bias is already pre-loaded in res_mem — no b_extended needed
                end
                
                READ_ACCUM_2: tile_done <= 0;
                
                ACCUMULATE_2: begin
                    tile_done <= 1;
                end

                WAIT_NEXT: begin
                    tile_done <= 0;
                    if (last_in_row) begin
                        tile_idx <= '0;
                        if (row_idx < rows[ROW_IDX_WIDTH-1:0] - 1) row_idx <= row_idx + 1;
                    end else begin
                        tile_idx <= tile_idx + 1;
                    end
                    x_load_elem <= 0; // Reset for LOAD_X_TILE
                end

                READ_MAX: begin
                    // Just wait 1 cycle for BRAM read
                end
                FIND_MAX: begin
                    if (current_abs > max_abs_reg) max_abs_reg <= current_abs;
                    if (int'(max_idx) < MAX_ROWS - 1) max_idx <= max_idx + 1;
                    else if (max_abs_reg == 0) max_abs_reg <= 1;
                end

                COMPUTE_SCALE: begin
                    if (scale_ready) begin
                        quant_in_idx <= 0;
                        quant_out_idx <= 0;
                    end
                end

                READ_QUANTIZE: begin
                    // Pipeline cycle
                end
                QUANTIZE: begin
                    quant_valid_in <= 0;
                    if (int'(quant_in_idx) < MAX_ROWS) begin
                        int32_value <= res_dout;
                        quant_in_idx <= quant_in_idx + 1;
                        quant_valid_in <= (quant_in_idx < rows[ROW_IDX_WIDTH-1:0]);
                    end
                    if (quant_valid_out) begin
                        if (int'(quant_out_idx) < MAX_ROWS) quant_out_idx <= quant_out_idx + 1;
                    end
                end
                
                READ_OUTPUT_Y: begin
                    // Pipeline cycle
                end
                OUTPUT_Y: begin
                    // Output y values tile by tile
                    if (y_tile_ready || !y_tile_valid) begin
                        // Read from BRAM and output
                        y_tile_out[y_elem_idx] <= res_dout[DATA_WIDTH-1:0];
                        output_y_idx <= output_y_idx + 1;
                        y_elem_idx <= y_elem_idx + 1;
                        
                        if (y_elem_idx == TILE_SIZE - 1) begin
                            y_tile_valid <= 1;
                            y_tile_idx <= y_output_tile_count;
                            y_output_tile_count <= y_output_tile_count + 1;
                            y_elem_idx <= 0;
                        end
                    end
                    
                    if (y_tile_ready && y_tile_valid) begin
                        y_tile_valid <= 0;
                    end
                end

                DONE_STATE: begin
                    done <= 1;
                    y_tile_valid <= 0;
                end

                default: ;
            endcase
        end
    end

    // PE Instantiation
    generate
        for (genvar i = 0; i < TILE_SIZE; i++) begin : pe_array
            pe #(.DATA_WIDTH(DATA_WIDTH)) pe_inst (.clk(clk), .rst(rst), .w(w_latched[i]), .x(x_in[i]), .y(pe_out[i]));
        end
    endgenerate

    // Submodules
    scale_calculator scale_inst (.clk(clk), .reset_n(~rst), .max_abs(max_abs_reg), .start(state == COMPUTE_SCALE), .reciprocal_scale(reciprocal_scale), .ready(scale_ready));
    quantizer_pipeline quantize_inst (.clk(clk), .reset_n(~rst), .int32_value(int32_value), .reciprocal_scale(reciprocal_scale), .valid_in(quant_valid_in), .int8_value(int8_value), .valid_out(quant_valid_out));

endmodule
