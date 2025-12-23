// Execution Unit - Handles different operation types
// This module isolates execution logic from instruction fetch and decode
module execution_unit #(
    parameter DATA_WIDTH = 8,
    parameter TILE_WIDTH = 256,
    parameter ADDR_WIDTH = 24,
    parameter MAX_ROWS = 1024,
    parameter MAX_COLS = 1024
)(
    input logic clk,
    input logic rst,
    
    // Control interface
    input logic start,
    input logic [4:0] opcode,
    input logic [4:0] dest,
    input logic [9:0] length_or_cols,  // increased width to handle larger values
    input logic [9:0] rows,
    input logic [ADDR_WIDTH-1:0] addr,
    // verilator lint_off UNUSED
    input logic [4:0] b_id, x_id, w_id,
    // verilator lint_on UNUSED

    
    // Results
    output logic signed [DATA_WIDTH-1:0] result [0:MAX_ROWS-1],
    output logic done
);

    localparam TILE_ELEMS = TILE_WIDTH / DATA_WIDTH;
    
    typedef enum logic [3:0] {
        IDLE,
        LOAD_VECTOR,
        LOAD_MATRIX, 
        GEMV_READ_X,
        GEMV_READ_X_TILES,
        GEMV_READ_BIAS,
        GEMV_READ_BIAS_TILES,
        GEMV_COMPUTE,
        EXECUTE_RELU,
        STORE_RESULT,
        COMPLETE
    } exec_state_t;
    
    exec_state_t state;
    // wires for debugging
    // logic nonzero_load_v;
    // Load_v module signals and buffer
    logic load_v_start, load_v_done, load_v_tile_ready;
    logic signed [DATA_WIDTH-1:0] load_v_buffer [0:TILE_ELEMS-1];
    
    // Load_m module signals and buffer
    logic load_m_start, load_m_done, load_m_tile_ready;
    logic [TILE_WIDTH-1:0] load_m_buffer;
    // logic signed [DATA_WIDTH-1:0] load_m_unpacked [0:TILE_ELEMS-1];
    
    // Tile counters for writing to buffer file
    logic [9:0] write_tile_count;

    // Buffer file signals - separate for vectors and matrices
    logic vector_buffer_write_enable, matrix_buffer_write_enable;
    // verilator lint_off UNUSED
    logic [4:0] vector_buffer_read_addr;
    // verilator lint_on UNUSED
    logic [4:0] matrix_buffer_read_addr;
    logic [TILE_WIDTH-1:0] vector_buffer_write_tile, matrix_buffer_write_tile;
    logic signed [DATA_WIDTH-1:0] vector_buffer_read_data [0:TILE_ELEMS-1];
    logic signed [DATA_WIDTH-1:0] matrix_buffer_read_data [0:TILE_ELEMS-1];
    
    // Tile reading counters and control
    logic [9:0] tile_read_count, total_tiles_needed;
    logic [9:0] current_element_offset;
    
    // Local storage for GEMV inputs
    logic signed [DATA_WIDTH-1:0] gemv_x_buffer [0:MAX_COLS-1];
    logic signed [DATA_WIDTH-1:0] gemv_bias_buffer [0:MAX_ROWS-1];
    
    // GEMV signals
    logic gemv_start, gemv_done, gemv_w_ready;
    logic w_valid, vector_read_enable, matrix_read_enable;
    logic signed [DATA_WIDTH-1:0] y_gemv [0:MAX_ROWS-1];
    // Delayed read-enable signals to sample buffer_file outputs one cycle after read_enable is asserted
    logic vector_read_enable_d, matrix_read_enable_d;

    // ReLU signals
    logic signed [DATA_WIDTH-1:0] relu_out [0:MAX_ROWS-1];

    // Load_v module instantiation
    load_v #(
        .TILE_WIDTH(TILE_WIDTH),
        .DATA_WIDTH(DATA_WIDTH)
    ) load_v_inst (
        .clk(clk),
        .rst(rst),
        .valid_in(load_v_start),
        .dram_addr(addr),
        .length(length_or_cols), 
        .data_out(load_v_buffer),
        .tile_out(load_v_tile_ready),
        .valid_out(load_v_done)
    );
    
    // Load_m module instantiation
    load_m #(
        .TILE_WIDTH(TILE_WIDTH)
    ) load_m_inst (
        .clk(clk),
        .rst(rst),
        .valid_in(load_m_start),
        .dram_addr(addr),
        .length(rows * length_or_cols),
        .data_out(load_m_buffer),
        .tile_out(load_m_tile_ready),
        .valid_out(load_m_done)
    );

    // Vector Buffer file instantiation (for vectors like x, bias)
    buffer_file #(
        .BUFFER_WIDTH(8192),
        .BUFFER_COUNT(32),  // Smaller buffer for vectors
        .TILE_WIDTH(TILE_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .TILE_SIZE(TILE_ELEMS)
    ) vector_buffer_inst (
        .clk(clk),
        .reset_n(~rst),
        .write_enable(vector_buffer_write_enable),
        .read_enable(vector_read_enable),
        .write_data(vector_buffer_write_tile),
        .write_buffer(dest),
        .read_buffer(vector_buffer_read_addr),
        .read_data(vector_buffer_read_data),
        .reset_indices_enable(1'b0),
        .reset_indices_buffer(5'b0),
        // verilator lint_off PINCONNECTEMPTY
        .writing_done(),
        .reading_done()
        // verilator lint_on PINCONNECTEMPTY
    );
    
    // Matrix Buffer file instantiation (for weight matrices)
    buffer_file #(
        .BUFFER_WIDTH(802820),
        .BUFFER_COUNT(32),  // Larger buffer for matrix tiles
        .TILE_WIDTH(TILE_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .TILE_SIZE(TILE_ELEMS)
    ) matrix_buffer_inst (
        .clk(clk),
        .reset_n(~rst),
        .write_enable(matrix_buffer_write_enable),
        .read_enable(matrix_read_enable),
        .write_data(matrix_buffer_write_tile),
        .write_buffer(dest),
        .read_buffer(matrix_buffer_read_addr),
        .read_data(matrix_buffer_read_data),
        .reset_indices_enable(1'b0),
        .reset_indices_buffer(5'b0),
        // verilator lint_off PINCONNECTEMPTY
        .writing_done(),
        .reading_done()
        // verilator lint_on PINCONNECTEMPTY
    );
    
    // GEMV instance - matrix buffer data is already properly formatted as array
    top_gemv #(
        .DATA_WIDTH(DATA_WIDTH),
        .MAX_ROWS(MAX_ROWS),
        .MAX_COLUMNS(MAX_COLS),
        .TILE_SIZE(TILE_ELEMS)
    ) gemv_unit (
        .clk(clk),
        .rst(rst),
        .start(gemv_start),
        .w_ready(gemv_w_ready),
        .w_valid(w_valid),
        .w_tile_row_in(matrix_buffer_read_data), // Use matrix buffer data directly
        .x(gemv_x_buffer),
        .bias(gemv_bias_buffer),
        .rows(rows),
        .cols(length_or_cols),
        .y(y_gemv),
        // verilator lint_off PINCONNECTEMPTY
        .tile_done(), // we just don't need this signal here
        // verilator lint_on PINCONNECTEMPTY
        .done(gemv_done)
    );
    
    // Select ReLU input from vector buffer file
    logic signed [DATA_WIDTH-1:0] relu_input [0:MAX_COLS-1];
    always_comb begin
        for (int i = 0; i < MAX_COLS; i++) begin
            if (i < TILE_ELEMS) begin
                relu_input[i] = vector_buffer_read_data[i];
            end else begin
                relu_input[i] = '0;
            end
        end
    end
    
    // ReLU instance
    relu #(
        .DATA_WIDTH(DATA_WIDTH),
        .LENGTH(MAX_COLS)
    ) relu_unit (
        .in_vec(relu_input),
        .out_vec(relu_out)
    );
    
    // Gate w_valid for GEMV - valid when we have matrix buffer data ready after read request
    assign w_valid = (state == GEMV_COMPUTE) && matrix_read_enable_d;
    
    // Buffer file control - separate control for vector and matrix buffers
    always_comb begin
        // Pack vector tile data for writing
        vector_buffer_write_tile = '0;
        if (load_v_tile_ready) begin
            for (int i = 0; i < TILE_ELEMS; i++) begin
                vector_buffer_write_tile[i*DATA_WIDTH +: DATA_WIDTH] = load_v_buffer[i];
            end
        end
        
        // Pack matrix tile data for writing  
        matrix_buffer_write_tile = '0;
        if (load_m_tile_ready) begin
            matrix_buffer_write_tile = load_m_buffer;
        end
    end
    //reg load_v_tile_ready_d;
    //always_ff @(posedge clk) load_v_tile_ready_d <= load_v_tile_ready;
    //wire tile_ready_pulse = load_v_tile_ready & ~load_v_tile_ready_d;

    always_comb vector_buffer_write_enable = load_v_tile_ready;

    // Buffer file control signals
    always_comb begin
        // Vector buffer control (for load_v)
        // if (load_v_tile_ready) begin
        //     vector_buffer_write_enable = 1;
        // end else begin
        //     vector_buffer_write_enable = 0;
        // end
        
        // Matrix buffer control (for load_m)
        if (load_m_tile_ready) begin
            matrix_buffer_write_enable = 1;
        end else begin
            matrix_buffer_write_enable = 0;
        end
    end
    
    // Main execution FSM
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            done <= 0;
            gemv_start <= 0;
            load_v_start <= 0;
            load_m_start <= 0;
            vector_read_enable <= 0;
            matrix_read_enable <= 0;
            vector_buffer_read_addr <= 0;
            matrix_buffer_read_addr <= 0;
            tile_read_count <= 0;
            total_tiles_needed <= 0;
            current_element_offset <= 0;
            write_tile_count <= 0;
            for (int i = 0; i < MAX_ROWS; i++) begin
                result[i] <= 0;
            end
        end else begin
            // Default values
            done <= 0;
            gemv_start <= 0;
            load_v_start <= 0;
            load_m_start <= 0;
            // Default read enables to 0; will be explicitly asserted in states that need them
            vector_read_enable <= 0;
            matrix_read_enable <= 0;

            case (state)
                IDLE: begin
                    if (start) begin
                        // $display("length_or_cols is %0d and length_or_cols*DATA_WIDTH is %0d", length_or_cols, length_or_cols*DATA_WIDTH);
                        // $display("opcode is %0h", opcode);
                        //nonzero_load_v <= 0;
                        case (opcode)
                            5'h00: begin // NOP
                                state <= COMPLETE;
                            end
                            5'h01: begin // LOAD_V
                                load_v_start <= 1;
                                write_tile_count <= 0;
                                state <= LOAD_VECTOR;
                            end
                            5'h02: begin // LOAD_M
                                load_m_start <= 1;
                                write_tile_count <= 0;
                                state <= LOAD_MATRIX;
                            end
                            5'h03: begin // STORE (placeholder)
                                state <= COMPLETE;
                            end
                            5'h04: begin // GEMV
                                vector_read_enable <= 1;
                                vector_buffer_read_addr <= x_id;
                                matrix_buffer_read_addr <= w_id; // Initialize weight buffer read
                                tile_read_count <= 0; // Reset tile counter for weight tiles
                                total_tiles_needed <= (length_or_cols + TILE_ELEMS - 1) / TILE_ELEMS;
                                current_element_offset <= 0;
                                state <= GEMV_READ_X;
                            end
                            5'h05: begin // RELU
                                vector_read_enable <= 1;
                                vector_buffer_read_addr <= dest;
                                tile_read_count <= 0;
                                total_tiles_needed <= (length_or_cols + TILE_ELEMS - 1) / TILE_ELEMS;
                                current_element_offset <= 0;
                                state <= EXECUTE_RELU;
                            end
                            default: begin
                                state <= COMPLETE;
                            end
                        endcase
                    end
                end
                
                LOAD_VECTOR: begin
                    // Increment tile count when a tile is written
                    //$display("In LOAD_VECTOR state");
                    if (load_v_tile_ready) begin
                        write_tile_count <= write_tile_count + 1;
                    end
                    if (load_v_done) begin
                        state <= COMPLETE;
                    end
                end
                
                LOAD_MATRIX: begin
                    // Increment tile count when a tile is written
                    //$display("In LOAD_MATRIX state");
                    if (load_m_tile_ready) begin
                        write_tile_count <= write_tile_count + 1;
                    end
                    if (load_m_done) begin
                        state <= COMPLETE;
                    end
                end
                
                GEMV_READ_X: begin
                    // Start reading first tile of x vector
                    //$display("In GEMV_READ_X state");
                    vector_read_enable <= 1;  // Pulse read enable for first x tile
                    state <= GEMV_READ_X_TILES;
                end
                
                GEMV_READ_X_TILES: begin
                    // Default: don't assert read_enable (edge-triggered)
                    vector_read_enable <= 0;
                    
                    // Wait one cycle for buffer read, then copy tile data to appropriate position
                    //$display("In GEMV_READ_X_TILES state, tile_read_count=%0d", tile_read_count);
                    for (int i = 0; i < TILE_ELEMS; i++) begin
                        // Use delayed read-enable to capture the tile output from buffer_file
                        if (vector_read_enable_d && int'(current_element_offset) + i < MAX_COLS && int'(current_element_offset) + i < length_or_cols) begin
                            gemv_x_buffer[int'(current_element_offset) + i] <= vector_buffer_read_data[i];
                            // if (vector_buffer_read_data[i] != 0) begin
                            //     nonzero_load_v <= 1;
                            //     $display("Read nonzero x[%0d] = %0d from vector buffer", current_element_offset + i, vector_buffer_read_data[i]);
                            // end
                        end
                    end
                    
                    tile_read_count <= tile_read_count + 1;
                    current_element_offset <= current_element_offset + TILE_ELEMS;
                    if(tile_read_count + 1 >= total_tiles_needed) begin
                        vector_buffer_read_addr <= b_id;
                        tile_read_count <= 0;
                        total_tiles_needed <= (rows + TILE_ELEMS - 1) / TILE_ELEMS; // Bias vector tiles
                        current_element_offset <= 0;
                        state <= GEMV_READ_BIAS;
                    end else begin
                        // Pulse vector_read_enable for next tile
                        vector_read_enable <= 1;
                    end
                end
                
                GEMV_READ_BIAS: begin
                    // Start reading first tile of bias vector
                    //$display("In GEMV_READ_BIAS state");
                    vector_read_enable <= 1;  // Pulse read enable for first bias tile
                    state <= GEMV_READ_BIAS_TILES;
                end
                
                GEMV_READ_BIAS_TILES: begin
                    // Default: don't assert read_enable (edge-triggered)
                    vector_read_enable <= 0;
                    
                    // Wait one cycle for buffer read, then copy tile data to appropriate position
                    //$display("In GEMV_READ_BIAS_TILES state, tile_read_count=%0d", tile_read_count); 
                    for (int i = 0; i < TILE_ELEMS; i++) begin
                        // Use delayed read-enable to capture the tile output from buffer_file
                        if (vector_read_enable_d && int'(current_element_offset) + i < MAX_ROWS && {int'(current_element_offset)+i} < rows) begin
                            gemv_bias_buffer[int'(current_element_offset) + i] <= vector_buffer_read_data[i];
                            // if(vector_buffer_read_data[i] != 0) begin
                            //     $display("Read nonzero bias[%0d] = %0d from vector buffer", current_element_offset + i, vector_buffer_read_data[i]);
                            // end
                        end
                    end
                    
                    tile_read_count <= tile_read_count + 1;
                    current_element_offset <= current_element_offset + TILE_ELEMS;
                    if (tile_read_count + 1 >= total_tiles_needed) begin
                        gemv_start <= 1;
                        tile_read_count <= 0; // Reset for weight tile counting
                        // Calculate total weight tiles needed: rows × tiles_per_row
                        total_tiles_needed <= rows * ((length_or_cols + TILE_ELEMS - 1) / TILE_ELEMS);
                        state <= GEMV_COMPUTE;
                        matrix_buffer_read_addr <= w_id;
                        matrix_read_enable <= 1; // Pulse for first weight tile
                        
                    end else begin
                        // Pulse vector_read_enable for next tile
                        vector_read_enable <= 1;
                    end
                end
                
                GEMV_COMPUTE: begin
                    // Default: don't assert read_enable (edge-triggered)
                    matrix_read_enable <= 0;
                    
                    // Provide weight tiles when GEMV is ready and we haven't sent all tiles
                    if (gemv_w_ready && !gemv_done) begin
                        // Check if we have more tiles to send
                        if (tile_read_count < total_tiles_needed) begin
                            matrix_read_enable <= 1;  // Pulse for one cycle
                            tile_read_count <= tile_read_count + 1;
                            //$display("Providing weight tile %0d of %0d to GEMV", tile_read_count + 1, total_tiles_needed);
                        end else begin
                            // All tiles sent, but GEMV not done - this indicates an error
                            //$display("ERROR: All %0d tiles sent but GEMV not done! Forcing completion.", total_tiles_needed);
                            state <= COMPLETE;
                        end
                    end
                    
                    if (gemv_done) begin
                        // Copy GEMV results
                        // $display("Starting GEMV with biases: ");
                        // for (int i = 0; i < MAX_ROWS; i++) begin
                        //     if (gemv_bias_buffer[i] != 0) begin
                        //         $display("non-zero bias[%0d] = %0d", i, gemv_bias_buffer[i]);
                        //     end
                        // end

                        // $display("Starting GEMV with inputs: ");
                        // for (int i = 0; i < MAX_COLS; i++) begin
                        //     if (gemv_x_buffer[i] != 0) begin
                        //         $display("non-zero input[%0d] = %0d", i, gemv_x_buffer[i]);
                        //     end
                        // end

                        // $display("GEMV done, copying results.");
                        for (int i = 0; i < MAX_ROWS; i++) begin
                            result[i] <= y_gemv[i];
                            if (y_gemv[i] != 0) begin
                                // $display("non-zero result[%0d] = %0d", i, y_gemv[i]);
                            end
                        end

                        state <= COMPLETE;
                    end
                end
                
                EXECUTE_RELU: begin
                    // Default: don't assert read_enable (edge-triggered)
                    vector_read_enable <= 0;
                    
                    // Read the GEMV results from buffer and apply ReLU using relu_out from relu module
                    //$display("In EXECUTE_RELU state, tile_read_count=%0d", tile_read_count);
                    
                    // Copy data from buffer to result with ReLU applied from relu_out
                    // The relu module processes vector_buffer_read_data combinationally via relu_input
                    for (int i = 0; i < TILE_ELEMS; i++) begin
                        if (vector_read_enable_d && int'(current_element_offset) + i < MAX_ROWS && int'(current_element_offset) + i < length_or_cols) begin
                            // Use ReLU output from the relu module
                            result[int'(current_element_offset) + i] <= relu_out[i];
                            
                            if (vector_buffer_read_data[i] != 0) begin
                                // $display("ReLU input[%0d] = %0d, output = %0d", 
                                //         current_element_offset + i, 
                                //         vector_buffer_read_data[i],
                                //         relu_out[i]);
                            end
                        end
                    end
                    
                    tile_read_count <= tile_read_count + 1;
                    current_element_offset <= current_element_offset + TILE_ELEMS;
                    
                    if (tile_read_count + 1 >= total_tiles_needed) begin
                        //$display("ReLU execution done");
                        // for (int i = 0; i < length_or_cols; i++) begin
                        //     if (result[i] != 0) begin
                        //         $display("non-zero ReLU result[%0d] = %0d", i, result[i]);
                        //     end
                        // end
                        state <= COMPLETE;
                    end else begin
                        // Pulse vector_read_enable for next tile
                        vector_read_enable <= 1;
                    end
                end
                
                STORE_RESULT: begin
                    // Placeholder for store operations
                    //$display("In STORE_RESULT state");
                    state <= COMPLETE;
                end
                
                COMPLETE: begin
                    //$display("In COMPLETE state, execution done.");
                    done <= 1;
                    state <= IDLE;
                end
                
                default: begin
                    state <= IDLE;
                end
            endcase
            // Update delayed versions of read-enable signals (capture current cycle values)
            vector_read_enable_d <= vector_read_enable;
            matrix_read_enable_d <= matrix_read_enable;
        end
    end
    

endmodule
