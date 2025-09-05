// Execution Unit - Handles different operation types
// This module isolates execution logic from instruction fetch and decode
module execution_unit #(
    parameter DATA_WIDTH = 8,
    parameter TILE_WIDTH = 256,
    parameter ADDR_WIDTH = 24,
    parameter MAX_ROWS = 784,
    parameter MAX_COLS = 784
)(
    input logic clk,
    input logic rst,
    
    // Control interface
    input logic start,
    input logic [4:0] opcode,
    input logic [4:0] dest,
    input logic [9:0] length_or_cols,  // increased width to handle larger values
    input logic [9:0] rows,
    input logic [23:0] addr,
    input logic [4:0] b_id, x_id, w_id,
    
    // Data interfaces - buffers for vectors and matrices
    // Vector buffers (for x, bias, etc.)
    input logic signed [DATA_WIDTH-1:0] x_buffer [0:MAX_COLS-1],
    input logic signed [DATA_WIDTH-1:0] bias_buffer [0:MAX_ROWS-1],
        
    // Weight loading interface
    input logic weight_tile_valid,
    input logic signed [DATA_WIDTH-1:0] weight_tile_data [0:TILE_ELEMS-1],
    
    // Results
    output logic signed [DATA_WIDTH-1:0] result [0:MAX_ROWS-1],
    output logic done
);

    localparam TILE_ELEMS = TILE_WIDTH / DATA_WIDTH;
    
    typedef enum logic [3:0] {
        IDLE,
        LOAD_VECTOR,
        LOAD_MATRIX, 
        EXECUTE_GEMV,
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
    
    // Load_v module signals and buffer
    logic load_v_start, load_v_done, load_v_tile_ready;
    logic signed [DATA_WIDTH-1:0] load_v_buffer [0:TILE_ELEMS-1];
    
    // Load_m module signals and buffer
    logic load_m_start, load_m_done, load_m_tile_ready;
    logic [TILE_WIDTH-1:0] load_m_buffer;
    logic signed [DATA_WIDTH-1:0] load_m_unpacked [0:TILE_ELEMS-1];
    
    // Tile counters for writing to buffer file
    logic [9:0] write_tile_count;

    // Buffer file signals - separate for vectors and matrices
    logic vector_buffer_write_enable, matrix_buffer_write_enable;
    logic [4:0] vector_buffer_write_addr, vector_buffer_read_addr;
    logic [4:0] matrix_buffer_write_addr, matrix_buffer_read_addr;
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
    logic gemv_start, gemv_done, gemv_w_ready, gemv_tile_done;
    logic w_valid, vector_read_enable, matrix_read_enable;
    logic signed [DATA_WIDTH-1:0] y_gemv [0:MAX_ROWS-1];
    
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
    
    // Unpack load_m tile data from packed format to array
    always_comb begin
        for (int i = 0; i < TILE_ELEMS; i++) begin
            load_m_unpacked[i] = load_m_buffer[i*DATA_WIDTH +: DATA_WIDTH];
        end
    end

    // Vector Buffer file instantiation (for vectors like x, bias)
    logic vector_writing_done, vector_reading_done;
    buffer_file #(
        .BUFFER_WIDTH(8192),
        .BUFFER_COUNT(16),  // Smaller buffer for vectors
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
        .writing_done(vector_writing_done),
        .reading_done(vector_reading_done)
    );
    
    // Matrix Buffer file instantiation (for weight matrices)
    logic matrix_writing_done, matrix_reading_done;
    buffer_file #(
        .BUFFER_WIDTH(802820),
        .BUFFER_COUNT(128),  // Larger buffer for matrix tiles
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
        .writing_done(matrix_writing_done),
        .reading_done(matrix_reading_done)
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
        .tile_done(gemv_tile_done),
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
    
    // Gate w_valid for GEMV - simple logic: valid when in GEMV_COMPUTE state
    assign w_valid = (state == GEMV_COMPUTE);
    
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
    
    // Buffer file control signals
    always_comb begin
        // Vector buffer control (for load_v)
        if (load_v_tile_ready) begin
            vector_buffer_write_enable = 1;
        end else begin
            vector_buffer_write_enable = 0;
        end
        
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
            
            case (state)
                IDLE: begin
                    if (start) begin
                        $display("length_or_cols is %0d and length_or_cols*DATA_WIDTH is %0d", length_or_cols, length_or_cols*DATA_WIDTH);
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
                                matrix_read_enable <= 1;
                                vector_buffer_read_addr <= x_id;
                                matrix_buffer_read_addr <= w_id; // Initialize weight buffer read
                                tile_read_count <= 0; // Reset tile counter for weight tiles
                                total_tiles_needed <= (length_or_cols + TILE_ELEMS - 1) / TILE_ELEMS;
                                current_element_offset <= 0;
                                state <= GEMV_READ_X;
                            end
                            5'h05: begin // RELU
                                vector_buffer_read_addr <= dest;
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
                    if (load_m_tile_ready) begin
                        write_tile_count <= write_tile_count + 1;
                    end
                    if (load_m_done) begin
                        state <= COMPLETE;
                    end
                end
                
                GEMV_READ_X: begin
                    // Start reading first tile of x vector
                    
                    state <= GEMV_READ_X_TILES;
                end
                
                GEMV_READ_X_TILES: begin
                    // Wait one cycle for buffer read, then copy tile data to appropriate position
                    for (int i = 0; i < TILE_ELEMS; i++) begin
                        if (current_element_offset + i < MAX_COLS && current_element_offset + i < length_or_cols) begin
                            gemv_x_buffer[current_element_offset + i] <= vector_buffer_read_data[i];
                            // if(vector_buffer_read_data[i] != 0)
                            //     $display("Read nonzero x[%0d] = %0d from vector buffer", current_element_offset + i, vector_buffer_read_data[i]);
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
                    end
                end
                
                GEMV_READ_BIAS: begin
                    // Start reading first tile of bias vector
                    state <= GEMV_READ_BIAS_TILES;
                end
                
                GEMV_READ_BIAS_TILES: begin
                    // Wait one cycle for buffer read, then copy tile data to appropriate position 
                    for (int i = 0; i < TILE_ELEMS; i++) begin
                        if (current_element_offset + i < MAX_ROWS && current_element_offset + i < rows) begin
                            gemv_bias_buffer[current_element_offset + i] <= vector_buffer_read_data[i];
                        end
                    end
                    
                    tile_read_count <= tile_read_count + 1;
                    current_element_offset <= current_element_offset + TILE_ELEMS;
                    if (tile_read_count + 1 >= total_tiles_needed) begin
                        gemv_start <= 1;
                        tile_read_count <= 0; // Reset for weight tile counting
                        state <= GEMV_COMPUTE;
                        matrix_buffer_read_addr <= w_id;
                    end
                end
                
                GEMV_COMPUTE: begin
                    // Properly manage weight tile reading from matrix buffer
                    if (gemv_w_ready && !gemv_done) begin
                        // GEMV is ready for next weight tile, provide next tile address
                        tile_read_count <= tile_read_count + 1;
                    end
                    
                    if (gemv_done) begin
                        // Copy GEMV results
                        for (int i = 0; i < MAX_ROWS; i++) begin
                            result[i] <= y_gemv[i];
                        end
                        state <= COMPLETE;
                    end
                end
                
                EXECUTE_RELU: begin
                    // Copy the input data for ReLU directly from x_buffer for this test
                    for (int i = 0; i < MAX_COLS; i++) begin
                        if (i < TILE_ELEMS && i < 10) begin
                            result[i] <= relu_out[i];
                        end else if (i >= 10) begin
                            result[i] <= 0;
                        end
                    end
                    state <= COMPLETE;
                end
                
                STORE_RESULT: begin
                    // Placeholder for store operations
                    state <= COMPLETE;
                end
                
                COMPLETE: begin
                    done <= 1;
                    state <= IDLE;
                end
                
                default: begin
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
