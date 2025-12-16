// GEMV Execution Module
// Handles General Matrix-Vector multiplication (GEMV) operations
// Orchestrates reading x vector, bias vector, and weight matrix from buffers,
// performs computation, and writes results back to buffer
//
// Operation Flow:
// 1. Read x vector tiles from buffer
// 2. Read bias vector tiles from buffer
// 3. Read weight matrix tiles and stream to GEMV unit
// 4. Wait for GEMV computation to complete
// 5. Write results back to destination buffer

module gemv_execution #(
    parameter DATA_WIDTH = 8,
    parameter TILE_WIDTH = 256,
    parameter TILE_ELEMS = TILE_WIDTH / DATA_WIDTH,
    parameter MAX_ROWS = 1024,
    parameter MAX_COLS = 1024
)(
    input logic clk,
    input logic rst,
    
    // Control interface
    input logic start,
    input logic [4:0] dest_buffer_id,   // Destination buffer for results
    input logic [4:0] w_buffer_id,      // Weight matrix buffer
    input logic [4:0] x_buffer_id,      // Input vector buffer
    input logic [4:0] b_buffer_id,      // Bias vector buffer
    input logic [9:0] cols,             // Matrix columns (vector length)
    input logic [9:0] rows,             // Matrix rows (output length)
    output logic done,
    
    // Buffer controller interface - vector reads
    output logic vec_read_enable,
    output logic [4:0] vec_read_buffer_id,
    input logic signed [DATA_WIDTH-1:0] vec_read_tile [0:TILE_ELEMS-1],
    input logic vec_read_valid,
    
    // Buffer controller interface - matrix reads
    output logic mat_read_enable,
    output logic [4:0] mat_read_buffer_id,
    input logic signed [DATA_WIDTH-1:0] mat_read_tile [0:TILE_ELEMS-1],
    input logic mat_read_valid,
    
    // Buffer controller interface - vector writes (for results)
    output logic vec_write_enable,
    output logic [4:0] vec_write_buffer_id,
    output logic signed [DATA_WIDTH-1:0] vec_write_tile [0:TILE_ELEMS-1],
    
    // Result output (for populating result register in parent module)
    output logic signed [DATA_WIDTH-1:0] result [0:MAX_ROWS-1]
);

    // FSM states
    typedef enum logic [3:0] {
        IDLE,
        READ_X_TILES,
        READ_BIAS_TILES,
        GEMV_COMPUTE,
        PACK_RESULT_TILE,
        WRITE_RESULT_TILES,
        COMPLETE
    } gemv_state_t;
    
    gemv_state_t state;
    
    // Tile counters and tracking
    logic [9:0] tile_read_count;
    logic [9:0] total_tiles_needed;
    logic [9:0] current_element_offset;
    logic [9:0] result_tile_write_count;
    logic read_pending;  // Track if a buffer read is in flight
    
    // Local storage for GEMV inputs
    logic signed [DATA_WIDTH-1:0] gemv_x_buffer [0:MAX_COLS-1];
    logic signed [DATA_WIDTH-1:0] gemv_bias_buffer [0:MAX_ROWS-1];
    logic signed [DATA_WIDTH-1:0] gemv_result [0:MAX_ROWS-1];
    
    // GEMV unit signals
    logic gemv_start, gemv_done, gemv_w_ready, w_valid;
    logic signed [DATA_WIDTH-1:0] y_gemv [0:MAX_ROWS-1];
    
    // GEMV top_gemv module instantiation
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
        .w_tile_row_in(mat_read_tile),
        .x(gemv_x_buffer),
        .bias(gemv_bias_buffer),
        .rows(rows),
        .cols(cols),
        .y(y_gemv),
        // verilator lint_off PINCONNECTEMPTY
        .tile_done(),
        // verilator lint_on PINCONNECTEMPTY
        .done(gemv_done)
    );
    
    // w_valid signal: assert when matrix tile is valid from buffer
    assign w_valid = (state == GEMV_COMPUTE) && mat_read_valid;
    
    // Main FSM for GEMV execution
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            done <= 0;
            gemv_start <= 0;
            vec_read_enable <= 0;
            mat_read_enable <= 0;
            vec_write_enable <= 0;
            vec_read_buffer_id <= 0;
            mat_read_buffer_id <= 0;
            vec_write_buffer_id <= 0;
            tile_read_count <= 0;
            total_tiles_needed <= 0;
            current_element_offset <= 0;
            result_tile_write_count <= 0;
            read_pending <= 0;
            
            for (int i = 0; i < MAX_COLS; i++) begin
                gemv_x_buffer[i] <= 0;
            end
            for (int i = 0; i < MAX_ROWS; i++) begin
                gemv_bias_buffer[i] <= 0;
                gemv_result[i] <= 0;
                result[i] <= 0;
            end
        end else begin
            // Default signal values
            done <= 0;
            gemv_start <= 0;
            vec_read_enable <= 0;
            mat_read_enable <= 0;
            vec_write_enable <= 0;
            
            case (state)
                IDLE: begin
                    if (start) begin
                        // Initialize for reading x vector
                        vec_read_buffer_id <= x_buffer_id;
                        vec_read_enable <= 1;
                        tile_read_count <= 0;
                        total_tiles_needed <= (cols + 10'd31) / 10'd32;
                        current_element_offset <= 0;
                        state <= READ_X_TILES;
                        
                        //$display("[GEMV_EXEC] Starting GEMV: rows=%0d, cols=%0d", rows, cols);
                        //$display("[GEMV_EXEC] Reading x from buffer %0d", x_buffer_id);
                    end
                end
                
                READ_X_TILES: begin
                    // Default: don't request reads
                    vec_read_enable <= 0;
                    
                    // Capture x vector tiles as they arrive
                    if (vec_read_valid) begin
                        for (int i = 0; i < TILE_ELEMS; i++) begin
                            if (int'(current_element_offset) + i < MAX_COLS && 
                                int'(current_element_offset) + i < cols) begin
                                gemv_x_buffer[int'(current_element_offset) + i] <= vec_read_tile[i];
                            end
                        end
                        
                        tile_read_count <= tile_read_count + 1;
                        current_element_offset <= current_element_offset + 10'd32;
                        
                        //$display("[GEMV_EXEC] Read x tile %0d/%0d, data[0:7]=%d,%d,%d,%d,%d,%d,%d,%d", 
                                //  tile_read_count + 1, total_tiles_needed,
                                //  vec_read_tile[0], vec_read_tile[1], vec_read_tile[2], vec_read_tile[3],
                                //  vec_read_tile[4], vec_read_tile[5], vec_read_tile[6], vec_read_tile[7]);
                        
                        if (tile_read_count + 1 >= total_tiles_needed) begin
                            // Done reading x, start reading bias
                            vec_read_buffer_id <= b_buffer_id;
                            tile_read_count <= 0;
                            total_tiles_needed <= (rows + 10'd31) / 10'd32;
                            current_element_offset <= 0;
                            state <= READ_BIAS_TILES;
                            //$display("[GEMV_EXEC] Reading bias from buffer %0d", b_buffer_id);
                        end
                        
                        // Request next tile if more to read
                        if (tile_read_count + 1 < total_tiles_needed) begin
                            vec_read_enable <= 1;
                        end else begin
                            vec_read_enable <= 0;
                        end
                    end
                end
                
                READ_BIAS_TILES: begin
                    // Capture bias vector tiles as they arrive
                    if (vec_read_valid) begin
                        for (int i = 0; i < TILE_ELEMS; i++) begin
                            if (int'(current_element_offset) + i < MAX_ROWS && 
                                int'(current_element_offset) + i < rows) begin
                                gemv_bias_buffer[int'(current_element_offset) + i] <= vec_read_tile[i];
                            end
                        end
                        
                        tile_read_count <= tile_read_count + 1;
                        current_element_offset <= current_element_offset + 10'd32;
                        
                        //$display("[GEMV_EXEC] Read bias tile %0d/%0d", tile_read_count + 1, total_tiles_needed);
                        
                        if (tile_read_count + 1 >= total_tiles_needed) begin
                            // Done reading bias, start GEMV computation
                            gemv_start <= 1;
                            mat_read_buffer_id <= w_buffer_id;
                            tile_read_count <= 0;
                            read_pending <= 0;  // Reset for weight reads
                            // Total weight tiles = rows * tiles_per_row
                            total_tiles_needed <= rows * ((cols + 10'd31) / 10'd32);
                            state <= GEMV_COMPUTE;
                            //$display("[GEMV_EXEC] Starting GEMV compute, reading weights from buffer %0d", w_buffer_id);
                        end
                        
                        // Request next tile if more to read
                        if (tile_read_count + 1 < total_tiles_needed) begin
                            vec_read_enable <= 1;
                        end else begin
                            vec_read_enable <= 0;
                        end
                    end else begin
                        // No data yet but we need it - keep vec_read_enable high
                        // (it was set when entering this state)
                        vec_read_enable <= 1;
                    end
                end
                
                GEMV_COMPUTE: begin
                    // Clear read_pending when data arrives
                    if (mat_read_valid) begin
                        read_pending <= 0;
                    end
                    
                    // Stream weight tiles to GEMV unit when ready
                    // Only request a new read if no read is pending
                    if (gemv_w_ready && !gemv_done && !read_pending) begin
                        if (tile_read_count < total_tiles_needed) begin
                            mat_read_enable <= 1;
                            tile_read_count <= tile_read_count + 1;
                            read_pending <= 1;  // Mark read as pending
                        end
                    end
                    
                    // When GEMV completes, copy results and prepare to write back
                    if (gemv_done) begin
                        //$display("[GEMV_EXEC] GEMV computation complete, y_gemv[0:9]=%d,%d,%d,%d,%d,%d,%d,%d,%d,%d", 
                        //         y_gemv[0], y_gemv[1], y_gemv[2], y_gemv[3], y_gemv[4],
                        //         y_gemv[5], y_gemv[6], y_gemv[7], y_gemv[8], y_gemv[9]);
                        for (int i = 0; i < MAX_ROWS; i++) begin
                            gemv_result[i] <= y_gemv[i];
                            result[i] <= y_gemv[i];  // Also expose via result output
                        end
                        
                        // Prepare to write results back to buffer
                        vec_write_buffer_id <= dest_buffer_id;
                        result_tile_write_count <= 0;
                        total_tiles_needed <= (rows + 10'd31) / 10'd32;
                        current_element_offset <= 0;
                        state <= PACK_RESULT_TILE;
                        //$display("[GEMV_EXEC] Writing %0d result tiles to buffer %0d",
                        //         (rows + TILE_ELEMS - 1) / TILE_ELEMS, dest_buffer_id);
                    end
                end
                
                PACK_RESULT_TILE: begin
                    // Pack current tile from results
                    for (int i = 0; i < TILE_ELEMS; i++) begin
                        if (int'(current_element_offset) + i < MAX_ROWS && 
                            int'(current_element_offset) + i < rows) begin
                            vec_write_tile[i] <= gemv_result[int'(current_element_offset) + i];
                        end else begin
                            vec_write_tile[i] <= 0;
                        end
                    end
                    
                    // Transition to write state on next cycle
                    state <= WRITE_RESULT_TILES;
                end
                
                WRITE_RESULT_TILES: begin
                    // Write result tiles back to buffer
                    vec_write_enable <= 1;
                    
                    //$display("[GEMV_EXEC] Writing result tile %0d to buffer %0d, data[0:7]=%d,%d,%d,%d,%d,%d,%d,%d",
                    //         result_tile_write_count + 1, dest_buffer_id,
                    //         vec_write_tile[0], vec_write_tile[1], vec_write_tile[2], vec_write_tile[3],
                    //         vec_write_tile[4], vec_write_tile[5], vec_write_tile[6], vec_write_tile[7]);
                    
                    result_tile_write_count <= result_tile_write_count + 1;
                    current_element_offset <= current_element_offset + 10'd32;
                    
                    //$display("[GEMV_EXEC] Writing result tile %0d/%0d",
                    //         result_tile_write_count + 1, total_tiles_needed);
                    
                    if (result_tile_write_count + 1 >= total_tiles_needed) begin
                        state <= COMPLETE;
                    end else begin
                        state <= PACK_RESULT_TILE;  // Pack next tile
                    end
                end
                
                COMPLETE: begin
                    //$display("[GEMV_EXEC] GEMV execution complete");
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
