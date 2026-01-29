// ReLU Execution Module
// Handles ReLU (Rectified Linear Unit) activation operations
// Reads input vector from buffer, applies ReLU, writes result to destination buffer
//
// Operation Flow:
// 1. Read input vector tiles from source buffer
// 2. Apply ReLU activation (max(0, x)) combinationally
// 3. Write activated results to destination buffer
//
// CRITICAL FIX: This module correctly reads from source buffer (x_buffer_id)
// and writes to destination buffer (dest_buffer_id), unlike the original
// execution_unit which incorrectly read from dest.

module relu_execution #(
    parameter DATA_WIDTH = accelerator_config_pkg::DATA_WIDTH,
    parameter TILE_WIDTH = accelerator_config_pkg::TILE_WIDTH,
    parameter TILE_ELEMS = accelerator_config_pkg::TILE_ELEMS,
    parameter MAX_ROWS = accelerator_config_pkg::MAX_ROWS
)(
    input logic clk,
    input logic rst,
    
    // Control interface
    input logic start,
    input logic [4:0] dest_buffer_id,   // Destination buffer for results
    input logic [4:0] x_buffer_id,      // Source buffer to read from
    input logic [9:0] length,           // Vector length (number of elements)
    output logic done,
    
    // Buffer controller interface - vector reads
    output logic vec_read_enable,
    output logic [4:0] vec_read_buffer_id,
    input logic signed [DATA_WIDTH-1:0] vec_read_tile [0:TILE_ELEMS-1],
    input logic vec_read_valid,
    
    // Buffer controller interface - vector writes
    output logic vec_write_enable,
    output logic [4:0] vec_write_buffer_id,
    output logic signed [DATA_WIDTH-1:0] vec_write_tile [0:TILE_ELEMS-1],
    
    // Result output (for populating result register in parent module)
    output logic signed [DATA_WIDTH-1:0] result [0:MAX_ROWS-1]  // Max MAX_ROWS elements
);

    // FSM states
    typedef enum logic [1:0] {
        IDLE,
        READ_AND_WRITE_TILES,
        COMPLETE
    } relu_state_t;
    
    relu_state_t state;
    
    // Tile counters
    logic [9:0] tile_count;
    logic [9:0] total_tiles_needed;
    logic [9:0] current_element_offset;
    
    // Result accumulator
    logic signed [DATA_WIDTH-1:0] result_accumulator [0:MAX_ROWS-1];
    
    // ReLU input and output for current tile
    logic signed [DATA_WIDTH-1:0] relu_input [0:TILE_ELEMS-1];
    logic signed [DATA_WIDTH-1:0] relu_output [0:TILE_ELEMS-1];
    
    // ReLU module instantiation
    // Performs element-wise max(0, x) operation
    relu #(
        .DATA_WIDTH(DATA_WIDTH),
        .LENGTH(TILE_ELEMS)
    ) relu_unit (
        .in_vec(relu_input),
        .out_vec(relu_output)
    );
    
    // Connect read tile to ReLU input
    always_comb begin
        relu_input = vec_read_tile;
    end
    
    // Main FSM for ReLU execution
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            done <= 0;
            vec_read_enable <= 0;
            vec_write_enable <= 0;
            vec_read_buffer_id <= 0;
            vec_write_buffer_id <= 0;
            tile_count <= 0;
            total_tiles_needed <= 0;
            current_element_offset <= 0;
            for (int i = 0; i < MAX_ROWS; i++) begin
                result_accumulator[i] <= 0;
                result[i] <= 0;
            end
        end else begin
            // Default signal values
            done <= 0;
            vec_read_enable <= 0;
            vec_write_enable <= 0;
            
            case (state)
                IDLE: begin
                    if (start) begin
                        // CRITICAL: Read from SOURCE buffer (x_buffer_id), not dest!
                        vec_read_buffer_id <= x_buffer_id;
                        vec_write_buffer_id <= dest_buffer_id;
                        
                        vec_read_enable <= 1;  // Pulse read enable for one cycle
                        tile_count <= 0;
                        total_tiles_needed <= (length + TILE_ELEMS - 1) / TILE_ELEMS;
                        current_element_offset <= 0;
                        state <= READ_AND_WRITE_TILES;
                        
                        //$display("[RELU_EXEC] Starting ReLU: length=%0d, src_buf=%0d, dst_buf=%0d",
                        //         length, x_buffer_id, dest_buffer_id);
                        //$display("[RELU_EXEC] Total tiles to process: %0d",
                        //         (length + TILE_ELEMS - 1) / TILE_ELEMS);
                    end
                end
                
                READ_AND_WRITE_TILES: begin
                    // Default: don't request reads (only pulse when needed)
                    vec_read_enable <= 0;
                    
                    // When read tile is valid, apply ReLU and write immediately
                    if (vec_read_valid) begin
                        vec_write_enable <= 1;
                        
                        //$display("[RELU_EXEC] Read from buffer %0d, input[0:7]=%d,%d,%d,%d,%d,%d,%d,%d",
                        //         x_buffer_id,
                        //         vec_read_tile[0], vec_read_tile[1], vec_read_tile[2], vec_read_tile[3],
                        //         vec_read_tile[4], vec_read_tile[5], vec_read_tile[6], vec_read_tile[7]);
                        
                        // Pack ReLU output to write tile
                        // Handle partial tiles by zero-padding elements beyond length
                        for (int i = 0; i < TILE_ELEMS; i++) begin
                            if (int'(current_element_offset) + i < length) begin
                                vec_write_tile[i] <= relu_output[i];
                            end else begin
                                vec_write_tile[i] <= 0;
                            end
                        end
                        
                        //$display("[RELU_EXEC] Writing to buffer %0d, output[0:7]=%d,%d,%d,%d,%d,%d,%d,%d",
                        //         dest_buffer_id,
                        //         relu_output[0], relu_output[1], relu_output[2], relu_output[3],
                        //         relu_output[4], relu_output[5], relu_output[6], relu_output[7]);
                        
                        tile_count <= tile_count + 1;
                        current_element_offset <= current_element_offset + TILE_ELEMS;
                        
                        //$display("[RELU_EXEC] Processed tile %0d/%0d",
                        //         tile_count + 1, total_tiles_needed);
                        
                        if (tile_count + 1 >= total_tiles_needed) begin
                            // All tiles processed
                            state <= COMPLETE;
                        end else begin
                            // Request next tile - pulse for one cycle only
                            vec_read_enable <= 1;
                        end
                    end
                end
                
                COMPLETE: begin
                    //$display("[RELU_EXEC] ReLU execution complete: %0d tiles processed", tile_count);
                    done <= 1;
                    // Copy results to output
                    for (int i = 0; i < MAX_ROWS; i++) begin
                        result[i] <= result_accumulator[i];
                    end
                    state <= IDLE;
                end
                
                default: begin
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
