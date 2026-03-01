// ReLU Execution Module - Optimized Tile Streaming
// Handles ReLU (Rectified Linear Unit) activation operations
// Reads input vector from buffer, applies ReLU, writes result to destination buffer
// Uses tile-by-tile processing without storing full vectors
//
// CRITICAL FIX: This module correctly reads from source buffer (x_buffer_id)
// and writes to destination buffer (dest_buffer_id)

module relu_execution #(
    parameter DATA_WIDTH = 8,
    parameter TILE_ELEMS = 32,
    parameter MAX_ROWS = 784
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
    
    // Result output - just current tile for status (not full vector)
    output logic signed [DATA_WIDTH-1:0] result [0:TILE_ELEMS-1]
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
    
    // ReLU input and output for current tile (small - just TILE_ELEMS)
    logic signed [DATA_WIDTH-1:0] relu_input [0:TILE_ELEMS-1];
    logic signed [DATA_WIDTH-1:0] relu_output [0:TILE_ELEMS-1];
    
    // ReLU module instantiation
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
    
    // Connect ReLU output to result for status
    always_comb begin
        result = relu_output;
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
            for (int i = 0; i < TILE_ELEMS; i++) begin
                vec_write_tile[i] <= 0;
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
                    end
                end
                
                READ_AND_WRITE_TILES: begin
                    // Default: don't request reads (only pulse when needed)
                    vec_read_enable <= 0;
                    
                    // When read tile is valid, apply ReLU and write immediately
                    if (vec_read_valid) begin
                        vec_write_enable <= 1;
                        
                        // Pack ReLU output to write tile
                        // Handle partial tiles by zero-padding elements beyond length
                        for (int i = 0; i < TILE_ELEMS; i++) begin
                            if (int'(current_element_offset) + i < length) begin
                                vec_write_tile[i] <= relu_output[i];
                            end else begin
                                vec_write_tile[i] <= 0;
                            end
                        end
                        
                        tile_count <= tile_count + 1;
                        current_element_offset <= current_element_offset + TILE_ELEMS;
                        
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
