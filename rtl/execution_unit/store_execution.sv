// Store Execution Module
// Handles STORE operations - writes vector data from buffer to memory
// Currently a placeholder for future implementation
//
// Operation Flow:
// 1. Read vector tiles from source buffer
// 2. Write data to DRAM at specified address
// 3. Signal done when complete
//
// NOTE: This is a simplified placeholder. Full implementation would require
// a store_v module similar to load_v for DRAM writes.

module store_execution #(
    parameter DATA_WIDTH = 8,
    parameter TILE_WIDTH = 256,
    parameter TILE_ELEMS = TILE_WIDTH / DATA_WIDTH,
    parameter ADDR_WIDTH = 24
)(
    input logic clk,
    input logic rst,
    
    // Control interface
    input logic start,
    input logic [4:0] src_buffer_id,    // Source buffer to read from
    input logic [9:0] length,           // Vector length to store
    input logic [ADDR_WIDTH-1:0] addr,  // DRAM address to write to
    output logic done,
    
    // Buffer controller interface - vector reads
    output logic vec_read_enable,
    output logic [4:0] vec_read_buffer_id,
    input logic signed [DATA_WIDTH-1:0] vec_read_tile [0:TILE_ELEMS-1],
    input logic vec_read_valid
    
    // TODO: Add DRAM write interface when store_v module is implemented
);

    // FSM states
    typedef enum logic [1:0] {
        IDLE,
        READING_TILES,
        COMPLETE
    } store_state_t;
    
    store_state_t state;
    
    // Tile counters
    logic [9:0] tile_count;
    logic [9:0] total_tiles_needed;
    logic [9:0] current_element_offset;
    
    // TODO: Add store_v module instantiation here when implemented
    // For now, this is a placeholder that just reads from buffer and discards
    
    // Main FSM for store execution
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            done <= 0;
            vec_read_enable <= 0;
            vec_read_buffer_id <= 0;
            tile_count <= 0;
            total_tiles_needed <= 0;
            current_element_offset <= 0;
        end else begin
            // Default signal values
            done <= 0;
            vec_read_enable <= 0;
            
            case (state)
                IDLE: begin
                    if (start) begin
                        vec_read_buffer_id <= src_buffer_id;
                        vec_read_enable <= 1;
                        tile_count <= 0;
                        total_tiles_needed <= (length + TILE_ELEMS - 1) / TILE_ELEMS;
                        current_element_offset <= 0;
                        state <= READING_TILES;
                        
                        $display("[STORE_EXEC] Starting STORE: src_buf=%0d, length=%0d, addr=0x%h",
                                 src_buffer_id, length, addr);
                        $display("[STORE_EXEC] NOTE: Placeholder implementation - data not written to DRAM");
                    end
                end
                
                READING_TILES: begin
                    // Read tiles from buffer
                    if (vec_read_valid) begin
                        // TODO: Write tile to DRAM via store_v module
                        // For now, just log that we read the tile
                        $display("[STORE_EXEC] Read tile %0d/%0d from buffer (not stored to DRAM yet)",
                                 tile_count + 1, total_tiles_needed);
                        
                        // Log non-zero data for debugging
                        for (int i = 0; i < TILE_ELEMS; i++) begin
                            if (int'(current_element_offset) + i < length && vec_read_tile[i] != 0) begin
                                $display("[STORE_EXEC] Element %0d: %0d",
                                         current_element_offset + i, vec_read_tile[i]);
                            end
                        end
                        
                        tile_count <= tile_count + 1;
                        current_element_offset <= current_element_offset + TILE_ELEMS;
                        
                        if (tile_count + 1 >= total_tiles_needed) begin
                            state <= COMPLETE;
                        end else begin
                            vec_read_enable <= 1;
                        end
                    end else if (tile_count < total_tiles_needed) begin
                        vec_read_enable <= 1;
                    end
                end
                
                COMPLETE: begin
                    $display("[STORE_EXEC] STORE execution complete (placeholder): %0d tiles read", tile_count);
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
