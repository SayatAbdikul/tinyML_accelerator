// Buffer Controller Module
// Encapsulates vector and matrix buffer files, providing a unified interface
// for reading and writing tiles. Manages addressing and tile-based I/O operations.
//
// Features:
// - Dual buffer system: separate vector and matrix buffers
// - Tile-based read/write operations
// - Automatic tile indexing
// - Support for multiple concurrent buffer IDs

module buffer_controller #(
    parameter DATA_WIDTH = 8,
    parameter TILE_WIDTH = 256,
    parameter TILE_ELEMS = TILE_WIDTH / DATA_WIDTH,  // 32 elements per tile
    parameter VECTOR_BUFFER_WIDTH = 8192,           // Smaller for vectors
    parameter MATRIX_BUFFER_WIDTH = 802820,         // Larger for matrices
    parameter BUFFER_COUNT = 32                     // Number of logical buffers
)(
    input logic clk,
    input logic rst,
    
    // Vector buffer write interface
    input logic vec_write_enable,
    input logic [4:0] vec_write_buffer_id,
    input logic signed [DATA_WIDTH-1:0] vec_write_tile [0:TILE_ELEMS-1],
    
    // Vector buffer read interface
    input logic vec_read_enable,
    input logic [4:0] vec_read_buffer_id,
    output logic signed [DATA_WIDTH-1:0] vec_read_tile [0:TILE_ELEMS-1],
    output logic vec_read_valid,
    
    // Matrix buffer write interface
    input logic mat_write_enable,
    input logic [4:0] mat_write_buffer_id,
    input logic [TILE_WIDTH-1:0] mat_write_tile,
    
    // Matrix buffer read interface
    input logic mat_read_enable,
    input logic [4:0] mat_read_buffer_id,
    output logic signed [DATA_WIDTH-1:0] mat_read_tile [0:TILE_ELEMS-1],
    output logic mat_read_valid,
    
    // Status signals
    output logic vec_write_done,
    output logic vec_read_done,
    output logic mat_write_done,
    output logic mat_read_done
);

    // Packed tile for vector writes
    logic [TILE_WIDTH-1:0] vec_write_tile_packed;
    
    // Delayed read enable signals for valid generation (buffer has 1-cycle read latency)
    logic vec_read_enable_d;
    logic mat_read_enable_d;
    
    // Pack vector tile data from unpacked array to packed bus
    always_comb begin
        vec_write_tile_packed = '0;
        for (int i = 0; i < TILE_ELEMS; i++) begin
            vec_write_tile_packed[i*DATA_WIDTH +: DATA_WIDTH] = vec_write_tile[i];
        end
    end
    
    // Vector Buffer File Instance
    // Optimized for smaller vectors (x, bias, intermediate activations)
    buffer_file #(
        .BUFFER_WIDTH(VECTOR_BUFFER_WIDTH),
        .BUFFER_COUNT(BUFFER_COUNT),
        .TILE_WIDTH(TILE_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .TILE_SIZE(TILE_ELEMS)
    ) vector_buffer_inst (
        .clk(clk),
        .reset_n(~rst),
        .write_enable(vec_write_enable),
        .read_enable(vec_read_enable),
        .write_data(vec_write_tile_packed),
        .write_buffer(vec_write_buffer_id),
        .read_buffer(vec_read_buffer_id),
        .read_data(vec_read_tile),
        .writing_done(vec_write_done),
        .reading_done(vec_read_done)
    );
    
    // Matrix Buffer File Instance
    // Optimized for larger weight matrices
    buffer_file #(
        .BUFFER_WIDTH(MATRIX_BUFFER_WIDTH),
        .BUFFER_COUNT(BUFFER_COUNT),
        .TILE_WIDTH(TILE_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .TILE_SIZE(TILE_ELEMS)
    ) matrix_buffer_inst (
        .clk(clk),
        .reset_n(~rst),
        .write_enable(mat_write_enable),
        .read_enable(mat_read_enable),
        .write_data(mat_write_tile),
        .write_buffer(mat_write_buffer_id),
        .read_buffer(mat_read_buffer_id),
        .read_data(mat_read_tile),
        .writing_done(mat_write_done),
        .reading_done(mat_read_done)
    );
    
    // Generate read valid signals (1 cycle after read enable)
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            vec_read_enable_d <= 0;
            mat_read_enable_d <= 0;
            vec_read_valid <= 0;
            mat_read_valid <= 0;
        end else begin
            vec_read_enable_d <= vec_read_enable;
            mat_read_enable_d <= mat_read_enable;
            vec_read_valid <= vec_read_enable_d;
            mat_read_valid <= mat_read_enable_d;
        end
    end

endmodule
