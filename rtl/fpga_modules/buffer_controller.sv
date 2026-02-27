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
    parameter TILE_ELEMS = 32,
    parameter VECTOR_BUFFER_WIDTH = 8192,     // 1KB per buffer  
    parameter MATRIX_BUFFER_WIDTH = 32768,    // 4KB per buffer (saves 8 BRAM blocks)
    parameter VECTOR_BUFFER_COUNT = 8,        // Reduced to 8 to save BRAM
    parameter MATRIX_BUFFER_COUNT = 2         // 2 matrix buffers
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
    output logic mat_read_done,
    
    // Cache clear (force reset of internal tracking)
    input logic clr_cache
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
    
    // Track last accessed buffer for each operation type
    logic [4:0] vec_last_read_buffer_id, vec_last_write_buffer_id;
    logic [4:0] mat_last_read_buffer_id, mat_last_write_buffer_id;
    
    // Internal reset signals
    logic vec_read_reset, vec_write_reset;
    logic mat_read_reset, mat_write_reset;
    logic vec_reset_indices, mat_reset_indices;
    
    // Detect when we switch buffers within the SAME operation type
    always_comb begin
        // Vector resets
        vec_read_reset = vec_read_enable && (vec_read_buffer_id != vec_last_read_buffer_id);
        vec_write_reset = vec_write_enable && (vec_write_buffer_id != vec_last_write_buffer_id);
        
        // Matrix resets
        mat_read_reset = mat_read_enable && (mat_read_buffer_id != mat_last_read_buffer_id);
        mat_write_reset = mat_write_enable && (mat_write_buffer_id != mat_last_write_buffer_id);
        
        // Combined resets (Prioritize Read reset if collision - rare)
        vec_reset_indices = vec_read_reset || vec_write_reset;
        mat_reset_indices = mat_read_reset || mat_write_reset;
    end
    
    always_ff @(posedge clk or posedge rst) begin
        if (rst || clr_cache) begin
            vec_last_read_buffer_id <= 5'h1F;
            vec_last_write_buffer_id <= 5'h1F;
            mat_last_read_buffer_id <= 5'h1F;
            mat_last_write_buffer_id <= 5'h1F;
        end else begin
            // Track last operation
            if (vec_read_enable) vec_last_read_buffer_id <= vec_read_buffer_id;
            if (vec_write_enable) vec_last_write_buffer_id <= vec_write_buffer_id;
            
            if (mat_read_enable) mat_last_read_buffer_id <= mat_read_buffer_id;
            if (mat_write_enable) mat_last_write_buffer_id <= mat_write_buffer_id;
        end
    end
    
    // Select which buffer to reset
    /* verilator lint_off UNUSEDSIGNAL */
    logic [4:0] vec_reset_buffer_id, mat_reset_buffer_id;
    /* verilator lint_on UNUSEDSIGNAL */
    
    // If Read Reset is active, use Read ID. Else use Write ID.
    assign vec_reset_buffer_id = vec_read_reset ? vec_read_buffer_id : vec_write_buffer_id;
    assign mat_reset_buffer_id = mat_read_reset ? mat_read_buffer_id : mat_write_buffer_id;
    
    // Vector Buffer File Instance
    // Optimized for smaller vectors (x, bias, intermediate activations)
    localparam VEC_BUF_IDX_W = $clog2(VECTOR_BUFFER_COUNT);
    localparam MAT_BUF_IDX_W = $clog2(MATRIX_BUFFER_COUNT);

    buffer_file #(
        .BUFFER_WIDTH(VECTOR_BUFFER_WIDTH),
        .BUFFER_COUNT(VECTOR_BUFFER_COUNT),
        .TILE_WIDTH(TILE_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .TILE_ELEMS(TILE_ELEMS),
        .USE_DISTRIBUTED_RAM(0)
    ) vector_buffer_inst (
        .clk(clk),
        .reset_n(~rst),
        .write_enable(vec_write_enable),
        .read_enable(vec_read_enable),
        .write_data(vec_write_tile_packed),
        .write_buffer(vec_write_buffer_id[VEC_BUF_IDX_W-1:0]),
        .read_buffer(vec_read_buffer_id[VEC_BUF_IDX_W-1:0]),
        .read_data(vec_read_tile),
        .writing_done(vec_write_done),
        .reading_done(vec_read_done),
        .reset_indices_enable(vec_reset_indices),
        .reset_indices_buffer(vec_reset_buffer_id[VEC_BUF_IDX_W-1:0]),
        .debug_w_tile_index()
    );

    // Matrix Buffer File Instance
    // Optimized for larger weight matrices
    buffer_file #(
        .BUFFER_WIDTH(MATRIX_BUFFER_WIDTH),
        .BUFFER_COUNT(MATRIX_BUFFER_COUNT),
        .TILE_WIDTH(TILE_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .TILE_ELEMS(TILE_ELEMS),
        .USE_DISTRIBUTED_RAM(0)
    ) matrix_buffer_inst (
        .clk(clk),
        .reset_n(~rst),
        .write_enable(mat_write_enable),
        .read_enable(mat_read_enable),
        .write_data(mat_write_tile),
        .write_buffer(mat_write_buffer_id[MAT_BUF_IDX_W-1:0]),
        .read_buffer(mat_read_buffer_id[MAT_BUF_IDX_W-1:0]),
        .read_data(mat_read_tile),
        .writing_done(mat_write_done),
        .reading_done(mat_read_done),
        .reset_indices_enable(mat_reset_indices),
        .reset_indices_buffer(mat_reset_buffer_id[MAT_BUF_IDX_W-1:0]),
        .debug_w_tile_index()
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
