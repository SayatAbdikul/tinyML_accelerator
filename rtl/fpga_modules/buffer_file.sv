
// Optimized buffer_file.sv with Sliced Inferred Block RAM
module buffer_file #(
    parameter DATA_WIDTH = 8,
    parameter BUFFER_WIDTH = 8192,
    parameter BUFFER_COUNT = 2,
    parameter TILE_WIDTH = 256, 
    parameter TILE_ELEMS = 32,
    parameter USE_DISTRIBUTED_RAM = 0 
)(
    input clk,
    input reset_n,
    input write_enable,
    input read_enable,
    input [TILE_WIDTH-1:0] write_data,
    input [$clog2(BUFFER_COUNT)-1:0] write_buffer,
    input [$clog2(BUFFER_COUNT)-1:0] read_buffer,
    output logic [DATA_WIDTH-1:0] read_data [0:TILE_ELEMS-1],
    output logic writing_done,
    output logic reading_done,
    input reset_indices_enable,
    input [$clog2(BUFFER_COUNT)-1:0] reset_indices_buffer,
    output logic [TILE_INDEX_WIDTH-1:0] debug_w_tile_index
);

    // [Counter Logic - Same as before]
    localparam TILE_COUNT = BUFFER_WIDTH / TILE_WIDTH;
    localparam TILE_INDEX_WIDTH = (TILE_COUNT == 1) ? 1 : $clog2(TILE_COUNT);

    logic [TILE_INDEX_WIDTH-1:0] w_tile_index [0:BUFFER_COUNT-1];
    assign debug_w_tile_index = w_tile_index[0];
    logic [TILE_INDEX_WIDTH-1:0] r_tile_index [0:BUFFER_COUNT-1];
    logic read_enable_prev;
    integer i;

    logic [TILE_INDEX_WIDTH-1:0] effective_w_index;
    assign effective_w_index = (reset_indices_enable && reset_indices_buffer == write_buffer) ? 0 : w_tile_index[write_buffer];

    logic [TILE_INDEX_WIDTH-1:0] effective_r_index;
    assign effective_r_index = (reset_indices_enable && reset_indices_buffer == read_buffer) ? 0 : r_tile_index[read_buffer];

    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            read_enable_prev <= 0;
            for (i = 0; i < BUFFER_COUNT; i++) begin
                w_tile_index[i] <= 0;
                r_tile_index[i] <= 0;
            end
            writing_done <= 0;
            reading_done <= 0;
        end else begin
            writing_done <= 0;
            reading_done <= 0;
            read_enable_prev <= read_enable;

             if (write_enable) begin
                 /* verilator lint_off WIDTHEXPAND */
                 if (effective_w_index == TILE_COUNT - 1) begin
                 /* verilator lint_on WIDTHEXPAND */
                     w_tile_index[write_buffer] <= 0;
                     writing_done <= 1;
                 end else begin
                     w_tile_index[write_buffer] <= effective_w_index + 1;
                 end
            end

            if (read_enable) begin
                 //$display("[BUF_DBG] Read! buf=%d idx=%d effective=%d addr=%0d data=%h", read_buffer, r_tile_index[read_buffer], effective_r_index, flat_r_addr, read_data[0]);
                 /* verilator lint_off WIDTHEXPAND */
                 if (effective_r_index == TILE_COUNT - 1) begin
                 /* verilator lint_on WIDTHEXPAND */
                     r_tile_index[read_buffer] <= 0;
                     reading_done <= 1;
                 end else begin
                     r_tile_index[read_buffer] <= effective_r_index + 1;
                 end
            end
        end
    end

    // [Memory Logic - FLAT]
    localparam TOTAL_TILES = BUFFER_COUNT * TILE_COUNT;
    localparam BUFFER_BITS = (BUFFER_COUNT > 1) ? $clog2(BUFFER_COUNT) : 0;
    localparam FLAT_ADDR_WIDTH = (BUFFER_COUNT > 1) ? (BUFFER_BITS + TILE_INDEX_WIDTH) : TILE_INDEX_WIDTH;

    logic [TILE_INDEX_WIDTH-1:0] curr_w_idx_local;
    logic [TILE_INDEX_WIDTH-1:0] curr_r_idx_local;
    assign curr_w_idx_local = (reset_indices_enable && reset_indices_buffer == write_buffer) ? 0 : w_tile_index[write_buffer];
    assign curr_r_idx_local = (reset_indices_enable && reset_indices_buffer == read_buffer) ? 0 : r_tile_index[read_buffer];

    logic [FLAT_ADDR_WIDTH-1:0] flat_w_addr;
    logic [FLAT_ADDR_WIDTH-1:0] flat_r_addr;
    
    generate
        if (BUFFER_COUNT > 1) begin
             assign flat_w_addr = {write_buffer, curr_w_idx_local};
             assign flat_r_addr = {read_buffer, curr_r_idx_local};
        end else begin
             assign flat_w_addr = curr_w_idx_local;
             assign flat_r_addr = curr_r_idx_local;
        end
    endgenerate

    // SIMPLE FLAT MEMORY
    (* syn_ram_style = "block_ram" *)
    logic [TILE_WIDTH-1:0] mem [0:TOTAL_TILES-1];
    logic [TILE_WIDTH-1:0] mem_rdata;
    
    // Write
    always_ff @(posedge clk) begin
        if (write_enable) begin
            mem[flat_w_addr] <= write_data;
        end
    end
    
    // Read
    always_ff @(posedge clk) begin
        if (read_enable) begin
            mem_rdata <= mem[flat_r_addr];
        end
    end
    
    // Unpack Logic
    genvar j;
    generate
        for (j = 0; j < TILE_ELEMS; j = j + 1) begin : unpack_loop
            assign read_data[j] = mem_rdata[j*DATA_WIDTH +: DATA_WIDTH];
        end
    endgenerate

endmodule
