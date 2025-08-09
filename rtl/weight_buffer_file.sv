module weight_buffer_file #(
    parameter BUFFER_WIDTH = 1024, // in testing purposes, should be 100352
    parameter BUFFER_COUNT = 2,
    parameter TILE_WIDTH = 256,
    parameter DATA_WIDTH = 8,
    parameter TILE_SIZE = 32
)(
    input clk,
    input reset_n,
    input write_enable,
    input read_enable,
    input [TILE_WIDTH-1:0] write_data,
    input [$clog2(BUFFER_COUNT)-1:0] write_buffer,
    input [$clog2(BUFFER_COUNT)-1:0] read_buffer,
    output reg [TILE_SIZE-1:0][DATA_WIDTH-1:0] read_data,
    output reg writing_done,
    output reg reading_done
);

// Calculate buffer parameters
localparam TILE_COUNT = BUFFER_WIDTH / TILE_WIDTH;
localparam TILE_INDEX_WIDTH = (TILE_COUNT == 1) ? 1 : $clog2(TILE_COUNT);
localparam INDEX_WIDTH = $clog2(BUFFER_WIDTH);

// Memory buffers
logic [BUFFER_WIDTH-1:0] buffers [0:BUFFER_COUNT-1];

// Tile indices
logic [TILE_INDEX_WIDTH-1:0] w_tile_index, r_tile_index;

// Bit indices (calculated from tile indices)
wire [INDEX_WIDTH-1:0] w_bit_index = w_tile_index * TILE_WIDTH;
wire [INDEX_WIDTH-1:0] r_bit_index = r_tile_index * TILE_WIDTH;

// Reset and operation logic
integer i;
always @(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
        // Reset buffers and indices
        for (i = 0; i < BUFFER_COUNT; i++) begin
            buffers[i] <= {BUFFER_WIDTH{1'b0}};
        end
        w_tile_index <= 0;
        r_tile_index <= 0;
        read_data <= 0;
        writing_done <= 0;
        reading_done <= 0;
    end else begin
        // Default done signals
        writing_done <= 0;
        reading_done <= 0;

        // Write operation
        if (write_enable) begin
            buffers[write_buffer][w_bit_index +: TILE_WIDTH] <= write_data;
            
            if ({ {(32-TILE_INDEX_WIDTH){1'b0}}, w_tile_index } == TILE_COUNT - 1) begin
                w_tile_index <= 0;
                writing_done <= 1;
                $display("Writing done, the value is %0h", buffers[write_buffer]);
            end else begin
                w_tile_index <= w_tile_index + 1;
            end
        end 
        // Read operation
        else if (read_enable) begin
            for (i = 0; i < TILE_SIZE; i++) begin
                read_data[i] <= buffers[read_buffer][{ {(32-INDEX_WIDTH){1'b0}}, r_bit_index }+i*8 +: DATA_WIDTH];
                //$display("Read data[%0d]: %0h", i, buffers[read_buffer][r_bit_index+i*8 +: DATA_WIDTH]);
            end
            if ({ {(32-TILE_INDEX_WIDTH){1'b0}}, r_tile_index } == TILE_COUNT - 1) begin
                r_tile_index <= 0;
                reading_done <= 1;
                //$display("Reading done here, the value is %0h", read_data);
            end else begin
                r_tile_index <= r_tile_index + 1;
            end
        end
    end
end

endmodule
